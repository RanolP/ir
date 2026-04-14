// Document chunker: 512-token chunks with 15% overlap and break-point scoring.
// Ported from qmd's chunkDocument() in store.ts.
//
// Token approximation: 4 chars ≈ 1 token (same as qmd).
// Break-point scoring (higher = preferred split location):
//   h1=100, h2=90, h3=80, h4=70, h5=60, h6=50
//   code fence boundary = 80
//   blank line = 20
//   list item  = 5
//   newline    = 1
//
// Scoring uses quadratic distance decay toward the target position:
//   final = break_score * (1 - (norm_dist^2) * 0.7)

use std::sync::atomic::{AtomicUsize, Ordering};

const CHARS_PER_TOKEN: usize = 4;
const DEFAULT_CHUNK_SIZE_TOKENS: usize = 512;
const CHUNK_OVERLAP_PERCENT: usize = 15;
/// Minimum chunk size: chunks shorter than this are merged with their predecessor.
const MIN_CHUNK_SIZE_TOKENS: usize = 100;
/// Window before the target end position in which to search for a break point.
const BREAK_WINDOW_CHARS: usize = 800;
static CHUNK_SIZE_OVERRIDE_TOKENS: AtomicUsize = AtomicUsize::new(0);

#[allow(dead_code)] // used by eval binary
pub fn set_chunk_size_tokens_override(tokens: Option<usize>) {
    CHUNK_SIZE_OVERRIDE_TOKENS.store(tokens.unwrap_or(0), Ordering::Relaxed);
}

pub fn chunk_size_tokens() -> usize {
    let v = CHUNK_SIZE_OVERRIDE_TOKENS.load(Ordering::Relaxed);
    if v > 0 { v } else { DEFAULT_CHUNK_SIZE_TOKENS }
}

fn chunk_overlap_tokens(chunk_size_tokens: usize) -> usize {
    ((chunk_size_tokens * CHUNK_OVERLAP_PERCENT) + 50) / 100
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub seq: usize,
    /// Byte offset of this chunk's start in the original document.
    pub pos: usize,
    pub text: String,
}

pub fn chunk_document(doc: &str) -> Vec<Chunk> {
    let chunk_size_chars = chunk_size_tokens() * CHARS_PER_TOKEN;
    let chunk_overlap_chars = chunk_overlap_tokens(chunk_size_tokens()) * CHARS_PER_TOKEN;
    let min_chunk_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;

    if doc.len() <= chunk_size_chars {
        return vec![Chunk { seq: 0, pos: 0, text: doc.to_string() }];
    }

    let break_points = precompute_break_points(doc);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0usize;
    let mut prev_end = 0usize;

    while start < doc.len() {
        let doc_tail = doc.len() - start;
        // Don't pick a break that produces a sub-floor chunk or re-uses the previous boundary
        // (which would happen when a semantic break falls inside the overlap region).
        let min_break_pos = (start + min_chunk_chars).max(prev_end + 1);

        let end = if doc_tail <= chunk_size_chars {
            doc.len()
        } else {
            let naive_target = start + chunk_size_chars;
            let remaining_after = doc.len() - naive_target;
            if remaining_after >= min_chunk_chars {
                // Healthy tail room — normal split.
                best_break(doc, &break_points, start, naive_target, min_break_pos)
            } else if doc_tail >= 2 * min_chunk_chars {
                // Would leave a sub-min tail but room for two min-sized chunks:
                // rebalance so the tail hits the floor, keeping this chunk ≤ chunk_size.
                let rebalanced_target = doc.len() - min_chunk_chars;
                best_break(doc, &break_points, start, rebalanced_target, min_break_pos)
            } else {
                // Can't fit two min-sized chunks — absorb the rest.
                doc.len()
            }
        };

        chunks.push(Chunk { seq: chunks.len(), pos: start, text: doc[start..end].to_string() });
        prev_end = end;

        if end == doc.len() {
            break;
        }

        let prev_start = start;
        start = end.saturating_sub(chunk_overlap_chars);
        while start < end && !doc.is_char_boundary(start) {
            start += 1;
        }
        // Defense-in-depth: guarantee forward progress for incoherent configs
        // (chunk_size < min_chars + overlap) that would otherwise loop forever.
        if start <= prev_start {
            start = end;
        }
    }

    chunks
}

/// Precompute all break point positions and their scores.
fn precompute_break_points(doc: &str) -> Vec<(usize, f64)> {
    let mut points: Vec<(usize, f64)> = Vec::new();
    let mut in_code_fence = false;
    let mut pos = 0usize;

    for line in doc.lines() {
        let line_start = pos;
        let line_end = pos + line.len();

        if line.starts_with("```") || line.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            // The fence boundary itself is a good split point.
            points.push((line_start, 80.0));
        } else if !in_code_fence {
            let score = line_break_score(line);
            if score > 0.0 {
                points.push((line_start, score));
            }
        }

        // +1 for the newline char (lines() strips it)
        pos = line_end + 1;
    }

    points
}

fn line_break_score(line: &str) -> f64 {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return 20.0;
    }
    if trimmed.starts_with("# ") || trimmed == "#" {
        return 100.0;
    }
    if trimmed.starts_with("## ") || trimmed == "##" {
        return 90.0;
    }
    if trimmed.starts_with("### ") || trimmed == "###" {
        return 80.0;
    }
    if trimmed.starts_with("#### ") || trimmed == "####" {
        return 70.0;
    }
    if trimmed.starts_with("##### ") || trimmed == "#####" {
        return 60.0;
    }
    if trimmed.starts_with("###### ") || trimmed == "######" {
        return 50.0;
    }
    if trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || trimmed
            .split_once(". ")
            .map(|(n, _)| n.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    {
        return 5.0;
    }
    1.0 // bare newline between non-empty lines
}

/// Find the best split position within BREAK_WINDOW_CHARS before target_end.
/// Only considers break points at or after min_break_pos (to prevent sub-floor chunks
/// and to avoid re-selecting the previous chunk's boundary when it falls in the overlap).
/// Falls back to a char boundary at target_end if no qualifying break is found.
fn best_break(
    doc: &str,
    break_points: &[(usize, f64)],
    start: usize,
    target_end: usize,
    min_break_pos: usize,
) -> usize {
    let window_start = target_end.saturating_sub(BREAK_WINDOW_CHARS).max(start);
    let window_size = (target_end - window_start) as f64;

    let best = break_points
        .iter()
        .filter(|(pos, _)| *pos > window_start && *pos <= target_end && *pos >= min_break_pos)
        .map(|(pos, score)| {
            // Distance from target_end, normalized to [0, 1] (0 = at target).
            let dist = (target_end - pos) as f64;
            let norm_dist = if window_size > 0.0 { dist / window_size } else { 0.0 };
            let adjusted = score * (1.0 - norm_dist.powi(2) * 0.7);
            (*pos, adjusted)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((pos, _)) = best {
        let mut p = pos;
        while p < doc.len() && !doc.is_char_boundary(p) {
            p += 1;
        }
        return p;
    }

    // No qualifying break found — fall back to target_end.
    let mut p = target_end;
    while p < doc.len() && !doc.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Extract a title from the document.
/// Priority: YAML frontmatter `title` or `name` field → first `# Heading` → first non-empty line → filename.
pub fn extract_title(doc: &str, path_hint: &str) -> String {
    let mut lines = doc.lines().peekable();

    // Parse YAML frontmatter
    if lines.peek() == Some(&"---") {
        lines.next(); // consume opening ---
        let mut fm_title: Option<String> = None;
        for line in lines.by_ref() {
            if line == "---" || line == "..." {
                break;
            }
            // Match `title: value` or `name: value`
            if let Some(rest) = line
                .strip_prefix("title:")
                .or_else(|| line.strip_prefix("name:"))
            {
                let val = rest.trim().trim_matches('"').trim_matches('\'');
                if !val.is_empty() {
                    fm_title = Some(val.to_string());
                    // Keep consuming until end of frontmatter
                }
            }
        }
        if let Some(t) = fm_title {
            return t;
        }
        // Fall through to scan the rest of the document for headings.
    }

    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("# ") {
            return trimmed[2..].trim().to_string();
        }
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // Filename without extension as final fallback.
    std::path::Path::new(path_hint)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_hint.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_doc_is_single_chunk() {
        let doc = "Hello world";
        let chunks = chunk_document(doc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, doc);
        assert_eq!(chunks[0].pos, 0);
    }

    #[test]
    fn long_doc_splits_into_multiple_chunks() {
        // Build a doc > CHUNK_SIZE_CHARS
        let line = "word ".repeat(100); // 500 chars
        let doc = (line + "\n").repeat(10); // 5010 chars > 3600 (chunk_size)
        let chunks = chunk_document(&doc);
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        // Each chunk should be non-empty
        for c in &chunks {
            assert!(!c.text.is_empty());
        }
    }

    #[test]
    fn chunks_cover_full_document() {
        let line = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let doc = (line + "\n\n## Section\n\n").repeat(10);
        let chunks = chunk_document(&doc);
        // Last chunk should end at doc end
        let last = chunks.last().unwrap();
        assert_eq!(last.pos + last.text.len(), doc.len());
    }

    #[test]
    fn doc_at_chunk_size_boundary_is_single_chunk() {
        // chunk_size=200 tokens=800 chars. Doc of 500 chars ≤ 800 → early-return single chunk.
        set_chunk_size_tokens_override(Some(200));
        let doc = "word ".repeat(100); // 500 chars
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 1, "doc ≤ chunk_size should be a single chunk");
        assert_eq!(chunks[0].text.len(), doc.len());
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn rebalances_when_tail_would_be_sub_min() {
        // chunk_size=200 tokens=800 chars, min=100 tokens=400 chars.
        // Doc of 1000 chars: naive split → remaining_after=200 < min(400), sub-min tail.
        // doc_tail=1000 ≥ 2*min=800 → rebalance: target pulled to 1000-400=600.
        // Result: ≥2 chunks, all ≥ min_chars.
        set_chunk_size_tokens_override(Some(200));
        let doc = "word ".repeat(200); // 1000 chars
        let chunks = chunk_document(&doc);
        assert!(chunks.len() >= 2, "rebalanced doc should produce ≥2 chunks");
        let min_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;
        for c in &chunks {
            assert!(c.text.len() >= min_chars, "chunk {} below floor: {} chars", c.seq, c.text.len());
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn normal_split_when_tail_is_healthy() {
        // chunk_size=200 tokens=800 chars, min=400 chars.
        // Doc ~1414 chars: heading break at ~702 (section boundary).
        // remaining_after naive split ≈ 614 ≥ 400 → normal split, rebalance not triggered.
        // min_break_pos prevents the overlapping chunk from re-selecting the heading.
        // All chunks must be ≥ min.
        set_chunk_size_tokens_override(Some(200));
        let section = "word ".repeat(140); // 700 chars
        let doc = format!("{section}\n\n## Section\n\n{section}"); // ~1414 chars
        let chunks = chunk_document(&doc);
        assert!(chunks.len() >= 2, "healthy-tail doc should produce ≥2 chunks");
        let min_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;
        for c in &chunks {
            assert!(c.text.len() >= min_chars, "chunk {} below floor: {} chars", c.seq, c.text.len());
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn extract_title_from_heading() {
        let doc = "# My Title\n\nContent here.";
        assert_eq!(extract_title(doc, "file.md"), "My Title");
    }

    #[test]
    fn extract_title_fallback_to_filename() {
        let doc = "";
        assert_eq!(extract_title(doc, "my-note.md"), "my-note");
    }
}
