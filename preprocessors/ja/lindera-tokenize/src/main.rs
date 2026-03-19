// Japanese morphological tokenizer using lindera + ipadic (embedded).
// Protocol: reads lines from stdin, writes one tokenized line per input line to stdout.
// Outputs content morphemes only — filters 助詞 (particles), 助動詞 (aux verbs), 記号 (symbols).
// Mode::Decompose with default penalties decomposes kanji compound nouns (threshold=2 for kanji).
use std::io::{self, BufRead, Write};

use lindera::dictionary::load_dictionary;
use lindera::mode::{Mode, Penalty};
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer;

fn is_content(pos: &str) -> bool {
    !matches!(pos, "助詞" | "助動詞" | "記号" | "接続詞" | "感動詞")
}

fn main() -> lindera::LinderaResult<()> {
    let dictionary = load_dictionary("embedded://ipadic")?;
    // Default penalty: kanji_threshold=2, other_threshold=7 — correct for Japanese kanji compounds.
    let segmenter = Segmenter::new(Mode::Decompose(Penalty::default()), dictionary, None);
    let tokenizer = Tokenizer::new(segmenter);

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line.expect("stdin read error");
        if line.is_empty() {
            writeln!(out).unwrap();
            out.flush().unwrap();
            continue;
        }
        match tokenizer.tokenize(&line) {
            Ok(mut tokens) => {
                let mut parts: Vec<String> = Vec::new();
                for t in tokens.iter_mut() {
                    let pos = t.get("part_of_speech").unwrap_or("*");
                    if is_content(pos) {
                        parts.push(t.surface.to_string());
                    }
                }
                let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
                writeln!(out, "{}", parts.join(" ")).unwrap();
            }
            Err(_) => {
                writeln!(out, "{}", line).unwrap();
            }
        }
        out.flush().unwrap();
    }
    Ok(())
}
