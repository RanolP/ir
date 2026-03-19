// Chinese bigram tokenizer — no dictionary required.
// Protocol: reads lines from stdin, writes one tokenized line per input line to stdout.
// Algorithm: CJK runs → overlapping bigrams; non-CJK runs → whitespace-split tokens.
// Matches Lucene CJKAnalyzer baseline used in Anserini.
// Example: "信息检索系统" → "信息 息检 检索 索系 系统"
use std::io::{self, BufRead, Write};

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'  // CJK Extension A
        | '\u{4E00}'..='\u{9FFF}'  // CJK Unified Ideographs
        | '\u{F900}'..='\u{FAFF}'  // CJK Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
        | '\u{2A700}'..='\u{2CEAF}' // Extensions C/D/E
        | '\u{2CEB0}'..='\u{2EBEF}' // Extension F
    )
}

fn tokenize_line(line: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if is_cjk(chars[i]) {
            // Emit bigrams for the CJK run.
            let start = i;
            while i < n && is_cjk(chars[i]) {
                i += 1;
            }
            let run = &chars[start..i];
            if run.len() == 1 {
                out.push(run.iter().collect());
            } else {
                for w in run.windows(2) {
                    out.push(w.iter().collect());
                }
            }
        } else if chars[i].is_whitespace() {
            i += 1;
        } else {
            // Non-CJK token: collect until whitespace or CJK.
            let start = i;
            while i < n && !chars[i].is_whitespace() && !is_cjk(chars[i]) {
                i += 1;
            }
            let tok: String = chars[start..i].iter().collect();
            let tok = tok.to_lowercase();
            if !tok.is_empty() {
                out.push(tok);
            }
        }
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut tokens: Vec<String> = Vec::new();

    for line in stdin.lock().lines() {
        let line = line.expect("stdin read error");
        if line.is_empty() {
            writeln!(out).unwrap();
            out.flush().unwrap();
            continue;
        }
        tokens.clear();
        tokenize_line(&line, &mut tokens);
        let parts: Vec<&str> = tokens.iter().map(String::as_str).collect();
        writeln!(out, "{}", parts.join(" ")).unwrap();
        out.flush().unwrap();
    }
}
