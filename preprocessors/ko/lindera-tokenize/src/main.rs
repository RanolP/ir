// Korean morphological tokenizer using lindera + mecab-ko-dic (embedded).
// Protocol: reads lines from stdin, writes one tokenized line per input line to stdout.
// Outputs content morphemes only — filters particles (J*), endings (E*), affixes, symbols.
// POS tag is details[0] ("part_of_speech_tag") in mecab-ko-dic schema.
use std::io::{self, BufRead, Write};

use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer;

fn is_content(tag: &str) -> bool {
    !matches!(tag.chars().next(), Some('J') | Some('E'))
        && !matches!(
            tag,
            "XPN" | "XSN" | "XSV" | "XSA" | "SF" | "SP" | "SS" | "SE" | "SO" | "SW" | "SWK"
        )
}

fn main() -> lindera::LinderaResult<()> {
    let dictionary = load_dictionary("embedded://ko-dic")?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
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
                    let tag = t.get("part_of_speech_tag").unwrap_or("*");
                    if is_content(tag) {
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
