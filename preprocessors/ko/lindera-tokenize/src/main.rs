// Korean morphological tokenizer using lindera + mecab-ko-dic (embedded).
// Protocol: reads lines from stdin, writes one tokenized line per input line to stdout.
// Outputs content morphemes only — filters particles (J*), endings (E*), affixes, symbols.
// POS tag is details[0] ("part_of_speech_tag") in mecab-ko-dic schema.
use std::io::{self, BufRead, Write};

use lindera::dictionary::load_dictionary;
use lindera::mode::{Mode, Penalty};
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
    // ! other_penalty_length_threshold=2: Hangul is "other" (not kanji), default=7 skips all short compounds
    let segmenter = Segmenter::new(
        Mode::Decompose(Penalty {
            kanji_penalty_length_threshold: 2,
            kanji_penalty_length_penalty: 3000,
            other_penalty_length_threshold: 2,
            other_penalty_length_penalty: 3000,
        }),
        dictionary,
        None,
    );
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
                let parts: Vec<&str> = tokens
                    .iter_mut()
                    .filter_map(|t| {
                        let tag = t.get("part_of_speech_tag").unwrap_or("*");
                        is_content(tag).then_some(t.surface.as_ref())
                    })
                    .collect();
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
