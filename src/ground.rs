//! Model-free grounding checks, ported from bench/harness.mjs.
//!
//! These are the load-bearing safety net: they took refusal on a mismatched context from
//! 4/6 to 6/6 on Qwen3.5-0.8B, and they hold at 6/6 on LFM2.5-230M, which refuses
//! *nothing* unaided (F26, F27, F44, F45, F46). Every rule here exists because a specific
//! fabrication got past the previous version.
//!
//! Rust's `regex` has no lookbehind, so `(?<![\w-])` is done by inspecting the characters
//! around each match instead — see `bounded`.

use regex::Regex;
use std::sync::LazyLock;

macro_rules! re {
    ($name:ident = $src:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($src).expect("bad regex"));
    };
}

re!(CITE = r"(?i)\[\s*([0-9]+|edit|citation needed|note [0-9]+)\s*\]");
re!(WS = r"\s+");
re!(SCRIPT = r"(?s)<script.*?</script>");
re!(STYLE = r"(?s)<style.*?</style>");
re!(TAG = r"<[^>]+>");
re!(ENT_NUM = r"&#([0-9]+);");
re!(FENCE = r"(?s)```[A-Za-z0-9]*\s*(.*?)```");
re!(INLINE = "`([^`\n]+)`");
re!(PROMPT = r"(?m)^[ \t]*[$#][ \t]+(.+)$");
re!(LEAD = r"^[$#]\s*");
re!(SPLITTOK = r"[\s|;]+");
re!(CMD_SHAPE = r"^[A-Za-z0-9_.-]+$");
re!(VOCAB_SHAPE = r"^[a-z][A-Za-z0-9_.-]*$");
re!(CODE_TAG = r"(?is)<(?:code|kbd|pre)[^>]*>(.*?)</(?:code|kbd|pre)>");
re!(A_TAG = r"(?s)<a[^>]*>([^<]{2,40})</a>");
re!(CMP_WORDS = r"(?i)\b(versus|vs\.?|difference between)\b");
re!(CMP_SPLIT = r"(?i)\s+(?:versus|vs\.?|or)\s+|\s*\bdifference between\s+|\s+and\s+");
re!(SCAFFOLD = r"(?i)^\s*(?:what(?:'s| is)? the |how do i |should i |which |what )?");
re!(NONALNUM = r"[^a-z0-9]+");
re!(QBACK = r"^[^.!]*\?\s*$");
re!(NUM = r"[0-9][0-9.,]*");
re!(FLAG = r"--?[A-Za-z0-9_-]{2,}");
re!(IDENT = r"[A-Za-z0-9_.]*(?:::|_|/)[A-Za-z0-9_./:-]+|\b[a-z]+[A-Z][A-Za-z0-9_]+");
re!(NOTFOUND = r"(?i)not found");
re!(THINK = r"(?s)<think>.*?</think>");
// F45: "how many" is not "how to"; "why did this fail" may legitimately be prose.
re!(HOWTO = r"(?i)^(?:how (?:do|can|would) i|how to|what command)\b|^(?:create|set|mount|encrypt|generate|check|list|install|enable|configure|disable|remove|start|stop|make)\b");

fn is_wordish(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// `(?<![\w-])needle(?![\w-])` — both boundaries. `du` must not be satisfied by
/// "produce", and a 2-char minimum would drop du/df/ls/ip, the commands asked about most.
pub fn word_in(hay: &str, needle: &str) -> bool {
    bounded(hay, needle, true)
}

/// `(?<![\w-])needle` — left boundary only, matching F38's comparison-side check, which
/// deliberately allows suffixes ("netctl" satisfied by "netctld").
pub fn word_start_in(hay: &str, needle: &str) -> bool {
    bounded(hay, needle, false)
}

fn bounded(hay: &str, needle: &str, right: bool) -> bool {
    if needle.is_empty() {
        return false;
    }
    hay.match_indices(needle).any(|(at, m)| {
        let left = hay[..at].chars().next_back().map_or(true, |c| !is_wordish(c));
        let end = at + m.len();
        let r = !right || hay[end..].chars().next().map_or(true, |c| !is_wordish(c));
        left && r
    })
}

/// F8: a 350M model cannot tell a citation marker from a datum.
pub fn denoise(s: &str) -> String {
    let s = CITE.replace_all(s, " ");
    WS.replace_all(&s, " ").trim().to_string()
}

pub fn html2txt(h: &str) -> String {
    let s = SCRIPT.replace_all(h, " ");
    let s = STYLE.replace_all(&s, " ");
    let s = TAG.replace_all(&s, " ");
    let s = s
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");
    let s = ENT_NUM.replace_all(&s, |c: &regex::Captures| {
        c[1].parse::<u32>().ok().and_then(char::from_u32).map(String::from).unwrap_or_default()
    });
    // last: &amp; must not resurrect an entity decoded above
    denoise(&s.replace("&amp;", "&"))
}

fn first_token(s: &str) -> Option<&str> {
    SPLITTOK.split(s).find(|t| !t.is_empty())
}

/// A command reaches us three ways: inline `cmd`, a fenced block, or a prompt line
/// ("# timedatectl set-timezone"). Missing the third made the echo rule misfire on a
/// correct answer — 1 false reject in 18 samples before it was widened. Paths are not
/// commands: an answer quoting `/home/username/.ssh/id_ed25519` was falsely rejected
/// because the reference spells that example differently.
pub fn commands_in(text: &str) -> Vec<String> {
    let mut blocks: Vec<&str> = Vec::new();
    for c in FENCE.captures_iter(text) {
        blocks.push(c.get(1).map_or("", |m| m.as_str()));
    }
    for c in INLINE.captures_iter(text) {
        blocks.push(c.get(1).map_or("", |m| m.as_str()));
    }
    for c in PROMPT.captures_iter(text) {
        blocks.push(c.get(1).map_or("", |m| m.as_str()));
    }
    let mut out: Vec<String> = Vec::new();
    for block in blocks {
        for line in block.split('\n') {
            let line = LEAD.replace(line.trim(), "");
            let Some(first) = first_token(&line) else { continue };
            if first.len() >= 2 && CMD_SHAPE.is_match(first) && !out.iter().any(|c| c == first) {
                out.push(first.to_string());
            }
        }
    }
    out
}

/// The reference marks its own commands up; `html2txt` throws that away, so pull the
/// vocabulary from raw HTML. `<code>` alone is not enough — Core_utilities has six code
/// tokens (rm mv cp arch kill ln) and names ncdu/gdu/dust only as wiki *links*, which
/// false-rejected a correct "du alternatives include ncdu, gdu…" answer (F45).
pub fn command_vocab(html: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |s: &str, out: &mut Vec<String>| {
        let s = s.trim();
        if s.len() >= 2 && VOCAB_SHAPE.is_match(s) && !out.iter().any(|c| c == s) {
            out.push(s.to_string());
        }
    };
    for c in CODE_TAG.captures_iter(html) {
        let txt = html2txt(c.get(1).map_or("", |m| m.as_str()));
        let txt = LEAD.replace(txt.trim(), "");
        if let Some(first) = first_token(&txt) {
            push(first, &mut out);
        }
    }
    for c in A_TAG.captures_iter(html) {
        let t = c.get(1).map_or("", |m| m.as_str()).to_string();
        push(&t, &mut out);
    }
    out
}

/// F35/F37: "versus | vs | difference between … and". Both sides come from the question's
/// own grammar, so this is model-free. Must run BEFORE `prep`, which strips the very words
/// the split needs. Fires 6/6 on comparisons, 0/26 on normal questions.
pub fn split_compare(q: &str) -> Option<(String, String)> {
    let ql = q.trim();
    if !CMP_WORDS.is_match(ql) {
        return None;
    }
    let cleaned = SCAFFOLD.replace(ql, "");
    let parts: Vec<&str> = CMP_SPLIT.split(&cleaned).filter(|s| !s.trim().is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    // carry the shared tail: "bash versus zsh startup files" -> both sides get it, or the
    // bare left side retrieves a shell index page instead of the topic (F37).
    let left = parts[0].trim();
    let right_full = parts[1].trim().trim_end_matches('?').trim();
    let words: Vec<&str> = right_full.split_whitespace().collect();
    let (right, tail) = if words.len() > 1 {
        (words[0], words[1..].join(" "))
    } else {
        (right_full, String::new())
    };
    let join = |s: &str| if tail.is_empty() { s.to_string() } else { format!("{s} {tail}") };
    Some((join(left), join(right)))
}

/// F27: returns "" when grounded, else the reason.
///
/// `reference` is the whole fetched article (F32: the 1.5 KB slice rejected a correct
/// answer for citing `cryptsetup` from a neighbouring section). `seen` is the slice
/// actually shown, because a claim about the *other* side of a comparison must be
/// supported by what was shown (F38).
pub fn ungrounded(answer: &str, reference: &str, q: &str, seen: &str) -> String {
    let stripped = strip_think(answer);
    let a = stripped.trim();
    if a.is_empty() {
        return "empty".into();
    }
    let ref_l = reference.to_lowercase();
    let cmds = commands_in(a);
    let absent: Vec<String> = cmds
        .iter()
        .filter(|c| !word_in(&ref_l, &c.to_lowercase()))
        .cloned()
        .collect();

    // F38: a comparison answer may assert about a tool whose article was never retrieved
    // ("chrony is the recommended alternative", chrony absent from the slice). No command
    // appears, so the command rule below cannot see it.
    if let Some((sa, sb)) = split_compare(q) {
        let al = a.to_lowercase();
        let seen_l = seen.to_lowercase();
        let unsupported: Vec<String> = [sa, sb]
            .into_iter()
            .filter(|s| word_start_in(&al, &s.to_lowercase()))
            .filter(|s| !word_start_in(&seen_l, &s.to_lowercase()))
            .collect();
        if !unsupported.is_empty() {
            return format!("asserts about {}, absent from what was shown", unsupported.join(", "));
        }
    }
    if !absent.is_empty() {
        return format!("command not in reference: {}", absent.join(", "));
    }
    if cmds.is_empty() {
        // Narrow on purpose: a word-overlap ratio caught one real echo and produced two
        // false rejects, and a false reject is worse — it turns a correct answer into
        // "not found".
        let norm = |s: &str| NONALNUM.replace_all(&s.to_lowercase(), " ").trim().to_string();
        if norm(q).contains(&norm(a)) {
            return "restates the question".into();
        }
        // observed degeneration variant: the model asks a question back instead of answering
        if QBACK.is_match(a) {
            return "asks a question back".into();
        }
    }
    String::new()
}

/// F44: "headline right, elaboration invented" — a fabricated release date, an invented
/// `--keyring` flag. Every multi-digit number and code-shaped identifier must appear in
/// the reference. Single digits are exempt: usually enumeration ("three files").
pub fn ungrounded_detail(answer: &str, reference: &str) -> String {
    let a = strip_think(answer);
    let ref_l = reference.to_lowercase();
    let has = |s: &str| ref_l.contains(&s.to_lowercase());
    // F108: separators were stripped from the answer's number and never from the reference, so
    // a reference "5,895 m" rejected a correct "5895" — the most likely benign transformation a
    // model makes, punished as fabrication. Normalise both sides, once.
    let strip = |s: &str| s.replace([',', '.', ' '], "");
    let ref_n = strip(&ref_l);

    let mut nums: Vec<String> = Vec::new();
    for m in NUM.find_iter(&a) {
        let s = m.as_str().trim_end_matches(['.', ',']).to_string();
        if s.chars().filter(char::is_ascii_digit).count() >= 2 && !nums.contains(&s) {
            nums.push(s);
        }
    }
    let bad: Vec<String> = nums
        .into_iter()
        .filter(|x| !has(x) && !ref_n.contains(&strip(x)))
        .collect();
    if !bad.is_empty() {
        return format!("number not in reference: {}", bad.join(", "));
    }

    let mut ids: Vec<String> = Vec::new();
    // The flag pattern needs a left boundary: without it "self-contained" yielded the
    // fragment "-contained" and false-rejected a correct answer. No lookbehind in `regex`,
    // so the boundary is checked on the character before the match.
    for m in FLAG.find_iter(&a) {
        if a[..m.start()].chars().next_back().map_or(true, |c| !is_wordish(c)) {
            ids.push(m.as_str().to_string());
        }
    }
    for m in IDENT.find_iter(&a) {
        ids.push(m.as_str().to_string());
    }
    let mut uniq: Vec<String> = Vec::new();
    for id in ids {
        let id = id.trim_end_matches(['.', ',', ')']).to_string();
        if id.len() >= 3 && !id.starts_with("http") && !uniq.contains(&id) {
            uniq.push(id);
        }
    }
    let bad: Vec<String> = uniq.into_iter().filter(|x| !has(x)).collect();
    if !bad.is_empty() {
        return format!("identifier not in reference: {}", bad.join(", "));
    }
    String::new()
}

/// F45: the commandless-prose blind spot. F27 only inspects commands, so an answer with
/// none is invisible to it — exactly how Q4_K_M evaded it (F43), answering "open your file
/// explorer or command prompt and navigate to the C…".
///
/// Judged against the reference's own command vocabulary, not markup: the first version
/// wanted marked-up commands and false-rejected 3/3 of LFM2.5-350M's correct answers,
/// which write "Use timedatectl set-timezone" bare (F45, F46).
pub fn ungrounded_shape(answer: &str, q: &str, ref_cmds: &[String]) -> String {
    let stripped = strip_think(answer);
    let a = stripped.trim();
    if a.is_empty() || NOTFOUND.is_match(a) {
        return String::new(); // empty and refusal are F27's business
    }
    if !HOWTO.is_match(q.trim()) {
        return String::new();
    }
    if ref_cmds.is_empty() {
        return String::new();
    }
    // F111: a backticked token was accepted as evidence on its own, so "you must use the
    // `enable` command" passed — `enable` is not a command, and the reference documents
    // `systemctl`. The token the answer proposes has to be one the reference demonstrates,
    // whether it arrives marked up or bare.
    let al = a.to_lowercase();
    if ref_cmds.iter().any(|c| word_in(&al, &c.to_lowercase())) {
        return String::new();
    }
    "how-to answer names no command from the reference".into()
}

fn strip_think(s: &str) -> String {
    THINK.replace_all(s, "").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from bench/harness.mjs `ground` — the pure self-test: no servers, no network.
    // Every case is a fabrication that got past an earlier version of a rule, or a correct
    // answer an earlier version wrongly rejected.

    #[test]
    fn f27_command_and_echo_rules() {
        let tz = "Set system clock. Use timedatectl set-timezone to change the timezone.";
        let swap = "Create a swap file using `mkswap` then activate with `swapon /swapfile`.";
        let du = "Use `du -h` to see what is using disk space, or `ncdu`.";

        // must catch
        assert!(!ungrounded("Use `mkfs.ext4 /dev/sdb1`.", swap, "create a swap file", swap).is_empty());
        assert_eq!(ungrounded("create a swap file", swap, "create a swap file", swap), "restates the question");
        assert_eq!(ungrounded("", swap, "create a swap file", swap), "empty");
        assert_eq!(ungrounded("Which filesystem do you want?", swap, "create a swap file", swap), "asks a question back");
        // `du` must not be satisfied by "produce"
        assert!(!ungrounded("Use `du -h`.", "This will produce output.", "check disk", "This will produce output.").is_empty());

        // must allow
        assert_eq!(ungrounded("Run `mkswap /swapfile` then `swapon /swapfile`.", swap, "create a swap file", swap), "");
        assert_eq!(ungrounded("Use timedatectl set-timezone to set it.", tz, "set the system timezone", tz), "");
        assert_eq!(ungrounded("# timedatectl set-timezone Europe/London", tz, "set the system timezone", tz), "");
        assert_eq!(ungrounded("Use `du -h` for that.", du, "check what is using disk space", du), "");
    }

    #[test]
    fn f38_comparison_side_rule() {
        let q = "what is the difference between netctl and NetworkManager";
        let shown = "netctl is a systemd-based network manager. See also NetworkManager.";
        // answering only the retrieved side is fine
        assert_eq!(ungrounded("netctl is profile-based.", shown, q, shown), "");
        // both sides present in the reference is fine
        assert_eq!(ungrounded("netctl is profile-based; NetworkManager is dynamic.", shown, q, shown), "");
        // asserting about a side absent from what was shown is not
        let narrow = "netctl is profile-based.";
        assert!(!ungrounded("NetworkManager is better for laptops.", narrow, q, narrow).is_empty());
    }

    #[test]
    fn f44_detail_rule() {
        let ver = "The current stable version is 1.97.1, released on 2026-08-12.";
        assert!(ungrounded_detail("The current stable version is 4.0.0, released on 2026-07-16.", ver)
            .starts_with("number not in reference"));
        let enc = "Encrypt a partition using cryptsetup luksFormat /dev/sdX.";
        assert!(ungrounded_detail("Encrypt with `cryptsetup --keyring /dev/mapper/x`.", enc)
            .starts_with("identifier not in reference"));
        // hyphenated prose is not a flag: "self-contained" must not yield "-contained"
        let ssh = "ssh-keygen -t ed25519 creates a key pair.";
        assert_eq!(ungrounded_detail("Use `ssh-keygen` with `-t ed25519` for a secure, self-contained key pair.", ssh), "");
        // real numbers pass
        let df = "/dev/mapper/root 220G 209G 1.2G 99% /";
        assert_eq!(ungrounded_detail("The root filesystem is 99% full, with only 1.2G available.", df), "");
    }

    #[test]
    fn f45_shape_rule() {
        let vocab = vec!["timedatectl".to_string()];
        // how-to with no command from the reference
        assert!(!ungrounded_shape(
            "You can change it from your desktop environment's date and time settings panel.",
            "how do I set the system timezone",
            &vocab
        )
        .is_empty());
        // bare command counts: no backticks needed (350M writes prose)
        assert_eq!(ungrounded_shape("Use timedatectl set-timezone to change it.", "how do I set the system timezone", &vocab), "");
        // vocabulary from links, not just <code>
        let du_vocab: Vec<String> = ["du", "ncdu", "gdu", "dust"].iter().map(|s| s.to_string()).collect();
        assert_eq!(ungrounded_shape("du alternatives include dust, gdu, and ncdu.", "check what is using disk space", &du_vocab), "");
        // refusal is exempt
        assert_eq!(ungrounded_shape("not found", "how do I set the system timezone", &vocab), "");
        // "how many" is not "how to"
        assert_eq!(ungrounded_shape("It has 302 neurons.", "how many neurons does C. elegans have", &vocab), "");
        // diagnosis may be prose
        assert_eq!(ungrounded_shape("The disk is full.", "why did this fail", &vocab), "");
    }

    #[test]
    fn vocab_reads_code_and_links() {
        let html = r#"<p><code>rm</code> and <code>mv</code>, see <a href="/x">ncdu</a> or <a href="/y">gdu</a>.</p>"#;
        let v = command_vocab(html);
        for want in ["rm", "mv", "ncdu", "gdu"] {
            assert!(v.iter().any(|c| c == want), "missing {want} in {v:?}");
        }
    }

    #[test]
    fn compare_split_carries_shared_tail() {
        let (a, b) = split_compare("bash versus zsh startup files").expect("should split");
        assert_eq!(a, "bash startup files");
        assert_eq!(b, "zsh startup files");
        assert!(split_compare("how do I create a swap file").is_none());
    }
}
