//! tny — a small-model terminal answerer.
//!
//! `tny "how do I create a swap file"` → grounded answer on stdout, source on stderr.
//!
//! The design is measured, not guessed: see NOTES.md (47 findings) and PLAN.md. The two
//! load-bearing results are that the model must never *select* anything — every choice it
//! was given lost to a deterministic rule — and that a regex grounding check, not a bigger
//! model, is what stops a fabricated shell command being served as fact.

mod corpus;
mod ground;
mod retrieve;

use ground::{command_vocab, html2txt, split_compare, ungrounded, ungrounded_detail, ungrounded_shape};
use retrieve::{article, pick_sections, prep, rank_articles, search_union, select_terms};
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// F19/F20/F47: thinking OFF is mandatory (Qwen opens `<think>` and never closes it inside
// 512 tokens, burning 95 s for zero answer). The second sentence is measured too: it took
// model-alone refusal on a mismatched context from 4/6 to 6/6 while cutting output from 31
// to 29 tokens. Command-only variants ("reply with that command and nothing else") lost a
// correct answer, 5/6.
const SYS: &str = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies. Use only facts written in the reference. Never add a flag, option, version, or path that does not appear there.";

const CHAT_PORT: u16 = 8080;
const KIWIX_PORT: u16 = 8082;
// F26/F43/F46: 0.8B is the floor. 350M degenerates, 230M refuses nothing unaided, 2B costs
// 2.2× for the same 6/6, and Q4_K_M halves RAM but breaks the grounding check's recovery.
const MODEL: &str = "ggml-org/Qwen3.5-0.8B-GGUF:Q8_0";
const MODEL_DIR: &str = "models--ggml-org--Qwen3.5-0.8B-GGUF";
// F31: lexical section selection needs top-5 × 600 chars for 14/14. F80: also the prompt
// size, and prefill is 19–22 s of a 20–40 s answer — so these are latency constants as much
// as accuracy ones. Overridable so a sweep costs a run rather than a rebuild.
fn top_sections() -> usize {
    std::env::var("TNY_TOP_SECTIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(5)
}
fn per_section() -> usize {
    std::env::var("TNY_PER_SECTION").ok().and_then(|v| v.parse().ok()).unwrap_or(600)
}
// F58: the answer is in the top-1 article for 36/58 cases and in the top-3 for 45/58.
// F82: overridable, to measure what a cheap first pass over one article would reach.
fn top_articles() -> usize {
    std::env::var("TNY_TOP_ARTICLES").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}
// F63: hits per book. Deeper costs nothing — one request per book either way, just a longer
// response — and it lifts the recall ceiling from 54/58 to 55/58 (`Hippocampus` answers
// "how does the brain consolidate long term memory" from its book's 6th hit). 12 adds nothing
// further, and neither 8 nor 12 disturbs the ranking: article@1 and answer@3 are unmoved, so
// the extra candidates are inert rather than noise.
const PER_BOOK: usize = 8;
/// F68: how many terms reach kiwix. Measured on one book: 8 terms returned the most hits of
/// any length tried, and 24 returned none. Overridable so a sweep costs a run rather than a
/// rebuild — `TNY_SEARCH_TERMS=999` reproduces the unlimited query this replaced.
fn search_terms() -> usize {
    std::env::var("TNY_SEARCH_TERMS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

const NEED_LLAMA: &str = "llama-server not on PATH. Install llama.cpp:\n  \
    arch:   pacman -S llama.cpp\n  \
    nix:    nix-shell -p llama-cpp\n  \
    other:  prebuilt binaries at github.com/ggml-org/llama.cpp/releases";
const NEED_KIWIX: &str = "kiwix-serve not on PATH. Install kiwix-tools:\n  \
    debian: apt install kiwix-tools\n  \
    fedora: dnf install kiwix-tools\n  \
    arch:   pacman -S kiwix-tools\n  \
    nix:    nix-shell -p kiwix-tools\n  \
    other:  download.kiwix.org/release/kiwix-tools";

struct Cfg {
    chat: String,
    kiwix: String,
    zim: PathBuf,
    models: PathBuf,
    cache: PathBuf,
    verbose: bool,
    /// Retrieval only: print the shortlist and stop, so ranking is measurable without
    /// paying 21 s of generation per case (bench/rank-cli.mjs).
    rank_only: bool,
    /// Dump every retrieved candidate as JSON and stop, for offline scorer sweeps.
    dump: bool,
    /// Print the exact text handed to the model and stop, without loading it. Answers the
    /// only question that separates a retrieval failure from a model failure (F67).
    context: bool,
    /// Skip the answer cache for this question and replace what is in it.
    fresh: bool,
}

fn main() {
    let mut question = String::new();
    let mut verbose = false;
    let mut rank_only = false;
    let mut dump = false;
    let mut context = false;
    let mut fresh = false;
    let mut follow = false;
    let mut corpus_args: Option<Vec<String>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-v" | "--verbose" => verbose = true,
            "--rank" => rank_only = true,
            "--dump" => dump = true,
            "--context" => context = true,
            "--fresh" => fresh = true,
            "-f" | "--follow" => follow = true,
            "--corpus" => corpus_args = Some(args.by_ref().collect()),
            "-h" | "--help" => {
                usage();
                return;
            }
            _ if a.starts_with('-') => {
                eprintln!("tny: unknown flag {a}");
                std::process::exit(1);
            }
            _ => {
                if !question.is_empty() {
                    question.push(' ');
                }
                question.push_str(&a);
            }
        }
    }

    let cfg = match config(verbose, rank_only, dump, context, fresh) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tny: {e}");
            std::process::exit(1);
        }
    };

    let r = match corpus_args {
        Some(sub) => corpus_cmd(&cfg, &sub),
        None if question.trim().is_empty() => {
            usage();
            std::process::exit(1);
        }
        None => run(&cfg, &question, follow),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("tny: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() {
    eprintln!(
        "tny \"question\"               grounded answer from local ZIM corpora\n\
         \n\
           -f, --follow               treat this as a follow-up to the last question\n\
               --fresh                re-answer instead of reusing the cached answer\n\
           -v, --verbose              per-stage timings on stderr\n\
         \n\
         tny --corpus list            mounted ZIM files\n\
         tny --corpus search <text>   find ZIMs in the kiwix library\n\
         tny --corpus pack [name]     download a whole shelf: mini small medium large huge\n\
         tny --corpus add <name>      download a ZIM (resumable, byte-verified)\n\
         tny --corpus update          check the library for newer editions\n\
         \n\
         needs llama-server and kiwix-serve on PATH\n\
         env: TNY_ZIM, TNY_MODELS, TNY_CHAT, TNY_KIWIX"
    );
}

fn config(verbose: bool, rank_only: bool, dump: bool, context: bool, fresh: bool) -> Result<Cfg, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME is not set")?;
    let xdg_data = std::env::var("XDG_DATA_HOME").unwrap_or(format!("{home}/.local/share"));
    let xdg_cache = std::env::var("XDG_CACHE_HOME").unwrap_or(format!("{home}/.cache"));
    // F77: the corpus location must not depend on where the user is standing. This used to
    // prefer `./zim` and `./models` when they existed, so the same question answered from a
    // checked-out repo and from $HOME searched different libraries — "no local corpus matched
    // (2 mounted)" in one directory and a correct answer in another. One fixed place, and
    // TNY_ZIM/TNY_MODELS for the cases that genuinely need another (the benchmarks, a corpus
    // on external disk). ZIMs run to gigabytes and models to hundreds of megabytes, so they
    // live under XDG *data*, not config; `~/.cache/tny` keeps only what is regenerable.
    let cache = PathBuf::from(format!("{xdg_cache}/tny"));
    std::fs::create_dir_all(&cache).map_err(|e| format!("cannot create {}: {e}", cache.display()))?;
    Ok(Cfg {
        chat: std::env::var("TNY_CHAT").unwrap_or(format!("http://127.0.0.1:{CHAT_PORT}")),
        kiwix: std::env::var("TNY_KIWIX").unwrap_or(format!("http://127.0.0.1:{KIWIX_PORT}")),
        zim: env_path("TNY_ZIM", format!("{xdg_data}/tny/zim")),
        models: env_path("TNY_MODELS", format!("{xdg_data}/tny/models")),
        cache,
        verbose,
        rank_only,
        dump,
        context,
        fresh,
    })
}

fn env_path(var: &str, fallback: String) -> PathBuf {
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(fallback))
}

// ------------------------------------------------------------------ corpus commands

fn corpus_cmd(cfg: &Cfg, args: &[String]) -> Result<i32, String> {
    match args.first().map(String::as_str).unwrap_or("list") {
        "list" => {
            let local = corpus::local(&cfg.zim);
            if local.is_empty() {
                eprintln!("tny: no ZIMs in {} — try: tny --corpus search bash", cfg.zim.display());
                return Ok(3);
            }
            for stem in local {
                println!("{stem}");
            }
            Ok(0)
        }
        "search" => {
            let q = args[1..].join(" ");
            if q.trim().is_empty() {
                return Err("usage: tny --corpus search <text>".into());
            }
            let entries = match corpus::cached(&cfg.cache) {
                Some(e) => e,
                None => corpus::fetch(&cfg.cache)?,
            };
            let hits = corpus::suggest(&entries, &q, 10);
            if hits.is_empty() {
                eprintln!("tny: nothing in the library matched {q:?}");
                return Ok(3);
            }
            for e in hits {
                println!("{:<34} {:>9}  {:>8} arts  {}", e.name, e.size_human(), e.articles, e.title);
            }
            Ok(0)
        }
        "add" => {
            let name = args.get(1).ok_or("usage: tny --corpus add <name>")?;
            corpus::add(&cfg.zim, &cfg.cache, name)?;
            // kiwix-serve mounts its library once at startup, so a new ZIM needs a new
            // process; drop it and the next query picks up the wider library.
            remount(cfg);
            Ok(0)
        }
        "pack" => {
            let entries = match corpus::cached(&cfg.cache) {
                Some(e) => e,
                None => corpus::fetch(&cfg.cache)?,
            };
            let Some(name) = args.get(1) else {
                println!("tny --corpus pack <name>\n");
                for p in corpus::PACK_NAMES {
                    let (keys, what) = corpus::pack(p).expect("named pack exists");
                    let plan = corpus::pack_plan(&cfg.zim, &entries, &keys);
                    println!(
                        "  {:<7} {:>9} to fetch   {} books, {} already here\n          {}",
                        p,
                        human_bytes(plan.bytes),
                        keys.len(),
                        plan.present.len(),
                        what
                    );
                    if !plan.missing.is_empty() {
                        eprintln!("          not in the catalog: {}", plan.missing.join(", "));
                    }
                }
                return Ok(0);
            };
            let (keys, _) = corpus::pack(name)
                .ok_or_else(|| format!("no pack {name:?} — {}", corpus::PACK_NAMES.join(", ")))?;
            let plan = corpus::pack_plan(&cfg.zim, &entries, &keys);
            let want = plan.want;
            if !plan.missing.is_empty() {
                eprintln!("tny: not in the catalog, skipping: {}", plan.missing.join(", "));
            }
            if want.is_empty() {
                println!("pack {name} is complete — all {} books are mounted", plan.present.len());
                return Ok(0);
            }
            println!("pack {name}: {} to download, {}", want.len(), human_bytes(plan.bytes));
            for (n, b) in &want {
                println!("  {:<40} {:>9}", n, human_bytes(*b));
            }
            // A pack can be tens of gigabytes. Never start that without a keypress, and
            // never ask for one when nobody is watching (a script gets the plan and stops).
            if !std::io::stdin().is_terminal() {
                eprintln!("tny: not a terminal — re-run interactively to confirm");
                return Ok(0);
            }
            eprint!("download? [y/N] ");
            std::io::stderr().flush().ok();
            if !matches!(key(), Some(b'y') | Some(b'Y')) {
                eprintln!("no");
                return Ok(0);
            }
            eprintln!("yes");
            let mut failed = 0;
            for (i, (n, _)) in want.iter().enumerate() {
                eprintln!("\n[{}/{}] {n}", i + 1, want.len());
                if let Err(e) = corpus::add(&cfg.zim, &cfg.cache, n) {
                    eprintln!("tny: {n} failed — {e}");
                    failed += 1;
                }
            }
            remount(cfg);
            if failed > 0 {
                eprintln!("\n{failed} of {} failed — re-run to resume", want.len());
            }
            Ok(if failed > 0 { 3 } else { 0 })
        }
        "update" => {
            let entries = corpus::fetch(&cfg.cache)?;
            let stale = corpus::outdated(&cfg.zim, &entries);
            corpus::write_stale_note(&cfg.cache, &stale);
            if stale.is_empty() {
                println!("all {} corpora are current", corpus::local(&cfg.zim).len());
            } else {
                for (name, have, newest) in &stale {
                    println!("{name:<34} {have} → {newest}");
                }
                eprintln!("tny: refresh with: tny --corpus add <name>");
            }
            Ok(0)
        }
        other => Err(format!("unknown corpus command {other:?} — list, search, add, pack, update")),
    }
}

/// F88: `--corpus update` only ever warned, and only when the user thought to run it, so a
/// library quietly ages until someone notices an answer citing a two-year-old page. The check
/// comes to the user instead — after the answer, never before it, because nobody asked a
/// question in order to be told about downloads.
///
/// Weekly at most, and declining snoozes it for another week: a prompt that appears every
/// time is a prompt that gets answered without reading. Offline or catalog fetch failing is
/// not an error here — it silently counts as "checked" so a laptop on a train stays quiet.
const UPDATE_EVERY: u64 = 7 * 24 * 60 * 60;

fn maybe_offer_update(cfg: &Cfg) {
    let path = cfg.cache.join("update.json");
    let state: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let now = now_secs();
    if now.saturating_sub(state["checked"].as_u64().unwrap_or(0)) < UPDATE_EVERY {
        return;
    }
    let mark = |asked: bool| {
        let _ = std::fs::write(
            &path,
            serde_json::json!({ "checked": now, "asked": asked }).to_string(),
        );
    };
    let Ok(entries) = corpus::fetch(&cfg.cache) else {
        mark(false);
        return;
    };
    let stale = corpus::outdated(&cfg.zim, &entries);
    corpus::write_stale_note(&cfg.cache, &stale);
    mark(!stale.is_empty());
    if stale.is_empty() {
        return;
    }
    let bytes: u64 = stale
        .iter()
        .filter_map(|(k, _, d)| entries.iter().find(|e| e.key() == *k && e.date() == *d))
        .map(|e| e.bytes)
        .sum();
    eprintln!(
        "\n\x1b[2m  {} newer edition{} available ({}): {}\x1b[0m",
        stale.len(),
        if stale.len() == 1 { "" } else { "s" },
        human_bytes(bytes),
        stale.iter().map(|(k, _, _)| k.as_str()).collect::<Vec<_>>().join(", ")
    );
    eprint!("  update now? [y/N] ");
    std::io::stderr().flush().ok();
    if !matches!(key(), Some(b'y') | Some(b'Y')) {
        eprintln!("not now — asking again in a week");
        return;
    }
    eprintln!("yes");
    for (k, _, d) in &stale {
        if let Some(e) = entries.iter().find(|e| e.key() == *k && e.date() == *d) {
            if let Err(err) = corpus::add(&cfg.zim, &cfg.cache, &e.name) {
                eprintln!("tny: {} failed — {err}", e.name);
            }
        }
    }
    corpus::write_stale_note(&cfg.cache, &[]);
    remount(cfg);
}

fn human_bytes(b: u64) -> String {
    const U: [(&str, u64); 3] = [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)];
    for (unit, div) in U {
        if b >= div {
            return format!("{:.1} {unit}", b as f64 / div as f64);
        }
    }
    format!("{b} B")
}

// ------------------------------------------------------------------ the pipeline
/// A question takes 20–60 s on a CPU, and silence for that long reads as a hang. The spinner
/// is stderr-only and TTY-only, so pipes, the harness and `tny … > file` are untouched.
///
/// Streaming the answer as it generates would be better and cannot be done: F27/F44/F45 reject
/// ungrounded answers *after* reading them, and an answer that was printed cannot be
/// unprinted. Progress is the honest substitute — it reports the stage, never the content.
struct Spin {
    tx: Option<std::sync::mpsc::Sender<Option<String>>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Spin {
    fn start(on: bool, first: &str) -> Spin {
        if !on {
            return Spin { tx: None, join: None };
        }
        let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
        let mut label = first.to_string();
        let t0 = Instant::now();
        let join = std::thread::spawn(move || {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            for i in 0.. {
                match rx.recv_timeout(std::time::Duration::from_millis(90)) {
                    Ok(Some(next)) => label = next,
                    Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(_) => {}
                }
                eprint!("\r\x1b[2K\x1b[2m{} {label}  {:.0}s\x1b[0m", FRAMES[i % 10], t0.elapsed().as_secs_f64());
                let _ = std::io::stderr().flush();
            }
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        });
        Spin { tx: Some(tx), join: Some(join) }
    }

    fn say(&self, label: impl Into<String>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Some(label.into()));
        }
    }

    fn stop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(None);
        }
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// `wikipedia_en_top_nopic_2026-06` is a filename, not a source. Strip the language, flavour
/// and date a ZIM name carries so the line under an answer reads like a citation.
fn human_book(name: &str) -> String {
    let stem = name.split("_en_").next().unwrap_or(name);
    match stem {
        "wikipedia" => "Wikipedia".into(),
        "archlinux" => "Arch Wiki".into(),
        s if s.starts_with("devdocs") => {
            let topic = name.split("_en_").nth(1).unwrap_or("").split('_').next().unwrap_or("");
            format!("devdocs {topic}").trim().to_string()
        }
        s => s.trim_end_matches(".com").to_string(),
    }
}

/// An article the answer was built from, kept so the prompt can open it.
struct Source {
    book: String,
    path: String,
    title: String,
}

enum Next {
    Ask(String, bool),
    Open(usize),
    Again,
    Quit,
}

/// F75: a terminal answerer that exits after one answer makes the user retype `tny` and pay
/// the library and model start-up again for a question they already had in mind. The prompt
/// keeps the process — and both servers — warm, and costs nothing when stdin is not a
/// terminal, so pipes and the benchmark harness behave exactly as before.
fn run(cfg: &Cfg, question: &str, follow: bool) -> Result<i32, String> {
    let mut q = question.to_string();
    let mut follow = follow;
    loop {
        let (code, sources) = answer_once(cfg, &q, follow)?;
        let interactive = std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal()
            && !cfg.rank_only
            && !cfg.dump
            && !cfg.context;
        if !interactive {
            return Ok(code);
        }
        // After the answer, and only here: a check that delays an answer is a check that
        // gets disabled.
        maybe_offer_update(cfg);
        if sources.is_empty() {
            return Ok(code);
        }
        loop {
            match prompt(&sources)? {
                Next::Quit => return Ok(code),
                Next::Again => continue,
                Next::Open(i) => open_source(cfg, &sources, i),
                Next::Ask(text, f) => {
                    q = text;
                    follow = f;
                    break;
                }
            }
        }
    }
}

/// One keypress, not a line: `q` must quit on the key, not on the key plus Enter. `stty` is
/// POSIX and tny already shells out to two servers, so this costs no raw-mode dependency.
/// `-isig` rather than plain `-icanon` matters: with signals left on, a Ctrl-C during the read
/// would kill the process with echo still off and leave the user's shell unusable. Instead
/// Ctrl-C arrives as a byte we handle, and the terminal is always restored by the line below.
fn key() -> Option<u8> {
    let tty = |args: &[&str]| {
        Command::new("stty").args(args).stdin(Stdio::inherit()).output().ok()
    };
    let saved = tty(&["-g"])?;
    let saved = String::from_utf8_lossy(&saved.stdout).trim().to_string();
    tty(&["-icanon", "-echo", "-isig", "min", "1", "time", "0"])?;
    let mut b = [0u8; 1];
    let n = std::io::stdin().read(&mut b).unwrap_or(0);
    tty(&[&saved]);
    (n == 1).then_some(b[0])
}

fn prompt(sources: &[Source]) -> Result<Next, String> {
    let open = if sources.len() > 1 { format!("1-{} open", sources.len()) } else { "1 open".into() };
    eprint!("\x1b[2m  ▸ ask a follow-up · n new topic · {open} · q quit\x1b[0m\n> ");
    std::io::stderr().flush().ok();
    // No stty means no single-key mode; one answer and out beats a broken terminal.
    let Some(k) = key() else { return Ok(Next::Quit) };
    Ok(match k {
        b'q' | 3 | 4 | 27 => {
            eprintln!();
            Next::Quit
        }
        b'1'..=b'9' => {
            eprintln!();
            Next::Open((k - b'1') as usize)
        }
        b'\r' | b'\n' => Next::Again,
        // F29: a follow-up carries the previous turn into the retrieval query; `n` drops it,
        // for when the subject changes and the old question would only poison the search.
        b'n' => {
            eprint!("\x1b[2m new topic\x1b[0m\n> ");
            std::io::stderr().flush().ok();
            match line()? {
                Some(q) => Next::Ask(q, false),
                None => Next::Quit,
            }
        }
        // Any other key is the first character of a question. Raw mode ate it, so echo it and
        // read the rest cooked - Backspace and the shell's line editing then work as usual.
        c if c.is_ascii_graphic() || c == b' ' => {
            eprint!("{}", c as char);
            std::io::stderr().flush().ok();
            match line()? {
                Some(rest) => Next::Ask(format!("{}{rest}", c as char), true),
                None => Next::Quit,
            }
        }
        _ => Next::Again,
    })
}

/// The rest of a typed line. `None` is EOF, which is a quit rather than an error.
fn line() -> Result<Option<String>, String> {
    let mut s = String::new();
    if std::io::stdin().read_line(&mut s).map_err(|e| e.to_string())? == 0 {
        eprintln!();
        return Ok(None);
    }
    let s = s.trim().to_string();
    Ok(if s.is_empty() { None } else { Some(s) })
}

/// The corpora are already served over HTTP by kiwix-serve, so "open the source" is a URL the
/// browser can read — offline, from the same ZIM the answer came from.
fn open_source(cfg: &Cfg, sources: &[Source], i: usize) {
    let Some(s) = sources.get(i) else { return };
    let url = format!("{}/content/{}/{}", cfg.kiwix, s.book, s.path);
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    match Command::new(opener).arg(&url).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
        Ok(_) => eprintln!("\x1b[2m  opening {}\x1b[0m", s.title),
        Err(_) => eprintln!("  {url}"),
    }
}

/// One question, one answer. Returns the exit code and the articles the answer was built
/// from, so the prompt underneath can offer to open them.
fn answer_once(cfg: &Cfg, question: &str, follow: bool) -> Result<(i32, Vec<Source>), String> {
    let t_all = Instant::now();
    let quiet = cfg.rank_only || cfg.dump || cfg.context;
    let mut spin = Spin::start(std::io::stderr().is_terminal() && !quiet, "starting the library");
    serve_kiwix(cfg)?;

    let prev = if follow { recent_turns(cfg, 2) } else { Vec::new() };
    // F85: a repeat costs a file read instead of 40 s. The key includes the turn being
    // followed, so the same words after a different question are a different question.
    let key = cache_key(cfg, question, prev.last());
    if !cfg.fresh && !cfg.rank_only && !cfg.dump && !cfg.context {
        if let Some((answer, sources)) = cached(cfg, &key) {
            spin.stop();
            println!("{answer}");
            eprintln!("\n\x1b[2m  {}   cached\x1b[0m", cite_lines(&sources).join("\n  "));
            save_turn(cfg, question, &answer);
            return Ok((0, sources));
        }
    }
    // F29: the retrieval query is `<prev question> <this question>`. NEVER a model rewrite:
    // asked to rephrase, 0.8B inverted "how do I turn it off" into "how do I turn it back
    // on". Concatenation scored 5/6 against the rewrite's 4/6, and it is free. Only the
    // immediately previous question joins it — F84 widened the model's history, not this.
    let retrieval_q = match prev.last() {
        Some((q, _)) => format!("{q} {question}"),
        None => question.to_string(),
    };

    // F37 measured synthesis from two topics at 2/5 and made comparisons refuse: "ask about
    // one". But the failure it recorded was the model inventing *the side it was not shown* —
    // a context bug, not a reasoning limit, because retrieval returned one article for a
    // two-sided question. F86 shows both sides instead of refusing. The split is model-free,
    // both names coming from the question's own grammar, and it must run BEFORE `prep`, which
    // strips the very words it needs. Fires 6/6, silent on 26/26.
    let books = corpus::local(&cfg.zim);
    let compare = split_compare(&retrieval_q).and_then(|(a, b)| {
        let one = |s: &str| {
            let c = search_union(&cfg.kiwix, &prep(s), &books, PER_BOOK);
            rank_articles(s, &c).into_iter().next()
        };
        match (one(&a), one(&b)) {
            // Both sides landing on one article is not a comparison — "SIGTERM vs SIGKILL"
            // is answered by the signals page, and splitting it would read it twice.
            (Some(x), Some(y)) if x.title != y.title => Some(vec![x, y]),
            _ => None,
        }
    });

    let t = Instant::now();
    // F68: a sentence is not a search query. kiwix scores 0 hits at 24 terms, so the query is
    // cut to its most informative 8 and widened only if that finds nothing — the backoff is
    // free on every query that already works, and the two extra searches cost ~300 ms on the
    // ones that do not. Ranking still sees the *whole* question: only the search is cut.
    // F71: rarity was measured and rejected as a term-selection objective. Real document
    // frequency, summed per book over every mounted ZIM and cached on disk, ordered terms
    // correctly - `the` 2500, `uname` 556, `harina` 107 - and selecting the rarest eight
    // scored 37/67 against the shape heuristic's 63/67, reproduced on one server instance.
    // Xapian already weights rarity inside the engine, so pre-selecting rare terms counts it
    // twice and spends the query budget on incidentals; what the engine cannot recover is
    // topic coverage. That is the second time an IDF signal has lost here, and the first
    // (23/58 against 32/58) was blamed on biased statistics. The statistics were fine.
    let prepped = prep(&retrieval_q);
    let mut query = select_terms(&prepped, search_terms());
    spin.say(format!("searching {} corpora", books.len()));
    let mut cands = search_union(&cfg.kiwix, &query, &books, PER_BOOK);
    // F72: an empty shortlist means one of two very different things, and tny reported the
    // wrong one. kiwix-serve dies under sustained load on a small machine — measured here
    // repeatedly, SIGSEGV then SIGABRT with 4.5 GB of ZIMs and 400 MB of free RAM — after
    // which every book's search fails silently and the user is told "no local corpus matched",
    // about an answer sitting on their disk. Ask whether the server is alive before believing
    // its silence, and give it one restart.
    if cands.is_empty() && !up(&format!("{}/", cfg.kiwix)) {
        eprintln!("tny: kiwix-serve is not responding — restarting it");
        serve_kiwix(cfg)?;
        cands = search_union(&cfg.kiwix, &query, &books, PER_BOOK);
    }
    // F70: widening a thin shortlist was measured and rejected. Three of five held-out misses
    // had six candidates or fewer, so backing off at twelve looked obvious — and it scored
    // *worse*, 60/67 against 62/67, because the extra candidates outranked the answer. Recall
    // was never the constraint: every one of those pages was already inside its own book's
    // top 50. Ranking precision is the constraint, so the backoff stays empty-only.
    for cap in [5, 3] {
        if !cands.is_empty() {
            break;
        }
        query = select_terms(&prepped, cap);
        for c in search_union(&cfg.kiwix, &query, &books, PER_BOOK) {
            if !cands.iter().any(|x| x.book == c.book && x.path == c.path) {
                cands.push(c);
            }
        }
    }
    let t_search = t.elapsed();
    if cands.is_empty() {
        spin.stop();
        return Ok((no_local_match(cfg, &query), vec![]));
    }
    // A comparison replaces the shortlist with one article per side, so the model is shown
    // both things it is being asked to compare rather than one and its own imagination.
    let ranked = match compare {
        Some(pair) => pair,
        None => rank_articles(&retrieval_q, &cands),
    };
    // Retrieval is 2 % of a query's wall time, so measuring ranking through full generation
    // costs 80 s per case and hides the thing under test. `--rank` stops here and prints the
    // whole shortlist, because rank-1 alone cannot distinguish a scoring miss from a
    // candidate that was never retrieved (F49's actual lesson).
    if cfg.rank_only {
        for c in ranked.iter().take(8) {
            println!("{}\t{}\t{}", c.book, c.title, c.path);
        }
        return Ok((0, vec![]));
    }
    if cfg.dump {
        // Every candidate, unranked, as JSON: lets a scorer be tried offline in
        // milliseconds instead of a 100-second run against live kiwix per variant.
        let rows: Vec<serde_json::Value> = cands
            .iter()
            .map(|c| {
                serde_json::json!({
                    "book": c.book, "title": c.title, "path": c.path, "snip": c.snip,
                    "rank": c.rank, "kind": format!("{:?}", c.kind),
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&rows).unwrap_or_default());
        return Ok((0, vec![]));
    }
    // Only now is the model needed: everything above is retrieval. `--context` stops before
    // the load, so inspecting what the model was given costs a search and a fetch.
    if !cfg.context {
        spin.say("loading the model");
        serve_chat(cfg)?;
    }


    // F58: three articles, not one. Rank-1 carries the answer for 36 of 58 verified cases,
    // but the top *three* carry it for 45 — retrieval's misses are near-misses, and a
    // ranking gap of 9 cases is closed by fetching 2 more articles at 15-41 ms each rather
    // than by a better scorer (F53 closed content reranking; F56 exhausted lexical signals).
    // F39b measured the same shape at the section level: 3 articles x 1 section beat
    // 1 article x 3 sections, 5/6 vs 4/6.
    spin.say(format!("reading {}", ranked[0].title));
    let t = Instant::now();
    let mut docs: Vec<(&retrieve::Candidate, String)> = Vec::new();
    for c in ranked.iter().take(top_articles()) {
        match article(&cfg.kiwix, &c.book, &c.path) {
            Ok(html) => docs.push((c, html)),
            Err(e) if docs.is_empty() => return Err(e),
            Err(_) => {}
        }
    }
    let t_fetch = t.elapsed();

    // The budget is split, not multiplied: the same ~3 KB of context, sourced from three
    // articles. F41 measured that a bigger window does not buy accuracy — placement does.
    let per_doc = top_sections().div_ceil(docs.len().max(1));
    let mut parts = Vec::new();
    let mut heads = Vec::new();
    for (c, html) in &docs {
        let p = pick_sections(html, &retrieval_q, per_doc, per_section());
        if p.text.trim().is_empty() {
            continue;
        }
        // Name the source inline: with three articles in context the model must be able to
        // attribute, and the grounding check compares against the union below.
        parts.push(format!("## {}\n{}", c.title, p.text));
        heads.extend(p.heads);
    }
    let picked = retrieve::Picked { heads, text: parts.join("\n\n") };
    if cfg.context {
        println!("{}", picked.text);
        return Ok((0, vec![]));
    }

    // F32: grounding reads the whole article, not the slice sent to the model — the slice
    // rejected a correct answer for citing `cryptsetup` from a neighbouring section. With
    // three sources the reference is the union of all three, or a correct answer taken from
    // the second article would be rejected as ungrounded.
    let full = docs.iter().map(|(_, h)| html2txt(h)).collect::<Vec<_>>().join("\n");
    let vocab = docs.iter().flat_map(|(_, h)| command_vocab(h)).collect::<Vec<_>>();

    spin.say("answering");
    let t = Instant::now();
    let answer = ask(cfg, question, &picked.text, &prev)?;
    let t_gen = t.elapsed();

    // F27/F44/F45: three rules, each with its own reference. A false reject is the worst
    // outcome — it turns a correct answer into "not found" — so each was tuned against
    // correct answers as hard as against fabrications.
    let why = [
        ungrounded(&answer, &full, question, &picked.text),
        ungrounded_detail(&answer, &full),
        ungrounded_shape(&answer, question, &vocab),
    ]
    .into_iter()
    .find(|r| !r.is_empty())
    .unwrap_or_default();

    // F83: cite the articles the answer came from, not every article that was read. Three go
    // into context and "why is the sky blue" credited `Sky Blue Sky`, a Wilco album, because
    // it ranked first — the answer came from the other two. An article earns its line by
    // containing the answer's distinctive words: the same evidence the grounding check uses,
    // applied per article instead of to the union of all three.
    let sources: Vec<Source> = supporting(&answer, &docs)
        .into_iter()
        .map(|(c, _)| Source {
            book: c.book.clone(),
            path: c.path.clone(),
            title: c.title.clone(),
        })
        .collect();

    spin.stop();
    if cfg.verbose {
        eprintln!(
            "  search {} ms · fetch {} ms · generate {} ms · §{}",
            t_search.as_millis(),
            t_fetch.as_millis(),
            t_gen.as_millis(),
            picked.heads.join(", §")
        );
    }
    if let Some(note) = corpus::stale_note(&cfg.cache) {
        eprint!("{note}");
    }

    if !why.is_empty() {
        eprintln!("tny: rejected — {why}");
        // F57: a rejection means the mounted corpora did not carry this answer. That is the
        // one moment a download suggestion is certainly not noise.
        suggest_corpus(cfg, &query);
        println!("not found");
        return Ok((3, sources));
    }
    println!("{}", answer.trim());
    // F75: the source line was `wikipedia_en_top_nopic_2026-06 · Sky Blue Sky · §Release and
    // reception, §Composition, §Sky and sea, §Artificial blues, …` — a filename, a rank-1
    // article that was not where the answer came from, and six section names that mean
    // something only to whoever wrote the ranker. What a reader needs is which works were
    // consulted; sections are diagnostics and moved behind `-v`.
    // Numbered, because the prompt underneath opens them by number.
    eprintln!(
        "\n\x1b[2m  {}   {:.1}s\x1b[0m",
        cite_lines(&sources).join("\n  "),
        t_all.elapsed().as_secs_f64()
    );
    save_turn(cfg, question, &answer);
    cache_put(cfg, &key, answer.trim(), &sources);
    Ok((0, sources))
}

fn cite_lines(sources: &[Source]) -> Vec<String> {
    sources
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{} {} · {}", i + 1, human_book(&s.book), s.title))
        .collect()
}

/// Which of the read articles actually support this answer. Distinctive words only — short
/// and common ones match everywhere and would put every article back on the list. An article
/// is kept when it carries most of what the best one carries, so a fact stated in two
/// articles cites both, and the Wilco album cites nothing.
fn supporting<'a>(
    answer: &str,
    docs: &'a [(&'a retrieve::Candidate, String)],
) -> Vec<&'a (&'a retrieve::Candidate, String)> {
    let words: Vec<String> = answer
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 5)
        .map(str::to_string)
        .collect();
    if words.is_empty() || docs.len() < 2 {
        return docs.iter().collect();
    }
    let share = |html: &str| {
        let text = html2txt(html).to_lowercase();
        words.iter().filter(|w| text.contains(w.as_str())).count() as f64 / words.len() as f64
    };
    let scored: Vec<(f64, &(&retrieve::Candidate, String))> =
        docs.iter().map(|d| (share(&d.1), d)).collect();
    let best = scored.iter().map(|(s, _)| *s).fold(0.0, f64::max);
    // Nothing matched anywhere: the answer is odd, so name everything read rather than
    // silently citing nothing.
    if best <= 0.0 {
        return docs.iter().collect();
    }
    scored.into_iter().filter(|(s, _)| *s >= best * 0.8).map(|(_, d)| d).collect()
}

/// F40/F57: the catalog is the index — 1,286 English ZIMs, cached locally at 1.5 MB, so a
/// suggestion costs a file read and no network. A lexical match over catalog metadata names
/// the right ZIM for 8/8 queries the local corpus cannot answer, with 0 false suggestions on
/// 5 it can.
///
/// Two triggers, both a *measured* failure signal rather than a guess: zero retrieved
/// candidates, and a grounding rejection. The second was specified in F40 and never wired —
/// "not found" is precisely the moment the user needs to know which corpus would have held
/// the answer.
fn suggest_corpus(cfg: &Cfg, query: &str) {
    match corpus::cached(&cfg.cache) {
        Some(entries) => {
            let hits = corpus::suggest(&entries, query, 3);
            if !hits.is_empty() {
                eprintln!("     the library has:");
                for e in hits {
                    eprintln!("       {:<30} {:>9}  tny --corpus add {}", e.title, e.size_human(), e.name);
                }
            }
        }
        None => eprintln!("     tny --corpus search {query}   (fetches the library catalog)"),
    }
}

fn no_local_match(cfg: &Cfg, query: &str) -> i32 {
    println!("not found");
    eprintln!("tny: no local corpus matched {query:?} ({} mounted)", corpus::local(&cfg.zim).len());
    suggest_corpus(cfg, query);
    3
}

fn ask(cfg: &Cfg, question: &str, reference: &str, prev: &[(String, String)]) -> Result<String, String> {
    let mut messages = vec![serde_json::json!({ "role": "system", "content": SYS })];
    // F28: keep the prior turns in the message list. History carries the antecedent for
    // elliptical follow-ups ("how do I unlock *it* at boot") — 83 % vs 75 % stateless — and
    // it is cheaper than it looks, because turn 1's prefix is still in the KV cache.
    // F84: two exchanges rather than one, so a thread's third question can still see its
    // first. Not more: every turn is prefill, and prefill is 85 % of the answer.
    for (q, a) in prev {
        messages.push(serde_json::json!({ "role": "user", "content": q }));
        messages.push(serde_json::json!({ "role": "assistant", "content": a }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": format!("Reference:\n{reference}\n\nQuestion: {question}")
    }));

    let body = serde_json::json!({
        "messages": messages,
        "temperature": 0.1,
        "top_k": 50,
        "repeat_penalty": 1.05,
        "max_tokens": 160,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let resp = ureq::post(&format!("{}/v1/chat/completions", cfg.chat))
        .timeout(Duration::from_secs(300))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("chat request failed: {e}"))?
        .into_string()
        .map_err(|e| format!("chat body: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&resp).map_err(|e| format!("chat json: {e}"))?;
    let msg = &j["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    // F19: empty content with reasoning present is an ERROR, not an answer.
    if content.trim().is_empty() && !msg["reasoning_content"].as_str().unwrap_or("").is_empty() {
        return Err("model emitted reasoning only — thinking is not disabled".into());
    }
    Ok(content.to_string())
}

// ------------------------------------------------------------------ servers

fn on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
        .unwrap_or(false)
}

/// Reuse a live server, else spawn it. F25: `--no-mmproj` always — without it llama.cpp
/// downloads a vision projector nobody uses.
///
/// Split in two because retrieval does not need the model: `--rank` measures ranking on a
/// machine with no llama.cpp at all, and demanding it there cost a 15-minute benchmark run
/// that reported 0/58 for a missing PATH entry.
fn serve_kiwix(cfg: &Cfg) -> Result<(), String> {
    if !up(&format!("{}/", cfg.kiwix)) {
        let zims: Vec<PathBuf> = std::fs::read_dir(&cfg.zim)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "zim"))
            .collect();
        if zims.is_empty() {
            return Err(format!("no ZIM files in {} — try: tny --corpus search bash", cfg.zim.display()));
        }
        if !on_path("kiwix-serve") {
            return Err(NEED_KIWIX.into());
        }
        let mut cmd = Command::new("kiwix-serve");
        cmd.arg("--port").arg(KIWIX_PORT.to_string());
        for z in &zims {
            cmd.arg(z);
        }
        spawn(cmd, cfg, "kiwix")?;
        wait_up(&format!("{}/", cfg.kiwix), 120, "kiwix-serve")?;
    }
    Ok(())
}

fn serve_chat(cfg: &Cfg) -> Result<(), String> {

    if !up(&format!("{}/health", cfg.chat)) {
        if !on_path("llama-server") {
            return Err(NEED_LLAMA.into());
        }
        std::fs::create_dir_all(&cfg.models).map_err(|e| format!("cannot create {}: {e}", cfg.models.display()))?;
        // llama.cpp fetches `-hf` models into LLAMA_CACHE itself; say so, because a silent
        // 800 MB first run looks like a hang.
        if !cfg.models.join(MODEL_DIR).is_dir() {
            eprintln!("tny: downloading {MODEL} (~800 MB, once) into {}", cfg.models.display());
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut cmd = Command::new("llama-server");
        cmd.args(["-hf", MODEL, "--no-mmproj", "--jinja", "--host", "127.0.0.1"])
            .args(["-t", &threads.to_string(), "-c", "8192", "--port", &CHAT_PORT.to_string()])
            .env("LLAMA_CACHE", &cfg.models);
        spawn(cmd, cfg, "chat")?;
        wait_up(&format!("{}/health", cfg.chat), 1800, "llama-server")?;
    }
    Ok(())
}

fn spawn(mut cmd: Command, cfg: &Cfg, what: &str) -> Result<(), String> {
    let log = cfg.cache.join(format!("{what}.log"));
    let out = std::fs::File::create(&log).map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    let err = out.try_clone().map_err(|e| format!("log dup: {e}"))?;
    cmd.stdin(Stdio::null()).stdout(out).stderr(err);
    let child = cmd.spawn().map_err(|e| format!("cannot start {what}: {e}"))?;
    let _ = std::fs::write(cfg.cache.join(format!("{what}.pid")), child.id().to_string());
    Ok(())
}

fn remount(cfg: &Cfg) {
    let pidfile = cfg.cache.join("kiwix.pid");
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        if let Ok(pid) = pid.trim().parse::<u32>() {
            let _ = Command::new("kill").arg(pid.to_string()).status();
            let _ = std::fs::remove_file(&pidfile);
            eprintln!("tny: kiwix-serve will restart with the new corpus on the next query");
        }
    }
}

fn up(url: &str) -> bool {
    ureq::builder()
        .timeout(Duration::from_millis(1500))
        .build()
        .get(url)
        .call()
        .is_ok()
}

fn wait_up(url: &str, secs: u64, what: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if up(url) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    Err(format!("{what} did not come up within {secs}s — see the log in the tny cache dir"))
}

/// F84: the conversation is more than one turn deep. `last.json` kept exactly one exchange,
/// so the third question in a thread could not see the first. Five is a rolling window, not a
/// transcript: the retrieval query still concatenates only the previous question (F29, which
/// was measured), while the model gets the last two exchanges for pronouns and continuity.
fn recent_turns(cfg: &Cfg, n: usize) -> Vec<(String, String)> {
    let raw = std::fs::read_to_string(cfg.cache.join("turns.json")).unwrap_or_default();
    let j: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap_or_default();
    j.iter()
        .rev()
        .take(n)
        .rev()
        .filter_map(|t| Some((t["q"].as_str()?.to_string(), t["a"].as_str()?.to_string())))
        .collect()
}

fn save_turn(cfg: &Cfg, q: &str, a: &str) {
    let mut turns = recent_turns(cfg, 4);
    turns.push((q.to_string(), a.to_string()));
    let j: Vec<serde_json::Value> =
        turns.iter().map(|(q, a)| serde_json::json!({ "q": q, "a": a })).collect();
    let _ = std::fs::write(cfg.cache.join("turns.json"), serde_json::Value::Array(j).to_string());
}

/// F85: asking the same question twice cost 40 s twice. The corpora are static files and the
/// sampling is near-deterministic, so a repeat is a lookup. Keyed by the question and the
/// turn it follows, because a follow-up's answer depends on what came before; `--fresh`
/// bypasses it, and a corpus change invalidates it through the book list in the key.
fn cache_key(cfg: &Cfg, question: &str, prev: Option<&(String, String)>) -> String {
    let books = corpus::local(&cfg.zim).join(",");
    let prev_q = prev.map(|(q, _)| q.as_str()).unwrap_or("");
    format!("{}\u{0}{}\u{0}{}", question.trim().to_lowercase(), prev_q.to_lowercase(), books)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Thirty days — and not the main invalidation. Corpora carry their edition date in the
/// filename (`wikipedia_en_top_nopic_2026-06`), so a refresh lands as a *new* file and changes
/// the book list in the key; `--corpus update` only ever warns, it never replaces anything
/// behind the user's back. The TTL covers what the key cannot see: a file swapped by hand
/// under the same name, and unbounded growth in a cache nobody prunes.
const CACHE_TTL: u64 = 30 * 24 * 60 * 60;

fn cached(cfg: &Cfg, key: &str) -> Option<(String, Vec<Source>)> {
    let raw = std::fs::read_to_string(cfg.cache.join("answers.json")).ok()?;
    let j: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let hit = j.get(key)?;
    if now_secs().saturating_sub(hit["at"].as_u64().unwrap_or(0)) > CACHE_TTL {
        return None;
    }
    let sources = hit["s"]
        .as_array()?
        .iter()
        .filter_map(|s| {
            Some(Source {
                book: s["book"].as_str()?.to_string(),
                path: s["path"].as_str()?.to_string(),
                title: s["title"].as_str()?.to_string(),
            })
        })
        .collect();
    Some((hit["a"].as_str()?.to_string(), sources))
}

fn cache_put(cfg: &Cfg, key: &str, answer: &str, sources: &[Source]) {
    let path = cfg.cache.join("answers.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    let mut j: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).unwrap_or_default();
    let now = now_secs();
    // Expire on write: nothing else ever walks this file, so the sweep belongs where it is
    // already open. Then a cap, oldest first, for the pathological case of 200 questions in
    // one month.
    j.retain(|_, v| now.saturating_sub(v["at"].as_u64().unwrap_or(0)) <= CACHE_TTL);
    while j.len() >= 200 {
        let oldest = j
            .iter()
            .min_by_key(|(_, v)| v["at"].as_u64().unwrap_or(0))
            .map(|(k, _)| k.clone());
        match oldest {
            Some(k) => {
                j.remove(&k);
            }
            None => break,
        }
    }
    j.insert(
        key.to_string(),
        serde_json::json!({
            "a": answer,
            "at": now,
            "s": sources.iter().map(|s| serde_json::json!({
                "book": s.book, "path": s.path, "title": s.title
            })).collect::<Vec<_>>(),
        }),
    );
    let _ = std::fs::write(path, serde_json::Value::Object(j).to_string());
}
