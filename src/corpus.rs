//! ZIM corpus management: catalog, download, update detection.
//!
//! Measured constraints this file exists to honour:
//!
//! * **F21** — large downloads die mid-transfer. Every ZIM I fetched from the foreground
//!   during measurement was killed by an interruption, and a truncated 945 MB Qwen file
//!   produced garbage output that cost an hour of wrong conclusions. So: `Range` resume plus
//!   a byte-length check against the catalog, never a bare fetch.
//! * **F40** — the catalog is the index. A lexical match over catalog metadata suggests the
//!   right ZIM for 8/8 queries the local corpus cannot answer, and stays silent on all 5 it
//!   can. Long terms substring-match, short terms need a word boundary: dropping terms under
//!   four characters killed `css`, `git`, and `php`.
//! * **F11** — kiwix's `books.name` wants the *filename stem* (`devdocs_en_bash_2026-04`),
//!   while the catalog's `<name>` is undated (`devdocs_en_bash`). The date lives in the
//!   download filename, which is exactly what makes update detection a string compare.

use regex::Regex;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

macro_rules! re {
    ($name:ident = $src:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($src).expect("bad regex"));
    };
}

re!(E_NAME = r"(?s)<name>(.*?)</name>");
re!(E_TITLE = r"(?s)<title>(.*?)</title>");
re!(E_SUMMARY = r"(?s)<summary>(.*?)</summary>");
re!(E_TAGS = r"(?s)<tags>(.*?)</tags>");
re!(E_ARTS = r"(?s)<articleCount>(.*?)</articleCount>");
re!(E_ZIM = r#"(?s)<link[^>]*type="application/x-zim"[^>]*href="([^"]+)"[^>]*length="([0-9]+)""#);
re!(DATED = r"^(.*)_(\d{4}-\d{2})$");
re!(WORDS = r"[a-z0-9.]+");
// F40: docs-category ZIMs answer "how do I use X" questions; general encyclopaedias are the
// fallback tier. Restricting to docs and falling back otherwise is what reached 8/8.
re!(IS_DOCS = r"(?i)devdocs|readthedocs|docs\.|php\.net|_category:(?:other|stack_exchange)");

const CATALOG: &str = "https://library.kiwix.org/catalog/v2/entries?lang=eng&count=-1";

pub struct Entry {
    pub name: String,
    pub title: String,
    pub summary: String,
    pub tags: String,
    pub articles: u64,
    pub href: String,
    pub bytes: u64,
}

impl Entry {
    /// The dated filename stem, which is what kiwix-serve and `books.name` use (F11).
    pub fn stem(&self) -> String {
        self.href
            .rsplit('/')
            .next()
            .unwrap_or("")
            .trim_end_matches(".meta4")
            .trim_end_matches(".zim")
            .to_string()
    }
    pub fn date(&self) -> String {
        DATED
            .captures(&self.stem())
            .and_then(|c| c.get(2))
            .map_or(String::new(), |m| m.as_str().to_string())
    }
    /// The stem without its date — the only safe identity for a book.
    ///
    /// The catalog reuses `<name>` across flavours: `wikipedia_en_all` is simultaneously a
    /// 124 GB `maxi`, a 52.7 GB `nopic`, and a 12.5 GB `mini` edition. Matching on `<name>`
    /// alone would download an arbitrary one of the three, so every lookup and every update
    /// comparison keys on this instead.
    pub fn key(&self) -> String {
        let stem = self.stem();
        DATED.captures(&stem).and_then(|c| c.get(1)).map_or(stem.clone(), |m| m.as_str().to_string())
    }
    pub fn size_human(&self) -> String {
        let mb = self.bytes as f64 / 1e6;
        if mb >= 1000.0 {
            format!("{:.1} GB", mb / 1000.0)
        } else {
            format!("{mb:.0} MB")
        }
    }
}

fn unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'")
}

pub fn parse(xml: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for item in xml.split("<entry>").skip(1) {
        let grab = |re: &Regex| -> String {
            re.captures(item).and_then(|c| c.get(1)).map_or(String::new(), |m| unescape(m.as_str().trim()))
        };
        let Some(zim) = E_ZIM.captures(item) else { continue };
        let name = grab(&E_NAME);
        if name.is_empty() {
            continue;
        }
        out.push(Entry {
            name,
            title: grab(&E_TITLE),
            summary: grab(&E_SUMMARY),
            tags: grab(&E_TAGS),
            articles: grab(&E_ARTS).parse().unwrap_or(0),
            href: zim[1].to_string(),
            bytes: zim[2].parse().unwrap_or(0),
        });
    }
    out
}

fn cache_path(cache: &Path) -> PathBuf {
    cache.join("catalog-eng.xml")
}

/// Fetch and cache. 1,286 English ZIMs, ~2.8 TB if you took everything, so the catalog is
/// the only part worth keeping locally (~2 MB of XML).
pub fn fetch(cache: &Path) -> Result<Vec<Entry>, String> {
    eprintln!("tny: fetching kiwix catalog…");
    let xml = ureq::get(CATALOG)
        .set("User-Agent", "tny/0.1")
        .timeout(std::time::Duration::from_secs(120))
        .call()
        .map_err(|e| format!("catalog fetch failed: {e}"))?
        .into_string()
        .map_err(|e| format!("catalog body: {e}"))?;
    let entries = parse(&xml);
    if entries.is_empty() {
        return Err("catalog parsed to zero entries — format changed?".into());
    }
    let _ = std::fs::write(cache_path(cache), &xml);
    Ok(entries)
}

/// Read-only, offline-safe: used by the zero-hit suggestion so a query never blocks on the
/// network to tell you what you are missing.
pub fn cached(cache: &Path) -> Option<Vec<Entry>> {
    let xml = std::fs::read_to_string(cache_path(cache)).ok()?;
    let e = parse(&xml);
    if e.is_empty() {
        None
    } else {
        Some(e)
    }
}

/// F40's hybrid term rule: terms of 4+ characters match as substrings, shorter ones need a
/// word boundary. Without the length split, `css`/`git`/`php` were dropped; with substring
/// matching everywhere, "postgres" wrongly matched unrelated books.
fn hits(query_terms: &[String], hay: &str) -> usize {
    let hay = hay.to_lowercase();
    let words: Vec<&str> = WORDS.find_iter(&hay).map(|m| m.as_str()).collect();
    query_terms
        .iter()
        .filter(|t| {
            if t.len() >= 4 {
                hay.contains(t.as_str())
            } else {
                words.iter().any(|w| w == &t.as_str())
            }
        })
        .count()
}

pub fn suggest<'a>(entries: &'a [Entry], query: &str, want: usize) -> Vec<&'a Entry> {
    let terms: Vec<String> = crate::retrieve::terms(query);
    if terms.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(&Entry, usize)> = entries
        .iter()
        .map(|e| {
            let name = e.name.replace(['_', '-', '.'], " ");
            let n = hits(&terms, &name) * 3 + hits(&terms, &e.title) * 2 + hits(&terms, &e.summary);
            (e, n)
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    // docs first, then the general tier; within a tier, score then smaller size
    scored.sort_by(|a, b| {
        let da = IS_DOCS.is_match(&format!("{} {}", a.0.name, a.0.tags));
        let db = IS_DOCS.is_match(&format!("{} {}", b.0.name, b.0.tags));
        db.cmp(&da).then(b.1.cmp(&a.1)).then(a.0.bytes.cmp(&b.0.bytes))
    });
    scored.into_iter().take(want).map(|(e, _)| e).collect()
}

/// Local ZIM stems, dated: `devdocs_en_bash_2026-04`.
pub fn local(zim: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(zim)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "zim"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    out.sort();
    out
}

/// (book key, local date, catalog date) for every local ZIM the catalog has newer.
/// Dates are `YYYY-MM`, so a string compare is a date compare.
///
/// Matches on `Entry::key` — the stem minus its date — so a local `nopic` edition is only
/// ever compared against `nopic` editions. Keying on the catalog's `<name>` would report a
/// 124 GB `maxi` release as an "update" to a 2.2 GB `nopic` corpus.
pub fn outdated(zim: &Path, entries: &[Entry]) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for stem in local(zim) {
        let Some(c) = DATED.captures(&stem) else { continue };
        let (key, have) = (c[1].to_string(), c[2].to_string());
        // several editions of the same book may be listed; take the newest matching one
        let newest = entries
            .iter()
            .filter(|e| e.key() == key)
            .map(|e| e.date())
            .filter(|d| !d.is_empty())
            .max();
        if let Some(newest) = newest {
            if newest > have {
                out.push((key, have, newest));
            }
        }
    }
    out
}

/// One line for `tny` to print on a normal query, written only when a catalog refresh
/// actually found something. Costs a `read` of a few bytes per query, no network.
pub fn write_stale_note(cache: &Path, stale: &[(String, String, String)]) {
    let p = cache.join("stale.txt");
    if stale.is_empty() {
        let _ = std::fs::remove_file(p);
        return;
    }
    let names: Vec<String> = stale.iter().map(|(n, have, new)| format!("{n} {have}→{new}")).collect();
    let _ = std::fs::write(p, format!("tny: {} corpus update(s): {}\n     refresh with: tny --corpus add <name>\n", stale.len(), names.join(", ")));
}

pub fn stale_note(cache: &Path) -> Option<String> {
    std::fs::read_to_string(cache.join("stale.txt")).ok().filter(|s| !s.trim().is_empty())
}

/// Several editions share one `<name>`. tny reads text and throws images away in
/// `html2txt`, so `nopic` is strictly the right default: same articles, a fraction of the
/// bytes (Wikipedia is 52.7 GB nopic against 124 GB maxi). `mini` is rejected outright —
/// it holds lead paragraphs only, which defeats section selection and detail answers.
fn pick_flavour<'a>(entries: &'a [Entry], name: &str) -> Option<&'a Entry> {
    let same: Vec<&Entry> = entries.iter().filter(|e| e.name == name).collect();
    if same.len() > 1 {
        let rank = |e: &Entry| match () {
            _ if e.stem().contains("_nopic") => 0,
            _ if e.stem().contains("_mini") => 3,
            _ if e.stem().contains("_maxi") => 2,
            _ => 1,
        };
        let best = same.iter().copied().min_by_key(|e| (rank(e), e.bytes))?;
        eprintln!(
            "tny: {name} has {} editions; taking {} ({}) — tny reads text only",
            same.len(),
            best.stem(),
            best.size_human()
        );
        return Some(best);
    }
    same.into_iter().next()
}

/// Mirrors, tried in order after the catalog's own URL.
///
/// Measured the hard way: `lb.download.kiwix.org` served 4 GB happily, then went
/// unreachable mid-transfer — as did `download.kiwix.org` and even `library.kiwix.org`.
/// curl sat at exactly 1,651,695,728 bytes indefinitely because a stalled socket is not an
/// error. Three independent mirrors were up throughout and all reported byte-identical
/// lengths. A single hard-coded host is a design defect, and so is an unbounded read.
const MIRRORS: &[&str] = &[
    "https://download.kiwix.org/",
    "https://mirrors.dotsrc.org/kiwix/",
    "https://ftp.fau.de/kiwix/",
    "https://mirror.accum.se/mirror/kiwix.org/",
];

fn mirror_urls(href: &str) -> Vec<String> {
    let direct = href.trim_end_matches(".meta4").to_string();
    let mut urls = vec![direct.clone()];
    // href is `https://<host>/zim/<collection>/<file>.zim.meta4`
    if let Some(rel) = direct.split("/zim/").nth(1) {
        for m in MIRRORS {
            let candidate = format!("{m}zim/{rel}");
            if candidate != direct {
                urls.push(candidate);
            }
        }
    }
    urls
}

/// The authoritative length, asked of the mirrors.
///
/// The catalog's `length` is rounded UP to a KiB boundary — it advertises 2,239,865,856 for
/// a file that is really 2,239,864,871 bytes (985 short, and 2239865856/1024 is exactly
/// 2187369). Verifying against it rejects a perfectly complete download and retries forever,
/// so the catalog figure is only ever used for display and as a fallback.
fn remote_len(urls: &[String]) -> Option<u64> {
    let agent = ureq::builder()
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(20))
        .build();
    for u in urls {
        if let Ok(r) = agent.head(u).set("User-Agent", "tny/0.1").call() {
            if let Some(len) = r.header("content-length").and_then(|s| s.parse::<u64>().ok()) {
                if len > 0 {
                    return Some(len);
                }
            }
        }
    }
    None
}

/// One attempt against one mirror. Returns the number of bytes on disk afterwards.
///
/// `timeout_read` is the stall guard: without it a hung mirror blocks forever, which is the
/// exact failure this function exists to survive.
fn pull(dest: &Path, url: &str, want: u64) -> Result<u64, String> {
    let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if want > 0 && have >= want {
        return Ok(have);
    }
    let agent = ureq::builder()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(45))
        .build();
    let mut req = agent.get(url).set("User-Agent", "tny/0.1");
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // 416 carries `Content-Range: bytes */<total>` — the server's own word on the size.
        // If we already hold that many bytes the file is complete, which is exactly what
        // happened when the catalog's KiB-rounded length said otherwise.
        Err(ureq::Error::Status(416, r)) => {
            let total = r
                .header("content-range")
                .and_then(|h| h.rsplit('/').next().map(str::trim).and_then(|n| n.parse::<u64>().ok()));
            return match total {
                Some(t) if have >= t => Ok(have),
                Some(t) => Err(format!("range {have} rejected, server holds {t} bytes")),
                None => Err("range not satisfiable".into()),
            };
        }
        Err(e) => return Err(format!("{e}")),
    };
    let resuming = resp.status() == 206;
    if have > 0 && !resuming {
        return Err(format!("mirror ignored Range (status {}), refusing to restart", resp.status()));
    }
    let mut file = if resuming {
        std::fs::OpenOptions::new().append(true).open(dest).map_err(|e| format!("cannot append: {e}"))?
    } else {
        std::fs::File::create(dest).map_err(|e| format!("cannot create {}: {e}", dest.display()))?
    };
    let mut written = if resuming { have } else { 0 };
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 16];
    let mut tick = written;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            // keep what arrived: the caller resumes from here, possibly on another mirror
            Err(e) => {
                let _ = file.flush();
                return Err(format!("stalled at {:.0} MB: {e}", written as f64 / 1e6));
            }
        };
        file.write_all(&buf[..n]).map_err(|e| format!("write failed: {e}"))?;
        written += n as u64;
        if written - tick >= 32 << 20 {
            tick = written;
            eprint!("\r     {:.0}/{:.0} MB", written as f64 / 1e6, want as f64 / 1e6);
            let _ = std::io::stderr().flush();
        }
    }
    file.flush().map_err(|e| format!("flush failed: {e}"))?;
    Ok(written)
}

/// Download with `Range` resume, mirror fallback, and a byte-length check against the
/// catalog (F21). A truncated ZIM is worse than a missing one: kiwix-serve refuses to mount
/// its whole library when one file is short, so a single dead mirror took every book down
/// (F60 — a 376 KB fragment of a 24 MB book made every query fail). Resume needs the partial
/// bytes to survive, so they live under `.part`, which `serve_kiwix` cannot mount.
pub fn add(zim: &Path, cache: &Path, name: &str) -> Result<PathBuf, String> {
    let entries = match cached(cache) {
        Some(e) => e,
        None => fetch(cache)?,
    };
    // Exact stem or exact key first; only then the ambiguous catalog `<name>`.
    let entry = entries
        .iter()
        .find(|e| e.stem() == name || e.key() == name)
        .or_else(|| pick_flavour(&entries, name))
        .ok_or_else(|| format!("no catalog entry named {name} — try: tny --corpus search {name}"))?;

    std::fs::create_dir_all(zim).map_err(|e| format!("cannot create {}: {e}", zim.display()))?;
    let dest = zim.join(format!("{}.zim", entry.stem()));
    let part = dest.with_extension("zim.part");
    let urls = mirror_urls(&entry.href);
    let disk = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);

    // The mirrors' Content-Length is the only exact figure; the catalog's is KiB-rounded.
    let target = match remote_len(&urls) {
        Some(len) => len,
        None if disk(&part) + disk(&dest) > 0 => {
            return Err(format!(
                "no mirror is reachable to verify {} ({:.0} MB on disk) — re-run when one is",
                entry.stem(),
                (disk(&part) + disk(&dest)) as f64 / 1e6
            ))
        }
        None => return Err(format!("no mirror is reachable for {}", entry.stem())),
    };

    // A file already under the final name is either a finished corpus or, from a tny that
    // downloaded straight there, resumable bytes that poison the whole mount. Size decides,
    // and a short one is moved out of the way rather than deleted: those bytes still count.
    if disk(&dest) > 0 {
        if disk(&dest) == target {
            eprintln!("tny: {} already complete ({})", dest.display(), entry.size_human());
            return Ok(dest);
        }
        eprintln!("tny: {} is short — resuming it as {}", dest.display(), part.display());
        std::fs::rename(&dest, &part).map_err(|e| format!("cannot move {}: {e}", dest.display()))?;
    }
    let have = disk(&part);
    if have > target {
        return Err(format!(
            "{} is {have} bytes but the mirror serves {target} — delete it and re-run",
            part.display()
        ));
    }
    if have > 0 {
        eprintln!("tny: resuming {} at {:.0} MB of {}", entry.stem(), have as f64 / 1e6, entry.size_human());
    } else {
        eprintln!("tny: downloading {} ({}, {} articles)", entry.stem(), entry.size_human(), entry.articles);
    }

    let mut last = String::from("no mirror attempted");
    // Passes, not infinite retries: a whole pass that moves zero bytes means every mirror is
    // failing, and hammering them is not a strategy.
    for pass in 0..4 {
        let before = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        for url in &urls {
            let host = url.split('/').nth(2).unwrap_or(url);
            match pull(&part, url, target) {
                Ok(n) if n >= target => {
                    eprintln!("\r     {:.0} MB from {host}", n as f64 / 1e6);
                    // Only a complete file gets the name kiwix-serve will mount.
                    std::fs::rename(&part, &dest)
                        .map_err(|e| format!("cannot rename {} to {}: {e}", part.display(), dest.display()))?;
                    eprintln!("tny: {} ready", dest.display());
                    return Ok(dest);
                }
                Ok(n) => {
                    last = format!("{host} stopped short at {:.0} MB", n as f64 / 1e6);
                    eprintln!("\ntny: {last}, trying the next mirror");
                }
                Err(e) => {
                    last = format!("{host}: {e}");
                    eprintln!("\ntny: {last}");
                }
            }
        }
        let after = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if after == before {
            return Err(format!("no mirror is serving {} — last error: {last}", entry.stem()));
        }
        eprintln!("tny: pass {} advanced to {:.0} MB, continuing", pass + 1, after as f64 / 1e6);
    }
    Err(format!(
        "{} still incomplete after 4 passes — re-run to resume. Last error: {last}",
        dest.display()
    ))
}
