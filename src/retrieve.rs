//! Retrieval: kiwix-serve search, article fetch, section selection.
//!
//! Every constant here is measured (see NOTES.md). Nothing in this file calls a model:
//! ranking is RRF over Xapian order and lexical title+snippet scoring (F31, 9/10 —
//! identical to the embedding version), and section selection is lexical (F31, 14/14 at
//! top-5). The model only phrases what this stage hands it (F16, F29).

use crate::ground::{denoise, html2txt};
use regex::Regex;
use std::sync::LazyLock;

macro_rules! re {
    ($name:ident = $src:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($src).expect("bad regex"));
    };
}

// F15/F35: raw conversational queries under-retrieve. kiwix ANDs every term, so one term
// no document shares zeroes the whole query — "string versus str slice" returned 0 hits,
// and dropping "versus" returns `str` first. Comparison words are the common culprit and
// they carry no retrieval signal.
re!(STOP = r"(?i)^(how|do|i|the|a|an|my|is|are|why|what|when|where|to|in|on|from|of|for|can|does|with|and|it|not|be|get|set|make|use|versus|vs|difference|differences|between|tradeoff|tradeoffs|pros|cons|better|worse|should|or|choose|choosing|compare|comparison|alternative|alternatives)$");
re!(NONQUERY = r"[^A-Za-z0-9\s.:+#-]");
// F31: deliberately NOT `STOP` — that list strips set/make/use/get, exactly the verbs a
// section head uses ("Set system clock"). This strips follow-up filler instead. 14/14 was
// measured with this list.
re!(STOP_LEX = r"(?i)^(how|do|i|the|a|an|my|is|are|why|what|when|where|to|in|on|from|of|for|can|does|with|and|it|only|ones|one|again|instead|all)$");
re!(TERM = r"[a-z0-9-]{2,}");
re!(HEADING = r"(?s)<h[2345][^>]*>(.*?)</h[2345]>");
re!(TAG = r"<[^>]+>");
re!(WS = r"\s+");
re!(ITEM_TITLE = r"(?s)<title>(.*?)</title>");
re!(ITEM_LINK = r"(?s)<link>(.*?)</link>");
re!(ITEM_DESC = r"(?s)<description>(.*?)</description>");
re!(CONTENT_BOOK = r"/content/([^/]+)");
re!(TRAIL_PAREN = r"\s*\(.*\)$");
// F14: the English _maxi ZIM is full of localised duplicates — half the candidate list.
re!(LOCALISED = r"\((Magyar|Deutsch|Español|Français|Português|Italiano|Polski|Русский|简体|正體|日本語|한국어|Türkçe|Nederlands|Čeština|Ελληνικά|עברית|فارسی|العربية|Indonesia|Tiếng Việt|Norsk|Dansk|Svenska|Suomi|Română|Български|Українська|Hrvatski|Slovenčina|Lietuvių|Català)\)");

// F48: page kind, from path structure and title shape — no model, no per-book config.
// Mounting Stack Exchange put "Highest Voted 'pacman' Questions" at rank 1 for a how-to
// query: a tag index page that lists questions and answers none. Answer pages are
// `questions/<id>/<slug>`; tag and user pages are navigation.
re!(SE_INDEX = r"(?i)^(highest voted|newest|active|unanswered|top|recent)\b|\bquestions$");
//
// F61: those patterns MUST be anchored. Unanchored, `/tags?/` matched devdocs'
// `engine/reference/commandline/tag/index`, so `docker tag` was classified as a navigation
// page and dropped from every candidate list — the exact article the question asked for. In
// a real corpus these pages only ever appear at the path root, verified against the mounted
// Stack Exchange ZIM: `questions/tagged/bash`, never nested.
re!(NAV_PATH = r"^questions/tagged/|^users?/|^tags?/");
re!(QA_PATH = r"questions/[0-9]+/");
// F49: intent is INFERRED from the query — there is no label at runtime. Measured at 12/18.
re!(DIAGNOSE = r"(?i)\b(error|errors|fail|failed|failing|refused|denied|cannot|can't|won't|does ?n't|broken|why (is|are|does|do|did|would|am|can't)|no such|not found|timed? ?out|exit code|permission)\b");
re!(HOWTO_Q = r"(?i)^(how (do|can|would) i|how to|what command)\b|^(create|set|mount|encrypt|generate|check|list|install|enable|configure|disable|remove|start|stop|make)\b");
re!(TITLE_TERM = r"[a-z0-9_.:-]{2,}");
re!(TITLE_STOP = r"^(the|a|an|of|in|to|and|or|for)$");
// F68: prose glue that survives `prep` — it is not a stopword (it can be a section head or a
// title term), but among twenty candidates it is the first thing to drop from a search query.
re!(GLUE = r"^(am|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|can|could|should|may|might|must|my|me|it|its|this|that|these|those|there|here|then|than|but|so|if|when|while|from|into|onto|with|without|about|like|just|also|very|really|some|any|all|not|only|even|still|want|wanted|need|needed|try|tried|trying|get|got|getting|see|seen|know|think|seems|new|old|good|bad)$");
// F64: apparatus, not content — citation and navigation sections that answer nothing but match
// query terms densely because they are lists of titles.
re!(APPARATUS = r"(?i)^\s*(references?|external links?|see also|further reading|bibliography|notes?|citations?|sources?|footnotes?|related pages?|external resources?)\s*$");

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Article,
    Qa,
    Index,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub title: String,
    pub book: String,
    pub path: String,
    pub snip: String,
    pub kind: Kind,
    /// Xapian's position *within its own book* — the only place that order means anything.
    pub rank: usize,
}

pub fn page_kind(title: &str, path: &str) -> Kind {
    if NAV_PATH.is_match(path) || SE_INDEX.is_match(title) {
        Kind::Index
    } else if QA_PATH.is_match(path) {
        Kind::Qa
    } else {
        Kind::Article
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Intent {
    Howto,
    Diagnose,
    Other,
}

pub fn infer_intent(q: &str) -> Intent {
    if DIAGNOSE.is_match(q) {
        Intent::Diagnose
    } else if HOWTO_Q.is_match(q.trim()) {
        Intent::Howto
    } else {
        Intent::Other
    }
}

pub struct Section {
    pub head: String,
    pub text: String,
}

pub struct Picked {
    pub heads: Vec<String>,
    pub text: String,
}

pub fn prep(q: &str) -> String {
    let lower = q.to_lowercase();
    let cleaned = NONQUERY.replace_all(&lower, " ");
    cleaned
        .split_whitespace()
        .filter(|w| !STOP.is_match(w))
        .collect::<Vec<_>>()
        .join(" ")
}

/// F68: kiwix returns *nothing* for a long query. Measured on one book: 4 terms → 13 hits,
/// 8 → 60, 12 → 40, 20 → 6, 24 → **0**. Every fixture case is a headline question of six
/// words, so this never showed; every real question is a sentence, and the held-out arm that
/// asks with the user's own words routed 18/53 against 52/53 for the page's own title.
///
/// So the query is cut to its most informative terms, in their original order. Rarity has to
/// be guessed — kiwix exposes no corpus statistics, and the one attempt to estimate IDF from
/// the retrieved candidates made ranking worse (23/58 against 32/58), because a biased sample
/// is not a corpus. What survives instead is shape and position.
///
/// The first cut of this ranked *any* token with a digit or punctuation as an identifier, and
/// that was wrong in a way worth keeping written down: asked "Why does uname show x86_64 three
/// times? This is Ubuntu 12.04.4", it searched for `x86 64 12.04.4.` and found nothing, having
/// dropped `uname` and `architecture` — the only two words that name the subject. Version
/// strings and pasted shell output are rare, but they describe the asker's machine, not the
/// question. A name with letters in it (`ext4`, `x86_64`, `gksudo`) is worth keeping; a run of
/// digits and dots is not. Position matters for the same reason: people put their subject in
/// the first clause and their environment in the last.
pub fn select_terms(q: &str, cap: usize) -> String {
    let words: Vec<&str> = q.split_whitespace().collect();
    if words.len() <= cap {
        return q.to_string();
    }
    let weight = |w: &str, i: usize| -> i32 {
        if GLUE.is_match(w) {
            return -200;
        }
        let letters = w.chars().filter(|c| c.is_alphabetic()).count();
        let digits = w.chars().filter(|c| c.is_ascii_digit()).count();
        // `12.04.4`, `3.2.0-59`, `408`. These are *rare*, so real IDF ranks them top — and
        // they describe the asker's machine, not the question. The veto stays above rarity:
        // it is about what kiwix can tokenise, not about what is informative.
        if letters == 0 || digits > letters {
            return -100;
        }
        let named = digits > 0 || w.contains(['-', '_', '/']);
        let base = match (named, w.chars().count()) {
            (true, _) => 60,
            (_, n) if n >= 8 => 40,
            (_, n) if n >= 6 => 20,
            _ => 0,
        };
        // The subject is stated before the setup, so an early term breaks ties over a late one.
        base + if i < 8 { 2 } else { 0 }
    };
    let mut ranked: Vec<(usize, &str)> = words.iter().copied().enumerate().collect();
    // Stable by construction: equal weights keep the order the user typed them in, so a
    // truncated query still reads as a phrase rather than a bag.
    ranked.sort_by_key(|&(i, w)| (-weight(w, i), i));
    let mut keep: Vec<usize> = ranked.into_iter().take(cap).map(|(i, _)| i).collect();
    keep.sort_unstable();
    keep.iter().map(|&i| words[i]).collect::<Vec<_>>().join(" ")
}

pub fn terms(q: &str) -> Vec<String> {
    let lower = q.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for m in TERM.find_iter(&lower) {
        let w = m.as_str();
        if !STOP_LEX.is_match(w) && !out.iter().any(|x| x == w) {
            out.push(w.to_string());
        }
    }
    out
}

/// Head/title hits weigh 3; body hits saturate at 5 so one long section cannot win by
/// repetition alone.
pub fn lex_score(head: &str, body: &str, t: &[String]) -> usize {
    let h = head.to_lowercase();
    let b = body.to_lowercase();
    let mut s = 0;
    for w in t {
        if h.contains(w) {
            s += 3;
        }
        s += b.matches(w.as_str()).count().min(5);
    }
    s
}

/// F31: the answer is often deeper into a section than the head of it. Selecting the right
/// section and then slicing its first 600 chars threw the answer away — §Protection in
/// OpenSSH mentions PermitRootLogin at offset 4,704. Centre the window on the densest run
/// of query terms instead of the section start.
pub fn window(text: &str, t: &[String], budget: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= budget {
        return text.to_string();
    }
    let step = (budget / 4).max(80);
    let mut best = 0usize;
    let mut best_score = -1i32;
    let mut at = 0usize;
    while at < chars.len() {
        let end = (at + budget).min(chars.len());
        let slice: String = chars[at..end].iter().collect::<String>().to_lowercase();
        let s = t.iter().filter(|w| slice.contains(w.as_str())).count() as i32;
        if s > best_score {
            best_score = s;
            best = at;
        }
        at += step;
    }
    // never start mid-word
    let cut = if best == 0 {
        0
    } else {
        chars[best..]
            .iter()
            .position(|c| *c == ' ')
            .map(|off| best + off + 1)
            .unwrap_or(best)
    };
    let end = (cut + budget).min(chars.len());
    let body: String = chars[cut..end].iter().collect();
    if cut > 0 {
        format!("… {body}")
    } else {
        body
    }
}

/// F31: split on h2–h5, not h2–h3. OpenSSH's §Protection is one 12.9 KB h2 chunk with
/// PermitRootLogin at offset 4,704: selection ranked it #1 and the window still missed the
/// answer. Splitting deeper yields 77 sections of ≤3.4 KB and fixed it for free — selection
/// went 12/14 → 14/14 at identical context size.
pub fn sections_of(html: &str) -> Vec<Section> {
    let heads: Vec<(String, usize, usize)> = HEADING
        .captures_iter(html)
        .map(|c| {
            let whole = c.get(0).unwrap();
            let inner = c.get(1).map_or("", |m| m.as_str());
            let text = TAG.replace_all(inner, " ");
            (WS.replace_all(&text, " ").trim().to_string(), whole.start(), whole.end())
        })
        .collect();
    let mut out = Vec::new();
    // F67: the lead was never a section. This loop starts *at* the first heading, so every
    // character before it was dropped — and on Wikipedia that is exactly where the
    // definitional fact lives: "Jupiter has 97 known moons", "Kilimanjaro is 5,895 m high".
    // Asked "how many moons does jupiter have", tny sent §Origin and evolution and §Irregular
    // satellites, which mention moons constantly and count none of them. The lead now
    // competes on score like any other section rather than being handed a slot: on a Stack
    // Exchange page the lead is the question, and the answer is below it.
    if let Some((_, first, _)) = heads.first() {
        let lead = html2txt(&html[..*first]);
        if lead.len() > 80 {
            out.push(Section { head: "(lead)".into(), text: lead });
        }
    }
    for (i, (head, _, end)) in heads.iter().enumerate() {
        let stop = heads.get(i + 1).map_or(html.len(), |(_, at, _)| *at);
        let text = html2txt(&html[*end..stop]);
        if text.len() > 80 {
            out.push(Section { head: head.clone(), text });
        }
    }
    out
}

/// F31: lexical section selection, the no-server arm. 14/14 at top-5 in ~2,680 chars; the
/// embedding arm reaches the same 14/14 at top-3 in ~1,640 chars, so bge-small buys a 39 %
/// smaller prompt and nothing else. v1 ships without it — one fewer supervised process —
/// and the trade is re-measurable from `bench/harness.mjs select`.
pub fn pick_sections(html: &str, q: &str, top_n: usize, per: usize) -> Picked {
    let t = terms(q);
    let mut secs = sections_of(html);
    // F73: a lead answers "what is X" and mis-answers "how do I X". F67 made the lead
    // reachable and the product fell 40/58 to 34/58, because a lead is a definition: asked how
    // to let a user run commands with sudo, tny explained what sudo *is*; asked what an HTTP
    // cookie is used *for*, it defined one. Both were graded wrong and deserved it.
    //
    // Suppressing the lead by page *kind* was tried first and reverted: a Stack Exchange page
    // has no h2 above its question, so its "lead" runs into the first answer and carries real
    // content — dropping it cost three cases of fact-in-context, stable over three runs. The
    // split that matches the evidence is the question's own intent, which `infer_intent`
    // already recovers from the query with no label at runtime.
    if !matches!(infer_intent(q), Intent::Other) {
        secs.retain(|s| s.head != "(lead)");
    }
    if secs.is_empty() {
        // Stack Exchange and DevDocs pages have no h2–h5 structure at all.
        return Picked {
            heads: vec!["(lead)".into()],
            text: window(&html2txt(html), &t, per * top_n),
        };
    }
    let mut scored: Vec<(usize, usize)> = secs
        .iter()
        .enumerate()
        // F64: an apparatus section is never an answer. `References` is a wall of article
        // titles, so it matches query terms by sheer density and outscored the prose: asked
        // "when did the berlin wall fall", tny picked §References twice plus §20th anniversary
        // celebrations — an event held in **2009** — and the model dutifully answered "November
        // 9, 2009" from the section it was shown, with the right article on screen. A wrong
        // section is indistinguishable from a hallucination in the output.
        .filter(|(_, s)| !APPARATUS.is_match(&s.head))
        .map(|(i, s)| (i, lex_score(&s.head, &s.text, &t)))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    // Wikipedia repeats `References` under several parents, so the same head can win more than
    // one slot and spend the budget twice on identical junk.
    let mut heads_seen: Vec<&str> = Vec::new();
    let top: Vec<(usize, usize)> = scored
        .into_iter()
        .filter(|(i, _)| {
            let h = secs[*i].head.as_str();
            !heads_seen.contains(&h) && {
                heads_seen.push(h);
                true
            }
        })
        .take(top_n)
        .collect();
    Picked {
        heads: top.iter().map(|(i, _)| secs[*i].head.clone()).collect(),
        text: top
            .iter()
            // F67: a lead is inverted-pyramid, so its first sentence is the definitional fact
            // — "Jupiter has N known moons", "Kilimanjaro is 5,895 m high" — while `window`
            // centres on the densest run of query terms and cut exactly that opening.
            // Prefix-only would be the opposite mistake: it optimises headline questions and
            // loses the deep ones, where the evidence sits further down the lead. So the lead
            // splits its budget, a third for the opening and the rest centred, and every other
            // section is a fragment of an argument where centring is simply right.
            .map(|(i, _)| {
                let s = &secs[*i];
                let body = if s.head == "(lead)" {
                    let open = per / 3;
                    let rest = window(&s.text, &t, per - open);
                    let opening = window(&s.text, &[], open);
                    if rest.starts_with(opening.trim_end()) { window(&s.text, &t, per) } else { format!("{opening}\n…\n{rest}") }
                } else {
                    window(&s.text, &t, per)
                };
                format!("## {}\n{}", s.head, body)
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

/// Length-normalised title scoring: BM25's idea without the corpus statistics kiwix refuses
/// to expose (its search XML carries no score at all).
///
/// F49: `lex_score` weighted every title hit +3 with no normalisation, so a 74-character
/// Stack Exchange question title beat the Arch Wiki's "Swap" on a how-to query purely by
/// matching more terms. That single defect cost 14 of 32 cases.
///
/// F59: the +3 that replaced it was still too heavy. A grid over the weights, scored offline
/// against all 58 cases (bench/sweep.mjs), puts every one of the top 14 configurations at
/// +2: article@1 36/58 against 32/58, and the answer reaches the top three for 47 cases
/// against 43. The win is a plateau (title 2 x cover 3-4 x rank 4-5 all score alike), and it
/// gains in all four fixtures, so it is not a fit to one of them.
fn title_body_score(c: &Candidate, t: &[String]) -> f64 {
    let title = c.title.to_lowercase();
    let tw = TERM.find_iter(&title).count().max(1) as f64;
    let title_hits = t.iter().filter(|w| title.contains(w.as_str())).count() as f64;
    let body = denoise(&c.snip).to_lowercase();
    let body_len = body.chars().count().min(400);
    let body = &body[..body.char_indices().nth(body_len).map_or(body.len(), |(i, _)| i)];
    let body_hits = t.iter().filter(|w| body.contains(w.as_str())).count() as f64;
    title_hits * 2.0 / tw.sqrt() + body_hits / (t.len().max(1) as f64)
}

/// Is the title an entity the query names? Reference questions do exactly this — "what does
/// the `--rm` option do in *docker run*" — and term coverage cannot see it, because "docker
/// run" is two terms against a question title that matches five by sheer length. Took
/// `reference` from 3/6 to 4/6 (F49).
fn title_covered(c: &Candidate, t: &[String]) -> f64 {
    let title = c.title.to_lowercase();
    let words: Vec<&str> = TITLE_TERM
        .find_iter(&title)
        .map(|m| m.as_str())
        .filter(|w| !TITLE_STOP.is_match(w))
        .collect();
    if words.is_empty() || words.len() > 5 {
        return 0.0;
    }
    let joined = t.join(" ");
    if !words.iter().all(|w| joined.contains(w)) {
        return 0.0;
    }
    // F54: a subset test alone promotes the most GENERIC article. "Docker" is one word,
    // present in "docker image tag for a registry", so it collected the full entity bonus
    // and beat the page that answers the question — same for `Plant`, `Ocean`, `Memory`,
    // `PostgreSQL`. Scale by the share of the question the title actually accounts for: a
    // title naming 2 of 4 query terms is twice the entity match of one naming 1 of 4.
    words.len() as f64 / t.len().max(1) as f64
}

/// A how-to question wants instructions, and a Q&A title that is itself a question is not
/// evidence that it answers *this* one. Diagnosis is left alone: that is what Q&A is best at
/// (F42 measured 5/6 on real tool errors), and the QA fixture's own construction showed the
/// wiki answering 6 of 15 candidate diagnosis questions better.
fn kind_prior(intent: Intent, kind: Kind) -> f64 {
    match (intent, kind) {
        (Intent::Howto, Kind::Qa) => -2.0,
        (Intent::Diagnose, Kind::Qa) => 1.0,
        _ => 0.0,
    }
}

/// F49: the measured winner. On 32 verified cases across four intents this reaches 17/32
/// right-article rank-1 and 25/32 right-book, against 12/32 and 20/32 for the fused lexical
/// scoring it replaces, and 5/32 for rescoring on fetched article text — kiwix's snippet is
/// the query-matched passage, so full text only dilutes the signal (the same result F39 hit).
pub fn rank_articles(q: &str, cands: &[Candidate]) -> Vec<Candidate> {
    let t = terms(q);
    let intent = infer_intent(q);
    let mut scored: Vec<(f64, &Candidate)> = cands
        .iter()
        .map(|c| {
            let s = title_body_score(c, &t)
                + kind_prior(intent, c.kind)
                + title_covered(c, &t) * 3.0
                // F59: Xapian's own within-book order is real evidence, and a hundredth of a
                // point made it a tie-break nobody could win. At /5 a book's 4th hit must
                // beat its 1st by 0.6 to overtake it — worth 4 cases on the same fixture.
                - c.rank as f64 / 5.0;
            (s, c)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, c)| c.clone()).collect()
}

pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn parse_items(xml: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    for item in xml.split("<item>").skip(1) {
        let grab = |re: &Regex| -> String {
            re.captures(item).and_then(|c| c.get(1)).map_or(String::new(), |m| m.as_str().to_string())
        };
        let title = grab(&ITEM_TITLE);
        let link = grab(&ITEM_LINK);
        let book = CONTENT_BOOK
            .captures(&link)
            .and_then(|c| c.get(1))
            .map_or(String::new(), |m| m.as_str().to_string());
        if book.is_empty() || LOCALISED.is_match(&title) {
            continue;
        }
        let path = link
            .rfind("/content/")
            .and_then(|i| link[i + 9..].find('/').map(|j| link[i + 9 + j + 1..].to_string()))
            .unwrap_or_default();
        let kind = page_kind(&title, &path);
        if kind == Kind::Index {
            continue; // F48: navigation pages are never answers
        }
        let snip = WS.replace_all(&TAG.replace_all(&grab(&ITEM_DESC), " "), " ").trim().to_string();
        let rank = out.len();
        out.push(Candidate { title, book, path, snip, kind, rank });
    }
    out
}

/// One book. `books.name` wants the ZIM filename stem (F11).
pub fn search_book(kiwix: &str, query: &str, book: &str, want: usize) -> Result<Vec<Candidate>, String> {
    let url = format!(
        "{kiwix}/search?books.name={book}&pattern={}&format=xml&pageLength={want}",
        urlencode(query)
    );
    let xml = ureq::get(&url)
        .call()
        .map_err(|e| format!("kiwix search failed: {e}"))?
        .into_string()
        .map_err(|e| format!("kiwix search body: {e}"))?;
    let mut rows = parse_items(&xml);
    rows.truncate(want);
    Ok(rows)
}

/// F49: candidate GENERATION was the wall, not scoring. One unrouted global query surfaced
/// the right article for only 10 of 32 verified cases — 413k Stack Exchange pages crowd out
/// a 132-article devdocs book entirely — while asking each book separately reaches 30/32.
/// No scorer can rank a candidate that was never retrieved. Nine requests at ~57 ms each,
/// against a query whose generation step costs 21 s.
pub fn search_union(kiwix: &str, query: &str, books: &[String], per_book: usize) -> Vec<Candidate> {
    // F78: the requests are independent and kiwix-serve is threaded, so they overlap. Four at
    // a time, not sixteen: kiwix-serve SIGSEGVs under heavier concurrency on a loaded machine
    // (F72), and a laptop answering one question is not the place to saturate a server.
    let threads = books.len().min(4).max(1);
    let chunk = books.len().div_ceil(threads);
    let mut parts: Vec<Vec<Vec<Candidate>>> = Vec::with_capacity(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = books
            .chunks(chunk)
            .map(|ch| s.spawn(move || ch.iter().map(|b| search_book(kiwix, query, b, per_book).unwrap_or_default()).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            parts.push(h.join().unwrap_or_default());
        }
    });

    // Reassembled in book order, never completion order: F59's rank tie-break and the dedupe
    // below both depend on a deterministic candidate order, and threads do not provide one.
    let mut out: Vec<Candidate> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for rows in parts.into_iter().flatten().flatten() {
        // dedupe per book: the same title in two books is two distinct answers
        let key = format!("{}\u{0}{}", rows.book, TRAIL_PAREN.replace(&rows.title, ""));
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(rows);
    }
    out
}

pub fn article(kiwix: &str, book: &str, path: &str) -> Result<String, String> {
    let url = format!("{kiwix}/content/{book}/{path}");
    ureq::get(&url)
        .call()
        .map_err(|e| format!("article fetch failed: {e}"))?
        .into_string()
        .map_err(|e| format!("article body: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apparatus_sections_never_win_a_slot() {
        // F63: `References` is a wall of article titles, so it matches query terms by density
        // and beat the prose; Wikipedia repeats the head, so it took two of five slots. The
        // answer then came from `§20th anniversary celebrations` — an event held in 2009 — for
        // "when did the berlin wall fall".
        // `sections_of` drops any section under 80 characters, so each body is real prose.
        let cite = "<p>Berlin wall fall Berlin wall fall Berlin wall fall archived from the \
                    original Berlin wall fall retrieved Berlin wall fall press release</p>";
        let html = format!(
            "<h2>References</h2>{cite}\
             <h2>Fall</h2><p>The Berlin wall fell on 9 November 1989 after the announcement \
             that crossings would be permitted, and crowds crossed that evening.</p>\
             <h2>References</h2>{cite}\
             <h2>See also</h2>{cite}"
        );
        let html = html.as_str();
        let p = pick_sections(html, "when did the berlin wall fall", 3, 600);
        assert!(!p.heads.iter().any(|h| h == "References"), "apparatus section selected: {:?}", p.heads);
        assert!(!p.heads.iter().any(|h| h == "See also"), "apparatus section selected: {:?}", p.heads);
        assert!(p.heads.iter().any(|h| h == "Fall"), "lost the prose section: {:?}", p.heads);
        assert!(p.text.contains("1989"), "the answer is not in the sent text: {}", p.text);
    }

    #[test]
    fn section_heads_are_deduped() {
        // One head must not take two slots and spend the budget twice on identical text.
        let html = "<h2>Usage</h2><p>To create a swap file of a chosen size, allocate it, set \
                    its permissions, and format it before enabling. Alpha variant text.</p>\
                    <h2>Usage</h2><p>To create a swap file of a chosen size, allocate it, set \
                    its permissions, and format it before enabling. Beta variant text.</p>\
                    <h2>Notes on swap</h2><p>A swap file must not be sparse, so create it with \
                    a tool that writes real blocks rather than reserving a hole.</p>";
        let p = pick_sections(html, "how do I create a swap file", 3, 600);
        assert_eq!(p.heads.iter().filter(|h| *h == "Usage").count(), 1, "duplicate head: {:?}", p.heads);
    }

    #[test]
    fn nav_filter_never_eats_command_pages() {
        // F61: `/tags?/` unanchored matched devdocs' command reference and deleted the very
        // page the question named. Both directions must hold, on real paths from the ZIMs.
        assert_eq!(page_kind("docker tag", "engine/reference/commandline/tag/index"), Kind::Article);
        assert_eq!(page_kind("git tag", "git-tag/index"), Kind::Article);
        assert_eq!(page_kind("Highest Voted 'bash' Questions", "questions/tagged/bash"), Kind::Index);
        assert_eq!(page_kind("Swap file management", "questions/659914/swap-file-management"), Kind::Qa);
    }

    #[test]
    fn prep_strips_comparison_words() {
        // F35: "string versus str slice" returned 0 hits because kiwix ANDs every term
        assert_eq!(prep("what is the difference between string versus str slice"), "string str slice");
        assert_eq!(prep("How do I create a swap file?"), "create swap file");
    }

    #[test]
    fn terms_keeps_section_verbs() {
        // F31: `set` must survive here even though `prep` strips it — section heads use it
        let t = terms("how do I set the system timezone");
        assert!(t.contains(&"set".to_string()));
        assert!(t.contains(&"timezone".to_string()));
        assert!(!t.contains(&"the".to_string()));
    }

    #[test]
    fn window_centres_on_terms_not_start() {
        // the F31 failure: the answer sits far past a 600-char slice of the section
        let filler = "a ".repeat(400);
        let text = format!("{filler}PermitRootLogin no is the setting.{filler}");
        let w = window(&text, &[String::from("permitrootlogin")], 200);
        assert!(w.to_lowercase().contains("permitrootlogin"), "window missed the needle: {w}");
    }

    #[test]
    fn window_never_starts_mid_word() {
        let text = format!("{}needle here", "word ".repeat(200));
        let w = window(&text, &[String::from("needle")], 100);
        assert!(w.starts_with("… "));
        assert!(!w.contains("… ord"), "started mid-word: {w}");
    }

    #[test]
    fn lex_score_saturates_body_hits() {
        let t = vec!["swap".to_string()];
        let many = "swap ".repeat(50);
        // 5 body hits max, plus 3 for the head
        assert_eq!(lex_score("Swap", &many, &t), 8);
        assert_eq!(lex_score("other", &many, &t), 5);
    }

    #[test]
    fn sections_split_on_h2_to_h5() {
        let html = format!(
            "<h2>First</h2><p>{}</p><h4>Deep</h4><p>{}</p>",
            "alpha ".repeat(30),
            "beta ".repeat(30)
        );
        let secs = sections_of(&html);
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].head, "First");
        assert_eq!(secs[1].head, "Deep");
    }

    #[test]
    fn pick_sections_falls_back_without_headings() {
        // Stack Exchange / DevDocs pages have no h2–h5 at all
        let html = format!("<p>{}</p>", "mkswap and swapon ".repeat(80));
        let p = pick_sections(&html, "create a swap file", 5, 600);
        assert_eq!(p.heads, vec!["(lead)".to_string()]);
        assert!(p.text.contains("mkswap"));
    }
}
