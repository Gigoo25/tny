//! F96: search-first terminal UI.
//!
//! Ranked titles and snippets are the default surface. Selection previews a result, Enter reads
//! its cleaned article without leaving the terminal, and local-model synthesis remains explicit.
//! Work runs on a worker thread so the UI redraws on a 100 ms clock and `Esc` stays available
//! while a model is generating.

use crate::ground::article_text;
use crate::retrieve::{best_passage, prep};
use crate::{
    answer_once, article, cite_lines, human_book, model_name, open_source, prose_text,
    resolve_model, save_prefs, stop_chat, Answered, Cfg, Len, Mode, Source,
};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::crossterm::{execute, terminal};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::collections::HashSet;
use std::io::Write;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const DIM: Style = Style::new().fg(Color::DarkGray);
type Job = (Receiver<Result<Answered, String>>, Instant, bool);

/// One question and what came back.
struct Turn {
    q: String,
    a: String,
    cites: Vec<String>,
    generated: bool,
}

struct App {
    cfg: Cfg,
    input: String,
    question: String,
    /// F97: a follow-up is a conversation, and erasing the answer you are following up on
    /// hides the thing you are asking about. Turns accumulate and the view scrolls, so
    /// "difference between X and Y" can be read next to the answer about X.
    turns: Vec<Turn>,
    list: Vec<Source>,
    /// How many results supported the extracted or generated text. Kept as a marker, not as a
    /// reason to dim the other search results.
    read: usize,
    sel: Option<usize>,
    scroll: u16,
    stage: Arc<Mutex<String>>,
    /// Some while a worker thread is searching or synthesizing. The flag records whether
    /// that job used a model, even if the user changes mode before it lands.
    job: Option<Job>,
    /// A follow-up carries the previous turn; a new topic does not.
    thread: bool,
    note: String,
    /// Full cleaned text of the selected source, read inside the TUI.
    viewing: Option<(usize, String)>,
    /// Sources whose search-engine snippets were replaced with passages from article text.
    previewed: HashSet<(String, String)>,
    tick: usize,
    /// Vim's two states. Insert is the default at a cold start, because the first thing a
    /// user does is type a question; a seeded question from the command line starts in
    /// normal, where the answer is already there to be driven.
    insert: bool,
    /// Query-term highlighting is visible by default and toggled for this session with `h`.
    highlight: bool,
    /// `Some` while a `:` line is being typed.
    cmd: Option<String>,
    /// `Some` while a `/` article search is being typed.
    find: Option<String>,
    last_find: String,
    find_line: Option<usize>,
    pending_g: bool,
    /// Wrapped height of the transcript and of its viewport, both measured during the last
    /// draw — scrolling is clamped against them so the keys cannot leave the text.
    height: u16,
    view: u16,
    width: u16,
}

pub fn run(cfg: &Cfg, question: &str) -> Result<i32, String> {
    let stage = Arc::new(Mutex::new(String::from("ready")));
    let mut app = App {
        cfg: Cfg {
            progress: Some(stage.clone()),
            ..cfg.clone()
        },
        input: String::new(),
        question: String::new(),
        turns: vec![],
        list: vec![],
        read: 0,
        sel: None,
        scroll: 0,
        stage,
        job: None,
        thread: false,
        note: String::new(),
        viewing: None,
        previewed: HashSet::new(),
        tick: 0,
        insert: question.trim().is_empty(),
        highlight: true,
        cmd: None,
        find: None,
        last_find: String::new(),
        find_line: None,
        pending_g: false,
        height: 0,
        view: 0,
        width: 8,
    };

    let mut out = std::io::stdout();
    terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    execute!(out, terminal::EnterAlternateScreen).map_err(|e| e.to_string())?;
    // A panic in raw mode leaves the terminal unusable and the backtrace unreadable; restore
    // first, then let the default hook print into a working terminal.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen);
        hook(p);
    }));
    let backend = ratatui::backend::CrosstermBackend::new(out);
    let mut term = Terminal::new(backend).map_err(|e| e.to_string())?;

    if !question.trim().is_empty() {
        app.ask(question.trim().to_string(), false);
    }
    let res = app.event_loop(&mut term);

    terminal::disable_raw_mode().map_err(|e| e.to_string())?;
    execute!(term.backend_mut(), terminal::LeaveAlternateScreen).map_err(|e| e.to_string())?;
    let _ = term.show_cursor();
    let _ = std::io::stdout().flush();
    res
}

impl App {
    fn event_loop<B: ratatui::backend::Backend>(
        &mut self,
        term: &mut Terminal<B>,
    ) -> Result<i32, String> {
        loop {
            term.draw(|f| self.draw(f)).map_err(|e| e.to_string())?;
            self.tick += 1;
            // Poll rather than block: the worker has to be able to finish, and the clock in
            // the status bar has to keep moving, while nobody is touching the keyboard.
            if event::poll(Duration::from_millis(100)).map_err(|e| e.to_string())? {
                if let Event::Key(k) = event::read().map_err(|e| e.to_string())? {
                    if k.kind != event::KeyEventKind::Press {
                        continue;
                    }
                    if self.key(k) {
                        return Ok(0);
                    }
                }
            }
            self.collect();
        }
    }

    /// Returns true to quit.
    ///
    /// Modal input separates typing from source selection: insert types, normal drives.
    fn key(&mut self, k: KeyEvent) -> bool {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        self.note.clear();
        if ctrl && k.code == KeyCode::Char('c') {
            return true;
        }
        if self.cmd.is_some() {
            return self.key_cmd(k);
        }
        if self.find.is_some() {
            self.key_find(k);
            return false;
        }
        if self.insert {
            self.key_insert(k);
            return false;
        }
        self.key_normal(k)
    }

    fn key_insert(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc => self.insert = false,
            KeyCode::Enter if !self.input.trim().is_empty() => {
                let q = std::mem::take(&mut self.input);
                let follow = self.thread;
                self.insert = false;
                self.ask(q, follow);
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            // ^W and ^U, because every readline in the terminal has them and the fingers
            // that reach for them do not stop to ask whether this is vim.
            KeyCode::Char('w') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                let t = self.input.trim_end();
                self.input
                    .truncate(t.rfind(' ').map(|i| i + 1).unwrap_or(0));
            }
            KeyCode::Char('u') if k.modifiers.contains(KeyModifiers::CONTROL) => self.input.clear(),
            KeyCode::Char(c) => self.input.push(c),
            _ => {}
        }
    }

    fn key_normal(&mut self, k: KeyEvent) -> bool {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let half = (self.view / 2).max(1);
        // `gg` is the only two-key sequence, so one flag is the whole parser.
        let pending_g = std::mem::take(&mut self.pending_g);
        match k.code {
            KeyCode::Char('q') => return true,
            KeyCode::Char(':') => self.cmd = Some(String::new()),
            KeyCode::Char('/') => {
                if self.viewing.is_some() {
                    self.find = Some(String::new());
                } else {
                    self.note = "read a source before searching it".into();
                }
            }
            KeyCode::Char('i') | KeyCode::Char('a') => self.insert = true,
            KeyCode::Char('o') => match self.sel {
                Some(i) => open_source(&self.cfg, &self.list, i),
                None => self.note = "pick a source first: 1-9".into(),
            },
            KeyCode::Char('h') => {
                self.highlight = !self.highlight;
                self.note = format!("highlight: {}", if self.highlight { "on" } else { "off" });
            }
            KeyCode::Char('j') | KeyCode::Down => self.scroll = self.bottom().min(self.scroll + 1),
            KeyCode::Char('k') | KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Char('d') if ctrl => self.scroll = self.bottom().min(self.scroll + half),
            KeyCode::Char('u') if ctrl => self.scroll = self.scroll.saturating_sub(half),
            KeyCode::Char('f') if ctrl => {
                self.scroll = self.bottom().min(self.scroll + self.view.max(2) - 1)
            }
            KeyCode::Char('b') if ctrl => {
                self.scroll = self.scroll.saturating_sub(self.view.max(2) - 1)
            }
            KeyCode::PageDown => {
                self.scroll = self.bottom().min(self.scroll + self.view.max(2) - 1)
            }
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(self.view.max(2) - 1),
            KeyCode::Char('g') if pending_g => self.scroll = 0,
            KeyCode::Char('g') => self.pending_g = true,
            KeyCode::Char('G') => self.follow_view(),
            // ^N/^P and J/K walk search results without taking j/k away from article scrolling.
            KeyCode::Char('n') if ctrl => self.move_sel(1),
            KeyCode::Char('p') if ctrl => self.move_sel(-1),
            KeyCode::Char('J') => self.move_sel(1),
            KeyCode::Char('K') => self.move_sel(-1),
            KeyCode::Char('n') => self.find_again(false),
            KeyCode::Char('N') => self.find_again(true),
            KeyCode::Char(c @ '1'..='9') => {
                let i = c as usize - '1' as usize;
                if i < self.list.len() {
                    self.sel = Some(i);
                    self.viewing = None;
                    self.scroll = 0;
                    self.preview_selected();
                }
            }
            KeyCode::Enter => self.read_selected(),
            // Repeat at the current speed. Generation is grounded in the selected source;
            // clear selection with Esc to synthesize from the normal top results.
            KeyCode::Char('r') => {
                if !self.question.is_empty() {
                    let q = self.question.clone();
                    let focus = self
                        .cfg
                        .mode
                        .generates()
                        .then(|| self.sel.and_then(|i| self.list.get(i).cloned()))
                        .flatten();
                    self.start(q, self.thread, focus);
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => match self.cfg.mode.next() {
                Some(m) => self.set_mode(m),
                None => self.note = "already at molasses".into(),
            },
            KeyCode::Char('-') => match self.cfg.mode.prev() {
                Some(m) => self.set_mode(m),
                None => self.note = "already at ultrafast".into(),
            },
            // `>` and `<` widen and narrow the answer, the way they indent in vim.
            KeyCode::Char('>') => match self.cfg.len.next() {
                Some(l) => self.set_len(l),
                None => self.note = "already at max".into(),
            },
            KeyCode::Char('<') => match self.cfg.len.prev() {
                Some(l) => self.set_len(l),
                None => self.note = "already at low".into(),
            },
            KeyCode::Char('O') => {
                self.new_topic();
                self.insert = true;
            }
            // Cancel current work and stop a supervised model process if it is generating.
            KeyCode::Char('x') if ctrl => {
                if self.cancel_job() {
                    self.note = "cancelled".into();
                }
            }
            KeyCode::Esc => {
                self.sel = None;
                self.viewing = None;
                self.scroll = 0;
            }
            _ => {}
        }
        false
    }

    /// `:` commands. The speed modes read better as words than as keys you have to remember,
    /// and `:q` is muscle memory nobody should have to unlearn.
    fn key_cmd(&mut self, k: KeyEvent) -> bool {
        let Some(buf) = self.cmd.as_mut() else {
            return false;
        };
        match k.code {
            KeyCode::Esc => self.cmd = None,
            KeyCode::Backspace => {
                if buf.pop().is_none() {
                    self.cmd = None;
                }
            }
            KeyCode::Char(c) => buf.push(c),
            KeyCode::Enter => {
                let line = self.cmd.take().unwrap_or_default();
                let mut it = line.split_whitespace();
                let (verb, arg) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                match verb {
                    "q" | "quit" | "q!" => return true,
                    "new" => self.new_topic(),
                    "r" | "re" => {
                        if !self.question.is_empty() {
                            let q = self.question.clone();
                            self.ask(q, self.thread);
                        }
                    }
                    "open" => match arg.parse::<usize>() {
                        Ok(n) if n >= 1 && n <= self.list.len() => {
                            open_source(&self.cfg, &self.list, n - 1)
                        }
                        _ => self.note = format!("open 1-{}", self.list.len()),
                    },
                    m if Mode::parse(m).is_some() => {
                        if let Some(m) = Mode::parse(m) {
                            self.set_mode(m);
                        }
                    }
                    "len" => match Len::parse(arg) {
                        Some(l) => self.set_len(l),
                        None => self.note = "len low|medium|max".into(),
                    },
                    "model" if !arg.is_empty() => {
                        let cancelled = self.cancel_job();
                        self.cfg.model = resolve_model(arg);
                        self.note =
                            dial_note(cancelled, format!("model: {}", model_name(&self.cfg.model)));
                        save_prefs(&self.cfg);
                    }
                    "model" => self.note = format!("model: {}", model_name(&self.cfg.model)),
                    "" => {}
                    other => self.note = format!("no such command: {other}"),
                }
            }
            _ => {}
        }
        false
    }

    fn key_find(&mut self, k: KeyEvent) {
        let Some(buf) = self.find.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => self.find = None,
            KeyCode::Backspace => {
                if buf.pop().is_none() {
                    self.find = None;
                }
            }
            KeyCode::Char(c) => buf.push(c),
            KeyCode::Enter => {
                let query = self.find.take().unwrap_or_default();
                if !query.trim().is_empty() {
                    self.last_find = query;
                    self.find_line = None;
                }
                self.find_again(false);
            }
            _ => {}
        }
    }

    fn find_again(&mut self, reverse: bool) {
        let Some((_, text)) = &self.viewing else {
            return;
        };
        if self.last_find.is_empty() {
            self.note = "no previous search".into();
            return;
        }
        let Some(line) = next_match(text, &self.last_find, self.find_line, reverse) else {
            self.note = format!("pattern not found: {}", self.last_find);
            return;
        };
        self.scroll = 4 + text
            .lines()
            .take(line)
            .map(|line| wrapped(line, self.width))
            .sum::<u16>();
        self.find_line = Some(line);
        self.note = format!(
            "{} match: {}",
            if reverse { "previous" } else { "next" },
            self.last_find
        );
    }

    fn cancel_job(&mut self) -> bool {
        let generated = self
            .job
            .as_ref()
            .is_some_and(|(_, _, generated)| *generated);
        let cancelled = self.job.take().is_some();
        if generated {
            stop_chat(&self.cfg);
        }
        if cancelled {
            if let Ok(mut stage) = self.stage.lock() {
                *stage = String::from("cancelled");
            }
        }
        cancelled
    }

    fn set_mode(&mut self, m: Mode) {
        let cancelled = self.cancel_job();
        self.cfg.mode = m;
        self.note = dial_note(cancelled, format!("speed: {}", m.name()));
        save_prefs(&self.cfg);
    }

    fn set_len(&mut self, l: Len) {
        let cancelled = self.cancel_job();
        self.cfg.len = l;
        self.note = dial_note(cancelled, format!("length: {}", l.name()));
        save_prefs(&self.cfg);
    }

    fn new_topic(&mut self) {
        self.cancel_job();
        self.thread = false;
        self.turns.clear();
        self.list.clear();
        self.viewing = None;
        self.find = None;
        self.last_find.clear();
        self.find_line = None;
        self.previewed.clear();
        self.sel = None;
        self.scroll = 0;
        self.note = "new topic".into();
    }

    fn move_sel(&mut self, d: i32) {
        self.sel = move_index(self.sel, self.list.len(), d);
        self.viewing = None;
        self.find_line = None;
        self.preview_selected();
        self.scroll = 0;
    }

    fn preview_selected(&mut self) {
        let Some((i, source)) = self
            .sel
            .and_then(|i| self.list.get(i).map(|s| (i, s.clone())))
        else {
            return;
        };
        let key = (source.book.clone(), source.path.clone());
        if self.previewed.contains(&key) {
            return;
        }
        match article(&self.cfg.kiwix, &source.book, &source.path) {
            Ok(html) => {
                let passage = best_passage(&prose_text(&html), &self.question);
                if !passage.trim().is_empty() {
                    self.list[i].snip = passage;
                }
                self.previewed.insert(key);
            }
            Err(e) => self.note = e,
        }
    }

    fn read_selected(&mut self) {
        let Some((i, source)) = self
            .sel
            .and_then(|i| self.list.get(i).map(|s| (i, s.clone())))
        else {
            self.note = "pick a result first: 1-9".into();
            return;
        };
        match article(&self.cfg.kiwix, &source.book, &source.path) {
            Ok(html) => {
                let text = article_text(&html);
                if text.trim().is_empty() {
                    self.note = "source has no readable text".into();
                } else {
                    self.viewing = Some((i, text));
                    self.find_line = None;
                    self.scroll = 0;
                    self.note = "reading source · o opens browser".into();
                }
            }
            Err(e) => self.note = e,
        }
    }

    fn ask(&mut self, q: String, follow: bool) {
        self.start(q, follow, None);
    }

    fn start(&mut self, q: String, follow: bool, focus: Option<Source>) {
        if self.job.is_some() {
            self.note = "still answering — ^X to cancel".into();
            return;
        }
        if q != self.question {
            self.list.clear();
            self.previewed.clear();
            self.sel = None;
        }
        self.viewing = None;
        self.question = q.clone();
        self.follow_view();
        if let Ok(mut s) = self.stage.lock() {
            *s = String::from("starting");
        }
        let (tx, rx) = mpsc::channel();
        let cfg = self.cfg.clone();
        let generated = cfg.mode.generates();
        std::thread::spawn(move || {
            let _ = tx.send(answer_once(&cfg, &q, follow, focus.as_ref()));
        });
        self.job = Some((rx, Instant::now(), generated));
    }

    fn collect(&mut self) {
        let Some((rx, _, generated)) = &self.job else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(a)) => {
                let preview = !*generated;
                self.read = a.sources.len();
                self.turns.push(Turn {
                    q: self.question.clone(),
                    a: a.text,
                    cites: cite_lines(&a.sources),
                    generated: *generated,
                });
                self.list = if a.shortlist.is_empty() {
                    a.sources
                } else {
                    a.shortlist
                };
                self.previewed.clear();
                self.sel = (!self.list.is_empty()).then_some(0);
                self.viewing = None;
                self.thread = true;
                self.job = None;
                if preview {
                    self.preview_selected();
                }
                self.follow_view();
            }
            Ok(Err(e)) => {
                self.turns.push(Turn {
                    q: self.question.clone(),
                    a: format!("tny: {e}"),
                    cites: vec![],
                    generated: *generated,
                });
                self.list.clear();
                self.job = None;
                self.follow_view();
            }
            Err(mpsc::TryRecvError::Disconnected) => self.job = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let sources_h = if self.list.is_empty() {
            0
        } else {
            (self.list.len().min(8) + 2) as u16
        };
        let [top, mid, bottom] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(sources_h),
            Constraint::Length(3),
        ])
        .areas(area);

        self.draw_answer(f, top);
        if sources_h > 0 {
            self.draw_sources(f, mid);
        }
        self.draw_input(f, bottom);
    }

    /// Search results show a selected snippet first. Generated turns keep transcript history;
    /// reading a source replaces preview with its cleaned full text.
    fn draw_answer(&mut self, f: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(2).max(8);
        self.width = width;
        let lines = self.transcript();
        let model = if self.cfg.mode.generates() {
            model_name(&self.cfg.model)
        } else {
            "no model".into()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(DIM)
            .title(Span::styled(
                " tny ",
                Style::new().add_modifier(Modifier::BOLD),
            ))
            .title_bottom(Line::from(vec![
                Span::styled(
                    format!(" {} ", self.cfg.mode.name()),
                    Style::new().fg(mode_colour(self.cfg.mode)),
                ),
                Span::styled(format!("· {} ", self.cfg.len.name()), DIM),
                Span::styled(format!("· {model} "), DIM),
            ]));
        // Wrapped height, so `follow_view` can pin the newest turn to the bottom and PageUp
        // cannot scroll past the end of the conversation.
        let height = lines
            .iter()
            .map(|l| wrapped(&l.to_string(), width))
            .sum::<u16>();
        let view = area.height.saturating_sub(2);
        let scroll = self.scroll.min(height.saturating_sub(view));
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((scroll, 0))
                .block(block),
            area,
        );
        self.height = height;
        self.view = view;
        self.scroll = scroll;
    }

    fn transcript(&self) -> Vec<Line<'_>> {
        if let Some((i, text)) = &self.viewing {
            let Some(source) = self.list.get(*i) else {
                return vec![];
            };
            let mut out = vec![
                Line::styled("SOURCE", DIM),
                Line::styled(
                    source.title.as_str(),
                    Style::new().add_modifier(Modifier::BOLD),
                ),
                Line::styled(human_book(&source.book), DIM),
                Line::raw(""),
            ];
            let query = if self.last_find.is_empty() {
                &self.question
            } else {
                &self.last_find
            };
            out.extend(highlighted_lines(text, query, self.highlight));
            return out;
        }

        if self.turns.last().is_some_and(|t| !t.generated) && !self.list.is_empty() {
            let mut out = vec![
                Line::styled("QUERY", DIM),
                Line::styled(
                    self.question.as_str(),
                    Style::new().add_modifier(Modifier::BOLD),
                ),
                Line::raw(""),
            ];
            match self.sel.and_then(|i| self.list.get(i).map(|s| (i, s))) {
                Some((i, source)) => {
                    out.push(Line::styled(
                        format!("RESULT {} OF {}", i + 1, self.list.len()),
                        Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ));
                    out.push(Line::styled(
                        source.title.as_str(),
                        Style::new().add_modifier(Modifier::BOLD),
                    ));
                    out.push(Line::styled(human_book(&source.book), DIM));
                    out.push(Line::raw(""));
                    out.extend(highlighted_lines(
                        &source.snip,
                        &self.question,
                        self.highlight,
                    ));
                    out.push(Line::raw(""));
                    out.push(Line::styled(
                        "Enter reads this source here · o opens it in browser · J/K changes result",
                        DIM,
                    ));
                }
                None => out.push(Line::styled(
                    format!(
                        "{} ranked results · press 1-9 or J/K to select",
                        self.list.len()
                    ),
                    DIM,
                )),
            }
            return out;
        }

        let mut out: Vec<Line> = vec![];
        for t in self.turns.iter().filter(|t| t.generated) {
            if !out.is_empty() {
                out.push(Line::raw(""));
            }
            out.push(Line::styled(
                format!("› {}", t.q),
                Style::new().add_modifier(Modifier::BOLD),
            ));
            out.push(Line::raw(""));
            out.extend(t.a.lines().map(Line::raw));
            if !t.cites.is_empty() {
                out.push(Line::raw(""));
                out.extend(t.cites.iter().map(|c| Line::styled(format!("  {c}"), DIM)));
            }
        }
        if self.job.is_some() {
            if !out.is_empty() {
                out.push(Line::raw(""));
            }
            out.push(Line::styled(
                format!("› {}", self.question),
                Style::new().add_modifier(Modifier::BOLD),
            ));
        } else if out.is_empty() {
            out.push(Line::styled(
                "Search local knowledge. Results and sources stay offline; generation is optional.",
                DIM,
            ));
        }
        out
    }

    fn bottom(&self) -> u16 {
        self.height.saturating_sub(self.view)
    }

    /// Pin the view to the newest turn. Called when a question starts and when it lands;
    /// PageUp is then free to go back without being yanked forward on the next redraw.
    fn follow_view(&mut self) {
        self.scroll = u16::MAX;
    }

    fn draw_sources(&mut self, f: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(6) as usize;
        let items: Vec<ListItem> = self
            .list
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mark = if i < self.read { "·" } else { " " };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{mark}{:>2} ", i + 1), DIM),
                    Span::raw(clip(
                        &format!("{} · {}", s.title, human_book(&s.book)),
                        width,
                    )),
                ]))
            })
            .collect();
        let mut state = ListState::default();
        state.select(self.sel);
        f.render_stateful_widget(
            List::new(items)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(DIM)
                        .title(Span::styled(" results ", DIM)),
                )
                .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan)),
            area,
            &mut state,
        );
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let hint = if self.insert {
            "⏎ search · esc normal"
        } else if self.viewing.is_some() {
            "/ find · n/N match · jk scroll · ⏎ reread · o browser · q quit"
        } else {
            "i search · JK results · ⏎ read · o browser · h highlight · +r generate · q quit"
        };
        let right = match &self.job {
            Some((_, t0, _)) => {
                let stage = self.stage.lock().map(|s| s.clone()).unwrap_or_default();
                format!(
                    " {} {stage} {:.0}s ",
                    FRAMES[self.tick % 10],
                    t0.elapsed().as_secs_f64()
                )
            }
            None if !self.note.is_empty() => format!(" {} ", self.note),
            None => format!(" {hint} "),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(DIM)
            .title_bottom(Line::from(Span::styled(right, DIM)).right_aligned());
        // The line reads like vim's: what mode you are in on the left, what is happening on
        // the right, and the text you are typing in between.
        let (tag, tag_style, body) = if let Some(c) = &self.cmd {
            (String::new(), DIM, format!(":{c}"))
        } else if let Some(c) = &self.find {
            (String::new(), DIM, format!("/{c}"))
        } else if self.insert {
            (
                String::from("-- INSERT -- "),
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                self.input.clone(),
            )
        } else {
            (
                String::new(),
                DIM,
                if self.input.is_empty() {
                    String::new()
                } else {
                    self.input.clone()
                },
            )
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(tag.clone(), tag_style),
                Span::raw(body.clone()),
            ]))
            .block(block),
            area,
        );
        // Terminal cursor, not a drawn block: it blinks, and it is where the terminal's own
        // IME and paste land.
        if self.insert || self.cmd.is_some() || self.find.is_some() {
            let x = area.x + 1 + (tag.chars().count() + body.chars().count()) as u16;
            f.set_cursor_position((x.min(area.x + area.width - 2), area.y + 1));
        }
    }
}

fn move_index(selected: Option<usize>, len: usize, delta: i32) -> Option<usize> {
    if len == 0 {
        return None;
    }
    Some(match selected {
        None if delta > 0 => 0,
        None => len - 1,
        Some(i) => (i as i32 + delta).clamp(0, len as i32 - 1) as usize,
    })
}

fn next_match(text: &str, needle: &str, current: Option<usize>, reverse: bool) -> Option<usize> {
    let needle = needle.to_lowercase();
    let matches = |line: &&str| line.to_lowercase().contains(&needle);
    if reverse {
        let mut before = None;
        let mut last = None;
        for (i, line) in text.lines().enumerate() {
            if line.to_lowercase().contains(&needle) {
                last = Some(i);
                if current.is_none_or(|at| i < at) {
                    before = Some(i);
                }
            }
        }
        before.or(last)
    } else {
        text.lines()
            .enumerate()
            .find(|(i, line)| current.is_none_or(|at| *i > at) && matches(line))
            .or_else(|| text.lines().enumerate().find(|(_, line)| matches(line)))
            .map(|(i, _)| i)
    }
}

fn highlighted_lines<'a>(text: &'a str, query: &str, enabled: bool) -> Vec<Line<'a>> {
    if !enabled {
        return text.lines().map(Line::raw).collect();
    }
    let query = prep(query);
    let terms: Vec<&str> = query.split_whitespace().collect();
    text.lines()
        .map(|line| {
            let mut spans = Vec::new();
            let mut start = 0;
            let mut in_word = false;
            for (i, c) in line.char_indices() {
                let is_word = c.is_alphanumeric() || matches!(c, '-' | '_');
                if is_word && !in_word {
                    if start < i {
                        spans.push(Span::raw(&line[start..i]));
                    }
                    start = i;
                    in_word = true;
                } else if !is_word && in_word {
                    let word = &line[start..i];
                    if terms.iter().any(|term| word.eq_ignore_ascii_case(term)) {
                        spans.push(Span::styled(
                            word,
                            Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                        ));
                    } else {
                        spans.push(Span::raw(word));
                    }
                    start = i;
                    in_word = false;
                }
            }
            if start < line.len() {
                let tail = &line[start..];
                if in_word && terms.iter().any(|term| tail.eq_ignore_ascii_case(term)) {
                    spans.push(Span::styled(
                        tail,
                        Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::raw(tail));
                }
            }
            Line::from(spans)
        })
        .collect()
}

fn clip(s: &str, width: usize) -> String {
    let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= width {
        return one_line;
    }
    match width {
        0 => String::new(),
        1 => "…".into(),
        _ => format!("{}…", one_line.chars().take(width - 1).collect::<String>()),
    }
}

fn dial_note(cancelled: bool, note: String) -> String {
    if cancelled {
        format!("cancelled · {note}")
    } else {
        note
    }
}

#[cfg(test)]
mod tests {
    use super::{clip, dial_note, highlighted_lines, move_index, next_match};
    use ratatui::style::Color;

    #[test]
    fn result_snippet_is_one_bounded_line() {
        assert_eq!(clip("one\n  two three", 20), "one two three");
        assert_eq!(clip("one two three", 8), "one two…");
        assert_eq!(clip("éclair", 3), "éc…");
    }

    #[test]
    fn dial_change_reports_cancelled_job() {
        assert_eq!(
            dial_note(true, "speed: fast".into()),
            "cancelled · speed: fast"
        );
        assert_eq!(dial_note(false, "length: max".into()), "length: max");
    }

    #[test]
    fn highlighting_preserves_text_and_can_be_disabled() {
        let on = highlighted_lines("Love is patient.", "What is love?", true);
        assert_eq!(on[0].to_string(), "Love is patient.");
        assert_eq!(on[0].spans[0].style.fg, Some(Color::Yellow));

        let off = highlighted_lines("Love is patient.", "What is love?", false);
        assert_eq!(off[0].to_string(), "Love is patient.");
        assert_eq!(off[0].spans[0].style.fg, None);
    }

    #[test]
    fn result_navigation_stops_at_both_ends() {
        assert_eq!(move_index(Some(7), 8, 1), Some(7));
        assert_eq!(move_index(Some(0), 8, -1), Some(0));
        assert_eq!(move_index(None, 8, 1), Some(0));
        assert_eq!(move_index(None, 8, -1), Some(7));
    }

    #[test]
    fn article_search_moves_and_wraps_in_both_directions() {
        let text = "alpha\nneedle one\nmiddle\nneedle two";
        assert_eq!(next_match(text, "NEEDLE", None, false), Some(1));
        assert_eq!(next_match(text, "needle", Some(1), false), Some(3));
        assert_eq!(next_match(text, "needle", Some(3), false), Some(1));
        assert_eq!(next_match(text, "needle", Some(1), true), Some(3));
        assert_eq!(next_match(text, "absent", None, false), None);
    }
}

/// How many terminal rows a logical line occupies once wrapped. Ratatui can only report this
/// behind an unstable feature, and the answer text is plain prose, so count it here.
fn wrapped(line: &str, width: u16) -> u16 {
    let cols = line.chars().count().max(1) as u16;
    cols.div_ceil(width.max(1))
}

fn mode_colour(m: Mode) -> Color {
    match m {
        Mode::Ultrafast => Color::Green,
        Mode::Fast => Color::Cyan,
        Mode::Medium => Color::Blue,
        Mode::Slow => Color::Yellow,
        Mode::Molasses => Color::Red,
    }
}
