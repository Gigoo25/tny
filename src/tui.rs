//! F96: the terminal UI.
//!
//! The line-based prompt it replaces had three problems that are not fixable with escape
//! codes: a 20-90 s answer left the screen frozen with a spinner and no way to change your
//! mind, steering competed with typing for the digit keys, and every answer scrolled the
//! previous one away, so "which of these two did it say?" meant scrolling back through
//! spinner frames. A framed layout fixes all three at once — the answer holds still and
//! scrolls on its own, the shortlist is always visible and selectable with arrows, and the
//! status bar can carry the stage, the clock and the mode without stealing a line from the
//! answer.
//!
//! Answering runs on a worker thread. The UI redraws on a 100 ms clock whatever the worker
//! is doing, which is the whole point: `Esc` still quits while the model is 40 s into a
//! molasses answer.

use crate::{
    answer_once, cite_lines, human_book, model_name, open_source, resolve_model, save_prefs, Answered, Cfg,
    Len, Mode, Source,
};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::crossterm::{execute, terminal};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io::Write;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const DIM: Style = Style::new().fg(Color::DarkGray);

/// One question and what came back.
struct Turn {
    q: String,
    a: String,
    cites: Vec<String>,
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
    /// How many of `list` the answer was actually built from — the rest were retrieved and
    /// not read, which is exactly what steering is for.
    read: usize,
    sel: Option<usize>,
    scroll: u16,
    stage: Arc<Mutex<String>>,
    /// Some while a worker thread is answering.
    job: Option<(Receiver<Result<Answered, String>>, Instant)>,
    /// A follow-up carries the previous turn; a new topic does not.
    thread: bool,
    note: String,
    tick: usize,
    /// Vim's two states. Insert is the default at a cold start, because the first thing a
    /// user does is type a question; a seeded question from the command line starts in
    /// normal, where the answer is already there to be driven.
    insert: bool,
    /// `Some` while a `:` line is being typed.
    cmd: Option<String>,
    pending_g: bool,
    /// Wrapped height of the transcript and of its viewport, both measured during the last
    /// draw — scrolling is clamped against them so the keys cannot leave the text.
    height: u16,
    view: u16,
}

pub fn run(cfg: &Cfg, question: &str) -> Result<i32, String> {
    let stage = Arc::new(Mutex::new(String::from("ready")));
    let mut app = App {
        cfg: Cfg { progress: Some(stage.clone()), ..cfg.clone() },
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
        tick: 0,
        insert: question.trim().is_empty(),
        cmd: None,
        pending_g: false,
        height: 0,
        view: 0,
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
    fn event_loop<B: ratatui::backend::Backend>(&mut self, term: &mut Terminal<B>) -> Result<i32, String> {
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
    /// F98: modal, because this is a terminal tool and the alternative is worse. A single
    /// mode has to choose between digits that type and digits that select, and the CLI
    /// prompt it replaces already lost that fight. Insert types, normal drives.
    fn key(&mut self, k: KeyEvent) -> bool {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        self.note.clear();
        if ctrl && k.code == KeyCode::Char('c') {
            return true;
        }
        if self.cmd.is_some() {
            return self.key_cmd(k);
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
                self.input.truncate(t.rfind(' ').map(|i| i + 1).unwrap_or(0));
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
            KeyCode::Char('i') | KeyCode::Char('a') => self.insert = true,
            // `o` is vim's "open a line and type": here, a new topic and a cursor.
            KeyCode::Char('o') => {
                self.new_topic();
                self.insert = true;
            }
            KeyCode::Char('j') | KeyCode::Down => self.scroll = self.bottom().min(self.scroll + 1),
            KeyCode::Char('k') | KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Char('d') if ctrl => self.scroll = self.bottom().min(self.scroll + half),
            KeyCode::Char('u') if ctrl => self.scroll = self.scroll.saturating_sub(half),
            KeyCode::Char('f') if ctrl => self.scroll = self.bottom().min(self.scroll + self.view.max(2) - 1),
            KeyCode::Char('b') if ctrl => self.scroll = self.scroll.saturating_sub(self.view.max(2) - 1),
            KeyCode::PageDown => self.scroll = self.bottom().min(self.scroll + self.view.max(2) - 1),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(self.view.max(2) - 1),
            KeyCode::Char('g') if pending_g => self.scroll = 0,
            KeyCode::Char('g') => self.pending_g = true,
            KeyCode::Char('G') => self.follow_view(),
            // ^N/^P walk the shortlist — vim's own keys for moving through a completion
            // menu, which is exactly what this list is.
            KeyCode::Char('n') if ctrl => self.move_sel(1),
            KeyCode::Char('p') if ctrl => self.move_sel(-1),
            KeyCode::Char('J') => self.move_sel(1),
            KeyCode::Char('K') => self.move_sel(-1),
            // A digit picks a source outright: no list navigation for the common case, and
            // in normal mode there is nothing for it to collide with.
            KeyCode::Char(c @ '1'..='9') => {
                let i = c as usize - '1' as usize;
                if i < self.list.len() {
                    self.sel = Some(i);
                }
            }
            // Enter reads the selected source; the steer.
            KeyCode::Enter => {
                if let (Some(i), false) = (self.sel, self.question.is_empty()) {
                    let focus = self.list.get(i).cloned();
                    let q = self.question.clone();
                    self.sel = None;
                    self.start(q, self.thread, focus);
                }
            }
            // Repeat the last question at the current speed — the "read that again, deeper"
            // gesture, and the reason `+` and `-` exist next to it.
            KeyCode::Char('r') => {
                if !self.question.is_empty() {
                    let q = self.question.clone();
                    self.ask(q, self.thread);
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
            KeyCode::Char('O') => match self.sel {
                Some(i) => open_source(&self.cfg, &self.list, i),
                None => self.note = "pick a source first: 1-9".into(),
            },
            // A question in flight is worth abandoning: the worker is detached and its answer
            // still lands in the cache, so it costs nothing the next time it is asked.
            KeyCode::Char('x') if ctrl => {
                if self.job.take().is_some() {
                    self.note = "cancelled".into();
                }
            }
            KeyCode::Esc => self.sel = None,
            _ => {}
        }
        false
    }

    /// `:` commands. The speed modes read better as words than as keys you have to remember,
    /// and `:q` is muscle memory nobody should have to unlearn.
    fn key_cmd(&mut self, k: KeyEvent) -> bool {
        let Some(buf) = self.cmd.as_mut() else { return false };
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
                        Ok(n) if n >= 1 && n <= self.list.len() => open_source(&self.cfg, &self.list, n - 1),
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
                    // Switching model restarts llama-server on the next question — which is
                    // why it is a typed command and not a key you can lean on.
                    "model" if !arg.is_empty() => {
                        self.cfg.model = resolve_model(arg);
                        self.note = format!("model: {}", model_name(&self.cfg.model));
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

    /// Both dials are written through to disk as they change: the next run of `tny` starts
    /// where this one left off, which is the only behaviour that makes a setting worth having.
    fn set_mode(&mut self, m: Mode) {
        self.cfg.mode = m;
        self.note = format!("speed: {}", m.name());
        save_prefs(&self.cfg);
    }

    fn set_len(&mut self, l: Len) {
        self.cfg.len = l;
        self.note = format!("length: {}", l.name());
        save_prefs(&self.cfg);
    }

    fn new_topic(&mut self) {
        self.thread = false;
        self.turns.clear();
        self.list.clear();
        self.sel = None;
        self.scroll = 0;
        self.note = "new topic".into();
    }

    fn move_sel(&mut self, d: i32) {
        if self.list.is_empty() {
            // No shortlist yet: the arrows are the answer's scrollbar instead of dead keys.
            self.scroll =
                if d < 0 { self.scroll.saturating_sub(1) } else { self.bottom().min(self.scroll + 1) };
            return;
        }
        let n = self.list.len() as i32;
        self.sel = match self.sel {
            None if d > 0 => Some(0),
            None => Some((n - 1) as usize),
            Some(i) => {
                let next = i as i32 + d;
                if next < 0 || next >= n {
                    None
                } else {
                    Some(next as usize)
                }
            }
        };
    }

    fn ask(&mut self, q: String, follow: bool) {
        self.start(q, follow, None);
    }

    fn start(&mut self, q: String, follow: bool, focus: Option<Source>) {
        if self.job.is_some() {
            self.note = "still answering — ^X to cancel".into();
            return;
        }
        self.question = q.clone();
        self.follow_view();
        if let Ok(mut s) = self.stage.lock() {
            *s = String::from("starting");
        }
        let (tx, rx) = mpsc::channel();
        let cfg = self.cfg.clone();
        std::thread::spawn(move || {
            let _ = tx.send(answer_once(&cfg, &q, follow, focus.as_ref()));
        });
        self.job = Some((rx, Instant::now()));
    }

    fn collect(&mut self) {
        let Some((rx, _)) = &self.job else { return };
        match rx.try_recv() {
            Ok(Ok(a)) => {
                self.read = a.sources.len();
                self.turns.push(Turn {
                    q: self.question.clone(),
                    a: a.text,
                    cites: cite_lines(&a.sources),
                });
                self.list = if a.shortlist.is_empty() { a.sources } else { a.shortlist };
                self.thread = true;
                self.job = None;
                self.follow_view();
            }
            Ok(Err(e)) => {
                self.turns.push(Turn { q: self.question.clone(), a: format!("tny: {e}"), cites: vec![] });
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
        let sources_h = if self.list.is_empty() { 0 } else { (self.list.len().min(8) + 2) as u16 };
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

    /// The transcript. Every turn keeps its question above it, because three answers deep
    /// "it" and "that" only mean something next to what they referred to.
    fn draw_answer(&mut self, f: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(2).max(8);
        let lines = self.transcript();
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(DIM)
            .title(Span::styled(" tny ", Style::new().add_modifier(Modifier::BOLD)))
            .title_bottom(Line::from(vec![
                Span::styled(format!(" {} ", self.cfg.mode.name()), Style::new().fg(mode_colour(self.cfg.mode))),
                Span::styled(format!("· {} ", self.cfg.len.name()), DIM),
                Span::styled(format!("· {} ", model_name(&self.cfg.model)), DIM),
            ]));
        // Wrapped height, so `follow_view` can pin the newest turn to the bottom and PageUp
        // cannot scroll past the end of the conversation.
        self.height = lines.iter().map(|l| wrapped(&l.to_string(), width)).sum::<u16>();
        self.view = area.height.saturating_sub(2);
        self.scroll = self.scroll.min(self.height.saturating_sub(self.view));
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((self.scroll, 0)).block(block),
            area,
        );
    }

    fn transcript(&self) -> Vec<Line<'static>> {
        if self.turns.is_empty() && self.job.is_none() {
            return vec![Line::styled(
                "Ask anything. Every answer comes from the ZIM corpora on this disk — no network, no API key.",
                DIM,
            )];
        }
        let mut out: Vec<Line> = vec![];
        for (i, t) in self.turns.iter().enumerate() {
            if i > 0 {
                out.push(Line::raw(""));
            }
            out.push(Line::styled(format!("› {}", t.q), Style::new().add_modifier(Modifier::BOLD)));
            out.push(Line::raw(""));
            out.extend(t.a.lines().map(|l| Line::raw(l.to_string())));
            if !t.cites.is_empty() {
                out.push(Line::raw(""));
                out.extend(t.cites.iter().map(|c| Line::styled(format!("  {c}"), DIM)));
            }
        }
        // The question in flight sits where its answer will appear, so the transcript does
        // not jump when it lands.
        if self.job.is_some() {
            if !out.is_empty() {
                out.push(Line::raw(""));
            }
            out.push(Line::styled(format!("› {}", self.question), Style::new().add_modifier(Modifier::BOLD)));
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
        let items: Vec<ListItem> = self
            .list
            .iter()
            .enumerate()
            .map(|(i, s)| {
                // `·` marks what the answer was actually built from. Everything below it was
                // retrieved and passed over, and is one Enter away from being read.
                let mark = if i < self.read { "·" } else { " " };
                let style = if i < self.read { Style::new() } else { DIM };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{mark}{:>2} ", i + 1), DIM),
                    Span::styled(format!("{} · {}", human_book(&s.book), s.title), style),
                ]))
            })
            .collect();
        let mut state = ListState::default();
        state.select(self.sel);
        f.render_stateful_widget(
            List::new(items)
                .block(Block::default().borders(Borders::ALL).border_style(DIM).title(Span::styled(
                    if self.sel.is_some() { " ⏎ read this one " } else { " sources " },
                    DIM,
                )))
                .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan)),
            area,
            &mut state,
        );
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let hint = if self.insert {
            "⏎ ask · esc normal"
        } else {
            "i ask · jk scroll · 1-9 ⏎ source · r repeat · +- speed · <> length · : cmd · q quit"
        };
        let right = match &self.job {
            Some((_, t0)) => {
                let stage = self.stage.lock().map(|s| s.clone()).unwrap_or_default();
                format!(" {} {stage} {:.0}s ", FRAMES[self.tick % 10], t0.elapsed().as_secs_f64())
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
        let (tag, tag_style, body) = match &self.cmd {
            Some(c) => (String::new(), DIM, format!(":{c}")),
            None if self.insert => (
                String::from("-- INSERT -- "),
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                self.input.clone(),
            ),
            None => (String::new(), DIM, if self.input.is_empty() { String::new() } else { self.input.clone() }),
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(tag.clone(), tag_style), Span::raw(body.clone())]))
                .block(block),
            area,
        );
        // Terminal cursor, not a drawn block: it blinks, and it is where the terminal's own
        // IME and paste land.
        if self.insert || self.cmd.is_some() {
            let x = area.x + 1 + (tag.chars().count() + body.chars().count()) as u16;
            f.set_cursor_position((x.min(area.x + area.width - 2), area.y + 1));
        }
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
