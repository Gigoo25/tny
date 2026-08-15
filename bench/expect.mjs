// Answer-shaped ground truth: does a free-text ANSWER carry the fact.
//
// The fixtures' `needleRe` matches the *article's* prose, which is the wrong shape for grading
// output — the article says "115 known moons", a correct answer says "Jupiter has 115 moons".
// Each regex here is authored against the source article's fact, accepts paraphrase, and
// rejects the plausible wrong answer. Indexed by fixture and case order.
//
// `fixture-ambiguous.mjs` carries its own inline `expectRe` per case (authored with a stricter
// harness that also tests each regex against the *sibling* article of the word-sense pair), so
// it is absent here and `bench/answer-cli.mjs` prefers the inline one.
//
// Rule followed throughout: judge the fact, never the wording, and never widen a pattern just
// to admit the answer the model happens to give today.

export const EXPECT = {
  instructional: [
    // 0 hostname: the persistent mechanism, not the volatile `hostname foo`
    /hostnamectl|\/etc\/hostname/i,
    // 1 must be `enable`, not `start` — `start` alone does not survive a reboot. The verb has
    // to sit next to the tool or `--now`, or it just matches the question restated.
    /systemctl\s+enable|enable[\s\S]{0,40}--now|--now[\s\S]{0,40}enable/i,
    // 2 -m/--create-home is the load-bearing option
    /useradd[\s\S]{0,60}(-m\b|--create-home)|(-m\b|--create-home)[\s\S]{0,60}useradd/i,
    /resize2fs/i,
    /UUID=/i,
    /locale[-.]gen/i,
    // 6 wheel group or the sudoers mechanism
    /wheel|visudo|sudoers/i,
    // 7 --soft keeps the changes the question asks to keep; --hard destroys them, and HEAD~3
    // drops three commits instead of one. Both are rejected outright.
    /^(?![\s\S]*--hard)(?![\s\S]*HEAD~[2-9])[\s\S]*reset[\s\S]{0,30}(--soft|--mixed)/i,
    /--amend/i,
    /docker\s+tag/i,
    /CREATE\s+ROLE[\s\S]{0,80}PASSWORD/i,
    /python3?\s+-m\s+venv|virtualenv/i,
    // 12 unwrap_or_default returns T::default() when the Option is None
    /default[\s\S]{0,20}value|T::default|Default::default|default\(\)/i,
    // 13 the form is ${parameter:-word}; "a colon command" is not it
    /\$\{[^}]*:-|:-\s*word|parameter expansion/i,
    // 14 shorthand for --pretty=oneline --abbrev-commit: one commit per line
    /--pretty=oneline|abbrev-commit|(single|one) line/i,
    /skipkeys|ensure_ascii/i,
    // 16 the actual documented limits, not "arbitrary precision" hand-waving
    /NUMERIC\s*\(|precision\s*,\s*scale|131072|16383/i,
    // 17 removes the container when it exits
    /remove[\s\S]{0,60}(exit|terminat|finish|stop)/i,
  ],

  qa: [
    // 0 the execute bit — not ownership, not a missing shebang
    /execute\s+(permission|bit)|chmod\s*\+x|\bx\s+permission/i,
    // 1 nothing is listening: no sshd, wrong port, or a firewall
    /(no|not|isn.t)[\s\S]{0,40}(sshd|ssh daemon|listening|running)|nothing[\s\S]{0,20}listening|wrong port|firewall|port[\s\S]{0,20}closed/i,
    /lsof|fuser/i,
    // 3 CRLF line endings, or a missing `do` — never "a missing quote mark"
    /\\r|carriage return|CRLF|dos2unix|tr\s+-d|line ending|missing\s+`?do`?\b/i,
    // 4 127 means specifically that the command was not found
    /command\s+(cannot be |not )found|not found/i,
    /ssh-keygen[\s\S]{0,30}-R|known_hosts/i,
    // 6 a process still holds the deleted file open — the fix is finding or restarting it.
    // "check inode count with tune2fs to identify if the deletion caused a process to open
    // the file" is backwards and gets no credit: `open` alone is not the mechanism.
    /\blsof\b|fuser|still\s+(open|held|holding)|(open|file)\s+(file\s+)?(handle|descriptor)|holds?\s+(it\s+)?open|restart[\s\S]{0,30}(the\s+)?process/i,
    // 7 ARG_MAX: use find -delete/-exec or xargs
    /find[\s\S]{0,40}(-delete|-exec)|xargs|ARG_MAX|too many arguments/i,
    // 8 inodes or the directory index are exhausted, not the blocks
    /inode|dir_index|directory index|hash collision/i,
    // 9 the shell cached the old path
    /hash\s+-[dr]\b|hash(ed)?\s+(table|cache)|rehash|shell[\s\S]{0,30}cache/i,
    // 10 by-design clarify prompt; the fact it would need is the shared inode
    /inode/i,
    /replace[\s\S]{0,40}(process|shell|image)|in place/i,
    // 12 filenames may contain newlines and spaces, so the output is unparseable
    /newline|special character|space[\s\S]{0,30}(filename|name)|arbitrary character|unpredictab/i,
    // 13 by-design clarify prompt; SIGKILL cannot be caught
    /(can.?t|cannot|unable to)\s+be\s+(caught|ignored|handled)|uncatchable/i,
  ],

  general: [
    // 0 the article says 115; "95" and "99" are the observed fabrications
    /\b115\b/,
    /\b104\b/,
    // 2 5,895 m, equivalently ~19,341 ft. "19,710 feet" is a different mountain.
    /5,?895|19,?3\d\d/,
    /\b1989\b/,
    /NaCl|sodium chloride/i,
    /Fleming|\b1928\b/i,
    /Leonardo|da Vinci/i,
    /42\.195|26\.2/,
    // 8 any genuinely correct mechanism, not "reflection off the oceans"
    /calvin|light[- ]independent|chlorophyll|thylakoid|carbon dioxide[\s\S]{0,60}(water|glucose|sugar)/i,
    /moon[\s\S]{0,60}gravit|gravit[\s\S]{0,60}moon/i,
    // 10 Rayleigh scattering, or shorter wavelengths scattering more
    /rayleigh|scatter/i,
    /lithosphere|tectonic plate|seven or eight|plates?[\s\S]{0,30}mov/i,
    /immune\s+(system|response)|antigen|antibod|memory\s+(cell|B)/i,
    // 13 the rate is the fact; "supine versus prone" does not answer how to perform it
    /(100|120)[\s\S]{0,40}(compress|per minute|minute)|30:2|chest compression[\s\S]{0,40}\d/i,
    // 14 the C:N ratio, layering, turning, or a named aerated method. A bare `aerat` stem
    // would pass on the word alone; the method has to be named.
    /25:1|carbon[- ]to[- ]nitrogen|green[\s\S]{0,30}brown|aerated static|forced aeration|turn(ing)? the pile/i,
    // 15 rooting hormone, or striking the cutting in a medium
    /hormone|rooting|root[\s\S]{0,40}(medium|soil|water)|vegetative/i,
  ],
};
