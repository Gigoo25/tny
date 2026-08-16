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
    // 1 must be `enable`, not `start` — `start` alone does not survive a reboot. The tool has
    // to be named: "use the `enable` command with `--now`" names nothing runnable, asserts a
    // requirement `--now` does not have, and is the question restated.
    /systemctl[\s\S]{0,20}enable|enable[\s\S]{0,20}systemctl/i,
    // 2 -m/--create-home is the load-bearing option
    /useradd[\s\S]{0,60}(-m\b|--create-home)|(-m\b|--create-home)[\s\S]{0,60}useradd/i,
    /resize2fs/i,
    /UUID=/i,
    /locale[-.]gen/i,
    // 6 the wheel group, or a well-formed sudoers entry. The bare word `sudoers` is not
    // evidence — the question invites it, and `archie hostname = (ALL:ALL)` is not a rule.
    /wheel|visudo|ALL\s*=\s*\(/i,
    // 7 --soft keeps the changes the question asks to keep; HEAD~3 drops three commits instead
    // of one. Both are rejected — but only `reset --hard` as the *instruction*: an answer that
    // names --hard to warn against it is the safest answer there is.
    /^(?![\s\S]*reset\s+--hard)(?![\s\S]*HEAD~[2-9])[\s\S]*reset[\s\S]{0,30}(--soft|--mixed)/i,
    /--amend/i,
    /docker\s+tag/i,
    /CREATE\s+ROLE[\s\S]{0,80}PASSWORD/i,
    /python3?\s+-m\s+venv|virtualenv/i,
    // 12 unwrap_or_default returns T::default() when the Option is None
    /default[\s\S]{0,20}value|T::default|Default::default|default\(\)/i,
    // 13 the form is ${parameter:-word}; "a colon command" is not it
    /\$\{[^}]*:[-=]|:[-=]\s*word|parameter expansion/i,
    // 14 shorthand for --pretty=oneline --abbrev-commit: one commit per line. "single-line" is
    // the fact and a hyphen must not decide the verdict; "disables tab expansion" is the
    // recorded fabrication for this case (F92) and is what rejects it.
    /^(?![\s\S]*tab expansion)[\s\S]*(--pretty=oneline|abbrev-commit|(single|one)[- ]line)/i,
    // 15 any of the documented keyword arguments. skipkeys/ensure_ascii are two of twelve, and
    // naming indent/sort_keys/allow_nan/cls answers the question just as well.
    /skipkeys|ensure_ascii|sort_keys|check_circular|allow_nan|separators|\bindent\b|\bcls\b/i,
    // 16 the actual documented limits, not "arbitrary precision" hand-waving
    /NUMERIC\s*\(|precision\s*,\s*scale|131072|16383/i,
    // 17 removes the container when it exits
    /remove[\s\S]{0,100}(exit|terminat|finish|stop)/i,
  ],

  qa: [
    // 0 the execute bit — not ownership, not a missing shebang. "requires the script to be
    // executable" is the fact without the words "permission" or "bit".
    /chmod|execut\w*\s+(permission|bit)|\bx\s+permission|(be|not)\s+executable/i,
    // 1 nothing is listening: no sshd, wrong port, or a firewall
    /(no|not|isn.t)[\s\S]{0,40}(sshd|ssh daemon|listening|running)|nothing[\s\S]{0,20}listening|wrong port|firewall|port[\s\S]{0,20}closed/i,
    /lsof|fuser/i,
    // 3 CRLF line endings, or a missing `do`. "a missing quote mark" is the recorded
    // fabrication and rejects the answer outright, however many real terms follow it.
    /^(?![\s\S]*quote)[\s\S]*(\\r|carriage return|CRLF|dos2unix|tr\s+-d|line ending|missing\s+`?do`?\b)/i,
    // 4 127 means specifically that the command was not found
    /command\s+(cannot be |not )found|not found/i,
    /ssh-keygen[\s\S]{0,30}-R|known_hosts/i,
    // 6 a process still holds the deleted file open — the fix is finding or restarting it.
    // "check inode count with tune2fs to identify if the deletion caused a process to open
    // the file" is backwards and gets no credit: `open` alone is not the mechanism.
    /\blsof\b|fuser|still\s+(open|held|holding)|(open|file)\s+(file\s+)?(handle|descriptor)|holds?\s+(it\s+)?open|restart[\s\S]{0,30}(the\s+)?process/i,
    // 7 ARG_MAX: use find -delete/-exec or xargs
    /find[\s\S]{0,40}(-delete|-exec)|xargs|ARG_MAX|too many arguments|exceed\w*[\s\S]{0,40}(limit|maximum|argument)/i,
    // 8 inodes or the directory index are exhausted, not the blocks
    /inode|dir_index|directory index|hash collision/i,
    // 9 the shell cached the old path. "bash hashed the original version" names the mechanism
    // without the word table or cache; the stem is the fact, not the noun after it.
    /hash\s+-[dr]\b|hash(ed|es)?\s+(table|cache)|(bash|shell|it)\s+hash(ed|es)?\b|rehash|shell[\s\S]{0,30}cache/i,
    // 10 by-design clarify prompt; the fact it would need is the shared inode
    /inode/i,
    /replace[\s\S]{0,40}(process|shell|image)|in place/i,
    // 12 filenames may contain newlines and spaces, so the output is unparseable
    /newline|special character|space[\s\S]{0,30}(filename|name)|arbitrary character|unpredictab/i,
    // 13 SIGKILL cannot be caught — stating the consequence ("no chance to clean up") is
    // stating the fact. A bare "SIGKILL is stronger" still fails, which is the point.
    /(can.?t|cannot|unable to)\s+be\s+(caught|ignored|handled)|uncatchable|without[\s\S]{0,40}(chance|opportunity)[\s\S]{0,25}(clean|handl|catch|exit)/i,
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
    /calvin|light[- ]independent|chlorophyll|thylakoid|light energy into chemical|carbon dioxide[\s\S]{0,60}(water|glucose|sugar)/i,
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
