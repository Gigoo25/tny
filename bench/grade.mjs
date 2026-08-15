// The grader, shared by every answering arm.
//
// Extracted from answer-cli.mjs when a second arm appeared (F76: is the 0.8B model even the
// right tool for this?). Two arms scored by two graders would compare nothing, so there is
// one grader here and the arms differ only in how the answer text is produced.
import { CASES as INST } from "./fixture-instructional.mjs";
import { CASES as QA } from "./fixture-qa.mjs";
import { CASES as GEN } from "./fixture-general.mjs";
import { CASES as AMB } from "./fixture-ambiguous.mjs";
import { EXPECT } from "./expect.mjs";

// A case carries [query, intent, book, titleRe, needleRe, expectRe?]. `fixture-ambiguous.mjs`
// authors its `expectRe` inline; the other three take theirs from `expect.mjs` by position, so
// the retrieval fixtures stay about articles and answer grading lives in one file.
export const CASES = [["inst", INST, EXPECT.instructional], ["qa", QA, EXPECT.qa],
                      ["gen", GEN, EXPECT.general], ["amb", AMB, []]]
  .flatMap(([set, cs, exp]) => cs.map((c, i) => [set, ...c.slice(0, 5), c[5] ?? exp[i]]));

// `needleRe` matches the *article's* prose, so it cannot be applied to an answer directly:
// the article says "115 known moons", a correct answer says "Jupiter has 115 moons". What
// transfers is the fact itself, and it is mechanically extractable — no hand-written expected
// answers, which would only encode my own idea of the right wording.
const COMMON = /^(the|and|for|are|was|were|with|that|this|from|have|has|been|into|than|then|when|what|which|about|their|there|these|those|being|other|also|more|most|some|such|only|over|after|before|between|called|named|later|primarily|known|used|using|use|its|his|her)$/i;

// An escaped `\.` in a needle is a decimal point ("42\.195" is one number); a bare `.` is
// regex-any, so "100.120 compressions" is really the range 100-120 and must be read as two.
//
// The largest number in the needle is the fact whenever there is one (1989, 5895, 42.195,
// 104): a wrong answer changes it, which is exactly the failure to catch — this rule flags
// "the Berlin Wall fell on November 9, 2009" and "Jupiter has 95 known moons" against
// articles that say 1989 and 115, both of which read as confidently grounded. With no number,
// the needle's rarest words carry it (`NaCl`, `Leonardo`, `Calvin`, `scatter`).
export function factOf(needleRe) {
  const raw = String(needleRe).replace(/^\//, "").replace(/\/[a-z]*$/, "");
  const src = raw.replace(/\\\./g, "\u0001").replace(/\\/g, "");
  const nums = (src.match(/\d[\d\u0001,]*\d|\d+/g) ?? []).map(s => s.replace(/,/g, "").replace(/\u0001/g, "."));
  const words = (src.match(/[A-Za-z][A-Za-z-]{2,}/g) ?? []).filter(w => !COMMON.test(w));
  const num = nums.length ? nums.reduce((a, b) => (parseFloat(b) > parseFloat(a) ? b : a)) : null;
  return { num, words };
}

// The number must stand alone: "195" inside "42.195" is not a match for 195, but "42.195" is.
const hasNum = (ans, n) => new RegExp(String.raw`(?<![\d.,])${n.replace(".", "\\.")}(?![\d])`).test(ans.replace(/,/g, ""));
const hasWord = (ans, words) => words.some(w => new RegExp(String.raw`\b${w}`, "i").test(ans));

// Four outcomes, and the difference between WRONG and REFUSED is the whole point of the
// grounding rules (F27/F44/F45): a refusal is safe, a confident wrong answer is not.
//   ok        the fact is in the answer
//   REFUSED   the arm declined ("not found" / rejected)
//   WRONG     an answer that does not carry the fact
//   ERROR     the arm never answered — F107: a 300 s request timeout on the 4B produced an
//             empty answer, and counting those as refusals credited the model with judgement
//             it never exercised. Infrastructure failures are their own column.
export function verdictOf(ans, err, needleRe, expectRe) {
  if (/chat request failed|chat body|chat json|kiwix|not responding/.test(err ?? "")) return "ERROR";
  const refused = /^not found$/im.test(ans) || ans.trim() === "" || /rejected/.test(err ?? "");
  // An authored `expectRe` grades the answer directly: it was written against the source
  // article to accept any paraphrase of the fact and reject the plausible wrong answer, which
  // the needle-derived rule cannot do. Cases without one fall back to that rule.
  const { num, words } = factOf(needleRe);
  const correct = !refused && (expectRe ? expectRe.test(ans) : num === null ? hasWord(ans, words) : hasNum(ans, num));
  return refused ? "REFUSED" : correct ? "ok" : "WRONG";
}

// One tally, one summary line, so two arms print comparable numbers.
export function tallyOf() {
  const t = {};
  return {
    add(set, verdict, ms) {
      t[set] ??= { ok: 0, REFUSED: 0, WRONG: 0, ERROR: 0, n: 0, ms: 0 };
      t[set][verdict]++;
      t[set].n++;
      t[set].ms += ms;
    },
    report(label) {
      console.log("");
      let C = 0, R = 0, W = 0, E = 0, N = 0, ms = 0;
      for (const [set, r] of Object.entries(t)) {
        const errs = r.ERROR ? ` errored ${r.ERROR}` : "";
        console.log(`${set.padEnd(5)} correct ${r.ok}/${r.n} refused ${r.REFUSED} wrong ${r.WRONG}${errs} ${(r.ms / r.n / 1000).toFixed(1)}s per answer`);
        C += r.ok; R += r.REFUSED; W += r.WRONG; E += r.ERROR; N += r.n; ms += r.ms;
      }
      console.log(`TOTAL ${label} correct ${C}/${N} (${((C / N) * 100).toFixed(0)}%) refused ${R} wrong ${W}${E ? ` errored ${E}` : ""} ${(ms / N / 1000).toFixed(2)}s/answer ${(ms / 60000).toFixed(1)} min`);
    },
  };
}
