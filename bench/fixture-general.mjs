// General-knowledge retrieval fixture: 16 cases with nothing to do with computing,
// shells, or system administration. Companion to fixture-instructional.mjs and
// fixture-qa.mjs, whose 32 cases are entirely technical — a ranking formula tuned only
// on those has never been measured on the question a search engine over an 875k-article
// Wikipedia actually gets asked. Format: [query, intent, bookStem, titleRe, needleRe].
//
// Every case verified live against kiwix-serve: searchBook(prep(query), bookStem, 8)
// returned exactly ONE hit matching titleRe, that article was fetched, converted with
// html2txt, and the needle read in context to confirm it answers the question. 16/16.
//
// Wikipedia-corpus overlap check (required, and the answer is uniform): each case was
// also run against wikipedia_en_computer_nopic_2026-06 and wikipedia_en_100_2026-07.
// Neither corpus contains ANY of these 16 articles — not out-ranked, absent. Searching
// the 5k-article top-100 subset for "mona lisa" returns Michael Jackson; searching the
// computer corpus for "cheetah" returns Cheetah Mobile and macOS version history. So
// general-knowledge routing here is a one-book decision, and the three wiki ZIMs do not
// compete on subject articles: their overlap is titles, not topics. The per-case trailing
// comment records the rank in _top plus the noise it had to beat.
//
// Union recall (searchUnion(prep(query), 5)): 16/16. Every expected article appears in
// the cross-book union, so no case here exposes a candidate-generation gap — unlike the
// technical fixtures, where mounting a 413k-article Stack Exchange corpus cost recall.
// The failures this fixture can expose are therefore ranking failures, not recall ones.
//
// Distribution — all 16 in wikipedia_en_top_nopic_2026-06:
//   reference 8  astronomy, biology, geography, history, chemistry, medicine, art, sport
//   concept   5  biology, oceanography, physics, geology, medicine
//   howto     3  first aid, gardening, horticulture — deliberately non-computing, to test
//                that the howto prior is not just "prefer a wiki to Stack Exchange"
//
// [query, intent, bookStem, titleRe, needleRe]
const TOP = "wikipedia_en_top_nopic_2026-06";

export const CASES = [
  // ------------------------------------------------- reference: one checkable datum
  ["how many moons does jupiter have", "reference", TOP,
    /^Moons of Jupiter$/i, /115 known moons/i],
  // r=1, behind "Natural satellite"; Jupiter's own article is not in the top 8 at all.
  // The lead sentence answers outright: 115 known moons as of 9 April 2026.

  ["how fast can a cheetah run", "reference", TOP,
    /^Cheetah$/i, /running speed is 104 km\/h/i],
  // r=1, behind "Antelope", and the top 8 also carries macOS version history + Mac OS X
  // Tiger on the Apple codename. §Speed: highest reliably reported speed 104 km/h, and
  // it explicitly discredits the popular 114 km/h figure.

  ["how tall is mount kilimanjaro", "reference", TOP,
    /^Mount Kilimanjaro$/i, /5,895\s*metres/i],
  // r=0. Needle is pinned to Uhuru Peak at 5,895 m — the article also states 5,803 m in a
  // climate-station table, so a needle of /5,8[0-9]{2}/ would have passed on the wrong number.

  ["when did the berlin wall fall", "reference", TOP,
    /^Fall of the Berlin Wall$/i, /fell on 9 November 1989/i],
  // r=0, with "Berlin Wall" at r=1 — anchored titleRe keeps them apart. "when did world
  // war two end" was tried first and dropped (see the drop list at the foot of this file).

  ["what is the chemical formula of table salt", "reference", TOP,
    /^Sodium chloride$/i, /chemical formula NaCl/i],
  // r=0, ahead of "Potassium chloride" and "Chloride". Lead: "an ionic compound with the
  // chemical formula NaCl".

  ["who discovered penicillin", "reference", TOP,
    /^Alexander Fleming$/i, /discovery in 1928 of what was later named benzylpenicillin/i],
  // r=0, with "Penicillin" at r=1 and "Howard Florey" at r=2 — three defensible answers,
  // and the person, not the substance, wins. Needle carries both the who and the year.

  ["who painted the mona lisa", "reference", TOP,
    /^Mona Lisa$/i, /portrait painting by the Italian artist Leonardo da Vinci/i],
  // r=0, ahead of "Lisa del Giocondo" (r=1) and "Leonardo da Vinci" (r=2). Needle is the
  // lead clause, so it names the painter rather than merely co-occurring with him.

  ["how long is a marathon race", "reference", TOP,
    /^Marathon$/i, /42\.195 kilometres/i],
  // r=4 — the deepest reference case. Four runner/athletics articles outrank the event
  // itself ("Long-distance running", Bekele, Tadese, Radcliffe), so any top-3 cutoff loses
  // this one. Anchored titleRe excludes "Half marathon" and "2013 Boston Marathon".

  // ------------------------------------------------- concept: explanation, not a datum
  ["how does photosynthesis work", "concept", TOP,
    /^Photosynthesis$/i, /light-independent reactions called the Calvin cycle/i],
  // r=1, behind "Terence McKenna" — a pure prep() artifact, since "work" survives the
  // stoplist. Needle sits in the two-stage mechanism summary in the lead.

  ["what causes ocean tides", "concept", TOP,
    /^Tide$/i, /differential gravitational forces exerted primarily by the Moon/i],
  // r=0, with "Tidal force" at r=1 — both are correct sources, and the anchored title
  // picks the one whose lead states the cause.

  ["why is the sky blue", "concept", TOP,
    /^Rayleigh scattering$/i, /blue light wavelengths scatter more/i],
  // r=1, beaten by the Wilco album "Sky Blue Sky" at r=0, with "Blue moon" and "Blue"
  // also in the top 8: the strongest title-vs-topic distractor in this fixture. The
  // article that actually explains it is the one whose title shares no query term.

  ["what is plate tectonics", "concept", TOP,
    /^Plate tectonics$/i, /fractured into seven or eight major plates/i],
  // r=0, ahead of Orogeny/Subduction/Continental drift. Easiest case here: the query is
  // the title.

  ["how does a vaccine produce immunity", "concept", TOP,
    /^Vaccine$/i, /recognizes vaccine agents as foreign/i],
  // r=3. Seven near-identical titles compete ("Immunization", "Inactivated vaccine",
  // "Conjugate vaccine", "Attenuated vaccine", three named vaccines) and the general
  // article loses to all but one — the general-knowledge analogue of the specific-page
  // pile-up the technical fixtures hit. §Effectiveness carries the mechanism.

  // ------------------------------------------------- howto: procedures, none computing
  ["how do you perform CPR on an adult", "howto", TOP,
    /^Cardiopulmonary resuscitation$/i, /100.120 compressions per minute/i],
  // r=0. Needle is the AHA rate of 100-120 compressions per minute (en dash in the ZIM,
  // hence the wildcard). A real procedure with a number, not a description of one.

  ["how do you make compost at home", "howto", TOP,
    /^Compost$/i, /carbon-to-nitrogen ratio of about 25:1/i],
  // r=0. The article gives the steps — shred, water, aerate by turning the pile — plus
  // the ratio the needle pins, so the answer is executable rather than definitional.

  ["how do I propagate a plant from cuttings", "howto", TOP,
    /^Vegetative reproduction$/i, /cuttings are treated with hormones before being planted/i],
  // r=0, ahead of "Grafting" and "Horticulture". §Artificial means gives the procedure:
  // cut a stem or leaf, treat with hormone, plant, adventitious roots follow.
];

// Dropped candidates, and why — kept here so the same dead ends are not re-tried:
//   "when did world war two end"                 no WWII article in the top 8 for either
//     phrasing; the corpus answers via "Surrender of Japan" (r=4), which is a different
//     question. Replaced with the Berlin Wall case.
//   "what is the atomic number of gold" /
//     "how many protons does a gold atom have"   both retrieve the general articles
//     ("Atomic number", "Chemical element", "Atomic nucleus") and never "Gold". Chemistry
//     is covered by the table-salt case instead.
//   "how do I tie a bowline knot" / "reef knot"  no knot article in the corpus at all;
//     4 hits total, best of them "Quipu". Knot-tying is simply not in this ZIM.
//   "how do you perform the heimlich maneuver"   4 hits, all actors and one film.
//   "how do you apply a tourniquet"              3 hits, none procedural.
//   "how do you make bread with sourdough starter"  "Bread" at r=0 describes bread, never
//     a method; no procedure to verify.
//   "how do you make water safe to drink by boiling"  "Water purification" at r=0 does
//     hold "boiling water for ten minutes", but as historical advice inside an
//     explanation; the three kept howtos give stronger procedures.
