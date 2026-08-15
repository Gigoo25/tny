// F105: a measurement pins its own configuration.
//
// `tny` remembers the speed, length and model you last chose in the UI (`~/.cache/tny/prefs`),
// which is right for a daily tool and poison for a benchmark: a stray `fast · low` left over
// from an afternoon's use silently rescored the fixture at 41/58 instead of 46/58, and
// nothing in the output said the run had been reconfigured.
//
// Anything explicitly exported by the caller still wins, so `TNY_MODEL=4b bun
// bench/answer-cli.mjs` measures the 4B — the pin is a default, not a cage.
export const TNY_ENV = {
  ...process.env,
  TNY_MODE: process.env.TNY_MODE ?? "medium",
  TNY_LEN: process.env.TNY_LEN ?? "medium",
  TNY_MODEL: process.env.TNY_MODEL ?? "0.8b",
};
