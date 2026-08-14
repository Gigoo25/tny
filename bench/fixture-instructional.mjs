// Retrieval fixture: queries whose correct source is an INSTRUCTIONAL corpus — a wiki
// page or a documentation page, never a Stack Exchange discussion. Companion to the
// Q&A fixture; together they measure cross-book routing, which the 15-case fixture in
// harness.mjs cannot (its expectations live in 3 books).
//
// Every case was verified live against kiwix-serve: the article was retrieved by
// search(prep(query), bookStem, 8), fetched, converted with html2txt, and the needle
// read in context to confirm it answers the question. Needles are literal commands,
// flags, or signatures — never common English.
//
// Corpora and per-book counts (18 cases):
//   archlinux_en_all_maxi_2026-07    7   howto 7
//   devdocs_en_git_2026-07           3   howto 2, reference 1
//   devdocs_en_docker_2026-07        2   howto 1, reference 1
//   devdocs_en_postgresql_2026-08    2   howto 1, reference 1
//   devdocs_en_python_2026-08        2   howto 1, reference 1
//   devdocs_en_bash_2026-04          1   reference 1
//   devdocs_en_rust_2026-07          1   reference 1
// Intents: howto 12, reference 6.
//
// [query, intent, bookStem, titleRe, needleRe]
export const CASES = [
  // ---------------------------------------------------------------- howto: Arch Wiki
  ["how do I change the hostname of my machine", "howto", "archlinux_en_all_maxi_2026-07",
    /^Network configuration$/i, /hostnamectl hostname/i],
  ["how do I enable a systemd service so it starts at boot", "howto", "archlinux_en_all_maxi_2026-07",
    /^Systemd$/i, /systemctl enable\b/i],
  ["how do I add a new user with useradd", "howto", "archlinux_en_all_maxi_2026-07",
    /^Users and groups$/i, /useradd -m/i],
  ["how do I resize an ext4 filesystem", "howto", "archlinux_en_all_maxi_2026-07",
    /^Ext4$/i, /resize2fs/i],
  ["how do I mount a filesystem by UUID in fstab", "howto", "archlinux_en_all_maxi_2026-07",
    /^Fstab$/i, /UUID=[0-9a-f]{8}-/i],
  ["how do I enable the en_US.UTF-8 locale", "howto", "archlinux_en_all_maxi_2026-07",
    /^Locale$/i, /locale-gen/i],
  ["how do I let a user run commands with sudo", "howto", "archlinux_en_all_maxi_2026-07",
    /^Sudo$/i, /%wheel ALL=\(ALL/i],

  // ------------------------------------------------------------------ howto: devdocs
  ["how do I undo the last commit but keep my changes", "howto", "devdocs_en_git_2026-07",
    /^git reset$/i, /--soft Leave your working tree files and the index unchanged/i],
  ["how do I change the message of my last commit", "howto", "devdocs_en_git_2026-07",
    /^git commit$/i, /--amend Replace the tip of the current branch/i],
  ["how do I create a docker image tag for a registry", "howto", "devdocs_en_docker_2026-07",
    /^docker tag$/i, /docker tag SOURCE_IMAGE/],
  ["how do I add a login role with a password in postgresql", "howto", "devdocs_en_postgresql_2026-08",
    /^CREATE ROLE$/i, /CREATE ROLE \w+ WITH LOGIN PASSWORD/i],
  ["how do I create a virtual environment in python", "howto", "devdocs_en_python_2026-08",
    /^venv$/i, /python -m venv/],

  // -------------------------------------------------------------- reference: devdocs
  ["what does unwrap_or_default return on an Option", "reference", "devdocs_en_rust_2026-07",
    /^std::option::Option$/i, /pub fn unwrap_or_default\(self\) -> T/],
  ["how does bash substitute a default value if a variable is unset", "reference", "devdocs_en_bash_2026-04",
    /^Shell Parameter Expansion$/i, /\$\{ ?parameter ?:- ?word ?\}/],
  ["what does the --oneline flag do in git log", "reference", "devdocs_en_git_2026-07",
    /^git log$/i, /--oneline This is a shorthand for --pretty=oneline/],
  ["what arguments does json.dumps take", "reference", "devdocs_en_python_2026-08",
    /^json$/i, /json\.dumps\(obj, \*, skipkeys=False/],
  ["what precision does the numeric data type have in postgresql", "reference", "devdocs_en_postgresql_2026-08",
    /^Numeric Types$/i, /NUMERIC\(precision, scale\)/],
  ["what does the --rm option do in docker run", "reference", "devdocs_en_docker_2026-07",
    /^docker run$/i, /--rm Automatically remove the container when it exits/],
];
