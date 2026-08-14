// Q&A-corpus retrieval fixture: 14 cases whose correct source is the Unix & Linux
// Stack Exchange ZIM, not a wiki. Format: [query, intent, bookStem, titleRe, needleRe].
//
// Selection rule, applied to every case below: the query was also run against
// archlinux_en_all_maxi_2026-07 (top 5), and the case was kept only when the wiki had
// nothing that answers it. The trailing comment on each line records that check —
// the Arch hits observed, and the answer the SE thread actually supplies. Six candidates
// were dropped because the wiki *did* answer better (notably "why does sudo ask for my
// password every time": Arch's Sudo article has "Reduce the number of times you have to
// type a password" with timestamp_timeout).
//
// needleRe always matches text inside the *answer*, never only the asker's question.

const SE = "unix.stackexchange.com_en_all_2026-02";

export const CASES = [
  // ---- diagnose: error messages and failure symptoms -----------------------------
  ["why does ./script.sh give permission denied but bash script.sh works", "diagnose", SE,
    /^Run \.\/script\.sh vs bash script\.sh - permission denied$/i, /execute permission bit/i],
  // wiki check: Etckeeper|Chroot|Users and groups|VeraCrypt|Firejail — "Users and groups" is the
  // one plausible competitor, but it documents permission bits, never this asymmetry. SE r=0
  // answers it outright: ./script.sh needs the execute bit, bash script.sh needs only read.

  ["ssh gives connection refused what does that mean", "diagnose", SE,
    /^ssh Connection refused: how to troubleshoot/i, /don.t have an SSH daemon running/i],
  // wiki check: GNOME/Keyring|SSH keys|Simple stateful firewall|GnuPG|Enlightenment — "SSH keys"
  // is on-topic for ssh but covers key auth, not a refused TCP connect. SE r=0: no sshd running.

  ["how do I find out which processes are preventing unmounting of a device", "diagnose", SE,
    /^How do I find out which processes are preventing unmounting/i, /lsof \| grep/],
  // wiki check: Ext4|LVM|Installation guide|Flashing BIOS from Linux|Security — none discuss
  // "target is busy". SE r=0: lsof | grep /mountpoint, plus fuser -mv and umount -l.

  ["bash syntax error near unexpected token done", "diagnose", SE,
    /while read line do.*cause/i, /tr -d '\\015'/],
  // wiki check: zero Arch hits at all. SE r=0 diagnoses the trailing-quote position in the error
  // as CR characters in the script and gives tr -d '\015' as the fix.

  ["what does exit code 127 mean", "diagnose", SE,
    /^How do I get the list of exit codes/i, /127\s*-\s*if a command cannot be found/i],
  // wiki check: PCI passthrough via OVMF/Troubleshooting|Zsh|Mutt|PCI passthrough via OVMF —
  // pure keyword noise on "code"/"exit". SE r=4 tabulates 126/127/128+n: 127 = command not found.

  ["ssh says host key verification failed what do I do", "diagnose", SE,
    /REMOTE HOST IDENTIFICATION HAS CHANGED/i, /ssh-keygen -f "[^"]*known_hosts"\s*-R/],
  // wiki check: GnuPG|YubiKey|Simple stateful firewall|PCI passthrough via OVMF — nothing on
  // known_hosts. SE r=5: ssh-keygen -f ~/.ssh/known_hosts -R host, or sed -i '<line>d'.

  ["disk space not freed after deleting a large log file", "diagnose", SE,
    /^Disk space is not freed up when deleting files/i, /lsof \+L1/],
  // wiki check: Btrfs|QEMU — both about images/allocation, not unlinked-but-open files.
  // SE r=1: a process still holds the deleted file open; lsof +L1 lists them.

  ["rm gives argument list too long", "diagnose", SE,
    /rm thinks argument list is too long/i, /-delete/],
  // wiki check: Mutt|ALSA|Steam/Troubleshooting|Systemd|Chromium — noise on "list"/"long".
  // SE r=1: stop passing an expanded list, let find do the removal with -delete.

  ["no space left on device but there is plenty of free space", "diagnose", SE,
    /^How to fix intermittant/i, /dir_index/],
  // wiki check: ZFS only, on unrelated pool behaviour. SE r=0 names the real cause — ext4
  // dir_index hash collisions — and the fix, tune2fs -O ^dir_index plus e2fsck -fD.

  ["command not found even though the program is installed and in my path", "diagnose", SE,
    /^Why is program not found in PATH$/i, /hash -d/],
  // wiki check: Desktop entries|Dd|Emacs|Map scancodes to keycodes|Pacman — none cover shell
  // command hashing. SE r=1: bash cached the old path; hash -d <cmd> clears that entry.

  // ---- concept: answered by discussion, not procedure ----------------------------
  ["what is the difference between symbolic and hard links", "concept", SE,
    /^What is the difference between symbolic and hard links\?$/i, /inode/i],
  // wiki check: Lenovo ThinkPad X201|Help:Laptop page guidelines|Beets|Rsync|ThinkPad X200 —
  // Rsync only mentions --hard-links. SE r=1 contrasts the semantics: hard links share an inode,
  // survive renaming the original, cannot cross filesystems; symlinks store a path and break.

  ["why do some shell scripts use exec to run commands", "concept", SE,
    /^Why do some Linux shell scripts use exec to run commands\?$/i, /replaces the process image/i],
  // wiki check: Fish|Xinit|Sway|Universal Wayland Session Manager|Tmux — Xinit uses exec in an
  // example but never explains it. SE r=0: exec replaces the process image, so no shell lingers
  // waiting on the child (and it must therefore be the last line).

  ["what is wrong with parsing the output of ls", "concept", SE,
    /^Why \*not\* parse .ls. \(and what to do instead\)\?$/i, /ls separates filenames with newlines/i],
  // wiki check: LightDM|Systemd-boot|Chromium|Mutt|QEMU — all incidental `ls` in code blocks.
  // SE r=1: ls delimits with newlines while filenames may contain any byte but NUL and /;
  // use a glob or find -print0.

  ["what is the difference between SIGTERM and SIGKILL", "concept", SE,
    /does SIGTERM behave identically to SIGKILL/i, /can.t be caught or ignored/i],
  // wiki check: Atrium|Keyboard shortcuts|OpenSSH — each only names the signals in passing.
  // SE r=0: default behaviour is equivalent, but SIGKILL cannot be caught, blocked or ignored
  // while SIGTERM can, and the parent still learns which signal killed the child.
];
