#!/usr/bin/env python3
"""
Build a personal human-writing baseline from a Gmail Takeout mbox.

The shipped human control is Moby-Dick, which is a generous anchor: literary
19th-century prose sits far from modern business writing, so separation looks
easier than it is. Your own pre-LLM email is the register this tool actually
gets pointed at.

    python3 mbox_baseline.py Sent.mbox --me you@gmail.com

Writes, by default into samples/private/ (gitignored):

    baseline_corpus.txt          every kept message
    baseline_corpus_pre2023.txt  the pre-LLM era on its own
    baseline_corpus_2023on.txt   the LLM era on its own
    baseline_manifest.csv        date, era, subject, words, burstiness, per
                                 message -> read this before trusting anything

Then score the two eras and compare them:

    ./voiceprint ppl samples/private/baseline_corpus_pre2023.txt
    ./voiceprint ppl samples/private/baseline_corpus_2023on.txt

Use the per-message MEDIAN burstiness this script prints, not the figure
`ppl` reports for the stitched file. Burstiness is the coefficient of
variation of sentence length; across a concatenation of short messages it
measures the stitching, not the writing.

What gets filtered out, and why each one matters:

  not from you        a thread contains other people's prose
  nothing by date     every year is kept. Messages are split into eras at
                      --split (default 2023-01-01, when ChatGPT was new) and
                      reported separately, because mail you may have had help
                      with should sit next to mail you did not rather than be
                      averaged into it
  quoted replies      '>' lines, 'On <date>, X wrote:', '-----Original
                      Message-----', 'From: ... Sent: ...' blocks
  signatures          everything after a '-- ' line, plus phone/address rows
  legal footers       everything from the first confidentiality line on. These
                      wrap over several lines and score near-zero perplexity,
                      so leaving them in makes your own writing look machine
  boilerplate         greetings, sign-offs, 'Sent from my iPhone'
  automated mail      no-reply senders, calendar invites, delivery reports
  short messages      under --min-words, which are noise at this length
"""
import os
import re
import csv
import sys
import email
import mailbox
import argparse
import statistics as st
from email import policy

SPLIT = '2023-01-01'        # ChatGPT shipped Nov 2022

# A reply's quoted body starts at one of these and runs to the end.
QUOTE_START = re.compile(
    r'''^(
        On\ .{0,120}\ wrote:        |
        -{2,}\s*Original\ Message   |
        _{5,}                       |
        -{2,}\s*Forwarded\ message  |
        From:\s                     |
        Sent\ from\ my\ \w+         |
        Get\ Outlook\ for\ \w+
    )''', re.X | re.I)

# Lines that are structure or courtesy, not prose.
DROP_LINE = re.compile(
    r'''^(
        (Hi|Hello|Hey|Dear|Good\ (morning|afternoon|evening))\b.{0,40}$  |
        (Thanks|Thank\ you|Cheers|Regards|Best|Best\ regards|Sincerely|
         Kind\ regards|Talk\ soon|Please\ advise|Let\ me\ know)[\s,.!]*$  |
        [A-Z][a-z]+\s*$                      |   # a lone first name
        (Tel|Phone|Cell|Mobile|Fax|Direct|Office|Email|E|M|T|D|W)[\s:.|]  |
        (https?://|www\.)                    |
        \+?[\d()\s.-]{9,}$                   |   # phone number
        .{0,60}(Confidential|confidentiality|privileged).{0,200}$
    )''', re.X)

# A legal footer runs to the end of the message and wraps over several lines,
# so only the first one carries a keyword. Break, do not filter line by line.
DISCLAIMER = re.compile(
    r'''(
        confidential                    |
        privileged                      |
        intended\ (recipient|only\ for) |
        unauthoriz(ed|ation)\ (review|use|disclosure) |
        notify\ (us|the\ sender)        |
        delete\ this\ (message|e-?mail) |
        disclose\ this\ message
    )''', re.X | re.I)

AUTOMATED = re.compile(
    r'(no[-_.]?reply|noreply|donotreply|do-not-reply|notifications?@|'
    r'mailer-daemon|postmaster|calendar-notification|bounce)', re.I)


def body_text(msg):
    """Plain text if the message has it, otherwise HTML with tags stripped."""
    try:
        part = msg.get_body(preferencelist=('plain',))
        if part is not None:
            return part.get_content()
        part = msg.get_body(preferencelist=('html',))
        if part is None:
            return ''
        html = part.get_content()
    except (LookupError, ValueError, KeyError, AssertionError):
        return ''
    html = re.sub(r'(?is)<(script|style).*?</\1>', ' ', html)
    html = re.sub(r'(?i)<br\s*/?>|</p>|</div>|</tr>', '\n', html)
    html = re.sub(r'(?s)<[^>]+>', ' ', html)
    for a, b in (('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'),
                 ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")):
        html = html.replace(a, b)
    return html


def own_text(raw):
    """Strip everything that is not prose this person typed in this message."""
    lines = []
    for ln in raw.replace('\r\n', '\n').split('\n'):
        s = ln.strip()
        if QUOTE_START.match(s):
            break                      # quoted reply runs to the end
        if DISCLAIMER.search(s):
            break                      # legal footer runs to the end
        if s == '--' or s == '-- ':
            break                      # signature delimiter
        if s.startswith('>'):
            continue
        if not s:
            lines.append('')
            continue
        if DROP_LINE.match(s):
            continue
        lines.append(s)
    out = '\n'.join(lines)
    out = re.sub(r'\n{3,}', '\n\n', out)
    return re.sub(r'[ \t]+', ' ', out).strip()


def sentences(text):
    return [x for x in re.split(r'(?<=[.!?])\s+', text) if len(x.split()) >= 3]


def burstiness(text):
    s = [len(x.split()) for x in sentences(text)]
    if len(s) < 2 or not st.mean(s):
        return None
    return st.pstdev(s) / st.mean(s)


def addr(value):
    if not value:
        return ''
    m = re.search(r'<([^>]+)>', str(value))
    return (m.group(1) if m else str(value)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mbox', nargs='+', help='Takeout .mbox file(s)')
    ap.add_argument('--me', action='append', default=[],
                    help='your address; repeatable. Default: every address '
                         'that appears as a sender more than 20 times')
    ap.add_argument('--split', default=SPLIT,
                    help='era boundary (YYYY-MM-DD). Mail before it is '
                         'reported as pre-LLM, mail on or after it separately')
    ap.add_argument('--since', default=None, help='drop mail before this date')
    ap.add_argument('--until', default=None,
                    help='drop mail on or after this date')
    ap.add_argument('--min-words', type=int, default=40)
    ap.add_argument('--out', default='samples/private')
    a = ap.parse_args()

    mine = {x.lower() for x in a.me}

    rows, blocks = [], []
    seen = set()
    counts, dropped = {}, {}

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1

    for path in a.mbox:
        print('reading %s ...' % path)
        box = mailbox.mbox(path, factory=None)
        for key in box.keys():
            try:
                msg = email.message_from_bytes(box.get_bytes(key),
                                              policy=policy.default)
            except Exception:
                drop('unparseable')
                continue
            frm = addr(msg.get('From'))
            counts[frm] = counts.get(frm, 0) + 1

            if mine and frm not in mine:
                drop('not from you')
                continue
            if AUTOMATED.search(frm):
                drop('automated sender')
                continue

            try:
                dt = email.utils.parsedate_to_datetime(msg.get('Date'))
                date = dt.date().isoformat()
            except (TypeError, ValueError):
                drop('no usable date')
                continue
            if a.since and date < a.since:
                drop('before %s' % a.since)
                continue
            if a.until and date >= a.until:
                drop('on or after %s' % a.until)
                continue
            era = 'pre2023' if date < a.split else '2023on'

            text = own_text(body_text(msg))
            words = len(text.split())
            if words < a.min_words:
                drop('under %d words' % a.min_words)
                continue
            fp = text[:200]
            if fp in seen:
                drop('duplicate')
                continue
            seen.add(fp)

            subj = re.sub(r'\s+', ' ', str(msg.get('Subject') or ''))[:80]
            b = burstiness(text)
            rows.append({'date': date, 'era': era, 'subject': subj,
                         'words': words, 'sentences': len(sentences(text)),
                         'burstiness': '' if b is None else round(b, 4)})
            blocks.append((era, text))

    if not mine:
        # No --me given: anything that sent more than 20 messages is probably you
        likely = sorted(((n, e) for e, n in counts.items() if n > 20),
                        reverse=True)[:5]
        print('\nNo --me given, so nothing was filtered by sender.')
        print('Most frequent senders in these files:')
        for n, e in likely:
            print('  %6d  %s' % (n, e))
        print('Re-run with --me <your address> for each one that is you.')

    os.makedirs(a.out, exist_ok=True)
    manifest = os.path.join(a.out, 'baseline_manifest.csv')
    written = []
    for label, keep in (('', None), ('_pre2023', 'pre2023'),
                        ('_2023on', '2023on')):
        txt = [t for e, t in blocks if keep is None or e == keep]
        if not txt:
            continue
        path = os.path.join(a.out, 'baseline_corpus%s.txt' % label)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(txt) + '\n')
        written.append(path)
    with open(manifest, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'era', 'subject', 'words',
                                          'sentences', 'burstiness'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['date']):
            w.writerow(r)

    total = sum(r['words'] for r in rows)
    print('\n' + '=' * 68)
    print('%-9s %7s %8s %12s %24s'
          % ('era', 'msgs', 'words', 'burstiness', 'dates'))
    print('-' * 68)
    for label, name in (('pre2023', 'pre-LLM'), ('2023on', '2023 on'),
                        (None, 'all')):
        rs = [r for r in rows if label is None or r['era'] == label]
        if not rs:
            continue
        bs = [r['burstiness'] for r in rs if r['burstiness'] != '']
        print('%-9s %7d %8d %12s %24s'
              % (name, len(rs), sum(r['words'] for r in rs),
                 ('%.3f' % st.median(bs)) if bs else 'n/a',
                 '%s..%s' % (min(r['date'] for r in rs),
                             max(r['date'] for r in rs))))
    print('\nBurstiness is the per-message median, which is the figure to')
    print('quote. Higher is more human on this measure.')
    print('\ndropped:')
    for why in sorted(dropped, key=lambda k: -dropped[k]):
        print('  %6d  %s' % (dropped[why], why))
    print()
    for path in written:
        print('wrote %s' % path)
    print('wrote %s' % manifest)
    if total < 1000:
        print('\nUnder 1,000 words. The numbers will be unstable; widen the')
        print('cutoff or lower --min-words, and say so when you quote them.')
    print('\nNext:')
    print('  read %s and delete any row you had help writing' % manifest)
    for path in written:
        if path.endswith('_pre2023.txt') or path.endswith('_2023on.txt'):
            print('  ./voiceprint ppl %s' % path)
    print('  compare the eras. If the later one scores more machine-like,')
    print('  that is either your style changing or assistance in the corpus,')
    print('  and the manifest is the only way to tell which')


if __name__ == '__main__':
    main()
