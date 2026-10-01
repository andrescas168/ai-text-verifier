#!/usr/bin/env python3
"""
Stylometric "reads-as-AI" scorer.

This is NOT a detector in the forensic sense and does not output a probability.
It measures the surface features that perplexity-based tools respond to, and
shows which ones are driving the impression, so you can judge for yourself.

Signals are heuristic bands from published stylometry, not trained thresholds.
Usage:  python detect.py FILE [FILE ...]
"""
import os
import re
import sys
import zipfile
import subprocess
import statistics as st
from collections import Counter

# ---------------------------------------------------------------- extraction

def from_docx(path):
    z = zipfile.ZipFile(path)
    xml = z.read('word/document.xml').decode('utf-8')
    paras = []
    for p in re.findall(r'<w:p[ >].*?</w:p>', xml, re.S):
        t = ''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.S))
        t = (t.replace('&amp;', '&').replace('&lt;', '<')
              .replace('&gt;', '>').replace('&quot;', '"').replace('&apos;', "'"))
        if t.strip():
            paras.append(t.strip())
    return '\n\n'.join(paras)


def from_md(path):
    out = []
    for ln in open(path, encoding='utf-8').read().split('\n'):
        s = ln.strip()
        if s.startswith('#'):
            continue                                   # headings are not prose
        s = re.sub(r'^[-*+]\s+', '', s)
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = re.sub(r'\*(.+?)\*', r'\1', s)
        s = re.sub(r'`(.+?)`', r'\1', s)
        s = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', s)
        if s:
            out.append(s)
    return '\n\n'.join(out)


def from_pdf(path):
    r = subprocess.run(['pdftotext', '-enc', 'UTF-8', path, '-'],
                       capture_output=True, timeout=120)
    return r.stdout.decode('utf-8', 'replace')


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.docx':
        return from_docx(path)
    if ext in ('.md', '.markdown'):
        return from_md(path)
    if ext == '.pdf':
        return from_pdf(path)
    return open(path, encoding='utf-8', errors='replace').read()


# ------------------------------------------------------------- segmentation

ABBR = {'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct',
        'nov', 'dec', 'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st',
        'approx', 'etc', 'eg', 'ie', 'vs', 'inc', 'ltd', 'co', 'no', 'fig'}


def sentences(text):
    text = re.sub(r'\s+', ' ', text.replace('\n', ' '))
    parts, buf = [], ''
    for tok in re.split(r'(?<=[.!?])\s+', text):
        buf = (buf + ' ' + tok).strip() if buf else tok
        m = re.search(r'([A-Za-z]+)\.$', buf)
        if m and m.group(1).lower() in ABBR:
            continue                                   # false stop, keep going
        if re.search(r'\b[A-Z]\.$', buf):              # initials like U.S.
            continue
        parts.append(buf)
        buf = ''
    if buf:
        parts.append(buf)
    return [p for p in parts if len(p.split()) >= 3]


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text)


# ------------------------------------------------------------------ lexicons

TIER1 = ['delve', 'delving', 'tapestry', 'testament', 'intricate', 'realm',
         'myriad', 'plethora', 'ever-evolving', 'ever-changing', 'underscore',
         'underscores', 'underscoring', 'showcase', 'showcasing', 'pivotal',
         'paradigm', 'synergy', 'holistic', 'multifaceted', 'game-changer',
         'burgeoning', 'unwavering', 'meticulous', 'meticulously', 'profound']

TIER1_PHRASES = ["it's worth noting", 'it is worth noting', 'in today’s',
                 "in today's", 'at the forefront', 'plays a crucial role',
                 'a testament to', 'when it comes to', 'dive into', 'deep dive',
                 'fast-paced', 'in conclusion', 'navigating the', 'the world of',
                 'not only', 'serves as a', 'stands as a']

# Normal business vocabulary. Common in LLM output, but also in real resumes,
# so these are reported separately and weighted lightly.
TIER2 = ['leverage', 'leveraging', 'robust', 'streamline', 'streamlining',
         'comprehensive', 'seamless', 'seamlessly', 'foster', 'fostering',
         'empower', 'empowering', 'crucial', 'vital', 'transformative',
         'navigate', 'landscape', 'elevate', 'harness', 'harnessing',
         'cutting-edge', 'state-of-the-art', 'spearhead', 'spearheaded',
         'furthermore', 'moreover', 'additionally', 'consequently', 'ultimately']


# -------------------------------------------------------------------- report

class Band:
    """A measured signal with a heuristic band and a direction.

    scored=False marks a signal that is reported but kept out of the verdict.
    Punctuation rates are house style, not authorship: measured against this
    author's own hand-written files, em-dash rate pointed the wrong way.
    """
    def __init__(self, name, value, fmt, lo, hi, ai_side, note='', scored=True):
        self.name, self.value, self.fmt = name, value, fmt
        self.lo, self.hi, self.ai_side, self.note = lo, hi, ai_side, note
        self.scored = scored

    @property
    def level(self):
        if self.value is None:
            return 'n/a'
        if self.value < self.lo:
            return 'LOW'
        if self.value > self.hi:
            return 'HIGH'
        return 'mid'

    @property
    def leans_ai(self):
        return self.level != 'mid' and self.level == self.ai_side

    def line(self):
        v = 'n/a' if self.value is None else format(self.value, self.fmt)
        if not self.scored:
            mark = '  (weak, not scored)'
        else:
            mark = '  <-- leans formulaic' if self.leans_ai else ''
        return '  %-26s %8s  %-4s%s' % (self.name, v, self.level, mark)


def analyse(path):
    text = load(path)
    sents = sentences(text)
    wl = words(text)
    low = text.lower()
    n_w = len(wl)
    paras = [p for p in text.split('\n\n') if p.strip()]

    print('\n' + '=' * 72)
    print(os.path.basename(path))
    print('=' * 72)
    if n_w < 120:
        print('  (!) %d words. Below ~200 these measures are noise.' % n_w)
    print('  %d words, %d sentences, %d blocks' % (n_w, len(sents), len(paras)))

    slen = [len(s.split()) for s in sents]
    bands = []

    # 1. Burstiness: variation in sentence length. The single biggest signal
    #    that perplexity tools respond to. Humans vary; models even out.
    cv = (st.pstdev(slen) / st.mean(slen)) if len(slen) > 1 and st.mean(slen) else None
    bands.append(Band('burstiness (CV)', cv, '.2f', 0.40, 0.85, 'LOW',
                      'sentence-length variation'))

    # 2. Mean sentence length. LLM prose clusters ~17-24 words.
    bands.append(Band('mean sentence length', st.mean(slen) if slen else None,
                      '.1f', 11, 26, None))

    # 3. MATTR: vocabulary richness, window-normalised so length cannot skew it.
    if n_w >= 60:
        win, ratios = 50, []
        for i in range(0, n_w - win + 1, 10):
            chunk = [w.lower() for w in wl[i:i + win]]
            ratios.append(len(set(chunk)) / win)
        mattr = st.mean(ratios) if ratios else None
    else:
        mattr = None
    bands.append(Band('lexical richness MATTR', mattr, '.3f', 0.70, 0.86, 'LOW'))

    # 4. Hapax ratio: share of words used exactly once.
    c = Counter(w.lower() for w in wl)
    hapax = sum(1 for w, k in c.items() if k == 1) / len(c) if c else None
    bands.append(Band('hapax ratio', hapax, '.3f', 0.45, 0.75, 'LOW'))

    # 5. Tricolon: the "A, B, and C" cadence models overuse.
    tri = sum(1 for s in sents if re.search(r'\w+,\s+\w[\w\s\-]*,\s+and\s+\w+', s))
    tri_rate = tri / len(sents) if sents else None
    bands.append(Band('tricolon rate', tri_rate, '.2f', 0.0, 0.22, 'HIGH',
                      'share of sentences with 3-item lists'))

    # 6. Sentence-opener diversity.
    openers = [s.split()[0].lower() for s in sents if s.split()]
    odiv = len(set(openers)) / len(openers) if openers else None
    bands.append(Band('opener diversity', odiv, '.2f', 0.55, 1.01, 'LOW'))

    # 7. Punctuation per 1000 words.
    per_k = lambda n: 1000.0 * n / n_w if n_w else None
    bands.append(Band('em-dash /1k words', per_k(text.count('—')), '.1f',
                      0.0, 2.5, 'HIGH', scored=False))
    bands.append(Band('semicolon /1k words', per_k(text.count(';')), '.1f',
                      0.0, 6.0, 'HIGH', scored=False))

    for b in bands:
        print(b.line())

    # ---- lexical tells
    t1 = Counter()
    for w in TIER1:
        n = len(re.findall(r'\b%s\b' % re.escape(w), low))
        if n:
            t1[w] = n
    for p in TIER1_PHRASES:
        n = low.count(p)
        if n:
            t1[p] = n
    t2 = Counter()
    for w in TIER2:
        n = len(re.findall(r'\b%s\b' % re.escape(w), low))
        if n:
            t2[w] = n

    print('  %-26s %8d  %s' % ('strong tells', sum(t1.values()),
                               ', '.join('%s x%d' % (k, v) for k, v in t1.most_common(8)) or '-'))
    print('  %-26s %8d  %s' % ('business-normal words', sum(t2.values()),
                               ', '.join('%s x%d' % (k, v) for k, v in t2.most_common(8)) or '-'))

    # ---- read
    leaning = [b.name for b in bands if b.leans_ai and b.scored]
    print('  ' + '-' * 68)
    frag = (st.mean(slen) < 14 and len(paras) > 8) if slen else False
    if frag:
        print('  NOTE: short blocks, bullet-style. Burstiness and opener')
        print('        diversity are not meaningful on this kind of text.')
    if n_w < 120:
        print('  NO VERDICT: %d words extracted. Need ~200+.' % n_w)
        if n_w == 0:
            print('  Nothing extracted at all: scanned image PDF, or empty file.')
        return
    score = len(leaning) + (1 if sum(t1.values()) >= 3 else 0)
    verdict = ('reads natural' if score <= 1 else
               'mixed' if score <= 3 else 'reads formulaic')
    print('  signals leaning formulaic: %d  ->  %s' % (len(leaning), verdict))
    if leaning:
        print('  driven by: ' + ', '.join(leaning))
    print('  Not evidence of authorship. Formal, edited human writing scores')
    print('  the same way, which is why these tools misflag non-native writers.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for f in sys.argv[1:]:
        try:
            analyse(f)
        except Exception as e:
            print('\n%s: FAILED %s' % (os.path.basename(f), e))
    print()
