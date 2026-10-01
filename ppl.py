#!/usr/bin/env python3
"""
Voiceprint: perplexity analysis.

Primary output is the SENTENCE breakdown, not a document score. A document
score tells you nothing you can act on; "these three sentences read as
boilerplate" does.

Scores every token's negative log-likelihood under a causal LM, in context,
with a sliding window so long documents are never truncated. Also reports GLTR
rank buckets (Gehrmann et al. 2019).

Usage: ppl.py [--model ID] [--top N] [--all] FILE [FILE ...]
"""
import os
import re
import sys
import math
import zipfile
import argparse
import subprocess
import statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = 'Qwen/Qwen3-0.6B'
MAXLEN = 1024
STRIDE = 512


# ---------------------------------------------------------------- extraction

def from_docx(path):
    xml = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    out = []
    for p in re.findall(r'<w:p[ >].*?</w:p>', xml, re.S):
        t = ''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.S))
        t = (t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
              .replace('&quot;', '"').replace('&apos;', "'"))
        if t.strip():
            out.append(t.strip())
    return '\n'.join(out)


def from_md(path):
    out = []
    for ln in open(path, encoding='utf-8').read().split('\n'):
        s = ln.strip()
        if s.startswith('#'):
            continue
        s = re.sub(r'^[-*+]\s+', '', s)
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = re.sub(r'\*(.+?)\*', r'\1', s)
        s = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', s)
        if s:
            out.append(s)
    return '\n'.join(out)


def load(path):
    e = os.path.splitext(path)[1].lower()
    if e == '.docx':
        return from_docx(path)
    if e in ('.md', '.markdown'):
        return from_md(path)
    if e == '.pdf':
        return subprocess.run(['pdftotext', '-enc', 'UTF-8', path, '-'],
                              capture_output=True, timeout=120
                              ).stdout.decode('utf-8', 'replace')
    return open(path, encoding='utf-8', errors='replace').read()


# ------------------------------------------------------------- segmentation

ABBR = {'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct',
        'nov', 'dec', 'mr', 'mrs', 'ms', 'dr', 'approx', 'etc', 'eg', 'ie',
        'vs', 'inc', 'ltd', 'co', 'no'}


def sentence_spans(text):
    spans, start = [], 0
    for m in re.finditer(r'[.!?]["\')\]]*\s+|\n+', text):
        end = m.end()
        chunk = text[start:end]
        head = re.search(r'([A-Za-z]+)\.\s*$', chunk)
        if head and head.group(1).lower() in ABBR:
            continue
        if re.search(r'\b[A-Z]\.\s*$', chunk):
            continue
        if chunk.strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def is_prose(s):
    """Headings, title rows and fragments are not sentences. They score
    absurdly low perplexity and would dominate any 'most predictable' list,
    which is how a resume's job-title lines drown out its actual writing."""
    w = s.split()
    if len(w) < 8:
        return False
    if ' | ' in s:                       # 'ROLE | Company | City | Dates'
        return False
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False                     # ALL CAPS heading
    if not re.search(r'[.!?]', s):
        return False                     # no terminal punctuation
    return True


# ------------------------------------------------------------------- scoring

def score(text, tok, model):
    enc = tok(text, return_offsets_mapping=True, return_tensors='pt')
    ids = enc['input_ids'][0]
    offs = enc['offset_mapping'][0].tolist()
    n = ids.size(0)
    if n < 2:
        return [], [], []

    nlls = [None] * n
    ranks = [None] * n
    start = 0
    while start < n - 1:
        end = min(start + MAXLEN, n)
        with torch.no_grad():
            logits = model(ids[start:end].unsqueeze(0)).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        lo = 0 if start == 0 else STRIDE - 1
        for i in range(lo, end - start - 1):
            tgt = ids[start + i + 1].item()
            lp = logprobs[i]
            nlls[start + i + 1] = -lp[tgt].item()
            ranks[start + i + 1] = int((lp > lp[tgt]).sum().item())
        if end == n:
            break
        start += STRIDE

    keep = [i for i in range(n) if nlls[i] is not None]
    return ([offs[i] for i in keep], [nlls[i] for i in keep],
            [ranks[i] for i in keep])


def bar(value, lo, hi, width=10):
    """Log-scaled bar: perplexity is log-distributed."""
    if hi <= lo:
        return '#' * (width // 2)
    f = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    f = max(0.0, min(1.0, f))
    n = int(round(f * width))
    return '#' * n + '.' * (width - n)


def clip(s, n=88):
    s = ' '.join(s.split())
    return s if len(s) <= n else s[:n - 3] + '...'


def analyse(path, tok, model, args):
    text = re.sub(r'[ \t]+', ' ', load(path)).strip()
    print('\n' + '=' * 78)
    print(os.path.basename(path))
    print('=' * 78)

    offs, nlls, ranks = score(text, tok, model)
    if len(nlls) < 50:
        print('  only %d scored tokens, too short for a reading' % len(nlls))
        return None

    # group token NLLs into sentences
    sents, skipped = [], 0
    for a, b in sentence_spans(text):
        raw = text[a:b].strip()
        sel = [nlls[i] for i, (s, e) in enumerate(offs) if s >= a and e <= b]
        if len(sel) < 4:
            continue
        if not is_prose(raw):
            skipped += 1
            continue
        sents.append((math.exp(sum(sel) / len(sel)), raw))

    doc_ppl = math.exp(sum(nlls) / len(nlls))
    print('  %d words · %d prose sentences · %s'
          % (len(text.split()), len(sents), args.model))

    if sents:
        vals = [v for v, _ in sents]
        lo, hi = min(vals), max(vals)
        ordered = sorted(sents)
        n = len(ordered) if args.all else min(args.top, len(ordered))

        print('\n  REWRITE CANDIDATES — lowest perplexity prose')
        print('  the model found these the easiest to predict:')
        for v, s in ordered[:n]:
            print('    %7.1f  %s  %s' % (v, bar(v, lo, hi), clip(s)))

        if not args.all and len(ordered) > n:
            print('\n  CARRIES THE MOST SIGNAL — highest perplexity')
            for v, s in ordered[-min(3, len(ordered)):][::-1]:
                print('    %7.1f  %s  %s' % (v, bar(v, lo, hi), clip(s)))

        cv = st.pstdev(vals) / st.mean(vals) if len(vals) > 1 else 0.0
    else:
        cv = 0.0
        print('\n  (no prose sentences found: all structural lines)')

    b = {'t1': 0, 't10': 0, 't100': 0, 't1k': 0, 'beyond': 0}
    for r in ranks:
        k = ('t1' if r == 0 else 't10' if r < 10 else 't100' if r < 100
             else 't1k' if r < 1000 else 'beyond')
        b[k] += 1
    tot = len(ranks)
    top10 = (b['t1'] + b['t10']) / tot

    print('\n  ' + '─' * 74)
    print('  document perplexity %.1f · sentence burstiness %.2f · top-10 share %.1f%%'
          % (doc_ppl, cv, 100 * top10))
    print('  GLTR  top-1 %.0f%%  top-10 %.0f%%  top-100 %.0f%%  top-1k %.0f%%  beyond %.0f%%'
          % tuple(100 * b[k] / tot for k in ('t1', 't10', 't100', 't1k', 'beyond')))
    if skipped:
        print('  %d structural line(s) skipped (headings, title rows)' % skipped)
    print('  Not evidence of authorship. Formal human writing scores the same way.')

    return {'file': os.path.basename(path), 'ppl': doc_ppl, 'cv': cv,
            'top10': top10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='HF model id (default %s; "gpt2" for the old baseline)'
                         % DEFAULT_MODEL)
    ap.add_argument('--top', type=int, default=4,
                    help='how many rewrite candidates to show (default 4)')
    ap.add_argument('--all', action='store_true', help='show every sentence')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()

    print('loading %s ...' % a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 4)

    rows = []
    for f in a.files:
        try:
            r = analyse(f, tok, model, a)
            if r:
                rows.append(r)
        except Exception as e:
            print('\n%s FAILED: %s' % (os.path.basename(f), e))

    if len(rows) > 1:
        print('\n' + '=' * 78)
        print('%-46s %8s %7s %9s' % ('file', 'ppl', 'burst', 'top-10%'))
        print('-' * 78)
        for r in rows:
            print('%-46s %8.1f %7.2f %8.1f%%'
                  % (r['file'][:46], r['ppl'], r['cv'], 100 * r['top10']))
        print('\nLower ppl, lower burstiness, higher top-10% lean model-generated.')
        print('Compare only within one model: the numbers are not portable.')
    print()


if __name__ == '__main__':
    main()
