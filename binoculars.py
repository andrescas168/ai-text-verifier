#!/usr/bin/env python3
"""
Voiceprint: Binoculars (Hans et al., 2024, arXiv:2401.12070).

Two models from one family:
  observer  = the base model
  performer = the instruct-tuned sibling

  score = log-perplexity(performer vs actual tokens)
          ---------------------------------------------
          cross-perplexity( observer dist || performer dist )

LOW score => machine-generated.

Why the denominator matters, and why this is the fair version: raw perplexity
punishes anyone whose writing is merely unusual, which is how detectors end up
flagging ~61% of non-native English essays. Cross-perplexity measures how much
the two models disagree about this text in general. Unusual-but-human writing
raises the numerator AND the denominator, so it cancels. Text that is
predictable in the specific way sampling makes it predictable does not.

Thresholds are model-pair specific. The paper's 0.9015 / 0.8536 are for a
Falcon-7B pair and DO NOT transfer. Run --baseline to see this pair's spread.

Usage: binoculars.py [--observer ID] [--performer ID] FILE [FILE ...]
"""
import os
import re
import sys
import zipfile
import argparse
import subprocess

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

OBSERVER = 'Qwen/Qwen3-0.6B-Base'
PERFORMER = 'Qwen/Qwen3-0.6B'
CHUNK = 512          # logits are [L, 151936] fp32; 512 keeps that ~312 MB/model
MIN_TOKENS = 64      # the paper considers shorter text unreliable


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


# ------------------------------------------------------------------- scoring

def binoculars(text, tok, observer, performer):
    """Returns (score, log_ppl, x_ppl, n_tokens). Accumulates over chunks so
    full logit tensors are never held for the whole document."""
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    n = ids.size(0)
    if n < MIN_TOKENS:
        return None, None, None, n

    ce_sum = 0.0      # performer cross-entropy against the real next token
    x_sum = 0.0       # H(observer_dist, performer_dist)
    count = 0

    start = 0
    while start < n - 1:
        end = min(start + CHUNK, n)
        chunk = ids[start:end].unsqueeze(0)
        with torch.no_grad():
            perf_logits = performer(chunk).logits[0].float()
            obs_logits = observer(chunk).logits[0].float()

        # position i predicts token i+1
        tgt = ids[start + 1:end]
        perf = perf_logits[:-1]
        obs = obs_logits[:-1]
        if tgt.numel() == 0:
            break

        ce = F.cross_entropy(perf, tgt, reduction='sum')
        obs_p = torch.softmax(obs, dim=-1)
        perf_lp = torch.log_softmax(perf, dim=-1)
        x = -(obs_p * perf_lp).sum(-1).sum()

        ce_sum += ce.item()
        x_sum += x.item()
        count += tgt.numel()

        del perf_logits, obs_logits, obs_p, perf_lp
        if end == n:
            break
        start = end - 1        # overlap one token so nothing is unpredicted

    if count == 0:
        return None, None, None, n
    log_ppl = ce_sum / count
    x_ppl = x_sum / count
    return log_ppl / x_ppl, log_ppl, x_ppl, n


def bar(score, lo=0.70, hi=1.15, width=30):
    f = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    i = int(round(f * (width - 1)))
    return '[' + '.' * i + '|' + '.' * (width - 1 - i) + ']'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observer', default=OBSERVER)
    ap.add_argument('--performer', default=PERFORMER)
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()

    print('observer : %s' % a.observer)
    print('performer: %s' % a.performer)
    tok = AutoTokenizer.from_pretrained(a.observer)
    obs = AutoModelForCausalLM.from_pretrained(a.observer, dtype=torch.float32)
    perf = AutoModelForCausalLM.from_pretrained(a.performer, dtype=torch.float32)
    obs.eval(); perf.eval()
    torch.set_num_threads(os.cpu_count() or 4)

    tv = tok.vocab_size
    if obs.config.vocab_size != perf.config.vocab_size:
        print('WARNING: vocab mismatch between the two models; scores invalid.')

    rows = []
    for f in a.files:
        try:
            text = re.sub(r'[ \t]+', ' ', load(f)).strip()
            s, lp, xp, n = binoculars(text, tok, obs, perf)
            name = os.path.basename(f)
            if s is None:
                print('\n%-46s  only %d tokens, need %d+' % (name, n, MIN_TOKENS))
                continue
            rows.append((name, s, lp, xp, n))
            print('\n%s' % name)
            print('  tokens %d   log-ppl %.4f   x-ppl %.4f' % (n, lp, xp))
            print('  Binoculars  %.4f  %s' % (s, bar(s)))
        except Exception as e:
            print('\n%s FAILED: %s' % (os.path.basename(f), e))

    if len(rows) > 1:
        rows.sort(key=lambda r: r[1])
        print('\n' + '=' * 70)
        print('%-44s %10s %8s' % ('file (lowest = most machine-like)', 'score', 'tokens'))
        print('-' * 70)
        for name, s, lp, xp, n in rows:
            print('%-44s %10.4f %8d' % (name[:44], s, n))
        spread = rows[-1][1] - rows[0][1]
        print('\nobserved spread: %.4f' % spread)
        print('Thresholds are NOT calibrated for this pair. The paper\'s 0.9015 /')
        print('0.8536 are Falcon-7B numbers and do not transfer. Use the spread')
        print('between your own known-AI and known-human anchors instead.')
    print()


if __name__ == '__main__':
    main()
