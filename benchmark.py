#!/usr/bin/env python3
"""
Evaluate the detectors against HC3 (Human ChatGPT Comparison Corpus).

Reports AUROC and, more importantly, TPR at a low false-positive rate. A
detector that catches 90% of machine text while falsely accusing 10% of humans
is not usable on real people; the low-FPR column is the one that matters.

Writes every per-document score to CSV as it goes, so a long CPU run is
resumable and the raw data stays inspectable.

Usage: benchmark.py --n 150 --out hc3_scores.csv
"""
import os
import re
import csv
import math
import time
import argparse
import statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MAXLEN, STRIDE = 1024, 512


# ------------------------------------------------------------------ metrics

def auroc(scores, labels):
    """P(score[machine] > score[human]), via rank sum. No sklearn needed."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):                      # average ranks over ties
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(1 for _, l in pairs if l == 1)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    rpos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rpos - npos * (npos + 1) / 2.0) / (npos * nneg)


def tpr_at_fpr(scores, labels, target_fpr):
    """Highest TPR with FPR <= target.

    Higher score = machine, so classify as machine when score > threshold.
    Allow k = floor(target_fpr * n_neg) human docs above the threshold; the
    threshold is then the (k+1)-th largest human score.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted((s for s, l in zip(scores, labels) if l == 0), reverse=True)
    if not pos or not neg:
        return float('nan')
    k = int(math.floor(target_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else neg[-1]
    return sum(1 for s in pos if s > thr) / len(pos)


# ------------------------------------------------------------------ scoring

def sentences_cv(text):
    s = [len(x.split()) for x in re.split(r'(?<=[.!?])\s+', text) if len(x.split()) >= 3]
    if len(s) < 2 or not st.mean(s):
        return float('nan')
    return st.pstdev(s) / st.mean(s)


def ppl_and_top10(text, tok, model):
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    n = ids.size(0)
    if n < 2:
        return None, None
    nlls, ranks, start = [], [], 0
    while start < n - 1:
        end = min(start + MAXLEN, n)
        with torch.no_grad():
            lg = model(ids[start:end].unsqueeze(0)).logits[0].float()
        lp = torch.log_softmax(lg, dim=-1)
        lo = 0 if start == 0 else STRIDE - 1
        for i in range(lo, end - start - 1):
            t = ids[start + i + 1].item()
            nlls.append(-lp[i][t].item())
            ranks.append(int((lp[i] > lp[i][t]).sum().item()))
        if end == n:
            break
        start += STRIDE
    if not nlls:
        return None, None
    return math.exp(sum(nlls) / len(nlls)), sum(1 for r in ranks if r < 10) / len(ranks)


def binoculars(text, tok, obs, perf):
    import torch.nn.functional as F
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    n = ids.size(0)
    if n < 64:
        return None
    ce_s, x_s, c, start = 0.0, 0.0, 0, 0
    while start < n - 1:
        end = min(start + 512, n)
        ch = ids[start:end].unsqueeze(0)
        with torch.no_grad():
            pl = perf(ch).logits[0].float()
            ol = obs(ch).logits[0].float()
        tgt = ids[start + 1:end]
        if tgt.numel() == 0:
            break
        p, o = pl[:-1], ol[:-1]
        ce_s += F.cross_entropy(p, tgt, reduction='sum').item()
        x_s += -(torch.softmax(o, -1) * torch.log_softmax(p, -1)).sum(-1).sum().item()
        c += tgt.numel()
        del pl, ol
        if end == n:
            break
        start = end - 1
    return (ce_s / c) / (x_s / c) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150, help='documents per class')
    ap.add_argument('--out', default='hc3_scores.csv')
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--no-bino', action='store_true')
    a = ap.parse_args()

    # HC3 ships a legacy loading script that current `datasets` rejects, so
    # pull the raw jsonl from the Hub and parse it directly.
    import json
    from huggingface_hub import hf_hub_download
    print('loading HC3 ...')
    path = hf_hub_download('Hello-SimpleAI/HC3', 'all.jsonl', repo_type='dataset')
    ds = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                ds.append(json.loads(line))
    print('  %d HC3 rows' % len(ds))

    # Stratify across HC3's five domains. Sampling in file order would take
    # everything from reddit_eli5 and overstate single-genre performance.
    import random
    random.seed(0)
    by_src = {}
    for row in ds:
        src = row.get('source', '?')
        h = (row.get('human_answers') or [None])[0]
        m = (row.get('chatgpt_answers') or [None])[0]
        if not h or not m:
            continue
        if not (80 <= len(h.split()) <= 600) or not (80 <= len(m.split()) <= 600):
            continue
        by_src.setdefault(src, []).append((h.strip(), m.strip()))

    srcs = sorted(by_src)
    per = max(1, a.n // max(1, len(srcs)))
    hum, mac = [], []
    for src in srcs:
        pool = by_src[src]
        random.shuffle(pool)
        for h, m in pool[:per]:
            hum.append((h, 0, src))
            mac.append((m, 1, src))
    print('  domains: %s' % ', '.join('%s=%d' % (s_, min(per, len(by_src[s_]))) for s_ in srcs))
    docs = hum + mac
    print('sampled %d human + %d machine' % (len(hum), len(mac)))

    print('loading %s ...' % a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()
    obs = perf = None
    if not a.no_bino:
        print('loading Binoculars pair ...')
        obs = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.float32)
        perf = model
        obs.eval()
    torch.set_num_threads(os.cpu_count() or 4)

    f = open(a.out, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['idx', 'label', 'source', 'words', 'ppl', 'top10', 'burstiness', 'bino'])

    t0 = time.time()
    rows = []
    for i, (text, label, src) in enumerate(docs):
        try:
            p, t10 = ppl_and_top10(text, tok, model)
            cv = sentences_cv(text)
            b = None if a.no_bino else binoculars(text, tok, obs, perf)
            rows.append((label, p, t10, cv, b))
            w.writerow([i, label, src, len(text.split()),
                        '' if p is None else round(p, 4),
                        '' if t10 is None else round(t10, 5),
                        '' if cv != cv else round(cv, 5),
                        '' if b is None else round(b, 6)])
            f.flush()
        except Exception as e:
            print('  doc %d failed: %s' % (i, str(e)[:70]))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print('  %d/%d   %.1fs elapsed, %.2fs/doc' % (i + 1, len(docs), el, el / (i + 1)))
    f.close()

    print('\n' + '=' * 72)
    print('HC3  %d human / %d machine  ·  %s' % (len(hum), len(mac), a.model))
    print('=' * 72)
    print('%-26s %7s %12s %12s' % ('metric', 'AUROC', 'TPR@1%FPR', 'TPR@5%FPR'))
    print('-' * 72)

    # direction: for each metric, the sign that makes "higher = machine"
    specs = [('perplexity', 1, lambda r: r[1], -1),
             ('top-10 share', 2, lambda r: r[2], +1),
             ('burstiness', 3, lambda r: r[3], -1),
             ('Binoculars', 4, lambda r: r[4], -1)]
    for name, _, get, sign in specs:
        pairs = [(sign * get(r), r[0]) for r in rows
                 if get(r) is not None and get(r) == get(r)]
        if len(pairs) < 20:
            print('%-26s %7s %12s %12s' % (name, 'n/a', '-', '-'))
            continue
        sc = [p[0] for p in pairs]
        lb = [p[1] for p in pairs]
        print('%-26s %7.3f %11.1f%% %11.1f%%'
              % (name, auroc(sc, lb),
                 100 * tpr_at_fpr(sc, lb, 0.01),
                 100 * tpr_at_fpr(sc, lb, 0.05)))
    print('\nraw scores -> %s' % a.out)
    print('TPR@1%%FPR is the honest column: how much machine text you catch')
    print('while wrongly flagging 1 in 100 humans.')


if __name__ == '__main__':
    main()
