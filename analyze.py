#!/usr/bin/env python3
"""Recompute benchmark metrics from the saved per-document scores.

Convention throughout: score is oriented so HIGHER = more machine-like.
"""
import csv
import math
import sys
from collections import defaultdict


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(1 for _, l in pairs if l == 1)
    nneg = len(pairs) - npos
    if not npos or not nneg:
        return float('nan')
    rpos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rpos - npos * (npos + 1) / 2.0) / (npos * nneg)


def tpr_at_fpr(scores, labels, target_fpr):
    """Highest TPR with FPR <= target.

    Higher score = machine, so we classify as machine when score > threshold.
    Allow k = floor(target_fpr * n_neg) human documents above the threshold;
    the threshold is then the (k+1)-th largest human score.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted((s for s, l in zip(scores, labels) if l == 0), reverse=True)
    if not pos or not neg:
        return float('nan')
    k = int(math.floor(target_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else neg[-1]
    return sum(1 for s in pos if s > thr) / len(pos)


def fpr_at_tpr(scores, labels, target_tpr):
    """Lowest FPR that still catches target_tpr of machine text."""
    pos = sorted((s for s, l in zip(scores, labels) if l == 1), reverse=True)
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float('nan')
    k = max(0, min(len(pos) - 1, int(math.ceil(target_tpr * len(pos))) - 1))
    thr = pos[k]
    return sum(1 for s in neg if s > thr) / len(neg)


METRICS = [('perplexity', 'ppl', -1),
           ('top-10 share', 'top10', +1),
           ('burstiness', 'burstiness', -1),
           ('Binoculars', 'bino', -1)]


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def table(rows, title):
    print('\n' + '=' * 76)
    nh = sum(1 for r in rows if r['label'] == '0')
    nm = len(rows) - nh
    print('%s   (%d human / %d machine)' % (title, nh, nm))
    print('=' * 76)
    print('%-16s %7s %11s %11s %12s %12s'
          % ('metric', 'AUROC', 'TPR@1%FPR', 'TPR@5%FPR', 'FPR@80%TPR', 'FPR@95%TPR'))
    print('-' * 76)
    for name, col, sign in METRICS:
        sc, lb = [], []
        for r in rows:
            v = r.get(col, '')
            if v == '' or v is None:
                continue
            try:
                sc.append(sign * float(v))
                lb.append(int(r['label']))
            except ValueError:
                continue
        if len(sc) < 20 or len(set(lb)) < 2:
            print('%-16s %7s' % (name, 'n/a'))
            continue
        print('%-16s %7.3f %10.1f%% %10.1f%% %11.1f%% %11.1f%%'
              % (name, auroc(sc, lb),
                 100 * tpr_at_fpr(sc, lb, 0.01),
                 100 * tpr_at_fpr(sc, lb, 0.05),
                 100 * fpr_at_tpr(sc, lb, 0.80),
                 100 * fpr_at_tpr(sc, lb, 0.95)))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'hc3_scores.csv'
    rows = load(path)
    table(rows, 'HC3 — all domains')

    by = defaultdict(list)
    for r in rows:
        by[r['source']].append(r)
    for src in sorted(by):
        if len(by[src]) >= 40:
            table(by[src], 'HC3 — %s' % src)

    # where do the two classes actually sit?
    print('\n' + '=' * 76)
    print('distributions (median)')
    print('=' * 76)
    print('%-16s %14s %14s' % ('metric', 'human', 'machine'))
    print('-' * 76)
    for name, col, _ in METRICS:
        for lab, tag in ((0, 'human'), (1, 'machine')):
            vals = sorted(float(r[col]) for r in rows
                          if r['label'] == str(lab) and r.get(col))
            if not vals:
                continue
            med = vals[len(vals) // 2]
            if tag == 'human':
                h = med
            else:
                print('%-16s %14.3f %14.3f' % (name, h, med))
    print()
