#!/usr/bin/env python3
"""
Re-run the HC3 Binoculars benchmark with the model pair the paper's author
suggested (ahans30/Binoculars#22), on a Modal GPU.

Four Binoculars configurations over the SAME 442-document HC3 sample, so the
things that could explain our weak 0.883 are separated:

  bino              official implementation, Qwen2.5-0.5B pair, bf16, 512-token
                    truncation. Their defaults, their code, only the model names
                    swapped. This is what was asked for.
  bino_ref_fp32     same, float32. Isolates dtype.
  bino_ours_fp32    OUR sliding-window implementation, Qwen2.5 pair, full
                    document, float32. Isolates our reimplementation.
  bino_q3_fp32      official implementation, the Qwen3-0.6B pair we originally
                    reported. The control: if this also lands near 0.88, the
                    pair was the problem, not our code.

Plus perplexity and top-10 share under Qwen2.5-0.5B-Instruct, so the
"perplexity beat Binoculars" comparison is apples to apples on the new pair.

Sampling is copied verbatim from benchmark.py (seed 0) so the document set is
identical to the shipped hc3_scores.csv.
"""
import modal

app = modal.App("voiceprint-binoculars")

# Their requirements.txt pins transformers 4.31.0, which predates Qwen2, so we
# install a modern 4.x and use their binoculars/ package unmodified.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.45,<5",
        "numpy<3",
        "huggingface_hub>=0.26",
        "sentencepiece",
        "accelerate",
    )
    .run_commands("git clone --depth 1 https://github.com/ahans30/Binoculars /opt/Binoculars")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)

cache = modal.Volume.from_name("voiceprint-hf-cache", create_if_missing=True)

MAXLEN, STRIDE = 1024, 512
CHUNK = 512
MIN_TOKENS = 64


# --------------------------------------------------------------- our scorers

def sentences_cv(text):
    import re
    import statistics as st
    s = [len(x.split()) for x in re.split(r'(?<=[.!?])\s+', text) if len(x.split()) >= 3]
    if len(s) < 2 or not st.mean(s):
        return float('nan')
    return st.pstdev(s) / st.mean(s)


def ppl_and_top10(text, tok, model, device):
    """Verbatim from benchmark.py, moved to the GPU."""
    import math
    import torch
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    n = ids.size(0)
    if n < 2:
        return None, None
    nlls, ranks, start = [], [], 0
    while start < n - 1:
        end = min(start + MAXLEN, n)
        with torch.no_grad():
            lg = model(ids[start:end].unsqueeze(0).to(device)).logits[0].float()
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


def bino_ours(text, tok, obs, perf, device):
    """Verbatim from benchmark.py: full document, sliding 512 with 1 overlap."""
    import torch
    import torch.nn.functional as F
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    n = ids.size(0)
    if n < MIN_TOKENS:
        return None, None, None
    ce_s, x_s, c, start = 0.0, 0.0, 0, 0
    while start < n - 1:
        end = min(start + CHUNK, n)
        ch = ids[start:end].unsqueeze(0).to(device)
        with torch.no_grad():
            pl = perf(ch).logits[0].float()
            ol = obs(ch).logits[0].float()
        tgt = ids[start + 1:end].to(device)
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
    if not c:
        return None, None, None
    lp, xp = ce_s / c, x_s / c
    return lp / xp, lp, xp


# ------------------------------------------------------------------- metrics

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
    import math
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted((s for s, l in zip(scores, labels) if l == 0), reverse=True)
    if not pos or not neg:
        return float('nan')
    k = int(math.floor(target_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else neg[-1]
    return sum(1 for s in pos if s > thr) / len(pos)


# ---------------------------------------------------------------- the sample

def hc3_sample(n):
    """Copied from benchmark.py so the document set is identical."""
    import json
    import random
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('Hello-SimpleAI/HC3', 'all.jsonl', repo_type='dataset')
    ds = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                ds.append(json.loads(line))
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
    per = max(1, n // max(1, len(srcs)))
    hum, mac = [], []
    for src in srcs:
        pool = by_src[src]
        random.shuffle(pool)
        for h, m in pool[:per]:
            hum.append((h, 0, src))
            mac.append((m, 1, src))
    return hum + mac


@app.function(image=image, gpu="L4", volumes={"/cache": cache},
              timeout=5400, retries=0)
def run(n: int = 250):
    import io
    import csv
    import sys
    import time
    import torch
    sys.path.insert(0, '/opt/Binoculars')
    from binoculars.detector import Binoculars, DEVICE_1
    from binoculars.metrics import perplexity as ref_perplexity, entropy as ref_entropy
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    docs = hc3_sample(n)
    say('sampled %d documents (%d human / %d machine)'
        % (len(docs), sum(1 for d in docs if d[1] == 0),
           sum(1 for d in docs if d[1] == 1)))
    say('gpu: %s' % torch.cuda.get_device_name(0))

    scores = {}        # column -> {idx: value}

    def put(col, i, v):
        scores.setdefault(col, {})[i] = v

    # ---- official implementation, several configs -------------------------
    def ref_terms(det, text):
        """compute_score's own three lines, kept open so the terms are visible."""
        enc = det._tokenize([text])
        obs_logits, perf_logits = det._get_logits(enc)
        ppl = ref_perplexity(enc, perf_logits)
        xppl = ref_entropy(obs_logits.to(DEVICE_1), perf_logits.to(DEVICE_1),
                           enc.to(DEVICE_1), det.tokenizer.pad_token_id)
        return float(ppl[0] / xppl[0]), float(ppl[0]), float(xppl[0])

    REF_CONFIGS = [
        ('bino',          'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct', True),
        ('bino_ref_fp32', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct', False),
        ('bino_q3_fp32',  'Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-0.6B', False),
    ]

    for col, obs_id, perf_id, bf16 in REF_CONFIGS:
        say('\n=== official impl: %s / %s  %s ==='
            % (obs_id, perf_id, 'bfloat16' if bf16 else 'float32'))
        t0 = time.time()
        det = Binoculars(observer_name_or_path=obs_id,
                         performer_name_or_path=perf_id,
                         use_bfloat16=bf16)
        # prove the open-coded version equals their shipped entry point
        chk_open = ref_terms(det, docs[0][0])[0]
        chk_ship = det.compute_score(docs[0][0])
        say('  compute_score %.6f vs open-coded %.6f  (delta %.2e)'
            % (chk_ship, chk_open, abs(chk_ship - chk_open)))
        say('  loaded dtype: %s' % next(det.performer_model.parameters()).dtype)
        for i, (text, label, src) in enumerate(docs):
            try:
                s, lp, xp = ref_terms(det, text)
                put(col, i, s)
                put(col + '_logppl', i, lp)
                put(col + '_xppl', i, xp)
            except Exception as e:
                say('  doc %d failed: %s' % (i, str(e)[:70]))
            if (i + 1) % 100 == 0:
                say('  %d/%d  %.0fs' % (i + 1, len(docs), time.time() - t0))
        del det
        torch.cuda.empty_cache()
        say('  done in %.0fs' % (time.time() - t0))

    # ---- our sliding-window implementation, Qwen2.5 pair ------------------
    say('\n=== our impl (full document, sliding 512, fp32): Qwen2.5 pair ===')
    t0 = time.time()
    dev = 'cuda:0'
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    obs = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', torch_dtype=torch.float32).to(dev).eval()
    perf = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float32).to(dev).eval()
    say('  dtypes: observer %s performer %s'
        % (next(obs.parameters()).dtype, next(perf.parameters()).dtype))
    assert next(perf.parameters()).dtype == torch.float32
    for i, (text, label, src) in enumerate(docs):
        try:
            s, lp, xp = bino_ours(text, tok, obs, perf, dev)
            if s is not None:
                put('bino_ours_fp32', i, s)
                put('bino_ours_logppl', i, lp)
                put('bino_ours_xppl', i, xp)
        except Exception as e:
            say('  doc %d failed: %s' % (i, str(e)[:70]))
        if (i + 1) % 100 == 0:
            say('  %d/%d  %.0fs' % (i + 1, len(docs), time.time() - t0))
    del obs
    torch.cuda.empty_cache()
    say('  done in %.0fs' % (time.time() - t0))

    # ---- perplexity + top-10 under the new performer ---------------------
    say('\n=== perplexity / top-10 share: Qwen2.5-0.5B-Instruct, fp32 ===')
    t0 = time.time()
    for i, (text, label, src) in enumerate(docs):
        try:
            p, t10 = ppl_and_top10(text, tok, perf, dev)
            if p is not None:
                put('ppl', i, p)
                put('top10', i, t10)
        except Exception as e:
            say('  doc %d failed: %s' % (i, str(e)[:70]))
        if (i + 1) % 100 == 0:
            say('  %d/%d  %.0fs' % (i + 1, len(docs), time.time() - t0))
    say('  done in %.0fs' % (time.time() - t0))

    # ---- write the CSV ---------------------------------------------------
    cols = ['bino', 'bino_logppl', 'bino_xppl',
            'bino_ref_fp32', 'bino_ref_fp32_logppl', 'bino_ref_fp32_xppl',
            'bino_ours_fp32', 'bino_ours_logppl', 'bino_ours_xppl',
            'bino_q3_fp32', 'bino_q3_fp32_logppl', 'bino_q3_fp32_xppl',
            'ppl', 'top10']
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['idx', 'label', 'source', 'words', 'burstiness'] + cols)
    for i, (text, label, src) in enumerate(docs):
        cv = sentences_cv(text)
        row = [i, label, src, len(text.split()),
               '' if cv != cv else round(cv, 5)]
        for c in cols:
            v = scores.get(c, {}).get(i)
            row.append('' if v is None else round(v, 6))
        w.writerow(row)

    # ---- report ----------------------------------------------------------
    labels_by_idx = {i: d[1] for i, d in enumerate(docs)}
    say('\n' + '=' * 72)
    say('HC3  %d docs  .  higher = more machine-like' % len(docs))
    say('=' * 72)
    say('%-34s %7s %11s %11s' % ('score', 'AUROC', 'TPR@1%FPR', 'TPR@5%FPR'))
    say('-' * 72)
    SPECS = [('Binoculars Qwen2.5 bf16 (theirs)', 'bino', -1),
             ('Binoculars Qwen2.5 fp32 (theirs)', 'bino_ref_fp32', -1),
             ('Binoculars Qwen2.5 fp32 (ours)', 'bino_ours_fp32', -1),
             ('Binoculars Qwen3 fp32 (theirs)', 'bino_q3_fp32', -1),
             ('perplexity Qwen2.5-Instruct', 'ppl', -1),
             ('top-10 share Qwen2.5-Instruct', 'top10', +1),
             ('burstiness', None, -1)]
    for name, col, sign in SPECS:
        if col is None:
            vals = {i: sentences_cv(d[0]) for i, d in enumerate(docs)}
            vals = {i: v for i, v in vals.items() if v == v}
        else:
            vals = scores.get(col, {})
        if len(vals) < 20:
            say('%-34s %7s' % (name, 'n/a'))
            continue
        sc = [sign * v for v in vals.values()]
        lb = [labels_by_idx[i] for i in vals]
        say('%-34s %7.3f %10.1f%% %10.1f%%'
            % (name, auroc(sc, lb), 100 * tpr_at_fpr(sc, lb, 0.01),
               100 * tpr_at_fpr(sc, lb, 0.05)))

    say('\nmedian numerator / denominator by class (official impl, Qwen2.5 bf16)')
    for lab, tag in ((0, 'human  '), (1, 'machine')):
        for term in ('bino_logppl', 'bino_xppl'):
            vals = sorted(v for i, v in scores.get(term, {}).items()
                          if labels_by_idx[i] == lab)
            if vals:
                say('  %s  %-12s %.4f' % (tag, term, vals[len(vals) // 2]))

    # how many fall on the AI side of the paper's shipped thresholds
    say('\nshipped Falcon thresholds applied unchanged')
    for col in ('bino', 'bino_q3_fp32'):
        vals = scores.get(col, {})
        if not vals:
            continue
        for thr, tname in ((0.9015310749276843, 'accuracy'),
                           (0.8536432310785527, 'low-fpr')):
            m = [i for i, v in vals.items() if v < thr and labels_by_idx[i] == 1]
            h = [i for i, v in vals.items() if v < thr and labels_by_idx[i] == 0]
            nm = sum(1 for i in vals if labels_by_idx[i] == 1)
            nh = sum(1 for i in vals if labels_by_idx[i] == 0)
            say('  %-14s %-9s %.4f -> TPR %.1f%%  FPR %.1f%%'
                % (col, tname, thr, 100 * len(m) / max(1, nm),
                   100 * len(h) / max(1, nh)))

    return buf.getvalue(), '\n'.join(log)


@app.local_entrypoint()
def main(n: int = 250, out: str = 'hc3_scores_qwen25.csv'):
    csv_text, report = run.remote(n)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_text)
    with open(out.replace('.csv', '.log.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print('\nwrote %s and %s' % (out, out.replace('.csv', '.log.txt')))
