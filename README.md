# Voiceprint

**An open evidence base on whether AI text detection actually works — and the
tool that produced it.**

Most tools in this space publish an accuracy number and not a false-positive
rate. That is not an oversight. A detector vendor cannot publish the rate at
which it falsely accuses people without damaging the product, so the most
important number in the field is the one nobody with distribution will state.

This repo states it, measures it on a public corpus, and ships the cases where
the methods fail.

## Headline result

Measured on [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), 221 human
and 221 ChatGPT answers, stratified across five domains, scored with
Qwen3-0.6B:

| metric | AUROC | **TPR @ 1% FPR** | TPR @ 5% FPR | FPR @ 95% TPR |
|---|---|---|---|---|
| **top-10 share** | 0.998 | **95.0%** | 98.6% | 0.9% |
| **perplexity** | 0.997 | **92.8%** | 99.5% | 1.4% |
| Binoculars | 0.883 | 32.1% | 57.0% | 47.5% |
| burstiness | 0.721 | 0.5% | 11.3% | 87.5% |

**On raw, unedited ChatGPT output written in native English, perplexity-based
detection genuinely works: 95% of machine text caught while wrongly flagging 1
human in 100.**

That is also the easy case, and it is not the case anyone is actually worried
about. See [what this does not show](#what-this-does-not-show).

## The number that matters is TPR at 1% FPR

AUROC flatters detectors. Burstiness scores 0.721 AUROC here, which sounds like
a weak-but-real signal. At a 1% false-positive rate it catches **0.5%** of
machine text. It is not a weak signal; it is nothing.

The question that decides whether a tool can be pointed at a person is: *how
much machine text does it catch while wrongly accusing only 1 human in 100?*
Report that, or you are not reporting anything.

## Three findings

**1. Burstiness barely works.** The "AI writes with uniform sentence length"
heuristic is the most widely repeated idea in this space. Measured: AUROC 0.721
overall, and **97% of human documents fall inside the machine range**. In the
`open_qa` domain it scores AUROC **0.127**, meaning it is not merely useless
there but actively inverted — human answers were *less* bursty than ChatGPT's.

**2. Binoculars underperforms raw perplexity here.** [Binoculars](https://arxiv.org/abs/2401.12070)
is the stronger method in the literature and the one designed to be fair. With a
Qwen3-0.6B base/instruct pair it reaches 0.883 AUROC against perplexity's 0.997.
The cause is visible in the intermediate values: cross-perplexity tracks
perplexity almost exactly, because the two models are too closely matched to
disagree usefully. The paper used 7B models. This is a limitation of the cheap
pairing, not of the method.

**3. Published thresholds do not transfer.** Binoculars' paper gives 0.9015 and
0.8536 for a Falcon-7B pair. Scores here land at a different operating point
entirely; applying those constants would misclassify a deliberately
machine-written control as human. Any threshold must be recalibrated per model
pair.

Two smaller ones, from earlier testing:

- **The em-dash folklore is backwards.** On the shipped controls, the human text
  uses 13.4 em-dashes per 1,000 words and the AI text uses zero. Punctuation
  rates are reported but deliberately not scored.
- **Structural lines dominate if unfiltered.** Headings and
  `ROLE | Company | Dates` rows score near-zero perplexity and crowd out real
  prose, so they are detected and excluded.

## What this does not show

This is the part other tools leave out.

**HC3 is the easy case on every axis that matters.** It is raw, unedited model
output, from a 2022-era generator, answering questions, written by native
English speakers. Every number above is a **best case**, not field performance.

Specifically, these results say nothing about:

- **Non-native English writers.** The central fairness claim in this field is
  that detectors punish them: [Liang et al. 2023](https://arxiv.org/abs/2304.02819)
  found ~61% of TOEFL essays flagged as AI. HC3 cannot test this, so this repo
  has not tested it. **This is the biggest gap and the one most worth closing.**
- **Edited or hybrid text.** Human-drafted then AI-polished, or AI-drafted then
  rewritten, is the normal case in real documents and the hardest to detect.
  Untested here.
- **Modern generators.** HC3 is ChatGPT-era.
- **Anything but English.**
- **Short text.** Under ~200 words the stylometric signals are noise; Binoculars
  needs 64+ tokens.

A detector that scores 95% on HC3 may well be useless, or harmful, on a
second-language student's essay. Nothing here licenses that use.

## The tool

```bash
git clone https://github.com/andrescas168/ai-text-verifier && cd ai-text-verifier
python3 -m venv ~/venvs/voiceprint
~/venvs/voiceprint/bin/pip install -r requirements.txt
```

```bash
./voiceprint detect  draft.docx     # stylometry, stdlib only, instant
./voiceprint ppl     draft.docx     # per-sentence perplexity (Qwen3-0.6B)
./voiceprint bino    draft.docx     # Binoculars, loads 2 models
```

Reads `.docx` `.md` `.txt` `.pdf`. `detect` needs only Python 3.8+.

The per-document output ranks **sentences**, not documents, because "these three
sentences read as boilerplate" is actionable and a score is not:

```
REWRITE CANDIDATES — lowest perplexity prose
      8.7  ..........  I would welcome the opportunity to bring that experience to...
     13.7  ###.......  My focus has been on identifying manual and repetitive work...

CARRIES THE MOST SIGNAL — highest perplexity
     47.9  ##########  This work also identified automation opportunities that...
```

### Reproduce the benchmark

```bash
~/venvs/voiceprint/bin/pip install datasets huggingface_hub
~/venvs/voiceprint/bin/python benchmark.py --n 250 --out hc3_scores.csv
python3 analyze.py hc3_scores.csv
```

Roughly 3 seconds per document on an 8-core CPU, no GPU required. Per-document
scores are written as it runs, so the raw data stays inspectable and the run is
resumable.

## Contributing

**The most valuable contribution is not code — it is text this benchmark cannot
reach**, starting with human writing by non-native English speakers. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the ranked list of evidence gaps and the
corpus requirements.

Contributions that make the numbers **worse** will be merged. The point is to be
right, not impressive.

## Please do not use this to accuse anyone

Every limitation above is a reason this cannot establish authorship. It is built
to improve your own drafts by finding sentences that read as generic. Using a
number from a tool like this against a student, an applicant or an employee is
the failure mode the research documents, and it falls hardest on people writing
in a second language.

## References

- Hans et al. 2024, *Spotting LLMs With Binoculars* — [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)
- Mitchell et al. 2023, *DetectGPT* — [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)
- Gehrmann et al. 2019, *GLTR* — [arXiv:1906.04043](https://arxiv.org/abs/1906.04043)
- Liang et al. 2023, *GPT detectors are biased against non-native English writers* — [arXiv:2304.02819](https://arxiv.org/abs/2304.02819)
- Sadasivan et al. 2023, *Can AI-Generated Text be Reliably Detected?* — [arXiv:2303.11156](https://arxiv.org/abs/2303.11156)
- Guo et al. 2023, *HC3 / How Close is ChatGPT to Human Experts?* — [arXiv:2301.07597](https://arxiv.org/abs/2301.07597)

## License

MIT
