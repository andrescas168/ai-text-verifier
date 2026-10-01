# Voiceprint

**Which sentences read as machine-written — not a verdict on who wrote it.**

AI text detectors are unreliable, and the honest ones say so. Detectors flagged
roughly **61% of TOEFL essays by non-native English writers** as AI-generated
([Liang et al., 2023](https://arxiv.org/abs/2304.02819)), because careful,
correct, slightly formal prose is low-perplexity prose. OpenAI withdrew its own
classifier in 2023 for poor accuracy.

So this tool does not output a probability that a human wrote something. It
shows you **which specific sentences** a language model finds easy to predict,
and it ships its own failure modes as first-class documentation.

The useful output is not a score. It is: *these three closing sentences are
boilerplate, and the two with concrete numbers in them are not.*

## Install

```bash
git clone https://github.com/andrescas168/ai-text-verifier && cd ai-text-verifier
python3 -m venv ~/venvs/voiceprint
~/venvs/voiceprint/bin/pip install -r requirements.txt   # torch (CPU ok) + transformers
```

`detect` needs nothing but Python 3.8+. Only `ppl` and `bino` need the venv.
Point the CLI at your venv with `VP_VENV=~/venvs/voiceprint` if you use a
different path.

## Use

```bash
./voiceprint detect  draft.docx     # stylometry, stdlib only, instant
./voiceprint ppl     draft.docx     # per-sentence perplexity (Qwen3-0.6B)
./voiceprint bino    draft.docx     # Binoculars, loads 2 models
./voiceprint both    draft.docx
```

Reads `.docx` `.md` `.txt` `.pdf`.

```
REWRITE CANDIDATES — lowest perplexity prose
      4.4  ..........  This requires careful planning, clear communication, and a deep...
      4.5  ..........  When combined with a culture of continuous improvement, these...

CARRIES THE MOST SIGNAL — highest perplexity
     19.5  ##########  Moreover, measuring outcomes is essential to ensuring that...

  document perplexity 7.5 · sentence burstiness 0.54 · top-10 share 87.8%
```

## The three methods

**`detect.py`** — surface stylometry, no model, no dependencies. Burstiness
(coefficient of variation of sentence length), mean sentence length, MATTR
lexical richness, hapax ratio, tricolon rate, sentence-opener diversity, two
punctuation rates, and a two-tier tell lexicon that keeps strong tells
(`delve`, `tapestry`, `testament`) separate from business-normal vocabulary
(`leverage`, `robust`, `streamline`) that real documents legitimately contain.

**`ppl.py`** — per-token negative log-likelihood under a causal LM, in context,
with a sliding window so documents over 1024 tokens are never truncated. Reports
per-sentence perplexity first, then document perplexity and GLTR rank buckets
([Gehrmann et al., 2019](https://arxiv.org/abs/1906.04043)).

**`binoculars.py`** — [Binoculars](https://arxiv.org/abs/2401.12070) (Hans et
al., 2024). Perplexity divided by cross-perplexity across a base/instruct model
pair. The denominator is the point: raw perplexity punishes anyone whose writing
is merely unusual, while cross-perplexity measures how much the two models
disagree about the text in general, so unusual-but-human writing raises both
terms and cancels.

## Reproducible results

Both files are in `samples/` so every number below can be re-run:

- `control_ai.txt` — deliberate LLM business-register prose, written as a
  positive control (215 words).
- `human_public_domain.txt` — the opening of *Moby-Dick* (1851, public domain),
  as a known-human negative control (954 words).

| | AI control | Human control |
|---|---|---|
| perplexity (Qwen3-0.6B) | **7.5** | **63.0** |
| perplexity (GPT-2) | 17.2 | 87.9 |
| top-10 share | **87.8%** | 60.9% |
| Binoculars | **0.9751** | 1.0205 |
| stylometric burstiness | 0.25 | 0.75 |
| tricolon rate | 0.36 | 0.12 |
| em-dashes per 1k words | **0.0** | **13.4** |
| `detect` verdict | mixed | reads natural |

## What testing actually showed

**Swapping the scorer mattered more than anything else.** The method assumes the
scoring model approximates the generator, which is weak for a 2019-era GPT-2 on
2026 text. Moving to Qwen3-0.6B widened the separation from **5.1x to 8.4x**.
`--model gpt2` still works for comparison.

**The em-dash folklore did not survive contact with data.** The popular belief is
that em-dashes signal AI. Measured here, the *human* control uses **13.4 per
1,000 words** and the AI control uses **zero**. Em-dash and semicolon rates are
reported but deliberately **not scored**; they measure house style, not
authorship.

**Structural lines have to be filtered or they dominate.** Headings, ALL-CAPS
lines and `ROLE | Company | Dates` table rows score near-zero perplexity and
crowd out real prose. `is_prose()` drops sub-8-word lines, ALL-CAPS headings,
` | ` title rows and lines with no terminal punctuation, and reports the count.

**Genre dominates authorship.** Résumés and other list-heavy documents run high
tricolon rates and low burstiness no matter who wrote them. Treat any verdict on
a structured document as noise.

**High perplexity can mean novelty, not humanity.** Text dense with unusual
proper nouns and jargon scores high simply because no model has seen it.

**Binoculars thresholds do not transfer between model pairs.** The paper's
0.9015 / 0.8536 are Falcon-7B numbers. Every score measured here landed at
0.97–1.03, *above* both, so applying the published thresholds would have
labelled a deliberately machine-written control as human.

## Limitations

- **Binoculars is implemented but under-powered here.** The Qwen3-0.6B
  base/instruct pair is too closely matched: cross-perplexity tracks perplexity
  almost exactly, compressing the spread to 0.045 versus 8.4x from raw
  perplexity. The paper used 7B models. Ordering is correct, resolution is poor.
- **The fairness claim is untested here.** Binoculars exists to cut false
  positives on non-native English. That is not demonstrated by this two-document
  control set. Do not cite these results as evidence of it.
- **The human control is 1851 literary prose**, stylistically distant from
  modern business writing, so it is a generous anchor. Build your own baseline
  from your own writing (see `samples/README.md`); it is far more useful.
- **Thresholds are uncalibrated.** The LOW/mid/HIGH bands in `detect.py` are
  hand-chosen, not fitted to a labelled corpus. Proper calibration would report
  percentiles of a human distribution at a stated false-positive rate.
- **Scores are not portable across models.** Only compare runs using the same
  `--model`; vocabulary size alone shifts the GLTR buckets.
- Under ~200 words the stylometric signals are noise and `detect` refuses a
  verdict. Binoculars needs 64+ tokens.

## Please do not use this to accuse anyone

Every limitation above is a reason this cannot establish authorship. It is built
to improve your own drafts by finding the sentences that read as generic. Using
a number from a tool like this to accuse a student, an applicant or an employee
is exactly the failure mode the research documents, and it falls hardest on
people writing in a second language.

## References

- Hans et al. 2024, *Spotting LLMs With Binoculars* — [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)
- Mitchell et al. 2023, *DetectGPT* — [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)
- Gehrmann et al. 2019, *GLTR* — [arXiv:1906.04043](https://arxiv.org/abs/1906.04043)
- Liang et al. 2023, *GPT detectors are biased against non-native English writers* — [arXiv:2304.02819](https://arxiv.org/abs/2304.02819)
- Sadasivan et al. 2023, *Can AI-Generated Text be Reliably Detected?* — [arXiv:2303.11156](https://arxiv.org/abs/2303.11156)

## License

MIT
