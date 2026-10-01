# Contributing

This project is an evidence base first and a tool second. The most valuable
contribution is not code. **It is text that the current benchmark cannot reach.**

## The gap that matters most

The central claim about AI detectors is that they are unfair to people writing
in a second language. [Liang et al. (2023)](https://arxiv.org/abs/2304.02819)
found detectors flagged roughly **61% of TOEFL essays by non-native English
writers** as AI-generated.

**This repository cannot currently test that claim.** HC3 is native-English
question answering. Every number here is measured on a population that
detectors treat kindly. That is the single biggest hole in the evidence, and it
is the one that decides whether any of this is safe to use on real people.

If you have, or know of, an openly licensed corpus of **human writing by
non-native English speakers**, that is the most useful thing you can bring.

## Other evidence gaps, roughly in order of value

1. **Non-native English human writing** (above).
2. **Human text from before 2021.** Anything written after ChatGPT shipped is
   contaminated: "human" samples may be LLM-assisted. Pre-2021 corpora give a
   clean negative class.
3. **Edited and hybrid text.** Real documents are rarely pure. Human-written
   then AI-polished, or AI-drafted then heavily rewritten, is the common case
   and the hardest one. Almost nothing public covers it.
4. **Non-English languages.** Every result here is English-only.
5. **Technical, legal and medical registers.** Formal domain prose is
   low-perplexity by nature and should be over-flagged. Quantify it.
6. **Newer generators.** HC3 is ChatGPT-era. Text from current models is the
   live question.
7. **A larger Binoculars pair.** The 0.6B base/instruct pair is too closely
   matched (see README). A 7B pair on a GPU would test the method properly.

## Adding a corpus

Datasets are referenced, not vendored. Add a loader that returns
`(text, label, source)` where `label` is `0` for human and `1` for machine,
then open a PR with the measured results table.

```python
# benchmark.py, alongside the HC3 loader
def load_yourcorpus(n):
    """Return [(text, label, source), ...]; 0 = human, 1 = machine."""
```

Requirements, so results stay comparable:

- **License the data permissively**, and say what the licence is. Do not commit
  scraped text of uncertain provenance.
- **Stratify across domains.** Sampling in file order overstates performance;
  the HC3 loader stratifies across five domains for this reason.
- **80–600 words per document.** Shorter is noise; longer is slow and
  unrepresentative.
- **Report TPR at 1% FPR**, not just AUROC. See below.
- **No personal data.** No real resumes, letters, student work or anything
  identifying a living person, even with a name removed.

## Why TPR at 1% FPR is the number that matters

AUROC flatters detectors. A detector with 0.95 AUROC sounds excellent and can
still falsely flag one student in ten.

The question that decides whether a tool is usable on real people is: *how much
machine text does it catch while wrongly accusing only 1 human in 100?* That is
`TPR@1%FPR`, and it is usually far worse than the AUROC implies. Report it.

If your contribution makes the numbers **worse**, that is a result worth
publishing, and it will be merged. The point of this repo is to be right, not
to be impressive.

## Code contributions

Welcome, but please keep the two properties that make this repo worth existing:

- **Limitations stay first-class.** Every claim ships with what would falsify
  it. No marketing language in the README.
- **No hosted upload service.** This repo will not become a site where people
  paste someone else's writing to judge it. That is the misuse the research
  documents, and it is out of scope here on purpose.

Run the benchmark before and after your change and include both tables.
