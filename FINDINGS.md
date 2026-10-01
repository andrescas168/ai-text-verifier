# The false-positive rate behind an AI detector's accuracy claim

An AI detector's accuracy claim leaves me with a question: how often does it flag human writing?

If a teacher uses the result to accuse a student, that error matters. A detector can catch machine text and still be a poor basis for judging people. An accuracy figure alone does not tell me how often that happens, or whose writing gets flagged.

I ran a benchmark of four detection methods and published every document score. Two methods performed well. One barely caught anything when I limited false positives. The more sophisticated method underperformed my simplest baseline.

I also found a bug in my own evaluation.

## What I measured

I used [HC3, the Human/ChatGPT Comparison Corpus](https://arxiv.org/abs/2301.07597), introduced by Guo et al. in 2023. My sample contained 221 human answers and 221 ChatGPT answers, stratified across finance, medicine, open question answering, Reddit ELI5 and computer science topics. I filtered both answers in each pair to 80 to 600 words.

Qwen3-0.6B scored the text in float32 on an eight-core CPU. The run took about 21 minutes, roughly 2.9 seconds per document.

I measured perplexity, how predictable the text is to the model; top-10 share, the fraction of actual next tokens among its ten highest-ranked predictions; burstiness, variation in sentence length; and Binoculars, a score that compares two related models.

These are four methods, not a test of four commercial products.

## The column that matters

True positive rate, or TPR, measures how much machine text gets caught. False positive rate, or FPR, measures how much human text gets wrongly flagged.

How much machine text can I catch before the threshold starts flagging human answers? Here I used a 1% false-positive budget. That is an evaluation constraint, not permission to accuse people at that rate.

| Method | AUROC | TPR at 1% FPR | TPR at 5% FPR | FPR at 95% TPR |
|---|---|---|---|---|
| Top-10 share | 0.998 | 95.0% | 98.6% | 0.9% |
| Perplexity | 0.997 | 92.8% | 99.5% | 1.4% |
| Binoculars | 0.883 | 32.1% | 57.0% | 47.5% |
| Burstiness | 0.721 | 0.5% | 11.3% | 87.5% |

AUROC summarizes how well a score ranks human and machine text across thresholds. It does not tell me how much detection survives a low false-positive budget.

Burstiness makes the difference obvious. Its 0.721 AUROC suggests some separation. But at the 1% FPR budget, it catches just 0.5% of machine text. Catching 95% requires wrongly flagging 87.5% of human text.

The thresholds were selected from the same sample I evaluated. These are descriptive benchmark results, not an independently validated operating threshold. With 221 human documents, the low-FPR estimate also rests on very few cases.

## Burstiness changes direction

The familiar claim is that humans vary sentence length while AI produces uniform prose. In this sample, 97.0% of human documents fell inside the machine burstiness range.

The domain results were worse than the pooled result suggests:

| Domain | Burstiness AUROC |
|---|---|
| Medicine | 0.906 |
| Computer science (`wiki_csai`) | 0.831 |
| Reddit ELI5 | 0.755 |
| Finance | 0.455 |
| Open question answering (`open_qa`) | 0.127 |

The open-QA result reverses the expected direction. Human answers were less bursty than ChatGPT's. Finance also falls below chance in the expected direction.

Sentence-length variation can describe a passage. I would not use it to infer who wrote one.

## Perplexity works on this easy case

Top-10 share caught 95.0% of ChatGPT answers within the false-positive budget. Perplexity caught 92.8%. Those are strong results.

The median perplexity was 27.8 for human text and 5.1 for machine text. Only 5.4% of human documents fell inside the machine perplexity range. Qwen found these ChatGPT answers much easier to predict.

But HC3 is an easy case: raw, unedited output from a 2022-era generator, answering questions. It does not test current generators, rewritten output or documents drafted by a person and polished by a model. It does not establish performance on student essays or workplace writing.

I treat these as best-case evidence, not field performance. They are not a mathematical upper bound on every possible dataset, either.

## My small Binoculars pair did worse

[Binoculars](https://arxiv.org/abs/2401.12070) normalizes log-perplexity by a cross-entropy term computed from two models' predictions. The paper also evaluates false positives on non-native English writing. I wanted to see how the approach behaved with a cheap model pair.

I used Qwen3-0.6B-Base as the observer and Qwen3-0.6B as the performer. It reached 0.883 AUROC and caught 32.1% of machine text at the 1% FPR budget. Perplexity alone did better.

Earlier control runs show how the ratio can compress. The AI control's numerator was 2.0132 and its denominator was 2.0645. A separate private human control produced 4.4668 and 4.1848. The denominator moved with the numerator, bringing both ratios close to one.

That helps explain the weaker separation. It does not establish that model size or similarity caused it. The published setup used Falcon-7B models; my experiment evaluates this small Qwen pairing, not Binoculars in general.

The [reference implementation's thresholds](https://github.com/ahans30/Binoculars/blob/main/binoculars/detector.py), 0.9015 and 0.8536, were selected using the Falcon pair. My shipped AI control scored 0.9751, so either threshold would label it human. A published constant is not a calibration for a different model pair.

## The most important question remains untested

Does this benchmark show that detectors penalize non-native English writers? No.

[Liang et al.](https://arxiv.org/abs/2304.02819) found an average false-positive rate of roughly 61% across the detectors they tested on TOEFL essays by non-native English writers. That is their finding. My benchmark cannot confirm it.

I used English-language HC3 answers without an evaluation separating native and non-native writers. English text is not evidence that its author is a native speaker. The good pooled results tell me nothing reliable about that difference.

I have not measured the error rate for the writers this question is about. A method that separates these HC3 answers well still needs testing on the people and writing it would actually judge.

## My threshold bug

My first run reported 0.0% TPR alongside 0.997 AUROC. The discrepancy sent me back to the threshold function.

It took the 1st percentile of human scores instead of the 99th and compared on the wrong side of the threshold. I had mishandled the score direction. The reported TPR was wrong.

Those numbers are not universally impossible together: AUROC and detection at a restrictive threshold measure different things. In my run, they were a warning worth investigating.

Every document score was already on disk. I corrected the evaluation and recomputed the table without repeating the model run.

## What I use the tool for

I use Voiceprint to rank sentences in my own drafts. Low-perplexity lines get another read: did I write something specific, or fill space with a familiar phrase?

A predictable sentence can be clear and necessary. An unpredictable one can be awkward. I make the editing decision. The score cannot determine authorship.

## Reproduce it

The [repository](https://github.com/andrescas168/ai-text-verifier) includes all 442 document scores in `hc3_scores.csv`. To recompute the tables without loading a model:

```bash
git clone https://github.com/andrescas168/ai-text-verifier
cd ai-text-verifier
python3 analyze.py hc3_scores.csv
```

To repeat the scoring, write a new CSV so the shipped results remain available:

```bash
python3 -m venv ~/venvs/voiceprint
~/venvs/voiceprint/bin/pip install -r requirements.txt huggingface_hub
~/venvs/voiceprint/bin/python benchmark.py --n 250 --out hc3_scores_reproduced.csv
python3 analyze.py hc3_scores_reproduced.csv
```

Scoring runs on CPU. Initial setup downloads the public models and corpus; no API keys or document uploads are required. For sentence rankings in your own draft:

```bash
VP_VENV=~/venvs/voiceprint ./voiceprint ppl draft.md
```

The evidence I most need next is an openly licensed corpus of human writing by non-native English speakers. Pre-2021 human text and edited human/AI drafts would also address gaps. The [contributing guide](https://github.com/andrescas168/ai-text-verifier/blob/main/CONTRIBUTING.md) describes the requirements. Worse results would still be results worth publishing.
