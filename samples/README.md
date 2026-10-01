# samples

Two controls, so every number in the README can be reproduced.

- `control_ai.txt` — deliberate LLM business-register prose, written as a
  positive control. 215 words.
- `human_public_domain.txt` — the opening of *Moby-Dick* (Melville, 1851,
  public domain), as a known-human negative control. 954 words.

```bash
./voiceprint ppl    samples/control_ai.txt samples/human_public_domain.txt
./voiceprint bino   samples/control_ai.txt samples/human_public_domain.txt
./voiceprint detect samples/control_ai.txt samples/human_public_domain.txt
```

## Build your own baseline

The shipped human control is 19th-century literary prose, which is a generous
anchor: it is stylistically far from modern business writing, so separation
looks easier than it is. A baseline from your own writing is far more useful.

1. Pick 1,000+ words you wrote yourself, before you ever used an LLM.
2. `./voiceprint ppl your_writing.txt`
3. Record the perplexity, burstiness and top-10 share.

The point of the tool is the distance between a draft and *your* numbers, not a
universal verdict.
