# Answering evaluation

Retriever `dense` at top-10, 5 passages in the prompt, model `openai/gpt-oss-120b`, golden set v1.

## Headline

| measure | count | rate | reading |
| --- | ---: | ---: | ---: |
| abstained on unanswerable | 10/10 | 100% | higher is better |
| abstained on answerable | 4/40 | 10% | lower is better |
| citation precision | 24 answers cited | 100% | cited a passage that existed |
| citation grounding | 40 answerable | 78% | cited a paper labelled relevant |
| answers with no citation | 12 | - | least grounded output possible |
| hallucinated citation numbers | 0 | - | pointed at a passage never supplied |

## Abstention failures

| id | class | query | what went wrong |
| --- | ---: | ---: | ---: |
| para_02 | paraphrase | improving a search engine by re-reading its own top  | refused a real question |
| para_05 | paraphrase | how bright was the beam when the collisions were rec | refused a real question |
| hop_01 | multi_hop | neutrino mass and cosmological structure formation | refused a real question |
| hop_06 | multi_hop | axion searches with haloscopes and with colliders | refused a real question |

## What is not measured here

Citation *precision* asks whether a cited number referred to a passage that
was actually supplied, and citation *grounding* asks whether that passage was
a paper the golden set labels relevant. Neither asks whether the cited passage
genuinely supports the sentence it is attached to. That requires a human
reader or a judge model, and neither has been used, so no claim is made about
it.
