## Retrieval ablation

Golden set v1: 40 answerable queries scored, 10 unanswerable queries held out of the averages (recall over an empty relevant set is 0/0, not 0).

Corpus: 3377 papers / 3465 chunks. top_k=10. Commit `010720aa`.

| mode | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.100 | 0.374 | 0.527 | 0.565 | 0.473 | 5 |
| dense | 0.166 | 0.563 | 0.744 | 0.775 | 0.691 | 25 |
| hybrid | 0.158 | 0.533 | 0.682 | 0.776 | 0.649 | 36 |

### recall@10 by query class

| class | bm25 | dense | hybrid |
| --- | ---: | ---: | ---: |
| exact_term | 0.855 | 0.905 | 0.933 |
| paraphrase | 0.206 | 0.615 | 0.397 |
| conceptual | 0.409 | 0.660 | 0.620 |
| multi_hop | 0.637 | 0.796 | 0.779 |

### nDCG@10 by query class

| class | bm25 | dense | hybrid |
| --- | ---: | ---: | ---: |
| exact_term | 0.864 | 0.884 | 0.909 |
| paraphrase | 0.153 | 0.558 | 0.366 |
| conceptual | 0.342 | 0.604 | 0.586 |
| multi_hop | 0.535 | 0.719 | 0.737 |

### Is the difference real?

Paired bootstrap over the same 40 queries, 10,000 resamples, fixed seed. W/L/T counts queries where the right-hand mode won, lost or tied.

| comparison (ndcg@10) | difference | 95% CI | p | W/L/T | significant |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense - bm25 | +0.218 | [+0.128, +0.308] | 0.0001 | 29/7/4 | yes |
| hybrid - bm25 | +0.176 | [+0.122, +0.234] | 0.0001 | 30/4/6 | yes |
| hybrid - dense | -0.041 | [-0.103, +0.020] | 0.1940 | 15/20/5 | no |

### Unanswerable queries

| mode | queries | results returned | returned nothing |
| --- | ---: | ---: | ---: |
| bm25 | 10 | 94 | 0 |
| dense | 10 | 100 | 0 |
| hybrid | 10 | 100 | 0 |

### Where hybrid loses

22 of 40 scored queries, by nDCG@10.

| query | class | hybrid | beaten by | their score | deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| para_07 | paraphrase | 0.290 | dense | 0.855 | 0.565 |
| para_05 | paraphrase | 0.264 | dense | 0.613 | 0.349 |
| para_06 | paraphrase | 0.521 | dense | 0.837 | 0.317 |
| conc_10 | conceptual | 0.588 | dense | 0.897 | 0.309 |
| para_02 | paraphrase | 0.000 | dense | 0.307 | 0.307 |
| para_01 | paraphrase | 0.704 | dense | 1.000 | 0.296 |
| hop_07 | multi_hop | 0.629 | dense | 0.906 | 0.277 |
| exact_06 | exact_term | 0.521 | dense | 0.760 | 0.239 |
| hop_04 | multi_hop | 0.482 | dense | 0.650 | 0.169 |
| conc_04 | conceptual | 0.590 | dense | 0.747 | 0.156 |
| conc_07 | conceptual | 0.849 | dense | 1.000 | 0.151 |
| hop_02 | multi_hop | 0.676 | dense | 0.825 | 0.149 |
| para_09 | paraphrase | 0.000 | dense | 0.123 | 0.123 |
| conc_08 | conceptual | 0.584 | dense | 0.692 | 0.108 |
| hop_01 | multi_hop | 0.920 | dense | 1.000 | 0.080 |
| conc_06 | conceptual | 0.842 | bm25 | 0.883 | 0.041 |
| exact_08 | exact_term | 0.906 | dense | 0.947 | 0.041 |
| exact_04 | exact_term | 0.962 | dense | 0.983 | 0.021 |
| para_03 | paraphrase | 0.983 | dense | 1.000 | 0.017 |
| hop_10 | multi_hop | 0.514 | dense | 0.527 | 0.013 |
| para_08 | paraphrase | 0.299 | dense | 0.308 | 0.009 |
| exact_03 | exact_term | 0.892 | bm25 | 0.893 | 0.001 |

Leaders: recall@1 dense; recall@5 dense; recall@10 dense; mrr hybrid; ndcg@10 dense.

