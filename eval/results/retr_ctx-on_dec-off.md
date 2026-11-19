# Evaluation report

`2026-11-18 14:07 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
113 answerable + 13 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.730  [0.65-0.81] |
| Recall@3 | 0.929  [0.89-0.97] |
| Recall@5 | 0.965  [0.93-0.99] |
| Recall@10 | 0.991  [0.97-1.00] |
| MRR@10 | 0.865  [0.82-0.91] |
| Recall@5 (single-chunk questions, n=100) | 0.970  [0.93-1.00] |
| Recall@5 (multi-chunk questions, n=13) | 0.923  [0.81-1.00] |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9728** (mean 0.648 answerable vs 0.3644 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 23% |
| 0.3 | 100% | 31% |
| 0.35 | 100% | 31% |
| 0.4 | 99% | 54% |
| 0.5 | 99% | 85% |
| 0.6 | 74% | 92% |

## Retrieval misses at k=5 (5)

- **g013** What should be avoided to prevent liver damage after consuming alcohol?  
  gold pages [480] · retrieved pages [486, 478, 488, 487, 487] · evidence ranks [9]
- **g035** What are the major functions of calcium in the body?  
  gold pages [688] · retrieved pages [656, 655, 655, 657, 682] · evidence ranks [None]
- **g043** What are the definitions of Recommended Dietary Allowance (RDA) and Adequate Intake (AI) in relation to nutrient intake?  
  gold pages [1193] · retrieved pages [754, 754, 754, 756, 755] · evidence ranks [9]
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  gold pages [858, 883] · retrieved pages [883, 878, 879, 884, 884] · evidence ranks [10, 1]
- **g110** How does the consumption of whole grains impact blood glucose levels and the risk of Type 2 diabetes compared to other carbohydrate types?  
  gold pages [288, 309] · retrieved pages [309, 308, 310, 310, 1158] · evidence ranks [7, 1]
