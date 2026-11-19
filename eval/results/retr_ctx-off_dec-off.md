# Evaluation report

`2026-11-18 14:06 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
113 answerable + 13 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.730  [0.65-0.81] |
| Recall@3 | 0.920  [0.87-0.96] |
| Recall@5 | 0.965  [0.93-0.99] |
| Recall@10 | 0.991  [0.97-1.00] |
| MRR@10 | 0.865  [0.82-0.91] |
| Recall@5 (single-chunk questions, n=100) | 0.970  [0.93-1.00] |
| Recall@5 (multi-chunk questions, n=13) | 0.923  [0.81-1.00] |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9864** (mean 0.6668 answerable vs 0.3705 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 15% |
| 0.3 | 100% | 31% |
| 0.35 | 100% | 31% |
| 0.4 | 99% | 46% |
| 0.5 | 98% | 85% |
| 0.6 | 81% | 100% |

## Retrieval misses at k=5 (5)

- **g035** What are the major functions of calcium in the body?  
  gold pages [688] · retrieved pages [655, 656, 51, 682, 652] · evidence ranks [None]
- **g043** What are the definitions of Recommended Dietary Allowance (RDA) and Adequate Intake (AI) in relation to nutrient intake?  
  gold pages [1193] · retrieved pages [754, 754, 755, 754, 752] · evidence ranks [9]
- **g060** What are some key recommendations from the 2015 Dietary Guidelines for Americans aimed at weight management?  
  gold pages [545] · retrieved pages [547, 545, 814, 812, 546] · evidence ranks [6]
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  gold pages [858, 883] · retrieved pages [879, 883, 884, 863, 878] · evidence ranks [9, 2]
- **g110** How does the consumption of whole grains impact blood glucose levels and the risk of Type 2 diabetes compared to other carbohydrate types?  
  gold pages [288, 309] · retrieved pages [309, 308, 310, 310, 1158] · evidence ranks [7, 1]
