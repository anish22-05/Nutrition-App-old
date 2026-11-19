# Evaluation report

`2026-11-18 14:06 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
113 answerable + 13 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.690  [0.61-0.77] |
| Recall@3 | 0.903  [0.85-0.95] |
| Recall@5 | 0.956  [0.92-0.99] |
| Recall@10 | 0.987  [0.96-1.00] |
| MRR@10 | 0.833  [0.78-0.89] |
| Recall@5 (single-chunk questions, n=100) | 0.970  [0.93-1.00] |
| Recall@5 (multi-chunk questions, n=13) | 0.846  [0.65-1.00] |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9687** (mean 0.6534 answerable vs 0.3733 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 15% |
| 0.3 | 100% | 31% |
| 0.35 | 100% | 31% |
| 0.4 | 99% | 46% |
| 0.5 | 97% | 85% |
| 0.6 | 76% | 92% |

## Retrieval misses at k=5 (6)

- **g035** What are the major functions of calcium in the body?  
  gold pages [688] · retrieved pages [655, 656, 51, 682, 652] · evidence ranks [None]
- **g043** What are the definitions of Recommended Dietary Allowance (RDA) and Adequate Intake (AI) in relation to nutrient intake?  
  gold pages [1193] · retrieved pages [754, 754, 754, 755, 756] · evidence ranks [8]
- **g060** What are some key recommendations from the 2015 Dietary Guidelines for Americans aimed at weight management?  
  gold pages [545] · retrieved pages [547, 545, 814, 812, 546] · evidence ranks [6]
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  gold pages [858, 883] · retrieved pages [879, 883, 863, 884, 878] · evidence ranks [8, 2]
- **g110** How does the consumption of whole grains impact blood glucose levels and the risk of Type 2 diabetes compared to other carbohydrate types?  
  gold pages [288, 309] · retrieved pages [309, 1158, 310, 308, 289] · evidence ranks [9, 1]
- **g113** What are the deficiency diseases associated with niacin and vitamin C, and what are their common symptoms?  
  gold pages [53, 631] · retrieved pages [608, 594, 607, 595, 610] · evidence ranks [None, 6]
