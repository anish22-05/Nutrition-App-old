# Evaluation report

`2026-11-18 14:07 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
113 answerable + 13 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.681  [0.60-0.76] |
| Recall@3 | 0.898  [0.85-0.95] |
| Recall@5 | 0.943  [0.90-0.98] |
| Recall@10 | 0.987  [0.96-1.00] |
| MRR@10 | 0.828  [0.78-0.88] |
| Recall@5 (single-chunk questions, n=100) | 0.950  [0.90-0.99] |
| Recall@5 (multi-chunk questions, n=13) | 0.885  [0.77-1.00] |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9646** (mean 0.6344 answerable vs 0.3644 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 23% |
| 0.3 | 100% | 31% |
| 0.35 | 100% | 31% |
| 0.4 | 99% | 54% |
| 0.5 | 96% | 85% |
| 0.6 | 67% | 92% |

## Retrieval misses at k=5 (8)

- **g013** What should be avoided to prevent liver damage after consuming alcohol?  
  gold pages [480] · retrieved pages [486, 478, 488, 487, 487] · evidence ranks [9]
- **g020** What has been the trend in obesity rates among children, including infants and toddlers, over the past thirty years?  
  gold pages [900] · retrieved pages [901, 938, 901, 902, 901] · evidence ranks [9]
- **g035** What are the major functions of calcium in the body?  
  gold pages [688] · retrieved pages [656, 655, 655, 657, 682] · evidence ranks [None]
- **g043** What are the definitions of Recommended Dietary Allowance (RDA) and Adequate Intake (AI) in relation to nutrient intake?  
  gold pages [1193] · retrieved pages [754, 754, 754, 755, 756] · evidence ranks [8]
- **g053** What are the age ranges that define early childhood and the remaining part of childhood?  
  gold pages [910] · retrieved pages [823, 823, 823, 894, 913] · evidence ranks [7]
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  gold pages [858, 883] · retrieved pages [883, 863, 884, 879, 878] · evidence ranks [7, 1]
- **g110** How does the consumption of whole grains impact blood glucose levels and the risk of Type 2 diabetes compared to other carbohydrate types?  
  gold pages [288, 309] · retrieved pages [309, 310, 308, 1158, 289] · evidence ranks [9, 1]
- **g113** What are the deficiency diseases associated with niacin and vitamin C, and what are their common symptoms?  
  gold pages [53, 631] · retrieved pages [608, 595, 607, 594, 631] · evidence ranks [None, 5]
