# Evaluation report

`2026-11-17 17:56 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
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

## Generation (top-5, similarity gate 0.35)

| metric | score |
|---|---|
| Answer correctness (vs reference, partial = 0.5) | 0.951  [0.92-0.98] |
| Faithfulness (claims supported by context) | 1.000  [1.00-1.00] |
| Citation validity (answers citing real sources) | 1.000  [1.00-1.00] |
| False-refusal rate (answerable but refused; lower is better) | 0.000  [0.00-0.00] |
| Abstention accuracy (unanswerable, refused correctly) | 1.000  [1.00-1.00] |
| Mean latency per answer | 1.55 s |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9864** (mean 0.6668 answerable vs 0.3706 unanswerable).

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

## Answers not fully correct (10)

- **g004** What is the process by which cells break down glucose to capture energy?  
  missing: Cellular respiration is a controlled burning of glucose.
- **g024** What are the end products of the digestion of glucose, lipids, and proteins?  
  missing: The digestion of glucose results in 3 molecules of glucose.; Proteins are converted into 3 molecules of amino acids.
- **g025** What are the symptoms of heat stroke?  
  missing: Symptoms of heat stroke include unconsciousness.
- **g026** What are mycotoxins and how can they affect health?  
  missing: They are contained in and around mold threads.; They may spread throughout the food.
- **g060** What are some key recommendations from the 2015 Dietary Guidelines for Americans aimed at weight management?  
  missing: This is essential for maintaining a balance with physical activity.
- **g084** What are the eight foods that account for ninety percent of food allergies?  
  contradicts: The eight foods that cause ninety percent of food allergies are milk, eggs, peanuts, tree nuts, and fish.
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  missing: Breastfed infants continue to breastfeed on demand.; Formula-fed infants have a more structured feeding schedule with specific ounce recommendations.
- **g102** What are the dietary sources of vitamin K that contribute to its role in blood coagulation?  
  missing: Vitamin K helps to form a plug when a blood vessel is injured.; Broccoli provides 133%, asparagus provides 28%, and cabbage provides 47% of the daily value of vitamin K.
- **g105** What are the differences between qualified health claims and nutrient content claims regarding the evidence required to support them?  
  missing: The standard for nutrient content claims is stricter than that for qualified health claims.
- **g108** How does increasing fruit and vegetable intake relate to cancer prevention and overall mortality rates?  
  missing: This includes deaths from various cancers.

## Unanswerable questions that got an answer instead of a refusal (0)

