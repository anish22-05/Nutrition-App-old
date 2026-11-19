# Evaluation report

`2026-11-17 18:08 UTC` · commit `cefe714+dirty` · corpus `Human-Nutrition-2020.pdf` (1902 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
113 answerable + 13 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.792  [0.72-0.86] |
| Recall@3 | 0.951  [0.92-0.98] |
| Recall@5 | 0.965  [0.93-0.99] |
| Recall@10 | 0.982  [0.96-1.00] |
| MRR@10 | 0.905  [0.86-0.95] |
| Recall@5 (single-chunk questions, n=100) | 0.980  [0.95-1.00] |
| Recall@5 (multi-chunk questions, n=13) | 0.846  [0.65-1.00] |

## Generation (top-5, similarity gate 0.35)

| metric | score |
|---|---|
| Answer correctness (vs reference, partial = 0.5) | 0.920  [0.88-0.96] |
| Faithfulness (claims supported by context) | 0.989  [0.97-1.00] |
| Citation validity (answers citing real sources) | 1.000  [1.00-1.00] |
| False-refusal rate (answerable but refused; lower is better) | 0.000  [0.00-0.00] |
| Abstention accuracy (unanswerable, refused correctly) | 1.000  [1.00-1.00] |
| Mean latency per answer | 1.79 s |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **0.9932** (mean 0.6543 answerable vs 0.3087 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 23% |
| 0.3 | 100% | 31% |
| 0.35 | 100% | 62% |
| 0.4 | 99% | 85% |
| 0.5 | 96% | 85% |
| 0.6 | 76% | 100% |

## Retrieval misses at k=5 (5)

- **g035** What are the major functions of calcium in the body?  
  gold pages [688] · retrieved pages [655, 656, 657, 51, 655] · evidence ranks [None]
- **g060** What are some key recommendations from the 2015 Dietary Guidelines for Americans aimed at weight management?  
  gold pages [545] · retrieved pages [812, 545, 812, 546, 814] · evidence ranks [None]
- **g105** What are the differences between qualified health claims and nutrient content claims regarding the evidence required to support them?  
  gold pages [776, 773] · retrieved pages [776, 113, 775, 776, 773] · evidence ranks [1, 7]
- **g108** How does increasing fruit and vegetable intake relate to cancer prevention and overall mortality rates?  
  gold pages [794, 1150] · retrieved pages [794, 794, 799, 796, 799] · evidence ranks [1, 7]
- **g113** What are the deficiency diseases associated with niacin and vitamin C, and what are their common symptoms?  
  gold pages [53, 631] · retrieved pages [607, 1020, 595, 608, 633] · evidence ranks [10, 8]

## Answers not fully correct (15)

- **g015** What is the term used to describe the force that drives water movement through a selectively permeable membrane due to differences in solute concentration?  
  missing: It is the force exerted by solutes at different concentrations on either side of a selectively permeable membrane.
- **g024** What are the end products of the digestion of glucose, lipids, and proteins?  
  missing: The digestion of glucose results in 3 molecules of glucose.; Proteins are converted into 3 molecules of amino acids.
- **g025** What are the symptoms of heat stroke?  
  missing: Symptoms of heat stroke include unconsciousness.
- **g026** What are mycotoxins and how can they affect health?  
  missing: They are contained in and around mold threads and may spread throughout the food.
- **g060** What are some key recommendations from the 2015 Dietary Guidelines for Americans aimed at weight management?  
  missing: This is essential for maintaining a balance with physical activity.
- **g061** What is the temperature set point of the human body, and what happens if the body temperature deviates too far from this point?  
  contradicts: If the temperature is too low or too high, enzymes stop functioning and metabolism halts, which can lead to muscle failure or even death.
- **g084** What are the eight foods that account for ninety percent of food allergies?  
  contradicts: The eight foods that cause ninety percent of food allergies include shellfish, wheat, and soy.
- **g086** What are the main causes of food infections?  
  missing: These infections grow inside the body and lead to symptoms.
- **g101** How do the feeding recommendations for breastfed infants compare to those for formula-fed infants during the first six months of life?  
  missing: Breastfed infants continue to breastfeed on demand.; Formula-fed infants have a more structured feeding schedule with specific ounce recommendations.
- **g102** What are the dietary sources of vitamin K that contribute to its role in blood coagulation?  
  missing: Broccoli provides 133% of the daily value of vitamin K.; Asparagus provides 28% of the daily value of vitamin K.; Cabbage provides 47% of the daily value of vitamin K.
- **g105** What are the differences between qualified health claims and nutrient content claims regarding the evidence required to support them?  
  missing: Nutrient content claims must meet specific requirements and cannot be made unless the food actually meets those criteria.; A food cannot be labeled as low-fat unless it contains three or fewer grams of fat.; The standard for nutrient content claims is stricter than that for qualified health claims.
- **g106** What nutritional challenges do toddlers face regarding iron intake and self-feeding, and how can parents address these issues?  
  missing: Parents should provide age-appropriate foods that toddlers can self-feed, such as sliced fruits and mashed vegetables.
- **g108** How does increasing fruit and vegetable intake relate to cancer prevention and overall mortality rates?  
  missing: This includes deaths from various cancers.
- **g110** How does the consumption of whole grains impact blood glucose levels and the risk of Type 2 diabetes compared to other carbohydrate types?  
  missing: Whole grains have beneficial effects compared to other carbohydrate sources that may cause quicker spikes in blood sugar.
- **g113** What are the deficiency diseases associated with niacin and vitamin C, and what are their common symptoms?  
  contradicts: Scurvy presents symptoms such as bleeding gums, joint pain, and poor wound healing.

## Unanswerable questions that got an answer instead of a refusal (0)

