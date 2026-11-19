# Evaluation report

`2026-11-17 18:08 UTC` · commit `cefe714+dirty` · corpus `introduction-to-nutrition.pdf` (61 chunks) · embed `text-embedding-3-small` · chat `gpt-4o-mini` · judge `gpt-4o-mini`
50 answerable + 11 unanswerable questions. Values are mean [95% bootstrap CI].

## Retrieval (answerable questions, no similarity cut-off)

| metric | score |
|---|---|
| Recall@1 | 0.880  [0.80-0.95] |
| Recall@3 | 0.960  [0.91-1.00] |
| Recall@5 | 1.000  [1.00-1.00] |
| Recall@10 | 1.000  [1.00-1.00] |
| MRR@10 | 0.952  [0.91-0.99] |
| Recall@5 (single-chunk questions, n=46) | 1.000  [1.00-1.00] |
| Recall@5 (multi-chunk questions, n=4) | 1.000  [1.00-1.00] |

## Generation (top-5, similarity gate 0.35)

| metric | score |
|---|---|
| Answer correctness (vs reference, partial = 0.5) | 0.950  [0.91-0.99] |
| Faithfulness (claims supported by context) | 1.000  [1.00-1.00] |
| Citation validity (answers citing real sources) | 1.000  [1.00-1.00] |
| False-refusal rate (answerable but refused; lower is better) | 0.000  [0.00-0.00] |
| Abstention accuracy (unanswerable, refused correctly) | 1.000  [1.00-1.00] |
| Mean latency per answer | 1.21 s |

## Can similarity alone tell answerable from unanswerable?

AUC of top-1 similarity = **1.0** (mean 0.6227 answerable vs 0.2133 unanswerable).

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.2 | 100% | 46% |
| 0.3 | 100% | 82% |
| 0.35 | 100% | 91% |
| 0.4 | 100% | 91% |
| 0.5 | 92% | 100% |
| 0.6 | 66% | 100% |

## Retrieval misses at k=5 (0)


## Answers not fully correct (5)

- **g011** What are some food sources of Vitamin C?  
  missing: Food sources of Vitamin C include guava.; Food sources of Vitamin C include cabbage.; Food sources of Vitamin C include broccoli.; Food sources of Vitamin C include sprouted grains.
- **g021** What are the essential amino acids and how many are there?  
  missing: There are a total of 22 different amino acids, of which 9 are classified as essential amino acids.
- **g040** What are the food sources of complex carbohydrates and how do they contribute to energy and dietary fiber needs in the body?  
  missing: Cereals like wheat and pulses also provide energy and fulfill the body's requirement for dietary fiber.
- **g042** How do calcium and phosphorus work together with vitamin D to support bone health?  
  missing: Together, they ensure that bones remain healthy and prevent conditions like rickets and osteoporosis.
- **g043** How do the food sources of Vitamin B-complex and Vitamin E compare in terms of their commonality?  
  missing: This overlap in food sources highlights the nutritional value of these items in providing essential vitamins for the body.

## Unanswerable questions that got an answer instead of a refusal (0)

