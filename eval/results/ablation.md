# Retrieval ablation (113 answerable questions, 1902 chunks)

| strategy | R@1 | R@3 | R@5 | R@10 | MRR | ms/query (excl. embed) |
|---|---|---|---|---|---|---|
| vector (current) | 0.730 | 0.920 | 0.965 | 0.991 | 0.865 | 0 |
| bm25 only | 0.721 | 0.872 | 0.916 | 0.947 | 0.840 | 2 |
| hybrid (vector+bm25, RRF) | 0.730 | 0.951 | 0.956 | 0.996 | 0.872 | 1 |
| vector + rerank MiniLM-L6 | 0.792 | 0.951 | 0.965 | 0.982 | 0.905 | 926 |
| hybrid + rerank MiniLM-L6 | 0.792 | 0.951 | 0.965 | 0.978 | 0.905 | 910 |
| hybrid + rerank MiniLM-L12 | 0.792 | 0.960 | 0.982 | 1.000 | 0.913 | 1983 |
