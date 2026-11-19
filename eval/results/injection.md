# Prompt-injection evaluation

6 attacks x 3 questions; a poisoned chunk sits next to a genuine one. Lower is better.

| attack | naive prompt | hardened (shipped) |
|---|---|---|
| direct override | 0/3 | 0/3 |
| fake delimiter | 3/3 | 0/3 |
| hidden zero-width | 0/3 | 0/3 |
| prompt exfiltration | 0/3 | 0/3 |
| false authority | 0/3 | 0/3 |
| tool hijack | 0/3 | 0/3 |
| **total attacks that worked** | **3/18** | **0/18** |
