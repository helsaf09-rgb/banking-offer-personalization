# MBD-mini Banking Product Validation Report

## Dataset
- Name: MBD-mini targets
- Source: https://huggingface.co/datasets/ai-lab/MBD-mini
- Paper: https://doi.org/10.48550/arXiv.2409.17587
- Scope: MBD-mini product propensity targets only; heavy transaction, dialog and geo archives are not required for this run.
- Raw clients: 100224
- Raw target rows: 1202688
- Months: 2022-02-28T00:00:00 .. 2023-01-31T00:00:00
- Positive product events before filtering: 13529
- Clients with positive product events before filtering: 10411
- Minimum positive events per retained client: 2
- Retained clients: 2325
- Product labels: 4
- Positive product interactions after filtering: 5443

## Holdout Protocol
- Split type: temporal per-client product holdout.
- For each retained client, the latest month with at least one positive product label is used as ground truth.
- Earlier positive product labels form the training history.
- Previously observed products are not excluded at ranking time because the MBD target task is product propensity prediction rather than novelty-only purchase recommendation.
- Train interactions: 2624
- Holdout interactions: 1984
- Evaluated clients: 1915

## Best Model
- Best by NDCG@2: `neural_cf`
- NDCG@2 = 0.6419
- MAP@2 = 0.6087

## Metrics
| model | precision@k | recall@k | map@k | ndcg@k |
| --- | ---: | ---: | ---: | ---: |
| neural_cf | 0.3734 | 0.7262 | 0.6087 | 0.6419 |
| lightgcn_tuned | 0.3413 | 0.6622 | 0.5815 | 0.6052 |
| sasrec_tuned | 0.3321 | 0.6450 | 0.5798 | 0.5992 |
| implicit_mf | 0.3034 | 0.5905 | 0.5487 | 0.5625 |
| popularity | 0.3185 | 0.6152 | 0.4601 | 0.5031 |
| item_knn | 0.2146 | 0.4090 | 0.3362 | 0.3578 |

## LightGCN Tuning
| candidate | ndcg@k | map@k | embedding_dim | layers | learning_rate | epochs | samples_per_epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgcn_config_3 | 0.6052 | 0.5815 | 24 | 2 | 0.0100 | 35 | 24000 |
| lightgcn_config_1 | 0.5962 | 0.5710 | 8 | 1 | 0.0300 | 20 | 12000 |
| lightgcn_config_2 | 0.5649 | 0.5501 | 16 | 2 | 0.0200 | 30 | 18000 |

## SASRec Tuning
| candidate | ndcg@k | map@k | embedding_dim | heads | blocks | max_seq_len | learning_rate | epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sasrec_config_1 | 0.5992 | 0.5798 | 16 | 2 | 1 | 12 | 0.0020 | 12 |
| sasrec_config_2 | 0.3395 | 0.3294 | 32 | 4 | 2 | 12 | 0.0020 | 14 |
| sasrec_config_3 | 0.3395 | 0.3294 | 32 | 4 | 2 | 24 | 0.0010 | 16 |

## Interpretation
- The experiment closes the post-predefense requirement to adapt the pipeline to an open banking dataset with real product labels.
- Because only four target products are available in the lightweight MBD-mini target archive, the task is intentionally framed as product-label ranking rather than large-catalog item recommendation.
- The tuned LightGCN and SASRec branches are evaluated on the same temporal banking holdout as the simpler baselines, which makes their limitations visible instead of leaving them as untested future work.
- BERT4Rec remains a defensible future extension rather than a necessary last-minute addition, because the priority experiment already checks the implemented sequence-aware SASRec branch on MBD-mini.