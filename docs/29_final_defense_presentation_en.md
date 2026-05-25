# Final Defense Presentation: 10-Slide Structure

## Slide 1. Thesis Topic and Result
- Topic: Personalization of customer offers based on banking services user transaction activity.
- Student: Elena Safonova.
- Supervisor: Dmitry Ilvovsky.
- Main claim: the thesis delivers a reproducible recommendation pipeline, offline validation, real-data checks, and a working API prototype.

## Slide 2. Motivation and Problem Statement
- Banks operate broad product ecosystems: cards, deposits, insurance, partner subscriptions, investment products, and services.
- Mass offer distribution creates irrelevant communication and lowers customer trust.
- Transaction history is a strong behavioral signal because it reflects actual financial behavior.
- Task: rank candidate banking offers for a user and return a top-K personalized list.
- Evaluation framing: implicit-feedback recommendation with Precision@K, Recall@K, MAP@K, and NDCG@K.

## Slide 3. Data Strategy
- Main benchmark: synthetic banking-like dataset.
- Scale: 800 users, 112,626 transactions, 15 banking offers, 6,442 positive interactions.
- Why synthetic data: public banking transaction logs with an arbitrary offer catalog are rarely available.
- Real-data checks:
  - Online Retail as a real transaction-log transferability test.
  - MBD-mini from Sber AI Lab as a real banking product-label holdout.

## Slide 4. Pipeline Architecture
- Data generation and preprocessing.
- Exploratory analysis and automatic reporting.
- Model training under a unified interface.
- Offline top-K evaluation.
- Report artifacts in CSV, JSON, PNG, and Markdown.
- Serving layer: FastAPI endpoint and Streamlit interface.

## Slide 5. Implemented Models
- Profile similarity baseline.
- Implicit matrix factorization.
- Neural Collaborative Filtering.
- Hybrid semantic model with TF-IDF offer descriptions.
- Time-decay model for recent behavioral signals.
- LightGCN as a graph-based branch.
- SASRec as a sequence-aware branch.
- BERT4Rec is kept as future work, because the final priority was tuning the already implemented LightGCN and SASRec branches.

## Slide 6. Main Synthetic Benchmark Results
- Best single-run model: time-decay.
- Time-decay: NDCG@5 = 0.2936, MAP@5 = 0.2387.
- Profile similarity: NDCG@5 = 0.2906, MAP@5 = 0.2349.
- Hybrid semantic: NDCG@5 = 0.2888, MAP@5 = 0.2334.
- Interpretation: recency improves ordering quality inside the top-K list, but the margin is modest.

## Slide 7. Robustness Analysis
- Bootstrap analysis: time-decay has a positive average gain, but confidence intervals include zero.
- Multi-seed benchmark: hybrid semantic, time-decay, and profile similarity form a close top group.
- Segment analysis: time-decay helps more for segments where behavior changes over time, such as family, student, and investor.
- Main conclusion: the thesis does not claim one universal winner; it provides a reproducible comparison and honest limitation analysis.

## Slide 8. Real-Data Validation
- Online Retail:
  - 2,992 users, 1,499 items, 201,226 positive interactions.
  - Best model: implicit MF, NDCG@10 = 0.1296, MAP@10 = 0.1025.
  - Meaning: the pipeline transfers to a real transaction log, but this is still a retail proxy.
- MBD-mini:
  - Open banking dataset from Sber AI Lab.
  - Used component: product propensity targets.
  - 2,325 retained clients, 4 product labels, 1,984 holdout product-label events.
  - Best model: Neural CF, NDCG@2 = 0.6419, MAP@2 = 0.6087.
  - Tuned LightGCN NDCG@2 = 0.6052; tuned SASRec NDCG@2 = 0.5992.

## Slide 9. Service Prototype
- FastAPI endpoint: GET /recommend/{user_id}?top_k=5.
- Returns a JSON response with user id, known-user flag, and ranked recommendations.
- Streamlit interface supports browsing users and recommendations.
- The service uses the time-decay profile configuration as the practical serving model.
- Value: the thesis result is not only an offline experiment, but also a working demonstration system.

## Slide 10. Contributions, Limitations, and Future Work
- Contributions:
  - reproducible banking-offer recommendation pipeline;
  - synthetic benchmark with documented generation assumptions;
  - comparison of interpretable, neural, graph-based, and sequence-aware models;
  - robustness analysis;
  - real-data validation on Online Retail and MBD-mini;
  - API and UI prototype.
- Limitations:
  - synthetic main benchmark;
  - compact offer catalog;
  - offline evaluation;
  - MBD-mini validation uses product targets, not the full multimodal dataset.
- Future work:
  - richer real banking data;
  - full MBD transaction, dialog, and geo modalities;
  - BERT4Rec-style bidirectional sequential model;
  - online A/B or uplift evaluation.
