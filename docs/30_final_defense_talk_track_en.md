# Final Defense Talk Track, 8-10 Minutes

## Slide 1. Thesis Topic and Result
Good afternoon. My thesis is devoted to personalization of customer offers based on banking services user transaction activity.

The main result of the work is a reproducible recommendation pipeline. It includes data generation, exploratory analysis, several recommendation models, offline top-K evaluation, real-data validation, and a working API prototype. So the project is not only a theoretical discussion of recommender systems. It is an end-to-end research and software contour that can be reproduced and extended.

## Slide 2. Motivation and Problem Statement
The motivation comes from the structure of modern banking ecosystems. A bank now offers not only cards or deposits, but also insurance, investment products, partner subscriptions, bundled services, and loyalty offers. In such a setting, sending the same offer to every client is inefficient. The system needs to select a short and relevant top-K list for each user.

I use transaction history as the main behavioral signal. Transactions are useful because they reflect real user actions rather than declared preferences. Formally, the task is to rank candidate banking offers for a user based on the user's transaction history. The output is a top-K list, and the quality is evaluated with Precision@K, Recall@K, MAP@K, and NDCG@K.

## Slide 3. Data Strategy
The main difficulty is data access. Public banking transaction logs with a flexible catalog of offers are extremely limited. For this reason, the main benchmark is synthetic, but it is banking-like and fully documented.

The generated dataset contains 800 users, 112,626 transactions, 15 offers, and 6,442 positive user-offer interactions. Users are divided into behavior segments, such as daily life, family, traveler, student, investor, and digital professional. These segments influence transaction categories, transaction amounts, and offer relevance.

To avoid making the thesis purely synthetic, I added two real-data checks. Online Retail is used as a real transaction-log transferability test. MBD-mini from Sber AI Lab is used as a real banking-domain product-label holdout.

## Slide 4. Pipeline Architecture
The project is organized as a pipeline rather than a single notebook. The main stages are data generation and preprocessing, exploratory analysis, model training, offline evaluation, reporting, and service delivery.

All key artifacts are saved in the project structure: CSV metrics, JSON summaries, plots, Markdown reports, Word documents, and PDF exports. This is important because the results can be checked and reproduced later. The same evaluation functions are reused across different models, which makes the comparison more consistent.

The final layer is a service prototype. It includes a FastAPI endpoint and a Streamlit interface for browsing users and recommendations.

## Slide 5. Implemented Models
The thesis implements seven recommendation approaches.

The first group contains classic and neural baselines: profile similarity, implicit matrix factorization, Neural Collaborative Filtering, and item-based logic. The second group contains more advanced branches: hybrid semantic ranking, time-decay modeling, LightGCN, and SASRec.

Profile similarity is the interpretable baseline. Hybrid semantic adds TF-IDF representation of offer descriptions. Time-decay adds recency weighting, so recent behavior has more influence than older behavior. LightGCN is the graph-based branch, and SASRec is the sequence-aware branch.

I did not add BERT4Rec as a last-minute model. Instead, I focused on tuning the already implemented LightGCN and SASRec branches on a real banking holdout. This was the more defensible experimental priority.

## Slide 6. Main Synthetic Benchmark Results
On the synthetic benchmark, the strongest single-run model is time-decay. It reaches NDCG@5 of 0.2936 and MAP@5 of 0.2387.

Profile similarity and hybrid semantic are very close: NDCG@5 is 0.2906 and 0.2888. This means that the interpretable behavioral profile is already strong, and the improvement from a more complex model is modest.

The important point is that time-decay does not increase Precision@5 and Recall@5 compared with the profile baseline. Its advantage is mainly in MAP and NDCG, which means it improves the ordering of relevant offers inside the top-K list. This is still useful, because in recommendation systems the order of the first few offers matters a lot.

## Slide 7. Robustness Analysis
I also checked the stability of the results. The bootstrap analysis shows a positive average gain for time-decay over the profile baseline, but the confidence interval includes zero. So I do not claim a statistically decisive victory.

The multi-seed benchmark gives a similar message. Hybrid semantic, time-decay, and profile similarity form a close top group. The ranking between them can change depending on the random seed.

Segment analysis adds more detail. Time-decay is more useful for segments where behavior changes over time, for example family, student, and investor. For more stable segments, a static profile can work almost as well.

The conclusion is deliberately careful: the thesis is not about declaring one universal winner. It is about building a reproducible comparison and understanding when each modeling idea is useful.

## Slide 8. Real-Data Validation
The first real-data check is Online Retail. After cleaning and filtering, it contains 2,992 users, 1,499 items, and 201,226 positive interactions. On this dataset, implicit matrix factorization performs best with NDCG@10 of 0.1296 and MAP@10 of 0.1025.

This result is important because the model ranking changes outside the synthetic setup. Collaborative structure becomes more useful on real purchase logs. At the same time, Online Retail is still a retail proxy, not a banking dataset.

Therefore, after pre-defense I added MBD-mini validation. MBD-mini is an open banking dataset from Sber AI Lab. I used the product propensity targets, because they provide monthly product labels and make the validation feasible on a regular laptop.

After filtering, the MBD-mini experiment used 2,325 clients, 4 product labels, and 1,984 holdout product-label events. The best model was Neural CF with NDCG@2 of 0.6419 and MAP@2 of 0.6087. Tuned LightGCN reached NDCG@2 of 0.6052, and tuned SASRec reached 0.5992. This closes the main post-pre-defense requirement: the pipeline was adapted to real banking product labels, and the graph and sequence branches were tuned on the same banking holdout.

## Slide 9. Service Prototype
The practical part is a working prototype. The API endpoint is GET /recommend/{user_id}, with a top_k parameter. It returns a JSON response with the user id, a known-user flag, and a ranked list of recommendations.

There is also a Streamlit interface, which makes the recommendations easier to inspect. The current serving configuration uses the time-decay profile model, because it is strong, interpretable, and easy to explain.

This service layer is not a production banking platform, but it demonstrates that the research result can be wrapped into an application interface. For a project-oriented master's thesis, this is an important practical output.

## Slide 10. Contributions, Limitations, and Future Work
The main contributions are: a reproducible recommendation pipeline, a documented synthetic banking-like benchmark, comparison of interpretable, neural, graph-based, and sequence-aware models, robustness analysis, real-data validation on Online Retail and MBD-mini, and a working API prototype.

The main limitations are also clear. The primary benchmark is synthetic. The offer catalog is compact. The evaluation is offline. MBD-mini validation uses product targets, but not the full transaction, dialog, and geo modalities.

Future work should therefore focus on richer real banking data, deeper use of the full MBD modalities, heavier sequential models such as BERT4Rec, and online evaluation such as A/B testing or uplift modeling.

To summarize, the thesis shows that transaction-driven personalization can be studied in a reproducible way even under limited access to real banking data. The final result is not a claim of production-ready uplift, but a complete and extensible research and engineering foundation.
