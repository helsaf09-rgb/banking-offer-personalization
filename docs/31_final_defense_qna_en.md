# Final Defense Q&A Cheat Sheet

## 1. Why did you use synthetic data?
Because open banking transaction logs with a flexible catalog of offers are difficult to access due to privacy, legal, and commercial restrictions. Synthetic data make the benchmark reproducible and allow the full pipeline to be tested end to end. I do not present synthetic data as a replacement for real banking production logs. I present it as a controlled research environment, complemented by Online Retail and MBD-mini validation.

## 2. What exactly does MBD-mini add?
MBD-mini adds a real banking-domain validation layer. Unlike Online Retail, it comes from a banking context and contains monthly product propensity targets. In the final experiment, the pipeline was adapted to these targets: 2,325 clients were retained after filtering, and the holdout contained 1,984 product-label events. This checks that the pipeline is not only a synthetic offer benchmark.

## 3. Why did you use only the targets part of MBD-mini?
The full MBD-mini repository contains larger transaction, dialog, and geo archives. For this thesis stage, the target component was the most relevant and tractable part because it directly contains product labels. It allows a banking product-ranking validation without turning the final thesis into a separate large-scale data engineering project.

## 4. Why is K equal to 2 on MBD-mini?
The lightweight MBD target task contains only four product labels. With such a compact product set, K = 2 is a reasonable top-K setting. It tests whether the model places the most relevant products near the top without making the ranking task artificially trivial.

## 5. Why is the best model different across datasets?
Because the datasets and tasks are different. The synthetic benchmark is category-driven and has explicit offer descriptions, so profile and time-aware models are strong. Online Retail is a larger real user-item purchase graph, where implicit MF benefits from collaborative structure. MBD-mini is a compact banking product-label task, where Neural CF and tuned LightGCN/SASRec become competitive. This is one of the thesis findings: model choice depends on data structure and validation protocol.

## 6. Why did you not implement BERT4Rec?
BERT4Rec is a relevant future direction, but after pre-defense the main experimental priority was to tune the already implemented LightGCN and SASRec branches on real-data holdouts. Adding BERT4Rec at the last moment would increase complexity, but not necessarily improve the scientific reliability of the final comparison. The thesis already contains a sequence-aware SASRec branch, and it was evaluated on MBD-mini.

## 7. Why does time-decay win on the synthetic benchmark?
The synthetic dataset contains temporal behavior patterns. Time-decay gives more weight to recent activity, so it improves the ordering of offers when recent interests matter. The gain is mainly visible in MAP@5 and NDCG@5 rather than Precision@5 and Recall@5, which means it improves ranking order inside the top-K list.

## 8. Is time-decay statistically better than the profile baseline?
Not decisively. The average bootstrap gain is positive, but the confidence interval includes zero. I interpret the result cautiously: time-decay is a useful refinement, but not an unconditional winner. The multi-seed benchmark also shows that time-decay, profile similarity, and hybrid semantic are close.

## 9. Why does Neural CF perform better on MBD-mini than on the synthetic benchmark?
On the synthetic benchmark, the interaction signal is strongly shaped by explicit category logic, so interpretable profile-based methods are very competitive. MBD-mini uses real banking product-label patterns, where nonlinear user-product interactions may be more useful. This makes Neural CF stronger in the compact banking holdout.

## 10. What is the practical value of the API prototype?
The API prototype shows that the recommendation logic can be wrapped into a service interface. It supports GET /recommend/{user_id} and returns a ranked JSON response. This demonstrates practical completion: the thesis is not only an offline metric table, but also a working recommendation service prototype.

## 11. What are the main limitations?
The main benchmark is synthetic. The offer catalog is compact. All evaluation is offline. Online Retail is a retail proxy rather than a banking dataset. MBD-mini validation uses product targets, but not the full transaction, dialog, and geo modalities. These limitations are explicitly stated and guide future work.

## 12. What is the main scientific contribution?
The thesis contributes a reproducible experimental and software pipeline for banking-offer personalization under limited access to real banking data. It compares interpretable, neural, graph-based, and sequence-aware methods; includes robustness checks; validates transferability on real data; and documents the limits of each result.

## 13. What would be the first production-oriented next step?
The first step would be to connect the pipeline to a real banking feature store and a larger real offer catalog, then run temporal backtesting and eventually online A/B or uplift evaluation. A production system would also need product eligibility rules, compliance checks, monitoring, and fairness analysis.

## 14. How would you explain the difference between Online Retail and MBD-mini validation?
Online Retail checks whether the ranking pipeline works on a real transaction log with many items. MBD-mini checks whether the pipeline can be adapted to a banking-domain product-label task. Online Retail is broader as a transaction graph, while MBD-mini is closer by domain but more compact by target catalog.

## 15. One-sentence final answer if asked what the thesis proves
The thesis proves that a transaction-based banking-offer recommender can be built as a reproducible end-to-end pipeline, compared across several model families, validated on real-data proxies including MBD-mini, and delivered as a working API prototype, while keeping the limitations of synthetic and offline evaluation explicit.
