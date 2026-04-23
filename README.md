# Abstract

This study investigates the impact of feature engineering and feature fusion strategies on human mobility prediction using the HuMob 2023 dataset. Building on the LP-BERT baseline, trajectory prediction is formulated as a masked sequence imputation task, where missing segments of user trajectories are reconstructed using bidirectional context.

The study introduces three extensions to the baseline framework: spatial data cleaning to mitigate GPS drift, the incorporation of contextual and behavioural features, and an alternative concatenation-based feature fusion strategy. These features include temporal indicators, POI-based spatial representations, LDA-derived functional zones, and mobility motif-based behavioural features.

Experimental results show that spatial data cleaning provides the most significant improvement in performance, highlighting the importance of data quality in mobility modelling. Temporal features contribute the most consistent gains, while additional contextual and behavioural features provide limited or inconsistent improvements. The comparison of fusion strategies shows that concatenation outperforms additive embedding. 

Overall, the findings suggest that data preprocessing and effective feature representation are more critical than increasing feature complexity. This study provides a systematic evaluation of feature contributions and offers practical insights for the design of mobility prediction models.

![image](./figures/Flowchart.png)
