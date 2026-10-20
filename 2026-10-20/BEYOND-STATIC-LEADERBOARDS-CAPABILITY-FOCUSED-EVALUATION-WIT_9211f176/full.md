# BEYOND STATIC LEADERBOARDS: CAPABILITY-FOCUSED EVALUATION WITH STRUCTURAL DIVER-SITY FOR TIME SERIES FOUNDATION MODELS

Hongyin Zhang<sup>1∗</sup>, Mingyi Xiang<sup>1∗</sup>, Rongzhao Chen<sup>2</sup>, Yankai Wang<sup>1</sup>, Jialin Qiao<sup>2</sup>, Xiangdong Huang<sup>1</sup>, Xiaoou Ding<sup>3</sup>, Chen Wang<sup>1</sup>, Zhongyi Pei<sup>1,⊠</sup>, Jianmin Wang<sup>1</sup>

<sup>1</sup>School of Software, BNRist, Tsinghua University

<sup>2</sup>Timecho Ltd, Beijing, China

<sup>3</sup>Harbin Institute of Technology

{hy-zhang25, xiangmy21}@mails.tsinghua.edu.cn

⊠ peizhyi@tsinghua.edu.cn

## ABSTRACT

Static public benchmarks for time series foundation models (TSFMs) typically have fixed and narrow test distributions. Their repeated reuse encourages benchmark-specific tuning, making reported performance reflect adaptation to the fixed test set rather than genuine forecasting generalization. In addition, a single aggregate score cannot fully characterize model capabilities across distinct temporal structures and input relationships. We propose CaFE (Capability-Focused Evaluation), a refreshable evaluation extension grounded in original benchmark data that enriches static benchmarks with structural diversity and enriches single-score black-box evaluation with multi-dimensional, capabilityoriented assessment. CaFE first profiles the original benchmark and defines eight Structural Features (SFs) that characterize its temporal structure. For each SF, it resamples the original data according to its structural support, removes edgeof-distribution samples, and augments the retained samples by varying the SF’s salience, producing samples with different structural profiles. After filtering out out-of-distribution samples while preserving diversity, CaFE yields a refreshed evaluation set. TSFMs are then evaluated on this set using both conventional forecast-error metrics and intervention-response metrics, giving each model a perfeature capability profile alongside endpoint accuracy. Experiments on GIFT-Eval and FEV-Mini20 show that CaFE yields stable evaluation results and reveals perfeature strengths and weaknesses that a single aggregate score hides. Moreover, on genuine data filtered by structural condition, models selected by CaFE outperform those chosen by the aggregate leaderboard. CaFE therefore makes model selection conditional on the structural response of interest.

## 1 INTRODUCTION

Time series foundation models (TSFMs), pretrained on large and diverse data corpora, are increasingly used for forecasting across domains such as energy, transportation, finance, and cloud operations (Ansari et al., 2024; Das et al., 2024; Woo et al., 2024). As more TSFMs become publicly available, selecting the right model for a specific forecasting task has become a critical practical challenge. Existing benchmarks evaluate TSFMs on widely reused datasets such as ETT, Electricity, and Traffic (Li et al., 2025; Aksu et al., 2024; Zhou et al., 2021; Trindade, 2015; Lai et al., 2018), typically summarizing performance with a single aggregate score. These leaderboards rank models by overall performance and have become the de facto standard for TSFM comparison.

Fixed public test sets create two intertwined problems. First, their repeated reuse conflates forecasting ability with familiarity with the specific test content, and their narrow structural coverage leaves structural diversity underrepresented. Second, a single aggregate score collapses many distinct observed cases into one ranking. As a result, the model that attains the best endpoint accuracy may differ from the model that responds faithfully to a controlled structural change, and existing benchmarks offer no direct way to inspect that difference (Figure 1).

To address these two problems, we present CAFE   
(Capability-Focused Evaluation), a refreshable   
evaluation extension grounded in original bench  
mark data. CaFE enriches static benchmarks with   
structural diversity and enriches single-score eval  
uation with multi-dimensional, capability-oriented   
assessment. Concretely, it first profiles the orig  
inal benchmark and defines eight Structural Fea  
tures (SFs) spanning intrinsic and relational struc  
ture. Table 2 summarizes the eight SFs and their   
controlled changes. Guided by these SFs, it re  
samples the original data along specific structural   
dimensions and removes edge-of-distribution sam  
ples to establish controlled structural conditions.   
It then augments the resampled data by varying   
the salience of each SF, producing samples with   
different structural profiles, and filters out out-of  
distribution augmented samples while preserving   
diversity, yielding a refreshed evaluation set. By   
spanning a wider range of structural conditions than   
the fixed test set, this refreshed set lowers the risk   
that reported performance reflects adaptation to a narrow distribution rather than genuine forecasting that reported performance reflects adaptation to a nar generalization. generalization.

![](images/a4f3a69df0c30bef9a828a6d01226c96f4fed0f066992d28180e1c69079be3a7.jpg)  
Figure 1: A static test set provides one finite snapshot of structural conditions, while a single aggregate score obscures how forecasts respond across those conditions.

On the evaluation side, CaFE measures each model along two axes. The first retains conventional forecast-error metrics, capturing endpoint accuracy on the refreshed set, as in existing benchmarks. The second introduces intervention-response metrics: for each Structural Feature, the evaluator varies the feature’s salience and measures how the model’s forecast changes in response, yielding a per-feature capability profile. The profile reveals structural sensitivity that endpoint accuracy alone misses: a model can forecast accurately while ignoring cross-series relationships or covariate conditions. CaFE therefore reports endpoint accuracy alongside a per-feature capability profile, exposing condition-specific strengths and weaknesses that a single aggregate score cannot express. In summary, we make the following main contributions:

• We propose CAFE, a refreshable evaluation extension grounded in original benchmark data that enriches static benchmarks with structural diversity and enriches single-score evaluation with multidimensional, capability-oriented assessment.

• We introduce Structural Features (SFs) and, guided by them, resample and augment the original data under controlled structural conditions, filtering out out-of-distribution samples to yield a refreshed evaluation set. We then expose per-feature response profiles through intervention-response metrics alongside conventional forecast-error metrics, adding a multi-dimensional, capabilityoriented axis to evaluation.

• Across GIFT-Eval and FEV-Mini20, we show that CaFE’s per-feature response profiles expose condition-specific strengths and weaknesses a single score hides, that repeated regenerations preserve the overall ranking while refreshing local winners, that training recipes can trade endpoint accuracy against response fidelity, and that models selected by CaFE outperform the aggregate leaderboard on genuine data filtered by structural condition.

## 2 RELATED WORK

Time series foundation model evaluation spans four broad lines. Fixed retrospective and TSFM suites, including M4/M5, TFB, BasicTS+, ProbTS, TSFM-Bench, GIFT-Eval, and fev-bench, provide data, pipelines, and standardized comparisons (Makridakis et al., 2020; 2022; Qiu et al., 2024; Shao et al., 2025; Zhang et al., 2024; Li et al., 2025; Aksu et al., 2024; Shchur et al., 2025). Structureaware suites such as It’s TIME and QuitoBench organize observed patterns or balanced regimes, and live platforms such as TS-Arena and Impermanent use preregistration or rolling streams (Qiao et al., 2026; Xue et al., 2026; Meyer et al., 2026; Garza et al., 2026). Synthetic-generation benchmarks such as GRATIS, TSGM, TimeSynth, and SynTSBench provide programmable mechanisms and behavior or fidelity diagnostics (Gupta, 2025; Garcia & Remy, 2026; Sun et al., 2026; Kang et al., 2020; Yoon et al., 2019; Desai et al., 2021; Yuan & Qiao, 2024; Ang et al., 2023; Nikitin et al., 2024; Tan et al., 2025; Haque et al., 2026). These lines vary fixed collections, observed regimes, incoming streams, or generators; the fixed suites emphasize endpoint comparisons, leaving source-task anchoring and controlled paired changes as separate design choices.

For CaFE, four properties distinguish an evaluation design: whether it anchors to original benchmark instances, makes structure explicit, applies a controlled paired change, and is refreshable. Retrospective, TSFM, and covariate-aware suites anchor to original instances but keep structure implicit and apply no paired changes; observed-structure suites make structure explicit but do not refresh; live suites refresh streams but apply no paired changes; synthetic suites expose mechanisms but abandon the source task context. CaFE combines all four, applying one Structural Feature change to an original instance under screening and then reporting endpoint error on the References and generated samples together with intervention-response fidelity. Table 1 summarizes this positioning.

Table 1: Benchmark families compared on four design properties: anchored to original instances, explicit structure, paired change, and refresh. CaFE is the only family that combines all four.
<table><tr><td>Family</td><td></td><td>Original Explicit Paired instances structure change Refresh</td><td></td><td></td></tr><tr><td>Retrospective suites</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>TSFM suites</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Covariate-aware suites</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Observed-structure suites</td><td>√</td><td>√</td><td>×</td><td>×</td></tr><tr><td>Live evaluation</td><td>√</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Synthetic mechanisms</td><td>×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>CaFE(Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 3 THE CAFE FRAMEWORK

CAFE addresses two failures of static benchmarks, a narrow structural distribution and a single aggregate score, by enriching a benchmark’s own data with structural diversity and its evaluation with capability-oriented assessment. Concretely, it proceeds in three generation stages followed by an evaluation stage (Figure 2). Profiling defines eight Structural Features (SFs) and maps each eligible source forecast window to an unchanged Reference. Augmentation resamples the References by structural support, removes edge-of-distribution windows, and, for each retained Reference, draws random salience coefficients to generate a family of samples spanning diverse structural conditions. Inspection and Sifting admits only complete families whose samples realize the intended change without drifting out of distribution. Finally, Evaluation scores each model along two axes.

The first axis retains forecast-error metrics, capturing endpoint accuracy on the refreshed set. The second adds intervention-response metrics that measure how faithfully a forecast tracks each induced change. Because every generated sample differs from its Reference only through a known change in one SF, the same benchmark yields both an endpoint-accuracy profile and a per-SF response profile, so model selection can follow capability for each specific SF rather than a single aggregate score.

## 3.1 PROFILING

Profiling anchors every intervention to the original forecasting task. It defines eight Structural Features (SFs) and maps each eligible source forecast window to an unchanged Reference R that pre serves the targets, forecast origin and horizon, input visibility, and scoreable positions. Because the Reference is a real, unmodified forecasting example, every generated sample stays grounded in real benchmark data rather than synthetic data. A Structural Feature is a temporal pattern or relationship whose salience can be varied in a controlled way to probe a specific forecasting capability. CaFE groups the eight features into two classes: intrinsic structure, which operates within a single target series, and relational structure, which operates across target series or between covariates and targets. Figure 3 summarizes the eight features, and Appendix A provides details.

Intrinsic Structure. Intrinsic structure comprises five features (SF1–SF5) spanning three forms of within-series variation. Trend and Seasonal Patterns (SF1–SF2) vary in persistent direction and concurrent periodicity, testing whether forecasts extrapolate long-term structure and combine multiple temporal scales. Nonstationary Dynamics (SF3–SF4) vary how seasonal behavior and temporal regimes evolve, testing whether forecasts track changes over time. Intermittent Event Dynamics (SF5) varies the regularity of sparse recurring events, testing whether forecasts recognize predictable structure when observations are mostly inactive. Moving from persistent patterns to evolving dynamics and sparse events, these interventions distinguish different forms of temporal generalization while retaining the same task anchor.

![](images/f7854b1b7b888d83e75560516d99ad2f97dba7eef31e00b2773efa66722fea29.jpg)  
Figure 2: The CAFE workflow. Profiling maps source forecast windows to unchanged References; Augmentation generates, for each Reference, a family of samples at varying salience; Inspection and Sifting admit only complete families that realize the intended change; Evaluation scores each model on forecast-error and intervention-response metrics.

Relational Structure. Relational structure comprises three features (SF6–SF8) spanning shared and directed dependencies. Shared Cross-Series Structure (SF6) varies a temporal component common to multiple target series, testing whether forecasts recognize movement shared across targets. Directed Predictive Relations (SF7–SF8) vary a lagged relation from a driver to a responder or a target’s response to a dynamic covariate, testing whether forecasts use signals from other series or covariates. Together, these features reveal whether a forecast uses information shared across targets or directed information from a specific series or covariate, while the series involved and the forecast task remain fixed. Because SF6–SF8 expose auxiliary inputs, CaFE additionally reports input utility by removing each input and measuring the resulting change in forecast error.

![](images/a170815d1fbdd4de43540fffe6b3366d7bfb57b6fee73f92658536b975568e10.jpg)  
Figure 3: The Structural Features of CAFE: intrinsic structure for within-series patterns and relational structure for cross-series and covariate dependencies.

## 3.2 AUGMENTATION

Augmentation turns each Reference into a

family of generated samples that differ from it only in a single Structural Feature. It first resamples the References according to their structural support for each SF, removing edge-of-distribution windows so that the refreshed set stays within the benchmark’s typical structural range. For each retained Reference and selected SF, it then draws random salience coefficients, each controlling how strongly the feature is expressed, and generates one sample per coefficient. Because the coefficients are drawn over declared ranges, the resulting families span a wider and more diverse set of structural conditions than the fixed test set.

For each Reference R and Structural Feature c, Augmentation draws a random salience coefficient λ and generates a sample $V = T _ { c } ( R , \lambda )$ . All samples generated from the same Reference and feature differ only in λ, so the family spans a range of salience levels while the forecast origin, horizon, target order, input visibility, and scoreable positions remain fixed. Each Structural Feature specifies its own transformation $T _ { c } ,$ which injects the designated pattern or relation into the Reference while leaving all other structure untouched. The concrete construction of each feature and the range from which λ is drawn are given in Appendix $\mathbf { A } ;$ for the present purpose, every generated sample is a controlled perturbation of its Reference with a known salience coefficient.

Together, the eight features and their randomly drawn salience coefficients give the refreshed set a structural coverage far wider than the fixed test set, so that reported accuracy reflects generalization across structure rather than familiarity with a narrow distribution. The generated families are then screened by Inspection and Sifting before they enter the evaluation set.

## 3.3 INSPECTION AND SIFTING

Augmentation produces candidate families, and Inspection and Sifting determine whether they can serve as evaluation data. Inspection works on each generated sample, verifying that the intended structural change is present, that non-target structure stays within tolerance, and that the sample remains valid and scoreable. Sifting works on the family linked to one Reference, requiring the family to be complete, checking that all members share the same Reference and configuration, and confirming that the family stays within the source distribution while preserving the structural diversity introduced by augmentation.

A family is admitted into the Refreshed Benchmark only when every member passes Inspection, and the family as a whole passes Sifting. Rejected or incomplete families receive an explicit reason and do not contribute to the evaluation, while their References remain available for the original task. The checks cover basic validity, realization of the intended structural change, non-target tolerance, source distance, scoreable support, distribution difference, and diversity. See Appendix B for details.

## 3.4 EVALUATION

CaFE reports two metric classes. Forecast-Error Metrics measure endpoint accuracy on the References and the generated samples, using Reference MASE and Probe MASE. Intervention-Response Metrics measure whether forecasts respond to the structural change introduced by augmentation along each SF, using Forecast-Change NRMSE, the normalized root-mean-square error between the induced target change and the model’s forecast change, with the Auxiliary-Input Gap (∆MASE) reported additionally for SF6–SF8. Together, these metrics characterize endpoint accuracy, structural response, and auxiliary-input utility.

For a given model and Structural Feature, Forecast-Change NRMSE measures how closely the model’s forecast change tracks the target change induced by varying the feature’s salience. Let $\Delta y$ and $\Delta \hat { y }$ denote the target change and the forecast change between a generated sample and its Reference on the affected positions. Forecast-Change NRMSE is

$$
\mathrm { N R M S E } = \sqrt { \frac { \sum _ { r = 1 } ^ { | \Delta y | } ( \Delta \hat { y } _ { r } - \Delta y _ { r } ) ^ { 2 } } { \sum _ { r = 1 } ^ { | \Delta y | } ( \Delta y _ { r } ) ^ { 2 } } } ,\tag{1}
$$

where r indexes the affected positions. A value of 0 indicates an exact response, 1 indicates no forecast response, and lower values mean closer agreement with the induced change. Eligibility and aggregation details are given in Appendix B.

The Auxiliary-Input Gap measures the change in MASE after removing the designated auxiliary input, with positive values indicating that the full input helps. The two metric classes keep their own rankings: Forecast-Error Metrics summarize accuracy over References and generated samples,

while Intervention-Response Metrics remain within each SF, whose controls, scales, and scoreable populations differ. Any cross-feature summary is a secondary equal-weight macro over directioncorrected feature-local ranks and does not pool raw values or define a cross-feature capability scale.

## 4 EXPERIMENTS

The experiments establish CaFE as an evaluation framework for time series foundation models. We first apply CaFE to GIFT-Eval (Short, Medium, and Long horizons) and fev-bench Mini-20, reporting endpoint accuracy and response fidelity for each model. On genuine data filtered by intrinsic structure, models selected by CaFE outperform those chosen by the aggregate leaderboard. We then validate the relational features SF6–SF8, verify that the augmentation is faithful, and check that the rankings are stable under regeneration. Finally, we show that the two metrics trade off under a changed training objective.

## 4.1 EXPERIMENTAL SETUP

We evaluate CaFE on three horizon subsets of GIFT-Eval (Aksu et al., 2024) (Short, Medium, and Long) and the Mini-20 subset of fev-bench (Shchur et al., 2025), using one fixed augmentation realization per suite. The comparison roster contains seven foundation models, Chronos-2 (Ansari et al., 2025), Timer-3.5 (Liu et al., 2026), TimesFM 2.5 (Google Research, 2025), TimesFM 3.0 (Das et al., 2024), TiREx-2 (Podest et al., 2026), Moirai 2 (Liu et al., 2025), and Toto 2.0 (Khwaja et al., 2026), plus four simple statistical baselines (Drift, Naive, Seasonal Naive, and Linear Trend). TiREx-2 is reported on GIFT-Eval Short and fev-bench Mini-20 because its output-length limit leaves others unsupported. Each Structural Feature is controlled by a salience coefficient λ, instantiated as five ordered settings L1–L5 (Appendix A); we report Reference MASE and Probe MASE for endpoint accuracy, Forecast-Change NRMSE for response fidelity, and Auxiliary-Input Gap for input utility, with model support and aggregation details in Appendices B and C.

## 4.2 ACCURACY AND RESPONSE FIDELITY

CaFE reports, for each model and Structural Feature, endpoint accuracy (Probe MASE on the Refreshed Benchmark) and response fidelity (Forecast-Change NRMSE); Figures 4 and 5 show the two views across the four suites, and Figure 6 condenses the fidelity view into a per-feature profile. Accuracy ranks models overall, while response fidelity reveals where each model’s response is strong or weak: a model can be accurate overall yet respond poorly to a structural change, such as a cross-series or covariate signal. The leaders differ by what is measured, TimesFM 3.0 winning 5/8, 7/8, 5/8, and 7/8 feature tracks by Probe MASE but only 4/8, 6/8, 6/8, and 3/8 by Forecast-Change NRMSE, while Linear Trend has the lowest SF1 NRMSE despite weak endpoint accuracy.

At the aggregate level the two rankings are largely consistent: the Spearman correlation between Probe-MASE rank and NRMSE rank is 0.029, 0.800, 0.900, and 0.771 on the four suites, and the refreshed set does not distort accuracy rankings (Reference MASE and Probe MASE are rankcorrelated at 0.943, 0.700, 0.900, and 0.657; Appendix C). The complementarity therefore lies in per-feature granularity rather than a reordering of the leaderboard: response fidelity flags the specific structural conditions under which a model’s endpoint accuracy does not transfer. The Linear Trend baseline exposes a further limit of the response metric: it attains the lowest SF1 NRMSE because a linear model trivially reproduces an injected linear trend, so the metric is most informative on the relational features (SF6–SF8), where matching the induced change requires modeling cross-series or covariate structure rather than reproducing a functional form.

## 4.3 SELECTING MODELS BY INTRINSIC STRUCTURE

An aggregate leaderboard selects a single model for every scenario, yet the best model can depend on the target series’ structural condition. For each intrinsic Structural Feature (SF1–SF5), we gather genuine tasks with strong corresponding structure, hold out one source family at a time, and compare the three models ranked highest by aggregate Reference MASE with those ranked highest by featurespecific Probe MASE. Figure 7 shows that CaFE raises Top-3 coverage of the Reference-MASE-best model from 41.4% (12/29) to 72.4% (21/29). The gain concentrates on SF1 (37.5% → 62.5%), SF3 (57.1% → 85.7%), and SF4 (42.9% → 85.7%), while SF2 (multi-seasonality) and SF5 (intermittency) stay flat, consistent with their few eligible families (4 and 3) and weaker feature-local signal.

![](images/42570950f0f1ce0fb904da60bd5313e38a125a6a6d3c5310189c52596f80ca03.jpg)

![](images/77c8bd78ac8a94b8e45e6d5d3bcfa2c852c526c522084d4b3ec349a7ec67e357.jpg)

![](images/87cab28cfb15f262957425269c5ccafb0e715ab07795f5e745d1deb0cdb7041d.jpg)

![](images/66ba2d3f3b2ab455e628bf82393dd223a153e25b62e504101aa9567271a05dfc.jpg)

Figure 4: Probe MASE rank by suite, model, and Structural Feature for the four suites in one horizontal strip. Within each feature, rank 1 is the most accurate model (lowest Probe MASE), and the underlying values are reported in Appendix C.  
![](images/18fa941b2726f5e8bf17ad1bca9891848f8dc1b9305fdf88f4bfec1cd4855799.jpg)

![](images/356c507f9bb2821be4eda8690787afeff50cfdcf89dd0f2349f8486680f9ba9f.jpg)

![](images/855e03f8d2ff576a77c3587f3b36866b5d4aa6ec17f1c28c4718d07d36ac9030.jpg)

![](images/93b6446c44e1eefe26a8273ada0306b29c3592b260980651f7e441c6d3517111.jpg)  
Figure 5: Forecast-Change NRMSE rank by suite, model, and Structural Feature for the four suites in one horizontal strip. Within each feature, rank 1 is the most faithful response to the induced change; NRMSE = 1 is the zero-response reference, and the underlying values are reported in Appendix C.

## 4.4 VALIDATION OF RELATIONAL FEATURES

The relational features SF6–SF8 test whether a model exploits auxiliary information: a shared latent factor across targets (SF6), a driver series (SF7), or a dynamic covariate (SF8). To validate these features, we remove the auxiliary input and measure how much the assessed target’s error changes, where a positive $\Delta \mathrm { M A S E } ~ = ~$ $\mathrm { M A S E _ { a b l a t e d } - M A S E _ { f u l l } }$ is the gain from the input. Figure 8 shows this gain against response fidelity: TimesFM 3.0 has the largest SF6 gap, Toto 2.0 the largest SF7 gap, and TimesFM 2.5 the largest SF8 gap, while several models lie near or below zero, marking models for which

![](images/9f567c86e5ad01319f8bce50cf0e0de8b99f6c6b56b9f94d18fd53d86eff94e2.jpg)  
Figure 7: Top-3 coverage of the Reference-MASE-best model on genuine strong-structure tasks from held-out source families (higher is better). SF1–SF5 use 8, 4, 7, 7, and 3 eligible families, respectively.

the auxiliary input acts as a distractor. The two axes are deliberately different: the Gap measures whether a model uses an input in the original task, whereas NRMSE measures whether a forecast responds to an injected change, so their lack of a clean monotone relationship is expected rather than a failure of either metric. The gains are small and model-specific, and the strongest input gain is not consistently held by the model with the lowest NRMSE, so the value of these inputs cannot be read from a single aggregate score. As a calibration, the simple baselines that cannot exploit cross-series or covariate structure sit near the zero-response reference on SF6–SF8, whereas the foundation models that use these inputs score lower; this confirms that the response metric separates models that track the induced change from those that ignore it.

![](images/c7d9701ce4fb3f14625f39ba37cb352371fff02f2b6b0ad399a7fc751220b3b4.jpg)  
Figure 6: Feature-level Forecast-Change NRMSE values (lower is better) across four suites. Each SF uses an independent range over the six shared models, with larger values toward the center and smaller values on extended outer spokes; endpoint labels show both bounds.

Figure 8 makes the two roles visible in one plot: the horizontal axis is response fidelity and the vertical axis is input utility. In the displayed means, TimesFM 3.0 has the largest SF6 gap, Toto 2.0 the largest SF7 gap, and TimesFM 2.5 the largest SF8 gap, while several other models lie near zero or below zero. The model with the strongest input gain is therefore not consistently the model with the lowest NRMSE, and the three SFs should be read on their own supports rather than as one difficulty scale. The task counts, descriptive intervals and details are reported in Appendix C.

![](images/4e433107381873ff35a9d7ef76706f0c0992f1804d1e1b4d2660a74cf1234ded.jpg)  
Figure 8: Model-level response fidelity and input utility for SF6–SF8. Each point is an equal-weight model mean across datasets; lower NRMSE and positive Gap are preferred.

## 4.5 FAITHFULNESS OF THE AUGMENTATION

A useful augmentation extends a benchmark without distorting model ordering; otherwise the structural signals CaFE adds would be noise rather than faithful evaluation. We verify this by ranking five small models on the References and the augmented Reference Families. Figure 9 shows the paired Reference and Augmented-Reference-Family MASE across 2,436 common source pairs and 21,793 valid augmented samples: agreement is Spearman $\rho ~ =$ 0.90 and Kendall $\tau _ { b } ~ = ~ 0 . 8 0$ with nine of ten pairwise orderings preserved, though heterogeneous across groups $( \rho = 1 . 0 0$ multivariate, 0.30 univariate, −0.10 Trend; Appendix C.7). We therefore treat this as conditional evidence for paired rank stability on this suite, not universal rank preservation. The single exception is Trend $( \rho ~ = ~ - 0 . 1 0 )$ , the same phenomenon seen in Section 4.2: a linear intervention is nearly trivial to reproduce, so the Trend feature is the least discriminative and the least orderpreserving; faithfulness is strongest where the paired change carries genuine relational information.

![](images/104ae430d400c3bac1c7503a758738b7b138bc5c58fb3c481045e373a8ae25d1.jpg)  
Figure 9: MASE on the References and on the Augmented Reference Families for five small models, ordered by Reference MASE. Points are three-seed means with ±1 sample SD; near-identical pairs within a model indicate that augmentation preserves that model’s accuracy ordering.

## 4.6 STABILITY UNDER BENCHMARK REFRESH

Because CaFE is refreshable, its conclusions should survive regeneration. Regenerating the GIFT Eval Short benchmark five times, changing only the augmentation seed, leaves the overall ranking unchanged, with TimesFM 3.0 first on both metrics (Table 10); the scores vary by 0.26%–0.51% (Probe MASE) and 0.59%–1.22% (Forecast-Change NRMSE). Feature-level winners are less stable, the lowest-NRMSE model changing across seeds on SF3, SF6, and SF8. Overall ordering is thus robust to regeneration, while feature-level winners reveal where a comparison depends on the generated conditions. This feature-level sensitivity means the per-feature winners reported in Section 4.2 are themselves seed-dependent; the robust finding is the stable overall ordering, while the specific per-feature leaders should be read as indicative rather than fixed.

## 4.7 TRADEOFF BETWEEN TWO METRICS

The two metrics also move in opposite directions when the training objective changes. Fine-tuning Chronos-2 on the same 11,269 CaFE-generated samples, official quantile-loss training lowers MASE by 13.46% but improves NRMSE by only 3.96%, whereas the Forecast-Change-NRMSE objective lowers NRMSE by 26.12% while raising MASE by 13.04% (Figure 11). The training objective therefore trades endpoint accuracy against response fidelity, so a practitioner can steer the model toward the capability their scenario needs. This is a single training realization. Optimizing Forecast-Change NRMSE teaches the model to track the injected changes, which trades away the endpoint fit on the refreshed distribution; the two objectives therefore move a single accuracy–fidelity operating point rather than demonstrating that one objective dominates the other.

Figure 10: Five-seed GIFT-Eval Short stability on the generated cohorts. Values are mean ± seed SD of Probe MASE and Forecast-Change NRMSE; ordering is unchanged across five seeds.
<table><tr><td>Model</td><td>Forecast-Change Probe MASE NRMSE</td></tr><tr><td>TimesFM 3.0  $1 . 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 5 8 0 \pm 0 . 0 0 5$ </td></tr><tr><td>TiREx-2  $1 . 2 0 \pm 0 . 0 0 3$ </td><td> $0 . 6 3 7 \pm 0 . 0 0 7$ </td></tr><tr><td>Chronos-2  $1 . 2 2 \pm 0 . 0 0 4$ </td><td> $0 . 6 4 6 \pm 0 . 0 0 7$ </td></tr><tr><td>Toto 2.0</td><td> $1 . 2 4 \pm 0 . 0 0 4$   $0 . 6 7 1 \pm 0 . 0 0 4$ </td></tr><tr><td>TimesFM 2.5  $1 . 2 5 \pm 0 . 0 0 4$ </td><td> $0 . 6 8 4 \pm 0 . 0 0 8$ </td></tr><tr><td>Timer-3.5  $1 . 2 8 \pm 0 . 0 0 5$ </td><td> $0 . 7 8 8 \pm 0 . 0 0 9$ </td></tr><tr><td>Moirai 2</td><td> $1 . 3 0 \pm 0 . 0 0 5$   $0 . 8 0 5 \pm 0 . 0 1 0$ </td></tr></table>

## 5 DISCUSSION

The novelty of CaFE is that a single, controlled, salience-varying perturbation of an unchanged Reference yields two measurements at once: endpoint accuracy and response fidelity. This paired Reference–sample design is what separates CaFE from fixed leaderboards, which report only endpoint accuracy, and from synthetic benchmarks, which replace the source task. Because every intervention stays anchored to a real instance, one refreshed benchmark produces a per-feature capability profile alongside the leaderboard, so the contribution is a capability-oriented axis that is grounded in, rather than divorced from, the original evaluation.

![](images/ca57885941f06c296a85ab13a136282269051688ddf221a3a52288e8c049c5d5.jpg)

![](images/9b8ff5ce9c49837381904deefaa2e10fd78dc8577ce41b41d541e7240ff04015.jpg)  
Figure 11: Chronos-2 fine-tuned on 11,269 CAFEgenerated samples under two objectives. The two panels track MASE and Forecast-Change NRMSE over 0–10k steps; official quantile-loss training improves MASE, while the Forecast-Change-NRMSE objective improves response fidelity at the cost of endpoint accuracy.

Across GIFT-Eval and fev-bench Mini-20, the two metrics reveal condition-specific strengths and weaknesses that a single aggregate score hides, the augmentation preserves model ordering, and the rankings are stable under regeneration. Most importantly, on genuine data filtered by intrinsic structure, models selected by CaFE’s per-feature profiles outperform the aggregate leaderboard pick, so structural response can guide model selection where a single score cannot. Several limits remain. Stability rests on five seeds on one suite, fine-tuning on a single realization, and the relational validation has few significant cells; selection is shown only for intrinsic structures. Future work include extending to relational and covariate conditions, more sources, and multi-seed recipes, along with controlled calibration of the response metric.

## 6 CONCLUSION

CaFE extends static time series foundation model benchmarks with refreshable structural diver sity and reports endpoint accuracy, response fidelity, and auxiliary-input utility as separate views. On genuine data filtered by intrinsic structure, models selected by CaFE outperform the aggregate leaderboard pick, so model selection can be made conditional on the structural response of inter est. The augmentation preserves model ordering, the rankings are stable under regeneration, and endpoint accuracy and response fidelity trade off under a changed training objective.

## AI USE DISCLOSURE

Generative AI tools were used to assist with limited aspects of this work, including code implementation support, experiment analysis, and manuscript refinement. These tools were used for tasks such as language editing, text organization, formatting conversion, compilation and layout checks, and consistency verification of terminology and figures. We did not use generative AI tools to generate datasets, experimental results, or mathematical claims. All AI-assisted outputs were reviewed and validated by the authors, who take full responsibility for the final manuscript, including all text, claims, analyses, figures, code, and research artifacts.

## REPRODUCIBILITY STATEMENT

The anonymous artifact, including code and compact numerical inputs for the reported analyses, is available at https://anonymous.4open.science/r/CaFE-ICLR2027-5210. Appendix B specifies data construction, scoring, and aggregation; Appendix C.12 records the evaluation panels, model support, and evidence scope; and Appendix C.13 documents the computing environments and procedures for reconstructing the tables and figures and rerunning the evaluation. The reported results use the declared panels and one training realization.

## REFERENCES

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval: A benchmark for general time series forecasting model evaluation. In NeurIPS Workshop on Time Series in the Age of Large Models, 2024. URL https://open review.net/forum?id=Z2cMOOANFX.

Yihao Ang, Qiang Huang, Yifan Bao, Anthony K. H. Tung, and Zhiyong Huang. TSGBench: Time series generation benchmark. Proceedings of the VLDB Endowment, 17(3):305–318, 2023. doi: 10.14778/3632093.3632097. URL https://www.vldb.org/pvldb/vol17/p305-hua ng.pdf.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. URL https: //openreview.net/forum?id=gerNCVqqtR.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Kuken, Andreas Auer, Boran Han, Pedro Mercado,¨ Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting, 2025. URL https://arxiv.org/abs/2510.15821.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 10148–10167, Vienna, Austria, July 2024. PMLR. URL https://proceedings.mlr.press/v235/das24c.html.

Abhyuday Desai, Cynthia Freeman, Zuhui Wang, and Ian Beaver. TimeVAE: A variational autoencoder for multivariate time series generation, 2021. URL https://arxiv.org/abs/21 11.08095.

Cyril Garcia and Guillaume Remy. Statistical benchmarking of transformer models in low signalto-noise time-series forecasting, 2026. URL https://arxiv.org/abs/2602.09869.

Azul Garza, Renee Rosillo, Rodrigo Mendoza-Smith, David Salinas, Andrew Robert Williams, Ar-´ jun Ashok, Mononito Goswami, and Jose Mart ´ ´ın Juarez. Impermanent: A live benchmark for tem-´ poral generalization in time series forecasting. In ICLR 2026 Workshop on Time Series in the Age ofLarge Models, 2026. URL https://openreview.net/forum?id=EOoinHRJ0P.

Google Research. TimesFM model card, 2025. URL https://huggingface.co/google/ timesfm-2.5-200m-pytorch. Checkpoint: timesfm-2.5-200m-pytorch.

Syamantak Datta Gupta. Evaluating time series foundation models on noisy periodic time series, 2025. URL https://arxiv.org/abs/2501.00889.

Md Rakibul Haque, Vishwa Goudar, Shireen Elhabian, and Warren Woodrich Pettine. TimeSynth: A framework for uncovering systematic biases in time series forecasting, 2026. URL https: //arxiv.org/abs/2602.11413.

Rob J. Hyndman and Anne B. Koehler. Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4):679–688, 2006. doi: 10.1016/j.ijforecast.2006.03.001. URL https://doi.org/10.1016/j.ijforecast.2006.03.001.

Yanfei Kang, Rob J. Hyndman, and Feng Li. GRATIS: GeneRAting TIme Series with diverse and controllable characteristics. Statistical Analysis and Data Mining: The ASA Data Science Journal, 13(4):354–376, 2020. doi: 10.1002/sam.11461. URL https://doi.org/10.1002/sam. 11461.

Emaad Khwaja, Chris Lettieri, Gerald Woo, Eden Belouadah, Marc Cenac, Guillaume Jarry, Enguerrand Paquin, Xunyi Zhao, Viktoriya Zhukov, Othmane Abou-Amal, Chenghao Liu, Ameet Talwalkar, and David Asker. Toto 2.0: Time series forecasting enters the scaling era, 2026. URL https://arxiv.org/abs/2605.20119.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In Proceedings ofthe 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 95–104, Ann Arbor, MI, USA, 2018. Association for Computing Machinery. doi: 10.1145/3209978.3210006. URL https://doi.org/10.1145/3209978.3210006.

Zhe Li, Xiangfei Qiu, Peng Chen, Yihang Wang, Hanyin Cheng, Yang Shu, Jilin Hu, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, and Bin Yang. TSFM-Bench: A comprehensive and unified benchmark of foundation models for time series forecasting. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 5595–5606, New York, NY, USA, 2025. Association for Computing Machinery. doi: 10.1145/3711896.3737442. URL https://doi.org/10.1145/3711896.3737442.

Chenghao Liu, Taha Aksu, Juncheng Liu, Xu Liu, Hanshu Yan, Quang Pham, Silvio Savarese, Doyen Sahoo, Caiming Xiong, and Junnan Li. Moirai 2.0: When less is more for time series forecasting, 2025. URL https://arxiv.org/abs/2511.11698. Evaluated checkpoint: Moirai-2.0-R-Small; model card: https://huggingface.co/Salesforce/moirai-2.0-R-small.

Yong Liu, Xingjian Su, Shiyu Wang, Haoran Zhang, Haixuan Liu, Yuxuan Wang, Zhou Ye, Yang Xiang, Jianmin Wang, and Mingsheng Long. Timer-s1: A billion-scale time series foundation model with serial scaling, 2026. URL https://arxiv.org/abs/2603.04791. Evaluated checkpoint: Timer-S1; model card: https://huggingface.co/thuml/Timer-S1.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The M4 competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1): 54–74, 2020. doi: 10.1016/j.ijforecast.2019.04.014. URL https://doi.org/10.1016/ j.ijforecast.2019.04.014.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. M5 accuracy competition: Results, findings, and conclusions. International Journal ofForecasting, 38(4):1346–1364, 2022. doi: 10.1016/j.ijforecast.2021.11.013. URL https://doi.org/10.1016/j.ijforeca st.2021.11.013.

Marcel Meyer, Sascha Benjamin Kaltenpoth, Henrik Albers, Kevin Zalipski, and Oliver Muller.¨ TS-Arena – a live forecast pre-registration platform. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 9558–9568. Association for Computing Machinery, 2026. doi: 10.1145/3770855.3817515. URL https://doi.org/10.1 145/3770855.3817515.

Alexander Nikitin, Letizia Iannucci, and Samuel Kaski. TSGM: A flexible framework for generative modeling of synthetic time series. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 37, pp. 129042–129061, Red Hook, NY, USA, 2024. Curran Associates, Inc. doi: 10.52202/079017-4098. URL https://proceedi ngs.neurips.cc/paper\_files/paper/2024/hash/e92bb8c76d1aa420171f5 1495ce56eaf-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Patrick Podest, Marco Pichler, Elias B”urger, Levente Z’olyomi, Bernhard Voggenberger, Wilhelm Berghammer, Daniel Klotz, Sebastian B”ock, G”unter Klambauer, and Sepp Hochreiter. Tirex-2: Generalizing tirex to multivariate data and streaming, 2026. URL https://arxiv.org/ abs/2607.01204. Evaluated checkpoint: TiRex-2; model card: https://huggingface.co/NX-AI/TiRex-2.

Zhongzheng Qiao, Sheng Pan, Anni Wang, Viktoriya Zhukova, Yong Liu, Xudong Jiang, Qingsong Wen, Mingsheng Long, Ming Jin, and Chenghao Liu. It’s TIME: Towards the next generation of time series forecasting benchmarks. In International Conference on Machine Learning (ICML), 2026. URL https://icml.cc/virtual/2026/poster/66086.

Xiangfei Qiu, Jilin Hu, Lekui Zhou, Xingjian Wu, Junyang Du, Buang Zhang, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, Zhenli Sheng, and Bin Yang. TFB: Towards comprehensive and fair benchmarking of time series forecasting methods. Proceedings of the VLDB Endowment (PVLDB), 17(9):2363–2377, 2024. doi: 10.14778/3665844.3665863. URL https://doi.org/10.14778/3665844.3665863.

Zezhi Shao, Fei Wang, Yongjun Xu, Wei Wei, Chengqing Yu, Zhao Zhang, Di Yao, Tao Sun, Guangyin Jin, Xin Cao, Gao Cong, Christian S. Jensen, and Xueqi Cheng. Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis. IEEE Transactions on Knowledge and Data Engineering (TKDE), 37(1):291–305, 2025. doi: 10.1109/TKDE.2024.3484454. URL https://doi.org/10.1109/TKDE.2024.3484 454.

Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, and Yuyang Wang. fev-bench: A realistic benchmark for time series forecasting, 2025. URL https://arxiv.org/abs/2509.26468.

Jiaze Sun, Kelvin J. L. Koa, Ruiyang Ni, Yize Liu, Haonan Chen, and Ke-Wei Huang. FinStressTS: A parametric synthetic benchmark for time-series forecasting in finance. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, 2026. doi: 10.1145/3770855.3817578. URL https://doi.org/10 .1145/3770855.3817578.

Qitai Tan, Yiyun Chen, Mo Li, Ruiwen Gu, Yilin Su, and Xiao-Ping (Steven) Zhang. SynTSBench: Rethinking temporal pattern learning in deep learning models for time series. In Danielle Belgrave, Cheng Zhang, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, and Nan Chen (eds.), Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 38, pp. 161198–161242, San Diego, California, USA, 2025. Curran Associates, Inc. doi: 10.52202/085713-4860. URL https://proceedings.neurips.cc /paper\_files/paper/2025/hash/d60361f3d0021c76626b2597957261d6-A bstract-Datasets\_and\_Benchmarks\_Track.html.

Artur Trindade. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. URL https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams2 0112014.

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 53140–53164, Vienna, Austria, July 2024. PMLR. URL https://proceedings.mlr.pr ess/v235/woo24a.html.

Siqiao Xue, Zhaoyang Zhu, Wei Zhang, Rongyao Cai, Rui Wang, Yixiang Mu, Fan Zhou, Jianguo Li, Peng Di, and Hang Yu. QuitoBench: A high-quality open time series forecasting benchmark, 2026. URL https://arxiv.org/abs/2603.26017.

Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar. Time-series generative adversarial networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, pp. 5508– 5518, Red Hook, NY, USA, 2019. Curran Associates, Inc. URL https://proceedings. neurips.cc/paper\_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7 d26d490-Abstract.html.

Xinyu Yuan and Yan Qiao. Diffusion-TS: Interpretable diffusion for general time series generation. In The Twelfth International Conference on Learning Representations, Vienna, Austria, 2024. OpenReview.net. URL https://proceedings.iclr.cc/paper\_files/paper/20 24/hash/b5b66077d016c037576cc56a82f97f66-Abstract-Conference.ht ml.

Jiawen Zhang, Xumeng Wen, Zhenwei Zhang, Shun Zheng, Jia Li, and Jiang Bian. ProbTS: Benchmarking point and distributional forecasting across diverse prediction horizons. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 37, pp. 48045–48082, Vancouver, BC, Canada, 2024. Neural Information Processing Systems Foundation, Inc. doi: 10.52202/079017-1523. URL https://doi.org/10.52202/079017-1 523.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 35(12):11106–11115, 2021. doi: 10.1609/aaai.v 35i12.17325. URL https://doi.org/10.1609/aaai.v35i12.17325.

## A PROFILING, AUGMENTATION, AND STRUCTURAL FEATURE CONSTRUCTIONS

The contracts below define task-qualified References and the eight Structural Feature constructions.   
The admitted GIFT-Eval tasks form a narrower population than the complete official benchmark;   
Appendix C reports the support of each analysis.

## A.1 STAGE 1: PROFILING

## A.1.1 TASK CONTRACT FOR EACH BENCHMARK

Let t denote one benchmark task. Its adapter contract identifies the source data and fixes the forecast targets and their order, forecast-origin enumeration and horizon, the input history in the benchmark’s native representation, auxiliary inputs allowed by the task and their visibility roles, observation policy, and validity conditions. A univariate setting has one prediction target, whereas a multivariate setting preserves multiple ordered targets as one forecasting object. Either target setting is covariateinformed when the task additionally provides non-target covariates allowed by the task. The adapter maps the native representation to CaFE’s common interface while preserving these semantics; any model-required channel decomposition is recorded as an inference adaptation rather than a change to the task’s target setting.

For the ith valid forecast origin of task $t ,$ the adapter returns a Reference, an adapter-qualified forecast instance from the original benchmark task in CAFE’s common representation, written as

$$
R _ { i } = \left( H _ { i } , Y _ { i } ^ { + } , Z _ { i } , M _ { i } , o _ { i } , \iota _ { i } \right) .\tag{2}
$$

Here, $H _ { i }$ is the target history, $Y _ { i } ^ { + }$ is the forecast truth, $Z _ { i }$ contains the auxiliary inputs permitted by $t , M _ { i }$ records the applicable observation masks, including the future-target observation mask that identifies scoreable cells, $o _ { i }$ is the forecast origin, and $\iota _ { i }$ identifies the benchmark task and native window used for the forecast. The task definition provides the information needed to reconstruct the corresponding Reference. The References of task t form

$$
{ \mathcal { A } } _ { t } = \{ R _ { i } : R _ { i } { \mathrm { ~ s a t i s f i e s ~ t h e ~ a d a p t e r ~ c o n t r a c t ~ f o r ~ } } t \} .\tag{3}
$$

Reference materialization and information boundary. The task adapter owns forecast-origin enumeration and future-label eligibility; it may require a fully observed horizon or retain a futuretarget observation mask, with the resulting scoreability state carried by the Reference. History repair, when required, follows a deterministic adapter-resolved rule derived from history, and the corre sponding scale policy declares whether repaired values or originally observed pairs enter historybased statistics. The source history, task contract, and recorded augmentation seed determine Structural Feature selection and every structural, control-setting, and random choice; future target values enter the workflow as scoreable truth after these choices. The future-target observation mask may constrain where a constructed effect must remain visible and determines which future cells are scoreable. Horizon visibility follows the declaration of task t: known-future dynamic covariates extend across the horizon, and past-only fields end at the forecast origin. Profiling records the adapter, task, source, and native-window identities together with the bound task contract for each $R _ { i } ,$ , supporting later contract checks and source reconstruction under the verification policy selected for the evaluation round.

## A.2 STAGE 2: AUGMENTATION

## A.2.1 CONTRACT FOR AN ORDERED SET OF SAMPLES

For Structural Feature $s \in \{ 1 , \ldots , 8 \}$ , Augmentation first applies an SF-specific source filter $F _ { s } ^ { \mathrm { p r e } }$ to References from the original benchmark. Its inputs are source history, Structural Feature evidence, and the task contract. It retains $R _ { i }$ only when the selected SF has sufficient source support, every declared setting admits a feasible construction that can satisfy the Structural Feature Expectation, and the inputs and construction domain are physically valid. The retained source Reference is written $R _ { i } ^ { \prime }$ to mark the selection step; the prime changes no task field or provenance. For each selected Reference $R _ { i } ^ { \prime }$ and Structural Feature s, Augmentation selects a shared structural configuration $\xi _ { i s }$ . It fixes the affected targets and feature-specific temporal form or predictive relation across the ordered index set $\mathcal { L } _ { s }$ . The evaluation round declares this set; the experimental setting counts and parameter ranges appear in Appendix C and Appendix A. The index $\ell \in \mathcal { L } _ { \varepsilon }$ marks a position within s, and $\lambda _ { i s \ell }$ is the setting of the feature-specific Control Parameter. The corresponding generated sample and sample set are

$$
\begin{array} { r l r } & { } & { V _ { i s \ell } = T _ { s } ( R _ { i } ^ { \prime } ; \xi _ { i s } , \lambda _ { i s \ell } ) , } \\ & { } & { \widetilde { \gamma } _ { i s } = \{ V _ { i s \ell } : \ell \in \mathcal { L } _ { s } \} , } \end{array}\tag{4}
$$

where $T _ { s }$ changes the feature-declared fields while preserving the task, source identifier, origin, horizon, target order, visibility roles, and shared configuration $\xi _ { i s } .$ The Control Parameter may represent amplitude, change location, period count, or event spacing, so its setting order and scale are local to each Structural Feature. Applying a setting may determine compatible details such as the phase used by SF5; all settings retain $R _ { i } ^ { \prime }$ as their common baseline.

The protocol permits source selection to resample existing References or structural regions accord ing to a declared Structural Feature Distribution. Resampling changes the structural conditions supplied to Augmentation and records the selected view $\bar { \boldsymbol { R } } _ { i } ^ { \prime } .$ The unchanged Reference population remains the estimand for original-task results, and each run records how its selection policy is instantiated.

Field-level transformation contract. The feature-specific transformation can be written componentwise as

$$
\begin{array} { c } { { V _ { i s \ell } = \left( H _ { i } + \Delta H _ { i s \ell } , \mathrm { ~ } Y _ { i } ^ { + } + \Delta Y _ { i s \ell } , \right. } } \\ { { \left. Z _ { i } + \Delta Z _ { i s \ell } , \mathrm { ~ } M _ { i } , \mathrm { ~ } o _ { i } , \mathrm { ~ } \iota _ { i } \right) . } } \end{array}\tag{5}
$$

The deltas identify the fields changed by Structural Feature s; zero deltas retain the corresponding selected-Reference fields. SF1–SF7 act on target histories and their corresponding generated future targets, so $\Delta Z _ { i s \ell } = 0$ . SF8 adds a dynamic-covariate impulse allowed by the task and its corresponding target response, with its future covariate component restricted by the task’s visibility declaration. Thus, every transformation changes only the feature-declared fields of $R _ { i } ^ { \prime } ;$ Evaluation measures the resulting forecast changes.

Structure selection and setting feasibility. Each mechanism determines applicability and constructs a feasible shared structure. For amplitude-controlled features, the generator normalizes the selected component by its source distance and selects settings only from the feasible interval; SF2, SF4, and SF5 vary period count, transition location, and event spacing. The selected structure is fixed across the ordered settings.

Source-distance evidence. Qualification uses a common distance to keep each generated change visible and bounded relative to its source. Let $\mathcal { I } _ { i s }$ be the affected target set, C a protocol-declared history view, and $\sigma _ { i d } > 0$ the source-history standard deviation for target d. A nonfinite or degenerate standard deviation falls back first to the standard deviation of first differences and then to the declared positive value 1. This quantity normalizes construction distance, while the MASE scale $\sigma _ { i d } ^ { \mathrm { M A S E } }$ normalizes forecast error. Stage 2 records

$$
d _ { i s \ell C } = \frac { 1 } { | \mathcal { T } _ { i s } | } \sum _ { d \in \mathcal { I } _ { i s } } \sqrt { \frac { 1 } { | C | } \sum _ { u \in C } \left( \frac { H _ { i s \ell , u , d } ^ { V } - H _ { i , u , d } ^ { R } } { \sigma _ { i d } } \right) ^ { 2 } } .\tag{6}
$$

This distance uses the selected Reference $R _ { i } ^ { \prime }$ as the comparator for every declared setting. For every required model-visible context, the macro value must lie in [0.10, 2.0]; each affected channel has an upper bound of 3.0. Structural variables such as period count or event spacing serve as featurespecific Control Parameters and express structural progression.

## A.3 STRUCTURAL FEATURE CONSTRUCTIONS

Table 2 summarizes the eight constructions. The reported GIFT-Eval and fev-bench Mini-20 experiments use $\mathcal { L } _ { s } = \{ 1 , \ldots , \bar { 5 } \}$ , displayed as L1–L5. The formulas specify the changed fields, shared structural configuration, and feature-specific control settings; the numerical ranges below belong to this experimental configuration.

Table 2: Structural features, controlled changes, control parameters, and input requirements. Full constructions appear in this appendix.
<table><tr><td>Structural feature</td><td>Controlled change</td><td>Control parameter</td><td>Applicability</td></tr><tr><td>SF1 Trend</td><td>Persistent directional continuation</td><td>Normalized trend magnitude</td><td>Uni./multi.</td></tr><tr><td>SF2 Multi-Seasonality</td><td>Concurrent stable periods</td><td>Number of active periods</td><td>Uni./multi.</td></tr><tr><td>SF3 Time-Varying Seasonality</td><td>Evolving amplitude or phase</td><td>Normalized evolution magnitude</td><td>Uni./multi.</td></tr><tr><td>SF4 Regime Switching</td><td>Bounded transition to a persistent state</td><td>Transition location</td><td>Uni./multi.</td></tr><tr><td>SF5 Intermittency</td><td>Sparse recurring events</td><td>Mean event gap</td><td>Uni./multi.</td></tr><tr><td>SF6 Common Factor</td><td>Shared trajectory across targets</td><td>Normalized shared-profile mag- nitude</td><td>Multi., ≥ 3 targets</td></tr><tr><td>SF7 Cross-Series Dependence</td><td>Directed driver-responder relation</td><td>Normalized transfer magnitude Joint normalized response mag-</td><td>Multi.,  $\geq 2$  targets</td></tr><tr><td>SF8 Covariate Response</td><td>Covariate impulse with matched target response</td><td>nitude</td><td>Native dynamic co- variate</td></tr></table>

## A.4 SHARED NOTATION AND CONTROL CONTRACT

For Reference $R _ { i } .$ , let $L _ { i }$ and $H _ { i } ^ { \mathrm { { h o r } } }$ denote its history length and forecast horizon. We use $u \in$ $\{ 0 , \ldots , L _ { i } + H _ { i } ^ { \mathrm { h o r } } - 1 \}$ on the concatenated target path

$$
X _ { i , d } ( u ) = \left\{ \begin{array} { l l } { H _ { i } [ u , d ] , } & { 0 \leq u < L _ { i } , } \\ { Y _ { i } ^ { + } [ u - L _ { i } , d ] , } & { L _ { i } \leq u < L _ { i } + H _ { i } ^ { \mathrm { h o r } } , } \end{array} \right.\tag{7}
$$

where $d$ indexes prediction targets. Let $U _ { i , q } ( u )$ denote task-permitted dynamic covariate $q$ on its declared visible range. A generated sample changes these fields by

$$
X _ { i s \ell } ^ { V } = X _ { i } + \Delta X _ { i s \ell } , \qquad U _ { i s \ell } ^ { V } = U _ { i } + \Delta U _ { i s \ell } .\tag{8}
$$

The future values in Equation (7) provide scoreable truth after construction. Structural configuration $\xi _ { i s }$ is selected from source history and task information before those values enter the workflow.

For the amplitude-controlled Structural Features SF1, SF3, SF6, SF7, and SF8, let $G _ { i s } ^ { X } ( u , d )$ and $G _ { i s } ^ { U } ( u , q )$ be the raw target and covariate components fixed by $\xi _ { i s }$ . With affected-target set $\mathcal { I } _ { i s }$ and source-history standard-deviation scale $\sigma _ { i d }$ , define the full-history macro normalized root-meansquare

$$
\mathcal { D } _ { i } ( G _ { i s } ^ { X } ; \mathcal { T } _ { i s } ) = \frac { 1 } { | \mathcal { T } _ { i s } | } \sum _ { d \in \mathcal { I } _ { i s } } \sqrt { \frac { 1 } { L _ { i } } \sum _ { u = 0 } ^ { L _ { i } - 1 } \left( \frac { G _ { i s } ^ { X } ( u , d ) } { \sigma _ { i d } } \right) ^ { 2 } } .\tag{9}
$$

Writing $q _ { i s } = \mathcal { D } _ { i } ( G _ { i s } ^ { X } ; \mathcal { T } _ { i s } )$ , Salience Tuning at control setting $\ell \in \mathcal { L } _ { \varepsilon }$ <sub>s</sub> gives

$$
\left( \Delta X _ { i s \ell } , \Delta U _ { i s \ell } \right) = \frac { \lambda _ { i s \ell } } { q _ { i s } } \left( G _ { i s } ^ { X } , G _ { i s } ^ { U } \right) .\tag{10}
$$

In this experimental instantiation, the five nominal intervals for $\lambda _ { i s \ell }$ are [0.10, 0.15], [0.17, 0.22], [0.25, 0.32], [0.36, 0.46], and [0.50, 0.65], intersected with the feasible interval of the selected structure. SF2, SF4, and SF5 instead use period count, transition location, and event gap as their featurespecific Control Parameters. Each SF therefore interprets L1–L5 through its own Control Parameter and Control Settings, and Evaluation measures the forecast changes under those settings.

Figure 12 illustrates the constructed components defined below.

## A.5 SF1 TREND

SF1 generates a trend by selecting a nonempty target subset $\mathcal { I } _ { i 1 }$ with history-stable trend signs $s _ { i d } ^ { \mathrm { t r } } \in \{ - 1 , 1 \}$ . For each selected target, the raw component is

$$
G _ { i 1 } ^ { X } ( u , d ) = \mathbf { 1 } \{ d \in \mathcal { I } _ { i 1 } \} \sigma _ { i d } s _ { i d } ^ { \mathrm { t r } } p _ { i 1 } ( u ) , \qquad \Delta X _ { i 1 \ell } = \frac { \lambda _ { i 1 \ell } } { q _ { i 1 } } G _ { i 1 } ^ { X } ,\tag{11}
$$

SF7 Cross-Series Dependence  
![](images/4e80c7e34ec65e2c752b7ed530b8f518e21bb9652dd539589317fd3cd00cf2bb.jpg)

![](images/80cf048ecadf3762ad46cb8852c4decce86c56fc0cba52d5f738f012d371cede.jpg)

![](images/42e47659064442f2c8684f3d01ec87c2ac4e330b5cd5a6f2c892dc55fcf810f4.jpg)

![](images/da5b1cf464037eab99ed20cd9899089ebcbe06512696def95ff0d7b07e7ab6a9.jpg)

![](images/f0ffe1e26d7966a6e35bcb8f0138c9518353f4e8314f492b9771746ea146a3c7.jpg)

![](images/0087de3beabf8c98434162c7011dbd7f25576cc0e4a840207254c9239f84bc97.jpg)

![](images/f05f3b486b37c01e79bda60ac802000af5f3cd31e2538066f88fe12fc3e15541.jpg)

![](images/7147d5c791de8acd3e51519cea68a6cbbeb1e82e2473d3dc7c15895f7d7afe9e.jpg)  
Figure 12: Illustrative interventions for SF1–SF8. Each panel joins the modified history to its analytic continuation over the forecast horizon and preserves the task's target and auxiliary-input roles.

where $p _ { i 1 }$ is one structure shared across all settings:

$$
\begin{array} { r l r } {  { p _ { \mathrm { l i n } } ( u ) = \frac { u } { L _ { i } - 1 } , } } \\ & { } & \\ & { p _ { \mathrm { d e l } } ( u ) = \frac { ( u - u _ { 0 } ) _ { + } } { L _ { i } - 1 } , } \\ & { } & \\ & { p _ { \mathrm { c u r v } } ( u ) = ( \frac { u } { L _ { i } - 1 } ) ^ { r } , } \\ & { } & \\ & { p _ { \mathrm { p w } } ( u ) = \displaystyle \frac { 1 } { L _ { i } - 1 } \sum _ { v = 0 } ^ { u - 1 } b _ { [ ( v + \phi ) / K ] \mathrm { ~ m o d ~ } B } , } \\ & { } & \\ & { p _ { i 1 } \in \{ p _ { \mathrm { l i n } } , p _ { \mathrm { d e l } } , p _ { \mathrm { c u r v } } , p _ { \mathrm { p w } } \} . } \end{array}\tag{12}
$$

The four alternatives are whole-history linear, delayed linear, slow-curvature, and recurring piecewise-linear continuation. The onset $u _ { 0 } .$ , curvature $r ,$ knot interval $K ,$ , phase $\phi ,$ slope cycle $\left( b _ { 0 } , \ldots , b _ { B - 1 } \right)$ , affected targets, and signs belong to $\xi _ { i 1 } ;$ only normalized magnitude $\lambda _ { i 1 \ell }$ changes across settings. Construction requires a stable history-derived direction and continues the same profile law through the horizon.

## A.6 SF2 MULTI-SEASONALITY

For each target, SF2 constructs multi-seasonality by selecting a harmonic anchor from up to three qualified history-spectrum candidates, with a protocol-generated anchor when none qualifies. For target $d ,$ let the selected anchor and five additional independent periods be $P _ { i d 0 } , \ldots , \bar { P } _ { i d 5 }$ . Component r is either a sinusoid or a two-harmonic waveform,

$$
\begin{array} { l l r } { \displaystyle { h _ { i d r } ( u ) = \sum _ { m = 1 } ^ { M _ { i d r } } a _ { i d r m } \sin \left( \frac { 2 \pi m u } { P _ { i d r } } \right) } } \\ { \displaystyle { + \sum _ { m = 1 } ^ { M _ { i d r } } b _ { i d r m } \cos \left( \frac { 2 \pi m u } { P _ { i d r } } \right) , } } \\ { \displaystyle { M _ { i d r } \in \{ 1 , 2 \} . } } \end{array}\tag{13}
$$

Each component is history-normalized to the target scale before aggregation. At Control Setting $\ell \in \{ 1 , \ldots , 5 \}$ , the generated sample contains the anchor and the first ℓ additional periods:

$$
S _ { i 2 \ell } ( u , d ) = \sum _ { r = 0 } ^ { \ell } h _ { i d r } ( u ) , \qquad \Delta X _ { i 2 \ell } ( u , d ) = \frac { \rho _ { i 2 } } { { \cal D } _ { i } ( S _ { i 2 \ell } ; { \mathcal T } _ { i 2 } ) } S _ { i 2 \ell } ( u , d ) ,\tag{14}
$$

where $\mathcal { I } _ { i 2 }$ contains all targets and the shared full-history distance $\rho _ { i 2 } \in [ 0 . 2 2 , 0 . 2 8 ]$ is fixed across the five settings. Hence the Control Parameter is the number of additional independent periods, producing two through six periods while holding aggregate history amplitude fixed. Periods, phases, waveform families, and their order are selected from history-qualified or protocol-generated candidates without future target values.

## A.7 SF3 TIME-VARYING SEASONALITY

SF3 constructs time-varying seasonality. For an eligible target $d ,$ the selected carrier, its quadrature component, and a slower envelope are

$$
\begin{array} { r l r l } & { c _ { i d } ( u ) = a _ { i d } ^ { c } \sin ( \omega _ { i d } ^ { c } u ) + b _ { i d } ^ { c } \cos ( \omega _ { i d } ^ { c } u ) , } & & { } \\ & { c _ { i d } ^ { \perp } ( u ) = b _ { i d } ^ { c } \sin ( \omega _ { i d } ^ { c } u ) - a _ { i d } ^ { c } \cos ( \omega _ { i d } ^ { c } u ) , } & & { } \\ & { e _ { i d } ( u ) = a _ { i d } ^ { e } \sin ( \omega _ { i d } ^ { e } u ) + b _ { i d } ^ { e } \cos ( \omega _ { i d } ^ { e } u ) , } & & { \omega _ { i d } ^ { e } < \omega _ { i d } ^ { c } . } \end{array}\tag{15}
$$

The raw component uses the carrier for amplitude modulation and the quadrature carrier for the periodic phase-drift component:

$$
\begin{array} { r l } & { b _ { i d } ^ { \mathrm { m o d } } ( u ) = \left\{ \begin{array} { l l } { c _ { i d } ( u ) , } & { \mathrm { a m p l i t u d e ~ m o d u l a t i o n , } } \\ { c _ { i d } ^ { \perp } ( u ) , } & { \mathrm { p e r i o d i c ~ p h a s e ~ d r i f t , } } \end{array} \right. } \\ & { G _ { i 3 } ^ { X } ( u , d ) = { \bf 1 } \{ d \in \mathcal { I } _ { i 3 } \} e _ { i d } ( u ) b _ { i d } ^ { \mathrm { m o d } } ( u ) , } \\ & { ~ \Delta X _ { i 3 \ell } = \frac { \lambda _ { i 3 \ell } } { q _ { i 3 } } G _ { i 3 } ^ { X } . } \end{array}\tag{16}
$$

History fits determine the carrier and slower envelope; for each eligible target, the seed selects one qualified carrier and modulation from a leading pool of at most eight, then selects a nonempty target subset. Control Setting ℓ changes only the normalized magnitude. Qualification requires the envelope to remain active on the declared fraction of observed horizon positions.

## A.8 SF4 REGIME SWITCHING

SF4 constructs regime switching. Let $\tau _ { i 4 \ell }$ be the setting-specific transition location, $w _ { i 4 }$ the shared transition width, and

$$
\begin{array} { r l } & { p ( u ) = \mathrm { c l i p } \bigg ( \frac { u - \tau } { w } , 0 , 1 \bigg ) , } \\ & { ~ r _ { \kappa } ( u ; \tau , w ) = \displaystyle \Bigg \{ \frac { 1 \{ u \geq \tau \} , } { p ( u ) } \qquad \kappa = \mathrm { s t e p } , } \\ & { \kappa = \mathrm { r a m p } , } \\ & { 3 p ( u ) ^ { 2 } - 2 p ( u ) ^ { 3 } , ~ \kappa = \mathrm { s i g m o i d } , } \end{array}\tag{17}
$$

With affected-target signs $s _ { i d } ^ { \mathrm { r g } } .$ , the five treatments are

$$
\begin{array} { r } { \Delta X _ { i 4 \ell } ( u , d ) = \alpha _ { i 4 } \mathbf { 1 } \{ d \in \mathcal { I } _ { i 4 } \} \sigma _ { i d } s _ { i d } ^ { \mathrm { r g } } r _ { \kappa _ { i 4 } } ( u ; \tau _ { i 4 \ell } , w _ { i 4 } ) . } \end{array}\tag{18}
$$

The seed fixes $\mathcal { I } _ { i 4 }$ , the signs, transition family $\kappa _ { i 4 }$ , and width. The Control Parameter is the transition location, represented by the ordered settings $\tau _ { i 4 \ell } / L _ { i } ;$ larger settings place the same state change closer to the forecast origin and therefore expose less post-transition history. The shared gain $\alpha _ { i 4 }$ is chosen once for the five settings to meet the history-distance contract.

## A.9 SF5 INTERMITTENCY

SF5 constructs intermittency. Let $w _ { i 5 }$ be the shared pulse width. For offset $r ,$ the selected pulse kernel is

$$
\kappa _ { i 5 } ( r ) = \left\{ \begin{array} { l l } { 1 , } & { 0 \leq r < w _ { i 5 } , \quad \mathrm { r e c t a n g u l a r } , } \\ { 1 - \displaystyle \frac { \left| r - ( w _ { i 5 } - 1 ) / 2 \right| } { \operatorname* { m a x } ( 1 , ( w _ { i 5 } + 1 ) / 2 ) } , } & { 0 \leq r < w _ { i 5 } , \quad \mathrm { t r i a n g u l a r } , } \\ { \exp \left[ - \displaystyle \frac { ( \log 2 ) r } { \operatorname* { m a x } ( 1 , w _ { i 5 } / 2 ) } \right] , } & { 0 \leq r < w _ { i 5 } , \quad \mathrm { e x p o n e n t i a l } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{19}
$$

For Control Setting $\ell ,$ event centers $\mathcal { E } _ { i 5 \ell }$ follow mean gap g<sub>i5ℓ</sub> with bounded alternating jitter $\epsilon _ { i 5 \ell } .$ so consecutive gaps are max $( w _ { i 5 } + 1 , g _ { i 5 \ell } \pm \epsilon _ { i 5 \ell } )$ . The target increment is

$$
\Delta X _ { i 5 \ell } ( u , d ) = \alpha _ { i 5 } \mathbf { 1 } \{ d \in \mathcal { I } _ { i 5 } \} \sigma _ { i d } s _ { i d } ^ { \mathrm { e v } } \sum _ { t \in \mathcal { E } _ { i 5 \ell } } \kappa _ { i 5 } ( u - t ) .\tag{20}
$$

The shared structure fixes the affected targets, event signs, pulse family, width, and jitter fraction. The Control Parameter is the mean-gap fraction of the maximum legal gap; larger settings are sparser. A deterministic compatible phase retains at least three history events and an event on an observed future position. The common gain $\alpha _ { i 5 }$ is fixed across settings and satisfies the historydistance and minimum future-effect gates.

## A.10 SF6 COMMON FACTOR

SF6 constructs common factor. Standardize the target history as $Z _ { i } ( u , d ) = ( H _ { i } [ u , d ] - \mu _ { i d } ) / \sigma _ { i d }$ and write its singular value decomposition as $Z _ { i } = \bar { U } _ { i } \Sigma _ { i } V _ { i } ^ { \top }$ . Candidate loading vectors are masked and renormalized members of

$$
\begin{array} { r l } & { \mathcal { B } _ { i } = \big \{ v _ { 1 } , v _ { 1 } \pm 0 . 5 5 v _ { 2 } , v _ { 1 } \pm 0 . 4 5 v _ { 3 } \big \} , } \\ & { \mathcal { W } _ { i } = \big \{ \mathrm { m a s k } _ { \eta } ( w ) : w \in \mathcal { B } _ { i } , \eta \in \{ 0 . 1 0 , 0 . 2 0 , 0 . 3 5 \} \big \} . } \end{array}\tag{21}
$$

where $v _ { j }$ is row $j$ of $V _ { i } ^ { \top }$ and $\mathrm { m a s k } _ { \eta }$ removes loadings below fraction $\eta$ of the largest absolute loading while retaining at least three targets. For a selected loading $w _ { i d }$ , the history-fitted factor $Z _ { i } w _ { i }$ supplies a stable harmonic carrier $f _ { i } ( u \bar { \boldsymbol { u } } ) = a _ { i } ^ { f } \sin ( \omega _ { i } ^ { f } u ) + b _ { i } ^ { f } \cos ( \omega _ { i } ^ { f } u )$ . The controlled component is

$$
G _ { i 6 } ^ { X } ( u , d ) = { \bf 1 } \{ d \in \mathcal { I } _ { i 6 } \} \sigma _ { i d } w _ { i d } f _ { i } ( u ) , \qquad \Delta X _ { i 6 \ell } = \frac { \lambda _ { i 6 \ell } } { q _ { i 6 } } G _ { i 6 } ^ { X } .\tag{22}
$$

After qualification and deduplication, the seed selects one of at most 12 leading loading–carrier candidates; the loading, participating targets, and carrier remain fixed across settings. The Control Parameter is normalized common-factor magnitude. This construction measures observable comovement under the selected loading and carrier.

## A.11 SF7 CROSS-SERIES DEPENDENCE

SF7 constructs lagged dependence by enumerating ordered driver–responder pairs and positive lags up to 24 or one eighth of the Reference-history length, whichever is smaller. For each ordered driver $p ,$ responder $d ,$ and positive lag τ , SF7 fits the history-only predictive comparison

$$
\begin{array} { r l } & { \quad z _ { i , d } ( u ) = \beta _ { 0 } + \beta _ { 1 } z _ { i , d } ( u - 1 ) + \beta _ { 2 } z _ { i , p } ( u - \tau ) + \varepsilon _ { u } , } \\ & { \quad G _ { i 7 } ^ { X } ( u , j ) = \mathbf { 1 } \{ j = d \} \sigma _ { i d } \beta _ { 2 } \widetilde { z } _ { i , p } ( u - \tau ) , } \\ & { \quad \Delta X _ { i 7 \ell } = \frac { \lambda _ { i 7 \ell } } { q _ { i 7 } } G _ { i 7 } ^ { X } . } \end{array}\tag{23}
$$

Here $z _ { i , d }$ is the standardized target history, and $\widetilde { z } _ { i , p }$ equals the observed driver history before the forecast origin and its history-fitted linear continuation afterward. Candidate edges must improve predictive fit over the responder-only model. SF7 ranks them by incremental predictive fit, scans at most 48 edges, and retains at most 12 that satisfy source-distance feasibility; the seed selects one retained edge. The driver, responder, lag, and coefficient $\beta _ { 2 }$ are fixed across settings, while $\lambda _ { i 7 \ell }$ controls transfer magnitude. Under this history-only construction, the resulting edge is interpreted as a predictive diagnostic for SF7.

## A.12 SF8 COVARIATE RESPONSE

SF8 constructs covariate response by selecting a legal dynamic covariate $q ,$ target $d ,$ event set $\mathcal { E } _ { i 8 }$ covariate sign $s _ { i } ^ { U }$ , response sign $s _ { i } ^ { X }$ , terminal pre-origin impulse $t _ { 0 }$ with multiplier $\gamma _ { 0 } .$ , and a onesided response kernel. With delay δ and time-scale parameter h (the half-life for the exponential family), the raw kernel is

$$
\begin{array} { r } { \bar { k } _ { i 8 } ( r ) = \left\{ \begin{array} { l l } { 0 , } & { r < \delta , } \\ { \exp [ - ( \log 2 ) ( r - \delta ) / h ] , } & { \mathrm { e x p o n e n t i a l } , } \\ { \displaystyle \frac { r - \delta } { h } \exp [ 1 - ( r - \delta ) / h ] , } & { \mathrm { d e l a y e d \ g a m m a } , } \\ { \operatorname* { m a x } \left( 0 , 1 - \frac { r - \delta } { 2 h } \right) , } & { \mathrm { t r i a n g u l a r } , } \end{array} \right. } \end{array}\tag{24}
$$

The peak-normalized kernel is

$$
k _ { i 8 } ( r ) = \frac { \bar { k } _ { i 8 } ( r ) } { \operatorname* { m a x } _ { v \geq 0 } \bar { k } _ { i 8 } ( v ) } , \qquad \operatorname* { m a x } _ { r \geq 0 } k _ { i 8 } ( r ) = 1 .\tag{25}
$$

Define the signed impulse path

$$
I _ { i 8 } ( u ) = s _ { i } ^ { U } \left[ \sum _ { t \in \mathcal { E } _ { i 8 } } \mathbf { 1 } \{ u = t \} + \gamma _ { 0 } \mathbf { 1 } \{ u = t _ { 0 } \} \right] .\tag{26}
$$

Let $\sigma _ { i q } ^ { U }$ be the source covariate scale. The raw covariate and target components and their shared setting scaling are

$$
\begin{array} { c } { { G _ { i 8 } ^ { U } ( u , j ) = { \bf 1 } \{ j = q \} \sigma _ { i q } ^ { U } I _ { i 8 } ( u ) , } } \\ { { G _ { i 8 } ^ { X } ( u , j ) = { \bf 1 } \{ j = d \} \sigma _ { i d } s _ { i } ^ { X } \displaystyle \sum _ { v \leq u } I _ { i 8 } ( v ) k _ { i 8 } ( u - v ) , } } \\ { { { } } } \\ { { ( \Delta X _ { i 8 \ell } , \Delta U _ { i 8 \ell } ) = \displaystyle \frac { \lambda _ { i 8 \ell } } { q _ { i 8 } } ( G _ { i 8 } ^ { X } , G _ { i 8 } ^ { U } ) . } } \end{array}\tag{27}
$$

Repeated historical impulses and the terminal impulse use the same kernel. A future covariate impulse is included only when the task declares that covariate known over the horizon; past-only covariates stop at the forecast origin. SF8 constructs at most eight candidate structural configurations; after feasibility qualification, the seed selects one qualified structural configuration, and $\lambda _ { i 8 \ell }$ jointly scales the covariate impulse and target response without changing their signs, timing, target, covariate, or kernel.

## B INSPECTION, SIFTING, AND EVALUATION PROTOCOL

This appendix gives the admission rules, execution contracts, and metric definitions for the constructions in Appendix A. Empirical support is reported in Appendix C.

## B.1 INSPECTION AND SIFTING

Check scopes and experiment binding. Structural Feature Expectation, Source Distance, and physical or task checks operate on individual samples; Distribution Difference and Diversity operate on the generated sample set relative to a declared source population. The contracts below specify both qualification scopes. A reported run binds each check to its descriptor, population, statistic, calibrated threshold or coverage requirement, and observed decision.

## B.1.1 SAMPLE INSPECTION

Sample Inspection evaluates each realized generated sample $V _ { i s \ell }$ against four recorded requirements: consistency with the structure selected in Augmentation; the realized Structural Feature Expectation and non-target tolerance; Source Distance and physical or task constraints on every required model-visible history suffix; and any feature-specific horizon or future-effect support condition. The Source Distance evidence uses Equation (6): every required history view must have macro distance in [0.10, 2.0], and every affected channel must have distance at most 3.0. The verification record stores pass or nonnumeric failure states, and numeric aggregation includes scored samples with their recorded support.

## B.1.2 ADMITTING A COMPLETE SAMPLE SET

A candidate sample set $\widetilde { \mathcal { V } } _ { i s }$ qualifies by satisfying all of the following conditions:

1. Structural Feature s is applicable to $R _ { i } ^ { \prime } { . }$ , a generated sample is present for every $\ell \in \mathcal { L } _ { \varepsilon }$ , and every sample passes Sample Inspection;

2. all generated samples share the selected Reference $R _ { i } ^ { \prime } .$ , fields, and structural configuration required by Equations (4) and (5); and

3. the ordered Control Settings and other sample-set construction invariants of Structural Feature s hold across the complete set; and

4. the set-level Structural Feature Descriptor distribution satisfies the declared Distribution Difference bound and its structural configurations and Control Settings satisfy the declared Diversity requirement.

Each $( R _ { i } ^ { \prime } , s )$ pair contributes all $| { \mathcal { L } } _ { s } |$ generated samples or a recorded nonnumeric state, while its original Reference remains in $\boldsymbol { A } _ { t }$ . Denote a passing set by $\nu _ { i s } ;$ the qualified sample sets of task t form

$$
\mathcal { P } _ { t } = \{ ( R _ { i } ^ { \prime } , \mathcal { V } _ { i s } ) : \widetilde { \mathcal { V } } _ { i s } \mathrm { p a s s e s ~ s a m p l e - s e t ~ q u a l i f i c a t i o n } \} .\tag{28}
$$

The record stores the reason and non-numeric state for each inapplicable or rejected construction. Structural Features that declare a minimum future change enforce it during sample-set construction; other features apply truth-change sufficiency at NRMSE scoring, while an eligible MASE value remains available.

Task-level contract verification. Verification checks treatment records for source distance, mechanism support, and applicable horizon support, recording accepted and rejected contracts. When the source data support exact replay, it also reconstructs the References and generated samples and compares their task records. Write $\mathrm { V e r O K } _ { t } = 1$ when the declared policy accepts the task.

Verified evaluation bundle. For an accepted task, Stage 3 assembles

$$
\mathcal { D } _ { t } = ( \boldsymbol { A } _ { t } , \mathcal { P } _ { t } ) \quad \mathrm { i f ~ } \operatorname { V e r O K } _ { t } = 1 .\tag{29}
$$

The bundle carries the links between each Reference and generated sample together with the accepted verification policy and report. Inference receives only bundles with an accepted report. Any SF6–SF8 ablated-input view is derived from an admitted sample after verification and recorded under a separate inference-ablation contract.

## B.2 EVALUATION

## B.2.1 MODEL CONFIGURATION AND EXECUTION STATES

Each paired Reference–sample comparison uses one fixed model configuration, including model identity, preprocessing, target/covariate mapping, context policy, and permitted adaptation. Auxiliary-input pairs additionally fix the randomization policy and change only the declared input view.

Input projection and output restoration. The Evaluator checks horizon and input compatibility, projects each constructed object to the model context, restores target order after any permitted channel-wise adaptation, and records unsupported, failed, and nonfinite requests separately from valid forecasts. Auxiliary removal is evaluated only for configurations that receive the designated input.

Metric scoreability is recorded separately from execution: each metric uses only valid forecasts and required pairings, while unsupported and nonnumeric states retain their coverage counts.

## B.2.2 MASE SCALE FROM THE REFERENCE HISTORY

Mean absolute scaled error (MASE) (Hyndman & Koehler, 2006) measures forecast error across target scales. Let $L _ { i }$ be the history length, let $p _ { t }$ be the adapter-resolved seasonality, and set $\tau _ { i } = \operatorname* { m i n } ( \operatorname* { m a x } ( 1 , p _ { t } )$ $\operatorname* { m a x } ( 1 , L _ { i } - \overset { \cdot } { 1 } ) )$ . For target $d ,$ let $\mathcal { Q } _ { i d } ( \tau _ { i } )$ contain the history-index pairs admitted by the adapter-resolved scale policy. A complete repaired-history policy admits the repaired cells, whereas a history-observation-preserving scale policy admits pairs whose two source cells are observed. The Reference-history scale is

$$
\sigma _ { i d } ^ { \mathrm { M A S E } } = \frac { 1 } { | \mathscr { Q } _ { i d } ( \tau _ { i } ) | } \sum _ { u \in \mathscr { Q } _ { i d } ( \tau _ { i } ) } | H _ { i , u , d } - H _ { i , u - \tau _ { i } , d } | .\tag{30}
$$

The evaluator uses the declared fallback when the seasonal-pair scale is unavailable or degenerate. The resulting Reference-history scale is shared by the Reference, generated samples, and any eligible ablation pair.

For each Reference and SF, its generated samples and any eligible full/ablated inference pair share the resulting $\sigma _ { i d } ^ { \mathrm { M A S E } }$

For object type $X \in \{ R , V \}$ , let $\Omega _ { i } ^ { X }$ contain its observed future cells. Its instance-level score is

$$
\mathrm { M A S E } _ { i } ^ { X } = \frac { 1 } { | \Omega _ { i } ^ { X } | } \sum _ { ( h , d ) \in \Omega _ { i } ^ { X } } \frac { | \widehat { Y } _ { i , h , d } ^ { X } - Y _ { i , h , d } ^ { X } | } { \sigma _ { i d } ^ { \mathrm { M A S E } } } .\tag{31}
$$

Within a fixed task and result unit, the evaluator averages instance-level values equally. Lower MASE therefore denotes smaller forecast error relative to the Reference history’s variation scale. For $X = R$ , this result object is Reference MASE; for $X = V$ , it is Probe MASE. The input-ablation comparison uses the same formula restricted to the assessed target under the full and ablated input views.

## B.2.3 NRMSE FOR RESPONSE TO CONTROLLED CHANGES

Fix task t, model configuration m, Structural Feature $s ,$ and control setting ℓ. For an affected and observed future cell $\boldsymbol { k } = ( h , d )$ of instance i, let superscript (a) denote the generated-sample view and superscript (0) denote its selected Reference. Define the truth and forecast changes as

$$
\Delta Y _ { i k } = Y _ { i k } ^ { ( a ) } - Y _ { i k } ^ { ( 0 ) } , \qquad \Delta \widehat { Y } _ { i k } = \widehat { Y } _ { i k } ^ { ( a ) } - \widehat { Y } _ { i k } ^ { ( 0 ) } .\tag{32}
$$

Let $\Omega _ { i } ^ { \Delta }$ contain the affected observed cells declared by $T _ { s } ,$ , and use the Reference-history scale $\sigma _ { i k } ^ { \mathrm { M A S E } } = \sigma _ { i d } ^ { \mathrm { M A S E } }$ . The standardized truth-change root-mean-square is

$$
q _ { i } = \sqrt { \frac { 1 } { | \Omega _ { i } ^ { \Delta } | } \sum _ { k \in \Omega _ { i } ^ { \Delta } } { \left( \frac { \Delta Y _ { i k } } { \sigma _ { i k } ^ { \mathrm { M A S E } } } \right) } ^ { 2 } } .\tag{33}
$$

The eligible set $\mathcal { T } _ { t , m , s , \ell } ^ { \Delta }$ contains instances with two valid forecasts, nonempty $\Omega _ { i } ^ { \Delta }$ , and $q _ { i } \geq 0 . 0 5$ The task-level normalized root-mean-square error (NRMSE) is

$$
\mathrm { N R M S E } _ { t , m , s , \ell } = \sqrt { \frac { \displaystyle \sum _ { i \in \mathcal { T } _ { t , m , s , \ell } ^ { \Delta } } \sum _ { k \in \Omega _ { i } ^ { \Delta } } \left( \frac { \Delta \widehat { Y } _ { i k } - \Delta Y _ { i k } } { \sigma _ { i k } ^ { \mathrm { M A S E } } } \right) ^ { 2 } } { \displaystyle \sum _ { i \in \mathcal { T } _ { t , m , s , \ell } ^ { \Delta } } \sum _ { k \in \Omega _ { i } ^ { \Delta } } \left( \frac { \Delta Y _ { i k } } { \sigma _ { i k } ^ { \mathrm { M A S E } } } \right) ^ { 2 } } } .\tag{34}
$$

The ratio pools standardized squared sums over eligible instances and cells and requires a nonempty set with positive truth-change energy. Equivalently, its square is the mean of the squared pair-level ratios weighted by each pair’s standardized target-change energy, which includes its number of affected observed cells. Instances below the 0.05 truth-change threshold remain available to MASE and retain a non-numeric NRMSE state.

## B.2.4 AUXILIARY-INPUT GAP AFTER REMOVING AN AUXILIARY INPUT

The inference ablation applies to model/view pairs that expose the designated auxiliary input. For the SF6 intervention, the highest-absolute-loading target is assessed and the ablated view retains only that target, deleting every other target channel. For the SF7 intervention, the responder is assessed and the ablated view retains only the responder, again deleting every other target channel. For the SF8 intervention, the target panel is retained in both views while the ablated view deletes the designated dynamic-covariate carrier together with its corresponding name, availability, futurevisibility, and type metadata. For both SF6 and SF7, task-permitted non-target covariates remain available in the full and ablated views. The SF6/SF7 comparison includes configurations receiving other target channels; the SF8 comparison includes configurations receiving the designated carrier.

Let $\Omega _ { i } ^ { \mathrm { a b l } }$ contain the observed future cells of the feature-specific assessed target. Let $A _ { i } ^ { ( a ) }$ and $A _ { i } ^ { \mathrm { a b l } }$ denote Equation (31) restricted to $\Omega _ { i } ^ { \mathrm { a b l } }$ for the full generated-sample and ablated-input views, respectively. The two views share the assessed-target history and future, forecast origin, horizon, observation mask, and Reference-history scale. $\operatorname { L e t } { \bar { \mathcal { T } } } _ { t , m , s , \ell } ^ { \mathrm { a b l } }$ contain the eligible pairs for which both forecasts are scoreable. Auxiliary-Input Gap is

$$
\Delta \mathrm { M A S E } _ { t , m , s , \ell } = \frac { 1 } { | \mathcal { T } _ { t , m , s , \ell } ^ { \mathrm { a b l } } | } \sum _ { i \in \mathcal { T } _ { t , m , s , \ell } ^ { \mathrm { a b l } } } \left( A _ { i } ^ { \mathrm { a b l } } - A _ { i } ^ { ( a ) } \right) .\tag{35}
$$

Positive values mean that auxiliary-input removal increases assessed-target error, zero denotes no average signed difference, and negative values mean that removal decreases the error. The quantity records net predictive contribution under the declared deletion.

## B.2.5 RECORDED ANALYSIS AGGREGATION AND UNCERTAINTY

The analyses in this paper use the following CAFE-native task-equal rule; source-benchmark evaluator and leaderboard parity are separate validation targets. For a result unit r selected by the analysis contract for suite-level reporting, let $\mathrm { M e t r i c } _ { t , r , m }$ be the task-level value for model configuration $m _ { : }$ and let $\mathcal { T } _ { r , m }$ be the tasks on which that value is computed. The unit identifies one metric and its evaluation object or comparison condition, plus a structural feature and control setting where applicable. Its recorded task-equal suite result is

$$
\mathrm { S u i t e M e t r i c } _ { r , m } = \frac { 1 } { | T _ { r , m } | } \sum _ { t \in { \mathcal T } _ { r , m } } \mathrm { M e t r i c } _ { t , r , m } , \qquad | { \mathcal T } _ { r , m } | > 0 .\tag{36}
$$

Thus, each benchmark task contributes one value regardless of its number of References or qualified sample sets. The suite row records the task-set cardinality, and task-level values remain available for paired comparisons.

For paired model comparisons, the task set is $\mathcal { T } _ { r , m _ { 1 } } \cap \mathcal { T } _ { r , m _ { 2 } }$ , and the analysis averages one difference per common task with equal weight. The bootstrap resamples tasks for a single-model result and paired task differences for a model comparison; the analysis contract records its seed, repetition count, interval rule, and task identities. Each metric retains its own scoreable support. The secondary structural-response macro uses equally weighted, direction-corrected SF-local ranks on common model support and excludes raw NRMSE pooling and Auxiliary-Input Gap.

## B.3 EVALUATION AND REPLAY SCOPE

Evaluation records. Each result unit retains its benchmark task, Reference, structural feature, control setting, model configuration, and scoreable support. These fields keep paired comparisons aligned while allowing unsupported or nonnumeric cases to remain explicit:

$$
\mathfrak { S } _ { t } \xrightarrow { \mathrm { S t a g e s ~ 1 - 2 } } \mathfrak { G } _ { t } \xrightarrow { \mathrm { S t a g e ~ 3 } } ( \mathcal { D } _ { t } , \mathfrak { V } _ { t } ) \xrightarrow { m } \mathfrak { P } _ { t , m } \xrightarrow { \mathrm { M e t r i c s } } \mathfrak { U } _ { t , m } ,\tag{37}
$$

where ${ \mathfrak { S } } _ { t }$ denotes the benchmark-task input, ${ \mathfrak { G } } _ { t }$ the generation conditions, ${ \mathfrak { V } } _ { t }$ the qualification result, $\mathfrak { P } _ { t , m }$ the predictions, and $\mathfrak { A } _ { t , m }$ the metric results. The analysis records each result unit’s task-set cardinality and keeps the task-level values available for paired comparisons.

The retained records support progressively broader checks, from treatment verification to construction and metric reconstruction and then end-to-end forecasting replay. Appendix C.13 specifies the required inputs and procedures for each reproduction level.

Table 3: Displayed model support of the four latest result packages. Short and FEV use 11 models; Medium and Long use 10 because TiREx-2 is outside the supported horizon.
<table><tr><td>Suite</td><td>Horizon</td><td>Configurations</td><td>Models</td></tr><tr><td>GIFT-Eval Short</td><td>30-60</td><td>55</td><td>11</td></tr><tr><td>GIFT-Eval Medium</td><td>480-600</td><td>21</td><td>10</td></tr><tr><td>GIFT-Eval Long</td><td>720-900</td><td>21</td><td>10</td></tr><tr><td>fev-bench Mini-20</td><td>5-288</td><td>20</td><td>11</td></tr></table>

## C SUPPORTING RESULTS AND REPRODUCIBILITY

The following sections supply experiment-specific support, aggregation details, and complete numerical results in the order of the main experiments. Earlier analysis blocks retain their original populations; reproduction requirements follow the results.

## C.1 RANK-CORRELATION DEFINITIONS

For two quantities X and Y observed on the same n items, Spearman’s rank correlation is the Pearson correlation of their ranks, $\rho _ { s } = \mathrm { c o r r } ( R _ { X } , R _ { Y } )$ ; when there are no ties, this is equivalently $\textstyle \rho _ { s } = 1 - 6 \sum _ { i = 1 } ^ { n } d _ { i } ^ { 2 } / [ n ( n ^ { 2 } - 1 ) ]$ ], where $d _ { i }$ is the difference between the two ranks (?). We use average ranks for tied values.

Kendall’s tie-adjusted rank correlation is reported as

$$
\tau _ { b } = \frac { C - D } { \sqrt { ( C + D + T _ { X } ) ( C + D + T _ { Y } ) } } ,\tag{38}
$$

where C and D count concordant and discordant item pairs, and $T _ { X }$ and $T _ { Y }$ count pairs tied only in X and only in $Y$ , respectively (?). Thus $\rho _ { s }$ summarizes agreement after rank transformation, whereas $\tau _ { b }$ measures pairwise ordering agreement while accounting for ties; both statistics are computed on the stated common model support.

## C.2 COMPUTATIONAL ENVIRONMENT

Experiments used a standard KVM-based compute-node configuration with 48 Intel Xeon Gold 6530 vCPUs, 188 GiB of system memory, and four NVIDIA GeForce RTX 5090 GPUs with 32,607 MiB of memory per GPU, running Ubuntu 24.04.4 LTS. The CAFE data-generation, validation, and aggregation pipeline used Python 3.12.3. Model inference services used Python 3.11.15, PyTorch 2.10.0, CUDA 12.8, cuDNN 9.10, and NVIDIA driver 575.57.08. This profile establishes the standard per-node hardware and software environment. Appendix C.12 summarizes the corresponding evaluation support and analysis scope.

## C.3 EVALUATION SUPPORT AND AGGREGATION

Table 3 records the supported horizons, configuration counts, and model roster for each frozen suite.   
Timer-3.5 denotes the evaluated Timer-S1 checkpoint (Liu et al., 2026).

Suite aggregation follows Equation (36). Probe MASE and Forecast-Change NRMSE first average eligible task-level values within each SF–setting cell, then apply equal setting weights within the feature; NRMSE pools standardized squared terms within each task as in Equation (34). Featurelocal rankings use common support and average ranks for ties. The secondary macro maps directioncorrected ranks to [0, 1] before averaging SF1–SF8 equally; Auxiliary-Input Gap is excluded.

## C.4 ACCURACY AND RESPONSE FIDELITY

The winner table uses the complete evaluated roster, whereas the common ten-model heatmap display omits TiREx-2. The two views therefore compare different model sets on Short and fev-bench Mini-20.

Table 4: Secondary equal-weight structural-response ranking on common model support. Each SF score is converted to a direction-corrected feature-local rank score before averaging equally across SF1–SF8; raw NRMSE values are not pooled.
<table><tr><td>Suite</td><td>Equal-weight diagnostic ranking</td></tr><tr><td>GIFT-Eval Short</td><td>1. TimesFM 3.0 (0.925); 2. Toto 2.0 (0.625); 3. Chronos-2 (0.600); 4. TimesFM 2.5 (0.550); 5. Timer-3.5 (0.175); 6. Moirai 2 (0.125)</td></tr><tr><td>GIFT-Eval Medium</td><td>1. TimesFM 3.0 (0.925); 2. Timer-3.5 (0.650); 3. Chronos-2 (0.600); 4. TimesFM 2.5 (0.375); 5. Toto 2.0 (0.300); 6. Moirai 2 (0.150)</td></tr><tr><td>GIFT-Eval Long</td><td>1. TimesFM 3.0 (0.950); 2. Chronos-2 (0.600); 3. Timer-3.5 (0.600); 4. TimesFM 2.5 (0.400); 5. Toto 2.0 (0.375); 6. Moirai 2 (0.075)</td></tr><tr><td>fev-bench Mini-20</td><td>1. Chronos-2 (0.750); 2. TimesFM 3.0 (0.700); 3. TimesFM 2.5 (0.500); 4. Toto 2.0 (0.375); 5. Timer-3.5 (0.350); 6. Moirai 2 (0.325)</td></tr></table>

Table 5: Inventory of the four latest result panels after restricting each view to the comparison roster.
<table><tr><td>Suite</td><td>Configs</td><td>Models shown</td><td>Panel predictions</td></tr><tr><td>GIFT-Eval Short</td><td>55</td><td>11</td><td>399,740</td></tr><tr><td>GIFT-Eval Medium</td><td>21</td><td>10</td><td>127,380</td></tr><tr><td>GIFT-Eval Long</td><td>21</td><td>10</td><td>121,810</td></tr><tr><td>fev-bench Mini-20</td><td>20</td><td>11</td><td>121,902</td></tr></table>

## C.4.1 LATEST FOUR-PANEL INVENTORY AND WINNERS

Table 5 records the panel prediction counts. These panels retain the source-window, unorderedvariant, and configuration aggregation supplied by their numerical inputs.

Figures 4 and 5 display the cell values. The main-paper radar restricts the NRMSE view to six models shared by all suites and uses a separate scale for each feature.

## C.5 INTRINSIC MODEL SELECTION

The regime-switching coverage table below belongs to an earlier frozen score-cache analysis. It retains that analysis’s seven-family slice, separately from the current intrinsic-selection result in the main paper.

The model-shortlisting analysis uses a frozen score-cache population distinct from the four latest suite panels. The cache contains 55 configurations, 19 source families, 15,673 common-scored original windows, and 6,016 capability–window rows; a capability row can correspond to the same original window as another capability. Twelve model columns are retained, while the recommendation analysis uses eight frozen foundation-model configurations. The analysis record retains their names and complete input order.

For the leave-source-family-out analysis, the held-out family is excluded from all A-side rankings and threshold fitting. The generic strategy uses all A-side original windows, the qualified strategy uses capability-qualified A-side original windows, and the augmented strategy averages the matched augmentation variants within each window and ranks by augmented MASE. Within each family, windows are averaged by configuration before equal-family Borda ranks are averaged across families. The A-side Q75 strength cutoff is applied to B; B is scored only on original windows and the family-balanced champion is determined after equal configuration aggregation. The effect-NRMSE ranking is reported as a separate response diagnostic; MASE determines the recommendations.

The pooled case uses the underlying collection plus original item as the source-group key. All windows, frequency variants, and augmentation variants for one group remain on the same side. Five seeds and both directions yield ten A/B directions; Q75 is fitted on A and applied unchanged to B. The reported Bitbrains subset is an exploratory post-hoc slice selected after inspection of the completed grid and is evaluated only on original MASE. The ten directions share a source pool and are summarized descriptively; no significance test is applied, so this slice provides exploratory descriptive evidence. The A-only restriction applies to threshold fitting and ranking; upstream qualification retains its offline future-quality and feasibility checks.

Table 6: Winners for each structural feature in the four latest result panels. Entries give the model and the number of feature tracks won; ties would be reported jointly.
<table><tr><td>Suite</td><td>Lowest Probe MASE</td><td>Lowest Forecast-Change NRMSE</td></tr><tr><td>GIFT-Eval Short</td><td>TimesFM 3.0 (5/8); Toto 2.0 (2/8); Chronos-2 (1/8)</td><td>TimesFM 3.0 (4/8); Toto 2.0 (2/8); Linear Trend (1/8); TimesFM 2.5 (1/8)</td></tr><tr><td>GIFT-Eval Medium</td><td>TimesFM 3.0 (7/8); Chronos-2 (1/8)</td><td>TimesFM 3.0 (6/8); Chronos-2 (1/8); Linear Trend (1/8)</td></tr><tr><td>GIFT-Eval Long</td><td>TimesFM 3.0 (5/8); Timer-3.5 (2/8); Chronos-2 (1/8)</td><td>TimesFM 3.0 (6/8); Linear Trend (1/8); Timer-3.5 (1/8)</td></tr><tr><td>fev-bench Mini-20</td><td>TimesFM 3.0 (7/8); TiREx-2 (1/8)</td><td>TimesFM 3.0 (3/8); Chronos-2 (1/8); Linear Trend (1/8); Naive (1/8); TimesFM 2.5 (1/8); Toto 2.0 (1/8)</td></tr></table>

Table 7: Regime-switching candidate coverage in the leave-source-family-out analysis. A hit means that the B-side family-balanced original-MASE champion is present in the frozen Top-K list.
<table><tr><td>Recommendation</td><td>Top-1</td><td>Top-3</td><td>Top-5</td></tr><tr><td>Generic original MASE</td><td>1/7</td><td>3/7</td><td>7/7</td></tr><tr><td>Qualified original MASE</td><td>1/7</td><td>4/7</td><td>7/7</td></tr><tr><td>Capability-augmented MASE</td><td>1/7</td><td>6/7</td><td>7/7</td></tr></table>

## C.6 RELATIONAL FEATURE VALIDATION

The historical numerical block retains the input-removal analysis’s task counts and descriptive intervals. Its population is separate from the latest suite panels; the deletion and assessed-target contracts are specified in Appendix B.

## C.7 AUGMENTATION FAITHFULNESS: PAIRED-RANKING STUDY

The small-model paired-ranking study contains five architectures, eight native GIFT-Eval configurations, three training seeds, and three augmentation seeds; the seven analyzed capabilities are trend, multi-seasonal, time-varying seasonality, regime switching, intermittency, common factor, and cross-series dependence. A is the sampled original-source union (2,294 windows), B is the percapability subset of original windows whose paired treatment enters C, and C is the valid treatment set. Across capabilities there are 2,712 source pairs; the three augmentation seeds retain 2,468, 2,466, and 2,460 B pairs, with 2,436 common valid pairs and 21,793 valid treatment samples across the three rounds. These counts define the analyzed support for the paired-ranking result.

For each source, unordered treatment variants are averaged first, followed by equal source averaging within each configuration–feature cell, equal averaging over valid configurations within each feature, and equal averaging over the seven feature summaries; absent features are not imputed. B uses the same source membership, target mask, weights, and original full-history seasonal MASE denominator as C, rather than a denominator recomputed from the augmented history or the 512-point model context. Training-seed scores are averaged after the three augmentation seeds; the reported ± values are sample SD over the three training seeds.

Training uses only the original train/validation splits with context 512 and horizon 48. Minimum original-validation MASE selects the checkpoint across eligible epochs and at most one prescribed regularized retry; selected weights are frozen before testing. Eight model trajectories used a retry and eight used a bounded refinement, while four selected trajectories met the late-overfit diagnostic and therefore used earlier validation-best checkpoints. The analysis reports selected-checkpoint behavior for this suite.

The rank uncertainty analysis uses 1,000 paired bootstrap replicates with fixed selected-suite item clusters, or circular blocks of four consecutive forecast origins for single-item configurations. Training and augmentation seeds are averaged before resampling. The conditional intervals are $\rho \in [ 0 . 7 0 , 0 . 9 0 ]$ and $\tau _ { b } \in [ 0 . 6 0 , 0 . 8 0 ]$ ]; the near-first C difference between iTransformer and DLinear has interval [−0.0269, 0.0114], which spans zero under this resampling. Support is heterogeneous: the multivariate group has three native items, while common-factor and cross-series dependence have 59 and 39 source windows concentrated in three and two items, respectively. These counts define the scope of the selected-suite result. Effect NRMSE remains a separate response diagnostic, while original MASE supplies the ranking metric.

Table 8: Bitbrains exploratory case under Q75 family-Borda selection. Values are direction-equal descriptive means over ten disjoint source-group directions; the B-side support has at least 16 A groups and 14 B groups per direction.
<table><tr><td>Recommendation</td><td>Mean gain vs generic</td><td>Wins-ties-losses</td></tr><tr><td>Qualified original MASE</td><td>11.02%</td><td></td></tr><tr><td>Capability-augmented MASE</td><td>9.06%</td><td>8-1-1</td></tr></table>

Table 9: Small-model B-to-C rank agreement by input group and structural feature. Correlations use five architectures; the multivariate group has only three native items.
<table><tr><td>Group or structural feature</td><td>Spearman ρ</td></tr><tr><td>Multivariate</td><td>1.00</td></tr><tr><td>Univariate</td><td>0.30</td></tr><tr><td>Common factor</td><td>1.00</td></tr><tr><td>Cross-series dependence</td><td>1.00</td></tr><tr><td>Intermittency</td><td>1.00</td></tr><tr><td>Regime switching</td><td>1.00</td></tr><tr><td>Multi-seasonal</td><td>0.90</td></tr><tr><td>Time-varying seasonality</td><td>0.90</td></tr><tr><td>Trend</td><td>-0.10</td></tr></table>

Table 9 preserves the input-group and feature-level results; six of the seven features have $\rho \ge 0 . 9 0$ Table 10 localizes selected trend changes, while the causes of the rank reversals remain unresolved.

## C.8 STABILITY UNDER BENCHMARK REFRESH

The ten-seed local-winner table below belongs to an earlier fixed ten-task panel. The main paper’s five-seed analysis uses the current 55-configuration Short panel and seven-model roster; the two seed studies retain separate supports.

## C.9 TRADE-OFF BETWEEN ACCURACY AND RESPONSE FIDELITY

The full checkpoint trajectories are retained below. The matched training conditions and checkpoint schedule are specified under Reproduction Levels and Procedures.

## C.10 HISTORICAL AND EXPANDED NUMERICAL BLOCKS

The expanded Short results and the earlier six-model cross-suite results use separate populations and aggregation rules. The earlier ten-seed stability and input-removal tables retain their original support; the fine-tuning trajectories describe one training realization.

The expanded Short block pairs Linear Trend’s SF1 response NRMSE of $3 . 2 4 \times 1 0 ^ { - 6 }$ with Probe MASE of 8.074. Its input-removal estimates include an SF6 gap of 0.157 for TimesFM 3.0 and −0.020 for Chronos-2, whose 95% interval includes zero.

## C.11 EARLIER SIX-MODEL CROSS-SUITE DATA BY STRUCTURAL FEATURE

The expanded GIFT-Eval Short package is reported separately from the earlier six-model crosssuite block. The compact cross-suite tables below report Probe MASE / Forecast-Change NRMSE pairs (P/N); lower is better for each component, and NRMSE = 1 is the zero-response reference. The expanded Short tables then report Official MASE, matched original / augmented MASE, and Forecast-Change NRMSE in separate blocks. Values are configuration-equal after the package’s source-window and unordered-variant reductions.

## C.12 EVALUATION COVERAGE AND SCOPE

## C.12.1 EVALUATION SETTINGS

Table 24 distinguishes the main panels from the supplementary analyses and records their support.

Table 10: Trend-treatment sensitivity retained in the small-model package. B is the paired original MASE and C is the valid treatment MASE; lower is better.
<table><tr><td>Configuration-model</td><td>B</td><td>C</td></tr><tr><td>M4 Hourly–Transformer</td><td>1.8348</td><td>9.3486</td></tr><tr><td>M4 Hourly-DLinear</td><td>3.0342</td><td>1.8950</td></tr><tr><td>M4 Hourly-PatchTST</td><td>1.4468</td><td>2.1678</td></tr><tr><td>SZ Taxi-PatchTST</td><td>0.6568</td><td>1.4263</td></tr><tr><td>SZ Taxi-TSMixer</td><td>0.9936</td><td>0.6860</td></tr></table>

Table 11: Probe MASE / Forecast-Change NRMSE pairs (P/N) for the structural features on GIFT-Eval Short. Lower is better for each component; NRMSE = 1 is the zero-response reference.
<table><tr><td>SF</td><td>N</td><td>Chronos-2</td><td>Timer-3.5</td><td>TimesFM 2.5</td><td>TiREx-2</td><td>Moirai 2</td><td>Toto 2.0</td></tr><tr><td>SF1</td><td>18</td><td>2.132 / 0.082</td><td>1.966 / 0.243</td><td>1.898 / 0.216</td><td>1.843 / 0.256</td><td>2.152 / 0.215</td><td>1.876 / 0.243</td></tr><tr><td>SF2</td><td>20</td><td>2.370 / 0.779</td><td>2.758 / 0.946</td><td>2.499 / 0.802</td><td>2.532 / 0.844</td><td>4.171 / 1.010</td><td>1.934 / 0.637</td></tr><tr><td>SF3</td><td>19</td><td>1.526 / 0.529</td><td>1.738 / 0.674</td><td>1.462 / 0.679</td><td>1.484 / 0.580</td><td>1.853 / 0.740</td><td>1.572 / 0.571</td></tr><tr><td>SF4</td><td>20</td><td>1.703 / 0.096</td><td>1.717 / 0.133</td><td>1.627 / 0.091</td><td>1.597 / 0.071</td><td>1.910 / 0.134</td><td>1.574 / 0.144</td></tr><tr><td>SF5</td><td>20</td><td>2.929 / 0.905</td><td>2.872 / 0.914</td><td>2.833 / 0.837</td><td>2.745 / 0.972</td><td>4.211 / 0.965</td><td>2.699 / 0.955</td></tr><tr><td>SF6</td><td>4</td><td>0.618 / 0.257</td><td>0.644 / 0.301</td><td>0.648 / 0.333</td><td>0.658 / 0.259</td><td>0.695 / 0.353</td><td>0.625 / 0.291</td></tr><tr><td>SF7</td><td>8</td><td>2.895 / 0.989</td><td>2.982 / 1.043</td><td>2.896 / 1.187</td><td>2.865 / 1.005</td><td>3.060 / 1.044</td><td>2.746 / 1.460</td></tr><tr><td>SF8</td><td>6</td><td>3.638 / 0.626</td><td>3.833 / 0.554</td><td>3.491 / 0.495</td><td>3.485 / 0.850</td><td>5.907 / 0.684</td><td>3.522 / 0.632</td></tr></table>

The model-shortlisting analysis uses a frozen window-level score cache and history-profile strengths. Its reconstruction record contains the source-group mapping, analysis procedure, results, reference checks, and redacted provenance; the raw time series, point forecasts, model weights, and private inference endpoints are outside this compact record.

The small-model paired-ranking analysis separates training, generation, evaluation, and analysis inputs. The compact handoff contains stage-level summaries; a full end-to-end replay additionally requires the model checkpoints, full predictions, and source time series.

## C.12.2 SUMMARY OF EVIDENCE

Table 25 connects each claim family to the corresponding evaluation evidence and presentation in the paper.

## C.12.3 EVALUATION INVENTORY AND ANALYSIS COVERAGE

Table 26 gives the evaluation inventory for the four main panels. Candidates and References count evaluated windows, generated samples count admitted materializations, and predictions count model forecasts. failure counts refer to issued requests, and metric-specific inapplicable cases remain separate from scored cases.

The four latest panel summaries contribute 88, 80, 80, and 88 model–feature cells for Short, Medium, Long, and FEV, respectively, with the filtered prediction counts shown above. The stability block contains 35 overall seed–model rows in the seven-model view, with feature-level rows retained in the compact summaries. The input-removal block preserves task-level and coverage summaries under the retained model roster. The fine-tuning block contains 22 curve rows and two objective summaries, all from one training realization and a shared 11,269-sample corpus. These counts index downstream records: panel configuration is the latest reduction unit, seed indexes the observed stability panel, and the fine-tuning curves share one training realization. The analysis encodes unavailable, unsupported, not-run, failed, invalid, and metric-undefined states as nonnumeric and preserves their denominators; NRMSE additionally distinguishes scoreable instances from lowtruth-effect and no-observed-effect states as defined in Appendix B.

Table 12: Expanded GIFT-Eval Short full55 package: Official MASE and matched original / augmented MASE for the first five model configurations.
<table><tr><td>Model</td><td>Official</td><td>SF1</td><td>SF2</td><td>SF3</td><td>SF4</td><td>SF5</td><td>SF6</td><td></td><td>SF7</td></tr><tr><td>Chronos-2</td><td>1.567</td><td>2.208/2.216</td><td>0.716/0.730</td><td>1.232/1.241</td><td>1.575/1.859</td><td>0.871/0.896</td><td>0.828/1.194</td><td>0.773/0.802</td><td>0.734/0.765</td></tr><tr><td>TimesFM 2.5</td><td>1.601</td><td>2.140/2.220</td><td>0.713/0.722</td><td>1.244/1.246</td><td>1.911/2.318</td><td>0.969/1.052</td><td>0.828/1.232</td><td>0.782/0.813</td><td>0.745/0.782</td></tr><tr><td>TimesFM 3.0</td><td>1.464</td><td>2.003/2.050</td><td>0.667/0.686</td><td>1.166/1.165</td><td>1.628/1.916</td><td>0.885/0.923</td><td>0.807/1.188</td><td>0.748/0.772</td><td>0.697/0.732</td></tr><tr><td>TiREx-2</td><td>1.610</td><td>2.239/2.216</td><td>0.706/0.732</td><td>1.241/1.246</td><td>1.722/1.914</td><td>0.876/0.921</td><td>0.838/1.233</td><td>0.788/0.817</td><td>0.751/0.801</td></tr><tr><td>Moirai 2</td><td></td><td>1.701 2.258/2.2600.766/0.7971.353/1.372</td><td></td><td></td><td>1.688/1.998</td><td></td><td></td><td></td><td>0.904/1.0300.922/1.327 0.824/0.863 0.809/0.877</td></tr></table>

Table 13: Expanded GIFT-Eval Short full55 panel: Official MASE and matched original / augmented MASE for the remaining six model configurations.
<table><tr><td>Model</td><td>Official</td><td>SF1</td><td>SF2</td><td>SF3</td><td>SF4</td><td>SF5</td><td>SF6</td><td>SF7</td><td>SF8</td></tr><tr><td>Timer-3.5</td><td>1.757</td><td>2.534/2.644</td><td>0.711/0.741</td><td>1.185/1.182</td><td>1.644/1.931</td><td>0.865/0.899</td><td>0.855/1.260</td><td>0.797/0.828</td><td>0.723/0.772</td></tr><tr><td>Toto 2.0</td><td>1.452</td><td>1.971/2.065</td><td>0.673/0.700</td><td>1.186/1.177</td><td>1.580/1.848</td><td>0.934/0.963</td><td>0.837/1.220</td><td>0.771/0.793</td><td>0.690/0.726</td></tr><tr><td>Naive</td><td>2.579</td><td>3.531/4.370</td><td>1.543/1.891</td><td>2.041/2.015</td><td>2.210/2.595</td><td>1.064/1.237</td><td>1.215/1.779</td><td>1.171/1.239</td><td>1.678/2.001</td></tr><tr><td>Seasonal Naive</td><td>2.193</td><td>2.945/3.920</td><td>1.276/1.549</td><td>1.722/1.728</td><td>1.953/2.313</td><td>1.045/1.254</td><td>1.143/1.659</td><td>1.086/1.137</td><td>1.250/1.335</td></tr><tr><td>Drift</td><td>2.565</td><td>3.431/3.431</td><td>1.542/1.897</td><td>2.060/2.036</td><td>2.631/3.269</td><td>1.100/1.283</td><td>1.248/1.821</td><td>1.191/1.258</td><td>1.686/2.012</td></tr><tr><td>Linear Trend</td><td>4.806</td><td>8.074/8.074</td><td>2.211/2.449</td><td>2.566/2.585</td><td>5.219/7.105</td><td>1.599/1.867</td><td>2.155/3.016</td><td>1.905/2.028</td><td>1.841/2.168</td></tr></table>

Table 14: Expanded GIFT-Eval Short full55 package: Forecast-Change NRMSE by structural feature for the first five model configurations.
<table><tr><td>Model</td><td>SF1</td><td>SF2</td><td>SF3</td><td>SF4</td><td>SF5</td><td>SF6</td><td>SF7</td><td>SF8</td></tr><tr><td>Chronos-2</td><td>1.371</td><td>0.437</td><td>0.655</td><td>0.401</td><td>0.991</td><td>0.392</td><td>0.427</td><td>0.420</td></tr><tr><td>TimesFM 2.5</td><td>1.506</td><td>0.472</td><td>0.677</td><td>0.396</td><td>0.901</td><td>0.419</td><td>0.537</td><td>0.475</td></tr><tr><td>TimesFM 3.0</td><td>1.179</td><td>0.340</td><td>0.624</td><td>0.387</td><td>0.928</td><td>0.392</td><td>0.403</td><td>0.389</td></tr><tr><td>TiREx-2</td><td>1.119</td><td>0.534</td><td>0.650</td><td>0.403</td><td>0.941</td><td>0.418</td><td>0.497</td><td>0.504</td></tr><tr><td>Moirai 2</td><td>2.109</td><td>0.609</td><td>0.642</td><td>0.419</td><td>1.024</td><td>0.449</td><td>0.609</td><td>0.617</td></tr></table>

Table 15: Expanded GIFT-Eval Short full55 panel: Forecast-Change NRMSE by structural feature for the remaining six model configurations.
<table><tr><td>Model</td><td>SF1</td><td>SF2</td><td>SF3</td><td>SF4</td><td>SF5</td><td>SF6</td><td>SF7</td><td>SF8</td></tr><tr><td>Timer-3.5</td><td>2.229</td><td>0.498</td><td>0.678</td><td>0.423</td><td>0.965</td><td>0.442</td><td>0.545</td><td>0.454</td></tr><tr><td>Toto 2.0</td><td>1.611</td><td>0.479</td><td>0.624</td><td>0.397</td><td>0.951</td><td>0.422</td><td>0.433</td><td>0.386</td></tr><tr><td>Naive</td><td>1.000</td><td>01.156</td><td>0.750</td><td>0.560</td><td>1.091 0.554</td><td></td><td>0.6500.789</td><td></td></tr><tr><td>Seasonal Naive</td><td></td><td>2.213 1.020</td><td>0.747</td><td>0.532</td><td>1.068</td><td></td><td></td><td>0.563 0.628 0.760</td></tr><tr><td>Drift</td><td> $3 . 4 2 \times 1 0 ^ { - 6 }$ </td><td></td><td>1.175 0.753</td><td>0.578</td><td></td><td></td><td></td><td>1.112 0.560 0.657 0.794</td></tr><tr><td>Linear Trend</td><td> $3 . 2 4 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.079 1.002 0.711 1.174 0.920 1.354 0.880</td></tr></table>

Table 16: Expanded GIFT-Eval Short full55 input-removal results. Positive ∆MASE favors retaining the auxiliary input; intervals are descriptive task-bootstrap 95% intervals.
<table><tr><td>Model</td><td>SF</td><td>Tasks</td><td>∆MASE</td><td>95% interval</td></tr><tr><td>Chronos-2</td><td>SF6</td><td>13</td><td>-0.020</td><td>[-0.113, 0.052]</td></tr><tr><td>Chronos-2</td><td>SF7</td><td>12</td><td>0.000</td><td>[-0.119, 0.113]</td></tr><tr><td>Chronos-2</td><td>SF8</td><td>8</td><td>0.006</td><td>[0.000, 0.013]</td></tr><tr><td>TimesFM 3.0</td><td>SF6</td><td>13</td><td>0.157</td><td>[0.064, 0.259]</td></tr><tr><td>TimesFM 3.0</td><td>SF7</td><td>11</td><td>0.051</td><td>[-0.020, 0.160]</td></tr><tr><td>TimesFM 3.0</td><td>SF8</td><td>7</td><td>0.003</td><td>[-0.000, 0.009]</td></tr><tr><td>Toto 2.0</td><td>SF6</td><td>13</td><td>-0.022</td><td>[-0.163, 0.081]</td></tr><tr><td>Toto 2.0</td><td>SF7</td><td>12</td><td>0.033</td><td>[-0.010, 0.107]</td></tr></table>

Table 17: Probe MASE / Forecast-Change NRMSE pairs (P/N) for the structural features on GIFT-Eval Medium. Lower is better for each component; NRMSE = 1 is the zero-response reference.
<table><tr><td>SF</td><td>N</td><td>Chronos-2</td><td>Timer-3.5</td><td>TimesFM 2.5</td><td>TiREx-2</td><td>Moirai 2</td><td>Toto 2.0</td></tr><tr><td>SF1</td><td>9</td><td>2.982 / 0.051</td><td>2.808 / 0.264</td><td>2.880 / 0.259</td><td></td><td>3.469 / 0.293</td><td>12.785 / 0.590</td></tr><tr><td>SF2</td><td>11</td><td>5.309 / 1.102</td><td>5.533 / 1.206</td><td>4.016 / 0.961</td><td></td><td>10.671 / 1.729</td><td>10.488 / 1.131</td></tr><tr><td>SF3</td><td>10</td><td>7.051 / 0.787</td><td>7.292 /1.130</td><td>3.507 / 0.980</td><td></td><td>10.616 /1.565</td><td>9.955 / 1.296</td></tr><tr><td>SF4</td><td>11</td><td>2.378 / 0.053</td><td>2.581 / 0.191</td><td>2.453 / 0.131</td><td></td><td>4.026 / 0.276</td><td>7.597 / 0.176</td></tr><tr><td>SF5</td><td>11</td><td>2.851 / 1.402</td><td>3.039 / 1.738</td><td>2.936 / 1.322</td><td></td><td>3.592 / 1.115</td><td>6.459 / 1.556</td></tr><tr><td>SF6</td><td>4</td><td>0.919 / 0.257</td><td>0.952 / 0.474</td><td>0.966 / 0.524</td><td></td><td>1.324 / 0.571</td><td>1.163 / 0.377</td></tr><tr><td>SF7</td><td>6</td><td>7.690 / 0.992</td><td>8.041 / 1.070</td><td>8.089 / 1.053</td><td></td><td>9.902 / 1.051</td><td>15.988 / 1.189</td></tr><tr><td>SF8</td><td>4</td><td>5.151 / 0.533</td><td>6.848 / 0.498</td><td>6.849 / 0.373</td><td></td><td>9.773 / 0.457</td><td>19.072 / 0.550</td></tr></table>

Table 18: Probe MASE / Forecast-Change NRMSE pairs (P/N) for the structural features on GIFT-Eval Long. Lower is better for each component; NRMSE = 1 is the zero-response reference.
<table><tr><td>SF</td><td>N</td><td>Chronos-2</td><td>Timer-3.5</td><td>TimesFM 2.5</td><td>TiREx-2</td><td>Moirai 2</td><td>Toto 2.0</td></tr><tr><td>SF1</td><td>10</td><td>3.597 / 0.062</td><td>3.549 / 0.191</td><td>3.682 / 0.124</td><td></td><td>4.609 / 0.213</td><td>16.115 / 0.257</td></tr><tr><td>SF2</td><td>11</td><td>6.252 / 0.914</td><td>6.780 / 1.168</td><td>4.890 / 0.966</td><td></td><td>9.542 / 1.638</td><td>21.869 / 0.998</td></tr><tr><td>SF3</td><td>10</td><td>4.860 / 0.859</td><td>8.021 / 1.223</td><td>5.632 / 0.961</td><td></td><td>9.717 /1.190</td><td>23.432 / 1.119</td></tr><tr><td>SF4</td><td>11</td><td>2.934 / 0.064</td><td>3.100 / 0.175</td><td>3.253 / 0.106</td><td></td><td>4.442 / 0.326</td><td>16.994 / 0.265</td></tr><tr><td>SF5</td><td>11</td><td>3.131 / 1.051</td><td>3.087 / 1.118</td><td>3.185 / 1.069</td><td></td><td>4.636 / 1.683</td><td>16.407 / 1.184</td></tr><tr><td>SF6</td><td>4</td><td>1.039 / 0.385</td><td>1.144 / 0.739</td><td>1.113 / 0.772</td><td></td><td>1.749 / 1.084</td><td>1.324 / 0.663</td></tr><tr><td>SF7</td><td>6</td><td>6.669 / 0.893</td><td>7.005 / 0.995</td><td>7.446 / 1.058</td><td></td><td>10.128 / 1.019</td><td>36.761 / 1.326</td></tr><tr><td>SF8</td><td>3</td><td>9.909 / 0.397</td><td>13.459 / 0.594</td><td>9.512 / 0.335</td><td></td><td>14.300 / 0.490</td><td>53.238 / 0.857</td></tr></table>

Table 19: Probe MASE / Forecast-Change NRMSE pairs (P/N) for the structural features on fevbench Mini-20. Lower is better for each component; NRMSE = 1 is the zero-response reference.
<table><tr><td>SF</td><td>N</td><td>Chronos-2</td><td>Timer-3.5</td><td>TimesFM 2.5</td><td>TiREx-2</td><td>Moirai 2</td><td>Toto 2.0</td></tr><tr><td>SF1</td><td>18</td><td>1.938 / 0.176</td><td>1.807 / 0.181</td><td>1.469 / 0.111</td><td>1.736 / 0.130</td><td>1.625 / 0.087</td><td>1.591 / 0.124</td></tr><tr><td>SF2</td><td>20</td><td>2.656 / 1.202</td><td>3.003 / 1.143</td><td>2.582 / 0.954</td><td>2.703 / 1.042</td><td>3.144 / 1.213</td><td>1.839 / 0.809</td></tr><tr><td>SF3</td><td>20</td><td>2.190 / 0.812</td><td>2.140 / 0.756</td><td>1.744 / 0.734</td><td>2.040 / 0.794</td><td>2.045 / 0.797</td><td>1.659 / 0.663</td></tr><tr><td>SF4</td><td>20</td><td>1.913 / 0.213</td><td>1.917 / 0.163</td><td>1.562 / 0.120</td><td>1.853 / 0.136</td><td>1.647 / 0.103</td><td>1.603 / 0.110</td></tr><tr><td>SF5</td><td>18</td><td>2.188 / 0.934</td><td>1.987 / 0.963</td><td>1.712 / 0.864</td><td>2.154 / 0.894</td><td>2.055 / 1.062</td><td>1.731 / 0.852</td></tr><tr><td>SF6</td><td>9</td><td>1.766 / 0.286</td><td>2.044 / 0.359</td><td>1.630 / 0.301</td><td>1.819 / 0.291</td><td>1.791 / 0.380</td><td>1.663 / 0.295</td></tr><tr><td>SF7</td><td>9</td><td>1.766 / 0.718</td><td>2.100 / 0.782</td><td>1.678 / 0.763</td><td>1.693 / 0.717</td><td>1.711 / 0.834</td><td>1.826 / 0.752</td></tr><tr><td>SF8</td><td>8</td><td>3.073 / 0.776</td><td>3.046 / 0.642</td><td>2.480 / 0.691</td><td>2.972 / 0.651</td><td>3.025 / 0.674</td><td>2.303 / 0.551</td></tr></table>

Table 20: Spearman rank correlations among Reference MASE (R), Probe MASE (P), and NRMSE (N) for the displayed model sets; the table reports point correlations, with interval estimation treated separately.
<table><tr><td>Suite</td><td>Models</td><td>ρR,P</td><td>ρR,N</td><td>ρP,N</td></tr><tr><td>GIFT-Eval Short</td><td>6</td><td>0.943</td><td>-0.086</td><td>0.029</td></tr><tr><td>GIFT-Eval Medium</td><td>5</td><td>0.700</td><td>0.800</td><td>0.800</td></tr><tr><td>GIFT-Eval Long</td><td>5</td><td>0.900</td><td>0.800</td><td>0.900</td></tr><tr><td>fev-bench Mini-20</td><td>6</td><td>0.657</td><td>0.543</td><td>0.771</td></tr></table>

Table 21: Local-winner stability across ten GIFT-Eval Short seeds for the structural features. Unanimous settings retain one lowest-NRMSE model in all seeds; modal fractions are descriptive.
<table><tr><td>SF</td><td>Unanimous / 5</td><td>Mean modal</td><td>Min modal</td><td>Setting-modal models</td></tr><tr><td>SF1</td><td>5</td><td>1.00</td><td>1.00</td><td>Chronos-2 (5)</td></tr><tr><td>SF2</td><td>0</td><td>0.82</td><td>0.70</td><td>TimesFM 2.5 (1), Toto 2.0 (4)</td></tr><tr><td>SF3</td><td>5</td><td>1.00</td><td>1.00</td><td>Chronos-2 (5)</td></tr><tr><td>SF4</td><td>0</td><td>0.68</td><td>0.50</td><td>TiREx-2 (3), TimesFM 2.5 (1), Chronos-2 (1)</td></tr><tr><td>SF5</td><td>0</td><td>0.74</td><td>0.50</td><td>TimesFM 2.5 (4), Chronos-2 (1)</td></tr><tr><td>SF6</td><td>0</td><td>0.64</td><td>0.50</td><td>Chronos-2 (5)</td></tr><tr><td>SF7</td><td>0</td><td>0.60</td><td>0.50</td><td>TiREx-2 (4), Chronos-2 (1)</td></tr><tr><td>SF8</td><td>2</td><td>0.86</td><td>0.70</td><td>TimesFM 2.5 (5)</td></tr></table>

Table 22: Complete Auxiliary-Input Gap results for the evaluated suite–model–feature cells. Each estimate is $\mathrm { M A \bar { S } E _ { a b l a t e d } - M A S \bar { E } _ { f u l l } }$ with a descriptive task-bootstrap 95% interval; positive values favor the full auxiliary input.
<table><tr><td>Suite</td><td>SF</td><td>Model</td><td>Tasks</td><td>∆MASE [95% interval]</td><td>Task median</td><td>Positive tasks</td></tr><tr><td></td><td></td><td></td><td></td><td>0.038</td><td></td><td></td></tr><tr><td>GIFT-Eval Short</td><td>SF6</td><td>Chronos-2</td><td>4</td><td>[0.009, 0.068] -0.004</td><td>0.042</td><td>3/4</td></tr><tr><td>GIFT-Eval Short</td><td>SF6</td><td>TiREx-2</td><td>4</td><td>[-0.006, -0.001] 0.005</td><td>-0.005</td><td>1/4</td></tr><tr><td>GIFT-Eval Short</td><td>SF6</td><td>Toto 2.0</td><td>4</td><td>[-0.012, 0.030] 0.012</td><td>-0.004</td><td>2/4</td></tr><tr><td>GIFT-Eval Short</td><td>SF7</td><td>Chronos-2</td><td>8</td><td>[-0.001, 0.030] 0.003</td><td>0.002</td><td>6/8</td></tr><tr><td>GIFT-Eval Short</td><td>SF7</td><td>TiREx-2</td><td>8</td><td>[-0.002, 0.008] 0.031</td><td>0.001</td><td>5/8</td></tr><tr><td>GIFT-Eval Short</td><td>SF7</td><td>Toto 2.0</td><td>8</td><td>[-0.002, 0.077] -0.001</td><td>-0.001</td><td>4/8</td></tr><tr><td>GIFT-Eval Short</td><td>SF8</td><td>Chronos-2</td><td>6</td><td>[-0.012, 0.010] 0.001</td><td>0.001</td><td>4/6</td></tr><tr><td>GIFT-Eval Short</td><td>SF8</td><td>TiREx-2</td><td>6</td><td> $[ - 1 . 2 1 \times 1 0 ^ { - 4 } , 0 . 0 0 3 ] 2 . 2 9 \times 1 0 ^ { - 4 }$  -0.007</td><td></td><td>4/6</td></tr><tr><td>GIFT-Eval Medium</td><td>SF6</td><td>Chronos-2</td><td>4</td><td>[-0.082, 0.050] -0.024</td><td>0.012</td><td>3/4</td></tr><tr><td>GIFT-Eval Medium</td><td>SF6</td><td>Toto 2.0</td><td>4</td><td>[-0.095, 0.038] 0.005</td><td>-0.014</td><td>2/4</td></tr><tr><td>GIFT-Eval Medium</td><td>SF7</td><td>Chronos-2</td><td>6</td><td>[-0.048, 0.048] -0.039</td><td>0.011</td><td>4/6</td></tr><tr><td>GIFT-Eval Medium</td><td>SF7</td><td>Toto 2.0</td><td>6</td><td>[-0.093, 0.002]</td><td>-0.016</td><td>1/6</td></tr><tr><td>GIFT-Eval Medium</td><td>SF8</td><td>Chronos-2</td><td>4</td><td>0.011 [-0.002, 0.033]</td><td>0.001</td><td>2/4</td></tr><tr><td>GIFT-Eval Long</td><td>SF6</td><td>Chronos-2</td><td>4</td><td>0.034 [-0.064, 0.153]</td><td>0.011</td><td>3/4</td></tr><tr><td>GIFT-Eval Long</td><td>SF6</td><td>Toto 2.0</td><td>4</td><td>-0.012 [-0.029, 0.005]</td><td>-0.012</td><td>1/4</td></tr><tr><td>GIFT-Eval Long</td><td>SF7</td><td>Chronos-2</td><td>6</td><td>-0.006 [-0.092, 0.082]</td><td>-0.014</td><td>2/6</td></tr><tr><td>GIFT-Eval Long</td><td>SF7</td><td>Toto 2.0</td><td>6</td><td>0.195 [-0.034, 0.629]</td><td>-0.008</td><td>3/6</td></tr><tr><td>GIFT-Eval Long</td><td>SF8</td><td>Chronos-2</td><td>3</td><td>-0.005 [-0.010, -0.002]</td><td>-0.003</td><td>0/3</td></tr><tr><td>fev-bench Mini-20</td><td>SF6</td><td>Chronos-2</td><td>8</td><td>0.048 [-0.005, 0.128]</td><td>0.003</td><td>6/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF6</td><td>TiREx-2</td><td>8</td><td>-0.020 [-0.056, 0.001]</td><td>-0.003</td><td>2/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF6</td><td>Toto 2.0</td><td>8</td><td>0.135 [-0.010, 0.402]</td><td>-0.001</td><td>4/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF7</td><td>Chronos-2</td><td>8</td><td>0.013 [0.004, 0.024]</td><td>0.009</td><td>7/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF7</td><td>TiREx-2</td><td>8</td><td>-0.010 [-0.029, 0.003]</td><td>0.001</td><td>5/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF7</td><td>Toto 2.0</td><td>8</td><td>0.505 [0.002, 1.495]</td><td>0.014</td><td>7/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF8</td><td>Chronos-2</td><td>8</td><td>2.935 [-0.299, 8.982]</td><td>0.150</td><td>5/8</td></tr><tr><td>fev-bench Mini-20</td><td>SF8</td><td>TimesFM 2.5</td><td>6</td><td>0.037 [0.004, 0.079]</td><td>0.028</td><td>5/6</td></tr><tr><td>fev-bench Mini-20</td><td>SF8</td><td>TiREx-2</td><td>8</td><td>0.038 [-0.530, 0.706]</td><td>0.004</td><td>5/8</td></tr></table>

Table 23: Complete single-realization Chronos-2 trajectories for official quantile-loss and Forecast-Change NRMSE objectives. Values use the panel aggregation on training seed 2026092406 and the same 11,269 samples. This block reports one training seed, so repeated-seed uncertainty is a separate analysis.
<table><tr><td>Step</td><td>Official MASE</td><td>Official FC-NRMSE</td><td>FC-NRMSE-objective MASE</td><td>FC-NRMSE-objective</td></tr><tr><td>0</td><td>0.725</td><td>0.641</td><td>0.725</td><td>0.641</td></tr><tr><td>1k</td><td>0.681</td><td>0.593</td><td>0.778</td><td>0.533</td></tr><tr><td>2k</td><td>0.665</td><td>0.556</td><td>0.804</td><td>0.488</td></tr><tr><td>3k</td><td>0.655</td><td>0.582</td><td>0.816</td><td>0.473</td></tr><tr><td>4k</td><td>0.645</td><td>0.587</td><td>0.815</td><td>0.483</td></tr><tr><td>5k</td><td>0.641</td><td>0.606</td><td>0.814</td><td>0.477</td></tr><tr><td>6k</td><td>0.635</td><td>0.605</td><td>0.819</td><td>0.475</td></tr><tr><td>7k</td><td>0.632</td><td>0.607</td><td>0.820</td><td>0.476</td></tr><tr><td>8k</td><td>0.630</td><td>0.614</td><td>0.819</td><td>0.478</td></tr><tr><td>9k</td><td>0.629</td><td>0.616</td><td>0.820</td><td>0.471</td></tr><tr><td>10k</td><td>0.628</td><td>0.615</td><td>0.820</td><td>0.473</td></tr></table>

Table 24: Evaluation panels and their declared support. Detailed source provenance and run identities are retained with the analysis record.
<table><tr><td>Panel</td><td>Evaluation setting</td><td>Main support</td><td>Provenance</td></tr><tr><td>GIFT-Eval Short latest</td><td>Frozen latest Short panel</td><td>55 configurations; 3 unordered variants; 11 models</td><td>Analysis record</td></tr><tr><td>GIFT-Eval Medium latest</td><td>Frozen latest Medium panel</td><td>21 configurations; 10 models</td><td>Analysis record</td></tr><tr><td>GIFT-Eval Long latest</td><td>Frozen latest Long panel</td><td>21 configurations; 10 models</td><td>Analysis record</td></tr><tr><td>fev-bench Mini-20 latest</td><td>Frozen latest Mini-20 panel</td><td>20 configurations; 11 models</td><td>Analysis record</td></tr><tr><td>Input removal</td><td>Matched full-input and target-only views</td><td>Same treatment contracts; assessed target history and future fixed</td><td>Separate block</td></tr><tr><td>Stability (latest)</td><td>Five augmentation seeds on Short</td><td>55 configurations; 7 models</td><td>Analysis record</td></tr><tr><td>Stability (earlier)</td><td>Fixed ten-task Short panel</td><td>Ten augmentation seeds</td><td>Historical block</td></tr><tr><td>Fine-tuning</td><td>One shared generated corpus and two objectives</td><td>11,269 samples; 10,000 steps; check- points every 1,000 steps</td><td>Single realization</td></tr><tr><td>Model shortlisting</td><td>Frozen window-level score cache</td><td>15,673 common-scored windows; 55 configurations; 19 source families; 8- model roster</td><td>CPU reconstruction</td></tr></table>

Table 25: Summary of evaluation evidence used for the current results.
<table><tr><td>Claim family</td><td>Evaluation evidence</td><td>Result presentation</td></tr><tr><td>SF1-SF8 interventions</td><td>Structural constructions and their ordered control Feature-level response profiles and heatmaps settings</td><td></td></tr><tr><td>Generated-sample accuracy and response</td><td>Four evaluation panels with suite-specific support</td><td>MASE and Forecast-Change NRMSE tables</td></tr><tr><td>Structural-response profiles</td><td>Feature-local comparisons on the eligible model Main and supplementary heatmaps support</td><td></td></tr><tr><td>Refresh stability</td><td>Five-seed and earlier fixed-panel comparisons</td><td>Stability summary and figure</td></tr><tr><td>Auxiliary-input utility</td><td>Matched full-input and target-only comparisons</td><td>Auxiliary-Input Gap table and figure</td></tr><tr><td>Training-objective trade-off</td><td>Two objectives trained on the same sample set</td><td>Fine-tuning curves and objective comparison</td></tr></table>

Table 26: Evaluation inventory across the four main panels.
<table><tr><td>Suite</td><td>Configs</td><td>Candidates/References</td><td>Generated samples</td><td>Models</td><td>Predictions</td><td>Failures</td><td>Horizon</td></tr><tr><td>GIFT-Eval Short</td><td>55</td><td>15,745</td><td>17,773</td><td>11</td><td>399,740</td><td>0</td><td>30-60</td></tr><tr><td>GIFT-Eval Medium</td><td>21</td><td>4,259</td><td>6,875</td><td>10</td><td>127,380</td><td>0</td><td>480-600</td></tr><tr><td>GIFT-Eval Long</td><td>21</td><td>4,179</td><td>6,552</td><td>10</td><td>121,810</td><td>0</td><td>720-900</td></tr><tr><td>fev-bench Mini-20</td><td>20</td><td>2,809</td><td>5,609</td><td>11</td><td>121,902</td><td>0</td><td>5-288</td></tr></table>

## C.13 REPRODUCTION LEVELS AND PROCEDURES

This subsection defines three reproducibility levels: checking the consistency of reported summaries, reconstructing the tables and figures from compact numerical summaries, and repeating the complete data construction and forecasting evaluation. The first two levels are the reconstruction scope described by the compact records; the third states the inputs required for a fresh end-to-end run.

## C.13.1 REPRODUCTION LEVELS

Table 27: Reproduction levels and resource requirements.
<table><tr><td>Level</td><td>Declared environment</td><td>Purpose</td></tr><tr><td>Summary consistency</td><td>Linux or macOS; CPU; less than 1 GB memory and disk</td><td>Checks row counts, dimensions, and numerical agreement for rep- resentative tables and figures</td></tr><tr><td>Table and figure reconstruc- tion</td><td>Linux or macOS; Python 3.11 or later; 41 CPU cores, 8 GB RAM, and 5 GB free compact numerical summaries disk recommended</td><td>Recomputes the four analysis blocks reported in the paper from</td></tr><tr><td>End-to-end evaluation</td><td>VRAM per worker; 16 CPU cores and 64 GB RAM recommended</td><td>Linux x86-64; Python 3.12; CUDA Defines the environment for reconstructing treatments, evaluating 12.8-compatible NVIDIA driver; 32 GB the models, and recomputing the reported analyses</td></tr></table>

## C.13.2 RECONSTRUCTION PROCEDURE

Summary consistency checks compare the numerical summaries with the displayed tables and figures. Table and figure reconstruction recomputes the four main evaluation summaries, the auxiliaryinput comparison, the stability analysis, and the fine-tuning curves from compact numerical inputs. End-to-end evaluation is the broader procedure: it requires the benchmark data, model checkpoints, full predictions, and computational resources needed to repeat profiling, augmentation, inspection, forecasting, and metric aggregation.

The model-shortlisting package provides a CPU-only reconstruction route from the frozen windowlevel score cache. It rebuilds the source splits, leave-family rankings, pooled sensitivity grid, tables, figures, and verification without CaFE, GPU hardware, model weights, raw time series, or network access. Its numerical checks reproduce the frozen package outputs for the declared reconstruction scope.

## C.13.3 CHRONOS-2 OBJECTIVE COMPARISON

The Chronos-2 comparison uses the same 11,269-sample corpus for the standard quantile-loss and response-focused objectives. Both branches use the same learning rate, model modules, training seed, schedule, context lengths, and evaluation conditions, with checkpoints every 1,000 steps through 10,000 steps. The comparison therefore isolates the change in training objective for this realization.