# Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons

Huseyin Cavus<sup>1</sup> <sup>⋆</sup>, Sebin Sabu<sup>2⋆</sup>, Joshua Spear<sup>2</sup>, Jaskaran Singh Kawatra<sup>2</sup>, and Pavithra Rajendran<sup>2</sup>

<sup>1</sup> Trakya University, Edirne, Türkiye

2 DRIVE, Great Ormond Street Hospital for Children NHS Foundation Trust, London, UK sebin.sabu@gosh.nhs.uk

Abstract. Interpretable machine learning for Large Language Models (LLMs) increasingly relies on sparse probing methods that identify small sets of feedforward neurons claimed to detect and causally influence behaviors such as factuality recall, safety alignment, and hallucination. These claims have important implications for model auditing and behavioral steering, yet they are rarely tested against known failure modes of L -regularized probing in correlated, high-dimensional feature spaces. We propose a five-step diagnostic protocol covering feature correlation, bootstrap stability, sparse versus dense ranking disagreement, intervention baselines, and cross-dataset evaluation as a minimum standard for sparse-neuron localization claims. In this paper, we investigate and study prior work [4] using our proposed approach, specifically on H-neurons using open-source LLMs (Gemma 3 4B and MedGemma 4B) across TriviaQA, BioASQ, and NQ-Open datasets. Our results demonstrate detection replicates across both models and datasets, and exceeds the original reported AUROC gaps for TriviaQA and BioASQ datasets. Gemma 3 4B consistently outperforms MedGemma 4B on matched datasets, with AUROC gaps of +0.311 versus +0.235 on TriviaQA, +0.474 versus +0.455 on BioASQ, and +0.128 versus +0.112 on NQ-Open respectively. Causal validation at n = 500 with five random seeds shows statistically significant efects beyond random same-layer baselines. At the same time, the diagnostic results indicate that the selected neurons are not uniquely localized. Across the three Gemma 3 4B settings, 19 of 22 selected H-Neurons have Pearson $| r | > 0 . 7$ with other features, bootstrap selections show only moderate stability, and sparse and dense rankings overlap only weakly. Our findings show that sparse predictive structure can coexist with non-unique neuron selection. Routine diagnostic validation is necessary to distinguish detection claims from localization claims in mechanistic interpretability.

Keywords: Interpretable machine learning · Mechanistic interpretability · Sparse probing · Large language models · Hallucination · Diagnostic methodology

<sup>⋆</sup> These authors contributed equally.

## 1 Introduction

Mechanistic interpretability is a growing area of research claiming that high-level behaviors of Large Language Models (LLMs) can be localized to sparse sets of internal units. Examples include skill neurons [1], safety neurons [2], knowledge neurons [3], and most recently hallucination-associated neurons or H-Neurons [4]. These claims share a common pattern: a small fraction of feedforward neurons is reported to (i) detect a behavior with high accuracy, (ii) causally control it under intervention, and (iii) emerge during pre-training rather than alignment. If valid, such claims ofer powerful afordances for model auditing, behavioral steering, and theoretical understanding of how capabilities arise during training [5,6].

The methodological pipeline underlying these claims, typically $L _ { 1 } .$ -regularized probing or activation contrasting over feedforward activations [7], is known to be vulnerable to specific failure modes when applied to high-dimensional, correlated feature spaces [8,9,10]. $L _ { 1 }$ regularization in the presence of correlated features tends to select arbitrary cluster representatives rather than uniquely informative units [9]. When the underlying feature space is dense with correlated activations [7,11], a single probe run can produce a sparse selection that is predictive without being uniquely localized. Recent peer-reviewed work in this lineage [1,2,25] validates localization claims by measuring direct neuron-set overlap across tasks, objectives, or datasets, a diagnostic the original H-Neurons report does not include.

In this paper, we propose a diagnostic evaluation of H-Neurons [4] as a case study in structured validation for sparse-neuron interpretability using a fivestep protocol as a minimum standard for sparse-neuron localization claims. Our contributions are as follows:

– Our proposed protocol to H-neurons is evaluated across two open-source LLMs (gemma-3-4b-it [12] and medgemma-4b-it) and three Question Answering datasets (TriviaQA [13], BioASQ [14], and NQ-Open [15]).

Our findings refine rather than refute the original H-Neurons claims. First, we successfully replicate and exceed the reported detection performance, confirming through rigorous evaluation that targeted H-Neuron interventions produce statistically significant causal shifts beyond random baselines. However, our diagnostics also reveal that these underlying neuron sets are not uniquely localized. The identified units are drawn from highly correlated clusters and exhibit only moderate stability across bootstrap samples.

Furthermore, cross-dataset and cross-model analyses reveal a previously undocumented partial-sharing pattern. While no single neuron is universally predictive across all evaluated domains, we identify specific feedforward units most notably the (L16, N4146) that persist across both dataset and model shifts. This suggests the existence of a partially-shared core mechanism coexisting with substantial domain-specific structure.

– Our results illustrate a broader property of sparse-probing based localization: a method can simultaneously produce sparse selections that are predictively useful and causally efective, even if those selections fail stronger criteria for unique localization.

The remainder of this paper is structured as follows: Section 3 formalizes the five-step diagnostic protocol. Section 4 applies the protocol to H-Neurons, detailing detection, causal validation, and overlap diagnostics. Finally, Section 5 discusses the implications of these findings and proposes our protocol as a minimum validation standard for future sparse-neuron interpretability work in LLMs.

## 2 Related Work

## 2.1 Sparse-Neuron Localization in LLMs

Wang et al. [1] introduced skill neurons, units whose activations predict task labels after prompt tuning, and validated their localization claim through crosstask neuron-importance analysis, showing that similar tasks share more skill neurons than dissimilar ones. Chen et al. [2] identified safety neurons via generationtime activation contrasting, reporting that intervention on roughly 5% of neurons restores 90% of safety behavior and that the identified sets emerge stably across random trials. They also measure direct neuron overlap between safety and helpfulness, finding significant overlap with difering activation patterns. Dai et al. [3] proposed knowledge neurons for factual recall, though subsequent work [16] questions whether knowledge-neuron interventions truly localize knowledge or operate through more difuse mechanisms. Gurnee et al. [7] formalized k-sparse linear probing as a methodology for localizing such features, finding apparent monosemanticity for context-level features in middle layers while explicitly cautioning that conclusive proofs of monosemanticity remain methodologically out of reach.

## 2.2 Hallucination in LLMs

Hallucination, the generation of content not supported by context or facts [17,18], has motivated both detection methods and theoretical analyses. Farquhar et al. [19] propose semantic entropy as a black-box uncertainty estimator for confabulations; internal-state methods detect hallucination directly from hidden representations [20,21]. The recent H-Neurons paper [4] extends this lineage by claiming neuron-level localization, and frames the underlying mechanism as a unified over-compliance signal spanning hallucination, sycophancy, and jailbreak susceptibility. Theoretical work by Kalai et al. [22] argues hallucination is an inevitable consequence of next-token prediction under finite data, which H-Neurons cites as motivation for its pre-training origin claim.

## 2.3 Cross-Task and Cross-Domain Neuron Overlap as Validation

Several peer-reviewed works treat direct neuron-set overlap across tasks or domains as essential evidence for localization claims. Wang et al. [1] measure pairwise neuron-importance overlap across nine tasks; Chen et al. [2] measure safety vs helpfulness overlap; Leng and Xiong [25] make cross-task overlap their central methodology, finding that the overlap of task-specific neurons is strongly associated with generalization and specialization across tasks. Similar overlap-based analyses appear in multilingual and cross-domain interpretability work [26]. The original H-Neurons report includes cross-dataset classifier transfer but does not measure neuron-set overlap directly, a gap this work addresses.

## 2.4 Critiques of Probing Methodology

Probing classifiers have a known set of pathologies. Hewitt and Liang [10] show that probes can achieve high accuracy on random control tasks, complicating interpretability claims drawn from probe performance alone. Belinkov [24] surveys advances and shortcomings of probing classifiers. From the statistical learning side, $L _ { 1 }$ regularization is well known to be unstable under feature correlation: Zou and Hastie [9] show that the Lasso [8] arbitrarily selects from groups of correlated features, motivating elastic-net regularization. These results predict that sparse probes on correlated activation spaces should produce non-unique selections, a property whose interpretability consequences have not been systematically diagnosed in recent neuron-localization work. Ferrando et al. [23] use sparse autoencoders rather than probing to identify directions corresponding to entity knowledge in LLMs, and find that mechanisms identified in base models causally afect chat-model behavior. This convergence across methodologies and target behaviors is the empirical foundation for the broader claim that sparse, pre-training-origin functional structures exist in LLMs.

## 3 Method

## 3.1 H-Neuron Identification Pipeline

The original identification pipeline proposed by Gao et al. [4] identifies H-Neurons by relying on token-level labeling to distinguish between faithful and hallucinated model generations. To ensure a more replicable evaluation and isolate high-confidence signals, we adapt their methodology by shifting to a stricter, response-level labeling approach. While this adaptation yields fewer candidate samples, it ensures cleaner, higher-quality data for probe training.

Our adapted identification pipeline consists of the following steps:

– Generation: For each input x and model M, we generate K = 10 independent responses using temperature sampling $( T = 1 . 0 , \mathrm { t o p } – p = 0 . 9$ top-k = 50).

– Judging and Filtering: A rule-based judge evaluates the correctness of each response using normalized substring matching (case-folded and punctuationstripped). We apply an uncertainty filter that judges any response containing explicit refusal as incorrect.

– Response-Level Labeling: We assign a sample the label of faithful if all 10 responses match the gold answer, and hallucinated if none match.

– Pruning: All intermediate cases (i.e., those with 1 to 9 correct responses) are entirely excluded from the training data.

Once the evaluation dataset is curated, we proceed with feature extraction and probe training to isolate the targeted neurons:

– Activation Extraction: For each labeled response, we extract the feedforward neuron activations across all L layers.

– CETT Score Computation: Following Gao et al. [4], we compute the CETT score for each neuron. This score quantifies the neuron’s normalized contribution to its layer’s residual stream, calculated separately for answer tokens and non-answer tokens.

– Feature Aggregation: We aggregate these neuron-level scores into a unified feature vector $x _ { i } \in \mathbb { R } ^ { D }$ . For the gemma- $- 3 - 4 b - \mathrm { i t }$ model, this results in a high-dimensional feature space where $D = L \times d _ { \mathrm { F F N } } = 3 4 8 , 1 6 0$

– Sparse Probing: These feature vectors serve as input to a sparse logistic regression classifier, trained with an $L _ { 1 }$ penalty (using $C = 1 . 0$ and the liblinear solver).

H-Neuron Selection: Finally, any features that retain non-zero coeficients after the regression fit are considered as probable H-Neurons.

## 3.2 The Five-Step Diagnostic Protocol

We define five diagnostics that collectively test whether a sparse set of identified neurons constitutes a unique, stable, and causally privileged functional unit. Each diagnostic targets a specific failure mode of sparse-probing based localization, and each is computationally cheap, requiring at most one additional probe-fitting step beyond the standard identification pipeline.

D1. Feature Correlation Analysis. For each identified neuron $n _ { i } \in S$ we compute $\rho _ { i } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j \not \in S } \left| \rho ( n _ { i } , n _ { j } ) \right|$ across the training activation matrix and report the proportion with $\rho ^ { \mathrm { m a x } } > 0 . 7$ . A localization concern is flagged when more than approximately 30% of neurons have a high-correlation partner outside $S ,$ since $L _ { 1 }$ selection from correlated clusters can produce predictive but nonunique selections [9].

D2. Bootstrap Stability. We fit the identification procedure $B \geq 5$ times on bootstrap-resampled training sets and compute mean pairwise Jaccard similarity $J ( S _ { i } , S _ { j } ) = | S _ { i } \cap S _ { j } | / | S _ { i } \cup S _ { j } |$ . A localization concern is flagged when mean Jaccard falls below 0.7, indicating that the identification procedure is sensitive to data composition rather than tracking a stable underlying signal.

D3. $L _ { 1 }$ vs $L _ { 2 }$ Ranking Disagreement. We refit the probe with $L _ { 2 }$ regularization (which distributes weight across correlated features), rank features by absolute weight magnitude, take the top-|S|, and compute overlap with the $L _ { 1 ^ { - } }$ selected set. A localization concern is flagged when overlap falls below $5 0 \%$ , since high disagreement between sparse and dense selection is a canonical signature of feature collinearity [8,9].

D4. Control Neuron Intervention. We compare intervention efects against random neurons drawn from the same layers (matched count per layer). Causal efect is measured via activation scaling: for scaling factor $\alpha \in \{ 0 , 1 , 2 \}$ ， we multiply the down\_proj input activations of the target neurons by α and measure judge accuracy. Statistical significance is assessed via McNemar’s test comparing α = 0 (suppression) and α = 1 (baseline). A stronger cluster-based alternative, sampling from features highly correlated with any identified neuron, remains proposed for future validation.

D5. Cross-Dataset and Cross-Model Overlap. We run the identification procedure independently on multiple datasets and models. Pairwise Jaccard similarity is computed and significance assessed via the hypergeometric test against the null of random selection from the full feature pool. A localization concern is flagged when mean Jaccard is at or near zero, indicating dataset or model specific predictive signals rather than a shared mechanism.

## 3.3 Experimental Setup

Models. We evaluate two instruction-tuned Gemma family checkpoints, google/gemma-3-4b-it and google/medgemma-4b-it [12,27]. Both models have L = 34 transformer layers and d = 10,240, yielding D = 348,160 feedforward features. We refer to these models as Gemma and MedGemma, respectively, throughout the paper.

Datasets. We use three publicly available open-ended question-answering datasets spanning diferent knowledge domains: TriviaQA [13] (general trivia, 2,000 samples), BioASQ [14] (biomedical, 2,000 samples), and NQ-Open [15] (open-domain web questions, 2,000 samples).

Hyperparameters. Identification uses $L _ { 1 }$ logistic regression with C = 1.0 and the liblinear solver. All features (no variance pre-filtering; top-k = 0) are passed to the probe. Causal validation uses n = 500 held-out samples per condition with five random seeds. Bootstrap diagnostics use B = 5 resamples.

Diagnostic Validation. For the TriviaQA condition, diagnostics (D1–D3) and cross-dataset overlap were evaluated on an independent replicate of the Gemma 3 4B experiment (10 H-Neurons, AUROC 0.820, gap +0.404) to verify that diagnostic findings were not specific to a single experimental run.

Reproducibility. All experimental code, neuron indices, and aggregated results will be released upon acceptance.

## 4 Results

## 4.1 Detection

Table 1 reports detection performance across six model-dataset combinations. The original H-Neurons report [4] finds accuracy gaps of approximately +0.149 (TriviaQA), +0.150 (BioASQ), and +0.110 (NQ-Open) on a diferent model family. Our AUROC gaps replicate the qualitative pattern (gap > 0 across all conditions) and are strongest on BioASQ.

Detection holds across two distinct model checkpoints and three domain settings, with consistently stronger performance on BioASQ than on more openended QA tasks. The weaker detection on NQ-Open (gap +0.128) likely reflects greater inherent variance in open-domain QA responses and a sparser supervisable signal.

Table 1. Detection results across model-dataset combinations. HN = number of H-Neurons selected by $L _ { 1 }$ (post-fit non-zero coeficients). AUROC is reported on held-out evaluation samples; Gap is the diference versus the majority-class baseline.
<table><tr><td>Model</td><td>Dataset HN HN</td><td></td><td> $( \text{‰}$ </td><td>AUROC</td><td>Gap</td><td>Bal Acc</td></tr><tr><td>Gemma 3 4B</td><td>TriviaQA</td><td>9</td><td>0.026</td><td>0.811</td><td>+0.311</td><td>0.712</td></tr><tr><td>Gemma 3 4B</td><td>BioASQ</td><td>8</td><td>0.023</td><td>0.974</td><td>+0.474</td><td>0.916</td></tr><tr><td>Gemma 3 4B</td><td>NQ-Open</td><td>4</td><td>0.011</td><td>0.628</td><td>+0.128</td><td>0.629</td></tr><tr><td>MedGemma 4B TriviaQA</td><td></td><td>4</td><td>0.011</td><td>0.735</td><td>+0.235</td><td>0.658</td></tr><tr><td>MedGemma 4B BioASQ</td><td></td><td>7</td><td>0.020</td><td>0.955</td><td>+0.455</td><td>0.885</td></tr><tr><td>MedGemma 4B NQ-Open</td><td></td><td>2</td><td>0.006</td><td>0.612</td><td>+0.112</td><td>0.532</td></tr></table>

## 4.2 Causal Validation

Table 2 reports activation-scaling intervention results for both models. We compute judge accuracy under suppression $( \alpha = 0 )$ , baseline $( \alpha = 1 )$ , and amplification $\left( \alpha = 2 \right)$ of H-Neuron down\_proj weights, with five random seeds and $n = 5 0 0$ per condition. Statistical significance is assessed via McNemar’s test comparing $\alpha = 0$ vs $\alpha = 1$

Table 2. Causal validation via activation scaling. Suppression of H-Neurons produces statistically significant judge accuracy shifts on both models tested, with efects not reproduced by random same-layer baselines. $p \textmd { - }$ values from McNemar’s test (paired binary outcomes).
<table><tr><td>Model</td><td>Dataset HN Suppress</td><td></td><td> $( \alpha = 0 )$ </td><td>Baseline  $( \alpha = 1 )$ </td><td>Amplify  $( \alpha = 2 )$ </td><td>McNemar p Random baseline</td><td></td></tr><tr><td>Gemma 3 4B</td><td>TriviaQA 9</td><td></td><td> $0 . 4 7 2 \pm 0 . 0 0 4$ </td><td> $0 . 4 9 6 \pm 0 . 0 0 3$ </td><td> $0 . 4 9 0 \pm 0 . 0 0 3$ </td><td>&lt; 0.001</td><td>0.482</td></tr><tr><td>MedGemma 4B BioASQ</td><td></td><td>7</td><td> $0 . 4 7 7 \pm 0 . 0 0 8$ </td><td> $0 . 4 8 6 \pm 0 . 0 0 9$ </td><td> $0 . 4 6 7 \pm 0 . 0 0 9$ </td><td>0.026</td><td>0.504</td></tr></table>

The McNemar test rejects the null of equal performance between suppression and baseline at $p < 0 . 0 5$ on both conditions. We interpret this as confirming the original H-Neurons paper’s causal-control claim under our stricter evaluation: H-Neuron targeted suppression produces a measurable, statistically significant decrease in judge accuracy that random same-layer baselines do not reproduce.

Efect magnitudes are modest (approximately 2.4 percentage points on TriviaQA, 0.9 on BioASQ) and substantially smaller than the original paper’s reported behavioral efects under amplification. Two factors plausibly contribute: stricter response-level rather than token-level labeling, which produces fewer training samples; and the relatively small number of selected H-Neurons (7–9) in our intervention. The qualitative finding of significant causal efect beyond random baseline is nevertheless robust.

## 4.3 Within-Dataset Diagnostics (D1–D3)

Table 3 reports diagnostic results on Gemma 3 4B across all three datasets. The feature correlation analysis (D1) reveals that 19 of 22 H-Neurons across the three

Gemma evaluations exhibit Pearson $| r | > 0 . 7$ with other features in the training matrix, with maximum correlations ranging 0.613–0.973. This indicates that the $L _ { 1 }$ procedure overwhelmingly selects from correlated neuron clusters rather than from uniquely informative units.

Table 3. Within-dataset diagnostics (D1–D3) for Gemma 3 4B. Correlation = proportion of H-Neurons with $| r | > 0 . 7$ to any non-selected feature. Bootstrap Jaccard = mean pairwise Jaccard across 5 random-seed bootstrap fits. $L _ { 1 } \cap L _ { 2 }$ overlap = fraction of $L _ { 1 }$ -selected H-Neurons appearing in the top-|S| $L _ { 2 } .$ -ranked features.
<table><tr><td></td><td></td><td>Dataset HN D1: Correlation</td><td>D2: Bootstrap Jaccard</td><td>D3:  $L _ { 1 } \cap L _ { 2 }$  overlap</td></tr><tr><td>TriviaQA 10</td><td></td><td>7/10 (70%)</td><td>0.47</td><td>0.10</td></tr><tr><td>BioASQ</td><td>8</td><td>8/8 (100%)</td><td>0.69</td><td>0.25</td></tr><tr><td>NQ-Open 4</td><td></td><td>4/4 (100%)</td><td>0.45</td><td>0.00</td></tr></table>

Bootstrap stability (D2) is moderate at best, with mean Jaccard across resamples of 0.45–0.69, below the suggested 0.7 threshold for stable localization. BioASQ exhibits the highest stability, consistent with its stronger detection signal. $L _ { 1 }$ versus $L _ { 2 }$ ranking disagreement (D3) is severe across all three datasets: only 0–25% of $L _ { 1 }$ -selected H-Neurons appear among the top-ranked features under $L _ { 2 }$ regularization. This pattern is the canonical signature of $L _ { 1 }$ selection from correlated clusters [9].

The combined results from D1–D3 indicate that the identified H-Neurons should not be interpreted as a unique localization of the hallucination signal. They are sparse, predictive, and partially causally efective, but they are not uniquely necessary, since many highly correlated alternatives exist in the feedforward representation. This needs to be investigated in the future work.

## 4.4 Cross-Dataset and Cross-Model Overlap (D5)

Table 4 reports cross-dataset overlap of H-Neurons within Gemma 3 4B. Crossmodel analysis between Gemma 3 4B (TriviaQA) and MedGemma 4B (BioASQ) yields 3 shared neurons (Jaccard 0.231, $p = 4 . 2 \times 1 0 ^ { - 1 3 } )$ . Statistical significance is assessed via the hypergeometric test against the null of random selection from the full feature pool $( D = 3 4 8 , 1 6 0 )$

## 4.5 Cross-Dataset Classifier Transfer

To evaluate the generalization of the identified hallucination signal, we measure zero-shot cross-dataset transfer across all three QA datasets on Gemma 3 4B.

Table 5 reports detection scores alongside all six cross-dataset transfer directions. All transfers yield positive AUROC gaps, confirming that sparse probes capture generalizable structure. The TriviaQA probe achieves the strongest outbound transfer (BioASQ 0.868, NQ-Open 0.721), exceeding its own in-domain performance on BioASQ. Conversely, the BioASQ probe has the highest indomain detection (AUROC 0.969) but the weakest outbound transfers (AU-ROC 0.626 and 0.646), while the NQ-Open probe, the weakest detector (AUROC 0.702), transfers to BioASQ at 0.809. Detection rows are from independent runs; values difer slightly from Table 1 due to run-to-run variation in $L _ { 1 }$ selection.

Table 4. Cross-dataset H-Neuron overlap on Gemma 3 4B. Shared = number of neurons in both identified sets; Jaccard = $| A \cap B | / | A \cup B | ; p$ = hypergeometric significance against the null of random selection.
<table><tr><td>Comparison</td><td colspan="4">Shared Jaccard Hypergeometric p Interpretation</td></tr><tr><td>TQA ∩ BioASQ</td><td>4</td><td>0.286</td><td> $2 . 4 \times 1 0 ^ { - 1 7 }$ </td><td>Significant overlap</td></tr><tr><td>TQA ∩ NQ-Open</td><td>1</td><td>0.077</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td><td>Significant overlap</td></tr><tr><td>BioASQ ∩ NQ-Open</td><td>0</td><td>0.000</td><td></td><td>No overlap</td></tr><tr><td>All three (∩)</td><td>0</td><td></td><td></td><td>No universal H-Neuron</td></tr></table>

Table 5. Bidirectional cross-dataset classifier transfer on Gemma 3 4B. Detection rows report in-domain probe performance; Transfer rows report zero-shot evaluation on other datasets. Random baseline = 0.500.
<table><tr><td>Source → Target AUROC Gap Bal Acc</td></tr><tr><td></td></tr><tr><td>TriviaQA → TriviaQA (detection) 0.811 +0.311 0.712 TriviaQA → BioASQ 0.868 +0.368 0.781</td></tr><tr><td>TriviaQA → NQ-Open 0.721 +0.221 0.653</td></tr><tr><td>BioASQ → BioASQ (detection) 0.969 +0.469 0.913</td></tr><tr><td>BioASQ → TriviaQA 0.626 +0.126 0.502</td></tr><tr><td>BioASQ → NQ-Open 0.646 +0.146 0.500</td></tr><tr><td>NQ-Open → NQ-Open (detection) 0.702 +0.202 0.676</td></tr><tr><td>NQ-Open → TriviaQA 0.684 +0.184 0.622</td></tr><tr><td> $\mathrm { N Q - O p e n }  \mathrm { B i o A S Q }$  0.809 +0.309 0.540</td></tr></table>

These transfer results extend the neuron overlap findings. While cross-dataset neuron-set overlap is partial (Table 4), sparse probes consistently transfer positively across all dataset pairs, indicating that each $L _ { 1 }$ -selected neuron set captures a generalizable hallucination subspace. A probe can efectively detect hallucinations in new domains even when an independent $L _ { 1 }$ fit on that domain selects a diferent, non-overlapping set of H-Neurons.

## 4.6 Cross-Model Evaluation

Table 6 compares Gemma 3 4B and MedGemma 4B on the same three datasets. Across all matched settings, Gemma 3 4B outperforms MedGemma 4B on AU-ROC, AUROC gap, and balanced accuracy, while also selecting slightly more H-Neurons. The largest performance diference appears on TriviaQA, whereas BioASQ remains the strongest condition for both models.

Table 6. Cross-model detection comparison on matched datasets. ∆ denotes Gemma 3 4B minus MedGemma 4B.
<table><tr><td colspan="3">Dataset ∆ HN ∆ AUROC ∆ Gap ∆ Bal Acc</td></tr><tr><td>TriviaQA +5 +0.076</td><td>+0.076</td><td>+0.054</td></tr><tr><td>BioASQ +1</td><td>+0.019 +0.019</td><td>+0.031</td></tr><tr><td>NQ-Open +2</td><td>+0.016 +0.016</td><td>+0.097</td></tr></table>

The qualitative pattern is consistent across models. BioASQ yields the strongest detection performance, NQ-Open yields the weakest, and TriviaQA lies in between. This suggests that the main domain level trend is stable across the two checkpoints, although Gemma 3 4B produces stronger separability than MedGemma 4B in every matched comparison.

The cross-dataset and cross-model analyses reveal a partial sharing pattern not previously documented. TriviaQA and BioASQ share four H-Neurons within Gemma 3 4B, far more than expected under random selection (hypergeometric $p = 2 . 4 \times 1 0 ^ { - 1 7 } )$ . Three H-Neurons are shared between Gemma 3 4B (TriviaQA) and MedGemma 4B (BioASQ), spanning both a model fine-tuning shift and a dataset shift. No H-Neuron is universal across all three Gemma-QA datasets, and BioASQ and NQ-Open share zero neurons despite both probing factual recall.

This pattern is consistent with partial domain specificity in the sparse probe selections: a small number of feedforward units recur across closely related domains and even across model fine-tuning, while the majority of identified H-Neurons appear to be dataset-specific.

## 4.7 Candidate Cross-Domain Neurons

Within the cross-dataset and cross-model intersections, three neurons appear in multiple overlap analyses. The strongest candidate is (L16, N4146), which appears in both the Gemma 3 4B TriviaQA ∩ BioASQ intersection (cross-dataset, same model) and the Gemma 3 4B TriviaQA ∩ MedGemma 4B BioASQ intersection (cross-model and cross-dataset). A secondary candidate, (L26, N3593), appears in Gemma TriviaQA ∩ NQ-Open and again in Gemma TriviaQA ∩ MedGemma BioASQ. A third unit, (L29, N5754), appears only in the crossmodel overlap.

We do not claim these neurons uniquely encode a hallucination mechanism. The D1–D3 diagnostics show that $L _ { 1 }$ selection is from correlated clusters, so the specific named neurons may be cluster representatives rather than the only causally privileged units. Their persistence across distinct datasets and model checkpoints does, nonetheless, suggest they index a hallucination-relevant subspace that is more stable than typical H-Neuron selections. We propose them as falsifiable targets for follow-up causal work, including single-neuron ablation studies and SAE decomposition of the surrounding cluster.

## 5 Discussion

## 5.1 What the Diagnostic Reveals About H-Neurons

Our results refine the H-Neurons claim along three dimensions. Detection clearly holds: sparse $L _ { 1 }$ probing on CETT features identifies highly predictive neuron sets, with AUROC gaps that exceed the original report’s quantitative claims across multiple models and datasets. Causal control also holds at the population level: H-Neuron suppression produces statistically significant accuracy shifts beyond random baselines, with efects robust to $n = 5 0 0$ evaluation samples and five seeds. Localization in the strong sense, that these specific neurons uniquely encode the behavior, is not supported. The D1–D3 diagnostics indicate that $L _ { 1 }$ selection draws representatives from correlated clusters, and D5 shows that the selections themselves are partially domain-specific, with no universal neurons across all three QA datasets.

The most natural reconciliation is that the H-Neurons procedure identifies some subspace of the feedforward representation that carries hallucinationrelevant signal, but the specific neurons returned by a single $L _ { 1 }$ fit are not the only such units. They are a sparse projection of a larger correlated structure. This is consistent with known properties of representation in transformer feedforward layers, including feature superposition [11] and the layer-wise distribution of context features [7].

## 5.2 Generalizable Methodological Takeaways

Three implications for sparse-neuron interpretability research follow from our results, beyond the specific H-Neurons case.

Detection and localization should be evaluated separately. A method can produce sparse selections that are highly predictive yet non-unique. Conflating predictive sparsity with mechanistic localization risks claiming more than the evidence supports.

$L _ { 1 } .$ -based identification on correlated activation spaces should be accompanied by collinearity diagnostics by default. The statistical properties of $L _ { 1 }$ regularization under correlated features are well established [8,9] and have predictable consequences for interpretability claims. D1 and D3 are direct adaptations of standard practice from statistical learning to interpretability evaluation.

Cross-dataset and cross-model overlap analyses are the most direct test of mechanism-sharing claims. Classifier-level transfer is necessary but not suficient:

a probe trained on one dataset can transfer to another because of distributed signal in the representation, even if the specific neurons identified difer. Direct neuron-set overlap measures the stronger property that interpretability claims usually require.

## 5.3 Limitations

Several limitations qualify our findings. We evaluate only two models, both in the Gemma 3 4B family; claims about cross-architecture generality require evaluation on Llama, Qwen, or Mistral variants. Our causal-validation efect sizes are smaller than the original H-Neurons report, plausibly due to our stricter response-level labeling and our smaller H-Neuron set sizes. We did not perform single-neuron ablation on the candidate units (L16, N4146) and (L26, N3593), which is the most natural follow-up. The diagnostic thresholds in our protocol $( | r | > 0 . 7 $ Jaccard > 0.7, overlap > 0.5) are empirical guidelines calibrated to current methodology rather than principled bounds; we recommend the field develop these thresholds via systematic study. We do not test capability preservation (for example, MMLU performance after intervention), so our causal claims do not rule out general-capability degradation as an alternative explanation. We flag this as the most important follow-up for any future intervention-based use of identified H-Neurons.

## 6 Conclusion

Sparse-probing based localization claims in mechanistic interpretability are becoming increasingly common, yet they are often evaluated primarily through predictive performance rather than through tests of stability, uniqueness, and generality. In this work, we proposed a five-step diagnostic protocol for assessing such claims and applied it to a representative case study: H-Neurons for hallucination in large language models.

Our results refine the original claim rather than reject it. We find that sparse probing over CETT features reliably identifies neuron sets with strong detection performance, and that targeted interventions on these sets produce statistically significant causal efects beyond random same-layer baselines. At the same time, the identified neuron sets are not uniquely localized: they are embedded in highly correlated feature clusters, show only moderate bootstrap stability, and exhibit substantial disagreement between sparse and dense probe rankings.

Cross-dataset and cross-model analyses further show that hallucination-relevant structure is only partially shared. While no single neuron is universal across all evaluated settings, a small number of units, most notably (L16, N4146), recur across both dataset and model shifts, suggesting a partially shared core mechanism alongside substantial domain-specific structure.

Taken together, our findings support a more careful interpretation of sparseneuron claims: predictive sparsity and population-level causal efect do not by themselves establish unique mechanistic localization. We therefore argue that diagnostic validation should become a routine part of sparse-neuron interpretability research, and that future work should treat detection and localization as distinct evaluation targets when developing transparent and reliable accounts of internal model behavior.

## Data and Code Availability

To facilitate reproducibility, all experimental code, extracted features, and resulting sparse probes are publicly available. The curated evaluation datasets, neuron indices for both models, and instructions for replicating the diagnostic protocol can be accessed at https://github.com/huseyincavusbi/hprobes-protocol.

## References

1. Wang, X., Wen, K., Zhang, Z., Hou, L., Liu, Z., Li, J.: Finding Skill Neurons in Pre-trained Transformer-based Language Models. In: EMNLP (2022)

2. Chen, J., Wang, X., Yao, Z., Bai, Y., Hou, L., Li, J.: Finding Safety Neurons in Large Language Models. arXiv:2406.14144 (2024)

3. Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., Wei, F.: Knowledge Neurons in Pretrained Transformers. In: ACL (2022)

4. Gao, X., et al.: H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs. arXiv:2512.01797 (2025)

5. Doshi-Velez, F., Kim, B.: Towards a Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608 (2017)

6. Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., Carter, S.: Zoom In: An Introduction to Circuits. Distill (2020)

7. Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., Bertsimas, D.: Finding Neurons in a Haystack: Case Studies with Sparse Probing. TMLR (2023)

8. Tibshirani, R.: Regression Shrinkage and Selection via the Lasso. JRSS-B 58(1), 267–288 (1996)

9. Zou, H., Hastie, T.: Regularization and Variable Selection via the Elastic Net. JRSS-B 67(2), 301–320 (2005)

10. Hewitt, J., Liang, P.: Designing and Interpreting Probes with Control Tasks. In: EMNLP (2019)

11. Elhage, N., et al.: Toy Models of Superposition. Transformer Circuits Thread (2022)

12. Gemma Team: Gemma 3 Technical Report. Google DeepMind (2025)

13. Joshi, M., Choi, E., Weld, D.S., Zettlemoyer, L.: TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In: ACL (2017)

14. Tsatsaronis, G., et al.: An Overview of the BIOASQ Large-scale Biomedical Semantic Indexing and Question Answering Competition. BMC Bioinformatics 16, 138 (2015)

15. Kwiatkowski, T., et al.: Natural Questions: A Benchmark for Question Answering Research. TACL 7, 453–466 (2019)

16. Niu, J., Liu, A., Zhu, Z., Penn, G.: What does the Knowledge Neuron Thesis Have to do with Knowledge? In: ICLR (2024)

17. Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On Faithfulness and Factuality in Abstractive Summarization. In: ACL (2020)

18. Ji, Z., et al.: Survey of Hallucination in Natural Language Generation. ACM Computing Surveys 55(12), 1–38 (2023)

19. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y.: Detecting Hallucinations in Large Language Models Using Semantic Entropy. Nature 630, 625–630 (2024)

20. Ji, Z., Yu, T., Xu, Y., Lee, N., Ishii, E., Fung, P.: LLM Internal States Reveal Hallucination Risk Faced with a Query. In: BlackboxNLP (2024)

21. Zhang, et al.: ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs. In: ACL (2025)

22. Kalai, A.T., Nachum, O., Vempala, S.S., Zhang, E.: Why Language Models Hallucinate. arXiv (2025)

23. Ferrando, J., Obeso, O., Rajamanoharan, S., Nanda, N.: Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models. In: ICLR (2025)

24. Belinkov, Y.: Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics 48(1), 207–219 (2022)

25. Leng, Y., Xiong, D.: Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons. arXiv:2407.06488 (2024)

26. Stańczak, K., et al.: Same Neurons, Diferent Languages: Probing Morphosyntax in Multilingual Pre-trained Models. In: NAACL (2022)

27. MedGemma Team: MedGemma Technical Report. Google DeepMind (2025)