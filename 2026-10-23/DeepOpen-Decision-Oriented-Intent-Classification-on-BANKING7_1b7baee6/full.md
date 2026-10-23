# DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150

Shuang Song

Zhongqiao Yuan

Shenzhen Depei Open Source Data Technology Co., Ltd. Shenzhen, China

## Abstract

We present DeepOpen, a decision-Oriented intent-classification framework with datasetspecific systems for BANKING77 and CLINC150. On BANKING77, a single English classifier continued with sharpness-aware minimization (SAM) achieves 94.1234% test accuracy (2,899/3,080). This is the best observed result among three fixed SAM seeds, selected for display after test evaluation; their mean is 94.0368%. On the 150 in-scope CLINC150 classes, probability fusion of one Laya specialist and three DeBERTa-v3-large classifiers achieves 98.0222% (4,411/4,500). We describe the two configurations, reproduce the reference comparisons displayed in the accompanying evaluation explorer, and separate headline scores from evidence about improvement. Under a uniform BANKING77 evaluation path, the SAM mean exceeds the historical checkpoint by 0.0433 percentage points, with a shared-row 95% interval of [-0.2273, +0.3247]. Thus, the continuation does not establish a reliable accuracy gain. The report documents strong observed task-specific performance, selection efects, and unresolved historical execution-path diferences, rather than a new backbone or a unified algorithmic advantage.

## 1 Introduction

Intent classification requires mapping an utterance to a predefined outcome rather than generating an open-ended response. This distinction is particularly relevant to fine-grained benchmarks such as BANKING77 and CLINC150, which contain 77 banking intents and 150 in-scope intents, respectively [2, 3]. When classification is implemented through autoregressive text generation, the generated answer must be mapped back to an admissible label. A direct decision model instead produces a distribution over the permitted classes without decoding a textual answer. This motivates a prediction interface aligned with the task’s finite output space, although the formulation alone does not guarantee better accuracy or lower computational cost than alternative implementations.

Laya provides a non-generative decision-model foundation [1], but adapting its flexible candidate interface to fixed-label benchmarks introduces an input-representation challenge. The original interface serializes questions, candidate descriptions, and utterance text into a shared encoder sequence, allowing the candidate inventory to compete with the utterance for the available token budget. Our archived CLINC150 audit identified incomplete candidate coverage under the evaluated input configuration. To address this issue, DeepOpen retains the Laya encoder while replacing the flexible candidate-scoring interface with task-specific classification heads. The encoder processes the utterance without repeatedly serializing the entire label inventory, and the output head represents every class explicitly. This adaptation trades arbitrary candidate flexibility for a fixed-label decision interface; it does not introduce a new backbone or preserve all capabilities of the original typed-decision model.

Building on this adaptation, we investigate two complementary routes to task-specific performance. For BANKING77, we apply sharpness-aware minimization (SAM) to continue training a single classifier, testing whether perturbation-aware optimization improves the existing decision boundary without increasing inference-time parameter count [5]. For CLINC150, we combine an R-Drop-trained Laya specialist with three DeBERTa-v3-large classifiers through probability fusion, testing whether their predictions provide complementary information [4, 6]. These interventions address diferent questions: SAM modifies optimization, whereas fusion combines multiple predictors at inference. We therefore evaluate them as separate system configurations and distinguish their efects from the fixed-label adaptation that resolves candidate-coverage constraints.

The resulting systems achieve observed test accuracies of 94.1234% on BANKING77 and 98.0222% on in-scope CLINC150 [7]. The BANKING77 result is the best observed among three SAM seeds and uses approximately 396M parameters, compared with the nominal 7B Llama 2 reference; the CLINC150 result uses a four-encoder ensemble totaling approximately 1.697B parameters [9, 11]. Our contributions are the documented adaptation of a flexible decision-model interface to fixed-label intent classification, the evaluation of task-specific optimization and fusion strategies, and an analysis connecting accuracy with parameter footprint and reproducibility. The experiments distinguish headline performance from evidence of improvement: the BANK-ING77 mean gain remains statistically inconclusive, while the CLINC150 fusion results must be interpreted alongside their additional inference resources.

## 2 Methodology

Figure 1 presents the overall design of DeepOpen. Panel A shows the adaptation from Laya’s flexible candidate interface to fixed-label intent classification. Panels B and C instantiate this design with separately trained systems: a single classifier continued with sharpness-aware minimization (SAM) for BANKING77, and probability fusion of four classifiers for CLINC150. SAM and R-Drop afect training, whereas the arrows in Panels B and C describe inference.

![](images/e99cda7df7a1947dc1c7cb6b840a9c5bc56f5ce5c3abc30a58a33b659a05984b.jpg)  
Figure 1: Overview of deepopen. (A) Fixed-label adaptation removes candidate descriptions from the input. The two tasks use separately trained models: (B) a single SAM-trained model for BANKING77 and (C) weighted probability fusion for CLINC150. SAM and R-Drop are used only during training.

## 2.1 SAM-Based Optimization for BANKING77

The BANKING77 system uses a Laya-based English encoder and a 77-class classification head. Its starting checkpoint was trained on all 10,003 oficial training examples, including a subset previously used for internal development. That subset therefore cannot serve as independent validation for this checkpoint.

All continuation candidates start from the same historical full-training weights. Three fixed seeds (13, 42, 87) are run for the control and SAM candidates. The recorded protocol uses two epochs, 626 updates, 10,003 training rows, encoder learning rate 2e-6, head learning rate 2e-5, and BF16 training. SAM uses rho = 0.05 [5]. Fixed final checkpoints are evaluated without validation-based checkpoint selection. Selecting seed 13 as the headline is a separate, post-test display choice.

Evaluation uses eager FP32, scaled dot-product attention (SDPA), batch size 32, and a maximum length of 128 tokens for every continuation checkpoint and the historical reference. The headline model performs inference with one encoder; no seed ensemble is used.

## 2.2 Probability Fusion for CLINC150

The CLINC150 system combines a Laya-based English specialist with three DeBERTa-v3-large classifiers [6]. The specialist uses masked mean pooling and a 150-class linear head; the historical selected configuration uses R-Drop training [4] and seed 42. The DeBERTa members use seeds 13, 42, and 87. Each member uses its own tokenization, and probabilities must share the same 150-class label order before fusion.

Let pL denote the specialist probabilities and p13, p42, and p87 the DeBERTa probabilities. The final distribution is:

$$
\begin{array} { r } { p _ { \mathrm { e n s e m b l e } } = \frac { 1 } { 2 } p _ { L } + \frac { 1 } { 6 } ( p _ { 1 3 } + p _ { 4 2 } + p _ { 8 7 } ) . } \end{array}\tag{1}
$$

The predicted intent is argmax(pEnsemble). All four encoders are required at inference. The archived hybrid protocol evaluated Laya weights in {0, 0.25, 0.5, 0.75, 1}, using validation accuracy and then NLL for selection. The final system uses weight 0.5 and equal weighting across the three DeBERTa members; selection was frozen before this evaluation. Earlier rounds had already exposed benchmark test summaries, so the overall development history is not a fresh blind evaluation [11].

## 2.3 Relationship to Existing Methods

Laya supplies the starting English model [1]. R-Drop [4], SAM [5], and DeBERTa-v3 [6] are established components, not methods introduced here. The paper reports their task-specific use and observed system behavior. It does not isolate the contribution of each component through a common cross-dataset ablation.

## 3 Experiment

## 3.1 Dataset for Intent Classification

Table 1: Dataset scope and headline aggregation.
<table><tr><td>Dataset</td><td>Training / evaluation scope</td><td>Test size</td><td>Headline aggregation</td></tr><tr><td>BANKING77</td><td>10,003 full-training examples; 77 intents</td><td>3,080</td><td>SAM seed 13; best observed of three</td></tr><tr><td>CLINC150</td><td>150 in-scope intents; original train/dev/test split</td><td>4,500</td><td>Four-encoder probability fusion</td></tr></table>

Accuracy is correct predictions divided by test examples. BANKING77 continuation uncertainty uses 10,000 shared-row bootstrap resamples of fixed checkpoints. These intervals exclude model-selection efects and uncertainty over the broader training-seed population. They do not turn the observed tests into a new independent confirmation set. The public protocol records no subsequent tuning following these tests.

## 3.2 Benchmark Results

Table 2: Headline results, aligned with the evaluation explorer retrieved on 21 October 2026 [7].
<table><tr><td>System</td><td>Correct / test</td><td>Accuracy (%) Interpretation</td><td></td></tr><tr><td>BANKING77 SAM seed 13</td><td>2,899 / 3,080</td><td>94.1234</td><td>Best observed SAM seed</td></tr><tr><td>CLINC150 four-encoder ensemble</td><td>4,411 / 4,500</td><td>98.0222</td><td>In-scope ensemble result</td></tr></table>

The BANKING77 headline is one correct prediction above the archived 94.0909% result, but the execution settings difer; Section 5 gives the matched-path continuation analysis. The CLINC150 archive also includes specialist and DeBERTa-only ensemble comparisons, which are reported with their resource costs in Section 4.2.

## 3.3 Separate Benchmark and Model Size Comparisons

Accuracies below retain the selected evaluation-explorer values [7]. Model sizes include embeddings and all ensemble members where verified. M denotes one million parameters. NV means the size of the exact displayed result configuration was not verified; it does not mean zero. These contextual comparisons are not reruns under one shared data, selection, or compute protocol.

Table 3: BANKING77 test accuracy comparison.
<table><tr><td>Method</td><td>Accuracy (%)</td><td>Inference configuration</td></tr><tr><td>SPACE (SPACE-3)</td><td>94.94</td><td>Single Transformer + classification head</td></tr><tr><td>SPACE 2.0 (SPACE-2)</td><td>94.77</td><td>Single encoder + classification head</td></tr><tr><td>Llama 2 7B (FT)</td><td>94.35</td><td>Single decoder-only model</td></tr><tr><td>ConvFiT (RoBERTa-base)</td><td>94.16</td><td>RoBERTa-base + cosine 1-NN</td></tr><tr><td>DeepOpen (SAM)</td><td>94.1234</td><td>Single encoder + classification head</td></tr><tr><td>PPTOD-base</td><td>93.86</td><td>T5-base; intent-label generation</td></tr></table>

The BANKING77 classifier uses approximately 17.7 times fewer parameters than the nominal 7B reference, with accuracy 0.2266 percentage points lower. This supports a parameter-footprint comparison for that reference pair, not an equal-compute or latency claim.

The archived member counts are 394,935,446 for the Laya specialist and 434,165,910 for each DeBERTa classifier, totaling 1,697,433,176. The DeBERTa-only comparison reuses the same checkpoints selected within the hybrid validation procedure; it is not a separately optimized DeBERTa-only baseline. Adding Laya increases accuracy by 0.5778 percentage points over that component ensemble, while the final system exceeds the specialist by 0.2667 points (twelve net correct predictions).

Table 4: CLINC150 in-scope test accuracy comparison.
<table><tr><td>Method</td><td>Accuracy (%)</td><td>Inference configuration</td></tr><tr><td>TOATOD-small</td><td>98.45</td><td>T5-small + task adapter</td></tr><tr><td>DeepOpen (ensemble)</td><td>98.0222</td><td>Laya + 3 DeBERTa-v3-large models</td></tr><tr><td>TOATOD-base</td><td>98.01</td><td>T5-base + task adapter</td></tr><tr><td>SPACE (SPACE-3)</td><td>97.89</td><td>Single Transformer + classification head</td></tr><tr><td>SPACE 2.0 (SPACE-2)</td><td>97.80</td><td>Single encoder + classification head</td></tr></table>

Notes. Results cover the 150 in-scope classes only. The DeepOpen ensemble combines four encoders through probability fusion. Boldface identifies our system, not the best result.

The hybrid report records conditional paired 95% bootstrap intervals of [0.2444, 0.9111] points against the DeBERTa-only ensemble and [0.0667, 0.4667] against the Laya specialist. These intervals condition on the selected weights and do not cover repeated research selection or the distribution of possible training runs. This revision carries forward those archived analyses; it does not present them as new independent experiments.

The deployed four-encoder weights total 6,789,898,104 bytes in the archived package. On an NVIDIA L20, the archived batch-1, padding-64, BF16 measurement with 20 warmup passes and 100 measured passes reports p50 49.411 ms and p95 49.632 ms for resident forward computation and probability fusion. Loading, tokenization, and network time are excluded. The specialist has an historical p50 of 10.104 ms under a similar protocol, but this is not a same-process paired speed test.

## 3.4 Parameter Accounting and Interpretation

The BANKING77 total is derived, not recounted from a newly loaded checkpoint. The shared encoder count is 394,935,446 minus the CLINC150 linear head (1024 + 1) x 150, giving 394,781,696. The archived BANKING77 configuration and the Transformers implementation [11] add a biasfree 1024 x 1024 dense layer, 1024 LayerNorm weights, and a (1024 + 1) x 77 output head: 395,910,221 parameters. SAM adds no inference parameters. The evidence and calculation are saved in the companion parameter-accounting file.

Parameter count is a model-footprint measure, not a substitute for latency, peak memory, FLOPs, or energy. Sequence length, precision, batching, software kernels, and the number of model evaluations all afect deployment cost. The BANKING77 result supports a smaller parameter footprint than the 7B reference; the CLINC150 result illustrates an accuracy-resource trade-of from four-model fusion.

## 4 BANKING77 Continuation and Execution Audit

Table 5: All reported continuation candidates under the uniform eager FP32 path.
<table><tr><td>Checkpoint</td><td>Correct / 3,080</td><td>Accuracy (%)</td></tr><tr><td>Historical checkpoint, uniform path</td><td>2895</td><td>93.9935</td></tr><tr><td>Control seed 13</td><td>2895</td><td>93.9935</td></tr><tr><td>Control seed 42</td><td>2894</td><td>93.9610</td></tr><tr><td>Control seed 87</td><td>2892</td><td>93.8961</td></tr><tr><td>SAM seed 13 (headline)</td><td>2899</td><td>94.1234</td></tr><tr><td>SAM seed 42</td><td>2896</td><td>94.0260</td></tr><tr><td>SAM seed 87</td><td>2894</td><td>93.9610</td></tr><tr><td>Control three-seed mean</td><td></td><td>93.9502</td></tr><tr><td>SAM three-seed mean</td><td></td><td>94.0368</td></tr></table>

Table 6: Fixed-checkpoint retrospective diferences. All reported intervals include zero.
<table><tr><td>Comparison</td><td>Delta (pp)</td><td>Shared-row 95% interval (pp)</td></tr><tr><td>SAM seed 13 vs historical</td><td>+0.1299</td><td> $[ - 0 . 1 6 2 3 , + 0 . 4 5 4 5 ]$ </td></tr><tr><td>SAM mean vs historical</td><td>+0.0433</td><td> $[ - 0 . 2 2 7 3 , + 0 . 3 2 4 7 ]$ </td></tr><tr><td>SAM mean vs control mean</td><td>+0.0866</td><td> $\left[ - 0 . 1 2 9 9 , + 0 . 3 1 3 9 \right]$ </td></tr></table>

The SAM mean is 94.0368%, compared with 93.9935% for the historical checkpoint under the same evaluation path. Its +0.0433-point mean diference has an interval spanning zero. The SAM-control mean diference is +0.0866 points, also with an interval spanning zero. Neither comparison establishes a reliable gain. The 94.1234% headline additionally carries post-test best-seed selection efects that these intervals do not model.

## 4.1 Historical Prediction Drift

Table 7: Distinct evaluation records for the same historical BANKING77 checkpoint.
<table><tr><td>Execution record</td><td>Correct / 3,080</td><td>Accuracy (%)</td></tr><tr><td>Archived historical predictions</td><td>2,898</td><td>94.0909</td></tr><tr><td>Current replay of original BF16 script</td><td>2,897</td><td>94.0584</td></tr><tr><td>Uniform eager FP32 evaluation</td><td>2,895</td><td>93.9935</td></tr></table>

The audit reports matching data and weight hashes but unresolved exact prediction reproduction. The current original-script replay difers on three predicted labels from the archive. Eight audited eager conditions yield 2,895 correct predictions. Execution-path sensitivity is demonstrated, but the responsible operator and remaining historical drift have not been isolated. These records must remain separate; selecting a favorable execution path would not establish a robust training improvement.

## 5 Reproducibility and Limitations

The evaluation explorer and its linked JSON records [7] provide the result source for this revision. The continuation protocol includes hashes for the historical model and all six continuation checkpoints, training configuration, precision, and the bootstrap scope. The downloaded page and three JSON files are retained with SHA-256 hashes in the companion source manifest. No new training or model inference was performed for this editorial revision.

The archived hybrid report records a full 4,500-example rerun of the exported CLINC150 system with zero prediction diferences. Separate historical package checks on two synthetic examples establish loading behavior only. Member configurations, parameter counts, and fusion selection are available in the archive; Historical hardware provenance and a uniformly rerun cross-model benchmark remain incomplete.

Several limits afect interpretation. The two tasks use diferent architectures and objectives, so their scores do not establish a common causal improvement. The BANKING77 headline is selected after test evaluation, while its mean gain remains uncertain. Its bootstrap intervals exclude selection and seed-population uncertainty. CLINC150 component comparisons condition on hybrid-selected checkpoints and repeated benchmark exposure. Some external reference model sizes remain unverified, and parameter counts do not establish compute-matched eficiency. Finally, pretraining contamination and semantic overlap have not been ruled out; in-scope accuracy does not establish out-of-scope, robustness, or deployment performance.

This report is scoped to two English benchmarks and should not be interpreted as a complete inventory of all project experiments or evidence of multilingual generalization. The public page is a result display, not itself proof of accepted leaderboard submission or verified delivery of every model package.

## 6 Conclusion

DeepOpen achieves an observed BANKING77 accuracy of 94.1234% with a single SAM-continued classifier and CLINC150 in-scope accuracy of 98.0222% with a four-encoder ensemble. These figures match the specified evaluation explorer. Reporting the complete BANKING77 continuation results and execution audit gives necessary context: the three-seed mean is 94.0368%, and its gain over the uniform-path historical checkpoint remains statistically inconclusive. The systems provide documented task-specific results; stronger causal, eficiency, or generalization claims require additional controlled evidence.

## References

[1] B. Warner et al. 2025. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Eficient, and Long Context Finetuning and Inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics.

[2] I. Casanueva, T. Temcinas, D. Gerz, M. Henderson, and I. Vulic. 2020. Eficient Intent Detection with Dual Sentence Encoders. In Proceedings of NLP4ConvAI.

[3] S. Larson et al. 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. In Proceedings of EMNLP-IJCNLP.

[4] X. Liang et al. 2021. R-Drop: Regularized Dropout for Neural Networks. Advances in Neural Information Processing Systems.

[5] P. Foret, A. Kleiner, H. Mobahi, and B. Neyshabur. 2021. Sharpness-Aware Minimization for Eficiently Improving Generalization. In International Conference on Learning Representations.

[6] P. He, J. Gao, and W. Chen. 2021. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv preprint arXiv:2111.09543.

[7] DeepOpen authors. 2026. Evaluation Explorer: Benchmark Comparison. Online evaluation resource, version ensemble-2.

[8] N. Bang, J. Lee, and M.-W. Koo. 2023. Task-Optimized Adapters for an End-to-End Task-Oriented Dialogue System. In Findings of the Association for Computational Linguistics: ACL 2023.

[9] Meta. Llama 2 7B Model Card. Hugging Face model documentation.

[10] Microsoft. DeBERTa-v3-large Model Card. Hugging Face model documentation.

[11] Hugging Face. Transformers: ModernBERT Implementation. Software source code, version 4.57.6.