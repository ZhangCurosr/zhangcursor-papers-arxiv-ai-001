# Relational Over-Regularization: Graph-Based AI-Generated Text Detection via Sentence Transition Deviation

Hyeonchu Park, Bugeun Kim

Department of Artificial Intelligence, Chung-Ang University, Republic of Korea {phchu0429, bgnkim}@cau.ac.kr

## Abstract

Detecting AI-generated text (AIGT) remains challenging because existing approaches rely on token-level statistical signals or independent stylometric features, causing them to overfit to specific generators and fail under distribution shift. We identify a structural signal at the sentence-pair level: LLMs produce inter-sentence transition variance that deviates from human writing through inflated variance driven by recurring similarity bursts at paragraph boundaries and templated transitions. We formalize this as Relational Over-Regularization (ROR) and validate it across four benchmarks (p < 0.001). The central contribution is this relational problem formulation, not a novel GNN architecture; CSFG is one concrete instantiation for operationalizing ROR. To exploit this signal, we propose the Cross-Source Stylometric Fingerprint Graph (CSFG), a graph-based framework that encodes positional, sequential, semantic, and transition deviation signals as learnable GNN edge features. The per-edge signed deviation $\delta _ { i j }$ operationalizes ROR without hand-crafted thresholds and acts as a false-positive calibrator. CSFG achieves 97.14% accuracy under binary detection, outperforming the strongest graph-based baseline by 11.14 pp, with a false positive rate of 1.57% and robust generalization to unseen LLMs in the inflated-variance regime; detection degrades for generators whose transition variance falls at or below the human baseline.

## 1 Introduction

Recent advances in Large Language Models (LLMs) have enabled AI-generated text (AIGT) to reach a level of fluency indistinguishable from human writing, raising concerns about misinformation, ghostwriting, and academic misconduct. Existing detection approaches, perplexity-based statistical signals (Mitchell et al., 2023; Hans et al., 2024) and Transformer-based classifiers (Guo et al., 2023; Bao et al., 2023), share two limitations:

they overfit to generator-specific surface distributions and degrade under unseen conditions LLMs (Dugan et al., 2024), and they treat linguistic features as independent variables, failing to capture sentence-level relational structure.

We argue that the key structural signal lies at the sentence-pair level. Human writing exhibits organically irregular inter-sentence similarity distributions from topic drift, affective variation, and revision inconsistency. LLMs optimized via next-token prediction and RLHF (Ouyang et al., 2022), by contrast, predominantly produce recurring similarity bursts, sharp local spikes at paragraph boundaries, topic restatements, and templated transitions, that inflate and pattern-skew transition variance in a manner systematically distinguishable from human writing. While Wang et al. (2023b) observed document-level semantic stability in AIGT, we identify a complementary pattern: structured local bursts define the dominant relational signature of AI-generated discourse across the generators we study. We formalize this as Relational Over-Regularization (ROR) and validate it empirically across four benchmarks (§3.1). Importantly, ROR constitutes the central problem formulation of this work: we frame AIGT detection as identifying deviations in relational transition structure, rather than introducing another architecture-specific detection mechanism. CSFG implements this formulation, designed to operationalize ROR through learnable edge representations. We note, however, that certain generators deviate from this pattern in the opposite direction—producing hyper-uniform text whose transition variance falls at or below the human baseline, a boundary condition whose implications for the detector scope we discuss in the context of our generalization experiments.

To exploit this signal, we represent each document as a sentence graph whose edges carry a transition deviation feature $\delta _ { i j }$ , the signed difference between a sentence pair’s cosine similarity and the document-level mean. A positive $\delta _ { i j }$ flags a local similarity burst; a negative value indicates a suppressed transition. This per-edge feature captures the inflated-variance regime of ROR without document-level aggregation or hand-crafted thresholds, while the signed encoding preserves sensitivity to both directional deviations for generators observed during training. We propose Cross-Source Stylometric Fingerprint Graph (CSFG), a GNN that detects AIGT from the structural pattern of transition deviations across all edges.

Our contributions are: (1) ROR, a sentence-pairlevel structural signal of AIGT validated across four benchmarks; (2) transition deviation $\delta _ { i j }$ , a novel per-edge GNN feature encoding local similarity burst signatures; (3) CSFG, a graph-based framework jointly modeling sequential and semantic sentence relations with $\delta _ { i j }$ as a learnable edge feature; and (4) 97.14% binary detection accuracy, outperforming the strongest graph-based baseline by 11.14pp with low false positive rates on unseen LLMs in the inflated-variance regime.

## 2 Related Work

## 2.1 AIGT Detection

AIGT detection (AIGTD) has emerged as a major research problem with the rapid advancement of LLMs (Wu et al., 2023). Existing approaches fall into two categories: perplexity-based zeroshot detection (Mitchell et al., 2023; Bao et al., 2023; Hans et al., 2024) and supervised Transformer classifiers (Solaiman et al., 2019; Hu et al., 2023). Despite strong benchmark performance, the RAID (Dugan et al., 2024) and M4 (Wang et al., 2024b) benchmarks show that detection degrades substantially under unseen LLMs, domain shifts, and adversarial rewriting, suggesting over-reliance on token-level surface signals rather than discourselevel regularity patterns.

Recent work has explored structural consistency as a detection signal. Wang et al. (2023b) demonstrated that AIGT exhibits more stable semantic structures than human text via masking-based selfconsistency analysis. However, macro-level stability cannot capture finer-grained sentence-pair phenomena: Recurring local bursts at paragraph boundaries may be absorbed into a globally stable mean, remaining invisible to aggregate analysis. We identify this sentence-pair transition signal as a complementary discriminator and formalize it in $\ S 3$

## 2.2 Discourse Structure and Relational Stylometry

Traditional stylometry summarises text as an independent document-level scalar features (Stamatatos, 2009), which, while discriminative (Uchendu et al., 2021), are structurally blind to relational signals: whether a given transition is anomalous relative to the document’s own distributional baseline. Discourse coherence modelling (Barzilay and Lapata, 2008; Lapata, 2003) establishes that textual naturalness emerges from relational structures between sentences, motivating sentence-pair transitions as the primary detection unit.

We bridge these traditions using the transition deviation $\delta _ { i j } \mathbf { \cdot }$ the signed difference between a sentence pair’s cosine similarity and the documentlevel mean, capturing the extent to which each transition departs from the document’s baseline. Human text exhibits organic irregularity in this deviation, whereas LLMs produce structurally biased patterns (Relational Over-Regularization; §3.1)—a signal captured by neither independent stylometric features nor absolute coherence scores.

## 2.3 Graph-based Text Representation Learning

GNNs have been widely adopted in NLP for relational text modelling (Yao et al., 2019; Wang et al., 2024a). For AIGT detection, CoCo (Liu et al., 2023) combines coherence graphs with contrastive learning at the sentence level, but encodes coherence as an absolute scalar similarity without capturing the signed deviation $\delta _ { i j }$ from the document mean—and thus cannot distinguish local similarity bursts from uniformly high similarity, precisely the distinction that characterizes AI-generated discourse under ROR. Valdez-Valenzuela et al. (2025) exploits syntactic dependency graphs to improve detection performance, but focuses on token-level syntactic structures rather than sentence-level semantic transition patterns. More recently, Kim et al. (2024) proposes $\mathrm { L M ^ { 2 } o t i f s }$ , an explainable GNN framework that constructs word co-occurrence graphs and extracts interpretable motifs to differentiate human and machine-generated text; however, operating at the word level, it does not model inter-sentence transition dynamics and thus cannot capture the document-level relational variance that characterizes ROR. We address this gap with a sentence-level graph that encodes transition deviation as an explicit edge feature.

## 3 Method

We propose the Cross-Source Stylometric Fingerprint Graph (CSFG), a GNN-based framework that models sentence-level relational dynamics for AIGTD. The cross-source property reflects joint training on human-written text and multiple LLM outputs, with an auxiliary source-attribution head that discourages generator-specific surface artifacts. A GNN is chosen because the distinguishing signature of AIGT lies not in individual sentence properties but in inter-sentence transition patterns—the structurally biased variance of sentence-to-sentence similarity. By representing sentences as nodes and pairwise relationships as typed, feature-rich edges, the GNN aggregates local and long-range relational context through iterative message passing, an inductive bias unavailable to independent sentence classifiers or sequence models.

## 3.1 Hypothesis: Relational Over-Regularization

LLMs optimized for next-token prediction and RLHF produce discourse with a structurally biased transition variance pattern (Ouyang et al., 2022). This deviation manifests in two directions: (1) inflated variance driven by recurring similarity bursts, sharp local spikes from paragraph boundaries, topic restatements, and templated transitions; or (2) suppressed variance (hyper-uniform generation), where inter-sentence similarity is held below the organic irregularity of human discourse. Human writing, by contrast, exhibits organic irregularity from topic drift, affective fluctuation, and revision inconsistency.

We attribute this structural bias to two complementary training pressures. Next-token prediction encourages the model to maintain local semantics coherence by generating tokens that are strongly consistent with the preceding context, producing high-similarity plateaus within topically coherent spans. RLHF further reinforces structured discourse organization: human raters reward responses with clear paragraph boundaries, explicit topic restatements, and summary sentences, which are perceived as well-organized and easy to follow (Ouyang et al., 2022). These two pressures operate jointly: next-token prediction creates intraspan similarity plateaus, while RLHF reward shaping introduces sharp cross-span spikes at structurally salient boundaries. The result is a transition variance pattern that is both higher in mean and more burst-skewed than the organic irregularity of human discourse, a signal we formalize as Relational Over-Regularization (ROR).

Formally, let ${ \mathcal { T } } ( D ) = \{ \sin ( s _ { i } , s _ { i + 1 } ) \} _ { i = 1 } ^ { n - 1 }$ denote consecutive-sentence cosine similarities for document D. The distribution of $\mathrm { V a r } ( \mathcal { T } ( D ) )$ differs systematically between AIGT and human text in a direction determined by the generator’s training objective and decoding procedure. Critically, this signal is invisible to document-level mean similarity $\bar { \sigma } ( D )$ : two documents may share an identical σ¯ while differing markedly in local burst structure, as recurring spikes are absorbed into the global average (Wang et al., 2023b). $\mathrm { V a r } ( \mathcal { T } ( D ) )$ exposes this structure directly, making it a complementary discriminator to macro-level stability analysis. As $\delta _ { i j }$ is a signed deviation from the document mean (Eq. 4), it captures both regimes without explicit directional thresholds; however, generalization to hyper-uniform generators depends on their presence during training (see Appendix C). We empirically validate the inflated-variance direction across four benchmarks in §5.1.

## 3.2 Document Graph Construction

Each document D is a graph $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } )$ with sentences as nodes and edges encoding relational dependencies.

Node Features. Each node $v _ { i }$ concatenates a frozen roberta-base [CLS] embedding (Liu et al., 2019) with a 10-dimensional stylometric vector:

$$
\mathbf { h } _ { i } ^ { ( 0 ) } = \left[ \mathbf { h } _ { i } ^ { \mathrm { R o B E R T a } } \parallel \mathbf { h } _ { i } ^ { \mathrm { s t y l e } } \right] \in \mathbb { R } ^ { 7 7 8 } .\tag{1}
$$

The stylometric vector covers five discoursefunction features (hedge-word, modal-verb, firstperson pronoun ratios; question/exclamation marks) and five surface-complexity features (token count, type-token ratio, comma density, discoursemarker presence, long-word ratio), all of which have been shown to differentiate human and LLMgenerated text (Reinhart et al., 2025; Opara, 2024; Aityan et al., 2025).

Sequential Edges. Adjacent sentences are connected to model a linear discourse trajectory: $e _ { i , i + 1 } ^ { \mathrm { s e q } } \in \mathcal { E } , i = 1 , \dots , n - 1$

Semantic Edges. Long-range dependencies are captured by connecting sentence pairs whose cosine similarity exceeds the threshold θ:

![](images/7849369a67c303fb67dfc20594f15c5594b4b0f427a3801e48c552dea18f4869.jpg)  
Figure 1: Overview of the CSFG framework.

$$
e _ { i j } ^ { \mathrm { s e m } } \in \mathcal { E } \quad \mathrm { i f } \quad \mathrm { s i m } ( s _ { i } , s _ { j } ) > \theta , i < j .\tag{2}
$$

We set $\theta = 0 . 6$ by validation AUROC; sensitivity analysis is in Appendix B.2.

Edge Features. Each edge carries a fourdimensional vector:

$$
\mathbf { e } _ { i j } = \left[ \sin ( s _ { i } , s _ { j } ) , ~ \tilde { d } _ { i j } , ~ { \mathcal { N } } _ { \mathrm { s e q } } , ~ \delta _ { i j } \right] \in \mathbb { R } ^ { 4 } ,\tag{3}
$$

where $\tilde { d } _ { i j } = | i - j | / ( n - 1 )$ is normalised positional distance, ${ \mathcal { k } } _ { \mathrm { s e q } } = \mathbf { 1 } [ | i - j | = 1 ]$ flags sequential edges, and

$$
\delta _ { i j } = \sin ( s _ { i } , s _ { j } ) - \bar { \sigma } , \quad \bar { \sigma } = \frac { 1 } { \binom { n } { 2 } } \sum _ { i < j } \sin ( s _ { i } , s _ { j } ) .\tag{4}
$$

Computing σ¯ over all pairs (rather than consecutive pairs only) captures the global similarity baseline against which each local burst is measured. Feature contributions are validated in Appendix B.3.

## 3.3 GNN Encoder and Graph-level Readout

Node features are projected to a $d _ { h }$ -dimensional space, then updated over L = 3 EdgeConv layers (Wang et al., 2019):

$$
\mathbf { h } _ { i } ^ { \mathrm { p r o j } } = \mathrm { R e L U } \Big ( \mathbf { W } _ { \mathrm { i n } } \mathbf { h } _ { i } ^ { ( 0 ) } \Big ) ,\tag{5}
$$

$$
\begin{array} { r } { \mathbf { m } _ { i j } ^ { ( \ell ) } = \mathrm { R e L U } \left( \mathbf { W } _ { \mathrm { m s g } } ^ { ( \ell ) } \big [ \mathbf { h } _ { i } ^ { ( \ell ) } \big \| \mathbf { h } _ { j } ^ { ( \ell ) } \big \| \mathbf { e } _ { i j } \big ] \right) , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \mathbf { h } _ { i } ^ { ( \ell + 1 ) } = \mathrm { B N } \Big ( \mathbf { W } _ { \mathrm { u p d } } ^ { ( \ell ) } \Big [ \mathbf { h } _ { i } ^ { ( \ell ) } \Big \| \frac { 1 } { | \mathcal { N } ( i ) | } \sum _ { j \in \mathcal { N } ( i ) } \mathbf { m } _ { i j } ^ { ( \ell ) } \Big ] \Big ) } \end{array}\tag{7}
$$

Because $\delta _ { i j }$ enters every message (Eq. 6), transition deviation modulates all layers without explicit

thresholds. The graph representation concatenates mean and max pooling over final-layer nodes:

$$
\begin{array} { r } { \mathbf { h } _ { \mathcal { G } } = [ \frac { 1 } { | \mathcal { V } | } \sum _ { i } \mathbf { h } _ { i } ^ { ( L ) } \| \mathbf { \epsilon } \operatorname* { m a x } _ { i } \mathbf { h } _ { i } ^ { ( L ) } \} \in \mathbb { R } ^ { 2 d _ { h } } , } \end{array}\tag{8}
$$

preserving complementary global (mean) and local (max) relational signals (Xu et al., 2018). This is passed to a binary detection head $f _ { \mathrm { b i n } } : \mathbb { R } ^ { 2 d _ { h } } $ $\mathbb { R } ^ { 2 } \mathbf { : }$ ; an auxiliary source attribution head $f _ { \mathrm { s r c } } :$ $\mathbb { R } ^ { 2 d _ { h } }  \mathbb { R } ^ { K }$ is used only during training and discarded at inference. Both heads are two-layer MLPs (linear–ReLU–dropout–linear).

## 3.4 Training Objective

The model minimizes a joint objective:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { b i n } } + \lambda \mathcal { L } _ { \mathrm { s r c } } , } \end{array}\tag{9}
$$

where $\mathcal { L } _ { \mathrm { b i n } }$ is binary cross-entropy and $\mathcal { L } _ { \mathrm { s r c } }$ is a cross-entropy source attribution loss that acts as a domain-adversarial regulariser, preventing the encoder from collapsing to generator-specific surface cues. We set λ = 1.0 (sensitivity analysis in Appendix B.1); full implementation details are in Appendix A.

## 4 Experiments

We evaluate CSFG under three settings: (1) binary detection (binary human vs. AI classification), (2) unseen model generalization (trained on known LLMs, tested on held-out generators), and (3) robustness evaluation (paraphrasing, humanization, and back-translation perturbations applied to AI-generated text, with no retraining). Prior to the main detection experiments, we empirically validate the ROR hypothesis by comparing

Var(T(D)) distributions between human-written and AI-generated text across all four benchmarks (§5.1). All experiments are run 10 times; we report the mean Accuracy (primary metric), False Positive Rate (FPR) and False Negative Rate (FNR) across runs. Accuracy is chosen as the primary metric because the evaluation sets are balanced; FPR and FNR are reported to distinguish the social costs of false accusations of human authors (FPR: human text incorrectly classified as AI-generated) from missed detections (FNR: AI-generated text incorrectly classified as human). All supervised models use a 70/15/15 train/validation/test split; The checkpoint with the highest validation AUROC is selected for the test evaluation.

Datasets. We use four benchmarks: HC3 (Guo et al., 2023) (expert vs. ChatGPT, professional domains), AIGTBench (Sun et al., 2025) (socialmedia style, 12 LLMs), M4 (Wang et al., 2024b) (multi-domain, multi-generator robustness), and MULTITuDE (Macko et al., 2023) (8 LLMs). Following Kim et al. (2024), we use up to 2,000 samples per generator for Setting 1; Settings 2 and 3 use 1,000 samples in total.

Evaluation Settings. For Setting 1 (binary detection), we compare CSFG against five baselines: the perplexity-based methods Fast-DetectGPT (Bao et al., 2023), DNAGPT (Zhang et al., 2023), LASTDE (Xu et al., 2025), Likelihood (Gehrmann et al., 2019) and , the supervised Transformer classifiers RoBERTa (Solaiman et al., 2019), RADAR (Hu et al., 2023),DetectAnyLLM (Fu et al., 2025), DeTeCtive (Guo et al., 2024), and SeqXGPT (Wang et al., 2023a)

and the graph-based CoCo (Liu et al., 2023) with contrastive learning. All supervised baselines are retrained from scratch under identical data conditions to ensure a fair comparison.

For Setting 2 (unseen model generalization), we construct a separate test set entirely excluded from training, using three recent LLMs: Claude Sonnet 4.6 (Anthropic, 2026), GPT-5 (Singh et al., 2025), and Gemini 2.5 Flash Lite (Comanici et al., 2025). Each generator contributes 500 AI-generated samples paired with a shared set of 500 human-written documents, yielding balanced binary test sets per generator. All Setting 2 evaluations use the model trained in Setting 1 without any retraining or finetuning on the held-out generator data.

For Setting 3 (robustness evaluation), we apply three perturbation conditions to Claude Sonnet 4.6,

GPT-5 and Gemini 2.5 Flash Lite -generated text: paraphrasing, humanization, and back-translation. We use Claude Sonnet 4.6 as the perturbation engine for all three conditions. We acknowledge that using the same model for generation and humanization is a limitation, but our aim is to measure detector robustness under realistic conditions LLMbased rewriting rather than stylistic humanization per se. The detector is not retrained: the model from Setting 1 is applied directly to perturbed texts, making this a zero-shot out-of-distribution evaluation. Human-written texts are also perturbed to isolate whether performance changes reflect detector robustness or a shift in the decision boundary; implications are discussed in §5.4.

Text generation and perturbation follow a unified prompt protocol; full prompts are provided in Appendix E.

## 5 Results and Discussion

## 5.1 Empirical Validation of Relational Over-Regularization

We empirically validate the ROR hypothesis by measuring Var(T (D)) for each document and comparing the resulting distributions between human-written and AIGT, using 1,000 samples per group across all four benchmarks. Statistical significance is assessed via the Mann-Whitney U test; effect size is reported as rank-biserial correlation r. Results are summarised in Table 1 and Figure 2.

AIGT consistently exhibits higher Var(T (D)) than human-written text across all four benchmarks $( p < 0 . 0 0 1$ in all cases), with effect sizes ranging from moderate on HC3 $( r = + 0 . 3 8 )$ and MULTI-TuDE $( r = + 0 . 3 2 )$ to small on M4 $( r = + 0 . 2 5 )$ and AIGTBench $( r ~ = ~ + 0 . 2 2 )$ , confirming the inflated-variance regime of the ROR hypothesis for the generators represented in our training benchmarks. The distributional difference manifests not only in higher mean variance but also in a characteristically heavier upper tail (Figure 2), reflecting recurring similarity bursts that are rare in humanwritten documents. As $\delta _ { i j }$ encodes a signed deviation from the document-level mean, the GNN captures structural departure from human transition variance in both directions, including the hyperuniform suppression observed in some frontier generators (Appendix C), without explicit directional thresholds.

<table><tr><td>Dataset</td><td> $n _ { \mathrm { h u m a n } }$ </td><td> $n _ { \mathrm { A I } }$ </td><td>μhuman</td><td> $\mu _ { \mathrm { A I } }$ </td><td>p-value</td><td>r</td></tr><tr><td>HC3</td><td>1000</td><td>1000</td><td> $5 . 7 2 \times 1 0 ^ { - 7 }$ </td><td> $1 . 0 5 \times 1 0 ^ { - 6 }$ </td><td> $3 . 3 4 \times 1 0 ^ { - 4 5 }$ </td><td>+0.38</td></tr><tr><td>M4</td><td>1000</td><td>1000</td><td> $2 . 5 2 \times 1 0 ^ { - 6 }$ </td><td> $3 . 0 6 \times 1 0 ^ { - 6 }$ </td><td> $5 . 2 1 \times 1 0 ^ { - 2 2 }$ </td><td>+0.25</td></tr><tr><td>MULTITuDE</td><td>1000</td><td>1000</td><td> $1 . 1 3 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 2 \times 1 0 ^ { - 6 }$ </td><td> $9 . 7 3 \times 1 0 ^ { - 6 0 }$ </td><td>+0.32</td></tr><tr><td>AIGTBench</td><td>1000</td><td>1000</td><td> $1 . 9 5 \times 1 0 ^ { - 6 }$ </td><td> $2 . 8 4 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 4 \times 1 0 ^ { - }$  -73</td><td>+0.22</td></tr></table>

Table 1: Mann–Whitney U test results for Var $( \mathcal { T } ( D ) )$ across four benchmarks. µ denotes the per-group mean variance. r: rank-biserial correlation as effect size. Positive r indicates $\mathrm { V a r } ( \mathcal { T } ( D ) )$ is higher in AI-generated text than in human-written text. All differences are statistically significant $( p < 0 . 0 0 1 )$ .

![](images/bdba66bc43e513e62be52475734781662fb0976ed1776a965745e7cefe78a6f0.jpg)  
Figure 2: Distribution of $\mathrm { V a r } ( \mathcal { T } ( D ) )$ for human-written (blue) and AI-generated (orange) text across four benchmarks, as violin plots. The horizontal bar: median. AIGT consistently exhibits higher variance and a heavier upper tail than human-written text, driven by recurring similarity bursts at paragraph boundaries and templated transitions. Significance: $^ { * * * } p < 0 . 0 0 1$ . Effect sizes (r) and exact p-values below each panel.

<table><tr><td>Detector</td><td>ACC (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td>F-DetectGPT</td><td>70.45</td><td>17.30</td><td>17.30</td></tr><tr><td>DNAGPT</td><td>81.23</td><td>24.60</td><td>19.15</td></tr><tr><td>LASTDE</td><td>58.93</td><td>38.82</td><td>43.31</td></tr><tr><td>Likelhood</td><td>60.75</td><td>36.96</td><td>41.54</td></tr><tr><td>DetectAnyLLM</td><td>64.56</td><td>18.65</td><td>52.23</td></tr><tr><td>DeTeCtive</td><td>87.78</td><td>6.74</td><td>17.70</td></tr><tr><td>SeqXGPT</td><td>93.02</td><td>12.43</td><td>1.53</td></tr><tr><td>RoBERTa</td><td>82.80</td><td>5.85</td><td>28.55</td></tr><tr><td>RADAR</td><td>61.38</td><td>28.65</td><td>24.23</td></tr><tr><td>CoCo</td><td>86.00</td><td>23.00</td><td>5.00</td></tr><tr><td>CSFG (Ours)</td><td>97.14</td><td>1.57</td><td>4.15</td></tr></table>

Table 2: Binary detection performance under standard evaluation (Setting 1). <sup>†</sup>RoBERTa is evaluated in an API-only setting via the Hugging Face inference endpoint.

## 5.2 Binary Detection

Table 2 reports binary detection performance across all baselines under Setting 1. CSFG achieves the highest accuracy of 97.14% and the lowest FPR of 1.57%, outperforming all baselines while maintaining a low FNR of 4.15%. The low FPR indicates that $\delta _ { i j }$ effectively suppresses false accusations of human-written text, consistent with its calibration role observed in the ablation study (§B.3).

Among the baselines, SeqXGPT is the strongest competitor, achieving 93.02% accuracy and the lowest FNR of 1.53%, but with a substantially higher FPR of 12.43%. RoBERTa achieves the lowest baseline FPR of 5.85% but exhibits the highest FNR of 28.55%, indicating a conservative decision boundary. The graph-based baseline CoCo achieves 86.00% accuracy with an FNR of 5.00%, but its FPR remains high at 23.00%, suggesting that discourse-level coherence alone is insufficient without explicitly modeling transition deviations.

Compared with SeqXGPT, CSFG improves accuracy by 4.12 percentage points while reducing the FPR from 12.43% to 1.57%. Compared with the graph-based baseline CoCo, CSFG further improves accuracy by 11.14 percentage points, reduces FPR by 21.43 percentage points, and lowers FNR by 0.85 percentage points. These improvements reflect the combined contributions of the four edge features—positional distance, sequential adjacency, cosine similarity, and transition deviation—together with the source-attribution regularizer. As confirmed by the ablation study (§B.3), positional distance is the primary contributor to detection accuracy, whereas $\delta _ { i j }$ mainly improves calibration by reducing FPR without affecting overall accuracy. Finally, the consistently low standard deviation across 10 runs $( \sigma \le 0 . 5 4 \mathrm { p p }$ for all metrics) demonstrates the stability of the proposed framework.

## 5.3 Unseen Model Generalization

Table 3 reports detection performance on three unseen LLMs: Claude Sonnet 4.6, GPT-5, and Gemini 2.5 Flash Lite. Performance varies substantially across generators, indicating that cross-model generalization remains highly model-dependent. CSFG achieves the highest accuracy on Claude Sonnet 4.6 and Gemini 2.5 Flash Lite, reaching 89.45% and 92.26%, respectively, while consistently maintaining the lowest FPR of 0.66%. These results suggest that the proposed relational representation generalizes well to unseen generators exhibiting transition-variance patterns similar to those observed during training.

Among the baselines, LASTDE, Likelihood, and DetectAnyLLM perform close to random guessing, with accuracies generally below 65% and unstable FPR/FNR trade-offs across generators. Recent learning-based methods, including DeTeCtive and SeqXGPT, exhibit more balanced behavior but remain below 62% accuracy on all three unseen generators. CoCo is the strongest baseline, achieving competitive performance on Claude Sonnet 4.6 and Gemini 2.5 Flash Lite, although its FPR remains consistently at around 18%, substantially higher than CSFG’s.

GPT-5 presents a markedly different challenge. Both CSFG and CoCo achieve the same accuracy of 68.85%, but their error profiles differ considerably. CoCo yields an FPR of 18.06% with an FNR of 44.37%, whereas CSFG reduces the FPR to only 0.66% at the expense of an FNR of 61.64%, indicating that it misses a large proportion of GPT-5- generated documents. As discussed in Appendix C, this limitation is not merely a calibration issue but stems from the proposed ROR hypothesis: GPT-5 exhibits a hyper-uniform transition-variance pattern that falls outside the variance regime learned during training. Additional experiments in Appendix C.2 further demonstrate that incorporating GPT-5 into training improves in-distribution detection but does not fully resolve this limitation.

Overall, the results demonstrate that CSFG consistently minimizes false accusations of humanwritten text while outperforming existing methods on generators that preserve the inflated transitionvariance characteristic. At the same time, the GPT-5 results highlight an important limitation of the current framework and motivate future work on modeling both inflated-variance and hyper-uniform generation patterns.

## 5.4 Robustness Against Text Perturbation

Table 4 reports detection performance under three perturbation settings: paraphrasing, humanization, and back-translation. Since the same overall trend is consistently observed across different generators, we present results for Claude Sonnet 4.6 in the main paper for brevity, while the corresponding results for GPT-5 and Gemini 2.5 Flash Lite are provided in Appendix D. No retraining is performed; the detector trained under Setting 1 is directly applied to perturbed texts, making this a zero-shot robustness evaluation.

Unlike Settings 1 and 2, the FPR in this setting is measured on perturbed human-written text. Therefore, it reflects the extent to which perturbation causes human documents to resemble AI-generated text and is not directly comparable to the FPR reported under standard evaluation.

Most baseline detectors exhibit substantial performance degradation under perturbation. LASTDE, Likelihood, and DetectAnyLLM achieve accuracies below 75% across all settings, while De-TeCtive and SeqXGPT show improved robustness but remain sensitive to rewriting strategies, particularly under humanization. Among all baselines, CoCo consistently achieves the strongest performance, obtaining accuracies of 91.55%, 89.22%, and 89.55% for paraphrasing, humanization, and back-translation, respectively.

CSFG remains competitive across all perturbation settings, achieving the highest accuracy under back-translation at 90.96% and performance comparable to CoCo under paraphrasing (90.59% vs. 91.55%) and humanization (87.43% vs. 89.22%). Similar to $\mathrm { C o C o } ,$ CSFG exhibits substantially elevated FPRs under paraphrasing and humanization while maintaining consistently low FNRs between 1.22% and 5.10%. As discussed in Appendix 5.4, back-translation produces a substantially lower FPR than direct stylistic rewriting, suggesting that it better preserves the original transition structure.

We note a limitation of the humanization setting: Claude Sonnet 4.6 is used as both the text generator and the humanization model, which may preserve generator-specific relational patterns. A more rigorous evaluation would employ human rewriting or a model from a different family, which we leave for future work.

<table><tr><td>Detector</td><td>Generator</td><td>Accuracy (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td rowspan="3">LASTDE</td><td>Claude Sonnet 4.6</td><td>50.50</td><td>72.00</td><td>26.40</td></tr><tr><td>GPT-5</td><td>51.80</td><td>89.00</td><td>6.10</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>52.30</td><td>18.50</td><td>77.80</td></tr><tr><td rowspan="3">Likelihood</td><td>Claude Sonnet 4.6</td><td>55.50</td><td>68.20</td><td>20.10</td></tr><tr><td>GPT-5</td><td>55.70</td><td>37.40</td><td>51.40</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>52.10</td><td>39.00</td><td>57.10</td></tr><tr><td rowspan="3">DetectAnyLLM</td><td>Claude Sonnet 4.6</td><td>64.00</td><td>29.90</td><td>42.10</td></tr><tr><td>GPT-5</td><td>60.80</td><td>18.70</td><td>59.80</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>54.00</td><td>19.50</td><td>72.60</td></tr><tr><td rowspan="3">DeTeCtive</td><td>Claude Sonnet 4.6</td><td>52.30</td><td>46.20</td><td>49.40</td></tr><tr><td>GPT-5</td><td>56.40</td><td>48.40</td><td>38.60</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>56.50</td><td>32.30</td><td>55.10</td></tr><tr><td rowspan="3">SeqXGPT</td><td>Claude Sonnet 4.6</td><td>61.20</td><td>33.30</td><td>44.50</td></tr><tr><td>GPT-5</td><td>57.90</td><td>42.70</td><td>41.50</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>56.90</td><td>32.50</td><td>54.10</td></tr><tr><td rowspan="3">CoCo</td><td>Claude Sonnet 4.6</td><td>85.83</td><td>18.06</td><td>10.38</td></tr><tr><td>GPT-5</td><td>68.85</td><td>18.06</td><td>44.37</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>89.74</td><td>17.89</td><td>3.19</td></tr><tr><td rowspan="3">CSFG (Ours)</td><td>Claude Sonnet 4.6</td><td>89.45</td><td>0.66</td><td>20.46</td></tr><tr><td>GPT-5</td><td>68.85</td><td>0.66</td><td>61.64</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>92.26</td><td>0.66</td><td>14.81</td></tr></table>

Table 3: Detection performance on unseen LLMs (Setting 2). Generators are excluded from training.

These robustness results assume ordinary-length documents drawn from conventional benchmarks. We additionally test two boundary conditions outside this regime: documents under 500 characters, and human-written text from a domain with formulaic discourse structure (news). Under both conditions, FPR rises sharply—to 64–71% on short documents for two of three unseen generators, and to 25.00% on structured human news text (XSum; Narayan et al., 2018), against 0.66% and 1.57% respectively under standard evaluation. We report the full results and analysis in Appendix D, since they identify conditions under which CSFG’s low-FPR advantage does not hold.

## 6 Conclusion

We propose CSFG (Cross-Source Stylometric Fingerprint Graph), a graph-based AIGT detection framework grounded in Relational Over-

Regularization (ROR), the observation that LLMs produce recurring inter-sentence similarity bursts that inflate and pattern-skew transition variance in a manner systematically distinguishable from human writing. We operationalize this as a learnable edge feature $\delta _ { i j }$ within an edge-featured GNN, enabling end-to-end detection without manual thresholds.

Across three evaluation settings, CSFG achieves 97.14% binary detection accuracy (+11.14pp over CoCo, FPR = 1.57%), generalizes to unseen LLMs with near-zero FPR, and reveals that LLMbased perturbation contaminates human text with generator-specific transition structure, an effect attributable to the rewriting process rather than detector failure. Ablation studies confirm that $\delta _ { i j }$ acts as a false-positive calibrator rather than a raw detection booster.

CSFG fails structurally on GPT-5 (FNR = 61.64%), whose transition variance falls below the human baseline, a pattern we term hyper-uniform generation. This failure is not addressable by threshold adjustment; it reflects a fundamental boundary of ROR-based detection, which targets the presence of similarity bursts rather than the absence of organic irregularity, and will affect any detector trained exclusively on inflated-variance generators.

<table><tr><td>Detector</td><td>Perturbation</td><td>Accuracy (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td rowspan="3">LASTDE</td><td>Paraphrasing</td><td>49.45</td><td>19.61</td><td>53.77</td></tr><tr><td>Humanization</td><td>54.16</td><td>33.33</td><td>47.14</td></tr><tr><td>Back-translation</td><td>66.73</td><td>27.45</td><td>33.88</td></tr><tr><td rowspan="3">Likelihood</td><td>Paraphrasing</td><td>67.53</td><td>27.45</td><td>32.99</td></tr><tr><td>Humanization</td><td>66.36</td><td>19.61</td><td>35.10</td></tr><tr><td>Back-translation</td><td>74.31</td><td>25.49</td><td>25.71</td></tr><tr><td rowspan="3">DetectAnyLLM</td><td>Paraphrasing</td><td>61.76</td><td>56.86</td><td>19.61</td></tr><tr><td>Humanization</td><td>61.76</td><td>56.86</td><td>19.61</td></tr><tr><td>Back-translation</td><td>63.73</td><td>50.98</td><td>21.57</td></tr><tr><td rowspan="3">DeTeCtive</td><td>Paraphrasing</td><td>80.63</td><td>45.10</td><td>16.70</td></tr><tr><td>Humanization</td><td>66.73</td><td>49.02</td><td>31.63</td></tr><tr><td>Back-translation</td><td>78.74</td><td>37.25</td><td>19.59</td></tr><tr><td rowspan="3">SeqXGPT</td><td>Paraphrasing</td><td>70.48</td><td>9.80</td><td>31.57</td></tr><tr><td>Humanization</td><td>71.90</td><td>29.41</td><td>27.96</td></tr><tr><td>Back-translation</td><td>87.43</td><td>19.61</td><td>11.84</td></tr><tr><td rowspan="3">CoCo</td><td>Paraphrasing</td><td>91.55</td><td>84.75</td><td>1.67</td></tr><tr><td>Humanization</td><td>89.22</td><td>81.54</td><td>4.76</td></tr><tr><td>Back-translation</td><td>89.55</td><td>76.17</td><td>3.52</td></tr><tr><td rowspan="3">CSFG (Ours)</td><td>Paraphrasing</td><td>90.59</td><td>88.24</td><td>1.22</td></tr><tr><td>Humanization</td><td>87.43</td><td>84.31</td><td>5.10</td></tr><tr><td>Back-translation</td><td>90.96</td><td>49.80</td><td>4.79</td></tr></table>

Table 4: Detection performance under text perturbation (Setting 3) on Claude Sonnet 4.6. Results for GPT-5 and Gemini 2.5 Flash Lite are provided in Appendix D.

Future work may incorporate complementary discourse signals (e.g., rhetorical relations, argument structure) to extend coverage to hyperuniform generators, explore sparse or hierarchical graph construction for scalability, and develop perturbation-aware training protocols to mitigate the contamination effect identified in Setting 3.

## Limitations

Scope of ROR and robustness to future generators. The ROR hypothesis may not hold consistently across all generators and writing domains. As demonstrated by the GPT-5 results in Setting 2 (FNR = 61.64%), certain LLMs produce transitionvariance patterns that closely resemble or fall below the human baseline, rendering the similarity-burst signature an unreliable discriminative signal for those generators. This boundary is structural: because $\delta _ { i j }$ primarily targets the presence and distribution of similarity bursts characteristic of the inflated-variance regime, it may not reliably distinguish generators whose relational patterns deviate in the opposite direction. Any detector trained exclusively on inflated-variance generators will face the same constraint. Addressing this limitation may require bidirectional relational modeling that explicitly captures both inflated and suppressed transition patterns, together with complementary discourse-level signals beyond $\delta _ { i j }$ . More broadly, the hypothesis may not generalize to creative writing or high-temperature generation, where idiosyncratic bursts differ in kind from the templated transitions our model targets, nor to highly structured human-written documents such as academic papers or news articles, which naturally exhibit periodic spikes in similarity—a prediction confirmed empirically in Appendix D, where FPR on news text (XSum) reaches 25%, and where short documents (under 500 characters) similarly destabilize $\delta _ { i j }$ estimation, raising FPR to 64–71% for two of three

unseen generators.

Perturbation-induced contamination. Robustness evaluation (Setting 3) reveals that LLM-based perturbation introduces a systematic side effect: rewriting human text with the same model used to generate AIGT imposes generator-specific transition patterns, inflating FPR attributable to rewriting rather than detector failure. Critically, this contamination effect is not specific to CSFG—CoCo exhibits similarly elevated FPR under the same conditions—suggesting it reflects a property of LLM rewriting rather than a detector deficiency. Future robustness evaluations should control for this effect by using a rewriting model distinct from the generator, and perturbation-aware training protocols may be necessary to mitigate it.

Sensitivity to representations and computational scalability. The framework relies on sentence-level cosine similarity computed over roberta-base CLS embeddings, which capture only part of the discourse structure and do not represent higher-level properties such as rhetorical organization or argument flow. Furthermore, pretrained encoders may share representational biases with AIGT, potentially attenuating the discriminative signal of $\delta _ { i j }$ for certain generators. Regarding scalability, semantic edge computation scales quadratically with document length; future work may explore sparse connectivity or hierarchical graph construction to address this constraint without sacrificing the relational signal captured by $\delta _ { i j }$

## The Use of Large Language Models

We used AI-assisted tools during the writing process for this manuscript. Specifically, we employed Grammarly for grammar checking and Claudesonnet 4.6 for language polishing and to improve clarity of expression. These tools were used for editorial purposes.

## Acknowledgments

This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education (RS-2025-25434151) and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)].

## References

Sergey K Aityan, William Claster, Karthik Sai Emani, Sohni Rais, and Thy Tran. 2025. A lightweight approach to detection of ai-generated texts using stylometric features. arXiv preprint arXiv:2511.21744.

Anthropic. 2026. Introducing claude sonnet 4.6. https://www.anthropic.com/news/claude-sonnet-4-6. Accessed: 2026-05-18.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023. Fast-detectgpt: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In The Twelfth International Conference on Learning Representations.

Regina Barzilay and Mirella Lapata. 2008. Modeling local coherence: An entity-based approach. Computational Linguistics, 34(1):1–34.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. 2024. RAID: A shared benchmark for robust evaluation of machinegenerated text detectors. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12463– 12492, Bangkok, Thailand. Association for Computational Linguistics.

Jiachen Fu, Chun-Le Guo, and Chongyi Li. 2025. Detectanyllm: Towards generalizable and robust detection of machine-generated text across domains and models. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 11229– 11238.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander Rush. 2019. GLTR: Statistical detection and visualization of generated text. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pages 111–116, Florence, Italy. Association for Computational Linguistics.

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. 2023. How close is chatgpt to human experts? comparison corpus, evaluation, and detection. arXiv preprint arxiv:2301.07597.

Xun Guo, Shan Zhang, Yongxin He, Ting Zhang, Wanquan Feng, Haibin Huang, and Chongyang Ma. 2024. Detective: Detecting ai-generated text via multi-level contrastive learning. Advances in Neural Information Processing Systems, 37:88320–88347.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting llms with binoculars: Zero-shot detection of machine-generated text. Preprint, arXiv:2401.12070.

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023. Radar: Robust ai-text detection via adversarial learning. Advances in neural information processing systems, 36:15077–15095.

Zae Myung Kim, Kwang Lee, Preston Zhu, Vipul Raheja, and Dongyeop Kang. 2024. Threads of subtlety: Detecting machine-generated texts through discourse motifs. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5449–5474, Bangkok, Thailand. Association for Computational Linguistics.

Mirella Lapata. 2003. Probabilistic text structuring: Experiments with sentence ordering. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics, pages 545–552, Sapporo, Japan. Association for Computational Linguistics.

Yafu Li, Zhilin Wang, Leyang Cui, Wei Bi, Shuming Shi, and Yue Zhang. 2024. Spotting ai’s touch: Identifying llm-paraphrased spans in text. In Findings of the Association for Computational Linguistics: ACL 2024, pages 7088–7107.

Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. 2023. CoCo: Coherenceenhanced machine-generated text detection under low resource with contrastive learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16167–16188, Singapore. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Dominik Macko, Robert Moro, Adaku Uchendu, Jason Lucas, Michiharu Yamashita, Matúš Pikuliak, Ivan Srba, Thai Le, Dongwon Lee, Jakub Simko, and Maria Bielikova. 2023. MULTITuDE: Large-scale multilingual machine-generated text detection benchmark. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9960–9987, Singapore. Association for Computational Linguistics.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. 2023. Detectgpt: zero-shot machine-generated text detection using probability curvature. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org.

Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary!

topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 1797–1807.

Chidimma Opara. 2024. Styloai: Distinguishing aigenerated content with stylometric analysis. In International conference on artificial intelligence in education, pages 105–114. Springer.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Alex Reinhart, Ben Markey, Michael Laudenbach, Kachatad Pantusen, Ronald Yurko, Gordon Weinberg, and David West Brown. 2025. Do llms write like humans? variation in grammatical and rhetorical styles. Proceedings ofthe National Academy of Sciences, 122(8):e2422455122.

Jenna Russell, Marzena Karpinska, and Mohit Iyyer. 2025. People who frequently use chatgpt for writing tasks are accurate and robust detectors of ai-generated text. In Proceedings ofthe 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5342–5373.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, Gretchen Krueger, Jong Wook Kim, Sarah Kreps, Miles McCain, Alex Newhouse, Jason Blazakis, Kris McGuffie, and Jasmine Wang. 2019. Release strategies and the social impacts of language models. Preprint, arXiv:1908.09203.

Efstathios Stamatatos. 2009. A survey of modern authorship attribution methods. Journal ofthe American Society for Information Science and Technology, 60(3):538–556.

Zhen Sun, Zongmin Zhang, Xinyue Shen, Ziyi Zhang, Yule Liu, Michael Backes, Yang Zhang, and Xinlei He. 2025. Are we in the AI-generated text world already? quantifying and monitoring AIGT on social media. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 22975–23005, Vienna, Austria. Association for Computational Linguistics.

Adaku Uchendu, Zeyu Ma, Thai Le, Rui Zhang, and Dongwon Lee. 2021. TURINGBENCH: A benchmark environment for Turing test in the age of neural text generation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2001–2016, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Andric Valdez-Valenzuela, Helena Gómez-Adorno, and Manuel Montes-y Gómez. 2025. Text graph neural networks for detecting AI-generated content. In Proceedings of the 1stWorkshop on GenAI Content Detection (GenAIDetect), pages 134–139, Abu Dhabi, UAE. International Conference on Computational Linguistics.

Kunze Wang, Yihao Ding, and Soyeon Caren Han. 2024a. Graph neural networks for text classification: a survey. Artificial Intelligence Review, 57(8).

Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. 2023a. Seqxgpt: Sentencelevel ai-generated text detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1144–1156.

Rongsheng Wang, Qi Li, and Sihong Xie. 2023b. Detectgpt-sc: improving detection of text generated by large language models through selfconsistency with masked predictions. arXiv preprint arXiv:2310.14479.

Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. 2019. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12.

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Chenxi Whitehouse, Osama Mohammed Afzal, Tarek Mahmoud, Toru Sasaki, Thomas Arnold, Alham Aji, Nizar Habash, Iryna Gurevych, and Preslav Nakov. 2024b. M4: Multi-generator, multi-domain, and multilingual black-box machine-generated text detection. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1369– 1407, St. Julian’s, Malta. Association for Computational Linguistics.

Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Hong Lin. 2023. Ai-generated content (aigc): A survey. Preprint, arXiv:2304.06632.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826.

Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, and Fei Wu. 2025. Trainingfree llm-generated text detection by mining token probability sequences. In International Conference on Learning Representations, volume 2025, pages 19072–19098.

Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Graph convolutional networks for text classification. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 7370–7377.

Xiao Yu, Yuang Qi, Kejiang Chen, Guoqiang Chen, Xi Yang, Pengyuan Zhu, Xiuwei Shang, Weiming Zhang, and Nenghai Yu. 2024. Dpic: Decoupling

prompt and intrinsic characteristics for llm generated text detection. Advances in Neural Information Processing Systems, 37:16194–16212.

Daoan Zhang, Weitong Zhang, Bing He, Jianguo Zhang, Chenchen Qin, and Jianhua Yao. 2023. Dnagpt: A generalized pretrained tool for multiple dna sequence analysis tasks. bioRxiv, pages 2023–07.

Xu Zheng, Zhuomin Chen, Esteban Schafir, Sipeng Chen, Hojat Allah Salehi, Haifeng Chen, Farhad Shirani, Wei Cheng, and Dongsheng Luo. 2025. Lm<sup>2</sup>otifs : An explainable framework for machine-generated texts detection. Preprint, arXiv:2505.12507.

## A Environment

Hardware configuration: The experiments were conducted on a system with an AMD Ryzen Threadripper 3960X 24-Core Processor and four NVIDIA RTX A6000 GPUs. The four NVIDIA RTX A6000 GPUs are used to train existing detectors.

LLM APIs: All text generation and perturbation experiments are conducted via the OpenRouter API. For Setting 2, we access Claude Sonnet 4.6 (Anthropic), GPT-5 (OpenAI), and Gemini 2.5 Flash Lite (Google). For Setting 3, perturbation prompts are applied to Claude Sonnet 4.6- generated text. All models are queried with a fixed random seed and temperature = 1.0 to ensure reproducibility.

Implementation Details Sentence embeddings are produced by roberta-base, encoding each sentence via the [CLS] token with $\ell _ { 2 }$ normalisation (hidden dimension 768, max\_length= 128, batch size 128). The encoder weights are frozen throughout training. Graphs are pre-built and cached to disk prior to training.

The GNN comprises L = 3 EdgeConv layers with hidden dimension $d _ { h } = 1 2 8$ and dropout rate 0.2. The model is optimised with AdamW (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $5 \times 1 0 ^ { - 4 } )$ with batch size 32. All experiments use a $7 0 / 1 5 / 1 5$ stratified train/validation/test split, and the checkpoint with the highest validation AUROC is selected for test evaluation.

Following Zheng et al. (2025), we use up to 2,000 samples per generator for Setting 1; Settings 2 and 3 use 1,000 samples in total.

## B Ablation Study

## B.1 Effect of Auxiliary Loss Weight (λ)

The joint training objective combines the binary detection loss $\mathcal { L } _ { \mathrm { b i n } }$ and the source attribution loss $\mathcal { L } _ { \mathrm { s r c } }$ via a scalar weight λ. The source attribution head is designed to prevent the shared encoder from collapsing to shallow, generator-specific surface cues; however, an excessively large λ risks over-regularising the encoder toward source discrimination at the expense of binary detection. To quantify this trade-off, we train CSFG with $\lambda \in \{ 0 . 0 , 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \}$ , where $\lambda = 0 . 0$ corresponds to training without the auxiliary objective. All other hyperparameters are held at their default values.

Table 5 reports the results. Removing the source attribution objective entirely $( \lambda = 0 . 0 )$ yields the highest FPR (2.27%) and FNR (5.23%), confirming that the auxiliary task provides meaningful regularisation for both error types. As λ increases from 0.0 to 1.0, accuracy improves monotonically from 96.25% to 97.22%, and both FPR and FNR decline consistently, indicating that stronger source attribution pressure continues to benefit binary detection throughout the evaluated range. Unlike configurations where an excessively large λ competes with the primary detection objective, performance does not degrade at $\lambda = 1 . 0$ , suggesting that the two objectives are complementary rather than competing at this scale. We therefore adopt $\lambda = 1 . 0$ as the default configuration.

## B.2 Sensitivity to Semantic Edge Threshold (θ)

The semantic edge threshold θ controls the sparsity of long-range edges in the document graph: a low θ yields a dense graph in which the sequential backbone is supplemented by many weak semantic links, whereas a high θ produces a near-sequential graph that captures only strongly similar sentence pairs. We vary $\theta \in \{ 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 \}$ and report detection performance across all three metrics.

Table 6 reports the results. Performance is notably stable across the evaluated range: accuracy varies by only 0.09pp (97.14%-97.23%) and FNR by 0.13pp (4.01%-4.16%), indicating that CSFG is not sensitive to the precise choice of θ. The lowest FPR is observed at $\theta = 0 . 7 \ : ( 1 . 4 5 \% )$ , while the highest accuracy and lowest FNR are achieved at $\theta = 0 . 6 ( 9 7 . 2 3 \% , 4 . 0 1 \% )$ . FPR exhibits a nonmonotonic pattern across thresholds, with no single value dominating all three metrics simultaneously.

We adopt $\theta = 0 . 6$ as the default configuration on the basis of validation-split AUROC, which shows a consistent advantage for this value across training runs. At $\theta = 0 . 6$ , the graph connects sentence pairs with meaningfully high semantic similarity while avoiding the spurious long-range edges introduced by lower thresholds, aligning with the theoretical motivation of capturing genuine thematic recurrence rather than noisy co-occurrence. The marginal FPR advantage of $\theta = 0 . 7 \ : ( \Delta \mathrm { F P R } =$ 0.08pp) does not outweigh the accuracy and FNR gains at the default setting.

<table><tr><td rowspan="2">λ</td><td colspan="2">Accuracy (%)</td><td colspan="2">FPR (%)</td><td colspan="2">FNR (%)</td></tr><tr><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td></tr><tr><td>0.0 (no aux.)</td><td>96.25</td><td>0.42</td><td>2.27</td><td>0.31</td><td>5.23</td><td>0.81</td></tr><tr><td>0.1</td><td>96.95</td><td>0.26</td><td>1.68</td><td>0.53</td><td>4.42</td><td>0.83</td></tr><tr><td>0.3</td><td>97.02</td><td>0.24</td><td>1.51</td><td>0.40</td><td>4.44</td><td>0.58</td></tr><tr><td>0.5</td><td>97.19</td><td>0.17</td><td>1.60</td><td>0.38</td><td>4.02</td><td>0.46</td></tr><tr><td>1.0</td><td>97.22</td><td>0.22</td><td>1.48</td><td>0.39</td><td>4.08</td><td>0.45</td></tr></table>

Table 5: Effect of auxiliary loss weight λ on binary detection performance. All results are averaged over 10 runs; σ denotes standard deviation in percentage points (pp). No aux. $( \lambda = 0 . 0 )$ denotes training without the source attribution head. Bold indicates the best value per column.

<table><tr><td rowspan="2">θ</td><td colspan="2">Accuracy (%)</td><td colspan="2">FPR (%)</td><td colspan="2">FNR (%)</td></tr><tr><td>μ</td><td>σ</td><td>μ σ</td><td></td><td>μ</td><td>σ</td></tr><tr><td>0.3</td><td>97.20</td><td>0.24</td><td>1.48</td><td>0.30 4.11</td><td>0.55</td><td></td></tr><tr><td>0.4</td><td>97.22</td><td>0.21</td><td>1.51</td><td>0.29</td><td>4.05</td><td>0.51</td></tr><tr><td>0.5</td><td>97.14</td><td>0.21</td><td>1.58</td><td>0.36</td><td>4.14</td><td>0.48</td></tr><tr><td>0.6</td><td>97.23</td><td>0.17</td><td>1.53</td><td>0.43</td><td>4.01</td><td>0.45</td></tr><tr><td>0.7</td><td>97.19</td><td>0.22</td><td>1.45</td><td>0.30</td><td>4.16</td><td>0.44</td></tr></table>

Table 6: Sensitivity to semantic edge threshold θ. Lower θ yields denser graphs with more long-range semantic edges. All results are averaged over 10 runs; σ denotes standard deviation in percentage points (pp). Bold indicates the best value per column.

## B.3 Contribution of Edge Feature Components

The four-dimensional edge feature vector ${ \bf e } _ { i j } = { \bf \Phi }$ $[ \sin ( s _ { i } , s _ { j } ) , \tilde { d } _ { i j } , { \mathcal { H } } _ { \mathrm { s e q } } , \delta _ { i j } ]$ encodes complementary relational signals: semantic proximity, positional distance, edge type, and transition deviation. To assess the individual contribution of each component, we ablate them independently by zeroing out the corresponding index in ${ \bf e } _ { i j }$ at collation time, keeping the model architecture and graph structure unchanged.

Table 7 reports the effect of removing each edge feature component individually. The full model achieves the highest accuracy (97.14%) across all variants, confirming that all four components contribute complementary signal. Among the ablated configurations, removing positional distance $( \tilde { d } _ { i j } )$ produces the largest degradation: accuracy drops to 96.89% $( \Delta \ = \ - 0 . 2 5 \mathrm { p p } )$ and FNR rises to 4.67%, the highest among all variants, indicating that position-aware relational encoding is the most critical component for distinguishing discourse topology between human and AIGT. Removing the sequential edge indicator $( \boldsymbol { \mathcal { H } } _ { \mathrm { s e q } } )$ and cosine similarity similarly degrade accuracy by 0.20pp and 0.22pp respectively, and raise FNR to 4.42% and

4.57%, confirming that each component carries non-redundant discriminative signal.

The result for w/o trans\_dev warrants careful interpretation. Removing $\delta _ { i j }$ yields a higher accuracy than the w/o sim and w/o dist variants (96.97%) and the lowest FNR among all ablations (4.40%), yet its FPR (1.65%) exceeds that of the full model (1.57%). This pattern confirms the expected calibrating role of $\delta _ { i j } \mathbf { \dot { \mathrm { : } } }$ rather than maximising raw detection rate, it acts as a regulariser on false positives, suppressing the tendency to classify borderline human-written text as AI-generated. The full model achieves the most balanced error profile across all three metrics, which is preferable in deployment settings where falsely accusing human authors carries a higher social cost than missing AIGT.

## B.4 Effect of GNN Depth

The number of EdgeConv layers determines the receptive field of each node in the sentence graph: a k-layer network aggregates relational signals from sentences up to k hops away. In a sentence-level graph, one hop corresponds to an adjacent or semantically linked sentence, so three layers approximate the span of a coherent paragraph-level discourse unit. Deeper networks risk over-smoothing, in which node representations converge and lose local distinctiveness, while shallower networks may fail to capture document-level topology. We compare configurations with 1, 2, 3, and 4 EdgeConv layers, keeping all other settings fixed at their default values $( \theta = 0 . 6 , \lambda = 1 . 0 )$ .

<table><tr><td rowspan="2">Edge Features</td><td colspan="2">Accuracy (%)</td><td colspan="2">FPR (%)</td><td colspan="2">FNR (%)</td></tr><tr><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td></tr><tr><td>Full (ours)</td><td>97.14</td><td>0.27</td><td>1.57</td><td>0.38</td><td>4.15</td><td>0.54</td></tr><tr><td>w/o sim</td><td>96.92</td><td>0.21</td><td>1.59</td><td>0.49</td><td>4.57</td><td>0.66</td></tr><tr><td>w/o dist</td><td>96.89</td><td>0.24</td><td>1.54</td><td>0.51</td><td>4.67</td><td>0.76</td></tr><tr><td>w/o is_seq</td><td>96.94</td><td>0.17</td><td>1.69</td><td>0.45</td><td>4.42</td><td>0.61</td></tr><tr><td>w/o trans_dev</td><td>96.97</td><td>0.23</td><td>1.65</td><td>0.34</td><td>4.40</td><td>0.67</td></tr></table>

Table 7: Contribution of each edge feature component. Each variant zeros out the corresponding dimension of ${ \bf e } _ { i j }$ at collation time; graph structure and model architecture remain unchanged. All results are averaged over 10 runs; σ denotes standard deviation in percentage points (pp). Bold indicates the best value per column.
<table><tr><td rowspan="2"># Layers</td><td colspan="2">Accuracy (%)</td><td colspan="2">FPR (%)</td><td colspan="2">FNR (%)</td></tr><tr><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td></tr><tr><td>1</td><td>96.59</td><td>0.19</td><td>2.06</td><td>0.31</td><td>4.75</td><td>0.63</td></tr><tr><td>2</td><td>97.09</td><td>0.26</td><td>1.53</td><td>0.28</td><td>4.29</td><td>0.71</td></tr><tr><td>3 (ours)</td><td>97.15</td><td>0.25</td><td>1.42</td><td>0.43</td><td>4.27</td><td>0.67</td></tr><tr><td>4</td><td>97.10</td><td>0.31</td><td>1.53</td><td>0.46</td><td>4.26</td><td>0.68</td></tr></table>

Table 8: Effect of GNN depth on binary detection performance. All results are averaged over 10 runs; σ denotes standard deviation in percentage points (pp). All variants use the full edge feature vector and default configuration $( \theta = 0 . 6 , \lambda = 1 . 0 )$ . Bold indicates the best value per column.

Table 8 reports detection performance as a function of GNN depth. The single-layer network yields the lowest accuracy (96.59%) and the highest FPR (2.06%) and FNR (4.75%), indicating that a receptive field limited to one sentence hop is insufficient to capture the document-level discourse topology required for reliable detection. Performance improves substantially at two layers (97.09%), with both FPR (1.53%) and FNR (4.29%) declining, confirming that aggregating relational context beyond immediate neighbours provides meaningful discriminative signal.

Three layers yield the best accuracy (97.15%) and the lowest FPR (1.42%), supporting the design rationale that aggregating relational context up to three sentence hops approximates a coherent paragraph-level discourse unit. Extending to four layers produces marginal performance degradation in accuracy (97.10%) and a slight FPR increase (1.53%), consistent with the over-smoothing phenomenon in which deeper message passing causes node representations to converge and lose local discriminative structure. FNR continues to decrease marginally beyond three layers $( 4 . 2 7 \%  4 . 2 6 \% )$ , but the gain of 0.01pp does not justify the accompanying increase in false positives. Three layers therefore strikes the best balance across all three metrics and is adopted as the default configuration.

## C Generator-Level Transition Variance Analysis

Section 5.3 reports that CSFG achieves a false negative rate of 61.64% on GPT-5 while attaining FNR $\leq 2 0 . 4 6 \%$ on all other unseen generators. To determine whether this failure reflects a detector deficiency or a qualitative difference in the generative behaviour of GPT-5, we perform a generator-level analysis of transition variance Var(T(D)) on the Setting 2 test set.

## C.1 Per-Generator Transition Variance Statistics

Table 9 reports $\mathrm { V a r } ( T ( D ) )$ statistics for each group alongside the corresponding Setting 2 FNR. Statistical significance is assessed via the Mann-Whitney U test against the human reference group; distributional distance is measured by the Kolmogorov-

Smirnov (KS) statistic D.

## C.2 GPT-5: Hyper-Uniform Generation

As shown in Table 9 and Figure 3, GPT-5 exhibits a mean transition variance of $0 . 8 \times 1 0 ^ { - 3 }$ , lower than that of human-written text $( 1 . 1 \times 1 0 ^ { - 3 } )$ . This reverses the directional prediction of the ROR hypothesis, under which AIGT is expected to exhibit higher transition variance than human writing. We term this pattern hyper-uniform generation: intersentence similarity is suppressed below the organic irregularity of human discourse, rather than being inflated by recurring similarity bursts.

The KS distance between GPT-5 and human transition variance distributions $( D = 0 . 2 3 8 9 )$ is less than half that of Gemini 2.5 Flash Lite $( D =$ 0.6482), and the CDF of GPT-5 closely overlaps with that of human text across the bulk of the distribution (Figure 3). This near-indistinguishability places GPT-5 outside the high-variance decision region learned from training generators, directly explaining the elevated FNR of 61.64%.

Critically, the failure is structural rather than a threshold calibration issue: because $\delta _ { i j }$ is defined as the signed deviation from the document-level mean similarity, a hyper-uniform document produces uniformly small $| \delta _ { i j } |$ values that do not trigger the burst signature targeted by the ROR hypothesis. Simple threshold adjustment cannot resolve this, as lowering the detection boundary would simultaneously increase false positives on human text. Reliable detection of hyper-uniform generators likely requires either retraining with GPT-5 examples or incorporating complementary features that capture the absence of organic irregularity rather than the presence of similarity bursts.

## C.3 Gemini 2.5 Flash Lite: Amplified ROR Signal

In contrast, Gemini 2.5 Flash Lite exhibits a mean transition variance of $2 . 3 \times 1 0 ^ { - 3 }$ , more than twice that of human text, with a KS distance of 0.6482 (Figure 3, green curve). This amplified ROR signal—arising from markedly abrupt topic transitions between sentences—is strongly discriminable by CSFG, consistent with the low FNR of 14.81% observed in Setting 2.

## C.4 Implications for ROR-Based Detection

Taken together, these results reveal that the transition variance landscape of frontier LLMs is not unimodal. Generators differ qualitatively in the direction of their deviation from human transition variance: some amplify the ROR burst pattern (Gemini), some partially exhibit it (Claude Sonnet 4.6), and some invert it (GPT-5). The ROR hypothesis is most predictive for generators whose training or decoding procedure introduces recurring similarity spikes; it is least applicable to generators whose outputs fall within or below the human variance range. Future work should investigate whether hyper-uniform generation is a deliberate property of GPT-5’s training objective or an incidental effect of scale, and whether it will generalize to subsequent model generations.

![](images/f4fa016a63df1801654e6854f920494ecfb0655956f812b7bd25b8659be31e3d.jpg)  
Figure 3: Cumulative distribution functions of $\mathrm { V a r } ( T ( D ) )$ for human-written text and three unseen generators (Setting 2 test set). GPT-5 (red) lies to the left of human text (blue), indicating hyper-uniform generation with suppressed transition variance. Gemini 2.5 Flash Lite (green) is displaced far to the right, reflecting an amplified ROR signal. Claude Sonnet 4.6 (orange) occupies an intermediate position. The close overlap between GPT-5 and human CDFs (KS $D = 0 . 2 3 8 9 )$ directly explains the elevated FNR reported in Table 3.

## D Full Results of Setting 3

## D.1 Document Length and Domain Structure

The evaluations in §5.2–§5.4 use documents with moderate-to-long sentence counts, providing sufficient sentence pairs for estimating $\delta _ { i j }$ and $\mathrm { V a r } ( T ( D ) )$ . We examine two boundary conditions where this assumption may break down: (1) short documents, where too few sentence pairs may yield unstable transition-variance estimates, and (2) highly structured human writing, where conventional discourse patterns may produce similarity spikes resembling the ROR signature.

Short-text performance. We construct a subset of documents shorter than 500 characters from the Setting 2 generators and evaluate the Setting 1 model without retraining. Table 12 summarizes the results.

<table><tr><td>Group</td><td>n</td><td>TV mean</td><td>TV std</td><td>p-value</td><td>KS D</td><td>FNR (%)</td></tr><tr><td>Human</td><td>500</td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td></td></tr><tr><td>GPT-5</td><td>500</td><td> $0 . 8 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $6 . 0 5 \times 1 0 ^ { - 1 3 }$ </td><td>0.2389</td><td>61.64</td></tr><tr><td>Claude Sonnet 4.6</td><td>500</td><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 9 \times 1 0 ^ { - 1 9 }$ </td><td>0.2915</td><td>20.46</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>500</td><td> $2 . 3 \times 1 0 ^ { - 3 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $8 . 8 8 \times 1 0 ^ { - 1 0 2 }$ </td><td>0.6482</td><td>14.81</td></tr></table>

Table 9: Per-generator transition variance statistics on the Setting 2 test set. TV mean and TV std denote the mean and standard deviation of $\mathrm { V a r } ( T ( D ) )$ per group. p-value: Mann-Whitney U test against the human group. KS D: Kolmogorov-Smirnov distance against the human CDF. FNR: miss rate of CSFG from Table 3.

Short documents substantially degrade detection performance, particularly by increasing false positives. For Claude Sonnet 4.6 and Gemini 2.5 Flash Lite, FPR rises by more than an order of magnitude relative to the full-length setting. This behavior follows directly from the construction of $\delta _ { i j } \colon$ with fewer sentences, the number of sentence pairs  <sup>n</sup><sub>2</sub> decreases rapidly, making the documentlevel mean similarity σ¯ more sensitive to individual sentence pairs. Consequently, a single locally similar pair can disproportionately influence the estimated transition-variance pattern.

The resulting low FNR but high FPR suggests that the detector tends to classify uncertain short documents as AI-generated. This is consistent with a decision boundary calibrated on longer documents, where positive $\delta _ { i j }$ values more reliably indicate genuine similarity bursts rather than sampling variation. GPT-5 exhibits a different failure mode. Because its transition variance is already low (Appendix D), short-document sampling does not produce the same inflated-variance pattern. Its FPR therefore remains comparatively lower, but FNR increases to 40.48%, indicating reduced sensitivity to GPT-5’s hyper-uniform transition structure.

Structured human domains. We further evaluate CSFG on 100 human-written documents from XSum (Narayan et al., 2018), a collection of professionally edited BBC news articles characterized by a relatively formulaic discourse structure, including lead sentences, inverted-pyramid organization, and recurring topic restatement. Table 13 reports the results.

CSFG produces an FPR of 25.00% on XSum, substantially higher than the 1.57% observed on the balanced Setting 1 test set. This result demonstrates a domain-specific confound: professionally edited news articles naturally contain paragraphboundary restatements and recurring transition patterns that can resemble the similarity bursts targeted by ROR. Thus, the relational signature is not inherently unique to AIGT; it can also arise from conventionalized human discourse structures.

This failure mode cannot be addressed simply by adding more LLM generators to the training set, because the error stems from the interaction between ROR and a particular human writing convention rather than generator-specific variation. Mitigating it will likely require domain-aware calibration or complementary discourse features that distinguish AI-generated regularities from legitimate structural regularities in human writing.

Overall, these results identify two important boundary conditions for CSFG. Short documents provide too few relational observations for stable estimation, whereas highly structured human genres can produce relational patterns that resemble ROR. Accordingly, CSFG’s low FPR should not be assumed to generalize uniformly across document lengths or writing domains. We report these cases explicitly to characterize when the proposed relational signal becomes unreliable.

## D.2 Cross-Generator Comparison

Table 14 summarizes the Setting 3 results across all three unseen generators, drawing together the pergenerator results reported in Table 4, 10, and 11. Two patterns generalize across generators. First, CSFG and CoCo both exhibit substantially elevated FPR under paraphrasing and humanization relative to back-translation, consistent with the contamination effect discussed in §5.4: LLM-based rewriting imposes generator-specific transition structure onto human text regardless of the source generator. Second, back-translation consistently yields the most favorable FPR/FNR trade-off for CSFG across all three generators (49.80%, 37.25%, and 22.58% FPR for Claude, Gemini, and GPT-5, respectively), supporting our claim that it better preserves the original transition structure than direct stylistic rewriting.

<table><tr><td>Detector</td><td>Perturbation</td><td>Accuracy (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td rowspan="3">LASTDE</td><td>Paraphrasing</td><td>42.62</td><td>9.80</td><td>62.32</td></tr><tr><td>Humanization</td><td>77.72</td><td>56.86</td><td>18.70</td></tr><tr><td>Back-translation</td><td>54.14</td><td>23.53</td><td>48.17</td></tr><tr><td rowspan="3">Likelihood</td><td>Paraphrasing</td><td>60.70</td><td>33.33</td><td>39.92</td></tr><tr><td>Humanization</td><td>46.41</td><td>21.57</td><td>56.91</td></tr><tr><td>Back-translation</td><td>76.61</td><td>27.45</td><td>22.97</td></tr><tr><td rowspan="3">DetectAnyLLM</td><td>Paraphrasing</td><td>51.96</td><td>78.43</td><td>17.65</td></tr><tr><td>Humanization</td><td>53.92</td><td>49.02</td><td>43.14</td></tr><tr><td>Back-translation</td><td>66.67</td><td>47.06</td><td>19.61</td></tr><tr><td rowspan="3">DeTeCtive</td><td>Paraphrasing</td><td>36.72</td><td>5.88</td><td>69.25</td></tr><tr><td>Humanization</td><td>76.61</td><td>66.67</td><td>18.90</td></tr><tr><td>Back-translation</td><td>79.37</td><td>31.37</td><td>19.51</td></tr><tr><td rowspan="3">SeqXGPT</td><td>Paraphrasing</td><td>58.30</td><td>7.84</td><td>45.21</td></tr><tr><td>Humanization</td><td>64.64</td><td>37.25</td><td>35.16</td></tr><tr><td>Back-translation</td><td>89.50</td><td>9.80</td><td>10.57</td></tr><tr><td rowspan="3">CoCo</td><td>Paraphrasing</td><td>0.90</td><td>0.70</td><td>0.04</td></tr><tr><td>Humanization</td><td>0.79</td><td>0.76</td><td>0.15</td></tr><tr><td>Back-translation</td><td>0.91</td><td>0.96</td><td>0.01</td></tr><tr><td rowspan="3">CSFG (Ours)</td><td>Paraphrasing</td><td>0.86</td><td>0.92</td><td>0.06</td></tr><tr><td>Humanization</td><td>0.76</td><td>0.82</td><td>0.17</td></tr><tr><td>Back-translation</td><td>0.92</td><td>0.37</td><td>0.05</td></tr></table>

Table 10: Detection performance under text perturbation (Setting 3) on Gemini 2.5 Flash Lite.

GPT-5 humanization is a notable exception to this trend. Under this condition, CSFG’s accuracy drops to 47.93%, with FNR rising to 55.35%—below SeqXGPT’s accuracy (71.07%) on the same condition and substantially worse than CSFG’s own performance on Claude and Gemini humanization. This compounds the hyper-uniform generation behavior of GPT-5 identified in Appendix C: because GPT-5’s unperturbed transition variance already falls near or below the human baseline, humanization-induced rewriting appears to push a large fraction of GPT-5-generated documents further into the region CSFG associates with human text, rather than introducing the burst-like artifacts that paraphrasing or back-translation partially retain. This result reinforces the boundary condition noted in §6 and Appendix C: ROR-based detection is structurally weakest for generators whose transition-variance regime departs from the inflated-variance direction, and this weakness is amplified, not mitigated, by humanization-style perturbation.

## E LLM Prompt

All text generation and perturbation experiments use the OpenRouter API. Table 15 lists the prompts used for each condition.

For Setting 2 (unseen model generalization), we adopt the generation prompt from Mitchell et al. (2023), which instructs the model to produce a coherent continuation given the first 30 tokens of a document.

For Setting 3 (robustness evaluation), we employ three perturbation prompts drawn from prior work: the paraphrase prompt follows Li et al. (2024), the humanization prompt follows Russell et al. (2025), and the back-translation prompt follows Yu et al. (2024), which performs a two-stage English→Chinese→English pipeline to introduce surface-level variation while preserving semantic content.

<table><tr><td>Detector</td><td>Perturbation</td><td>Accuracy (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td rowspan="3">LASTDE</td><td>Paraphrasing</td><td>43.51</td><td>22.45</td><td>60.00</td></tr><tr><td>Humanization</td><td>75.62</td><td>60.00</td><td>20.73</td></tr><tr><td>Back-translation</td><td>70.11</td><td>35.48</td><td>29.36</td></tr><tr><td rowspan="3">Likelihood</td><td>Paraphrasing</td><td>65.84</td><td>26.53</td><td>34.95</td></tr><tr><td>Humanization</td><td>38.02</td><td>15.56</td><td>66.74</td></tr><tr><td>Back-translation</td><td>65.92</td><td>16.13</td><td>35.78</td></tr><tr><td rowspan="3">DetectAnyLLM</td><td>Paraphrasing</td><td>64.29</td><td>51.02</td><td>20.41</td></tr><tr><td>Humanization</td><td>52.22</td><td>26.67</td><td>68.89</td></tr><tr><td>Back-translation</td><td>61.29</td><td>6.45</td><td>70.97</td></tr><tr><td rowspan="3">DeTeCtive</td><td>Paraphrasing</td><td>72.52</td><td>44.90</td><td>25.68</td></tr><tr><td>Humanization</td><td>52.27</td><td>24.44</td><td>50.11</td></tr><tr><td>Back-translation</td><td>76.26</td><td>54.84</td><td>20.80</td></tr><tr><td rowspan="3">SeqXGPT</td><td>Paraphrasing</td><td>80.53</td><td>14.29</td><td>20.00</td></tr><tr><td>Humanization</td><td>71.07</td><td>20.00</td><td>29.84</td></tr><tr><td>Back-translation</td><td>77.09</td><td>6.45</td><td>24.46</td></tr><tr><td rowspan="3">CoCo</td><td>Paraphrasing</td><td>0.84</td><td>0.60</td><td>0.12</td></tr><tr><td>Humanization</td><td>0.59</td><td>0.34</td><td>0.42</td></tr><tr><td>Back-translation</td><td>0.88</td><td>0.67</td><td>0.07</td></tr><tr><td rowspan="3">CSFG (Ours)</td><td>Paraphrasing</td><td>0.78</td><td>0.23</td><td>0.13</td></tr><tr><td>Humanization</td><td>0.48</td><td>0.20</td><td>0.55</td></tr><tr><td>Back-translation</td><td>0.86</td><td>0.37</td><td>0.21</td></tr></table>

Table 11: Detection performance under text perturbation (Setting 3) on GPT-5.

<table><tr><td>Generator</td><td>N</td><td>Acc. (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td>Claude Sonnet 4.6</td><td>125</td><td>56.00</td><td>63.86</td><td>4.76</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>125</td><td>51.20</td><td>71.08</td><td>4.76</td></tr><tr><td>GPT-5</td><td>125</td><td>74.40</td><td>18.07</td><td>40.48</td></tr></table>

Table 12: Detection performance on documents shorter than 500 characters. FPR increases sharply relative to the 0.66% observed on the full-length Setting 1 test set, while FNR remains low for Claude Sonnet 4.6 and Gemini 2.5 Flash Lite but increases substantially for GPT-5.

Table 13: CSFG performance on human-written news text (XSum). FPR is substantially elevated relative to the 1.57% observed on the balanced Setting 1 test set.
<table><tr><td>Dataset</td><td>N</td><td>Acc. (%)</td><td>FPR (%)</td><td>FNR (%)</td></tr><tr><td>XSum</td><td>100</td><td>85.00</td><td>25.00</td><td>6.00</td></tr></table>

<table><tr><td></td><td></td><td colspan="3">SeqXGPT</td><td colspan="3">CoCo</td><td colspan="3">CSFG (Ours)</td></tr><tr><td>Generator</td><td>Perturbation</td><td>Acc</td><td>FPR</td><td>FNR</td><td>Acc</td><td>FPR</td><td>FNR</td><td>Acc</td><td>FPR</td><td>FNR</td></tr><tr><td rowspan="3">Claude Sonnet 4.6</td><td>Paraphrasing</td><td>70.48</td><td>9.80</td><td>31.57</td><td>91.55</td><td>84.75</td><td>1.67</td><td>90.59</td><td>88.24</td><td>1.22</td></tr><tr><td>Humanization</td><td>71.90</td><td>29.41</td><td>27.96</td><td>89.22</td><td>81.54</td><td>4.76</td><td>87.43</td><td>84.31</td><td>5.10</td></tr><tr><td>Back-translation</td><td>87.43</td><td>19.61</td><td>11.84</td><td>89.55</td><td>76.17</td><td>3.52</td><td>90.96</td><td>49.80</td><td>4.79</td></tr><tr><td rowspan="3">Gemini 2.5 Flash Lite</td><td>Paraphrasing</td><td>58.30</td><td>7.84</td><td>45.21</td><td>89.92</td><td>69.51</td><td>3.53</td><td>85.98</td><td>92.16</td><td>5.91</td></tr><tr><td>Humanization</td><td>64.64</td><td>37.25</td><td>35.16</td><td>79.35</td><td>76.17</td><td>14.83</td><td>76.43</td><td>82.35</td><td>17.48</td></tr><tr><td>Back-translation</td><td>89.50</td><td>9.80</td><td>10.57</td><td>90.91</td><td>95.92</td><td>1.24</td><td>91.53</td><td>37.25</td><td>5.49</td></tr><tr><td rowspan="3">GPT-5</td><td>Paraphrasing</td><td>80.53</td><td>14.29</td><td>20.00</td><td>84.04</td><td>60.25</td><td>11.92</td><td>77.86</td><td>36.73</td><td>20.63</td></tr><tr><td>Humanization</td><td>71.07</td><td>20.00</td><td>29.84</td><td>59.05</td><td>34.44</td><td>41.62</td><td>47.93</td><td>20.00</td><td>55.35</td></tr><tr><td>Back-translation</td><td>77.09</td><td>6.45</td><td>24.46</td><td>88.37</td><td>66.54</td><td>6.64</td><td>86.31</td><td>22.58</td><td>12.84</td></tr></table>

Table 14: Cross-generator summary of Setting 3 robustness results (%). FPR is measured on perturbed humanwritten text. Detailed per-generator results are reported separately in Table 4, 10, and 11.

<table><tr><td>Set.</td><td>Condition</td><td>Prompt</td></tr><tr><td>2</td><td>Generation</td><td>Please provide a continuation for the following content to make it coherent: {first 30 tokens}</td></tr><tr><td></td><td>Paraphrase</td><td>Paraphrase the following text: {Text}</td></tr><tr><td>3</td><td>Humanization</td><td>Paraphrase the given sentence. Only return the paraphrased sentence in your response. Make it seem like a human wrote the article and that it is from the YOUR SECTION section of YOUR PUBLICATION. Sentence to paraphrase:</td></tr><tr><td></td><td>Back-translation</td><td>YOUR SENTENCE Your paraphrase of the sentence: {Text} You are a professional translator. Translate the following English text into Chinese. Rules: (1) Preserve the original meaning and tone faithfully. (2) Do not add explanations or commentary. (3) Output only the translated Chinese text. Text: {input_text}</td></tr></table>

Table 15: Prompts used for text generation (Setting 2) and perturbation (Setting 3).