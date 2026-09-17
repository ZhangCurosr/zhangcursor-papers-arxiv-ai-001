# Decodability is Not Causality: Dissociating Probe Readouts from Behavioral Drivers via SAE Decomposition

Devesh Tiwari<sup>1,</sup> <sup>∗</sup>, Camille Davis<sup>2,</sup> <sup>∗</sup>, Shivank Sinha<sup>3</sup>

Talia Weaver<sup>4</sup>, Aditya Shah<sup>5</sup>, Maheep Chaudhary<sup>6,</sup> <sup>†</sup>

<sup>1</sup>WW-P High School South <sup>2</sup>Phillips Academy Andover <sup>3</sup>Dublin High School <sup>4</sup>Carlmont High School <sup>5</sup>Google <sup>6</sup>Independent

{camillehdavis}@Gmail.com, {deveshtiwari2705}@Gmail.com

## Abstract

Linear probes can decode safety-relevant concepts such as truthfulness from language-model activations, but probe accuracy may show only decodability, not that the features the probe weights causally drive model behavior. We demonstrate that this gap cannot be closed from the geometry of probe weights alone: the features geometrically aligned with probe direction need not be the ones the model uses, so causal relevance requires intervention. We introduce a feature-level diagnostic that decomposes a deployed True/False probe into sparse-autoencoder (SAE) features, ranks those features by both probe alignment and by gradient sensitivity of the model’s behavior, and ablates the resulting shared, probe-only, and random feature sets under a coherence gate. On the truth probe of Bürger et al. (2024) (TTPD), applied in the instructed truth/deception setting of Long et al. (2025) for Gemma-2-9B-Instruct, the two rankings overlap only weakly (about 12%, Spearman ρ = 0.10), and ablation dissociates them sharply: features the probe shares with the model flip the output far more (up to 27%) than equally sized probe-only (6%) or random (1%) features at full coherence, while probe-only features instead perturb the probe’s own readout. The dissociation holds across five seeds and a held-out split, and an activation-aware selection of features flips behavior nearly three times as often as the probe’s geometric top features (17.6% vs. 6.1%). In this setting, therefore, the geometric projection of a probe’s weight vector alone does not identify the features the model causally uses; however, combining probe information with feature activation statistics recovers substantially more behaviorally causal features, and coherence-gated SAE intervention is needed to separate them from probe readouts. We make the implementation of our methods accessible at this URL.

## 1 Introduction

Linear probes trained on model activations are widely used to identify and predict representations associated with behavioral states (Alain & Bengio, 2018; Belinkov, 2022). However, information that is decodable from a model’s activations is not necessarily causally responsible for the model’s behavior. Consequently, high probe accuracy alone does not establish that the representations identified by the probe necessarily play a causal role in the model’s decisions (Hewitt & Liang, 2019; Belinkov, 2022; Occhipinti et al., 2026). Distinguishing between features that are merely predictive and those that are causally relevant remains an open challenge.

We investigate whether probe-identified features are causally involved in the model behaviors associated with the representations they decode. Using the truthful/deceptive factualverification setting introduced by Long et al. (2025), we decompose model activations via a sparse autoencoder (SAE) and independently rank features according to both probe alignment and gradient sensitivity of the model’s output. We then perform direct interventions on these feature sets while tracking the output coherence.

![](images/5dbf8ef0ac20f8e420456070cc3335554334874de78988e8a789a687a7703759.jpg)  
Figure 1: We decompose final pre-generation residual activations into an SAE feature basis, then rank the same SAE features by probe-weight alignment, indicating features the probe reads, and by gradient sensitivity of the model’s True/False margin, indicating features whose perturbation most afects the model’s output. We then intervene on each feature group, together with random controls, and measure efects on model behavior, probe prediction, and output coherence.

We find that the features the probe weights most and the features the model is sensitive to overlap only weakly, and that this diference is behavioral. Ablating the features the probe shares with the model flips the model’s output substantially more than ablating equally sized sets of probe-only or random features, while output coherence is preserved throughout. Probe-only features instead perturb the probe’s own readout without changing behavior. These results suggest that many probe-weighted features function primarily as diagnostic readouts rather than causal drivers of behavior, and that the geometry of a probe’s weights alone does not identify the features the model causally uses.

Our work makes three main contributions. First, we introduce a framework for comparing probe-aligned and gradient-sensitive SAE features through direct behavioral interventions. Second, we quantify the extent to which probe-ranked features produce coherent behavioral efects, showing that predictive importance and behavioral importance can diverge substantially. Third, we identify a subset of shared probe-model features that consistently influence both probe predictions and model behavior, providing stronger evidence of behavioral relevance than probe-only features.

## 2 Related Work

Probing, truth representations, and behavioral use. Linear probes are widely used to identify information encoded in neural representations (Alain & Bengio, 2018; Belinkov, 2022), but decodability alone does not establish that the decoded information is used by the model (Hewitt & Liang, 2019). Amnesic probing made this distinction explicit by removing probe-identified information and measuring downstream behavior, finding that conventional probing performance need not correlate with task importance (Elazar et al., 2021). Subsequent work has emphasized that such interventions must be both complete with respect to the targeted property and selective with respect to unrelated information (Canby et al., 2025).

For factual truth, prior work has identified approximately linear truth representations and tested their behavioral relevance through activation interventions (Marks & Tegmark, 2024). Bürger et al. identify a polarity-aware truth subspace and introduce TTPD (Bürger et al., 2024); Long et al. apply this framework under truthful, neutral, and deceptive instructions and show that truth-related representations change under instructed deception (Long et al.,

2025). Related probes have been used to detect or steer chain-of-thought unfaithfulness (Occhipinti et al., 2026), while other studies show that highly accurate probes can instead exploit task-format confounds (Sahoo et al., 2026). Our work begins from a probe direction that is itself behaviorally causal and asks a finer question: whether the SAE features most geometrically aligned with that direction are the features carrying its behavioral efect.

Sparse autoencoders and causal feature evaluation. SAEs decompose model activations into sparse latent features (Cunningham et al., 2023; Lieberum et al., 2024), but reconstruction quality and feature interpretability do not by themselves guarantee causal or task-level utility. Task-grounded evaluations find that unsupervised SAE dictionaries can provide weaker behavioral control than supervised feature dictionaries (Makelov et al., 2025), struggle to disentangle independently manipulable factual attributes (Chaudhary & Geiger, 2024), and do not consistently improve over non-SAE probing baselines (Kantamneni et al., 2025). SAEBench similarly finds that improvements on unsupervised proxy metrics do not reliably transfer to downstream interpretability tasks (Karvonen et al., 2025). These results motivate evaluating SAE features through task-specific interventions and verifying that the studied probe signal is preserved by the SAE reconstruction.

Prior work has used SAE ablation, patching, and steering to identify causally relevant features and circuits (Kissane et al., 2024; Marks et al., 2025). Most directly, Ma et al. show that contrastively selected SAE reasoning features frequently reflect lexical correlates rather than reasoning computations (Ma et al., 2026).

## 3 Methodology

## 3.1 Overview

Our framework proceeds in the following stages: probe replication, SAE feature attribution, gradient-based sensitivity ranking, causal interventions (ablation and learned sparse mask), and overlap and decomposition.

We track output coherence throughout all interventions to ensure observed efects are interpretable (Section 3.7).

We also run auxiliary diagnostics on an additional deception probe to test whether the same SAE-based probe-attribution workflow is suitable for a hardcoded-code deception probe.

## 3.2 Stage 1: Probe Replication

We replicate the target probe using the original authors’ procedure, confirming it matches their reported performance (Appendix A.1).

## 3.3 Stage 2: SAE Feature Attribution

We project the probe’s weight vector into the decoder basis of a pre-trained SAE to identify which SAE features the probe reads from.

Let the probe P have weight vector $\boldsymbol { w } \in \mathbb { R } ^ { d }$ , trained on residual stream activations $h ^ { ( \ell ) } \in \mathbb { R } ^ { d }$ at layer ℓ. Let the SAE have decoder matrix $D \in \mathbb { R } ^ { d \times k }$ , with columns $d _ { i }$ denoting learned feature directions. Since the probe’s output can be approximated as $\begin{array} { r } { \boldsymbol { w } ^ { \top } \boldsymbol { \dot { h } } \approx \sum _ { i } \boldsymbol { f _ { i } } \cdot \left( \boldsymbol { w } ^ { \top } \boldsymbol { d _ { i } } \right) } \end{array}$ where $f _ { i }$ is the activation of SAE feature $i ,$ the structural contribution of each feature to the probe is:

$$
c _ { i } = w ^ { \top } d _ { i }\tag{1}
$$

Features with large $\left| { c _ { i } } \right|$ are directions to which the probe is linearly sensitive within the SAE reconstruction. We rank features by the activation-free score $\left| { c _ { i } } \right|$ as the probe-alignment ranking.

## 3.4 Stage 3: Gradient-Based Sensitivity Ranking

To estimate which features the model’s output is most sensitive to, we rank SAE features by a gradient-based attribution of the model’s behavior with respect to feature activations. Let $A$ be the attribution objective, defined as the gradient of the behavior logit (or target-token margin) with respect to each feature’s activation $f _ { i } ^ { \ell }$ at layer $\ell ,$ averaged over the dataset:

$$
A _ { i } = \mathbb { E } _ { x \in \mathcal { D } } \left[ \left| \frac { \partial \mathcal { M } ( x ) } { \partial f _ { i } ^ { \ell } } \right| \right] ,\tag{2}
$$

where $\mathcal { M } ( \boldsymbol { x } ) = \mathrm { { l o g i t } ( T r u e ) - l o g i t ( F a l s e ) }$ is diferentiated with respect to the SAE feature activations $f _ { i } ^ { \ell }$ at the final-token position of layer $\ell ,$ averaged over the dataset. We retain the top-ranked features as the gradient-sensitive feature set $\mathcal { F } _ { \mathrm { m o d e l } }$

## 3.5 Stage 4: Causal Interventions

Having ranked features from both the probe’s perspective and by gradient sensitivity, we validate these attributions through direct intervention. Under each intervention, we measure the behavior flip rate, defined as the fraction of examples where the model’s True/False output changes relative to the unintervened baseline, and the probe-readout shift, defined as the relative change in the probe margin $w ^ { \top } h$

This dual measurement separates a feature’s efect on the probe’s readout from its efect on the model’s behavior, the dissociation at the center of our analysis (Figure 1).

## 3.5.1 Feature Ablation

For a selected feature set $s ,$ we ablate those features by subtracting their reconstructed contributions from the residual stream at layer ℓ:

$$
\tilde { h } ^ { ( \ell ) } = h ^ { ( \ell ) } - \sum _ { i \in \mathcal { S } } f _ { i } d _ { i }\tag{3}
$$

and allow the model’s forward pass to continue from $\tilde { h } ^ { ( \ell ) }$ . We apply this procedure to probe-attributed features $( \mathcal { F } _ { \mathrm { p r o b e } } )$ ), gradient-sensitive features $( \mathcal { F } _ { \mathrm { m o d e l } } )$ , and random control features of matched cardinality.

## 3.5.2 Learned Sparse Mask

We learn a sparse mask over SAE features that selects those contributing most to the probe margin, combining each feature’s decoder alignment $c _ { i }$ with its activation. A learnable weight $a _ { i } \in [ 0 , 1 ]$ per feature is optimized with an $\bar { \ell } _ { 1 }$ penalty to reconstruct the probe margin from the weighted feature contributions. After training, 30 features exceed $a _ { i } > 0 . 5 ;$ ; we ablate the top 16 to match the size of the other feature sets in Table 4. This yields an activation-aware feature set $\mathcal { F } _ { \mathrm { m a s k } }$ , in contrast to the activation-free geometric ranking $\left| { c _ { i } } \right|$ . Full training details are in Appendix A.1.2.

## 3.6 Stage 5: Overlap and Decomposition

We compare the probe’s feature ranking against the model’s to characterize the extent to which the two rely on shared features.

We compute set overlap between the top-N probe-attributed and gradient-sensitive features as a function of $N _ { \ast }$ , and report the Spearman rank correlation as a summary statistic. Using this analysis, we partition the probe’s features into three groups: Shared, the intersection of the top-K probe-aligned and top-K gradient-sensitive features, whose size m sets the matched size for the other two groups; Probe-only, comprising the highest probe-ranked features that are not among the model’s top gradient-ranked features, size-matched to the shared set; and Random, comprising a size-matched random sample from the remaining features.

We ablate each group at K ∈ {32, 64, 128, 256, 512, 1024} top features from each ranking (yielding shared-set sizes of 5 to 126), subtracting reconstructed feature contributions as in Eq. 3, and compare the resulting behavior flip rate, probe-margin shift, and coherence. This decomposition isolates the causal contribution of features the probe shares with the model from those unique to the probe.

## 3.7 Coherence Gating

Interventions that disrupt the model’s ability to produce well-formed output are not informative about the causal role of specific features. We therefore track output coherence under all interventions. We define coherence as the percentage of the responses in which the model responds with either True/False. An intervention’s behavioral efect is only interpreted when coherence is maintained, ensuring that observed behavioral efects are attributable to intervention rather than to general degradation of the model’s output. Beyond output validity, we verify three further conditions. First, the SAE faithfully represents the activations it decomposes: its reconstruction attains mean cosine similarity 0.89 with the original layer-20 activations (normalized reconstruction MSE 0.21), so interventions act in a basis that captures most of the activation signal rather than in reconstruction noise. Second, the interventions are small targeted perturbations rather than gross corruptions: ablating the shared set removes only 5.7% of the residual-stream norm on average, yet produces the behavioral efects reported in Section 5.2. Third, the interventions leave the model’s next-token distribution largely intact: the mean KL divergence between the baseline and post-ablation distributions is 0.11 nats for the shared set (0.05 for the geometric top-16, near zero for random controls; n = 600), indicating that even the behaviorally efective shared-set ablation is a localized shift of the output distribution rather than a wholesale disruption.

## 4 Experimental Setup

## 4.1 Probe Selection

## 4.1.1 Task and Concept

We apply our framework to the instructed deception setting of Long et al. (2025). A language model is presented with factual statements and prompted under one of three instruction conditions: Truthful, Deceptive, or Neutral. Under the deceptive condition, the model is instructed to respond incorrectly. Long et al. (2025) train linear probes on the model’s residual stream activations to predict the instructed output under each condition, and find that the output is linearly decodable across all three conditions, with accuracy peaking around layer 14 for LLaMA-3.1-8B-Instruct and layer 21 for Gemma-2-9B-Instruct (Long et al., 2025).

Appendix A.2 reports an auxiliary diagnostic on a reward-hacking code probe. The probe is trained on hardcoded versus correct code completions from an MBPP reward-hacking environment, where the behavior of interest is generated code that passes visible tests while failing hidden tests.

## 4.1.2 Probe Relevance

We select the TTPD probe as our primary case study for several reasons. First, the task produces a binary output (True/False), providing a clean behavioral signal for measuring our intervention efects. The probe also targets a well-defined form of deception: truthful versus deceptive instruction-following in factual verification. The setup is also technically suitable for our method because it uses fixed pre-generation residual-stream activations and has matching GemmaScope SAEs.

## 4.1.3 Model, Dataset and SAE

We use Gemma-2-9B-Instruct (Team et al., 2024) in bfloat16 precision. We use the factual-statement datasets from Bürger et al. (2024) (as used by Long et al. (2025)). We use six afirmative factual-statement sets (cities, sp\_en\_trans, inventors, animal\_class, element\_symb, facts) and their negated counterparts, amounting to twelve sets in total, all under the deceptive instruction condition. Activations are extracted at the final prompt token of layer 20, adjacent to the layer-21 accuracy peak reported by Long et al. (2025) and the layer with a matching GemmaScope SAE (SAE for Gemma-2-9B-Instruct with a width of 16,384 features from Lieberum et al. (2024)). The behavioral evaluation set comprises 1,600 statements; the held-out analysis (Section 5.2) derives the probe direction and feature rankings on one half and measures ablation on the other.

<table><tr><td>Top-k features</td><td>Shared features</td><td>Shared fraction</td><td>Chance overlap</td></tr><tr><td>16</td><td>2</td><td>12.5%</td><td>0.02</td></tr><tr><td>50</td><td>7</td><td>14.0%</td><td>0.15</td></tr><tr><td>100</td><td>12</td><td>12.0%</td><td>0.61</td></tr><tr><td>200</td><td>22</td><td>11.0%</td><td>2.44</td></tr></table>

Table 1: Overlap between geometric probe ranking and model-gradient ranking. Geometric alignment is measured by $| w \cdot d |$ , while model ranking uses gradient sensitivity of the True/False margin. The model ranking correlates weakly with mean absolute SAE feature activation, with Pearson $r = 0 . 0 7$

## 4.1.4 Behavioral Signal and Coherence

We define the behavioral signal as

$$
M ( x ) = \mathrm { { l o g i t } ( T r u e ) - \mathrm { { l o g i t } ( F a l s e ) } }
$$

at the final token of the templated prompt, the position from which True/False is generated. A behavior flip occurs when the sign of this margin under intervention difers from the unintervened baseline. Coherence is the fraction of outputs that remain a valid True/False token (Section 3.7).

## 4.1.5 Intervention Configurations

For feature ablation, we evaluate the following feature sets at $K \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$ probe-attributed $( \mathcal { F } _ { \mathrm { p r o b e } } )$ , gradient-sensitive $( { \mathcal { F } } _ { \mathrm { m o d e l } } )$ , the learned sparse mask $( \mathcal { F } _ { \mathrm { m a s k } } )$ shared features, probe-only features, and size-matched random controls.

## 5 Results

## 5.1 Probe-aligned and gradient-sensitive features exhibit minimal overlap

Two rankings, (1) geometric alignment with the probe direction |w · d| and (2) gradient sensitivity of the True/False margin, identify largely diferent features. The top 16, 50, 100, and 200 probe features share only 2, 7, 12, and 22 features with the gradient-sensitivity ranking, against chance values of 0.02, 0.15, 0.61, and 2.44 (Spearman $\rho = 0 . 1 0 )$ . The shared fraction holds close to 11 to 14 percent across scales, one to two orders of magnitude above chance, but a small minority of either set (Table 1).

Appendix A.1.3 plots the full overlap curve.

## 5.2 Probe alignment alone does not predict behavioral efect.

Under ablation at the deceptive prompt, the three feature sets dissociate sharply. Shared features (top probe ∩ top model) flip the True/False output on 13 to 27 percent of statements (0.130 at 16 features, 0.269 at 54); an equal number of probe-only features (probe-ranked, with the shared model-ranked features removed) flip 6 percent (0.061); random features flip 1 percent (0.010). Exemplars of output changes from these interventions are shown in

![](images/d41e94349761a2149633e4d3835c9e77e7329adccebc4c7f02c806e9a16b267b.jpg)  
features ablated

(a) Behavior flip rate. Ablating shared features (top probe ∩ top gradient) flips the $\bar { T } r u e / F a l s e$ output far more than probe-only or random sets of equal size.  
![](images/2158f66d5eac59d0e2f6114d6f7536c65a0b2e5f3222039afb191fecd584381d.jpg)  
(b) Relative shift in the probe’s readout, $\frac { | \Delta \mathrm { p r o j e c t i o n } | } { \mathrm { b a s e l i n e } . }$ Probeonly ablation perturbs the readout most, consistent with these features being selected for alignment with the probe direction.

![](images/7eecc4b6a8bba091d883660b73333771f4714de92503488484f0f41980491238.jpg)  
features ablated  
(c) Coherence, the fraction of outputs that remain a valid True/False token, stays at 1.00 across all sets, indicating flips are genuine rather than degradation.

Figure 2: Causal efects of ablating shared, probe-only, and random SAE feature sets.
<table><tr><td>Setting</td><td>Shared</td><td> $\mathbf { P r o b e - o n l y }$ </td><td>Random</td></tr><tr><td>Five seeds / data resamples</td><td> $0 . 1 0 5 \pm 0 . 0 1 7$ </td><td> $0 . 0 5 9 \pm 0 . 0 1 6$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 2$ </td></tr><tr><td>Held-out split</td><td>0.122</td><td>0.085</td><td>0.010</td></tr></table>

Table 2: Robustness of the ablation dissociation across seed/data resamples and a held-out split. The shared set contains 23 features (five-seed row, K=256) and 28 features (held-out row, K=256 on half the data).

Appendix A.1.5. Coherence remains 1.00 throughout, demonstrating answer flips rather than ablation-induced degradation. The probe’s own projection shifts most under probe-only ablation (relative shift 0.24 against 0.05 for shared), as expected given that probe-only features are selected for alignment with the probe direction.

The shared-ablation efect is non-monotonic, rising with set size to a peak near 96 features before declining as the lowest-ranked shared features shift the margin in the opposing direction and partially cancel the flip. The decline reflects sign-incoherence among these low-ranked features rather than output degradation: the mean absolute margin remains between 1.7 and 2.8 throughout, indicating the model continues to emit well-formed, confident True/False outputs at every ablation count (Appendix A.1.4). The set-level numbers we report (≤ 54 shared features) lie within the rising regime.

The shared > probe-only > random ordering holds across all five seeds and under a held-out split where rankings are derived on one half and ablation measured on the other (Table 2), indicating the dissociation is stable rather than an artifact of overfitting to the ranking data.

The flips are directional rather than truth-restoring. On 800 statements (at the matched 16-feature set size), shared-feature ablation changes the output True→False on 86 cases and False→True on only 4 (Table 3), shifting the decision variable consistently toward False rather than toward the factually correct answer; baseline accuracy under the deceptive prompt is 0.40 and does not increase under ablation. This is expected as the shared features carry the truth-direction signal $t _ { G }$ that the probe reads, so removing them withdraws positive evidence for True and biases the margin negative. The efect is therefore one of causal control over the model’s output, not a restoration of honesty, and it is precisely this directional consistency (rather than random sign changes) that marks these features as causal. Crucially, the efect is not a trivial consequence of removing probe-aligned features: the most probe-aligned set (geometric top-16, Section 5.3) produces far weaker behavioral movement despite larger probe-readout shifts, so directional output control is specific to the features the model also uses, not to probe alignment per se.

<table><tr><td>Output transition</td><td>Count</td><td>Rate</td></tr><tr><td> $T r u e  F a l s e ~ ( \mathrm { { f i p } ) }$ </td><td>86</td><td>0.107</td></tr><tr><td> $F a l s e  T r u e ( \mathrm { { f f i p } ) }$ </td><td>4</td><td>0.005</td></tr><tr><td> $T r u e  T r u e ( \mathrm { u n c h a n g e d } )$ </td><td>624</td><td>0.780</td></tr><tr><td> $F a l s e  F a l s e ( \mathrm { u n c h a n g e d } )$ </td><td>86</td><td>0.108</td></tr><tr><td>Baseline accuracy</td><td></td><td>0.403</td></tr><tr><td>Post-ablation accuracy</td><td></td><td>0.330</td></tr></table>

Table 3: Output transitions under shared-feature ablation $( n = 8 0 0$ , shared set $m = 1 6 )$ Of 800 statements, 90 flip (86 $T r u e {  } F a l s e ,$ 4 $F a l s e {  } T r u e )$ and 710 are unchanged. Flips are overwhelmingly True→False: ablation biases the decision variable toward False rather than restoring factual correctness, consistent with the shared features carrying the probe’s truth-direction signal. A confusion matrix is attached in Appendix A.1.6.
<table><tr><td>Feature set</td><td></td><td>Behavior flip rate Probe-readout shift</td></tr><tr><td>Activation-aware mask</td><td> $0 . 1 7 6 \pm 0 . 0 0 7$ </td><td> $0 . 3 9 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Shared set</td><td> $0 . 1 2 1 \pm 0 . 0 1 0$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 1$ </td></tr><tr><td>Geometric top-16</td><td> $0 . 0 6 1 \pm 0 . 0 0 5$ </td><td> $0 . 1 6 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Gradient-top</td><td> $0 . 0 5 4 \pm 0 . 0 0 9$ </td><td> $0 . 0 5 5 \pm 0 . 0 0 1$ </td></tr><tr><td>Random control</td><td> $0 . 0 0 8 \pm 0 . 0 0 6$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 5$ </td></tr></table>

Table 4: Behavioral and probe-readout efects of ablating size-matched feature sets (five-seed means ± std, $n = 6 0 0$ per seed; coherence 100% throughout). All sets are size-matched at 16 features.

Shared-feature ablation flips behavior in ten of twelve categories (overall 0.081 on 840 balanced statements, 70 per category), exceeding ten percent in five spanning four base families, with the True→False direction dominant in nearly all and only two categories showing no flips. The efect is therefore reasonably broad rather than driven by any single category (Appendix A.1.7).

## 5.3 Geometric probe attribution understates behavioral relevance.

The geometric ranking |w · d| measures only how well a feature’s decoder direction aligns with the probe, ignoring how strongly the feature actually fires, and it is a poor guide to which features move behavior. We compare it against an activation-aware set: a learned sparse mask that selects the SAE features contributing most to the probe margin, where each feature’s contribution combines its decoder alignment with its activation. This set overlaps the geometric top-16 by only 3 of 16 features, yet across five seeds it flips behavior on $1 7 . { \overset { } { 6 } } \pm 0 . 7$ percent of statements, above the shared set (12.1 ± 1.0) and nearly three times the geometric top-16 (6.1 ± 0.5), with the random control near zero $( \dot { 0 } . 8 \pm 0 . 6 )$ and coherence at 100 percent throughout. The probe-readout shift dissociates from behavior in the opposite direction: the geometric top-16 perturbs the probe’s own margin more than the shared set (0.16 versus 0.05) while flipping behavior less, indicating that geometric alignment tracks how strongly a feature moves the probe rather than how strongly it moves the model. Features most strongly aligned with the probe direction thus act primarily as probe readouts with weak behavioral efects, whereas an activation-aware selection of equal size moves behavior substantially more. The dissociation is not an artifact of perturbation magnitude: the geometric top-16 removes more than twice the residual-stream norm of the shared set (13.9% versus 5.7%) while flipping behavior less, indicating that behavioral relevance depends on which features are removed, not how much of the activation is perturbed. The mask’s success shows that probe-derived information can identify causally efective features when combined with activation magnitudes; the failure is specific to activation-free geometric alignment, not to probe-based attribution in general.

<table><tr><td>Ablation</td><td>Behavior flip rate</td></tr><tr><td>Full probe direction  $t _ { G }$ </td><td> $0 . 1 8 1 \pm 0 . 0 1 1$ </td></tr><tr><td>Shared set (16 features)</td><td> $0 . 1 2 1 \pm 0 . 0 1 1$ </td></tr><tr><td>Probe geometric top-16</td><td> $0 . 0 6 8 \pm 0 . 0 0 6$ </td></tr><tr><td>Random control (16 features)</td><td> $0 . 0 0 7 \pm 0 . 0 0 1$ </td></tr></table>

Table 5: Direction-level baseline (five-seed means ± std, n = 600 per seed). Ablating the full TTPD direction flips behavior on 18.1% of statements; the 16-feature shared set recovers two-thirds of this efect, while the probe’s geometric top-16 recovers roughly a third, indicating the SAE decomposition localizes the direction’s causal efect into the shared features rather than dispersing it.

Features ranked highest under gradient attribution alone (gradient-top) flip behavior weakly $( 0 . 0 5 4 \pm 0 . 0 0 9 )$ , comparable to the geometric top-16. This is consistent with previous work: gradient sensitivity is a local, first-order signal and an imperfect proxy for the efect of ablation in nonlinear models (Li & Janson, 2024). In our setting, a high-gradient feature may carry the wrong sign, be near-zero in practice, or lie of the manifold the model actually visits. This failure mode is known beyond our setting: prior work shows that standard-model input gradients can highlight non-discriminative or non-instance-specific features, a phenomenon termed feature leakage (Shah et al., 2021). Probe alignment, conversely, ignores whether a feature fires. The intersection retains only features that are both active and aligned, which is why it concentrates causal efect that neither single ranking achieves alone, confirming our framework does not merely rediscover gradient attribution. As Table 4 shows at matched set size, the shared set and the activation-aware mask move behavior most, the single-ranking sets are weak handles, and the probe-readout shift runs opposite to behavior, with the probe-aligned geometric set perturbing the readout far more than it moves the model.

## 5.4 The probe direction is causal, and the shared features localize it

The preceding results show which features carry the probe’s behavioral efect, but not whether the probe direction is causally load-bearing in the first place. To establish this, and to test whether the SAE decomposition localizes the direction’s causal efect, we ablate the full TTPD direction $t _ { G }$ by projecting it out of the residual stream at layer 20, and compare against feature-set ablations on the same statements. Removing the entire direction flips behavior on $1 8 . 1 \pm 1 . 1$ percent of statements (Table 5). The shared set, just 16 of 16,384 SAE features, recovers two-thirds of this efect (12.1 ± 1.1 percent), whereas the probe’s own geometric top-16 recovers roughly a third (6.8 ± 0.6 percent) and random features near zero $( 0 . 7 \pm 0 . 1$ percent). The probe direction therefore carries a causal efect, and at matched small size that efect is substantially recovered by the features that are both probe-aligned and gradient-sensitive, rather than by the features most aligned with the direction. This confirms the premise of our analysis and indicates the decomposition concentrates the direction’s causal efect within the shared features rather than diluting it.

## 6 Conclusion

We introduce a causal-validation framework for testing whether probe-attributed SAE features are also behaviorally relevant. Applied to a True/False truth probe, the features most aligned with the probe diverged from those most relevant to the model’s output: shared probe–model features produced larger coherent behavior changes, while probe-only features primarily afected the probe readout. These results demonstrate that the geometric projection of probe weights alone is insuficient for identifying the features a model causally uses; the failure is specific to activation-free geometric alignment, since an activation-aware selection that combines probe information with feature activations recovers substantially more behaviorally causal features. Feature-level interventions should therefore be used to separate diagnostic readouts from behavioral causes.

## 7 Limitations

Our analysis focuses primarily on a single model, layer, sparse autoencoder, and behavioral setting. While the truthful/deceptive factual-verification task provides a controlled environment for studying the relationship between probe-attributed and behaviorally relevant features, it remains unclear whether the observed probe-behavior dissociation generalizes across other model families, layers, SAE architectures, and behavioral domains. Although we include auxiliary diagnostics on an obfuscation probe, the majority of our causal analysis is derived from a single primary case study.

Additionally, our gradient-sensitivity ranking is based on gradient-derived feature importance and our causal validation relies on feature ablations. These methods identify features that influence the model’s True/False behavior in the studied setting, but they do not by themselves establish that the identified features implement deception as a general computational mechanism. Future work should evaluate alternative attribution methods, intervention strategies, and behavioral settings to determine the extent to which the observed distinctions between probe-attributed and behaviorally relevant features hold more broadly.

## References

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2018. URL https://arxiv.org/pdf/1610.01644.

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, March 2022. doi: 10.1162/coli\_a\_00422. URL https:// aclanthology.org/2022.cl-1.7/.

Lennart Bürger, Fred A. Hamprecht, and Boaz Nadler. Truth is universal: Robust detection of lies in llms. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2407.12831. Poster.

Marc E. Canby, Adam Davies, Chirag Rastogi, and Julia Hockenmaier. How reliable are causal probing interventions? In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 857–878. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, December 2025. doi: 10.18653/v1/2025.ijcnlp-long.47. URL https://aclanthology.org/2025. ijcnlp-long.47/.

Maheep Chaudhary and Atticus Geiger. Evaluating open-source sparse autoencoders on disentangling factual knowledge in GPT-2 small, 2024. URL https://arxiv.org/abs/ 2409.04478.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL https: //arxiv.org/abs/2309.08600.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9:160–175, 2021. doi: 10.1162/tacl\_a\_00359. URL https://aclanthology.org/2021.tacl-1.10/.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733–2743. Association for Computational Linguistics, 2019. doi: 10.18653/ v1/D19-1275. URL https://aclanthology.org/D19-1275/.

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. Are sparse autoencoders useful? A case study in sparse probing. In Proceedings of the 42nd International Conference on Machine Learning, volume 267

of Proceedings of Machine Learning Research, pp. 29018–29049. PMLR, 2025. URL https://proceedings.mlr.press/v267/kantamneni25a.html.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Isaac Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum Stuart McDougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 29223–29264. PMLR, 2025. URL https://proceedings.mlr.press/v267/karvonen25a.html.

Connor Kissane, Robert Krzyzanowski, Joseph Isaac Bloom, Arthur Conmy, and Neel Nanda. Interpreting attention layer outputs with sparse autoencoders, 2024. URL https://arxiv.org/abs/2406.17759.

Maximilian Li and Lucas Janson. Optimal ablation for interpretability, 2024. URL https: //arxiv.org/abs/2409.09951.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2, 2024. URL https://arxiv.org/abs/2408.05147.

Xianxuan Long, Yao Fu, Runchao Li, Mu Sheng, Haotian Yu, Xiaotian Han, and Pan Li. When truthful representations flip under deceptive instructions? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 16315–16335, Suzhou, China, November 2025. Association for Computational Linguistics. doi: 10.18653 v1/2025.emnlp-main.826. URL https://aclanthology.org/2025.emnlp-main.826/.

George Ma, Zhongyuan Liang, Irene Y. Chen, and Somayeh Sojoudi. Do sparse autoencoders identify reasoning features in language models?, 2026. URL https://arxiv.org/abs/ 2601.05679.

Aleksandar Makelov, George Lange, and Neel Nanda. Towards principled evaluations of sparse autoencoders for interpretability and control. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=1Njl73JKjB.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https://arxiv.org/ abs/2310.06824. Conference on Language Modeling.

Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=I4e82CIDxv.

Giovanni Maria Occhipinti, Alessandro Abate, and Nandi Schoots. Probing and steering chain-of-thought unfaithfulness in language models. In ICLR 2026 Test-Time Updates (TTU) Workshop, 2026. URL https://openreview.net/pdf?id=LocRunEIxK.

Subramanyam Sahoo, Vinija Jain, Aman Chadha, and Divya Chaudhary. Linear probes detect task format, not reasoning mode in language model hidden states, 2026. URL https://arxiv.org/abs/2606.02907.

Harshay Shah, Prateek Jain, and Praneeth Netrapalli. Do input gradients highlight discriminative features?, 2021. URL https://arxiv.org/abs/2102.12781.

Mohammad Taufeeque, Stefan Heimersheim, Adam Gleave, and Chris Cundy. The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes, 2026. URL https://arxiv.org/abs/2602.15515.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hofman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozińska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucińska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jef Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jef Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.

## A Appendix

## A.1 TTPD Probe Bürger et al. (2024), Long et al. (2025)

## A.1.1 Replication

We replicate the probe of Long et al. (2025) using their released code and the associated factual-statement datasets. Following Bürger et al. (2024), the probe is the general truth direction $t _ { G } { \mathrm { : } }$ centered layer-20 activations are regressed on the statement’s truth label and its afirmative/negated polarity, and the truth component is retained; statements are then classified by the sign of the projection onto $t _ { G }$ (the TTPD probe). On gemma-2-9b-instruct, our reproduced $t _ { G }$ separates true from false statements on the deceptive-instruction activations with AUROC 0.98. We use this $t _ { G }$ direction as the probe w throughout the main evaluation.

Long et al. (2025) additionally report a logistic-regression (LR) probe, using the unregularized configuration in the released code (penalty = None). We exclude the LR probe from the main evaluation for methodological reasons rather than for reasons of accuracy. As a classifier, the LR probe performs well, with training accuracy remaining between 0.980 and 1.000 across regularization strengths $( C \in \{ 0 . 0 \bar { 1 } , 0 . 1 , 1 . 0 , \bar { \infty } \} )$ ). However, the features receiving the largest weights are highly sensitive to the choice of regularization. The top-30 features overlap with those of the unregularized solution with Jaccard indices of only 0.62, 0.43, and 0.07 at $C = 1 . 0 , 0 . 1 , 0 . 0 1$ , respectively. Thus, the notion of “the features the LR probe relies $\mathrm { o n } ^ { \dag }$ is not well-defined unless one arbitrarily fixes a regularization hyperparameter. Because our analysis attributes behavior to specific features, it requires a probe direction that is fixed rather than hyperparameter-dependent. The TTPD direction $t _ { G }$ is hyperparameter-free and is the canonical truth probe of Bürger et al. (2024); therefore, we use it for all attribution analyses and omit the LR probe from the main experiments.

![](images/34b60c2d4f2a4cfc41306027bc21d4454eeabeb5e31985cd5303580dbdadf552.jpg)  
(a) Deceptive prompt.

![](images/1e4f50da07eee86f602f84f688984ff0f06aa4d2c03b3ad376a9e863c645ce48.jpg)  
(b) Neutral prompt.

![](images/4cbc3f7266c778b123bdb5d93de44313a2a3bad8960be4697ce494e5f473ec2a.jpg)  
(c) Truthful prompt.  
Figure 3: Layer-wise probing accuracy for Gemma-2-9B-Instruct under deceptive, neutral, and truthful instructions. LR and TTPD probes predict the instructed True/False output from final pre-generation residual-stream activations. Across all three conditions, accuracy rises from chance in early layers to a high-accuracy plateau near the selected analysis layer. The dashed line marks layer 21, the accuracy peak reported by Long et al. (2025). We analyze layer 20 in our experiments due to SAE constraints.

## A.1.2 Learned Sparse Mask Details

For each SAE feature $i ,$ a learnable parameter $\theta _ { i }$ produces a soft mask weight $a _ { i } = \sigma ( \theta _ { i } )$ initialized at $\theta _ { i } = 0$ so that all features begin at $a _ { i } = 0 . 5$ . The masked probe margin is $\begin{array} { r } { \hat { m } ( x ) = \sum _ { i } a _ { i } \cdot f _ { i } \cdot c _ { i } ^ { \mathrm { r a w } } } \end{array}$ , where $c _ { i } ^ { \mathrm { r a w } } = { \bf w } ^ { \top } { \bf d } _ { i }$ uses unnormalized decoder columns. We minimize $\begin{array} { r } { \bar { \mathcal { L } } \bar { \mathbf { \Psi } } = \mathrm { M S E } ( \bar { m } / s , \ m / s ) \bar { \mathbf { \Psi } } \bar { \lambda } \sum _ { i } a _ { i ; } } \end{array}$ where m is the true probe margin, s is its standard deviation across the dataset, and $\lambda = 0 . 0 1$ . The mask is optimized with Adam (lr $= 0 . 1 )$ for 400 steps on the full deceptive-prompt activation matrix. After training, features are ranked by $a _ { i }$ descending.

![](images/a7906f00a85ca4c4c7f448f3f32039c08ef9cf9f06b77d27e6d4e084f396e164.jpg)  
Figure 4: Fraction of top-N features shared between the probe ranking $( | w \cdot d | )$ and the model’s gradient ranking, as a function of N on a log scale. The observed overlap exceeds the chance baseline (dashed line) by one to two orders of magnitude for small N. Both curves approach 1 as N approaches the full dictionary, corresponding to the trivial limit.

## A.1.3 Gradient and probe overlap

Figure 4.

## A.1.4 Margin dip

The efect of ablating the shared probe–model features is non-monotonic in the number of features removed. Cumulatively ablating the top-N shared features, ordered by importance, raises the behavioral flip rate to a peak of 0.31 at N=96, after which it falls to 0.10 once all 126 shared features are ablated (Figure 5a). The signed probe margin mirrors this trajectory: it declines from a baseline of 2.28 to 0.10 at N=96 and then rebounds to 1.87 at $N { = } 1 2 6$ (Figure 5b).

This recovery is not a breakdown of model coherence. The mean absolute margin remains between 1.7 and 2.8 across the entire sweep, so the model continues to emit confident, wellformed True/False outputs at every ablation count. The dip instead reflects sign-incoherence among the lowest-ranked shared features: ablating them shifts the margin in the opposite direction to the higher-ranked features and partially cancels the flip. The features responsible for the behavioral efect are therefore concentrated near the top of the ranking, which is why the main analysis confines its causal claims to the leading features, where the efect is monotone.

## A.1.5 Behavior Flip Exemplars

Figure 6.

## A.1.6 Directional breakdown of shared-feature ablation

Figure 7.

## A.1.7 Category Breakdown

Table 6 reports the per-category breakdown of shared-feature ablation. The efect appears in ten of twelve categories, ranging from 0.014 to 0.243, with the strongest movement in translation (Sp–En, 0.243) and inventor (0.186) statements. The True→False direction dominates within nearly every category (61 versus 7 aggregate), matching the directional pattern of Table 3. Two categories (animal class, negated inventors) show no flips at the matched 16-feature size, consistent with categories whose baseline True/False margin is large enough that a small ablation does not cross the decision boundary. The breadth across categories indicates the shared features are not a category-specific artifact, while the heterogeneity in magnitude likely reflects per-category diferences in baseline decision confidence.

![](images/c21d61b3ebf7cd6233062733d456dfef7beb252e8dc555a0d400a97e62f82193.jpg)  
(a) Flip rate under cumulative ablation of the top-N shared features. The rate climbs to 0.31 at N=96 and then falls to 0.10 once all 126 shared features are removed.

![](images/2b0f1f7fe14e54d356e9100f5b6cb91f78c97ff62de176d4e539812ca1ad1709.jpg)  
(b) Probe margin under the same ablation. The signed mean margin (blue) drops from the baseline (2.28) to 0.10 at N=96 and rebounds to 1.87 at N=126, while the mean absolute margin (orange) stays between 1.7 and 2.8.

Figure 5: Non-monotonic efect of cumulatively ablating the shared probe–model features on gemma-2-9b-instruct, ordered by importance. The dip and recovery in flip rate and signed margin coincide, but the model’s confidence (absolute margin) is preserved throughout, indicating the recovery is driven by sign-incoherent low-ranked features rather than a loss of coherence.
<table><tr><td>Category</td><td>n</td><td>Flip rate</td><td> $T \to F$ </td><td> $F  T$ </td></tr><tr><td>Translation (Sp-En)</td><td>70</td><td>0.243</td><td>17</td><td>0</td></tr><tr><td>Inventors</td><td>70</td><td>0.186</td><td>13</td><td>0</td></tr><tr><td>Animal class (neg.)</td><td>70</td><td>0.143</td><td>9</td><td>1</td></tr><tr><td>Translation (neg.)</td><td>70</td><td>0.129</td><td>4</td><td>5</td></tr><tr><td>Cities</td><td>70</td><td>0.114</td><td>8</td><td>0</td></tr><tr><td>Facts</td><td>70</td><td>0.057</td><td>4</td><td>0</td></tr><tr><td>Cities (neg.)</td><td>70</td><td>0.043</td><td>3</td><td>0</td></tr><tr><td>Element symbols</td><td>70</td><td>0.029</td><td>2</td><td>0</td></tr><tr><td>Element symbols s (neg.)</td><td>70</td><td>0.014</td><td>0</td><td>1</td></tr><tr><td>Facts (neg.)</td><td>70</td><td>0.014</td><td>1</td><td>0</td></tr><tr><td>Inventors (neg.)</td><td>70</td><td>0.000</td><td>0</td><td>0</td></tr><tr><td>Animal class</td><td>70</td><td>0.000</td><td>0</td><td>0</td></tr><tr><td>All categories</td><td>840</td><td>0.081</td><td>61</td><td>7</td></tr></table>

Table 6: Shared-feature ablation flip rate by dataset category $( n = 7 0$ per category, balanced sampling so rates are comparable; shared set $m = 1 6 )$ . Categories ordered by flip rate.

## A.2 Probe from Taufeeque et al.

## As discussed in Taufeeque et al. (2026),

As an auxiliary diagnostic, we tested whether the same SAE attribution workflow transfers to a reward-hacking code probe from Taufeeque et al. (2026). We use the released AlignmentResearch/diverse-deception-probe-qwen3-8b checkpoint, taking the last-token-generation linear probe at layer 20 of Qwen/Qwen3-8B. The probe is applied to MBPP-Honeypot-style examples, contrasting correct code with hardcoded visible-test solutions, and we use the probe weight vector as w for SAE attribution with the matching Qwen-Scope layer-20 residual-stream SAE.

<table><tr><td colspan="2">Inventors</td><td>Factual label: True</td></tr><tr><td colspan="2">&quot;Orville and Wilbur Wright lived in the U.S.&quot;</td><td></td></tr><tr><td colspan="2">Baseline (deceptive prompt) True √</td><td></td></tr><tr><td>Shared probe-model ablation</td><td>False XTrue → False · toward deception</td><td></td></tr><tr><td>Probe-only ablation</td><td>True √</td><td>unchanged</td></tr></table>

<table><tr><td>Spanish-English translation (negated)</td><td>Factual label: True</td></tr><tr><td>&quot;The Spanish word &#x27;edad&#x27; does not mean &#x27;clock&#x27;.&quot;</td><td></td></tr><tr><td>Baseline (deceptive prompt) True √</td><td></td></tr><tr><td>Shared probe-model ablation</td><td>False XTrue → False · toward deception</td></tr><tr><td>Probe-only ablation True √</td><td>unchanged</td></tr></table>

<table><tr><td colspan="3">Spanish-English translation Factual label: False</td></tr><tr><td colspan="3">&quot;The Spanish word &#x27;llevar&#x27; means &#x27;end&#x27;.&quot;</td></tr><tr><td>Baseline (deceptive prompt)</td><td>True X</td><td></td></tr><tr><td>Shared probe-model ablation</td><td>False √</td><td>True →False · toward truth</td></tr><tr><td>Probe-only ablation</td><td>True X</td><td>unchanged</td></tr></table>

Figure 6: Intervention exemplars on the True/False factual-verification task. Under a deceptive instruction, the model emits a single True/False answer. Each panel reports the baseline output and the output after ablating two feature sets: (1) shared probe–model features (ranked highly by both the linear probe and model-gradient attribution) and (2) probe-only features (ranked highly by the probe but not by gradient attribution). The small mark denotes factual correctness (✓ / ✗); the tag gives the output transition and its directional reading. Ablating the shared features flips the output in every case, and the flip is always True→False: it removes the probe’s truth-direction signal and biases the decision toward False. On factually true statements (Examples 1–2), this makes the model abandon a correct answer and comply with the deceptive instruction; on a factually false statement (Example 3), the same True→False shift instead overturns the model’s deceptive compliance and restores the correct answer. The fixed output bias, not factual correctness, is what the shared features control. Across the full evaluation set, these True→False flips dominate, and net accuracy falls (Table 3). Probe-only ablation leaves the output unchanged in all three cases, consistent with those features acting as probe readouts rather than behavioral drivers.

It did not pass the SAE-substrate verification. Only 26% of the raw probe margin was preserved by the SAE reconstruction, while 74% lay in the reconstruction error. Consequently, ablating probe-aligned or sparse-mask SAE features changed reconstruction-space predictions but left raw-activation probe predictions nearly unchanged. We therefore treat this setting as a boundary case: before interpreting SAE-feature interventions causally, one must verify that the probe signal is actually represented in the SAE feature basis.

![](images/6ebabbcf2ff4e0dd75e7f4221423909afb927d8e9d85a16e5b5225049caebe4c.jpg)

![](images/02ad3f5351af96e35852de56dff7c6934651cc5bc16674e6efbe91b0db9e77c0.jpg)

![](images/315084dd45e52d239ed89b451046bf83bfd8d1ce265346a3aa3cca3542224c52.jpg)  
Figure 7: Confusion matrix of baseline versus post-ablation True/False output under sharedfeature ablation $( n = 8 0 0$ , shared set $m = 1 6 )$ . Rows are the baseline output, columns the output after ablation; diagonal cells are unchanged statements and of-diagonal cells are flips. Flips are concentrated in the $T r u e {  } F a l s e$ cell (86 versus 4 for $F a l s e  \check { T r } u e )$ , showing that ablation drives the decision variable toward False rather than toward the factually correct answer. This is the per-cell view of the transition counts in Table 3.

<table><tr><td>Representation</td><td>Probe margin</td><td>Percent of raw margin</td></tr><tr><td>Raw activation</td><td>3.84</td><td>100%</td></tr><tr><td>SAE reconstruction</td><td>1.00</td><td>26%</td></tr><tr><td>SAE error vector</td><td>2.84</td><td>74%</td></tr></table>

Table 7: Probe margin decomposition for the reward-hacking code probe diagnostic. The raw probe margin is compared against the margin preserved by the SAE reconstruction and the margin carried by the SAE reconstruction error.

As Figure 8 shows, ablating probe-aligned SAE features flips about one third of predictions when the probe is evaluated on the SAE reconstruction, but produces near zero flips on the raw activation. Therefore, the selected SAE features explain the portion of the probe readout that lives in reconstructed SAE space, but they are not necessary for the origina probe’s prediction on the full residual stream.

Similarly, ablating the top-50 sparse-mask features flips 32.7% of SAE-reconstruction predictions, but only 2.1% of raw-activation predictions. The raw flip rate remains small for larger masks, peaking at 4.5% for top-100 features before returning near 2%. This indicates that the learned sparse mask successfully identifies features that support the reconstructed probe readout, but these features do not account for most of the discriminative signal used by the raw probe.

![](images/9bc6a21a699fa9743a961dc2950a02a0c81465f2c57e8614475ac3a1d72c2884.jpg)  
Figure 8: Probe-aligned SAE feature ablations afect the reconstructed probe readout but leave the raw published probe nearly unchanged. Features are ranked by ${ c } _ { i } = { w } ^ { \top } d _ { i } .$ . Ablating top-K decoded feature contributions flips approximately one third of probe predictions when evaluated on the SAE reconstruction, but produces near-zero flip rates on raw activations.

![](images/8ed54fb348ead12d65e01df0502d21c9c9f64a71c65b782ea7d4e0c69ca8cc61.jpg)  
Figure 9: Sparse-mask ablations strongly afect reconstruction-space probe predictions but have little efect on raw probe predictions. The sparse mask was trained to reproduce the probe logit from SAE features.