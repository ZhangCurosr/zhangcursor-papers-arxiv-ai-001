# Beyond Causal Masks: Auditing and Localizing Hidden Future-Token Leakage in Adaptive Sequence Models

Huiting Liu<sup>a</sup>, Jinsong Wu<sup>a,b,</sup>∗

<sup>a</sup>School of Artificial Intelligence, Guilin University of Electronic Technology, Guilin, 541004, China

<sup>b</sup>Department of Electrical Engineering, University of Chile, Santiago, 8370451, Chile

## Abstract

Hidden future-token leakage can create false architectural gains: an autoregressive model may appear substantially better under teacher-forced evaluation while its free-running generation deteriorates. We investigate this failure in an adaptive residual-routing decoder whose sequence-level routing summary exposes prefix positions to future tokens despite a correct causal attention mask. An anomalous 24-layer checkpoint reduces validation loss from the baseline’s 3.5958 to 1.4259, yet it fails a prefix-invariance audit and has substantially higher reference-model perplexity during generation. Layer-wise diagnostics and implementation inspection trace the violation to a full-sequence summary broadcast through the dynamic routing-query path. Two independently trained 12-layer causal repairs eliminate detectable leakage and return validation loss and reference perplexity to the control scale, removing the large apparent advantage. As diagnostic tools, we use the established prefix-invariance contract, maximum prefix-output discrepancy (LeakScore), complementary behavioral metrics, and layer-wise profiles. Supporting experiments cover six synthetic implementation pairs spanning five module families, clean pretrained controls, controlled fault injection, and cache consistency across numerical precisions. These results illustrate how hidden future-token access can produce misleading claims of architectural progress and why lower teacher-forced loss should be interpreted together with whole-model causal checks and free-running evaluation.

Keywords: autoregressive models, causal masking, leakage detection, sequence modeling

## 1. Introduction

In decoder-only Transformers, causal self-attention restricts the output at position t to information from the prefix $x _ { 1 : t }$ . The same restriction must hold for every operation contributing to that output. Sequence-level pooling, temporal convolution, state aggregation, routing, and cross-time normalization can create additional information paths outside of masked attention. Prior work on linear attention and recurrent-style sequence compression likewise makes autoregressive behavior depend on prefix-consistent state aggregation [1]. Adaptive computation and selective state updates further motivate checking the complete dependency structure [2, 3].

We use hidden future-token leakage to describe sufix influence on prefix outputs through an auxiliary operation outside standard masked selfattention, despite a correct attention mask. Such access can lower teacherforced validation loss without improving valid autoregressive prediction. Broader studies of leakage have shown how invalid information access can inflate empirical performance claims [4]; here we examine this risk within the model’s computation. The practical concern is a false architectural gain: an apparent improvement that depends on information unavailable during causal generation.

Figure 1 illustrates the dependency structure. At position i, the causal path uses only $x _ { 1 : i }$ , while a non-causal auxiliary operation introduces an additional dependency on $x _ { i + 1 : T }$ . The auxiliary operation may also use past and present inputs; the violation is the future-to-prefix dependency, not the presence of the auxiliary module itself.

Our motivation comes from a follow-up implementation of CDR-AttnRes (Content-Conditioned Depth Routing for Attention Residuals), a residualrouting decoder variant inspired by Attention Residuals [5]. A full-sequence summary was broadcast as a low-rank correction to the depth-routing query. An unusually low teacher-forced validation loss suggested a strong improvement, but causal auditing detected leakage and free-running generation had substantially higher reference-model perplexity than the controls. Localization and code inspection identified the query path; independently trained 12-layer repairs removed detectable leakage and the large apparent advantage. Section 7 distinguishes the anomalous 24-layer trigger from the 12-layer repair comparisons.

Our diagnostic framework uses the established prefix-invariance principle: changing the sufix should leave shared-prefix outputs unchanged. We denote the maximum prefix-output discrepancy by LeakScore, supplement it with behavioral metrics, and inspect layer-wise profiles to guide implementation review. Controlled module comparisons and pretrained-model checks support interpretation of these diagnostics.

![](images/9b52ba439e967a2326925299804b0dac05de4f338b7c939b3e3e9e29d4e54c90.jpg)  
Figure 1: Mechanism of hidden future-token leakage. Solid paths represent permitted dependencies on past and present inputs, while the orange dashed path highlights a forbidden dependency on future inputs through a non-causal auxiliary operation. This is a schematic dependency diagram rather than a particular architecture: pooling, temporal mixing, state aggregation, routing, and cross-time normalization can introduce such paths when implemented non-causally.

The main contributions are threefold. First, we document a false architectural gain in a real adaptive architecture: a large teacher-forced loss advantage accompanies detectable leakage and generation degradation, and disappears once the non-causal path is repaired. Second, we provide crossmodule diagnostic evidence across six implementation pairs spanning five module families, together with clean controls and fault-injected pretrained models. Third, we use layer-wise localization to support root-cause analysis: FirstLeakLayer identifies the first monitored layer with detectable sufix influence and guides inspection of the CDR query path. Prefix invariance and LeakScore serve as diagnostic tools rather than a new causal contract.

Sections 3–4 define the problem and audit. Sections 5–6 present supporting experiments, followed by the CDR case in Section 7 and practical implications in Section 8.

## 2. Related Work

Leakage and empirical reliability. Data leakage can produce overoptimistic empirical conclusions [4], while overlap between benchmark content and training data complicates evaluation [6, 7]. These concerns arise at the dataset or evaluation-protocol level. Our focus is an internal dependency violation: future tokens influence prefix computation even when the dataset split and attention mask are correct.

Causality contracts. FBS describes a no-leakage unit test comparing sharedprefix logits under diferent sufixes by $\ell _ { \infty }$ discrepancy, together with previewcache consistency checks [8]. TorchLean formalizes causality-related contracts for attention and state-space sequence models, including prefix preservation, cache-position alignment, and scan consistency. Its diagnostic examples also address normalization state and padding conventions [9]. We build on the established prefix-invariance principle. Our empirical emphasis is a documented case in which a false gain disappears under causal repair, supported by cross-module diagnosis and layer-wise localization.

Sequence mixing beyond standard attention. Masked self-attention enforces causal token access in decoder-only Transformers [10]; segment-level and sparse-window variants support longer contexts [11, 12]. Linear attention and recurrent-form Transformers aggregate history into prefix states [1]. Causal convolutions prevent access to future inputs [13], and state-space models use structured or selective state updates [14, 3, 15]. RetNet, RWKV, and Hyena explore retention, recurrent formulations of linear attention, and longconvolution/gating mechanisms, respectively [16, 17, 18]. Their diferent computational forms motivate examining the causal structure of state aggregation and temporal mixing beyond the attention mask. Our controlled violations concern the auxiliary implementations specified in the experiments, rather than a claim that these cited architectures are non-causal.

Adaptive computation. Mixture-of-experts models route tokens to selected experts [19, 20]. Expert-choice routing assigns tokens to experts by ranking across sequence positions, so a position’s assignment can depend on later positions [21]. Expert threshold routing addresses causal routing and computation allocation [22], while Mixture-of-Depths allocates computation across tokens and layers [2]. These works make the information available to routing decisions a relevant implementation concern.

Testing, verification, and residual routing. Diferential and whitebox testing detect behavioral inconsistencies [23], whereas certified robustness methods derive provable bounds under specified perturbations [24], and formalization frameworks express semantic contracts [9]. Our behavioral audit requires access to position-wise outputs, and layer-wise localization additionally requires access to the monitored hidden states; compatibility with arbitrary checkpoints is not assumed. Attention Residuals provides the residual-aggregation setting for our follow-up CDR implementation [5]. The case concerns the fullsequence query correction added in that implementation.

## 3. Problem Definition

We consider a decoder-only autoregressive sequence model f operating on an input sequence

$$
x = ( x _ { 1 } , x _ { 2 } , \dots , x _ { T } ) .
$$

Causal correctness requires that, for any two inputs x and $x ^ { \prime }$ with $x _ { 1 : t } = x _ { 1 : t } ^ { \prime } .$ the outputs satisfy $f ( x ) _ { 1 : t } = f ( x ^ { \prime } ) _ { 1 : t }$ . This requirement concerns the complete computation of $f ,$ , including auxiliary operations as well as attention.

We use hidden future-token leakage to denote a violation of this condition mediated by an auxiliary operation outside masked self-attention, while the attention mask itself is causal. The term “hidden” refers to a forbidden dependency that is not visible in the attention mask.

This notion should be clearly distinguished from ordinary train/test leakage or benchmark contamination in machine learning [4, 6]. In conventional train/test leakage, the main problem is that evaluation data, labels, or benchmark content improperly overlaps with the training process, which leads to overoptimistic performance estimates at the dataset or protocol level. By contrast, hidden future-token leakage is an intra-model causal violation. The training and validation split may be entirely correct, yet the model can still behave non-causally at inference time because sufix information flows into prefix computation through internal modules. In other words, ordinary leakage concerns whether the model has seen the wrong data, whereas hidden future-token leakage concerns whether the model computes the wrong dependency structure.

Our audit examines which auxiliary operations violate this contract, how strongly their outputs depend on sufix perturbations, and where that dependence first becomes detectable. We use the maximum prefix-output discrepancy, denoted LeakScore, together with complementary behavioral metrics and layer-wise profiles.

## 4. Audit Methodology

To detect hidden future-token leakage in a module-agnostic way, we use a simple causal auditing principle based on prefix invariance. The core idea is that, for a causally correct autoregressive model, changing only the sufix of an input sequence should not afect the model outputs on the shared prefix.

## 4.1. Prefix-Invariance Criterion

Let x and $x ^ { \prime }$ be two input sequences such that

$$
x _ { 1 : t } = x _ { 1 : t } ^ { \prime } , \qquad x _ { t + 1 : T } \neq x _ { t + 1 : T } ^ { \prime } .
$$

If the model f is causally correct, then its outputs on the shared prefix should remain unchanged up to numerical tolerance:

$$
f ( x ) _ { 1 : t } \approx f ( x ^ { \prime } ) _ { 1 : t } .
$$

This is the criterion used throughout the paper. The prefix is held fixed and the sufix is perturbed. A discrepancy exceeding a suitably calibrated numerical tolerance, with other sources of variation controlled, is evidence of sufix-to-prefix influence. Because this test is defined only in terms of inputoutput behavior, it can be applied uniformly across a wide range of sequence modules, regardless of their internal implementation.

## 4.2. LeakScore

We summarize the discrepancy between the two prefix outputs induced by sufix perturbation using the following maximum-diference statistic, which we denote LeakScore:

$$
{ \mathrm { L e a k S c o r e } } ( f ; x , x ^ { \prime } , t ) = \operatorname* { m a x } _ { i \leq t } \| f _ { i } ( x ) - f _ { i } ( x ^ { \prime } ) \| _ { \infty } .
$$

This quantity measures the largest absolute output deviation over all prefix positions and output coordinates. For language models, the output coordinates are vocabulary logits. A score exceeding the numerical tolerance indicates detectable sufix-to-prefix influence for the tested pair; an undetected dependency may still exist for other inputs or below the tolerance.

For completeness, one may also consider an averaged version,

$$
\mathrm { L e a k S c o r e } _ { \mathrm { m e a n } } ( f ; x , x ^ { \prime } , t ) = \frac { 1 } { t d } \sum _ { i = 1 } ^ { t } \| f _ { i } ( x ) - f _ { i } ( x ^ { \prime } ) \| _ { 1 } ,
$$

where d is the output dimension (vocabulary size for logits). This is the elementwise mean absolute diference reported as Mean dif; batches are additionally averaged over examples. LeakScore determines the pair-level verdict, while the mean discrepancy summarizes the overall perturbation magnitude. We also report the top-1 flip rate (the fraction of prefix positions whose highest-scoring token changes) and the afected-token rate (the fraction whose maximum coordinate diference exceeds ϵ). The normalized maximum divides LeakScore by the maximum absolute reference-prefix logit plus $1 0 ^ { - 1 2 }$ ; mean Jensen–Shannon (JS) divergence compares the two softmax distributions and is averaged over prefix positions. These auxiliary metrics characterize the discrepancy but do not replace the absolute-tolerance verdict.

## 4.3. Audit Procedure

Algorithm 1 summarizes the audit procedure. We construct a pair of sequences with the same prefix and diferent sufixes, evaluate both with the same model parameters and runtime settings, and compare their prefix outputs. Dropout is disabled and other stochastic efects are controlled so that input perturbation is the intended source of variation.

Figure 2 illustrates the test: $x = ( p , s )$ and $\boldsymbol { x } ^ { \prime } = \left( p , s ^ { \prime } \right)$ have the same prefix $p = x _ { 1 : t } .$ , and only the shared-prefix logits are compared. The score S in the figure is LeakScore $( f ; x , x ^ { \prime } , t )$ defined above. A passing verdict applies only to the tested inputs and execution conditions.

Algorithm 1 Prefix-invariance audit procedure used throughout this work.   
Require: model $f ,$ sequence pair $( x , x ^ { \prime } )$ , prefix index $t ,$ tolerance ϵ, with   
$x _ { 1 : t } = x _ { 1 : t } ^ { \prime }$ and $x _ { t + 1 : T } \neq x _ { t + 1 : T } ^ { \prime }$   
Ensure: audit outcome: pass or fail   
1: Compute $y = f ( x )$ and $y ^ { \prime } = f ( x ^ { \prime } )$   
2: Extract prefix outputs $y _ { 1 : t }$ and $y _ { 1 : t } ^ { \prime }$   
3: Compute LeakScore $\begin{array} { r } { ( f ; x , x ^ { \prime } , t ) = \operatorname* { m a x } _ { i \leq t } \| y _ { i } - y _ { i } ^ { \prime } \| _ { \infty } } \end{array}$   
4: if LeakScore $( f ; x , x ^ { \prime } , t ) > \epsilon$ then   
5: report fail   
6: else   
7: report pass   
8: end if

![](images/abf351562c0cde1da0528d7b62f73dadff2a890da10c33d422f63b8f0fa37158.jpg)  
Passing applies only to the tested conditions.  
Figure 2: Prefix-invariance audit. Two inputs share a prefix $p = x _ { 1 : t }$ but have diferent sufixes s and $s ^ { \prime } .$ . Both are evaluated by the same model under the same evaluation settings, with stochastic efects controlled. LeakScore is the largest absolute prefix-logit diference across positions and vocabulary entries. A score above the numerical tolerance ϵ indicates an audit failure; passing provides evidence only for the tested inputs and conditions.

In practice, the two sequences can be generated by sampling a base sequence and replacing only the sufix segment after position t. The same procedure applies to logits, hidden states, or other position-wise model outputs, as long as the compared quantity is the object whose autoregressive causality we wish to audit.

## 4.4. Tolerance and Numerical Robustness

Because the test compares floating-point outputs, a small numerical tolerance is necessary even for causally correct implementations. In our experiments, we use

$$
\epsilon = 1 0 ^ { - 5 }
$$

as the default threshold. This choice is appropriate for our toy auditing setup in standard float32 evaluation and provides a stable separation between causal and non-causal modules. As shown later in the robustness checks, our main conclusions are unchanged when the tolerance is varied over $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ : causally repaired variants remain near numerical zero, whereas non-causal variants consistently yield LeakScore values far above this range. Thus, the observed separation is not an artifact of a particular threshold choice.

## 4.5. Generation Diagnostics

In the CDR checkpoint case study, audit verdicts are compared with open-ended generation diagnostics. The reference-model perplexity $\mathrm { p p l } _ { \mathrm { r e f } }$ is computed using a fixed GPT-2 medium reference LM. The frozen evaluation code scores the saved text, including the prompt and generated continuation, in windows of at most 1024 tokens with stride 512. For each sample, it exponentiates the token-count-weighted mean window negative log-likelihood; the reported value is the arithmetic mean of these sample perplexities over the 50 prompts. Higher perplexity means the reference model assigns lower likelihood to the text. This is a model-dependent diagnostic of generation quality, not a direct measure of semantic coherence or a human evaluation. The text is re-encoded with the reference tokenizer. Sampling entropy $\begin{array} { r } { H = - \sum _ { v } p _ { v } } \end{array}$ log $p _ { v }$ is measured in nats from the distribution after temperature scaling and top-k truncation, averaged over generated steps and then prompts. Distinct-2 is the proportion of unique bigrams, and Repeat-3 is the proportion of trigram occurrences belonging to repeated trigrams; both are computed on whitespace-separated units of the saved prompt-inclusive text. Bootstrap intervals quantify uncertainty in the prompt-averaged perplexity.

## 5. Synthetic Auditing Experiments

We next use the prefix-invariance test to audit a set of synthetic sequence modules beyond masked attention. The goal of these experiments is not to maximize task performance, but to isolate whether representative auxil iary components can introduce hidden future-token leakage under controlled conditions. We construct causally repaired and non-causal counterparts for each module family and evaluate them under the same auditing procedure described in Section 4.

## 5.1. Experimental Design

Our synthetic audit covers six implementation pairs across five module families: pooling (two pairs), temporal convolution, linear attention, dynamic routing, and cross-time normalization. Each pair contains one non-causal implementation and one causally repaired counterpart, giving twelve implementations in total. The selected families are intended to span several common ways in which future information may re-enter prefix computation, including sequence-level summary broadcast, temporal mixing, full-sequence state aggregation, routing based on global context, and cross-time normalization.

The synthetic modules are randomly initialized and audited without training. Input token IDs are sampled uniformly from the vocabulary; each pair shares the same sampled prefix and has independently sampled sufixes. These experiments test whether the implemented computation respects prefix invariance, rather than evaluating learned language-model quality. For each audited module, we compare outputs on the shared prefix. We then compute the LeakScore on prefix outputs using the default tolerance $\epsilon = 1 0 ^ { - 5 }$ . A model is considered to fail the audit if the resulting LeakScore exceeds the tolerance, and to pass otherwise.

## 5.2. Audited Modules

Table 1 summarizes the audited modules and their main results. The six non-causal variants all yield strictly positive LeakScore values, whereas all six causally repaired counterparts produce zero leakage under the test.

## 5.3. Main Findings and Interpretation

All six non-causal implementations fail the audit, while their paired causal repairs have zero discrepancy under the same conditions (Table 1). The failures span summary broadcast, temporal mixing, full-sequence KV aggregation, routing, and normalization, showing that the tested risk is not confined to pooling. LeakScore ranges from 0.0033 to 0.6625 across modules with diferent output scales, so these values should not be read as a cross-module severity ranking. The relevant comparison is within each implementation pair: every non-causal implementation lies well above the tolerance, and every causal repair is exactly zero.

Table 1: Six synthetic non-causal/causal implementation pairs. Score before and score after denote LeakScore for the non-causal implementation and its causal repair, respectively, under the same audit $( \epsilon = 1 0 ^ { - 5 } )$ . Scores are reported on each module’s own output scale and should not be used to rank diferent modules.
<table><tr><td>Module</td><td>Non-causal operation</td><td>Causal repair</td><td>Before</td><td>After</td></tr><tr><td>Pooling</td><td>Full-sequence mean</td><td>Token-wise routing</td><td>0.0755</td><td>0.0000</td></tr><tr><td>Pooling</td><td>Last-token broadcast</td><td>Prefix-mean routing</td><td>0.3232</td><td>0.0000</td></tr><tr><td>Conv1d</td><td>Symmetric padding</td><td>Left-only padding</td><td>0.6625</td><td>0.0000</td></tr><tr><td>Linear attention</td><td>Full-sequence KV</td><td>Prefix-only KV</td><td></td><td>0.1168 0.0000</td></tr><tr><td>Dynamic routing</td><td>aggregation Full-sequence context</td><td>aggregation Prefix-only context</td><td></td><td>0.0033 0.0000</td></tr><tr><td>Normalization statistics</td><td>Full-sequence</td><td>Prefix-only statistics</td><td></td><td>0.0255 0.0000</td></tr></table>

These controlled comparisons connect the dependency criterion to concrete implementations without conflating causal validity with learned languagemodel quality. They also provide a check that the same behavioral test separates the evaluated failures from their causal counterparts. The following robustness checks vary the split, random seed, and tolerance; the real-model experiments then examine clean controls, injected violations, and runtime efects.

## 5.4. Robustness Checks

We further verified that these conclusions do not depend on a particular experimental setting. We consider three sources of variability: the prefix length used to define the split, the random seed controlling both model initialization and input sampling, and the tolerance used to decide pass or fail. Unless stated otherwise, the following checks use the same default configuration as the main experiments above (sequence length 32, prefix length 16, random seed 42, $\epsilon = 1 0 ^ { - 5 } )$ .

Table 2: LeakScore of non-causal modules under diferent prefix lengths (sequence length 80, random seed 42, $\epsilon = 1 0 ^ { - 5 } )$ . All causally repaired variants remain at 0.0 and are omitted for clarity.
<table><tr><td>Module</td><td>8</td><td>16</td><td>32</td><td>64</td></tr><tr><td>noncausal_pool_mean</td><td>0.0484</td><td>0.0425</td><td>0.0328</td><td>0.0212</td></tr><tr><td>noncausal_pool_last</td><td>0.6819</td><td>0.5959</td><td>0.6969</td><td>0.6041</td></tr><tr><td>noncausal_samepad_conv1d</td><td>0.8203</td><td>0.7067</td><td>0.9781</td><td>0.9717</td></tr><tr><td>noncausal_fullkv_linear_attn</td><td>0.1343</td><td>0.1134</td><td>0.0958</td><td>0.0711</td></tr><tr><td>noncausal_global_route</td><td>0.00421</td><td>0.00466</td><td>0.00359</td><td>0.00217</td></tr><tr><td>noncausal_global_norm</td><td>0.0358</td><td>0.0306</td><td>0.0199</td><td>0.0178</td></tr></table>

Varying the prefix length.. Table 2 reports the LeakScore of each non-causal module under prefix lengths of 8, 16, 32, and 64. In order to accommodate prefix length 64, this check uses a longer sequence length of 80; this is why the numbers in Table 2 difer from those in Table 1, which used sequence length 32. Under all four split points, every non-causal module remains clearly above the tolerance, while every causally repaired variant remains at zero. The prefix-invariance failure is therefore not an artifact of a particular split position.

Varying the random seed.. We repeated the audit with three random seeds (0, 1, 2) under the default configuration. Table 3 summarizes the mean and population standard deviation of the LeakScore over the three seeds, together with the number of passing runs. All non-causal modules fail in all three runs $( 0 / 3 )$ , whereas every causally repaired variant passes in all three runs $( 3 / 3 )$ . The exact LeakScore values fluctuate with the seed, but the categorical conclusion is fully stable, and no module lands near the decision boundary.

Varying the tolerance.. Finally, we repeated the audit with tolerance values of $1 0 ^ { - 6 } , \ 1 0 ^ { - 5 }$ , and $1 0 ^ { - 4 }$ . The pass/fail classification is identical across all three thresholds: every causally repaired variant passes and every non-causal module fails. Moreover, the weakest non-causal signal in our experiments remains above $1 0 ^ { - 4 }$ (e.g., the non-causal global-routing variant at $\approx 0 . 0 0 3 3 )$ ), which remains well above all three tested tolerances and the observed zero baseline of the repaired controls. The observed separation is therefore not an artifact of the tolerance choice.

Table 3: Seed stability of the audit over three random seeds (sequence length 32, prefix length 16, $\epsilon = 1 0 ^ { - 5 } )$ . All causally repaired variants achieve 3/3 pass with LeakScore $0 . 0 \pm 0 . 0 .$
<table><tr><td>Module</td><td>Passed runs LeakScore</td><td> $( \mathrm { m e a n } \pm \mathrm { s t d } )$ </td></tr><tr><td>noncausal_pool_mean</td><td>0/3</td><td> $0 . 0 7 6 0 \pm 0 . 0 2 1 1$ </td></tr><tr><td>noncausal_pool_last</td><td>0/3</td><td> $0 . 5 1 0 4 \pm 0 . 1 4 2 0$ </td></tr><tr><td>noncausal_samepad_conv1d</td><td>0/3</td><td> $0 . 6 1 7 3 \pm 0 . 0 1 7 5$ </td></tr><tr><td>noncausal_fullkv_linear_attn</td><td>0/3</td><td> $0 . 1 2 6 9 \pm 0 . 0 2 1 3$ </td></tr><tr><td>noncausal_global_route</td><td>0/3</td><td> $0 . 0 0 3 4 7 \pm 0 . 0 0 0 0 8$ </td></tr><tr><td>noncausal_global_norm</td><td>0/3</td><td> $0 . 0 3 1 6 \pm 0 . 0 1 0 4$ </td></tr></table>

## 6. Real-Model Auditing

The synthetic experiments establish that the audit has high specificity on controlled modules. We now ask whether the same test remains useful on real pretrained models, where implementations are black boxes and numerical noise is a practical concern. This section reports four additional layers of evidence: negative controls on trusted models, fault injection to establish sensitivity, cache and precision consistency across runtime paths, and layerwise leakage localization of the first leak layer. Together they address a limitation of the toy setup: a behavioral test that works on synthetic modules might, in principle, either fire spuriously on real implementations or become dificult to interpret once runtime-path and precision efects introduce nonzero clean discrepancies.

## 6.1. Experimental Setup

We use three publicly released decoder-only models: the Transformers GPT-2 and Qwen2.5-0.5B, and the selective state-space model Mamba-130M [3]. The sufix-perturbation and localization audits use float32 inference with dropout disabled; the cache-consistency experiment separately evaluates float32, float16, and bfloat16.

The perturbation protocol follows the same fixed-prefix principle as the synthetic audit, with a separate evaluation grid. We set sequence length $T = 6 4$ , prefix lengths $t \in \{ 1 6 , 3 2 , 4 8 \}$ , random seeds {0, 1, 2}, and generate 8 sufix-perturbed pairs per combination of prefix length and seed, giving 72 pairs per model. Perturbations operate directly on token IDs, so the two inputs share exactly the same prefix tokens and difer only in the sufix, avoiding tokenizer boundary artifacts. For each pair we report the max and mean prefix-output discrepancy, the top-1 flip rate, the afected-token rate, and the pass/fail verdict under tolerance $\epsilon = 1 0 ^ { - 5 }$

## 6.2. Clean Negative Controls

GPT-2, Qwen2.5-0.5B, and Mamba-130M each pass all 72 tested input pairs in float32 with $\epsilon = 1 0 ^ { - 5 } ;$ maximum and mean prefix-logit discrepancy and top-1 flip rate are all exactly zero. No false positives are observed under this same-path protocol. Exact zero does not imply that floating-point inference is generally noise-free: comparing diferent execution paths can yield non-zero discrepancies, as Section 6.4 shows. The clean controls support the chosen tolerance for this protocol, rather than a universal threshold for other precisions or runtime paths.

## 6.3. Fault Injection and Sensitivity

A clean negative control only establishes specificity. To verify that the audit can actually detect a hidden leak when one is present, we inject a causal violation into the same models without touching the pretrained weights. Following the failure pattern found in the CDR case (Section 7), the wrapper first pools the full hidden-state sequence $h _ { 1 : T }$ into a single summary $s = \mathrm { M e a n } ( h _ { 1 : T } )$ , then produces a correction $\Delta z = \alpha g ( s )$ broadcast back to every position, i.e. $z _ { i } \gets z _ { i } + \Delta z$ for all i. This is exactly the sequence-level summary-to-broadcast pattern that our synthetic pooling routers instantiate, and it is applied identically to all three model families. We sweep the injection strength $\alpha \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 0 \}$ ; GPT-2 and Qwen are tested at all three strengths, and Mamba at $\alpha = 0 . 0 5$ , using the same 72-pair protocol.

Table 4 shows the results. Two findings stand out. First, sensitivity is high: at every tested non-zero injection strength, every evaluated model fails the audit and the pass rate drops from 1.0 to 0.0, while the afected-token rate stays at or above 0.993, i.e. the shared correction reaches essentially every prefix position. Second, the continuous diagnostics behave diferently from the binary verdict. The max discrepancy increases monotonically with α (approximately linearly for Qwen: $1 . 6 \times 1 0 ^ { - 3 } , 7 . 9 \times 1 0 ^ { - 3 } , 1 . 6 \times 1 0 ^ { - 2 } )$ , yet even the smallest tested injection $\alpha = 0 . 0 1$ already fails the binary test. The audit is therefore both sensitive at the decision level and quantitative in the diagnostic scores, which is exactly the combination needed for the multi-metric diagnosis emphasized throughout this paper.

Table 4: Fault-injection sensitivity with a sequence-level summary-broadcast wrapper and unchanged pretrained weights. Max dif and Norm. max are maxima over 72 pairs; JS divergence and afected-token rate are per-pair means, and pass rate is the fraction of passing pairs $( \epsilon = 1 0 ^ { - 5 } )$
<table><tr><td>Model</td><td>α</td><td>Max diff</td><td>Norm. max JS div</td><td></td><td>Affected-token rate Pass rate</td></tr><tr><td>GPT-2</td><td>0.01</td><td>2.1e-4</td><td>1.3e-6</td><td>1.9e-8</td><td>0.993 0.000</td></tr><tr><td>GPT-2</td><td>0.05</td><td>1.0e-3</td><td>6.3e-6</td><td>1.8e-8 1.000</td><td>0.000</td></tr><tr><td>GPT-2</td><td>0.10</td><td>2.1e-3</td><td>1.3e-5</td><td>2.0e-8 1.000</td><td>0.000</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.01</td><td>1.6e-3</td><td>7.1e-5</td><td>1.1e-8 1.000</td><td>0.000</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.05</td><td>7.9e-3</td><td>3.5e-4</td><td>9.9e-8</td><td>1.000 0.000 1.000</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.10</td><td>1.6e-2</td><td>7.0e-4</td><td>3.9e-7</td><td>0.000</td></tr><tr><td>Mamba-130M</td><td>0.05</td><td>3.0e-3</td><td>3.7e-5</td><td>2.0e-8</td><td>1.000 0.000</td></tr></table>

## 6.4. Cache and Precision Consistency

Cache consistency tests whether full-prefix evaluation and token-by-token decoding produce comparable logits for the same input and checkpoint. A discrepancy between these paths can arise from numerical precision, kernel diferences, or cache implementation errors; it does not by itself demonstrate future-token leakage. We compare the two paths for GPT-2 and Qwen2.5- 0.5B over a prefix of length 32 in float32, float16, and bfloat16. Mamba is outside the scope of this KV-cache experiment because it uses recurrent state rather than a Transformer key-value cache.

Table 5: Cache-consistency audit between full-prefix forward and incremental decoding, summarized over the 32 prefix steps by the discrepancy distribution of the per-step maximum logit diference. The key calibration quantities are the clean-control P50/P95/max values for each runtime setting.
<table><tr><td>Model</td><td>Precision P50 max diff P95 max diff</td><td></td><td></td><td>Max diff</td><td>Top-1 agreement</td></tr><tr><td>GPT-2</td><td>fp32</td><td> $1 . 9 \times 1 0 ^ { - 4 }$ </td><td> $4 . 5 \times 1 0 ^ { - 4 }$ </td><td> $7 . 6 \times 1 0 ^ { - 4 }$ </td><td>32/32</td></tr><tr><td>GPT-2</td><td>fp16</td><td>0.125</td><td>0.25</td><td>0.625</td><td>32/32</td></tr><tr><td>GPT-2</td><td>bf16</td><td>0.75</td><td>2.0</td><td>4.0</td><td>31/32</td></tr><tr><td>Qwen2.5-0.5B</td><td>fp32</td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td><td> $3 . 9 \times 1 0 ^ { - 5 }$ </td><td> $8 . 1 \times 1 0 ^ { - 5 }$ </td><td>32/32</td></tr><tr><td>Qwen2.5-0.5B</td><td>fp16</td><td>0.0313</td><td>0.0476</td><td>0.1475</td><td>32/32</td></tr><tr><td>Qwen2.5-0.5B</td><td>bf16</td><td>0.2461</td><td>0.3703</td><td>0.5</td><td>32/32</td></tr></table>

As Table 5 shows, the clean-control discrepancy distribution broadens sharply as precision is reduced. In fp32, the two runtime paths remain close but not numerically identical: the per-step P95 maximum discrepancy is $4 . 5 \times 1 0 ^ { - 4 }$ for GPT-2 and $3 . 9 \times 1 0 ^ { - 5 }$ for Qwen2.5-0.5B. In reduced precision the envelope becomes much wider, reaching a maximum of 4.0 for GPT-2 in bf16 and 0.5 for Qwen2.5-0.5B in bf16. Top-1 predictions remain fully consistent in every setting except GPT-2 bf16, where one of the 32 prefix steps has a diferent top-1 prediction. The implication is not that these clean checkpoints are architecturally non-causal, but that cache consistency is itself precision-sensitive: any behavioral threshold used across runtime paths must therefore be calibrated empirically on trusted clean controls, and should be reported together with the relevant precision/runtime setting rather than treated as universal. This operational dimension is complementary to, and in the spirit of, the cache-consistency checks used in recent work on causal correctness of sequence models [8, 9].

## 6.5. Layer-wise Leakage Localization

A failing verdict indicates detectable sufix-to-prefix influence, but does not identify where it first enters the network. To turn the audit from a binary decision into an actionable location, we compute a per-layer leakage profile. Let $h ^ { ( \ell ) } ( x )$ denote the hidden state at the output of layer ℓ. For a sequence pair $( x , x ^ { \prime } )$ sharing prefix t, the layer-ℓ leakage is

$$
L _ { \operatorname* { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i \leq t } \left\| h _ { i } ^ { ( \ell ) } ( x ) - h _ { i } ^ { ( \ell ) } ( x ^ { \prime } ) \right\| _ { \infty } ,
$$

and the first leak layer is defined as

$$
\mathrm { F i r s t L e a k L a y e r } = \operatorname* { m i n } \left\{ \ell : L _ { \mathrm { m a x } } ^ { ( \ell ) } > \epsilon _ { \ell } \right\} ,
$$

where layers are indexed from zero and $\epsilon _ { \ell } = 1 . 0 5 c _ { \ell } + 1 0 ^ { - 8 }$ is calibrated from clean controls under the same localization protocol, with $c _ { \ell }$ the maximum clean discrepancy observed at layer ℓ across 24 calibration pairs. The threshold is set slightly above the empirically observed clean-control noise ceiling, so that floating-point variation is absorbed without masking genuine leakage. If no layer exceeds its threshold, we report that no first leak layer is detected for that input pair. To verify localization on the three public models, we inject a summary-broadcast perturbation into the hidden-state output of layer 3 with $\alpha = 0 . 0 5$ and check whether this layer is recovered on one representative input pair $( T = 6 4 , t = 3 2 )$ per model. These profiles are not maxima over the 72-pair output audit. The representative pair is also among the clean calibration pairs, so this check is not an independent holdout evaluation. This hidden-state intervention difers from the output-logit wrapper used in the sensitivity experiment; their discrepancy magnitudes are not directly comparable.

As Table 6 shows, the injected layer is recovered in all three tested architectures: on clean models no layer exceeds its threshold, and after injection the audit identifies layer 3 as the first monitored layer with detectable sufix influence. The final-layer discrepancy is larger than at the injection layer. These results support using layer-wise profiles to guide implementation in spection under the evaluated conditions. The same diagnostic is applied to the CDR checkpoints in Section 7, where the first detectable leak occurs at the bottom of the network.

Table 6: Hidden-state localization on one representative pair per model (t = 32) after injection at zero-based layer 3 (α = 0.05). Final-layer discrepancy is measured at the last monitored layer.
<table><tr><td>Model</td><td></td><td>Layers Inject → FirstLeakLayer Final-layer discrepancy</td><td></td></tr><tr><td>GPT-2</td><td>12</td><td>3→3</td><td>3.47</td></tr><tr><td>Qwen2.5-0.5B</td><td>24</td><td>3→3</td><td>0.86</td></tr><tr><td>Mamba-130M</td><td>24</td><td>3→3</td><td>18.15</td></tr></table>

Figure 3 shows the full per-layer profile. For all three models the leakage is exactly zero before the injection point, jumps to a non-zero value at layer 3, and then propagates through subsequent layers with local decreases as well as increases; the final-layer discrepancy exceeds the discrepancy at the injection layer in every model. The clean models remain at zero in every layer. Because the models difer in depth (12 vs. 24 layers), the same absolute layer 3 appears at diferent normalized depths in the figure (3/11 for GPT-2 vs. 3/23 for the others); the recovered FirstLeakLayer is nonetheless the exact absolute layer index in every case.

![](images/8783134a1f2b651bdb8ab71c7b747fefad84ba231d1168863ec7019912dceeb2.jpg)  
Figure 3: Layer-wise hidden-state discrepancy after fault injection at zero-based layer 3 $( \alpha = 0 . 0 5 )$ . Dotted lines indicate the injection depth. Zero discrepancies are displayed at $1 0 ^ { - 7 }$ on the logarithmic axis.

## 7. Case Study: False Architectural Gains in CDR-AttnRes

We examine the main empirical case from our architectural exploration: CDR-AttnRes, a residual-routing decoder variant developed in follow-up experiments inspired by Attention Residuals [5]. In this case, an anomalously low teacher-forced validation loss does not indicate an architectural improvement: the variants also exhibit detectable prefix leakage and markedly degraded free-running generation; layer-wise localization and implementation inspection trace the leakage to a non-causal query pathway; and causal repairs eliminate both the leakage and the large apparent advantage.

We use a compact naming scheme for the checkpoints audited in this section. Baseline denotes the standard decoder, and Static AttnRes uses learned base routing queries without the content-conditioned query correction introduced by CDR; its depth-wise mixing weights remain input-dependent (AttnRes abbreviates Attention Residuals [5]). The 12-layer pre-fix variants share the CDR-AttnRes routing and rank $r = 6 4$ , and difer in the sequencelevel pooling used to produce the query correction: CDR-mean, CDR-gated, and CDR-last correspond to mean, gated, and last-token pooling, respectively. Causally repaired reimplementations carry a $- \mathit { f i x }$ sufix (CDR-tokenfix and CDR-prefix-mean-fix). A name without a $\mathcal { Z } \mathcal { L }$ sufix denotes the

12-layer configuration.

## 7.1. Anomalous Training and Validation Behavior

The first sign of the problem was an unusually large reduction in validation loss. In a 24-layer configuration, the baseline model achieved a validation loss of 3.5958, whereas the pre-fix CDR-last 24L variant reached 1.4259. This large gap motivated a causal audit before interpreting the lower loss as an architectural improvement.

The same pattern also appeared in 12-layer experiments. We therefore use the anomalous 24-layer checkpoint as the trigger case, while the 12-layer variants provide the controlled and repair comparisons used to test whether the same mechanism recurs and can be causally repaired. Several pre-fix variants based on sequence-level pooling all achieved much lower validation losses than the trusted controls, Baseline and Static AttnRes. However, these gains were not accompanied by correspondingly credible autoregressive behavior.

## 7.2. Direct Audit with Prefix Invariance and LeakScore

We next audited the real checkpoints using the same prefix-invariance test introduced earlier. For each checkpoint, we used sequence length $T = 6 4$ prefix lengths $t \in \{ 1 6 , 3 2 , 4 8 \}$ , seeds {0, 1, 2}, and eight sufix perturbations per combination, giving 72 input pairs with identical prefixes and diferent sufixes. We compared the prefix logits with tolerance $\epsilon = 1 0 ^ { - 5 }$ . At the checkpoint level, we report the maximum LeakScore over the 72 pairs; mean absolute discrepancy, top-1 flip rate, and afected-token rate are arithmetic means of their per-pair values. Thus, rates give equal weight to each pair rather than pooling prefix tokens across diferent prefix lengths. For the anomalous pre-fix 24-layer checkpoint, the audit fails decisively: the maximum prefix-output diference reaches 34.22, and the mean per-pair top-1 flip rate is 95.1%.

The checkpoint-level failure establishes detectable sufix-to-prefix influence before localization or implementation inspection. It links the validationloss anomaly to a direct violation of the causal contract; Table 10 reports the corresponding audit results across the core checkpoints.

## 7.3. Layer-wise Localization of the CDR Leak

We also applied the same layer-wise localization criterion used in Section 6 to the CDR checkpoints, using causal-safe references for the per-layer thresholds: the token-wise repaired model for 12-layer pre-fix variants and the 24-layer baseline for the 24-layer pre-fix variant. Controls and repaired variants use self-calibration. The localization profiles use a representative input pair with sequence length 64 and prefix length 32, while the per-layer thresholds are calibrated on 24 pairs. The representative pair is included among the calibration inputs, so this is not an independent hold-out evaluation. These profiles are not maxima over the 72-pair checkpoint audit. Table 7 reports the representative trusted and pre-fix variants. Every trusted control has no first leak layer and zero final-layer discrepancy. By contrast, every pre-fix variant shown in the table, including the 24-layer checkpoint, begins to leak immediately at layer 0. The discrepancy then propagates through subsequent layers with local decreases as well as increases; in every leaking variant, the final-layer discrepancy exceeds that at layer 0.

Table 7: CDR hidden-state localization on a representative input pair with prefix length 32. Final-layer discrepancy is measured at the last monitored layer; – denotes no detected leaking layer. Unmarked variants are 12-layer.
<table><tr><td>Variant</td><td>FirstLeakLayer</td><td>Final-layer discrepancy</td></tr><tr><td>Baseline Static AttnRes</td><td></td><td>0.00 0.00</td></tr><tr><td>CDR-gated</td><td>0</td><td>2.20</td></tr><tr><td>CDR-mean</td><td>0</td><td>3.55</td></tr><tr><td>CDR-last</td><td>0</td><td>4.64</td></tr><tr><td>CDR-last 24L</td><td>0</td><td>16.96</td></tr></table>

## 7.4. Generation Diagnostics of Controls and Leaking Checkpoints

To assess whether the audit failure accompanies degraded generation, we evaluated the checkpoints under a fixed open-ended generation protocol. For each variant, we generated 50 continuations from fixed prompts taken from the OpenWebText validation set (50 prompts derived from the first 32 GPT-2 tokens of each selected document, with whitespace normalized before generation, 300 new tokens, temperature 0.8, top-k 200, random seed 1337), and measured reference-model perplexity $\mathrm { p p l } _ { \mathrm { r e f } }$ (sliding-window perplexity of the GPT-2 medium reference LM; definition in Section 4), sampling-distribution entropy H, and lexical diversity and repetition statistics. Table 8 reports prompt-averaged values. As defined in Section 4, reference perplexity and lexical statistics use the saved prompt-inclusive text, whereas entropy is averaged over generated steps. The validation-loss column records checkpoint validation results and is not an average over the generation prompts. Percheckpoint 95% confidence intervals for mean reference perplexity are computed by resampling the 50 prompts 2000 times; the interval envelopes across checkpoint groups are summarized below.

Table 8: Validation loss and generation diagnostics for controls and pre-fix CDR checkpoints. Generation metrics are averaged over 50 prompts, with one continuation per prompt; validation loss is the checkpoint validation result. Unmarked variants are 12- layer.
<table><tr><td>Variant</td><td>Val loss</td><td> $\mathrm { p p l } _ { \mathrm { r e f } }$ </td><td>Entropy</td><td>Distinct-2</td><td>Repeat-3</td></tr><tr><td>Static AttnRes</td><td>3.6676</td><td>17.2</td><td>1.64</td><td>0.804</td><td>0.174</td></tr><tr><td>Baseline</td><td>3.6900</td><td>19.8</td><td>1.85</td><td>0.808</td><td>0.153</td></tr><tr><td>Baseline 24L</td><td>3.5958</td><td>16.8</td><td>1.90</td><td>0.827</td><td>0.139</td></tr><tr><td>CDR-gated</td><td>2.6015</td><td>198.5</td><td>2.31</td><td>0.822</td><td>0.136</td></tr><tr><td>CDR-mean</td><td>2.8473</td><td>299.9</td><td>2.80</td><td>0.725</td><td>0.198</td></tr><tr><td>CDR-last</td><td>2.5488</td><td>695.5</td><td>2.87</td><td>0.784</td><td>0.134</td></tr><tr><td>CDR-last 24L</td><td>1.4259</td><td>700.5</td><td>2.13</td><td>0.849</td><td>0.086</td></tr></table>

Table 8 shows a clean separation. Every trusted control has $\mathrm { p p l } _ { \mathrm { r e f } }$ between 16.8 and 19.8, whereas every pre-fix variant lies between 198.5 and 700.5—roughly a 10–42× gap with no overlap. The per-prompt uncertainty estimates reinforce the separation: the 95% bootstrap confidence intervals of the trusted controls (whose individual interval endpoints span [15.2, 22.8]) and the leaking variants (whose endpoints span [169.0, 839.7]) do not overlap. Sampling entropy provides a second descriptive separation: controls occupy 1.64–1.90, and leaking variants 2.13–2.87. The same checkpoints that fail the causal audit have much higher reference perplexity and sampling entropy during generation.

Lexical diversity alone does not capture the observed degradation. The pre-fix 24-layer variant has the highest Distinct-2 (0.849) and lowest Repeat-3 (0.086) in Table 8, yet also the highest reference perplexity (700.5). These lexical scores therefore do not establish generation quality. Reference perplexity and the illustrative samples in Appendix A.3 support the diagnosis of of-target generation, while sampling entropy describes the distribution from which tokens are selected. The small perplexity diference between the two last-pooling checkpoints should not be interpreted as a reliable ordering, given their overlapping confidence intervals.

## 7.5. Root Cause: Sequence-Level Pooling in Dynamic Query Construction

To explain why the checkpoint fails the audit and why localization points to the very bottom of the network, we inspected the implementation paths that inject auxiliary signals into the backbone, which led us to the dynamic query pathway. Here, the query controls depth-wise routing: at each token position, it scores residual sources from diferent depths, and a softmax over those sources determines their weighted mixture. This routing is applied before both the attention and MLP sublayers; the query is distinct from the token-to-token query in masked self-attention. In the pre-fix implementation, a sequence-level summary was first computed from the entire hidden-state sequence, using mean pooling, last-token pooling, or related summary operations. The hidden-state sequence of shape $B \times T \times D$ was pooled along the time axis to a vector of shape $B \times D$ , and the resulting summary was then projected into a shared dynamic query correction $\Delta q _ { \mathrm { s h a r e d } }$ that was broadcast back to all token positions. The same correction signal therefore influenced prefix positions as well as later ones.

This design is non-causal because the sequence-level summary already contains information from sufix tokens. As a result, even though the masked self-attention path remained causally correct, prefix positions were indirectly afected by $x _ { t + 1 : T }$ through the auxiliary dynamic-query route. This mechanism also explains the localization result above: because the shared correction is injected immediately into the backbone computation, the first detectable leak appears at the bottom of the network and is then amplified by subsequent layers.

## 7.6. Repair Validation

We finally asked whether a causality-preserving reimplementation would remove both the audit failure and the apparent empirical gain. The repair replaces the shared full-sequence summary with a position-dependent causal summary of the current hidden states $h _ { 1 : T }$ . Token-wise repair uses $s _ { i } = h _ { i }$ whereas prefix-mean repair uses $s _ { i } = i ^ { - 1 } \textstyle \sum _ { j = 1 } ^ { i } h _ { j }$ . Both retain the low-rank correction $\Delta q _ { i } = W _ { \mathrm { u p } } \mathrm { G E L U } ( W _ { \mathrm { d o w n } } s _ { i } )$ and add it to the learned base routing query at position i, so the correction has no access to future positions. The two repaired variants were trained from scratch using the shared training recipe in Appendix A.1; they are separate checkpoints rather than inferencetime switches applied to a pre-fix checkpoint. Table 9 summarizes their results. Under both token-wise and prefix-mean repair, the LeakScore returns to zero, no first leak layer is found, the validation loss is close to that of the trusted controls, and the generation perplexity (15.3 and 18.1) returns to the same scale as the trusted controls (16.8–19.8) while remaining an order of magnitude below every leaking variant (198.5–700.5).

Table 9: Audit and evaluation results for the two independently trained 12-layer causal repairs. – denotes no detected leaking layer. Both repairs pass the audit and have validation loss and reference perplexity on a similar scale to the controls.
<table><tr><td>Variant</td><td></td><td></td><td>Val loss LeakScore FirstLeakLayer</td><td> $\mathrm { p p l } _ { \mathrm { r e f } }$ </td></tr><tr><td>CDR-token-fix</td><td>3.6766</td><td>0.0</td><td></td><td>15.3</td></tr><tr><td>CDR-prefix-mean-fix</td><td>3.6731</td><td>0.0</td><td></td><td>18.1</td></tr></table>

Figure 4 compares all seven 12-layer checkpoints using stacked panels sharing a common horizontal axis: validation loss (a), checkpoint-level LeakScore (b), and generation perplexity (c). The leaking variants combine lower validation loss with higher generation perplexity, whereas the repaired variants have zero leakage and return to the control scale. Table 7 provides layer-wise localization, and Table 9 reports the repair results numerically.

## 7.7. Quantitative Consistency Between Leakage and Apparent Gain

Beyond the focused anomaly and repair comparisons above, we also asked whether the leak–gain relationship holds over the nine core checkpoints rather than only the representative examples above. We therefore ran the same audit on the full set of nine core CDR variants. Table 10 reports validation loss, maximum leakage, top-1 flip rate, and the audit verdict for each.

Across the nine checkpoints, lower validation loss is associated with greater leakage (Pearson r = −0.868 for LeakScore; full descriptive statistics are provided in the supplementary materials). This pattern is largely a separation between the five passing controls/repairs and four failing pre-fix checkpoints. Within the four leaking checkpoints alone, the association is weaker and not statistically significant (Pearson $r = - 0 . 5 5 9 , \ p = 0 . 4 4 1 )$ These related checkpoints are not independent experimental replications, and the leaking variants difer in pooling mode and depth while sharing rank $r \ : = \ : 6 4$ The small sample and these diferences preclude a dose–response interpretation. Table 10 supports the case-level observation that the apparent low-loss advantage accompanies audit failure; it does not establish that leakage magnitude determines validation loss.

(a) 12L Validation Loss  
![](images/f1f864926161f00b4f015e2175f8a6dab71fb3141ffb779372ad286795282a6e.jpg)  
(b) 12L LeakScore Audit

![](images/be00a574e83d4656a0a6bc87e4f439e6fb638318d4a6e1ac1bc47b11fb12edb6.jpg)  
(c) 12L Generation (ppl<sub>ref</sub>)

![](images/67a43f65e80d74da22cfb04682dee1b299d962cadff2aafc4e8c3eab1d01d643.jpg)  
Figure 4: Diagnostics for seven 12-layer checkpoints in a shared order: (a) validation loss, (b) checkpoint-level LeakScore, and (c) mean reference perplexity over 50 prompts. Open and shaded bars denote passing and failing checkpoints, respectively. Panel (c) uses a logarithmic axis with untransformed perplexity labels. CDR-token and CDR-prefix-mean abbreviate the two repaired variants.

Table 10: Prefix-invariance audit on nine core CDR checkpoints $( \epsilon = 1 0 ^ { - 5 } )$ . LeakScore is the maximum over 72 pairs; top-1 flip rate is the mean of per-pair rates. Controls and repairs pass on all tested pairs, while all pre-fix variants fail. Unmarked variants are 12-layer.
<table><tr><td>Variant</td><td>Val loss</td><td>LeakScore</td><td>Top-1 flip rate</td><td>Verdict</td></tr><tr><td>Baseline</td><td>3.6900</td><td>0.0</td><td>0.0</td><td>pass</td></tr><tr><td>Static AttnRes</td><td>3.6676</td><td>0.0</td><td>0.0</td><td>pass</td></tr><tr><td>Baseline 24L</td><td>3.5958</td><td>0.0</td><td>0.0</td><td>pass</td></tr><tr><td>CDR-token-fix</td><td>3.6766</td><td>0.0</td><td>0.0</td><td>pass</td></tr><tr><td>CDR-prefix-mean-fix</td><td>3.6731</td><td>0.0</td><td>0.0</td><td>pass</td></tr><tr><td>CDR-gated</td><td>2.6015</td><td>8.52</td><td>0.483</td><td>fail</td></tr><tr><td>CDR-last</td><td>2.5488</td><td>11.31</td><td>0.509</td><td>fail</td></tr><tr><td>CDR-mean</td><td>2.8473</td><td>28.64</td><td>0.720</td><td>fail</td></tr><tr><td>CDR-last 24L</td><td>1.4259</td><td>34.22</td><td>0.951</td><td>fail</td></tr></table>

Together, the audit, generation diagnostics, implementation inspection, and 12-layer repair comparisons support the interpretation that the large pre-fix gain depends on future-token access. The independently trained repairs do not isolate every efect of the implementation change, and we do not report a repaired 24-layer checkpoint. Within these boundaries, the joint disappearance of audit failure and anomalously low loss means that the original result cannot be treated as evidence of a valid autoregressive improvement.

## 7.8. Implications of the Case Study

The practical implication of this case is to audit the complete computation before interpreting a lower teacher-forced validation loss as an architectural improvement. Layer-wise localization can guide implementation inspection, and independently trained causal repairs can test whether the apparent advantage persists under a valid dependency structure.

## 8. Guidelines and Limitations

## 8.1. Causality-Safe Design Guidelines

The implementation question is what information can influence an output at position i. Every contributing summary, state update, routing decision, and normalization statistic must respect the prefix boundary. Table 1 gives concrete repairs: token-local or prefix-only summaries, left-padded convolution, prefix-KV aggregation, and prefix-consistent routing and normalization. A causal attention mask cannot compensate for future access elsewhere in the computation.

The CDR case gives a specific example: the query controls mixing across depth, yet the summary used to construct it mixes across time. Restricting the token-to-token attention pattern therefore leaves this auxiliary path unchecked. Auditing should cover such control signals as well as the primary sequence-mixing operator, with clean controls matched to the execution conditions. Prefix-invariance tests can reveal violations and guide inspection, while a passing result remains limited to the tested inputs and tolerance.

## 8.2. Limitations

Our results should be interpreted with several limitations in mind. First, the prefix-invariance audit used here should be viewed as a necessary practical check rather than a formal proof of causality. Passing the test provides evidence that a model does not exhibit detectable sufix-to-prefix influence under the evaluated perturbations and tolerance, but it does not establish that every possible execution path or implementation detail is causally correct in a formal verification sense. A failure exceeding a calibrated tolerance under controlled evaluation is evidence of sufix-to-prefix influence; attributing it to a hidden auxiliary path requires implementation or localization evidence. New architectural components should therefore undergo implementation review as well as behavioral testing.

Second, although we study several representative families of auxiliary modules, our coverage is not exhaustive. In particular, we do not claim to cover all state-space models, all mixture-of-experts designs, or the full range of hybrid architectures combining routing, compression, recurrence, retrieval, and state updates. Some SSM or MoE variants may be causally safe by construction, whereas others may contain subtler non-causal paths than the ones examined here. Extending the audit to a broader set of modern sequence architectures remains an important direction for future work.

Third, the synthetic experiments are intentionally simplified in order to isolate causal structure from unrelated optimization efects. For that reason, they should be interpreted as controlled behavioral audits rather than as tasklevel benchmark evaluations of architectural quality. The purpose of these experiments is to test whether a module can leak future information, not to provide a comprehensive comparison of model performance across datasets and training regimes.

Finally, the CDR-AttnRes case study ofers a realistic and informative example, but it is still a single concrete case from our own architectural exploration. Additional case studies on independent codebases and publicly released implementations would further strengthen the empirical scope of the paper. We therefore regard the present results as evidence for the importance of causal auditing, rather than as a claim that every adaptive sequence model in the literature is afected by hidden future-token leakage.

## 9. Conclusion

Hidden future-token leakage can create false architectural gains even when the causal attention mask is correct. In the CDR-AttnRes case, anomalously low teacher-forced validation loss co-occurs with detectable prefix leakage and substantially worse reference-model perplexity during free-running generation. Layer-wise diagnostics and implementation inspection identify a full-sequence summary broadcast through the depth-routing query path. Two independently trained 12-layer causal repairs remove detectable leakage, return validation loss and reference perplexity to the control scale, and remove the large apparent advantage.

The main empirical finding is that, with a correct causal attention mask, hidden future-token leakage is still suficient to produce a large performance advantage that invites interpretation as an architectural improvement. That advantage disappears under causal repair and is therefore not a valid autoregressive improvement. The established prefix-invariance contract, LeakScore, complementary metrics, and FirstLeakLayer provide diagnostic evidence for the case. Six synthetic implementation pairs, clean pretrained controls, controlled fault injection, and cache/precision checks support interpretation of those diagnostics under the tested conditions. They do not constitute a proof of causality for all inputs or implementations.

The practical implication is to check the complete dependency structure before treating lower teacher-forced loss as architectural progress. Pooling, temporal mixing, state aggregation, routing, and cross-time normalization can introduce future-token access outside masked attention. Combining causal checks with free-running evaluation, root-cause inspection, and repair validation can help distinguish a valid autoregressive improvement from an apparent gain enabled by an invalid dependency path.

## Appendix A. Reproducibility

## Appendix A.1. CDR Training Configuration

All CDR checkpoints were trained with the shared configuration in Table A.1; both the 12-layer and the 24-layer variants use the same hyperparameters.

Table A.1: Shared CDR training configuration. Parameter counts are unique trainable parameters; the input embedding and the output head share a single weight matrix, counted once. CDR adds rank-r down- and up-projections for dynamic routing-query corrections in the residual aggregation paths before both the attention and MLP sublayers.
<table><tr><td>Data</td><td>OpenWebText, GPT-2 tokenizer (vocabulary padded from 50257 to 50304), block size 512</td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta = ( 0 . 9 , 0 . 9 5 )$  , weight decay 0.1, gradi- ent clip 1.0), dropout 0.0, no bias</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$  , cosine decay to  $3 \times 1 0 ^ { - 5 }$  , warmup 1000 iterations</td></tr><tr><td>Iterations</td><td>20,000</td></tr><tr><td>Effective batch</td><td>64 sequences/step (batch size 4 per device, gradient accumulation  $1 6 / 2 \mathrm { G P U s } = 8 $  per device, 2 devices)</td></tr><tr><td>Precision</td><td>bfloat16 1337 (all nine core checkpoints)</td></tr><tr><td>Training seed</td><td>Checkpoint selection best validation loss (evaluation every 1000 itera-</td></tr><tr><td></td><td>tions, 100 evaluation iterations)</td></tr><tr><td>Model sizes</td><td>12 layers, 8 heads, 512-dimensional embedding: 63.8M parameters (baseline), 65.4M with CDR (r = 64); 24 layers, 8 heads, 512-dimensional em- bedding: 101.5M (baseline), 104.7M with CDR  $( r = 6 4 )$ </td></tr></table>

## Appendix A.2. Auditing Protocols

Public-model audit conditions are specified in Section 6; CDR audit and generation conditions are specified in Section 7, with metric definitions in Section 4. The supplementary materials provide per-experiment result files, the fixed prompt set, saved continuations, and the checkpoint-to-implementation mapping. The nine core checkpoints use training seed 1337; auxiliary rank and seed explorations are not part of the core checkpoint comparison.

## Appendix A.3. Representative Generation Samples

We show the first 80 whitespace-delimited units of the saved continuation for the same prompt (idx = 0) from three 12-layer checkpoints: Baseline, pre-fix CDR-last, and repaired CDR-token-fix. Wording and generated errors are preserved; line breaks and quotation marks are normalized for typesetting, and [. . . ] marks truncation. The displayed <|endoftext|> marker is present in the saved baseline output. These illustrative excerpts do not establish factual accuracy or replace the aggregate diagnostics in Table 8. Full outputs for all 50 prompts, including the 24-layer variants, are provided in the supplementary materials.

## Shared prompt.

Reporting from Perry, Iowa — Defending himself against charges of isolationism, Republican presidential candidate Ron Paul told voters in Iowa on Thursday that western sanctions against Iran are

## Baseline (12L control).

too high. “Don’t believe me when you get to war,” he said by phone. “This is something that the government is supposed to protect,” he said.<|endoftext|>There’s a reason for the Trump administration’s decision to use climate "perform" heathens to decry the risks of climate change under the President-elect Donald Trump. "Because the climate change ban on its own will not be renewed," said Kelly "I am sure that it will be renewed as soon as possible.” How much does warming [. . . ]

## CDR-last (12L pre-fix).

self un un state un low un low low heavy self heavy un highly un heavy un self un self un heavy inf self un un highly low heavy deep heavy kind low large form form un particularly large very largely very small low un high large inf human super small heavy low heavily small large small un heavy large single super completely large single un large form small traditional large large large single super large large large super low [. . . ]

CDR-token-fix (12L repair).

too sweeping but he continued with an comments he made earlier this month when he asked reporters to listen to him. “His statements were going to be something, and it was a hard thing to get him to do it, to attack people,” Paul said. “He was, after all, a very, very good friend. He had a really good message. But he knew he thought it was important to have a back-up.” Paul, a senator from Virginia, said he has [. . . ]

## Funding

This paper was supported in part by the China Guangxi Science and Technology Plan Project under grant AD23026096, Chile CONICYT FONDE-CYT Regular under grant 1181809, and Chile CONICYT FONDEF under grant ID16I10466.

## Data and Code Availability

The accompanying supplementary materials contain the auditing and generation code, saved results for the synthetic and public-model experiments, CDR audit and generation summaries, the fixed OpenWeb-Text prompt set, raw continuations, descriptive correlation statistics, and checkpoint-to-implementation mappings. Model checkpoints are not included because of storage constraints. Checkpoint inference requires the corresponding weights; retraining additionally requires preprocessed Open-WebText data and the training environment. Public-model audits and reference scoring require the relevant public and reference models. Paths to external checkpoints and datasets must be configured before running the corresponding experiments; exact numerical agreement across environments is not guaranteed.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used DeepSeek-V4-Flash in order to improve the clarity and readability of the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret, Transformers are RNNs: Fast autoregressive transformers with linear attention, in: Proceedings of the 37th International Conference on Machine Learning, Vol. 119 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 5156–5165. URL https://proceedings.mlr.press/v119/katharopoulos20a.ht ml

[2] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, A. Santoro, Mixture-of-Depths: Dynamically allocating compute in Transformer-based language models, arXiv preprint arXiv:2404.02258 (2024). doi:10.48550/arXiv.2404.02258. URL https://arxiv.org/abs/2404.02258

[3] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, arXiv preprint arXiv:2312.00752 (2024). doi:10.48550/a rXiv.2312.00752. URL https://arxiv.org/abs/2312.00752

[4] S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (9) (2023) 100804. doi: 10.1016/j.patter.2023.100804. URL https://doi.org/10.1016/j.patter.2023.100804

[5] Kimi Team, G. Chen, Y. Zhang, J. Su, W. Xu, S. Pan, Y. Wang, Y. Wang, G. Chen, B. Yin, Y. Chen, J. Yan, M. Wei, Y. Zhang, F. Meng, C. Hong, X. Xie, S. Liu, E. Lu, Y. Tai, Y. Chen, X. Men, H. Guo, Y. Charles, H. Lu, L. Sui, J. Zhu, Z. Zhou, W. He, W. Huang, X. Xu, Y. Wang, G. Lai, Y. Du, Y. Wu, Z. Yang, X. Zhou, Attention Residuals, arXiv preprint arXiv:2603.15031 (2026). doi:10.48550/arXiv.2603. 15031.

[6] C. Xu, S. Guan, D. Greene, M.-T. Kechadi, Benchmark data contamination of large language models: A survey, arXiv preprint arXiv:2406.04244 (2024). doi:10.48550/arXiv.2406.04244. URL https://arxiv.org/abs/2406.04244

[7] J. Dodge, M. Sap, A. Marasović, W. Agnew, G. Ilharco, D. Groeneveld, M. Mitchell, M. Gardner, Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus, in: Proceedings of the

2021 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, 2021, pp. 1286–1305. doi: 10.18653/v1/2021.emnlp-main.98.

[8] T. Wang, FBS: Modeling native parallel reading inside a Transformer, in: Findings of the Association for Computational Linguistics: ACL 2026, Association for Computational Linguistics, San Diego, California, United States, 2026, pp. 4106–4137. doi:10.18653/v1/2026.finding s-acl.200.

[9] R. J. George, J. Cruden, W. Adkisson, X. Zhong, H. Zhang, A. Anandkumar, TorchLean: Formalizing neural networks in Lean, arXiv preprint arXiv:2602.22631v2 (2026). doi:10.48550/arXiv.2602.22631. URL https://arxiv.org/abs/2602.22631v2

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017, pp. 5998–6008. URL https://papers.nips.cc/paper/7181-attention-is-all-you -need

[11] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, R. Salakhutdinov, Transformer-XL: Attentive language models beyond a fixed-length context, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, 2019, pp. 2978–2988. doi:10.18653/v1/P19-1285. URL https://aclanthology.org/P19-1285/

[12] I. Beltagy, M. E. Peters, A. Cohan, Longformer: The long-document Transformer, arXiv preprint arXiv:2004.05150 (2020). doi:10.48550/a rXiv.2004.05150. URL https://arxiv.org/abs/2004.05150

[13] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, K. Kavukcuoglu, WaveNet: A generative model for raw audio, arXiv preprint arXiv:1609.03499 (2016). doi:10.48550/arXiv.1609.03499.

[14] A. Gu, K. Goel, C. Ré, Eficiently modeling long sequences with structured state spaces, in: International Conference on Learning Representations, 2022.

URL https://openreview.net/forum?id=uXl3bZLkr3c

[15] T. Dao, A. Gu, Transformers are SSMs: Generalized models and eficient algorithms through Structured State Space Duality, in: Proceedings of the 41st International Conference on Machine Learning, Vol. 235 of Proceedings of Machine Learning Research, PMLR, 2024, pp. 10041– 10071.

[16] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, F. Wei, Retentive Network: A successor to Transformer for large language models, arXiv preprint arXiv:2307.08621 (2023). doi:10.48550/arXiv.230 7.08621.

[17] B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, S. Biderman, H. Cao, X. Cheng, M. Chung, L. Derczynski, X. Du, M. Grella, K. Gv, X. He, H. Hou, P. Kazienko, J. Kocon, J. Kong, B. Koptyra, H. Lau, J. Lin, K. S. I. Mantri, F. Mom, A. Saito, G. Song, X. Tang, J. Wind, S. Woźniak, Z. Zhang, Q. Zhou, J. Zhu, R.-J. Zhu, RWKV: Reinventing RNNs for the Transformer era, in: H. Bouamor, J. Pino, K. Bali (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, Association for Computational Linguistics, Singapore, 2023, pp. 14048–14077. doi:10.18653/v1/2023.findings-emn lp.936.

[18] M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, C. Re, Hyena Hierarchy: Towards larger convolutional language models, in: A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, J. Scarlett (Eds.), Proceedings of the 40th International Conference on Machine Learning, Vol. 202 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 28043–28078. URL https://proceedings.mlr.press/v202/poli23a.html

[19] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hin ton, J. Dean, Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, in: International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=B1ckMDqlg

[20] W. Fedus, B. Zoph, N. Shazeer, Switch Transformers: Scaling to trillion parameter models with simple and eficient sparsity, Journal of Machine

Learning Research 23 (120) (2022) 1–39. URL https://jmlr.org/papers/v23/21-0998.html

[21] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M. Dai, Z. Chen, Q. V. Le, J. Laudon, Mixture-of-experts with Expert Choice Routing, in: Advances in Neural Information Processing Systems, Vol. 35, Curran Associates, Inc., 2022, pp. 7103–7114. doi:10.52202/068431-0515. URL https://papers.nips.cc/paper\_files/paper/2022/hash/2f0 0ecd787b432c1d36f3de9800728eb-Abstract-Conference.html

[22] H. Sun, Y. Liu, Y. Wu, L. Sun, Expert threshold routing for autoregressive language modeling with dynamic computation allocation and load balancing, arXiv preprint arXiv:2603.11535 (2026). doi: 10.48550/arXiv.2603.11535. URL https://arxiv.org/abs/2603.11535

[23] K. Pei, Y. Cao, J. Yang, S. Jana, DeepXplore: Automated whitebox testing of deep learning systems, in: Proceedings of the 26th Symposium on Operating Systems Principles, ACM, 2017, pp. 1–18. doi:10.1145/ 3132747.3132785. URL https://doi.org/10.1145/3132747.3132785

[24] K. Xu, Z. Shi, H. Zhang, Y. Wang, K.-W. Chang, M. Huang, B. Kailkhura, X. Lin, C.-J. Hsieh, Automatic perturbation analysis for scalable certified robustness and beyond, in: Advances in Neural Information Processing Systems, Vol. 33, Curran Associates, Inc., 2020, pp. 1129–1141. URL https://proceedings.neurips.cc/paper/2020/hash/0cbc567 1ae26f67871cb914d81ef8fc1-Abstract.html