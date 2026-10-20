# Fine-Grained Units, Flat Decisions: Hierarchical Micro-Expert Pruning for Mixture-of-Experts Models

Zukang Xu<sup>1</sup>\* Xinpei Gao<sup>1,2</sup>\* Xing Hu<sup>1,3</sup>

Zhe Jiang<sup>3</sup> Dawei Yang<sup>1†</sup>

<sup>1</sup>Houmo AI

<sup>2</sup>University of Science and Technology of China

<sup>3</sup>Southeast University

## Abstract

Micro-expert pruning reduces the capacity of mixture-of-experts (MoE) models by removing coupled channels within routed experts. However, finer pruning granularity alone does not guarantee the preservation of downstream capabilities, as similar language-modeling perplexity can accompany substantially different task performance. We revisit micro-expert mask construction as two distinct decisions: allocating retained capacity across experts and selecting channels within that capacity. We propose UC-RAMP, a two-stage post-training pruning framework that explicitly separates these decisions. CAPE uses cross-domain calibration evidence to redistribute expert widths under an exact layer-wise budget, while RAMP refines channel composition at fixed widths through one-out, one-in exchanges guided by a channel-interaction surrogate. Across three MoE models at 25% and 50% routed-expert channel pruning, UC-RAMP achieves the highest ten-task mean accuracy among evaluated pruning methods in five of six settings. At 50% pruning, it outperforms the strongest evaluated baseline by 8.19, 3.96, and 6.14 percentage points across the three models, respectively. In a single-A100 deployment at 25% pruning, UC-RAMP reduces measured time to first token by 19.7–23.9% and time per output token by 20.2–22.3% relative to the unpruned model. The code is available at https: //github.com/xzktx003/UC\_RAMP.

## 1 Introduction

Sparse mixture-of-experts (MoE) models reduce per-token computation by routing each input to only a subset of experts while retaining a substantially larger parameter pool (Shazeer et al., 2017; Fedus et al., 2022). However, sparse activation does not eliminate the storage and deployment costs of the full expert pool. Post-training pruning can reduce these costs without retraining the model from scratch. Existing pruning methods operate at different granularities, ranging from individual weights (Frantar and Alistarh, 2023; Sun et al., 2024) and coupled model dimensions (Ma et al., 2023; Ashkboos et al., 2024) to entire experts (Lu et al., 2024; Lasby et al., 2026). While wholeexpert pruning removes all channels of a selected expert, micro-expert pruning instead removes corresponding gate and up-projection rows together with a down-projection column (Xu et al., 2026), allowing expert capacity to be reduced without necessarily removing the entire expert.

Channel-level pruning offers greater flexibility in constructing structural masks, but finer granularity alone does not guarantee the preservation of downstream capabilities. Figure 1 illustrates this limitation on Qwen3-30B-A3B-Instruct-2507. At 50% routed-expert intermediate-channel pruning, CAMERA’s WikiText-2 perplexity remains close to that of the dense model (7.61 versus 7.26), whereas MATH500 accuracy drops from 80.6% to 10.2% and HumanEval pass@1 from 95.7% to 0%. In contrast, UC-RAMP achieves substantially higher task accuracy under the same pruning budget, while exhibiting similarly low perplexity. These observations highlight the importance of mask construction beyond pruning granularity alone, motivating a closer examination of how individual channel scores are translated into a structural pruning mask.

A micro-expert pruning mask encodes two distinct structural decisions: capacity allocation, which determines the number of channels retained in each expert, and channel composition, which specifies the identities of those channels. Under a fixed layer-wise channel budget, a global Top-K ranking determines both decisions simultaneously: selecting a channel not only includes it in the retained set but also allocates a portion of the budget to its expert. However, these decisions can benefit from different sources of information. Cross-domain response statistics can guide expertcapacity allocation, whereas interactions among channels provide complementary information for evaluating channel sets at fixed widths. While prior methods such as MoE-Slimming consider nonuniform expert widths (Ding et al., 2026), our focus is on explicitly separating capacity allocation from channel composition, allowing each decision to be guided by its corresponding evidence.

![](images/2091ab9e44f69f94198aa8b9786b655001ea9c017767a29da6306824ce765e54.jpg)  
Figure 1: Perplexity and downstream task performance at 50% routed-expert channel pruning on Qwen3-30B-A3B-Instruct-2507. CAMERA exhibits substantial degradation on MATH500 and HumanEval despite a small change in WikiText-2 perplexity. At the same nominal pruning ratio, UC-RAMP achieves substantially higher task accuracy. Lower perplexity and higher task accuracy indicate better performance.

To operationalize this separation, we introduce UC-RAMP, a two-stage micro-expert pruning framework consisting of Capability-Aware Pruning via Envelopes (CAPE) and Reconstruction-Aware Micro-expert Pruning (RAMP). In the first stage, CAPE establishes a shared channel ranking from cross-domain calibration statistics and evaluates each expert on a common reference channel set. An expert-level multiplier then adjusts the competition for retained capacity, enabling width redistribution under an exact layer-wise budget while preserving the channel ranking within each expert. In the second stage, RAMP freezes the allocated widths and incorporates pairwise channel interactions into a reconstruction-based surrogate. It then refines the retained channel sets through one-out, one-in exchanges that reduce the surrogate without changing expert widths. Both stages operate without gradient updates or modifications to the router.

Our main contributions are summarized as follows:

• We formulate micro-expert pruning as two distinct structural decisions: allocating retained capacity across experts and selecting channels within the allocated capacity. This formulation makes explicit the two decisions jointly determined by conventional global channel ranking.

• We propose UC-RAMP, a two-stage posttraining pruning framework. CAPE uses cross-domain calibration evidence to redistribute expert capacity under an exact channel budget, while RAMP incorporates channel interactions to refine retained channel sets without changing expert widths.

• We evaluate UC-RAMP across three MoE models and two pruning ratios. UC-RAMP achieves the highest mean accuracy among the evaluated pruning methods in five of six settings, including all three models at 50% pruning. Component ablations characterize the contributions of the two stages, and deployment measurements demonstrate inference latency reductions with nonuniform expert widths.

## 2 Related Work

Post-training and whole-expert pruning. Posttraining pruning has been widely studied for reducing the storage and computational costs of large language models. SparseGPT (Frantar and Alistarh, 2023) and Wanda (Sun et al., 2024) perform weight-level pruning using reconstruction information and weight-activation saliency, respectively, while LLM-Pruner (Ma et al., 2023) and SliceGPT (Ashkboos et al., 2024) remove coupled model dimensions. For MoE models, another line of work focuses on expert-level compression. Lu et al. (2024) investigate expert pruning and skipping, REAP (Lasby et al., 2026) evaluates routed expert contributions, and DiEP (Bai et al., 2025) optimizes nonuniform expert pruning across layers. More recently, AIMER (Liu et al., 2026a) employs calibration-free weight statistics, whereas MAN/MSAN (Liu et al., 2026b) leverages routedtoken activation statistics. These approaches primarily operate at the granularity of complete experts, in contrast to micro-expert pruning, which allows capacity to be reduced within individual experts.

Expert-internal pruning and dependent channel selection. Finer-grained MoE compression retains greater flexibility in selecting expert-internal structures. MoE-Pruner (Xie et al., 2024) incorporates routing information into expert-internal parameter pruning, while CAMERA (Xu et al., 2026) introduces micro-experts as coupled channels across the gate, up, and down projections of SwiGLU experts. MoE-Slimming (Ding et al., 2026) further considers nonuniform expert widths through attribution-guided and coverage-based structural compression. Beyond independent importance estimates, The Sparsity Whisperer (Kong et al., 2026) highlights the importance of dependencies among jointly pruned units. Building on this line of research, UC-RAMP explicitly separates expert-capacity allocation from fixed-width channel composition. CAPE uses cross-domain responses evaluated on shared reference channel sets to redistribute expert widths under an exact layerwise budget, whereas RAMP incorporates pairwise channel interactions to refine channel membership without changing the allocated widths.

## 3 Method

## 3.1 Capacity and Composition of Micro-Expert Masks

Consider a routed expert e in MoE layer ℓ with intermediate width C. For a token representation $x _ { t } .$ , the activation of its c-th SwiGLU channel is

$$
z _ { t , e , c } = \mathrm { S i L U } ( w _ { g , e , c } ^ { \top } x _ { t } ) ( w _ { u , e , c } ^ { \top } x _ { t } ) .\tag{1}
$$

Here and below, we omit the layer index in channel activations and weight vectors for clarity. Let $\boldsymbol { r } _ { t , e }$ denote the routing weight used in the model’s forward pass. The routed expert contribution can

be expressed as

$$
r _ { t , e } E _ { e } ( x _ { t } ) = \sum _ { c = 1 } ^ { C } r _ { t , e } z _ { t , e , c } w _ { d , e , c } .\tag{2}
$$

The corresponding gate and up-projection rows and down-projection column constitute a micro-expert, which is removed as a single structured unit.

Let $m _ { \ell , e , c } \in \{ 0 , 1 \}$ indicate whether channel c is retained. The retained width of expert e and the exact layer-wise channel budget are defined as

$$
k _ { \ell , e } = \sum _ { c = 1 } ^ { C } m _ { \ell , e , c } , \sum _ { e } k _ { \ell , e } = K _ { \ell } .\tag{3}
$$

Here $K _ { \ell }$ is the retained channel budget in layer ℓ. A pruning mask therefore specifies two structural decisions: capacity allocation, represented by the expert widths $k _ { \ell , e } .$ , and channel composition, represented by the retained sets $S _ { \ell , e } = \{ c : m _ { \ell , e , c } = 1 \}$ where $| S _ { \ell , e } | = k _ { \ell , e }$

A global Top- $K _ { \ell }$ ranking of individual channel scores determines both decisions simultaneously. Changing an expert’s scores can redistribute capacity across experts even when the channel ordering within that expert remains unchanged. Conversely, exchanging a retained channel with a pruned channel within the same expert changes its composition without affecting its width. UC-RAMP explicitly separates these operations: CAPE determines the expert widths and initial retained sets, whereas RAMP refines channel composition while preserving the allocated widths. Figure 2 illustrates this two-stage interface.

## 3.2 CAPE: Allocating Expert Capacity

CAPE redistributes retained channels across experts under a fixed layer-wise budget $K _ { \ell }$ . It first constructs a cross-domain channel ranking and then adjusts expert-level capacity competition without changing the channel ordering within each expert.

Cross-domain channel importance. Let $\mathcal { D }$ denote the calibration domains, including a generallanguage domain G and complementary domains. For tokens routed to expert $e ,$ we define $a _ { t , e , c } =$ $\bar { r } _ { t , e } z _ { t , e , c }$ , where $\bar { r } _ { t , e }$ is the routing probability normalized over the selected experts. To account for the scale of each channel’s output contribution, we weight its down-projection vector by the next layer’s RMSNorm coefficient $\gamma ^ { ( \ell + 1 ) }$ , obtaining $\widetilde { w } _ { d , e , c } = \gamma ^ { ( \ell + 1 ) } \odot w _ { d , e , c }$ . The domain-specific channel importance combines accumulated response energy and peak response:

![](images/e2c30249f46905e193ae0ef0c8e0e5c66fe8f42dc772bb223e87c35ddf7c9322.jpg)  
Figure 2: Overview of UC-RAMP. CAPE combines cross-domain channel importance with expert-level capacity adjustment to allocate retained widths under an exact layer-wise budget. RAMP then constructs an interaction-aware surrogate and refines channel composition through one-out, one-in exchanges while preserving the allocated widths.

$$
\begin{array} { r l r } {  { I _ { e , c } ^ { ( d ) } = [ ( 1 - \alpha ) \sum _ { t \in d } a _ { t , e , c } ^ { 2 }  } } \\ & { } & {  + \alpha \operatorname* { m a x } _ { t \in d } a _ { t , e , c } ^ { 2 } ] \| \widetilde { w } _ { d , e , c } \| _ { 2 } ^ { 2 } . } \end{array}\tag{4}
$$

Because calibration domains differ in content and routing patterns, their raw importance scores may have different scales. Within each layer, we compute $s _ { d } = \mathrm { m e a n } _ { e , c } I _ { e , c } ^ { ( d ) }$ and the geometric reference $\begin{array} { r } { s _ { \mathrm { r e f } } = ( \prod _ { d \in \mathcal { D } } { s _ { d } } ) ^ { 1 / | \mathcal { D } | } } \end{array}$ . We then normalize each domain as $\widehat { I } _ { e , c } ^ { ( d ) } = I _ { e , c } ^ { ( d ) } s _ { \mathrm { r e f } } / s _ { d }$ . This alignment equalizes domain-wise mean scores while preserving their relative channel rankings.

The normalized scores are combined into a general-language-weighted channel anchor:

$$
A _ { e , c } = \lambda \widehat { I } _ { e , c } ^ { ( G ) } + \frac { 1 - \lambda } { | \mathscr { D } | - 1 } \sum _ { d \neq G } \widehat { I } _ { e , c } ^ { ( d ) } .\tag{5}
$$

The anchor $A _ { e , c }$ provides a unified ranking of individual channels across calibration domains.

Expert-level capacity adjustment. Within layer $\ell ,$ a global Top- $K _ { \ell }$ selection based on A defines reference widths $k _ { e } ^ { 0 }$ and retained channel sets $S _ { e } ^ { \mathrm { r e f } }$ satisfying $\textstyle \sum _ { e } k _ { e } ^ { 0 } = K _ { \ell }$ . We evaluate every calibration domain on the same expert-specific reference set and define the expert multiplier

$$
\rho _ { e } = \sqrt { \frac { \operatorname* { m a x } _ { d } \sum _ { c \in S _ { e } ^ { \mathrm { r e f } } } \widehat { I } _ { e , c } ^ { ( d ) } } { \operatorname* { m a x } ( \sum _ { c \in S _ { e } ^ { \mathrm { r e f } } } A _ { e , c } , \varepsilon ) } } .\tag{6}
$$

For an expert with zero reference width, its highest-ranked anchor channel is used only to compute the multiplier, without changing the reference allocation.

The maximum cross-domain response forms a capability envelope over the expert’s reference channels. The multiplier $\rho _ { e }$ compares this envelope with the fused anchor response on the same <sup>W̃ W̃</sup>channel set, providing an expert-level signal for capacity redistribution. Applying it uniformly to an expert’s channels gives $T _ { e , c } = \rho _ { e } A _ { e , c } .$ For $\rho _ { e } > 0$ this reweighting changes competition for capacity across experts while preserving the channel ranking within each expert. Thus, cross-domain evidence adjusts expert-level capacity priorities without redefining the channel anchor.

Budget-constrained width allocation. Let $A _ { e , ( j ) }$ denote the j-th largest anchor score within expert e. To constrain capacity migration from the reference allocation, we impose lower and upper width bounds $k _ { e } ^ { \mathrm { m i n } }$ and $k _ { e } ^ { \mathrm { m a x } }$ around $k _ { e } ^ { 0 }$ . CAPE determines the retained widths by solving

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \{ k _ { e } \} } } & { \displaystyle \sum _ { e } \rho _ { e } \sum _ { j = 1 } ^ { k _ { e } } A _ { e , ( j ) } } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { e } k _ { e } = K _ { \ell } , \qquad k _ { e } \in \mathbb { Z } , } \\ & { k _ { e } ^ { \mathrm { m i n } } \leq k _ { e } \leq k _ { e } ^ { \mathrm { m a x } } . } \end{array}\tag{7}
$$

The bounds define a local trust region around the reference widths. Since the weighted marginal scores are nonincreasing within each expert, the objective is optimized exactly by retaining all lowerbound prefixes and selecting the largest eligible remaining marginal scores until the budget is filled.

CAPE outputs the allocated widths $k _ { e }$ and initial retained sets $S _ { e } ^ { ( 0 ) } = \mathrm { T o p K } ( A _ { e } , k _ { e } )$ . The width bounds and the optimality argument are detailed in Appendices $\mathbf { A }$ and B, respectively. These widths remain fixed during the subsequent RAMP refinement.

## 3.3 RAMP: Fixed-Width Channel Refinement

CAPE constructs initial retained sets using individual channel importance scores, without explicitly accounting for interactions among simultaneously removed channels. RAMP addresses this limitation by refining channel composition while keeping the expert widths determined by CAPE fixed. To estimate these interactions, RAMP collects routed calibration statistics from the intermediate model $\mathcal { M } ^ { ( 0 ) }$ obtained after applying CAPE’s initial pruning masks.

Let $m _ { e } \in \{ 0 , 1 \} ^ { C }$ denote the retained-channel mask of expert e, with the layer index omitted, and let $p _ { e } = { \bf 1 } - m _ { e }$ indicate deleted channels. For tokens from domain d routed to expert $e ,$ define $Z _ { t , c } ^ { ( d , e ) } = r _ { t , e } z _ { t , e , c } ,$ , where $\boldsymbol { r } _ { t , e }$ is the routing weight used in the forward pass. Under fixed channel responses, the output contribution of the deleted channels is

$$
q _ { t } ( p _ { e } ) = \sum _ { c } p _ { e , c } Z _ { t , c } ^ { ( d , e ) } \widetilde { w } _ { d , e , c } .\tag{8}
$$

Its mean squared norm can be written as $p _ { e } ^ { \top } G _ { e } ^ { ( d ) } p _ { e }$ where

$$
G _ { e } ^ { ( d ) } = \frac { ( Z _ { e } ^ { ( d ) } ) ^ { \top } Z _ { e } ^ { ( d ) } } { N _ { d , e } } \odot ( \widetilde { W } _ { e } ^ { \top } \widetilde { W } _ { e } ) .\tag{9}
$$

Here $N _ { d , e }$ is the number of routed calibration tokens, and $\breve { W } _ { e }$ contains the scaled down-projection vectors as columns. We aggregate the domainspecific Gram matrices as

$$
G _ { e } = \sum _ { d \in \mathcal { D } } \omega _ { d } G _ { e } ^ { ( d ) } ,\tag{10}
$$

where $\omega _ { G } = \lambda$ and $\omega _ { d } = ( 1 - \lambda ) / ( | \mathcal { D } | - 1 )$ for d $\neq$ $G ,$ , following the same domain weighting scheme as CAPE in $\mathrm { E q . } \left( 5 \right)$ . The off-diagonal entries of $G _ { e }$ capture channel co-activation and the alignment of their down-projection outputs.

Rather than directly using the diagonal energies of $G _ { e } ,$ RAMP preserves CAPE’s individual channel importance while incorporating the interaction structure of $G _ { e }$

We obtain a normalized Gram matrix $R _ { e }$ by rescaling the pairwise interaction entries of $G _ { e }$ :

$$
R _ { e , i j } = \left\{ \begin{array} { l l } { 1 , \quad } & { i = j , } \\ { \frac { G _ { e , i j } } { \sqrt { G _ { e , i i } G _ { e , j j } } } , \quad } & { i \neq j , } \\ { 0 , \quad } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{11}
$$

Let $D _ { A _ { e } } = \mathrm { d i a g } ( \sqrt { A _ { e } } )$ . We then construct the interaction-aware surrogate:

$$
\begin{array} { c } { { H _ { e } = D _ { A _ { e } } \big [ ( 1 - \beta ) I + \beta R _ { e } \big ] D _ { A _ { e } } , } } \\ { { \xi _ { e } ( p _ { e } ) = p _ { e } ^ { \top } H _ { e } p _ { e } . } } \end{array}\tag{12}
$$

By construction, the diagonal and off-diagonal entries of $H _ { e }$ satisfy

$$
\begin{array} { l l } { H _ { e , i i } = A _ { e , i } , } \\ { H _ { e , i j } = \beta \sqrt { A _ { e , i } A _ { e , j } } R _ { e , i j } } & { ( i \neq j ) . } \end{array}\tag{13}
$$

The diagonal preserves CAPE’s individual channel importance, whereas the off-diagonal terms incorporate pairwise channel interactions, controlled by $\beta .$ The resulting quadratic form evaluates deletedchannel sets using both sources of information.

RAMP minimizes this surrogate locally through fixed-width channel exchanges. For a retained channel i and a deleted channel $j ,$ the candidate update is $p _ { e } ^ { \prime } = p _ { e } + { \bf e } _ { i } - { \bf e } _ { j }$ . We evaluate the exact surrogate change $\Delta _ { i \to j } = \mathcal { E } _ { e } ( p _ { e } ^ { \prime } ) - \mathcal { E } _ { e } ( p _ { e } )$ and accept the best candidate only if $\Delta _ { i  j } < - \tau$ , where $\tau \geq 0$ . The main configuration permits at most one accepted exchange per layer-expert. Each exchange preserves $k _ { e } ,$ and the acceptance criterion guarantees a strict decrease in the defined surrogate. For $0 \leq \beta \leq 1$ $H _ { e }$ is positive semidefinite, as shown in Appendix B.

The two stages follow a fixed execution order. CAPE applies the initial pruning masks sequentially from shallow to deep layers, producing the intermediate model $\mathcal { M } ^ { ( 0 ) }$ . RAMP then collects calibration statistics using this model and refines channel composition without changing the expert widths allocated by CAPE. The exact exchange formula and execution details are provided in Appendices B and A, respectively.

## 4 Experiments

## 4.1 Models, Budgets, and Evaluation

We evaluate UC-RAMP on Qwen3-30B-A3B-Instruct-2507, Gemma-4-26B-A4B-it, and Qwen3.6-35B-A3B at pruning ratios of 25% and 50%. These ratios refer exclusively to the intermediate channels of routed experts, excluding attention, embeddings, routers, shared experts, Gemma’s parallel dense MLP, and multimodal modules. We distinguish channel-level and expert-level pruning baselines in Table 1, as identical nominal pruning ratios can lead to different structural configurations and runtime characteristics.

UC-RAMP uses four calibration domains, each consisting of sequences of length 2048: 128 WikiText-2 training examples for general language (Merity et al., 2017); 64 GSM8K and 64 MathQA training examples for general reasoning (Cobbe et al., 2021; Amini et al., 2019); 128 competition-MATH training examples for advanced mathematics (Hendrycks et al., 2021); and 64 MBPP training examples for code (Austin et al., 2021). Calibration requires only forward-pass statistics, without gradient updates. We set the general-domain weight to λ = 0.9, distributing the remaining weight equally among the three specialized domains. The other hyperparameters are fixed at $\alpha = 0 . 9$ in Eq. (4) and $\beta = 0 . 2$ in Eq. (12). Details of the width trustregion constraints are provided in Appendix A.

Our channel-level baselines include CAMERA (Xu et al., 2026) and MoE-Slimming (Ding et al., 2026), while the whole-expert baselines comprise AIMER (Liu et al., 2026a), REAP (Lasby et al., 2026), and MAN/MSAN (Liu et al., 2026b). We also report results for the unpruned models. The baseline results retain the respective scoring procedures and calibration settings of each method.

Table 2 further disentangles the contributions of CAPE’s capacity redistribution and RAMP’s fixedwidth channel refinement.

All evaluations are conducted using EvalScope 1.4.1 with temperature set to 0, a maximum generation length of 2048 tokens, and a fixed random seed of 1234. We report WikiText-2 perplexity separately from accuracy on ten downstream benchmarks: PIQA (Bisk et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), GSM8K (Cobbe et al., 2021), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), RACE (Lai et al., 2017), GPQA-Diamond (Rein et al., 2024), MATH500 (Hendrycks et al., 2021; Lightman et al., 2024), and HumanEval pass@1 (Chen et al., 2021). The average accuracy reported in Tables 1 and 2 is the unweighted mean across the ten benchmarks, excluding perplexity. Detailed per-task results are provided in Appendix C.

## 4.2 Results across models and budgets

Overall Performance. UC-RAMP achieves the highest ten-task mean accuracy among pruned models in five of the six settings in Table 1, including all three models at 50% pruning. At this ratio, it reaches 75.43% on Qwen3-30B-A3B, 67.88% on Gemma-4-26B, and 74.23% on Qwen3.6-35B-A3B. These scores exceed the respective strongest baselines, MSAN (67.24%), MoE-Slimming (63.92%), and MAN (68.09%), by 8.19, 3.96, and 6.14 percentage points.

Capability Recovery. On Qwen3 at 50% pruning, UC-RAMP achieves near-dense MATH500 accuracy: 78.0%, compared with 80.6% for the unpruned model and 10.2% for CAMERA. HumanEval pass@1 shows substantial recovery over CAMERA (59.15% versus 0%), compared with 95.73% for the unpruned model. WikiText-2 perplexities remain similar: 7.68 for UC-RAMP and 7.61 for CAMERA, compared with 7.26 for the unpruned model. These task-level gains at similar perplexity address the capability loss highlighted in Figure 1.

Performance Across Architectures. The advantages at 50% pruning extend to Gemma-4 and Qwen3.6, although recovery varies across individual tasks and architectures. Their MATH500 accuracies reach 73.8% and 52.8%, compared with CAMERA’s 4.4% and 12.2%. HumanEval pass@1 reaches 14.02% and 75.61%, compared with zero for CAMERA on both models. Qwen3.6 at 25% pruning is the single exception in Table 1: UC-RAMP obtains 67.18% mean accuracy, while MSAN leads with 75.59%. Appendix C provides the complete task scores.

<table><tr><td colspan="2"></td><td colspan="2">Qwen3-30B-A3B</td><td colspan="2">Gemma-4-26B</td><td colspan="2">Qwen3.6-35B-A3B</td></tr><tr><td>Method</td><td>Unit</td><td>25%</td><td>50%</td><td>25%</td><td>50%</td><td>25%</td><td>50%</td></tr><tr><td>Dense</td><td>none</td><td>86.19</td><td>86.19</td><td>90.15</td><td>90.15</td><td>76.94</td><td>76.94</td></tr><tr><td>CAMERA</td><td>channel</td><td>68.63</td><td>58.46</td><td>69.12</td><td>56.12</td><td>69.90</td><td>56.13</td></tr><tr><td>MoE-Slimming</td><td>channel</td><td>78.43</td><td>67.00</td><td>79.85</td><td>63.92</td><td>64.71</td><td>49.62</td></tr><tr><td>AIMER</td><td>expert</td><td>80.82</td><td>64.46</td><td>62.27</td><td>41.83</td><td>70.97</td><td>48.89</td></tr><tr><td>REAP</td><td>expert</td><td>79.29</td><td>66.30</td><td>79.74</td><td>27.92</td><td>69.37</td><td>65.41</td></tr><tr><td>MAN</td><td>expert</td><td>81.89</td><td>64.82</td><td>69.96</td><td>49.21</td><td>74.26</td><td>68.09</td></tr><tr><td>MSAN</td><td>expert</td><td>79.94</td><td>67.24</td><td>83.99</td><td>48.08</td><td>75.59</td><td>67.05</td></tr><tr><td>UC-RAMP</td><td>channel</td><td>82.81</td><td>75.43</td><td>84.96</td><td>67.88</td><td>67.18</td><td>74.23</td></tr></table>

Table 1: Ten-task mean accuracy (%) for UC-RAMP and pruning baselines. Columns specify nominal pruning ratios, and Unit identifies each method’s pruning granularity. UC-RAMP’s ratios apply to routed-expert intermediate channels. Bold denotes the highest pruned-model mean in each setting.

<table><tr><td>Model</td><td>CAPE</td><td>RAMP</td><td>Mean ACC (%)</td></tr><tr><td>Qwen3</td><td>×</td><td>×</td><td>73.11</td></tr><tr><td></td><td>√</td><td>×</td><td>75.19</td></tr><tr><td></td><td>√</td><td>√</td><td>75.43</td></tr><tr><td>Gemma-4</td><td>×</td><td>X</td><td>65.51</td></tr><tr><td></td><td>√</td><td>×</td><td>67.60</td></tr><tr><td></td><td>√</td><td>√</td><td>67.88</td></tr></table>

Table 2: Component ablation at 50% routed-expert channel pruning. $\times / \times$ denotes Multi-domain Top-K, retaining UC-RAMP’s multi-domain scoring and calibration; it is distinct from CAMERA. The CAPE column toggles expert multipliers and capacity redistribution, while RAMP toggles fixed-width swaps. Bold marks the highest mean for each model.

## 4.3 Ablation Studies

## 4.3.1 Component Ablation

Table 2 isolates expert-capacity redistribution and fixed-width channel refinement while retaining multi-domain scoring. At 50% pruning, adding CAPE’s expert multipliers and capacity redistribution raises mean accuracy from 73.11% to 75.19% on Qwen3 and from 65.51% to 67.60% on Gemma, gains of 2.08 and 2.09 percentage points. RAMP then improves the means to 75.43% and 67.88% with these expert widths held fixed. Appendix D reports the other four CAPE–UC-RAMP comparisons.

These results distinguish the contributions of the two structural decisions. The multi-domain Top-K baseline already provides a channel ranking, while CAPE additionally changes how the retained-channel budget is distributed across experts. RAMP operates within these allocated widths and provides a smaller incremental gain through interaction-aware set refinement. The larger improvement from capacity redistribution in both settings suggests that expert width allocation is an important factor in constructing effective micro-expert masks.

## 4.3.2 Hyperparameter Sensitivity

We examine the three hyperparameters on Qwen3- 30B-A3B-Instruct-2507 at 50% pruning (Figure 3). The α and λ sweeps evaluate CAPE only, while the β sweep adds RAMP to the CAPE checkpoint. The main experiments use the fixed defaults $\alpha = 0 . 9$ $\lambda = 0 . 9 , \mathrm { a n d } \ \beta = 0 . 2$

Balancing activation energy and peak response yields similar mean accuracy across $\alpha \in [ 0 . 6 , 0 . 9 ]$ ranging from 75.178% to 75.268%. The tested endpoints, $\alpha = 0 . 3$ and $\alpha = 1 . 0 \ :$ , give 60.891% and 40.627%, respectively, supporting a blend of the two response statistics.

The fusion weight λ favors a balance between general and specialized evidence. The highest recorded CAPE mean is 77.119% at $\lambda = 0 . 8 .$ , compared with 75.187% at the fixed default $\lambda = 0 . 9$ Both tested endpoints (λ = 0.4 and 1.0) score below 65%.

For fixed-width refinement, moderate interaction strength improves mean accuracy from 75.187% at $\beta = 0$ to 75.432% at $\beta \ : = \ : 0 . 2$ . Increasing the weight to $\beta ~ = ~ 0 . 6$ yields 74.11%. This trend supports incorporating channel interactions with a moderate weight in this setting.

## 4.4 Inference Efficiency

We compare unpruned and 25%-pruned Qwen3 in BF16 using vLLM 0.27.1 (TP=1) on a single NVIDIA A100-SXM4 (80 GB). UC-RAMP packs expert weights at their retained widths $k _ { e } ,$ preserving the logical budget $\textstyle \sum _ { e } k _ { e } = K _ { \ell }$ . Its custom Triton kernels use masked channel tiles (16 for gate/up and 32 for down reduction), while temporary activations are allocated by the maximum expert width in each layer. Table 3 includes these execution overheads.

![](images/06ba31303fa41de727b02dfc14dfca283938c5bc24069f8cd3d11b48cee4278e.jpg)

![](images/3b9de483efea48c46b26544160c53ff41157483eb5b51be4accee692e80f9798.jpg)

![](images/d1f0faec37775780ac4f13ac2948284948299f29562732bd7872e8cf08908d81.jpg)  
Figure 3: Hyperparameter sensitivity on Qwen3-30B-A3B-Instruct-2507 at 50% routed-expert channel pruning. Panels (a) α and (b) λ report CAPE-only results. Panel (c) β reports RAMP initialized from CAPE, with expert widths fixed. Diamonds mark the fixed defaults (0.9, 0.9, 0.2). The panels use separate y-axis ranges.

Requests generate 32 tokens with temperature 0 and EOS ignored. Time to first token (TTFT) and time per output token (TPOT) are measured through local HTTP/SSE streaming using actual output token IDs. We report medians over two rounds, each with one warmup and three timed requests per configuration, reversing model startup order between rounds. TTFT decreases by 19.7– 23.9% and TPOT by 20.2–22.3% relative to the dense reference. At the same 25% pruning ratio, UC-RAMP retains 82.81% ten-task mean accuracy, compared with 86.19% for the unpruned model (Table 1). These measurements demonstrate lower inference latency under the evaluated deployment configurations, including custom-kernel and workspace overheads. Appendix A gives the deployment details.

## 5 Conclusion

Micro-expert pruning involves two structural decisions: allocating retained capacity across experts and selecting channels within that capacity. UC-RAMP separates these decisions through crossdomain capacity redistribution in CAPE and fixedwidth channel refinement in RAMP. Across three MoE models and two pruning ratios, UC-RAMP achieves the highest mean accuracy among evaluated pruning methods in five of six settings, including all three models at 50% pruning. Deployment measurements further demonstrate reduced inference latency at 25% pruning. These results highlight the value of treating expert capacity and channel composition as distinct decisions in posttraining MoE compression.

<table><tr><td>TTFT (ms)</td><td colspan="2"></td><td colspan="2">TPOT (ms/token)</td></tr><tr><td>Input Batch</td><td></td><td>Dense UC-RAMP</td><td>Dense</td><td>UC-RAMP</td></tr><tr><td>256</td><td>1</td><td>106.61</td><td>81.71 47.45</td><td>36.94</td></tr><tr><td>512</td><td>1</td><td>107.92</td><td>82.14 47.54</td><td>36.92</td></tr><tr><td>1024</td><td>1</td><td>103.84</td><td>83.42 46.68</td><td>36.88</td></tr><tr><td>1024</td><td></td><td>2131.38</td><td>102.79 48.48</td><td>38.70</td></tr><tr><td>1024</td><td></td><td>4159.12</td><td>126.92</td><td>48.44 38.57</td></tr></table>

Table 3: Inference latency on Qwen3-30B-A3B-Instruct-2507 at 25% routed-expert channel pruning, using one NVIDIA A100-SXM4 (80 GB). Both models use BF16. Input is prompt length in tokens; Batch is batch size.

## Limitations

The experiments cover three MoE checkpoints and two pruning rates defined only over routed-expert intermediate channels. The calibration evidence spans four selected domains, so the observed allocation behavior may depend on which capabilities those domains represent. RAMP optimizes a local output-reconstruction surrogate within each expert; its guarantee concerns that surrogate, and its mixed task-level effects show that it should not be read as a general monotonic improvement to downstream accuracy.

## References

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, Minneapolis, Minnesota. Association for Computational Linguistics.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. SliceGPT: Compress large language models by deleting rows and columns. In International Conference on Learning Representations, pages 11682–11701.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Sikai Bai, Haoxi Li, Jie Zhang, Zicong Hong, and Song Guo. 2025. DiEP: Adaptive mixture-of-experts compression through differentiable expert pruning. In Advances in Neural Information Processing Systems, volume 38, pages 56090–56115. Curran Associates, Inc.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about physical commonsense in natural language. Proceedings ofthe AAAI Conference on Artificial Intelligence, 34(05):7432–7439.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Yifu Ding, Jiacheng Wang, Ge Yang, Yongcheng Jing, Jinyang Guo, Xianglong Liu, and Dacheng Tao. 2026. Attribution-guided and coverage-maximized pruning for structural MoE compression. arXiv preprint arXiv:2606.18304.

William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39.

Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive language models can be accurately pruned in one-shot. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 10323–10337. PMLR.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

Linghao Kong, Inimai Subramanian, Micah Adler, Dan Alistarh, Dan Gutfreund, and Nir Shavit. 2026. The sparsity whisperer. arXiv preprint arXiv:2608.06630.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale reading comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785– 794, Copenhagen, Denmark. Association for Computational Linguistics.

Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, and Vithursan Thangarasa. 2026. REAP the experts: Why pruning prevails for one-shot MoE compression. In International Conference on Learning Representations, pages 146883–146912.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, pages 39578– 39601.

Zongfang Liu, Guangyi Chen, Shengkun Tang, Yifan Shen, Huan Wang, and Xin Yuan. 2026a. AIMER: Calibration-free task-agnostic MoE expert pruning. arXiv preprint arXiv:2603.18492.

Zongfang Liu, Jinghui Zhang, Zijian Ma, Guangyi Chen, and Xin Yuan. 2026b. How to score experts for one-shot MoE expert pruning: A unified formulation and selection principle. arXiv preprint arXiv:2606.15716.

Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. 2024. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6159–6172, Bangkok, Thailand. Association for Computational Linguistics.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. LLM-Pruner: On the structural pruning of large language models. In Advances in Neural Information Processing Systems, volume 36, pages 21702–21720.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial winograd schema challenge at scale. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8732–8740.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations.

Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024. A simple and effective pruning approach for large language models. In International Conference on Learning Representations, pages 4942–4964.

Yanyue Xie, Zhi Zhang, Ding Zhou, Cong Xie, Ziang Song, Xin Liu, Yanzhi Wang, Xue Lin, and An Xu. 2024. MoE-Pruner: Pruning mixture-of-experts large language model using the hints from its router. arXiv preprint arXiv:2410.12013.

Yuzhuang Xu, Xu Han, Yuanchi Zhang, Yixuan Wang, Yijun Liu, Shiyu Ji, Qingfu Zhu, and Wanxiang Che. 2026. CAMERA: Multi-matrix joint compression for MoE models via micro-expert redundancy analysis. Proceedings of the AAAI Conference on Artificial Intelligence, 40(32):27395–27404.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

## A Implementation and protocol details

The general-language calibration set contains 128 WikiText-2 training windows. The generalreasoning set contains 64 GSM8K and 64 MathQA training windows. The harder-mathematics set contains 128 sampled competition-MATH training windows, and the code set contains 64 MBPP training windows. Each window has length 2048. Each domain maintains separate channel statistics. The general domain receives weight 0.9 and the other three domains share the remaining 0.1 equally.

For layer $\ell ,$ the reference ranking first assigns $k _ { \ell , e } ^ { 0 }$ retained channels to expert e. With C intermediate channels per expert and pruning rate $r _ { \ell } ,$ the trust-region radius is

$$
\Delta _ { \ell } = \operatorname* { m a x } ( 1 , \lfloor \mathrm { r o u n d } [ C ( 1 - r _ { \ell } ) ] / 8 \rfloor ) .\tag{14}
$$

The allocation bounds are $k _ { \ell , e } ^ { \mathrm { m i n } } = \operatorname* { m a x } ( 0 , k _ { \ell , e } ^ { 0 }$ $\Delta _ { \ell } )$ and $k _ { \ell , e } ^ { \operatorname* { m a x } } = \operatorname* { m i n } ( C , k _ { \ell , e } ^ { 0 } + \Delta _ { \ell } )$ . CAPE processes layers from shallow to deep while retaining the masks already chosen for earlier layers. Once all CAPE masks are fixed, RAMP gathers routed trajectories from that checkpoint, computes withinexpert Gram matrices, and considers at most one accepted exchange per layer-expert. Final masks are checked against every layer’s exact budget.

Inference deployment. Table 3 compares the unpruned Qwen3-30B-A3B-Instruct-2507 checkpoint with UC-RAMP at 25% routed-expert intermediate-channel pruning. Both models use BF16 without quantization on one NVIDIA A100-SXM4 (80 GB), with tensor parallelism set to 1 in vLLM 0.27.1. Both use FlashAttention 2 and enforce\_eager, with torch.compile and CUDA Graph disabled. The maximum context length is 2048 tokens, and gpu\_memory\_utilization is 0.90.

The dense model uses vLLM’s built-in TritonExperts with its default MoE configuration. The pruned model uses custom Triton kernels in two stages: gate/up projections with SwiGLU, followed by down projection and routing-weight aggregation. Expert weights are packed contiguously at their exact retained widths and located through offsets. The gate/up kernels use channel tiles of 16; the down kernel uses reduction tiles of 32 channels and output tiles of 64 hidden dimensions. Partial tiles are masked. Temporary activations and the gate/up launch grid are sized by the largest retained expert width in each layer. Their workspace and launch overheads are included in the reported latency.

Latency measurement. The five inputlength/batch-size pairs are 256/1, 512/1, 1024/1, 1024/2, and 1024/4. Each request generates exactly 32 tokens with temperature 0 and EOS ignored. A local HTTP/SSE client measures TTFT and TPOT using actual emitted token IDs. Each configuration is tested in two rounds, with one warmup request followed by three timed requests per model in each round. The second round reverses model startup order; the reported values are medians pooled across both rounds.

## B Properties of the two stages

Exact solution of the capacity proxy. For a fixed width $k _ { e }$ and positive $\rho _ { e } ,$ a channel set omitting a higher-scoring channel in favor of a lower-scoring one cannot maximize Eq. (7). Hence an optimum exists in which each expert retains an anchor-score prefix. The lower-bound prefixes are mandatory. After retaining them, each eligible additional position has marginal value $\rho _ { e } A _ { e , ( j ) }$ . Choosing the globally largest remaining marginals until the exact layer budget is reached maximizes the objective; upper-bound positions are excluded from the candidate pool. A zero multiplier gives equal marginal values for that expert, resolved by a fixed tie-break.

RAMP surrogate and exchanges. For a domain and expert, $Z ^ { \top } Z / N$ and $\widetilde { W } ^ { \mp } \widetilde { W }$ are positive semidefinite. Their Hadamard product $G ^ { ( d , e ) }$ is positive semidefinite by the Schur product theorem, as is the nonnegative weighted sum across domains. Normalizing the Gram matrix on positive-diagonal channels preserves positive semidefiniteness. Zerodiagonal channels have zero off-diagonal entries and receive unit diagonal in R, so $R \succeq 0$ . For $0 \leq \beta \leq 1 , ( 1 - \beta ) I + \beta R \succeq 0$ and therefore $H = D _ { A } [ ( 1 - \beta ) I + \beta R ] D _ { A } \succeq 0 .$

For $p ^ { \prime } = p + { \bf e } _ { i } - { \bf e } _ { j }$ , expanding the quadratic form gives the exact objective change

$$
\begin{array} { c } { \Delta _ { i \to j } = 2 ( H p ) _ { i } - 2 ( H p ) _ { j } } \\ { + H _ { i i } + H _ { j j } - 2 H _ { i j } . } \end{array}\tag{15}
$$

Accepting only $\Delta _ { i  j } < - \tau$ strictly lowers the surrogate. The simultaneous one-out and one-in update leaves the expert width unchanged. These statements concern the defined integer and reconstruction objectives, not model accuracy.

## C Full task results

Tables 4–9 report the ten task accuracies and their unweighted mean for each model and pruning rate. WikiText-2 perplexity is used only for the Qwen3 example in Figure 1 and the corresponding discussion in the main text.
<table><tr><td>Method</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>GSM8K</td><td>Hella</td><td>Wino</td><td>RACE</td><td>GPQA-D</td><td>MATH</td><td>HE</td><td>Mean</td></tr><tr><td>Dense</td><td>91.29</td><td>94.40</td><td>93.00</td><td>96.29</td><td>87.12</td><td>75.30</td><td>91.61</td><td>56.57</td><td>80.60</td><td>95.73</td><td>86.19</td></tr><tr><td>CAMERA</td><td>90.75</td><td>94.02</td><td>93.17</td><td>93.48</td><td>87.39</td><td>73.64</td><td>91.26</td><td>34.34</td><td>28.20</td><td>0.00</td><td>68.63</td></tr><tr><td>MoE-Slimming</td><td>90.64</td><td>93.43</td><td>91.89</td><td>96.06</td><td>86.91</td><td>72.38</td><td>89.85</td><td>35.35</td><td>79.00</td><td>48.78</td><td>78.43</td></tr><tr><td>AIMER</td><td>85.42</td><td>92.85</td><td>89.93</td><td>94.84</td><td>84.05</td><td>72.30</td><td>89.57</td><td>30.81</td><td>77.00</td><td>91.46</td><td>80.82</td></tr><tr><td>REAP</td><td>89.01</td><td>93.90</td><td>92.15</td><td>96.06</td><td>86.27</td><td>71.74</td><td>89.05</td><td>43.43</td><td>78.20</td><td>53.05</td><td>79.29</td></tr><tr><td>MAN</td><td>86.78</td><td>94.02</td><td>92.66</td><td>95.07</td><td>81.74</td><td>68.51</td><td>88.38</td><td>39.39</td><td>78.40</td><td>93.90</td><td>81.89</td></tr><tr><td>MSAN</td><td>87.65</td><td>93.52</td><td>91.47</td><td>95.22</td><td>81.65</td><td>68.03</td><td>88.26</td><td>40.40</td><td>79.40</td><td>73.78</td><td>79.94</td></tr><tr><td>CAPE</td><td>90.53</td><td>94.40</td><td>92.32</td><td>96.44</td><td>86.67</td><td>74.35</td><td>90.78</td><td>32.83</td><td>80.80</td><td>89.02</td><td>82.81</td></tr><tr><td>UC-RAMP</td><td>90.70</td><td>94.40</td><td>92.49</td><td>96.59</td><td>86.76</td><td>73.80</td><td>90.90</td><td>33.84</td><td>79.00</td><td>89.63</td><td>82.81</td></tr></table>

Table 4: Task accuracy (%) for Qwen3-30B-A3B-Instruct-2507; 25%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.
<table><tr><td>Method</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>GSM8K</td><td>Hella</td><td>Wino</td><td>RACE</td><td>GPQA-D</td><td>MATH</td><td>HE</td><td>Mean</td></tr><tr><td>Dense</td><td>91.29</td><td>94.40</td><td>93.00</td><td>96.29</td><td>87.12</td><td>75.30</td><td>91.61</td><td>56.57</td><td>80.60</td><td>95.73</td><td>86.19</td></tr><tr><td>CAMERA</td><td>79.76</td><td>91.62</td><td>85.92</td><td>74.00</td><td>71.65</td><td>57.38</td><td>86.79</td><td>27.27</td><td>10.20</td><td>0.00</td><td>58.46</td></tr><tr><td>MoE-Slimming</td><td>84.93</td><td>86.74</td><td>80.80</td><td>96.13</td><td>79.07</td><td>66.46</td><td>83.77</td><td>15.66</td><td>73.40</td><td>3.05</td><td>67.00</td></tr><tr><td>AIMER</td><td>74.92</td><td>81.48</td><td>75.34</td><td>71.49</td><td>72.50</td><td>62.90</td><td>86.24</td><td>14.65</td><td>49.00</td><td>56.10</td><td>64.46</td></tr><tr><td>REAP</td><td>79.87</td><td>91.46</td><td>87.12</td><td>93.63</td><td>72.75</td><td>59.67</td><td>83.82</td><td>21.72</td><td>64.40</td><td>8.54</td><td>66.30</td></tr><tr><td>MAN</td><td>65.13</td><td>85.65</td><td>79.78</td><td>90.60</td><td>46.44</td><td>55.56</td><td>72.50</td><td>34.34</td><td>81.00</td><td>37.20</td><td>64.82</td></tr><tr><td>MSAN</td><td>68.23</td><td>88.01</td><td>81.48</td><td>92.19</td><td>49.56</td><td>55.25</td><td>66.56</td><td>34.85</td><td>81.40</td><td>54.88</td><td>67.24</td></tr><tr><td>CAPE</td><td>82.43</td><td>92.13</td><td>88.48</td><td>95.45</td><td>81.71</td><td>68.35</td><td>89.12</td><td>21.72</td><td>77.00</td><td>55.49</td><td>75.19</td></tr><tr><td>UC-RAMP</td><td>82.43</td><td>92.09</td><td>88.82</td><td>95.53</td><td>81.70</td><td>68.51</td><td>88.91</td><td>19.19</td><td>78.00</td><td>59.15</td><td>75.43</td></tr><tr><td>Dense</td><td>93.42</td><td>94.95</td><td>95.39</td><td>96.66</td><td>79.43</td><td>85.40</td><td>92.56</td><td>72.73</td><td>94.00</td><td>96.95</td><td>90.15</td></tr><tr><td>CAMERA</td><td>90.70</td><td>94.91</td><td>95.31</td><td>94.24</td><td>78.38</td><td>81.77</td><td>91.87</td><td>43.43</td><td>20.60</td><td>0.00</td><td>69.12</td></tr><tr><td>MoE-Slimming</td><td>87.32</td><td>91.25</td><td>89.76</td><td>94.54</td><td>73.32</td><td>73.95</td><td>88.36</td><td>39.39</td><td>75.20</td><td>85.37</td><td>79.85</td></tr><tr><td>AIMER</td><td>83.24</td><td>84.13</td><td>82.59</td><td>62.47</td><td>73.64</td><td>76.95</td><td>91.57</td><td>22.73</td><td>36.20</td><td>9.15</td><td>62.27</td></tr><tr><td>REAP</td><td>88.47</td><td>94.23</td><td>92.06</td><td>95.91</td><td>81.51</td><td>71.43</td><td>90.78</td><td>60.10</td><td>89.40</td><td>33.54</td><td>79.74</td></tr><tr><td>MAN</td><td>65.02</td><td>87.46</td><td>86.77</td><td>43.21</td><td>55.84</td><td>55.88</td><td>75.36</td><td>54.04</td><td>85.20</td><td>90.85</td><td>69.96</td></tr><tr><td>MSAN</td><td>87.60</td><td>93.64</td><td>91.72</td><td>95.91</td><td>76.08</td><td>68.43</td><td>90.88</td><td>57.07</td><td>81.60</td><td>96.95</td><td>83.99</td></tr><tr><td>CAPE</td><td>84.55</td><td>94.57</td><td>94.37</td><td>96.59</td><td>73.17</td><td>78.06</td><td>91.51</td><td>56.57</td><td>91.80</td><td>87.80</td><td>84.90</td></tr><tr><td>UC-RAMP</td><td>84.93</td><td>94.32</td><td>94.45</td><td>96.59</td><td>74.18</td><td>77.58</td><td>91.41</td><td>56.57</td><td>91.20</td><td>88.41</td><td>84.96</td></tr></table>

Table 5: Task accuracy (%) for Qwen3-30B-A3B-Instruct-2507; 50%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.

Table 6: Task accuracy (%) for Gemma4-26B-A4B-it; 25%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.
<table><tr><td>Method</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>GSM8K</td><td>Hella</td><td>Wino</td><td>RACE</td><td>GPQA-D</td><td>MATH</td><td>HE</td><td>Mean</td></tr><tr><td>Dense</td><td>93.42</td><td>94.95</td><td>95.39</td><td>96.66</td><td>79.43</td><td>85.40</td><td>92.56</td><td>72.73</td><td>94.00</td><td>96.95</td><td>90.15</td></tr><tr><td>CAMERA</td><td>69.80</td><td>89.73</td><td>84.56</td><td>63.38</td><td>70.30</td><td>61.17</td><td>84.48</td><td>33.33</td><td>4.40</td><td>0.00</td><td>56.12</td></tr><tr><td>MoE-Slimming</td><td>77.86</td><td>75.72</td><td>67.66</td><td>90.60</td><td>64.60</td><td>61.01</td><td>82.47</td><td>31.82</td><td>58.20</td><td>29.27</td><td>63.92</td></tr><tr><td>AIMER</td><td>64.25</td><td>57.66</td><td>52.22</td><td>15.31</td><td>56.52</td><td>62.04</td><td>86.71</td><td>19.19</td><td>4.40</td><td>0.00</td><td>41.83</td></tr><tr><td>REAP</td><td>13.11</td><td>25.72</td><td>23.89</td><td>86.13</td><td>14.63</td><td>11.13</td><td>74.34</td><td>17.68</td><td>12.60</td><td>0.00</td><td>27.92</td></tr><tr><td>MAN</td><td>55.17</td><td>70.41</td><td>61.69</td><td>74.68</td><td>33.26</td><td>26.20</td><td>75.74</td><td>24.75</td><td>58.00</td><td>12.20</td><td>49.21</td></tr><tr><td>MSAN</td><td>49.78</td><td>53.87</td><td>50.77</td><td>84.76</td><td>30.29</td><td>33.86</td><td>83.91</td><td>24.24</td><td>61.40</td><td>7.93</td><td>48.08</td></tr><tr><td>CAPE</td><td>73.45</td><td>89.73</td><td>85.67</td><td>94.84</td><td>62.70</td><td>63.93</td><td>86.32</td><td>31.31</td><td>73.40</td><td>14.63</td><td>67.60</td></tr><tr><td>UC-RAMP</td><td>72.69</td><td>90.53</td><td>87.20</td><td>95.45</td><td>63.50</td><td>66.54</td><td>86.81</td><td>28.28</td><td>73.80</td><td>14.02</td><td>67.88</td></tr><tr><td>Dense</td><td>94.34</td><td>94.15</td><td>93.86</td><td>86.50</td><td>90.44</td><td>91.32</td><td>89.60</td><td>20.20</td><td>24.80</td><td>84.15</td><td>76.94</td></tr><tr><td>CAMERA</td><td>95.16</td><td>94.78</td><td>95.48</td><td>87.19</td><td>91.17</td><td>92.82</td><td>92.70</td><td>16.67</td><td>31.20</td><td>1.83</td><td>69.90</td></tr><tr><td>MoE-Slimming</td><td>89.39</td><td>90.03</td><td>91.64</td><td>41.93</td><td>82.24</td><td>83.66</td><td>79.43</td><td>11.62</td><td>15.60</td><td>61.59</td><td>64.71</td></tr><tr><td>AIMER</td><td>91.95</td><td>90.78</td><td>90.61</td><td>73.16</td><td>82.70</td><td>88.24</td><td>84.99</td><td>11.62</td><td>29.20</td><td>66.46</td><td>70.97</td></tr><tr><td>REAP</td><td>94.67</td><td>92.72</td><td>94.20</td><td>75.44</td><td>90.76</td><td>91.08</td><td>89.68</td><td>12.12</td><td>27.40</td><td>25.61</td><td>69.37</td></tr><tr><td>MAN</td><td>94.67</td><td>92.72</td><td>93.94</td><td>73.77</td><td>90.55</td><td>88.63</td><td>90.82</td><td>15.66</td><td>24.40</td><td>77.44</td><td>74.26</td></tr><tr><td>MSAN</td><td>94.34</td><td>93.73</td><td>94.88</td><td>75.97</td><td>91.03</td><td>90.61</td><td>91.95</td><td>18.18</td><td>27.80</td><td>77.44</td><td>75.59</td></tr><tr><td>CAPE</td><td>92.00</td><td>92.38</td><td>90.78</td><td>77.79</td><td>80.49</td><td>86.50</td><td>83.68</td><td>3.03</td><td>28.80</td><td>72.56</td><td>70.80</td></tr><tr><td>UC-RAMP</td><td>91.13</td><td>91.50</td><td>90.78</td><td>77.18</td><td>82.20</td><td>85.64</td><td>83.04</td><td>5.56</td><td>29.40</td><td>35.37</td><td>67.18</td></tr></table>

Table 7: Task accuracy (%) for Gemma4-26B-A4B-it; 50%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.

Table 8: Task accuracy (%) for Qwen3.6-35B-A3B; 25%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.
<table><tr><td>Method</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>GSM8K</td><td>Hella</td><td>Wino</td><td>RACE</td><td>GPQA-D</td><td>MATH</td><td>HE</td><td>Mean</td></tr><tr><td>Dense</td><td>94.34</td><td>94.15</td><td>93.86</td><td>86.50</td><td>90.44</td><td>91.32</td><td>89.60</td><td>20.20</td><td>24.80</td><td>84.15</td><td>76.94</td></tr><tr><td>CAMERA</td><td>85.26</td><td>88.85</td><td>89.93</td><td>33.74</td><td>59.62</td><td>81.93</td><td>89.58</td><td>20.20</td><td>12.20</td><td>0.00</td><td>56.13</td></tr><tr><td>MoE-Slimming</td><td>71.38</td><td>86.95</td><td>82.17</td><td>20.24</td><td>66.71</td><td>67.01</td><td>73.19</td><td>6.57</td><td>11.00</td><td>10.98</td><td>49.62</td></tr><tr><td>AIMER</td><td>65.18</td><td>67.30</td><td>63.40</td><td>13.80</td><td>49.22</td><td>79.79</td><td>57.70</td><td>25.76</td><td>43.00</td><td>23.78</td><td>48.89</td></tr><tr><td>REAP</td><td>90.21</td><td>88.05</td><td>86.69</td><td>89.84</td><td>78.71</td><td>88.71</td><td>86.90</td><td>18.18</td><td>25.60</td><td>1.22</td><td>65.41</td></tr><tr><td>MAN</td><td>90.81</td><td>92.21</td><td>91.98</td><td>90.30</td><td>80.38</td><td>87.37</td><td>88.55</td><td>8.59</td><td>20.20</td><td>30.49</td><td>68.09</td></tr><tr><td>MSAN</td><td>89.45</td><td>92.47</td><td>91.89</td><td>75.44</td><td>81.67</td><td>87.21</td><td>90.01</td><td>12.12</td><td>20.40</td><td>29.88</td><td>67.05</td></tr><tr><td>CAPE</td><td>76.44</td><td>86.99</td><td>88.05</td><td>93.10</td><td>75.91</td><td>85.56</td><td>89.18</td><td>8.08</td><td>52.20</td><td>71.34</td><td>72.69</td></tr><tr><td>UC-RAMP</td><td>78.78</td><td>88.09</td><td>88.91</td><td>93.40</td><td>76.89</td><td>85.40</td><td>89.74</td><td>12.63</td><td>52.80</td><td>75.61</td><td>74.23</td></tr></table>

Table 9: Task accuracy (%) for Qwen3.6-35B-A3B; 50%. MATH denotes MATH500 and HE denotes HumanEval pass@1. The 25% or 50% rate applies to routed-expert intermediate channels.

## D Additional component comparisons

Table 10 reports the four remaining model-budget settings, complementing the two 50% component ablations in Table 2. Every CAPE–UC-RAMP pair uses the same expert widths.

<table><tr><td>Model / ratio</td><td>CAPE</td><td>UC-RAMP</td><td>∆ RAMP</td></tr><tr><td>Qwen3, 25%</td><td>82.81</td><td>82.81</td><td>0.00</td></tr><tr><td>Gemma-4, 25%</td><td>84.90</td><td>84.96</td><td>+0.06</td></tr><tr><td>Qwen3.6, 25%</td><td>70.80</td><td>67.18</td><td>-3.62</td></tr><tr><td>Qwen3.6, 50%</td><td>72.69</td><td>74.23</td><td>+1.54</td></tr></table>

Table 10: Additional fixed-width refinement comparisons in ten-task mean accuracy (%). ∆ RAMP is the difference between the displayed UC-RAMP and CAPE means.

On Qwen3.6 at 50%, refinement increases GPQA-Diamond accuracy from 8.08% to 12.63% and HumanEval pass@1 from 71.34% to 75.61%, while MATH500 changes from 52.2% to 52.8%. At 25%, HumanEval changes from 72.56% to 35.37% and MATH500 from 28.8% to 29.4%. Appendix C provides the complete task scores for these configurations.