# Anchoring What Matters: A Dual-Level Learning Framework for Visually-Grounded Multimodal Reasoning

Xinxin Song<sup>1</sup>\*, Siyuan Li<sup>1</sup>\*, Tingxiong Xiao<sup>1</sup>, Jinli Suo<sup>1,2†</sup>

<sup>1</sup>Department of Automation, Tsinghua University <sup>2</sup>Institute for Brain and Cognitive Science, Tsinghua University jlsuo@tsinghua.edu.cn

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has significantly improved the reasoning capabilities of large vision-language models (LVLMs). However, standard on-policy RLVR algorithms face a critical optimization bottleneck in preserving and reinforcing visually grounded reasoning behaviors: valuable visually-grounded reasoning trajectories are discarded after a single update, while uniform token advantage allocation prevents the model from reinforcing critical perception or reasoning steps. To bridge this gap, we propose PIVOT, a dual-level learning framework that anchors policy optimization around informative visual reasoning signals. Specifically, PIVOT introduces a self-calibrated experience replay mechanism, which selectively collects and replays visually-grounded historical experiences as stable reference anchors for policy optimization. Building upon this, we further design a vision-guided advantage allocation mechanism to allocate additional vision-aware advantages to tokens based on their local visual support and impact on downstream reasoning. Extensive experiments across diverse benchmarks demonstrate that PIVOT achieves highly competitive performance in enhancing the multimodal reasoning capabilities of LVLMs.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become a prevailing post-training paradigm for enhancing the reasoning capabilities of large language models (Zhang et al., 2025c). Given its success in the text domain, recent studies have extended RLVR to large vision-language models (LVLMs) and achieved promising results in visual reasoning tasks (Liu et al., 2025b). However, directly applying standard RLVR algorithms to LVLMs exposes a critical optimization bottleneck: sparse yet informative learning signals are neither sufficiently preserved nor effectively exploited, leading to inefficient optimization for visually grounded reasoning. Specifically, this inefficiency manifests at two interconnected levels.

At the trajectory level, standard on-policy RL algorithms typically discard the generated trajectories<sup>1</sup> after each update phase. For LVLMs, obtaining high-quality visually grounded trajectories is non-trivial due to the expanded multimodal search space. These trajectories explicitly demonstrate how to ground visual evidence into logical steps, containing richer information than a sparse binary reward. Discarding them indiscriminately results in a waste of optimization signals. At the token level, the issue of uniform credit assignment further dilutes the learning signals. Outcome-based RLVR algorithms typically assign the same scalar advantage to all tokens within a trajectory. However, the visual reasoning chain is highly heterogeneous: certain tokens act as pivotal points for visual perception or logical branching, whereas others are mere linguistic fillers. Assigning uniform optimization strength to all tokens prevents the model from anchoring on the critical perception or reasoning steps, causing informative token-level signals to be diluted across the entire response.

To address the above limitations, we propose PIVOT, a dual-level learning framework that Preserves and reinforces sparse but Informative learning signals for Visually-grOunded reasoning at both the trajectory and Token levels. Specifically, at the trajectory level, we introduce a selfcalibrated experience replay mechanism, which builds a dynamic experience pool to collect and selectively replay informative prompts alongside their high-quality, visually-grounded historical trajectories. These trajectories act as retrospective anchors rather than direct training objectives, gently constraining the policy without stifling exploration. Beyond trajectory-level anchoring, we further introduce a vision-guided advantage allocation strategy. This mechanism obtains fine-grained visual learning signals by calculating the counterfactual visual support of sampled tokens and estimating their impact on future reasoning. These signals are then used for fine-grained advantage allocation to concentrate the optimization strength on more valuable perception and reasoning tokens. The main contributions of our method are summarized as follows:

• We identify a key optimization bottleneck in multimodal RLVR: valuable visual reasoning signals are diluted across both trajectory sampling and token-level optimization, making it difficult for the policy to consistently reinforce visually grounded reasoning behaviors.

• We propose PIVOT, a unified dual-level framework that preserves high-value visual reasoning experiences as retrospective anchors and reinforces optimization on visually grounded tokens via fine-grained advantage modulation.

• Extensive experiments on various data and model scales demonstrate that PIVOT consistently improves multimodal reasoning performance over baselines.

## 2 Related Works

RLVR for Multimodal Reasoning. Given the success of RLVR in the text domain, recent works have begun to explore its application to LVLMs. From data perspective, methods like Vision-R1 (Huang et al., 2026b) and MM-EUREKA (Meng et al., 2025) construct highquality CoT datasets, while NoisyRollout (Liu et al., 2025a) injects perceptual diversity through data augmentation. For reward design, Perception-R1 (Yu et al., 2025a) designs specialized reward mechanisms for different visual tasks, and Vision-SR1 (Li et al., 2026b) decomposes visual perception and language reasoning to formulate a selfreward mechanism. For optimization objectives, PAPO (Wang et al., 2026b) introduces an additional KL loss to improve perceptual capabilities. Unlike prior work that mainly improves multimodal RL through stronger supervision signals, we revisit the optimization process and study how high-value visual reasoning signals can be more effectively preserved and exploited during policy learning.

Experience-based RL. Experience replay (Lin, 1992) has been extensively studied in classical RL tasks (Wang et al., 2024). Recent studies have extended it to LLMs. For instance, RLEP (Zhang et al., 2025a) introduces a two-stage framework of collection followed by replay; ReMix (Liang et al., 2026) formulates a stable optimization objective for off-policy training; and ExGRPO (Zhan et al., 2026) prioritizes the replay of high-value trajectories based on systematic metric analysis. For LVLMs, EFRame (Wang et al., 2025a) replays successful trajectories for hard prompts, while VL-Rethinker (Wang et al., 2025b) mitigates the vanishing advantage problem via selective sample replay. In this work, we emphasize visual perception quality during experience collection while treating historical experience as a reference anchor for policy updates rather than a direct optimization objective, which yields better results. CalibRL (Huang et al., 2026c) adopts a similar optimization objective to ours, but it relies on external expert demonstrations instead of self-generated experience.

Token Optimization in RLVR. Recent works have revealed the heterogeneity and sparsity of the token optimization signal in RLVR updates (Wang et al., 2025c). In LVLMs, prior works assess tokenlevel visual dependence using counterfactual output divergence (Ye et al., 2026; Huang et al., 2026a), hidden state similarities (Wang et al., 2026a), or visual attention scores (Jiao et al., 2026; Luo et al., 2026) to reinforce tokens with strong perceptual grounding. Methods like PEPO (Li et al., 2026a) and ToR (Lu et al., 2026) further integrate this with high-entropy reasoning tokens. Building upon counterfactual support for local visual dependence, we further formulate an entropy-gated future visual support to capture downstream impact, enabling fine-grained advantage allocation.

## 3 Preliminaries

## 3.1 Multimodal RLVR

Consider a multimodal reasoning problem where each input x consists of a textual query q and an image I, denoted by $x = \left( q , I \right)$ . An LVLM parameterized by π<sub>θ</sub> generates a textual response sequence $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } )$ autoregressively according to

$$
\pi _ { \boldsymbol { \theta } } ( \mathbf { y } \mid x ) = \prod _ { t = 1 } ^ { T } \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid q , I , y _ { < t } ) .\tag{1}
$$

For reasoning tasks, the generated sequence y typically consists of an intermediate multi-step reasoning chain followed by the final answer. In the RLVR framework, a rule-based verifier assigns a binary reward $R \ \in \ \{ 0 , 1 \}$ to each response based solely on whether its final extracted answer matches the ground truth a.

DAPO (Yu et al., 2025b) is a widely used RLVR algorithm for reasoning tasks. It samples a group of G candidate responses $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { G }$ from old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ and computes the advantage $\hat { A } _ { i }$ for the i-th candidate response ${ \bf y } _ { i }$ by

$$
\hat { A } _ { i } = \frac { R _ { i } - \operatorname* { m e a n } ( \{ R _ { k } \} _ { k = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { k } \} _ { k = 1 } ^ { G } ) } .\tag{2}
$$

The policy parameters $\theta$ are then updated by minimizing the following clipping objective:

$$
\begin{array} { c } { \displaystyle \mathcal { L } ( \theta ) = - \mathbb { E } \left[ \frac { 1 } { \sum _ { i = 1 } ^ { G } | \mathbf { y } _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | \mathbf { y } _ { i } | } \operatorname* { m i n } \Big ( \rho _ { i , t } ( \theta ) \hat { A } _ { i , t } , } \\ { \displaystyle \mathrm { c l i p } \big ( \rho _ { i , t } ( \theta ) , 1 - \epsilon _ { l } , 1 + \epsilon _ { h } \big ) \hat { A } _ { i , t } \Big ) \right] , } \end{array}\tag{3}
$$

where $\begin{array} { r } { \rho _ { i , t } ( \theta ) ~ = ~ \frac { \pi _ { \theta } \left( y _ { i , t } | q , I , \mathbf { y } _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { i , t } | q , I , \mathbf { y } _ { i , < t } \right) } } \end{array}$ is the importance sampling ratio, $\epsilon _ { l }$ and $\epsilon _ { h }$ are the clipping hyperparameters.

## 3.2 Trajectory Quality Estimation

To evaluate the quality of generated trajectories in multimodal RLVR, we consider two complementary signals that capture different aspects of trajectory reliability.

We first consider the visual dependency of a trajectory. Following previous work (Huang et al., 2026a), we quantify it by computing the KL divergence between the predictive distribution of the policy conditioned on the original image and a corrupted version. Given the multimodal input $( q , I )$ and the generated trajectory y, we will perform counterfactual interventions (such as random patch masking (Wang et al., 2026b)) on the original image I to obtain a corrupted image ${ \widetilde { I } } ,$ then we define the visual dependency as:

$$
\begin{array} { l }  { \displaystyle V ( \mathbf { y } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \Big ( \pi _ { \theta } ( \cdot  { | \mathbf { \nabla } q _ { \star } I , \mathbf { y } _ { < t } ) } } \\ { { \displaystyle \parallel \pi _ { \theta } ( \cdot  { | \mathbf { \nabla } q _ { \star } \mathbf { y } _ { < t } ) } \Big ) . } } \end{array}\tag{4}
$$

A higher $V ( \mathbf { y } )$ indicates that the trajectory is more grounded in visual evidence rather than relying solely on language priors. Then, we consider the

stability of the reasoning process and use the average trajectory entropy as its proxy:

$$
H ( \mathbf { y } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { H } \big ( \pi _ { \theta } ( \cdot \mid q , I , \mathbf { y } _ { < t } ) \big ) ,\tag{5}
$$

where $\begin{array} { r } { \mathcal { H } ( p ) = - \sum _ { x } p ( x ) \log p ( x ) } \end{array}$ denotes the entropy of the token distribution. ExGRPO (Zhan et al., 2026) has observed that lower-entropy trajectories tend to exhibit more stable and higher-quality reasoning behaviors under RLVR. Together, $V ( \mathbf { y } )$ and $H ( \mathbf { y } )$ provide complementary views of trajectory quality: the former captures grounding in visual evidence, while the latter reflects the stability of the reasoning process. These signals are used as complementary criteria for selecting high-quality trajectories in experience replay.

## 4 Method

In this section, we introduce PIVOT, a dual-level learning framework designed to preserve and amplify sparse yet informative learning signals for visually-grounded reasoning. Our approach extends the standard RLVR paradigm with two synergistic components: (1) a self-calibrated experience replay mechanism that collects and reuses high-quality past experiences to guide the policy optimization, and (2) a vision-guided advantage allocation strategy that reallocate token advantages to prioritize visually critical reasoning steps. These components work in tandem to transform rare successes into dense and informative signals, allowing the model to more effectively internalize visually grounded reasoning patterns. Figure 1 provides a schematic overview of our framework.

## 4.1 Self-Calibrated Experience Replay

This mechanism primarily addresses the issue of high-quality trajectories being discarded during on-policy sampling, which results in wasted optimization signals. It achieves this by collecting and selectively replaying high-quality trajectories and using them as reference anchors to guide policy optimization.

Experience Pool Curation. During the rollout phase, the model generates G candidate responses $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { G }$ for a given multimodal input $x ^ { * } = ( q ^ { * } , I ^ { * } )$ , which are then evaluated by a rulebased verifier. We denote the subset of k successful responses as $y ^ { + } = \{ y _ { + } ^ { * } \}$ , and estimate the empirical success probability for this input as ${ \hat { p } } ( x ^ { * } ) = k / G$ . To efficiently manage these experiences, we store them in an experience buffer B as a structured mapping $x ^ { * } \mapsto \{ \hat { p } ( x ^ { * } ) , y ^ { + } \}$ . This design uniquely associates each input with its empirical success probability and a set of successful responses.

![](images/08f9dcf7f2fe8674d45253f3c8f84514201895f2fb295d6c2dff2bc04fd18bcd.jpg)  
Figure 1: Overview of the PIVOT framework. Standard RLVR is augmented by two synergistic modules: (1) Self-Calibrated Experience Replay module selectively collects and replays valuable prompts alongside high-quality experiences to guide policy optimization through a self-calibration loss; and (2) Vision-Guided Advantage Allocation module allocates token advantages by combining local counterfactual visual support with entropy-gated future influence to reinforce pivotal perception or reasoning steps.

Selective Experience Replay. A naive replay strategy involves randomly selecting prompts and successful trajectories from the experience pool. However, previous work (Zhan et al., 2026) has shown that this approach is suboptimal. We argue that prompts at the frontier of the model’s capabilities provide the most informative training signal. And we identify these prompts by estimating their reward variance, which has been shown to be a good proxy both theoretically and empirically (Foster et al., 2025; Jiang et al., 2025). For a multimodal prompt $x ^ { * }$ in the experience buffer, the variance can be estimated by $\hat { p } ( x ^ { * } ) ( 1 - \hat { p } ( x ^ { * } ) )$ . When constructing a training batch of size N, we sample $N _ { \mathrm { o f f } }$ prompts from the experience buffer with a probability proportional to their estimated variance:

$$
P ( { \mathrm { s a m p l e ~ p r o m p t } } x ^ { * } ) \propto { \hat { p } } ( x ^ { * } ) ( 1 - { \hat { p } } ( x ^ { * } ) )\tag{6}
$$

This variance-driven sampling naturally creates a curriculum that focuses the replay mechanism on partially solved problems.

For a selected prompt, the experience buffer may contain multiple successful historical trajectories. Instead of random selection, we will select a trajectory that is both visually grounded and stable under the current model. Specifically, based on the metrics defined in Section 3.2, we use the visual dependency score $V ( \mathbf { y } )$ to quantify how strongly a trajectory relies on visual information, and the trajectory entropy $H ( \mathbf { y } )$ to measure the stability of the reasoning process. We employ a two-stage filtering process to select the optimal one from the candidate set $y +$ . First, we rank all trajectories in $y +$ by their entropy $H ( \mathbf { y } )$ in ascending order, and retain the top $\alpha \% ( { \bf e . g . } , \alpha = 5 0 )$ with the lowest entropy to form a stable subset ${ \mathcal { V } } ^ { * }$ . Subsequently, among these remaining high-quality candidates, we select the one that maximizes the visual dependency score as our trajectory-level anchor $\mathbf { y } ^ { \mathrm { e x p } }$

$$
\mathbf { y } ^ { \mathrm { e x p } } = \underset { \mathbf { y } \in \mathcal { Y } ^ { * } } { \arg \operatorname* { m a x } } V ( \mathbf { y } )\tag{7}
$$

Experience Calibration Loss. After the experience sampling process described above, we have constructed a batch of N prompts. Among them, $N _ { \mathrm { o f f } }$ prompts are selectively sampled from the experience buffer, each paired with its optimal historical response $\mathbf { y } ^ { \mathrm { e x p } }$ . The model will perform a standard rollout phase on this batch, and each prompt will get a group of G responses, denoted as $\{ \mathbf { y } _ { i } ^ { \mathrm { c u r } } \} _ { i = 1 } ^ { G } .$

For prompts replayed from the experience buffer, we introduce an additional self-calibration loss to uncover valuable signals from historical experiences. Rather than treating the historical experiences as a direct optimization target, we utilize them as stable reference anchors for selfcalibration. We measure the policy’s relative preference between its newly generated trajectory ${ \bf y } _ { i } ^ { \mathrm { c u r } }$ and the historical experience $\mathbf { y } ^ { \mathrm { e x p } }$ by defining the log-confidence gap:

$$
\Delta \ell _ { i } = \log \pi _ { \boldsymbol { \theta } } ( \mathbf { y } _ { i } ^ { \mathrm { c u r } } \mid \boldsymbol { q } , I ) - \mathrm { s g } \left[ \log \pi _ { \boldsymbol { \theta } } ( \mathbf { y } ^ { \mathrm { e x p } } \mid \boldsymbol { q } , I ) \right]\tag{8}
$$

where $\mathrm { s g } [ \cdot ]$ denotes a stop-gradient operator, ensuring $\mathbf { y } ^ { \mathrm { e x p } }$ serves as a stable reference anchor. To dynamically guide the policy based on the correctness of its current exploration, we formulate a contrastive margin loss:

$$
{ \mathcal { L } } _ { \mathrm { e x p } } = \left| A _ { i } \right| \cdot \mathrm { s o f t p l u s } \left( - s \cdot \Delta \ell _ { i } \right)\tag{9}
$$

where $\left| A _ { i } \right|$ denotes the magnitude of the groupwise advantage for ${ \bf y } _ { i } ^ { \mathrm { c u r } }$ , and s is an indicator variable set to +1 if ${ \bf y } _ { i } ^ { \mathrm { c u r } }$ is correct and −1 if it is incorrect. In practice, we implement this objective at the token level and normalize it over all valid response tokens, providing more fine-grained calibration signals.

This objective acts as a dynamic self-calibration mechanism. For successful explorations $( s = + 1 )$ it increases the policy’s confidence in ${ \bf y } _ { i } ^ { \mathrm { c u r } }$ , while the softplus operator naturally decays gradients once $\Delta \ell _ { i } > 0$ to prevent over-optimization. Conversely, for incorrect trajectories $( s ~ = ~ - 1 )$ , the historical anchor acts as a dynamic threshold: the loss penalizes the flawed ${ \bf y } _ { i } ^ { \mathrm { c u r } }$ until its confidence drops safely below the proven baseline.

## 4.2 Vision-Guided Advantage Allocation

Although the self-calibrated experience replay mechanism preserves and leverages high-quality historical experiences by introducing an additional calibration loss, standard RLVR still broadcasts a uniform advantage to all tokens within a trajectory, resulting in the dilution of optimization signals at the token level. We address this by modulating the token advantage based on its visual utility.

Quantifying Token Visual Utility. We begin by estimating the direct counterfactual visual support for each generated token. This is achieved by comparing the model’s predictive probability under the original image I against the counterfactually corrupted image Ie (see Section 3.2). We formulate this metric as:

$$
c _ { t } = 1 - { \frac { \pi _ { \theta } ( y _ { t } \mid q , { \widetilde { I } } , \mathbf { y } _ { < t } ) } { \pi _ { \theta } ( y _ { t } \mid q , I , \mathbf { y } _ { < t } ) } } .\tag{10}
$$

This formulation captures the relative change in the model’s support for the sampled token after visual perturbation. A positive $c _ { t }$ indicates that the sampled token receives positive support from the visual evidence, suggesting it is more likely to be a perception-critical token. Conversely, $c _ { t } \approx 0$ or $c _ { t } < 0$ implies a lack of visual support, indicating that the token might be generated from language priors or visual hallucinations. Detailed justification is provided in Appendix A.

However, using only local counterfactual support as visual utilities may overlook the impact of tokens on future visual reasoning processes. To capture this impact, we define the future-discounted visual support score:

$$
F _ { t } = { \frac { \sum _ { k > t } \gamma ^ { k - t - 1 } c _ { k } } { \sum _ { k > t } \gamma ^ { k - t - 1 } } } ,\tag{11}
$$

where $\gamma \in \ [ 0 , 1 ]$ is a discount factor. In implementation, the summation is truncated to a finite window of size W, which removes noisy long-tail effects from distant tokens.

To avoid indiscriminately propagating future visual credit to all preceding tokens, we further introduce an entropy-based gate to highlight reasoningpivot tokens (Wang et al., 2025c). Let $H _ { t }$ denote the token entropy and $\bar { H }$ the average entropy over the batch. We define

$$
u _ { t } = 1 - \exp \left( - \frac { H _ { t } } { \bar { H } } \right) .\tag{12}
$$

This gate assigns larger weights to high-entropy positions, which are more likely to be reasoning decision points. The final token visual utility is

$$
U _ { t } = c _ { t } + \lambda \cdot \mathrm { D e t r e n d i n g } ( u _ { t } F _ { t } ) ,\tag{13}
$$

where λ balances local visual support and futureaware influence. Detrending(·) is a standard OLS detrending operator (Wu et al., 2007) to remove potential systematic position bias; implementation details are provided in Appendix B.

Advantage Modification. The final token visual utility score $U _ { t }$ not only measures local visual support but also captures its impact on downstream visual reasoning. We then use it to modulate the original advantage $\hat { A } _ { i , t }$ (which is uniform across tokens within the trajectory):

$$
\hat { A } _ { i , t } ^ { \prime } = \hat { A } _ { i , t } + \beta | \hat { A } _ { i , t } | U _ { t } ,\tag{14}
$$

where $\beta$ is a scaling factor. Then we apply the following sign protection operation to $\hat { A } _ { i , t } ^ { \prime } \mathrm { : }$

$$
\hat { A } _ { i , t } ^ { \mathrm { f i n a l } } = \left\{ \begin{array} { l l } { \operatorname* { m a x } ( \hat { A } _ { i , t } ^ { \prime } , 0 ) , \mathrm { i f } R _ { i } = 1 , } \\ { \quad } \\ { \operatorname* { m i n } ( \hat { A } _ { i , t } ^ { \prime } , 0 ) , \mathrm { i f } R _ { i } = 0 . } \end{array} \right.\tag{15}
$$

This operation prevents the modulation from reversing the optimization directions of positive and negative samples. All computations within this module are detached from the gradient graph, as they are only utilized to modulate the scalar token advantages. The policy gradient is then computed using the refined token advantages.

## 4.3 Final Training Objective

Our dual-level learning framework is integrated into the standard policy optimization process. The final optimization objective combines the RLVR loss with our modulated advantages $\hat { A } _ { i , t } ^ { \mathrm { f i n a l } }$ , and the self-calibration experience loss $\mathcal { L } _ { \mathrm { e x p } } \mathrm { . }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f i n a l } } ( \theta ) = \mathcal { L } _ { \mathrm { R L V R } } ( \theta ; \hat { A } _ { i , t } ^ { \mathrm { f i n a l } } ) + \lambda _ { \mathrm { e x p } } \mathcal { L } _ { \mathrm { e x p } } ( \theta ) , } \end{array}\tag{16}
$$

where $\mathcal { L } _ { \mathrm { R L V R } } ( \theta ; \hat { A } _ { i , t } ^ { \mathrm { f i n a l } } )$ is the standard RLVR loss using the modified advantage $\hat { A } _ { i , t } ^ { \mathrm { f i n a l } }$ , and $\lambda _ { \mathrm { e x p } }$ is a coefficient balancing the two components. This objective not only preserves the on-policy learning signal but also anchors policy optimization to highquality historical experiences and visually critical tokens, improving the utilization efficiency of highvalue optimization signals in RLVR.

## 5 Experiments

## 5.1 Experimental Setup

Models and Baselines. We adopt Qwen-2.5-VL-3B and Qwen-2.5-VL-7B as our base models. To evaluate data scalability, we first conducted experiments on the Geometry3K (Lu et al., 2021) dataset and further extended the training to the larger-scale VIRL39K dataset (Wang et al., 2025b). We reproduce the algorithms GRPO (Shao et al., 2024), DAPO (Yu et al., 2025b), PAPO (Wang et al., 2026b), and VPPO (Huang et al., 2026a) across different model scales and datasets.

Evaluation. The evaluation benchmarks include MathVista (Lu et al., 2024), MathVerse (Zhang et al., 2025d), We-Math (Qiao et al., 2025), MMK12 (Meng et al., 2025) and Geometry3K (Lu et al., 2021), LogicVista (Xiao et al., 2024), SuperClevr-Counting (Li et al., 2023), MMMU-Pro (Yue et al., 2025) and MathVerse-V (Zhang et al., 2025d). We also test on ScienceQA (Lu et al., 2022), HallusionBench (Guan et al., 2024), ChartQAPro (Masry et al., 2025), InfographicVQA (Mathew et al., 2022) and RealWorldQA from lmms-eval (Zhang et al., 2025b) to evaluate out-of-domain performance. To reduce reliance on LLM-as-a-judge systems, we adopt an exactmatch scoring protocol and report the average accuracy@8 with an inference temperature of 1.0 following PAPO (Wang et al., 2026b). More details are provided in Appendix C.1.

Implementation Details. Our training framework follows the DAPO (Yu et al., 2025b) recipe with a learning rate of 1e-6, a rollout batch size of 384, and a maximum response length of 2048. Models are trained for 15 epochs on Geo3k and 2 epochs on VIRL39K. We set the rollout group size $n = 5$ for the 3B model, and $n = 8$ for the 7B models. Following VPPO (Huang et al., 2026a), we apply a small entropy penalty (coefficient 0.06) to all models during training to ensure training stability and fair comparison. More details are provided in Appendix C.2.

## 5.2 Main Results

Multimodal Reasoning Performance. Table 1 and Table 2 present the performance of PIVOT compared to other baselines. When trained on the Geometry3K dataset, PIVOT consistently achieves the highest average accuracy for both the 3B and 7B models, reaching 51.25 (11.27% relative gains over DAPO) and 57.85 (10.57% relative gains over DAPO), respectively. The performance improvements are observed in both general mathematical and vision-dependent reasoning tasks. To validate data scalability, we further extend the training of the 3B model to the larger-scale VIRL39K dataset. As shown in Table 2, PIVOT maintains its superiority, yielding the highest average score of 55.68 and outperforming competitive baselines across most evaluated benchmarks. These results demonstrate the effectiveness of PIVOT in enhancing the multimodal reasoning capabilities of LVLMs.

<table><tr><td rowspan="2">Model</td><td colspan="5">Mathematical &amp; Geometric Reasoning</td><td colspan="3">Vision-Dependent Reasoning</td><td rowspan="2">Avg.</td></tr><tr><td></td><td>Geo3Ktest MathVerse MathVista WeMath MMK12</td><td></td><td></td><td></td><td>LogicVista Counting MathVersey</td><td></td><td></td></tr><tr><td>Qwen2.5-VL-3B</td><td>18.95</td><td>33.33</td><td>46.49</td><td>30.10</td><td>36.47</td><td>29.42</td><td>37.25</td><td>30.55</td><td>32.82</td></tr><tr><td>+ GRPO</td><td>35.88</td><td>42.92</td><td>52.43</td><td>47.25</td><td>40.57</td><td>36.66</td><td>47.25</td><td>39.28</td><td>42.78</td></tr><tr><td>+ DAPO</td><td>41.60</td><td>46.88</td><td>55.19</td><td>53.00</td><td>40.53</td><td>37.11</td><td>50.75</td><td>43.38</td><td>46.06</td></tr><tr><td>+ PAPOD</td><td>43.68</td><td>50.46</td><td>55.54</td><td>54.87</td><td>40.92</td><td>38.17</td><td>56.31</td><td>47.59</td><td>48.44</td></tr><tr><td>+ VPPO</td><td>43.16</td><td>51.89</td><td>54.28</td><td>58.05</td><td>42.14</td><td>37.78</td><td>55.31</td><td>46.78</td><td>48.67</td></tr><tr><td>+ PIVOT (Ours)</td><td>43.07</td><td>53.90</td><td>56.25</td><td>60.07</td><td>43.05</td><td>38.20</td><td>65.63</td><td>49.83</td><td>51.25</td></tr><tr><td>Qwen2.5-VL-7B</td><td>35.73</td><td>39.02</td><td>63.50</td><td>45.80</td><td>43.11</td><td>42.84</td><td>66.75</td><td>34.58</td><td>|46.42</td></tr><tr><td>+ GRPO</td><td>47.40</td><td>47.84</td><td>66.46</td><td>56.61</td><td>45.52</td><td>43.82</td><td>72.69</td><td>43.56</td><td>52.99</td></tr><tr><td>+ DAPO</td><td>51.73</td><td>43.77</td><td>65.56</td><td>49.73</td><td>43.74</td><td>46.09</td><td>80.50</td><td>37.41</td><td>52.32</td></tr><tr><td>+ PAPOD</td><td>51.66</td><td>48.70</td><td>65.96</td><td>54.27</td><td>44.73</td><td>45.69</td><td>84.56</td><td>43.14</td><td>54.84</td></tr><tr><td>+ VPPO</td><td>52.06</td><td>51.03</td><td>68.94</td><td>55.78</td><td>47.11</td><td>44.97</td><td>80.13</td><td>44.24</td><td>55.53</td></tr><tr><td>+ PIVOT (Ours)</td><td>53.14</td><td>54.44</td><td>68.38</td><td>60.88</td><td>49.64</td><td>46.28</td><td>82.50</td><td>47.52</td><td>57.85</td></tr></table>

Table 1: Performance comparison of PIVOT against various baselines. Models are trained on the Geometry3K dataset using Qwen2.5-VL-3B and 7B as backbones. Bold and underlined indicate the best and second-best results.
<table><tr><td>Method Geo3K MathVista MathVerse</td><td></td><td></td><td></td><td></td><td></td><td>WeMath MMK12 LogicVista</td><td></td><td>Counting MMMU-Pro MathVersev</td><td></td><td>Avg.</td></tr><tr><td>Base</td><td>19.22</td><td>46.49</td><td>33.33</td><td>30.10</td><td>36.47</td><td>29.42</td><td>37.25</td><td>19.66</td><td>30.55</td><td>31.39</td></tr><tr><td>GRPO</td><td>30.99</td><td>59.15</td><td>55.37</td><td>60.05</td><td>58.18</td><td>39.71</td><td>58.81</td><td>27.06</td><td>52.08</td><td>49.04</td></tr><tr><td>DAPO</td><td>36.29</td><td>60.14</td><td>60.87</td><td>62.97</td><td>62.11</td><td>42.51</td><td>75.44</td><td>28.32</td><td>57.23</td><td>53.99</td></tr><tr><td>PAPOD</td><td>37.44</td><td>61.45</td><td>61.88</td><td>65.48</td><td>62.57</td><td>43.85</td><td>77.13</td><td>28.36</td><td>58.92</td><td>55.23</td></tr><tr><td>VPPO</td><td>36.04</td><td>60.33</td><td>59.78</td><td>64.18</td><td>62.26</td><td>42.25</td><td>74.56</td><td>28.14</td><td>56.95</td><td>53.83</td></tr><tr><td>PIVOT</td><td>37.73</td><td>63.04</td><td>62.52</td><td>65.01</td><td>61.96</td><td>44.44</td><td>77.81</td><td>28.98</td><td>59.67</td><td>55.68</td></tr></table>

Table 2: Performance evaluation of PIVOT and baselines trained on the larger-scale VIRL39K dataset using Qwen2.5-VL-3B as base model. Bold and underlined indicate the best and second-best results.

Training Dynamics. Figure 2 illustrates the training dynamics of the 7B model on Geo3k dataset. As shown in Figure 2a, PIVOT demonstrates superior learning efficiency, achieving higher training accuracy rewards compared to other baselines. Furthermore, the corresponding validation accuracy (Figure 2b) confirms that this efficient optimization directly translates into better generalization results on unseen validation data. These improvements demonstrate that our proposed method provides a more robust and effective optimization process.

## 5.3 Quantitative Analysis

Ablation Study. To isolate the contributions of different modules in our framework, we conduct ablation studies using the Qwen2.5-VL-3B model trained on the Geo3k dataset. PIVOT consists of two core modules: Self-calibrated Experience Replay (SER) and Vision-guided Advantage Allocation (VAA). We ablate each module to verify its influence. Moreover, we further conduct ablations on the replay and token-utility designs. Specifically,

![](images/4788e5ab22d9dc21b42dc53d1bc25790ac40d5f0c9c304afd21a30ca0788c6ef.jpg)

![](images/3819f7208558afe88f0b75c63c118508ab340637ef5199723057830f6cc9b87c.jpg)  
(a) Accuracy Rewards.  
(b) Validation Accuracy.  
Figure 2: Training dynamics of Qwen2.5-VL-7B on the Geometry3k dataset: (a) training accuracy rewards, and (b) validation accuracy on the Geometry3k@test data (Lu et al., 2021).

Off-PG+VAA reuses successful experiences via off policy gradient updates, whereas Off-SFT+VAA uses them as supervised fine-tuning targets. In addition, $S E R + V A A _ { l o c a l }$ removes the entropy-gated future visual utility from VAA, isolating the effect of local visual utility alone. As shown in Table 3, both SER-only and VAA-only yields noticeable performance gains over the baseline, and their combination yields the best average performance across evaluation benchmarks. Off-PG+VAA and

<table><tr><td>Variant  $\mathbf { G e o 3 k _ { t e s t } }$ </td><td colspan="8">MathVerse MathVista WeMath MMK12 LogicVista Counting MathVersey Avg.</td><td> $\Delta$ </td></tr><tr><td>DAPO (Baseline)</td><td>41.60</td><td>46.88</td><td>55.19</td><td>53.00</td><td>40.53</td><td>37.11</td><td>50.75</td><td>43.38</td><td>46.06</td><td></td></tr><tr><td>+ SER only</td><td>42.37</td><td>51.19</td><td>55.23</td><td>57.67</td><td>42.51</td><td>38.62</td><td>55.38</td><td>48.05</td><td></td><td> $4 8 . 8 8 ~ + 6 . 1 2 \%$ </td></tr><tr><td>+ VAA only</td><td>44.09</td><td>48.56</td><td>57.08</td><td>57.51</td><td>41.74</td><td>38.93</td><td>65.94</td><td>45.39</td><td></td><td> $4 9 . 9 1 ~ + 8 . 3 6 \%$ </td></tr><tr><td> $\mathrm { O f f - P G + V A A }$ </td><td>44.20</td><td>51.06</td><td>56.89</td><td>54.31</td><td>42.18</td><td>39.23</td><td>67.75</td><td>46.99</td><td></td><td> $5 0 . 3 3 ~ + 9 . 1 7 \%$ </td></tr><tr><td> $\mathrm { O f f - S F T + V A A }$ </td><td>44.03</td><td>51.26</td><td>55.88</td><td>55.94</td><td>41.91</td><td>38.95</td><td>61.50</td><td>46.78</td><td></td><td> $4 9 . 5 3 ~ + 7 . 1 0 \%$ </td></tr><tr><td> $\mathrm { S E R + V A A _ { l o c a l } }$ </td><td>44.57</td><td>49.76</td><td>55.68</td><td>57.52</td><td>42.43</td><td>38.42</td><td>59.38</td><td>45.60</td><td></td><td> $4 9 . 1 7 \ + 6 . 7 5 \%$ </td></tr><tr><td>CalibRL</td><td>41.43</td><td>44.75</td><td>52.85</td><td>50.79</td><td>38.70</td><td>36.52</td><td>56.81</td><td>40.75</td><td></td><td> $4 5 . 3 3 \ \mathrm { ~  ~ { ~ - 1 . 5 8 \% } ~ }$ </td></tr><tr><td> $\mathbf { C a l i b R L + V A A }$ </td><td>44.05</td><td>49.59</td><td>54.86</td><td>54.56</td><td>41.06</td><td>37.98</td><td>60.00</td><td>45.75</td><td></td><td> $4 8 . 4 8 ~ + 5 . 2 5 \%$ </td></tr><tr><td>PIVOT (full)</td><td>43.07</td><td>53.90</td><td>56.25</td><td>60.07</td><td>43.05</td><td>38.20</td><td>65.63</td><td>49.83</td><td></td><td> $\mathbf { 5 1 . 2 5 + 1 1 . 2 7 \% }$ </td></tr></table>

Table 3: Ablation of Self-calibrated Experience Replay (SER) module and Vision-guided Advantage Allocation (VAA) module. Their combination yields the best results, confirming the effectiveness of our dual-level framework.

$O f f - S F T + V A A$ improve over baseline but lag behind full PIVOT, indicating the advantage of using high-quality experiences as calibration anchors. The gap between $S E R + V A A _ { l o c a l }$ and full PIVOT further validates the contribution of entropy-gated future visual utility. It is worth noting that PIVOT slightly sacrifices the in-domain Geo3K test set performance relative to several ablations while improving broader generalization. This arises from SER’s regularization against source-domain overspecialization and VAA’s modeling of downstream visual impact beyond local token–image correspondence. Together, these designs balance exploitation and exploration, enabling broader generalization.

In addition, to isolate the specific benefits of our self-generated visually grounded anchors, we also conducted experiments evaluating both CalibRL and CalibRL+VAA. As shown in Table 3, PIVOT consistently outperforms CalibRL and the CalibRL+VAA variant on average and across almost all benchmarks. This superiority can be attributed to two main factors: (1) PIVOT’s self-generated anchors, which are inherently compatible with the evolving policy to ensure better-calibrated confidence comparisons, and (2) the proposed entropy and visual-dependency filtering mechanisms. We also ablate different visual intervention strategies; details are shown in Appendix D.1.

Out-of-Domain Generalization. To ensure that our model is not simply over-optimized for math and geometry benchmarks, we evaluate its Out-of-Domain generalization capabilities on ScienceQA, HallusionBench, ChartQAPro, InfographicVQA, and RealWorldQA. As shown in Table 4, our method consistently achieves improvements over baselines across different OOD datasets. These results suggest that the visual reasoning capabilities acquired through our training paradigm can generalize to broader multimodal tasks and help mitigate multimodal hallucinations.

![](images/832c09679cae31a73a8655dfbdb4d5339fb71f2d20860db1e7b8d9eaea111a04.jpg)

![](images/2d6f94415234185bd251af4d77fbb3a0b9b18eac12d1d6a9aace42cdc0531e11.jpg)  
(a) Sensitivity analysis on $\beta$ (b) Sensitivity analysis on $\lambda _ { \mathtt { e x p } }$  
Figure 3: Sensitivity Analysis of $\beta$ and $\lambda _ { \mathrm { e x p } }$

Sensitivity Analysis To evaluate the robustness of our method, we conduct sensitivity analyses on two key hyperparameters: the advantage scaling factor $\beta$ and the loss balancing coefficient $\lambda _ { \mathrm { e x p } }$ . As shown in Figure 3, both hyperparameters exhibit a similar pattern: moderate values lead to the best performance. This is consistent with our design intuition. A small $\beta$ weakens the effect of visual advantage modulation, whereas a large $\beta$ may overamplify visual signals and destabilize optimization. Similarly, $\lambda _ { \mathrm { e x p } }$ needs to balance the experience calibration loss with the main RL objective: insufficient weighting limits its benefit, while excessive weighting may interfere with on-policy learning.

Generalize to GRPO To verify the generalizability of PIVOT, we extended our training recipe to Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which serves as another widely adopted baseline in the RLVR paradigm. Specifically, we seamlessly integrated our self-calibrated experience replay and vision-guided advantage modulation into the standard GRPO optimization

<table><tr><td>Method</td><td colspan="5">ScienceQA HallusionBench ChartQAPro InfographicVQA RealWorldQA Avg.</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>68.75</td><td>51.76</td><td>26.90</td><td>49.73</td><td>44.59 48.35</td></tr><tr><td>+ DAPO</td><td>80.25</td><td>56.02</td><td>32.94</td><td>51.64 46.90</td><td>53.55</td></tr><tr><td>+ PIVOT (Ours)</td><td>82.04</td><td>57.64</td><td>34.73</td><td>52.86</td><td>47.47 54.95</td></tr></table>

Table 4: Out-of-Domain generalization results. Though trained only on the Geometry3K dataset, our method shows strong generalization to out-of-domain multimodal tasks.
<table><tr><td>Strategy</td><td colspan="10">Geo3K MathVerse MathVista WeMath MMK12 LogicVista Counting MathVersev</td></tr><tr><td>Qwen2.5-VL-3B</td><td>18.95</td><td>33.33</td><td>46.49</td><td>30.10</td><td>36.47</td><td>29.42</td><td>37.25</td><td>30.55</td><td>Avg 32.82</td><td>∆</td></tr><tr><td>GRPO (Baseline)</td><td>35.88</td><td>42.92</td><td>52.43</td><td>47.25</td><td>40.57</td><td>36.66</td><td>47.25</td><td>39.28</td><td>42.78 +30.35%</td><td></td></tr><tr><td>PIVOTG</td><td>37.79</td><td>45.29</td><td>55.04</td><td>47.57</td><td>41.81</td><td>37.00</td><td>53.94</td><td>42.84</td><td></td><td>45.16 +37.60%</td></tr></table>

Table 5: Results of extending PIVOT to the GRPO algorithm.

loop.

As shown in Table 5, incorporating PIVOT consistently improves standard GRPO’s performance across multimodal reasoning tasks. This consistent improvement indicates that our dual-level learning framework is highly orthogonal to the choice of the specific RLVR algorithms. Rather than being confined to a specific algorithm, PIVOT functions as a general framework that effectively enhances the visual grounding capability of LVLMs during reinforcement learning.

## 5.4 Additional Results

We also provide several additional results to complement the main experiments. Specifically, Appendix D.2 provides the token visual utility visualizations, which demonstrate that PIVOT concentrates the RL optimization signal on visually grounded perception tokens and reasoning pivots that are more relevant to solving the problem. Appendix D.3 presents a layer-wise attention analysis, which provides complementary evidence that PIVOT encourages the model to attend more to visual tokens when performing multimodal reasoning. Appendix D.4 reports the computational overhead of the proposed framework. These results offer further insights into the behavior and practical cost of PIVOT.

## 6 Conclusion

In this paper, we proposed PIVOT, a dual-level learning framework that addresses key optimization bottlenecks in multimodal RLVR by preserving and amplifying visual reasoning signals. The framework collects and selectively replays high-quality experiences as optimization anchors through a selfcalibration loss. Building upon this, PIVOT further allocates fine-grained visual advantages to tokens based on their local visual support and downstream impact on reasoning. Extensive experiments demonstrate that PIVOT consistently improves multimodal reasoning capabilities and out-of-domain generalization of LVLMs. Future work could consider extending this dual-level learning framework to broader multimodal agent tasks.

## Limitations

Despite the promising performance of PIVOT, several limitations remain to be addressed in future work. First, the self-calibrated experience replay module and the counterfactual intervention strategy inevitably introduce additional computational and storage overhead during training. Although these overheads are acceptable, exploring more resource-efficient designs and lightweight perturbation strategies remains an important direction for future optimization. Secondly, due to constrained computational resources, we primarily evaluated our framework on vision-language models up to the 7B parameter scale. While PIVOT demonstrates consistent improvements on these models, validating our framework on much larger scales, such as 32B variants, would be beneficial to fully verify its scalability. Finally, our current selection of high-quality trajectories relies on empirical proxies, specifically visual dependency and trajectory entropy. While these metrics are effective in practice, they may not fully capture the semantic correctness of complex multi-step reasoning. Future work could integrate structured, rubric-based evaluation criteria (Gunjal et al., 2026) to define and filter high-quality visual reasoning paths more robustly.

## Ethical Considerations

This work focuses on improving multimodal reasoning capabilities of large vision-language models. All experiments are conducted on public datasets, without using private user data, personally identifiable information, or human-subject data. Nevertheless, models trained with our method may still produce incorrect or biased outputs, especially under distribution shifts or in high-stakes real-world scenarios. Thus, such models should be carefully validated and used with human oversight before deployment in domains such as medicine, law, or safety-critical inspection. AI-assisted tools were used only for language polishing and coding assistance.

## Acknowledgments

This work is jointly supported by the National Key R&D Program of China (Grant No. 2024YFF0505703) and Beijing Municipal Natural Science Foundation (Grant No. L257009).

## References

Thomas Foster, Anya Sims, Johannes Forkel, and Jakob Foerster. 2025. Lilo: Learning to reason at the frontier of learnability. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 48941–48974. Curran Associates, Inc.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375– 14385.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean M. Hendryx. 2026. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In The Fourteenth International Conference on Learning Representations.

Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. 2026a. Spotlight on token perception for multimodal reinforcement learning. In The Fourteenth International Conference on Learning Representations.

Wenxuan Huang, Bohan Jia, Shaosheng Cao, Zheyu Ye, Fei zhao, Zhe Xu, Yao Hu, and Shaohui Lin. 2026b. Vision-r1: Incentivizing reasoning capability in multimodal large language models. In The Fourteenth International Conference on Learning Representations.

Zhuoxu Huang, Mengxi Jia, Hao Sun, Xuelong Li, and Jungong Han. 2026c. Controllable exploration in hybrid-policy RLVR for multi-modal reasoning. In The Fourteenth International Conference on Learning Representations.

Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, and Hao Wang. 2025. Vcrl: Variance-based curriculum reinforcement learning for large language models. arXiv preprint arXiv:2509.19803.

Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, and Linfeng Zhang. 2026. Credit where it is due: Cross-modality connectivity drives precise reinforcement learning for mllm reasoning. arXiv preprint arXiv:2602.11455.

Yunheng Li, Hangyi Kuang, Hengrui Zhang, Jiangxia Cao, Zhaojie Liu, Qibin Hou, and Ming-Ming Cheng. 2026a. Rethinking token-level policy optimization for multimodal chain-of-thought. arXiv preprint arXiv:2603.22847.

Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. 2023. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14963–14973.

Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen Liang, Rui Liu, Fuxiao Liu, Jingxi Chen, Dian Yu, Jordan Lee Boyd-Graber, Haitao Mi, and Dong Yu. 2026b. Vision-SR1: Self-rewarding vision-language model via reasoning decomposition and multi-reward policy optimization. In The Fourteenth International Conference on Learning Representations.

Jing Liang, Jinyi Liu, Yi Ma, Hongyao Tang, YAN ZHENG, Shuyue Hu, LEI BAI, and Jianye HAO. 2026. Squeeze the soaked sponge: Efficient offpolicy RFT for large language model. In The Fourteenth International Conference on Learning Representations.

Long-Ji Lin. 1992. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3):293–321.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. 2026. GDPO: Group reward-decoupled normalization policy optimization for multi-reward RL optimization. In Forty-third International Conference on Machine Learning.

Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. 2025a. Noisyrollout: Reinforcing visual reasoning with data augmentation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. 2025b. Visual-rft: Visual reinforcement fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2034– 2044.

Jinda Lu, Junkang Wu, Jinghan Li, Kexin Huang, Shuo Yang, Guoyin Wang, Jiancan Wu, Xiang Wang, and Xiangnan He. 2026. Bridging perception and reasoning: Token reweighting for rlvr in multimodal llms. arXiv preprint arXiv:2603.25077.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2024. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774– 6786, Online. Association for Computational Linguistics.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems.

Ruilin Luo, Chufan Shi, Yizhen Zhang, Cheng Yang, Songtao Jiang, Tongkun Guan, Ruizhe Chen, Ruihang Chu, Peng Wang, Mingkun Yang, Lei Wang, Yujiu Yang, Junyang Lin, and Zhibo Yang. 2026. From narrow to panoramic vision: Attention-guided cold-start reshapes multimodal reasoning. In The Fourteenth International Conference on Learning Representations.

Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, Megh Thakkar, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. 2025. ChartQAPro: A more diverse and challenging benchmark for chart question answering. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19123–19151, Vienna, Austria. Association for Computational Linguistics.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2582–2591. IEEE.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. 2025. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. Preprint, arXiv:2503.07365.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, YiFan Zhang, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Xiao Zong, Yida Xu, Peiqing Yang, Zhimin Bao, Muxi Diao, Chen Li, and Honggang Zhang. 2025. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 20023– 20070, Vienna, Austria. Association for Computational Linguistics.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yuzhi Zhang, and Yue Wang. 2025a. Eframe: Deeper reasoning via exploration-filter-replay reinforcement learning framework. arXiv preprint arXiv:2506.22200.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. 2025b. Vlrethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 30865–30891. Curran Associates, Inc.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. 2025c. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 115452–115486. Curran Associates, Inc.

Xu Wang, Sen Wang, Xingxing Liang, Dawei Zhao, Jincai Huang, Xin Xu, Bin Dai, and Qiguang Miao. 2024. Deep reinforcement learning: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(4):5064–5078.

Zengbin Wang, Feng Xiong, Liang Lin, Xuecai Hu, Yong Wang, Yanlin Wang, Man Zhang, and Xiangxiang Chu. 2026a. Visually-guided policy optimization for multimodal reasoning. arXiv preprint arXiv:2604.09349.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru WANG, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, and Heng Ji. 2026b. Perception-aware policy optimization for multimodal reasoning. In The Fourteenth International Conference on Learning Representations.

Zhaohua Wu, Norden E. Huang, Steven R. Long, and Chung-Kang Peng. 2007. On the trend, detrending, and variability of nonlinear and nonstationary time series. Proceedings ofthe National Academy ofSciences, 104(38):14889–14894.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. 2024. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973.

Zekai Ye, Qiming Li, Xiaocheng Feng, Ruihan Chen, Ziming Li, Haoyu Ren, Kun Chen, Dandan Tu, and Bing Qin. 2026. Not all tokens see equally: Perception-grounded policy optimization for large vision-language models. arXiv preprint arXiv:2604.01840.

En Yu, Kangheng Lin, Liang Zhao, jisheng yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wenbing Tao. 2025a. Perceptionr1: Pioneering perception policy with reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 94827–94853. Curran Associates, Inc.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, juncai liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025b. Dapo: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 113222–113244. Curran Associates, Inc.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. 2025. MMMU-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, Vienna, Austria. Association for Computational Linguistics.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, and Yu Cheng. 2026. ExGRPO: Learning to reason from experience. In The Fourteenth International Conference on Learning Representations.

Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, and Guorui Zhou. 2025a. Rlep: Reinforcement learning with experience replay for llm reasoning. arXiv preprint arXiv:2507.07451.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. 2025b. LMMs-eval: Reality check on the evaluation of large multimodal models. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 881–916, Albuquerque, New Mexico. Association for Computational Linguistics.

Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, and 20 others. 2025c. A survey of reinforcement learning for large reasoning models. Preprint, arXiv:2509.08827.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, Peng Gao, and Hongsheng Li. 2025d. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In Computer Vision – ECCV 2024, pages 169–186, Cham. Springer Nature Switzerland.

## A Justification of the Counterfactual Visual Support Score

In Section 4.2, we define the direct counterfactual support score as a probability ratio rather than a standard log-likelihood difference. Here, we provide the theoretical justification for this design choice.

For simplicity of notation at generation step t, let $p _ { t } ( y ) = \pi _ { \boldsymbol { \theta } } ( y _ { t } = y \mid q , I , \mathbf { y } _ { < t } )$ denote the token distribution conditioned on the original image, and $q _ { t } ( y ) = \pi _ { \theta } ( y _ { t } = y \mid q , \widetilde { I } , \mathbf { y } _ { < t } )$ denote the distribution conditioned on the corrupted image. For a sampled token $y _ { i , t } .$ , our metric is defined as:

$$
c _ { t } = 1 - \frac { q _ { t } ( y _ { t } ) } { p _ { t } ( y _ { t } ) } .\tag{17}
$$

And this computation is detached from the gradient graph. Intuitively, this quantity measures whether the sampled token receives more probability support from the original-image condition than from the corrupted-image condition. If $\dot { \mathbf { \nabla } } p _ { t } ( y _ { t } ) > q _ { t } ( y _ { t } )$ then $c _ { t } > 0$ , meaning that the token is more likely to be generated when the correct visual evidence is available. Such a token can be regarded as visually supported. If $p _ { t } ( y _ { t } ) \approx q _ { t } ( y _ { t } )$ , then $c _ { t } \approx 0$ , indicating that the token can be similarly explained with or without the original image, and is therefore more likely to correspond to language priors, formatting tokens, or generic connectors. If $p _ { t } ( y _ { t } ) < q _ { t } ( y _ { t } )$ then $c _ { t } < 0 .$ , suggesting that the token is even more favored under the corrupted-image condition, these tokens may correspond to visual hallucinations or some completely unrelated tokens and should be suppressed.

The key advantage of this form is that it estimates a signed distributional difference between the original-image and corrupted-image policies. Specifically, for any vocabulary token a, we have

$$
\begin{array} { c } { { \mathbb { E } _ { y \sim p _ { t } } \left[ c _ { t } ( y ) \mathbf { 1 } \{ y = a \} \right] = p _ { t } ( a ) \left( 1 - \frac { q _ { t } ( a ) } { p _ { t } ( a ) } \right) } } \\ { { = p _ { t } ( a ) - q _ { t } ( a ) . } } \end{array}\tag{18}
$$

Thus, under sampling from the original-image policy, the residual provides an unbiased estimate of the signed probability-mass difference $p _ { t } - q _ { t }$

This property also explains why the score is preferable to directly using a log-likelihood gap such as

$$
\log p _ { t } ( y _ { t } ) - \log q _ { t } ( y _ { t } ) .\tag{19}
$$

Although the log-likelihood gap is intuitive, its expectation under $p _ { t }$ is always non-negative:

$$
\mathbb { E } _ { y \sim p _ { t } } \left[ \log \frac { p _ { t } ( y ) } { q _ { t } ( y ) } \right] = D _ { \mathrm { K L } } ( p _ { t } \Vert q _ { t } ) \ge 0 .\tag{20}
$$

Therefore, if it is directly added to the token-level advantage, it tends to introduce a positive bonus on average, which may encourage longer responses or over-reward tokens from prefixes with large distributional divergence. By contrast, the proposed residual satisfies

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { y \sim p _ { t } } \left[ 1 - \frac { q _ { t } ( y ) } { p _ { t } ( y ) } \right] = \sum _ { y } p _ { t } ( y ) - \sum _ { y } q _ { t } ( y ) } \\ { = 0 . } \end{array}\tag{21}
$$

Hence, $c _ { t }$ is a zero-mean signed residual under the original-image distribution. It redistributes optimization strength from tokens more favored by the corrupted-image or language-prior condition to tokens more favored by the original-image condition, rather than injecting an additional uniformly positive reward.

When used as an additive token-level advantage correction,

$$
\begin{array} { r } { \widetilde { A } _ { i , t } = A _ { i } + \eta r _ { c , t } , } \end{array}\tag{22}
$$

the score induces a policy-gradient correction that pushes the model toward the excess probability mass of the original-image distribution. To see this, consider the expected correction term for the logit of a vocabulary token a. Since

$$
\begin{array} { r } { \mathbb { E } _ { y \sim p _ { t } } [ c _ { t } ( y ) ] = 0 , } \end{array}\tag{23}
$$

we have

$$
\begin{array} { r l } & { \mathbb { E } _ { y \sim p _ { t } } \left[ c _ { t } ( y ) \nabla _ { z _ { t } ( a ) } \log p _ { t } ( y ) \right] } \\ & { \ = \mathbb { E } _ { y \sim p _ { t } } \left[ c _ { t } ( y ) \big ( \mathbf { 1 } \{ y = a \} - p _ { t } ( a ) \big ) \right] } \\ & { \ = \mathbb { E } _ { y \sim p _ { t } } \left[ c _ { t } ( y ) \mathbf { 1 } \{ y = a \} \right] - p _ { t } ( a ) \mathbb { E } _ { y \sim p _ { t } } [ c _ { t } ( y ) ] } \\ & { \ = p _ { t } ( a ) - q _ { t } ( a ) . } \end{array}\tag{24}
$$

Therefore, this correction increases the logits of tokens whose probability mass is higher under the original-image condition and decreases those whose probability mass is higher under the corrupted-image condition. This gives $c _ { i , t }$ a clear optimization meaning: it encourages the model to reinforce the probability mass specifically contributed by visual evidence, while suppressing probability mass supported by language priors or visual hallucinations.

## B Details of the Detrending Operator

During preliminary experiments, we find that the future-aware visual score may introduce a mild length-related bias. Although the future score is computed within a finite window, tokens near different relative positions can still have systematically different future-support statistics due to sequence boundaries, response length variation, and heterogeneous reasoning structures. In some cases, this bias encourages overly long responses. To mitigate this issue, we apply a lightweight detrending step to the entropy-gated future visual score. Let $a _ { i , t } = u _ { i , t } F _ { i , t }$ denote the raw entropy-gated future visual score for token t in response i. Let $E _ { i }$ be the set of valid tokens in response i and $n _ { i } = | E _ { i } |$ . For each token $t \in E _ { i }$ , we define its relative position as

$$
r _ { i , t } = \frac { \mathrm { r a n k } _ { E _ { i } } ( t ) } { \operatorname* { m a x } ( n _ { i } - 1 , 1 ) } ,\tag{25}
$$

where rank $_ { E _ { i } } ( t ) \in \{ 0 , \dots , n _ { i } - 1 \}$ is the rank of token t among valid tokens. Thus, $r _ { i , t } \in [ 0 , 1 ]$ normalizes token position within each response.

For each response, we fit a response-specific linear trend between the raw future score and the relative position:

$$
( \alpha _ { i } , \beta _ { i } ) = \arg \operatorname* { m i n } _ { \alpha , \beta } \sum _ { t \in E _ { i } } \left( a _ { i , t } - \alpha - \beta r _ { i , t } \right) ^ { 2 } .\tag{26}
$$

The detrending operator is then defined as:

$$
\mathrm { D e t r e n d i n g } ( a _ { i , t } ) = a _ { i , t } - ( \alpha _ { i } + \beta _ { i } r _ { i , t } ) .\tag{27}
$$

This operation can be viewed as response-wise position detrending. It removes the component of the future-aware score that can be explained by relative token position, and preserves the tokenspecific deviation beyond this trend. Therefore, the retained signal does not simply reward a token for appearing in a favorable position of the response, but reflects whether the token has stronger future visual support than expected from its location.

## C Experimental Settings

## C.1 Data and Evaluation

Training Datasets. To examine data scalability, we conduct reinforcement learning on two training sets with different scales and data coverage.

• Geometry3K (Lu et al., 2021): a geometry problem-solving dataset containing 3,002 multiple-choice geometry problems. The dataset requires models to jointly parse geometric diagrams, understand problem statements, and perform symbolic geometric reasoning. We follow the standard split, using the 2,101 training problems for training and the 601 test problems for validation.

• ViRL39K (Wang et al., 2025b): a largerscale vision-language reinforcement learning dataset consisting of 38,870 verifiable question-answering instances. ViRL39K is constructed from newly collected problems and existing multimodal reasoning datasets through cleaning, reformatting, rephrasing, and rule-based verification. Compared with Geometry3K, it covers a broader range of visual reasoning scenarios, including gradeschool problems, STEM and social topics, charts, diagrams, tables, documents, and spatial reasoning.

Evaluation Benchmarks. We conduct evaluation on diverse benchmarks to evaluate multimodal reasoning performance, details are as follows:

• MathVista (Lu et al., 2024): a visual mathematical reasoning benchmark with 6,141 examples collected from 28 existing multimodal datasets and three newly created datasets. It evaluates mathematical reasoning in diverse visual contexts and requires fine-grained visual understanding as well as compositional reasoning.

• MathVerse (Zhang et al., 2025d): it contains 2,612 visual math problems, each reformulated into multiple modality variants with different amounts of textual and visual information. This design allows the benchmark to diagnose whether a model truly uses diagrams rather than relying only on textual cues.

• We-Math (Qiao et al., 2025): it consists of 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and five levels of knowledge granularity. It is designed to evaluate not only final-answer accuracy but also the underlying knowledge acquisition and generalization behavior of multimodal models.

• MMK12 (Meng et al., 2025): it is a K12 multimodal reasoning dataset that covers mathematics, physics, chemistry, and biology. Its evaluation set contains 2,000 multiple-choice questions, with 500 questions for each subject, and is designed to assess multidisciplinary visual reasoning under human-verified solutions.

• LogicVista (Xiao et al., 2024): it is a visual logical reasoning benchmark containing 448 human-annotated multiple-choice questions.

• SuperCLEVR-Counting (Li et al., 2023): it is derived from Super-CLEVR, a controllable visual reasoning benchmark designed to diagnose domain robustness under factors such as visual complexity, question redundancy and concept compositionality. We use its counting subset to evaluate fine-grained object perception and numerical reasoning.

• MMMU-Pro (Yue et al., 2025): it is a more robust version of MMMU that filters out questions answerable by text-only models, augments answer choices from four to ten, and introduces a vision-only setting where questions are embedded into screenshots or photos.

• MathVerse-V (Zhang et al., 2025d): it is a vision-centric subset of MathVerse, where solving the problem requires substantial information from the visual input.

## C.2 Implementation Details.

We trained Qwen2.5-VL-3B and Qwen2.5-VL-7B models on the Geometry3K and VIRL39K datasets. All experiments were conducted utilizing PyTorch 2.6.0 and CUDA 12.4. Following GDPO (Liu et al., 2026), we add a small group decoupled format advantage with an advantage weight of 0.1 to regularize the format of model responses. To perform intervention on images, we perform random patch masking (patch size 14, probability 0.6) on the input images following PAPO (Wang et al., 2026b). For future visual score computation, we use a discounted future window size W = 32, a discount factor γ = 0.8, and a future coefficient λ = 0.5. For training stability, the self-calibrated experience replay module is activated once the task solved ratio exceeds 0.45 or after a maximum number of warmup steps. Upon activation, 50% of each training batch is sampled from the experience buffer. We retain the lowest-entropy half of the candidates per prompt (retaining at least one), and then select the final experience that maximizes the trajectory dependency. For all training and evaluation experiments, we used the single, standardized prompt template shown below.

Reasoning Template   
SYSTEM:   
You are a helpful assistant.   
USER:   
{question}   
You first think through the reasoning process   
as an internal monologue, enclosed within   
<think> </think> tags. Then, provide your   
final answer enclosed within \boxed{ }.

## D Additional Results

## D.1 Ablation on Intervention Strategy

We compare three visual intervention strategies for computing visual support scores, including complete grey masking, additive Gaussian noise, and random patch masking following VPPO (Huang et al., 2026a).

• Complete masking replaces the entire image with a neutral grey canvas with RGB value (128, 128, 128), removing all visual information.

• Gaussian noise adds pixel-wise noise with a standard deviation of 189, which is calibrated such that each pixel has approximately a 50% probability of being saturated to its maximum or minimum value.

• Random patch masking follows the ViT-style patch structure of the base LVLM: the image is divided into 14×14 patches, and each patch is independently blackened with probability 0.6.

Table 6 shows that all intervention strategies outperform the DAPO baseline, confirming the usefulness of counterfactual visual perturbation for improving visual reasoning. Among them, random patch masking achieves the best average performance. Compared with complete masking, it avoids overly coarse removal of the whole image; compared with Gaussian noise, it introduces more severe visual corruption without severe pixel-level artifacts. Therefore, we adopt random patch masking as the default intervention strategy in all main experiments.

## D.2 Visualization of Token Utilities

To better understand how VAA reallocates tokenlevel optimization signals, we visualize the estimated token visual utilities in Figure 4. The left side shows the input image and question, while the right side displays the generated response with each token colored according to its visual utility. Warmer colors indicate higher utility values and thus stronger contribution to visually grounded reasoning. The visualization shows that VAA assigns higher utilities to tokens that are closely associated with visual perception and key reasoning operations. For example, tokens such as “interior”, “quadrilateral”, “angles”, “70”, “56”, and “marks” correspond to information that must be extracted from the image. Meanwhile, reasoning tokens such as “Given”, “Since”, and “divide” also receive high utility, since they connect the perceived visual facts to the final geometric derivation. In contrast, routine function words and formatting tokens generally receive lower utility. This suggests that VAA does not uniformly amplify all response tokens, but instead concentrates the RL optimization signal on visually grounded perception tokens and reasoning pivots that are more relevant to solving the problem.

<table><tr><td>Strategy</td><td colspan="8">Geo3K MathVerse MathVista WeMath MMK12 LogicVista Counting MathVersev Avg</td></tr><tr><td>DAPO (Baseline)</td><td>41.60</td><td>46.88</td><td>55.19</td><td>53.00</td><td>40.53</td><td>37.11</td><td>50.75</td><td>43.38</td><td>46.06</td></tr><tr><td>Complete Grey Mask</td><td>44.24</td><td>47.67</td><td>57.63</td><td>55.85</td><td>42.75</td><td>39.21</td><td>65.69</td><td>43.86</td><td>49.61</td></tr><tr><td>Gaussian Noise</td><td>44.30</td><td>52.87</td><td>56.06</td><td>60.02</td><td>43.17</td><td>40.24</td><td>64.56</td><td>48.69</td><td>51.24</td></tr><tr><td>Random Patch Masking†</td><td>43.07</td><td>53.90</td><td>56.25</td><td>60.07</td><td>43.05</td><td>38.20</td><td>65.63</td><td>49.83</td><td>51.25</td></tr></table>

Table 6: Comparison of three visual intervention strategies. <sup>†</sup> Our final choice.

## D.3 Layer-wise Attention Analysis

To further examine whether PIVOT improves the model’s reliance on visual information, we conduct a layer-wise attention analysis in Figure 5. Specifically, we measure the attention mass assigned by response tokens to image tokens at each transformer layer, and compare PIVOT with the DAPO baseline and the original Qwen2.5-VL-7B-Instruct model. The results show that PIVOT consistently assigns higher attention mass to image tokens across most layers, especially in the middle and later layers where multimodal reasoning is more actively integrated. Compared with the original base model, DAPO already increases visual attention to some extent, suggesting that RLVR encourages the model to exploit visual evidence for solving reasoning tasks. PIVOT further strengthens this trend, indicating that trajectory-level anchoring and token-level visual advantage allocation promote stronger visual grounding during response generation. Although attention alone cannot fully characterize the reasoning process, this analysis provides complementary evidence that PIVOT encourages the model to attend more to visual tokens when performing multimodal reasoning.

## D.4 Computational Overhead of PIVOT

We further analyze the computational overhead introduced by PIVOT in Table 7. All measurements are conducted with Qwen2.5-VL-7B on 4 NVIDIA A100 80GB GPUs. Compared with DAPO, VPPO increases the average training time per step from 698.8s to 830.4s, mainly due to the additional computation required for visual perturbation-based signal estimation. PIVOT further increases the average time per step to 860.8s, corresponding to a 23.2% overhead over DAPO and only a modest additional overhead over VPPO. The extra cost of PIVOT mainly comes from two sources: maintaining the buffer for experience replay and estimating token visual utilities for advantage allocation. The memory footprint of the experience buffer is lightweight, requiring only 21.1 MB of additional RAM in our setting (training on Geometry3K dataset). Therefore, although PIVOT introduces extra computation during training, its memory overhead remains negligible, and the overall training cost is still practical for LVLM reinforcement learning.

Token Visual Utilities over Response Tokens  
![](images/52cd9feb4206ad6d21afe29a6c7eeb8b88a3ebb1595e06a7662e3cf7a81f9106.jpg)

![](images/c83f2aeaa703a8775b7513fcd7294a003cbe8e93962de737e0caab56c81e53b5.jpg)  
Figure 4: Visualization of token visual utilities. Warmer colors indicate higher utility scores.

<table><tr><td>Model &amp; Hardware</td><td>Method</td><td>Experience Buffer RAM</td><td>Avg. Time / Step (s)</td></tr><tr><td rowspan="4">Qwen2.5-VL-7B (4 GPUs)</td><td>DAPO</td><td>一</td><td>698.8</td></tr><tr><td>VPPO</td><td></td><td>830.4+18.8%</td></tr><tr><td>PIVOT</td><td>+21.1 MB</td><td>860.8+23.2%</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 7: Computational overhead comparison.

![](images/15a1d5c319ffe7c8df3ad37a17c4aa0058bb50ee2297c82ea193a84941b68839.jpg)  
Figure 5: Layer-wise attention analysis. We report the average attention mass assigned by response tokens to image tokens across transformer layers.