# Density-Aware Reward Scaling: A Reinforcement Learning Framework for Reward-Proportional Sampling

Ria Shekhawat<sup>∗</sup>, Abhijeet Sinha<sup>∗</sup>, Yingtao Zhu, Dianbo Liu

National University of Singapore

## Abstract

Reinforcement learning objectives that maximize expected return can concentrate probability mass on a small subset of high-reward outcomes, limiting diversity when multiple distinct solutions are desirable. We introduce Density-Aware Reward Scaling (DARS), a simple reward-processing mechanism that promotes reward-proportional sampling in continuous outcome spaces while retaining standard policy-gradient training. DARS estimates the local density of outcomes within each on-policy rollout group and inversely scales rewards by this density. The method requires no learned density model, auxiliary objective, critic, or modification to the underlying policy-gradient objective. In a controlled multimodal navigation task, DARS-GRPO avoids the single-mode collapse exhibited by GRPO and closely recovers the target mode allocation. Applied to text-to-image prompt adaptation, DARS-GRPO consistently improves image and prompt diversity across datasets, Policy LMs, and reward settings while maintaining good semantic alignment. These results show that density-aware reward scaling provides a lightweight way to improve coverage of diverse high-reward outcomes within existing reinforcement-learning pipelines.

## 1 Introduction

Reinforcement learning is typically formulated to maximize expected return, which can drive outcome-level probability mass toward a small subset of high-reward outcomes (GX-Chen et al. 2025). While desirable for conventional control problems, this behavior is limiting in domains where multiple distinct outcomes are simultaneously valuable. In generative modeling, scientific discovery, and structured prediction, the objective is often not merely to identify a high-reward solution, but to learn a policy that maintains broad coverage over the high-reward outcome space.

Several approaches have been proposed to promote such diversity. Entropy-regularized RL encourages stochastic action selection, but action-level stochasticity does not in general specify a desired distribution over terminal outcomes (Haarnoja et al. 2018). Generative Flow Networks (GFlowNets) instead target reward-proportional sampling, providing a principled framework for generating diverse high-reward samples (Bengio et al. 2023). Applying

GFlowNet-style objectives to autoregressive pretrained models generally requires formulating generation through flowor trajectory-based objectives rather than retaining an unchanged policy-gradient training pipeline (Yun et al. 2025). Inverse Probability Scaling (IPS) ofers another route by inversely scaling reward according to terminal-outcome visitation probability (Sinha, Elango, and Liu 2026). In continuous or high-dimensional outcome spaces, however, exact repeated outcomes may be rare or may not provide a meaningful notion of semantic equivalence. We introduce Density-Aware Reward Scaling (DARS), a simple framework for encouraging a policy that has reward-proportional behavior within standard policy-gradient reinforcement learning. DARS is motivated by the limitations of IPS and replaces discrete visitation frequency with an estimate of local outcome density in continuous outcome spaces. For each on-policy rollout group, outcomes that lie in densely populated regions receive proportionally reduced rewards, while high-reward outcomes in underrepresented regions receive greater relative rewards. This density scaling is applied before advantage estimation and requires no modification to the underlying policy-gradient objective. Importantly, DARS introduces no learned density model, critic, auxiliary loss, or specialized policy architecture; instead it simply modifies reward processing within the existing RL pipeline.

Our central contribution is a simple mechanism for mitigating outcome collapse and promoting coverage of diverse, high-reward regions within standard policy-gradient training. We validate DARS through controlled synthetic experiments that establish its ability to approximate reward-proportional sampling, followed by text-to-image prompt adaptation experiments demonstrating improved diversity while preserving generation quality, and kernel-width ablations that provide mechanistic evidence for its density-aware reward scaling.

## 2 Related Work

Mode collapse in RL and loss of diversity: Mode collapse is widely associated with generative models that represent only a small subset of a multimodal data distribution. An analogous failure occurs in reinforcement learning at the level of terminal outcomes: even when multiple trajectories or solutions receive similar rewards, expected-return optimization concentrates the learned policy on a narrow subset of them, reducing the diversity of generated solutions in language and image generation (Kirk et al. 2024; Chen et al. 2026). Existing approaches mitigate this behavior through entropy regularization, KL constraints, reward shaping, or diversity-aware objectives (Schulman et al. 2017; Liu et al. 2026), yet they do not directly control the distribution of terminal outcomes. Recent work instead attributes outcome collapse to expected-return optimization itself, where an outcome’s aggregate learning signal is amplified by its current sampling probability, creating a rich-getricher feedback loop (Sinha, Elango, and Liu 2026). Rewardproportional methods address this issue by sampling outcomes in proportion to their rewards: GFlowNets achieve this through specialized flow-conservation objectives and have been extended to multi-objective optimization, continuous state spaces, prompt adaptation, and language-model reasoning (Bengio et al. 2021, 2023; Jain et al. 2023; Lahlou et al. 2023; Zhu et al. 2026), while IPS instead rescales rewards using inverse visitation probabilities estimated from empirical outcome frequencies (Sinha, Elango, and Liu 2026). However, empirical visitation frequencies become inefective in continuous outcome spaces, where exact outcome repetitions are exceedingly rare. Concurrent work (Chen et al. 2026), also adjusts GRPO rewards using within-group semantic redundancy for mathematical reasoning in LLMs. In contrast, we study density-aware reward scaling as a mechanism for reward-proportional sampling in continuous outcome spaces and directly evaluate recovery of the target reward distribution.

Prompt Adaptation and Image Generation: Loss of output diversity has become increasingly visible in reinforcement learning from human feedback and other forms of large-language-model post-training (Ouyang et al. 2022; Kirk et al. 2024; Chen et al. 2026). Reward optimization can favor a narrow set ofresponse styles or semantic patterns even when many responses are valid. In practice, this behavior is often moderated with KL regularization, reward mixing, early stopping, or stochastic decoding. Prompt-optimization methods learn rewrites that improve text-to-image quality or alignment (Hao et al. 2023), while PAG learns diverse effective prompts with a GFlowNet formulation over promptgeneration trajectories (Yun et al. 2025). CADS acts at diffusion inference time by annealing the conditioning signal during sampling.

## 3 Density-Aware Reward Scaling (DARS)

IPS shows that reward-proportional sampling can be achieved by inversely scaling rewards according to outcome probabilities under the current policy (Sinha, Elango, and Liu 2026), but estimating these probabilities through exact visitation frequencies does not extend naturally to continuous outcome spaces, where repeated outcomes are unlikely. DARS instead represents outcomes in a continuous feature space where distances capture meaningful similarity between outcomes. This representation allows visitation to be defined over local regions rather than exact outcome identities. DARS estimates the local density of each sampled outcome within its onpolicy rollout group and inversely scales its reward by this density, reducing reinforcement of densely sampled regions while relatively favoring underrepresented high-reward regions.

## 3.1 Density

Let O denote the outcome space and let $\phi : \mathcal { O }  \mathbb { R } ^ { n }$ be a representation map that embeds outcomes into a feature space. For an outcome $o _ { i } \in \mathcal { O }$ , let $y _ { i } = \phi ( o _ { i } ) \in \mathbb { R } ^ { n }$ denote its representation. Given an on-policy rollout group containing $\dot { G }$ such $y _ { i }$ and their corresponding rewards $R _ { i } ~ \geq ~ 0 .$ DARS computes the relative local density of each outcome only with respect to the outcomes in the same rollout group. For a nonnegative kernel $K _ { h }$ , this estimate is

$$
\rho _ { i } = \sum _ { j = 1 } ^ { G } K _ { h } ( y _ { i } , y _ { j } ) ,\tag{1}
$$

where $h$ controls how strongly nearby outcomes contribute to the local density. One choice is the Gaussian kernel,

$$
K _ { h } ( y _ { i } , y _ { j } ) = \exp \left( - \frac { \| y _ { i } - y _ { j } \| _ { 2 } ^ { 2 } } { 2 h ^ { 2 } } \right) ,\tag{2}
$$

which assigns greater weight to nearby outcomes and smoothly decreases the contribution of outcomes as their distance increases. The kernel width h determines the scale of the resulting visitation neighborhoods. As the Gaussian kernel includes each outcome itself with $K _ { h } ( y _ { i } , y _ { i } ) = 1$ , the local density estimate satisfies $\rho _ { i } \geq 1$

DARS then applies inverse local-density reward scaling to down-weight frequently visited outcomes and encourage exploration of less populated regions of the outcome space, yielding the scaled reward ${ \widetilde { R } } _ { i }$

$$
\widetilde { R } _ { i } = \frac { R _ { i } } { \rho _ { i } } ,\tag{3}
$$

## 3.2 Integration with GRPO

DARS modifies only the reward-processing stage of the GRPO pipeline. GRPO uses a clipped policy-ratio objective, and estimates advantages from groups of sampled outputs (Shao et al. 2024). It replaces each task reward $R _ { i }$ with the density-scaled reward ${ \widetilde { R } } _ { i }$ from Equation (3), after which the group-relative advantage is computed as

$$
A _ { i } = \frac { \widetilde { R } _ { i } - \overline { { \widetilde { R } } } } { s _ { \widetilde { R } } } ,\tag{4}
$$

where $\overline { { \widetilde { R } } }$ and $s _ { \widetilde { R } }$ are the within-group mean and standard deviation of the scaled rewards. The resulting advantages are passed to the standard GRPO objective without further modification. For each conditioning input, DARS-GRPO samples an on-policy rollout group and evaluates its task rewards. It then computes the within-group local-density scores, scales and normalizes the rewards to obtain group-relative advantages, and applies the standard GRPO policy update without further modification. Thus, DARS requires no change to the GRPO policy ratio, clipped surrogate objective, policy architecture, or optimization procedure; its sole intervention is density-aware reward scaling before advantage normalization.

## 3.3 Behavior

At either kernel width extremes, as $h \to 0 , \rho _ { i } \approx 1$ and the rewards remain unchanged; as $h \to \infty , \rho _ { i } \approx G$ and grouprelative advantage normalization cancels the shared reward scale. Intermediate bandwidths instead produce non-uniform reward scaling across the given rollout group, thereby modifying the resulting policy-gradient signal.

For d-dimensional outcome representations, pairwise kernel evaluation costs $O ( G ^ { 2 } d )$ time and $O ( \stackrel { { \cdot } } { G } ^ { 2 } )$ memory. DARS introduces no learned density model, critic, or auxiliary optimization objective.

## 4 Navigation Task: Controlled Distribution-Recovery Experiment

This experiment tests whether DARS can make a policygradient RL method act as an approximate rewardproportional sampler in a continuous outcome space. The primary comparison is against GRPO: both methods use the same policy, environment, and terminal reward, but only DARS corrects for local outcome collapse. GFlowNet provides a reward-proportional reference with a diferent training objective. The experiment is a controlled empirical test of whether DARS approaches the normalized reward distribution.

Environment and Target: We evaluate how closely a policy approaches a known normalized reward density on the unit square $\mathcal { X } = [ 0 , 1 ] ^ { 2 }$ . Each episode starts at $s _ { 0 } = ( 0 , 0 )$ At state $s _ { t } .$ , the policy either terminates or takes a nonnegative displacement satisfying

$$
0 \leq a _ { t , i } \leq \operatorname* { m i n } ( \delta , 1 - s _ { t , i } ) , \qquad i \in \{ 1 , 2 \} ,
$$

with deterministic transition $s _ { t + 1 } = s _ { t } + a _ { t }$ . Trajectories therefore move monotonically up and to the right. The step bound δ makes distant terminal states require several decisions to reach for the policy. The policy may stop at any visited state except the origin, and it is forced to stop at the boundary or when the rollout horizon is reached. Only the terminal state receives rewards.

The movement policy uses polar coordinates. A neural network predicts a Bernoulli stopping probability and separate mixtures of Beta distributions for a normalized angle and radius. The angle is mapped to $\theta _ { t } \in [ 0 , \pi / 2 ]$ , and the radius is rescaled by both δ and the distance to the boundary. The resulting action is

$$
a _ { t } = ( r _ { t } \cos \theta _ { t } , r _ { t } \sin \theta _ { t } ) .
$$

This guarantees valid first-quadrant movements and lets the policy jointly learn where to move, how far to move, and when to terminate. Since the radial parameterization bounds the Euclidean step length, it imposes the slightly stricter constraint $\| a _ { t } \| _ { 2 } \leq \delta$

The reward is an equally weighted mixture offour isotropic Gaussians, as shown in Figure 1. Each component has standard deviation $\sigma = 0 . 0 7 5$

## 4.1 Baselines and Evaluation Metrics

We compare GRPO and DARS-GRPO, using GFlowNet as a non-RL reward-proportional reference. We evaluate the terminal distribution produced by each trained policy using $1 0 ^ { 6 }$ sampled terminal states per trained policy. Target samples are drawn from the Gaussian mixture and rejected when they fall outside the unit square. We use complementary metrics because no single score captures all forms of distributional mismatch in a multimodal setting.

Table 1: Navigation Task Results (avg over 5 seeds)
<table><tr><td>Metric</td><td>GFlowNet</td><td>GRPO</td><td>DARS-GRPO</td></tr><tr><td colspan="4">Density and transport</td></tr><tr><td>Mean raw reward ↑</td><td>3.44</td><td>6.96</td><td>4.59</td></tr><tr><td>JSD↓</td><td>0.020</td><td>0.649</td><td>0.041</td></tr><tr><td>Hist. TV ↓</td><td>0.149</td><td>0.975</td><td>0.219</td></tr><tr><td> $\mathrm { S W } _ { 2 } \downarrow$ </td><td>0.036</td><td>0.242</td><td>0.015</td></tr><tr><td>Energy ↓</td><td>0.007</td><td>0.281</td><td>0.002</td></tr><tr><td colspan="4">Mode allocation and support</td></tr><tr><td>Mode TV ↓</td><td>0.104</td><td>0.750</td><td>0.014</td></tr><tr><td>Min. mode mass  $\uparrow$ </td><td>0.187</td><td>0.000</td><td>0.236</td></tr><tr><td>Precision ↑</td><td>0.998</td><td>0.997</td><td>0.998</td></tr><tr><td>Recall ↑</td><td>0.996</td><td>0.032</td><td>0.985</td></tr><tr><td>Hierarchical  $W \downarrow$ </td><td>0.128</td><td>0.359</td><td>0.073</td></tr><tr><td colspan="4">Per-mode and local diagnostics (desired mass: 0.25)</td></tr><tr><td>Mode 1 mass</td><td>0.318</td><td>0.000</td><td>0.254</td></tr><tr><td>Mode 2 mass</td><td>0.187</td><td>0.000</td><td>0.252</td></tr><tr><td>Mode 3 mass</td><td>0.286</td><td>0.000</td><td>0.258</td></tr><tr><td>Mode 4 mass</td><td>0.209</td><td>1.000</td><td>0.236</td></tr><tr><td>Between-mode  $W \downarrow$ </td><td>0.113</td><td>0.335</td><td>0.046</td></tr><tr><td>Within-mode W ↓</td><td>0.016</td><td>0.023†</td><td>0.027</td></tr><tr><td>HDR outside-mass error ↓</td><td>0.001</td><td>0.050</td><td>0.044</td></tr><tr><td colspan="4">RBF MMD2 by bandwidth</td></tr><tr><td> $\sigma = 0 . 0 2 0 \downarrow$ </td><td>0.001</td><td>0.703</td><td>0.002</td></tr><tr><td> $\sigma = 0 . 0 4 0 \downarrow$ </td><td>0.004</td><td>0.832</td><td>0.004</td></tr><tr><td> $\sigma = 0 . 0 8 0 \downarrow$ </td><td>0.007</td><td>0.803</td><td>0.006</td></tr><tr><td> $\sigma = 0 . 1 6 0 \downarrow$ </td><td>0.011</td><td>0.690</td><td>0.002</td></tr><tr><td> $\sigma = 0 . 3 2 0 \downarrow$ </td><td>0.010</td><td>0.372</td><td>0.001</td></tr><tr><td colspan="4">Sinkhorn divergence by regularization</td></tr><tr><td> $\varepsilon = 0 . 0 0 1 \downarrow$ </td><td>0.008</td><td>0.115</td><td>0.002</td></tr><tr><td> $\varepsilon = 0 . 0 0 5 \downarrow$ </td><td>0.008</td><td>0.115</td><td>0.002</td></tr><tr><td> $\varepsilon = 0 . 0 1 0 \downarrow$ </td><td>0.008</td><td>0.114</td><td>0.001</td></tr><tr><td> $\varepsilon = 0 . 0 2 0 \downarrow$ </td><td>0.008</td><td>0.112</td><td>0.001</td></tr></table>

<sup>†</sup>For GRPO, the hierarchical within-mode contribution treats empty target modes separately; the Gaussian $W _ { 2 }$ of its populated mode is 0.094.

Mean raw reward is estimated as the empirical average of $r ( x )$ over sampled terminal states. Jensen–Shannon divergence (JSD) and histogram total variation (Hist. TV) compare the empirical and target densities on a 100 × 100 grid. Sliced Wasserstein ${ - 2 ( \mathrm { S W } _ { 2 } ) }$ (Kolouri et al. 2019), energy distance (Székely, Rizzo et al. 2013), and Sinkhorn divergence (Feydy et al. 2019) compare generated and target samples directly and measure geometric mismatch. RBF maximum mean discrepancy (MMD) (Gretton et al. 2012) is reported at several bandwidths: small bandwidths emphasize local component shape, whereas large bandwidths emphasize mode allocation and global structure. Sliced Wasserstein-2 and energy distance use ten repetitions of 20,000 samples, while Sinkhorn divergence uses 500 samples.

![](images/2c4f8cf7a4416db9ee8d6b0fd992f45c5886a1af15b5c9bf8490df2f9bb2fb34.jpg)  
Figure 1: Terminal distributions on the navigation task.

For mode-level evaluation, each terminal state is assigned to its nearest target center. From these assignments, we compute the mass of each mode, the minimum mode mass, and mode TV which measures deviation from uniform target mode weights. Precision is the fraction of generated samples lying inside k-nearest-neighbor neighborhoods of target samples, while recall is the fraction of target samples covered by neighborhoods induced by generated samples (Kynkäänniemi et al. 2019). Precision therefore measures of-target generation, whereas recall diagnoses missing support and mode collapse.

We measure Wasserstein error at two scales: the betweenmode term evaluates mode weights and centers, while the within-mode term evaluates each mode’s mean and covariance. Their sum defines the hierarchical score,

$$
W _ { \mathrm { h i e r } } = W _ { \mathrm { b e t w e e n } } + W _ { \mathrm { w i t h i n } } .
$$

HDR outside mass is the fraction of generated samples outside the target’s 95% highest-density region. Since the target assigns approximately 0.05 probability mass outside this region, we report the calibration error $| m _ { \mathrm { o u t s i d e } } - 0 . 0 5 | ,$ for which lower values indicate better agreement with the target.

## 4.2 Results and Interpretation

The primary result is the comparison with GRPO. In the reported run, GRPO concentrates on a single high-reward mode, whereas DARS-GRPO assigns substantial mass to all four modes and approaches their target weights. The DARS-GRPO point estimates are closer to the target than GRPO across the metrics reported in Table 1. GRPO’s higher mean raw reward reflects tight concentration near one of four equally high peaks, rather than agreement with the normalized reward distribution. Thus, within this controlled environment, the reported run provides evidence that density reweighting prevents the evaluated policy from concentrating on one mode, recovers the correct mode weights, and improves global geometric agreement with the known target relative to GRPO. This is the central controlled contribution of the experiment.

DARS-GRPO closely matches, and on several key metrics surpasses, GFlowNet despite requiring only a simple reward-scaling modification within a standard policygradient pipeline. Both methods achieve identical precision, indicating equally accurate recovery of high-reward outcomes. DARS consistently improves metrics that capture the global reward distribution, including sliced Wasserstein, energy distance, mode total variation, minimum mode mass, hierarchical Wasserstein error, HDR outside mass, and most larger-bandwidth MMD and Sinkhorn metrics, demonstrating more faithful recovery of target mode proportions and global geometry.

GFlowNet performs slightly better on metrics emphasizing fine-scale local density, such as Jensen–Shannon divergence, histogram total variation, within-mode Wasserstein error, recall, and small-bandwidth MMD, reflecting its specialized flow-based objective. Nevertheless, these results show that a lightweight modification to conventional policygradient RL is suficient to achieve reward-proportional behavior that is competitive with a substantially more sophisticated algorithmic framework.

Having established in a controlled setting that DARS approximates reward-proportional sampling, we next ask whether this property is useful in a real continuous-output RL problem where the target reward distribution is unknown.

## 5 Text-to-Image Prompt Adaptation 5.1 Motivation

Prompt adaptation for image generation refers to the setting in which a language model rewrites the input prompt to match a desired set of outcomes before image generation. Text-to-image prompt adaptation provides a natural real-world benchmark for evaluating reward-proportional reinforcement learning. Unlike the synthetic experiments in Section 4, where the target reward distribution is explicitly known, the reward-proportional distribution over generated images is intractable and cannot be directly characterized. A single text prompt typically admits many semantically faithful realizations that difer in composition, viewpoint, lighting, object arrangement, or artistic style while remaining equally valid.

Since the underlying reward distribution is unavailable, directly measuring reward-proportional sampling is infeasible. Instead, we evaluate its consequences through observable proxies. If DARS successfully distributes probability mass across multiple high-reward regions, it should produce a more diverse set of images while preserving prompt alignment and image quality. We therefore measure visual diversity together with alignment and preference metrics as an indirect estimate of improved coverage of the latent reward landscape. Consequently, the objective is not merely to identify a single high-reward image, but to maintain coverage over the diverse set of high-quality outcomes consistent with the prompt.

Since input prompts are the conditioning signal in difusion models, semantically faithful prompt rewrites help steer the generation in desired directions. Learning a distribution over such rewrites therefore provides an efective mechanism for exploring multiple high-reward image modes while keeping the underlying image generator fixed.

Following this formulation, we fine-tune a language model using reinforcement learning to rewrite input prompts. Images generated from the rewritten prompts are rewarded as defined in section 5.2. Since multiple visually distinct images can achieve comparable rewards, broader coverage of the latent reward landscape should translate to greater image diversity without sacrificing fidelity. To quantify this diversity, we measure distances in the DINOv2 representation space as a proxy for perceptual similarity (Oquab et al. 2023).

![](images/2d591ae8332302c368543fab8e4b6f64b1f676adf9bed2b7e9a8e625f396d82c.jpg)  
Figure 2: Overview of DARS-GRPO for text-to-image prompt adaptation. A trainable language-model policy samples G rewrites of an input prompt, and a frozen text-to-image model generates one image per rewrite. Each image receives a task reward $r _ { i }$ and is mapped by a frozen image encoder into a representation space, where a Gaussian kernel computes its within-group density $\rho _ { i }$ . DARS forms the scaled reward $\tilde { r } _ { i } = r _ { i } / \rho _ { i }$ , reducing the contribution of high-reward images in dense representation-space regions and increasing the relative contribution of high-reward images in sparsely represented regions. GRPO then uses these scaled rewards to update only the prompt-rewriting policy.

## 5.2 Reward Definitions

The reward should preserve two complementary properties: semantic fidelity to the original prompt and visual appeal. Fidelity alone does not ensure that an image is aesthetically strong, whereas optimizing aesthetics alone can favor attractive images that drift from the requested content. We therefore evaluate on two diferent reward functions: alignment-only reward and a combined reward consisting of aesthetic and alignment components.

Let c denote the original prompt and $y _ { i }$ an image generated from its ith rewrite. We measure semantic alignment using the image-text cosine similarity from a CLIP model (Radford et al. 2021). A binary semantic gate determines whether the image is suficiently aligned with the original prompt. Thus the alignment reward is

$$
R _ { i } ^ { \mathrm { a l i g n } } = \mathbb { I } [ \mathrm { C L I P } ( c , y _ { i } ) > \phi ] ,\tag{5}
$$

where $\phi$ is the minimum accepted CLIP similarity. Images below this threshold receive zero reward to encourage strict adherence. This setting tests whether DARS can broaden coverage within the set of suficiently aligned images.

To measure visual appeal, we use LAION-Aesthetics Predictor (Schuhmann 2022), which maps images to an aesthetic score on a 1–10 scale. Let $a ( \cdot )$ denote this predictor, and let $y _ { c } ^ { \mathrm { r e f } }$ be an image generated from the original input prompt. We define the aesthetic improvement and its multiplicative reward factor as

$$
\Delta a _ { i } = a ( y _ { i } ) - a \left( y _ { c } ^ { \mathrm { r e f } } \right) ,\tag{6}
$$

$$
R _ { i } ^ { \mathrm { a e s } } = \exp ( \Delta a _ { i } ) .\tag{7}
$$

An aesthetic improvement produces a factor greater than one, no change produces a factor of one, and a decrease produces a factor between zero and one.

Thus, the combined reward is given as

$$
{ R } _ { i } ^ { \mathrm { c o m b } } = \exp ( \Delta a _ { i } ) \mathrm { C L I P } ( c , y _ { i } ) \mathbb { I } [ \mathrm { C L I P } ( c , y _ { i } ) > \phi ]\tag{8}
$$

## 5.3 Density Calculation and Reward Scaling

DARS estimates local density in the space of generated images, since our objective is to promote diversity over visual outcomes rather than textual rewrites. As discussed in Section 3.1, we instantiate ϕ using a frozen DINOv2 encoder, which maps each generated image i to its unnormalized class-token (CLS) embedding $y _ { i }$ . Local densities are then estimated in this representation space.

For a group of G images, let $\bar { y _ { i } } \in \mathbb { R } ^ { 7 6 8 }$ denote the feature vector of image i. We compute the local density score for image i using a Gaussian kernel as

![](images/8c70a680723073bee9ea2c59407068a1c95698dda81607c299d23fc95ac33277.jpg)  
Figure 3: Comparison of prompt rewrites and their corresponding generated images $\mathrm { f o r } ^ { 6 } \mathrm { \bar { A } }$ snowman standing in the yard” using GRPO, PAG, and DARS-GRPO. GRPO produces nearly identical rewrites and visually similar nighttime scenes, whereas PAG introduces moderate variation and DARS-GRPO generates more diverse rewrites, settings, and visual compositions.

$$
\rho _ { i } = \sum _ { j = 1 } ^ { G } K _ { h } ( y _ { i } , y _ { j } ) = \sum _ { j = 1 } ^ { G } \exp \left( - \frac { \| y _ { i } - y _ { j } \| _ { 2 } ^ { 2 } } { 2 h ^ { 2 } } \right) .\tag{9}
$$

Following Equation (3), DARS rescales the task reward as $\widetilde { R } _ { i } ~ = ~ R _ { i } / \rho _ { i }$ . We use this scaled reward to optimize the prompt-rewriting policy with DARS-GRPO.

## 5.4 Evaluation Metrics

We evaluate each prompt-rewriting policy along three complementary dimensions: image diversity, rewrite diversity, and image fidelity or preference. Image diversity is measured in DINOv2 CLS representation space using the Vendi Score (Friedman and Dieng 2023) of the cosine-similarity matrix of L2-normalized CLS embeddings, and the mean pairwise Euclidean distance between CLS embeddings. Rewrite diversity is measured analogously in CLIP text-embedding space using Prompt Vendi and the mean pairwise cosine distance among rewritten prompts. Finally, CLIP-align, HPSv2 (Wu et al. 2023), and PickScore (Kirstain et al. 2023) assess the semantic compatibility and human-preference-style quality of each generated image relative to the input prompt. CLIPalign is the nonnegative cosine similarity between the L2- normalized CLIP embeddings of a generated image and its original prompt. Diversity metrics are computed within each group of rewrites or images generated for the same original prompt and then averaged across the evaluation set, following within-prompt conditional evaluation. For all metrics, higher values indicate greater diversity or stronger fidelity and preference, respectively.

## 5.5 Experiment Details

We use a frozen Stable Difusion v1.4 generator (Rombach et al. 2022) and optimize only the prompt-rewriting policy. For all baselines and reward functions, both Qwen3- 0.6B(base) (Yang et al. 2025) and GPT-2 (124M) (Radford et al. 2019) are already supervised fine-tuned on a shared set ofcurated prompt rewrites. We then train the models on 1,024 randomly sampled MS-COCO prompts (Lin et al. 2014) and evaluate them on disjoint MS-COCO and Lexica prompts (Shen et al. 2023). DARS uses h = 20 for Gaussian kernels, and the alignment reward uses a CLIP-similarity threshold of 0.2.

We compare DARS-GRPO with vanilla GRPO, PAG, and GFlowNet prompt optimization, together with Stable Difusion v1.4 with random seed initialization and CADS (Sadat et al. 2024) as control baselines. All methods are evaluated using the same held-out prompts, generation budget, difusion model, and sampling procedure. Difusion seeds are sampled independently across methods, so image-diversity metrics also capture variation introduced by stochastic difusion sampling while prompt-diversity metrics are unafected by this confound.

## 5.6 Results and Interpretation

Across both datasets, Policy LMs, and reward settings, DARS-GRPO consistently improves upon vanilla GRPO across all three diversity metrics (Table 2) without significant loss of image fidelity. Further, among all prompt-adaptation methods, it achieves the highest DINO pairwise distances and Image Vendi scores, indicating that density-aware reward scaling broadens the learned rewrite distribution and increases coverage in DINOv2 image-representation space. The high Prompt Vendi and diversity scores further indicate that this image diversity arises from diverse prompt rewrites, while CLIP-align, HPSv2, and PickScore imply close image fidelity.

Changing the reward function afects performance across all metrics, showing that DARS responds to the reward landscape rather than producing diversity gains randomly. Consistent with our hypothesis, the binary gated alignment-only reward yields broader sampling from uniformly scored reward zones, whereas the combined reward forces the policy to learn a more dynamic reward landscape characterized by aesthetically-driven peaks, thus, sampling from a more varied distribution as reflected by the higher average Image Vendi scores.

Having demonstrated the efectiveness of DARS on a realworld RL problem with an intractable reward-proportional distribution, we next study the efect of the kernel width used for local density estimation on the learned outcome distribution.

Table 2: Prompt Adaptation for Text-to-Image Generation results on MS-COCO and Lexica (avg over 5 seeds).
<table><tr><td></td><td></td><td colspan="2">Image Vendi ↑</td><td colspan="2">CLIP-align ↑</td><td colspan="2">HPSv2 ↑</td><td colspan="2">PickScore ↑</td><td colspan="2">Prompt Vendi ↑</td><td colspan="2">Prompt diversity ↑</td><td colspan="2">DINO pairwise distance ↑</td></tr><tr><td>Policy LM</td><td>Method</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td><td>COCO</td><td>Lexica</td></tr><tr><td></td><td>SD v1.4</td><td>8.60</td><td>9.52</td><td>0.27</td><td>0.27</td><td>0.26</td><td>0.25</td><td>21.53</td><td>20.91</td><td>1</td><td>一</td><td>-</td><td>一</td><td>40.58</td><td>41.84</td></tr><tr><td>一</td><td>CADS</td><td>12.93</td><td>14.08</td><td>0.26</td><td>0.26</td><td>0.24</td><td>0.23</td><td>20.86</td><td>20.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>46.22</td><td>46.96</td></tr><tr><td colspan="10">Combined reward</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-0.6B</td><td>GRPO</td><td>8.76</td><td>9.81</td><td>0.24</td><td>0.24</td><td>0.23</td><td>0.23</td><td>20.92</td><td>20.69</td><td>1.01</td><td>1.04</td><td>0.00</td><td>0.01</td><td>40.90</td><td>42.10</td></tr><tr><td>Qwen3-0.6B</td><td>DARS-GRPO</td><td>13.04</td><td>14.45</td><td>0.24</td><td>0.24</td><td>0.23</td><td>0.23</td><td>20.80</td><td>20.37</td><td>1.43</td><td>1.67</td><td>0.05</td><td>0.06</td><td>46.92</td><td>47.89</td></tr><tr><td>Qwen3-0.6B</td><td>PAG</td><td>8.25</td><td>11.51</td><td>0.26</td><td>0.26</td><td>0.25</td><td>0.24</td><td>21.34</td><td>20.69</td><td>1.26</td><td>1.59</td><td>0.04</td><td>0.07</td><td>40.20</td><td>44.64</td></tr><tr><td>Qwen3-0.6B</td><td>GFN</td><td>8.74</td><td>11.69</td><td>0.26</td><td>0.26</td><td>0.25</td><td>0.24</td><td>21.31</td><td>20.69</td><td>1.22</td><td>1.59</td><td>0.04</td><td>0.07</td><td>40.50</td><td>44.71</td></tr><tr><td>GPT-2</td><td>GRPO</td><td>9.60</td><td>10.68</td><td>0.24</td><td>0.24</td><td>0.23</td><td>0.23</td><td>20.87</td><td>20.43</td><td>1.09</td><td>1.18</td><td>0.01</td><td>0.03</td><td>41.85</td><td>42.87</td></tr><tr><td>GPT-2</td><td>DARS-GRPO</td><td>12.72</td><td>14.67</td><td>0.23</td><td>0.23</td><td>0.23</td><td>0.23</td><td>20.68</td><td>20.26</td><td>1.63</td><td>1.78</td><td>0.08</td><td>0.10</td><td>47.75</td><td>48.60</td></tr><tr><td>GPT-2</td><td>PAG</td><td>9.22</td><td>14.03</td><td>0.26</td><td>0.24</td><td>0.25</td><td>0.23</td><td>21.16</td><td>20.35</td><td>1.61</td><td>2.14</td><td>0.09</td><td>0.13</td><td>41.36</td><td>46.43</td></tr><tr><td>GPT-2</td><td>GFN</td><td>9.02</td><td>13.67</td><td>0.25</td><td>0.24</td><td>0.24</td><td>0.22</td><td>21.12</td><td>20.26</td><td>1.15</td><td>1.69</td><td>0.02</td><td>0.08</td><td>41.03</td><td>46.67</td></tr><tr><td colspan="10">Alignment-only reward</td><td colspan="3"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>Qwen3-0.6B</td><td>GRPO</td><td>9.47</td><td>10.94</td><td>0.26</td><td>0.26</td><td>0.25</td><td>0.24</td><td>21.24</td><td>20.55</td><td>1.59</td><td>1.64</td><td>0.07</td><td>0.08</td><td>41.68</td><td>43.52</td></tr><tr><td>Qwen3-0.6B</td><td>DARS-GRPO</td><td>20.13</td><td>20.03</td><td>0.22</td><td>0.21</td><td>0.22</td><td>0.22</td><td>20.34</td><td>19.81</td><td>2.43</td><td>2.40</td><td>0.16</td><td>0.16</td><td>56.23</td><td>55.76</td></tr><tr><td>Qwen3-0.6B</td><td>PAG</td><td>9.53</td><td>13.88</td><td>0.25</td><td>0.24</td><td>0.25</td><td>0.24</td><td>21.22</td><td>20.48</td><td>1.79</td><td>2.33</td><td>0.11</td><td>0.15</td><td>41.99</td><td>46.86</td></tr><tr><td>Qwen3-0.6B</td><td>GFN</td><td>8.93</td><td>12.13</td><td>0.26</td><td>0.26</td><td>0.25</td><td>0.24</td><td>21.27</td><td>20.60</td><td>1.30</td><td>1.62</td><td>0.05</td><td>0.08</td><td>40.99</td><td>44.97</td></tr><tr><td>GPT-2</td><td>GRPO</td><td>8.76</td><td>10.22</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.24</td><td>21.42</td><td>20.66</td><td>1.03</td><td>1.11</td><td>0.01</td><td>0.02</td><td>40.93</td><td>42.52</td></tr><tr><td>GPT-2</td><td>DARS-GRPO</td><td>22.87</td><td>22.64</td><td>0.23</td><td>0.22</td><td>0.23</td><td>0.22</td><td>20.58</td><td>19.89</td><td>2.15</td><td>2.29</td><td>0.14</td><td>0.15</td><td>55.57</td><td>55.00</td></tr><tr><td>GPT-2</td><td>PAG</td><td>8.51</td><td>13.15</td><td>0.25</td><td>0.24</td><td>0.25</td><td>0.23</td><td>21.23</td><td>20.38</td><td>1.20</td><td>1.66</td><td>0.03</td><td>0.07</td><td>40.55</td><td>45.78</td></tr><tr><td>GPT-2</td><td>GFN</td><td>8.60</td><td>14.08</td><td>0.26</td><td>0.24</td><td>0.24</td><td>0.22</td><td>21.19</td><td>20.33</td><td>1.17</td><td>1.79</td><td>0.03</td><td>0.08</td><td>40.54</td><td>46.03</td></tr></table>

## 6 Ablation Study

As discussed in 3.3, kernel width h afects reward-scaling in DARS and hence needs to be calibrated for accurate density estimation. For the experiments in Section 5, we demonstrate the direct efects of varying kernel widths on image-diversity produced by DARS (Table 3). Diversity metrics consistently peak at intermediate kernel widths and degrade when the bandwidth is either too small or too large, while image quality remains largely unchanged. This behavior reflects how DARS estimates local outcome density. With a very small kernel width, nearly every generated image appears isolated, causing density estimates to be dominated by self-similarity and resulting in almost uniform reward scaling. Conversely, an excessively large kernel width treats distant images as belonging to the same neighborhood, oversmoothing density estimates and diminishing the relative reward diferences that encourage exploration. Intermediate kernel widths instead resolve meaningful local neighborhoods allowing for efective density estimation in DARS.

## 7 Limitations

DARS relies on a meaningful representation space and kernel bandwidth for estimating local outcome density. Its effectiveness may therefore degrade when embedding distances poorly reflect semantic similarity or when no single bandwidth captures the relevant neighborhood structure. The pairwise density computation also scales quadratically with rollout-group size, potentially limiting its use with large groups. Our experiments focus on a controlled navigation task and text-to-image prompt adaptation with fixed image representations and generators. Broader evaluation across RL domains, policy architectures, and learned representations is needed. Finally, our results establish empirical approximation to reward-proportional sampling rather than a general convergence guarantee.

Table 3: Kernel-width ablation results (avg over 5 seeds)
<table><tr><td>Width</td><td>Image Vendi ↑</td><td>CLIP-align ↑</td><td>Prompt Diversity ↑</td><td>Prompt Vendi ↑</td></tr><tr><td>10</td><td>9.99</td><td>0.26</td><td>0.12</td><td>1.99</td></tr><tr><td>15</td><td>14.12</td><td>0.25</td><td>0.21</td><td>3.34</td></tr><tr><td>20</td><td>18.01</td><td>0.24</td><td>0.17</td><td>2.74</td></tr><tr><td>25</td><td>19.77</td><td>0.23</td><td>0.22</td><td>3.47</td></tr><tr><td>30</td><td>12.14</td><td>0.25</td><td>0.11</td><td>2.09</td></tr></table>

## 8 Conclusion

We introduced Density-Aware Reward Scaling (DARS), a lightweight mechanism for promoting diverse, high-reward outcomes within standard policy-gradient training. By inversely scaling rewards using local outcome density, DARS counteracts repeated reinforcement of densely sampled regions without requiring a critic, auxiliary objective, learned density model, or modified policy-gradient loss. In a controlled multimodal navigation task, DARS-GRPO prevents the mode collapse observed with GRPO and closely recovers target mode allocation. In text-to-image prompt adaptation, it consistently increases prompt and image diversity across datasets, Policy LMs, and reward settings while retaining semantic fidelity. These results position density-aware reward scaling as a simple yet efective approach to improving outcome coverage.

## References

Bengio, E.; Jain, M.; Korablyov, M.; Precup, D.; and Bengio, Y. 2021. Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. In Advances in Neural Information Processing Systems, volume 34, 27381– 27394.

Bengio, Y.; Lahlou, S.; Deleu, T.; Hu, E. J.; Tiwari, M.; and Bengio, E. 2023. GFlowNet Foundations. Journal of Machine Learning Research, 24(210): 1–55.

Chen, Y.; Chakraborty, S.; Wolf, L.; Paschalidis, I. C.; and Pacchiano, A. 2026. Post-training Large Language Models for Diverse High-Quality Responses. In International Conference on Learning Representations.

Feydy, J.; Séjourné, T.; Vialard, F.-X.; Amari, S.-i.; Trouvé, A.; and Peyré, G. 2019. Interpolating between Optimal Transport and MMD using Sinkhorn Divergences. In Proceedings ofthe Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings ofMachine Learning Research, 2681–2690.

Friedman, D.; and Dieng, A. B. 2023. The Vendi Score: A Diversity Evaluation Metric for Machine Learning. Transactions on Machine Learning Research.

Gretton, A.; Borgwardt, K. M.; Rasch, M. J.; Schölkopf, B.; and Smola, A. 2012. A Kernel Two-Sample Test. Journal of Machine Learning Research, 13(25): 723–773.

GX-Chen, A.; Prakash, J.; Guo, J.; Fergus, R.; and Ranganath, R. 2025. KL-Regularized Reinforcement Learning is Designed to Mode Collapse. arXiv preprint arXiv:2510.20817.

Haarnoja, T.; Zhou, A.; Abbeel, P.; and Levine, S. 2018. Soft Actor-Critic: Of-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, 1861–1870.

Hao, Y.; Chi, Z.; Dong, L.; and Wei, F. 2023. Optimizing Prompts for Text-to-Image Generation. In Advances in Neural Information Processing Systems, volume 36, 66923– 66939.

Jain, M.; Raparthy, S. C.; Hernández-García, A.; Rector-Brooks, J.; Bengio, Y.; Miret, S.; and Bengio, E. 2023. Multi-Objective GFlowNets. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, 14631–14653.

Kirk, R.; Mediratta, I.; Nalmpantis, C.; Luketina, J.; Hambro, E.; Grefenstette, E.; and Raileanu, R. 2024. Understanding the Efects of RLHF on LLM Generalisation and Diversity. In International Conference on Learning Representations.

Kirstain, Y.; Polyak, A.; Singer, U.; Matiana, S.; Penna, J.; and Levy, O. 2023. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. arXiv preprint arXiv:2305.01569.

Kolouri, S.; Nadjahi, K.; Simsekli, U.; Badeau, R.; and Rohde, G. 2019. Generalized sliced wasserstein distances. Advances in neural information processing systems, 32.

Kynkäänniemi, T.; Karras, T.; Laine, S.; Lehtinen, J.; and Aila, T. 2019. Improved Precision and Recall Metric for Assessing Generative Models. In Advances in Neural Information Processing Systems, volume 32, 3927–3936.

Lahlou, S.; Deleu, T.; Lemos, P.; Zhang, D.; Volokhova, A.; Hernández-García, A.; Ezzine, L. N.; Bengio, Y.; and Malkin, N. 2023. A Theory of Continuous Generative Flow Networks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 18269–18300.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft COCO: Common Objects in Context. In European Conference on Computer Vision, 740–755. Springer.

Liu, H.; Huang, H.; Wang, J.; Liu, C.; Li, X.; and Ji, X. 2026. DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1864–1873.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2023. DINOv2: Learning Robust Visual Features without Supervision. arXiv preprint arXiv:2304.07193.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems, volume 35, 27730–27744.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8748–8763.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10684–10695.

Sadat, S.; Buhmann, J.; Bradley, D.; Hilliges, O.; and Weber, R. M. 2024. CADS: Unleashing the Diversity of Difusion Models through Condition-Annealed Sampling. In International Conference on Learning Representations.

Schuhmann, C. 2022. LAION-Aesthetics. https://laion.ai/ blog/laion-aesthetics/. Accessed: 2026-07-29.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D.

2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

Shen, X.; Qu, Y.; Backes, M.; and Zhang, Y. 2023. Prompt Stealing Attacks Against Text-to-Image Generation Models. CISPA Helmholtz Center for Information Security.

Sinha, A.; Elango, S.; and Liu, D. 2026. Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling. arXiv preprint arXiv:2601.21669.

Székely, G. J.; Rizzo, M. L.; et al. 2013. Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference, 143(8): 1249–1272.

Wu, X.; Hao, Y.; Sun, K.; Chen, Y.; Zhu, F.; Zhao, R.; and Li, H. 2023. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yun, T.; Zhang, D.; Park, J.; and Pan, L. 2025. Learning to Sample Efective and Diverse Prompts for Text-to-Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23625–23635.

Zhu, X.; Cheng, D.; Zhang, D.; Li, H.; Zhang, K.; Jiang, C.; Sun, Y.; Hua, E.; Zuo, Y.; Lv, X.; Zhang, Q.; Chen, L.; Shao, F.; Xue, B.; Song, Y.; Yang, Z.; Cui, G.; Ding, N.; Gao, J.; Liu, X.; Zhou, B.; Mei, H.; and Lin, Z. 2026. FlowRL: Matching Reward Distributions for LLM Reasoning. In International Conference on Learning Representations.

## A Mathematical Foundations for DARS A.1 Inverse policy-density scaling yields a reward-proportional policy

Let the continuous outcome space be $\mathcal { V } \subseteq \mathbb { R } ^ { d }$ . At training time t, let the policy induce a probability density $p _ { t } ( y )$ on $\mathcal { V } .$ Let the reward be a measurable function $r ( y ) \geq 0$ satisfying

$$
0 < \mathcal { R } : = \int _ { \mathcal { V } } r ( y ) d y < \infty ,\tag{10}
$$

and assume $p _ { t } ( y ) > 0$ wherever $r ( y ) > 0$ . The desired reward-proportional density is

$$
p ^ { \star } ( y ) = \frac { r ( y ) } { \mathcal { R } } .\tag{11}
$$

We prove the continuous analogue of inverse probability scaling. Parameterize an unrestricted policy density by a logit function $f ,$

$$
p _ { f } ( y ) = \frac { e ^ { f ( y ) } } { \int _ { y } e ^ { f ( u ) } d u } .\tag{12}
$$

At iteration $t ,$ freeze the current policy density in the reward denominator and use

$$
\widetilde r _ { t } ( y ) = { \frac { r ( y ) } { p _ { t } ( y ) } } .\tag{13}
$$

For the population objective

$$
J _ { t } ( f ) = \int _ { y } p _ { f } ( y ) \widetilde { r } _ { t } ( y ) d y ,\tag{14}
$$

the following result holds.

Theorem 1 At the current policy $p _ { t } = p _ { f _ { t } }$ , the functional policy-gradient direction is

$$
\Bigg | \frac { \delta J _ { t } } { \delta f } ( y ) \Bigg | _ { f = f _ { t } } = r ( y ) - \mathcal { R } p _ { t } ( y ) = \mathcal { R } \big ( p ^ { \star } ( y ) - p _ { t } ( y ) \big ) .\tag{15}
$$

Hence the reward-proportional density $p ^ { \star } ( y ) = r ( y ) / \mathcal { R }$ is the unique stationary probability density, interpreted in the closure of the exponential-family parameterization when r vanishes on part of the outcome space. Under the corresponding population gradient flow, every convergent density limit is $p ^ { \star }$

Proof 1 For any perturbation v, the normalized exponential density obeys

$$
\left. \frac { d } { d s } p _ { f _ { t } + s v } ( y ) \right| _ { s = 0 } = p _ { t } ( y ) \left( v ( y ) - \mathbb { E } _ { p _ { t } } [ v ] \right) .\tag{16}
$$

Because the denominator in Equation (13) is stop-gradient,

$$
{ \frac { d } { d s } } J _ { t } ( f _ { t } + s v ) { \bigg | } _ { s = 0 } = \int _ { \mathcal { V } } { \frac { r ( y ) } { p _ { t } ( y ) } } p _ { t } ( y )\tag{17}
$$

$$
\mathbf { \nabla } \times \left( v ( y ) - \mathbb { E } _ { p _ { t } } [ v ] \right) d y\tag{18}
$$

$$
= \int _ { y } r ( y ) v ( y ) d y\tag{19}
$$

$$
- \mathcal { R } \int _ { \mathcal { V } } p _ { t } ( y ) v ( y ) d y\tag{20}
$$

$$
= \int _ { \mathcal { V } } ( r ( y ) - \mathcal { R } p _ { t } ( y ) ) v ( y ) d y .\tag{21}
$$

Since this identity holds for every perturbation v, Equation (15) follows. A stationary density must satisfy

$$
r ( y ) - \mathcal { R } p _ { t } ( y ) = 0\tag{22}
$$

almost everywhere, and therefore

$$
p _ { t } ( y ) = \frac { r ( y ) } { \mathcal R } = p ^ { \star } ( y ) .\tag{23}
$$

The direction is not merely stationary at $p ^ { \star } ,$ ; it points toward it. Consider the population gradient flow

$$
\partial _ { t } f _ { t } ( y ) = \mathcal { R } \big ( p ^ { \star } ( y ) - p _ { t } ( y ) \big )\tag{24}
$$

and thefunctional

$$
\Phi ( f ) = \log \int _ { \mathcal { V } } e ^ { f ( y ) } d y - \int _ { \mathcal { V } } p ^ { \star } ( y ) f ( y ) d y .\tag{25}
$$

Its functional derivative is $\delta \Phi / \delta f = p _ { f } - p ^ { \star }$ . Therefore, along the gradient flow,

$$
\frac { d } { d t } \Phi ( f _ { t } ) = \int _ { \mathcal { V } } \bigl ( p _ { t } ( y ) - p ^ { \star } ( y ) \bigr ) \partial _ { t } f _ { t } ( y ) d y\tag{26}
$$

$$
= - \mathcal { R } \int _ { \mathcal { V } } \bigl ( p _ { t } ( y ) - p ^ { \star } ( y ) \bigr ) ^ { 2 } d y \leq 0 .\tag{27}
$$

Equality holds only when $p _ { t } ~ = ~ p ^ { \star }$ almost everywhere. Thus the only stationary density, and hence any convergent population-gradient-flow density limit, is the rewardproportional density in Equation (11).

The cancellation underlying the result can also be seen directly. A small region $d y$ around y is sampled with probability approximately $p _ { t } ( y ) d y$ . Its per-sample scaled reward is $r ( y ) / p _ { t } ( y )$ . Thus its aggregate expected learning weight is proportional to

$$
\underbrace { p _ { t } ( y ) d y } _ { \mathrm { s a m p l i n g ~ p r o b a b i l i t y } } \underbrace { \frac { r ( y ) } { p _ { t } ( y ) } } _ { \mathrm { i n v e r s e - d e n s i t y ~ r e w a r d } } = r ( y ) d y .\tag{28}
$$

The current policy density cancels, leaving learning mass proportional only to reward mass.

## A.2 The Gaussian kernel density score is proportional to the policy density

Draw a rollout group of G outcomes independently from the current policy,

$$
Y _ { 1 } , \dots , Y _ { G } \overset { \mathrm { i i d } } { \sim } p _ { t } .\tag{29}
$$

DARS uses the unnormalized Gaussian kernel

$$
K _ { h } ( y , z ) = \exp \left( { - \frac { \| y - z \| ^ { 2 } } { 2 h ^ { 2 } } } \right) .\tag{30}
$$

Instead of separating any sample from the rest of the group, define the density-score field produced by the entire rollout group,

$$
\rho _ { G } ( y ) = \sum _ { j = 1 } ^ { G } K _ { h } ( y , Y _ { j } ) .\tag{31}
$$

The score assigned to a sampled outcome is simply

$$
\rho _ { i } = \rho _ { G } ( Y _ { i } ) .\tag{32}
$$

Thus the same group of all G policy samples is used throughout.

Define

$$
C _ { h } = ( 2 \pi ) ^ { d / 2 } h ^ { d } , \qquad \phi _ { h } ( z ) = \frac { 1 } { C _ { h } } \exp \left( - \frac { | | z | | ^ { 2 } } { 2 h ^ { 2 } } \right)\tag{33}
$$

Then $\phi _ { h }$ is the $\mathcal { N } ( 0 , h ^ { 2 } I _ { d } )$ density and

$$
\widehat { p } _ { G , h } ( y ) : = \frac { \rho _ { G } ( y ) } { G C _ { h } } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \phi _ { h } ( y - Y _ { j } ) .\tag{34}
$$

Equation (34) is exactly the Gaussian kernel-density estimator formed from the full rollout group.

Theorem 2 (Gaussian density approximation) For every fixed y,

$$
\begin{array} { r } { \boxed { \mathbb { E } [ \widehat { p } _ { G , h } ( y ) ] = ( p _ { t } * \phi _ { h } ) ( y ) , } } \end{array}\tag{35}
$$

where

$$
( p _ { t } * \phi _ { h } ) ( y ) = \int _ { \mathcal { V } } p _ { t } ( u ) \phi _ { h } ( y - u ) d u .\tag{36}
$$

Assume that, on the region of interest, $p _ { t }$ is twice continuously diferentiable,

$$
\operatorname* { s u p } _ { y } \| \nabla ^ { 2 } p _ { t } ( y ) \| _ { \mathrm { o p } } \leq M ,\tag{37}
$$

and $p _ { t } ( y ) \leq P$ . Ifthe region has a boundary, assume either that p<sub>t</sub> admits a $C ^ { 2 }$ extension to a neighborhood of that region or that the statement is restricted to interior points at the bandwidth scale. Then

$$
\boxed { | \mathbb { E } [ \widehat { p } _ { G , h } ( y ) ] - p _ { t } ( y ) | \le \frac { d M } { 2 } h ^ { 2 } , }\tag{38}
$$

and

$$
\boxed { \mathrm { V a r } ( \widehat { p } _ { G , h } ( y ) ) \leq \frac { P } { 2 ^ { d } \pi ^ { d / 2 } G h ^ { d } } . }\tag{39}
$$

Consequently, for a fixed query point y and any $\delta \in ( 0 , 1 )$ with probability at least $1 - \delta ,$

$$
\boxed { | \widehat { p } _ { G , h } ( y ) - p _ { t } ( y ) | \leq \frac { d M } { 2 } h ^ { 2 } + \sqrt { \frac { P } { 2 ^ { d } \pi ^ { d / 2 } G h ^ { d } \delta } } . }\tag{40}
$$

Proof 2 Using the full group in Equation (34) and linearity of expectation,

$$
\mathbb { E } [ \widehat { p } _ { G , h } ( y ) ] = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \phi _ { h } ( y - Y _ { j } ) \right]\tag{41}
$$

$$
= \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \mathbb { E } [ \phi _ { h } ( y - Y _ { j } ) ]\tag{42}
$$

$$
= \int _ { \mathcal { V } } \phi _ { h } ( y - u ) p _ { t } ( u ) d u\tag{43}
$$

$$
{ \bf \Phi } = ( p _ { t } * \phi _ { h } ) ( y ) .\tag{44}
$$

No element ofthe rollout group is separatedfrom the others. Let $Z \sim \dot { \mathcal { N } } ( 0 , h ^ { 2 } I _ { d } )$ . Then, away from boundary efects as specified above,

$$
( p _ { t } * \phi _ { h } ) ( y ) = \mathbb { E } [ p _ { t } ( y - Z ) ] .\tag{45}
$$

A second-order Taylor expansion gives

$$
p _ { t } ( y - Z ) = p _ { t } ( y ) - \nabla p _ { t } ( y ) ^ { \top } Z + \frac { 1 } { 2 } Z ^ { \top } \nabla ^ { 2 } p _ { t } ( \xi ) Z\tag{46}
$$

for a point $\xi$ on the segment between y and $y - Z .$ Since $\mathbf { \mathbb { E } } [ Z ] = 0$ and

$$
\mathbb { E } \| Z \| ^ { 2 } = d h ^ { 2 } ,\tag{47}
$$

we obtain

$$
| ( p _ { t } \ast \phi _ { h } ) ( y ) - p _ { t } ( y ) | \leq \frac { M } { 2 } \mathbb { E } \| Z \| ^ { 2 } = \frac { d M } { 2 } h ^ { 2 } ,\tag{48}
$$

which proves Equation (38).

For fixed y, the G terms in Equation (34) are independent and identically distributed. Hence

$$
\operatorname { V a r } ( \widehat { p } _ { G , h } ( y ) ) = \frac { 1 } { G } \operatorname { V a r } ( \phi _ { h } ( y - Y ) )\tag{49}
$$

$$
\leq \frac { 1 } { G } \mathbb { E } [ \phi _ { h } ( y - Y ) ^ { 2 } ]\tag{50}
$$

$$
\leq \frac { P } { G } \int _ { \mathbb { R } ^ { d } } \phi _ { h } ( z ) ^ { 2 } d z .\tag{51}
$$

For the Gaussian kernel,

$$
\int _ {  { \mathbb { R } } ^ { d } } \phi _ { h } ( z ) ^ { 2 } d z = \frac { 1 } { 2 ^ { d } \pi ^ { d / 2 } h ^ { d } } ,\tag{52}
$$

which proves Equation (39). Chebyshev’s inequality together with Equation (38) gives Equation (40).

The preceding theorem shows directly how the approximation depends on bandwidth and group size:

$$
\boxed { \widehat { p } _ { G , h } ( y ) = p _ { t } ( y ) + O ( h ^ { 2 } ) + O _ { p } \Big ( ( G h ^ { d } ) ^ { - 1 / 2 } \Big ) . }\tag{53}
$$

The $O ( h ^ { 2 } )$ term is the Gaussian smoothing bias, while the stochastic error decreases as the efective number of samples in an h-scale neighborhood, $G h ^ { d }$ , increases.

Equation (40) is stated for a fixed query point because that is the elementary finite-sample calculation. To evaluate the same full-group estimator at the sampled outcomes $Y _ { i }$ without separating any sample, use the standard uniformconsistency property of Gaussian kernel-density estimators. On a compact region D satisfying the regularity assumptions above,

$$
\operatorname * { s u p } _ { y \in \mathcal { D } } \left| \widehat { p } _ { G , h } ( y ) - p _ { t } ( y ) \right| = O ( h ^ { 2 } ) + O _ { p } \left( \sqrt { \frac { \log G } { G h ^ { d } } } \right)\tag{54}
$$

provided $h  0$ and $G h ^ { d } / \log G \to \infty .$ . Because Equation (54) holds simultaneously for every $y \in \mathcal { D }$ , it also holds when y is any of the observed outcomes $Y _ { i } \in \mathcal { D }$ . Therefore,

$$
\boxed { \frac { \rho _ { i } } { G C _ { h } } = p _ { t } ( Y _ { i } ) + O ( h ^ { 2 } ) + O _ { p } \left( \sqrt { \frac { \log G } { G h ^ { d } } } \right) . }\tag{55}
$$

In particular, on any region where $p _ { t } ( y ) \geq m > 0 .$ , if h and $\dot { G }$ are chosen so that the right-hand error in Equation (55) is small relative to $m$ , then

$$
\boxed { \rho _ { i } = G C _ { h } p _ { t } ( Y _ { i } ) ( 1 + \varepsilon _ { i } ) , \qquad | \varepsilon _ { i } | \ll 1 . }\tag{56}
$$

Thus the DARS Gaussian score is approximately the policy probability density multiplied by the same global factor $G C _ { h }$ for all outcomes in the region of interest.

The bandwidth tradeof is now explicit. The bandwidth should be small enough that

$$
p _ { t } * \phi _ { h } \approx p _ { t } ,\tag{57}
$$

but large enough that

$$
G h ^ { d }\tag{58}
$$

is not too small. In the standard regime

$$
h  0 , \qquad { \frac { G h ^ { d } } { \log G } }  \infty ,\tag{59}
$$

the approximation error vanishes uniformly on the region of interest.

Finally, Equation (56) implies

$$
\frac { R _ { i } } { \rho _ { i } } = \frac { 1 } { G C _ { h } } \frac { R _ { i } } { p _ { t } ( Y _ { i } ) } \frac { 1 } { 1 + \varepsilon _ { i } } .\tag{60}
$$

Thus, when the Gaussian kernel-density estimate is accurate, DARS difers from exact inverse policy-density scaling only by a common positive factor and a small multiplicative estimation error. Section A.1 then gives the corresponding reward-proportional population target.

## B Navigation Task

Figure 4 visualizes the movement fields induced by the trained GRPO and DARS-GRPO policies. At each state $s = ( x , y ) \in [ 0 , 1 ] ^ { 2 }$ , actions are sampled repeatedly from the policy and used to estimate the mean one-step displacement $\mathbb { E } [ s _ { t + 1 } - s _ { t } \ | \ s _ { t } = \ s ]$ . The streamlines indicate its direction, their color indicates its magnitude, and the gray contours show the normalized reward landscape.

Terminal actions are treated as zero displacement. Consequently, a small field magnitude may result from small proposed movements, frequent termination, or cancellation among sampled actions. The DARS-GRPO opptimised the policy such that the coverage over all the modes could be optimised in proportion to the reward. The visualization thus shows how density-aware reward scaling changes the learned navigation dynamics. It should not be interpreted as a gradient or parameter-update field.

## C Text-to-Image Prompt Adaptation

LM model choice: GPT-2 vs Qwen3-0.6B Figures 5–8 compare GPT-2 and Qwen3-0.6B as policy LMs under the combined aesthetic and alignment reward. For each original prompt, GRPO generations are shown on the left and DARS generations on the right while keeping the difusion model, sampling budget, and evaluation protocol fixed. This isolates the efect of the language-model used from that of the reinforcement learning objective.

![](images/d107e75bf20288dbc5457a9519c67a2c2b7b310993668d4aa481f379c236d7aa.jpg)  
Figure 4: Policy-induced movement fields for GRPO (left) and DARS-GRPO (right) on the four-mode navigation task. Streamlines show the estimated mean one-step displacement, color indicates its magnitude, and gray contours show the normalized reward landscape.

Across both policy LMs, DARS consistently expands the diversity of generated images while preserving the semantic content of the original prompt. Compared with GRPO, DARS produces greater variation in scene layout, viewpoint, lighting, object appearance, and background context without substantially altering the primary subject. Although GPT-2 tends to make larger structural changes and Qwen3-0.6B generally preserves a more stable object identity, both exhibit the same qualitative benefit from density-aware reward scaling. These examples illustrate that the improvements obtained by DARS are not tied to a particular language model and generalize across diferent policy LMs.

## Reward choice: Combined vs Alignment-only

Figures 9–12 compare the two reward functions using the same Qwen3-0.6B policy LM. The combined reward combines semantic fidelity with improvement under the frozen aesthetic scorer, whereas the alignment-only reward applies only the minimum CLIP-alignment constraint. Each embedded grid compares GRPO and DARS under one reward, allowing the efect of reward design to be separated from the efect of density scaling.

The alignment-only objective permits DARS to explore a substantially broader visual region since it is free of aesthetic bounds that afect the combined reward. Across the examples, it produces larger changes in setting, illumination, camera distance, and composition. This is particularly visible for the bench, cat, scarecrow, and fox prompts in Figures 10,11, and 12 where the generated scenes include a wider range of architectural and natural backgrounds. The quantitative results show the same pattern with higher Image Vendi Scores and DINO pairwise distance for alignment-only rewards.

## Additional qualitative comparisons

Figures 13–15 provide additional examples comparing GRPO, PAG, GFN, and DARS-GRPO. Across these prompts, GRPO reuses a narrow rewriting pattern, whereas DARS produces more varied scene contexts consequently leading to more diverse images.

GPT-2 A snowman standing in the yard.  
![](images/378625739e35aba1fdbe8ec52b2b860e9c0d71fb7b0c2e95a91f9ef151488908.jpg)  
Figure 5: Policy LM comparison for “A snowman standing in the yard” (top) and “A bicycle leaning against a brick wall” (bottom). Each embedded grid compares GRPO with DARS for the indicated language-model.

![](images/decffc193ff919da323ae48b76252db64ff915e54128e78f34f291e96d150daa.jpg)  
GPT-2 A wooden bench under a tree.  
Figure 6: Policy LM comparison for “A wooden bench under a tree” (top) and “A cat sleeping on a windowsill” (bottom). DARS expands the visual contexts while preserving the object and scene requested by each prompt.

![](images/d6178a4a31ec503b1eaddc43dfd581286cab1c85a2c2eb2393d346435f361020.jpg)  
Figure 7: Policy LM comparison for “A scarecrow in the middle of a farm” (top) and “A castle on top of a hill” (bottom). The DARS samples show greater variation in character appearance, architecture, framing, and illumination than the corresponding GRPO samples.

GPT-2  
GPT-2 A scarecrow in the middle of a farm.  
![](images/376b77db22df597a6e96e57057d84dc16a135d36118494675f90b611fcad8c8b.jpg)  
A scarecrow in the middle of a farm.

![](images/5b844c1fe2624d80820919d573194cf73bfcb1b43e0ba99cef054501255012b8.jpg)  
A castle on top of a hill.

![](images/d8bcb4bb4cf7b0b3e5706d28243b959795e6ba40a3db28b3cac849818f392fba.jpg)  
Qwen3-0.6B  
A castle on top of a hill.

![](images/f604425643a46c8b094081cf6ad4620d6ed8e06684b117ddfd9550f3ce2a80d7.jpg)  
GPT-2 A lantern hanging outside a cottage.  
Figure 8: Policy LM comparison for “A lantern hanging outside a cottage” (top) and “A fox standing near a forest path” (bottom). Across both policy LMs, DARS broadens the within-prompt visual distribution while maintaining the prompt’s central subject.

Alignment-only

A bicycle leaning against a brick wall.

Combined A snowman standing in the yard.  
DARS  
![](images/ec6365200766dd7dbd220e38a3bc8b8bef2e47da0a27874278459717df079190.jpg)  
A snowman standing in the yard.

DARS  
![](images/8fc5421901408e8f7235b0e48191b48bac88b815a8592fbfcce7b23934c5f46d.jpg)

Alignment-only  
DARS  
DARS  
![](images/7ed65c04f0077d8225495485e13148c6c19731ad200722b3a77aa2d138d6bf55.jpg)  
A bicycle leaning against a brick wall.

![](images/0448eb4483c3ef7f4206af6ec459d7f544e6296aa39185e693fdd46c851af4c2.jpg)  
Figure 9: Reward comparison for “A snowman standing in the yard” (top) and “A bicycle leaning against a brick wall” (bottom), using Qwen3-0.6B.

DARS

Combined

## Combined

DARS  
DARS  
![](images/b1b5e5d42736f9027e3f25ca87c6512fe2d6a26d144f3779dcc291134c7a91b8.jpg)  
Alignment-only  
A wooden bench under a tree.

![](images/dbd1c2f30579c92d79660b660687cd43479a327f050f1a0428aeb9a8e3d0c9c5.jpg)  
A cat sleeping on a windowsill.

DARS  
![](images/b86fde9907424a8270cb31b1518cfaebdce251ccc019fa65036ff4c6282044b4.jpg)  
Alignment-only  
A cat sleeping on a windowsill.

![](images/3339413726ccc05e62ce28aecaec7012a768bd1e9b8937a299b26e56dcefb80d.jpg)

![](images/ea8c16a82873af03fe5d1440248049d1edb5b902d339c5823cb227cc09d15a57.jpg)  
Figure 10: Reward comparison for “A wooden bench under a tree” (top) and “A cat sleeping on a windowsill” (bottom).

![](images/403b4d73b20b175975a502d86031341eb819db20f1001062986ffab7e337aa70.jpg)  
Figure 11: Reward comparison for “A scarecrow in the middle of a farm” (top) and “A castle on top of a hill” (bottom).

DARS

DARS

GRPO

Combined

DARS

## Combined

A lantern hanging outside a cottage.

DARS  
![](images/fc2a4a813b43d40e82b4cdee8268fe97265e3c797449dbc4700d05f04cade5a7.jpg)  
Alignment-only  
A lantern hanging outside a cottage.

![](images/dd19383e0ed9969113439db20cbde6cee171b58468dfe1d3977c9ead06f48b76.jpg)  
A fox standing near a forest path

![](images/c5b591a954691e89dbe38842e0ec7f4aaa4f8e1c741e05780c2da31db0736716.jpg)  
Alignment-only  
A fox standing near a forest path.

Figure 12: Reward comparison for “A lantern hanging outside a cottage” (top) and “A fox standing near a forest path” (bottom).

![](images/cf4e55f28a2cb5681f2d4d5b49fa00bec8294351bddb35b26af908fd3c28af02.jpg)  
Figure 13: Qualitative comparison for “A bicycle leaning against a brick wall.” GRPO repeatedly generates a similar nighttime composition, while DARS-GRPO varies lighting, seasonal context, and surrounding details while preserving the bicycle and wall.

![](images/ee14d41e042246e0a631f9973731b1925c0480ce4584024d968262aa8c73a93c.jpg)  
Figure 14: Qualitative comparison for “A wooden bench under a tree.” DARS-GRPO explores more varied locations, tree appearances, and visual contexts than GRPO, while PAG and GFN produce more limited changes around the same central composition.

![](images/a3a61e8a49dafc509a137d2c924c0455232f5e28e077db3cafe00760db45a90a.jpg)  
Figure 15: Qualitative comparison for “A cat sleeping on a windowsill.” All methods preserve the primary subject, while DARS-GRPO introduces broader variation in background scenery, lighting, and framing.

## C.1 Implementation Details

Experimental Configuration Unless otherwise specified, all experiments use the same training, generation, and evaluation settings. Table 4 summarizes the policy optimization hyperparameters shared by DARS and the GRPO baseline, while Table 5 lists the hyperparameters specific to densityaware reward scaling. The generation pipeline, reward formulation, and datasets are held fixed across experiments unless explicitly varied in an ablation study.

DARS and GRPO are trained under identical optimization settings to ensure a fair comparison. Both methods use the same policy LM, optimizer hyperparameters, decoding configuration, and rollout schedule. They difer only in the reward scaling mechanism. Unless otherwise stated, these hyperparameters are used for all reported experiments.

Table 5 lists the hyperparameters specific to DARS. Density is estimated in the DINOv2 image embedding space using a Gaussian kernel with bandwidth $h = 2 0 .$ . To study the efect of density estimation, all kernel-width ablations retain the same training configuration while varying only the kernel bandwidth.

Kernel-width ablations use the same experimental setup while varying only the Gaussian kernel width,

$$
h \in \{ 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \} ,
$$

with all other hyperparameters kept fixed.

Computing Environment All experiments were conducted on a server equipped with four NVIDIA A100-SXM4 GPUs using the software environment summarized in Table 6. Mixed-precision training uses bf16, while Stable Difusion inference is performed in fp16 to improve computational eficiency without afecting evaluation.

Training Configuration Tables 4 and 5 summarize the optimization hyperparameters and DARS-specific settings used during training. Unless explicitly stated otherwise, all experiments employ these default configurations. DARS and GRPO difer only in the computation of policy rewards, while sharing identical optimization settings.

Generation Configuration All generated images are produced using the Stable Difusion v1.4 pipeline with identical sampling settings across methods. Each rewritten prompt is decoded independently using 20 difusion steps with a classifier-free guidance scale of 7.5. The same CLIP encoder, DINOv2 encoder, and aesthetic predictor are used throughout both training and evaluation to ensure consistency across reported metrics.

Reward Configuration Two reward variants are considered in this work. The primary configuration combines aesthetic improvement with CLIP-based semantic alignment, while the alignment-only variant removes the aesthetic component to isolate the contribution of density-aware reward scaling. Apart from the reward definition, all experimental settings remain unchanged.

Evaluation Protocol Evaluation follows the same protocol for all methods. For each evaluation prompt, the policy generates N = 64 rewritten prompts, each of which is used to generate one image under identical difusion sampling settings. Diversity metrics are computed only among images originating from the same source prompt and are then averaged across the evaluation set.

Table 4: Policy optimization hyperparameters (DARS and GRPO).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Policy LMs</td><td>SFT Qwen3-0.6B; SFT GPT-2</td></tr><tr><td>Training temperature</td><td>1.5</td></tr><tr><td>Evaluation temperature</td><td>1.0</td></tr><tr><td>max_new_tokens</td><td>64</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>PPO clipping €</td><td>0.25</td></tr><tr><td>KL coefficient</td><td> $1 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>Policy updates per rollout</td><td>1</td></tr><tr><td>Training batch size</td><td>2 prompts</td></tr></table>

Table 5: DARS-specific hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Density embedding space</td><td>dinov2_image</td></tr><tr><td>Density kernel</td><td>Gaussian</td></tr><tr><td>Kernel width (h)</td><td>20.0</td></tr></table>

Image-text alignment is measured using OpenAI CLIP ViT-L/14, while image diversity is computed from DI-NOv2 image embeddings. The aesthetic LAION predictor (sac+logos+ava1-l14-linearMSE) is used wherever aesthetic reward is required.

During training, model checkpoints are periodically evaluated on the validation set using the same evaluation protocol. The checkpoint used for all reported experiments is selected based on the highest validation Vendi Score after reward saturation, ensuring that model selection aligns with the diversity objective optimized by DARS.

Datasets The MS-COCO training split is used for policy optimization, while evaluation is performed on held-out MS-COCO prompts and the Lexica benchmark to assess both in-domain and out-of-domain generalization. Unless otherwise stated, all reported quantitative results use these dataset splits.

Table 6: Hardware and software environment used for training.
<table><tr><td>Component</td><td>Value</td></tr><tr><td>Operating System</td><td>Ubuntu 22.04.4 LTS (Linux 5.15.0-94-generic)</td></tr><tr><td>GPUs Host RAM</td><td>4× NVIDIA A100-SXM4-40GB</td></tr><tr><td>Python</td><td>~503 GiB 3.13.9</td></tr><tr><td>PyTorch</td><td>2.12.1+cu130</td></tr><tr><td>Key Libraries</td><td>transformers 5.13.0, diffusers0.39.0,</td></tr><tr><td>Mixed Precision</td><td>accelerate 1.14.0, peft 0.19.1</td></tr><tr><td>Diffusion Dtype</td><td>bf16(run.mixed_precision) fp16</td></tr></table>

Table 7: Shared generation and evaluation configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Diffusion model</td><td>CompVis/stable-diffusion-v1-4</td></tr><tr><td>Resolution / Steps / CFG</td><td>512 × 512, 20 steps, guidance scale 7.5</td></tr><tr><td>Scheduler</td><td>DPM-Solver Multistep</td></tr><tr><td>diffusion.seed_mode</td><td>random</td></tr><tr><td>run.seed</td><td>42</td></tr><tr><td>Rewrites per prompt (G = N)</td><td>64</td></tr><tr><td>CLIP encoder</td><td>OpenAI CLIP ViT-L/14</td></tr><tr><td>DINOv2 encoder</td><td>facebook/dinov2-base (CLS embedding)</td></tr><tr><td>Aesthetic predictor</td><td>sac+logos+ava1-114-linearMSE</td></tr><tr><td>CLIP alignment threshold</td><td>0.20</td></tr></table>

Table 8: Reward settings.
<table><tr><td>Setting</td><td>Implementation</td></tr><tr><td>Combined Alignment only</td><td>Product-style aesthetic gain with CLIP alignment gating CLIP alignment reward without aesthetic gain</td></tr></table>

Table 9: Training and evaluation datasets.
<table><tr><td>Split</td><td>Dataset</td></tr><tr><td>Training</td><td>dataset/coco/n_1024/train.jsonl</td></tr><tr><td>MS-COCO evaluation</td><td>dataset/coco/n_1024/test.jsonl</td></tr><tr><td>Lexica evaluation</td><td>dataset/lexica/n_1024/test.jsonl</td></tr></table>