# Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning

Yaning Jia<sup>1</sup>, Chunhui Zhang<sup>1</sup>, Wenxuan Xu<sup>1</sup>, Xingjian Diao<sup>1</sup>

Xiaoyuan Wang<sup>2</sup>, Soroush Vosoughi<sup>1</sup>\*

<sup>1</sup>Dartmouth College

yaning.jia.gr@dartmouth.edu

<sup>2</sup>Carnegie Mellon University

soroush.vosoughi@dartmouth.edu

## Abstract

Supervised fine-tuning (SFT) applies a uni form cross-entropy loss to all target tokens, even though different tokens provide unequal learning signals for mathematical reasoning. This uniform treatment can over-sharpen already mastered tokens while amplifying learn ing pressure on uncertain, low-confidence tokens, leading to suboptimal training dynam ics. We propose Trimmed Logit-Gap SFT (TrimSFT), a simple token-level reweighting method that scales the SFT loss according to the logit gap between the gold token and its strongest competitor. TrimSFT trims supervi sion away from both extremes: tokens already mastered (large logit gap) and tokens weakly supported by the current model (small or neg ative logit gap), concentrating learning within an intermediate logit-gap region between them. We instantiate this principle with a Gaussian weight centered at margin m with bandwidth τ, requiring no reference model or additional forward pass. We evaluate TrimSFT on six base models from the Llama, Qwen, and Deep Math families across five mathematical reasoning benchmarks. TrimSFT consistently im proves over standard SFT, achieving the best average performance on five out of six mod els, with gains of up to +26.9 points over SFT on MATH500. Further analyses show that the bandwidth τ matters more than the exact margin location, and that half-trim variants that remove supervision pressure from only one side yield inferior trade-offs. A token-level logit gap distribution analysis suggests that TrimSFT reshapes model confidence in a more balanced way than uniform SFT or monotonic reweighting methods. These results suggest that rea soning SFT can benefit from trimming both extremes rather than treating all tokens uniformly.

## 1 Introduction

Large language models (LLMs) have demonstrated strong capabilities on complex reasoning tasks (Wei et al., 2022b; Wang et al., 2024; Fan et al., 2024), with mathematical reasoning serving as a key testbed for studying multi-step inference (Jia et al., 2025; Liu et al., 2025). Supervised fine-tuning (SFT) is widely used to adapt pretrained models to mathematical reasoning data (Yu et al., 2024; Yue et al., 2024), and often serves as an initialization stage for downstream reinforcement learning or preference optimization (Ouyang et al., 2022; Bai et al., 2022).

Despite its effectiveness, standard SFT applies token-level cross-entropy with uniform weighting across all target tokens (Lin et al., 2026), ignoring that different tokens can provide very different learning signals (Wu et al., 2026; Gong et al., 2026). In reasoning trajectories, some tokens may already be well mastered and continue to receive unnecessary sharpening pressure, which can contribute to over-confidence (Pereyra et al., 2017; Chen et al., 2025; Wei et al., 2022a). Other tokens may be highly uncertain, noisy, or beyond the model’s current capability, yet still induce large losses and dominate the optimization signal. These two extremes suggest that treating all tokens uniformly can pull learning away from a potentially useful intermediate region (Wu et al., 2026; Lin et al., 2017; Han et al., 2018). This motivates a selective training objective that reduces supervision at both ends of the logit-gap spectrum while concentrating learning on an intermediate region.

We therefore propose Trimmed Logit-Gap SFT (TrimSFT), a token-level reweighting method that trims supervision away from both extremes: tokens already mastered (large logit gap) and tokens weakly supported by the current model (small or negative logit gap). For each target token, we compute its logit gap, defined as the margin between the gold token’s logit and that of its strongest competitor, and use a Gaussian weight centered at margin m with bandwidth τ to scale the cross-entropy loss (Figure 1, left). Tokens whose logit gaps lie near m receive stronger supervision, while tokens with much smaller or larger gaps are softly downweighted, concentrating learning within a bounded intermediate region of the logit-gap spectrum. The weight is computed from the model’s own logits in the same forward pass, requiring no reference model or additional forward pass. To examine the contribution of each side of this two-sided trimming profile, we further introduce two halftrim variants (Figure 1, right): Trim-Easy SFT (TrimSFT-E), which trims the hard side only and preserves full weight on high-gap (easy) tokens, and Trim-Hard SFT (TrimSFT-H), which trims the easy side only and preserves full weight on low-gap (hard) tokens.

![](images/3ac7806e847e9858d1d97dac35cb5f0da123da2226da2f39b42f862e594d20f4.jpg)

![](images/2ffaf433b76057c9c6aa82987829ca63f3542cad0264c43c381314ea50221030.jpg)  
Figure 1: Token weighting functions for SFT, TrimSFT, and its half-trim variants TrimSFT-E and TrimSFT-H. The margin m determines the center of the weighting function, and τ controls the width of the weighted region.

We evaluate TrimSFT on six base models from the Llama, Qwen, and DeepSeekMath families across five mathematical reasoning benchmarks. Our results show that TrimSFT consistently improves over standard SFT, and we further analyze the mechanism behind these gains. Our contributions are summarized as follows:

• We introduce TrimSFT, a token-level reweighting method that scales the SFT loss by the logit gap between the gold token and its strongest competitor. It requires no reference model or additional forward pass.

• TrimSFT achieves the best average performance on five out of six models, with gains of up to +26.9 points over SFT on MATH500. It also improves capability coverage and selfconsistency under repeated sampling, as measured by pass@8 and best-of-8, respectively.

• Ablations show that performance is more sensitive to the bandwidth τ than to the margin m, suggesting that the width of the selected logitgap region is more important than its center.

• Half-trim variants show that trimming only one side yields inferior trade-offs: TrimSFT-H collapses below SFT, and TrimSFT-E degrades on the hardest problems. Token-level logit-gap distribution analysis further shows that TrimSFT reshapes model confidence in a more balanced way than uniform SFT or monotonic reweighting methods.

## 2 Related Work

Supervised Fine-Tuning for Mathematical Reasoning. Mathematical reasoning has emerged as a central testbed for evaluating the multi-step inference capabilities of large language models (Cobbe et al., 2021; Hendrycks et al., 2021; He et al., 2024; Saxton et al., 2019; Lewkowycz et al., 2022). A common recipe for adapting pretrained models to mathematical tasks is supervised finetuning (SFT) on curated reasoning trajectories (Yu et al., 2024; Yue et al., 2024; Toshniwal et al., 2024; Li et al., 2024). Much of the progress in this line has come from data-centric improvements, including synthesizing high-quality chainof-thought solutions (Yu et al., 2024; Luo et al., 2025), distilling from stronger teachers (Shao et al., 2024; Yang et al., 2024), incorporating recent long chain-of-thought trajectories from reasoningspecialized models (Guo et al., 2025; Hugging Face, 2025; Ye et al., 2025), and constructing curated datasets through difficulty- or correctnessbased filtering (Toshniwal et al., 2024; Li et al., 2024). Despite these advances, the underlying objective typically remains the standard token-level cross-entropy loss applied uniformly to every token. In contrast, our work targets the per-token objective rather than the training data, making it complementary to existing data-centric approaches.

Token-Level Reweighting and Selection in SFT. While vanilla SFT applies a uniform cross-entropy loss across all tokens, recent work has explored token-level reweighting or selection to account for differences in training value across tokens (Lin et al., 2024; Ruan et al., 2025; Wu et al., 2026). These methods differ mainly in the signals used to score tokens and in the form of loss modulation. Some approaches rely on auxiliary signals or pre-computed token masks, such as referencemodel-based scoring in Rho-1 (Lin et al., 2024) and counterfactual selection in CFT (Ruan et al., 2025). DFT (Wu et al., 2026) and Focal Loss (Lin et al., 2017) both rescale token-level cross-entropy using the predicted probability of the gold token, though in opposite monotonic directions: DFT up-weights tokens already assigned high probability, while Focal Loss down-weights them. Our method departs from these probability-based schemes in two ways: it uses the logit gap as a more scale-sensitive signal, and applies a Gaussian band-pass weighting that targets a bounded intermediate region rather than following a monotonic trend. We elaborate on this comparison in Section 3.3.

Logit Gaps, Margins, and Confidence Shaping. Margin-related quantities have been widely used to shape confidence and compare competing predictions, but they are rarely used directly as token-level supervision weights. In classification calibration, label smoothing (Müller et al., 2019; Pereyra et al., 2017) and margin-based label smoothing (Liu et al., 2022) shape confidence and margin behavior to mitigate overconfidence, while logit normalization (Wei et al., 2022a) and calibration-oriented training objectives (Guo et al., 2017) regulate confidence and logit magnitude during training. Related concerns have also been studied in large language models, where recent work examines confidence calibration and overconfidence in generated outputs (Zhang et al., 2024; Leng et al., 2024). Recent evidence further shows that linguistic confidence can diverge substantially from internal, logit-based confidence, highlighting the importance of distinguishing verbalized confidence from model-internal confidence signals (Zhang et al., 2026). In preference optimization, methods such as SimPO (Meng et al., 2024) use sequence-level logit margins between preferred and rejected outputs. These works use margin-related quantities either to regulate confidence or to define sequencelevel preference signals. In contrast, TrimSFT uses the per-token logit gap as a supervision weight during SFT, trimming both extremes of the tokenconfidence spectrum rather than constraining or maximizing margins globally.

## 3 Method

We begin from the standard supervised finetuning (SFT) objective and then introduce TrimSFT, a token-level reweighting scheme driven by the model’s own logit gaps. The key idea is to allocate stronger supervision to tokens whose logits place them near a bounded decision region, while reducing learning pressure on tokens that are already well mastered or currently beyond the model’s effective reach. TrimSFT deliberately reduces supervision pressure at both ends of the logit-gap spectrum: tokens with very large gaps (already mastered) receive lower weight, as do tokens with very small or negative gaps (weakly supported by the current model).

Given an input prompt x and a target token sequence $y _ { 1 } , \ldots , y _ { T }$ , standard SFT trains a model $\pi \theta$ by minimizing the token-level cross-entropy loss

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t = 1 } ^ { T } \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) ,\tag{1}
$$

which assigns uniform weight to every target token. Let $z _ { t } \in \bar { \mathbb { R } ^ { V } }$ denote the pre-softmax logits at position t, and let $z _ { t , y _ { t } }$ denote the logit assigned to the gold token $y _ { t }$ . We define the logit gap at position t as

$$
\Delta _ { t } = z _ { t , y _ { t } } - \operatorname* { m a x } _ { v \neq y _ { t } } z _ { t , v } ,\tag{2}
$$

namely, the margin between the gold token’s logit and that of its strongest competing token. A positive $\Delta _ { t }$ indicates that the gold token is currently the top-1 prediction, while a larger magnitude reflects greater separation from the closest alternative. This quantity provides a local, token-level measure of how decisively the model distinguishes the correct token from competing candidates.

## 3.1 Trimmed Logit-Gap SFT

Our goal is to trim supervision away from both extremes of the logit-gap spectrum and concentrate learning on tokens whose gaps fall within an intermediate region. Intuitively, tokens with very large positive gaps are already well separated and may benefit little from continued sharpening, whereas tokens with very small or negative gaps may be highly uncertain, unstable, or beyond the model’s current capability. We therefore seek a weighting function that peaks around a moderate logit-gap region and decays on both sides.

We instantiate this with a Gaussian weight on the logit gap:

$$
w _ { t } = \exp \left( - \frac { ( \Delta _ { t } - m ) ^ { 2 } } { 2 \tau ^ { 2 } } \right) ,\tag{3}
$$

where m is the center of the weighting function and τ controls the width of the high-weight region. Tokens whose logit gaps lie near m receive the largest weights, while tokens whose gaps are substantially smaller or larger are smoothly down-weighted. In this sense, m determines where learning should be concentrated, and τ determines how broadly that concentration should spread.

The resulting TrimSFT objective is

$$
\mathcal { L } _ { \mathrm { T r i m S F T } } = - \sum _ { t = 1 } ^ { T } \operatorname { s g } ( w _ { t } ) \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) ,\tag{4}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. The weights w<sub>t</sub> are computed from the same forward pass as the logits and then detached, so they act purely as a rescaling per-token of the cross-entropy loss without contributing a gradient through θ. This does not require any auxiliary model or extra forward pass, and adds only minimal computational overhead.

Compared with probability-based reweighting, the logit gap is particularly suitable in our setting for two reasons. First, it directly measures the separation between the gold token and its strongest competitor, making it naturally aligned with how close a token is to the model’s decision boundary. Second, unlike the gold-token probability, which tends to saturate in high-confidence regimes, the logit gap remains discriminative even when probabilities would otherwise appear nearly indistinguishable. The Gaussian weighting then realizes a smooth two-sided trimming profile over token states: supervision is concentrated on tokens whose logit gaps lie near the chosen margin m, and decays on both sides. Unlike monotonic reweighting schemes that progressively emphasize either harder or easier tokens across the full confidence range, TrimSFT explicitly trims both extremes and focuses learning on a bounded intermediate region.

## 3.2 Half-Trim Variants

The weighting function in TrimSFT is symmetric around the margin $m ,$ assigning lower weights to tokens on both sides as their logit gaps move away from $m .$ To isolate the contribution of each side, we define two half-trim variants that retain Gaussian decay on only one side while keeping full weight on the other (Figure 1, right).

TrimSFT-E. TrimSFT-E trims the hard side only: it keeps full weight on tokens whose logit gaps exceed the margin m, and applies Gaussian decay only on the lower-gap side:

$$
w _ { t } ^ { \mathrm { E } } = \left\{ \begin{array} { l l } { \displaystyle \exp \left( - \frac { ( \Delta _ { t } - m ) ^ { 2 } } { 2 \tau ^ { 2 } } \right) , } & { \Delta _ { t } < m , } \\ { 1 , } & { \Delta _ { t } \geq m . } \end{array} \right.\tag{5}
$$

This preserves full supervision for tokens already beyond the chosen margin, while down-weighting tokens with smaller gaps.

TrimSFT-H. TrimSFT-H trims the easy side only: it keeps full weight on tokens whose logit gaps fall below the margin m, and applies Gaussian decay only on the higher-gap side:

$$
w _ { t } ^ { \mathrm { H } } = \left\{ \begin{array} { l r } { 1 , } & { \Delta _ { t } \leq m , } \\ { \exp \left( - \frac { ( \Delta _ { t } - m ) ^ { 2 } } { 2 \tau ^ { 2 } } \right) , } & { \Delta _ { t } > m . } \end{array} \right.\tag{6}
$$

This preserves full supervision for tokens with smaller gaps, while down-weighting tokens that are already well separated.

Together, these variants retain the same margin m and bandwidth τ as TrimSFT, but isolate the contribution of each side of the weighting profile. Comparing them with full TrimSFT allows us to determine whether its gains arise primarily from trimming high-gap tokens, low-gap tokens, or both.

## 3.3 A Unified View of Token Reweighting

Standard SFT, TrimSFT, and several representative reweighting methods, including DFT (Wu et al., 2026) and a focal-loss-based SFT baseline (FSFT) (Lin et al., 2017), can all be written in a unified token-reweighted form:

$$
\mathcal { L } = \sum _ { t = 1 } ^ { T } \mathrm { s g } ( w _ { t } ) \ell _ { t } ,\tag{7}
$$

where $\ell _ { t } = - \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } )$ is the standard token-level cross-entropy loss, $\operatorname { s g } ( \cdot )$ denotes stopgradient, and the methods differ only in the choice of $w _ { t }$ . Representative objectives then correspond to different weighting functions:

$$
w _ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { S F T } , } \\ { ( 1 - p _ { t , y _ { t } } ) ^ { \gamma } , } & { \mathrm { F S F T } , } \\ { p _ { t , y _ { t } } , } & { \mathrm { D F T } , } \\ { \mathrm { e x p } \left( - \frac { \left( \Delta _ { t } - m \right) ^ { 2 } } { 2 \tau ^ { 2 } } \right) , } & { \mathrm { T r i m S F T } . } \end{array} \right.\tag{8}
$$

In this framework, standard SFT assigns equal training weight to every token. FSFT and DFT instead reshape the SFT objective using the goldtoken probability: FSFT places larger weights on lower-confidence tokens, thereby emphasizing harder or less well-mastered positions, whereas DFT increases the relative contribution of higherprobability tokens to compensate for the implicit inverse-probability bias of standard cross-entropy. In contrast, TrimSFT uses the logit gap $\Delta _ { t }$ and applies a non-monotonic weighting profile that trims both extremes while concentrating learning on tokens whose gaps lie near a prescribed margin m.

This unified view makes two differences especially clear. First, TrimSFT operates on the logit gap rather than the probability, so it directly tracks the separation between the gold token and its strongest competitor and avoids the saturation that compresses high-confidence tokens to nearly identical weights under probability-based schemes. Second, TrimSFT targets a bounded intermediate region between well-mastered and weakly supported tokens, whereas existing probability-based reweighting methods follow a single monotonic trend over the full confidence range.

## 4 Experiments

## 4.1 Setup

Dataset. NuminaMath-CoT (Li et al., 2024) is a large-scale math reasoning dataset containing about 860K problem-solution pairs with chain-of-thought annotations. For all experiments, we randomly sample 20,000 problems for supervised fine-tuning.

Models. We evaluate TrimSFT and the baseline methods on six base models from three families, spanning parameter scales from 1.5B to 8B: Llama3.2-3B (Meta AI, 2024), Llama3.1- 8B (Grattafiori et al., 2024), DeepSeekMath-7B (Shao et al., 2024), Qwen2.5-Math-1.5B, Qwen2.5-Math-7B (Yang et al., 2024), and Qwen3- 4B-Base (Yang et al., 2025). We use only base models, rather than instruction-tuned variants, to reduce the influence of prior instruction tuning and enable a cleaner comparison of different supervised fine-tuning objectives.

Baselines. We compare TrimSFT with four representative baselines: (i) Base, the original pretrained model without supervised fine-tuning; (ii) SFT, standard supervised fine-tuning with uniform token-level weighting; (iii) FSFT, a focalloss-based variant that upweights lower-confidence tokens during training (Lin et al., 2017); and (iv) DFT (Wu et al., 2026), a probability-based reweighting approach that assigns larger weights to higher-probability tokens.

Evaluation benchmarks. We evaluate all methods on five mathematical reasoning benchmarks spanning a range of difficulty levels: MATH500 (Hendrycks et al., 2021), Olympiad-Bench (He et al., 2024), Minerva (Lewkowycz et al., 2022), AMC (AI-MO, 2024b), and AIME24 (AI-MO, 2024a). For each benchmark, we sample N = 8 generations per problem and report average@8 as the primary metric, which reflects the model’s average generation quality. In Section 4.3, we additionally report pass@8 and best-of-8, where majority voting is used as the selector, to characterize the model’s exploration ability and self-consistency, respectively. Detailed evaluation settings are provided in Appendix A.

Implementation details. For TrimSFT, we use a single hyperparameter setting, $( m , \tau ) = ( 1 . 5 , 0 . 8 )$ for all models and benchmarks in the main comparison (Table 1), without any model- or benchmarkspecific tuning. All results are reported using the checkpoint obtained after one training epoch. We study the effects of varying m and τ in Sections 4.4 and 4.5, and provide additional training and evaluation details in Appendix A. Code is available at https://github.com/karpning/TrimSFT.

## 4.2 Main Results

Table 1 reports the main results under the average@8 accuracy across six base models and five mathematical reasoning benchmarks, with TrimSFT evaluated using $( m , \tau ) = ( 1 . 5 , 0 . 8 )$

Overall, TrimSFT consistently improves over standard SFT across all six models and achieves the best average performance on five out of six models. The gains are especially large on math-oriented models: for example, TrimSFT improves the average score from 15.47 to 31.77 on Qwen2.5-Math-1.5B and from 20.95 to 35.98 on Qwen2.5-Math-7B. On MATH500, TrimSFT is the top-performing method across all six models, with the largest gain over SFT reaching +26.93 points on Qwen2.5- Math-1.5B (40.02 → 66.95). Compared with DFT, TrimSFT achieves stronger overall average performance on five of the six models, suggesting that logit-gap-based reweighting provides a useful alternative to probability-based token weighting. In contrast, FSFT performs poorly in most settings, indicating that simply emphasizing harder tokens is insufficient for reasoning SFT. These results provide initial evidence for our hypothesis that effective supervised fine-tuning for mathematical reasoning benefits from trimming both extremes of the logit-gap spectrum rather than treating all tokens uniformly.

Table 1: Average@8 accuracy (%) on math reasoning benchmarks. Bold marks the best result and underlined marks the second-best within each model block. Avg. denotes the mean across benchmarks.
<table><tr><td>Model</td><td>Method</td><td>MATH500</td><td>OlyBench</td><td>Minerva</td><td>AMC</td><td>AIME24</td><td>Avg.</td></tr><tr><td rowspan="5">Llama3.2-3B</td><td>Base</td><td>1.50</td><td>0.86</td><td>0.66</td><td>1.48</td><td>0.00</td><td>0.90</td></tr><tr><td>SFT</td><td>4.80</td><td>1.56</td><td>1.65</td><td>1.48</td><td>0.00</td><td>1.90</td></tr><tr><td>FSFT</td><td>1.90</td><td>0.96</td><td>0.83</td><td>1.19</td><td>0.00</td><td>0.98</td></tr><tr><td>DFT</td><td>8.78</td><td>2.35</td><td>3.60</td><td>1.05</td><td>0.00</td><td>3.16</td></tr><tr><td>TrimSFT</td><td>9.45</td><td>2.81</td><td>2.92</td><td>3.53</td><td>2.00</td><td>4.14</td></tr><tr><td rowspan="5">Llama3.1-8B</td><td>Base</td><td>2.10</td><td>0.93</td><td>1.65</td><td>1.20</td><td>0.00</td><td>1.18</td></tr><tr><td>SFT</td><td>12.05</td><td>2.59</td><td>3.26</td><td>3.40</td><td>0.41</td><td>4.34</td></tr><tr><td>FSFT</td><td>2.95</td><td>0.89</td><td>0.96</td><td>1.64</td><td>0.00</td><td>1.29</td></tr><tr><td>DFT</td><td>19.77</td><td>4.89</td><td>5.81</td><td>4.73</td><td>0.83</td><td>7.21</td></tr><tr><td>TrimSFT</td><td>20.95</td><td>5.13</td><td>5.52</td><td>6.17</td><td>1.41</td><td>7.84</td></tr><tr><td rowspan="5">DeepSeekMath-7B</td><td>Base</td><td>5.45</td><td>1.63</td><td>2.15</td><td>1.94</td><td>0.00</td><td>2.23</td></tr><tr><td>SFT</td><td>23.40</td><td>5.52</td><td>7.36</td><td>7.08</td><td>0.41</td><td>8.75</td></tr><tr><td>FSFT</td><td>6.80</td><td>1.70</td><td>2.26</td><td>2.80</td><td>0.00</td><td>2.71</td></tr><tr><td>DFT</td><td>38.42</td><td>12.16</td><td>13.30</td><td>13.23</td><td>1.24</td><td>15.67</td></tr><tr><td>TrimSFT</td><td>38.80</td><td>12.94</td><td>12.26</td><td>12.35</td><td>1.24</td><td>15.52</td></tr><tr><td rowspan="5">Qwen2.5-Math-1.5B</td><td>Base</td><td>31.45</td><td>16.23</td><td>8.45</td><td>13.68</td><td>3.32</td><td>14.63</td></tr><tr><td>SFT</td><td>40.02</td><td>12.11</td><td>10.08</td><td>12.64</td><td>2.50</td><td>15.47</td></tr><tr><td>FSFT</td><td>10.30</td><td>1.93</td><td>2.34</td><td>1.76</td><td>0.00</td><td>3.27</td></tr><tr><td>DFT</td><td>63.17</td><td>26.90</td><td>20.55</td><td>27.95</td><td>7.93</td><td>29.30</td></tr><tr><td>TrimSFT</td><td>66.95</td><td>29.41</td><td>24.96</td><td>27.95</td><td>9.58</td><td>31.77</td></tr><tr><td rowspan="5">Qwen2.5-Math-7B</td><td>Base</td><td>40.07</td><td>18.45</td><td>13.43</td><td>18.50</td><td>11.66</td><td>20.42</td></tr><tr><td>SFT</td><td>50.65</td><td>16.91</td><td>17.46</td><td>18.09</td><td>1.66</td><td>20.95</td></tr><tr><td>FSFT</td><td>18.12</td><td>3.06</td><td>4.02</td><td>4.12</td><td>0.41</td><td>5.95</td></tr><tr><td>DFT</td><td>69.75</td><td>33.08</td><td>26.31</td><td>34.41</td><td>11.66</td><td>35.04</td></tr><tr><td>TrimSFT</td><td>71.15</td><td>33.61</td><td>28.88</td><td>37.48</td><td>8.76</td><td>35.98</td></tr><tr><td rowspan="5">Qwen3-4B-Base</td><td>Base</td><td>34.35</td><td>17.96</td><td>11.76</td><td>17.06</td><td>4.15</td><td>17.06</td></tr><tr><td>SFT</td><td>47.45</td><td>15.90</td><td>15.10</td><td>16.16</td><td>1.66</td><td>19.25</td></tr><tr><td>FSFT</td><td>12.40</td><td>1.97</td><td>3.16</td><td>3.39</td><td>0.41</td><td>4.27</td></tr><tr><td>DFT</td><td>61.65</td><td>27.37</td><td>22.88</td><td>24.74</td><td>5.41</td><td>28.41</td></tr><tr><td>TrimSFT</td><td>61.80</td><td>26.08</td><td>25.03</td><td>27.93</td><td>4.15</td><td>29.00</td></tr></table>

## 4.3 Capability Ceiling and Self-Consistency

Beyond average@8, we further evaluate pass@8 and best-of-8 with majority voting to characterize two complementary aspects of model behavior. Pass@8 measures whether the model can produce at least one correct solution among multiple samples, reflecting its capability ceiling or exploration coverage. Best-of-8 with majority voting measures whether the model’s sampled solutions consistently support the correct answer, reflecting the self-consistency of its generation distribution.

We report results on Qwen2.5-Math-1.5B and Llama3.1-8B in Figure 2. For both models, TrimSFT uses the same fixed hyperparameter setting (m, τ ) = (1.5, 0.8) as in the main comparison in Table 1.

As shown in Figure 2, TrimSFT consistently improves pass@8 over standard SFT on both models across most benchmarks, indicating that two-sided logit-gap trimming expands the model’s ability to discover correct solutions under repeated sampling. This suggests that TrimSFT does not merely optimize a single deterministic output, but improves the broader solution space explored by the model. The improvement also extends to best-of-8 with majority voting, where TrimSFT remains stronger than or competitive with the baselines across the two models. Since majority voting requires multiple samples to converge toward the correct answer, these gains indicate that the sampled solutions become more reliably aligned rather than only occasionally correct. Together with the average@8 results in Table 1, these findings show that TrimSFT improves average generation quality, capability coverage, and self-consistency under repeated sampling.

![](images/1ca7b07284c66faf137cee0a2c7952e2265b830b8580546c46e9e918d618f7f5.jpg)  
Qwen2.5-Math-1.5B: Pass@8

![](images/0c5d3f907c0c686131759b01ebf1e60ebe0866a61610f0e5917df4cbc6555c95.jpg)

![](images/32d7dda82bf21453929eee48dab34a10ee70ea81a27fc8b29491c695d2ac5a83.jpg)  
Llama3.1-8B: Pass@8

Qwen2.5-Math-1.5B: Best-of-8  
![](images/403a3d3b7741ffdf47411c6b8069e66e0ee8dc57abe4b0f17f0c2b26e7a71f5b.jpg)  
Figure 2: Pass@8 and best-of-8 with majority voting on Qwen2.5-Math-1.5B and Llama3.1-8B across five mathematical reasoning benchmarks.

## 4.4 Ablation: Margin and Bandwidth

![](images/666693f69fa25c7725b6b213778d60c1ed1de68a545459c3d6353e0f2a151ae1.jpg)

![](images/2a28f17dd6c54f4612e3467505e171f3bbb8ad07a1b9000bb17033ee2c4d7e0a.jpg)  
Figure 3: Ablation of margin m and bandwidth τ on Qwen2.5-Math-1.5B. Left: varying τ with fixed m ∈ $\{ 1 , 6 \}$ ; right: varying m with fixed $\tau \in \{ 1 , 6 \}$ . The dashed line denotes the average SFT baseline.

We further study the effect of the two key hyperparameters in TrimSFT: the margin m, which determines the center of the weighted logit-gap region, and the bandwidth τ, which controls how broadly tokens around this margin are emphasized. Figure 3 reports average@8 accuracy, averaged across the five benchmarks, on Qwen2.5-Math-1.5B under different choices of m and τ, with the full numerical results provided in Appendix B.

Figure 3 shows that TrimSFT is more sensitive to the bandwidth τ than to the margin m. When m is fixed, increasing τ leads to a clear performance drop, especially for m = 1, indicating that an overly broad weighting function weakens the intended focus on an intermediate logit-gap region. In contrast, when $\tau = 1$ is fixed, TrimSFT remains consistently strong across different values of m and stays well above the SFT baseline, with only moderate variation as the margin changes. This relative insensitivity to m holds primarily when the bandwidth is small: with $\tau = 6 .$ performance varies more visibly with $m .$ , whereas with $\tau = 1$ the curve remains comparatively stable. These results suggest that the width of the selected region is more important than its exact center: TrimSFT benefits most when supervision is concentrated within a relatively narrow band of logit gaps, under which the choice of m becomes less critical. Additional token-weight diagnostics further show that τ primarily controls the selectivity and trimming strength of the objective, while m mainly shifts the selected logit-gap region; detailed results are provided in Appendix C.

## 4.5 Half-Trim Variants

We compare TrimSFT with its two half-trim variants, TrimSFT-E and TrimSFT-H, to examine whether the full two-sided trimming profile is necessary. TrimSFT-E trims the hard side only and keeps easy tokens at full weight, whereas TrimSFT-H trims the easy side only and keeps hard tokens at full weight. To avoid drawing conclusions from a single margin choice, we evaluate the three variants under representative margins $m \in \{ 1 , 3 , 5 \}$ while fixing $\tau = 1$ . Table 2 reports results on three representative benchmarks.

Table 2: Comparison between TrimSFT and its half-trim variants on Qwen2.5-Math-1.5B with fixed τ = 1 across representative margins m $\in \{ 1 , 3 , 5 \}$ . Bold marks the best result among the three variants.
<table><tr><td>m</td><td>Method</td><td>MATH500</td><td>AMC</td><td>AIME24</td></tr><tr><td rowspan="3">1</td><td>TrimSFT</td><td>62.80</td><td>31.74</td><td>12.07</td></tr><tr><td>TrimSFT-E</td><td>65.98</td><td>33.01</td><td>7.51</td></tr><tr><td>TrimSFT-H</td><td>28.75</td><td>8.84</td><td>0.41</td></tr><tr><td rowspan="3">3</td><td>TrimSFT</td><td>65.98</td><td>30.29</td><td>7.50</td></tr><tr><td>TrimSFT-E</td><td>64.30</td><td>26.18</td><td>4.56</td></tr><tr><td>TrimSFT-H</td><td>40.92</td><td>13.39</td><td>2.06</td></tr><tr><td rowspan="3">5</td><td>TrimSFT</td><td>65.96</td><td>31.03</td><td>7.50</td></tr><tr><td>TrimSFT-E</td><td>66.67</td><td>32.19</td><td>6.25</td></tr><tr><td>TrimSFT-H</td><td>42.50</td><td>16.03</td><td>1.65</td></tr></table>

Table 2 shows a consistent asymmetry between the two half-trim variants. TrimSFT-E is often competitive with or slightly stronger than TrimSFT on MATH500 and AMC, suggesting that preserving full supervision for higher-gap tokens can benefit relatively easier or medium-difficulty benchmarks. However, this advantage does not consistently transfer to the harder AIME24 benchmark, where TrimSFT achieves the best performance across all three reported margins. In contrast, TrimSFT-H performs substantially worse across all settings, mirroring the poor performance of FSFT in Table 1. These results reveal an asymmetric contribution from the two sides of the weighting profile: suppressing low-gap tokens appears to be the primary source of stability, whereas trimming high-gap tokens provides an additional regularization effect. This helps explain why TrimSFT-E can remain competitive on easier benchmarks while full TrimSFT provides a more robust trade-off on harder problems. Additional gradient-mass analysis supporting this interpretation is provided in Appendix E.

## 5 Mechanism: Logit Gap Distribution Analysis

To better understand how different fine-tuning objectives reshape token-level confidence, i.e., the model’s margin-based confidence in the gold response tokens, we analyze the distribution of logit gaps on the training data. We randomly sample

![](images/95da6c4fe302c9b8db858df5d858ae8de838b06fb3030a333f2bce55c26e64d3.jpg)  
Figure 4: Logit-gap distributions on 100 randomly sampled training examples for Qwen2.5-Math-1.5B, aggregated over all response tokens.

100 examples from the training set and compute teacher-forced logit gaps for every token in the response part of each example. We then aggregate all token-level gaps across the sampled examples and plot their kernel density estimates for Base, SFT, DFT, FSFT, and TrimSFT in Figure 4.

The resulting distributions reveal clear differences in how each objective reshapes token-level confidence. Standard SFT shifts the distribution to the right relative to the base model, indicating stronger separation between gold tokens and their competitors after fine-tuning. FSFT, in contrast, leaves more mass near small logit gaps, consistent with its emphasis on low-confidence tokens and its poor performance in Table 1. DFT also moves the distribution toward larger gaps and appears closer to TrimSFT than to standard SFT, which may help explain why DFT often achieves competitive performance. However, DFT and TrimSFT reach this behavior through different mechanisms: DFT monotonically emphasizes high-probability tokens, whereas TrimSFT trims supervision away from both extremes of the logit-gap spectrum. As a result, TrimSFT shifts the distribution toward larger gaps while preserving a broad shape and reducing mass in the small-gap region. This supports our interpretation that TrimSFT improves reasoning SFT by reshaping token-level confidence in a more balanced way, rather than simply over-emphasizing uncertain tokens or uniformly sharpening already confident ones.

Beyond these distributional analyses, additional token-category results in Appendix F further characterize how TrimSFT redistributes the optimization signal, showing a larger share of gradient mass on numeric and mathematical-symbol tokens and a reduced contribution from generic other tokens.

## 6 Conclusion

We presented TrimSFT, a token-level reweighting method that uses logit gaps to concentrate supervision on an intermediate confidence region. By trimming both already well-separated tokens and tokens with weak current support, TrimSFT provides a non-monotonic alternative to uniform SFT and monotonic reweighting methods. Across six base models and five mathematical reasoning benchmarks, TrimSFT consistently improves over standard SFT, while further analyses show gains in capability coverage, self-consistency, and better trade-offs than half-trim variants. Mechanistically, TrimSFT reshapes token-level confidence by moving tokens toward more confident regions while avoiding uniform over-sharpening. Overall, our results suggest that reasoning SFT can benefit from selectively concentrating supervision within an intermediate logit-gap region rather than applying uniform pressure across all tokens.

## Limitations

TrimSFT is evaluated primarily on mathematical reasoning benchmarks with supervised fine-tuning from base models, so its effectiveness on broader domains such as code generation, open-ended instruction following, or long-form reasoning remains to be further studied. In addition, TrimSFT uses two hyperparameters, the margin m and bandwidth τ , to define the emphasized logit-gap region. Our ablations show that the method is relatively robust to the exact margin choice and generally improves over standard SFT across a range of settings, suggesting that careful tuning is not necessary to obtain gains. Still, different model scales or data distributions may benefit from adaptive strategies that adjust the weighting region automatically during training.

## Ethical Considerations

TrimSFT is a token-level reweighting method for supervised fine-tuning and does not involve new data collection or human annotation. All experiments are conducted using publicly available pretrained models and mathematical reasoning datasets intended for research use. As with other supervised fine-tuning methods, improving reasoning capabilities may also strengthen broader generation abilities of language models, which could potentially be misused in downstream applications.

## Acknowledgments

This research was supported in part by the National Science Foundation under Grant No. 2452367.

## References

AI-MO. 2024a. AIMO validation AIME. https://hu ggingface.co/datasets/AI-MO/aimo-validat ion-aime.

AI-MO. 2024b. AIMO validation AMC. https://hu ggingface.co/datasets/AI-MO/aimo-validat ion-amc.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862. Available: https: //arxiv.org/pdf/2204.05862.

Feng Chen, Allan Raventos, Nan Cheng, Surya Ganguli, and Shaul Druckmann. 2025. Rethinking finetuning when scaling test-time compute: Limiting confidence improves mathematical reasoning. Advances in Neural Information Processing Systems. Available: https://papers.nips.cc/paper\_fil es/paper/2025/hash/e8f4eae0a41cab67fdead 3aa6b77f083-Abstract-Conference.html.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. Available: https://arxiv.org/pdf/2110.14168.

Lizhou Fan, Wenyue Hua, Lingyao Li, Haoyang Ling, and Yongfeng Zhang. 2024. NPHardEval: Dynamic benchmark on reasoning ability of large language models via complexity classes. In Annual Meeting of the Associationfor Computational Linguistics. Available: https://aclanthology.org/2024.acl-lon g.225.pdf.

Xuan Gong, Senmiao Wang, Hanbo Huang, Ruoyu Sun, and Shiyu Liang. 2026. Vcore: Variance-controlled optimization-based reweighting for chain-of-thought supervision. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Available: https: //aclanthology.org/2026.acl-long.1298.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Available: https://arxiv.org/pdf/2407.21783.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning. Available: https://arxiv.org/pdf/1706.0 4599.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Available: https://arxiv.org/pdf/2501.12948.

Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. 2018. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In Advances in Neural Information Processing Systems. Available: https://arxiv.org/pdf/1804.06872.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Annual Meeting of the Association for Computational Linguistics. Available: https: //aclanthology.org/2024.acl-long.211.pdf.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track. Available: https: //arxiv.org/pdf/2103.03874.

Hugging Face. 2025. Open r1: A fully open reproduction of deepseek-r1.

Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Zhongyu Ouyang, Chiyu Ma, and Soroush Vosoughi. 2025. What makes a good curriculum? disentangling the effects of data ordering on llm mathematical reasoning. arXiv preprint arXiv:2510.19099. Available: https://arxiv.org/pdf/2510.19099.

Jixuan Leng, Chengsong Huang, Banghua Zhu, and Jiaxin Huang. 2024. Taming overconfidence in llms: Reward calibration in rlhf. arXiv preprint arXiv:2410.09724. Available: https://arxiv.or g/pdf/2410.09724.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. 2022. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems. Available: https://arxiv.org/pdf/2206.14858.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. 2024. Numinamath: The largest public dataset in

ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository. Available: https://huggingface.co/collections/A I-MO/numinamath.

Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, Yufan Guo, Kai Zhong, Weiqi Zhang, sujay sanghavi, Changyou Chen, Hyokun Yun, and Lihong Li. 2026. SFT doesn’t always hurt general capabilities: Revisiting domain-specific fine-tuning in LLMs. In International Conference on Learning Representations. Available: https://arxiv.org/pdf/2509.20758.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal loss for dense object detection. In International Conference on Computer Vision. Available: https://arxiv.org/pdf/1708 .02002.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, et al. 2024. Rho-1: Not all tokens are what you need. arXiv preprint arXiv:2404.07965. Available: https://arxiv.org/pdf/2404.07965.

Bingyuan Liu, Ismail Ben Ayed, Adrian Galdran, and Jose Dolz. 2022. The devil is in the margin: Marginbased label smoothing for network calibration. In Conference on Computer Vision and Pattern Recognition. Available: https://arxiv.org/pdf/2111 .15430.

Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, and Ming Zhang. 2025. Safe: Enhancing mathematical reasoning in large language models via retrospective step-aware formal verification. In Annual Meeting of the Association for Computational Linguistics. Available: https://aclanthology.org /2025.acl-long.594.pdf.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-Guang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang, and Dongmei Zhang. 2025. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. In International Conference on Learning Representations. Available: https: //arxiv.org/pdf/2308.09583.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems. Available: https: //arxiv.org/pdf/2405.14734.

Meta AI. 2024. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. https: //ai.meta.com/blog/llama-3-2-connect-202 4-vision-edge-mobile-devices/. Meta AI Blog, accessed 2026.

Rafael Müller, Simon Kornblith, and Geoffrey E Hinton. 2019. When does label smoothing help? In

Advances in Neural Information Processing Systems. Available: https://arxiv.org/pdf/1906.02629.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems. Available: https: //arxiv.org/pdf/2203.02155.

Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. 2017. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548. Available: https://arxiv.org/pdf/1701.06548.

Zhiwen Ruan, Yixia Li, He Zhu, Yun Chen, Peng Li, Yang Liu, and Guanhua Chen. 2025. Enhancing large language model reasoning via selective critical token fine-tuning. arXiv preprint arXiv:2510.10974. Available: https://arxiv.org/pdf/2510.10974.

David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. 2019. Analysing mathematical reasoning abilities of neural models. arXiv preprint arXiv:1904.01557. Available: https://arxiv.or g/pdf/1904.01557.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Available: https://arxiv.org/pdf/2402.03300.

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. 2024. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. In Advances in Neural Information Processing Systems. Available: https: //arxiv.org/pdf/2402.10176.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems. Available: https://arxiv.org/pdf/2406.01574.

Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. 2022a. Mitigating neural network overconfidence with logit normalization. In International Conference on Machine Learning. Available: https://arxiv.org/pdf/2205.09310.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems. Available: https://arxiv.org/pdf/2201.11903.

Yongliang Wu, Yizhou Zhou, Zhou Ziheng, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi,

Ming-Hsuan Yang, and Xu Yang. 2026. On the generalization of SFT: A reinforcement learning perspective with reward rectification. In International Conference on Learning Representations. Available: https://arxiv.org/pdf/2508.05629.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388. Available: https://arxiv.org/pdf/2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. 2024. Qwen2.5- Math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122. Available: https://arxiv.or g/pdf/2409.12122.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387. Available: https://arxiv.org/pdf/2502.03387.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. Metamath: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations. Available: https://arxiv.org/pdf/2309.12284.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2024. MAmmoTH: Building math generalist models through hybrid instruction tuning. In International Conference on Learning Representations. Available: https://arxiv.org/pdf/2309.05653.

Hefan Zhang, Bingquan Zhang, Ming Cheng, Saeed Hassanpour, Weicheng Ma, and Soroush Vosoughi. 2026. When linguistic and internal confidence diverge in large language models. arXiv preprint arXiv:2608.28382. Available: https://arxiv.or g/pdf/2608.28382.

Mozhi Zhang, Mianqiu Huang, Rundong Shi, Linsen Guo, Chong Peng, Peng Yan, Yaqian Zhou, and Xipeng Qiu. 2024. Calibrating the confidence of large language models by eliciting fidelity. In Conference on Empirical Methods in Natural Language Processing. Available: https://aclanthology.o rg/2024.emnlp-main.173.pdf.

## Contents of Appendix

A Experiment Setting 12   
B Detailed Results for Ablations 12   
C Understanding the Roles of m and τ 12   
D Half-Trim Variants 13   
E Gradient-Mass Analysis 14   
F Token-Category Analysis 14

## A Experiment Setting

Parameter setting. We train all models for one epoch using the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ , cosine learning rate decay with warmup, and gradient clipping. Training is conducted with bfloat16 mixed precision and FlashAttention-2 acceleration under the FSDP framework. Unless otherwise specified, the global batch size is 24 with a per-device micro-batch size of 4, and the maximum sequence length is set to 2048. For TrimSFT, we set the margin to $m = 1 . 5$ and the bandwidth to $\tau = 0 . 8$

Baseline setting. All four methods, SFT, FSFT, DFT, and TrimSFT, are trained under an identical configuration, including optimizer, learning rate, schedule, batch size, sequence length, precision, and number of epochs. They differ only in the pertoken weight w<sub>t</sub> applied to the cross-entropy loss, as summarized in Section 3.3. The only methodspecific hyperparameters beyond this shared recipe are the margin m and bandwidth τ in TrimSFT, and the focusing parameter $\gamma$ in FSFT, for which we use γ = 2 following common practice in the focalloss literature (Lin et al., 2017). DFT introduces no additional hyperparameters.

Evaluation protocol. For evaluation, we sample eight generations per problem using temperature 1.0, top-p sampling with $p \ = \ 1 . 0 .$ , and a maximum generation length of 2048 tokens. We use temperature 1.0 to encourage diverse sampled solutions, which is necessary for evaluating average@8, pass@8, and best-of-8 under repeated sampling. Average@8 is computed as the mean correctness over the eight sampled generations, pass@8 checks whether at least one generation is correct, and bestof-8 uses majority voting over the extracted final answers. We extract final answers from the generated solutions and judge correctness using the same evaluation protocol across all methods and benchmarks.

Computing infrastructure. Experiments are conducted on NVIDIA RTX A6000 GPUs, each with 48GB memory, using CUDA 12.8 and NVIDIA driver 570.207. Each training run uses two GPUs with FSDP unless otherwise specified.

## B Detailed Results for Ablations

Table 3 and Table 4 report the full per-benchmark results for the margin and bandwidth ablation study shown in Figure 3. Each setting is denoted by $( m , \tau )$ , where m is the margin and τ is the bandwidth. While the main text summarizes the averaged trends across the evaluated benchmarks, this appendix provides the complete per-benchmark results for both varying the bandwidth under fixed margins and varying the margin under fixed bandwidths.

Table 3: Ablation results with fixed m and varying τ . Each setting is denoted as (m, τ ).
<table><tr><td>(m, τ)</td><td>MATH500</td><td>OlyBench</td><td>Minerva</td><td>AMC</td><td>AIME24</td><td>Avg.</td></tr><tr><td colspan="7">Fixed m = 1</td></tr><tr><td>(1,0.1)</td><td>66.83</td><td>29.53</td><td>23.86</td><td>29.02</td><td>9.61</td><td>31.77</td></tr><tr><td>(1, 0.5)</td><td>63.72</td><td>27.17</td><td>23.16</td><td>25.90</td><td>9.16</td><td>29.82</td></tr><tr><td>(1, 1)</td><td>62.80</td><td>27.66</td><td>18.65</td><td>31.74</td><td>12.07</td><td>30.58</td></tr><tr><td>(1, 2)</td><td>60.38</td><td>24.68</td><td>18.46</td><td>25.00</td><td>4.58</td><td>26.22</td></tr><tr><td>(1,3)</td><td>53.63</td><td>21.05</td><td>16.30</td><td>22.50</td><td>3.32</td><td>23.36</td></tr><tr><td>(1,4)</td><td>54.05</td><td>21.03</td><td>16.88</td><td>20.59</td><td>3.12</td><td>23.13</td></tr><tr><td>(1, 5)</td><td>46.83</td><td>16.38</td><td>14.15</td><td>17.34</td><td>2.07</td><td>19.35</td></tr><tr><td>(1, 6)</td><td>46.93</td><td>15.30</td><td>12.96</td><td>16.78</td><td>2.07</td><td>18.81</td></tr><tr><td colspan="7">Fixed m = 6</td></tr><tr><td>(6,0.1)</td><td>65.12</td><td>30.76</td><td>23.31</td><td>30.75</td><td>7.93</td><td>31.57</td></tr><tr><td>(6, 0.5)</td><td>68.60</td><td>30.69</td><td>26.83</td><td>31.45</td><td>7.94</td><td>33.10</td></tr><tr><td>(6, 1)</td><td>66.83</td><td>29.28</td><td>22.65</td><td>27.65</td><td>11.65</td><td>31.61</td></tr><tr><td>(6,2)</td><td>66.39</td><td>28.60</td><td>21.11</td><td>33.39</td><td>8.76</td><td>31.65</td></tr><tr><td>(6,3)</td><td>63.95</td><td>27.44</td><td>20.50</td><td>29.55</td><td>7.50</td><td>29.79</td></tr><tr><td>(6, 4)</td><td>60.70</td><td>26.40</td><td>19.21</td><td>26.77</td><td>5.42</td><td>27.70</td></tr><tr><td>(6,5)</td><td>56.90</td><td>24.88</td><td>18.65</td><td>25.45</td><td>4.15</td><td>26.01</td></tr><tr><td>(6, 6)</td><td>54.40</td><td>22.26</td><td>16.86</td><td>22.81</td><td>4.58</td><td>24.18</td></tr><tr><td>SFT</td><td>40.02</td><td>12.11</td><td>10.08</td><td>12.64</td><td>2.50</td><td>15.47</td></tr></table>

The per-benchmark results are consistent with the averaged trends in the main text. When the margin m is fixed, increasing the bandwidth τ generally reduces performance, especially when τ becomes large. In contrast, when $\tau = 1$ is fixed, the method remains stronger than standard SFT across a wide range of margin values, indicating that it does not require careful tuning of the exact margin location to obtain improvements.

## C Understanding the Roles of m and τ

To further explain the different sensitivities of TrimSFT to the bandwidth τ and the margin $m ,$ we conduct a token-weight analysis under the same experimental setting as Section 4.4, using Qwen2.5- Math-1.5B. Specifically, we compute logit gaps on 40,082 gold response tokens sampled from the training data and derive the corresponding TrimSFT weights:

Table 4: Ablation results with fixed τ and varying m. Each setting is denoted as (m, τ).
<table><tr><td>(m, τ)</td><td>MATH500</td><td>OlyBench</td><td>Minerva</td><td>AMC</td><td>AIME24</td><td>Avg.</td></tr><tr><td colspan="7">Fixed τ = 1</td></tr><tr><td>(0,1)</td><td>61.10</td><td>26.64</td><td>20.96</td><td>26.60</td><td>7.91</td><td>28.64</td></tr><tr><td>(0.5, 1)</td><td>66.75</td><td>30.73</td><td>23.45</td><td>31.76</td><td>9.99</td><td>32.54</td></tr><tr><td>(1, 1)</td><td>62.80</td><td>27.66</td><td>18.65</td><td>31.74</td><td>12.07</td><td>30.58</td></tr><tr><td>(2, 1)</td><td>62.25</td><td>26.06</td><td>19.39</td><td>27.36</td><td>7.09</td><td>28.43</td></tr><tr><td>(3, 1)</td><td>65.98</td><td>26.76</td><td>19.64</td><td>30.29</td><td>7.50</td><td>30.03</td></tr><tr><td>(4, 1)</td><td>63.60</td><td>26.90</td><td>22.03</td><td>27.51</td><td>5.83</td><td>29.17</td></tr><tr><td>(5, 1)</td><td>65.96</td><td>28.55</td><td>20.13</td><td>31.03</td><td>7.50</td><td>30.63</td></tr><tr><td>(6, 1)</td><td>66.83</td><td>29.27</td><td>22.65</td><td>27.65</td><td>11.65</td><td>31.61</td></tr><tr><td colspan="7">Fixed τ = 6</td></tr><tr><td>(0, 6)</td><td>46.97</td><td>15.98</td><td>13.83</td><td>16.46</td><td>2.47</td><td>19.14</td></tr><tr><td>(0.5, 6)</td><td>48.13</td><td>17.14</td><td>14.23</td><td>16.03</td><td>3.75</td><td>19.86</td></tr><tr><td>(1, 6)</td><td>46.93</td><td>15.30</td><td>12.96</td><td>16.78</td><td>2.07</td><td>18.81</td></tr><tr><td>(2, 6)</td><td>46.30</td><td>17.74</td><td>13.88</td><td>16.32</td><td>3.32</td><td>19.51</td></tr><tr><td>(3, 6)</td><td>54.58</td><td>21.48</td><td>17.74</td><td>24.26</td><td>5.84</td><td>24.78</td></tr><tr><td>(4, 6)</td><td>55.88</td><td>22.75</td><td>17.19</td><td>23.96</td><td>5.82</td><td>25.12</td></tr><tr><td>(5, 6)</td><td>51.50</td><td>20.25</td><td>16.53</td><td>21.46</td><td>2.49</td><td>22.45</td></tr><tr><td>(6, 6)</td><td>54.40</td><td>22.26</td><td>16.86</td><td>22.81</td><td>4.57</td><td>24.18</td></tr><tr><td>SFT</td><td>40.02</td><td>12.11</td><td>10.08</td><td>12.64</td><td>2.50</td><td>15.47</td></tr></table>

$$
w _ { t } = \exp \left( - \frac { ( \Delta _ { t } - m ) ^ { 2 } } { 2 \tau ^ { 2 } } \right) ,\tag{9}
$$

where $\begin{array} { r } { \Delta _ { t } = z _ { t , y _ { t } } - \operatorname* { m a x } _ { v \neq y _ { t } } z _ { t , v } . } \end{array}$

We report two diagnostics. High-Weight (%) is the fraction of tokens with $w _ { t } > 0 . 5$ , while the Effective Token Ratio, denoted as $R _ { \mathrm { e f f } }$ , is defined as

$$
R _ { \mathrm { e f f } } = \frac { \left( \sum _ { t = 1 } ^ { N } w _ { t } \right) ^ { 2 } } { N \sum _ { t = 1 } ^ { N } w _ { t } ^ { 2 } } .\tag{10}
$$

where N is the number of response tokens. A larger $R _ { \mathrm { e f f } }$ indicates that supervision is distributed more uniformly across tokens, with uniform SFT corresponding to $R _ { \mathrm { e f f } } = 1$

Table 5 shows that varying m from 1 to 6 with $\tau = 1$ leaves the fraction of high-weight tokens nearly unchanged, while increasing τ substantially broadens the weighted region and moves $R _ { \mathrm { e f f } }$ toward uniform SFT. These results provide additional evidence that m mainly shifts the location of the weighting band, whereas τ controls its selectivity and trimming strength.

## D Half-Trim Variants

Table 6 reports additional results for TrimSFT and its two half-trim variants, TrimSFT-E and

Table 5: Effect of margin and bandwidth on TrimSFT token weighting for Qwen2.5-Math-1.5B. Avg. denotes average@8 accuracy across the five mathematical reasoning benchmarks.
<table><tr><td> $( m , \tau )$ </td><td>High-Weight (%)</td><td> $R _ { \mathrm { e f f } }$ </td><td>Avg.</td></tr><tr><td>(1,1)</td><td>14.92</td><td>0.212</td><td>30.58</td></tr><tr><td>(3, 1)</td><td>14.79</td><td>0.223</td><td>30.03</td></tr><tr><td>(6, 1)</td><td>15.56</td><td>0.236</td><td>31.61</td></tr><tr><td>(1,3)</td><td>38.47</td><td>0.517</td><td>23.36</td></tr><tr><td>(1, 6)</td><td>66.42</td><td>0.831</td><td>18.11</td></tr><tr><td>(6, 3)</td><td>47.34</td><td>0.666</td><td>29.79</td></tr><tr><td>(6, 6)</td><td>86.34</td><td>0.923</td><td>24.18</td></tr><tr><td>SFT</td><td>100.00</td><td>1.000</td><td>15.47</td></tr></table>

TrimSFT-H, on Qwen2.5-Math-1.5B. The main text reports representative margins $m \in \{ 1 , 3 , 5 \}$ , while this appendix includes the full set of margins $m \in \{ 1 , 2 , 3 , 4 , 5 \}$ . All variants are evaluated with fixed $\tau = 1$

Table 6: Comparison between TrimSFT and its half-trim variants with fixed $\tau = 1$ . Bold marks the best result among the three variants.
<table><tr><td>m</td><td>Method</td><td>MATH500</td><td>AMC</td><td>AIME24</td></tr><tr><td rowspan="3">1</td><td>TrimSFT</td><td>62.80</td><td>31.74</td><td>12.07</td></tr><tr><td>TrimSFT-E</td><td>65.98</td><td>33.01</td><td>7.51</td></tr><tr><td>TrimSFT-H</td><td>28.75</td><td>8.84</td><td>0.41</td></tr><tr><td rowspan="3">2</td><td>TrimSFT</td><td>62.25</td><td>27.36</td><td>7.09</td></tr><tr><td>TrimSFT-E</td><td>65.32</td><td>32.06</td><td>7.10</td></tr><tr><td>TrimSFT-H</td><td>37.05</td><td>11.18</td><td>0.41</td></tr><tr><td rowspan="3">3</td><td>TrimSFT</td><td>65.98</td><td>30.29</td><td>7.50</td></tr><tr><td>TrimSFT-E</td><td>64.30</td><td>26.18</td><td>4.56</td></tr><tr><td>TrimSFT-H</td><td>40.92</td><td>13.39</td><td>2.06</td></tr><tr><td rowspan="3">4</td><td>TrimSFT</td><td>63.60</td><td>27.51</td><td>8.83</td></tr><tr><td>TrimSFT-E</td><td>64.58</td><td>27.49</td><td>7.51</td></tr><tr><td>TrimSFT-H</td><td>42.43</td><td>13.98</td><td>2.47</td></tr><tr><td rowspan="3">5</td><td>TrimSFT</td><td>65.96</td><td>31.03</td><td>7.50</td></tr><tr><td>TrimSFT-E</td><td>66.67</td><td>32.19</td><td>6.25</td></tr><tr><td>TrimSFT-H</td><td>42.50</td><td>16.03</td><td>1.65</td></tr></table>

The results are consistent with the trends discussed in the main text. TrimSFT-E is often competitive with TrimSFT on relatively easier or medium-difficulty benchmarks such as MATH500 and AMC, suggesting that preserving full supervision on high-gap tokens can sometimes be useful. However, TrimSFT achieves better results on the harder AIME24 benchmark across the reported margins. In contrast, TrimSFT-H consistently performs much worse than both TrimSFT and TrimSFT-E, indicating that preserving full supervision on low-gap tokens while trimming highgap tokens leads to an inferior trade-off. Overall, these results support the benefit of two-sided trimming over relying on either half-trim variant alone.

## E Gradient-Mass Analysis

To further examine the asymmetric behavior of the half-trim variants, we analyze how each weighting scheme redistributes the token-level optimization signal. We conduct this analysis on 100 randomly sampled training examples from NuminaMath-CoT, covering 40,082 response tokens, using Qwen2.5-Math-1.5B. Following the setting in Section 4.5, we fix $\tau = 1$ and consider $m \in \{ 1 , 3 , 5 \}$ For each response token, we compute the weighted logit-gradient magnitude

$$
g _ { t } = w _ { t } \left. \mathrm { s o f t m a x } ( z _ { t } ) - \mathrm { o n e h o t } ( y _ { t } ) \right. _ { 2 } ,\tag{11}
$$

and report the fraction of total gradient mass assigned to different logit-gap regions. For each $( m , \tau )$ setting, we define the low-, middle-, and high-gap regions as $\Delta _ { t } < m - \tau , m - \tau \leq \Delta _ { t } \leq$ $m + \tau .$ , and $\Delta _ { t } > m + \tau$ , respectively.

Table 7: Distribution of weighted gradient mass (%) across logit-gap regions for TrimSFT and its half-trim variants on Qwen2.5-Math-1.5B. All settings use $\tau = 1$
<table><tr><td>m</td><td>Method</td><td>Low</td><td>Middle</td><td>High</td></tr><tr><td>1</td><td>TrimSFT TrimSFT-E TrimSFT-H</td><td>22.18 18.41 71.78</td><td>73.14 64.05 26.71</td><td>4.68 17.54 1.50</td></tr><tr><td>3</td><td>TrimSFT TrimSFT-E TrimSFT-H</td><td>39.36 35.36 93.80</td><td>58.07 54.36 5.97</td><td>2.57 10.28 0.23</td></tr><tr><td>5</td><td>TrimSFT TrimSFT-E TrimSFT-H</td><td>45.00 40.91 99.04</td><td>52.60 49.75 0.93</td><td>2.39 9.34 0.04</td></tr></table>

Table 7 reveals a clear asymmetry between the two sides of the weighting profile. TrimSFT-H, which preserves full supervision on low-gap tokens, concentrates most of its gradient mass in the low-gap region, reaching 71.78%, 93.80%, and 99.04% as m increases. In contrast, both TrimSFT and TrimSFT-E substantially suppress this region and maintain a larger share of the optimization signal in the intermediate region. Full TrimSFT further reduces the contribution of high-gap tokens relative to TrimSFT-E, providing an additional regularization effect on already well-separated tokens.

These results support the interpretation that lowgap trimming is the primary source of stability, while high-gap trimming provides a complementary regularization effect.

## F Token-Category Analysis

Following the same experimental setting as Section E, we conduct the token-category analysis on 100 randomly sampled training examples from NuminaMath-CoT using Qwen2.5-Math-1.5B, covering 40,082 response tokens. We group the decoded response tokens into four lightweight categories and compare standard SFT with TrimSFT under $( m , \tau ) = ( 1 , 1 )$

Table 8: Token-category analysis on Qwen2.5-Math-1.5B. Num., MS, and RC denote numeric tokens, mathematical symbols, and reasoning connectors, respectively. Ratio is the token fraction, Weight is the average TrimSFT weight, Mid. is the intermediate-gap fraction, and Grad. is the gradient-mass fraction.
<table><tr><td>Type</td><td>Ratio</td><td>Weight</td><td>Mid.</td><td>SFT Grad.</td><td>Trim Grad.</td></tr><tr><td>Num.</td><td>15.39</td><td>0.132</td><td>11.25</td><td>5.55</td><td>10.31</td></tr><tr><td>MS</td><td>22.49</td><td>0.191</td><td>16.31</td><td>15.76</td><td>22.78</td></tr><tr><td>RC</td><td>0.82</td><td>0.053</td><td>3.98</td><td>2.55</td><td>0.65</td></tr><tr><td>Other</td><td>61.30</td><td>0.158</td><td>13.06</td><td>76.14</td><td>66.25</td></tr></table>

Table 8 shows that TrimSFT assigns a larger share of gradient mass to numeric and mathematical-symbol tokens than standard SFT, while reducing the contribution of the broad Other category. In particular, the gradient-mass share increases from 5.55% to 10.31% for numeric tokens and from 15.76% to 22.78% for mathematical symbols. These results suggest that the reweighted optimization signal remains closely associated with math-relevant token categories rather than being concentrated primarily on generic response tokens.