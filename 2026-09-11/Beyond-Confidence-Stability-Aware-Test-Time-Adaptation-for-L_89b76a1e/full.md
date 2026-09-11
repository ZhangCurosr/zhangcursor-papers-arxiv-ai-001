# Beyond Confidence: Stability-Aware Test-Time Adaptation for LLM Reasoning

Bincheng Gu<sup>1</sup>, Min Gao<sup>1∗</sup>, Zongwei Wang<sup>1</sup>, Yibing Bai<sup>1</sup>, Yulan He<sup>2</sup>, Junliang Yu<sup>3</sup>

<sup>1</sup>Key Laboratory of Dependable Service Computing in Cyber Physical Society, Chongqing University, China

<sup>2</sup>King’s College London, London, UK

<sup>3</sup>Grifith University, Brisbane, Australia

gaomin@cqu.edu.cn,

## Abstract

Test-time adaptation has emerged as a lightweight alternative to costly post-training for improving the reasoning capabilities of Large Language Models (LLMs) on downstream tasks. Predictive entropy provides a model-derived signal for such adaptation, guiding models toward higher-confidence reasoning states without external verifiers or reward models. However, higher confidence does not necessarily imply correctness, as LLMs may remain highly confident along incorrect reasoning trajectories. We observe that high-confidence reasoning is more likely to be correct when confidence remains stable under local perturbations. Based on this observation, we propose Test-Time Adaptation via Stability-Aware Confidence Optimization (TASCO), a framework that incorporates local stability into confidence-based test-time adaptation while keeping the LLM frozen. TASCO operationalizes local stability by optimizing a lightweight task-level prefix under two alternative perturbation strategies: Random Perturbation promotes distributional stability across trajectories induced by nearby perturbed prefixes, whereas Sharpness-Aware Perturbation targets worst-case local sensitivity. Experiments demonstrate that TASCO improves reasoning accuracy and token eficiency across diverse LLMs and reasoning benchmarks, while behavioral analyses show that it maintains stable confidence under local perturbations without prema turely concentrating the model’s predictive distribution.

## Introduction

Large Language Models (LLMs) have demonstrated strong reasoning capabilities on complex tasks such as mathematical reasoning and code generation (Guo et al. 2025; Jaech et al. 2024). Traditionally, adapting these models to downstream tasks relies on post-training methods that update model parameters, such as supervised fine-tuning or reinforcement learning (Shao et al. 2024; Lambert et al. 2024). Although efective, these methods require substantial computational resources, motivating recent work to keep model parameters frozen and instead optimize continuous inputs or internal representations at inference time (Hu et al. 2025b).

One efective paradigm is to optimize a small set of continuous variables, including soft prompts, steering vectors, and latent representations, to guide a frozen LLM toward task-specific reasoning states (Xu et al. 2025b; Li et al. 2025;

![](images/e34d2238fbe9c98e084c3da0d1094ab06e9fb26a40fac2cb1ac1e454030837dc.jpg)  
Figure 1: Comparison of confidence-only and stability-aware optimization under prefix perturbations across trajectories yielding correct and incorrect answers.

Sinii et al. 2025). These approaches difer mainly in the source of their optimization signals. Some rely on external verifiers or reward models to score answers or reasoning trajectories (Khalifa et al. 2025; Nguyen and Le 2026), whereas verifier-free methods use the model’s output entropy as an intrinsic confidence signal (Ye, Liang, and Shan 2025).

However, entropy-based test-time optimization assumes that an LLM is more likely to answer correctly when its predictive entropy is lower during reasoning (Agarwal et al. 2026). This assumption can fail on complex reasoning tasks, where LLMs can be poorly calibrated and remain highly confident along incorrect reasoning trajectories (Jurayj, Cheng, and Van Durme 2025; Huang et al. 2025). When confidence is the only test-time objective, entropy minimization cannot distinguish high confidence associated with reliable reasoning from spurious confidence along an incorrect trajectory, potentially steering the model toward a locally fragile highconfidence state (Chen et al. 2026a; Zhao et al. 2026).

How can we guide an LLM toward a high-confidence state that supports reliable reasoning? Inspired by the connection between flat minima and generalization (Keskar et al. 2016), we examine the local stability of high-confidence states. Specifically, high confidence that remains stable under local perturbations to the steering variable is more likely to stem from reliable reasoning than from a fragile overconfident state (Fig. 1). We measure this stability using trajectory-level confidence variance. As shown in Fig. 2, the low-variance group consistently achieves higher answer accuracy than the high-variance group across representative models from three families. Further analysis shows that the low-variance group remains more accurate among queries with comparable mean confidence (see Appendix B for details). Together, these results suggest that local stability complements confidence magnitude in characterizing reasoning reliability.

![](images/4fdb835971772461d10cb516ea99c4e2f9045e517d65dfb89d93c706a89c8ada.jpg)  
Figure 2: Answer accuracy for low- and high-variance query groups defined by the median trajectory-confidence variance.

Guided by this observation, we propose Test-Time Adaptation via Stability-Aware Confidence Optimization (TASCO), a label-free framework for guiding frozen LLMs toward reliable high-confidence reasoning. Concretely, TASCO optimizes a task-level prefix shared across unlabeled test queries to stabilize confidence during reasoning. Once optimized, this prefix is prepended to each query embedding, thereby steering the frozen model throughout trace generation. Since local stability of reasoning confidence can manifest as either variation across nearby perturbed prefixes or sensitivity to the most adversely perturbed prefix, TASCO operationalizes it using two complementary strategies: Random Perturbation measures distributional stability through confidence variation across trajectories independently decoded under Gaussian-perturbed prefixes, while Sharpness-Aware Perturbation targets worst-case local sensitivity by applying a single gradient-guided perturbation along the most sensitive direction in the prefix space for the current trajectory. By optimizing the prefix with these two schemes, TASCO improves accuracy by up to 17.2% while generating up to 28.1% fewer tokens across diverse LLMs and reasoning benchmarks. Behavioral analyses show TASCO promotes stable confidence under local perturbations without prematurely narrowing reasoning directions. The learned prefix improves reasoning on unseen datasets without further optimization.

Our contributions are summarized as follows:

• We introduce a stability-aware perspective on confidencedriven test-time optimization, identifying local stability as a complementary signal for assessing the reliability of high-confidence reasoning.

• We propose TASCO, a label-free framework that uses two alternative perturbation strategies to guide frozen LLMs toward high-confidence states that remain stable under local perturbations. We further characterize how both strategies control local prefix sensitivity.

• Extensive experiments across diverse LLMs and reasoning benchmarks demonstrate that TASCO improves reasoning accuracy and eficiency while promoting stable confidence under local perturbations.

## Related Work

## Reasoning in Large Language Models

Chain-of-thought (CoT) prompting (Wei et al. 2022) improves LLM reasoning by eliciting explicit intermediate steps. Building on this paradigm, subsequent methods enhance reasoning through trajectory aggregation (Wang et al. 2022), structured search (Yao et al. 2023), and step-level verification (Zhang et al. 2025b). Reasoning models further extend explicit CoT by performing longer inference-time deliberation (Chen et al. 2026b). In parallel, continuous reasoning methods move part of the intermediate computation from token space into latent space. COCONUT (Hao et al. 2024) and CODI (Shen et al. 2025), for example, replace explicit reasoning tokens with latent states, while SoftCoT (Xu et al. 2025b) and SemCoT (He et al. 2026) use auxiliary modules to construct continuous rationales that are subsequently aligned with and injected into the target model (Wei et al. 2025). Together, these methods establish latent states as an interface for steering reasoning without backbone updates.

## Test-Time Adaptation for Reasoning

Test-time adaptation improves LLM reasoning using inference-time signals. TTRL (Zuo et al. 2026) derives majority-vote pseudo-rewards from rollouts. Entropy-based approaches either use negative token entropy for policy optimization (Prabhudesai et al. 2025) or minimize output entropy to adapt model parameters on unlabeled samples (Gao et al. 2025). Such updates add computation and risk catastrophic forgetting (Hu et al. 2025a). Recent methods instead optimize continuous variables while freezing the model. Some rely on external verifiers or reward models to guide intermediate reasoning states (Han et al. 2026; Chen et al. 2025), whereas others use model-derived entropy to optimize continuous vectors (Kang, Shi, and Chen 2026; Ye, Liang, and Shan 2025). MTI selectively guides high-entropy tokens during decoding (Yang et al. 2026). However, confidencedriven methods rarely assess whether the resulting states are reliable, despite evidence that LLM confidence can be misaligned with correctness (Agarwal et al. 2026). Perturbationbased studies instead use sensitivity diagnostically: CCPS predicts answer correctness from changes in hidden representations (Khanmohammadi et al. 2025), while Wen et al. identify uncertain reasoning steps through embedding perturbations (Wen et al. 2026). Unlike these approaches, TASCO optimizes local stability to promote reliable reasoning.

## Method

In this section, we present TASCO, a stability-aware framework that optimizes a continuous prefix at test time while keeping the LLM frozen. We first formulate the optimization problem and define the confidence objective. We then introduce two alternative strategies for characterizing local stability from distributional and worst-case perspectives. We further analyze how these strategies guide confidence optimization. Figure 3 provides an overview of TASCO.

![](images/f4ac2806800f85cd68588aaa3b72bb9590e0237645ff8b20c679bff5298898cb.jpg)  
Figure 3: Overview of TASCO. TASCO optimizes a shared prefix for confidence and stability. Random Perturbation penalizes local confidence variance, while Sharpness-Aware Perturbation targets worst-case sensitivity.

## Preliminaries

Problem Formulation. Let $M _ { \theta }$ denote a pretrained LLM with frozen parameters θ, and let $\mathcal { D } _ { \mathrm { t e s t } } = \{ X _ { 1 } , . . . , X _ { N } \}$ be an unlabeled test set. Our goal is to improve the reasoning performance of $M _ { \theta }$ on $\mathcal { D } _ { \mathrm { t e s t } }$ without updating its parameters. To this end, we optimize a shared continuous prefix $V ~ \in ~ \mathbb { R } ^ { L \times d }$ . Here, L and d denote the prefix length and embedding dimension. Given a query X with token embeddings $E ( { \bar { X } } )$ , we prepend V to form the prefixconditioned input $E ^ { \prime } { \dot { ( X ) } } = [ { \dot { V } } ; { \dot { E } } ( X ) ]$ , using [·; ·] to denote concatenation along the sequence dimension. The frozen model then autoregressively generates a reasoning trajectory $Y = ( y _ { 1 } , \dots , { y _ { T } } )$ . At decoding step $t ,$ its next-token distribution is $p _ { \theta } ( \cdot \mid E ^ { \prime } ( X ) , y _ { < t } )$ . TASCO optimizes V using only label-free signals derived from $M _ { \theta } .$ , without external supervision or auxiliary models.

Confidence Objective. We use predictive entropy as a label-free confidence signal. For a generated trajectory $Y =$ $( y _ { 1 } , \dots , y _ { T } )$ , the confidence loss is defined as the average token-level entropy:

$$
\mathcal { L } _ { \mathrm { c o n f } } ( V ; X , Y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } H \left( p _ { \theta } ( \cdot \mid E ^ { \prime } ( X ) , y _ { < t } ) \right) ,\tag{1}
$$

where $\begin{array} { r } { H ( p ) = - \sum _ { w \in \mathcal { V } } p ( w ) \log p ( w ) } \end{array}$ denotes the entropy over the vocabulary V. In practice, we compute this loss using one generated trajectory per input and treat the generated tokens as fixed.

## Confidence Optimization with Local Stability

Although entropy minimization ofers a label-free objective for prefix optimization, it may steer the prefix toward a locally fragile high-confidence state. TASCO therefore introduces two local-stability strategies: (1) Random Perturbation for distributional stability and (2) Sharpness-Aware Perturbation for worst-case directional stability.

Random Perturbation. We promote distributional stability by examining reasoning trajectories decoded under nearby prefixes. At each optimization step, we draw K Gaussian perturbations around the current prefix:

$$
V ^ { ( k ) } = V + \epsilon ^ { ( k ) } , \quad \epsilon ^ { ( k ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) ,\tag{2}
$$

where σ is the standard deviation of the Gaussian noise applied to each prefix element. For each input X, the model greedily decodes a trajectory $Y ^ { ( k ) }$ under every perturbed prefix $\bar { V } ^ { ( k ) }$ . We summarize each rollout by the average loglikelihood of its generated tokens:

$$
c ^ { ( k ) } = \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } \log p _ { \theta } \left( y _ { t } ^ { ( k ) } \mid E ^ { \prime ( k ) } ( X ) , y _ { < t } ^ { ( k ) } \right) .\tag{3}
$$

Variation in $\{ c ^ { ( k ) } \} _ { k = 1 } ^ { K }$ reflects local behavioral instability and may capture trajectory switching because each perturbed prefix is re-decoded. We quantify the behavioral sharpness of V on X by the sample variance:

$$
\bar { S } ( V ; X ) = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } \left( c ^ { ( k ) } - \bar { c } \right) ^ { 2 } , \quad \bar { c } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } c ^ { ( k ) } .\tag{4}
$$

For a mini-batch B, the random-perturbation regularizer is obtained by averaging this quantity across inputs, $\begin{array} { r } { \mathcal { L } _ { \mathrm { r a n d } } ( V ) = | \mathcal { B } | ^ { - \bar { 1 } } \sum _ { X \in \mathcal { B } } \breve { S } ( V ; X ) } \end{array}$ . Combining it with the batch-averaged confidence loss gives:

$$
{ \mathcal { L } } _ { \mathrm { R P } } = { \mathcal { L } } _ { \mathrm { c o n f } } ( V ) + \lambda _ { \mathrm { r a n d } } { \mathcal { L } } _ { \mathrm { r a n d } } ( V ) ,\tag{5}
$$

where $\lambda _ { \mathrm { { r a n d } } }$ balances predictive confidence and local stability. Thus, ${ \mathcal { L } } _ { \mathrm { c o n f } }$ encourages low predictive entropy, whereas $\mathcal { L } _ { \mathrm { r a n d } }$ reduces behavioral variation across nearby prefixes.

Sharpness-Aware Perturbation. Unlike Random Perturbation, Sharpness-Aware Perturbation avoids generating multiple trajectories by constructing a single gradient-guided perturbation. At each update, we fix the trajectory Y decoded under V while minimizing the worst-case confidence loss:

$$
\operatorname* { m i n } _ { V } \operatorname* { m a x } _ { \| \epsilon \| _ { F } \leq \rho } { \mathcal { L } } _ { \mathrm { c o n f } } ( V + \epsilon ; X , Y ) ,\tag{6}
$$

where $\rho$ defines the perturbation radius under the Frobenius norm. Since the inner maximization in Eq. 6 is generally intractable, we use a first-order approximation of the confidence loss around V, based on

$$
g = \nabla _ { V } { \mathcal { L } } _ { \mathrm { c o n f } } ( V ; X , Y ) .\tag{7}
$$

The linearized inner problem yields the following worstcase perturbation and perturbed prefix:

$$
\widehat V = V + \epsilon ^ { \star } , \quad \epsilon ^ { \star } = \rho \frac { g } { \| g \| _ { F } } ,\tag{8}
$$

where $\epsilon ^ { \star }$ is the perturbation along the steepest ascent direction under $\| \epsilon \| _ { \bar { F } } \leq \rho .$ We then evaluate the confidence loss at $\widehat { V }$ using the same trajectory $Y \colon$

$$
\mathcal { L } _ { \mathrm { S A P } } = \mathcal { L } _ { \mathrm { c o n f } } ( \widehat { V } ; X , Y ) .\tag{9}
$$

When updating V with $\mathcal { L } _ { \mathrm { S A P } }$ , we treat $\epsilon ^ { \star }$ as constant to avoid second-order derivatives through g.

The variants trade of neighborhood coverage and computation: Random Perturbation captures variation across sampled perturbations, whereas SAP targets worst-case sensitivity with one gradient-guided perturbation. Complete algorithms appear in Appendix D.

## Theoretical Analysis

We analyze how the two objectives control local confidence sensitivity to prefix perturbations. Random Perturbation captures confidence variation across perturbed rollouts, whereas SAP targets worst-case sensitivity along a fixed trajectory.

Random Perturbation For $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ , let $Z$ denote the greedy trajectory decoded under $V + \epsilon ,$ , and let $C =$ $c ( V ^ { \bar { + } } + \epsilon )$ denote its trajectory-level confidence. The sample variance in Eq. 23 is an unbiased estimator of the following population behavioral sharpness:

$$
\begin{array} { r } { S _ { \sigma } ( V ; X ) = \mathrm { V a r } _ { \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) } \left[ c ( V + \epsilon ) \right] . } \end{array}\tag{10}
$$

Because each perturbed prefix is independently re-decoded, the law of total variance gives:

$$
\begin{array} { r } { S _ { \sigma } ( V ; X ) = \mathbb { E } _ { Z } [ \mathrm { V a r } ( C \mid Z ) ] + \mathrm { V a r } _ { Z } ( \mathbb { E } [ C \mid Z ] ) . } \end{array}\tag{11}
$$

The first term captures variation among perturbations yielding the same trajectory, whereas the second captures variation in mean confidence across decoded trajectories, without diferentiating through decoding.

For the within-trajectory term, consider a local neighborhood in which greedy decoding remains Y, and let $\bar { f } _ { Y } ( V )$ denote the teacher-forced confidence of Y in Eq. 22. If $f _ { Y }$ is suficiently smooth, then, as $\sigma  0 \mathrm { \cdot }$

$$
\begin{array} { r } { \mathrm { V a r } _ { \epsilon } [ f _ { Y } ( V + \epsilon ) ] = \sigma ^ { 2 } \| \nabla _ { V } f _ { Y } ( V ) \| _ { F } ^ { 2 } + \mathcal { O } ( \sigma ^ { 4 } ) . } \end{array}\tag{12}
$$

Thus, away from stationary points, Random Perturbation penalizes first-order confidence sensitivity to isotropic perturbations. At a stationary point of $f _ { Y }$ , the corresponding expansion is:

$$
\mathrm { V a r } _ { \epsilon } [ f _ { Y } ( V + \epsilon ) ] = \frac { \sigma ^ { 4 } } { 2 } \| \nabla _ { V } ^ { 2 } f _ { Y } ( V ) \| _ { F } ^ { 2 } + o ( \sigma ^ { 4 } ) .\tag{13}
$$

The latter reflects aggregate local curvature, while Eq. 11 additionally captures between-trajectory variation.

Sharpness-Aware Perturbation For a trajectory Y held fixed within an update, define $\ell ( V ) = { \mathcal { L } } _ { \mathrm { c o n f } } ( V ; { \dot { X } } , Y )$ . Its worst-case local increase over a Frobenius ρ-ball is:

$$
\Delta _ { \rho } ( V ) = \operatorname* { m a x } _ { \| \epsilon \| _ { F } \leq \rho } \left[ \ell ( V + \epsilon ) - \ell ( V ) \right] .\tag{14}
$$

Assume that ℓ has a $\beta - 1$ Lipschitz-continuous gradient on this ball, and let $\boldsymbol { g } = \nabla _ { V } \ell ( V )$ . Then:

$$
| \Delta _ { \rho } ( V ) - \rho | | g | | _ { F } | \leq \frac { \beta \rho ^ { 2 } } { 2 } .\tag{15}
$$

Thus, the gradient norm provides a first-order approximation to the worst-case local increase. For $g \neq 0 ,$ , the perturbation $\epsilon ^ { \star }$ in Eq. 8 satisfies:

$$
| \ell ( V + \epsilon ^ { \star } ) - \ell ( V ) - \rho \| g \| _ { F } | \leq \frac { \beta \rho ^ { 2 } } { 2 } .\tag{16}
$$

Together, Eqs. 15 and 16 show that the induced increase $\ell ( \bar { V } + \epsilon ^ { \star } ) - \bar { \ell } ( V )$ approximates $\Delta _ { \rho } ( V )$ with $\mathcal { O } ( \rho ^ { 2 } )$ error.

Separately, if V is a stationary point and ℓ is twice continuously diferentiable in a neighborhood of V , the ideal inner problem admits the following expansion as $\rho  0 \mathrm { : }$

$$
\Delta _ { \rho } ( V ) = \frac { \rho ^ { 2 } } { 2 } \left[ \lambda _ { \mathrm { m a x } } \bigl ( \nabla _ { V } ^ { 2 } \ell ( V ) \bigr ) \right] _ { + } + o ( \rho ^ { 2 } ) ,\tag{17}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ . Thus, the leading term of the ideal worst-case increase is governed by the largest positive local curvature.

Together, the two objectives control local confidence sensitivity under sampled and worst-case perturbations. These results characterize robustness rather than correctness; the empirical relationship between stability and answer accuracy is examined in Appendix B.

## Experiments

## Experimental Setup

Models. We evaluate TASCO on five open-source LLMs covering both general and reasoning models. General LLMs include Qwen2.5-Math (Yang et al. 2024) in two sizes (1.5B and 7B) and LLaMA3.1-8B-Instruct (Grattafiori et al. 2024). Reasoning LLMs include DeepSeek-R1-Distill-Qwen in two sizes (1.5B and 7B) (Guo et al. 2025).

Datasets. We evaluate on six reasoning benchmarks. For mathematics, we use MATH-500 (Hendrycks et al. 2021), AMC23, AIME24 (Li et al. 2024), and AIME25 (Balunovic et al. 2026), spanning high-school to olympiad-level problems. For science, we use Minerva Math (Lewkowycz et al. 2022), with undergraduate-level STEM problems, and GPQA Diamond (Rein et al. 2023), with graduate-level questions written by domain experts.

<table><tr><td>Model</td><td>Method</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>AIME24</td><td>GPQA</td><td>Avg.</td><td>∆</td></tr><tr><td rowspan="7">Qwen2.5-Math-1.5B</td><td>Zero-Shot CoT</td><td>39.4</td><td>34.7</td><td>8.8</td><td>6.7</td><td>20.2</td><td>22.0</td><td></td></tr><tr><td>SLOT</td><td>43.4</td><td>47.2</td><td>16.9</td><td>10.0</td><td>26.8</td><td>28.9</td><td>+6.9</td></tr><tr><td>LatentSeek</td><td>45.6</td><td>43.4</td><td>14.3</td><td>7.5</td><td>22.2</td><td>26.6</td><td>+4.6</td></tr><tr><td>LTPO</td><td>58.6</td><td>45.6</td><td>12.9</td><td>9.2</td><td>21.7</td><td>29.6</td><td>+7.6</td></tr><tr><td>TTSV</td><td>68.6</td><td>46.8</td><td>20.6</td><td>8.8</td><td>24.8</td><td>33.9</td><td>+11.9</td></tr><tr><td>TASCO-RP</td><td>70.8</td><td>50.6</td><td>22.4</td><td>10.4</td><td>27.8</td><td>36.4</td><td>+14.4</td></tr><tr><td>TASCO-SAP</td><td>71.8</td><td>52.2</td><td>21.3</td><td>12.1</td><td>26.8</td><td>36.8</td><td>+14.8</td></tr><tr><td rowspan="7">Qwen2.5-Math-7B</td><td>Zero-Shot CoT</td><td>52.2</td><td>44.4</td><td>12.9</td><td>14.6</td><td>31.8</td><td>31.2</td><td></td></tr><tr><td>SLOT</td><td>58.8</td><td>56.2</td><td>29.4</td><td>18.8</td><td>32.8</td><td>39.2</td><td>+8.0</td></tr><tr><td>LatentSeek</td><td>57.6</td><td>48.1</td><td>20.5</td><td>6.2</td><td>32.8</td><td>33.0</td><td>+1.8</td></tr><tr><td>LTPO</td><td>65.6</td><td>42.8</td><td>17.6</td><td>18.8</td><td>33.5</td><td>35.7</td><td>+4.5</td></tr><tr><td>TTSV</td><td>71.0</td><td>59.1</td><td>33.5</td><td>17.9</td><td>36.4</td><td>43.6</td><td>+12.4</td></tr><tr><td>TASCO-RP</td><td>75.8</td><td>68.8</td><td>37.1</td><td>21.2</td><td>37.4</td><td>48.1</td><td>+16.9</td></tr><tr><td>TASCO-SAP</td><td>73.4</td><td>66.2</td><td>39.1</td><td>24.2</td><td>38.9</td><td>48.4</td><td>+17.2</td></tr><tr><td rowspan="7">Llama3.1-8B-Instruct</td><td>Zero-Shot CoT</td><td>47.4</td><td>24.7</td><td>22.1</td><td>7.5</td><td>28.3</td><td>26.0</td><td></td></tr><tr><td>SLOT</td><td>48.8</td><td>23.8</td><td>22.8</td><td>9.2</td><td>33.8</td><td>27.7</td><td>1 +1.7</td></tr><tr><td>LatentSeek</td><td>49.4</td><td>26.8</td><td>23.2</td><td>3.3</td><td>32.3</td><td>27.0</td><td></td></tr><tr><td>LTPO</td><td>48.4</td><td>27.5</td><td>20.5</td><td>11.3</td><td>30.5</td><td>27.6</td><td>+1.0 +1.6</td></tr><tr><td>TTSV</td><td>47.8</td><td>22.8</td><td>22.1</td><td>6.2</td><td>27.8</td><td>25.3</td><td>-0.7</td></tr><tr><td>TASCO-RP</td><td>46.8</td><td>29.1</td><td>25.0</td><td>10.4</td><td>33.3</td><td>28.9</td><td>+2.9</td></tr><tr><td>TASCO-SAP</td><td>50.2</td><td>28.4</td><td>26.5</td><td>11.7</td><td>31.8</td><td>29.7</td><td>+3.7</td></tr></table>

Table 1: Main performance comparison across three models and five reasoning benchmarks (accuracy, %). ∆ denotes the absolute percentage-point change relative to Zero-Shot CoT. Best results are in bold; second-best results are underlined.

Baselines. For general LLMs, we include CoT (Wei et al. 2022) as the standard prompting baseline. We compare with SLOT (Hu et al. 2025b) and LatentSeek (Li et al. 2025), which perform test-time adaptation through final-layer or latent representations. We also compare with LTPO (Ye, Liang, and Shan 2025) and TTSV (Kang, Shi, and Chen 2026), which optimize continuous latent variables using modelinternal confidence signals while keeping the backbone frozen. For reasoning LLMs, we compare with s1 (Muennighof et al. 2025), CoD (Xu et al. 2025a), and α1 (Zhang et al. 2025a), which control reasoning behavior at test time.

Implementation Details. For general LLMs, we use a shared prefix of length L = 20 and a maximum generation length of 3,072 tokens. For reasoning-enhanced LLMs, we use L = 10 and allow up to 8,192 tokens for longer reasoning traces. We optimize the prefix using AdamW with a batch size of 16. Random Perturbation uses K = 8 perturbed rollouts with σ = 0.01, while Sharpness-Aware Perturbation uses a perturbation radius of $\rho = 0 . 8$ . We further analyze sensitivity to σ and ρ. All experiments run on a single NVIDIA A800 GPU. More details are provided in Appendix A.

Evaluation. We report accuracy on all benchmarks. For mathematical reasoning benchmarks, we extract the final answer and determine correctness using dataset-specific normalization and equivalence rules. For AMC23 and AIME24, we independently sample eight responses for each problem and report the mean accuracy across the 8 generations.

## Main Results

Tables 1 and 2 compare TASCO with baselines on general and reasoning-enhanced LLMs, respectively, showing accuracy gains across both model types. Furthermore, steering the model toward high-confidence reasoning states yields shorter outputs, improving token eficiency.

Performance Improvement. Tables 1 and 2 show TASCO improves both general and reasoning-enhanced LLMs. On Qwen2.5-Math-1.5B and Qwen2.5-Math-7B, TASCO raises average accuracy from 22.0% to 36.8% and from 31.2% to 48.4%, gains of 14.8 and 17.2 percentage points over Zero-Shot CoT. TASCO further outperforms TTSV by 2.9 and 4.8 points, respectively. TASCO also improves LLaMA3.1-8B-Instruct from 26.0% to 29.7%, demonstrating applicability beyond Qwen. On reasoning-enhanced models, TASCO improves the four-benchmark average from 45.8% to 52.6% on DeepSeek-R1-Distill-Qwen-1.5B and from 59.5% to 65.5% on DeepSeek-R1-Distill-Qwen-7B, gains of 6.8 and 6.0 percentage points. It surpasses the strongest reasoning-control baseline by 4.3 and 2.5 points, respectively. Overall, incorporating local stability into confidence optimization improves test-time reasoning across model families and reasoning paradigms.

Token Eficiency. TASCO also improves token eficiency. Figure 4 shows that TASCO-R and TASCO-S reduce average generation length by 28.1% and 24.0%, respectively, across general LLMs compared with CoT. Similar gains on reasoning-enhanced LLMs are reported in Appendix E. These reductions lower inference cost and suggest that TASCO improves accuracy without requiring longer reasoning traces. Instead, the optimized prefix promotes concise, reliable reasoning while improving accuracy.

## Analyses of TASCO

Ablation of Prefix Optimization. We exam whether TASCO’s gains arise from using local stability to improve confidence-guided reasoning. CoT-Unk and Confidence Only test whether additional prefix positions or confidence optimization alone sufice. Perturbed Confidence uses the same K trajectories as TASCO-RP but removes the stability objective, isolating additional rollout computation; Stability Only removes the confidence objective, testing whether stability regularization alone explains the gains. As shown in Fig. 5, TASCO-RP outperforms both controlled variants. Together with the improvements of both TASCO variants over Confidence Only, these results show that the gains arise neither from additional computation nor from stability regularization in isolation, but from using local stability to guide optimization toward reliable high-confidence reasoning.

![](images/89c96ad2656e415cb7398bbc53643f5730736c96c1a3abdeadb541527d4cd3fb.jpg)

![](images/e7ceb16ca833c8bf26b1e447d43b7359e8ef5122f7c04dd66d94fb024575bbc4.jpg)

![](images/5bfa4c6743d108c9a6ad6b7c91dfb9bfb7a0b11a5f1ee6c3eec237d474ccd0e0.jpg)  
Figure 4: Accuracy-token trade-of across general LLMs, averaged over all benchmarks.

![](images/b9dd9f06287c6ea5338ab51cafdbb45e4c1aa95b2d6a11de92648fdb18bf4dd3.jpg)

![](images/9baa804e3641254d7d497fb088d72c3e7660eaae97c31a540c19843db525251b.jpg)

![](images/1acd9d44f8cf653a4af094e768dc8d64b01b462e806001f2ba14cb397635d8e3.jpg)

![](images/4bac33a409e4b4fcbcadd34ab36325318cf5bd706f596661510dbd30b8ee8998.jpg)  
Figure 5: Impact of optimized prefixes on MATH500 and Minerva. The results indicate that the gains of TASCO come from learning stability-aware task-level prefixes rather than simply adding extra prefix positions.

<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td></tr><tr><td colspan="5">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Base</td><td>22.8</td><td>22.5</td><td>59.4</td><td>78.6</td></tr><tr><td>s1</td><td>25.8</td><td>25.4</td><td>61.9</td><td>80.2</td></tr><tr><td>CoD</td><td>26.7</td><td>20.8</td><td>66.2</td><td>79.4</td></tr><tr><td>α1</td><td>30.8</td><td>12.9</td><td>63.1</td><td>81.0</td></tr><tr><td>TASCO</td><td>30.4</td><td>29.2</td><td>69.4</td><td>81.4</td></tr><tr><td colspan="5">DeepSeek-R1-Distill-Qwen-7B</td></tr><tr><td>Base</td><td>44.2</td><td>25.0</td><td>79.4</td><td>89.4</td></tr><tr><td>s1</td><td>37.5</td><td>24.2</td><td>78.1</td><td>89.0</td></tr><tr><td>CoD</td><td>45.0</td><td>33.3</td><td>85.6</td><td>88.0</td></tr><tr><td>α1</td><td>36.7</td><td>32.5</td><td>90.6</td><td>89.8</td></tr><tr><td>TASCO</td><td>49.2</td><td>34.2</td><td>88.1</td><td>90.4</td></tr></table>

Table 2: Evaluation on reasoning-enhanced LLMs using DeepSeek-R1-Distill models as backbones. TASCO results are obtained with the TASCO-SAP variant.

Behavioral Efects of Stability-Aware Optimization. We examine how TASCO afects local robustness and confidence formation on Qwen2.5-Math-7B. To assess local robustness, we add Gaussian noise with σ = 0.01 to each learned prefix and generate K = 8 perturbed rollouts per input. As shown in Table 3, confidence-only optimization produces higher trajectory-confidence variance and lower answer consistency across datasets, whereas TASCO reduces this variation and improves consistency. We further analyze confidence formation on MATH500 by tracking the mean next-token entropy and top-1 probability over the first 5% of decoding steps. Figure 7 shows that confidence-only optimization rapidly concentrates probability mass early in generation, whereas TASCO retains higher entropy and lower top-1 probability before the gap narrows later. Together, these results show that TASCO produces reasoning behavior that is more stable under local perturbations and less prone to premature confidence concentration.

Analysis of Cross-Distribution Generalization. To assess the transferability of the guidance learned by TASCO, we optimize a prefix on one source benchmark and directly apply it to target benchmarks without re-optimization. As shown in Fig. 6, the transferred prefixes consistently outperform Zero-Shot CoT across all evaluated source-target pairs, indicating that the learned guidance is not confined to its optimization distribution. This transferability stems from TASCO’s stability-aware objective: by promoting confidence that remains stable under local perturbations, TASCO guides the model to learn reliable reasoning patterns shared across mathematical tasks rather than merely increasing confidence on the source distribution.

![](images/81597316ccdc26808a1c10077669fccfced3230b8489e76cdcefe778e4aa8bc9.jpg)  
(a) Source: MATH

![](images/4d7fc48ca920da7ec4471f28f07a975fc6c33cdd24e8e1bd50e161f470f67d6c.jpg)  
(b) Source: Minerva

![](images/44b4972e76069e98f686fee5a6db8e9c79f7ed6cbc48bf89cbf834be8eb4e11d.jpg)  
(c) Source: AIME24

![](images/3975df59b8dfce2c9a0fdb87d94ad768fd185d93b41f60dac88fbf6fa7f6ebd9.jpg)  
(d) Source: AMC23

Figure 6: Cross-distribution generalization of TASCO on Qwen2.5-Math-7B, measured by accuracy gains over Zero-Shot CoT.
<table><tr><td>Method</td><td colspan="3">MATH500 AMC23 Minerva GPQA</td></tr><tr><td></td><td colspan="3">Confidence Variance  $( \times 1 0 ^ { - 4 } ) \downarrow$ </td></tr><tr><td>Confidence Only</td><td>4.83 3.05</td><td>6.59</td><td>15.76</td></tr><tr><td>TASCO-RP</td><td>2.33 2.34</td><td>5.48</td><td>9.31</td></tr><tr><td>TASCO-SAP</td><td>2.89 2.32</td><td>4.51</td><td>12.26</td></tr><tr><td colspan="4">Answer Consistency ↑</td></tr><tr><td>Confidence Only</td><td>62.9 42.5</td><td>58.9</td><td>48.1</td></tr><tr><td>TASCO-RP</td><td>71.7 47.0</td><td>62.6</td><td>59.4</td></tr><tr><td>TASCO-SAP</td><td>73.8 50.5</td><td>66.6</td><td>52.8</td></tr></table>

Table 3: Local stability under prefix perturbations on Qwen2.5-Math-7B. Lower variance and higher consistency indicate more stable reasoning behavior.

![](images/448a11381e5028be46b578ed571e8289a3e02a2ab0d8668ebb7875e322045b2b.jpg)  
Normalized Position (\%)

![](images/d8b1d00adff3ceeaf38c0cd7b1be6f6daa9915447e19f658ce508ac210bd2cf2.jpg)  
Figure 7: Mean next-token entropy (left) and top-1 probability (right) over the first 5% of decoding steps for Qwen2.5- Math-7B on MATH500.

Performance Across Query Dificulty. We first estimate query dificulty from eight responses sampled by the unadapted model: queries answered correctly 0–2, 3–5, and 6–8 times are categorized as hard, medium, and easy, respectively. The resulting groups contain 195, 130, and 175 queries for Qwen2.5-Math-7B and 216, 92, and 192 queries for LLaMA3.1-8B-Instruct. We then compare TASCO with confidence-only optimization within the same groups. As shown in Table 4, TASCO yields clear gains on hard and medium queries across both models while maintaining comparable performance on easy queries. These results suggest that TASCO does more than reinforce solutions the model already finds easy; it learns reasoning guidance that supports more reliable reasoning on challenging problems.

Sensitivity to Perturbation Scale. We evaluate sensitivity to the perturbation scale on Qwen2.5-Math-1.5B with MATH500. For Random Perturbation, we vary the Gaussian scale σ; for Sharpness-Aware Perturbation, we vary the radius ρ, while fixing all other settings. Figure 8 shows that both variants outperform confidence-only optimization across a broad range of nonzero scales. Accuracy peaks near the selected defaults and decreases only when the perturbation becomes excessively large, indicating that TASCO does not rely on narrow hyperparameter tuning. Additional sensitivity analyses are provided in Appendix E.

<table><tr><td rowspan="2">Difficulty</td><td colspan="2">Qwen2.5-Math-7B</td><td colspan="2">LLaMA3.1-8B</td></tr><tr><td>Confidence</td><td>TASCO</td><td>Confidence</td><td>TASCO</td></tr><tr><td>Hard</td><td>46.7</td><td>54.4</td><td>9.7</td><td>14.4</td></tr><tr><td>Medium</td><td>75.4</td><td>80.8</td><td>48.9</td><td>54.3</td></tr><tr><td>Easy</td><td>94.9</td><td>96.0</td><td>90.1</td><td>88.5</td></tr><tr><td>Overall</td><td>71.0</td><td>75.8</td><td>47.8</td><td>50.2</td></tr></table>

Table 4: Accuracy (%) of confidence-only optimization and TASCO-RP across MATH500 dificulty groups.

![](images/9468773fadd4b5c520542e6d44f91ca2d647cb46824dccf292cfae05dd1c4422.jpg)  
Figure 8: Sensitivity of TASCO to the perturbation scale on Qwen2.5-Math-1.5B with MATH500.

## Conclusion

We observed that high-confidence reasoning is more likely to yield correct answers when it remains stable under local perturbations. Based on this observation, we proposed TASCO, a label-free framework that optimizes a shared task-level prefix for confidence and stability while keeping the LLM frozen. Random Perturbation captures distributional variation across nearby behaviors, whereas Sharpness-Aware Perturbation targets worst-case local sensitivity. Experiments across diverse LLMs and reasoning benchmarks show that stability-aware confidence optimization improves reasoning accuracy and generation eficiency without parameter updates or external supervision.

## References

Agarwal, S.; Zhang, Z.; Yuan, L.; Han, J.; and Peng, H. 2026. The unreasonable efectiveness of entropy minimization in llm reasoning. Advances in Neural Information Processing Systems, 38: 107150–107180.

Balunovic, M.; Dekoninck, J.; Petrov, I.; Jovanović, N.; and Vechev, M. 2026. Matharena: Evaluating llms on uncontaminated math competitions. Advances in Neural Information Processing Systems, 38.

Chen, G.; Niu, S.; Chen, D.; Yang, J.; Zhang, Z.; Tan, M.; Wu, P.; and Shen, Z. 2026a. ZeroSiam: An Eficient Asymmetry for Test-Time Entropy Optimization without Collapse. In The Fourteenth International Conference on Learning Representations.

Chen, Q.; Qin, L.; Liu, J.; Peng, D.; Guan, J.; Wang, P.; Hu, M.; Zhou, Y.; Gao, T.; and Che, W. 2026b. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. Science China Information Sciences, 69(6): 161101.

Chen, R.; Zhang, Z.; Hong, J.; Kundu, S.; and Wang, Z. 2025. Seal: Steerable reasoning calibration of large language models for free.

Gao, Z.; Chen, L.; Luo, H.; Zhou, J.; and Dai, B. 2025. One-shot entropy minimization.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

Han, P.; Xu, X.; Xuan, K.; Song, P.; Ouyang, S.; Tian, R.; Jiang, Y.; Qian, C.; Jiang, P.; Sun, J.; et al. 2026. Steer2Adapt: Dynamically Composing Steering Vectors Elicits Eficient Adaptation of LLMs.

Hao, S.; Sukhbaatar, S.; Su, D.; Li, X.; Hu, Z.; Weston, J.; and Tian, Y. 2024. Training large language models to reason in a continuous latent space.

He, Y.; Zheng, W.; Zhu, Y.; Zheng, Z.; Su, L.; Vasudevan, S.; Guo, Q.; Hong, L.; and Li, J. 2026. SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens. Advances in Neural Information Processing Systems, 38: 43455–43485.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring mathematical problem solving with the math dataset.

Hu, J.; Zhang, Z.; Chen, G.; Wen, X.; Shuai, C.; Luo, W.; Xiao, B.; Li, Y.; and Tan, M. 2025a. Test-time learning for large language models.

Hu, Y.; Zhang, X.; Fang, X.; Chen, Z.; Wang, X.; Zhang, H.; and Qi, G. 2025b. Slot: Sample-specific language model optimization at test-time.

Huang, C.; Huang, L.; Leng, J.; Liu, J.; and Huang, J. 2025. Eficient test-time scaling via self-calibration.

Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. Openai o1 system card.

Jurayj, W.; Cheng, J.; and Van Durme, B. 2025. Is that your final answer? test-time scaling improves selective question answering. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 636–644.

Kang, X.; Shi, D.; and Chen, L. 2026. Model Whisper: Steering Vectors Unlock Large Language Models’ Potential in Test-Time. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 31392–31400.

Keskar, N. S.; Mudigere, D.; Nocedal, J.; Smelyanskiy, M.; and Tang, P. T. P. 2016. On large-batch training for deep learning: Generalization gap and sharp minima.

Khalifa, M.; Agarwal, R.; Logeswaran, L.; Kim, J.; Peng, H.; Lee, M.; Lee, H.; and Wang, L. 2025. Process reward models that think.

Khanmohammadi, R.; Miahi, E.; Mardikoraem, M.; Kaur, S.; Brugere, I.; Smiley, C.; Thind, K. S.; and Ghassemi, M. M. 2025. Calibrating LLM confidence by probing perturbed representation stability. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 10459–10525.

Lambert, N.; Morrison, J.; Pyatkin, V.; Huang, S.; Ivison, H.; Brahman, F.; Miranda, L. J. V.; Liu, A.; Dziri, N.; Lyu, S.; et al. 2024. Tulu 3: Pushing frontiers in open language model post-training.

Lewkowycz, A.; Andreassen, A.; Dohan, D.; Dyer, E.; Michalewski, H.; Ramasesh, V.; Slone, A.; Anil, C.; Schlag, I.; Gutman-Solo, T.; et al. 2022. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35: 3843–3857.

Li, H.; Li, C.; Wu, T.; Zhu, X.; Wang, Y.; Yu, Z.; Jiang, E. H.; Zhu, S.-C.; Jia, Z.; Wu, Y. N.; et al. 2025. Seek in the dark: Reasoning via test-time instance-level policy gradient in latent space. arXiv preprint arXiv:2505.13308.

Li, J.; Beeching, E.; Tunstall, L.; Lipkin, B.; Soletskyi, R.; Huang, S.; Rasul, K.; Yu, L.; Jiang, A. Q.; Shen, Z.; et al. 2024. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9): 9.

Muennighof, N.; Yang, Z.; Shi, W.; Li, X. L.; Fei-Fei, L.; Hajishirzi, H.; Zettlemoyer, L.; Liang, P.; Candès, E.; and Hashimoto, T. B. 2025. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 20286–20332.

Nguyen, T.; and Le, T. 2026. ATLAS: Adaptive Test-Time Latent Steering with External Verifiers for Enhancing LLMs Reasoning.

Prabhudesai, M.; Chen, L.; Ippoliti, A.; Fragkiadaki, K.; Liu, H.; and Pathak, D. 2025. Maximizing confidence alone improves reasoning.

Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2023. Gpqa: A graduate-level google-proof q&a benchmark.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

Shen, Z.; Yan, H.; Zhang, L.; Hu, Z.; Du, Y.; and He, Y. 2025. Codi: Compressing chain-of-thought into continuous space via self-distillation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 677– 693.

Sinii, V.; Gorbatovski, A.; Cherepanov, A.; Shaposhnikov, B.; Balagansky, N.; and Gavrilov, D. 2025. Steering llm reasoning through bias-only adaptation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 9213–9222.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang, S.; Chowdhery, A.; and Zhou, D. 2022. Self-consistency improves chain of thought reasoning in language models.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837.

Wei, X.; Liu, X.; Zang, Y.; Dong, X.; Cao, Y.; Wang, J.; Qiu, X.; and Lin, D. 2025. SIM-CoT: Supervised Implicit Chain-of-Thought.

Wen, Q.; Wang, J.; Nan, Y.; He, P.; Tandon, R.; and Xu, H. 2026. Embedding Perturbation may Better Reflect the Uncertainty in LLM Reasoning.

Xu, S.; Xie, W.; Zhao, L.; and He, P. 2025a. Chain of draft: Thinking faster by writing less.

Xu, Y.; Guo, X.; Zeng, Z.; and Miao, C. 2025b. Softcot: Soft chain-of-thought for eficient reasoning with llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 23336– 23351.

Yang, A.; Zhang, B.; Hui, B.; Gao, B.; Yu, B.; Li, C.; Liu, D.; Tu, J.; Zhou, J.; Lin, J.; et al. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement.

Yang, Z.; Zhang, M.; Chen, F.; Ding, G.; Hou, L.; Tao, X.; and Chen, Y.-C. 2026. Less is more: Improving llm reasoning with minimal test-time intervention. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 20124–20137.

Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Grifiths, T.; Cao, Y.; and Narasimhan, K. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36: 11809–11822.

Ye, W.; Liang, Y.; and Shan, L. 2025. Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization. arXiv preprint arXiv:2510.04182.

Zhang, J.; Dong, R.; Wang, H.; Ning, X.; Geng, H.; Li, P.; He, X.; Bai, Y.; Malik, J.; Gupta, S.; et al. 2025a. Alphaone: Reasoning models thinking slow and fast at test time. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 11340–11365.

Zhang, Y.; Wang, X.; Wu, L.; and Wang, J. 2025b. Enhancing chain of thought prompting in large language models via reasoning patterns. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 25985–25993.

Zhao, C.; Yang, E.; Liu, Y.; Zhao, J.; and Guo, G. 2026. ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning.

Zuo, Y.; Zhang, K.; Sheng, L.; Qu, S.; Cui, G.; Zhu, X.; Li, H.; Long, X.; Hua, E.; Qi, B.; et al. 2026. Ttrl: Testtime reinforcement learning. Advances in Neural Information Processing Systems, 38: 131459–131483.

## Appendix

## A. Experimental Details

This section provides the implementation details of TASCO and the compared baselines. We first describe the prefix optimization and perturbation procedures, followed by the baseline implementations and evaluation protocol.

Optimization and Generation Settings. We optimize only the continuous prefix V using AdamW while keeping all parameters of $M _ { \theta }$ frozen. During optimization, trajectories are generated using the model-specific temperature, top-p, and repetition penalty reported in Table 5. The confidence objective is computed from at most the first $T _ { \mathrm { o p t } }$ generated tokens, reported as “Conf. tokens.” Final evaluation uses greedy decoding subject to the corresponding maximum generation length.

Prefix Initialization. TASCO can be optimized directly from a freshly initialized prefix. We use model-familyspecific initialization. Prefixes for Qwen-based models, including DeepSeek-R1-Distill-Qwen, are sampled from $\mathcal { N } ( 0 , I )$ . For LLaMA models, we first encode the unlabeled optimization set and compute the per-dimension empirical mean and variance of the resulting token embeddings. Each prefix vector is then sampled from the Gaussian distribution parameterized by these statistics. Since perturbation-based stability optimization requires additional rollouts, applying it before the prefix has been guided toward higher-confidence reasoning may be ineficient. In addition to direct optimization from scratch, we also support an initial confidence-only stage of 10 epochs, followed by five epochs of TASCO optimization. All stages use the same learning rate reported in Table 5.

Perturbation Settings. For Random Perturbation, we draw $K = 8$ Gaussian perturbations with $\sigma = 0 . 0 1$ at each optimization step. We set the stability weight to $\lambda _ { \mathrm { r a n d } } = 2 0$ for Qwen2.5-Math and $\lambda _ { \mathrm { r a n d } } = 5$ for LLaMA3.1-Instruct and DeepSeek-R1-Distill-Qwen. For Sharpness-Aware Perturbation, we use a perturbation radius of $\rho = 0 . 8 .$ . The two variants are optimized and evaluated separately.

Trajectory Generation and Updates. At each optimization step, the model generates one trajectory for every input in the current mini-batch using the current prefix. The generated tokens are held fixed during backpropagation, and gradients are computed only with respect to V. New trajectories are generated after each prefix update. Random Perturbation independently decodes a trajectory under each perturbed prefix, whereas SAP evaluates its gradient-guided perturbation using the trajectory fixed within the current update.

## Baseline Implementation Details

We use the oficial implementations and follow the configurations reported by the original authors whenever available.

Chain-of-Thought. CoT elicits explicit step-by-step reasoning through natural-language instructions without modifying the model.

SLOT. SLOT adapts each test query by optimizing a sample-specific vector added to the model’s final hidden representations.

LatentSeek. LatentSeek uses policy gradients and selfgenerated rewards to iteratively optimize instance-specific latent reasoning representations.

LTPO. LTPO optimizes latent thought vectors at test time using an intrinsic confidence reward while keeping the LLM frozen.

TTSV. TTSV learns a task-level continuous prefix by minimizing predictive entropy over the unlabeled test set.

s1. s1 applies budget forcing to extend reasoning by inserting additional wait tokens before the model terminates its thinking process.

Chain of Draft. CoD encourages concise reasoning by prompting the model to express each intermediate step using only a few words.

α1. α1 controls the transition from slow to fast reasoning through a dynamically scheduled α-moment.

Prompt Design. We use fixed model-family-specific prompt templates throughout optimization and evaluation. Table 11 provides a complete example of the qwen25-math-cot prompt, the embedding-level prefix placement, and the resulting response format. The remaining templates are included in the released implementation.

Evaluation. For MATH500, Minerva Math, and GPQA, the main tables report results from a single run. For AMC23 and AIME24, whose smaller test sets produce greater accuracy fluctuations, we independently run each experiment with eight random seeds and report the mean accuracy.

## B. Exploring Local Confidence Stability

We investigate whether local confidence stability is associated with answer correctness and whether this relationship persists after accounting for confidence magnitude and query dificulty. We first establish the basic association and then examine these two alternative explanations.

## Experimental Setup.

We conduct the analysis on the full MATH500 benchmark using five models from three families: Qwen2.5-Math-1.5B and 7B, LLaMA3.1-8B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B and 7B. For each model, we use the final tasklevel prefix V obtained through confidence-only optimization. For every query $X _ { i } ,$ , we sample $K = 8$ Gaussian perturbations with $\sigma = 0 . 0 1$

$$
V ^ { ( k ) } = V + \epsilon ^ { ( k ) } , \qquad \epsilon ^ { ( k ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) .\tag{18}
$$

Under each perturbed prefix, the model greedily decodes a trajectory $Y _ { i } ^ { \overline { { ( k ) } } }$ . Its trajectory-level confidence is defined as the length-normalized log-probability:

$$
c _ { i } ^ { ( k ) } = \frac { 1 } { T _ { i } ^ { ( k ) } } \sum _ { t = 1 } ^ { T _ { i } ^ { ( k ) } } \log p _ { \theta } \left( y _ { i , t } ^ { ( k ) } \mid X _ { i } , V ^ { ( k ) } , y _ { i , < t } ^ { ( k ) } \right) ,\tag{19}
$$

<table><tr><td>Model family</td><td>Prefix length</td><td>Batch size</td><td>Learning rate</td><td>Temp.</td><td>Top-p</td><td>Rep. penalty</td><td>Conf. tokens</td><td>Max. tokens</td></tr><tr><td>Qwen2.5-Math</td><td>20</td><td>16</td><td>1e-3</td><td>0.7</td><td>0.95</td><td>1.15</td><td>256</td><td>3,072</td></tr><tr><td>LLaMA3.1-Instruct</td><td>20</td><td>16</td><td>5e-6</td><td>0.7</td><td>0.95</td><td>1.05</td><td>256</td><td>3,072</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>10</td><td>16</td><td>5e-4</td><td>0.6</td><td>0.95</td><td>1.00</td><td>1,024</td><td>8,192</td></tr></table>

Table 5: Model-specific optimization and generation settings. “Conf. tokens” denotes the maximum number of generated tokens used to compute the confidence objective.

<table><tr><td>Model</td><td>Spearman&#x27;s ρ</td><td>AUROC</td></tr><tr><td>Qwen2.5-Math-1.5B</td><td>-0.369</td><td>0.668</td></tr><tr><td>Qwen2.5-Math-7B</td><td>-0.403</td><td>0.687</td></tr><tr><td>LLaMA3.1-8B-Instruct</td><td>-0.413</td><td>0.683</td></tr><tr><td>DS-Distill-Qwen-1.5B</td><td>-0.308</td><td>0.709</td></tr><tr><td>DS-Distill-Qwen-7B</td><td>-0.359</td><td>0.745</td></tr></table>

Table 6: Association between local confidence variance and unperturbed correctness. Spearman’s ρ measures the correlation between $s _ { i }$ and correctness, while AUROC uses $- { \mathcal { S } } _ { i }$ to distinguish correct from incorrect answers.

where $y _ { i , t } ^ { ( k ) }$ denotes the t-th token in $Y _ { i } ^ { ( k ) }$ and $T _ { i } ^ { ( k ) }$ is the total sequence length. We quantify local confidence stability using the sample variance:

$$
\mathcal { S } _ { i } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } \left( c _ { i } ^ { ( k ) } - \bar { c } _ { i } \right) ^ { 2 } , \qquad \bar { c } _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } c _ { i } ^ { ( k ) } ,\tag{20}
$$

where lower $s _ { i }$ indicates greater local stability, and $\bar { c } _ { i }$ denotes the mean confidence across the perturbed trajectories.

For each query, we additionally greedily decode a reference trajectory under the unperturbed prefix V . We evaluate its final answer correctness using the standard benchmark evaluator and compute its original trajectory-level confidence $c _ { i } ^ { ( 0 ) }$ following Eq. 19.

## Accuracy across Stability Groups

We first examine whether local confidence stability is associated with correctness at the unperturbed prefix. For each query, we estimate $\boldsymbol { S } _ { i }$ from perturbed trajectories and independently evaluate correctness using a trajectory decoded under the unperturbed prefix. We split queries at the modelspecific median of $\boldsymbol { S } _ { i }$ and report group accuracy with 95% bootstrap confidence intervals.

As shown in Figure 9, Low-Variance queries achieve higher unperturbed accuracy across all five models, with gaps ranging from 17.2 to 29.0 percentage points. This pattern holds for both general and reasoning-enhanced LLMs. Table 6 further shows consistently negative correlations between local variance and unperturbed correctness, with AU-ROCs ranging from 0.668 to 0.745. Importantly, stability is estimated from perturbed trajectories, whereas correctness is evaluated independently at the unperturbed prefix. The observed separation is therefore not merely induced by measuring variance and accuracy on the same rollouts. Instead, it suggests that neighborhood stability contains information about the reliability of the optimized prefix itself across model families.

<table><tr><td>Model</td><td>Gap</td><td> $p _ { \mathrm { p e r m } }$ </td><td>ρpartial</td><td>Ppartial</td></tr><tr><td>Qwen2.5-Math-1.5B</td><td>5.4</td><td>0.086</td><td>-0.108</td><td>0.128</td></tr><tr><td>Qwen2.5-Math-7B</td><td>11.6</td><td>&lt;0.001</td><td>-0.242</td><td>&lt;0.001</td></tr><tr><td>LLaMA3.1-8B</td><td>23.3</td><td>&lt;0.001</td><td>-0.283</td><td>&lt;0.001</td></tr><tr><td>DS-Distill-Qwen-1.5B</td><td>30.5</td><td>&lt;0.001</td><td>-0.560</td><td>&lt;0.001</td></tr><tr><td>DS-Distill-Qwen-7B</td><td>8.7</td><td>&lt;0.001</td><td>-0.297</td><td>&lt;0.001</td></tr></table>

Table 7: Confidence-controlled association between local variance and perturbation accuracy. $\mathrm { G a p } _ { \mathrm { s t r a t } }$ is the Lowminus High-Variance accuracy after five-way stratification by mean confidence. Partial Spearman correlation controls for mean confidence continuously.

## Controlling for Confidence Magnitude.

A remaining concern is that local variance may merely reflect confidence magnitude, as more confident queries may also be inherently more stable. To separate these efects, we divide queries into five equal-frequency strata according to their mean confidence across perturbed trajectories. Within each stratum, we split queries at the median of $\boldsymbol { S } _ { i }$ and compare their perturbation accuracy, thereby reducing confidence differences between the Low- and High-Variance groups. We estimate confidence intervals using stratified bootstrap sampling and assess the accuracy gap with a stratified permutation test. We additionally compute the partial Spearman correlation between local variance and perturbation accuracy while controlling for mean confidence.

As shown in Table 7, Low-Variance queries achieve higher perturbation accuracy after confidence stratification across all five models. The diferences are statistically significant on four models, with accuracy gaps ranging from 11.6 to 30.5 percentage points. Partial Spearman analysis yields the same pattern, showing significant negative correlations after controlling for mean confidence on these four models. These results suggest that local stability generally provides reliability information beyond confidence magnitude.

Qwen2.5-Math-1.5B is an exception, showing a nonsignificant 5.4-point gap and a near-zero partial correlation $( \rho = - 0 . 1 0 8 , p = 0 . 1 2 8 )$ . Its local variance is strongly correlated with mean confidence $( \rho = - 0 . 7 1 7 )$ , suggesting that variance provides little additional information beyond confidence magnitude for this model.

Overall, these analyses show a consistent association between local confidence stability and answer accuracy across model families. The relationship persists among queries with comparable mean confidence and within diferent dificulty groups, indicating that it cannot be explained solely by confidence magnitude or easier queries. These findings provide empirical motivation for incorporating local stability into confidence-based test-time optimization. The main experiments further examine whether optimizing this signal improves reasoning performance.

![](images/a67b3be76b3f0277ca852acf1b5ba71345c7701addb7f31cf8ac4194d5f5909b.jpg)  
Figure 9: Answer accuracy under the unperturbed prefix for low- and high-variance query groups across five models. Groups are defined by the model-specific median of $s _ { i }$ ; error bars show 95% bootstrap confidence intervals, and red annotations indicate accuracy gaps.

## C. Proofs and Additional Theoretical Analysis

The main paper presents the local-sensitivity results for Random Perturbation and Sharpness-Aware Perturbation. This section restates the quantities required by the proofs, makes the assumptions explicit, and provides the complete derivations. We distinguish behavioral confidence obtained after re-decoding from smooth confidence evaluated on a fixed trajectory, separating variation across decoded trajectories from local variation within a fixed decoding region.

## Setup and Assumptions

We identify the prefix $V \in \mathbb { R } ^ { L \times d }$ with a vector in $\mathbb { R } ^ { m }$ , where $m = L d$ . Under this vectorization, the Euclidean norm and inner product coincide with the Frobenius norm and inner product. For Random Perturbation, we draw K independent Gaussian perturbations:

$$
V ^ { ( k ) } = V + \epsilon ^ { ( k ) } , \qquad \epsilon ^ { ( k ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) .\tag{21}
$$

Under each perturbed prefix $V ^ { ( k ) }$ , the model greedily decodes a trajectory ${ \cal Y } ^ { ( k ) } = ( y _ { 1 } ^ { ( k ) } , \dots , y _ { T _ { k } } ^ { ( k ) } )$ . Following the notation in the main paper, its trajectory-level confidence is:

$$
c ^ { ( k ) } = \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } \log p _ { \theta } \left( y _ { t } ^ { ( k ) } \mid E ^ { \prime ( k ) } ( X ) , y _ { < t } ^ { ( k ) } \right) ,\tag{22}
$$

where $E ^ { \prime ( k ) } ( X ) = [ V ^ { ( k ) } ; E ( X ) ]$ denotes the input conditioned on $V ^ { ( k ) }$ . Let ${ \bar { c } } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } c ^ { ( k ) }$ . The Random Perturbation objective uses the sample variance:

$$
\mathcal { S } ( V ; X ) = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } \Big ( c ^ { ( k ) } - \bar { c } \Big ) ^ { 2 } .\tag{23}
$$

For the population analysis, draw $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ . Let Z denote the greedy trajectory under $V { + \epsilon }$ , and let $C = \operatorname { \dot { \it { c } } } ( V + \epsilon )$

denote its trajectory-level confidence, defined analogously to Eq. 22. Thus, each $c ^ { ( k ) }$ is an independent realization of C. We assume that $\mathbb { E } [ C ^ { 2 } ] < \infty$

For the smooth within-trajectory analysis, fix a trajectory $Y = ( y _ { 1 } , \dots , y _ { T } )$ and define its teacher-forced confidence as:

$$
f _ { Y } ( V ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \theta } \left( y _ { t } \mid E ^ { \prime } ( X ) , y _ { < t } \right) ,\tag{24}
$$

where $E ^ { \prime } ( X ) = [ V ; E ( X ) ]$ is the prefix-conditioned input.

We assume that V is an interior point of the decoding region associated with Y: there exists $r > 0$ such that every prefix $V + \delta$ satisfying $\| \delta \| _ { F } \le r$ greedily decodes Y. Let $g _ { Y } = \nabla _ { V } f _ { Y } ( V )$ and $H _ { Y } = \nabla _ { V } ^ { 2 } f _ { Y } ( V )$

We further assume that $f _ { Y }$ is suficiently smooth around V so that, for $U \sim { \mathcal { N } } ( 0 , I ) ;$

$$
\begin{array} { l } { { f _ { Y } ( V + \sigma U ) = f _ { Y } ( V ) + \sigma \langle g _ { Y } , U \rangle _ { \cal F } } } \\ { { \displaystyle ~ + \frac { \sigma ^ { 2 } } { 2 } \langle U , H _ { Y } U \rangle _ { \cal F } + R _ { \sigma } ( U ) , } } \end{array}\tag{25}
$$

where $\mathbb { E } [ R _ { \sigma } ( U ) ^ { 2 } ] = \mathcal { O } ( \sigma ^ { 6 } )$ . This condition follows, for example, from a locally bounded third derivative together with an integrable Taylor remainder. As $\sigma  0 .$ , the probability that a Gaussian perturbation leaves the fixed-decoding neighborhood decreases faster than any polynomial in σ. Under the same integrability condition, these tail events do not afect the polynomial orders derived below.

For Sharpness-Aware Perturbation, fix a trajectory Y within an update and let $\ell ( V ) = \mathcal { L } _ { \mathrm { c o n f } } ( V ; X , Y )$ denote the corresponding fixed-trajectory confidence loss. We assume that ℓ has a β-Lipschitz-continuous gradient on $\lbrace V + \epsilon :$ $\| \epsilon \| _ { F } \le \rho \}$ . The stationary-point result additionally assumes that ℓ is twice continuously diferentiable in a neighborhood of V.

## Random Perturbation

Proposition 1 (Population behavioral sharpness). Let $C _ { 1 } , \ldots , C _ { K }$ be the trajectory-level confidences induced by $K$ independent Gaussian perturbations, and let $S ( V ; X )$ be their sample variance in Eq. 23. Then:

$$
\begin{array} { r } { \mathbb { E } [ S ( V ; X ) ] = S _ { \sigma } ( V ; X ) = \operatorname { V a r } _ { \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) } [ C ] . } \end{array}\tag{26}
$$

Moreover, the population sharpness satisfies:

$$
\operatorname { \mathcal { S } } _ { \sigma } ( V ; X ) = \mathbb { E } _ { Z } [ \operatorname { V a r } ( C \mid Z ) ] + \operatorname { V a r } _ { Z } ( \mathbb { E } [ C \mid Z ] )\tag{27}
$$

Proof. Let $\mu = \mathbb { E } [ C ]$ and $\begin{array} { r } { \bar { C } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } C _ { k } } \end{array}$ . The following identity holds:

$$
\sum _ { k = 1 } ^ { K } ( C _ { k } - \bar { C } ) ^ { 2 } = \sum _ { k = 1 } ^ { K } ( C _ { k } - \mu ) ^ { 2 } - K ( \bar { C } - \mu ) ^ { 2 } .\tag{28}
$$

Taking expectations gives:

$$
\begin{array} { r l } {  { \mathbb { E } [ \sum _ { k = 1 } ^ { K } ( C _ { k } - \bar { C } ) ^ { 2 } ] = K \operatorname { V a r } ( C ) - K \operatorname { V a r } ( \bar { C } ) } } \\ & { = ( K - 1 ) \operatorname { V a r } ( C ) , } \end{array}\tag{29}
$$

where independence implies $\mathrm { V a r } ( \bar { C } ) = \mathrm { V a r } ( C ) / K$ . Dividing by $K - 1$ proves Eq. 26.

For the second result, define $m ( Z ) ~ = ~ \mathbb { E } [ C ~ | ~ Z ]$ . The centered confidence decomposes as:

$$
C - \mathbb { E } [ C ] = { \big ( } C - m ( Z ) { \big ) } + { \big ( } m ( Z ) - \mathbb { E } [ C ] { \big ) } .\tag{30}
$$

The cross term has zero expectation because $\mathbb { E } [ C - m ( Z ) \mid$ $Z ] = 0$ . Taking the expectation of the squared identity yields Eq. 27. This argument treats $Z$ as a discrete random variable and does not diferentiate through decoding. □

Corollary 1 (Control of large confidence deviations). For any $\tau > 0 .$ , Chebyshev’s inequality gives:

$$
\mathrm { P r } ( | C - \mathbb { E } [ C ] | \geq \tau ) \leq \frac { S _ { \sigma } ( V ; X ) } { \tau ^ { 2 } } .\tag{31}
$$

Thus, reducing population behavioral sharpness controls the probability of large confidence deviations under the perturbation distribution. This is a stability statement and does not by itself imply that the decoded answer is correct.

Corollary 2 (Contribution of behavioral changes). Let $Y _ { 0 }$ be the trajectory decoded at the unperturbed prefix, and let $\phi ( Z )$ denote a discrete behavioral descriptor of trajectory $Z ,$ such as its normalized final answer. Define $B = \bar { { \bf 1 } } \{ \phi ( \bar { Z } ) \neq$ $\phi ( Y _ { 0 } ) \}$ and $p _ { \mathrm { c h g } } = \mathrm { P r } ( B = 1 )$ . For $0 < p _ { \mathrm { c h g } } < \bar { 1 }$ , define the conditional means:

$$
\mu _ { \mathrm { s a m e } } = \mathbb { E } [ C \mid B = 0 ] , \qquad \mu _ { \mathrm { c h g } } = \mathbb { E } [ C \mid B = 1 ] .\tag{32}
$$

Because B is a function of $Z ,$ applying the law of total variance to $m ( Z ) = \mathbb { E } [ C \mid Z ]$ gives:

$$
\operatorname { V a r } _ { Z } ( \mathbb { E } [ C \mid Z ] ) \ge p _ { \mathrm { c h g } } ( 1 - p _ { \mathrm { c h g } } ) \left( \mu _ { \mathrm { c h g } } - \mu _ { \mathrm { s a m e } } \right) ^ { 2 } .\tag{33}
$$

The right-hand side is $\operatorname { V a r } _ { B } ( \mathbb { E } [ C \mid B ] )$ , which measures the confidence diference between the unchanged and changed groups. Thus, the bound is informative when behavioral changes are accompanied by a shift in mean confidence. If the two groups have identical mean confidence, it provides no information about how frequently behavior changes.

Proposition 2 (Smooth local sensitivity). Within a neighborhood where greedy decoding remains $Y ,$ , the local confidence variance satisfies the following expansion as $\sigma  0 \mathrm { : }$

$$
\begin{array} { r } { \mathrm { V a r } _ { \epsilon } [ f _ { Y } ( V + \epsilon ) ] = \sigma ^ { 2 } \| \nabla _ { V } f _ { Y } ( V ) \| _ { F } ^ { 2 } + \mathcal { O } ( \sigma ^ { 4 } ) . } \end{array}\tag{34}
$$

If $\nabla _ { V } f _ { Y } ( V ) = 0$ , the corresponding expansion is:

$$
\mathrm { V a r } _ { \epsilon } [ f _ { Y } ( V + \epsilon ) ] = \frac { \sigma ^ { 4 } } { 2 } \| \nabla _ { V } ^ { 2 } f _ { Y } ( V ) \| _ { F } ^ { 2 } + o ( \sigma ^ { 4 } ) .\tag{35}
$$

Proof. Write $\epsilon = \sigma U$ , where $U \sim \mathcal { N } ( 0 , I )$ , and abbreviate $g = g _ { Y }$ and $H = H _ { Y }$ . From Eq. 25, define:

$$
A = \langle g , U \rangle _ { F } , \qquad Q = \frac { 1 } { 2 } \langle U , H U \rangle _ { F } .\tag{36}
$$

Then, $f _ { Y } ( V + \sigma U ) = f _ { Y } ( V ) + \sigma A + \sigma ^ { 2 } Q + R _ { \sigma } ( U )$ . The Gaussian moments satisfy:

$$
\mathbb { E } [ A ] = 0 , \qquad \operatorname { V a r } ( A ) = \| g \| _ { F } ^ { 2 } , \qquad \operatorname { C o v } ( A , Q ) = 0 .\tag{37}
$$

The last equality follows because every third-order centered Gaussian moment is zero. Moreover, Var $( \sigma ^ { 2 } Q ) = \mathcal { O } ( \sigma ^ { 4 } )$ By the Cauchy–Schwarz inequality and $\begin{array} { r l } { \mathbb { E } [ \mathring { R } _ { \sigma } ( U ) ^ { 2 } ] } & { { } = } \end{array}$ $\dot { \mathcal { O } } ( \sigma ^ { 6 } )$ ), the remainder and its covariance terms contribute at most $\mathcal { O } ( \sigma ^ { 4 } )$ . Consequently, we obtain:

$$
\mathrm { V a r } [ f _ { Y } ( V + \sigma U ) ] = \sigma ^ { 2 } \| g \| _ { F } ^ { 2 } + \mathcal { O } ( \sigma ^ { 4 } ) .\tag{38}
$$

This proves Eq. 34.

At a stationary point of $f _ { Y } , g = 0$ , and the quadratic term becomes dominant:

$$
\operatorname { V a r } [ f _ { Y } ( V + \sigma U ) ] = \sigma ^ { 4 } \operatorname { V a r } ( Q ) + o ( \sigma ^ { 4 } ) .\tag{39}
$$

For a symmetric Hessian H and a standard Gaussian vector U, the following identities hold:

$$
\begin{array} { r } { \mathbb E [ \boldsymbol { U } ^ { \top } H \boldsymbol { U } ] = \mathrm { t r } ( H ) , \qquad \mathrm { V a r } ( \boldsymbol { U } ^ { \top } H \boldsymbol { U } ) = 2 \| H \| _ { F } ^ { 2 } . } \end{array}\tag{40}
$$

Therefore, $\begin{array} { r } { \mathrm { V a r } ( Q ) = \frac { 1 } { 2 } \lVert H \rVert _ { F } ^ { 2 } } \end{array}$ , which proves Eq. 35. □

This proposition characterizes the smooth withintrajectory component of behavioral sharpness; variation due to trajectory switching is captured by the between-trajectory term in Eq. 27.

Perturbation scale. The elementwise standard deviation σ corresponds to the following root-mean-square Frobenius radius:

$$
{ \sqrt { \mathbb { E } [ \left. \epsilon \right. _ { F } ^ { 2 } ] } } = \sigma { \sqrt { L d } } .\tag{41}
$$

Thus, choosing $\sigma = \rho / \sqrt { L d }$ would match the root-meansquare norm of the random perturbation to the SAP radius $\rho .$ This relation aligns only their geometric scales and does not make the two objectives equivalent.

## Sharpness-Aware Perturbation

Proposition 3 (First-order worst-case bound). Let the worst-case local increase be:

$$
\Delta _ { \rho } ( V ) = \operatorname* { m a x } _ { \| \epsilon \| _ { F } \leq \rho } \left[ \ell ( V + \epsilon ) - \ell ( V ) \right] .\tag{42}
$$

Let $\boldsymbol { g } = \nabla _ { V } \ell ( V )$ . If ℓ has a $\beta \mathrm { - I }$ Lipschitz-continuous gradient on the perturbation ball, the following bound holds:

$$
| \Delta _ { \rho } ( V ) - \rho | | g | | _ { F } | \leq \frac { \beta \rho ^ { 2 } } { 2 } .\tag{43}
$$

For $g \neq 0 ,$ , the gradient-guided perturbation $\epsilon ^ { \star } = \rho g / \| g \| _ { F }$ additionally satisfies:

$$
| \ell ( V + \epsilon ^ { \star } ) - \ell ( V ) - \rho \| g \| _ { F } | \leq \frac { \beta \rho ^ { 2 } } { 2 } .\tag{44}
$$

Accordingly, the implemented single-example SAP loss is ${ \mathcal { L } } _ { \mathrm { S A P } } = { \overline { { \ell } } } ( { \overline { { V } } } + \epsilon ^ { \star } )$

Proof. For any ϵ in the perturbation ball, the fundamental theorem of calculus gives:

$$
\begin{array} { l } { \displaystyle \ell ( V + \epsilon ) - \ell ( V ) - \langle g , \epsilon \rangle _ { F } } \\ { \displaystyle = \int _ { 0 } ^ { 1 } \langle \nabla _ { V } \ell ( V + t \epsilon ) - g , \epsilon \rangle _ { F } d t . } \end{array}\tag{45}
$$

Gradient Lipschitzness and Cauchy–Schwarz imply:

$$
| \ell ( V + \epsilon ) - \ell ( V ) - \langle g , \epsilon \rangle _ { F } | \leq \int _ { 0 } ^ { 1 } \beta t \| \epsilon \| _ { F } ^ { 2 } d t = \frac { \beta } { 2 } \| \epsilon \| _ { F } ^ { 2 } .\tag{46}
$$

For the upper bound, Eq. 46 gives:

$$
\Delta _ { \rho } ( V ) \le \operatorname* { m a x } _ { \| \epsilon \| _ { F } \le \rho } \langle g , \epsilon \rangle _ { F } + \frac { \beta \rho ^ { 2 } } { 2 } = \rho \| g \| _ { F } + \frac { \beta \rho ^ { 2 } } { 2 } .\tag{47}
$$

When $g \neq 0 ,$ , evaluating the inner objective at $\epsilon ^ { \star } = \rho g / \| g \| _ { F }$ gives:

$$
\Delta _ { \rho } ( V ) \geq \ell ( V + \epsilon ^ { \star } ) - \ell ( V ) \geq \rho \| g \| _ { F } - \frac { \beta \rho ^ { 2 } } { 2 } .\tag{48}
$$

When $g = 0 ,$ , choosing $\epsilon = 0$ gives $\Delta _ { \rho } ( V ) \geq 0$ , while the upper bound gives $\Delta _ { \rho } ( V ) \leq \beta \rho ^ { 2 } / 2$ . Hence, Eq. 43 holds in both cases. Finally, substituting $\epsilon ^ { \star }$ into Eq. 46 proves Eq. 44. □

Efect of stop-gradient. For $g \neq 0 ,$ , the bounds above concern the value of the perturbed objective and are unchanged by the stop-gradient operation. Treating $\epsilon ^ { \star }$ as constant gives the outer gradient:

$$
\nabla _ { V } ^ { \mathrm { s g } } \mathcal { L } _ { \mathrm { S A P } } = \nabla _ { V } \ell ( V + \epsilon ^ { \star } ) .\tag{49}
$$

Gradient Lipschitzness further gives:

$$
\left\| \nabla _ { V } ^ { \mathrm { s g } } \mathcal { L } _ { \mathrm { S A P } } - g \right\| _ { F } \leq \beta \rho .\tag{50}
$$

Thus, stop-gradient removes derivatives through the construction of $\epsilon ^ { \star }$ while retaining the gradient evaluated at the perturbed prefix.

Proposition 4 (Stationary-point curvature). Suppose that $\overset { \cdot } { \nabla } _ { V } \ell ( V ) = 0$ and that ℓ is twice continuously diferentiable in a neighborhood of $V . \operatorname { A s } \rho \to 0$ , the ideal inner maximization satisfies:

$$
\Delta _ { \rho } ( V ) = \frac { \rho ^ { 2 } } { 2 } \left[ \lambda _ { \mathrm { m a x } } \bigl ( \nabla _ { V } ^ { 2 } \ell ( V ) \bigr ) \right] _ { + } + o ( \rho ^ { 2 } ) ,\tag{51}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$

Proof. Let $H = \nabla _ { V } ^ { 2 } \ell ( V )$ . Twice continuous diferentiability and $\nabla _ { V } \ell ( V ) = { \ ' { 0 } }$ yield the following expansion, uniformly over $\| \epsilon \| _ { F } \le \rho$ as $\rho  0 \colon$

$$
\ell ( V + \epsilon ) - \ell ( V ) = \frac { 1 } { 2 } \langle \epsilon , H \epsilon \rangle _ { F } + o ( \| \epsilon \| _ { F } ^ { 2 } ) .\tag{52}
$$

Writing $\epsilon = \rho U$ with $\| U \| _ { F } \leq 1$ gives:

$$
\Delta _ { \rho } ( V ) = \frac { \rho ^ { 2 } } { 2 } \operatorname* { m a x } _ { \| U \| _ { F } \leq 1 } \langle U , H U \rangle _ { F } + o ( \rho ^ { 2 } ) .\tag{53}
$$

Algorithm 1: TASCO with Random Perturbation   
Require: Frozen LLM $M _ { \theta } ,$ test set $\mathcal { D } _ { \mathrm { t e s t } }$ , warm-start prefix   
$V _ { 0 } .$ , optimizer O, number of epochs E, perturbation scale   
$\sigma ,$ number of perturbations $\dot { K } \geq 2 ,$ , and stability weight   
$\lambda _ { \mathrm { { r a n d } } }$   
Ensure: Optimized prefix V   
1: Initialize $V  \dot { V _ { 0 } }$   
2: for $e = 1 , \ldots , E$ do   
3: Randomly shufle $\mathcal { D } _ { \mathrm { t e s t } }$ and partition it into mini  
batches   
4: for each mini-batch B do   
5: Generate trajectories $\mathbf { Y } _ { B }$ under V   
6: $\mathcal { L } _ { \mathrm { c o n f } }  \mathcal { L } _ { B } \mathbf { \bar { ( } } V ; \mathbf { Y } _ { B } )$   
7: for $k = 1 , \ldots , K$ do   
8: Sample $\epsilon ^ { ( k ) ^ { \prime } } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ and set ${ V ^ { ( k ) }  V + }$   
$\epsilon ^ { ( k ) }$   
9: Greedily generate trajectories $\mathbf { Y } _ { B } ^ { ( k ) }$ under $V ^ { ( k ) }$   
10: Compute $\{ c _ { X } ^ { ( k ) } : X \in B \}$ using Eq. 22   
11: end for   
12: Compute $\mathcal { L } _ { \mathrm { r a n d } }$ from $\{ c _ { X _ { . } } ^ { ( k ) } \}$ using Eq. 23   
13: $g \gets \nabla _ { V } \left( \mathcal { L } _ { \mathrm { c o n f } } + \lambda _ { \mathrm { r a n d } } \mathbf { \bar { \mathcal { L } } } _ { \mathrm { r a n d } } \right)$   
14: $V  \mathcal { O } ( \dot { V } , g )$   
15: end for   
16: end for   
17: return $V$

If $\dot { \operatorname { \lambda } } _ { \operatorname* { m a x } } ( H ) > 0$ , the maximum is attained by a unit eigenvector associated with $\lambda _ { \operatorname* { m a x } } ( H ) . \operatorname { I f } \lambda _ { \operatorname* { m a x } } ( H ) \dot { \leq } 0 ,$ , it is attained at $U = 0$ . Therefore:

$$
\operatorname* { m a x } _ { \| U \| _ { F } \leq 1 } \langle U , H U \rangle _ { F } = [ \lambda _ { \operatorname* { m a x } } ( H ) ] _ { + } .\tag{54}
$$

This proves Eq. 51.

This proposition characterizes the ideal worst-case objective at stationarity, whereas Proposition 3 characterizes the implemented gradient-guided perturbation when $g \neq 0$

Remark. Proposition 4 characterizes the ideal worst-case inner problem. At an exact stationary point, the first-order perturbation $\epsilon ^ { \star } = \rho g / \| g \| _ { F }$ is undefined and does not recover the leading Hessian eigenvector. The proposition therefore explains the local geometry of the robust objective rather than claiming that the implemented first-order perturbation solves the stationary second-order problem.

## Relationship Between the Two Local Objectives

The two analyses reveal distinct views of local geometry. To compare them, suppose temporarily that both perturbation strategies are applied to the same smooth scalar function $h ( V )$ and that decoding remains fixed. Away from stationary points, the random-perturbation variance satisfies:

$$
\operatorname { V a r } _ { \epsilon } [ h ( V + \epsilon ) ] = \sigma ^ { 2 } \| \nabla _ { V } h ( V ) \| _ { F } ^ { 2 } + { \mathcal { O } } ( \sigma ^ { 4 } ) .\tag{55}
$$

By contrast, the worst-case local increase satisfies:

$$
\operatorname* { m a x } _ { \| \epsilon \| _ { F } \leq \rho } [ h ( V + \epsilon ) - h ( V ) ] = \rho \| \nabla _ { V } h ( V ) \| _ { F } + \mathcal { O } ( \rho ^ { 2 } ) .\tag{56}
$$

Thus, the random objective aggregates sensitivity over isotropic directions, whereas the worst-case objective selects the most adverse direction.

At a stationary point with Hessian $H _ { h }$ , their leading curvature terms depend on $\| H _ { h } \| _ { F } ^ { 2 }$ and $[ \lambda _ { \operatorname* { m a x } } ( H _ { h } ) ] _ { + }$ , respectively. Their relationship is bounded by:

$$
[ \lambda _ { \operatorname* { m a x } } ( H _ { h } ) ] _ { + } \leq \| H _ { h } \| _ { 2 } \leq \| H _ { h } \| _ { F } .\tag{57}
$$

This distinguishes the largest positive curvature from aggregate curvature across directions. In TASCO, Random Perturbation operates on re-decoded trajectory-level loglikelihood, whereas SAP operates on the fixed-trajectory entropy loss. Therefore, these relations clarify their diferent geometric interpretations but do not establish numerical equivalence or an ordering between the objectives.

Summary. Together, these analyses show how the two TASCO objectives control complementary forms of local confidence sensitivity. Random Perturbation regularizes distributional variation across perturbed behaviors, whereas SAP controls worst-case sensitivity along an adverse local direction. These results establish local confidence robustness under prefix perturbations rather than a guarantee of answer correctness. Section B separately examines the empirical relationship between local stability and answer accuracy.

## D. Complete Optimization Algorithms

This section provides the complete optimization procedures for the two TASCO variants. Both algorithms optimize a task-level prefix V over the unlabeled test set while keeping the LLM parameters θ frozen. They difer only in how local confidence stability is estimated and incorporated into prefix optimization.

Random Perturbation. Algorithm 1 presents the Random Perturbation variant. At each update, it samples K Gaussian perturbations around the current prefix and generates a trajectory under each perturbed prefix. The resulting trajectorylevel confidence variance is averaged across the mini-batch and combined with the confidence loss to update V.

Sharpness-Aware Perturbation. Algorithm 2 presents the Sharpness-Aware Perturbation variant. It first generates one trajectory per input under the current prefix and holds these trajectories fixed throughout the update. A gradientguided perturbation is then constructed from the confidence loss, after which the prefix is updated using the loss evaluated at the perturbed prefix. The perturbation is treated as constant during the second backward pass.

Computational Cost. Random Perturbation captures richer distributional information but requires K independently decoded perturbed rollouts per input, introducing additional adaptation cost. Sharpness-Aware Perturbation avoids these rollouts through a single gradient-guided perturbation on the fixed trajectory, substantially reducing autoregressive decoding cost relative to Random Perturbation. The reported token-eficiency results characterize generation length after adaptation. Moreover, the learned prefix transfers to unseen datasets without re-optimization, avoiding additional adaptation cost in cross-distribution settings.

Algorithm 2: TASCO with Sharpness-Aware Perturbation   
Require: Frozen LLM $M _ { \theta } ,$ test set $\mathcal { D } _ { \mathrm { t e s t } }$ , warm-start prefix   
$V _ { 0 } .$ optimizer O, number of epochs E, and perturbation   
radius $\rho$   
Ensure: Optimized prefix $V$   
1: Initialize $V  \dot { V _ { 0 } }$   
2: for $e = 1 , \ldots , E$ do   
3: Randomly shufle $\mathcal { D } _ { \mathrm { t e s t } }$ and partition it into mini  
batches   
4: for each mini-batch $\boldsymbol { B }$ do   
5: Generate trajectories $\mathbf { Y } _ { B }$ under V and hold them   
fixed   
6: $g \gets \nabla _ { V } \mathcal { L } _ { B } ( V ; \mathbf { Y } _ { B } )$   
7: if $\| g \| _ { F } > 0$ then   
8: $\epsilon ^ { \star }  \rho g / \| g \| _ { F }$   
9: else   
10: $\epsilon ^ { \star }  0$   
11: end if   
12: $g _ { \mathrm { S A P } } \gets \nabla _ { V } \mathcal { L } _ { \mathcal { B } } \left( V + \mathrm { S t o p G r a d } ( \epsilon ^ { \star } ) ; \mathbf { Y } _ { \mathcal { B } } \right)$   
13: $V \gets \mathcal { O } ( V , g _ { \mathrm { S A P } } )$   
14: end for   
15: end for   
16: return V

## E. Further Analysis of Reasoning Efectiveness

This section examines TASCO beyond final-answer accuracy, focusing on whether its gains remain consistent across models and runs and are reflected in the reasoning process itself. We also assess its sensitivity to the perturbation scale.

## Evaluation on Reasoning-Enhanced Models.

We first examine whether TASCO also benefits reasoningenhanced LLMs, whose pretrained reasoning behavior differs substantially from that of general instruction-tuned models. We evaluate DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B on five mathematical reasoning benchmarks. In addition to the original model, we compare TASCO with test-time reasoning-control methods, including s1, CoD, and α1. We report answer accuracy and average generation length to jointly assess reasoning efectiveness and token eficiency.

As shown in Table 9, TASCO achieves the highest average accuracy at both model scales and consistently outperforms the base model across all five benchmarks. It improves average accuracy from 42.2% to 49.0% on DeepSeek-R1-Distill-Qwen-1.5B while reducing generation length by 14.1%, and from 56.1% to 61.1% on the 7B model with an 8.8% reduction. Although some reasoning-control baselines produce shorter outputs, their accuracy gains are smaller or inconsistent. TASCO thus ofers a stronger balance between accuracy and generation eficiency, demonstrating that stability-aware confidence optimization also benefits models already trained for deliberative reasoning.

## Evaluation Across Random Seeds

We additionally evaluate TASCO and selected baselines using three independent random seeds. To limit the cost of repeated test-time optimization, we include representative methods covering no adaptation, latent-variable optimization, and confidence-only prefix optimization: Zero-Shot CoT, LTPO, and TTSV. We evaluate both TASCO variants under the same protocol and report mean accuracy with standard deviation in Table 8. This evaluation provides more reliable performance estimates and quantifies variation across independent runs.

<table><tr><td>Method</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>AIME24</td><td>GPQA</td><td>Avg.</td></tr><tr><td colspan="7">Qwen2.5-Math-1.5B</td></tr><tr><td>Zero-Shot CoT</td><td> $4 0 . 2 \ : ( \pm 1 . 8 )$ </td><td> $3 4 . 1 \ : ( \pm 2 . 4 )$ </td><td> $9 . 4 \left( \pm 1 . 4 \right)$ </td><td> $6 . 2 \ : ( \pm 3 . 5 )$ </td><td> $1 9 . 4 ( \pm 2 . 1 ) $ </td><td> $2 1 . 9 \left( \pm 0 . 5 \right)$ </td></tr><tr><td>LTPO</td><td> $5 7 . 6 ( \pm 2 . 1 ) $ </td><td> $4 7 . 2 \left( \pm 2 . 6 \right)$ </td><td> $1 2 . 4 ( \pm 1 . 5 )$ </td><td> $8 . 8 \left( \pm 3 . 1 \right)$ </td><td> $2 1 . 5 \AA ( \pm 1 . 6 )$ </td><td> $2 8 . 6 ( \pm 2 . 2 ) $ </td></tr><tr><td>TTSV</td><td> $6 9 . 2 \left( \pm 1 . 6 \right)$ </td><td> $4 5 . 4 ( \pm 2 . 0 ) $ </td><td> $1 9 . 2 \ : ( \pm 1 . 4 ) $ </td><td> $6 . 2 \ : ( \pm 2 . 6 )$ </td><td> $2 4 . 2 \ : ( \pm 0 . 6 ) $ </td><td> $3 2 . 9 \left( \pm 1 . 4 \right)$ </td></tr><tr><td>TASCO-RP</td><td> $7 0 . 8 ( \pm 0 . 4 ) $ </td><td> $4 9 . 9 \left( \pm 0 . 9 \right)$ </td><td> $2 2 . 1 \ : ( \pm 1 . 3 )$ </td><td> $9 . 9 \left( \pm 1 . 0 \right)$ </td><td> $2 6 . 7 ( \pm 1 . 1 )$ </td><td> $3 5 . 9 \left( \pm 0 . 5 \right)$ </td></tr><tr><td>TASCO-SAP</td><td> $7 1 . 6 ( \pm 0 . 9 )$ </td><td> $4 9 . 4 \ : ( \pm 1 . 4 ) $ </td><td> $2 0 . 1 \ : ( \pm 2 . 2 )$ </td><td> $1 1 . 8 ( \pm 1 . 6 ) $ </td><td> $2 6 . 2 \ : ( \pm 0 . 6 )$ </td><td> $3 6 . 4 \ : ( \pm 0 . 7 )$ </td></tr><tr><td colspan="7">Qwen2.5-Math-7B</td></tr><tr><td>Zero-Shot CoT</td><td> $5 1 . 8 \left( \pm 1 . 6 \right)$ </td><td> $4 2 . 9 \left( \pm 2 . 0 \right)$ </td><td> $1 4 . 1 \ : ( \pm 2 . 9 )$ </td><td> $1 3 . 3 ( \pm 2 . 3 )$ </td><td> $3 0 . 7 \left( \pm 1 . 3 \right)$ </td><td> $3 0 . 8 \left( \pm 1 . 8 \right)$ </td></tr><tr><td>LTPO</td><td> $6 5 . 5 \left( \pm 2 . 2 \right)$ </td><td> $4 2 . 8 ( \pm 3 . 1 ) $ </td><td> $1 8 . 2 \ : ( \pm 2 . 6 ) $ </td><td> $1 7 . 5 ( \pm 2 . 8 ) $ </td><td> $3 3 . 6 ( \pm 2 . 5 ) $ </td><td> $3 5 . 7 \left( \pm 2 . 8 \right)$ </td></tr><tr><td>TTSV</td><td> $7 0 . 0 ( \pm 2 . 2 ) $ </td><td> $5 7 . 8 ( \pm 3 . 7 ) $ </td><td> $3 2 . 8 \ : ( \pm 2 . 5 )$ </td><td> $1 7 . 0 ( \pm 3 . 4 )$ </td><td> $3 6 . 0 ( \pm 2 . 5 )$ </td><td> $4 2 . 7 \ : ( \pm 3 . 2 )$ </td></tr><tr><td>TASCO-RP</td><td> $7 4 . 5 ( \pm 1 . 2 ) $ </td><td> $6 6 . 7 \left( \pm 2 . 0 \right)$ </td><td> $3 8 . 1 ( \pm 2 . 1 ) $ </td><td> $2 0 . 4 ( \pm 2 . 4 )$ </td><td> $3 7 . 7 \substack { ( \pm 2 . 6 ) }$ </td><td> $4 7 . 2 \left( \pm 2 . 1 \right)$ </td></tr><tr><td>TASCO-SAP</td><td> $7 2 . 6 ( \pm 0 . 8 )$ </td><td> $6 5 . 2 \ : ( \pm 2 . 0 ) $ </td><td> $3 9 . 0 ( \pm 2 . 4 ) $ </td><td> $2 3 . 4 ( \pm 2 . 4 )$ </td><td> $3 8 . 5 ( \pm 2 . 8 ) $ </td><td> $4 7 . 8 \ : ( \pm 1 . 8 )$ </td></tr><tr><td colspan="7">LLaMA3.1-8B-Instruct</td></tr><tr><td></td><td> $4 7 . 0 \left( \pm 0 . 8 \right)$ </td><td></td><td></td><td> $7 . 9 \left( \pm 1 . 2 \right)$ </td><td></td><td></td></tr><tr><td>Zero-Shot CoT LTPO</td><td> $4 8 . 9 \left( \pm 0 . 8 \right)$ </td><td> $2 5 . 1 \ : ( \pm 0 . 9 )$   $2 6 . 5 ( \pm 1 . 4 )$ </td><td> $2 1 . 7 \ : ( \pm 0 . 7 )$   $2 0 . 8 \left( \pm 2 . 8 \right)$ </td><td> $1 0 . 8 ( \pm 1 . 4 ) $ </td><td> $2 8 . 7 ( \pm 0 . 7 )$   $3 0 . 1 \ : ( \pm 0 . 8 )$ </td><td> $2 6 . 1 \ : ( \pm 0 . 8 )$ </td></tr><tr><td>TTSV</td><td> $4 7 . 1 \ : ( \pm 0 . 7 )$ </td><td> $2 2 . 5 ( \pm 2 . 8 ) $ </td><td> $1 9 . 6 ( \pm 2 . 7 ) $ </td><td> $5 . 2 ( \pm 3 . 1 )$ </td><td> $2 7 . 2 \left( \pm 1 . 7 \right)$ </td><td> $2 7 . 5 \ : ( \pm 0 . 9 )$ </td></tr><tr><td>TASCO-RP</td><td> $4 7 . 2 \left( \pm 1 . 2 \right)$ </td><td> $2 8 . 6 ( \pm 0 . 8 ) $ </td><td> $2 5 . 3 ( \pm 1 . 5 )$ </td><td> $1 0 . 9 ( \pm 1 . 2 ) $ </td><td> $3 2 . 7 \left( \pm 0 . 7 \right)$ </td><td> $2 5 . 0 ( \pm 1 . 6 )$ </td></tr><tr><td>TASCO-SAP</td><td> $4 9 . 7 \left( \pm 0 . 8 \right)$ </td><td> $2 6 . 4 \ : ( \pm 2 . 0 )$ </td><td> $2 6 . 0 ( \pm 0 . 7 )$ </td><td> $1 2 . 2 \left( \pm 1 . 2 \right)$ </td><td> $3 2 . 2 \left( \pm 0 . 8 \right)$ </td><td> $2 9 . 3 \ : ( \pm 1 . 5 )$   $2 9 . 2 \ : ( \pm 0 . 8 )$ </td></tr></table>

Table 8: Accuracy (%) across three independent random seeds. Each entry reports the mean accuracy followed by the standard deviation in gray. The average is computed across the five benchmarks for each run and then aggregated across seeds.

## Semantic Evaluation of Reasoning Processes

these annotations rather than directly assigned by the judge.

Experimental Setup. We use DeepSeek-V4-Flash as an external judge to examine how TASCO afects intermediate reasoning. We compare paired trajectories generated by Confidence Only and TASCO-SAP using Qwen2.5-Math-7B, LLaMA3.1-8B-Instruct, and DeepSeek-R1-Distill-Qwen-7B on the full MATH500 and Minerva Math test sets, containing 500 and 272 examples, respectively. For each trajectory, the judge receives only the original problem and the generated reasoning process; no ground-truth answer is provided. Common explicit final-answer spans are masked to prevent the judge from inferring reasoning quality from answer correctness. We enable thinking mode with high reasoning efort and request a structured JSON annotation.

External-Judge Prompt. The prompt template used for reasoning-process annotation is shown below. The placeholders are replaced with the original problem and the corresponding trajectory after final-answer masking.

External-Judge Prompt. We use the following condensed prompt to summarize the instructions given to the external judge. The complete prompt and JSON schema are provided in the released implementation.

The judge first divides each trajectory into semantically meaningful steps. Each step receives a correctness label from {correct, incorrect, uncertain}, an informativeness label from {informative, redundant, off\_track}, and a functional role. The judge also determines whether the trajectory commits prematurely to an answer or reasoning direction before suficient support has been established. All reported metrics are computed from

## Reasoning-Process Evaluation Prompt

System: You are an expert evaluator of mathematical reasoning. Evaluate the reasoning process independently of its final answer, writing style, verbosity, or expressed confidence. Divide the trajectory into semantically meaningful steps. Label each step by:

• correctness: correct, incorrect, or uncertain;   
• informativeness: informative, redundant, or   
off\_track.

Determine whether the trajectory commits prematurely to an answer or reasoning direction before suficient support is established. Return all annotations as a valid JSON object.

User: Evaluate the following reasoning trajectory.

Problem: <Problem>

Reasoning trajectory: <Reasoning Trajectory>

Evaluation Metrics. For trajectory i with $J _ { i }$ semantic steps, let $c _ { i j }$ and $u _ { i j }$ denote the correctness and informativeness labels of step j, respectively. Let $E _ { i }$ denote the subset of steps labeled either correct or incorrect. Step Correctness is defined as:

<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">MATH500</td><td colspan="2">AMC23</td><td colspan="2">Minerva</td><td colspan="2">Average</td></tr><tr><td>Acc.</td><td>Tok.</td><td>Acc.</td><td>Tok.</td><td>Acc.</td><td>Tok. Acc.</td><td>Tok.</td><td>Acc.</td><td>Tok.</td><td>Acc.</td><td></td><td>Tok.</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-1.5B</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>22.8</td><td>7580</td><td>22.5</td><td>7674</td><td>78.6</td><td>3700</td><td>59.4</td><td>6242</td><td>27.6</td><td>5057</td><td>42.2</td><td>6051</td></tr><tr><td>s1</td><td>25.8</td><td>7778</td><td>25.4</td><td>7982</td><td>80.2</td><td>4623</td><td>61.9</td><td>6428</td><td>29.4</td><td>6127</td><td>44.5↑2.4</td><td>6588↑8.9%</td></tr><tr><td>CoD</td><td>26.7</td><td>7196</td><td>20.8</td><td>7552</td><td>79.4</td><td>3267</td><td>66.2</td><td>5293</td><td>32.0</td><td>4041</td><td>45.0↑2.8</td><td>5470↓9.6%</td></tr><tr><td>α1</td><td>30.8</td><td>6841</td><td>12.9</td><td>6529</td><td>81.0</td><td>3541</td><td>63.1</td><td>4310</td><td>33.1</td><td>4589</td><td>44.2↑2.0</td><td>5162↓14.7%</td></tr><tr><td>TASCO-SAP</td><td>30.4</td><td>6751</td><td>29.2</td><td>6838</td><td>81.4</td><td>3616</td><td>69.4</td><td>4282</td><td>34.6</td><td>4507</td><td>49.0↑6.8</td><td>5199↓14.1%</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-7B</td><td colspan="3"></td></tr><tr><td>Base</td><td>44.2</td><td>6899</td><td>25.0</td><td>7099</td><td>89.4</td><td>3399</td><td>79.4</td><td>4672</td><td>42.6</td><td>4416</td><td>56.1</td><td>5297</td></tr><tr><td>s1</td><td>37.5</td><td>7368</td><td>24.2</td><td>7041</td><td>89.0</td><td>4072</td><td>78.1</td><td>5271</td><td>40.4</td><td>4986</td><td>53.8↓2.3</td><td>5748↑8.5%</td></tr><tr><td>CoD</td><td>45.0</td><td>6301</td><td>33.3</td><td>6582</td><td>88.0</td><td>2071</td><td>85.6</td><td>3645</td><td>40.1</td><td>2440</td><td>58.4↑2.3</td><td>4208↓20.6%</td></tr><tr><td>α1</td><td>36.7</td><td>6831</td><td>32.5</td><td>6638</td><td>89.8</td><td>3848</td><td>90.6</td><td>4529</td><td>42.6</td><td>4223</td><td>58.4↑2.3</td><td>5214↓1.6%</td></tr><tr><td>TASCO-SAP</td><td>49.2</td><td>6188</td><td>34.2</td><td>6652</td><td>90.4</td><td>3041</td><td>88.1</td><td>4287</td><td>43.4</td><td>3988</td><td>61.1↑4.9</td><td>4831↓8.8%</td></tr></table>

Table 9: Accuracy (%) and average generation length across five reasoning benchmarks. Accuracy is reported to one decimal place, and token counts are rounded to the nearest integer. Average results are computed across the five datasets. Arrows denote absolute accuracy changes and relative token changes compared with Base. Best results are in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">MATH500</td><td colspan="4">Minerva Math</td></tr><tr><td>SC↑</td><td>SI↑</td><td>PC↓</td><td>RR↓</td><td>SC↑</td><td>SI↑</td><td>PC↓</td><td>RR↓</td></tr><tr><td rowspan="2">Qwen2.5-Math-7B</td><td>Confidence Only</td><td>82.7</td><td>77.6</td><td>18.8</td><td>15.1</td><td>75.4</td><td>75.7</td><td>22.8</td><td>11.2</td></tr><tr><td>TASCO-SAP</td><td>85.2</td><td>80.4</td><td>17.2</td><td>11.6</td><td>77.5</td><td>80.5</td><td>19.5</td><td>10.1</td></tr><tr><td>LLaMA3.1-8B-Instruct</td><td>Confidence Only TASCO-SAP</td><td>65.6 67.6</td><td>66.2 67.6</td><td>43.0 41.4</td><td>10.2 8.8</td><td>51.8 63.7</td><td>58.8 69.6</td><td>51.8 40.4</td><td>12.8 10.2</td></tr><tr><td rowspan="2">DS-Distill-Qwen-7B</td><td>Confidence Only</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>91.2</td><td>83.6</td><td>6.8</td><td>11.6</td><td>69.5</td><td>69.7</td><td>31.3</td><td>11.9</td></tr><tr><td></td><td>TASCO-SAP</td><td>94.2</td><td>85.1</td><td>4.8</td><td>11.9</td><td>72.1</td><td>72.6</td><td>29.3</td><td>10.6</td></tr></table>

Table 10: Reasoning-process evaluation using DeepSeek-V4-Flash as the external judge. All values are percentages.

$$
\mathrm { S C } _ { i } = \frac { \sum _ { j \in E _ { i } } { \bf 1 } \{ c _ { i j } = \mathrm { c o r r e c t } \} } { \lvert E _ { i } \rvert } ,\tag{58}
$$

where 1{·} is the indicator function and uncertain steps are excluded. Higher SC indicates that a larger proportion of assessable steps are mathematically or logically valid.

Step Informativeness measures the proportion of steps that introduce useful information and advance the solution:

$$
\mathrm { S I } _ { i } = { \frac { 1 } { J _ { i } } } \sum _ { j = 1 } ^ { J _ { i } } \mathbf { 1 } \{ u _ { i j } = { \mathrm { i n f o r m a t i v e } } \} .\tag{59}
$$

Higher SI indicates more productive reasoning.

Redundancy Rate measures the proportion of steps that mainly repeat or restate previously established information:

$$
\mathrm { R R } _ { i } = \frac { 1 } { J _ { i } } \sum _ { j = 1 } ^ { J _ { i } } { \bf 1 } \{ u _ { i j } = \mathrm { r e d u n d a n t } \} .\tag{60}
$$

Lower RR indicates less repetitive reasoning.

Let $p _ { i }$ denote the trajectory-level premature-commitment judgment. We define the corresponding indicator and its aggregate rate over N trajectories as:

$$
\mathrm { P C } _ { i } = { \bf 1 } \{ p _ { i } = \mathrm { t r u e } \} , \qquad \mathrm { P C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { P C } _ { i } .\tag{61}
$$

Lower PC indicates that the model is less likely to commit prematurely to an insuficiently supported reasoning direction. We average SC, SI, and RR across trajectories and report all four metrics as percentages.

Sensitivity to Perturbation Scale. We further examine the sensitivity of TASCO to the Gaussian scale σ and perturbation radius $\rho$ on four additional benchmarks. As shown in Fig. 10, both variants outperform confidence-only optimization across a broad range of perturbation scales. The default settings, $\sigma = 0 . 0 1$ and $\rho = 0 . 8 ,$ , achieve the best or nearbest accuracy across datasets, while moderate changes generally produce comparable results. These findings indicate that TASCO is not sensitive to a narrowly tuned perturbation scale and that the selected settings generalize well across benchmarks.

![](images/4b96ed85af993cfa03b9152e86d7f167917a53655c9edd1d2d93d7317e090b7e.jpg)  
Figure 10: Sensitivity of TASCO to perturbation scales on four additional benchmarks using Qwen2.5-Math-1.5B. Each benchmark contains separate panels for Random Perturbation and Sharpness-Aware Perturbation.

Results. As shown in Table 10, TASCO consistently improves all four reasoning-process metrics across both datasets. It produces a larger proportion of correct and informative steps while reducing redundant reasoning and premature commitment. These results suggest that stability-aware optimization afects not only final-answer accuracy but also how the model develops and maintains its reasoning process.

## F. Case Studies

This section presents case studies that illustrate the experimental prompting protocol and compare the reasoning behaviors induced by Confidence Only and TASCO.

Prompt and Example. We first provide an AIME24 example illustrating the complete prompt and response format. As shown in Table 11, the system message instructs the model to reason step by step and place its final answer within \boxed{}, while the original problem is provided as the user message. TASCO adds no textual instructions; its learned continuous prefix is prepended only at the embedding level. The table reports the complete response and the final answer extracted by the benchmark parser. Subsequent case studies use the same prompt format and omit these repeated instructions.

Reasoning Patterns. Table 12 reveals a recurring contrast between the two methods. Confidence Only often commits early to a locally plausible solution and continues confidently after discarding a necessary condition. Across the three cases, it cancels a potentially zero factor, confuses simultaneous constraints with their union, or violates a coprimality requirement. TASCO instead preserves these constraints throughout the derivation and retains valid reasoning branches until they are properly resolved. These examples suggest that stabilityaware optimization mitigates premature commitment to fragile reasoning paths and promotes more consistent and reliable multi-step reasoning.

SYSTEM: Please reason step by step, and put your final answer within \boxed{}.   
USER: $[ \mathbf { v } _ { 1 } ] [ \mathbf { v } _ { 2 } ] [ \mathbf { v } _ { 3 } ] \cdot \cdot \cdot [ \mathbf { v } _ { 2 0 } ]$ Every morning Aya goes for a 9-kilometer-long walk and stops at a cofee shop   
afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t   
minutes spent in the cofee shop. When she walks at $s + 2$ kilometers per hour, the walk takes her 2 hours and 24   
minutes, including t minutes spent in the cofee shop. Suppose Aya walks at $s + { \frac { 1 } { 2 } }$ kilometers per hour. Find the   
number of minutes the walk takes her, including the t minutes spent in the cofee shop.   
GROUND TRUTH: 204   
TASCO REASONING:   
To solve this problem, we first determine the values of s and t using the two given walking times.   
## Step 1: Formulate the first time constraint. When Aya walks at s kilometers per hour, the walking time is $\frac { 9 } { s }$ hours,   
while the time spent in the cofee shop is $\frac { t } { 6 0 }$ hours. Since the total time is 4 hours,   
9 t   
+ = 4.   
s 60   
## Step 2: Formulate the second time constraint. When Aya walks at $s + 2$ kilometers per hour, the total duration is   
2 hours and 24 minutes, or 2.4 hours. Therefore,   
9 t   
+ = 2.4.   
s + 2 60   
## Step 3: Solve for s and t. Solving the two equations gives   
(s, t) = (−3, 240) or (s, t) = (2.5, 24).   
Because a speed of −3 kilometers per hour is not physically meaningful, we discard the first solution. Hence,   
s = 2.5, t = 24.   
## Step 4: Compute the requested duration. At the new speed,   
1   
s + = 2.5 + 0.5 = 3   
2   
kilometers per hour. Walking 9 kilometers therefore takes   
9   
= 3   
3   
hours. Including the 24 minutes spent in the cofee shop, the total duration is   
3 × 60 + 24 = 204   
minutes.   
Therefore, the walk takes $\boxed { 2 0 4 }$ minutes.  
Table 11: Prompt and TASCO reasoning example on AIME24. The vectors shown before the user question schematically represent the learned continuous prefix prepended at the embedding level rather than textual tokens.

CASE 1: POLYNOMIAL ROOT FINDING

QUESTION: Find the roots of

$$
( x - 3 ) ^ { 3 } + ( x - 7 ) ^ { 3 } = ( 2 x - 1 0 ) ^ { 3 } .
$$

GROUND TRUTH: 3, 5, 7

CONFIDENCE-ONLY REASONING (INCORRECT):

\## Step 1: To solve the equation, we use the identity

$$
a ^ { 3 } + b ^ { 3 } = ( a + b ) ( a ^ { 2 } - a b + b ^ { 2 } ) .
$$

\## Step 2: Let $a = x - 3$ and $b = x - 7 .$ Applying the identity gives

$$
( 2 x - 1 0 ) { \big ( } ( x - 3 ) ^ { 2 } - ( x - 3 ) ( x - 7 ) + ( x - 7 ) ^ { 2 } { \big ) } = ( 2 x - 1 0 ) ^ { 3 } .
$$

\## Step 3: Dividing both sides by $2 x - 1 0$ , we obtain

$$
( x - 3 ) ^ { 2 } - ( x - 3 ) ( x - 7 ) + ( x - 7 ) ^ { 2 } = ( 2 x - 1 0 ) ^ { 2 } .
$$

\## Step 4: Expanding both sides gives

$$
x ^ { 2 } - 6 x + 9 - \left( x ^ { 2 } - 1 0 x + 2 1 \right) + x ^ { 2 } - 1 4 x + 4 9 = 4 x ^ { 2 } - 4 0 x + 1 0 0 .
$$

Simplifying yields

$$
x ^ { 2 } - 1 0 x + 3 7 = 4 x ^ { 2 } - 4 0 x + 1 0 0 .
$$

\## Step 5: Rearranging and factoring gives

$$
3 x ^ { 2 } - 3 0 x + 6 3 = 0 , \qquad x ^ { 2 } - 1 0 x + 2 1 = ( x - 3 ) ( x - 7 ) = 0 .
$$

\## Step 6: Therefore, the roots are $x = 3$ and $x = 7$ , and the final answer is {3,7}.

TASCO REASONING (CORRECT):

\## Step 1: Let $y = x - 5$ . Then

$$
x - 3 = y + 2 , \qquad x - 7 = y - 2 , \qquad 2 x - 1 0 = 2 y .
$$

\## Step 2: The equation becomes

$$
( y + 2 ) ^ { 3 } + ( y - 2 ) ^ { 3 } = ( 2 y ) ^ { 3 } .
$$

Expanding both sides gives

$$
( y ^ { 3 } + 6 y ^ { 2 } + 1 2 y + 8 ) + ( y ^ { 3 } - 6 y ^ { 2 } + 1 2 y - 8 ) = 8 y ^ { 3 } .
$$

\## Step 3: Combining like terms, rearranging, and factoring gives

$$
2 y ^ { 3 } + 2 4 y = 8 y ^ { 3 } , \qquad 6 y ( y ^ { 2 } - 4 ) = 0 .
$$

\## Step 4: Thus,

$$
y = 0 , \qquad y = 2 , \qquad y = - 2 .
$$

Substituting $x = y + 5$ gives

$$
x = 5 , \qquad x = 7 , \qquad x = 3 .
$$

\## Step 5: Therefore, the complete set of roots is {3,5,7}.

CASE 2: DOMAIN CONSTRAINTS

QUESTION: For which real values of x is the following function undefined?

$$
f ( x ) = { \frac { 1 } { \left| x ^ { 2 } + 3 x - 4 \right| + \left| x ^ { 2 } + 9 x + 2 0 \right| } } .
$$

GROUND TRUTH: −4

CONFIDENCE-ONLY REASONING (INCORRECT):

\## Step 1: To determine the values outside the domain, we find where the denominator is zero. Since both terms are absolute values,

$$
| x ^ { 2 } + 3 x - 4 | + | x ^ { 2 } + 9 x + 2 0 | = 0
$$

if and only if both quadratic expressions are zero.

\## Step 2: The first quadratic factors as

$$
x ^ { 2 } + 3 x - 4 = ( x + 4 ) ( x - 1 ) = 0 ,
$$

giving $x = - 4$ or x = 1.

\## Step 3: The second quadratic factors as

$$
x ^ { 2 } + 9 x + 2 0 = ( x + 4 ) ( x + 5 ) = 0 ,
$$

giving $x = - 4 \operatorname { o r } x = - 5 .$

\## Step 4: The roots of the two equations are combined by union rather than intersection. The reasoning therefore concludes that the denominator is zero at $x = - 5 , - 4 , 1$ , giving {-5,-4,1}.

TASCO REASONING (CORRECT):

\## Step 1: The denominator is a sum of two nonnegative terms, so it is zero only when both quadratic expressions are zero simultaneously.

\## Step 2: Solving the first equation gives

$$
x ^ { 2 } + 3 x - 4 = ( x + 4 ) ( x - 1 ) = 0 ,
$$

whose roots are $x = - 4$ and $x = 1$

\## Step 3: Solving the second equation gives

$$
x ^ { 2 } + 9 x + 2 0 = ( x + 4 ) ( x + 5 ) = 0 ,
$$

whose roots are $x = - 4$ and $x = - 5$

\## Step 4: Because the two conditions must hold simultaneously, we take the intersection:

$$
\{ - 4 , 1 \} \cap \{ - 4 , - 5 \} = \{ - 4 \} .
$$

Therefore, the denominator is zero only at $x = - 4 ,$ and the final answer is {-4}.

## CASE 3: COPRIMALITY CONSTRAINTS

QUESTION: If a and b are positive integers such that $\operatorname* { g c d } ( a , b ) = 2 1 0 , \operatorname { l c m } ( a , b ) = 2 1 0 ^ { 3 }$ , and $a < b ,$ how many possible values are there for a?

GROUND TRUTH: 8

CONFIDENCE-ONLY REASONING (INCORRECT):

\## Step 1: Using

$$
\operatorname* { g c d } ( a , b ) \operatorname { l c m } ( a , b ) = a b ,
$$

we obtain

$$
a b = 2 1 0 \cdot 2 1 0 ^ { 3 } = 2 1 0 ^ { 4 } .
$$

\## Step 2: Since $\operatorname* { g c d } ( a , b ) = 2 1 0$ , write

$$
a = 2 1 0 m , \qquad b = 2 1 0 n ,
$$

where $\operatorname* { g c d } ( m , n ) = 1$ . Substituting gives

$$
m n = 2 1 0 ^ { 2 } .
$$

\## Step 3: The prime factorization is

$$
2 1 0 ^ { 2 } = 2 ^ { 2 } \cdot 3 ^ { 2 } \cdot 5 ^ { 2 } \cdot 7 ^ { 2 } .
$$

Therefore, the total number of divisors of $2 1 0 ^ { 2 }$ is

$$
( 2 + 1 ) ^ { 4 } = 8 1 .
$$

\## Step 4: Each divisor is treated as producing a coprime factor pair, even though many divisor pairs share prime factors. After removing symmetric pairs, the number of possible values of a is computed as

$$
{ \frac { 8 1 - 1 } { 2 } } = 4 0 .
$$

Therefore, the final answer is {40}.

TASCO REASONING (CORRECT):

\## Step 1: Using the relationship between the greatest common divisor and least common multiple gives

$$
a b = \operatorname* { g c d } ( a , b ) \operatorname { l c m } ( a , b ) = 2 1 0 \cdot 2 1 0 ^ { 3 } = 2 1 0 ^ { 4 } .
$$

\## Step 2: Write

$$
a = 2 1 0 m , \qquad b = 2 1 0 n ,
$$

where $\operatorname* { g c d } ( m , n ) = 1$ . Then

$$
( 2 1 0 m ) ( 2 1 0 n ) = 2 1 0 ^ { 4 } , \qquad m n = 2 1 0 ^ { 2 } .
$$

\## Step 3: Since

$$
2 1 0 ^ { 2 } = 2 ^ { 2 } \cdot 3 ^ { 2 } \cdot 5 ^ { 2 } \cdot 7 ^ { 2 }
$$

and m and n are coprime, each complete prime power must be assigned entirely to either m or n. Splitting a prime power between the two factors would violate gcd $( m , n ) = 1$

\## Step 4: There are four distinct prime powers, each of which can be assigned independently to either m or n. Hence, there are

$$
2 ^ { 4 } = 1 6
$$

ordered assignments. Since $a < b ,$ equivalently $m < n ,$ each unordered pair contributes exactly one valid value of a. Therefore,

$$
{ \frac { 1 6 } { 2 } } = 8 .
$$

The final answer is {8}.

Table 12: Three comparative case studies on polynomial roots, domain constraints, and coprime factorization. In each case, Confidence Only drops a condition identified earlier in the derivation, whereas TASCO preserves the condition and obtains the correct answer.