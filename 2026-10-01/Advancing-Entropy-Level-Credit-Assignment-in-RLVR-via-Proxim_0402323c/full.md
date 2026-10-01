# Advancing Entropy-Level Credit Assignment in RLVR via Proximal Entropy Policy Optimization<sup>∗</sup>

Yun Kim Seoul National University yunkimmy@snu.ac.kr

Nojun Kwak Seoul National University nojunk@snu.ac.kr

## Abstract

Value-model-free RLVR methods such as GRPO assign uniform advantages to all tokens in a rollout, ignoring that tokens contribute unequally. Recent methods use token entropy as an importance proxy but compute it globally across the batch, conflating importance with prompt difficulty and positional trends. We argue that importance should instead be measured relative to a token’s own local context, and introduce proximal entropy: a local measure of token importance relative to neighboring tokens, and prove it is invariant to both confounders. Proximal Entropy Policy Optimization (PEPO) uses it to weight per-token advantages and outperforms GRPO and entropy-based baselines on mathematical reasoning across Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B-Instruct. We also show the formulation generalizes to other algorithms where substituting proximal entropy into existing methods improves, and applying it to single-stream RL succeeds where global entropy fails.

## 1 Introduction

In Reinforcement Learning with Verifiable Rewards (RLVR) Lambert et al. [2024], Guo et al. [2025a] for large language models, value-model-free methods such as Group Relative Policy Optimization (GRPO) Shao et al. [2024] have become dominant for their efficiency. However, these methods assign a single sequence-level advantage to every generated token, ignoring the fact that tokens contribute unequally to the final answer. Several approaches address this through learned value models Schulman et al. [2017], Parthasarathi et al. [2025] or additional rollouts Kazemnejad et al. [2025], Guo et al. [2025b], but these require training a separate model or generating extra samples, adding significant overhead. Among lighter alternatives, token entropy has emerged as the most widely adopted proxy for token importance Wang et al. [2025a], Cheng et al. [2025a], Tan et al. [2025], requiring no overhead beyond the standard forward pass.

The premise of entropy-based credit assignment is that low-entropy tokens correspond to routine grammatical continuations, while high-entropy tokens mark pivotal decisions in the reasoning trajectory Wang et al. [2025a]. Methods such as 80/20 Wang et al. [2025a] exploit this by updating only the top 20% of tokens by entropy, ranked across all tokens in a training batch. However, we observe that this batch-level approach systematically biases updates toward hard prompts and early generation steps, where entropy is globally elevated. This causes low-importance tokens in hard prompts or early positions to be prioritized over genuine decision points, misguiding the learning signal. Therefore, we argue that a token’s importance should be measured relative to its own local context, not relative to the batch.

In this paper, we propose Proximal Entropy Policy Optimization (PEPO), which assigns each token an advantage weight based on its proximal entropy. Proximal entropy measures how uncertain the model is at a given position relative to its neighboring tokens, rather than relative to the entire batch. We define proximal entropy as a softmax over a sliding window of entropies centered on each token, providing a per-token importance signal that is invariant to prompt difficulty and robust to positional trends. We formally show that this formulation removes both confounders (Proposition 1) and empirically verify that proximal entropy remains constant across difficulty levels and generation positions where global entropy varies substantially. Importantly, computing proximal entropy adds negligible overhead, as token-level entropy is precomputed by standard LLM inference engines and the sliding window operations are fully vectorized.

![](images/db03f92cb6f71c33106b88aff2553fed821ab1a5afa3d60f225da82f2d832d5a.jpg)  
(a) Qwen3-1.7B

![](images/37e72428ffc1f31f1f5d438bd5a43727fed4ff18cbcaef5fc3c26ef3978c491c.jpg)  
(b) Qwen3-4B  
Figure 1: Validation accuracy over training steps for PEPO and the 80/20. The plotted values represent the average mean@k accuracy evaluated across the MATH, AIME 2024, AIME 2025, and AMC benchmarks for Qwen3-1.7B (a) and Qwen3-4B (b). PEPO achieves a higher validation accuracy compared to the 80/20 baseline.

We conduct extensive experiments on MATH500 Lightman et al. [2024], AMC Mathematical Association of America [2023], and AIME 2024/2025 Mathematical Association of America [2024] using Qwen3-1.7B, Qwen3-4B Yang et al. [2025], and Llama-3.2-3B-Instruct Grattafiori et al. [2024], where PEPO consistently outperforms batch-level methods on Qwen3-1.7B and Qwen3-4B (Figure 1). Beyond accuracy gains, we find that the proximal entropy formulation is itself the key ingredient. Substituting our local reference frame into 80/20 improves its performance without any other changes, isolating the contribution from the rest of PEPO’s design. The same formulation also transfers to single-stream RL (SPO) Xu and Ding [2025], where each prompt produces only a single rollout. In this setting, the diversity of prompts greatly increases, and batch-level entropy methods degrade below the baseline while PEPO continues to improve over it. Together, these results confirm that proximal entropy captures token importance more effectively than batch-level approaches, providing accurate credit assignment for value-model-free RL. We summarize our contributions as follows.

• We identify a systematic bias in batch-level entropy metrics Wang et al. [2025a], Cheng et al. [2025a] that concentrates updates on hard prompts and early generation steps.

• We propose PEPO, which weights per-token advantages using proximal entropy, a local measure of token importance that we formally prove is invariant to prompt difficulty and positional trends.

• We demonstrate that PEPO consistently outperforms batch-level methods across three model families (Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B-Instruct) on four benchmarks (MATH500, AMC, AIME 2024, and AIME 2025).

• We show that the proximal entropy formulation generalizes beyond PEPO, improving 80/20 when substituted in and transferring to single-stream RL where batch-level entropy fails.

## 2 Related Work

Value-model-free RLVR and credit assignment. GRPO [Shao et al., 2024] and its successors [Yu et al., 2025, Zheng et al., 2025, Zhao et al., 2025, Chen et al., 2025] eliminate the value network of PPO by estimating advantages through group-relative comparisons, and Single-stream Policy Optimization [Xu and Ding, 2025] further removes the group synchronization barrier. These methods differ in how they estimate and stabilize advantages but share a common limitation that every token in a rollout receives identical credit. A range of approaches address this through auxiliary signals. Learned value networks [Schulman et al., 2017, Parthasarathi et al., 2025], Monte Carlo estimates from extra rollouts or tree-structured branches [Kazemnejad et al., 2025, Guo et al., 2025b, Tran et al., 2025, Wang et al., 2026, Li et al., 2026], process reward models that score each step [Cheng et al., 2025b, Xie et al., 2025b, Zou et al., 2025], attention maps from selected heads [Li et al., 2025, Jiao et al., 2026, Nie et al., 2026], and gradient-based attribution [Zhang et al., 2026] all recover stronger credit signals at the cost of additional models, extra rollouts, or extra computation per token. Token entropy, by contrast, is computed from the same forward pass as the policy itself and adds essentially zero overhead, making it the lightweight signal we focus on in this work.

Token-entropy-based credit assignment. Token entropy has emerged as the dominant lightweight proxy for token importance, since it requires no overhead beyond the standard forward pass. Wang et al. [2025a] show that high-entropy tokens correspond to forking points in reasoning trajectories and propose 80/20, which restricts updates to the top 20% of tokens by entropy. Cheng et al. [2025a] augment the advantage with a clipped, gradient-detached entropy term to amplify updates at highentropy positions. GTPO [Tan et al., 2025] redistributes rewards by weighting each token by its entropy relative to other rollouts in its group, while UCAS [Xie et al., 2025a] modulates advantages using both response-level confidence and token-level certainty. ERPO [Yu et al., 2026] and EDGE-GRPO [Zhang et al., 2025] further explore entropy-aware gating and advantage diversity. A common thread across these methods is that entropy statistics are computed in a global reference frame, either across the entire batch or as absolute per-token quantities. As we demonstrate in Section 3.2, this global frame conflates token importance with prompt difficulty and positional trends, leading to systematic biases in which tokens receive credit.

Closest to our work, ARES [Chen et al., 2026] uses window-averaged entropy to shape per-token advantages in multimodal reasoning, adding a bonus on tokens whose window entropy exceeds a batch-derived threshold. PEPO differs in three respects. First, ARES uses a forward-looking window with arithmetic averaging to smooth entropy and thresholds it against a batch-global cutoff, whereas PEPO uses a centered window with softmax-relative weighting that measures each token against its local neighbors and is therefore invariant to the absolute scale. Second, the symmetric window is essential to our positional invariance result (Proposition 1), since the cancellation of linear positional trends in the proof relies on offsets ranging symmetrically around t; a forward-looking window would retain a residual trend bias. Third, we show that the local frame is a drop-in improvement that transfers across algorithms and to the single-stream setting where global entropy methods fail.

## 3 Background and Motivation

## 3.1 Preliminaries

We consider the RLVR setting [Lambert et al., 2024, Guo et al., 2025a], in which a policy $\pi _ { \theta }$ generates a response $y = ( y _ { 1 } , \dots , y _ { T } )$ ) conditioned on a prompt $x ,$ and a verifier assigns a binary reward $r ( x , y ) \bar { \in } \{ 0 , 1 \}$ indicating whether y is correct. The goal is to maximize the expected reward $\mathbb { E } _ { { x } \sim { p _ { d a t a } } , { y } \sim \pi _ { \theta } ( { . } | { x } ) } [ r ( \dot { x } , { y } ) ]$

GRPO. For each prompt $x ,$ GRPO [Shao et al., 2024] samples a group of $G$ rollouts $\{ y _ { i } \} _ { i = 1 } ^ { G }$ from the current policy and computes a group-normalized advantage

$$
A _ { i } = \frac { r _ { i } - \mathrm { m e a n } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } ,\tag{1}
$$

where $r _ { i } = r ( x , y _ { i } )$ . The same advantage $A _ { i }$ is assigned to every token $y _ { i , t }$ in the rollout, and the policy is updated by maximizing

$$
\mathcal { I } _ { \mathrm { g r o u p } } ( \theta ) = \frac { 1 } { \sum _ { j = 1 } ^ { G } T _ { j } } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \bigl ( \rho _ { i , t } ( \theta ) A _ { i } , \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \varepsilon _ { l } , 1 + \varepsilon _ { h } ) A _ { i } \bigr ) ,\tag{2}
$$

where $T _ { i }$ is the number of tokens generated in the i-th rollout, $\rho _ { i , t } ( \theta ) = \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid$ $x , y _ { i , < t } )$ is the importance ratio and $\varepsilon _ { l } , \varepsilon _ { h }$ are the lower and upper clipping thresholds [Schulman et al., 2017, Yu et al., 2025].

Token entropy. The entropy of the policy at step t of rollout i is

$$
H _ { i , t } = - \sum _ { v \in \mathcal { V } } \pi _ { \theta } ( v \mid x , y _ { i , < t } ) \log \pi _ { \theta } ( v \mid x , y _ { i , < t } ) ,\tag{3}
$$

where V is the vocabulary.

Fine-grained credit assignment. Because A in Eq. (1) is constant across t, GRPO assigns identical credit to every token in a rollout, regardless of its role in producing $r _ { i }$ . We refer to the problem of assigning per-token weights $c _ { i , t } .$ , such that the effective advantage $c _ { i , t } A _ { i }$ reflects each token’s contribution, asfine-grained credit assignment. The challenge is that r provides no direct signal about which tokens mattered, so any solution must rely on a proxy for token importance. A learned value model can provide such a proxy [Schulman et al., 2017, Parthasarathi et al., 2025], but introduces substantial memory and training overhead and reintroduces the instability issues that motivated value-free RLVR in the first place [Shao et al., 2024, Yu et al., 2025]. In this work, we explore token entropy as the importance signal, since it is a lightweight proxy that introduces no additional overhead, as it is already computed during standard LLM inference [Kwon et al., 2023].

## 3.2 Entropy as a Token Importance Signal

Token entropy has emerged as a model-free alternative to learned value models [Wang et al., 2025a, Cheng et al., 2025a, Tan et al., 2025]. During generation, routine continuations such as grammatical placeholders and function words carry little uncertainty, while tokens at logical branch points, intermediate numerical results, and operator choices exhibit high entropy and frequently determine whether a derivation proceeds correctly [Wang et al., 2025a]. A growing line of work exploits this by using entropy to instantiate the per-token weight $c _ { i , t }$ [Wang et al., 2025a, Cheng et al., 2025a, Xie et al., 2025a]. For instance, Wang et al. [2025a] show that high-entropy tokens correspond toforking points in the reasoning trajectory and propose 80/20, which updates only the top 20% of tokens by entropy:

![](images/08ac75f5d901b10b11307754e67f8b5a6f07a34fa1d83e706d847baf2cff75e0.jpg)  
Figure 2: Positional bias in token entropy on Llama-3.2-3B-Instruct. We plot the deviation of global and proximal entropy from its sequencelevel mean across generation percentiles.

$$
c _ { i , t } = \mathbb { 1 } \left[ H _ { i , t } \geq \tau _ { 8 0 } \right] ,\tag{4}
$$

where $\tau _ { 8 0 }$ is the 80th-percentile entropy across all tokens in the batch. Cheng et al. [2025a] similarly augments the advantage function with a global entropy term to reinforce exploratory reasoning behaviors. In both cases, the reference frame is global.

However, a token’s raw entropy is driven not only by its importance but also by two confounding factors: the token’s position in the sequence and the difficulty of the prompt. Figure 2 illustrates the positional bias on Llama-3.2-3B-Instruct by plotting the deviation of global entropy from its sequence-level mean across generation percentiles, computed over 1024 rollouts from the validation set. Global entropy exhibits a strong positional trend, with early tokens deviating up to 80% above the mean and late tokens falling well below it, reflecting that early generation steps carry higher uncertainty before the model commits to a reasoning direction [Guo, 2025]. Table 1 illustrates the prompt difficulty bias. We group prompts by pass rate (out of 8) into hard (0–1), moderate (2–6), and easy (7–8) categories and report the mean entropy for each group. Global entropy varies substantially across difficulty levels (0.315 for hard vs. 0.221 for easy), confirming that prompt difficulty elevates entropy across all tokens in a sequence. Together, these results demonstrate that batch-level entropy statistics systematically prioritize tokens from hard prompts and early positions, regardless of whether those tokens represent genuine decision points.

Table 1: Prompt difficulty bias in token entropy on Llama-3.2-3B-Instruct.
<table><tr><td>Global Entropy</td><td></td><td>Proximal Entropy</td></tr><tr><td>Hard</td><td>0.315</td><td>0.00987</td></tr><tr><td>Moderate</td><td>0.270</td><td>0.00987</td></tr><tr><td>Easy</td><td>0.221</td><td>0.00989</td></tr></table>

To understand why, consider the effect of prompt difficulty alone. Let $\begin{array} { r } { \bar { H } _ { i } = \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } H _ { i , t } } \end{array}$ <sub>t</sub> denote the mean entropy of rollout i, which serves as a measure of overall prompt difficulty. Each token’s entropy decomposes as

$$
\begin{array} { r } { H _ { i , t } = \underbrace { \bar { H } _ { i } } _ { \mathrm { s e q u e n c e - l e v e l } } + \underbrace { ( H _ { i , t } - \bar { H } _ { i } ) } _ { \mathrm { t o k e n - s p e c i f i c } } , } \end{array}\tag{5}
$$

where the token-specific component measures how surprising a token is relative to its own sequence, serving as a more reliable indicator of importance. Global methods, however, rank tokens by $H _ { i , t }$ across the entire batch, allowing tokens to receive disproportionate credit simply because they belong to a high-entropy sequence. Accounting for the positional bias only compounds this problem, as the systematic decay of entropy over the course of generation means that early tokens receive additional unwarranted credit due to their position alone.

These observations motivate a token importance measure that accounts for both prompt difficulty and positional variation. Rather than asking which tokens have high entropy relative to the global distribution, we should ask which tokens have high entropy relative to their own local context. In the next section, we formalize this local reference frame and introduce PEPO, which replaces global entropy statistics with a sliding window centered around each token.

## 4 Proximal Entropy Policy Optimization

The key idea behind PEPO is to replace the global entropy statistics used by prior methods with a local measure we call proximal entropy. For each token, proximal entropy captures how uncertain the model is at that position relative to its neighboring tokens, rather than relative to the entire batch. This provides a per-token importance signal that is invariant to prompt difficulty and position.

Proximal entropy. Given the entropy sequence $( H _ { i , 1 } , \ldots , H _ { i , T _ { i } } )$ for rollout i, we define the proximal entropy of token t using a softmax over a sliding window of size W centered on t:

$$
e _ { i , t } = \frac { \exp ( H _ { i , t } ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp ( H _ { i , k } ) } ,\tag{6}
$$

where $\mathcal { W } ( t ) = \{ k : | k - t | \leq W / 2 \}$ is the set of token positions within the window. The softmax formulation ensures that a token’s proximal entropy reflects its uncertainty relative to its neighbors, not its absolute value. A token with moderately high entropy in an otherwise low-entropy region receives a large proximal entropy, while a token with the same entropy surrounded by equally uncertain neighbors does not.

Per-token advantage weighting. We use proximal entropy to modulate the GRPO advantage. Since each token’s proximal entropy is computed over its own local window, the values do not sum to one across the sequence. We therefore normalize and scale them to preserve the total advantage magnitude:

$$
{ \hat { A } } _ { i , t } = c _ { i , t } \cdot A _ { i } , \quad { \mathrm { w h e r e } } \quad c _ { i , t } = { \frac { e _ { i , t } } { \sum _ { k = 1 } ^ { T _ { i } } e _ { i , k } } } \cdot T _ { i } .\tag{7}
$$

This ensures that $\textstyle \sum _ { t } c _ { i , t } = T _ { i }$ , so the total advantage magnitude is identical to that of standard GRPO. In practice, we apply proximal-entropy weighting only to positive-reward rollouts, consistent with prior entropy-weighted rewards for successful responses [Tan et al., 2025] and asymmetric weighting of positive and negative samples [Ma et al., 2026]. We leave failed rollouts unweighted because amplifying their negative updates at high-entropy positions may discourage exploration. The PEPO objective is then

$$
\mathcal { T } _ { \mathrm { P E P } } ( \theta ) = \frac { 1 } { \sum _ { j = 1 } ^ { G } T _ { j } } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \bigl ( \rho _ { i , t } ( \theta ) \hat { A } _ { i , t } , \ \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \varepsilon _ { l } , 1 + \varepsilon _ { h } ) \hat { A } _ { i , t } \bigr ) .\tag{8}
$$

This is identical to the standard GRPO objective in Eq. (2), with the only difference being that the uniform advantage $A _ { i }$ is replaced by the entropy-weighted $\hat { A } _ { i , t }$

Table 2: Main results on mathematical reasoning benchmarks. Trained methods are reported as mean ± sample standard deviation (SD) over three independent runs. We report avg@1 on MATH500 and avg@16 on AMC, AIME 2024, and AIME 2025; the Mean column summarizes each run’s four benchmark scores. Best results per model are in bold.
<table><tr><td>Model</td><td>Method</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td><td>Mean</td></tr><tr><td rowspan="5">Qwen3-1.7B</td><td>Base</td><td> $1 4 . 0 2 \pm 0 . 8 2$ </td><td> $1 7 . 5 8 \pm 0 . 7 4$ </td><td> $4 5 . 2 1 \pm 0 . 6 5$ </td><td> $7 5 . 5 3 \pm 0 . 4 8$ </td><td> $3 8 . 0 9 \pm 0 . 4 2 $ </td></tr><tr><td>GRPO</td><td> $1 6 . 5 8 \pm 0 . 4 4$ </td><td> $1 8 . 6 5 \pm 0 . 6 2$ </td><td> $5 0 . 2 8 \pm 0 . 9 9$ </td><td> $8 0 . 2 7 \pm 1 . 0 1$ </td><td> $4 1 . 4 5 \pm 0 . 5 0$ </td></tr><tr><td>Entropy Adv.</td><td> $2 1 . 4 4 \pm 1 . 8 1$ </td><td> $1 9 . 4 8 \pm 1 . 7 2$ </td><td> $5 2 . 5 2 \pm 1 . 0 6$ </td><td> $8 1 . 4 0 \pm 1 . 4 4$ </td><td> $4 3 . 7 1 \pm 0 . 5 2$ </td></tr><tr><td>80/20</td><td> $2 0 . 9 7 \pm 1 . 4 8$ </td><td> $2 0 . 8 3 \pm 0 . 7 5$ </td><td> $5 2 . 1 3 \pm 0 . 4 4$ </td><td> $8 0 . 7 3 \pm 1 . 5 0$ </td><td> $4 3 . 6 7 \pm 0 . 3 0$ </td></tr><tr><td>PEPO (Ours)</td><td> $\mathbf { 2 1 . 7 4 \ : \pm 1 . 0 6 }$ </td><td> $2 2 . 4 3 \pm 0 . 6 7$ </td><td> ${ \pm } 5 5 . 2 6 \pm 1 . 1 1$ </td><td> $\mathbf { 8 2 . 6 7 \ : \pm 0 . 3 1 }$ </td><td> $\mathbf { 4 5 . 5 3 \pm 0 . 4 6 }$ </td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Base</td><td> $2 5 . 6 2 \pm 0 . 9 5$ </td><td> $2 0 . 1 4 \pm 0 . 8 1$ </td><td> $5 5 . 3 9 \pm 0 . 7 2$ </td><td> $7 9 . 8 0 \pm 0 . 5 4$ </td><td> $4 5 . 2 4 \pm 0 . 5 1$ </td></tr><tr><td>GRPO</td><td> $3 2 . 7 5 \pm 0 . 5 4$ </td><td> $2 3 . 5 4 \pm 0 . 7 6$ </td><td> $5 9 . 9 6 \pm 0 . 6 8$ </td><td> $8 4 . 3 3 \pm 1 . 2 2$ </td><td> $5 0 . 1 5 \pm 0 . 4 3$ </td></tr><tr><td>Entropy Adv.</td><td> $3 5 . 8 8 \pm 1 . 5 4$ </td><td> $2 6 . 2 1 \pm 1 . 2 2$ </td><td> $6 7 . 8 3 \pm 0 . 5 7$ </td><td> $8 9 . 7 3 \pm 0 . 2 3$ </td><td> $5 4 . 9 1 \pm 0 . 2 3$ </td></tr><tr><td>80/20</td><td> $3 5 . 3 5 \pm 2 . 4 7$ </td><td> $3 0 . 6 2 \pm 0 . 2 1$ </td><td> $6 7 . 2 5 \pm 1 . 5 3$ </td><td> $8 9 . 7 3 \pm 1 . 1 5$ </td><td> $5 5 . 7 4 \pm 0 . 8 6$ </td></tr><tr><td>PEPO (Ours)</td><td> $3 9 . 2 4 \pm 1 . 2 7$ </td><td> ${ \bf 3 1 . 8 1 \pm 0 . 1 3 }$ </td><td> ${ \bf 6 9 . 5 8 \pm 0 . 5 9 }$ </td><td> $\mathbf { 9 1 . 4 0 \pm 1 . 8 3 }$ </td><td> ${ \pm 8 . 0 1 \pm 0 . 1 4 }$ </td></tr><tr><td rowspan="5">Llama-3.2-3B-Instruct</td><td>Base</td><td> $5 . 6 9 \pm 0 . 4 5$ </td><td> $0 . 6 2 \pm 0 . 1 8$ </td><td> $2 1 . 4 9 \pm 0 . 5 2$ </td><td> $4 5 . 9 3 \pm 0 . 6 1$ </td><td> $1 8 . 4 3 \pm 0 . 2 8$ </td></tr><tr><td>GRPO</td><td> $7 . 2 2 \pm 0 . 8 4$ </td><td> $0 . 9 0 \pm 0 . 5 2$ </td><td> $2 1 . 4 3 \pm 0 . 5 7$ </td><td> $4 7 . 6 7 \pm 1 . 3 3$ </td><td> $1 9 . 3 1 \pm 0 . 2 0$ </td></tr><tr><td>Entropy Adv.</td><td> $8 . 2 6 \pm 0 . 8 4$ </td><td> $0 . 5 6 \pm 0 . 2 4$ </td><td> $2 3 . 6 4 \pm 0 . 7 4$ </td><td> $4 7 . 6 0 \pm 0 . 6 0$ </td><td> $2 0 . 0 2 \pm 0 . 4 4$ </td></tr><tr><td>80/20</td><td> $8 . 6 0 \pm 0 . 6 1$ </td><td> $0 . 4 9 \pm 0 . 1 2$ </td><td> $2 2 . 5 7 \pm 0 . 9 1$ </td><td> $4 7 . 6 7 \pm 0 . 5 0$ </td><td> $1 9 . 8 3 \pm 0 . 0 3$ </td></tr><tr><td>PEPO (Ours)</td><td> ${ \bf 9 . 1 0 \pm 0 . 5 2 }$ </td><td> ${ \bf 1 . 4 6 \pm 0 . 5 5 }$ </td><td> $2 4 . 1 2 \pm 0 . 3 6$ </td><td> $\mathbf { 4 9 . 3 3 \ : \pm 0 . 1 2 }$ </td><td> ${ \bf 2 1 . 0 0 \pm 0 . 1 5 }$ </td></tr></table>

Theoretical justification. We now show that proximal entropy formally removes the confounders identified in Section 3.2.

Proposition 1. The proximal entropy $e _ { i , t }$ in $E q .$ . (6) satisfies:

(i) If all entropies in a sequence are shifted by a sequence-dependent constant $\delta _ { i } ,$ , then $e _ { i , t }$ is unchanged.

(ii) If entropies in a sequence follow a smooth monotonic trend, then $e _ { i , t }$ is approximately invariant to the trendfor sufficiently small W.

Property (i) implies that proximal entropy is invariant to prompt difficulty, as all tokens in a sequence share the same sequence-level component ${ \bar { H } } _ { i } .$ . Property (ii) implies robustness to positional entropy trends, which manifest as a smooth decay over the course of generation. Together, these properties ensure that proximal entropy isolates the token-specific signal from both confounders. The full proof is provided in Appendix B.

We note that Figure 2 and Table 1 empirically corroborate both properties. As shown in Figure 2, global entropy deviates by up to 80 points from its sequence-level mean across generation percentiles, while proximal entropy remains flat throughout, confirming the positional invariance of property (ii). Table 1 further shows that global entropy varies from 0.221 (easy) to 0.315 (hard) across prompt difficulty groups, whereas proximal entropy remains constant at ≈0.00988 across all three groups, confirming the shift invariance of property (i).

## 5 Experiments

## 5.1 Experimental Setup

Models. We evaluate on three base models spanning two model families: Qwen3-1.7B, Qwen3- 4B [Yang et al., 2025], and Llama-3.2-3B-Instruct [Grattafiori et al., 2024].

Training. For the main experiments, we use the ROLL framework [Wang et al., 2025b] with the DeepMath-103K dataset [He et al., 2025]. For single-stream RL experiments (Section 5.4), we use the VeRL framework [Sheng et al., 2024] with the DAPO-Math-17K dataset [Yu et al., 2025]. Al methods are implemented on top of GRPO [Shao et al., 2024] with $G = 8$ rollouts per prompt, a learning rate of $\mathrm { i } \times \mathrm { 1 0 ^ { - 6 } }$ , clipping threshold $\varepsilon _ { h } = 0 . 2 8$ , and 500 training steps, and follow the default library configurations. For PEPO, we set the window size $W = 1 0 1$ across all experiments. The average response length during training ranges from approximately 2000 to 3000 tokens depending on the model, so the window covers roughly 3–5% of each sequence. Full hyperparameter details are provided in Appendix.

Table 3: Effect of substituting proximal entropy into 80/20. Each cell reports the mean ± sample SD over three independent runs. All other components of 80/20 remain unchanged. PEPO results from Table 2 are included for reference. Best results within each model are bolded.
<table><tr><td>Model</td><td>Method</td><td>AIME2024 AIME2025</td><td></td><td>AMC</td><td>MATH500</td><td>Mean</td></tr><tr><td rowspan="3">Qwen3-1.7B</td><td>80/20</td><td> $2 0 . 9 7 \pm 1 . 4 8$ </td><td> $2 0 . 8 3 \pm 0 . 7 5$ </td><td> $5 2 . 1 3 \pm 0 . 4 4$ </td><td> $8 0 . 7 3 \pm 1 . 5 0$ </td><td> $4 3 . 6 7 \pm 0 . 3 0$ </td></tr><tr><td>80/20 + Proximal Entropy</td><td> $\pm 2 . 0 1 \pm 0 . 8 7$ </td><td> $2 0 . 6 9 \pm 0 . 2 4$ </td><td> $5 3 . 3 6 \pm 0 . 7 8$ </td><td> $8 2 . 2 7 \pm 0 . 3 1$ </td><td> $\mathbf { 4 4 . 5 8 \pm 0 . 4 3 }$ </td></tr><tr><td>PEPO (Ours)</td><td> $2 1 . 7 4 \pm 1 . 0 6$ </td><td> $2 2 . 4 3 \pm 0 . 6 7$ </td><td> ${ \pm } 5 5 . 2 6 \pm 1 . 1 1$ </td><td> $\mathbf { 8 2 . 6 7 \pm 0 . 3 1 }$ </td><td> $4 5 . 5 3 \pm 0 . 4 6$ </td></tr><tr><td rowspan="3">Qwen3-4B</td><td>80/20</td><td> $3 5 . 3 5 \pm 2 . 4 7$ </td><td> $3 0 . 6 2 \pm 0 . 2 1$ </td><td> $6 7 . 2 5 \pm 1 . 5 3$ </td><td> $8 9 . 7 3 \pm 1 . 1 5$ </td><td> $5 5 . 7 4 \pm 0 . 8 6$ </td></tr><tr><td>80/20 + Proximal Entropy 36.18 ± 2.61</td><td></td><td> $3 1 . 6 0 \pm 0 . 4 8$ </td><td> $6 9 . 3 8 \pm 1 . 4 1$ </td><td> $9 0 . 3 3 \pm 0 . 3 1$ </td><td> $5 6 . 8 7 \pm 0 . 5 6$ </td></tr><tr><td>PEPO (Ours)</td><td> ${ \bf 3 9 . 2 4 \pm 1 . 2 7 }$ </td><td> ${ \bf 3 1 . 8 1 \pm 0 . 1 3 }$ </td><td> ${ \bf 6 9 . 5 8 \pm 0 . 5 9 }$ </td><td> ${ \bf 9 1 . 4 0 \pm 1 . 8 3 }$ </td><td> ${ \bf 5 8 . 0 1 \pm 0 . 1 4 }$ </td></tr></table>

Baselines. We compare PEPO against three baselines: (1) standard GRPO [Shao et al., 2024] with uniform token weighting, (2) 80/20 [Wang et al., 2025a], which updates only the top 20% of tokens by global entropy, and (3) Entropy-augmented advantage [Cheng et al., 2025a], which augments the advantage function with a global entropy term.

Evaluation. We evaluate on MATH500 [Lightman et al., 2024], AMC [Mathematical Association of America, 2023], AIME 2024 [Mathematical Association of America, 2024], and AIME 2025 [Mathematical Association of America, 2025]. We report avg@1 accuracy on MATH500 and avg@16 accuracy on AMC, AIME 2024, and AIME 2025 over three runs. All detailed per-run results are provided in the Appendix.

## 5.2 Main Results

Table 2 presents the main results across all models and benchmarks. PEPO consistently outperforms both GRPO and batch-level entropy methods across all three models spanning two distinct model families, demonstrating that the improvements are robust and not an artifact of a particular architecture.

On Qwen3-4B, PEPO achieves the largest improvements, outperforming GRPO by 6.49%p on AIME 2024 and 9.62%p on AMC, and surpassing 80/20 by 2.27%p in mean accuracy across all benchmarks. On Qwen3-1.7B, PEPO improves over GRPO by 5.16%p on AIME 2024 and 4.98%p on AMC, with a mean accuracy gain of 1.82%p over Entropy Advantage. Notably, on Llama-3.2-3B-Instruct, 80/20 and Entropy Adv. degrade below GRPO on AIME 2025 (0.41%p and 0.34%p), while PEPO improves over GRPO on all benchmarks. The consistent gains of PEPO, even in cases where global entropy methods hurt performance, suggest that proximal entropy provides a more reliable token importance signal across model families.

## 5.3 Proximal Entropy as a General Formulation

To isolate the contribution of the proximal entropy formulation from the rest of PEPO’s design, we substitute our window-relative entropy into 80/20, replacing its global entropy ranking with proximal entropy while keeping all other components unchanged. Concretely, instead of selecting the top 20% of tokens by global entropy across the batch, we select the top 20% by proximal entropy within the same batch. This tests whether proximal entropy is indeed a more accurate measure of token importance compared to global entropy. Table 3 shows the results.

Replacing global entropy with proximal entropy improves 80/20 across benchmarks on both models, with no other changes to the algorithm. This confirms that proximal entropy is a more accurate measure of token importance regardless of the algorithm it is embedded in. PEPO, which further replaces binary token selection with continuous entropy-based weighting, achieves the best overall performance across both models.

## 5.4 Transfer to Single-Stream RL

To test whether the proximal entropy formulation transfers beyond GRPO, we apply it to Single-stream Policy Optimization (SPO) [Xu and Ding, 2025], where each prompt produces only a single rollout. We refer to this variant as PESPO (Proximal Entropy SPO). We use the same window size $\begin{array} { r l r } { W } & { { } = } & { 1 0 1 } \end{array}$ and all other hyperparameters as in the main experiments, without any task-specific tuning. Table 4 shows the results.

Both global entropy methods degrade below the SPO baseline in mean accuracy, with S-Entropy Adv. dropping to 54.65% and S-80/20 to 55.99%, compared to SPO’s 57.13% in mean accuracy. The degradation is particularly pronounced on AIME 2025, where S-Entropy Adv. falls to 21.80%, nearly 10%p below SPO. This is consistent with the failure mode identified in Section 3.2. Without multiple rollouts per prompt, the batch contains a more di-

Table 4: Results on single-stream RL (SPO) with Qwen3- 4B. Cells show the mean followed by sample SD in smaller type, computed over three independent runs. All methods are implemented on top of SPO. PESPO uses the same hyperparameters as in the main experiments without modification.
<table><tr><td>Method</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td><td>Mean</td></tr><tr><td>SPO</td><td> $3 5 . 7 0 \pm 1 . 9 7$ </td><td> $3 1 . 3 9 \pm 0 . 4 4$ </td><td> $7 0 . 8 1 \pm 1 . 2 8$ </td><td>90.60 ± 0.53</td><td> $5 7 . 1 3 \pm 0 . 4 2$ </td></tr><tr><td>S-Entropy Adv.</td><td> $3 8 . 8 9 \pm 1 . 2 6$ </td><td> $2 1 . 8 0 \pm 0 . 6 7$ </td><td> $6 8 . 4 2 \pm 0 . 4 3$ </td><td> $8 9 . 4 7 \pm 0 . 3 1$ </td><td> $5 4 . 6 5 \pm 0 . 5 8$ </td></tr><tr><td>S-80/20</td><td> $3 5 . 2 8 \pm 1 . 1 5$ </td><td> $2 7 . 6 4 \pm 1 . 0 5$ </td><td> $7 0 . 0 6 \pm 0 . 9 3$ </td><td> $\mathbf { 9 1 . 0 0 \ : \pm 0 . 4 0 }$ </td><td> $5 5 . 9 9 \pm 0 . 4 3$ </td></tr><tr><td>PESPO (Ours)</td><td> ${ \bf 4 1 . 7 4 \pm 0 . 5 2 }$ </td><td> $\mathbf { 3 1 . 6 0 \ : \pm 0 . 7 9 }$ </td><td> ${ \bf 7 1 . 7 1 \pm 1 . 0 6 }$ </td><td> $9 0 . 6 7 \pm 0 . 3 1$ </td><td> ${ \bf 5 8 . 9 3 \pm 0 . 4 8 }$ </td></tr></table>

verse set of prompts, amplifying the prompt difficulty bias in global entropy statistics. PESPO, in contrast, achieves the highest mean accuracy (58.93%) and improves over SPO on AIME 2024 by 6.04%p, demonstrating that proximal entropy transfers to single-stream settings without hyperparameter adjustment. This provides further evidence that proximal entropy effectively captures token importance even when prompt diversity is large, whereas global entropy does not.

## 5.5 Analysis

Token intervention. To test the causal effect of the token-importance metrics on accuracy, we use each metric to choose which generated tokens to replace, following prior token-replacement analyses [Meng et al., 2026, Huang et al., 2026]. For AIME 2024 avg@16, we generate responses with Qwen3-4B Base and rank their token positions by global entropy or proximal entropy. At each replacement budget, we replace the highest-ranked Base tokens with samples from a fixed GRPO checkpoint, then continue generation autoregressively. We hold the Base model, replacement checkpoint, evaluation set, and budget fixed, so only the ranking metric changes. Table 5 reports the resulting accuracy.

Table 5: Token-replacement intervention on AIME 2024 using Qwen3-4B. Results are avg@16 accuracy (%); each budget is the fraction of token positions replaced.
<table><tr><td>Budget</td><td>Global entropy</td><td>Proximal entropy</td></tr><tr><td>0%</td><td>25.62</td><td>25.62</td></tr><tr><td>5%</td><td>27.34</td><td>31.62</td></tr><tr><td>10%</td><td>29.91</td><td>33.33</td></tr><tr><td>15%</td><td>30.76</td><td>33.33</td></tr><tr><td>20%</td><td>32.47</td><td>33.33</td></tr><tr><td>30%</td><td>33.33</td><td>33.33</td></tr></table>

At a 5% replacement budget, proximal entropy reaches 31.62% accuracy versus 27.34% for global entropy, a 4.28-point gap. At 10%, it reaches the GRPO checkpoint’s 33.33% accuracy; global entropy reaches that level only at 30%. These results provide evidence that proximal entropy is a more accurate measure of token importance than global entropy in this setting, as it allows the GRPO checkpoint’s full accuracy to be recovered with a smaller replacement budget, further supporting the claims in the main section.

Window size. We ablate the window size W on Qwen3-4B, comparing $W \in \{ 5 1 , 1 0 1 , 1 5 1 \}$ Results are shown in Table 6. Performance is relatively robust across all three window sizes, suggesting that PEPO is not sensitive to this hyperparameter. This is consistent with the local smoothness assumption in Proposition 1. As long as the window is small enough relative to the sequence length, the positional trend is approximately cancelled. Among the three, W = 101 achieves the best results, which we use as the default across all experiments.

## Prompt:

What is the remainder when $1 2 9 ^ { \sim } 3 4 ~ + ~ 9 6 ^ { \sim } 3 8$ is divided by 11?

## (a) Global Entropy

![](images/4aae4e250c5dc4dbf1a8348562c19f7870f941a2ab95967633bfd8ba9ecbf201.jpg)

## (b) Proximal Entropy

![](images/e2ca710c231f92d91d61c366743cbcf9f3882c66da72e7fbca38d918deec8190.jpg)  
Figure 3: Qualitative comparison of token weighting on a Qwen3-4B rollout. (a) Global entropy assigns broad, near-uniform weight across structural and decision tokens alike (e.g., the entire phrases “We’ll calculate” and “property of periodicity”). (b) Proximal entropy evaluates each token relative to its local neighborhood, producing fine-grained sub-phrase weighting that singles out the tokens that steer the reasoning trajectory (e.g., calculate, periodicity) while still preserving signal across the derivation steps.

Table 6: Ablation studies on window size on Qwen3-4B. Each cell shows the mean followed by sample SD in smaller type over three independent runs.
<table><tr><td>W</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td></tr><tr><td> $W = 5 1$ </td><td> $3 6 . 2 5 \pm 1 . 5 0$ </td><td> $3 0 . 6 2 \pm 1 . 3 0$ </td><td> $6 8 . 3 9 \pm 0 . 5 9$ </td><td> $9 0 . 6 7 \pm 0 . 4 2$ </td></tr><tr><td> $W = 1 0 1$ </td><td> $3 9 . 2 4 \pm 1 . 2 7$ </td><td> $3 1 . 8 1 \pm 0 . 1 3$ </td><td> $6 9 . 5 8 \pm 0 . 5 9$ </td><td> $9 1 . 4 0 \pm 1 . 8 3 $ </td></tr><tr><td> $W = 1 5 1$ </td><td> $3 6 . 8 1 \pm 0 . 2 4$ </td><td> $2 8 . 2 7 \pm 1 . 2 6$ </td><td> $6 9 . 3 5 \pm 0 . 4 9$ </td><td> $9 2 . 2 7 \pm 0 . 4 6$ </td></tr></table>

Qualitative analysis. Figure 3 (Appendix) visualizes the token-level scores produced by global entropy and proximal entropy on a Qwen3-4B rollout for the modular arithmetic problem $1 \bar { 2 } 9 ^ { 3 4 } +$ $9 6 ^ { 3 8 }$ mod 11, where each token is colored by its score relative to the mean of each metric. Two properties of proximal entropy stand out. First, proximal entropy produces fine-grained, sub-phrase weighting that global entropy cannot. In “We’ll calculate,” for instance, global entropy lights up nearly uniformly across all three tokens, whereas proximal entropy distinguishes between them. We and calculate receive substantially more weight than ’ll, with calculate receiving the strongest weight as the token that commits to the action. The same effect appears in “property of periodicity,”

where global entropy assigns elevated weight to all three content words, while proximal entropy concentrates on periodicity, the noun that names the mathematical principle being invoked. This local reference frame allows proximal entropy to identify the specific tokens that steer the reasoning trajectory, even within short phrases where every token shares similar local context.

Second, proximal entropy does not ignore the derivation itself. The execution tokens that carry out the modular arithmetic — $^ { \ast \cdot } 8 ^ { 2 } = 6 4 { \overset { \sim } { = } } 1 1 \cdot 5 + 9 , ^ { , \ast } { } ^ { \ast } 8 1 = 1 1 \cdot 7 + 4 ^ { , \ast }$ receive visibly nontrivial weight under proximal entropy, even though their absolute entropy is near zero (each digit follows deterministically from the formula structure). Global entropy, in contrast, washes these tokens out entirely. Proximal entropy’s window-relative formulation preserves signal across the derivation while still concentrating attention on the higher-leverage decision points.

Additional evaluation and ablations, including held-out code generation, the weighting design, and temperature sensitivity, are reported in Appendix F.

## 6 Conclusion

We introduced Proximal Entropy Policy Optimization (PEPO), which replaces global entropy statistics with proximal entropy, a local measure of token importance for value-model-free RLVR. By computing entropy relative to a sliding window around each token, PEPO identifies tokens that are genuinely important given their local context, rather than tokens that merely reside in high-entropy regions of the batch. We formally showed that this formulation is invariant to prompt difficulty and approximately invariant to positional entropy trends (Proposition 1), directly addressing the confounders we identified in global entropy methods. Experiments across Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B-Instruct demonstrated consistent improvements over GRPO and batch-level entropy methods on mathematical reasoning benchmarks. On a held-out LiveCodeBench split, Qwen3-4B with PEPO also improved pass@1 over GRPO and 80/20. We further showed that the proximal entropy formulation generalizes beyond PEPO itself. Substituting it into 80/20 yields gains without any other changes, and applying it to single-stream RL (PESPO) improves performance where global entropy degrades.

Limitations and future work. Our main benchmark suite focuses on mathematical reasoning. The additional coding evaluation covers one held-out LiveCodeBench split and Qwen3-4B, so broader evaluation across coding benchmarks and model families is needed to establish how well PEPO generalizes to code generation, agentic tool use, and multi-turn dialogue. We did not compare against value-based methods such as PPO, which provide a stronger but more expensive form of credit assignment. Combining proximal entropy with a learned value model could offer complementary benefits. More broadly, the local reference frame could be applied beyond advantage weighting, for instance, to guide exploration strategies or to design entropy-aware curricula that adapt at the token level rather than the prompt level.

## Acknowledgments and Disclosure of Funding

This work received no external funding.

Competing interests: The authors declare no competing interests.

## References

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. MiniMax-M1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025. Introduces CISPO.

Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, and Nanyun Peng. ARES: Multimodal adaptive reasoning via difficulty-aware tokenlevel entropy shaping. In The Fourteenth International Conference on Learning Representations (ICLR), 2026.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective on reinforcement learning for LLMs. arXiv preprint arXiv:2506.14758, 2025a.

Jie Cheng, Gang Xiong, Ruixi Qiao, Lijun Li, Chao Guo, Junle Wang, Yisheng Lv, and Fei-Yue Wang. Stop summation: Min-form credit assignment is all process reward model needs for reasoning. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), 2025b.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Xu Guo. Measuring reasoning utility in LLMs via conditional entropy reduction. arXiv preprint arXiv:2508.20395, 2025.

Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, and Shuang Qiu. Segment policy optimization: Effective segment-level credit assignment in RL for large language models. arXiv preprint arXiv:2505.23564, 2025b.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. DeepMath-103K: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Kexin Huang, Haoming Meng, Junkang Wu, Jinda Lu, Chiyu Ma, Ziqian Chen, Xue Wang, Bolin Ding, Jiancan Wu, Xiang Wang, Xiangnan He, Guoyin Wang, and Jingren Zhou. On the direction of RLVR updates for LLM reasoning: Identification and exploitation. arXiv preprint arXiv:2603.22117, 2026.

Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, and Linfeng Zhang. Credit where it is due: Cross-modality connectivity drives precise reinforcement learning for MLLM reasoning. arXiv preprint arXiv:2602.11455, 2026.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. VinePPO: Refining credit assignment in RL training of LLMs. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tülu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, Bo Zheng, and Junchi Yan. Attention illuminates LLM reasoning: The preplan-and-anchor rhythm enables fine-grained policy optimization. arXiv preprint arXiv:2510.13554, 2025.

Yugu Li, Zehong Cao, Jianglin Qiao, and Siyi Hu. SSVPO: Effective step-level credit assignment for RL training of language models. In ICLR 2026 Conference (under review), 2026. OpenReview id g33DGvnHYd.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, and Jingren Zhou. FIPO: Eliciting deep reasoning with future-KL influenced policy optimization. arXiv preprint arXiv:2603.19835, 2026.

Mathematical Association of America. AMC 2023: American mathematics competitions. https: //artofproblemsolving.com/wiki/index.php/AMC\_Problems\_and\_Solutions, 2023.

Mathematical Association of America. AIME 2024: American invitational mathematics examination. https://artofproblemsolving.com/wiki/index.php/AIME\_Problems\_and\_ Solutions, 2024.

Mathematical Association of America. AIME 2025: American invitational mathematics examination. https://artofproblemsolving.com/wiki/index.php/AIME\_Problems\_and\_ Solutions, 2025.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in RLVR fine-tuning of LLMs. In International Conference on Learning Representations, 2026.

Shuaiyi Nie, Siyu Ding, Wenyuan Zhang, Linhao Yu, Tianmeng Yang, Yao Chen, Weichong Yin, Yu Sun, Hua Wu, and Tingwen Liu. ATTNPO: Attention-guided process supervision for efficient reasoning. arXiv preprint arXiv:2602.09953, 2026.

Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, and Sarath Chandar. GRPO-λ: Credit assignment improves LLM reasoning. arXiv preprint arXiv:2510.00194, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. arXiv preprint arXiv:2409.19256, 2024.

Hongze Tan, Zihan Wang, Jianfei Pan, Jinghao Lin, Hao Wang, Yifan Wu, Tao Chen, Zhihang Zheng, Zhihao Tang, and Haihua Yang. GTPO and GRPO-S: Token and sequence-level reward shaping with policy entropy. arXiv preprint arXiv:2508.04349, 2025.

Hieu Tran, Zonghai Yao, and Hong Yu. Exploiting tree structure for credit assignment in RL training of LLMs. arXiv preprint arXiv:2509.18314, 2025.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. arXiv preprint arXiv:2506.01939, 2025a.

Tao Wang, Suhang Zheng, and Xiaoxiao Xu. RTMC: Step-level credit assignment via rollout trees. arXiv preprint arXiv:2604.11037, 2026.

Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, et al. ROLL: Reinforcement learning optimization for large-scale learning – an efficient and user-friendly scaling library. arXiv preprint arXiv:2506.06122, 2025b.

Can Xie, Ruotong Pan, Xiangyu Wu, Yunfei Zhang, Jiayi Fu, Tingting Gao, and Guorui Zhou. Unlocking exploration in RLVR: Uncertainty-aware advantage shaping for deeper reasoning. arXiv preprint arXiv:2510.10649, 2025a.

Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, and Xiao Zhang. CAPO: Credit assignment policy optimization. arXiv preprint arXiv:2508.02298, 2025b.

Zhongwen Xu and Zihan Ding. Single-stream policy optimization. arXiv preprint arXiv:2509.13232, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Song Yu, Li Li, Wenwen Zhao, and Zhisheng Yang. ERPO: Token-level entropy-regulated policy optimization for large reasoning models. arXiv preprint arXiv:2603.28204, 2026.

Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. EDGE-GRPO: Entropy-driven GRPO with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025.

Zheng Zhang, Ao Lu, Yuanhao Zeng, Ziwei Shan, Jinjin Guo, Lufei Li, Yexin Li, and Kan Ren. Grad2Reward: From sparse judgment to dense rewards for improving open-ended LLM reasoning. arXiv preprint arXiv:2602.01791, 2026.

Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, and Furu Wei. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Jiaru Zou, Ling Yang, Jingwen Gu, Jiahao Qiu, Ke Shen, Jingrui He, and Mengdi Wang. ReasonFlux-PRM: Trajectory-aware PRMs for long chain-of-thought reasoning in LLMs. arXiv preprint arXiv:2506.18896, 2025.

## Appendix

## A Broader Impacts

PEPO is a methodological contribution to value-model-free RLVR that improves token-level credit assignment without additional rollouts, value models, or process reward models. As a general-purpose post-training technique, its societal impact is largely inherited from the broader trajectory of LLM reasoning research rather than tied to a specific deployment.

On the positive side, PEPO reduces the compute required to obtain a given level of reasoning performance: it matches or exceeds batch-level entropy methods at the same training budget (Table 9) and transfers to single-stream RL, where it removes the need for $G$ parallel rollouts per prompt. Lower training cost lowers the barrier to entry for academic and resource-constrained groups working on reasoning models, and reduces the energy footprint of RL post-training.

The negative considerations are those common to any technique that improves LLM reasoning. Stronger reasoning capabilities accelerate both beneficial applications (scientific assistance, education, code generation) and potentially harmful ones (more persuasive misinformation, automated vulnerability discovery, misuse in high-stakes decisions). PEPO does not introduce a new capability axis or unlock qualitatively new behaviors; it makes existing RLVR pipelines more sample- and credit-efficient. We do not release any new pretrained models or scraped datasets, and all experiments build on publicly available base models and mathematical reasoning datasets that have been released by their original authors with their own usage terms. We see no direct path from this work to specific harmful applications beyond those already discussed in the broader literature on LLM post-training.

## B Proof of Proposition 1

We restate the proposition for convenience.

Proposition 2. The proximal entropy $\boldsymbol { e } _ { i , t } \left( E q . \theta \right)$ satisfies:

(i) Ifall entropies in a sequence are shifted by a sequence-dependent constant $\delta _ { i } ,$ , then $e _ { i , t }$ is unchanged.

(ii) If entropies in a sequence follow a smooth monotonic trend, then $e _ { i , t }$ is approximately invariant to the trendfor sufficiently small W.

## B.1 Proof of (i): Invariance to Constant Shift

Let $\tilde { H } _ { i , t } = H _ { i , t } + \delta _ { i }$ for all t and some sequence-dependent constant $\delta _ { i } \in \mathbb { R }$ . The corresponding proximal entropy is

$$
\begin{array} { r l } & { \tilde { e } _ { i , t } = \frac { \exp \left( \tilde { H } _ { i , t } \right) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \left( \tilde { H } _ { i , k } \right) } = \frac { \exp \left( H _ { i , t } + \delta _ { i } \right) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \left( H _ { i , k } + \delta _ { i } \right) } } \\ & { ~ = \frac { \exp \left( \delta _ { i } \right) \exp \left( H _ { i , t } \right) } { \exp \left( \delta _ { i } \right) \sum _ { k \in \mathcal { W } ( t ) } \exp \left( H _ { i , k } \right) } = \frac { \exp \left( H _ { i , t } \right) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \left( H _ { i , k } \right) } = e _ { i , t } . } \end{array}\tag{9}
$$

Since all tokens within a window $\mathcal { W } ( t )$ belong to the same rollout $i ,$ they share the same $\delta _ { i } ,$ which factors out and cancels. This holds for any sequence-dependent shift, so proximal entropy is invariant to differences in overall entropy level across sequences.

## B.2 Proof of (ii): Approximate Invariance to Positional Trends

We decompose each token’s entropy as

$$
H _ { i , t } = f ( t ) + r _ { i , t } ,\tag{10}
$$

where $f \colon [ T _ { i } ]  \mathbb { R }$ is a smooth, monotonic, deterministic positional trend shared across rollouts and ${ \boldsymbol { r } } _ { i , t }$ is the residual that captures token-specific variation. Our goal is to show that $e _ { i , t }$ is approximately independent of f under mild regularity conditions.

Setup. Fix a token position t and consider the window $\mathcal { W } ( t ) = \{ k : | k - t | \leq W / 2 \}$ . Since $f$ is monotonic, $f ^ { \prime } ( t )$ does not change sign within the window, so the linear term dominates the local variation of $f$ and a first-order Taylor expansion of $f$ around t gives

$$
f ( k ) = f ( t ) + f ^ { \prime } ( t ) ( k - t ) + \mathcal { O } \bigg ( \frac { W ^ { 2 } } { T _ { i } ^ { 2 } } \bigg ) ,\tag{11}
$$

where we have used $| k - t | \leq W / 2$ and the fact that $f$ varies over the full sequence length $T _ { i } ,$ so sup $| f ^ { \prime \prime } | = \mathcal { O } ( 1 / T _ { i } ^ { 2 } )$ . Monotonicity ensures that the residual term captures only curvature rather than direction-changing oscillations, so the approximation is accurate when $W \ll T _ { i } , \mathrm { i . e . }$ ., the positional trend is approximately linear within each window.

Substitution. Substituting Eq. (11) into the definition of $e _ { i , t }$ and dropping the higher-order terms:

$$
\begin{array} { l } { \displaystyle e _ { i , t } = \frac { \exp ( H _ { i , t } ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp ( H _ { i , k } ) } = \frac { \exp \bigl ( f ( t ) + r _ { i , t } \bigr ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \bigl ( f ( k ) + r _ { i , k } \bigr ) } } \\ { \displaystyle \approx \frac { \exp \bigl ( f ( t ) + r _ { i , t } \bigr ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \bigl ( f ( t ) + f ^ { \prime } ( t ) ( k - t ) + r _ { i , k } \bigr ) } } \\ { \displaystyle = \frac { \exp \bigl ( r _ { i , t } \bigr ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp \bigl ( f ^ { \prime } ( t ) ( k - t ) + r _ { i , k } \bigr ) } , } \end{array}\tag{12}
$$

where the last step uses the translation-invariance property established in (i) to cancel $f ( t )$

Small-gradient regime. Eq. (12) still contains the linear term $f ^ { \prime } ( t ) ( k - t )$ . We show that this term is negligible when the trend slope is small relative to the inverse window size. For each $k \in \mathcal { W } ( t )$ we have $| f ^ { \prime } ( t ) ( k - t ) | \le | f ^ { \prime } ( t ) | \cdot W / 2$ . When $| f ^ { \prime } ( t ) | \cdot W / 2 \ll 1$ , a first-order expansion of the exponential gives

$$
\mathrm { e x p } \big ( f ^ { \prime } ( t ) ( k - t ) + r _ { i , k } \big ) \approx \mathrm { e x p } ( r _ { i , k } ) \big ( 1 + f ^ { \prime } ( t ) ( k - t ) \big ) .\tag{13}
$$

Substituting into the denominator of Eq. (12):

$$
\sum _ { k \in \mathcal { W } ( t ) } \exp \bigl ( f ^ { \prime } ( t ) ( k - t ) + r _ { i , k } \bigr ) \approx \sum _ { k \in \mathcal { W } ( t ) } \exp ( r _ { i , k } ) + f ^ { \prime } ( t ) \sum _ { k \in \mathcal { W } ( t ) } ( k - t ) \exp ( r _ { i , k } ) .\tag{14}
$$

The second term is a weighted sum of offsets $( k - t )$ , which range symmetrically from $- W / 2$ to $+ W / 2$ . Even without any assumption on the residuals, this term is $\mathsf { \bar { O } } ( \bar { f } ^ { \prime } ( t ) \cdot W )$ times the first term, and therefore negligible when $| { \bf \bar { \boldsymbol { f } } } ^ { \prime } ( t ) | \cdot { \boldsymbol { W } } / 2 \ll 1$ . The same expansion applies to the numerator at $k = t ,$ , where $( k - t ) = 0$ and the linear correction vanishes exactly. We therefore obtain

$$
e _ { i , t } \approx \frac { \exp ( r _ { i , t } ) } { \sum _ { k \in \mathcal { W } ( t ) } \exp ( r _ { i , k } ) } ,\tag{15}
$$

which depends only on the residuals and is independent of the trend $f .$

Summary. The conditions required are that the positional trend $f$ is smooth and monotonic, and that $| f ^ { \prime } ( t ) | \cdot W / 2 \ll 1 , \mathrm { i . e . }$ , the trend is slowly varying within each window. Under these conditions, the proximal entropy $e _ { i , t }$ is approximately invariant to $f$ up to corrections of order $\mathcal { O } ( f ^ { \prime } ( t ) \cdot W )$ and the Taylor remainder from Eq. (11). Combined with property (i), this implies that $e _ { i , t }$ isolates the token-specific signal from both prompt difficulty and positional entropy trends. □

Remark. The condition $| f ^ { \prime } ( t ) | \cdot W / 2 \ll 1$ is mild for typical reasoning rollouts. From Fig. 2, entropy decays from roughly 0.314 to 0.185 over the full sequence, giving $| f ^ { \prime } | \approx 0 . 1 3 / T _ { i }$ . For $T _ { i } = 2 0 0 0$ and $W = 1 0 1$ , we have $| f ^ { \prime } | \cdot W / 2 \approx 0 . 0 0 4$ , well within the small-gradient regime. The window size ablation in Fig 4 provides further empirical support, where performance is stable across $W \in \{ 3 1 , 5 1 , 1 0 1 \}$ }, suggesting that the approximation holds well for the window sizes used.

## C Implementation Details

## C.1 Hyperparameters

We follow the default configurations of the ROLL framework [Wang et al., 2025b] for our main experiments and the VeRL framework [Sheng et al., 2024] for single-stream experiments. Table 7 reports the hyperparameters used for the main and drop-in experiments (Tables 2, 3, 6), and Table 8 reports those used for the single-stream RL experiments (Table 4).

Table 7: Hyperparameters for main experiments on ROLL framework. These settings are used for GRPO, 80/20, Entropy Adv., and PEPO across all three models.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Framework Training data</td><td>ROLL DeepMath-103K</td></tr><tr><td>Rollouts per prompt (G) Learning rate</td><td>8  $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Lower clipping threshold (εl) Upper clipping threshold (εh)</td><td>0.2</td></tr><tr><td></td><td>0.28</td></tr><tr><td>Training steps</td><td></td></tr><tr><td></td><td>500</td></tr><tr><td>Sampling temperature</td><td>0.6</td></tr><tr><td>Top-p</td><td>0.6</td></tr><tr><td>Maximum response length</td><td></td></tr><tr><td></td><td>4096 tokens</td></tr><tr><td>Window size W (PEPO)</td><td>101</td></tr></table>

Table 8: Hyperparameters for single-stream experiments on VeRL framework. These settings are used for SPO, S-80/20, S-Entropy Adv., and PESPO.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Framework Training data Rollouts per prompt Learning rate Optimizer Lower clipping threshold (εl) Upper clipping threshold  $( \varepsilon _ { h } )$  Training steps Sampling temperature Top-p Maximum response length Window size W (PESPO)</td><td>VeRL DAPO-Math-17K 1 (single-stream)  $1 \times 1 \bar { 0 } ^ { - 6 }$  AdamW 0.2 0.28 500 0.6 0.6 4096 tokens 101</td></tr></table>

## C.2 Window Edge Handling

The window $\mathcal { W } ( t ) = \{ k : | k - t | \leq W / 2 \}$ extends beyond the sequence boundaries when $t < W / 2$ or $t > T _ { i } - W / 2$ . We use reflect padding to fill the missing positions, mirroring entropy values across the boundary. We chose reflect padding over zero padding because the latter would inject artificially low entropy values into the softmax denominator at boundary windows, distorting the relative weighting. Reflect padding instead uses values drawn from the same sequence, preserving the local entropy scale. This is also intuitively aligned with the proximal entropy formulation itself, which measures each token’s entropy relative to its neighbors; reflecting the sequence at the boundary effectively treats nearby tokens on the opposite side of t as the missing neighbors, consistent with the local-context interpretation of $e _ { i , t }$

Boundary effects are confined to roughly $W / T _ { i } \approx 5 \%$ of tokens for our default $W = 1 0 1$ and typical sequence lengths, and we found this choice to work well empirically.

## C.3 Software and Compute

All experiments are run on 2 NVIDIA H200 GPUs. We use the ROLL framework for the main experiments and the VeRL framework for single-stream experiments, with vLLM for rollout generation.

## D Runtime Analysis

To verify that proximal entropy weighting adds negligible overhead, we measure the total wall-clock training time of PEPO and the baselines under identical hardware (2×H200), framework (ROLL), and training configuration (500 steps, G = 8, max length 4096). Table 9 reports the results for the Qwen3-1.7B and Qwen3-4B.

Table 9: Total wall-clock training time on 2×H200 GPUs. All methods are run with identical training configurations for 500 steps. PEPO adds negligible overhead over GRPO across both models.
<table><tr><td>Method</td><td>Qwen3-1.7B</td><td>Qwen3-4B</td></tr><tr><td>GRPO</td><td>1d 7h 32m</td><td>1d 19h 25m</td></tr><tr><td>80/20</td><td>1d 7h 28m</td><td>1d 20h 14m</td></tr><tr><td>Entropy Adv.</td><td>1d 8h 28m</td><td>1d 21h 3m</td></tr><tr><td>PEPO (Ours)</td><td>1d 7h 54m</td><td>1d 19h 45m</td></tr></table>

As shown in Table 9, all entropy-based methods (80/20, Entropy Adv., PEPO) run within minutes of GRPO across both models, well within the margin of variation expected from differences in random seeds and hardware scheduling.

## E Additional Bias Analyses of Proximal Entropy

In Section 2.2 of the main paper, we showed that proximal entropy is empirically invariant to prompt difficulty (Table 1) and positional trends (Figure 2). In this section, we provide additional analyses showing that this invariance is robust across window sizes and is not an artifact of the softmax operation.

Invariance across window sizes and aggregation choices. Proposition 1(ii) establishes that proximal entropy is approximately invariant to positional trends for sufficiently small W. To empirically verify this, we plot the deviation of two window-based statistics from the sequence-level mean across generation percentiles for $W \in \{ 3 1 , 5 1 , 1 0 1 \}$ : (i) proximal entropy as defined in Eq. 6 and (ii) a simple windowed average of token entropies without the softmax operation. As shown in Figure 4, both statistics yield flat curves across all three window sizes, in contrast to global entropy (gray dashed) which decays substantially over the course of generation.

This result has two implications. First, the positional invariance of proximal entropy is robust to the choice of W, holding across the range of window sizes considered in our experiments. Second, the mitigation of positional bias is a property of the local windowing itself rather than the softmax operation: even a simple windowed average suffices to remove the positional trend. The softmax in PEPO serves a different purpose, which is to convert a window of entropies into a continuous, normalized per-token weight that emphasizes tokens with relatively higher entropy than their local neighbors. The two design choices (windowing and softmax-based weighting) are therefore separable, and the bias-removal property attaches to the former.

![](images/14821cdc57b817d517fb590985d9172ebb725698196d249262b4f5b5b5803859.jpg)  
Figure 4: Deviation of windowed entropy statistics from their sequence-level mean across generation percentiles, for window sizes $W \in \{ 3 \mathbf { \bar { 1 } } , 5 1 , 1 0 1 \}$ . We compare two aggregation choices: proximal entropy (solid purple), defined as a softmax over the window, and a simple windowed average (dashed orange). Both yield flat curves across all three window sizes, in contrast to global entropy (gray dashed) which decays substantially. This shows that the positional invariance is robust to the window size and is a property of the windowing itself rather than of the softmax operation. Computed over 992 rollouts on the validation set with Llama-3.2-3B-Instruct.

## F Additional Evaluation and Ablations

## F.1 Held-Out Code Generation: LiveCodeBench

To test transfer beyond mathematical reasoning, we evaluate the Qwen3-4B GRPO, 80/20, and PEPO checkpoints on the held-out release\_v6/code\_generation\_lite split of LiveCodeBench, which contains 1,055 problems. LiveCodeBench was not included in RL training; the ROLL training mixture did include separate verifiable coding problems from KodCode. The three checkpoints share the same base model, training data, and training budget. We use the official LiveCodeBench evaluation implementation (https://github.com/LiveCodeBench/LiveCodeBench) with temperature 0.2, top-p 0.95, a maximum of 4,096 generated tokens, thinking mode enabled, and bfloat16 inference. Table 10 reports overall pass@1, and Table 11 reports the difficulty and platform breakdowns.

PEPO reaches 46.35% pass@1, a 14.60 percentage-point improvement over GRPO and 5.31 points over 80/20. Its gains over GRPO are positive across all three difficulty groups and all listed platforms. The Codeforces subset contains only nine problems, so that row should be interpreted cautiously.

Table 10: Held-out LiveCodeBench release $. \mathtt { v } 6 / $ code\_generation\_lite pass@1 point estimates for Qwen3-4B.
<table><tr><td>Method</td><td>Correct</td><td>Pass@1</td><td>Gain vs. GRPO</td></tr><tr><td>GRPO</td><td>335 / 1,055</td><td>31.75%</td><td></td></tr><tr><td>80/20</td><td>433 / 1,055</td><td>41.04%</td><td>+9.29 pp</td></tr><tr><td>PEPO</td><td>489 / 1,055</td><td>46.35%</td><td>+14.60 pp</td></tr></table>

Table 11: LiveCodeBench pass@1 breakdown by difficulty and problem platform. The Codeforces subset contains only 9 problems.
<table><tr><td>Subset</td><td>Problems</td><td>GRPO</td><td>80/20</td><td>PEPO</td><td>PEPO – GRPO</td></tr><tr><td>By difficulty</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Easy</td><td>322</td><td>76.09%</td><td>84.78%</td><td>90.99%</td><td>+14.91 pp</td></tr><tr><td>Medium</td><td>383</td><td>22.72%</td><td>37.60%</td><td>43.86%</td><td>+21.15 pp</td></tr><tr><td>Hard</td><td>350</td><td>0.86%</td><td>4.57%</td><td>8.00%</td><td>+7.14 pp</td></tr><tr><td>By platform</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AtCoder</td><td>602</td><td>31.23%</td><td>38.87%</td><td>42.86%</td><td>+11.63 pp</td></tr><tr><td>LeetCode</td><td>444</td><td>32.66%</td><td>43.92%</td><td>50.68%</td><td>+18.02 pp</td></tr><tr><td>Codeforces</td><td>9</td><td>22.22%</td><td>44.44%</td><td>66.67%</td><td>+44.44 pp</td></tr></table>

## F.2 Weighting Design Ablation

We decompose the Qwen3-4B mean accuracy gain by changing one component at a time: global to proximal entropy for binary selection, binary selection to continuous weighting over all trajectories, and then positive-only application. Table 12 reports the aggregate results.

Table 12: Sequential weighting ablation on Qwen3-4B. Mean accuracy averages AIME 2024, AIME 2025, AMC, and MATH500; the first gain averages the benchmark-level differences in Table 3.
<table><tr><td>Configuration</td><td>Mean</td><td>Gain (pp)</td></tr><tr><td>Global entropy, binary selection (80/20)</td><td>55.74</td><td></td></tr><tr><td>Proximal entropy, binary selection</td><td>56.87</td><td>+1.14</td></tr><tr><td>Proximal entropy, continuous weighting on all rollouts</td><td>57.53</td><td>+0.66</td></tr><tr><td>Proximal entropy, continuous weighting on positive rollouts</td><td>58.01</td><td>+0.48</td></tr></table>

Localization adds 1.14 points, moving from binary to continuous weighting over all trajectories adds 0.66 points, and restricting continuous weighting to positive-reward rollouts adds a further 0.48 points. The first gain is the average of the four benchmark-level changes shown in Table 3; the all-rollout continuous-weighting result was reported without run-level uncertainty, so its comparison with positive-only weighting is descriptive. Even with all-rollout weighting, the mean remains 7.38 points above GRPO (57.53 versus 50.15).

## F.3 Temperature Sensitivity

We vary the proximal-entropy softmax temperature τ on Llama-3.2-3B-Instruct while keeping the remaining settings fixed. Table 13 gives the reported point estimates.

Table 13: Temperature sensitivity for Llama-3.2-3B-Instruct. Results are the reported point estimates; no seed-level uncertainty was provided for this sweep. Best result per column is bolded.
<table><tr><td>T</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Mean</td></tr><tr><td>0.5</td><td>8.75</td><td>1.25</td><td>23.72</td><td>49.40</td><td>20.78</td></tr><tr><td>1.0</td><td>9.10</td><td>1.46</td><td>24.12</td><td>49.33</td><td>21.00</td></tr><tr><td>2.0</td><td>8.54</td><td>1.46</td><td>23.72</td><td>49.00</td><td>20.68</td></tr></table>

Mean accuracy spans only 20.68–21.00 across the sweep, with the default τ = 1 giving the highest reported mean.

## Per-Run Experimental Results

## Main Results (Table 2)

Table 14: Per-run accuracy for the main results in Table 2. Each row reports the avg@1 accuracy on MATH500 and avg@16 accuracy on AMC, AIME 2024, and AIME 2025 for a single run. Mean rows match the values reported in Table 2.
<table><tr><td>Model</td><td>Method</td><td>Run</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td></tr><tr><td rowspan="18">Qwen3-1.7B</td><td>GRPO</td><td>1</td><td>16.42</td><td>18.26</td><td>51.22</td><td>79.20</td></tr><tr><td></td><td>2 3</td><td>17.07 16.24</td><td>19.36 18.33</td><td>50.38 49.25</td><td>81.20 80.40</td></tr><tr><td rowspan="6"></td><td>Mean</td><td>16.58</td><td>18.65</td><td>50.28</td><td>80.27</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>22.25</td><td>20.53</td><td>51.61</td><td>79.80</td></tr><tr><td>2</td><td>19.37</td><td>17.50</td><td>53.69</td><td>82.60</td></tr><tr><td>Entropy Adv. 3</td><td>22.71</td><td>20.41</td><td>52.26</td><td>81.80</td></tr><tr><td>Mean</td><td>21.44</td><td>19.48</td><td>52.52</td><td>81.40</td></tr><tr><td rowspan="6">80/20</td><td>1</td><td>22.29</td><td></td><td></td><td></td></tr><tr><td>23</td><td>21.25</td><td>21.46</td><td>51.95</td><td>79.20</td></tr><tr><td></td><td>19.37</td><td>21.04</td><td>52.63</td><td>80.80</td></tr><tr><td></td><td>20.97</td><td>20.00 20.83</td><td>51.80</td><td>82.20</td></tr><tr><td>Mean</td><td></td><td></td><td>52.13</td><td>80.73</td></tr><tr><td>1</td><td>20.84</td><td>22.71</td><td>54.54</td><td>82.40</td></tr><tr><td rowspan="10">Qwen3-4B</td><td>PEPO</td><td>2</td><td>22.91</td><td>21.67</td><td>56.54 54.69</td><td>83.00 82.60</td></tr><tr><td>Mean</td><td>3</td><td>21.46 21.74</td><td>22.91 22.43</td><td>55.26</td><td>82.67</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>23</td><td>1</td><td>32.68</td><td>22.92</td><td>60.25</td><td>83.00 85.40</td></tr><tr><td rowspan="5">GRPO</td><td></td><td>33.33 32.25</td><td>24.38 23.31</td><td>59.18 60.45</td><td>84.60</td></tr><tr><td>Mean</td><td>32.75</td><td>23.54</td><td>59.96</td><td>84.33</td></tr><tr><td></td><td>36.41</td><td>24.88</td><td>68.25</td><td></td></tr><tr><td>1 2</td><td>37.08</td><td>26.46</td><td>67.18</td><td>89.60 90.00</td></tr><tr><td>Entropy Adv. 3</td><td>34.15</td><td>27.29</td><td>68.06</td><td>89.60</td></tr><tr><td rowspan="5">80/20</td><td>Mean</td><td>35.88</td><td>26.21</td><td>67.83</td><td>89.73</td></tr><tr><td>1</td><td>32.50</td><td>30.83</td><td>67.77</td><td>88.40</td></tr><tr><td>23</td><td>36.87</td><td>30.62</td><td>68.46</td><td>90.40</td></tr><tr><td></td><td>36.67</td><td>30.42</td><td>65.53</td><td>90.40</td></tr><tr><td>Mean</td><td>35.35</td><td>30.62</td><td>67.25</td><td>89.73</td></tr><tr><td rowspan="5">PEPO</td><td>1</td><td>38.13</td><td>31.88</td><td></td><td>91.80</td></tr><tr><td>2</td><td>40.63</td><td>31.88</td><td>70.17 69.58</td><td>89.40</td></tr><tr><td>3</td><td>38.96</td><td>31.66</td><td>68.99</td><td>93.00</td></tr><tr><td>Mean</td><td>39.24</td><td>31.81</td><td>69.58</td><td>91.40</td></tr><tr><td>1</td><td></td><td>0.83</td><td></td><td></td></tr><tr><td rowspan="10">Llama-3.2-3B-Instruct</td><td>GRPO</td><td>6.46</td><td></td><td>21.69</td><td>48.00</td><td>48.80</td></tr><tr><td>3</td><td>2</td><td>7.08</td><td>1.46</td><td>20.78 21.83</td><td>46.20</td></tr><tr><td>Mean</td><td>8.12</td><td></td><td>0.42 0.90</td><td>21.43</td><td>47.67</td></tr><tr><td></td><td>7.22</td><td></td><td></td><td>24.24</td><td>47.60</td></tr><tr><td>Entropy Adv.</td><td>1 2</td><td>8.12 9.17</td><td>0.42 0.42</td><td>23.86</td><td>48.20</td></tr><tr><td></td><td>3</td><td>7.50</td><td>0.83</td><td>22.81</td><td>47.00</td></tr><tr><td></td><td>Mean</td><td>8.26</td><td>0.56</td><td>23.64</td><td>47.60</td></tr><tr><td></td><td>1</td><td>7.92</td><td>0.42</td><td>23.26</td><td>47.60</td></tr><tr><td>80/20</td><td>2</td><td>9.12</td><td>0.62</td><td>21.53</td><td>48.20</td></tr><tr><td>3</td><td></td><td>8.75</td><td>0.42</td><td>22.91</td><td>47.20</td></tr><tr><td rowspan="5">PEPO</td><td>Mean</td><td>8.60</td><td>0.49</td><td>22.57</td><td>47.67</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>8.54</td><td></td><td>1.67</td><td>23.71</td><td>49.40</td></tr><tr><td>2</td><td>9.58</td><td></td><td>0.83</td><td>24.40</td><td>49.40</td></tr><tr><td>3</td><td>9.17</td><td></td><td></td><td>24.24</td><td>49.20</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td>1.87 1.46</td><td></td><td></td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td>49.33</td></tr><tr><td></td><td></td><td></td><td></td><td>24.12</td><td></td></tr><tr><td></td><td>9.10</td><td></td><td></td><td></td><td></td></tr></table>

## Proximal Entropy as a General Formulation (Table 3)

Table 15: Per-run accuracy for the drop-in proximal entropy results in Table 3. Mean rows match the values reported in Table 3.
<table><tr><td>Model</td><td>Method</td><td>Run</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td></tr><tr><td rowspan="4">Qwen3-1.7B</td><td rowspan="4">80/20 + Prox.</td><td>1</td><td>22.71</td><td>20.83</td><td>53.16</td><td>82.60</td></tr><tr><td>2</td><td>21.04</td><td>20.42</td><td>52.71</td><td>82.20</td></tr><tr><td>3</td><td>22.29</td><td>20.83</td><td>54.22</td><td>82.00</td></tr><tr><td>Mean</td><td>22.01</td><td>20.69</td><td>53.36</td><td>82.27</td></tr><tr><td rowspan="4">Qwen3-4B</td><td rowspan="4">80/20 + Prox.</td><td>1</td><td>38.75</td><td>31.04</td><td>69.65</td><td>90.60</td></tr><tr><td>2</td><td>33.54</td><td>31.88</td><td>70.63</td><td>90.40</td></tr><tr><td>3</td><td>36.25</td><td>31.88</td><td>67.85</td><td>90.00</td></tr><tr><td>Mean</td><td>36.18</td><td>31.60</td><td>69.38</td><td>90.33</td></tr></table>

## Single-Stream RL (Table 4)

Table 16: Per-run accuracy for the single-stream RL results in Table 4 on Qwen3-4B. Mean rows match the values reported in Table 4.
<table><tr><td>Method</td><td>Run</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td></tr><tr><td rowspan="4">SPO</td><td>1</td><td>37.92</td><td>31.25</td><td>70.71</td><td>90.00</td></tr><tr><td>2</td><td>35.00</td><td>31.04</td><td>72.14</td><td>90.80</td></tr><tr><td>3</td><td>34.17</td><td>31.88</td><td>69.58</td><td>91.00</td></tr><tr><td>Mean</td><td>35.70</td><td>31.39</td><td>70.81</td><td>90.60</td></tr><tr><td rowspan="4">S-80/20</td><td>1</td><td>36.46</td><td>26.67</td><td>68.98</td><td>90.60</td></tr><tr><td>2</td><td>35.21</td><td>28.75</td><td>70.56</td><td>91.40</td></tr><tr><td>3</td><td>34.17</td><td>27.50</td><td>70.63</td><td>91.00</td></tr><tr><td>Mean</td><td>35.28</td><td>27.64</td><td>70.06</td><td>91.00</td></tr><tr><td rowspan="4">S-Entropy Adv.</td><td>1</td><td>38.75</td><td>22.29</td><td>68.30</td><td>89.80</td></tr><tr><td>2</td><td>40.21</td><td>22.08</td><td>68.90</td><td>89.40</td></tr><tr><td>3</td><td>37.71</td><td>21.04</td><td>68.07</td><td>89.20</td></tr><tr><td>Mean</td><td>38.89</td><td>21.80</td><td>68.42</td><td>89.47</td></tr><tr><td rowspan="4">PESPO</td><td>1</td><td>41.25</td><td>31.04</td><td></td><td></td></tr><tr><td>2</td><td>42.29</td><td>32.50</td><td>71.76 72.74</td><td>90.60 90.40</td></tr><tr><td>3</td><td>41.67</td><td>31.25</td><td>70.63</td><td>91.00</td></tr><tr><td>Mean</td><td>41.74</td><td>31.60</td><td>71.71</td><td>90.67</td></tr></table>

## Window Size Ablation (Table 6)

Table 17: Per-run accuracy for the window size ablation in Table 6 on Qwen3-4B. Mean rows match the values reported in Table 6.
<table><tr><td>Window Size</td><td>Run</td><td>AIME2024</td><td>AIME2025</td><td>AMC</td><td>MATH500</td></tr><tr><td rowspan="4">W = 51</td><td>1</td><td>34.59</td><td>30.20</td><td>68.36</td><td>90.20</td></tr><tr><td>2</td><td>37.50</td><td>29.59</td><td>67.82</td><td>91.00</td></tr><tr><td>3</td><td>36.67</td><td>32.08</td><td>68.99</td><td>90.80</td></tr><tr><td>Mean</td><td>36.25</td><td>30.62</td><td>68.39</td><td>90.67</td></tr><tr><td rowspan="4">W = 151</td><td>1</td><td>37.08</td><td>27.08</td><td>69.87</td><td>92.00</td></tr><tr><td>2</td><td>36.67</td><td>29.59</td><td>68.89</td><td>92.00</td></tr><tr><td>3</td><td>36.67</td><td>28.13</td><td>69.29</td><td>92.80</td></tr><tr><td>Mean</td><td>36.81</td><td>28.27</td><td>69.35</td><td>92.27</td></tr></table>