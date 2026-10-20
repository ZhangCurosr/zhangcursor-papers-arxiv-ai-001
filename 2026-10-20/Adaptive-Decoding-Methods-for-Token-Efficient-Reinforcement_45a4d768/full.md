# Adaptive Decoding Methods for Token-Efficient Reinforcement Learning

Zhaonan Li<sup>∗†∗</sup> School of Computing and Augmented Intelligence Arizona State University zhaonan2@asu.edu

Amritansh Mishra<sup>†</sup> AI Foundations Capital One amritansh.mishra@capitalone.com

Mann Patel AI Foundations Capital One mann.patel2@capitalone.com

Ben Zhou School of Computing and Augmented Intelligence Arizona State University benzhou@asu.edu

Zelei Cheng<sup>†</sup> AI Foundations Capital One zelei.cheng@capitalone.com

Chia-Hsuan Lee AI Foundations Capital One chiahsuan.lee@capitalone.com

Sambit Sahu AI Foundations Capital One sambit.sahu@capitalone.com

William Campbell AI Foundations Capital One william.campbell@capitalone.com

## Abstract

Rollout decoding affects both which trajectories are used for reinforcement learning with verifiable rewards (RLVR) and how many completion tokens are spent collecting them. We study whether candidate-token support should remain fixed or widen during training in the context of rollout sampling efficiency. Our main schedule moves from $\mathsf { t o p } { \cdot } K = 1 0$ to $K = 2 0$ and then to full support as the attempted-token budget increases. In a 4B code-RLVR setting, the schedule has a higher interpolated 150M-token validation value than its matched fixed- $K = 1 0$ continuation in all three runs, with a mean gain of $1 . 8 0 { \pm } 0 . 4 2 \mathrm { p o i n t s }$ . The schedule also performs comparably to a separately trained fixed-K = 20 baseline, the strongest fixed-support setting in our sweep. Follow-up experiments suggest that late widening to full support is the most promising variant, although the evidence remains noisy, and mixed-reward-group diagnostics do not clearly explain the gain. An exploratory single-run test replay yields identical point estimates for the schedule and fixed K = 10. Overall, our results show that rollout support is an important control for sampling efficiency in RLVR and motivate further study of adaptive support schedules.

## 1 Introduction

RLVR trains reasoning models from sampled solutions whose outcomes can be checked automatically. In group-relative training, a prompt whose rollouts are all correct or all incorrect supplies no reward contrast, but every attempted response still consumes generation tokens. The rollout policy therefore affects both the optimization signal and the cost of obtaining it.

Candidate-token support is a direct but understudied rollout control. Full-support sampling admits the entire vocabulary, including unlikely continuations; top-K sampling concentrates generation on the model’s K most likely next tokens. A narrow support may avoid unproductive tail samples early, but it may also suppress alternatives that become useful as the policy changes. We call training-time rules that change this support adaptive rollout decoding; the evaluation decoder remains fixed. This creates a concrete question: should rollout support remain fixed, or widen with training progress? Intuitively, the useful support may change as the policy improves. With a limited rollout-token budget, a narrow support concentrates sampling on high-probability continuations and avoids spending tokens on unlikely tail choices. However, the same restriction can exclude lower-ranked alternatives that may contain useful solution paths. This suggests a simple hypothesis: narrower support may be useful for concentrating sampling early in training, while broader support may become more useful later as the policy improves.

We test this hypothesis under an attempted-completion-token budget. Here, token-efficient refers to validation performance as a function of generated completion tokens. At a common 36.84Mtoken development budget, an observed $K = 1 0$ value from one development run is 17.73%, while independently trained $\bar { K } = 2 0$ and full-support trajectories interpolate to 16.18% and 16.90%. In the separate 150M-token primary cohort, fixed K = 20 achieves a higher estimated mean than fixed $K = 1 0$ . These cross-cohort observations suggest that the useful rollout support may depend on training progress and motivate a simple widening schedule: K = 10 initially, K = 20 after a nominal 50M attempted completion tokens, and full support after a nominal 100M tokens. Temperature, nucleus cutoff, group size, optimizer, verifier, and evaluation decoder are held fixed.

The common-budget estimates favor widening over every matched fixed-K = 10 continuation, while fixed K = 20 remains the strongest static control. A follow-up decomposition makes late direct widening the most promising next design and shows that the exact three-stage effect varies by cohort. Mixed-reward-group diagnostics and an exploratory test replay help clarify the scope and limitations of the result. Accordingly, our contributions are (i) budget-indexed and entropy-gated adaptive decoding rules for rollout support, with the latter treated as exploratory; (ii) transparent shared-parent and static-control estimates under a common token accounting rule; and (iii) a decomposition that separates the stable empirical pattern from the next hypotheses.

## 2 Related Work

Verifiable rewards and group-relative optimization. Cobbe et al. showed that learned verifiers can rank sampled solutions and substantially improve performance on GSM8K [Cobbe et al., 2021]. GRPO estimates relative advantages from groups of responses without a separate critic [Shao et al., 2024]. DAPO develops group-relative RLVR at scale through asymmetric clipping, token-level losses, and dynamic filtering of all-correct or all-incorrect groups [Yu et al., 2025]. Our optimizer follows this group-relative setting; our focus is the behavior policy that generates each group.

Efficient and informative rollout collection. Recent work improves RLVR sampling by deciding where additional responses are most useful. Gradient-variance minimization assigns prompt-specific rollout counts under a compute budget [Yao et al., 2025], while AR3PO adapts response allocation and reuses collected responses [Zhang et al., 2025]. DAPO’s dynamic sampler similarly avoids groups with no reward contrast [Yu et al., 2025]. These approaches change prompt or group allocation. We instead hold the group size at eight and control which token candidates may enter every rollout, measuring the resulting learning curve against attempted completion tokens.

Truncation and adaptive decoding. Top-K and nucleus sampling truncate the low-probability tail rather than merely rescale it [Holtzman et al., 2020]. Truncation can also be viewed as estimating the plausible token support, with entropy-dependent thresholds adapting that support to context [Hewit et al., 2022]. Mirostat provides another feedback-controlled decoder by adjusting top-K from observed token surprise [Basu et al., 2021]. These methods target inference-time generation. We use support control as a training-time intervention: it changes the trajectories, rewards, and token cost seen by RLVR, while evaluation decoding remains fixed.

Exploration schedules and entropy. Multi-temperature RLVR varies temperature at token or rollout levels [Zhuang et al., 2025], while difficulty-aware exploration couples adaptive rollout budgets with temperature control [Liao et al., 2025]. High-entropy tokens have also been identified as sparse reasoning forks that carry disproportionate policy-gradient value [Wang et al., 2025]. Our budget schedule is related to curricula that change a learning condition with progress [Bengio et al., 2009], but uses actual attempted-token consumption as its clock. Temperature remains fixed: the schedule changes token admissibility, and the entropy gate decides where widening is applied. The next section connects these controls to group-relative learning in one training loop.

## 3 Method

## 3.1 Method overview

Our method adapts the candidate-token support used to collect RLVR rollouts. At collection batch j, the accumulated attempted-completion-token budget $B _ { j }$ selects the nominal support: $K = 1 0$ before 50M tokens, $K = 2 0$ from 50M to 100M, and full support thereafter. The optional entropygated variant applies widening selectively at each token, retaining $K = 1 0$ when entropy of the renormalized top-20 distribution is below 0.25 nats and otherwise using the nominal wider support. We sample eight completions per prompt from the resulting behavior policy and assign binary code-execution rewards.

Within each prompt group, the verified rewards produce group-relative advantages. The actor is then updated with the correction-weighted, dual-clipped PPO objective and reference-KL penalty. Every generated completion token, including tokens from incorrect and length-truncated responses, increments $B _ { j }$ and advances the support schedule. Thus adaptive collection, verification, optimization, and resource accounting share one token clock. The following subsections define each component and explain why rollout diversity determines whether a group supplies reward-driven gradient signal.

## 3.2 Group-relative learning and the token clock

Let $x \sim \mathcal { D }$ be a prompt, $y _ { i } = ( a _ { i , 1 } , \dotsc , a _ { i , L _ { i } } )$ one of $G = 8$ sampled completions, and $r _ { i } \in \{ 0 , 1 \}$ its code-execution reward. The group-relative advantage is

$$
\widehat { A } _ { i } = \frac { r _ { i } - \bar { r } } { s _ { r } + \epsilon _ { A } } , \qquad \bar { r } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } ,\tag{1}
$$

where $s _ { r }$ is the within-group sample standard deviation and $\epsilon _ { A } = 1 0 ^ { - 6 }$ . Thus an all-correct or all-incorrect group has zero reward-driven relative advantage.

For the binary outcome rate $q _ { x } ^ { e }$ induced by rollout rule e, conditionally independent samples give

$$
P _ { \mathrm { m i x e d } } ( x ; e ) = 1 - ( q _ { x } ^ { e } ) ^ { G } - ( 1 - q _ { x } ^ { e } ) ^ { G } .\tag{2}
$$

This identity motivates recording mixed-reward groups, but it makes no directional prediction: a sampler can create contrast by producing additional failures. We therefore treat the mixed-group rate as a diagnostic and validation learning as the outcome.

Because support can also affect response length, we index training by attempted completion tokens rather than optimizer steps. If $\mathcal { R } _ { \leq j }$ contains every response attempted through collection batch $j ,$ then

$$
B _ { j } = \sum _ { y \in \mathcal { R } _ { \leq j } } | y | .\tag{3}
$$

The count covers generated completion tokens, including incorrect and length-truncated outputs, and serves as our rollout-efficiency measure.

## 3.3 Reward Contrast in Group-Relative Learning

Group-relative optimization requires within-group reward contrast for reward-driven learning. Under output-only rewards, Mishra et al. [Mishra et al., 2026] show that the variance of an individual rollout’s centered reward is proportional to the reward variance under the rollout policy. Consequently, when all rollouts for a prompt receive the same reward, their group-relative advantages vanish and the prompt contributes no reward-driven policy-gradient signal. We summarize the connection to our standardized GRPO objective in Appendix A.

Candidate support changes the rollout distribution and can therefore change the probability of obtaining mixed-reward groups. At the same time, wider support can change response length and the support-dependent correction weights in our objective. We therefore view support adaptation as a hypothesis about the trade-off between obtaining reward contrast and the token cost of rollout collection, rather than assuming that wider support is always beneficial. Accordingly, we use mixed reward groups as a mechanism diagnostic and validation performance per attempted token as the primary outcome.

## 3.4 Support-controlled rollout decoding and optimization

Let $z _ { t , a }$ be the old policy’s logit for token a at state $s _ { t } ,$ and let $\mathcal { C } _ { K } ( z _ { t } )$ contain the K largest-logit tokens, with $\mathcal { C } _ { \mathrm { f u l l } } ( z _ { t } ) = \nu$ . The behavior distribution is

$$
\mu _ { \mathrm { o l d } } ^ { K } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } ) = \frac { \mathbf { 1 } \{ \boldsymbol { a } \in \mathcal { C } _ { K } ( \boldsymbol { z } _ { t } ) \} \exp ( \boldsymbol { z } _ { t , a } ) } { \sum _ { b \in \mathcal { C } _ { K } ( \boldsymbol { z } _ { t } ) } \exp ( \boldsymbol { z } _ { t , b } ) } .\tag{4}
$$

All training arms use $T = 1$ and $p = 1$ , so changing K changes admissibility without changing relative log-odds among retained tokens.

For sampled tokens, the policy ratio is $\rho _ { i , t } = \pi _ { \theta } ( a _ { i , t } \mid s _ { i , t } ) / \pi _ { \mathrm { o l d } } ( a _ { i , t } \mid s _ { i , t } )$ . Let

$$
u ( \rho , A ) = \mathrm { m a x } \{ - \rho A , - \mathrm { c l i p } ( \rho , 1 - \epsilon , 1 + \epsilon ) A \} ,\tag{5}
$$

$$
d ( \rho , A ) = \left\{ \begin{array} { l l } { u ( \rho , A ) , } & { A \geq 0 , } \\ { \operatorname* { m i n } \{ - c A , u ( \rho , A ) \} , } & { A < 0 , } \end{array} \right.\tag{6}
$$

with $\epsilon = 0 . 2$ and $c = 3 .$ The first line is the negative per-sample clipped PPO surrogate—the minimization-loss form of the PPO clipped objective [Schulman et al., $2 0 1 7 ] ;$ the second is dual clipping [Ye et al., 2020]. The rollout engine records the processed log probability $\ell _ { i , t } ^ { \mathrm { r o l l } }$ , while the actor recomputes the old full-softmax log probability $\ell _ { i , t } ^ { \mathrm { a c t o r } }$ . We form

$$
w _ { i , t } = \operatorname* { m i n } \bigl \{ 2 , \exp ( \ell _ { i , t } ^ { \mathrm { a c t o r } } - \ell _ { i , t } ^ { \mathrm { r o l l } } ) \bigr \} , \qquad \mathcal { L } _ { \mathrm { p g } } = \frac { \sum _ { i , t } M _ { i , t } w _ { i , t } d ( \rho _ { i , t } , \widehat { A } _ { i } ) } { \sum _ { i , t } M _ { i , t } } ,\tag{7}
$$

where M masks padding and w is detached. This matches the implemented order: PPO and dual clipping precede the unnormalized rollout weight. Under an ideal synchronized implementation, $\mathrm { e x p } \dot { ( \ell ^ { \mathrm { a c t o r } } - \ell ^ { \mathrm { r o l l } } ) }$ for a retained top-K action equals the top-K mass and cannot exceed one. Separately computed actor and rollout-engine likelihoods can yield larger corrections, which are capped at two. For numerical stability, both log-ratios are clamped to [−20, 20] before exponentiation; the low-variance KL token term is additionally clamped to $[ - 1 \bar { 0 } , 1 0 ]$ . The loss includes that reference-KL penalty with coefficient $1 0 ^ { - 3 }$ . Top-K truncates the collection distribution rather than the actor’s parameterized softmax, thereby affecting both collected actions and their correction weights. The correction applies to sampled actions within the behavior support, and the resulting support-dependent weights can also change effective gradient scale.

## 3.5 Adaptive decoding by token budget and entropy

We evaluate the monotone schedule

$$
K ( B ) = \left\{ \begin{array} { l l } { 1 0 , } & { 0 \le B < 5 0 \mathrm { M } , } \\ { 2 0 , } & { 5 0 \mathrm { M } \le B < 1 0 0 \mathrm { M } , } \\ { \mathrm { f u l l } , } & { B \ge 1 0 0 \mathrm { M } . } \end{array} \right.\tag{8}
$$

The active value is fixed for an entire collection batch; the first batch beginning at or after a threshold uses the new support. Boundaries are therefore nominal and can be exceeded by one batch. This rule supplies budget-indexed adaptation, while the entropy gate below supplies state-dependent adaptation.

The budget schedule widens support globally as training progresses, but uncertainty is not uniform across token positions. At a low-entropy position, the next-token distribution is strongly peaked, so widening the candidate set may mostly expose low-probability alternatives. At a high-entropy position, several continuations are competitive, making broader support more likely to expose meaningful alternative paths. We therefore ask whether widening can be applied selectively at locally uncertain positions. As an exploratory extension, we also gate widening at the token level. We compute the entropy $H _ { 2 0 } ( s _ { t } )$ of the normalized top-20 distribution and retain $K = 1 0$ when $H _ { 2 0 } ( s _ { t } ) < 0 . 2 5$ nats; otherwise we use the arm’s nominal wider support. This single-run experiment tests the feasibility of state-dependent support adaptation.

We compare these budget-indexed and entropy-gated decoding rules with fixed-support controls under the common training and evaluation protocol described next.

## 4 Experimental Design

Model, data, and verifier. We train Llama-3.1-Nemotron-Nano-4B-v1.1 [Bercovich et al., 2025] on a difficulty-filtered subset derived from rStar-Coder [Liu et al., 2025]. We first sample the same base model five times per candidate problem at $T = 0 . 6$ and $p = 0 . 9 5$ (maximum 16,384 tokens), and retain problems solved in at most two of the five attempts. We focus on these lowsolvability problems to stress-test rollout sampling efficiency. On easy problems, useful solutions are already sampled frequently, leaving less room for the rollout distribution to matter. On harder problems, successful trajectories are relatively rare, so how the fixed rollout-token budget is allocated can have a larger effect on whether training observes informative successes and mixed-reward groups. A fixed stratified split yields 1,323 training, 300 validation, and 300 test problems with disjoint identifiers; validation and test each contain 100 problems from the $0 / 5 , 1 / 5 ,$ , and $2 / 5$ strata. A single user message requests a fenced Python program, and we execute the last fenced code block using SandboxFusion [Bytedance-Seed-Foundation-Code-Team et al., 2025]. Reward is one only if the program passes all supplied test cases within a 10-second per-test timeout; missing code, execution errors, and timeouts receive zero.

Training and evaluation. Each update samples 64 prompts and eight responses per prompt, with maximum prompt and response lengths of 2,048 and 24,576 tokens. We use a verl [Sheng et al., 2024] fork with vLLM collection and FSDP training, bfloat16, one policy epoch, AdamW with constant learning rate $5 \times 1 0 ^ { - 6 } , 0 . 0 1$ weight decay, (0.9, 0.999) betas, no warmup, gradient-norm clipping at 1, a 64-sequence PPO minibatch, and no entropy bonus. Runs use one node with eight H100 GPUs. Unless noted otherwise, ± denotes the sample standard deviation across three independent training runs.

Validation is configured every 20 optimizer updates. Because token use per update is large and variable, the resulting observations are irregular and often far apart. At each validation point, we sample five responses per problem and average their binary outcomes. We use the same evaluation decoder for every training arm so that differences in validation performance reflect the learned policies rather than different test-time sampling rules. The temperature and nucleus settings $( T = 0 . 6$ and $p = 0 . 9 5 )$ follow the model’s recommended sampling configuration, while $K = 2 0$ is held fixed across all evaluations. The average binary outcome is the same estimator later called pass@1. With $B ^ { \star } = 1 5 0 \mathbf { M }$ , we report piecewise-linear estimates

$$
\widehat { \mathrm { A c c } } _ { m } \ @ B ^ { \star } = \widehat { \mathrm { V a l A c c } } _ { m } ( B ^ { \star } ) , \qquad \widehat { \mathrm { A U C } } _ { m , B ^ { \star } } = \frac { 1 } { B ^ { \star } } \int _ { 0 } ^ { B ^ { \star } } \widehat { \mathrm { V a l A c c } } _ { m } ( B ) \mathrm { d } B ,\tag{9}
$$

The endpoint measures performance at the target token budget, whereas AUC captures how quickly a method acquires validation performance over the full training horizon. The hat denotes linear interpolation between actual evaluations; AUC uses trapezoidal integration of those segments. For a resumed comparison beginning at shared-parent budget $B _ { 0 }$ , post-parent AUC is $\begin{array} { r } { ( B ^ { \star } - B _ { 0 } ) ^ { - 1 } \int _ { B _ { 0 } } ^ { B ^ { \star } } \widehat { \mathrm { V a l A c c } _ { m } } ( B ) } \end{array}$ dB. Figure 1 exposes the raw knots, Table 1 reports bracket widths, and both quantities are labeled as common-budget interpolation estimates. Because validation checkpoints are sparse and irregular, these quantities are descriptive common-budget estimates rather than direct observations at $B ^ { \star }$ . Reported standard deviations quantify run-to-run variation conditional on the piecewise linear interpolation and do not account for uncertainty due to unobserved learning-curve shape between checkpoints.

Cohorts and controls. The primary cohort compares full support, fixed $K = 1 0 ,$ , fixed $K = 2 0$ and Equation (8). The scheduled and fixed- $K = 1 0$ arms share the model, optimizer, and data state at their within-run branch point; full support and fixed $K = 2 0$ are independently trained static controls with separate parents and aligned replicate indices. Owing to whole-batch switching, the primary scheduled runs actually enter $K = 2 0$ at 51.16–52.49M tokens and full support at 101.12–104.85M. A follow-up decomposition branches four continuations from each early $\dot { K } = 1 0$ parent: remain at $K = 1 0 ,$ , change only to $K = 2 0$ , change directly to full support at 100M, or use both transitions. Two decomposition runs are new continuations, while the third reuses the corresponding primary fixed and scheduled runs. The entropy-gate comparison uses a single shared parent and is reported separately.

For the final 100–150M interval, we additionally aggregate the fraction of prompt groups whose eight rollouts contain both reward values, the mean response length, and mixed groups per million attempted tokens. Groups and tokens from batches crossing the interval boundaries are prorated. These statistics connect Equation (2) to the observed rollouts as a mechanism diagnostic.

## 5 Results

## 5.1 Token-budgeted adaptive decoding versus fixed support

We first ask whether widening rollout support improves validation performance at a common attempted-token budget. Our primary comparison is between the widening schedule and its matched fixed- $K = 1 0$ continuation, which share a parent within each run. Fixed $K = 2 0$ and full support provide independently trained static baselines. Table 1 reports both endpoint performance and performance integrated over the 150M-token training horizon.

Table 1: Piecewise-linear common-budget validation estimates at $B ^ { \star } = 1 5 0 \mathrm { M }$ attempted completion tokens (percentages; three runs per method). Reported ± values are sample standard deviations across runs conditional on the interpolation and do not capture uncertainty from sparse evaluation checkpoints. The bold row reports the matched schedule–fixed- $K = 1 0$ difference. “Max bracket” is the widest run level interval between observed validation checkpoints surrounding 150M.
<table><tr><td>Method</td><td>Training support</td><td>Acc@150M</td><td> $\widehat { \mathbf { A U C } } _ { 1 5 0 \mathbf { M } }$ </td><td>Max bracket (M)</td></tr><tr><td>Full support</td><td>full</td><td> $1 8 . 3 4 \pm 2 . 5 1 $ </td><td> $1 6 . 2 6 { \pm } 1 . 8 0 $ </td><td>105.95</td></tr><tr><td>Fixed support</td><td>10</td><td> $2 1 . 4 9 { \pm } 2 . 5 8 $ </td><td> $1 8 . 5 6 { \pm } 1 . 5 0 $ </td><td>117.11</td></tr><tr><td>Fixed support</td><td>20</td><td> $2 3 . 3 5 { \pm } 1 . 2 8 $ </td><td> $1 9 . 1 6 { \pm } 0 . 8 9$ </td><td>42.92</td></tr><tr><td>Scheduled</td><td> $1 0 \to 2 0 \to \mathrm { f u l l }$ </td><td> $2 3 . 2 9 { \pm } 2 . 5 1 $ </td><td> $1 9 . 1 6 { \pm } 1 . 5 8 $ </td><td>102.48</td></tr><tr><td>Matched gain</td><td>schedule — fixed  $K = 1 0$ </td><td> $\mathbf { + 1 . 8 0 \pm 0 . 4 2 }$ </td><td> $\mathbf { + 0 . 6 0 \pm 0 . 0 8 }$ </td><td></td></tr></table>

![](images/7fdf8a8a2c6e24703eeddfc51db1ef831ec06815cdea34b60fd39fec71480c6e.jpg)  
Figure 1: Validation versus attempted completion tokens for the primary support strategies. Lines and bands show mean ±1 sample standard deviation after piecewise-linear interpolation; symbols mark actual evaluations. Vertical lines are nominal transitions; the $y \cdot$ -axis is truncated. Fixed $\dot { K } = 2 0$ is independently trained.

Scheduling versus the matched narrow continuation. Under the stated interpolation, the schedule is above its matched fixed-K = 10 continuation in every run. Estimated endpoint differences are 2.12, 1.32, and 1.95 points across the three matched runs, giving +1.80 ± 0.42 points; estimated paired AUC differs by $\mathbf { + 0 . 6 0 \pm 0 . 0 8 }$ points. The shared parents make this the cleanest comparison in the study. Per-run values and checkpoint brackets are reported in Appendix B.

A tuned static support remains competitive. Fixed K = 20 reaches an estimated 23.35±1.28% endpoint success, while the schedule reaches 23.29±2.51%. Corresponding-run schedule– $K = 2 0$ differences are $- 0 . 0 7 { \scriptstyle \pm 3 . 6 4 }$ endpoint points and 0.01±2.17 AUC points. The study does not separate these independently trained arms, and fixed K = 20 remains competitive. Thus the primary result supports widening over remaining at $K = 1 0$ , but does not show an advantage over a well-chosen fixed support. Figure 1 displays both the interpolation and the observation density underlying it.

Exploratory test replay. We additionally evaluate a single replay using the fixed-K = 10 and scheduled continuations from Table 2, together with a full-support checkpoint. At the 150M-token budget, estimated validation success is 24.05%, 24.40%, and 25.28% for full support, fixed $K = 1 0$ and the scheduled continuation, respectively. On the disjoint 300-problem test split, however, pass@1 is 23.07%, 23.87%, and 23.87%, respectively, so the scheduled and fixed- $K = \bar { 1 } 0$ continuations have identical test point estimates. Their test difference is 0.00 percentage points, with a paired-by-question cluster-bootstrap interval of [−2.80, 2.87] points, conditional on the five sampled generations per problem. We treat this result as exploratory because it is based on a single replay and because the test split was also used in an earlier 500M analysis.

## 5.2 Which transition matters?

The primary experiment shows that the full widening schedule improves over its matched $K = 1 0$ continuation, but it does not tell us when widening helps. We therefore run a matched-parent decomposition that isolates the two support transitions: 10 → 20 and 10 → full. Separately, we use a single-run entropy-gated experiment to ask whether widening needs to occur at every token position. Two of the three decomposition runs are new stochastic continuations from their earlier parents, while the third reuses the corresponding primary run; accordingly, baseline values need not match Table 1.

Table 2: Piecewise-linear follow-up estimates at 150M attempted completion tokens. Boldface marks the highest accuracy and AUC within each comparison block. Accuracy and AUC are percentages; actor entropy is the linearly estimated logged token-mean in nats, while $H _ { 2 0 }$ is the gate statistic. The decomposition reports mean ± sample standard deviation. Gate rows are single-run exploratory continuations.
<table><tr><td>Continuation or gate</td><td>n</td><td> $\widehat { { \bf A } { \bf c c } } \ @ { \bf B } ^ { \star }$ </td><td>Post-parent AUC</td><td>Actor entropy</td></tr><tr><td>Matched-parent decomposition</td><td></td><td></td><td></td><td></td></tr><tr><td> $K = 1 0 \mathrm { t h r o u g h o u t }$ </td><td>3</td><td> $2 2 . 9 7 { \scriptstyle \pm 3 . 9 4 }$ </td><td> $2 0 . 3 8 { \pm } 2 . 6 8$ </td><td>0.348±0.013</td></tr><tr><td> $1 0  2 0$ </td><td>3</td><td> $2 1 . 4 1 { \pm } 2 . 7 8$ </td><td> $1 9 . 7 0 { \pm } 1 . 9 9 $ </td><td>0.359±0.043</td></tr><tr><td>10 → full at 100M</td><td>3</td><td> $\mathbf { 2 4 . 2 3 \pm 1 . 4 2 }$ </td><td> ${ \bf 2 0 . 9 6 \pm 0 . 9 7 }$ </td><td>0.437±0.074</td></tr><tr><td> $1 0 \to 2 0 \to \mathrm { f u l l }$ </td><td>3</td><td> $2 3 . 3 5 { \pm } 2 . 5 4 $ </td><td> $2 0 . 4 4 { \pm } 2 . 0 3$ </td><td> $0 . 4 4 7 { \scriptstyle \pm 0 . 0 6 7 }$ </td></tr><tr><td>Exploratory entropy gate</td><td></td><td></td><td></td><td></td></tr><tr><td>Ungated schedule</td><td>1</td><td>24.30</td><td>21.55</td><td>0.440</td></tr><tr><td>Gated, nominal  $K = 2 0$ </td><td>1</td><td>24.71</td><td>20.66</td><td>0.392</td></tr><tr><td>Gated, nominal full</td><td>1</td><td>23.27</td><td>21.60</td><td>0.439</td></tr><tr><td>Gated schedule</td><td>1</td><td>25.62</td><td>21.74</td><td>0.428</td></tr></table>

Direct late expansion has the highest estimated mean in the decomposition, but the uncertainty is material: relative to its matched fixed continuation, its endpoint change is 1.26±3.06 points (positive in two of three runs) and its post-parent AUC change is 0.58±1.88 points (positive in one of three). Its fixed- $K = 2 0$ comparator comes from a separate lineage. We therefore treat late widening as a promising follow-up hypothesis rather than an established improvement. The exact three-stage schedule is more consistent against the middle-only continuation $\mathbf { \left( + 1 . 9 4 \pm 0 . 6 0 \right. }$ endpoint points, positive in all three runs), but changes by only $0 . 3 8 { \pm } 1 . 8 7$ points versus fixed $K \stackrel { = } { = } 1 0$ in this cohort. Sparse validation observations also prevent localizing these differences sharply to a transition boundary.

In the single-run comparison, the gated schedule has a higher estimated endpoint (25.62%) than the ungated schedule (24.30%), while its estimated average actor entropy is lower. This single-run result motivates replicated evaluation with direct gate-activation logging.

## 5.3 Do wider supports create more contrast-bearing groups?

We next ask whether wider support improves learning by producing more contrast-bearing groups for the same rollout-token budget. If this mechanism explains the validation gains, we would expect widening to increase the number of mixed-reward groups obtained per attempted token. We measure this over the final 100–150M-token interval.

Table 3: Rollout-group diagnostics over the final 100–150M-token interval (three runs per method). “Mixed” means that a prompt’s eight rollouts contain both correct and incorrect attempts. Widersupport arms have slightly more mixed groups but also longer completions, yielding no clear increase in mixed groups per attempted token.
<table><tr><td>Final-interval support</td><td>Mixed groups (%)</td><td>Mean length</td><td>Mixed groups/M tokens</td></tr><tr><td>Fixed  $K = 1 0$ </td><td>52.33±5.08</td><td> $^ { 8 , 1 5 2 \pm 1 , 5 2 7 }$ </td><td> $8 . 1 3 { \pm } 0 . 9 1$ </td></tr><tr><td>Direct late full</td><td>54.44±3.64</td><td> $^ { 9 , 1 2 1 \pm 8 5 5 }$ </td><td> $7 . 4 8 { \pm } 0 . 3 6 $ </td></tr><tr><td>Three-stage schedule</td><td>53.90±1.98</td><td> $^ { 8 , 4 2 1 \pm 1 , 4 6 3 }$ </td><td> $8 . 1 3 { \pm } 1 . 1 3$ </td></tr></table>

Relative to fixed $K = 1 0$ , the mixed-group fraction changes by $+ 2 . 1 1 { \pm } 3 . 6 7$ points for direct late expansion and +1.56±3.19 points for the three-stage schedule. The wider supports also produce longer completions: direct late expansion yields −0.66±0.92 mixed groups per million tokens, while the schedule’s mean change has magnitude below 0.01 (SD 0.37). Thus support changes the rollout distribution, but these results do not show that widening increases the number of contrast-bearing groups obtained per attempted token.

## 6 Discussion

Our results provide initial evidence that rollout support can influence token-budgeted RLVR learning dynamics. In the primary cohort, the widening schedule achieves a higher estimated 150M validation value than its matched fixed-K = 10 continuation in all three runs, while performing comparably to the independently trained fixed-K = 20 baseline. The follow-up decomposition highlights late widening to full support as the most promising variant, with some variability across runs that motivates further replication. Mixed-reward-group diagnostics do not fully account for the observed validation differences, leaving the mechanism of support adaptation open.

Several limitations make these conclusions preliminary. We study one 4B model in one code-RLVR setting, the primary results use three runs, and validation checkpoints are sparse. The entropy-gating and test replay analyses use only one run. In addition, changing support affects both the sampled trajectories and the rollout-correction weights, so we cannot attribute the observed differences solely to exploration. Denser evaluations, replicated late-widening experiments, and a fresh held-out test set are the most useful next steps.

## 7 Conclusion

In this 4B code-RLVR setting, rollout support affects validation performance as a function of attempted completion tokens. In the primary cohort, budget-indexed widening has a higher interpolated 150M-token validation value than its matched fixed-K = 10 continuation in all three runs, while its estimated mean is comparable to the independently trained fixed-K = 20 baseline. An exploratory test replay yields identical point estimates for the scheduled and fixed-K = 10 continuations.

The follow-up decomposition suggests that late widening to full support is the most promising component of the schedule, although the current evidence is not sufficient to establish a consistent improvement. Mixed-reward-group diagnostics also do not clearly explain the observed validation gain, leaving the mechanism of support adaptation open. More broadly, our results suggest that candidate support should be treated as an important rollout hyperparameter in RLVR rather than defaulting to either narrow or full support. Adaptive strategies should therefore be evaluated over the full training horizon, against strong fixed-support baselines, and under the rollout-token budget that they directly affect.

## References

Sourya Basu, Govardana Sachitanandam Ramachandran, Nitish Shirish Keskar, and Lav R. Varshney. Mirostat: A neural text decoding algorithm that directly controls perplexity. In International Conference on Learning Representations, 2021.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 41–48. ACM, 2009. doi: 10.1145/1553374.1553380.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. arXiv preprint arXiv:2505.00949, 2025.

Bytedance-Seed-Foundation-Code-Team, :, Yao Cheng, Jianfeng Chen, Jie Chen, Li Chen, Liyu Chen, Wentao Chen, Zhengyu Chen, Shijie Geng, Aoyan Li, Bo Li, Bowen Li, Linyi Li, Boyi Liu, Jiaheng Liu, Kaibo Liu, Qi Liu, Shukai Liu, Siyao Liu, Tianyi Liu, Tingkai Liu, Yongfei Liu, Rui Long, Jing Mai, Guanghan Ning, Z. Y. Peng, Kai Shen, Jiahao Su, Jing Su, Tao Sun, Yifan Sun, Yunzhe Tao, Guoyin Wang, Siwei Wang, Xuwu Wang, Yite Wang, Zihan Wang, Jinxiang Xia, Liang Xiang, Xia Xiao, Yongsheng Xiao, Chenguang Xi, Shulin Xin, Jingjing Xu, Shikun Xu, Hongxia Yang, Jack Yang, Yingxiang Yang, Jianbo Yuan, Jun Zhang, Yufeng Zhang, Yuyu Zhang, Shen Zheng, He Zhu, and Ming Zhu. Fullstack bench: Evaluating llms as full stack coders, 2025. URL https://arxiv.org/abs/2412.00535.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John

Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

John Hewitt, Christopher D. Manning, and Percy Liang. Truncation sampling as language model desmoothing. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3414–3427. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022. findings-emnlp.249.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020.

Mengqi Liao, Xiangyu Xi, Ruinian Chen, Jia Leng, Yangen Hu, Ke Zeng, Shuai Liu, and Huaiyu Wan. Enhancing efficiency and exploration in reinforcement learning for LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1451–1463. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.emnlp-main.75.

Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, Cheng Li, and Mao Yang. rstar-coder: Scaling competitive code reasoning with a large-scale verified dataset. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 58780–58807. Curran Associates, Inc., 2025. doi: 10.52202/085713-1967.

Amritansh Mishra, Supriyo Chakraborty, and Berkcan Kapusuzoglu. On the policy gradient foundations of group relative policy optimization: Credit assignment, gradient sparsity, and rank collapse. arXiv preprint arXiv:2606.29238, 2026.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 Rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 115452–115486. Curran Associates, Inc., 2025. doi: 10.52202/085713-3850.

Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, and Tong Zhang. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and RL. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 163245–163284. Curran Associates, Inc., 2025. doi: 10.52202/085713-5450.

Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang, Qiang Fu, Wei Yang, and Lanxiao Huang. Mastering complex control in MOBA games with deep reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04): 6672–6679, 2020. doi: 10.1609/aaai.v34i04.6144.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pages 113222–113244, 2025. doi: 10.52202/085713-3775.

Yuheng Zhang, Wenlin Yao, Changlong Yu, Yao Liu, Qingyu Yin, Bing Yin, Hyokun Yun, and Lihong Li. Improving sampling efficiency in RLVR through adaptive rollout and response reuse. arXiv preprint arXiv:2509.25808, 2025.

Haomin Zhuang, Yujun Zhou, Taicheng Guo, Yue Huang, Fangxu Liu, Kai Song, and Xiangliang Zhang. Exploring multi-temperature strategies for token- and rollout-level control in RLVR. arXiv preprint arXiv:2510.08892, 2025.

## A Connection Between Reward Contrast and GRPO Gradient Signal

Our motivation builds on the gradient analysis of Mishra et al. [Mishra et al., 2026]. We specialize the relevant result to our binary-reward setting and clarify its relationship to the standardized advantages used in our implementation.

For a prompt x, let $r _ { i } \in \{ 0 , 1 \}$ be the terminal reward of rollout i, let

$$
\bar { r } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } ,
$$

and define the centered reward

$$
U _ { i } = r _ { i } - { \bar { r } } .
$$

If the $G$ rollouts are conditionally independent with success probability $q _ { x } ^ { e }$ under rollout rule $e ,$ then

$$
\operatorname { V a r } ( U _ { i } \mid x , e ) = \frac { G - 1 } { G } \sigma _ { r , x , e } ^ { 2 } , \qquad \sigma _ { r , x , e } ^ { 2 } = q _ { x } ^ { e } ( 1 - q _ { x } ^ { e } ) .\tag{10}
$$

This is the binary-reward specialization of the advantage-variance result of Mishra et al. [Mishra et al., 2026]. By Chebyshev’s inequality,

$$
\operatorname* { P r } ( | U _ { i } | > \varepsilon \mid x , e ) \leq \frac { ( G - 1 ) \sigma _ { r , x , e } ^ { 2 } } { G \varepsilon ^ { 2 } } .\tag{11}
$$

As $q _ { x } ^ { e }$ approaches 0 or 1, reward variance decreases and mixed-reward groups become rare. In the limiting case of a uniform group, $r _ { 1 } = \cdots = r _ { G }$ , we have $U _ { i } = 0$ for every rollout. Our standardized advantage,

$$
\widehat { A } _ { i } = \frac { r _ { i } - \bar { r } } { s _ { r } + \epsilon _ { A } } ,
$$

is therefore also exactly zero, so the reward-driven PPO surrogate receives no contribution from that group.

## B Per-Run Evaluation Details

Table 4 gives the run-level quantities summarized in Table 1; Table 5 gives the raw checkpoints used to estimate the 150M endpoint.

Table 4: Per-run piecewise-linear estimates in percentage points at 150M attempted completion tokens.
<table><tr><td>Method</td><td>Run A Acc/AUC</td><td>Run B Acc/AUC</td><td>Run C Acc/AUC</td></tr><tr><td>Full support</td><td>20.95 / 18.28</td><td>15.95 / 14.83</td><td>18.14 / 15.67</td></tr><tr><td>Fixed  $\dot { K } = 1 0$ </td><td>23.17 / 19.79</td><td>22.77 / 19.01</td><td>18.52 / 16.89</td></tr><tr><td>Fixed  $K = 2 0$ </td><td>21.90 / 18.17</td><td>23.83 / 19.88</td><td>24.33 / 19.42</td></tr><tr><td>Scheduled</td><td>25.29 / 20.47</td><td>24.10 / 19.62</td><td>20.47 / 17.41</td></tr></table>

## C Broader Impacts

Reducing attempted completion tokens can lower one component of RLVR cost, while energy use remains a system-level property. Model, training-data, and deployment risks continue to require task-specific safety evaluation. More capable reasoning models may be used in harmful applications, and verifier-based training can amplify specification errors when the verifier rewards an incomplete proxy for the intended behavior.

Table 5: Actual evaluation checkpoints immediately bracketing 150M attempted completion tokens. As a descriptive check, the mean success rates at each run’s nearest raw checkpoint are 17.80, 21.18, 23.62, and 23.20% in row order; those checkpoints occur at different token counts.
<table><tr><td>Method</td><td>Run A (M)</td><td>Run B (M)</td><td>Run C (M)</td></tr><tr><td>Full support</td><td>138.1-244.1</td><td>116.2-158.4</td><td>132.8–225.2</td></tr><tr><td>Fixed K = 10</td><td>144.6-261.7</td><td>116.2–221.9</td><td>135.7-208.8</td></tr><tr><td>Fixed K = 20</td><td>108.1-151.0</td><td>134.5-153.9</td><td>134.8-154.7</td></tr><tr><td>Scheduled</td><td>51.4–153.5</td><td>119.1–221.5</td><td>81.9–152.8</td></tr></table>