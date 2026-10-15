# ACTIONOPD: DISTILLING DECISIONS, NOT TOKENS

Huining Yuan<sup>\*</sup>, Xiangmin Yi<sup>\*</sup>, Zelai Xu, Zixu Wang,

Xinlei Chen, Xiao-Ping Zhang<sup>†</sup>, Yu Wang<sup>†</sup>, Chao Yu<sup>†</sup>

Tsinghua University

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding authors.

<sup></sup> Project Page <sup>§</sup> Code Models

## ABSTRACT

Distilling capable language-model agents into smaller models requires aligning the student’s behavior with the teacher’s. This raises a fundamental question: at which states and at what granularity should this alignment be performed? Offline imitation provides dense but token-specific supervision and can suffer from state-distribution mismatch. Token-level on-policy distillation (OPD) addresses this mismatch but typically requires teacher token probabilities and compatible tokenization. We argue that semantic actions provide a natural unit of supervision for agent distillation. We introduce ActionOPD, an action-level on-policy distillation method motivated by maximizing expected teacher-action likelihood at student-encountered states. At each such state, ActionOPD samples actions from both the student and a black-box teacher, rewarding student actions according to their semantic agreement with the sampled teacher actions. Through policy optimization, ActionOPD aligns the student’s action-level behavior with the teacher’s decisions without requiring reproduction of teacher-written reasoning or access to teacher token probabilities. Training Qwen3-4B and Qwen3-8B with two teachers on diverse, curated tool-use tasks yields average gains of 3.3–6.9 percentage points across four out-of-distribution benchmarks. ActionOPD scales favorably with student size and teacher capability, without requiring the teacher and student to share a model family or tokenizer. On τ<sup>2</sup>-bench, ActionOPD achieves competitive performance against offline distillation and token-level OPD. We further explore Self-ActionOPD with a same-model teacher and privileged context. These results establish ActionOPD as a promising approach to agent distillation.

## 1 INTRODUCTION

Language models are increasingly deployed as agents that communicate with users, maintain state, and act through external tools (Yao et al., 2023; 2025). In these interactions, an agent’s decisions shape subsequent observations and the states it encounters. Distilling capable but expensive frontier agents into smaller models therefore raises two questions: at which states should the teacher supervise the student, and at what granularity should their behavior be aligned?

Distillation methods address these questions differently. Offline distillation provides dense supervision by fitting teacher-generated sequences (Kim & Rush, 2016), but can face teacher–student state-distribution mismatch (Ross et al., 2011). Token-level on-policy distillation (OPD) supervises student-generated sequences using teacher token probabilities, reducing mismatch while typically requiring compatible tokenization (Agarwal et al., 2024; Gu et al., 2024). Both supervise tokens without explicitly matching action semantics (Figure 2).

![](images/b80f32b7caa31d46337ae79acbbad16b2e23e979e270ce3b82e32b8c7c338f89.jpg)  
Figure 1: Student–teacher scaling of ActionOPD. Larger students attain higher final averages. DeepSeek-V4- Flash yields higher student averages than Qwen3.7-Flash.

![](images/7fffd1bb019da9ef44b2d8a5db508f0bb941e5b7610def9af4eff1a34b0f82ae.jpg)  
Figure 2: Supervision for agent learning. Offline distillation (KD) imitates teacher tokens at teacher induced states. Token-level OPD compares token distributions along student-generated interactions. Outcome-based RL uses terminal feedback. ActionOPD instead rewards semantic agreement with teacher actions at student-encountered states. Here, we sample only one response per state for demonstration. For implementation, we expand the sampled teacher multiset and student state group.

Our key observation is that semantic actions provide a natural unit of supervision for agent distillation. A tool call expresses a decision through its tool name and arguments; a natural-language response conveys communicative intent and information. The same decision can be realized through different wording and reasoning. This shifts the objective from reproducing a particular teacher sequence to increasing the likelihood of its underlying action.

We introduce ActionOPD, an action-level on-policy distillation method motivated by maximizing expected teacher-action likelihood at student-encountered states. Here, likelihood is the total probability mass of student outputs that semantically match a teacher action, not the probability of reproducing one teacher-written sequence. This objective yields a REINFORCE-form policy gradient with action-level semantic agreement as reward (Williams, 1992). In practice, ActionOPD samples actions from the teacher and responses from the current student policy, rewarding student responses according to their semantic agreement with teacher actions. It requires only generation access to a black-box teacher, not teacher token probabilities or tokenizer alignment. The resulting action level rewards train the student’s own reasoning and action tokens without imposing teacher-written reasoning as a token-level imitation target.

We train Qwen3-4B and Qwen3-8B students using teachers from two model families on curated tooluse tasks from Agent World Model (AWM) and EnvScaler (Wang et al., 2026b; Song et al., 2026). All four configurations improve average performance across four out-of-distribution agent benchmarks. ActionOPD scales favorably with student size and teacher capability: larger students achieve higher final averages, and DeepSeek-V4-Flash yields higher student averages than Qwen3.7-Flash at both sizes (Figure 1). Comparisons on τ<sup>2</sup>-bench (Barres et al., 2025) show competitive performance against offline distillation and token-level OPD. Ablation studies support the effectiveness of semantic action supervision. We further explore Self-ActionOPD, in which a same-model teacher receives privileged task information. This extension suggests a path toward agent self-improvemen through action-level self-distillation.

## Contributions. Our contributions are threefold:

• We introduce ActionOPD, an action-level on-policy distillation method that decouples teacher decision supervision from student token realization, enabling distillation from black-box teachers without teacher token probabilities or tokenizer alignment.

• We demonstrate ActionOPD’s applicability across teacher families and empirical scaling with student size and teacher capability. All tested configurations improve the fourbenchmark average over their corresponding base models by 3.3–6.9 percentage points.

• We show that ActionOPD achieves competitive performance against offline distillation and token-level OPD while requiring only generation access to the teacher.

## 2 ACTIONOPD

A language-model agent alternates between generating responses and receiving environment observations. At step t, its visible state $s _ { t }$ contains the task, tools, and retained interaction history. Its response $y _ { t }$ realizes a semantic action $a _ { t } = \phi _ { s _ { t } } ( y _ { t } )$ . Here, action level refers to the semantics of a tool call or user-facing message, rather than the token sequence used to express it. The function $\phi _ { s }$ maps responses to semantic actions at state s. We use $\pi _ { \theta }$ for student probabilities over semantic actions, complete responses, and tokens, distinguished by their arguments, and $\pi _ { \mathrm { T } }$ for the teacher. Student and teacher receive the same public context and tools through their native prompts.

To align the student’s action-level behavior with the teacher, we maximize expected teacher-action likelihood at student-encountered states:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { s \sim d _ { \mathrm { S } } , a ^ { \mathrm { T } } \sim \pi _ { \mathrm { T } } ( \cdot | s ) } \left[ \pi _ { \theta } \big ( a ^ { \mathrm { T } } \mid s \big ) \right] .\tag{1}
$$

Denote this expectation by $J _ { \mathrm { A c t i o n O P D } } ( \theta )$ . Here, $d _ { \mathrm { S } }$ denotes the distribution of student-encountered states collected by the rollout procedure in Section 2.3, held fixed during each policy update.

## 2.1 FROM ACTION LIKELIHOOD TO POLICY GRADIENTS

To optimize Equation 1, we express the likelihood of a teacher action as the total probability mass of student responses that realize it:

$$
\pi _ { \boldsymbol { \theta } } ( a ^ { \mathrm { { T } } } \mid s ) = \sum _ { y } \pi _ { \boldsymbol { \theta } } ( y \mid s ) \mathbf { 1 } \{ \phi _ { s } ( y ) = a ^ { \mathrm { { T } } } \} .\tag{2}
$$

Here, $\mathbf { \mathbb { 1 } } \{ \phi _ { s } ( y ) = a ^ { \mathrm { T } } \}$ is the semantic-agreement indicator: it equals one when $y$ realizes the teacher action and zero otherwise. Agreement is defined by action semantics, not textual identity. Substitut ing Equation 2 into Equation 1 and exchanging expectations yields an equivalent reward objective:

$$
\begin{array} { r l } & { r _ { \mathrm { T } } ( s , a ) = \mathbb { E } _ { a ^ { \mathrm { T } } \sim \pi _ { \mathrm { T } } ( \cdot | s ) } [ 1 \{ a = a ^ { \mathrm { T } } \} ] , } \\ & { J _ { \mathrm { A c t i o n O P D } } ( \theta ) = \mathbb { E } _ { s \sim d _ { \mathrm { S } } , y \sim \pi _ { \theta } ( \cdot | s ) } [ r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) ) ] . } \end{array}\tag{3}
$$

For a fixed state and student action, $r _ { \mathrm { T } }$ averages binary semantic agreement over the teacher’s action distribution. With the state distribution, teacher policy, and semantic mapping held fixed during the update, the score-function identity yields

$$
\nabla _ { \boldsymbol { \theta } } J _ { \mathrm { A c t i o n O P D } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { s , a ^ { \mathrm { T } } , y } \big [ { \mathbf { 1 } \big \{ \boldsymbol { \phi } _ { s } ( y ) = a ^ { \mathrm { T } } \big \} } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( y \mid s ) \big ] ,\tag{4}
$$

where the variables follow the distributions above. This is precisely the REINFORCE policygradient form (Williams, 1992), with semantic action agreement as reward (Appendix A.1).

This gradient can be estimated from sampled teacher actions and student responses using student log-probabilities, without teacher token probabilities or tokenizer alignment. Since log $\pi _ { \theta } ( y \mid s ) =$ $\begin{array} { r } { \sum _ { \ell = 1 } ^ { | y | } \log \pi _ { \theta } ( y _ { \ell } \ | \ s , y _ { < \ell } ) } \end{array}$ , where ℓ indexes response tokens, a single semantic-agreement reward weights every token’s contribution to the policy gradient. The supervision is therefore action-level, while optimization covers the student’s own reasoning and action tokens without imposing teacherwritten reasoning as a hard target.

## 2.2 SEMANTIC MATCHING AND REWARD DESIGN

We estimate the expected agreement reward $r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) )$ using sampled teacher responses and a practical semantic matcher. At state s, we request $K = 3$ teacher responses realizing $\overset { \bullet } { a } _ { k } ^ { \mathrm { T } } = \phi _ { s } ( y _ { k } ^ { \mathrm { T } } )$ and retain the multiset $\mathcal { M } _ { \mathrm { T } } ( s ) = [ y _ { 1 } ^ { \mathrm { T } } , \dots , y _ { K } ^ { \mathrm { T } } ]$ . The matcher $m _ { s } ( y , y _ { k } ^ { \mathrm { T } } ) \in \{ 0 , 1 \}$ approximates $\mathbf { 1 } \{ \phi _ { s } ( y ) = a _ { k } ^ { \mathrm { T } } \}$ . Invalid outputs never match. Define

$$
c ( s , y ) = \sum _ { k = 1 } ^ { K } m _ { s } ( y , y _ { k } ^ { \mathrm { T } } ) .\tag{5}
$$

Repeated teacher samples contribute separately to the count. Under independent teacher draws, $c ( s , y ) / K$ is an unbiased estimate of the probability that the matcher judges y to agree with a sampled teacher response. With accurate matching, this probability equals $r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) )$ . Appendix A.3 details sample retention and caching.

The matcher compares action content, not reasoning (Appendix A.3). Tool names and canonicalized arguments are checked programmatically, with a frozen LLM assessing remaining argument differences. For example, get weather calls with ${ \mathsf { c i t y } } = " { \mathrm { P a r i s } } "$ and $\mathtt { u n i t } { = } " \mathtt { c e l s i u s } ^ { \prime }$ are equivalent regardless of argument-key order. For communicative responses, we accept normalized exact matches and otherwise ask the LLM to compare intent and factual content. At the same state, “Basic-economy tickets cannot be changed” and “Changes are not permitted for basic-economy tickets” convey the same restriction. Each candidate is compared with every teacher sample.

Although $c / K$ estimates agreement, three samples provide only coarse frequency estimates. To limit sensitivity to sampled counts, we assign a unit reward to the first match and a smaller frequency bonus to each additional match. For $K > 1$ , we use

$$
r _ { \lambda } ( s , y ) = \left\{ \begin{array} { l l } { - 1 , } & { y \mathrm { ~ c o n t a i n s ~ a n ~ i n v a l i d ~ a c t i o n , } } \\ { 0 , } & { y \mathrm { ~ c o n t a i n s ~ a ~ v a l i d ~ a c t i o n ~ a n d ~ } c ( s , y ) = 0 , } \\ { 1 + \lambda \displaystyle \frac { c ( s , y ) - 1 } { K - 1 } , } & { c ( s , y ) > 0 . } \end{array} \right.\tag{6}
$$

With $\lambda = 0 . 5$ and $[ A , A , B ]$ , a response matching A earns 1.25 and one matching B earns 1: the additional A match contributes 0.25, compared with 1 for the first match. Invalid outputs incur a separate penalty for protocol violations. The reward in Equation 6 modifies the original likelihood objective. Appendix A.2 derives the objective and its single-sample, binary, and raw-count cases.

## 2.3 STATE-GROUP ROLLOUT AND OPTIMIZATION

State-group rollout. We use these rewards for both action selection and policy optimization (Figure 3). At state $s _ { g } ,$ group g contains $N \ = \ 4$ responses indexed by i, sampled as $y _ { g , i } ~ \sim ~ \pi _ { \theta _ { \mathrm { o l d } } } ( . ~ | ~ s _ { g } )$ with frozen $\theta _ { \mathrm { o l d } }$ . All candidates are scored against the same teacher multiset, yielding $r _ { g , i } = r _ { \lambda } ( s _ { g } , y _ { g , i } )$ . We execute only the parsed action in $y _ { g , i ^ { \star } }$ , where $i ^ { \star } \in \arg \operatorname* { m a x } _ { i } r _ { g , i } .$ , but train on all responses in retained groups. Repeated selection produces one trajectory per task with a teacher-guided visited-state distribution, rather than the ordinary single-sample rollout distribution.

![](images/cb8037718011431201f5fbb36528c74618e1dc2816b0090c8530aa4f6df53515.jpg)  
Figure 3: ActionOPD at one state. Student candidates are scored against a shared teacher multiset. One highest-reward action is committed. All candidates in retained groups contribute to training.

By repeatedly committing high-reward actions, state-group rollout can explore trajectories less likely under ordinary student rollouts, while introducing a degree of off-policy state visitation. Same-state candidate groups also provide local reward comparisons for policy optimization. We use $N = 4$ to balance teacher-guided exploration and deviation from the student’s rollout distribution.

Policy optimization. Motivated by the REINFORCE-form gradient in Equation 4, we use grouprelative advantages and a PPO-style clipped surrogate (Shao et al., 2024; Schulman et al., 2017). Groups without supervision or with identical rewards are excluded from optimization. Equal-reward groups may still advance the environment. Within each retained group, the mean reward provides a local baseline for all N candidates, including invalid actions with reward −1. We normalize the centered rewards by a shared batch-wide standard deviation:

$$
\bar { r } _ { g } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } r _ { g , j } , \qquad \widetilde { A } _ { g , i } = r _ { g , i } - \bar { r } _ { g } , \qquad A _ { g , i } = \frac { \widetilde { A } _ { g , i } } { \sigma + \epsilon } .\tag{7}
$$

Here $\sigma$ is the population standard deviation of centered advantages over all candidates in the retained groups, and $\epsilon > 0$ stabilizes division. This shared normalization preserves relative advantage magnitudes across different state groups.

Let $\tau _ { + }$ index active generated tokens $( g , i , \ell )$ in retained groups. To compare the updated and rollout policies, define the ratio of temperature-scaled token probabilities

$$
\rho _ { g , i , \ell } ( \theta ) = \frac { \pi _ { \theta } ( y _ { g , i , \ell } \mid s _ { g } , y _ { g , i , < \ell } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { g , i , \ell } \mid s _ { g } , y _ { g , i , < \ell } ) } .\tag{8}
$$

Following the token-level expansion in Section 2.1, $A _ { g , i }$ weights every active token in $y _ { g , i }$ . PPO clipping discourages excessively large changes in token probabilities. With clipping width $\eta > 0$ we maximize the token-mean surrogate

$$
J _ { \mathrm { c l i p } } ( \theta ) = \frac { 1 } { | { \cal T } _ { + } | } \sum _ { ( g , i , \ell ) \in { \cal T } _ { + } } \operatorname* { m i n } \{ \rho _ { g , i , \ell } ( \theta ) A _ { g , i } , \mathrm { c l i p } ( \rho _ { g , i , \ell } ( \theta ) , 1 - \eta , 1 + \eta ) A _ { g , i } \} .\tag{9}
$$

With the frequency bonus, differing match counts can yield nonzero advantages even when all candidates match a teacher sample. This allows optimization to distinguish among teacher-consistent responses. These updates are not unbiased estimates of the gradient in Equation 4 after group-relative normalization, masking, and clipping. Appendix A.4 details dual clipping and implementation.

## 3 EXPERIMENTS

We evaluate ActionOPD’s cross-benchmark generalization, student–teacher scaling, and performance relative to other distillation methods. We then use ablations and case studies to examine its components and decision-making behavior, and explore Self-ActionOPD with privileged teacher context.

## 3.1 CROSS-BENCHMARK TRANSFER SETUP

Training data. We trained on Agent World Model (AWM; Wang et al., 2026b) and EnvScaler (Song et al., 2026). AWM provides executable tools and persistent backend state; EnvScaler adds a simulated user to the interaction loop. The curated pool comprised 5,942 AWM tasks from 794 environments and 755 EnvScaler tasks from 49 environments: $\mathbf { 6 , 6 9 7 }$ tasks in total. Screening assessed environment integrity and static task feasibility, without requiring successful expert rollouts (Appendix B.1).

Models and training. We paired Qwen3-4B and Qwen3-8B students (Yang et al., 2025) with Qwen3.7-Flash (Alibaba Cloud, 2026) and DeepSeek-V4-Flash (DeepSeek-AI, 2026) action teachers in a 2×2 design. Each configuration used 100 training steps and 64 task starts per step (59 AWM, 5 EnvScaler), with N = 4 student candidates, $K = 3$ teacher samples per state, and frequency bonus scale $\lambda = 0 . 5$ . Task starts could revisit tasks in the curated pool. We report the final checkpoints, with Base denoting the corresponding student before ActionOPD training. Appendix B.2 details settings and auxiliary models.

Evaluation. We evaluated transfer on BFCL Multi-turn (Patil et al., 2025), ToolSandbox’s verified-local-only subset (Lu et al., 2025), AppWorld Normal and Challenge (Trivedi et al., 2024), and $\tau ^ { 2 }$ -bench Airline, Retail, and Telecom (Barres et al., 2025). These tasks were excluded from AWM/EnvScaler training. Students used greedy decoding without teacher-guided action selection. Appendix B.3 details coverage, trials, and interaction protocols.

Metrics. We report BFCL accuracy, ToolSandbox similarity (not binary success), AppWorld task/scenario goal completion (TGC/SGC), and Tau $\mathrm { { p a s s ^ { 1 } / p a s s ^ { 4 } } } ;$ : single-trial success and four-trial all-success reliability, not best-of-four success (Yao et al., 2025). The average in Table 1 weights the four benchmarks equally, using mean Normal/Challenge TGC for AppWorld and mean domain pass<sup>1</sup> for Tau. Appendix B.3 gives the estimators.

## 3.2 MAIN RESULTS

Table 1 compares distilled students with base models and directly evaluated teachers.

Cross-benchmark generalization. After training on AWM and EnvScaler, all four configurations improved the four-benchmark average over their base models by 3.3–6.9 percentage points. Gains extended across BFCL accuracy, ToolSandbox similarity, and AppWorld task completion. These gains on benchmarks excluded from training demonstrate that action-level distillation transfers beyond the training environments, improving average performance across diverse agent tasks.

Table 1: Cross-benchmark transfer after 100 ActionOPD steps (%). + denotes training with the named teacher. AppWorld: task/scenario success on Normal/Challenge; scenario success requires all task variants to succeed. Tau: pass $^ { 1 } / \mathrm { p a s s } ^ { 4 }$ . Average weights four benchmarks equally. Bold/underlined: best/second-best distinct scores per student size, including Base; ties share marks. Rankings and green average gains over Base (percentage points) use unrounded scores. Generation settings differ by role (Table 8).
<table><tr><td rowspan="2">Model</td><td rowspan="2">BFCL</td><td rowspan="2">Tool Sandbox</td><td colspan="2">AppWorld</td><td colspan="3">τ2-Bench</td><td rowspan="2">Average</td></tr><tr><td>Normal</td><td>Challenge</td><td>Airline</td><td>Retail</td><td>Telecom</td></tr><tr><td>Direct Teacher Evaluation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.7-Flash</td><td>58.0</td><td>76.2</td><td></td><td></td><td>70.2/55.436.5/23.078.0/54.0 55.9/18.4 90.6/71.9</td><td></td><td></td><td>65.6</td></tr><tr><td>DeepSeek-V4-Flash</td><td>57.0</td><td>72.3</td><td></td><td></td><td></td><td></td><td>84.5/75.074.8/54.777.0/52.051.8/15.898.9/95.6</td><td>71.2</td></tr><tr><td>Qwen3-4B Student</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>34.3</td><td>55.5</td><td>9.5/3.6</td><td>2.6/0.7</td><td>34.5/8.0</td><td>26.8/7.9</td><td>20.6/5.3</td><td>30.8</td></tr><tr><td>Qwen3.7-Flash +</td><td>42.3</td><td>61.8</td><td>14.3/8.9</td><td>4.8/0.7</td><td>33.5/12.0</td><td>34.4/9.6</td><td>22.4/7.9</td><td>35.9 +5.1</td></tr><tr><td>DeepSeek-V4-Flash X</td><td>44.1</td><td>63.3</td><td>13.1/5.4</td><td>6.5/2.2</td><td>39.0/20.0</td><td>37.1/8.8</td><td>24.1/16.7</td><td>37.7 +6.9</td></tr><tr><td>Qwen3-8B Student</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>39.9</td><td>57.7</td><td>15.5/5.4</td><td>4.6/0.7</td><td>35.0/8.0</td><td>38.2/14.0</td><td>31.8/14.9</td><td>35.6</td></tr><tr><td>Qwen3.7-Flash +</td><td>45.0</td><td>62.5</td><td>16.7/8.9</td><td>7.7/1.4</td><td>36.5/22.0</td><td>37.7/10.5</td><td>34.2/14.9</td><td>39.0 +3.3</td></tr><tr><td>DeepSeek-V4-Flash +</td><td>45.5</td><td>62.5</td><td>17.9/10.7</td><td>7.7/2.9</td><td>44.0/18.0</td><td>40.1/15.8</td><td>33.8/12.3</td><td>40.0 +4.4</td></tr></table>

Scaling with student size. With either teacher, Qwen3-8B achieved a higher final average than Qwen3-4B under the same task-start and training-step budgets. We used the same candidate count, teacher sample count, and reward formulation for both students. This pattern supports empirical scaling across the two tested sizes, with larger students yielding stronger distilled agents.

Scaling with teacher capability. DeepSeek-V4-Flash achieved a higher direct-evaluation average than Qwen3.7-Flash and yielded higher student averages at both sizes (Figure 1), supporting empirical scaling with teacher capability. The DeepSeek-to-Qwen results also demonstrate effective cross-family distillation through an action-level interface that requires neither a shared model family nor tokenizer alignment. This allows teacher selection to prioritize decision quality rather than compatibility with the student’s token space

## 3.3 COMPARISON WITH DISTILLATION BASELINES

We compared ActionOPD, offline distillation (KD), and sampled-token reverse-KL OPD using a Qwen3-4B student, a Qwen3-32B teacher, and the same 178 Tau training tasks. All methods used 24 task starts per step—eight per domain—and were evaluated at step 50. KD collected four independent teacher trajectories per task start and fine-tuned on all reasoning and action/message tokens from successful trajectories. OPD collected four student trajectories per task start and used teacher log-probabilities to supervise sampled tokens. ActionOPD used state-group rollout, sampling four candidates at each visited state while advancing one trajectory per task start. It used K = 3 teacher samples per state and λ = 0.5 (Appendix B.4).

KD achieved the highest test average pass<sup>4</sup> (19.2), indicating greater repeated-success reliability on limited tasks and states. Conversely, OPD and ActionOPD achieved higher test average pass<sup>1</sup> than KD. This pattern demonstrates the advantages of on-policy supervision: reducing mismatch between teacher demonstrations and student-encountered states, and improving performance across a broader range of student tasks. ActionOPD achieved higher training-set averages than OPD on both metrics and exceeded OPD’s test average pass<sup>1</sup> by 0.4 percentage points, although its test average pass<sup>4</sup> was lower. While these results do not establish uniform superiority, ActionOPD offers a competitive alternative with a more flexible teacher interface. Its generation-only interface requires neither teacher token log-probabilities nor tokenizer alignment, and does not require a shared model family. The cross-family transfer demonstrated in Section 3.2 complements this same-family comparison.

Table 2: Tau distillation comparison: $\mathrm { p a s s ^ { 1 } / p a s s ^ { 4 } }$ (%). Qwen3-4B students and a Qwen3-32B teacher. Trained students are evaluated at step 50. Four trials per task. Average equally weights domains within each split. ActionOPD uses the same run as in Table 3. Bold/underlined: best/second best distinct scores (unrounded); teacher excluded.
<table><tr><td rowspan="2">Model</td><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Telecom</td><td colspan="2">Average</td></tr><tr><td>Train (30)</td><td>Test (20)</td><td>Train (74)</td><td>Test (40)</td><td>Train (74)</td><td>Test (40)</td><td>Train</td><td>Test</td></tr><tr><td>Qwen3-32B</td><td>49.2/16.7</td><td>43.8/15.0</td><td>46.0/16.2</td><td>46.9/15.0</td><td>25.7/13.5</td><td>41.9/30.0</td><td>40.3/15.5</td><td>44.2/20.0</td></tr><tr><td>Base</td><td>35.0/10.0</td><td>33.8/5.0</td><td>26.7/6.8</td><td>26.9/10.0</td><td>17.2/4.1</td><td>26.9/7.5</td><td>26.3/6.9</td><td>29.2/7.5</td></tr><tr><td>KD</td><td>38.3/20.0</td><td>37.5/20.0</td><td>37.5/13.5</td><td>33.8/12.5</td><td>25.3/13.5</td><td>41.3/25.0</td><td>33.7/15.7</td><td>37.5/19.2</td></tr><tr><td>OPD</td><td>40.0/13.3</td><td>41.3/20.0</td><td>40.2/10.8</td><td>37.5/10.0</td><td>22.6/10.8</td><td>40.0/22.5</td><td>34.3/11.7</td><td>39.6/17.5</td></tr><tr><td>ActionOPD</td><td>42.5/16.7</td><td>40.0/15.0</td><td>42.9/10.8</td><td>39.4/12.5</td><td>28.4/13.5</td><td>40.6/15.0</td><td>37.9/13.7</td><td>40.0/14.2</td></tr></table>

Table 3: Tau ablations: $\mathrm { p a s s ^ { 1 } / p a s s ^ { 4 } } ( \% )$ . Qwen3-4B student, Qwen3-32B teacher; trained variants at step 50. Four trials per task. Parentheses: task counts. Average weights domains equally per split. Bold/underlined: best/second-best distinct unrounded scores, including Base; ties share marks.
<table><tr><td rowspan="2">Model</td><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Telecom</td><td colspan="2">Average</td></tr><tr><td>Train (30)</td><td>Test (20)</td><td>Train (74)</td><td>Test (40)</td><td>Train (74)</td><td>Test (40)</td><td>Train</td><td>Test</td></tr><tr><td>Base</td><td>35.0/10.0</td><td>33.8/5.0</td><td>26.7/6.8</td><td>26.9/10.0</td><td>17.2/4.1</td><td>26.9/7.5</td><td>26.3/6.9</td><td>29.2/7.5</td></tr><tr><td>Validity Reward</td><td>42.5/16.7</td><td>30.0/10.0</td><td>23.7/8.1</td><td>26.3/10.0</td><td>25.3/14.9</td><td>36.3/10.0</td><td>30.5/13.2</td><td>30.8/10.0</td></tr><tr><td>Binary Reward</td><td>35.8/10.0</td><td>33.8/10.0</td><td>35.8/8.1</td><td>36.3/17.5</td><td>28.4/13.5</td><td>40.6/15.0</td><td>33.3/10.5</td><td>36.9/14.2</td></tr><tr><td>Random Commit</td><td>34.2/3.3</td><td>27.5/10.0</td><td>41.6/16.2</td><td>40.6/15.0</td><td>24.0/8.1</td><td>44.4/12.5</td><td>33.2/9.2</td><td>37.5/12.5</td></tr><tr><td>ActionOPD</td><td>42.5/16.7</td><td>40.0/15.0</td><td>42.9/10.8</td><td>39.4/12.5</td><td>28.4/13.5</td><td>40.6/15.0</td><td>37.9/13.7</td><td>40.0/14.2</td></tr></table>

## 3.4 ABLATION STUDIES

Ablations shared the settings in Section 3.3, with N = 4 and K = 3. Validity Reward replaced the semantic training reward with action validity, while retaining semantic scores for action selection. Binary Reward removed the frequency bonus from Equation 6, using binary agreement for both training and action selection. Random Commit retained ActionOPD’s reward but selected uniformly from all candidates, including invalid or zero-reward actions. Appendix B.4 details the variants and their effects on trainable state groups.

Table 3 reports results on the training set and 100 held-out tasks. ActionOPD outperforms all three variants in average single-trial success on both splits, supporting the combined design. The largest drop occurs with Validity Reward, which assigns identical rewards to all valid candidates and thus provides no learning signal in all-valid groups. Semantic rewards can instead distinguish these candidates by their agreement with teacher actions, highlighting the value of supervising action meaning beyond protocol compliance.

## 3.5 PATTERN ANALYSIS

We compared Qwen3-4B Base with the same student after 100 ActionOPD steps using DeepSeek-V4-Flash as teacher. Each selected case has at least two additional successes after training over the first four trials. Table 4 summarizes one Base-failure/ActionOPD-success pair matched by task and trial index in each Tau domain, using replies, tool calls, and environment results.

In these cases, the student respected policy constraints, distinguished information sources from operation targets, and followed through on promised actions. These changes highlight the value of ActionOPD’s semantic action supervision for task-appropriate behavior.

Table 4: Selected positive cases: Qwen3-4B before/after 100 ActionOPD steps with DeepSeek-V4- Flash. Fractions: successes in the first four trials (Base → ActionOPD). Each pair contrasts Base failure with ActionOPD success on the same task and trial. Outcomes denote evaluation results, not training rewards.
<table><tr><td>Pattern / Task Context</td><td>Qwen3-4B Base</td><td>Qwen3-4B + ActionOPD</td></tr><tr><td>Following Policy Constraints Airline  $( 2 / 4  4 / 4 )$  . The user requests a flight change, but basic economy bookings cannot be modified.</td><td>Reads that the booking is basic economy, yet modifies the flight through the API, violating the policy. Outcome:Failure.</td><td>Explains the restriction and calls the transfer tool when the user requests escalation. Success means keeping the booking unchanged, not rebooking. Prohibited modifications:  $2 / 4  0 / 4 .$  Outcome: Success.</td></tr><tr><td>Tracking the Correct Target Retail  $( 0 / 4  3 / 4 )$  . Change the target order&#x27;s address and speaker. Retrieve the new address in New York from another order.</td><td>Treats the source order&#x27;s New York address as if it already belonged to the target. Changes only the speaker. The target&#x27;s address remains in Los Angeles. Outcome:Failure.</td><td>Retrieves the source address, then updates the target&#x27;s address followed by its speaker. The returned order contains both the New York address and the requested green speaker. Outcome:Success.</td></tr><tr><td>Completing and Verifying Actions Telecom  $( 0 / 4  2 / 4 )$  Restore service after an overdue-bill suspension and an unseated SIM. Payment alone does not reactivate the line.</td><td>Promises restoration after payment but never calls resume_line. The user pays and reboots, but service remains disconnected and the agent transfers the case. Outcome:Failure.</td><td>Recovers from an invalid tool call, resumes the line with resume_line, and guides the user to reseat the SIM. User-side tool output confirms connection, and the evaluator verifies that the debt is cleared. Outcome: Success</td></tr></table>

Table 5: Self-ActionOPD on Tau: $\mathrm { p a s s ^ { 1 } / p a s s ^ { 4 } }$ (%), using Qwen3-4B. Privileged Teacher is the ini tial Base model given privileged context. Self-ActionOPD is evaluated at step 50 using public context only. Four trials per task. Average equally weights domains within each split. Bold/underlined: best/second-best of Base and Self-ActionOPD; ties share marks; privileged reference excluded.
<table><tr><td rowspan="2">Model</td><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Telecom</td><td colspan="2">Average</td></tr><tr><td>Train (30)</td><td>Test (20)</td><td>Train (74)</td><td>Test (40)</td><td>Train (74)</td><td>Test (40)</td><td>Train</td><td>Test</td></tr><tr><td>Privileged Teacher 51.7/33.3</td><td></td><td>35.0/15.0</td><td>38.2/6.8</td><td>38.1/17.5</td><td>20.6/8.1</td><td>37.5/20.0</td><td>36.8/16.1</td><td>36.9/17.5</td></tr><tr><td>Base</td><td>35.0/10.0</td><td>33.8/5.0</td><td>26.7/6.8</td><td>26.9/10.0</td><td>17.2/4.1</td><td>26.9/7.5</td><td>26.3/6.9</td><td>29.2/7.5</td></tr><tr><td>Self-ActionOPD</td><td>44.2/26.7</td><td>35.0/10.0</td><td>35.1/8.1</td><td>35.6/10.0</td><td>19.3/5.4</td><td>31.3/10.0</td><td>32.9/13.4</td><td>34.0/10.0</td></tr></table>

## 3.6 SELF-ACTIONOPD

We further examine whether a model can provide useful action-level supervision for itself when given privileged task information. Self-ActionOPD shared current Qwen3-4B weights between teacher and student, as in OPSD (Zhao et al., 2026), but independently sampled their responses. Teacher responses were refreshed from the updated shared model at each collection step and held fixed during subsequent policy optimization. The student received only public interaction context, whereas the teacher additionally received action guidance, summaries of private user goals and constraints, and reference actions and target outcomes. Training used action-agreement rewards on student-generated candidates only. Appendix B.5 provides implementation details.

After 50 steps, Self-ActionOPD improved test average pass<sup>1</sup>/pass<sup>4</sup> over Base by 4.8/2.5 percentage points (Table 5). Single-trial success also improved across all three domains. Unlike the privilegedteacher reference, the trained student was evaluated using only public context, showing that the student retained benefits from the privileged guidance after training. Together, these results support the feasibility of using a same-model teacher given privileged context for action-level distillation.

## 4 RELATED WORK

Distillation and self-distillation. Classical distillation transfers predictive distributions (Hinton et al., 2015), while sequence distillation imitates generated outputs (Kim & Rush, 2016). Offline imitation uses token-level losses on fixed demonstrations that may not cover student-induced prefixes. GKD and MiniLLM instead supervise student-generated sequences through token-level divergences (Agarwal et al., 2024; Gu et al., 2024). These objectives require teacher distributions or sampled-token log-probabilities. Guidance can become unreliable under prefix drift or specialtoken mismatch (Fu et al., 2026). Cross-tokenizer OPD transfers probabilistic supervision through explicit alignment (Niu et al., 2026; Wang et al., 2026a). Privileged self-distillation augments teacher context: On-Policy Self-Distillation (OPSD) minimizes token-level divergence on student rollout using a shared-weight teacher conditioned on reference solutions (Zhao et al., 2026). SDPO distills feedback-conditioned predictions (Hubotter et al., 2026), while privileged-information distillation¨ studies shared-parameter training, including on-policy reverse KL (Penaloza et al., 2026). Self-ActionOPD instead uses sampled-action agreement.

Black-box and semantic distillation. GAD jointly trains a discriminator to reward teacher-like responses (Ye et al., 2025), whereas ROPD evaluates responses against rubrics derived from teacher– student contrasts (Fang et al., 2026). On-policy Verbal Distillation (OVD) uses verbal teacher scores to evaluate reasoning quality at step and trajectory levels (Xiong et al., 2026). OmniOPD compares selected student reasoning chunks with teacher continuations sampled at student prefixes (Zhou et al., 2026). MENTOR rewards trajectory-wide tool-set agreement, final correctness, and execution validity (Choi et al., 2025). Action Semantics Learning (ASL) derives a semantic-agreement policy gradient and uses a world model to compare UI transitions induced by generated and reference actions. Its RL variant combines teacher-action agreement with environment rewards (Tang et al., 2026). ActionOPD instead matches immediate tool calls or communicative responses against teacher samples at the same state. Its matcher is instructed to assess agreement, not correctness. ActionOPD requires no reasoning-text matching, complete reference trajectories, or terminal rewards.

Interactive agent learning and process supervision. DAgger queries experts at learner-visited states (Ross et al., 2011). SCoRe provides local teacher corrections before supervised and shorthorizon RL refinement (Lyu et al., 2026). Guided-OPD gradually withdraws teacher turns through a rollout curriculum; ATOD combines OPD and RL with annealed coefficients and turn-level weighting (Li et al., 2026; Tan et al., 2026). GiGPO retrospectively groups repeated states across trajectories and derives relative advantages from returns (Feng et al., 2025). ActionOPD scores directly sampled same-state candidates against a shared teacher multiset, without estimating returns.

## 5 CONCLUSION AND LIMITATIONS

ActionOPD decouples teacher decision supervision from student token realization through semantic agreement at student-encountered states, requiring only generation access to the teacher. Across tested configurations, it improves average performance on unseen benchmarks, with higher final averages from larger students and stronger teachers. ActionOPD also achieves competitive performance against offline distillation and token-level OPD. Ablations support semantic rewards over validity rewards. Self-ActionOPD improves held-out performance using a same-model teacher with privileged context, highlighting a potential path toward agent self-improvement. Together, these findings support semantic actions as a unit of supervision for agent distillation.

Three limitations warrant attention. First, agreement does not guarantee correctness: teacher errors or suboptimal decisions can become learning targets. Second, LLM matching of communicative responses and tool-argument differences can produce false positives or negatives, misassigning rewards. Third, fixed-size teacher and student sample sets may contain few semantically matching actions when plausible decisions are dispersed, as may occur in long-horizon interactions. Increasing K and N could improve coverage at greater generation and matching cost, but the benefit in this regime remains untested.

## AI USE STATEMENT

The authors developed the method and designed all experiments. Generative AI tools assisted with manuscript preparation, literature review, implementation, debugging, derivation checks, and result analysis, and provided feedback on methodological details. The use of language models for task screening, teacher supervision, semantic matching, and user simulation is documented in the experimental protocol. All final research decisions were made by the authors, who take responsibility for the manuscript and its findings.

## REPRODUCIBILITY STATEMENT

Appendices A and B document the derivations, algorithm, data preparation, and experimental protocols. Code, data, and model checkpoints for the core results will be made available at https://github.com/thu-nics/ActionOPD.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html.

Alibaba Cloud. Qwen3.7-Flash model documentation. Model Studio documentation, 2026. URL https://www.alibabacloud.com/help/en/model-studio/ qwen3-7-flash. Accessed September 24, 2026.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating conversational agents in a dual-control environment. arXiv:2506.07982, 2025. URL https://arxiv.org/abs/2506.07982.

ChangSu Choi, Hoyun Song, Dongyeon Kim, Minkyung Cho, WooHyeon Jung, Sunjin Park, NohHyeob Bae, Seona Yu, and KyungTae Lim. MENTOR: Reinforcement learning via flexible teacher-optimized rewards for tool-use distillation. arXiv:2510.18383v4, 2025. URL https: //arxiv.org/abs/2510.18383v4.

DeepSeek-AI. DeepSeek-V4-Flash. Model card, 2026. URL https://huggingface.co/ deepseek-ai/DeepSeek-V4-Flash. Accessed September 24, 2026.

Junfeng Fang, Zhepei Hong, Mao Zheng, Mingyang Song, Gengsheng Li, Houcheng Jiang, Dan Zhang, Haiyun Guo, Xiang Wang, and Tat-Seng Chua. Rubric-based on-policy distillation. arXiv:2605.07396v1, 2026. URL https://arxiv.org/abs/2605.07396v1.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv:2603.25562v2, 2026. URL https://arxiv.org/abs/2603.25562v2.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015. URL https://arxiv.org/abs/1503.02531.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation. arXiv:2601.20802v2, 2026. URL https://arxiv. org/abs/2601.20802v2.

Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327, 2016. doi: 10.18653/v1/D16-1139. URL https://aclanthology.org/D16-1139/.

Gengsheng Li, Mao Zheng, Mingyang Song, Ruiqi Liu, Tianyu Yang, Jie Sun, Qiyong Zhong, Haiyun Guo, Junfeng Fang, Dan Zhang, and Jinqiao Wang. On-policy distillation with curriculum turn-level guidance for multi-turn agents. arXiv:2606.15912v2, 2026. URL https://arxiv. org/abs/2606.15912v2.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1160–1183, 2025. doi: 10.18653/v1/2025.findings-naacl.65. URL https://aclanthology.org/2025. findings-naacl.65/.

Yuanjie Lyu, Chengyu Wang, Jun Huang, and Tong Xu. Student-centered distillation narrows the agentic gap between small and large LLMs. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2509.14257v3.

Yifan Niu, Han Xiao, Dongyi Liu, Zelong Wang, Dihong Gong, Yasheng Wang, and Jia Li. Breaking the tokenizer barrier: On-policy distillation across model families. arXiv:2606.09456, 2026. URL https://arxiv.org/abs/2606.09456.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 48371–48392, 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv:2602.04942v3, 2026. URL https://arxiv.org/abs/2602.04942v3.

Qwen Team. Qwen3.5-9B. Model card, 2026a. URL https://huggingface.co/Qwen/ Qwen3.5-9B. Accessed September 24, 2026.

Qwen Team. Qwen3.8-27B. Model card, 2026b. URL https://huggingface.co/Qwen/ Qwen3.8-27B. Accessed September 24, 2026.

Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and´ structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15, pp. 627–635, 2011. URL https: //proceedings.mlr.press/v15/ross11a.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017. URL https://arxiv.org/abs/1707. 06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300, 2024. URL https: //arxiv.org/abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. EnvScaler: Scaling tool-interactive environments for LLM agent via programmatic synthesis. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 8326– 8357, 2026. doi: 10.18653/v1/2026.findings-acl.407. URL https://aclanthology.org/ 2026.findings-acl.407/.

Qitai Tan, Zefang Zong, Mo Li, Yipeng Shi, Yang Li, and Peng Chen. ATOD: Annealed turn-aware on-policy distillation for multi-turn agentic tasks. arXiv:2606.27814v6, 2026. URL https: //arxiv.org/abs/2606.27814v6.

Bohan Tang, Dezhao Luo, Jianheng Liu, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, and Kun Shao. Beyond syntax: Action semantics learning for app agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 9444–9454, 2026. URL https://openaccess.thecvf.com/content/ CVPR2026F/html/Tang\_Beyond\_Syntax\_Action\_Semantics\_Learning\_for\_ App\_Agents\_CVPRF\_2026\_paper.html.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16022– 16076, 2024. doi: 10.18653/v1/2024.acl-long.850. URL https://aclanthology.org/ 2024.acl-long.850/.

Hao Wang, Kun Yuan, Wenlin Zhong, Minglei Zhang, Han Xiao, Ming Sun, and Honggang Qi. Cross-tokenizer on-policy distillation via byte-prefix marginalization. arXiv:2607.22334, 2026a. URL https://arxiv.org/abs/2607.22334.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. Agent world model: Infinity synthetic environments for agentic reinforcement learning. In International Conference on Machine Learning, 2026b. URL https://arxiv.org/abs/ 2602.10090v3.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992. doi: 10.1007/BF00992696. URL https:// doi.org/10.1007/BF00992696.

Jing Xiong, Hui Shen, Shansan Gong, Yuxin Cheng, Jianghan Shen, Chaofan Tao, Haochen Tan, Haoli Bai, Lifeng Shang, and Ngai Wong. OVD: On-policy verbal distillation. arXiv:2601.21968, 2026. URL https://arxiv.org/abs/2601.21968.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv:2505.09388, 2025. URL https://arxiv.org/abs/ 2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=roNSXZpUDN.

Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang,

Qiang Fu, Wei Yang, and Lanxiao Huang. Mastering complex control in MOBA games with deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 6672–6679, 2020. doi: 10.1609/aaai.v34i04.6144. URL https://ojs.aaai. org/index.php/AAAI/article/view/6144.

Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-box on-policy distillation of large language models. arXiv:2511.10643v3, 2025. URL https://arxiv. org/abs/2511.10643v3.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv:2601.18734v3, 2026. URL https://arxiv.org/abs/2601.18734v3.

Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Bo Peng, Jiayi Liu, Xiangjun Fan, and Zhuokai Zhao. OmniOPD: Logit-free on-policy distillation via speculative verification. arXiv:2606.01476v2, 2026. URL https://arxiv.org/abs/2606.01476v2.

## A DERIVATIONS AND IMPLEMENTATION DETAILS

## A.1 FROM TEACHER-ACTION LIKELIHOOD TO REINFORCE

Assumptions and induced actions. Let $\mathcal { V } ( s )$ contain student response sequences, including termination tokens. At each state, semantic action classes partition these sequences: $\phi _ { s } ( y )$ maps equivalent realizations to the same action, and protocol-invalid outputs to an invalid outcome ⊥. The induced $\pi _ { \theta } ( a \mid s )$ is normalized over the complete action space, including ⊥. Teacher targets from $\pi _ { \mathrm { T } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ are valid actions. Summation over actions counts each semantic class once, not each expression of that class.

Assume a normalized, differentiable response policy on a fixed support, with a finite vocabulary and bounded response length. These conditions permit exchanging differentiation and summation. The semantic mapping $\phi _ { s } ,$ teacher distribution $\pi _ { \mathrm { T } }$ , and collected-state distribution $d _ { \mathrm { S } }$ remain fixed during each update. Teacher action samples suffice; teacher token probabilities are not needed.

Expanding the likelihood. For a teacher semantic action, all its student realizations contribute to its likelihood,

$$
\pi _ { \boldsymbol { \theta } } ( a ^ { \mathrm { { T } } } \mid s ) = \sum _ { y \in \mathcal { V } ( s ) } \pi _ { \boldsymbol { \theta } } ( y \mid s ) \mathbf { 1 } \{ \phi _ { s } ( y ) = a ^ { \mathrm { { T } } } \} .\tag{10}
$$

Consequently,

$$
\begin{array} { l } { { \displaystyle J _ { \mathrm { A c t i o n O P D } } ( \theta ) = \mathbb { E } _ { s \sim d _ { \mathrm { S } } } \sum _ { g ^ { \mathrm { T } } } \pi _ { \mathrm { T } } ( a ^ { \mathrm { T } } \mid s ) \sum _ { y } \pi _ { \theta } ( y \mid s ) { \bf 1 } \{ \phi _ { s } ( y ) = a ^ { \mathrm { T } } \} } } \\ { { \displaystyle \ = \mathbb { E } _ { s \sim d _ { \mathrm { S } } } \sum _ { y } \pi _ { \theta } ( y \mid s ) r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) ) . } } \end{array}\tag{11}
$$

Differentiating the student factor gives

$$
\begin{array} { r l } & { \nabla _ { \theta } J _ { \mathrm { A c t i o n O P D } } ( \theta ) = \mathbb { E } _ { s } \displaystyle \sum _ { y } r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) ) \nabla _ { \theta } \pi _ { \theta } ( y \mid s ) } \\ & { \quad \quad \quad = \mathbb { E } _ { s , y } [ r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) ) \nabla _ { \theta } \log \pi _ { \theta } ( y \mid s ) ] } \\ & { \quad \quad \quad = \mathbb { E } _ { s , a ^ { \mathrm { T } } , y } \left[ \mathbf { 1 } \{ \phi _ { s } ( y ) = a ^ { \mathrm { T } } \} \displaystyle \sum _ { \ell = 1 } ^ { | y | } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { \ell } \mid s , y _ { < \ell } ) \right] . } \end{array}\tag{12}
$$

This score-function estimator (Williams, 1992) uses semantic action agreement without evaluating the marginal $\pi _ { \theta } ( a ^ { \mathrm { ~ T ~ } } | \ s )$ explicitly. The likelihood and agreement objectives are identical under this semantic action definition.

An action-independent baseline $b ( s )$ leaves this gradient unchanged because $\mathbb { E } _ { y } [ \nabla _ { \theta } \log \pi _ { \theta } ( y \mid s ) ] =$ 0. A sample group mean instead includes each candidate’s own reward. For independent candidates and an otherwise unmodified estimator, subtracting that mean scales the expected gradient by $( N - 1 ) / N$ . Whitening, masking, token normalization, and clipping further distinguish the practical surrogate from the ideal gradient.

Expected likelihood versus expected log-likelihood. Where $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } ^ { \mathrm { T } } \mid \boldsymbol { s } ) > 0$ , replacing the expected likelihood in Equation 1 with $\mathbb { E } _ { s , a ^ { \mathrm { { T } } } } [ \log \pi _ { \theta } ( a ^ { \mathrm { { T } } } \mid s ) ]$ yields the following gradient:

$$
\mathbb { E } _ { s , a ^ { \mathrm { T } } , y } \left[ \frac { \mathbf { 1 } \{ \phi _ { s } ( y ) = a ^ { \mathrm { T } } \} } { \pi _ { \theta } ( a ^ { \mathrm { T } } \mid s ) } \nabla _ { \theta } \log \pi _ { \theta } ( y \mid s ) \right] .\tag{13}
$$

The inverse-likelihood factor is absent from ActionOPD. Expected agreement also need not preserve teacher modes: for disjoint actions A, B with teacher probabilities 0.7, 0.3, a student placing all mass on A maximizes the unregularized likelihood objective.

Algorithm 1 ActionOPD   
Require: Student $\pi _ { \theta } ,$ teacher π , matcher $m _ { s } ,$ , tasks, $\overline { { K = 3 , N = 4 , \lambda = 0 . 5 } }$   
1: for each training iteration do   
2: Freeze $\theta _ { \mathrm { o l d } }  \theta ;$ initialize state-group buffer $B  \emptyset$   
3: for each sampled task do   
4: Reset the environment and obtain public state s   
5: while task is active and interaction budget remains do   
6: Set group index $g \gets | \boldsymbol { B } | + 1$ and state $s _ { g } \gets s$   
7: Sample teacher responses $\mathcal { M } _ { \mathrm { T } } ( s _ { g } ) = [ \dot { y _ { 1 } ^ { \mathrm { T } } } , \dots , y _ { K } ^ { \mathrm { T } } ]$ using π<sub>T</sub>   
8: Sample $y _ { g , i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid s _ { g } )$ and retain parsed actions, for $i = 1 , \ldots , N$   
9: Compute $c ( s _ { g } , y _ { g , i } )$ and $\dot { r } _ { g , i } = r _ { \lambda } ( s _ { g } , y _ { g , i } )$ (Equation 5; Equation 6)   
10: Add the complete state group $( s _ { g } , \{ y _ { g , i } , r _ { g , i } \} _ { i = 1 } ^ { N } )$ to B   
11: Select $i ^ { \star } \in$ arg max $r _ { g , i } ;$ execute the parsed action in $y _ { g , i ^ { \star } }$ and update s   
12: end while   
13: end for   
14: Exclude groups without supervision or with equal rewards   
15: Center all N rewards per retained group and whiten across the batch (Equation 7)   
16: Broadcast advantages to generated tokens and update θ with the clipped objective (Equation 9)   
17: end for

## A.2 OBJECTIVE INDUCED BY THE FREQUENCY BONUS

Fix a state s, a nominal teacher sample count $K > 1$ , and the retained teacher-response multiset $\mathcal { M } = \mathcal { M } _ { \mathrm { T } } ( s )$ with $1 \leq | { \mathcal { M } } | \leq K$ . Write $\begin{array} { r } { c ( y ) = c ( s , y ) = \sum _ { k = 1 } ^ { | \mathcal { M } | } m _ { s } ( y , y _ { k } ^ { \mathrm { T } } ) } \end{array}$ , suppressing the fixed state and multiset. Let

$$
\begin{array} { r l } & { C _ { \theta } ( s ; \mathcal { M } ) = \mathbb { E } _ { y \sim \pi _ { \theta } } [ c ( y ) ] , } \\ & { U _ { \theta } ( s ; \mathcal { M } ) = \mathbb { P } _ { y \sim \pi _ { \theta } } [ c ( y ) > 0 ] , } \\ & { \quad \quad V _ { \theta } ( s ) = \mathbb { P } _ { y \sim \pi _ { \theta } } [ y \mathrm { c o n t a i n s ~ a n ~ i n v a l i d ~ a c t i o n } ] . } \end{array}\tag{14}
$$

Here $C _ { \theta }$ is the expected match count, $U _ { \theta }$ the any-match probability, and $V _ { \theta }$ the invalid-action probability. With accurate matching, $\begin{array} { r } { C _ { \theta } = \sum _ { k } \pi _ { \theta } ( a _ { k } ^ { \mathrm { ~ T ~ } } | \ : s ) } \end{array}$ , but the decomposition below does not require this equality. Invalid actions have $c ( y ) = 0 . \ \mathrm { A n }$ y-match probability does not measure coverage of distinct teacher actions. Writing $\alpha = \lambda / ( K - 1 )$ , Equation 6 is exactly

$$
r _ { \lambda } ( s , y ) = \alpha c ( y ) + ( 1 - \alpha ) \mathbf { 1 } \{ c ( y ) > 0 \} - \mathbf { 1 } \{ y \mathrm { c o n t a i n s ~ a n ~ i n v a l i d ~ a c t i o n } \} .\tag{15}
$$

Its expected-reward objective is therefore

$$
J _ { \lambda } ( \theta \mid s , \mathcal { M } ) = \alpha C _ { \theta } ( s ; \mathcal { M } ) + ( 1 - \alpha ) U _ { \theta } ( s ; \mathcal { M } ) - V _ { \theta } ( s ) .\tag{16}
$$

The same score-function derivation applies to $r _ { \lambda }$ . For $0 \leq \lambda \leq K - 1$ , the positive terms combine expected match count and any-match probability with nonnegative weights. Beyond this range, the identity still holds but the any-match coefficient is negative. For $0 < \lambda < K - 1$ , each additional match contributes $\alpha < 1$ , whereas the first match earns one. This downweights additional matches relative to the first match, but does not imply reduced variance of the normalized policy update. This objective is conditional on the retained samples and does not calibrate teacher probabilities or confidence.

At $\lambda = 0 ,$ , valid actions receive binary agreement rewards. At $\lambda = K - 1$ , their reward is the raw match count $c ,$ while the invalid-action penalty remains −1. With $K = 3 , \lambda = 0 . 5 , \alpha = 0 . 2 5$ , and $[ A , A , B ] ,$ , rewards are $1 . 2 5 , 1 , 0 , - 1$ for an A match, a B match, a valid non-match, and an invalid action. One retained teacher action gives unit reward for a match and objective $U _ { \theta } - V _ { \theta }$ , as does an explicit $K = 1$ configuration. An empty multiset supplies no supervision, not negative labels.

## A.3 TEACHER SUPERVISION AND SEMANTIC MATCHING

Visible information and native tools. In standard ActionOPD, teacher and student receive the same retained public context and tool schemas through their native chat templates. Each teacher sample contains one immediate action. Hidden simulator state, verifier code, task solutions, and private user-simulator instructions are excluded. Self-ActionOPD adds the inputs in Appendix B.5. Tools are exposed directly, without list tools/call tool wrappers. Multiple student tool calls violate the single-action protocol; only the teacher’s first call is retained when its output contains several.

Teacher sampling and validation. The teacher multiset is fixed before student generation. Each of three requested samples permits two retries for parser/schema-invalid output; valid samples are retained without regeneration. This filters protocol validity, not task success. A nonempty partial multiset can supervise a group, retaining nominal $K = 3$ in the bonus denominator. With two retained copies of A, a matching student response receives 1.25 at $\lambda = 0 . 5$ . With no valid samples, collection ends without a student group or policy-failure label. Retries and partial retention differ from independent sampling in Section 2.2.

Caching. Teacher-cache keys include task/state context, tools, teacher/provider, prompt and protocol versions, decoding settings, and history-dependent state. Retained samples preserve multiplicity. Reuse does not provide fresh independent draws. Matcher-cache keys include the action pair, evidence, model, and prompt.

Context management. Native chat templates retain fixed task/system content and complete tool schemas, dropping the oldest removable exchanges until the prompt fits its token budget. The standard teacher uses the same retained public context; Self-ActionOPD has an additional teacher-only prompt allowance (Appendix B.5). If protected content cannot fit, the rollout terminates without truncating tool definitions or fabricating an action. Prompt construction leaves the environment’s full interaction history intact.

Semantic matching. The implemented $m _ { s } ( y , y ^ { \mathrm { T } } )$ compares retained tool-call fields or message text, not reasoning. It approximates $\mathbf { 1 } \{ \phi _ { s } ( y ) \stackrel {  } { = } \phi _ { s } ( \stackrel { \cdot } { y } ^ { \mathrm { T } } ) \}$ . Approximation errors can produce inconsistent judgments for semantically equivalent responses. Thus counts and rewards depend on sampled responses, not only their semantic classes. For fixed y and a complete multiset of K independent teacher draws, $c / K$ is unbiased for $\mathbb { E } _ { y ^ { \mathrm { T } } } [ m _ { s } ( y , y ^ { \mathrm { T } } ) ]$ . Accurate matching recovers $r _ { \mathrm { T } } ( s , \phi _ { s } ( y ) )$ . With any fixed matcher, the score-function identity still applies to its expected reward, but errors can make that objective differ from $J _ { \mathrm { A c t i o n O P D } }$

The matcher has fixed parameters and prompts. It is instructed to assess agreement in the immediate operation or communicative act, including facts, identifiers, numbers, negation, commitments, and exact-copy requirements. Shared usefulness or outcomes do not establish equivalence. Ambiguous judgments are negative; API or response-format failures mean unavailable supervision, not negative matches.

Tool calls must name the same tool. Canonical checks ignore dictionary-key order and normalize integral floating-point values, but distinguish booleans from numbers. Canonicalization generally preserves distinctions between missing and null values, as well as list order and multiplicity; toolsource evidence can justify specific defaults or normalization. Remaining differences are judged using the tool schema, both argument objects, differing paths, public context, and tool-implementation evidence. Every teacher sample contributes a separate binary match, even when identical pairs reuse cached judgments. No transitive closure is applied.

Tool-implementation evidence is available to the matcher, not the teacher policy. It supports argument interpretation, not inspection of hidden database contents, task answers, or action quality. Two calls do not match merely because both fail or produce no change. Missing required source evidence makes supervision unavailable. Message matching receives both messages, public context, and tools, but no hidden solution. Normalized exact matches bypass the LLM. Cross-kind pairs do not match, but tool, message, and invalid candidates remain in the same group.

## A.4 OPTIMIZATION AND EXECUTION DETAILS

Active groups and tokens. Training retains complete groups with available supervision and a reward range greater than $1 0 ^ { - 8 }$ . All N responses in each retained group contribute to reward centering, including invalid actions with reward −1. Centered rewards are divided by their population standard deviation across all retained groups, with $\epsilon = 1 0 ^ { - 8 }$ . No additional mean subtraction is needed. A positive-reward action can have negative advantage, and an all-matching group remains informative if match counts differ. An empty active batch skips optimization.

Advantages apply to all active student-generated tokens, including reasoning, but not to prompts, tool observations, or user-simulator outputs. The actor uses active-token mean aggregation. Padding rows serve computational alignment and contribute neither statistics nor gradients. Compact execution skips masked training rows while retaining rollout records and metrics, without introducing within-group candidate selection. Unselected responses in retained groups still contribute to training.

Clipped optimization. Let $h ( \rho , A ) = \operatorname* { m i n } \{ \rho A , \mathrm { c l i p } ( \rho , 1 - \eta , 1 + \eta ) A \}$ . The implementation additionally uses dual clipping (Ye et al., 2020),

$$
h _ { \mathrm { d u a l } } ( \rho , A ) = \left\{ { \operatorname* { m a x } \{ h ( \rho , A ) , c _ { \mathrm { d u a l } } A \} } , \quad A < 0 , \qquad \eta = 0 . 2 , \quad c _ { \mathrm { d u a l } } = 3 . \right.\tag{17}
$$

The actor maximizes the mean of $h _ { \mathrm { d u a l } }$ over active tokens. Teacher and matcher outputs, old-policy scores, and advantages remain fixed during the update. No reference-policy KL penalty or entropy bonus is added.

Sampling and asynchronous collection. Table 8 specifies student and teacher generation settings. In Equation 8, old and updated π scores use the full softmax at the rollout temperature, without topp/top-k truncation. Their ratio does not correct for truncation during candidate sampling.

Workers overlap teacher sampling and tool/API waiting with generation at other ready states. Within each group, teacher samples are fixed before student generation. Collection uses one behavior-policy snapshot, followed by optimization. Algorithm 1 specifies these dependencies without requiring sequential workers.

## B EXPERIMENTAL DETAILS

## B.1 TRAINING DATA AND TASK SELECTION

Table 6 records the transfer-training pools. AWM context selection retained tasks whose initial native prompt, including tool schemas, contained at most 16,000 tokens under the Qwen3-4B tokenizer. This selection cutoff is distinct from the training context limit. A deterministic audit checked scenario, database, and verifier integrity. Static feasibility screening then assessed each task using its tools, implementation, initial data, and verification evidence. It tested whether the environment supported the requested goal and success conditions, not whether a particular policy could succeed. EnvScaler used the same feasibility criterion after deterministic checks. Judge confidence was diag nostic, not an admission threshold.

Static feasibility screening for both data sources used DeepSeek-V4-Flash. Manifests link retained tasks to source data and screening records. Admission did not require expert rollout success. Static screening does not rule out runtime defects.

Each 100-step transfer run contains 5,900 AWM and 500 EnvScaler task starts, potentially revisiting tasks from the 6,697-task pool. Tau comparisons, ablations, and Self-ActionOPD use the official 178-task training and 100-task test splits.

Table 6: Task-selection funnel for the pools used in this study. Task counts denote unique tasks remaining after each stage, not rollout attempts. The final row reports environment coverage. Context selection is an AWM-specific stage. The dash denotes that no separate corresponding stage was used for EnvScaler.
<table><tr><td>Stage</td><td>AWM</td><td>EnvScaler</td></tr><tr><td>Source tasks</td><td>10,000</td><td>2,550</td></tr><tr><td>Context-eligible</td><td>9,380</td><td></td></tr><tr><td>Deterministic integrity pass</td><td>7,562</td><td>2,495</td></tr><tr><td>Static feasibility: healthy</td><td>5,942</td><td>755</td></tr><tr><td>Environments represented in the final pool</td><td>794</td><td>49</td></tr></table>

Table 7: Standard ActionOPD and ablation training configurations. Each configuration corresponds to one training run. The number of samples refers to responses at one state, not independent complete trajectories.
<table><tr><td>Setting</td><td>Transfer experiment</td><td>Tau ablations</td></tr><tr><td>Student</td><td>Qwen3-4B or Qwen3-8B</td><td>Qwen3-4B</td></tr><tr><td>Action teacher</td><td>Qwen3.7-Flash or DeepSeek-V4-Flash</td><td>Qwen3-32B</td></tr><tr><td>Training tasks</td><td>AWM/EnvScaler static-feasibility</td><td>Official train splits</td></tr><tr><td>Collection steps reported</td><td>pools 100</td><td>50</td></tr><tr><td>Task starts per step</td><td>64: 59 AWM + 5 EnvScaler</td><td>24: 8 per Tau domain</td></tr><tr><td>Student samples N</td><td>4</td><td>4</td></tr><tr><td>Teacher samples K</td><td>3</td><td>3</td></tr><tr><td>Frequency bonus scale λ</td><td>0.5</td><td>0.5 for ActionOPD/random commit; 0 for binary reward</td></tr><tr><td>Maximum agent decisions</td><td>20 AWM; 40 EnvScaler</td><td>20</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate / schedule</td><td>10−⁶ / constant, no warmup</td><td>10−⁶ / constant, no warmup</td></tr><tr><td>PPO epochs / minibatch size</td><td>1/32</td><td>1/ 32</td></tr><tr><td>PPO clip / dual clip</td><td>0.2 / 3</td><td>0.2 / 3</td></tr><tr><td>Advantage normalization</td><td>Group centering, batch whitening</td><td>Group centering, batch whitening</td></tr><tr><td>Loss aggregation KL penalty / entropy bonus</td><td>Active-token mean Neither</td><td>Active-token mean</td></tr><tr><td></td><td></td><td>Neither</td></tr><tr><td>Outcome reward in loss</td><td>No</td><td>No</td></tr></table>

## B.2 TRAINING CONFIGURATIONS AND AUXILIARY MODELS

Tables 7 and 8 summarize training and generation settings. Teacher/matcher inputs follow Appendix A.3. User simulators generate interaction turns, not distillation targets. Our implementation extends verl-agent (Feng et al., 2025), built on veRL/HybridFlow (Sheng et al., 2025).

One collection step can involve multiple optimizer updates: one PPO epoch traverses the collected candidates in minibatches rather than updating once per task batch. Tau’s 20-decision training limit is distinct from the native environment’s 200-internal-transition budget. KD, OPD, and Self-ActionOPD share ActionOPD’s optimizer, learning-rate, update, and sampling settings, but retain their respective objectives.

Main-experiment matcher and EnvScaler user-simulator settings are shared across all four student– teacher configurations. Tau training instead uses Qwen3.8-27B (Qwen Team, 2026b) for matching and user simulation; Tau evaluation uses Qwen3.5-9B (Qwen Team, 2026a). Self-ActionOPD’s private user summaries were prepared offline with DeepSeek-V4-Flash and reviewed, so auxiliary models remain part of its training setup.

In Table 8, T, p, and k denote temperature, top-p, and top-k, respectively. Response limits are in tokens, with 1k denoting 1,024 tokens.

## B.3 EVALUATION PROTOCOL AND SCORE DEFINITIONS

BFCL and ToolSandbox. Students and directly evaluated teachers use the same task subsets and interaction limits. BFCL includes 800 multi-turn tasks, 200 per category, with one trial and the native per-turn step limit of 20. ToolSandbox uses 490 strictly local scenarios, four scheduled trials each, and at most 30 messages. Similarity is averaged over all 1,960 scheduled trials.

AppWorld. AppWorld evaluation covers 168 Normal and 417 Challenge tasks for students and directly evaluated teachers, with one trial, 50 steps, and a 100,000-output-token budget per task. TGC is the fraction passing all task-goal tests; SGC measures completion of all variants within a

Table 8: Model roles and generation settings.
<table><tr><td colspan="5">Generation Settings</td></tr><tr><td>Experiment</td><td>Role and Model</td><td>Thinking</td><td>Sampling</td><td>Maximum Response</td></tr><tr><td rowspan="5">AWM/ EnvScaler Training</td><td>Student: Qwen3-4B/8B</td><td>Enabled</td><td> $\begin{array} { l c l } { { T = 0 . 6 ; p = 0 . 9 5 ; } } \\ { { k = 2 0 } } \end{array}$ </td><td>4k</td></tr><tr><td>Teacher: DeepSeek-V4-Flash</td><td>Enabled; effort max</td><td></td><td>4k</td></tr><tr><td>Teacher: Qwen3.7-Flash</td><td>Enabled</td><td> $T = 0 . 6 ; p = 0 . 9 5$ </td><td>8k</td></tr><tr><td>Matcher: DeepSeek-V4-Flash</td><td>Disabled</td><td>T = 0</td><td>128</td></tr><tr><td>User: DeepSeek-V4-Flash</td><td>Disabled</td><td>T = 1</td><td></td></tr><tr><td rowspan="6">Tau/Self- ActionOPD Training</td><td>Student: Qwen3-4B</td><td>Enabled</td><td> $\begin{array} { l c l } { { T = 0 . 6 ; p = 0 . 9 5 ; } } \\ { { k = 2 0 } } \end{array}$ </td><td>4k</td></tr><tr><td>Teacher: Qwen3-32B</td><td>Enabled</td><td> $\begin{array} { l c l } { { T = 0 . 6 ; p = 0 . 9 5 ; } } \\ { { k = 2 0 } } \end{array}$ </td><td>8k</td></tr><tr><td>Self Teacher: Qwen3-4B</td><td>Enabled</td><td> $\begin{array} { l c l } { { T = 0 . 6 ; p = 0 . 9 5 ; } } \\ { { k = 2 0 } } \end{array}$ </td><td>4k</td></tr><tr><td>Matcher: Qwen3.8-27B</td><td>Disabled</td><td> $\begin{array} { l c l l l } { T = 0 . 7 ; p = 0 . 8 ; } \\ { k = 2 0 } \end{array}$ </td><td>32k</td></tr><tr><td>User: Qwen3.8-27B</td><td>Disabled</td><td> $T = 0 . 7 ; p = 0 . 8 ;$  k = 20</td><td>8k</td></tr><tr><td>Student: Qwen3-4B/8B</td><td>Enabled</td><td>Greedy (T = 0)</td><td></td></tr><tr><td rowspan="4">Evaluation</td><td>Teacher: Qwen3.7-Flash</td><td>Enabled</td><td>T = 0.6; p = 0.95</td><td>4k</td></tr><tr><td>Teacher: DeepSeek-V4-Flash</td><td>Enabled; effort</td><td></td><td>32k 32k</td></tr><tr><td></td><td>max</td><td></td><td></td></tr><tr><td>Tau User: Qwen3.5-9B</td><td>Enabled</td><td>T = 1</td><td>8k</td></tr></table>

Tau transfer evaluation. Transfer evaluation covers all 50 Airline, 114 Retail, and 114 Telecom tasks in the official base splits, including the subsets used for Tau train/test comparisons. Transfer students were trained on AWM/EnvScaler, not these tasks. Evaluation uses native function calling and the generation settings in Table 8. All models use four trials per task. For the six student models in Table 1, scores use trials 0–3 from the historical evaluations, without outcome-based trial selection.

scenario. Normal and Challenge scores are reported separately. The average uses TGC only. Shared task and interaction budgets do not imply matched generation budgets (Table 8).

All Tau evaluations in this paper disable natural-language assertions, including directly evaluated teachers, method comparisons, ablations, Self-ActionOPD, and the Privileged Teacher reference. Tau’s evaluator supplies success labels under this scoring configuration, not the semantic matcher. Success therefore does not certify every utterance.

Tau method comparison, ablation, and Self-ActionOPD evaluation. Each model has four trials per task. Repeated Base and ActionOPD entries reuse the same results, not independent runs. Base scores partition the same trials 0–3 by official split task IDs. Airline, Retail, and Telecom train/test splits contain 30/20, 74/40, and 74/40 tasks. Training-set evaluation measures in-domain fit, not held-out generalization. Its episodes are distinct from training rollouts. Ordinary student evaluation excludes reference actions and private task criteria. The Privileged Teacher instead receives additional information and is not a same-input reference.

Success and reliability. Following Tau’s repeated-success metric (Yao et al., 2025), for task j with n scheduled trials and $c _ { j }$ successes, we compute

$$
\widehat { p } _ { j } ^ { 1 } = \frac { c _ { j } } { n } , \qquad \widehat { p } _ { j } ^ { 4 } = \frac { { \binom { c _ { j } } { 4 } } } { { \binom { n } { 4 } } } ,\tag{18}
$$

where $\binom { c _ { j } } { 4 } = 0$ when $c _ { j } < 4$ . Each domain score averages these values over its tasks. With $n = 4$ domain $\mathrm { \ p a s s ^ { 4 } }$ is the fraction of tasks successful in all four trials.

For the main table, let B be BFCL accuracy, S ToolSandbox similarity, $G _ { \mathrm { N } }$ and $G _ { \mathrm { C } }$ AppWorld Normal and Challenge TGC, and $P _ { d }$ Tau pass<sup>1</sup> in domain d. On a common percentage scale,

$$
\begin{array} { r l } & { G = \frac { 1 } { 2 } ( G _ { \mathrm { N } } + G _ { \mathrm { C } } ) , } \\ & { P = \frac { 1 } { 3 } ( P _ { \mathrm { A i r l i n e } } + P _ { \mathrm { R e t a i l } } + P _ { \mathrm { T e l e c o m } } ) , } \\ & { { \mathrm { A v g } } = \frac { 1 } { 4 } ( B + S + G + P ) . } \end{array}\tag{19}
$$

This study-specific summary weights benchmark families equally, without counting AppWorld SGC or Tau reliability again. Scores use one-decimal half-up rounding. Average gains are computed before display rounding, so subtracting displayed values can differ by 0.1 percentage points. Tauonly tables instead average the three domains separately for each metric and split; train and test are never pooled. Each configuration was trained once; repeated evaluation trials measure interaction variability, not variation across independent training runs.

## B.4 TAU BASELINES, ABLATIONS, AND COLLECTION BUDGETS

All configurations use the same 178-task Tau training set and Qwen3-4B initialization. KD, OPD, and standard ActionOPD use Qwen3-32B as action teacher. Each step starts eight tasks per domain, and Tables 2 and 3 report step 50. Their ActionOPD rows reuse one run.

ActionOPD. The reference configuration uses Equation 6 with $K = 3 , \lambda = 0 . 5$ , and uniform tie-breaking among maximum-reward candidates.

Validity Reward. Training assigns +1 to valid actions and −1 to invalid output or transferprotocol violations. Teacher sampling and semantic matching remain active for ActionOPD’s commitment rule, not training rewards. All-valid groups are masked as equal-reward groups, so this variant changes the number and composition of trainable groups as well as removing semantic reward distinctions.

Binary Reward. With the same three teacher samples, λ = 0 gives +1 for any match, zero for valid non-matches, and −1 for invalid actions. This reward also governs commitment, changing both learning signals and ties relative to ActionOPD.

Random Commit. Rewards follow ActionOPD, but commitment is uniform over all four candidates, including invalid and zero-reward actions. There is no success-based resampling. All candi dates in retained groups contribute to training.

A transfer notification without the preceding transfer call is treated as invalid output with reward −1. Equal-reward groups produce no learning signal.

KD: successful teacher-trajectory imitation. For each task start, Qwen3-32B independently generated four complete trajectories; only successful ones entered SFT. Following output imitation (Kim & Rush, 2016), loss covers all teacher-generated reasoning and action/message tokens, while user turns and tool observations provide context only. Thus outcome labels filter demonstra tions, but do not supply a policy-gradient reward.

OPD: sampled-token reverse-KL. OPD collected four student trajectories per task start. The teacher scored student-sampled tokens at their corresponding prefixes for sampled-token reverse KL supervision (Fu et al., 2026). It requires teacher log-probabilities and a compatible token space, but not full-vocabulary distributions.

Collection budgets. KD and OPD collect four trajectories per task start; ActionOPD samples four candidates per state and executes one, while all candidates in retained groups contribute to training. Trajectory lengths, teacher generation/scoring, matching costs, and KD filtering differ. Equal task starts and collection steps therefore do not equalize retained examples, tokens, teacher queries, or compute.

## B.5 SELF-ACTIONOPD: TEACHER INPUTS AND PARAMETER UPDATES

Current-weight teacher, fixed update targets. Self-ActionOPD used Qwen3-4B in both roles, with eight task starts per Tau domain per step. We report results at step 50. Four student candidates and three teacher samples were independently generated from the same current parameters. Teacher samples and matching results remained fixed during optimization. Subsequent collection used updated shared parameters. Self-teacher caches were cleared at each collection step.

Table 8 specifies generation settings. During Self-ActionOPD training, both roles used a 32,768- token context window and a 4,096-token response limit. Student and teacher prompts were capped at 24,576 and 28,672 tokens, respectively, providing an additional 4,096 tokens for teacher-only inputs. Teacher outputs and private prompt tokens were excluded from the student loss. Only student-generated candidate tokens received the action-level signal.

Teacher guidance and privileged information. The student received the domain policy, available tools, and observed interaction history. Additional teacher guidance covered immediate actions, preconditions, authorization, clarification, and avoiding repetition. Telecom device operations were rendered as user-facing instructions rather than user-tool syntax. This conversion added no private facts.

Privileged task information comprised reviewed summaries of private goals, constraints, preferences, and known/unknown information, including facts not yet disclosed in dialogue. It also supplied reference tool names and arguments, customer-side operations, target states, and required messages from evaluation criteria. These inputs did not replace required public confirmation or execution evidence. They were withheld from the student and matcher, although they could affect which student actions were rewarded. The trained student was evaluated without privileged inputs.