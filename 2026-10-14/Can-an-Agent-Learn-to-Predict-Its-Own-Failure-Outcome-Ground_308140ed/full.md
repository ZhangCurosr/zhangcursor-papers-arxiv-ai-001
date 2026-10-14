# Can an Agent Learn to Predict Its Own Failure? Outcome-Grounded Confidence Calibration in LLM Agents

Ashutosh Hathidara

SAP

ashutosh.hathidara@sap.com

## Abstract

Controlling an acting LLM agent means deciding, at each step of a long-horizon task, whether to trust its next action, abstain, or hand off to a human. That decision needs a per-step confidence signal predicting whether the current rollout will eventually succeed. The agent’s own self-reported verbalized confidence is the natural candidate, but existing reinforcement learning methods calibrate it against the cor rectness of a decision at the current step. We propose a method to calibrate it against the eventual outcome of a multi-step episode: under GRPO, the group of rollouts for a task supplies success labels, and a proper scoring rule scores each step’s confidence against the real ized outcome. In offline RL on recorded trajectories of a Qwen2.5-7B agent in ALFWorld, TextWorld, and WebShop environments, the AUROC of step confidence against episode outcome rises from near chance to 0.68–0.85, and the top-10% of steps the agent is most confident about succeed ∼4.3× more often than average. In online RL on the live ALFWorld agent, four calibration rewards improve early step AUROC, expected calibration error goes from 0.42 to below 0.12, and selective execution improves over both the untrained agent and a format-only RL control, without degrading task success. We also introduce an evaluation protocol that separates episode depth, task difficulty, and attempt-level foresight, which shows where the gain lies: the trained agent recognizes from its first step which tasks it will fail, while telling a succeeding attempt from a failing one at the same task is harder, with some early-window signal emerging on TextWorld.

## 1 Introduction

Acting LLM agents increasingly operate in longhorizon, partially observable environments (Shridhar et al., 2021; Yao et al., 2022; Côté et al., 2019; Shridhar et al., 2020; Hausknecht et al., 2020; Wang et al., 2022; Zhou et al., 2024; Liu et al.,

2024), where a single episode spans dozens of actions under compounding uncertainty. Controlling such an agent at runtime requires an estimate of the probability that the current attempt will succeed, so that it can continue with the action, verify with observation, or escalate to a human (Kadavath et al., 2022). Such a probability estimate can work as a confidence signal that predicts whether the current rollout will ultimately reach the goal, not whether the current step is locally correct.

The natural candidate is a self-reported verbalized confidence of an agent. Without additional focused training, smaller LLMs are poorly calibrated and barely predictive (§6). Previous works on Reinforcement Learning (RL) for calibration (Damani et al., 2025; Bani-Harouni et al., 2026; Wang et al., 2026b; Xuan et al., 2026; Zhang et al., 2026) focus on training confidence to predict either a single answer or step-level correctness. Such greedy optimization signals train the model to be locally predictive, promising no predictiveness for eventual outcome.

We propose a methodology to train a step-wise confidence signal that could predict the eventual outcome of the task. Under group-relative policy optimization (GRPO), the rollouts for a task supply success labels at no extra cost. We define the perstep reward as a proper scoring rule on the step confidence against the realized outcome, with other reward terms. We conduct offline & online RL training. Our evaluation protocol separates what the aggregate score mixes (§3.2).

Our approach reveals that outcome-grounded calibration can turn a weak agent’s confidence into a reliable early signal of eventual outcome, consistently across environments and training regimes. Our evaluation protocol then reveals precisely where this new ability comes from, and where it does not.

In summary, our contributions include: (i) Outcome-grounded calibration training under

GRPO, (ii) an evaluation protocol for episodic confidence, and (iii) an empirical analysis of what an LLM learns during such a training process.

## 2 Related Work

Measuring and eliciting confidence. A broad line of work reads a confidence out of a model without additional training: verbalized probabilities (Lin et al., 2022; Tian et al., 2023), self-consistency and self-evaluation (Kadavath et al., 2022), and elicitation strategies for black-box models (Xiong et al., 2024). A related line of work documents that post-hoc calibration of neural classifiers degrades with scale and training (Guo et al., 2017). These works primarily target synchronous single-answer correctness, where confidence degrades on weak models, precisely the area we target.

Reinforcement learning for confidence. The most closely related work trains calibration. RLCR (Damani et al., 2025) optimizes a Brier reward for verbalized confidence, and related efforts tie confidence to reward shaping, uncertainty, or abstention in non-agentic settings (Bani-Harouni et al., 2026; Xu et al., 2024; Zhang et al., 2026; Leng et al., 2025; Yaldiz et al., 2026; Li et al., 2026; Wang et al., 2026a). All optimize the calibration of a single synchronous output; none targets the eventual success of an agent through a long-horizon task.

Positioning against concurrent agentic methods. Three concurrent methods are closest. CAR/M-SCR (Xuan et al., 2026) and CAPO (Wang et al., 2026b) train confidence in tool-use and reasoning settings, and RLCR (Damani et al., 2025) in singleanswer QA; all optimize confidence against the synchronous correctness of the current answer or step, whereas we target the predictive success of the eventual rollout. They differ from us in how the signal is produced and rewarded: CAPO reads a perplexity-derived confidence and reshapes the advantage estimator through a consistency-and-regret construction, and MSCR uses a margin reward, while we train an explicit verbalized token under a lightweight per-completion calibration reward.

Selective prediction. Our governance mechanism is an instance of selective prediction: we evaluate it with risk-coverage curves (Geifman and El-Yaniv, 2017; Kamath et al., 2020) and train the signal with classical proper scoring rules (BRIER, 1950; Gneiting and Raftery, 2007). Closest in spirit, belief-uncertainty agents such as Singh et al. (2026)

estimate an agent’s internal belief rather than its outcome-predictive confidence; we distinguish the two in §4.

## 3 Problem Setup and Evaluation Protocol

We first state what a per-step confidence should predict (§3.1), then define the family of estimands that separates genuine foresight from the depth and difficulty confounds a pooled score mixes together (§3.2), and finally fix the calibration, selectiveexecution, and statistical conventions used throughout (§3.3).

## 3.1 Predictive Confidence Target

We model an acting agent as a partially observable decision process with policy π<sub>θ</sub>. At step t the policy observes history $h _ { t }$ (the prompt, prior actions, and environment feedback) and emits an action together with an inline confidence, $( a _ { t } , c _ { t } ) \sim$ $\pi _ { \theta } ( \cdot ~ | ~ h _ { t } )$ , where $c _ { t } ~ \in ~ [ 0 , 1 ]$ is parsed from a <conf>NN</conf> tag (an integer 0–100 rescaled to [0, 1]; a missing or malformed tag defaults to a neutral 0.5). A full rollout τ terminates and receives a binary success label $Y \in \{ 0 , 1 \} ( Y { = } 1$ iff the task is solved).

We define the predictive-calibration target: $c _ { t }$ should track

$$
c _ { t } \approx P ( Y = 1 \mid h _ { t } ) ,\tag{1}
$$

the probability that this episode ultimately succeeds given the history so far. We target eventual success rather than the synchronous P(current step correct) used by prior work. Long-horizon agentic tasks admit many valid solution paths, so multiple actions may be correct at each step; local correctness is thus ill-defined, and only the eventual outcome matters as a label.

## 3.2 Estimands

Task s has attempts $i = 1 , \ldots , G$ with outcomes $Y _ { s , i }$ and step confidences $c _ { s , i , t } .$ , every step inheriting its attempt’s outcome. A step’s confidence can rank correctly for three unrelated reasons: (i) the task is easy, (ii) the attempt is going well, or (iii) the episode is near its end. A single pooled metric conflates all three, so we report four estimands, each an AUROC against episode success—the probability that a step of a succeeding attempt is higher than a step of a failing one—differing only in which steps may be compared. We evaluate over all steps, the early window $( k \leq 2 ,$ , steps 0–2), and step 0 alone.

Pooled AUROC compares any success-step with any failure-step and holds none of the confounds fixed. Between-task AUROC collapses each task to its mean confidence, leaving only task difficulty (i). Within-task AUROC compares succeeding against failing attempts of the same task, cancelling difficulty to isolate attempt-level foresight (ii). Depthmatched within-task AUROC compares them only at the same step index, so neither difficulty nor depth contributes and only foresight survives (iii). Against each we report a depth-only model-free baseline scoring every step by −t: over all steps it can match or beat a real signal, but the early window and step 0 drive it to chance, so any gain there reflects foresight rather than position. At step 0 every attempt sees the same observation, so a gain can arise only from ranking tasks or from the first sampled action, before any environment feedback. All formal definitions appear in Appendix A.

## 3.3 Calibration & Selective Execution

We report two kinds of metrics. For calibration, how well confidence matches outcome frequency: Expected Calibration Error (ECE) with 10 equalwidth bins (Guo et al., 2017), and the Brier score (BRIER, 1950). For selective execution, how useful confidence is for gating actions: the success rate among the top-q% most confident steps in a window $( q = 1 0 , 2 5 )$ , and task-level triage (the success rate when only the top 50% of tasks by mean early confidence are attempted). Online intervals are 95% task-level cluster bootstraps. Offline intervals are 95% bootstraps over steps; steps of one trajectory share a label, so these intervals understate uncertainty. Paired deltas resample the tasks shared by both runs under comparison; within-task deltas use only the tasks that are mixed-outcome in both arms, a small subset, so these intervals rest on few clusters and are correspondingly wide. Tables 4 and 5 and the appendix report ∼100 paired deltas.

## 4 Methodology

Our method turns the rollouts that GRPO already samples into a training signal for confidence. The G attempts at a task each end in an outcome that labels every one of their steps, so a step can be scored by how well its stated confidence matches the attempt outcome it belongs to. We assemble this into three pieces: a group-relative advantage that credits a step for both the eventual outcome and the quality of its confidence (§4.1), a per-step reward built from a proper scoring rule that supplies the confidence term, and an optional spread term that keeps confidence away from the endpoints of the scale (§4.2). The same signal drives training on recorded offline trajectories and on the live agent alike (§4.3).

![](images/9206a675b6d5ac4806f5b1ecbb477e7debf607fa1550773695793d71db840b92.jpg)  
Figure 1: Outcome-grounded calibration under GRPO. The G attempts at a task end in outcomes Y that label every one of their steps. Each step’s confidence c is scored against that label by a proper scoring rule (Eq. 3), and the group-relative advantage credits the step for both the outcome and the quality of its confidence before updating π<sub>θ</sub>.

## 4.1 Group-Relative Training with Outcome

GRPO (Shao et al., 2024; Guo et al., 2025) samples G rollouts of one task and forms advantages relative to the group mean. In an episodic setting the realized outcomes $Y _ { i }$ of the group are the labels of Eq. (1) at no extra cost. Online, the advantage of every step is

$$
A _ { s , i , t } = \left( Y _ { s , i } - \bar { Y } _ { s } \right) + \left( r _ { s , i , t } - \bar { r } _ { s } \right) ,\tag{2}
$$

where $\begin{array} { r } { \bar { Y } _ { s } = \frac { 1 } { G } \sum _ { i } Y _ { s , i } } \end{array}$ is the group’s success rate, $\bar { r } _ { s }$ the group mean of the per-step rewards, and $r _ { s , i , t }$ is the per-step reward below. We also ablate two alternative optimization strategies against this GRPO objective (Section 6.1): RLOO (Ahmadian et al., 2024), a leave-one-out variant that baselines each rollout against the mean of the others rather than the full group, and PPO with a learned value head, a non-group-relative control that replaces the group baseline with a learned critic.

## 4.2 Per-Step Reward

The per-step reward is

$$
\boldsymbol { r } = \boldsymbol { w } _ { \mathrm { c a l } } \mathrm { c a l } ( c , Y ) + \boldsymbol { w } _ { \varphi } \varphi ( c ) + \boldsymbol { w } _ { \mathrm { f m t } } \boldsymbol { e } ,\tag{3}
$$

where $e \in \{ 0 , 1 \}$ is a format indicator that is 1 iff the step emits a well-formed <conf> tag (a missing or malformed tag sets $e = 0$ and defaults c to 0.5; §3.1), and $w _ { \mathrm { f m t } } = 0 . 2$ throughout. By default we instantiate the calibration term cal(·) of Eq. (3) with the Brier reward,

$$
\operatorname { c a l } ( c , Y ) = r _ { B } ( c , Y ) = 1 - ( c - Y ) ^ { 2 } ,\tag{4}
$$

which is strictly proper: with $p = P ( Y = 1 \mid h )$ $\operatorname { \mathbb { E } } [ r _ { B } \mid h ] = 1 - \operatorname { V a r } ( Y \mid h ) - ( c - p ) ^ { 2 }$ , maximized at $c = p$ (BRIER, 1950; Gneiting and Raftery, 2007). Any strictly proper rule shares this optimum, so the form should not matter; Section 6.1 tests this with other choices of cal(·): the logarithmic and spherical rules (proper), L1 (improper), a decision/selective reward, and a ranking surrogate. The ranking surrogate is a streaming soft-AUROC against moving averages of opposite-class confidence; it reads only the order of confidences (all forms are in Appendix I). Balanced Brier reweights wins and losses within a group so that both contribute equally, with weight $0 . 5 / \bar { Y } _ { s }$ on wins and $0 . 5 / ( 1 - \bar { Y } _ { s } )$ on losses.

The spread term $\varphi ( c ) = 4 c ( 1 - c )$ is concave, maximal at $\begin{array} { r } { c = \frac { 1 } { 2 } } \end{array}$ , and zero at the endpoints; with weight $w _ { \varphi }$ it is a regularizer that keeps confidence in the interior of the scale—preventing the calibration reward from collapsing it onto 0 or $1 { - } \mathrm { a t }$ some cost in ranking quality (§6.2). The dispersion variant $\varphi _ { \mathrm { d i s p } } ( c ) = \operatorname* { m i n } \bigr ( 1 , 4 ( c - \bar { c } _ { s } ) ^ { 2 } \bigl )$ , with $\bar { c } _ { s }$ the group’s mean confidence, replaces φ and so rewards spread around the group mean rather than toward $\begin{array} { r } { c = \frac { 1 } { 2 } } \end{array}$

## 4.3 Training Strategies

We use the standard distinction between offline RL, which trains on a fixed dataset of recorded trajectories without interacting with the environment, and online RL, which trains the agent on the trajectories it generates in the live environment (on-policy).

Offline Training. The offline route runs GRPO on (prefix → action + <conf>) rows cut from one recorded untrained Qwen2.5-7B-Instruct (Yang et al., 2024) trajectory per game, each prefix labeled with that trajectory’s outcome. The per-step reward is the weighted composite

$$
\boldsymbol { r } ^ { \mathrm { o f f } } = 0 . 5 \ : a + 0 . 2 \ : e + w _ { \mathrm { c a l } } \mathrm { c a l } ( c , Y ) + w _ { \varphi } \varphi ( c ) ,\tag{5}
$$

where $a \in \{ 0 , 1 \}$ indicates that the sampled action matches the recorded one, e and cal $( c , Y )$ are the format and calibration terms of Eq. (3), $w _ { \mathrm { c a l } } = 0 . 3$ by default, and the spread weight $w _ { \varphi }$ is nonzero only when regularization is enabled (Appendix I). The policy never acts on its own outputs. Because all eight generations for a prefix share the recorded outcome as their label, the group supplies no label contrast, the action-match term rewards reproducing the recorded action, and trained confidence is scored on recorded prefixes of the untrained policy. The reward-form arms of Table 2 use $w _ { \mathrm { c a l } } = 1 . 0$ in place of the 0.3 default.

Online Training. The online route instead trains the live agent on its own rollouts for 120 iterations under Eq. (2); each iteration samples a batch of 4 tasks from the training pool and performs 6 rollouts per task (the GRPO group, $G = 6 )$ , with learning rate $1 0 ^ { - 6 }$ , clip 0.2, and KL coefficient $1 0 ^ { - 3 }$ . Because each offline rollout contributes a single trajectory, the within-task estimands can be measured only on the online route.

## 5 Experimental Setup

Environments and models. We use ALFWorld, TextWorld, and WebShop, each with binary episode success, and Qwen2.5-7B-Instruct (Yang et al., 2024) trained with LoRA (Hu et al., 2022) at rank 32 and $\alpha = 6 4$ . We also perform experiments on a different-sized model Qwen2.5-14B-Instruct, with the same training configurations. The prompt asks for one admissible action on the first line and, on the next line, the confidence that this episode will ultimately succeed as <conf>NN</conf> (details in Appendix I).

Data. Balanced training data hold ALFWorld 1,068 rows from 200 games, TextWorld 943 from 200, and WebShop 1,066 from 372. For comparison, we train the model on each environment independently for 150 steps.

Training ablations. We train a format-only baseline $( w _ { \mathrm { c a l } } = w _ { \varphi } = 0 )$ to compare against four calibration experiments: Brier+φ $( w _ { \mathrm { c a l } } = 1 , w _ { \varphi }$ 0.10), balanced Brier, Brier+dispersion, and balanced+dispersion. Every calibration experiment carries a spread or dispersion term at $w _ { \varphi } = 0 . 1 0 $ so there is no pure-Brier online arm, and each online arm uses a single training seed. Three things were fixed in advance: (i) the estimand (within-task AUROC at k ≤ 2), (ii) the checkpoint rule for the balanced, dispersion, and balanced+dispersion arms (validation argmax over checkpoints {40, 80, 120}), and (iii) the evaluation criterion.

<table><tr><td rowspan="2"></td><td colspan="2">Pooled AUROC</td><td colspan="2">Tail success lift</td></tr><tr><td>Environment Untrained</td><td>Trained</td><td>Untrained</td><td>Trained</td></tr><tr><td>ALFWorld</td><td>0.517</td><td>0.681</td><td>1.55×</td><td>4.33×</td></tr><tr><td>TextWorld</td><td>0.578</td><td>0.722</td><td>1.57×</td><td>4.35×</td></tr><tr><td>WebShop</td><td>0.462</td><td>0.850</td><td>1.11×</td><td>4.35×</td></tr></table>

Table 1: Offline training results comparing base instruction tuned vs RL trained Qwen2.5-7B-Instruct model on held-out single-trajectory Qwen2.5-7B-Instruct recordings: pooled step-level AUROC, and the tail success lift (success rate in the top-10% confident samples vs base success rate). Bootstrap CIs in Appendix G.
<table><tr><td>Reward</td><td>Type</td><td>AUROC [95% CI]</td><td>Format</td></tr><tr><td>Spherical</td><td>proper</td><td>0.882 [.837,.921]</td><td>1.00</td></tr><tr><td>Brier</td><td>proper</td><td>0.864 [.822,.903]</td><td>1.00</td></tr><tr><td>Ranking</td><td>order</td><td>0.845 [.791,.894]</td><td>1.00</td></tr><tr><td>L1</td><td>improper</td><td>0.839 [.788,.887]</td><td>1.00</td></tr><tr><td>Logarithmic</td><td>proper</td><td>0.828 [.770,.880]</td><td>1.00</td></tr><tr><td>Decision</td><td></td><td>0.820 [.773,.863]</td><td>1.00</td></tr><tr><td>None (task only)</td><td></td><td>0.559 [.479,.636]</td><td>0.56</td></tr><tr><td>Untrained</td><td></td><td>0.502 [.465,.542]</td><td>0.08</td></tr></table>

Table 2: Offline training results on WebShop environment across different Calibration reward forms. Pooled AUROC with 95% step-level bootstrap intervals; “Format" is the fraction of samples that correctly predicts the tagged confidence tokens.

Baselines. We compare against untrained and a format reward-only trained model variants. We also compare against two white-box signals available offline: negative mean token entropy and sequence log-probability. Finally, we compare against a depth-only model-free baseline and a depth + length texture regression, which capture what episode position & output length alone predict.

## 6 Results

We first establish offline RL experiments, across all three environments, that outcome-grounded training makes verbalized confidence predictive of eventual episode success (§6.1); we then show that the effect carries over to the live agent trained online in ALFWorld, and transfers to a second live environment, TextWorld (§6.2). We then analyze the trained signal (§7).

<table><tr><td>Estimator</td><td>Seed 0</td><td>Seed 1</td></tr><tr><td>GRPO</td><td>0.613 [.529,.696]</td><td>0.676 [.598,.749]</td></tr><tr><td>RLOO</td><td>0.801 [.739,.856]</td><td>0.876 [.825,.922]</td></tr><tr><td>PPO, value critic</td><td>0.500</td><td>0.500</td></tr></table>

Table 3: Offline training results on WebShop environment across different advantage estimator under a matched ranking + spread reward (w=0.10). Pooled AUROC with 95% step-level bootstrap intervals; PPO with a learned value critic fails to learn parsable response format across both seeds.

## 6.1 Offline RL Training

Each offline training sample is a single recorded trajectory, so the within and between-task estimands of Section 3.2 cannot be formed; thus we report the pooled estimand with the depth-and-length controls it demands. Outcome-grounded training makes verbalized confidence predict eventual success in every environment (Table 1): pooled AUROC rises from 0.517, 0.578, and 0.462 to 0.681, 0.722, and 0.850, and success in the most-confident decile from 1.1–1.6× to about 4.3× the base rate (intervals and per-environment lifts in Appendix G).

Before training, a free white-box signal is competitive. The untrained model’s negated mean token entropy predicts success more accurately than its verbalized confidence in all three environments, and retains most of this power after residualizing on step depth and output length (Appendix G). Training inverts the comparison. After calibration, the verbalized confidence exceeds the token entropy in every environment—most sharply on TextWorld (0.722 vs. 0.399) and WebShop (0.850 vs. 0.430)—because the mean content-token entropy falls roughly sixfold as the policy sharpens and the entropy predictor degrades, where the token distribution grows most confident on the steps that fail.

The pooled confidence AUROC is partly explained by trajectory depth. A predictor built from step depth and output length alone, with no access to the confidence token, reaches AUROC 0.787, 0.855, and 0.799, matching or exceeding the trained confidence AUROC on ALFWorld and TextWorld though not on WebShop (Appendix G). A depth-independent component of the confidence nonetheless does not replicate the same result: residualizing the confidence on depth and output length leaves AUROC 0.626, 0.677, and 0.811, whereas the same residualization reduces the untrained model’s confidence to 0.534, 0.479, and 0.509, near chance. The depth-independent signal is thus a product of training. Because the pooled estimand cannot separate confidence from depth, we evaluate the depth-matched and within-task estimands in the online setting, which supports them (Section 6.2).

![](images/7f1d6a2c06b3b94dc60716dc0335b983d678cd51cc86fdb29a69c4b2249ed46b.jpg)  
Figure 2: Comparison of reliability over all steps of untrained 7B agent against two online RL trained agents on live ALFWorld. Marker area ∝ bin mass; same setting as of Figure 3).

The presence of a calibration term matters more than its functional form (Table 2). Six reward forms—spherical, Brier, ranking, L1, logarithmic, and decision—yield overlapping AUROC between 0.820 and 0.882, statistically indistinguishable, whereas training with the task reward alone reaches only 0.559, close to the untrained 0.502. The choice of advantage estimator, by contrast, is consequential (Table 3): under a matched rankingplus-spread reward, RLOO attains 0.801 and 0.876 against 0.613 and 0.676 for GRPO, and PPO with a cold-start value critic fails to learn (0.500 across two seeds) parsable response format.

## 6.2 Online RL Training

All four calibration rewards raise pooled earlywindow AUROC on the live acting 7B agent (Table 4): +0.072 [.026,.112] for Brier+φ, +0.076 [.038,.111] for balanced Brier, +0.101 [.069,.134] for Brier+dispersion, and +0.048 [.002,.086] for balanced+dispersion. The format-only control gives +0.019 [-.010,.051]. Two deltas are marginal: balanced+dispersion’s Pooled AUROC $( k ~ \leq ~ 2 )$ +0.048 and Brier+φ’s top-10 success (k ≤ 2) % +0.080 [-.000,.185].

![](images/a9748621d1348eef821f228e88ff5064b72c3c060b70476cc2fc6296b97db7cb.jpg)  
Figure 3: Comparison of Risk–Coverage of different live agents on ALFWorld: for the early window (k ≤ 2, top) and all steps (bottom).

Calibration. ECE over all steps (10 bins) goes from 0.419 to 0.053–0.115 and the Brier score from 0.250 to 0.044–0.054, all eight paired deltas with CIs above 0; format-only moves them by +0.010 and +0.003 respectively. Mean confidence falls from 0.475 to 0.082–0.163 against a step-level success rate of 0.057, so much of the ECE and Brier drop is the agent learning the base rate; AUROC and selective execution are the discrimination evidence (reliability diagrams in Figure 2). A constant prediction at the step-level base rate p has ECE near zero and a Brier score of $p ( 1 - p )$ , so these two columns do not by themselves show discrimination; in the early window, where the base rate is 0.201, ECE goes from 0.277 to 0.051–0.133 and Brier from 0.229 to 0.137–0.157 (Appendix F).

Selective execution. Success among the 10% most confident early steps rises from 0.493 untrained to 0.56–0.66 across the four calibration arms, three of four deltas with CIs above 0 (Table 4, Figure 3); among the top 25% it rises from 0.363 to 0.415–0.446, all four delta CIs above 0. The format-only control moves neither window (−0.024 and −0.012), so the gain comes from the calibration term rather than the RL format reward: against that control, Brier+φ adds +0.053 [.016,.090] and Brier+dispersion +0.082 [.050,.114] in pooled early-window AUROC. None of this costs task success, which holds at 0.189–0.203 against 0.201 untrained, while the <conf>format parsing correctness rises from 0.978 to 1.000. The untrained agent already sorts whole tasks well: its most confident quarter succeeds at 0.557 against a 0.201 base rate. Training adds little to whole-task triage here (top-50% 0.349–0.362 against 0.362 untrained, task-level AUROC deltas all include 0) and instead sharpens the finer step-level signal behind selective execution.

<table><tr><td rowspan="3">Arm</td><td rowspan="3">Episode success</td><td colspan="2">Pooled AUROC.  $k \leq 2$ </td><td rowspan="3">ECE</td><td rowspan="3">Brier</td><td colspan="2">Top-10% success, k ≤ 2</td></tr><tr><td>value</td><td> $\Delta$ </td><td>(all steps)</td><td></td></tr><tr><td>Untrained</td><td>0.201</td><td>0.689 [.640,.733]</td><td></td><td>0.419 [.398,.438]</td><td>0.250 [.237,.264]</td><td>0.493 [.323,.675]</td><td></td></tr><tr><td>Format-only</td><td>0.188</td><td>0.708 [.660,.749]</td><td>+0.019 [-.010,.051]</td><td>0.429 [.408,.447]</td><td>0.253 [.239,.267]</td><td>0.465 [.296,.626]</td><td>-0.024[-.094,.037]</td></tr><tr><td>Brier+φ</td><td>0.194</td><td>0.761 [.716,.798]</td><td>+0.072 [.026,.112]</td><td>0.115 [.105,.124]</td><td>0.052 [.045,.061]</td><td>0.561 [.391,.748]</td><td>+0.080 [-.000,.185]</td></tr><tr><td>Balanced Brier</td><td>0.197</td><td>0.765 [.721,.805]</td><td>+0.076 [.038,.111]</td><td>0.114 [.102,.125]</td><td>0.054 [.046,.064]</td><td>0.609 [.413,.765]</td><td>+0.115 [.035,.193]</td></tr><tr><td>Brier+dispersion</td><td>0.189</td><td>0.790 [.742,.828]</td><td>+0.101 [.069,.134]</td><td>0.054 [.050,.059]</td><td>0.044 [.034,.054]</td><td>0.635 [.426,.804]</td><td>+0.159 [.054,.224]</td></tr><tr><td>Balanced+dispersion</td><td>0.203</td><td>0.737 [.684,.783]</td><td>+0.048 [.002,.086]</td><td>0.053 [.049,.057]</td><td>0.047 [.037,.059]</td><td>0.657 [.448,.818]</td><td>+0.167[.068,.235]</td></tr></table>

Table 4: Online training results for Qwen2.5-7B-Instruct on ALFWorld. CIs are 95% task-level cluster bootstraps, ∆ = (trained - base instruction tuned) on the shared tasks; early-window ECE, Brier, top-25% success, and the ECE and Brier deltas are in Appendix F.
<table><tr><td rowspan="2">Arm</td><td>Pooled</td><td>Between-task</td><td colspan="4">Within-task</td></tr><tr><td>k=0</td><td> $\Delta , k \le 2$ </td><td> $k \leq 2 ( n \mathrm { m i x e d } )$ </td><td>∆, k ≤ 2 (n paired)</td><td>∆, all steps</td><td>∆ depth-matched, all</td></tr><tr><td>Untrained</td><td>0.562 [.530,.600]</td><td></td><td>0.606 (20) [.515,.696]</td><td></td><td></td><td></td></tr><tr><td>Format-only</td><td></td><td></td><td></td><td>0.568 [.534,.602] +0.033 [-.004,.082] 0.571 (15) [.484,.657] -0.054 (8) [-.118,.018]</td><td>-0.077[-.114,-.045]+0.013[-.080,.133]</td><td></td></tr><tr><td>Brier+φ</td><td></td><td></td><td></td><td>0.642[.584,.691]+0.037[-.017,.085] 0.585 (20)[.483,.680]-0.059 (12) [-.184,.059]</td><td>-0.013 [-.056,.029] +0.027[-.059,.124]</td><td></td></tr><tr><td>Balanced Brier</td><td></td><td></td><td></td><td>0.697 [.637,.744] +0.032 [-.019,.077] 0.657 (19)[.582,.733] −0.026 (11) [-.128,.055]</td><td>-0.038 [-.082,.005] +0.052[-.034,.143]</td><td></td></tr><tr><td>Brier+dispersion</td><td></td><td></td><td></td><td>0.700 [.634,.763] +0.072[.029,.115] 0.661 (15) [.568,.757] -0.092 (7) [-.262,.062]</td><td>-0.050[-.102,.004] +0.035[-.060,.140]</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Balanced+dispersion 0.670 [.612,.722] +0.023 [-.038,.083] 0.587 (20) [.486,.688] -0.031 (13) [-.120,.058] -0.072 [-.114,-.031] -0.026 [-.107,.062]</td><td></td><td></td></tr></table>

Table 5: Decomposition of the online training gain by estimand and step window on ALFWorld. Columns move left to right from no confound removed to difficulty and depth both removed; values are absolute AUROC except where marked ∆ (trained minus untrained). Within-task columns use only tasks with both a succeeding and a failing attempt: the absolute cell averages per-task AUROC over that arm’s n mixed-outcome tasks (n mixed), and each ∆ resamples the tasks mixed in both the trained and untrained arm (n paired); depth-matched is concordance over same-task, same-step-index pairs across all steps. Bold = paired 95% CI excludes 0.

![](images/eb06b25d41d7967183d119a07d9bfa44b768dd30830a6224dddf2c78c9348644.jpg)  
Figure 4: Online training results on TextWorld environment: paired within-task AUROC gain (trained − untrained) by step window; 95% CIs over shared heldout tasks.

Transfer to a second environment. The online results above come from ALFWorld environment. To test applicability to another environment, we repeat the live, on-policy training in TextWorld with the Brier+φ reward. Episode success rises from 0.146 to 0.262 and format correctness stays near-saturated (0.971 → 0.979). Unlike ALFWorld, where attempt-level foresight did not detectably change, within-task discrimination here improves in the early window: the paired (trained - untrained) within-task AUROC is +0.265 [.107,.428] at step 0 and +0.070 [.010,.130] over the first three steps, both delta CIs above 0, and decays to zero over the full episode (Figure 4). Detailed experiment results appear in Appendix D.

A larger model size. We also train the larger Qwen2.5-14B-Instruct agent online in ALFWorld with the Brier+φ reward under the same protocol. The pooled matched-emission AUROC rises from 0.697 to 0.729 (paired trained-untrained ∆ of +0.032), a further improvement in the same direction as the 7B gain that suggests the effect carries over to a larger model size, though fewer held-out samples make the evaluation minimal & fluctuating; the full results are in Appendix E.

## 7 Analysis of the Trained Signal

Results in §6 show that training improves confidence. As part of this section, we ask what it changes. We decompose the online gain by estimand to locate where the improvement comes from (§7.1), trace it back to the first step of each episode (§7.2), and benchmark it against depth and white-box signals (§7.3) before drawing out what it means for an agent controller (§7.4).

## 7.1 Decomposition

Training’s gain is specifically between-task: it sharpens which tasks the agent will succeed at, not how it reads individual attempts (Table 5; Appendix Figure 6). Between-task early-window AUROC rises for every calibration arm (0.825 to 0.848–0.897), and only Brier+dispersion’s +0.072 [.029,.115] is significant. The localization suggests: the within-task estimand does not move (0.606 untrained against 0.585–0.661; paired point estimates −0.026 to −0.092), so the measured improvement is concentrated in task ranking, and no experiment met the pre-registered within-task criterion of §5. Consistent with this, the within-task all-step estimates match format-only experiment (format-only −0.077, balanced+dispersion −0.072) and similar to the depth-confounded estimand of §3.2. Relative to the format-only control, the calibration term does incur gain on early-window within-task discrimination (+0.062 [-.034,.177] for Brier+φ, +0.127 [.062,.189] for Brier+dispersion), while the depth-matched within-task deltas are not significant (Appendix C).

Pooled and between-task early-window values move together under training (0.689/0.825 untrained to 0.761/0.862 for Brier+φ; details in Appendix C), while task-level AUROC, triage deltas, and within-task deltas show no significant gains. One reading consistent with all four estimands is that training sharpens and brings foresight forward to step 0, the task ranking the untrained agent already expresses in the early window (§7.2).

## 7.2 Foresight at Step 0

Training brings this task ranking forward to the very first step. Pooled step-0 AUROC rises from 0.562 [.530,.600] to 0.642–0.700 across the four calibration arms, even though the confidence is emitted after the sampled first action but before any environment feedback. The signal is task-level: withintask step-0 AUROC stays at chance (0.450–0.602, every CI spanning 0.5), so the agent is ranking which tasks it will solve rather than grading its first action. The untrained baseline confirms this is genuine improvement: at greedy decoding it emits exactly 0.50 at step 0 on every task of a 64-task probe (Appendix B), so its sampled 0.562 is noise around a near-constant value, and the task ranking it expresses (between-task 0.825) emerges only after the first feedback. The mechanism follows from Eq. (2): uniform-outcome groups, which still arise by sampling despite mixed-outcome task selection, carry no success/failure contrast, so their gradient moves a task’s confidence toward its shared outcome at every step, step 0 included.

## 7.3 Depth and White-Box Baselines

The depth-only baseline beats confidence on pooled all-step AUROC and matches or beats every trained experiment, yet collapses to 0.499–0.500 in the early window, whereas the early-window and depthmatched estimands are essential for a decent controller (Figure 5; per-size & per-experiment values in Appendix B, C). Token entropy is the strongest competitor for offline training; the online experiments comparison has no white-box competitor.

## 7.4 Confidence as a Triage Signal

Outcome-grounded RL makes verbalized confidence a calibrated early triage signal: the 10% most confident early steps succeed at 0.56–0.66 against 0.49 untrained and a base rate of 0.20, with ECE 0.05–0.12 (Figure 3). The gain is earlier recognition of which tasks the agent will fail (§7.2); attempt-level foresight does not improve on ALF-World, though it emerges in the early window on TextWorld (§6.2). Among the most confident steps (the top 10%), higher confidence does not reliably mean higher success (Figure 3), and over all steps the most-confident bin is overconfident (Figure 2), so a controller should rely on a band of high confidence rather than the very highest values alone. Finally, the within-task question needs training signals that contrast attempts at the same task.

## 8 Conclusion

We introduce a training method that calibrates a language agent’s verbalized confidence by rewarding it against eventual task outcomes under GRPO. On a live multi-turn agent it turns an unreliable confidence token into a usable early-triage signal without hurting task success, and a decomposition shows the gain is between tasks: the agent learns earlier which tasks it will fail, not which attempts.

## Limitations

Our evidence is strongest as a between-task effect: on the live ALFWorld agent, training sharpens which tasks the agent will succeed at, and establishing the same signal at the attempt level more broadly is left to future work, as the withintask estimands move only in the early window on TextWorld and not yet on ALFWorld. Our positive live multi-turn results span two text-based environments (ALFWorld and TextWorld) at the 7B scale, and a larger 14B agent corroborates the pooled effect in Appendix E. Our live WebShop results are inconclusive: held-out success stays low for both arms, leaving too few informative held-out tasks to assess the signal (Appendix D); extending the live study to additional model families is left to future work.

## Ethical considerations

We conducted experiments within the provisions of the ACL Ethics Policy and relevant researchintegrity guidelines. There are, to the best of our knowledge, no remaining ethical risks that have not been addressed.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, Bangkok, Thailand. Association for Computational Linguistics.

David Bani-Harouni, Chantal Pellegrini, Paul Stangel, Ege Özsoy, Kamilia Zaripova, Nassir Navab, and Matthias Keicher. 2026. Rewarding doubt: A reinforcement learning approach to calibrated confidence expression of large language models. In The Fourteenth International Conference on Learning Representations.

GLENN W. BRIER. 1950. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1 – 3.

Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2019. Textworld: A learning environment for text-based games. In Computer Games, pages 41–75, Cham. Springer International Publishing.

Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. 2025. Beyond binary rewards: Training lms to reason about their uncertainty. arXiv preprint arXiv:2507.16806.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 4885–4894, Red Hook, NY, USA. Curran Associates Inc.

Tilmann Gneiting and Adrian E. Raftery. 2007. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102:359 – 378.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning - Volume 70, ICML’17, page 1321–1330. JMLR.org.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. volume 645, pages 633–638.

J A Hanley and B J McNeil. 1982. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1):29–36.

Matthew Hausknecht, Prithviraj Ammanabrolu, Marc-Alexandre Côté, and Xingdi Yuan. 2020. Interactive fiction games: A colossal adventure. Preprint, arXiv:1909.05398.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (mostly) know what they know. Preprint, arXiv:2207.05221.

Amita Kamath, Robin Jia, and Percy Liang. 2020. Selective question answering under domain shift. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 5684– 5696, Online. Association for Computational Linguistics.

Jixuan Leng, Chengsong Huang, Banghua Zhu, and Jiaxin Huang. 2025. Taming overconfidence in LLMs: Reward calibration in RLHF. In The Thirteenth International Conference on Learning Representations.

Chen Li, Xiaoling Hu, Songzhu Zheng, Jiawei Zhou, and Chao Chen. 2026. Orce: Order-aware alignment of verbalized confidence in large language models. Preprint, arXiv:2605.12446.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others. 2024. Agentbench: Evaluating LLMs as agents. In The Twelfth International Conference on Learning Representations.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10737–10746.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. {ALFW}orld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations.

Joykirat Singh, Zaid Khan, Archiki Prasad, Justin Chih-Yao Chen, Akshay Nambi, Hyunji Lee, Elias Stengel-Eskin, and Mohit Bansal. 2026. Agent-brace: Decoupling beliefs from actions in long-horizon tasks via verbalized state uncertainty. Preprint, arXiv:2605.11436.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442, Singapore. Association for Computational Linguistics.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. 2020. TRL: Transformers Reinforcement Learning.

Liaoyaqi Wang, Chunsheng Zuo, William Jurayj, Benjamin Van Durme, and Anqi Liu. 2026a. Process supervision of confidence margin for calibrated LLM

reasoning. In Third Conference on Language Modeling.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. ScienceWorld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Ziqi Wang, Xingzhou Lou, Meiqi Wu, Zhengqi Wen, and Junge Zhang. 2026b. Calibration-aware policy optimization for reasoning LLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18375–18390, San Diego, California, United States. Association for Computational Linguistics.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations.

Tianyang Xu, Shujin Wu, Shizhe Diao, Xiaoze Liu, Xingyao Wang, Yangyi Chen, and Jing Gao. 2024. SaySelf: Teaching LLMs to express confidence with self-reflective rationales. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 5985–5998, Miami, Florida, USA. Association for Computational Linguistics.

Weihao Xuan, Qingcheng Zeng, Heli Qi, Yunze Xiao, Junjue Wang, and Naoto Yokoya. 2026. The confidence dichotomy: Analyzing and mitigating miscalibration in tool-use agents. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11325–11349, San Diego, California, United States. Association for Computational Linguistics.

Duygu Nur Yaldiz, Evangelia Spiliopoulou, Zheng Qi, Siddharth Varia, Srikanth Doss, and Nikolaos Pappas. 2026. Balancing classification and calibration performance in decision-making LLMs via calibration aware reinforcement learning. In Findings ofthe Association for Computational Linguistics: ACL 2026, pages 12537–12553, San Diego, California, United States. Association for Computational Linguistics.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and 40 others. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Shunyu Yao, Howard Chen, John Yang, and Karthik R Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. In Advances in Neural Information Processing Systems.

Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Nigel Collier, and Andreas Vlachos. 2026. LoVeC: Reinforcement learning for better verbalized confidence in long-form generation. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 33336–33363, San Diego, California, United States. Association for Computational Linguistics.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. Webarena: A realistic web environment for building autonomous agents. Preprint, arXiv:2307.13854.

## A Estimand Definitions

This appendix gives the formal definitions of the four estimands and the depth-only baseline introduced in §3.2. All five are the same AUROC statistic applied to different comparison sets and score transforms, so any difference between them is attributable to the confound each one cancels, not to a change of metric.

Notation. A task s is attempted $i = 1 , \dots , G$ times. Attempt $( s , i )$ has a binary episode outcome ${ Y _ { s , i } } \in \{ 0 , 1 \}$ (success or failure) and emits a step confidence $c _ { s , i , t } \in [ 0 , 1 ]$ at each decision step $t = 0 , 1 , \ldots ;$ every step inherits its attempt’s outcome, so step $( s , i , t )$ carries the label $Y _ { s , i }$ . A window $W \subseteq \mathbb { Z } _ { \geq 0 }$ is a set of step indices that restricts which steps are compared; we report results for three, $W _ { \mathrm { a l l } } = \mathbb { Z } _ { > 0 } = \{ 0 , 1 , 2 , . . . \}$ (all steps), $W _ { k \le 2 } ~ = ~ \{ 0 , 1 , 2 \}$ (the early window), and $W _ { 0 } ~ = ~ \{ 0 \}$ (step 0 alone). Write $\mathcal { S } _ { W }$ = $\{ ( s , i , t ) : t \in W , \ c _ { s , i , t }$ emitted} for the steps in play, and partition them by outcome into positives $\mathcal { P } _ { W } = \{ ( s , i , t ) \in \mathcal { S } _ { W } : Y _ { s , i } = 1 \}$ and negatives $\mathcal { N } _ { W } = \{ ( s , i , t ) \in S _ { W } : Y _ { s , i } = 0 \}$

The AUROC statistic. Fix a set of items each carrying a real-valued score and a binary label. Thresholding the score at $\tau$ calls an item positive when its score exceeds $\tau ;$ sweeping τ traces the receiver operating characteristic (ROC) curve of the true-positive rate $\mathrm { T P R } ( \tau ) ~ = ~ P ( \mathrm { s c o r e } ~ >$ $\tau \ | \ \mathrm { \ l a b e l } \ = \ 1 \ $ against the false-positive rate $\mathrm { F P R } ( \tau ) = P ( \mathrm { s c o r e } > \tau \mid \mathrm { l a b e l } = 0 )$ . The area under this curve (AUROC) summarizes separation across all thresholds at once: it is $\textstyle { \frac { 1 } { 2 } }$ when scores are independent of the label and 1 when every positive outscores every negative. A standard identity turns the area into a pairwise statistic rather than an integral, namely the probability that a uniformly drawn positive outranks a uniformly drawn negative (Hanley and McNeil, 1982), which for a finite set of positive scores A and negative scores $B ,$ with ties split evenly, is

$$
\operatorname { A U C } ( A , B ) = { \frac { 1 } { | A | \left| B \right| } } \sum _ { a \in A } \sum _ { b \in B } h ( a , b ) ,\tag{6}
$$

where the pairwise kernel $h ( a , b ) = \mathbf { 1 } [ a > b ] +$ ${ \bf \frac { 1 } { 2 } } { \bf 1 } [ a = b ]$ scores a correctly ranked pair 1, a tie $\frac { \mathrm { 1 } } { \mathrm { 2 } }$ , and a misranked pair 0. Each estimand below instantiates (6) with a particular score for every step and a particular partition into A and B.

Pooled AUROC. Score each step by its raw confidence and compare every success-step against every failure-step in the window, holding none of the confounds fixed:

$$
\mathrm { A U C } _ { \mathrm { p o o l } } ( W ) = \mathrm { A U C } \big ( c ( \mathcal { P } _ { W } ) , c ( \mathcal { N } _ { W } ) \big ) .\tag{7}
$$

This conflates task difficulty (i), attempt quality (ii), and depth (iii): a success-step may outrank a failure-step merely because its task is easy or its episode is nearly over.

Between-task AUROC. Collapse each task to a single number by replacing every step’s confidence with the mean confidence of its task over the window. Writing $S _ { s , W } = \{ ( i , t ) : ( s , i , t ) \in S _ { W } \}$ for the steps task s emits in the window,

$$
\bar { c } _ { s } ( W ) = \frac { 1 } { | S _ { s , W } | } \sum _ { ( i , t ) \in S _ { s , W } } c _ { s , i , t } ,\tag{8}
$$

and then apply (7) with $c _ { s , i , t }$ replaced by $\bar { c } _ { s } ( W )$ Because every step of a task now shares one score, all within-task comparisons tie and contribute exactly ${ \frac { 1 } { 2 } } ;$ ; only cross-task comparisons carry signal, so the statistic measures whether easier (highermean-confidence) tasks succeed more often, i.e. difficulty alone.

Within-task AUROC. Fix the task and compare its succeeding attempts against its failing attempts, which cancels difficulty because both sides share the task. For a task s with both outcomes present in the window (a mixed-outcome task, written $s \in$ $\mathcal { M } _ { W } )$ , define the per-task score

$$
a _ { s } ( W ) = \mathrm { A U C } \big ( c ( \mathcal { P } _ { W } ^ { s } ) , c ( \mathcal { N } _ { W } ^ { s } ) \big ) ,\tag{9}
$$

and average over the mixed-outcome tasks:

$$
\mathrm { A U C } _ { \mathrm { w i t h i n } } ( W ) = \frac { 1 } { | \mathcal { M } _ { W } | } \sum _ { s \in \mathcal { M } _ { W } } a _ { s } ( W ) .\tag{10}
$$

Only tasks that both succeed and fail in the window are estimable, so $| \mathcal { M } _ { W } |$ (reported as the mixed count in Table 6) sets how many clusters the estimate rests on; it shrinks as the window narrows, which is why the step-0 and early-window withintask intervals are wide. The statistic still mixes attempt quality (ii) with depth (iii), since a succeeding attempt’s step may outrank a failing one simply by sitting later in the episode.

Depth-matched within-task AUROC. Additionally fix the step index, so only confidences emitted at the same depth of the same task are ever compared. For each task s and index t with both outcomes present, let

$$
a _ { s , t } = \mathrm { A U C } \big ( c ( \mathcal { P } _ { W } ^ { s , t } ) , c ( \mathcal { N } _ { W } ^ { s , t } ) \big ) ,\tag{11}
$$

and, writing $\mathcal { C } _ { W } = \{ ( s , t ) : t \in W , \exists i , i ^ { \prime } Y _ { s , i } =$ 1, $Y _ { s , i ^ { \prime } } = 0 \}$ for the qualifying (mixed-outcome, same-step) cells, average over them:

$$
\mathrm { A U C } _ { \mathrm { d m } } ( W ) = \frac { 1 } { | { \mathcal C } _ { W } | } \sum _ { ( s , t ) \in { \mathcal C } _ { W } } a _ { s , t } .\tag{12}
$$

Neither difficulty nor depth can contribute, so what survives is attempt-level foresight alone: among attempts of one task that have reached the identical step, does the one that will ultimately succeed state higher confidence.

Depth-only baseline. To quantify how much of any estimand a trivial position signal could explain, we recompute it with each confidence replaced by the negated step index,

$$
\tilde { c } _ { s , i , t } = - t ,\tag{13}
$$

feeding c˜ in place of c through the same estimand ((7), (8), (10), or (12)). This baseline ranks shallow steps above deep ones and knows nothing of the agent. Over all steps it can match or exceed a real signal, because succeeding and failing attempts differ systematically in length. In the step-0 window every $\tilde { c } = 0 ,$ , so by (6) every comparison ties and the baseline is exactly $\scriptstyle { \frac { 1 } { 2 } } ;$ ; in the early window the three admissible depths leave it near chance. A trained signal that beats the depth-only baseline in the early window or at step 0 therefore exhibits foresight that position cannot explain.

## B Untrained Estimand Ladder

Table 6 gives the untrained agents under every estimand of Section 3.2, and Figure 5 shows the Qwen2.5-7B-Instruct agent before and after Brier+φ training against the depth-only baseline. Untrained episode success and format correctness are 20.1% / 97.8% at 7B and 36.9% / 100% at 14B. On a 64-task probe (one run per decoding mode; no interval), greedy decoding of the untrained 7B agent gives pooled all-step AUROC 0.784 against 0.793 at temperature 1.0, and at greedy the agent emits exactly 0.50 at step 0 on every task of the probe. On the 45-task training pool the untrained 7B agent has within-task AUROC 0.677 [.628,.721] at $k \leq 2$ and 0.763 [.739,.786] over all steps.

<table><tr><td>Estimand</td><td>7B (128 tasks)</td><td>14B (128 tasks)</td></tr><tr><td>Pooled, all steps</td><td>0.740 [.704,.776]</td><td>0.698 [.650,.746]</td></tr><tr><td>Depth-only baseline, all steps</td><td>0.817 [.777,.857]</td><td>0.755 [.722,.792]</td></tr><tr><td>Pooled,  $k \leq 2$ </td><td>0.689 [.640,.733]</td><td>0.722 [.667,.774]</td></tr><tr><td>Depth-only baseline,  $k \leq 2$ </td><td>0.499</td><td>0.500</td></tr><tr><td>Pooled, step 0</td><td>0.562 [.530,.600]</td><td>0.599 [.532,.677]</td></tr><tr><td>Between-task,  $k \leq 2$ </td><td>0.825 [.758,.881]</td><td>0.830 [.748,.896]</td></tr><tr><td>Between-task, all steps</td><td>0.884 [.820,.937]</td><td>0.741 [.650,.824]</td></tr><tr><td>Within-task,  $k \leq 2$  (n mixed)</td><td>0.606 [.515,.696] (20)</td><td>0.456 [.395,.519] (22)</td></tr><tr><td>Within-task, all steps</td><td>0.651 [.598,.704]</td><td>0.653 [.599,.707]</td></tr><tr><td>Within-task, step 0</td><td>0.529 [.432,.620]</td><td>0.400 [.309,.493]</td></tr><tr><td>Depth-matched within-task, all steps</td><td>0.633 [.565,.715]</td><td>0.616 [.572,.667]</td></tr><tr><td>Depth-only within-task, all steps</td><td>0.750</td><td>0.683</td></tr><tr><td>Attempt-level, pooled</td><td>0.798 [.735,.852]</td><td>0.787 [.714,.847]</td></tr><tr><td>Attempt-level, within-task</td><td>0.658 [.509,.806]</td><td>0.453 [.335,.571]</td></tr></table>

Table 6: Untrained Qwen2.5-7B-Instruct agents on live ALFWorld: AUROC of step confidence against eventual episode success under every estimand of §3.2.

![](images/8e91002219dcd216ecbd15206eb8b70420e995b89ea36a96357498e5120e8e1d.jpg)  
Figure 5: Qwen2.5-7B-Instruct on live ALFWorld: depth-only baseline, untrained agent, and the Brier+φ RL comparison. Bars are 95% task-level bootstrap intervals.

## C Detailed Results on Online RL Training

Table 7 gives absolute values for all six experiments under all estimands, and Table 8 the paired deltas; Figure 6 plots the early-window absolute values and Figure 7 the pooled and within-task early-window deltas. Format correctness is 0.978 untrained and 1.000 for every trained experiment, and the depth-only baseline on each experiment’s rows gives pooled all-step AUROC 0.817 / 0.853 / 0.844 / 0.845 / 0.827 / 0.833 in row order and 0.499–0.500 at $k \ \leq \ 2 .$ Its within-task all-step trajectory is non-monotone (untrained 0.651 → checkpoint $4 0 0 . 6 7 7  8 0 0 . 7 4 5  1 2 0 0 . 6 7 3 )$ , as is its within-task early-window trajectory (0.606 $ 0 . 6 0 3  0 . 6 4 9  0 . 5 8 5 )$ . Training dynamics for Brier+φ (one training seed; no interval): mean confidence 0.71 at iteration 0, 0.54 at 28, 0.29 at 60, 0.20 at 86, and about 0.15 at the end; KL to the reference policy $0 . 0 0 2  0 . 3 2$ . Attempt-level scores (mean confidence over steps $\leq ~ 2$ per attempt, one score per attempt) agree with the steplevel estimands: pooled deltas are +0.051 [-.004,.105], +0.099 [.057,.141], and +0.066 [.017,.111] for Brier+φ, Brier+dispersion, and balanced Brier, and withintask deltas are $- 0 . 1 6 3 \ [ - . 3 7 6 , . 0 5 4 ] , \ - 0 . 2 4 6 \ [$ [-.489,-.021] (7 tasks), and −0.036 [-.205,.118]. Calibration-rewardminus-format-only contrasts: Brier+φ minus format, within all +0.068 [.025,.107] (8), within $k \leq 2$ $+ 0 . 0 6 2 \ [ - . 0 3 4 , . 1 7 7 ]$ (8), pooled $k \leq 2 + 0 . 0 5 3$ [.016,.090]; Brier+dispersion minus format, within $k \ \leq \ 2$ +0.127 [.062,.189] (5), within all +0.036 [.005,.068] (5), pooled $k \leq 2 + 0 . 0 8 2$ [.050,.114].

![](images/d3bcfff9ce3ff1e03e7eacc5a6734937c72e5dd0203c67fc943d326405f15baf.jpg)  
Figure 6: Pooled, between-task, and within-task AU-ROC in the early window $( k \le 2 )$ for the untrained Qwen2.5-7B-Instruct agent and four calibration experiments (same setting as Table 4). Bars are 95% task-level bootstrap intervals.

<table><tr><td>Arm</td><td>Succ.</td><td>Pooled all Pooled k ≤ 2</td><td>Pooled t=0 Between</td><td></td><td> $k \leq 2$  Within all (n)</td><td></td><td>Within k ≤ 2 (n)</td><td>Within t=0 Depth-matched</td></tr><tr><td>Untrained</td><td>0.201</td><td>0.740 [.704,.776]</td><td>0.689 [.640,.733]</td><td>0.562 [.530,.600]</td><td>0.825 [.758,.881]</td><td>0.651 (20) [.598,.704]</td><td>0.606 (20) [.515,.696]</td><td>0.529 [.432,.620] [.565,.715]</td></tr><tr><td>Format-only</td><td>0.188</td><td>0.754 [.712,.792]</td><td>0.708 [.660,.749]</td><td>0.858 [.534,.602] [.792,.911] 0.862</td><td>0.665 (15) [.601,.727]</td><td>0.571 (15) [.484,.657]</td><td>0.507 [.393,.610]</td><td>0.646 [.575,.739]</td></tr><tr><td>Brier+φ</td><td>0.194</td><td>0.799 [.753,.838]</td><td>0.761 [.716,.798]</td><td>0.642 [.584,.691] [.798,.911]</td><td>0.673 (20) [.615,.733]</td><td>0.585 (20) [.483,.680]</td><td>0.500 [.378,.626]</td><td>0.659 [.607,.719]</td></tr><tr><td>Balanced Brier</td><td>0.197</td><td>0.782 [.730,.829]</td><td>0.765 [.721,.805]</td><td>0.697 [.637,.744]</td><td>0.856 0.683 (19) [.791,.907] [.621,.753]</td><td>0.657 (19) [.582,.733]</td><td>0.592 [.453,.725]</td><td>0.685 [.630,.752]</td></tr><tr><td>Brier+dispersion</td><td>0.189</td><td>0.777 [.734,.817]</td><td>0.790 [.742,.828]</td><td>0.700 [.634,.763]</td><td>0.897 [.847,.944]</td><td>0.686 (15) 0.661 (15) [.627,.746] [.568,.757]</td><td>0.602 [.483,.712]</td><td>0.667 [.607,.744]</td></tr><tr><td>Balanced+dispersion 0.203</td><td>0.742</td><td>0.737 [.696,.784] [.684,.783]</td><td></td><td>0.670 [.612,.722]</td><td>0.848 [.769,.911] [.564,.660]</td><td>0.612 (20)</td><td>0.587 (20) [.486,.688]</td><td>0.450 0.607 [.340,.566] [.552,.665]</td></tr></table>

Table 7: Absolute values for every online RL for Qwen2.5-7B-Instruct experiment. Succ. = episode success rate; intervals are 95% task-level cluster bootstraps
<table><tr><td>Arm</td><td>Pooled all</td><td>Pooled k  $\leq 2$ </td><td>Between k  $; \le 2$ </td><td>Within all (n)</td><td>Within k ≤ 2 (n)</td><td>Depth-matched</td></tr><tr><td rowspan="2">Format-only</td><td>+0.014</td><td>+0.019</td><td>+0.033</td><td>-0.077 (8)</td><td>-0.054 (8)</td><td>+0.013</td></tr><tr><td>[-.019,.054]</td><td>[-.010,.051]</td><td>[-.004,.082]</td><td>[-.114,-.045]</td><td>[-.118,.018]</td><td>[-.080,.133]</td></tr><tr><td rowspan="2">Brier+φ</td><td>+0.058</td><td>+0.072</td><td>+0.037</td><td>-0.013 (12)</td><td>-0.059 (12)</td><td>+0.027</td></tr><tr><td>[.014,.103]</td><td>[.026,.112]</td><td>[-.017,.085]</td><td>[-.056,.029]</td><td>[-.184,.059]</td><td>[-.059,.124]</td></tr><tr><td rowspan="2">Balanced Brier</td><td>+0.041</td><td>+0.076</td><td>+0.032</td><td>-0.038 (11)</td><td>-0.026 (11)</td><td>+0.052</td></tr><tr><td>[-.008,.084]</td><td>[.038,.111]</td><td>[-.019,.077]</td><td>[-.082,.005]</td><td>[-.128,.055]</td><td>[-.034,.143]</td></tr><tr><td rowspan="2">Brier+dispersion</td><td>+0.037</td><td>+0.101</td><td>+0.072</td><td>-0.050 (7)</td><td>-0.092 (7)</td><td>+0.035</td></tr><tr><td>[-.007,.084]</td><td>[.069,.134]</td><td>[.029,.115]</td><td>[-.102,.004]</td><td>[-.262,.062]</td><td>[-.060,.140]</td></tr><tr><td rowspan="2">Balanced+dispersion</td><td>+0.002</td><td>+0.048</td><td>+0.023</td><td>–0.072 (13)</td><td>-0.031 (13)</td><td>-0.026</td></tr><tr><td>[-.044,.047]</td><td>[.002,.086]</td><td>[-.038,.083]</td><td>[-.114,-.031]</td><td>[-.120,.058]</td><td>[-.107,.062]</td></tr></table>

Table 8: Paired deltas (trained - untrained) for every online RL Qwen2.5-7B-Instruct experiment; same setting as Table 7. Pooled and between-task deltas resample the shared evaluation tasks; within-task deltas resample the n tasks mixed in both arms.

![](images/7814d5ccce56adf0b941dee8a9858da09482735b5b8469d641dcc0ddfab8ba18.jpg)  
Figure 7: Paired deltas of AUROC (trained - untrained) in the early window $( k ~ \le ~ 2 )$ , pooled (squares) and within-task (circles), with 95% task-level bootstrap intervals.

## D Detailed results on Other Environments

Section 6.2 reports the live, on-policy TextWorld transfer in brief. Table 9 gives the win and emission rates and the within-task AUROC by window. Beyond the early-window gain of Figure 4, the paired trained-minus-untrained within-task AU-ROC is +0.017 [-.050,.082] by five steps and −0.029 [-.087,.029] over all steps, and the all-step within-task

<table><tr><td></td><td>Win</td><td>Emit</td><td colspan="2">Within-task AUROC</td></tr><tr><td>Arm</td><td>rate</td><td>rate</td><td> $k \leq 2$ </td><td>all steps</td></tr><tr><td>Untrained</td><td>0.146</td><td>0.971</td><td>0.476 [.433,.526]</td><td>0.697 [.668,.727]</td></tr><tr><td>Brier+φ</td><td>0.262</td><td>0.979</td><td>0.506 [.464,.548]</td><td>0.668 [.633,.703]</td></tr><tr><td>Format-only</td><td>0.172</td><td>0.744</td><td>0.517 [.488,.545]</td><td>0.684 [.647,.720]</td></tr></table>

Table 9: Online RL training results on TextWorld: win and emission rates over held-out tasks, and within-task AUROC by window. The paired early-window gain is in Figure 4.

AUROC is unchanged (0.697 → 0.668), so discrimination is concentrated in the early window, as in ALFWorld.

The early-window improvement might instead reflect de-censoring, where training raises the fraction of early steps that emit a confidence tag, rather than the calibration term in isolation. We probe this with a format-only control but cannot cleanly separate the two. Pooled over all emitting steps the three-way comparison is null (base 0.723, full 0.702; flip full−base −0.021 [-.052,.011]), so the pooled signal is not a censoring artifact. For the early-window gain we evaluate the control at an emission-matched checkpoint (tag rate 0.967, within 0.012 of both the base and full arms): neither the de-censoring component (format − base, +0.006 [-.020,.034]) nor the calibration component (full − format, −0.027 [-.055,.000]) is individually resolvable. By the trained endpoint the control’s own format correctness rate falls to 0.744, which confounds a like-for-like comparison. We therefore report the early-window transfer gain (withintask AUROC +0.265 [.107,.428] at step 0 and +0.070 [.010,.130] over steps 0–2) without attributing it to decensoring or calibration in isolation.

WebShop. On a live, on-policy WebShop run with the binary won reward, held-out episode success stays near the floor for both arms (0.9% untrained, 1.6% Brier+φ), leaving too little success variation to estimate a within-task early-window gain: only two to three held-out tasks contain both a successful and a failed seed, so the within-task AUROC is not meaningfully identified. Step-level pooled discrimination is 0.825 [.785,.889] untrained and 0.877 [.784,.951] trained, with overlapping intervals due to insufficient held-out samples.

## E Across Model Size

We repeat the online ALFWorld comparison on Qwen2.5-14B-Instruct, evaluating the untrained agent and a Brier+φ checkpoint under the same held-out protocol as the 7B experiments. Episode success is 36.9% untrained and 38.1% trained, with format parsing rate of 1.000 in both experiments. The pooled matched-emission AUROC is 0.697 [.615,.781] untrained and 0.729 [.635,.818] trained; the paired (trained-untrained) ∆ over the shared heldout tasks is +0.032 [-.016,.073], directionally consistent with the 7B pooled gain. The within-task earlywindow estimand is not identifiable here, as only six untrained and seven trained held-out tasks contain both a successful and a failed seed, too few for a reliable paired within-task delta. At 14B the trained agent still beats its untrained counterpart on both episode success and pooled AUROC, in the same direction as the 7B gain, so the pooled effect does not vanish with scale. Resolving the within-task estimand at this scale would require more held-out tasks across additional sizes and model families.

## F Calibration and Selective Execution

Tables 10 and 11 give the calibration and selectiveexecution metrics for all experiments with paired deltas, and Table 12 gives the untrained block at two model sizes. Full-coverage step-level success is 0.046–0.057 over all steps and 0.186–0.203 at $k \leq 2 ,$ and task-level triage (success if only the top 50% of tasks by mean early confidence are attempted) is 0.362 / 0.354 / 0.352 / 0.349 / 0.354 / 0.362 in row order. The task-level triage deltas (format-only −0.008, calibration arms −0.013 to 0.000) and the task-level AUROC deltas (−0.008 to +0.098) all have CIs including 0.

## G Details on Offline RL Training

Pooled AUROC intervals. The 95% step-level bootstrap intervals for Table 1, untrained vs. trained, are: ALFWorld 0.517 [.496,.538] vs. 0.681 [.662,.699] at step 300 (0.689 [.671,.706] at step 150); TextWorld 0.578 [.547,.611] vs. 0.722 [.692,.753]; Web-Shop 0.462 [.396,.550] vs. 0.850 [.773,.910] (18–23 successes, hence the wide interval). Untrained-totrained top-10% tail success lift is 1.55×→4.33× (ALFWorld; 2.73× at step 150), 1.57×→4.35× (TextWorld), and 1.11×→4.35× (WebShop).

Additional signals on the untrained rows. Negative mean token entropy gives AUROC 0.741 [.722,.762] (ALFWorld), 0.667 [.630,.703] (TextWorld), and 0.806 [.724,.881] (WebShop), with top-10% multipliers 4.22×, 2.97×, and 3.34×. Sequence logprobability gives AUROC 0.706 [.685,.726] (ALF-World), 0.593 [.560,.627] (TextWorld), and 0.762 [.653,.858] (WebShop), with top-10% multipliers 3.10×, 1.91×, and 3.34×. Negative last-token entropy gives 0.531, 0.493, and 0.401, with multipliers 0.81×, 1.06×, and 0.0×. Untrained confidence on emitted steps only gives 0.610 [.579,.639] and 1.93× in ALFWorld (6,184 steps / 410 successes) and 0.619 [.586,.652] and 1.61× in TextWorld (2,383 / 180); the WebShop emitted subset (265 steps, 2 successes) is not estimable.

Texture control. The AUROC of the OLS prediction of success from step depth and log(1 + content tokens) alone is 0.769 / 0.849 / 0.858 on the untrained rows and 0.787 / 0.855 / 0.799 on the trained rows (ALFWorld / TextWorld / Web-Shop). Residualized untrained confidence gives 0.534 [.510,.557] / 0.479 [.440,.516] / 0.509 [.386,.632]; residualized untrained token entropy 0.707 [.687,.727] / 0.586 [.549,.623] / 0.810 [.729,.882]; residualized trained confidence 0.626 [.603,.649] / 0.677 [.638,.715] / 0.811 [.712,.887].

White-box signals on the trained rows. On the trained models’ rows, ALFWorld (step 150) token entropy gives 0.660 and sequence log-probability 0.654 against trained confidence 0.689; TextWorld entropy 0.399 against 0.722; WebShop entropy

<table><tr><td>Arm</td><td>Mean conf all / k ≤ 2</td><td>ECE all</td><td> $\mathrm { E C E } k \leq 2$ </td><td>Brier all</td><td colspan="5">Brier k ≤ 2 Top-10% k ≤ 2 Top-25% k ≤ 2 Top-10% all Top-5% all</td></tr><tr><td>Untrained</td><td>0.475 / 0.477</td><td>0.419 [.398,.438]</td><td>0.277 [.217,.330]</td><td>0.250 [.237,.264]</td><td>0.229 [.212,.248]</td><td>0.493 [.323,.675]</td><td>0.363 [.241,.475]</td><td>0.241 [.151,.327]</td><td>0.241 [.151,.333]</td></tr><tr><td>Format-only</td><td>0.474 / 0.476</td><td>0.429 [.408,.447][.236,.347]</td><td>0.290</td><td>0.253 [.239,.267]</td><td>0.229 [.212,.248]</td><td>0.465 [.296,.626]</td><td>0.363 [.220,.464]</td><td>0.193</td><td>0.182 [.117,.272] [.114,.275]</td></tr><tr><td>Brier+φ</td><td>0.161 / 0.189</td><td>0.115</td><td>0.051 [.105,.124] [.041,.084] [.045,.061]</td><td>0.052</td><td>0.137 [.107,.168]</td><td>0.561 [.391,.748]</td><td>0.444 [.299,.599]</td><td>0.303 [.192,.428] [.326,.655]</td><td>0.491</td></tr><tr><td>Balanced Brier</td><td>0.163 / 0.183</td><td>0.114 [.102,.125]</td><td>0.054 [.042,.097]</td><td>0.054 [.046,.064]</td><td>0.139 [.108,.173]</td><td>0.609 [.413,.765]</td><td>0.444 [.293,.590]</td><td>0.290 [.182,.417]</td><td>0.469 [.303,.626]</td></tr><tr><td>Brier+dispersion</td><td>0.085 / 0.110</td><td>0.054</td><td>0.116 [.050,.059] [.066,.164] [.034,.054]</td><td>0.044</td><td>0.140 [.101,.178]</td><td>0.635 [.426,.804]</td><td>0.446 [.307,.585]</td><td>0.272</td><td>0.480 [.167,.395] [.295,.652]</td></tr><tr><td>Balanced+dispersion 0.082 / 0.100</td><td></td><td>0.053</td><td>0.133 [.049,.057] [.085,.186] [.037,.059]</td><td>0.047</td><td>0.157 [.118,.199]</td><td>0.657 [.448,.818]</td><td>0.415 [.281,.557]</td><td>0.273 [.173,.398] [.308,.673]</td><td>0.490</td></tr></table>

Table 10: Calibration and selective execution for every online training on Qwen2.5-7B-Instruct experiment; same setting as Table 7. ECE uses 10 equal-width bins, top-q% is the success rate among the q% most confident steps in the window, and intervals are 95% task-level cluster bootstraps.

<table><tr><td>Arm</td><td>ECE all</td><td> $\mathrm { E C E } k \leq 2$ </td><td>Brier all</td><td>Brier  $k \leq 2$ </td><td>Top-10%  $k \leq 2$ </td><td>Top-25%  $k \leq 2$ </td><td></td><td></td><td>Top-10% all Triage top-50% Task AUROC</td></tr><tr><td>Format-only</td><td>+0.010 [-.001,.021]</td><td>+0.013 [-.005,.030]</td><td>+0.003 [-.005,.010]</td><td>-0.001 [-.008,.006]</td><td>-0.024 [-.094,.037]</td><td>-0.012 [-.050,.036]</td><td>-0.047 [-.084,-.012]</td><td>-0.008 [-.034,.018]</td><td>+0.076 [-.013,.176]</td></tr><tr><td>Brier+φ</td><td>-0.304 [-.318,-.286][-.271,-.147][-.213,-.180]</td><td>-0.226</td><td>-0.198</td><td>-0.093 [-.126,-.057]</td><td>+0.080 [-.000,.185]</td><td>+0.081 [.038,.160]</td><td>+0.061 [.018,.128]</td><td>-0.010 [-.026,.008]</td><td>-0.008 [-.127,.111]</td></tr><tr><td>Balanced Brier</td><td>-0.305 [-.319,-.288][-.273,-.126] [-.213,-.180] [-.125,-.054]</td><td>-0.224</td><td>-0.196</td><td>-0.090</td><td>+0.115 [.035,.193]</td><td>+0.076 [.042,.155]</td><td>+0.050 [.007,.111]</td><td>-0.013 [-.034,.010]</td><td>+0.015 [-.092,.124]</td></tr><tr><td>Brier+dispersion</td><td>-0.364 [-.382,-.346] [-.270,-.057][-.225,-.187] [-.131,-.049]</td><td>-0.162</td><td>-0.206</td><td>-0.089</td><td>+0.159 [.054,.224]</td><td>+0.092 [.041,.143]</td><td>+0.035 [-.012,.095]</td><td>-0.008 [-.031,.016]</td><td>+0.098 [-.006,.210]</td></tr><tr><td>Balanced+dispersion</td><td>-0.366 [-.384,-.346][-.239,-.033][-.221,-.184][-.114,-.025]</td><td>-0.144</td><td>-0.203</td><td>-0.072</td><td>+0.167 [.068,.235]</td><td>+0.095 [.019,.119]</td><td>+0.033 [-.013,.105]</td><td>0.000 [-.023,.018]</td><td>+0.047 [-.064,.155]</td></tr></table>

Table 11: Paired deltas (trained - untrained) of the calibration and selective-execution metrics; same setting as Table 7. Task AUROC = AUROC of mean early confidence against “task solved at least once”.

0.430 against 0.850. Mean content-token entropy on ALFWorld falls from 0.145 to 0.025.

ALFWorld checkpoint curve. On the same heldout rows, pooled AUROC raw → residualized is 0.517 → 0.534 untrained; 0.690 → 0.634 at step 100; 0.689 → 0.626 at 150; 0.659 → 0.534 at 200; 0.677 → 0.612 at 250; 0.681 → 0.633 at 300. Success among the top 5% of steps goes from 0.137 untrained to 0.310 at step 300, and among the top 20% from 0.109 to 0.181; at step 150, success among the top 25% is 0.165 against 0.096 untrained.

Separate held-out probes. A second set of Web-Shop and TextWorld probes uses a different heldout split and counts WebShop wins differently (96 of 1,303 steps), so its values are not comparable to Table 1. On that split: WebShop untrained AU-ROC 0.518 [.485,.554] with ECE 0.452; ranking reward at 150 steps 0.728 [.676,.778], ECE 0.185, top-10% success 0.162 (2.2× of 0.074); ranking + spread w = 0.10 at 150 steps 0.795 [.738,.844], ECE 0.146, top-10% 0.369 (5.0×). TextWorld untrained 0.522 [.438,.610] with ECE 0.586; ranking at 150 steps 0.775 [.690,.853], ECE 0.100, top-10% 0.333 (5.4× of 0.062), on 600 steps with 37 successes.

## H Ablations

Learning rate and probe temperature. The Brier arm’s held-out WebShop AUROC rises steeply near a learning rate of $2 \times 1 0 ^ { - 6 }$ and plateaus by $5 \times 1 0 ^ { - 6 }$ at about 0.86 (one seed; no interval). Reading its confidence greedily rather than at T = 0.8 barely moves it: 0.863 [.819,.901] greedy against 0.846 [.795,.892] at $T = 0 . 8$ (emitted steps, n = 217 and 219), and 0.864 under the pooled convention of Table 2 (missing tag scored 0.5) for the same checkpoint.

Table 13 gives the spread-weight sweep. The second greedy w = 0.10 value is the run reported as GRPO seed 1 in Table 3, and a third run of that configuration gives 0.613 (GRPO seed 0 there); the same sweep on ALFWorld offline (checkpoint 150, 176 held-out prefixes / 58 successes, 3 seeds, 15 cells) gives 0.760–0.822 with overlapping intervals. Reliability by bin for that sweep (K = 8 sampled decodes at $T = 1 . 0$ , emitted decodes only). At w = 0: confidences ≥ 0.8 hold 0.39 of the mass with mean 0.86 and success 0.33; $0 . 2 \leq c < 0 . 8$ holds 0.60 with mean 0.61 and success 0.20; c < 0.2 holds 0.01. At w = 0.10: 0.29 / 0.86 / 0.37; 0.70 / 0.59 / 0.22; 0.01. The untrained model at T = 1.0 emits a tag on 17% of decodes (300 of

<table><tr><td></td><td>Qwen2.5-7B-Instruct</td><td>Qwen2.5-14B-Instruct</td></tr><tr><td>Mean conf, all  $/ k \leq 2$ </td><td>0.475 / 0.477</td><td>0.534 / 0.646</td></tr><tr><td>Step success, all  $/ k \leq 2$ </td><td>0.057 / 0.201</td><td>0.112 / 0.289</td></tr><tr><td>ECÉ, all</td><td>0.419 [.398,.438]</td><td>0.425 [.388,.458]</td></tr><tr><td> $\mathrm { E C E } , k \leq 2$ </td><td>0.277 [.217,330]</td><td>0.357 [.296,.417]</td></tr><tr><td>Brier, all</td><td>0.250 [.237,.264]</td><td>0.303 [.281,.328]</td></tr><tr><td>Brier,  $k \leq 2$ </td><td>0.229 [.212,.248]</td><td>0.310 [.283,.336]</td></tr><tr><td> $\mathrm { T o p - } 1 0 \% \ \mathrm { s u c c . } , k \leq 2$ </td><td>0.493 [.323,.675]</td><td>0.717 [.556,.839]</td></tr><tr><td> $\mathrm { T o p } { - } 2 5 \% \mathrm { s u c c } . , k \leq 2$ </td><td>0.363 [.241,.475]</td><td>0.571 [.406,.724]</td></tr><tr><td> $\mathrm { T o p } { - } 1 0 \% \ \mathrm { s u c c } . , \mathrm { a l l }$ </td><td>0.241 [.151,.327]</td><td>0.275 [.192,.375]</td></tr><tr><td>Triage, top-50% tasks</td><td>0.362 [.245,.497]</td><td>0.464 [.328,.591]</td></tr><tr><td>Triage, top-25% tasks</td><td>0.557 [.375,.714]</td><td>0.755 [.521,.964]</td></tr><tr><td>Task-level AUROC</td><td>0.680 [.578,.787]</td><td>0.756 [.663,.842]</td></tr><tr><td>Tasks solved at least once</td><td>38/128</td><td>51/128</td></tr></table>

Table 12: Calibration and selective execution of the untrained Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct agents on the held-out set. Intervals are 95% task-level cluster bootstraps.

<table><tr><td>w</td><td>AUROC, sampled (per seed)</td><td>Mass in [0.2, 0.8]</td><td>Mass ≥ 0.8</td><td>AUROC, greedy (2 seeds)</td></tr><tr><td>0.00</td><td>0.746 (0.717, 0.775)</td><td>0.64</td><td>0.34</td><td>0.720 / 0.762</td></tr><tr><td>0.05</td><td>0.678 (0.679, 0.676)</td><td>0.76</td><td>0.23</td><td>0.690 / 0.698</td></tr><tr><td>0.10</td><td>0.675 (0.675)</td><td>0.74</td><td>0.25</td><td>0.663 / 0.676</td></tr><tr><td>0.15</td><td>0.584 (0.594, 0.574)</td><td>0.88</td><td>0.12</td><td>0.593 / 0.597</td></tr><tr><td></td><td>0.30 0.556 (0.587, 0.525)</td><td>0.93</td><td>0.08</td><td>0.568 / 0.520</td></tr></table>

Table 13: Spread-weight sweep, WebShop offline RL training, ranking reward, evaluated on the held-out prefixes of Table 2; sampled columns use K = 8 decodes per prefix at temperature $T = 1 . 0$ , top-p 0.95, seedaveraged $( w = 0 . 1 0$ has one sampled seed), and the interior mass column is the closed interval [0.2, 0.8] (so $c = 0 . 8$ is counted in both mass columns).

1,760), with $c \ge 0 . 8$ holding 0.47 of the emitted mass at success 0.31. Emission is 42% at $w = 0$ and 50% at $w = 0 . 1 0$

## I Hyperparameters, Prompt & Rewards

## Prompt (used in both training & evaluation).

You are an agent solving a household task in a text world. Each turn you are given the task, a log of what you have already done this episode, what you currently observe, and the list of admissible actions. Choose exactly ONE admissible action that best progresses toward the task. Do not repeat actions that did not help. First output the chosen action text on its own line, copied verbatim from the admissible list, with no explanation. Then on the NEXT line output your confidence that THIS EPISODE will ultimately succeed, as an integer 0-100 wrapped exactly as <conf>NN</conf>. Output nothing else.

Models. Qwen2.5-7B-Instruct and Qwen2.5- 14B-Instruct; LoRA rank 32, α = 64 for all training.

Online training. 120 iterations $\times ~ 4$ tasks $\times ~ 6$ attempts (the group); advantage per step as in

Eq. (2); reward as in Eq. (3); learning rate $1 0 ^ { - 6 }$ PPO-style clip 0.2; KL-to-reference coefficient $1 0 ^ { - 3 } ;$ gradient-norm clip 1.0; micro-batch 1; checkpoints every 20 iterations; sampling temperature 1.0; up to 30 steps and 48 new tokens per step. Arms: format-only $( w _ { \mathrm { c a l } } = 0 , w _ { \varphi } = 0 )$ ; Brier+φ $( w _ { \mathrm { c a l } } = 1 , \varphi = 4 c ( 1 - c ) , w _ { \varphi } = 0 . 1 0 ) ;$ balanced Brier $( w _ { \mathrm { c a l } } = 1$ , Brier reweighted with $0 . 5 / \bar { Y } _ { s }$ on wins and $0 . 5 / ( 1 - \bar { Y } _ { s } )$ on losses, $\varphi = 4 c ( 1 - c )$ $w _ { \varphi } ~ = ~ 0 . 1 0 )$ ; Brier+dispersion $( w _ { \mathrm { c a l } } = 1$ , φ replaced by $\operatorname* { m i n } ( 1 , 4 ( c - \bar { c } _ { s } ) ^ { 2 } ) , w _ { \varphi } = 0 . 1 0 ) ;$ balanced+dispersion (both). Training tasks: 45 mixedoutcome tasks from seeds 18–245 (7B). Validation seeds 500–563; test seeds 1000–1127. All disjoint.

Offline training. GRPO on one recorded trajectory per game from the untrained 7B policy, cut into (prefix → recorded action, outcome) rows; balanced parquets ALFWorld 1,068 rows / 200 games, TextWorld 943 / 200, WebShop 1,066 / 372 (won rate 0.35 after balancing; raw WebShop 5,191 rows / 400 games / 7.2% won); no game has mixed outcomes. 8 generations per prefix; temperature 0.8; maximum completion 256 tokens; AdamW with a linear schedule; learning rate $5 \times 1 0 ^ { - 6 } ~ ( 1 0 ^ { - 6 }$ was too cold and produced artefacts); batch 8; KL coefficient $1 0 ^ { - 3 }$ ; 600 steps for the WebShop ablations (checkpoint 600) and 150 for the multienvironment checkpoints. Reward = 0.5 actionmatch + 0.2 format + 0.3 calibration, plus w φ(c) in the spread sweep; the reward-form arms of Table 2 use $w _ { \mathrm { c a l } } = 1 . 0$ in place of the 0.3 default. Held-out probes decode greedily (temperature 0, 64 new tokens) unless stated; the spread sweep and its reliability analysis use $K = 8$ sampled decodes at $T = 1 . 0 , \mathrm { t o p } { \cdot } p 0 . 9 5$

Calibration reward forms. With c the parsed confidence and Y the outcome (ϵ, the sigmoid gain 10, the abstain utility 0.5, and the EMA decay 0.99 are implementation constants of the reward code; $\tau = 0 . 1 )$ : Brier $1 - ( c - Y ) ^ { 2 } \colon$ ; logarithmic, normalized to $[ 0 , 1 ] , 1 + \lceil Y \log c + ( 1 -$ $Y ) \log ( 1 - c ) ] / ( - \log \epsilon )$ with c clipped to [ϵ, 1−ϵ] and $\epsilon = 1 0 ^ { - 3 } $ ; spherical $q _ { Y } / \sqrt { c ^ { 2 } + ( 1 - c ) ^ { 2 } }$ with $q _ { Y } ~ = ~ c { \mathrm { ~ i f ~ } } Y ~ = ~ 1$ and $1 - c$ otherwise; L1 $1 - | c - Y | ;$ decision/selective $g Y + ( 1 - g ) \cdot 0 . 5$ with $g = \sigma \big ( 1 0 ( c - 0 . 5 ) \big )$ , so the optimum commits when $P ( Y = 1 \mid h ) > 0 . 5 ;$ ranking surrogate $\sigma \big ( ( c - \mu _ { \mathrm { l o s e } } ) / \tau \big )$ for a winning completion and $\sigma \big ( ( \mu _ { \mathrm { w i n } } - c ) / \tau \big )$ for a losing one, where $\mu _ { \mathrm { { w i n } } }$ and µ<sub>lose</sub> are exponential moving averages (decay 0.99) of the confidence of the opposite class and $\tau = 0 . 1$

Reward-form and estimator details (Tables 2 and 3). Across seeds the proper forms overlap— Brier spans 0.823–0.864 and spherical 0.852– 0.882, of the same order as the between-form spread—so the six forms are indistinguishable at this seed count; the paired Brier-minus-ranking gap is +0.019 [-.024,.064] on seed 0, where the top-25% step-success gap is +0.018 [-.145,.073]. Task reward only gives 0.559 and 0.555 at format rates 0.56 and 0.59 over two seeds, against untrained 0.502 at 0.08. Table 3 carries the spread term at $w = 0 . 1 0$ and so is not comparable to the ranking run of Table 2 (0.845); the spread sweep of the same reward gives 0.663/0.676 at that weight (Appendix H). PPO’s three 0.500 runs (seed 2 at a 10× value-head learning rate) reflect a cold-start value head: large, noisy advantages for roughly the first 15 updates, tag emission collapsing to a constant, and zero withingroup reward variance.

## J Computational Budget and Reproducibility

Models and parameters. All training uses Qwen2.5-Instruct backbones at 7B and 14B parameters (Yang et al., 2024), adapted with LoRA (Hu et al., 2022) at rank 32 and α = 64 (Appendix I); only the adapter parameters are updated.

Hardware and compute budget. All models were trained on a single node with 8× NVIDIA H200 GPUs. Each run is a single-GPU LoRA job (adapters only), with experiments scheduled concurrently across the eight GPUs. Representative single-GPU wall-clock is about 8 h for a 7B online ALFWorld experiment, 12 h for the 14B experiment, 5 h for a TextWorld experiment, 2 h for WebShop, and roughly 10 min for an offline GRPO reward-form computation. By category: the offline GRPO reward-form and estimator sweeps total ≈2 GPU-hours; the online 7B ALFWorld arms (six reward forms plus four additional-seed runs) ≈79 GPU-hours; the TextWorld and WebShop online runs ≈18 GPU-hours; and the cross-size experiments ≈49 GPU-hours. Including preliminary and failed runs, total training compute is approximately 200 GPU-hours.

Software. PyTorch 2.13.0 (CUDA 13.0), Transformers 5.17.0, TRL 1.14.0, PEFT 0.21.0, Accelerate 1.15.0, Datasets 5.0.1, and NumPy 2.5.3. Offline training uses TRL GRPO; the online route uses the in-house on-policy learner (Appendix I).

Runs and statistical reporting. Each online training experiment uses a single training seed (Section 5); the offline reward-form and estimator comparisons use the seed counts stated in Appendix I and Table 13. All reported intervals are 95% tasklevel cluster bootstraps over held-out tasks. Training, validation, and test seeds are disjoint (Appendix I); held-out evaluation seeds never appear in training.

Reproducibility. Every ingredient needed to reproduce our results is specified in the paper: the on-policy learner configuration and full hyperparameters, the agent prompt template with the inline <conf>NN</conf> format, and the exact per-arm reward definitions (Appendix I); the pre-registered estimand (within-task AUROC at $k ~ \leq ~ 2 )$ , the checkpoint-selection rule, and the disjoint train/validation/test seed ranges (Section 5); and the 95% task-level cluster-bootstrap procedure used for all reported intervals. We provide anonymized training and evaluation code, configurations, and prompts in the supplementary material to reproduce every experiment.

## K Scientific Artifacts and Licenses

Our experiments build on three agentic environments and the Qwen2.5-Instruct model family (Yang et al., 2024), adapted with LoRA (Hu et al., 2022) using the TRL library (von Werra et al., 2020) for training. ALFWorld (Shridhar et al., 2021), TextWorld (Côté et al., 2019), and Web-Shop (Yao et al., 2022) are released under the MIT License, and TRL under the Apache-2.0 License. Among the models, Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct are released under the Apache-2.0 License. We use each artifact solely for research, consistent with its intended use as an environment or model for studying language agents, and within the terms of its license. The three environments are standard, publicly available English-language task suites (household and textadventure navigation and simulated web shopping) that carry no personally identifying information and no offensive content.

## L Human Subjects

Our study involves no human annotators, participants, or user-generated data.

## M Use of AI Assistants

We used the AI-based coding and writing assistant Claude Code with the Claude-4.8-Opus model for routine code implementation and language editing for the paper.