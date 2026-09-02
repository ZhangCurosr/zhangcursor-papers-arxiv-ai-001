# Beyond the Clock: Measuring the Value of Adaptive Revision

Ayushi Chadha ayushichadha48@gmail.com

## Abstract

As agentic systems become compound systems, increasingly important decisions move above task execution itself: when should a higher-level controller preserve the strategy guiding another process, and when should it revise it? We study this meta-level control problem in a hierarchical latent reasoner whose manager can retain or replace a commitment governing lower-level computation. Across three precommitted training seeds, learned revision timing produces qualitatively different policies, ranging from an almost deterministic early clock to substantially more state-conditioned schedule distributions, yet none outperforms the best forced timing policy evaluated on the same frozen checkpoint. This separates state dependence from decision value: a controller can vary its actions with internal state without turning that variation into a reproducible task-performance benefit. A deeper intervention study on the original checkpoint shows that timing itself is consequential and order-sensitive, while exhaustive enumeration reveals that a strong fixed schedule captures most of the measurable value available from timing at this decision budget. Counterfactual PERSIST/REPLAN diagnostics further show why score-level evidence can be misleading when predictability is dominated by decision position rather than within-position discrimination. Together, these results argue that learned meta-level control should be evaluated along three separate axes: whether its score depends on state, whether that dependence changes realized behavior, and whether those changes capture outcome value beyond a strong non-adaptive policy.

## 1 Introduction

Compound AI systems increasingly contain decisions that sit above ordinary task execution. A higher-level process may decide whether to preserve or revise a prompt, trajectory, program, subgoal, memory state, or agent harness. Recent meta-level optimizers make this structure explicit: GEPA learns from system trajectories to revise prompts (Agrawal et al., 2025), AlphaEvolve repeatedly proposes and evaluates program modifications (AlphaEvolve Team, 2025), and Meta-Harness searches over executable harnesses using scores and traces from prior candidates (Lee et al., 2026). These systems differ substantially, but they share a basic problem: a meta-level decision must be assigned credit for its downstream consequences. The quality of the score used to train or rank that decision is therefore not merely an implementation detail.

This paper isolates that problem in a setting where the relevant decisions can be measured directly. We use a compact hierarchical latent reasoner derived from the Hierarchical Reasoning Model (HRM) (Wang et al., 2025). A slow manager emits a directional latent commitment that biases a faster worker for multiple refinement passes. The meta-level action is deliberately simple: after observing the current pass, should the manager retain the active commitment or replace it for subsequent computation? This turns subgoal persistence into a small supervisory-control problem with a fixed compute budget and an exactly enumerable timing space.

The study began with a natural hypothesis: a state-conditioned trigger should allocate a limited revision budget more intelligently than a fixed clock. A first 400-step run instead collapsed almost completely to the earliest eligible second intervention. We then preregistered two additional adaptive seeds under the same training, budget-calibration, and evaluation protocol and retained all outcomes regardless of direction. They did not reproduce the same clock: one produced a high-entropy mixture of schedules and the other mostly retained the initial commitment. What did replicate was more consequential: in no seed did the learned adaptive policy outperform the best forced timing policy on the same frozen checkpoint. Thus the replicated failure is not universal clock collapse, but a lack of reproducible adaptive value despite qualitatively different degrees of state-conditioned behavior.

This distinction motivates the central claim of the paper: state dependence is not adaptation, and adaptation is not yet utility. A score may vary with internal state without changing actions; actions may vary substantially across episodes without improving outcomes; and even a genuinely state-aware policy may have little value if a strong fixed decision already captures most of the available gain. These possibilities are easy to conflate when only the controller’s score or schedule diversity is inspected.

Our contributions are fourfold. First, we provide a three-seed, precommitted matched-budget replication of learned revision timing and show that policy form is seed-sensitive while controlled adaptive advantage is absent in all three runs. Second, we use frozen-checkpoint interventions to localize the seed-0 failure to timing rather than generally weak representations. Third, we exhaustively enumerate the complete two-intervention timing space on that checkpoint and quantify both order effects and the oracle ceiling on episode-specific adaptation. Fourth, we use matched counterfactual PERSIST/REPLAN branches to separate cross-position predictability from withinposition decision signal. The resulting protocol, comprising replication, behavioral audit, frozen intervention, exhaustive decision-value measurement, and counterfactual branching, is intended as a diagnostic template for meta-level control, not as a claim that this small internal manager is itself a full meta-agent.

## 2 Supervisory Replanning in a Hierarchical Reasoner

We study supervisory revision in a compact two-level latent reasoner built on HRM (Wang et al., 2025). The architecture is used as a controlled testbed rather than as a claim about the necessity of hierarchy: it exposes a persistent, addressable latent commitment whose timing can be intervened on exactly. A slow manager proposes the commitment; a fast worker performs the task-conditioned refinement that consumes it.

## 2.1 Manager–worker commitments

Let $m \in \{ 1 , \ldots , M \}$ index outer refinement passes. The model maintains high- and low-level hidden sequences $z _ { m } ^ { H ^ { \prime } } , z _ { m } ^ { L }$ . From the post-pass high-level representation $h _ { m } = z _ { m } ^ { H } [ : , 0 ]$ ], the manager produces a normalized directional goal and a scalar gate,

$$
\tilde { g } _ { m } = s _ { g } \frac { W _ { g } h _ { m } + b _ { g } } { \operatorname* { m a x } ( \lVert W _ { g } h _ { m } + b _ { g } \rVert _ { 2 } , \epsilon ) } ,\tag{1}
$$

$$
\alpha _ { m } = \sigma \big ( ( w _ { \alpha } ^ { \top } h _ { m } + b _ { \alpha } ) / \tau \big ) .\tag{2}
$$

The goal is projected back into the shared hidden space and injected into subsequent low- and highlevel updates with strength $\alpha _ { m }$ . The pooled worker representation is $\begin{array} { r } { w _ { m } = \dot { T } ^ { - 1 } \sum _ { t } z _ { m , t } ^ { L } . } \end{array}$ . When a commitment is emitted after pass s, the contemporaneous worker representation is stored as an anchor $a = w _ { s }$ <sub>s</sub>. On later passes consuming that commitment, the directional auxiliary objective measures displacement relative to the anchor,

$$
\ell _ { m } ^ { F } = A _ { m } \alpha _ { m } \left[ 1 - \cos ( w _ { m } - a _ { m } , g _ { m } ) \right] ,\tag{3}
$$

where $A _ { m }$ masks passes on which no causal commitment has yet been consumed. The goal, gate, and anchor persist until the next manager intervention. The dwell of a commitment is therefore the number of worker passes for which this tuple remains active. This construction follows the directional-control lineage of FeUdal Networks (Vezhnevets et al., 2017), while our question concerns persistence and revision rather than reward-maximizing hierarchical control.

![](images/a9b771fb7f7523ec54db4acd30aa149e83936bfdf8bc2e6d7b8325ce8ee07d16.jpg)  
Figure 1: Manager–worker timing. A decision made after pass m changes the commitment first consumed on pass $m + 1$ . Bottom: [1, 2] and [1, 5] use the same $K = \bar { 2 }$ intervention budget but allocate persistence differently.

## 2.2 Causal intervention timing

All primary experiments execute a fixed M = 8 outer passes. At pass m, the worker first consumes the commitment already stored in recurrent carry; the hidden states are updated; only then does the manager observe the post-pass state and decide whether to retain or replace the commitment. A replacement decided after pass m is first consumed on pass $m + 1$ and can never affect pass m itself. Pass 1 is a bootstrap pass followed by a mandatory initial emission. Optional revision is eligible after passes 2–7, and no emission is allowed after pass 8 because no subsequent worker computation could consume it.

We write a schedule as the ordered list of emission positions. Thus [1, 4] denotes the mandatory emission after pass 1 followed by one replacement after pass 4, yielding commitment dwells of 3 and 4 worker passes. Under $M = 8$ and exactly two emissions $( K = 2 )$ , the complete fixed-clock space is [1, k] for $k \in \{ 2 , 3 , 4 , 5 , 6 , 7 \}$ . Figure 1 makes the emission/consumption boundary explicit.

## 2.3 From a clock to a state-conditioned trigger

A fixed-period manager implements a temporal policy $u _ { m } = f ( m )$ , where $u _ { m } \in \{ 0 , 1 \}$ denotes replacement after pass m. The adaptive manager instead observes a compact detached state summary

$$
\phi _ { m } = \left[ c _ { m } , d _ { m } , \rho _ { m } , \tau _ { m } / M , q _ { m } , \alpha _ { m } \right] ^ { \top } ,\tag{4}
$$

where $c _ { m } = \cos ( w _ { m } - a _ { m } , g _ { m } )$ is cumulative directional progress, $d _ { m } = \cos ( w _ { m } - w _ { m - 1 } , g _ { m } )$ is latest-pass progress, $\rho _ { m } = c _ { m } - c _ { m - 1 }$ is its trend, $\tau _ { m } / M$ is normalized dwell, $q _ { m }$ is a bounded halt-versus-continue confidence signal, and $\alpha _ { m }$ is the active gate. All six features are detached before entering the trigger. An affine Bernoulli policy produces

$$
\beta _ { m } = \sigma ( w _ { \beta } ^ { \top } \phi _ { m } + b _ { \beta } ) , \qquad u _ { m } \sim \mathrm { B e r n o u l l i } ( \beta _ { m } )\tag{5}
$$

during training, while deterministic evaluation uses $u _ { m } = \mathcal { H } [ \beta _ { m } > \theta ]$ . The six-feature trigger is a project-specific synthesis: options and Option-Critic motivate persistent behavior with statedependent termination (Sutton et al., 1999; Bacon et al., 2017), while ACT, PonderNet, and Skip-RNN motivate learned continuation decisions (Graves, 2016; Banino et al., 2021; Campos et al., 2018). Here the persistent object is a latent directional commitment inside iterative reasoning.

## 2.4 One-consuming-pass credit

The causal boundary also constrains training. A decision after pass m first changes pass $m + 1$ but the recurrent training loop performs backward propagation and an optimizer update between outer refinement calls. We therefore do not carry an unrestricted graph through the full future

trajectory. Cached commitments are stored as detached values and, on the first consuming pass, a local value-preserving Jacobian is reconstructed:

$$
v _ { \mathrm { u s e d } } = v _ { \mathrm { c a c h e d } } + I _ { \mathrm { p e n d i n g } } \left( v _ { \mathrm { r e c o m p u t e d } } - \mathrm { s t o p g r a d } ( v _ { \mathrm { r e c o m p u t e d } } ) \right) .\tag{6}
$$

The added term is numerically zero in the forward pass, so the worker consumes exactly the cached commitment; backward propagation follows the locally recomputed quantity. The construction is applied to goal, gate, and adaptive retain/replace selection. Thus the intervention receives credit from thefirst worker pass that consumes it, not from unrestricted backpropagation through the complete refinement trajectory. This implementation is causally correct for the intended local credit path, but whether that horizon values revision in a way aligned with final task performance is an empirical question addressed in Section 5.

## 3 Experimental Protocol and Controlled Evaluation

Our design separates three quantities that are easily confounded in learned supervisory control: reasoner quality, intervention budget, and intervention timing. We first establish fixed-clock controls, then train an adaptive controller to the same optimization budget, calibrate its realized intervention count without using task performance, and finally freeze model parameters while intervening directly on timing.

## 3.1 Task, compute, and evidence hierarchy

All primary experiments use ConceptARC-mini with $M = 8$ outer passes. Fixing M ensures that changing a schedule changes when commitments are replaced rather than how much outer reasoning compute is available. The corrected fixed-period baseline contains 15 deterministic training runs, $P \in \{ 1 , 3 , 4 , 6 , \infty \}$ and seeds {0, 1, 2}, each trained for 400 optimizer steps. The adaptive controller is trained for 400 steps under three precommitted seeds. Seed 0 is the checkpoint for which we additionally performed the exhaustive six-clock sweep, oracle analysis, and counterfactual diagnostics; seeds 1 and 2 are independent replications of the matched-budget adaptive audit and frozen forced-schedule comparison.

The primary evidence is the 400-step matched-budget adaptive audit, the frozen-checkpoint schedule intervention, and the preregistered exhaustive $K = 2$ sweep. Earlier 96-step adaptive runs establish mechanism sanity, including gradient flow, deterministic score ordering, causal retain/replace execution, and threshold behavior, but are not used for adaptive-versus-fixed performance claims. The later counterfactual-v2 experiment is a diagnostic and stops before training a new policy.

## 3.2 Why $K = 2$ is the controlled budget

Across the three-seed fixed-period matrix, differences between persistence periods are small relative to across-seed variation, so the fixed sweep does not identify a universally optimal period. It does, however, contain a useful matched-count contrast: $P = 4$ and $P = 6$ both emit exactly $K = 2$ commitments per episode but place the second emission at different positions. $P = 6$ exceeds $P = 4$ on all three seeds, with a mean held-out token-accuracy difference of 0.00248. The effect is modest, but it isolates timing from intervention count and motivates asking whether the second intervention can be allocated from state rather than a clock.

## 3.3 Leakage-free budget calibration

The adaptive controller is trained with no explicit intervention penalty $( \eta = 0 )$ . During the 400-step run, stochastic training averages 3.945 total emissions per completed episode. At deterministic evaluation with the nominal $\theta = 0 . 5 ,$ it retains after the mandatory initial commitment. Because the learned score is reproducibly non-constant, we treat this as an operating-point problem rather than evidence that the trigger is dead.

For matched evaluation, θ is calibrated on a split disjoint from final evaluation using intervention count only,

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } | \bar { K } _ { \mathrm { c a l } } ( \theta ) - 2 | .\tag{7}
$$

No accuracy or loss enters threshold selection. The chosen $\theta ^ { * } = 0 . 4 9 8 2 0 0$ gives $\bar { K } _ { \mathrm { c a l } } = 2 . 0 0 0 0 0 0$ and, once frozen, $\bar { K } _ { \mathrm { f i n a l } } = 2 . 0 0 0 8 1 4$ on the ordered 3,686-episode final split. We therefore match the decision budget without treating $\beta _ { m }$ as a calibrated probability of replanning benefit.

## 3.4 Frozen timing interventions and exhaustive enumeration

The strongest control freezes the exact adaptive step-400 checkpoint and overrides only the position of its second commitment. Under forced [1, k], the model follows the normal causal trajectory and uses the manager-generated goal at the forced emission state; no parameter is retrained. This holds fixed the learned substrate and changes only the supervisory timing rule.

Because $M = 8$ and $K = 2$ , all six legal schedules can then be evaluated on the same frozen checkpoint and the same ordered episodes. Before running this sweep, we froze the checkpoint, final split, compute semantics, six schedules, headline micro-token metric, and reversed-dwell hypotheses. Schedule and oracle comparisons use 10,000 paired episode-bootstrap resamples. Micro token accuracy is the preregistered headline for the sweep; earlier matched-controller tables use episodeaveraged token accuracy and are labeled separately. Exact sequence accuracy is zero in this compact regime and is not used for schedule comparison.

The original fixed $P = 4$ and $P = 6$ runs did not persist final weights. We deterministically reran their recorded seed-0 commands; the resulting training summaries matched the original records to reported precision. The matched-controller values reported here, 0.481321 and 0.485595, come from re-evaluating those reconstructed checkpoints on the ordered 3,686-episode final split. All causal frozen-schedule claims instead use the explicitly saved and hashed adaptive checkpoint.

## 4 Learned Replanning Is State-Dependent but Not Reliably Better

## 4.1 Three precommitted seeds produce different policies

The three adaptive runs reach final episode-averaged token accuracies of 0.467015, 0.486149, and 0.474947, with held-out intervention counts $K = 2 . 0 0 0 8 , 1 . 9 6 7 4 .$ , and 1.9135. Table 1 isolates the controlled performance comparison; the score-structure diagnostics below characterize how each learned policy realizes that budget. Their realized policies differ qualitatively. Seed 0 is almost deterministic: 99.19% of episodes use [1, 2], schedule entropy is 0.0776 bits, position explains 96.76% of $\beta$ variance, and a time-only score has Spearman $\rho = 0 . 9 7 7 6$ with the full trigger. Seed 1 instead has 2.8598 bits of schedule entropy, only 60.77% position-explained variance, and timeonly/full-score $\rho = - 0 . 2 7 9 9$ . Seed 2 mostly emits only the mandatory initial commitment ([1] on 78.59% of episodes), but its score is likewise not clock-like under the same diagnostic (54.51% position-explained variance; $\rho = - 0 . 2 4 5 7 )$ . Thus the exact clock-collapse phenomenon is specific to seed 0; learned policy form is not stable across training seeds.

Table 1: Adaptive timing does not outperform the best forced schedule across seeds. Accuracy is episode-averaged token accuracy on the same frozen checkpoint; gap is adaptive minus best forced, in percentage points.
<table><tr><td>Seed</td><td>Adaptive acc. (%)</td><td>Best forced acc. (%)</td><td>Gap (pp)</td><td>Dominant policy</td></tr><tr><td>0</td><td>46.70</td><td>[1, 4] 48.75</td><td>-2.05</td><td>[1, 2] (99.2%)</td></tr><tr><td>1</td><td>48.61</td><td>[1, 6] 48.95</td><td>-0.34</td><td>[1] (34.2%)</td></tr><tr><td>2</td><td>47.49</td><td>[1, 2] 47.55</td><td>-0.06</td><td>[1] (78.6%)</td></tr></table>

## 4.2 State-conditioned behavior does not establish useful adaptation

On every frozen checkpoint, the adaptive policy falls below the best tested forced two-update schedule: by 2.0496, 0.3386, and 0.0568 percentage points for seeds 0–2, respectively (Table 1). Mean adaptive accuracy is 47.6037%, versus 48.4187% for the best within-checkpoint forced schedule. We do not treat this mean as a population estimate from three seeds; its role is descriptive. More importantly, seed 1 rules out the simplest interpretation of the original failure. Its controller is behaviorally diverse and substantially less position-dominated, yet that diversity does not produce a controlled accuracy advantage. Seed 2 is approximately neutral but likewise provides no positive evidence of useful timing. The replicated result is therefore narrower and stronger than “the controller becomes a clock”: state-conditioned schedules emerge, but useful state-conditioned timing does not replicate.

![](images/23be750c1aedbc4c67a4c82f3f07b29bcb8b01b4964b2951867e677e4111cc04.jpg)  
(a) Complete K = 2 timing sweep.

![](images/34b42a5a20cd477d7234dd4737f0144a6ce0db7add16903396f3b1658be52f47.jpg)  
(b) Preregistered reversed-order tests.  
Figure 2: Timing is consequential and order-sensitive. Left: performance has a narrow persistence sweet spot later than the learned controller’s near-universal [1, 2] replacement. Right: placing the longer dwell first wins all three reversed pairs; intervals are paired episode-bootstrap 95% CIs.

## 4.3 Seed 0 localizes a timing failure under frozen intervention

A cross-checkpoint comparison cannot determine whether the adaptive model simply learned representations for which early replacement is appropriate. We therefore force alternative schedules on the same adaptive weights. Forced [1, 2] reproduces the calibrated adaptive result (0.467002 episode-averaged token accuracy). Delaying the second emission to [1, 4] or [1, 6] raises accuracy to 0.487511 and 0.486622 respectively, with no retraining and the same $K = 2$ budget. The checkpoint therefore contains manager and worker representations capable of better performance under later intervention. The immediate failure is the controller’s choice of when to replace the commitment.

## 4.4 The complete timing landscape is narrow but order-sensitive

The exhaustive sweep evaluates [1, k] for every $k \in \{ 2 , \ldots , 7 \}$ on the frozen checkpoint. Micro token accuracy rises sharply as the first commitment is allowed to persist, peaks at [1, 5], and declines gently thereafter (Figure 2). The best observed fixed clock reaches 0.521206 micro accuracy. The numerical identity of [1, 5] is checkpoint-specific; the important fact is that the learned policy concentrates on an early region that the controlled endpoint measurement identifies as poor.

With seven consuming passes and $K = 2 ,$ reversed schedules exchange the same pair of dwell lengths: $[ 1 , 2 ]  [ 1 , 7 ] , [ 1 , 3 ]  [ 1 , 6 ] , \mathrm { a n d } [ 1 , 4 ]  [ 1 , 5 ]$ . Before evaluation we predicted equality within each pair if only the unordered dwell multiset mattered. All three predictions are rejected. Expressed as longer-first minus shorter-first micro accuracy, the differences are 0.00924, 0.00405, and 0.00097, with paired-bootstrap 95% intervals excluding zero in every case (Figure 2). Thus persistence duration alone does not determine value: where a duration occurs in the reasoning trajectory matters. Because the two dwells sum to a constant, this design does not separately identify causal first- and second-dwell effects; it establishes order sensitivity within the tested anti-diagonal

Across seeds, the learned policy does not recover a reproducible timing advantage, even though its behavioral form ranges from nearly deterministic to highly diverse. Seed 0 then provides a controlled mechanistic case study: forcing later timing on the same weights recovers performance, and exhaustive intervention locates its learned schedule on the wrong side of a non-monotonic timing landscape. We next ask why score-level evidence can still look informative without establishing useful adaptation.

## 5 Auditing the Supervisory Signal

## 5.1 State-dependent score variation is not yet adaptive behavior

An early 96-step checkpoint is useful as a mechanism audit because its score remains dispersed enough to inspect without retraining. At the natural $\theta = 0 . 5$ , all optional decisions are retained, yet a threshold sweep reveals 16.724% within-position score variance and, at intermediate operating points, up to 21 distinct schedules with 2.344 bits of schedule entropy. Repeated extraction reproduces the ranking exactly and higher-precision recomputation produces no strict reversals between distinct float32 score levels. The trigger therefore learned a stable relative ordering even though the default threshold yields retain-all behavior.

The step-400 checkpoint sharpens the distinction: score variation remains, but temporal structure now explains 96.760% of it and the calibrated policy executes [1, 2] almost universally. Hence

$$
\mathrm { s t a t e \mathrm { - } d e p e n d e n t \ s c o r e } \ \not = \ \mathrm { s t a t e \mathrm { - } d e p e n d e n t \ r e a l i z e d \ p o l i c y } .\tag{8}
$$

The relevant audit must therefore ask whether states at the same decision position are ranked usefully, not merely whether scores differ across the trajectory.

## 5.2 Matched counterfactual branching

We freeze the step-400 checkpoint and construct a local counterfactual target at every eligible state. From the identical detached latent carry, one branch keeps the old commitment and the other installs the candidate commitment. Each branch uses the selected commitment for one consuming transition followed by one Q-lookahead transition. For branch $b ,$ let

$$
\begin{array} { r } { J _ { b } = \ell _ { \mathrm { L M } , b } + \frac { 1 } { 2 } ( \ell _ { \mathrm { h a l t } , b } + \ell _ { \mathrm { c o n t i n u e } , b } ) + \lambda _ { F } \ell _ { F , b } , } \end{array}\tag{9}
$$

and define detached replanning advantage

$$
A = J _ { \mathrm { p e r s i s t } } - J _ { \mathrm { r e p l a n } } ,\tag{10}
$$

so $A > 0$ favors replacement. This is more causal than reading the learned trigger score because the two branches begin from the same state and differ only in commitment choice. It remains a bounded local surrogate, not full-episode return.

Across 64 episodes and 384 eligible states, PERSIST wins 90.365% of branches. Mean and median A are negative. A tiny critic using only cumulative progress and old-versus-candidate goal disagreement nevertheless predicts the target well in aggregate: held-out $R ^ { 2 } = 0 . 4 0 6$ , Spearman $\rho = 0 . 7 2 9$ , and sign accuracy 95.6%. Taken alone, these metrics would suggest a strong state signal.

## 5.3 Conditioning on position changes the conclusion

Pass position explains 79.464% of the critic’s prediction variance. A pass-only predictor already achieves reconstructed held-out $R ^ { 2 } = 0 . 2 9 2$ . After removing training-split pass means from features and target, adding the residual critic improves full-advantage prediction, but held-out $R ^ { 2 }$ on the residual target itself $\mathrm { i s - 0 . 0 0 7 }$

Classification exposes the same issue more starkly. Of 114 held-out decision states, only 12 favor REPLAN, and all 12 occur at pass 2. A pass-only predictor has no ability to rank states within a pass, yet achieves pooled AUROC 0.966. High aggregate AUROC can therefore be obtained by learning where in time favorable decisions occur while assigning identical scores to all states facing the same decision. Adding residual state information improves rare-state balanced accuracy, so state information is not absent; the evidence is simply too sparse and position-concentrated to justify a new end-to-end controller.

## 5.4 Credit horizon versus endpoint value

The local target and endpoint sweep measure different quantities, but their disagreement is precisely the design warning. The local diagnostic concentrates its positive replanning cases at the earliest eligible position, while final-outcome intervention shows that repeatedly selecting this early position is poor and that later fixed timing is substantially better on the same checkpoint. We therefore do not claim a pointwise theorem that every locally favorable revision is globally harmful. The supported conclusion is narrower and more useful:

$$
\mathrm { l o c a l l y ~ p r e d i c t i v e ~ r e v i s i o n ~ s i g n a l ~ \neq ~ c o r r e c t ~ e n d p o i n t ~ t i m i n g ~ p o l i c y . }\tag{11}
$$

Implementation correctness guarantees that one-consuming-pass credit reaches the trigger; it does not guarantee that this horizon values intervention on the horizon that ultimately matters. The diagnostic was therefore used as a stop rule: globally predictive but weak within-position evidence was not treated as sufficient reason to launch another adaptive training run.

## 6 How Much Is Adaptation Worth?

Finding a bad learned clock and a better fixed clock still does not establish that a more sophisticated adaptive controller is worth building. We therefore ask a separate question: after selecting a strong fixed timing policy, how much additional value remains available to episode-specific timing at all?

## 6.1 Random, fixed, and oracle timing

Let $a _ { k }$ denote micro token accuracy of fixed schedule [1, k]. A uniform-random clock chooses one of the six schedules independently of state, giving expected accuracy

$$
A _ { \mathrm { r a n d } } = { \textstyle \frac { 1 } { 6 } } \sum _ { k = 2 } ^ { 7 } a _ { k } = 0 . 5 1 7 9 0 7 .\tag{12}
$$

The best single fixed schedule is [1, 5] with $A _ { \mathrm { f i x e d } } = 0 . 5 2 1 2 0 6$ . For each episode i, we then allow an oracle to inspect all six held-out outcomes and choose the schedule with highest episode accuracy. Pooling the selected outcomes gives $A _ { \mathrm { o r a c l e } } = 0 . 5 2 2 5 5 7$ . This oracle is deliberately unreachable: it uses held-out labels after all counterfactual trajectories have been evaluated.

The remaining headroom above the best fixed clock is only

$$
H = A _ { \mathrm { o r a c l e } } - A _ { \mathrm { f i x e d } } = 0 . 0 0 1 3 5 1 ,\tag{13}
$$

with paired-bootstrap 95% CI [0.000858, 0.001943]. Relative to random timing, the best fixed clock captures

$$
\frac { A _ { \mathrm { f i x e d } } - A _ { \mathrm { r a n d } } } { A _ { \mathrm { o r a c l e } } - A _ { \mathrm { r a n d } } } \approx 0 . 7 1\tag{14}
$$

of the measurable timing value. Timing matters globally, but after choosing a good clock, at most about 29% of this already-small interval remains for perfect episode-specific selection over the same action space.

## 6.2 The residual opportunity is sparse

The oracle gain is highly concentrated. Only 281 of 3,686 episodes (7.6%) obtain any additional correct tokens over fixed [1, 5], and eight episodes (0.2%) contribute half of the total micro headroom. Ties are pervasive: 3,252 episodes (88.2%) share their maximum across at least two schedules and 1,980 (53.7%) tie across all six. Tie-breaking therefore changes which schedule is nominally credited per episode but not the oracle ceiling itself.

## 6.3 Adaptation has two prerequisites

These measurements separate two prerequisites for adaptive supervisory control. First, there must be heterogeneity ofdecision value: different actions must change outcomes. Our frozen interventions clearly satisfy this condition. Second, substantial heterogeneity must remain after conditioning on a strong non-adaptive baseline. That condition is much weaker here. Once [1, 5] is selected globally, the held-out-label oracle has only a narrow margin left to exploit, and Section 5 finds limited evidence that the available state features can identify the rare residual cases.

This changes the design problem from “learn a timing decision independently on every episode” to

$$
\underbrace { \mathrm { c h o o s e ~ a ~ s t r o n g ~ g l o b a l ~ t i m i n g ~ p r i o r } } _ { \mathrm { m o s t ~ m e a s u r e d ~ v a l u e } } + \underbrace { \mathrm { l e a r n ~ r a r e ~ d e v i a t i o n s } } _ { \mathrm { s m a l l ~ r e s i d u a l ~ o p p o r t u n i t y } } .\tag{15}
$$

![](images/6c1a6764286dd809c441d4b5ce818ae981feeea39651cae01f9755f0de1ae68d.jpg)  
(a) Best fixed captures most timing value.

![](images/eefc40c4da2e1d626461d8add8cd1bf373f55673b5d934e27bcc014533e7ff17.jpg)  
(b) Residual headroom is concentrated.  
Figure 3: Decision-value decomposition over the complete K = 2 clock space. Left: the best fixed clock captures about 71% of the improvement from uniform-random timing to the held-out-label oracle. Right: only a small fraction of episodes contributes the remaining gain, and half of that gain comes from eight episodes.

The oracle is an upper bound, not an achievable target. In this checkpoint and decision space, little episode-specific value remains after selecting a strong fixed clock. This suggests a simple experimental order: measure a strong fixed rule, bound instance-specific value, and only then ask whether state predicts the residual. That separates learning the wrong timing policy from there being little adaptive value left to learn.

## 7 Discussion: From Internal Revision to Meta-Level Control

Our testbed is an internal manager, not an autonomous agent that rewrites another agent or harness; the transferable object is the evaluation problem. Meta-level systems such as GEPA, AlphaEvolve, and Meta-Harness likewise observe evidence, assign credit to revisions, and turn scores into actions (Agrawal et al., 2025; AlphaEvolve Team, 2025; Lee et al., 2026). Three questions should remain separate: Does the score vary with state? Does that variation change decisions? Do those changes add outcome value? Our seeds answer these differently: one becomes a clock and another is behaviorally diverse, yet none beats its strongest forced timing control.

The lesson is not “use clocks.” It is to measure adaptive value before treating state dependence or policy diversity as progress. Frozen interventions, counterfactual branching, and strong non-adaptive baselines provide this discipline even when exhaustive enumeration is impossible.

## 8 Limitations

The adaptive audit covers three precommitted seeds, but the exhaustive sweep, oracle analysis, and counterfactual diagnostic are seed-0 analyses; broader tasks, architectures, and compute regimes remain untested. ConceptARC-mini is a mechanism-oriented setting. The original fixed $P = \hat { 4 / P } =$ 6 seed-0 runs did not persist final weights, so those matched-controller checkpoints are deterministic reconstructions whose training summaries reproduce the recorded values to reported precision. The K = 2 sweep spans six schedules on one anti-diagonal of dwell pairs: it establishes order sensitivity, not independent dwell effects. The held-out-label oracle is an unattainable upper bound over these clocks. Micro and episode-averaged accuracy are labeled separately, and residual trigger variation is not interpreted causally.

## Responsible-Use Statement

This work highlights a risk offalse assurance: supervisory scores or diverse actions can appear adaptive without improving downstream outcomes. We recommend strong matched-compute baselines and frozen interventions or counterfactual rollouts to test realized decisions. This diagnostic testbed does not imply that fixed supervision is generally preferable.

## References

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, et al. GEPA: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025.

AlphaEvolve Team. AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms. Google DeepMind, 2025.

Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In AAAI, 2017.

Andrea Banino, Jan Balaguer, and Charles Blundell. PonderNet: Learning to ponder. In ICML Deep Learning Workshop, 2021.

Víctor Campos, Brendan Jou, Xavier Giró-i-Nieto, Jordi Torres, and Shih-Fu Chang. Skip RNN: Learning to skip state updates in recurrent neural networks. In ICLR, 2018.

Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-Harness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

Richard S. Sutton, Doina Precup, and Satinder Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1–2):181–211, 1999.

Alexander S. Vezhnevets, Simon Osindero, Tom Schaul, et al. FeUdal Networks for hierarchical reinforcement learning. In ICML, 2017.

Guan Wang, Jin Li, Yuhao Sun, et al. Hierarchical Reasoning Model. arXiv preprint arXiv:2506.21734, 2025.

## A Additional Evaluation Details

Evidence hierarchy. Headline claims use the three precommitted 400-step adaptive audits and their within-checkpoint forced-schedule comparisons. The exhaustive K = 2 sweep, oracle analysis, and counterfactual persistence diagnostic are deeper mechanistic analyses of seed 0. The corrected three-seed fixed-period matrix is contextual control evidence. The 96-step adaptive experiment is used only as mechanism sanity evidence; counterfactual persistence v2 stopped before policy training.

Schedule notation. An emission after pass m changes the commitment first consumed on pass m + 1. For M = 8, [1, k] therefore contains the mandatory initial emission plus one replacement. The corresponding two commitment dwells sum to seven worker passes. This convention explains why reversed pairs such as [1, 2] and [1, 7] exchange the two dwell lengths.

Calibration. Threshold calibration optimizes only the mean intervention count on the 921-episode calibration split. The selected threshold is then frozen and evaluated once on the disjoint 3,686- episode final split. This procedure prevents task labels or final accuracy from selecting the operating point.

Bootstrap. Schedule comparisons and oracle headroom use paired episode resampling with 10,000 bootstrap draws. Pairing preserves the fact that all six clocks are evaluated on identical episodes. Confidence intervals reported in the main text are percentile intervals.

## B Supplementary Quantitative Results

Table 2: Three-seed adaptive audit and frozen-checkpoint forced controls.
<table><tr><td>Seed</td><td>Final K</td><td>Adaptive acc.</td><td>Best forced</td><td>Best acc.</td><td>Gap (pp)</td></tr><tr><td>0</td><td>2.0008</td><td>0.467015</td><td>[1, 4]</td><td>0.487511</td><td>-2.0496</td></tr><tr><td>1</td><td>1.9674</td><td>0.486149</td><td>[1, 6]</td><td>0.489535</td><td>-0.3386</td></tr><tr><td>2</td><td>1.9135</td><td>0.474947</td><td>[1, 2]</td><td>0.475515</td><td>-0.0568</td></tr></table>

Table 3: Headline quantitative checks. Metrics are not mixed across aggregation families.
<table><tr><td>Experiment</td><td>Quantity</td><td>Result</td></tr><tr><td>Adaptive seed 0</td><td>final mean K</td><td>2.0008</td></tr><tr><td>Adaptive seed 0</td><td>episodes selecting [1, 2]</td><td>99.19%</td></tr><tr><td>Adaptive seed 0</td><td>position-explained β variance</td><td>96.76%</td></tr><tr><td>Matched training</td><td>adaptive ep.-avg. token acc.</td><td>0.4670</td></tr><tr><td>Matched training</td><td>fixed  $P = \bar { 4 } / \bar { P } = 6$ </td><td>0.4813 / 0.4856</td></tr><tr><td>Frozen adaptive weights</td><td>forced  $[ 1 , 2 ]$ </td><td>0.4670</td></tr><tr><td>Frozen adaptive weights</td><td>forced  $[ 1 , 4 ] / \left[ 1 , 6 \right]$ </td><td>0.4875 / 0.4866</td></tr><tr><td> $K = 2$  sweep</td><td>best fixed [1, 5] micro acc.</td><td>0.52121</td></tr><tr><td> $K = 2 { \mathrm { ~ s w e e p } }$ </td><td>uniform-random micro acc.</td><td>0.51791</td></tr><tr><td> $K = 2$  sweep</td><td>oracle micro acc.</td><td>0.52256</td></tr><tr><td> $K = 2 { \mathrm { ~ s w e e p } }$ </td><td>oracle headroom over best fixed</td><td>0.00135</td></tr></table>

Table 4: Preregistered reversed-order comparisons, reported as longer-first minus shorter-first micro token accuracy.
<table><tr><td>Comparison</td><td>Difference</td><td>95% paired bootstrap CI</td></tr><tr><td> $\left\lceil 1 , 7 \right\rceil - \left\lceil 1 , 2 \right\rceil$ </td><td>0.00924</td><td>[0.00847, 0.01006]</td></tr><tr><td> $[ 1 , 6 ] - [ 1 , 3 ]$ </td><td>0.00405</td><td>[0.00346, 0.00472]</td></tr><tr><td> $[ 1 , 5 ] - [ 1 , 4 ]$ </td><td>0.00097</td><td>[0.00075, 0.00123]</td></tr></table>

## C Reproducibility Notes

The primary model uses hidden and goal dimensions 32, two attention heads, one high-level and one low-level layer, and one cycle at each level. The primary objective combines Stablemax token loss, halt/continue losses, and directional loss with $\lambda _ { F } = 0 . 0 5$ and intervention weight $\eta = 0 .$ Training uses AdamATan2, batch size 4, learning rate $1 0 ^ { - 4 }$ , puzzle-embedding learning rate $1 0 ^ { - 2 }$ , weight decay 0.1, no warmup, 400 optimizer steps, and CPU execution. The adaptive replication uses seeds 0–2; the contextual fixed-period matrix also uses seeds 0–2.

Each adaptive step-400 checkpoint was frozen before its schedule-control evaluation. SHA-256 digests are seed 0: 165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889; seed 1: 688c870db32b2ccb0209f9eb93395e5bcf9cb886963fea5055b81284d627d8e8; and seed 2: 67526940b33f324fafe17565cc31f231bae01a84fc1adcb2710e2e254819d2a5. Fixedcompute completion was asserted at runtime and terminal goal emissions were suppressed. Counterfactual features and targets were detached, and matched branches were simulated from identical latent carries. The implementation uses numerically safe cosine normalization and bounded gate/halting features. The intervention-loss normalization assumes the fixed M = 8 regime and should be changed before varying the compute budget.