# FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience

Zixun Huang<sup>\*1,2</sup>, Kishan Panaganti<sup>\*†1</sup>, Haitao Mi<sup>1</sup>, Leowei Liang<sup>1</sup>

<sup>1</sup>Tencent HY LLM Frontier <sup>2</sup>University of Pennsylvania <sup>\*</sup>Equal contribution, <sup>†</sup>Project lead.

zixunh@wharton.upenn.edu, kpb@global.tencent.com

 GitHub: github.com/alexhuang13/FlowBalance

ø Blog: alexhuang13.github.io/FlowBalance-Blog/

A reasoning model can improve from its own on-policy experience, but this inner loop is fragile: terminal verifiers provide reliable yet sparse supervision, while dense same-model guidance can reinforce false confidence or overconcentrate learning on a narrow solution mode. We introduce FlowBalance, a verifier-grounded self-improvement method that learns a normalized distribution over complete responses. For each on-policy trajectory, a frozen training-time view of the same policy uses privileged context to produce token-level log-probability gains, which are aggregated into a trajectory-level selfguidance score. FlowBalance calibrates this score with the verifier-derived group advantage: guidance is retained on positive-advantage trajectories, reversed on negative-advantage trajectories, and disabled when the rollout group provides no outcome preference. The resulting energy exponentially reweights a reference policy, and profiled trajectory balance fits the normalized target with one log-partition estimate per rollout group. This realizes outcome-calibrated self-guidance via trajectory balance, without a separate token-level imitation loss. Our analysis establishes within-group contrast preservation, a minimum-change reverse-KL characterization, monotonic verifier control of target reward, and an exact correction against false-positive self-guidance on rejected responses. On mathematical reasoning, FlowBalance improves average performance over FlowRL on both Qwen3-4B and Qwen3-8B, while also improving training speed and stability, avoiding direct OPSD’s response-length collapse, and exhibiting higher correct-strategy diversity in a controlled AIME24 diagnostic.

Date: September 4, 2026

## 1 Introduction

A central promise of post-training is that a reasoning model can improve from its own experience: sample several solutions, evaluate what worked, and update the policy so that the next round of experience is better. We use self-improvement in this operational sense—repeated policy improvement from the model’s own on-policy trajectories under training-time feedback—rather than to claim a fully autonomous or closed-loop system. For long-horizon reasoning, this inner loop must learn more than a higher mean reward. It must move probability toward correct reasoning, preserve useful alternative strategies, and avoid repeatedly sharpening one locally preferred trace [1–3].

Two failure modes make this loop fragile. First, reinforcement learning with verifiable rewards (RLVR) provides dependable outcome grounding, but only through sparse terminal feedback [4, 5]. A response may contain hundreds or thousands of tokens while the verifier supplies one final reward; group-based objectives improve comparison across responses but still attach a response-level signal to every decision. Distributional methods such as FlowRL improve how this terminal evidence is translated into probability over complete responses [6], yet outcome-only energies cannot exploit fine-grained evidence along a sampled reasoning path.

Second, the model can reassess its own sampled trajectories using a more informed training-time view. A frozen copy of the current policy can condition on a reference solution or task feedback and score every sampled token [7, 8]. This same-model, privileged-hindsight signal is dense and inexpensive: it requires neither a larger external model nor additional generated responses. However, dense self-guidance is not automatically trustworthy. Because the scoring view observes information unavailable at inference, it may favor a plausible but ultimately rejected trajectory, shorten reasoning, suppress uncertainty-driven exploration, or concentrate updates around a narrow local mode [9, 10]. Blindly following such confidence creates self-confirmation: the model’s own erroneous preference becomes the next update’s supervision.

![](images/d5914ae8a31cab734ad0cd72e15c41a24180fe9df5104d59330698aa774fcc86.jpg)  
Figure 1 FlowBalance as a verifier-grounded self-improvement cycle. The current policy generates on-policy reasoning experience (1); a verifier supplies sparse but reliable outcome feedback (2); a frozen privileged-hindsight view of the same policy supplies dense self-guidance on the sampled tokens (3); sign gating grounds that guidance in verified outcomes (4); and profiled trajectory balance internalizes the resulting normalized response distribution (5). Refreshing the frozen snapshot repeats the inner loop.

We therefore ask a distributional self-improvement question:

Given on-policy reasoning experience, sparse verified outcomes, and dense but imperfect selfguidance, what normalized response distribution should the next policy learn?

We introduce FlowBalance, a verifier-grounded self-improvement operator that answers this question through outcome-calibrated self-guidance via trajectory balance. The current policy generates a rollout group. A verifier supplies stopped group-relative advantages. A frozen privileged-hindsight view of the same policy then scores the already sampled tokens using training-only context. FlowBalance aggregates these scores into a trajectorylevel guidance gain, uses the verifier to determine its direction, and fits the resulting reference-supported Gibbs target with profiled trajectory balance. The policy is optimized only through this complete-response distribution-matching objective; there is no separate token-level imitation loss.

For a prompt x, let $\mathcal { G } = ( y ^ { ( 1 ) } , \dots , y ^ { ( N ) } )$ be the on-policy rollout group and let $A _ { i } = A _ { \mathcal { G } } ( y ^ { ( i ) } )$ be its stopped group-relative verifier advantage. The privileged-hindsight scoring view $\pi _ { \mathrm { H } }$ conditions on training-only context c and yields clipped token-level log-probability gains relative to the reference policy. Their trajectory average is the self-guidance gain $G _ { \mathrm { H } } ( y ^ { ( i ) } \mid x , c )$ . FlowBalance defines

$$
E _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ( \boldsymbol { y } ^ { ( i ) } \mid \boldsymbol { x } , \boldsymbol { c } ) = \eta _ { A } A _ { i } + \beta _ { G } G _ { \mathrm { H } } ( \boldsymbol { y } ^ { ( i ) } \mid \boldsymbol { x } , \boldsymbol { c } ) \operatorname { s g n } ( A _ { i } ) .\tag{1}
$$

The verifier term anchors the direction of improvement. When $A _ { i } > 0 ,$ , positive self-guidance can reinforce a verified response; when $A _ { i } < 0$ , the same positive guidance is reversed rather than allowed to self-confirm a failure; and when $A _ { i } = 0$ , the dense branch is disabled. The guidance therefore refines the verifier’s direction instead of overriding it.

Conditioned on the realized group, FlowBalance defines the reference-supported target

$$
p _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ^ { \star } ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { c } ) \propto \pi _ { \mathrm { r e f } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \exp \left( \frac { E _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { c } ) } { \tau } \right) ,\tag{2}
$$

and fits it through the profiled trajectory-balance residual

$$
\Delta _ { \mathrm { T B } } ( y ; x , c , \mathcal { G } ) = \tau \log Z _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ( x , c ) + \tau \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } - E _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ( y \mid x , c ) .\tag{3}
$$

One log-partition estimate is profiled per rollout group. The reference policy controls support and drift, the verifier determines outcome direction, privileged hindsight provides dense within-trajectory evidence, and trajectory balance converts their composite energy into a normalized distribution over complete responses.

This construction can also be viewed as a distributional inner-loop update: a task source supplies prompts, the current policy supplies experience, FlowBalance maps $\{ ( y _ { i } , \bar { R _ { i } } , \bar { G } _ { \mathrm { H } , i } ) \} _ { i = 1 } ^ { N }$ to a target distribution, and policy fitting produces the next policy. Our experiments deliberately hold the task distribution fixed, isolating this experience-to-policy update from outer-loop task generation or curriculum evolution. The latter is complementary rather than assumed by the method.

Our analysis studies the induced target directly. We show that profiling the partition preserves every withingroup probability contrast; characterize FlowBalance as the unique minimum reverse-KL displacement from the reference at its attained composite-energy level; establish monotonic verifier control of the target reward statistic; and quantify exactly how outcome calibration converts false-positive self-guidance on a rejected response into a probability-ratio correction favoring a verified response. These are properties of the target and its local fitting objective, not a claim of global convergence for an arbitrary neural optimizer.

On mathematical reasoning, FlowBalance obtains the strongest overall averages on both Qwen3-4B and Qwen3-8B among GRPO, OPSD, RLSD, FlowRL, and FlowBalance. Relative to FlowRL, it improves the core four-benchmark average over AIME24, HMMT25, MATH500, and OlympiadBench by 1.67 points on Qwen3-4B and 1.98 points on Qwen3-8B, while also improving the full aggregate average. On Qwen3-8B, FlowBalance reaches 0.5 AIME24 validation accuracy in about 100 steps versus roughly 143 for GRPO, remains stable over 400 steps, and avoids the response-length collapse observed under direct OPSD. A controlled AIME24 diagnostic further finds higher correct-only semantic strategy diversity than GRPO and RLSD.

Our contributions are:

• A distributional self-improvement objective that improves over both verifier-only RL and RL–selfguidance hybrids. GRPO improves reasoning from sparse verifier outcomes, while RLSD augments verifier-based policy optimization with dense same-model guidance. FlowBalance addresses a diferent question: what normalized complete-response distribution should the next policy learn from these signals? It combines verifier advantages and privileged-hindsight guidance into one reference-supported trajectory energy, then fits the induced distribution through profiled trajectory balance. This makes the comparison to GRPO and RLSD especially informative: FlowBalance is not merely adding more supervision to GRPO or inserting another guidance term into an RL objective; it changes the policy-update object from a local optimization signal into an explicitly normalized distribution over the model’s own reasoning trajectories.

Concrete evidence. On the core four-benchmark average over AIME24, HMMT25, MATH500, and OlympiadBench, FlowBalance improves over GRPO by 2.60 points on Qwen3-4B (67.69 versus 65.10) and 2.44 points on Qwen3-8B (71.09 versus 68.65). Relative to RLSD, the corresponding gains are 5.83 points on Qwen3-4B (67.69 versus 61.87) and 4.18 points on Qwen3-8B (71.09 versus 66.92). The same ordering holds on the full five-benchmark aggregate: FlowBalance exceeds GRPO by 1.95 and 2.12 points and RLSD by 4.71 and 3.49 points on the 4B and 8B backbones, respectively. On Qwen3-8B, it also reaches 0.5 AIME24 validation accuracy in about 100 steps rather than roughly 143 for GRPO and remains near its peak over 400 steps while GRPO degrades after about step 180. These results isolate the practical contribution: learning a verifier-grounded trajectory distribution improves over both outcome-only RL and a direct RL–dense-guidance combination across two model scales.

• Outcome-calibrated self-guidance that converts false confidence into a verifier-grounded correction. The privileged-hindsight view is deliberately treated as useful but imperfect: it can recognize promising reasoning structure, yet it can also assign positive confidence to a trajectory that fails the verifier. FlowBalance therefore uses $+ G _ { \mathrm { H } }$ when the group-relative advantage is positive, $- G _ { \mathrm { H } }$ when it is negative, and zero guidance when the group provides no outcome preference. This is more than a heuristic filter.

Proposition 4 shows that, for a verified success and a rejected response, sign gating multiplies their target probability ratio by the exact factor $\exp ( 2 \beta _ { G } G _ { \mathrm { H } } ( y _ { - } ) / \tau )$ relative to ungated guidance. Dense same-model evidence can therefore refine verified successes without allowing a confidently scored failure to become self-reinforcing supervision.

Concrete evidence. In the exact four-mode diagnostic, reward-only shaping reaches success mass 0.818, ungated self-guidance reaches 0.832, and FlowBalance reaches 0.900 while also increasing the robust-success mode from 0.327 under reward-only shaping to 0.440. In the binary false-confidence sweep at $G _ { - } = 0 . 5$ FlowBalance attains target success probability 0.894, compared with 0.817 for reward-only shaping and 0.807 for ungated shaping. The language-model results show the same qualitative distinction: direct OPSD rapidly collapses to short responses, whereas FlowBalance maintains long reasoning traces; moreover, increasing the guidance coeficient from $\beta _ { G } = 1$ to 3 lowers AIME24 from 89.33 to 86.00 and HMMT25 from 34.67 to 30.00. The benefit therefore comes from calibrated self-guidance, not from simply making the dense signal stronger.

• A conservative and information-efficient trajectory-balance update with exact target-level guarantees. FlowBalance profiles one scalar log-partition value per rollout group, but this nuisance parameter removes only the common ofset: all N − 1 independent within-group probability contrasts remain available to train the policy. The induced target is also the unique minimum reverse-KL displacement from the reference among group distributions reaching its expected composite-energy level, and increasing the verifier coeficient monotonically increases the target’s expected verifier reward for fixed guidance. Together, these results identify the update as a principled minimum-change self-improvement step that uses the full relative structure of the rollout group while keeping verified reward as an explicit control variable.

Concrete evidence. The exact synthetic diagnostics quantify both efects. A matched-energy full-support alternative requires reverse KL 0.973, whereas the FlowBalance exponential tilt requires only 0.273; the alternative is therefore 3.6× farther from the reference. At rollout-group size N = 32, profiling all contrasts yields 2.92% of the local Gaussian parameter risk of a one-contrast-per-group estimator—about 1/34 as much risk. Consistent with this conservative, all-contrast update—without claiming that the target-level results alone prove optimizer convergence—FlowBalance reaches 0.5 AIME24 validation accuracy in about 100 steps rather than roughly 143 for GRPO and remains near its peak over 400 steps, while GRPO degrades sharply after approximately step 180.

• Performance and semantic-diversity evidence that self-improvement need not collapse to one reasoning mode. FlowBalance improves mathematical reasoning while preserving a broader successful response distribution. On Qwen3-4B, it improves over FlowRL on all four core reported benchmarks: AIME24, HMMT25, MATH500, and OlympiadBench. On Qwen3-8B, it obtains the best mean on every benchmark in the main table and improves the core four-benchmark average by 1.98 points over FlowRL. These results matter for the paper’s central claim because the largest gains are not confined to a single metric: they span both multi-sample competition reasoning and single-sample accuracy, while the training traces show that the policy does not obtain them by collapsing to the short-response behavior seen under direct OPSD.

Concrete evidence. In the controlled AIME24 diagnostic, correct-only Simpson strategy diversity is 0.2194 for FlowBalance, compared with 0.1017 for GRPO and 0.1456 for RLSD. The associated traces difer in their mathematical representation, not merely their wording: FlowBalance discovers a hidden $4 \times 5 \times 8$ box embedding where GRPO uses Cayley–Menger, and the appendix shows envelope versus multiple-root reasoning, meridian-section versus implicit-normal geometry, and coordinate elimination versus secant– tangent hyperbola parameterization. Although this is a one-seed LLM-judged diagnostic rather than a population-level diversity guarantee, it supplies concrete qualitative evidence that the learned distribution places mass on genuinely diferent correct derivations instead of stylistic rewrites of one dominant template.

## 2 Preliminaries

## 2.1 On-Policy Reasoning Experience

Let $x \sim \mathcal { D }$ be a reasoning prompt. A response is a variable-length token sequence $y = ( y _ { 1 } , \dots , y _ { T } ) \in \mathcal { V } ( x )$ At token $t ,$ the state is $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ , and the trainable policy factorizes as

$$
\pi _ { \boldsymbol { \theta } } ( y \mid x ) = \prod _ { t = 1 } ^ { T } \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid s _ { t } ) .\tag{4}
$$

At the start of each training iteration, we snapshot the current parameters as $\theta ^ { - }  \theta$ . The frozen rollout policy is $\pi _ { \theta ^ { - } } .$ while the reference policy $\pi _ { \mathrm { r e f } }$ is a fixed copy of the initial checkpoint.

For each prompt, we sample a group of responses $\{ y ^ { ( i ) } \} _ { i = 1 } ^ { N } \sim \pi _ { \theta ^ { - } } ( \cdot \mid x )$ . The verifier assigns the terminal reward $R _ { i } = R ( y ^ { ( i ) } ; x )$ according to final-answer correctness. We compute the stopped group-relative advantage

$$
A _ { i } = \frac { R _ { i } - \mu _ { R } ( x ) } { \sigma _ { R } ( x ) + \epsilon } ,\tag{5}
$$

where $\mu _ { R } ( x )$ and $\sigma _ { R } ( x )$ are the mean and standard deviation of the rewards in the response group. Rewards, group statistics, sampled trajectories, and advantages receive no gradient.

## 2.2 Privileged-Hindsight Self-Guidance

Each training prompt x is paired with training-only context $c ,$ such as a reference solution or task feedback. We evaluate the sampled token $y _ { t }$ through two views of the same frozen snapshot:

$$
\pi _ { \operatorname { r o l l } } ( \cdot \mid s _ { t } ) = \pi _ { \theta ^ { - } } ( \cdot \mid x , y _ { < t } ) ,\tag{6}
$$

$$
\pi _ { \mathrm { H } } ( \cdot \mid s _ { t } , c ) = \pi _ { \theta ^ { - } } ( \cdot \mid x , c , y _ { < t } ) .\tag{7}
$$

The rollout view generates the response without observing c. The privileged-hindsight view sees c only after the response has been sampled and scores the same tokens; it generates no replacement trajectory and receives no gradient. At inference, the deployed policy observes neither c nor the hindsight view.

This same-model scoring construction is related to on-policy self-distillation, but its role in FlowBalance is narrower: it supplies a stopped dense feature for defining a trajectory-level target. FlowBalance does not optimize a separate token-level imitation loss.

## 2.3 Reference-Supported Target Distribution

We consider a normalized target over complete responses,

$$
p ^ { \star } ( y \mid x , c ) = \frac { 1 } { Z ( x , c ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { E ( y \mid x , c ) } { \tau } \right) ,\tag{8}
$$

where $E ( y \mid x , c )$ is a stopped trajectory energy, $\tau > 0$ is its temperature, and $Z ( x , c )$ is the partition function. The reference policy fixes support and controls drift; the energy specifies relative preference among complete responses. Section 3 defines FlowBalance’s outcome-calibrated self-guidance energy and fits the target through trajectory balance.

## 3 FlowBalance: Verifier-Grounded Self-Improvement

FlowBalance maps a rollout group of self-generated reasoning trajectories into a normalized next-policy target. The verifier supplies sparse but grounded outcome evidence, while a privileged-hindsight view of the same frozen policy supplies dense self-guidance on the sampled tokens. FlowBalance first combines these stopped signals into an outcome-calibrated trajectory energy and then fits the induced complete-response distribution through trajectory balance. The dense branch therefore shapes which distribution is learned; it is not optimized as a separate token-level imitation objective.

## 3.1 Training-Time Self-Guidance from Privileged Hindsight

For a sampled response y, define the clipped token-level hindsight gain

$$
\begin{array} { r } { \delta _ { t } ^ { \mathrm { H } } ( y ; x , c ) = \mathrm { c l i p } \left( \log \pi _ { \mathrm { H } } ( y _ { t } \mid s _ { t } , c ) - \log \pi _ { \mathrm { r e f } } ( y _ { t } \mid s _ { t } ) , - B , B \right) . } \end{array}\tag{9}
$$

We aggregate these gains over the complete response:

$$
G _ { \mathrm { H } } ( y \mid x , c ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \delta _ { t } ^ { \mathrm { H } } ( y ; x , c ) .\tag{10}
$$

$G _ { \mathrm { H } }$ measures how much the privileged-hindsight view raises or lowers the average sampled-token log probability relative to the fixed reference policy. It is a trajectory feature computed on the policy’s own on-policy experience. No token is resampled from $\pi _ { \mathrm { H } }$ , and no gradient is propagated through π or $G _ { \mathrm { H } }$

## 3.2 Outcome-Calibrated Self-Improvement Target

Dense self-guidance can be useful while still being wrong about the verified outcome. FlowBalance therefore uses the group-relative advantage to determine its direction:

$$
E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) = \eta _ { A } A ( y ) + \beta _ { G } G _ { \mathrm { H } } ( y \mid x , c ) \mathrm { s g n } ( A ( y ) ) ,\tag{11}
$$

where $\eta _ { A } , \beta _ { G } \geq 0$ . If $A ( y ) > 0$ , positive guidance increases the trajectory energy. If $A ( y ) < 0$ , positive guidance is reversed, preventing a confidently scored failure from becoming self-reinforcing supervision. If $A ( y ) = 0$ , the dense branch is disabled. All quantities on the right-hand side are stopped during the policy update.

FlowBalance defines the unnormalized and normalized targets

$$
{ \widetilde p } _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( { \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) } { \tau } } \right) ,\tag{12}
$$

$$
p _ { \mathrm { F l o w B a l a n c e } } ^ { \star } ( y \mid x , c ) = \frac { \widetilde { p } _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) } { Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) } , \quad Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) = \sum _ { y \in \mathcal { Y } ( x ) } \widetilde { p } _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) .\tag{13}
$$

The three factors have distinct roles: $\pi _ { \mathrm { r e f } }$ retains reference support, A supplies verified outcome direction, and $G _ { \mathrm { H } }$ provides dense within-trajectory evidence. Because A is group-relative in implementation, the practical update uses the realized rollout group $\mathcal { G } \colon$

$$
p _ { \mathrm { F l o w B a l a n c e } , \mathcal { G } } ^ { \star } ( y ^ { ( i ) } \mid x , c ) = \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) \exp ( E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( i ) } \mid x , c ) / \tau ) } { \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \exp ( E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( j ) } \mid x , c ) / \tau ) } .\tag{14}
$$

The response-space notation describes the corresponding ideal Gibbs distribution; the objective and profiled partition are evaluated on sampled complete trajectories

Why a partition-normalized target? A local dense score specifies how individual sampled tokens look under privileged hindsight, but not where total probability mass should settle across complete responses. The partition term converts relative trajectory energies into a probability-conserving target. Consequently, selfguidance can refine the distribution within the verifier’s direction without becoming an independent local loss. Appendix A gives the corresponding equilibrium interpretation.

## 3.3 Trajectory Balance as a Normalized Self-Update

The complete-trajectory balance equation is

$$
\tau \log Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) + \tau \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } - E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) = 0 .\tag{15}
$$

The trajectory-balance residual is

$$
\Delta _ { \mathrm { T B } } ( y ^ { ( i ) } ; x , c ) = \tau \log Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) + \tau \log \frac { \pi _ { \theta } ( y ^ { ( i ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( i ) } \mid x , c ) .\tag{16}
$$

At zero residual, the partition cancels in pairwise probability ratios:

$$
\log \frac { \pi _ { \theta } ( y ^ { ( a ) } \mid x ) } { \pi _ { \theta } ( y ^ { ( b ) } \mid x ) } = \log \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( a ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( b ) } \mid x ) } + \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( a ) } \mid x , c ) - E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( b ) } \mid x , c ) } { \tau } .\tag{17}
$$

Thus the energy controls relative preference among the model’s own sampled experiences, while log $Z _ { \mathrm { F } }$ lowBalance absorbs only the common prompt-level ofset.

The main loss is

$$
\mathcal { L } _ { \mathrm { F l o w B a l a n c e } } ( \theta ) = \mathbb { E } _ { ( x , c ) \sim \mathcal { D } } \left[ \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \Delta _ { \mathrm { T B } } ( y ^ { ( i ) } ; x , c ) ^ { 2 } \right] .\tag{18}
$$

Rewards, advantages, self-guidance scores, partition estimates, and sampled responses are stopped. Gradients are taken only through the trainable-policy log probabilities log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { ( i ) } \mid \boldsymbol { x } )$

Subtrajectory balance. The same principle can be applied between intermediate states. Let $Z _ { \mathrm { F l o w B a l a n c e } } ( s )$ denote the continuation partition from state $s ,$ with $Z _ { \mathrm { F l o w B a l a n c e } } ( s _ { \mathrm { t e r m } } ) = 1$ . Define the per-token shaped increment

$$
r _ { t } ^ { \mathrm { F l o w B a l a n c e } } = \tau \log \pi _ { \mathrm { r e f } } ( y _ { t } \mid s _ { t } ) + { \frac { \beta _ { G } } { T } } \delta _ { t } ^ { \mathrm { H } } \operatorname { s g n } ( A ) + \eta _ { A } A \mathbf { 1 } \{ t = T \} .\tag{19}
$$

For an interval $i { : } j .$ , the subtrajectory residual is

$$
\Delta _ { i : j } = \tau \log Z _ { \mathrm { F l o w B a l a n c e } } ( s _ { i } ) + \tau \sum _ { t = i } ^ { j - 1 } \log \pi _ { \theta } ( y _ { t } \mid s _ { t } ) - \tau \log Z _ { \mathrm { F l o w B a l a n c e } } ( s _ { j } ) - \sum _ { t = i } ^ { j - 1 } r _ { t } ^ { \mathrm { F l o w B a l a n c e } } .\tag{20}
$$

The endpoint partitions account for diferent continuation distributions. Sampling intervals and minimizing $\mathbb { E } [ \Delta _ { i : j } ^ { 2 } ]$ provides a denser fitting objective along long responses; the experiments in this paper use the complete-response implementation unless otherwise stated.

## 3.4 Profiled Optimization over Rollout Groups

The partition can be represented by a learned prompt-conditioned estimator or profiled directly from the rollout group. We use the group estimator. Each response implies

$$
\widehat { \log Z } _ { i } ( x , c ) = \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { ( i ) } \mid x , c ) } { \tau } - \log \frac { \pi _ { \theta } ( y ^ { ( i ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } ,\tag{21}
$$

and we set

$$
\widehat { \log Z } _ { \mathrm { F l o w B a l a n c e } } ( x , c ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { \log Z } _ { i } ( x , c ) .\tag{22}
$$

Gradients are stopped through this estimate.

Algorithm 1 makes the self-improvement loop explicit. The frozen snapshot first generates experience without privileged context, then provides a training-only hindsight score on those same trajectories. The verifier grounds the direction of the score, and trajectory balance internalizes the resulting normalized distribution into the next policy snapshot.

## 4 Why FlowBalance Supports Verifier-Grounded Self-Improvement

Section 3 defines FlowBalance as a partition-normalized update over a realized group of the model’s own trajectories. We analyze four properties that are useful for a self-improvement inner loop. First, profiling the group partition should retain the relative evidence contained in all sampled responses. Second, the target should move conservatively from the reference. Third, the verifier should remain an explicit control on target reward even in the presence of dense self-guidance. Fourth, false-positive guidance on a rejected response should not become self-reinforcement. All statements below are conditional on the realized rollout group and treat verifier and guidance quantities as stopped. Proofs are in Appendix B.

Algorithm 1 FlowBalance: verifier-grounded self-improvement from on-policy experience   
Require: Prompt–context dataset $\mathcal { D } ;$ policy $\pi _ { \boldsymbol { \theta } } ;$ fixed reference $\pi _ { \mathrm { r e f } } ;$ verifier $R ;$ coeficients $\eta _ { A } , \beta _ { G } , \tau .$   
1: for each training iteration do   
2: Snapshot the current policy: $\theta ^ { - }  \theta .$   
3: Sample a minibatch $\{ ( x _ { b } , c _ { b } ) \} _ { b = 1 } ^ { B } \sim \mathcal { D } .$   
4: Generate N responses $y _ { b } ^ { ( i ) } \sim \pi _ { \theta ^ { - } } ( \cdot \mid x _ { b } )$ for every prompt.   
5: Evaluate rewards and compute stopped group-relative advantages $A _ { b , i }$ using $\operatorname { E q . } { ( 5 ) } .$   
6: Score sampled tokens under $\pi _ { \theta } , \pi _ { \mathrm { r e f } } ,$ and the frozen hindsight view $\pi _ { \mathrm { H } } = \pi _ { \theta ^ { - } } ( \cdot \mid x _ { b } , c _ { b } , y _ { < t } )$   
7: Compute $G _ { \mathrm { H } } ( y _ { b } ^ { ( i ) } \mid x _ { b } , c _ { b } )$ and $E _ { \mathrm { F l o w B a l a n c e } } ( y _ { b } ^ { ( i ) } )$ using Eqs. (10) and (11).   
8: Profile $\widehat { \log Z } _ { \mathrm { F l o w B a l a n c e } } ( x _ { b } , c _ { b } )$ by $\operatorname { E q . } { ( 2 2 ) } .$   
9: Compute $\Delta _ { \mathrm { T B } } ( y _ { b } ^ { ( i ) } ; x _ { b } , c _ { b } )$ and update θ with $\operatorname { E q . } { ( 1 8 ) } .$   
10: end for

## 4.1 Distributional Structure of the Self-Update

For each sampled response $\boldsymbol y ^ { ( i ) }$ , let

$$
E _ { i } = \eta _ { A } A _ { i } + \beta _ { G } G _ { \mathrm { H } } ( y ^ { ( i ) } \mid x , c ) \mathrm { s g n } ( A _ { i } )\tag{23}
$$

be its stopped FlowBalance energy. The normalized group target is

$$
p _ { i } ^ { \star } = \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) \exp ( E _ { i } / \tau ) } { \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \exp ( E _ { j } / \tau ) } .\tag{24}
$$

Let $p ^ { \star } = ( p _ { 1 } ^ { \star } , \ldots , p _ { N } ^ { \star } )$ . All results are conditional on the realized rollout group.

Proposition 1 (Profiled balance uses all within-group contrasts). For any candidate group distribution with positive mass on the realized responses, the profiled trajectory-balance loss is zero $i f$ and only $i f$ every relative probability contrast matches

$$
\frac { \pi ( y ^ { ( i ) } ) } { \pi ( y ^ { ( j ) } ) } = \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) } \exp \biggl ( \frac { E _ { i } - E _ { j } } { \tau } \biggr ) , \qquad i , j \in [ N ] .\tag{25}
$$

In particular, whenever the group target is representable, the global minimum is zero; the unknown prompt-level normalizer removes exactly one common ofset and preserves the remaining $N - 1$ contrast directions.

The group partition therefore does not collapse the rollout group to one comparison. It absorbs only the common energy shift while retaining all relative evidence supplied by the model’s own sampled experiences.

Proposition 2 (Conservative minimum-change self-update). Let the reference be restricted and renormalized to the realized rollout group when computing KL. For any group distribution $p \in \Delta _ { N }$ ，

$$
\mathrm { K L } ( p \| \pi _ { \mathrm { r e f } } ) = \mathrm { K L } ( p ^ { \star } \| \pi _ { \mathrm { r e f } } ) + \mathrm { K L } ( p \| p ^ { \star } ) + \frac { 1 } { \tau } \left( \sum _ { i = 1 } ^ { N } p _ { i } E _ { i } - \sum _ { i = 1 } ^ { N } p _ { i } ^ { \star } E _ { i } \right) .\tag{26}
$$

Consequently, among all group distributions that attain at least the expected FlowBalance energy of $p ^ { \star }$ , the FlowBalance target is the unique minimum-reverse-KL displacement from the reference.

FlowBalance is therefore the smallest reference-supported distributional tilt that reaches its composite selfimprovement energy level. This is a property of the target distribution, not a claim that an arbitrary finite neural-network update exactly reaches the target in one optimizer step.

## 4.2 Verifier Control against Self-Confirmation

Proposition 3 (Verifier weight monotonically improves target reward). Hold the self-guidance scores and the realized rollout group fixed. Increasing the verifier coeficient $\eta _ { A }$ monotonically shifts the FlowBalance target toward higher-reward responses. In particular,

$$
\frac { \partial } { \partial \eta _ { A } } \mathbb { E } _ { p ^ { \star } } [ R ] = \frac { \operatorname { V a r } _ { p ^ { \star } } ( R ) } { \tau ( \sigma _ { R } ( x ) + \epsilon ) } \geq 0 .\tag{27}
$$

The verifier remains an explicit control knob even when dense self-guidance is present: for fixed guidance, increasing its weight cannot reduce the target’s expected verifier reward.

Proposition 4 (Outcome calibration corrects false-positive self-guidance). Consider a verified success y<sub>+</sub> and a verifier-rejected response y<sub>−</sub> in the same mixed-outcome group. Relative to otherwise identical ungated self-guidance, sign gating changes their target probability ratio according to

$$
\frac { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { - } ) } = \frac { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { - } ) } \exp \left( \frac { 2 \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c ) } { \tau } \right) .\tag{28}
$$

Hence, when $G _ { \mathrm { H } } ( y _ { - } \mid x , c ) > 0$ , positive self-guidance on a rejected response is converted from self-reinforcement into a probability-ratio correction favoring the verified response.

This proposition captures the intended useful-but-imperfect regime: privileged hindsight can assign positive local likelihood both to correct reasoning and to plausible failures. FlowBalance preserves the ability of guidance to distinguish successful modes while reversing its false-positive support where the verifier rejects the response. Appendix C.1 provides exactly enumerable diagnostics for the same target, including a reliability sweep over imperfect self-guidance.

## 5 Experiments

We evaluate whether FlowBalance supports reliable policy self-improvement from on-policy reasoning experience along four axes. First, we compare final mathematical reasoning accuracy against a verifier-only RL baseline, direct on-policy distillation, a verifier–distillation hybrid, and outcome-only trajectory balance. Second, we inspect update eficiency, late-training stability, and response-length behavior. Third, we isolate verifier grounding and self-guidance strength. Fourth, we test whether the learned response distribution retains multiple successful strategies through an LLM-judged AIME24 diagnostic. Appendix D.1 gives the training and decoding details, Appendix D.3 gives the diversity protocol, and Appendix D.2 gives the full ablation tables.

## 5.1 Self-Improvement Evaluation Setup

We evaluate Qwen3-4B and Qwen3-8B [11]. Every deployed policy receives only the problem statement. For FlowBalance, a frozen current-policy snapshot additionally sees the training solution or task feedback only while scoring already sampled tokens; the resulting self-guidance is unavailable at inference. RLSD uses an analogous frozen current-policy scoring path, whereas OPSD uses a fixed privileged teacher according to its original formulation. All methods share the same training prompts, rollout group size, verifier, response-length cap, checkpoint schedule, and evaluation script within each backbone. We compare GRPO [4, 5], OPSD [7], RLSD [12], FlowRL [6], and FlowBalance. Table 1 reports final benchmark accuracy, and Figure 2 reports the intermediate traces used to diagnose update eficiency, stability, and response-length behavior. Appendix D.1, especially Table 5, lists the concrete training and decoding fields used for reproducibility.

## 5.2 Verifier-Grounded Self-Improvement on Mathematical Reasoning

Main Results. Table 1 summarizes mathematical reasoning results across two model backbones. All entries report step-180 results over five seeds. AIME24 is evaluated with Pass@16 to reflect competition-style sampling, while HMMT25, Minerva, MATH500, and OlympiadBench use Pass@1. The reported aggregate averages summarize broad performance rather than tuning to a single benchmark.

Table 1 Mathematical reasoning results across Qwen3-4B and Qwen3-8B. AIME24 uses Pass@16; all other benchmarks use Pass@1. Entries are percentages reported as mean ± sample standard deviation over five seeds at step 180. “Avg.” averages the five reported benchmark means.
<table><tr><td>Model</td><td>Method</td><td>AIME24@16</td><td>HMMT25@1</td><td>Minerva@1</td><td>MATH500@1</td><td>Olympiad@1</td><td>Avg.</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>GRPO</td><td> $7 8 . 0 0 \pm 1 . 8 3 $ </td><td> $2 6 . 6 7 \pm 2 . 3 6$ </td><td> $5 1 . 1 8 \pm 1 . 3 6$ </td><td> $9 2 . 0 4 \pm 0 . 9 8$ </td><td> $6 3 . 6 8 \pm 0 . 5 8$ </td><td>62.31</td></tr><tr><td>OPSD</td><td> $6 5 . 3 3 \pm 3 . 8 0$ </td><td> $1 4 . 6 7 \pm 2 . 9 8$ </td><td> $4 7 . 2 8 \pm 0 . 5 6$ </td><td> $8 7 . 5 6 \pm 1 . 3 4$ </td><td> $5 5 . 7 6 \pm 1 . 4 7$ </td><td>54.12</td></tr><tr><td>RLSD</td><td> $7 3 . 3 3 \pm 2 . 3 6$ </td><td> $2 1 . 3 3 \pm 3 . 8 0$ </td><td> $5 0 . 2 9 \pm 0 . 8 8$ </td><td> $9 1 . 4 4 \pm 0 . 5 2$ </td><td> $6 1 . 3 6 \pm 0 . 6 6$ </td><td>59.55</td></tr><tr><td>FlowRL</td><td> $7 5 . 3 3 \pm 1 . 8 3$ </td><td> $3 0 . 6 7 \pm 4 . 9 4$ </td><td> $\mathbf { 5 1 . 9 9 \pm 1 . 5 1 }$ </td><td> $9 2 . 8 4 \pm 0 . 6 2$ </td><td> $6 5 . 2 5 \pm 0 . 4 9$ </td><td>63.22</td></tr><tr><td>FlowBalance</td><td> $\mathbf { 8 0 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $\mathbf { 3 2 . 0 0 \pm 2 . 9 8 }$ </td><td> $5 0 . 5 1 \pm 0 . 5 6$ </td><td> $\mathbf { 9 3 . 2 8 \pm 0 . 5 9 }$ </td><td> ${ \bf 6 5 . 4 9 \pm 0 . 9 2 }$ </td><td>64.26</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>GRPO</td><td> $8 5 . 3 3 \pm 1 . 8 3$ </td><td> $3 1 . 3 3 \pm 7 . 6 7$ </td><td> $5 2 . 8 7 \pm { 1 . 0 2 }$ </td><td> $9 3 . 1 6 \pm 0 . 8 3$ </td><td> $6 4 . 7 8 \pm 0 . 6 2$ </td><td>65.49</td></tr><tr><td>OPSD</td><td> $4 8 . 6 7 \pm 3 . 8 0$ </td><td> $4 . 0 0 \pm 3 . 6 5$ </td><td> $3 8 . 4 6 \pm 3 . 8 5$ </td><td> $7 4 . 5 6 \pm 4 . 8 1$ </td><td> $4 0 . 0 9 \pm 3 . 5 7$ </td><td>41.16</td></tr><tr><td>RLSD</td><td> $8 2 . 6 7 \pm 3 . 6 5$ </td><td> ${ 2 8 . 0 0 \pm 1 . 8 3 }$ </td><td> $5 2 . 9 4 \pm 1 . 3 8$ </td><td> $9 3 . 4 4 \pm 0 . 1 7$ </td><td> $6 3 . 5 6 \pm 1 . 1 9$ </td><td>64.12</td></tr><tr><td>FlowRL</td><td> $8 6 . 6 7 \pm 0 . 0 0$ </td><td> $3 0 . 6 7 \pm 4 . 3 5$ </td><td> $5 2 . 7 9 \pm 1 . 3 7$ </td><td> $9 2 . 9 2 \pm 0 . 5 0$ </td><td> $6 6 . 2 0 \pm 1 . 0 5$ </td><td>65.85</td></tr><tr><td>FlowBalance</td><td> $\mathbf { 8 9 . 3 3 \pm 1 . 4 9 }$ </td><td> ${ \bf 3 4 . 6 7 \pm 9 . 8 9 }$ </td><td> ${ \bf 5 3 . 6 8 \pm 0 . 7 8 }$ </td><td> $\mathbf { 9 3 . 5 2 \pm 0 . 3 0 }$ </td><td> ${ \bf 6 6 . 8 5 \pm 0 . 4 6 }$ </td><td>67.61</td></tr></table>

On Qwen3-4B, FlowBalance reaches a five-benchmark average of 64.26, improving over GRPO by 1.95 points, OPSD by 10.14 points, RLSD by 4.71 points, and FlowRL by 1.04 points. It outperforms OPSD and RLSD on all five benchmarks and improves over GRPO on AIME24, HMMT25, MATH500, and OlympiadBench; FlowRL obtains the highest Minerva mean on this backbone. On Qwen3-8B, FlowBalance reaches 67.61 and obtains the best mean on every reported benchmark, improving over GRPO by 2.12 points, OPSD by 26.45 points, RLSD by 3.49 points, and FlowRL by 1.76 points. The comparison to FlowRL isolates the value of self-guidance inside the same trajectory-distribution family, while the comparisons to GRPO, OPSD, and RLSD show that neither verifier-only optimization nor direct local self-imitation explains the aggregate gain. A plausible explanation for OPSD’s poor reasoning performance is that imitating a solution-conditioned privileged teacher can suppress epistemic verbalization and shorten reasoning, weakening uncertainty-driven exploration and self-correction [9]; this is consistent with Figure 2c. The largest FlowBalance gain over stronger baselines appears on AIME24 Pass@16, where allocating mass across multiple correct trajectories is especially useful. The smaller but consistent gains on MATH500 and OlympiadBench indicate that this distributional update does not trade broad Pass@1 accuracy for a sampling-heavy metric.

Training Dynamics. Figure 2 summarizes the dynamics of repeated policy improvement on Qwen3-8B. The accuracy panels use matched rollout and evaluation budgets for FlowBalance and GRPO, while the response-length panel compares FlowBalance with direct OPSD. First, FlowBalance reaches 0.5 AIME24 validation accuracy in about 100 steps, compared with roughly 143 for GRPO, a 1.43× reduction in updates to the threshold. Second, FlowBalance remains near its peak accuracy throughout 400 steps, whereas GRPO degrades sharply after approximately step 180. Third, FlowBalance maintains substantially longer reasoning trajectories than direct OPSD, which rapidly collapses to short responses. These observations support a narrow but useful claim: in this fixed-task inner loop, verifier-grounded self-guidance accelerates and stabilizes the update while avoiding the shortcut behavior of ungrounded local imitation.

Ablation Study. We study the two coeficients in the FlowBalance energy:

$$
E _ { \mathrm { F l o w B a l a n c e } } ( y ) = \eta _ { A } A _ { \mathcal { G } } ( y ) + \beta _ { G } G _ { \mathrm { H } } ( y ) \mathrm { s g n } \big ( A _ { \mathcal { G } } ( y ) \big ) .\tag{29}
$$

We vary $\eta _ { A }$ to control verifier grounding and $\beta _ { G }$ to control self-guidance strength. The ablation uses onedimensional sweeps around the default setting rather than a full grid, so that each table isolates one mechanism while holding the other coeficient fixed. Table 2 sweeps $\eta _ { A } \in \{ 5 , 1 0 , 1 5 \}$ , and Table 3 sweeps the self-guidance coeficient $\beta _ { G } \in \{ 1 , 2 , 3 \}$ . The default run corresponds to $\eta _ { A } = 1 5$ and $\beta _ { G } = 1 ;$ the $\eta _ { A } = 1 0$ run and the full $\beta _ { G }$ sweep are now complete. Each ablation row uses the same backbone, training budget, seeds, and five-benchmark average as Table 1; in addition to accuracy, we track training stability to detect settings that improve one benchmark at the cost of unstable optimization.

## 5.3 Correct-Strategy Diversity under Self-Improvement

![](images/6a6faea97f797f06b662cb5609190ab1b17620c506cc340694058bb007888155.jpg)  
(a) Training acceleration.

![](images/a2b7c09e2b31a08d3d74411cb130555de6c81b4fe5552ebfad00e473f9fb6953.jpg)  
(b) Training stability.

![](images/a053a3c3db1b974c8f106b5942720a5c081e3d635f6bee5a1ea5494983bc63d6.jpg)  
(c) Response length.  
Figure 2 Training dynamics on mathematical reasoning with Qwen3-8B. (a) Training acceleration: FlowBalance reaches 0.5 AIME24 validation accuracy in about 100 steps, compared with roughly 143 steps for GRPO (1.43× faster). (b) Training stability: over 400 training steps, FlowBalance remains near its peak performance, whereas GRPO degrades sharply after approximately step 180. (c) Response length: direct OPSD rapidly collapses to substantially shorter responses, whereas FlowBalance maintains longer reasoning trajectories. In the accuracy panels, solid curves show smoothed trends and lighter curves show the corresponding per-step measurements.

Table 2 Ablation over verifier coeficient $\eta _ { A }$  
Table 3 Ablation over self-guidance coeficient $\beta _ { G }$
<table><tr><td> $\eta _ { A }$ </td><td>5</td><td>10</td><td>15</td></tr><tr><td>Avg.</td><td>65.65</td><td>65.41</td><td>67.61</td></tr></table>

<table><tr><td> $\beta _ { G }$ </td><td>1</td><td>2</td><td>3</td></tr><tr><td> $\operatorname { A v g } .$ </td><td>67.61</td><td>66.48</td><td>65.95</td></tr></table>

A reliable self-improvement update should not obtain accuracy only by sharpening one successful template. Flow-Balance also broadens the kinds of correct solutions represented by the policy. We measure this with an LLM-judged semantic strategy diagnostic on AIME24: a GPT-5.5 judge extracts each full trajectory’s mathematical representation and tools, clusters trajectories with the same core strategy, and we report Simpson strategy diversity,

$$
D _ { \mathrm { S i m p s o n } } = 1 - \sum _ { k } p _ { k } ^ { 2 } ,\tag{30}
$$

![](images/dafbba6c60e3257d5eaa6eef56907cc4f2a91fa2eefc277cf6c3133e80d618f5.jpg)  
Figure 3 LLM-judged strategy diversity. Correct only Simpson diversity on AIME24; protocol and scope are given in Appendix D.3.

where $p _ { k }$ is the fraction of correct trajectories in strategy cluster k. This metric is the probability that two sampled

correct trajectories use diferent semantic strategies; Appendix D.3 gives the full protocol and scope.

Figure 3 reports correct-only Simpson diversity. FlowBalance achieves 0.2194, compared with 0.1017 for GRPO and 0.1456 for RLSD. Thus, within this diagnostic setting, FlowBalance’s successful responses span a broader range of semantic solution strategies.

Case Study. Table 4 illustrates what the LLM-judged strategy clusters are intended to capture: diferences in the mathematical objects and representations that make a solution work. We choose this example for its structural contrast rather than for per-problem win rate. GRPO uses a standard and broadly applicable Cayley–Menger determinant solution: it treats the tetrahedron as six pairwise distances and obtains the volume from a distance invariant. FlowBalance follows a diferent route by recognizing $4 1 = 4 ^ { 2 } + 5 ^ { 2 } , 8 0 = 4 ^ { 2 } + 8 ^ { 2 }$ , and $8 9 = 5 ^ { 2 } + 8 ^ { 2 }$ , which reveals a hidden $4 \times 5 \times 8$ rectangular-box embedding and reduces the volume computation to a scalar triple product. This case study therefore grounds the aggregate LLM-judged result: FlowBalance can place probability mass on correct trajectories that are not merely rephrasings of the dominant solution template, but use a diferent mathematical representation altogether. Appendix D.5 gives three additional AIME24 examples in the same format, covering envelope versus multiple-root reasoning, planar circle tangency versus implicit-surface normals, and coordinate elimination versus trigonometric hyperbola parameterization.

Table 4 Case study on AIME24 Problem 23. Boxed spans indicate key reasoning actions in representative correct trajectories; “· · · ” denotes omitted intermediate text.
<table><tr><td colspan="10">Content (boxed = reasoning actions; “. . . &quot;= omitted)</td></tr><tr><td>Question</td><td colspan="7">Tetrahedron ABCD satisfies  $A B = C D = { \sqrt { 4 1 } } , A C = B D = { \sqrt { 8 0 } } ,$  and  $B C = A D = { \sqrt { 8 9 } } .$  If the common distance from the incenter to the four faces is  $m \sqrt { n } / p ,$  find m + n + p.</td></tr><tr><td rowspan="2">GRPO</td><td colspan="4">Common Cayley-Menger route.</td><td colspan="3">“... all faces have sides √41,√80,√89 S = 24√21</td><td rowspan="2"></td></tr><tr><td colspan="4">build Cayley-Menger matrix M from 41, 80, 89</td><td colspan="3">diagonalize the distance block by symmetry</td></tr><tr><td rowspan="5">FlowBalance</td><td>det M = 819200</td><td>V = 160/3</td><td colspan="2">r = 3V/S = 20√21/63 “.</td><td colspan="3">20 + 21 + 63 = 104 ..&quot;</td></tr><tr><td>Hidden</td><td>box-embedding route.</td><td rowspan="3"></td><td>observe</td><td colspan="3">41 = 42+52,80 = 4²+ 82,89 = 52+ 82</td><td rowspan="2"></td></tr><tr><td>embed in a 4× 5 ×8 box</td><td colspan="2"></td><td colspan="2">A = (0, 0, 0), B = (4, 5, 0), C = (4, 0, 8), D = (0, 5, 8)</td></tr><tr><td>verify all six edge lengths</td><td colspan="3">scalar triple product gives V = 160/3</td><td>r = 20√21/63</td><td></td></tr><tr><td></td><td>20 + 21 + 63 = 104 ...”</td><td colspan="3"></td><td></td><td></td></tr></table>

## 6 Related Work

Reinforcement learning from verifiable outcomes. RLVR is a major route to improving LLM reasoning with automatically checkable answers [4, 5, 13–16]. Group-based methods estimate advantages from multiple responses to one prompt without a separately learned critic. Recent work improves this optimization through sequence-level ratios, trust regions, variance-aware weighting, dynamic clipping, and adaptive baselines or learning rates [17–21]. These methods provide reliable outcome grounding, but their supervision remains response-level. FlowBalance addresses the complementary experience-to-policy question: how sparse outcomes and dense training-time evidence should define a normalized distribution over complete responses.

Privileged self-guidance and on-policy distillation. Knowledge distillation transfers behavior from a teacher to a student, while on-policy distillation evaluates student-generated trajectories under a teacher [22–24]. Recent methods use a privileged view of the same model, conditioned on reference answers, demonstrations, correct rollouts, or environment feedback [7, 8, 12, 25–27]. This produces dense token-level information, but it can also shorten reasoning, suppress uncertainty, leak unavailable context into the learning signal, or plateau and degrade [9, 10, 12, 28]. Existing work modifies, selects, or contrasts local teacher signals [29–35]; concurrent β-OPSD derives an eficient KL-regularized target with a tunable reference–teacher tradeof [36]. FlowBalance uses the same-model privileged scorer only as a stopped self-guidance feature. The algorithmic object is instead a verifier-grounded complete-response distribution fitted by trajectory balance, with no separate token-level imitation loss.

Guided self-improvement and self-evolving systems. Self-evolving systems add an outer loop that generates, selects, or schedules new learning experiences. R-Few, for example, stabilizes a Challenger–Solver loop with a small pool of human anchors and an online dificulty-based curriculum [37]. Its primary object is the evolving task and curriculum distribution. FlowBalance studies a complementary inner-loop object: given prompts and on-policy solution trajectories, how should those experiences update the policy without amplifying false self-confidence or collapsing successful-strategy diversity? We do not combine the two systems experimentally; rather, R-Few illustrates how an outer-loop experience generator could supply tasks to a FlowBalance-style distributional policy update.

Distribution matching and trajectory balance. GFlowNets learn stochastic construction policies that sample objects in proportion to an unnormalized reward rather than concentrating only on maximizers [38–42]. Trajectory balance enforces this condition over complete trajectories and improves long-horizon credit assignment [43]. For LLM reasoning, Flow of Reasoning and FlowRL instantiate trajectory balance with a prompt-conditioned partition term [6, 44], while GFlowRL replaces the auxiliary partition network with an in-batch rollout-group estimate [45]. FlowBalance builds on this distributional lineage but changes the target energy: verified outcome advantages are combined with outcome-calibrated privileged-hindsight guidance, and the resulting reference-supported distribution is profiled over the rollout group. The comparison to FlowRL in our experiments isolates this additional guidance inside the same broad trajectory-balance family.

## 7 Conclusions and Discussions

Conclusion. FlowBalance treats policy self-improvement as a distribution-learning problem over the model’s own on-policy reasoning experience. A frozen privileged-hindsight view of the same policy supplies dense selfguidance, a verifier determines whether that guidance should be retained or reversed, and profiled trajectory balance fits the resulting reference-supported target over complete responses. This separation is central: dense guidance informs the target, but the policy is updated only through the normalized trajectory-balance objective. The theory shows that the profiled update preserves all within-group contrasts, is the minimum reverse-KL displacement at its attained energy level, retains monotonic verifier control of target reward, and converts false-positive guidance on rejected trajectories into an anti-self-confirmation correction. Across Qwen3-4B and Qwen3-8B, FlowBalance achieves the strongest five-benchmark average among GRPO, OPSD, RLSD, FlowRL, and FlowBalance. It also reaches the AIME24 validation threshold in fewer updates than GRPO, remains stable over extended training, avoids direct OPSD’s response-length collapse, and exhibits higher correct-only semantic strategy diversity in the controlled AIME24 diagnostic.

Limitations and scope. Four limitations delimit these claims. First, the large-scale experiments focus on mathematical reasoning, so generalization to agentic, multimodal, or other long-horizon domains remains open. Second, the response-length comparison diagnoses a correlated failure mode but does not establish that longer responses cause higher accuracy. Third, the diversity analysis is an LLM-judged AIME24 diagnostic at one checkpoint and seed rather than a multi-seed human study. Fourth, we isolate the experience-to-policy inner loop on a fixed prompt distribution; FlowBalance is not by itself a full self-evolving system that generates or curates new tasks. A natural next step is to combine the distributional update with an outer-loop curriculum or guided task generator, including R-Few-style Challenger–Solver systems, while separately measuring task drift, policy stability, and strategy diversity.

## References

[1] Jiannan Guan, Qiguang Chen, Libo Qin, Dengyun Peng, Jinhao Liu, Liangyu Huo, Jian Xie, and Wanxiang Che. Beware of reasoning overconfidence: Pitfalls in the reasoning process for multi-solution tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 30843–30851, 2026.

[2] Zeno Kujawa, John Poole, Dobrik Georgiev, Danilo Numeroso, Henry Fleischmann, and Pietro Liò. Neural algorithmic reasoning with multiple correct solutions. In Workshop on Machine Learning on Graphs in the Era of Generative Artificial Intelligence at KDD, 2025.

[3] Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. In International Conference on Learning Representations, 2025.

[4] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024.

[5] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Li, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yan Lin, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale, 2025.

[6] Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, Qizheng Zhang, Lin Chen, Fanghao Shao, Bo Xue, Yunchong Song, Zhenjie Yang, Ganqu Cui, Ning Ding, Jianfeng Gao, Xiaodong Liu, Bowen Zhou, Hongyuan Mei, and Zhouhan Lin. FlowRL: Matching reward distributions for LLM reasoning, 2025.

[7] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026.

[8] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation, 2026.

[9] Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of LLMs?, 2026.

[10] Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, and Ge Liu. The many faces of on-policy distillation: Pitfalls, mechanisms, and fixes, 2026.

[11] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Kexin Yang, Le Yu, Xinyu Wang, Jing Zhou, and Qwen Team. Qwen3 technical report, 2025.

[12] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled RLVR, 2026.

[13] Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, and Dong Yu. Evolving language models without labels: Majority drives selection, novelty promotes variation, 2025.

[14] Dian Yu, Yulai Zhao, Kishan Panaganti, Linfeng Song, Haitao Mi, and Dong Yu. Every question has its own value: Reinforcement learning with explicit human values, 2025.

[15] Zhenwen Liang, Sidi Lu, Wenhao Yu, Kishan Panaganti, Yujun Zhou, Haitao Mi, and Dong Yu. Can LLMs guide their own exploration? gradient-guided reinforcement learning for LLM reasoning, 2025.

[16] Jiangnan Xia, Yucheng Shi, Yu Yang, Kishan Panaganti, Zhenwen Liang, and Ninghao Liu. Reasoning or memorization? direction-aware diversity exploration in LLM reinforcement learning, 2026.

[17] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025.

[18] Zhengpeng Xie, Qiang Zhang, Fan Yang, Marco Hutter, and Renjing Xu. Simple policy optimization, 2025.

[19] Kaichen Zhang, Yuzhong Hong, Junwei Bao, Hongfei Jiang, Yang Song, Dingqian Hong, and Hui Xiong. GVPO: Group variance policy optimization for large language model post-training, 2025.

[20] Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. DCPO: Dynamic clipping policy optimization, 2025.

[21] Zixun Huang, Jiayi Sheng, and Zeyu Zheng. Variance-aware baselines and adaptive learning rates for reinforcement learning with verifiable rewards, 2026.

[22] Cristian Buciluˇa, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 535–541, 2006.

[23] Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network, 2015.

[24] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024.

[25] Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models, 2026.

[26] Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning, 2026.

[27] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models, 2026.

[28] Jaehoon Kim and Dongha Lee. OPSD compresses what RLVR teaches: A post-RL compaction stage for reasoning models, 2026.

[29] Leyi Pan, Shuchang Tao, Yunpeng Zhai, Shun Zhang, Wenzhe Li, Xiao Liu, Jibin Wang, Jian Shen, Zhaopeng Tu, and Kaisheng Zhao. RLCSD: Reinforcement learning with contrastive on-policy self-distillation, 2026.

[30] Xinlei Yu, Gen Li, Qingyi Si, Guibin Zhang, Yuqi Xu, Congcong Wang, Shuai Dong, Kaiwen Tuo, Xiangyu Zeng, Kaituo Feng, Qunzhong Wang, Yang Shi, Xiaobin Hu, Xiangyu Yue, Jiaqi Wang, and Shuicheng Yan. DOPD: Dual on-policy distillation, 2026.

[31] Ziqi Zhao, Xinyu Ma, Liu Yang, Yujie Feng, Daiting Shi, Jingzhou He, Xin Xin, Zhaochun Ren, and Xiao-Ming Wu. Reflective on-policy self-distillation for language model reasoning across domains, 2026.

[32] Hongbin Zhang, Chaozheng Wang, Kehai Chen, Youcheng Pan, Yang Xiang, Jinpeng Wang, and Min Zhang. Tailoring teaching to aptitude: Direction-adaptive self-distillation for llm reasoning, 2026.

[33] Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level, 2026.

[34] Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. TIP: Token importance in on-policy distillation, 2026.

[35] Jeonghye Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Rebellious student: Reversing teacher signals for reasoning exploration with self-distilled RLVR, 2026.

[36] Haoran Xu, Chufan Shi, Chenyang Yang, Ming Li, Kaiyan Liu, Weinan Zhang, and Jun Wang. β-OPSD: Deriving with Policy Optimization, Training with Self-Distillation, 2026.

[37] Wenhao Yu, Zhenwen Liang, Chengsong Huang, Kishan Panaganti, Tianqing Fang, Haitao Mi, and Dong Yu. Guided self-evolving llms with minimal human supervision, 2025.

[38] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In Advances in Neural Information Processing Systems, volume 34, pages 27381–27394, 2021.

[39] Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, and Emmanuel Bengio. GFlowNet foundations. Journal of Machine Learning Research, 24(210):1–55, 2023.

[40] Zaiyan Xu, Sushil Vemuri, Kishan Panaganti, Dileep Kalathil, Rahul Jain, and Deepak Ramachandran. Robust LLM alignment via distributionally robust direct preference optimization. In Advances in Neural Information Processing Systems, 2025.

[41] Kishan Panaganti, Zhenwen Liang, Wenhao Yu, Haitao Mi, and Dong Yu. Group distributionally robust optimization-driven reinforcement learning for LLM reasoning, 2026.

[42] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust reinforcement learning using ofline data. In Advances in Neural Information Processing Systems, 2022.

[43] Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in GFlowNets. In Advances in Neural Information Processing Systems, volume 35, pages 5955–5967, 2022.

[44] Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. Flow of reasoning: Training LLMs for divergent reasoning with minimal examples, 2025. ICML 2025.

[45] Li Du, Kanglong Li, Yihang Cao, Kaiyan Lin, Yuanqi Ma, Yuxuan Lu, Xi Chen, Jinyang Yang, Zhiyuan Ma, Zhixuan Chu, Xuanjing Huang, Hui Yu, Jie M. Zhang, Furu Wei, Bowen Yu, and Lei Li. GFlowRL: Scaling distribution-matching RL to large language models, 2026.

## A Detailed-Balance View of FlowBalance

Detailed balance. Let $q$ be a probability distribution on the response space $y ( x )$ , and let $K ( y  y ^ { \prime } )$ be a Markov transition kernel. The kernel is reversible with respect to q if, for every pair $y , y ^ { \prime } \in \mathcal { V } ( x )$

$$
q ( y ) K ( y \to y ^ { \prime } ) = q ( y ^ { \prime } ) K ( y ^ { \prime } \to y ) .\tag{31}
$$

This is the detailed-balance condition: at equilibrium, the probability flow from $y$ to $y ^ { \prime }$ equals the reverse flow from $y ^ { \prime }$ to $y .$ . Summing $\operatorname { E q . }$ (31) over y shows that $q$ is a stationary distribution of K. Whenever both transition probabilities are positive, detailed balance also gives

$$
\frac { K ( y \to y ^ { \prime } ) } { K ( y ^ { \prime } \to y ) } = \frac { q ( y ^ { \prime } ) } { q ( y ) } .\tag{32}
$$

FlowBalance as an equilibrium distribution. For a fixed prompt–context pair $( x , c )$ , FlowBalance defines the Gibbs target

$$
p ^ { \star } ( y \mid x , c ) = { \frac { 1 } { Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) } } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( { \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) } { \tau } } \right) .\tag{33}
$$

If a reversible Markov kernel were constructed with $p ^ { \star }$ as its equilibrium distribution, its forward–reverse transition ratio would satisfy

$$
\frac { K ( y  y ^ { \prime } ) } { K ( y ^ { \prime }  y ) } = \frac { \pi _ { \mathrm { r e f } } ( y ^ { \prime } \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } \exp ( \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y ^ { \prime } \mid x , c ) - E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) } { \tau } ) .\tag{34}
$$

Thus the reference policy supplies the baseline relative mass, while the FlowBalance energy exponentially reweights transitions toward responses preferred by the verifier-grounded self-guidance energy. The partition function does not appear in this pairwise ratio, but it is required to turn these relative preferences into a normalized distribution.

Trajectory-balance interpretation. FlowBalance does not construct or simulate the Markov kernel K. Instead, trajectory balance directly imposes the global identity

$$
Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) \pi _ { \theta } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( { \frac { E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) } { \tau } } \right)\tag{35}
$$

for each complete response. Taking logarithms and multiplying by $\tau$ gives the trajectory-balance residual

$$
\tau \log Z _ { \mathrm { F l o w B a l a n c e } } ( x , c ) + \tau \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } - E _ { \mathrm { F l o w B a l a n c e } } ( y \mid x , c ) .\tag{36}
$$

When this residual is zero for all responses, $\pi _ { \theta }$ equals the normalized target in $\mathrm { E q . \ ( 3 3 ) }$ and therefore has the same pairwise probability ratios as the equilibrium distribution associated with Eq. (34). In this sense, detailed balance provides an equilibrium interpretation, whereas trajectory balance is the learning constraint that fits the desired distribution without explicitly defining pairwise response transitions.

## B Proofs for Theoretical Analyses

Throughout this appendix we condition on the same fixed prompt–context pair and realized rollout group as in Section 4. We use the main-text notation

$$
E _ { i } = \eta _ { A } A _ { i } + \beta _ { G } G _ { \mathrm { H } } ( y ^ { ( i ) } \mid x , c ) \mathrm { s g n } ( A _ { i } ) , \qquad p _ { i } ^ { \star } = \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) \mathrm { e x p } ( E _ { i } / \tau ) } { \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \mathrm { e x p } ( E _ { j } / \tau ) } .\tag{37}
$$

When KL is computed over the realized rollout group, $\pi _ { \mathrm { r e f } } ( \cdot \mid x )$ denotes its restriction and renormalization to that group.

## B.1 Distributional Properties

Proposition B.1 (Profiled balance matches all within-group contrasts). For a candidate policy π and scalar partition/intercept z, consider the realized-group trajectory-balance loss

$$
\mathcal { L } _ { N } ( \pi , z ) = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \tau z + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } \right] ^ { 2 } .\tag{38}
$$

After profiling out $z _ { i }$ , the trajectory-balance loss is zero if and only if every within-group relative probability contrast matches

$$
\frac { \pi ( y ^ { ( i ) } ) } { \pi ( y ^ { ( j ) } ) } = \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) } \exp \left( \frac { E _ { i } - E _ { j } } { \tau } \right) , \qquad i , j \in [ N ] .\tag{39}
$$

Whenever the group target is representable, the global minimum is therefore zero. Profiling removes only one common group-level ofset and preserves the remaining $N - 1$ contrast directions.

Proof. For a candidate policy π, the trajectory-balance residual for response $\boldsymbol y ^ { ( i ) }$ and scalar partition/intercept z is

$$
\tau z + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } .\tag{40}
$$

The empirical trajectory-balance loss over the realized group is therefore

$$
\mathcal { L } _ { N } ( \pi , z ) = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \tau z + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } \right] ^ { 2 } .\tag{41}
$$

For fixed $\pi ,$ this is a convex quadratic function of the single scalar z. Diferentiating Eq. (41) with respect to z gives

$$
\frac { \partial \mathcal { L } _ { N } ( \pi , z ) } { \partial z } = \frac { \tau } { N } \sum _ { i = 1 } ^ { N } \left[ \tau z + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } \right] .\tag{42}
$$

Thus the profiled intercept z is the unique solution of

$$
\sum _ { i = 1 } ^ { N } \left[ \tau \widehat { z } + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } \right] = 0 .\tag{43}
$$

After profiling, the loss is zero if and only if every squared residual in Eq. (41) is zero at $z = { \widehat { z } } .$ Hence, for every pair $i , j$

$$
\tau \widehat { z } + \tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i } = 0 ,\tag{44}
$$

$$
\tau \widehat { z } + \tau \log \frac { \pi ( y ^ { ( j ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) } - E _ { j } = 0 .\tag{45}
$$

Subtracting the second equality from the first cancels the profiled scalar $\tau \widehat { z }$ and yields

$$
\tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi ( y ^ { ( j ) } ) } - \tau \log \frac { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) } = E _ { i } - E _ { j } .\tag{46}
$$

Dividing by $\tau ,$ exponentiating, and rearranging the reference-policy ratio gives Eq. (39).

Conversely, if Eq. (39) holds for all pairs, then the quantities

$$
\tau \log \frac { \pi ( y ^ { ( i ) } ) } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } - E _ { i }\tag{47}
$$

are all equal to the same constant. Choosing z to be minus that constant divided by τ makes every trajectorybalance residual zero. Therefore zero profiled loss is equivalent to matching all within-group target log probability-ratio contrasts. The scalar z removes exactly one common ofset and cannot change any pairwise contrast. □

Proposition B.2 (Minimum reference displacement). Let $\pi _ { \mathrm { r e f } }$ denote the reference distribution restricted and renormalized to the realized rollout group. Then, for any group distribution $p \in \Delta _ { N }$

$$
\mathrm { K L } ( p \| \pi _ { \mathrm { r e f } } ) = \mathrm { K L } ( p ^ { \star } \| \pi _ { \mathrm { r e f } } ) + \mathrm { K L } ( p \| p ^ { \star } ) + \frac { 1 } { \tau } \left( \sum _ { i = 1 } ^ { N } p _ { i } E _ { i } - \sum _ { i = 1 } ^ { N } p _ { i } ^ { \star } E _ { i } \right) .\tag{48}
$$

Consequently, among all group distributions satisfying $\begin{array} { r } { \sum _ { i } p _ { i } E _ { i } \ge \sum _ { i } p _ { i } ^ { \star } E _ { i } } \end{array}$ , the FlowBalance target $p ^ { \star }$ uniquely minimizes the reverse-KL displacement $\mathrm { K L } ( p \| \pi _ { \mathrm { r e f } } )$

Proof. From Eq. (24), for each group element,

$$
\log \frac { p _ { i } ^ { \star } } { \pi _ { \mathrm { r e f } } ( y ^ { ( i ) } \mid x ) } = \frac { E _ { i } } { \tau } - \log \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \exp ( E _ { j } / \tau ) .\tag{49}
$$

Let $p \in \Delta _ { N }$ be any group distribution. Then

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( p | | \pi _ { \mathrm { r e f } } ) = \sum _ { i = 1 } ^ { N } p _ { i } \log \frac { p _ { i } } { p _ { i } ^ { \star } } + \sum _ { i = 1 } ^ { N } p _ { i } \log \frac { p _ { i } ^ { \star } } { \pi _ { \mathrm { r e f } } \left( y ^ { ( i ) } \mid x \right) } } \ ~ } \\ { { \displaystyle = \mathrm { K L } ( p | | p ^ { \star } ) + \frac { 1 } { \tau } \sum _ { i = 1 } ^ { N } p _ { i } E _ { i } - \log \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \exp ( E _ { j } / \tau ) } . } \end{array}\tag{50}
$$

(51)

Applying the same identity with $p = p ^ { \star }$ gives

$$
\mathrm { K L } ( p ^ { \star } \| \pi _ { \mathrm { r e f } } ) = \frac { 1 } { \tau } \sum _ { i = 1 } ^ { N } p _ { i } ^ { \star } E _ { i } - \log \sum _ { j = 1 } ^ { N } \pi _ { \mathrm { r e f } } ( y ^ { ( j ) } \mid x ) \exp ( E _ { j } / \tau ) .\tag{52}
$$

Subtracting this expression from Eq. (51) yields Eq. (48):

$$
\mathrm { K L } ( p \| \pi _ { \mathrm { r e f } } ) = \mathrm { K L } ( p ^ { \star } \| \pi _ { \mathrm { r e f } } ) + \mathrm { K L } ( p \| p ^ { \star } ) + \frac { 1 } { \tau } \left( \sum _ { i = 1 } ^ { N } p _ { i } E _ { i } - \sum _ { i = 1 } ^ { N } p _ { i } ^ { \star } E _ { i } \right) .\tag{53}
$$

If $p$ attains at least the expected FlowBalance energy of $p ^ { \star }$ , the last term is nonnegative; $\mathrm { K L } ( p \| p ^ { \star } ) \geq 0$ as well, with equality only when $p = p ^ { \star }$ . Hence $p ^ { \star }$ is the unique minimum-reverse-KL displacement from the reference among all such distributions. □

## B.2 Effects of Verifier Weighting and Sign Gating

Proposition B.3 (Verifier weight monotonically improves reward statistics). Hold the self-guidance scores and the realized rollout group fixed. For the GRPO-normalized advantage

$$
A _ { i } = \frac { R _ { i } - \bar { R } } { \sigma _ { R } ( x ) + \epsilon } ,\tag{54}
$$

increasing the verifier coeficient $\eta _ { A }$ monotonically increases the expected reward under the FlowBalance target:

$$
\frac { \partial } { \partial \eta _ { A } } \mathbb { E } _ { p ^ { \star } } [ R ] = \frac { \operatorname { V a r } _ { p ^ { \star } } ( R ) } { \tau ( \sigma _ { R } ( x ) + \epsilon ) } \geq 0 .\tag{55}
$$

Proof. Hold the self-guidance scores fixed. The only dependence of $p _ { i } ^ { \star }$ on $\eta _ { A }$ is through the term $\eta _ { A } A _ { i }$ Diferentiating the normalized exponential-family form gives

$$
\frac { \partial p _ { i } ^ { \star } } { \partial \eta _ { A } } = \frac { p _ { i } ^ { \star } } { \tau } \left( A _ { i } - \sum _ { j = 1 } ^ { N } p _ { j } ^ { \star } A _ { j } \right) .\tag{56}
$$

Therefore

$$
\frac { \partial } { \partial \eta _ { A } } \mathbb { E } _ { p ^ { \star } } [ R ] = \frac { 1 } { \tau } \mathrm { C o v } _ { p ^ { \star } } ( R , A ) .\tag{57}
$$

For the GRPO-normalized advantage used in the main text,

$$
A _ { i } = \frac { R _ { i } - \bar { R } } { \sigma _ { R } ( x ) + \epsilon } ,\tag{58}
$$

where $\bar { R }$ and $\sigma _ { R } ( x ) + \epsilon$ are fixed for the realized group. Hence

$$
\operatorname { C o v } _ { p ^ { \star } } ( R , A ) = { \frac { \operatorname { V a r } _ { p ^ { \star } } ( R ) } { \sigma _ { R } ( x ) + \epsilon } } ,\tag{59}
$$

and thus

$$
\frac { \partial } { \partial \eta _ { A } } \mathbb { E } _ { p ^ { \star } } [ R ] = \frac { \operatorname { V a r } _ { p ^ { \star } } ( R ) } { \tau ( \sigma _ { R } ( x ) + \epsilon ) } \geq 0 ,\tag{60}
$$

which proves Eq. (55).

Proposition B.4 (Sign gating corrects self-guidance support on failures). Consider a verified success $y _ { + }$ with positive advantage and a verifier-rejected response $y _ { - }$ with negative advantage in the same rollout group. Relative to otherwise identical ungated self-guidance shaping, sign gating changes their target probability ratio according to

$$
\frac { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { - } ) } = \frac { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { - } ) } \exp \left( \frac { 2 \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c ) } { \tau } \right) .\tag{61}
$$

Hence positive privileged support on a verifier-rejected response is converted from self-reinforcement pressure into a probability-ratio correction favoring the verified response.

Proof. Consider a verified success $y _ { + }$ and a verifier-rejected response y<sub>−</sub> in the same group. Their probability ratio under the gated target is

$$
\frac { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { - } ) } = \frac { \pi _ { \mathrm { r e f } } ( y _ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { - } \mid x ) } \exp \left( \frac { E _ { + } ^ { \mathrm { g a t e d } } - E _ { - } ^ { \mathrm { g a t e d } } } { \tau } \right) ,\tag{62}
$$

where the group normalizer cancels. The corresponding ungated target has

$$
\frac { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { - } ) } = \frac { \pi _ { \mathrm { r e f } } ( y _ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { - } \mid x ) } \exp \left( \frac { E _ { + } ^ { \mathrm { u n g a t e d } } - E _ { - } ^ { \mathrm { u n g a t e d } } } { \tau } \right) .\tag{63}
$$

The reference ratio cancels when comparing the gated and ungated probability ratios. Since $y _ { + }$ has positive advantage and $y _ { - }$ has negative advantage,

$$
E _ { + } ^ { \mathrm { g a t e d } } - E _ { - } ^ { \mathrm { g a t e d } } = \eta _ { A } \left( A _ { \mathcal { G } } ( y _ { + } ) - A _ { \mathcal { G } } ( y _ { - } ) \right) + \beta _ { G } G _ { \mathrm { H } } ( y _ { + } \mid x , c ) + \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c ) ,\tag{64}
$$

$$
E _ { + } ^ { \mathrm { u n g a t e d } } - E _ { - } ^ { \mathrm { u n g a t e d } } = \eta _ { A } \left( A _ { \mathcal { G } } ( y _ { + } ) - A _ { \mathcal { G } } ( y _ { - } ) \right) + \beta _ { G } G _ { \mathrm { H } } ( y _ { + } \mid x , c ) - \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c ) .\tag{65}
$$

The verifier diference $\eta _ { A } \left( A _ { \mathcal { G } } ( y _ { + } ) - A _ { \mathcal { G } } ( y _ { - } ) \right)$ is identical in the two expressions, and the self-guidance contribution difers by $2 \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c )$ . Therefore

$$
\frac { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { g a t e d } } ^ { \star } ( y _ { - } ) } = \frac { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { + } ) } { p _ { \mathrm { u n g a t e d } } ^ { \star } ( y _ { - } ) } \exp \left( \frac { 2 \beta _ { G } G _ { \mathrm { H } } ( y _ { - } \mid x , c ) } { \tau } \right) ,\tag{66}
$$

which proves Eq. (61).

## C Numerical Analysis Details

## C.1 Exact Numerical Diagnostics for Verifier-Grounded Self-Improvement

We instantiate the unchanged FlowBalance target on exactly enumerable response spaces. Every distribution below is obtained by normalizing

$$
q ( y ) \propto \rho ( y ) \exp \left( \frac { \eta A ( y ) + \beta G ( y ) \ : \mathrm { s g n } ( A ( y ) ) } { \tau } \right) ,\tag{67}
$$

with no surrogate objective or alternative gate. These are qualitative diagnostics of the target, not additional language-model training runs. We organize them as a short narrative. The first plot gives the basic geometric intuition for verifier-grounded self-guidance. The second isolates the exact anti-self-confirmation efect in the simplest mixed-outcome group. The third maps the reliability boundary of self-guidance. The fourth returns to the two structural properties emphasized by the theory: conservative reference change and eficient use of all within-group contrasts.

## C.1.1 Mechanism View: Grounded Self-Guidance across Multiple Modes

Figure 4 embeds four exact response modes as one-dimensional Gaussians: two failures and two successes. The verifier cannot distinguish the two successful modes, while privileged hindsight assigns its largest gain to the more robust success. It also assigns positive local likelihood to plausible failures, which is exactly the false-positive regime addressed by Proposition 4. We use

$$
\rho = ( 0 . 3 0 , 0 . 2 0 , 0 . 3 0 , 0 . 2 0 ) , \quad A = ( - 1 , - 1 , + 1 , + 1 ) , \quad G = ( 0 . 4 5 , 0 . 2 5 , 0 . 3 0 , 0 . 7 5 ) , \quad\tag{68}
$$

with $( \eta , \beta , \tau ) = ( 0 . 7 5 , 0 . 8 0 , 1 )$ . Reward-only shaping raises total success mass to 0.818 but cannot prefer the robust successful mode. Ungated self-guidance identifies that mode, yet also reinforces positive guidance on failures, reaching success mass 0.832. FlowBalance preserves the useful ranking among successes while reversing the failed-response contribution, reaching success mass 0.900 and robust-success mass 0.440. The Gaussian curves visualize exact categorical masses; they do not approximate the target.

(a) On-policy modes and self-guidance gains  
![](images/47ad2cda19b7ba4c1ff3174cb483721c0f5c8b973e1dc94f05041e56bc3e335a.jpg)  
(c) Ungated: +<sub>G</sub> on every mode

self-guidance in verified outcomes (b) Reward-only: outcome signal only  
![](images/2a9b4ced822f694d6a9063ea8b98789cd67eeb3cecd772730ee8d2a6195ac27a.jpg)

![](images/072fa4654088911308526195582f4c5467f3f6f5be0d80f190348282d98577b0.jpg)  
successes (A > 0)

![](images/71a4a7db3da9ac1075319697f3047695ce3c0be36c4e50f86a0483971aa133e5.jpg)

(d) FlowBalance: <sub>G</sub> on failures, +<sub>G</sub> on successes  
![](images/1378d747b97d738d3a8bf222ef4715475df1e217c237d5b80e4508344354ebfd.jpg)  
failures (A < 0)  
successes (A > 0)

![](images/8da064f7d6a2c9cded6eb5da9739135e27c8a8dea3e38aaa5fb2d32c6a654603.jpg)  
failures (A < 0)

![](images/9de83c43ecb8ad6eaf6463616935b533d14ec2ad0176105854659765b17f9df0.jpg)  
successes (A > 0)  
Figure 4 Mechanism view of verifier-grounded self-guidance. FlowBalance uses $+ G$ on positive-advantage responses and $- G$ on negative-advantage responses. It retains useful guidance among successful modes without converting positive local confidence on failed responses into self-reinforcement.

## C.1.2 Pairwise View: Exact Anti-Self-Confirmation in a Mixed Group

Figure 5 fixes positive self-guidance $G _ { + } > 0$ on a successful response and sweeps positive false-support $G \_ > 0$ on a plausible failure. Proposition 4 gives the exact log-odds advantages

$$
\mathrm { l o g i t } \ p _ { \mathrm { F l o w B a l a n c e } } - \mathrm { l o g i t } \ p _ { \mathrm { r e w a r d } } = \frac { \beta ( G _ { + } + G _ { - } ) } { \tau } > 0 , \qquad \mathrm { l o g i t } \ p _ { \mathrm { F l o w B a l a n c e } } - \mathrm { l o g i t } \ p _ { \mathrm { u n g a t e d } } = \frac { 2 \beta G _ { - } } { \tau } > 0 .\tag{69}
$$

Thus FlowBalance dominates both baselines throughout this useful-but-imperfect quadrant. As false-positive support grows, ungated shaping becomes less reward-aligned, whereas outcome calibration becomes more selective. At $G _ { - } = 0 . 5 ,$ the exact target success probabilities are 0.894 for FlowBalance, 0.817 for reward-only shaping, and 0.807 for ungated shaping.

Reliability-strength map for outcome-calibrated self-guidance

## Exact binary advantage of outcome calibration under useful but imperfect self-guidance

![](images/5b16d5d60b04312ba8a3ef3fa68ed5d0ae8b975dee25a3b889891f7682c4d51f.jpg)

![](images/fe963a62044738639d66aeba8a56aabedd55d94cc2c86b1374aeb2ad106eb444.jpg)  
Figure 5 Exact anti-self-confirmation advantage. The left panel shows target success probability as positive self-guidance on a failed response increases. The right panel plots the closed-form probability-ratio margins in Eq. (69); no fitted decision boundary is used.

## C.1.3 Boundary View: When Self-Guidance Helps and When It Stops Helping

Outcome calibration does not make arbitrary guidance safe. Figure 6 makes this boundary explicit on a five-mode response space. We interpolate the guidance vector as $G _ { q } = q G _ { \mathrm { u s e f u l } } + ( 1 - q ) G _ { \mathrm { a d v } }$ , where $q = 1$ is useful but still false-positive on failures and $q = 0$ is adversarial to the verifier. We then sweep both q and $\beta _ { G } / \tau .$ . Above a reliability threshold, FlowBalance improves verified-success mass over reward-only shaping while retaining several successful modes. Below that threshold, stronger guidance can reduce verified success. The reverse-KL panel quantifies the accompanying movement from the reference. This diagnostic therefore supports a bounded claim: sign gating corrects false-positive confidence on failures, but it does not justify arbitrarily strong guidance that is systematically wrong on successful responses.

![](images/0e87f9e475f0ed2443c73e006a92499662827833e707fb3dbf6a02ad80062524.jpg)

![](images/63225b98a461436ac416b5704507d627a20f1797d16a662093593d9291a37369.jpg)

![](images/1472d6639b9a9d957ef8476c00281a9a59fc970c52badaea48adbb7009ef9a8c.jpg)  
Figure 6 Reliability–strength map for self-guidance. Left: gain in verified-success mass over reward-only shaping. Middle: conditional Simpson diversity among successful modes; the white contour marks the reward-only level. Right: reverse KL to the reference. The reliability parameter is a synthetic interpolation, not an empirical calibration estimate.

## C.1.4 Structural View: Conservative Change and All-Contrast Efficiency

Figure 7 verifies two distributional consequences of the FlowBalance formulation. First, the exponentialtilt path traces the minimum reverse-KL frontier as expected sign-gated energy increases. A constructed full-support fixed-data target is matched to the exact FlowBalance energy, but requires reverse KL 0.973 instead of 0.273, a 3.6× larger displacement from the reference. Second, a local Gaussian calculation shows that profiling the group partition intercept retains all N − 1 within-group contrast directions. At group size $N = 3 2$ , its exact parameter risk is 2.92% of a one-contrast-per-group estimator, approximately 34× lower risk. Together, these diagnostics explain why the self-improvement update is both conservative in where it moves probability mass and statistically eficient in how it uses each rollout group.

Table 5 Shared reporting fields for the reasoning experiments. Entries marked “matched” are held identical across methods within a backbone.
<table><tr><td>Field</td><td>Setting</td></tr><tr><td>Backbones</td><td>Qwen3-4B, Qwen3-8B</td></tr><tr><td>Training supervision</td><td>final-answer verifier; OPSD, RLSD, and FlowBalance also use privileged training solution/feedback</td></tr><tr><td>Compared methods</td><td>GRPO, OPSD, RLSD, FlowRL, FlowBalance</td></tr><tr><td>Rollout group size</td><td>matched across methods; pending release</td></tr><tr><td>Training horizon</td><td>step-180 final table; up to 400 steps for dynamics and response-length</td></tr><tr><td>Seeds</td><td>curves five seeds for every main-table entry</td></tr><tr><td>Maximum response length</td><td>matched across methods; pending release</td></tr><tr><td>Reference policy Training-time scorer</td><td>fixed copy of the initial checkpoint</td></tr><tr><td></td><td>OPSD: fixed privileged teacher; RLSD/FlowBalance: frozen current-policy copy conditioned on c</td></tr><tr><td>FlowBalance default coefficients Evaluation metrics</td><td> $\eta _ { A } = 1 5 , \beta _ { G } = 1$  for the completed default run</td></tr><tr><td></td><td>AIME24 Pass@16; HMMT25/Minerva/MATH500/OlympiadBench Pass@1</td></tr><tr><td>Aggregate score</td><td>unweighted mean of the five benchmark means</td></tr></table>

Distributional guarantees of the FlowBalance self-update  
![](images/30542639f045c0d5b86cd78c97d439f8c64662d6a08f518b1c7303af4f5e2e78.jpg)

![](images/7c717227adac2f44beb38502bc960f765fde34e9f694d4b3ec4e7a48f519b5b5.jpg)  
Figure 7 Conservative change and all-contrast fitting. Left: FlowBalance is the minimum reverse-KL target at its attained composite-energy level. Right: profiling one nuisance intercept per rollout group preserves all N − 1 contrast directions and yields the exact local Gaussian risk reduction.

## D LLM Experiment Details

## D.1 Reasoning Self-Improvement Setup and Protocol

Scope and fairness controls. The large-scale experiments are designed to compare policy-update objectives rather than infrastructure choices. Within each backbone, all methods use the same training prompts, rollout group size, maximum response length, verifier, optimizer schedule, checkpoint cadence, and evaluation script. For methods that use privileged training information (OPSD, RLSD, and FlowBalance), that information is available only to a frozen scoring path during training. Rollout generation and evaluation never include the solution or feedback field c.

Training signal construction. For every prompt in a minibatch, the frozen rollout snapshot samples a group of responses without seeing c. A rule-based verifier assigns a terminal correctness reward, from which we compute the stopped group-relative advantage in Eq. (5). FlowBalance then reuses the same frozen snapshot as the privileged-hindsight view by conditioning it on c and scoring the already sampled tokens. The self-guidance contribution is the clipped, length-averaged log-probability gain in Eq. (10); no response is resampled from the guidance view and no guidance gradient is taken. The sign-gated trajectory energy is optimized through the trajectory-balance residual with the rollout-group partition estimate in Eq. (22).

Table 6 Full ablation over verifier coeficient $\eta _ { A }$ on Qwen3-8B at step 180. The self-guidance coeficient is held fixed at the default value $\beta _ { G } = 1 $ . Entries are percentages reported as mean ± sample standard deviation over five seeds. $^ { * 6 } \mathrm { A v g . } ^ { , * }$ averages the five benchmark means.
<table><tr><td> $\eta _ { A }$ </td><td></td><td></td><td></td><td>AIME24@16 HMMT25@1 Minerva@1 MATH500@1 Olympiad@1</td><td></td><td> $\operatorname { A v g . }$ </td></tr><tr><td>5</td><td> $8 6 . 0 0 \pm 1 . 4 9$ </td><td> $3 0 . 0 0 \pm 8 . 5 0$ </td><td> $5 3 . 4 6 \pm 1 . 2 4$ </td><td> $9 3 . 3 2 \pm 0 . 4 1$ </td><td> $6 5 . 4 6 \pm 0 . 6 9$ </td><td>65.65</td></tr><tr><td>10</td><td> $8 5 . 3 3 \pm 1 . 8 3$ </td><td> $3 0 . 0 0 \pm 3 . 3 3$ </td><td> $5 2 . 4 3 \pm 0 . 8 1$ </td><td> $9 3 . 2 0 \pm 0 . 3 2$ </td><td> $6 6 . 1 1 \pm 0 . 5 1$ </td><td>65.41</td></tr><tr><td>15</td><td> $8 9 . 3 3 \pm 1 . 4 9$ </td><td> $3 4 . 6 7 \pm 9 . 8 9$ </td><td> $5 3 . 6 8 \pm 0 . 7 8$ </td><td> $9 3 . 5 2 \pm 0 . 3 0$ </td><td> $6 6 . 8 5 \pm 0 . 4 6$ </td><td>67.61</td></tr></table>

Table 7 Full ablation over self-guidance coeficient $\beta _ { G }$ on Qwen3-8B at step 180. The verifier coeficient is held fixed at the default value $\eta _ { A } = 1 5$ . Entries are percentages reported as mean ± sample standard deviation over five seeds $^ { * 6 } \mathrm { A v g . } ^ { , * }$ averages the five benchmark means.
<table><tr><td> $\beta _ { G }$ </td><td></td><td></td><td></td><td>AIME24@16 HMMT25@1 Minerva@1 MATH500@1 Olympiad@1</td><td></td><td> $\operatorname { A v g . }$ </td></tr><tr><td>1</td><td> $8 9 . 3 3 \pm 1 . 4 9$ </td><td> $3 4 . 6 7 \pm 9 . 8 9$ </td><td> $5 3 . 6 8 \pm 0 . 7 8$ </td><td> $9 3 . 5 2 \pm 0 . 3 0$ </td><td> $6 6 . 8 5 \pm 0 . 4 6$ </td><td>67.61</td></tr><tr><td>2</td><td> $8 7 . 3 3 \pm 1 . 4 9$ </td><td> $3 2 . 0 0 \pm 5 . 0 6$ </td><td> $5 3 . 6 8 \pm 1 . 4 0$ </td><td> $9 3 . 6 4 \pm 0 . 3 3$ </td><td> $6 5 . 7 3 \pm 1 . 1 6$ </td><td>66.48</td></tr><tr><td>3</td><td> $8 6 . 0 0 \pm 1 . 4 9$ </td><td> $3 0 . 0 0 \pm 4 . 0 8$ </td><td> $5 3 . 4 6 \pm 1 . 3 9$ </td><td> $9 3 . 5 6 \pm 0 . 5 5$ </td><td> $6 6 . 7 4 \pm 0 . 6 6$ </td><td>65.95</td></tr></table>

Baseline implementations. GRPO uses the same verifier rewards and group-relative normalization as FlowBalance but optimizes the standard reward-policy objective rather than a normalized trajectory target. OPSD applies clipped forward-KL self-distillation from a fixed privileged teacher to the current policy’s on-policy trajectories [7]. RLSD combines verifier and teacher signals in its policy-optimization objective. FlowRL follows the original outcome-only trajectory-balance algorithm [6]. All baselines use the same sampled responses, answer verifier, and response-length cap whenever their objectives permit this sharing.

Evaluation protocol. For AIME24, each problem is decoded with 16 samples and is counted correct if any sample matches the final answer. For HMMT25, Minerva, MATH500, and OlympiadBench, we report Pass@1. The evaluator applies the same benchmark-specific final-answer extraction and normalization to every method. All main-table entries report the mean and sample standard deviation over five seeds at step 180. The accuracy-dynamics plots use matched evaluation checkpoints for FlowBalance and $\mathrm { G R P O ; }$ the response-length panel compares the logged training trajectories of FlowBalance and OPSD. Solid curves denote smoothed trends and lighter curves denote the corresponding per-step measurements.

## D.2 Grounding and Guidance Ablations

Coefficient sweeps. The ablations in Tables 2–3 are one-dimensional sweeps around the default setting. In the $\eta _ { A } \in \{ 5 , 1 0 , 1 5 \}$ sweep, β<sub>G</sub> and all other training settings are held fixed. In the $\beta _ { G } \in \{ 1 , 2 , 3 \}$ sweep, η<sub>A</sub> and all other settings are held fixed. Each sweep point is evaluated with the same five-benchmark average used in Table 1; all reported sweep points contain completed five-seed runs. Tables 6–7 provide the corresponding per-benchmark breakdown.

Coefficient ablations. Tables 6–7 expand the coeficient ablations in Section 5.2 to the same five benchmark columns used in the main result table. All rows use the Qwen3-8B backbone and report step-180 evaluation. AIME24 is evaluated with Pass@16, while HMMT25, Minerva, MATH500, and OlympiadBench are evaluated with Pass@1. The average is the unweighted mean of the five benchmark means.

## D.3 Semantic Diversity Diagnostic

The diversity study in Section 5.3 evaluates semantic strategy variation on AIME24 using the step-180 checkpoints of GRPO, RLSD, and FlowBalance. We use seed 0 only, decode 16 complete responses for each of the 30 problems and each method, and submit all $3 0 \times 3 \times 1 6 = 1 4 4 0$ full trajectories to a GPT-5.5 judge. No response is truncated or heuristically compressed before judging. These results are therefore a controlled diagnostic of strategy variation at one checkpoint and one seed, not a multi-seed stability estimate.

Two-stage LLM-judge protocol. The judge first performs strategy extraction independently for each full trajectory. The extracted fields include the primary solution strategy, central mathematical representation, main tools or theorems, a short reasoning outline, a compact strategy signature, and, for incorrect responses, the attempted failure mode. The instruction is to describe the strategy actually used by the trajectory, not to repair incorrect reasoning. Diferences in variable names, wording, notation, response length, arithmetic detail, or the order of equivalent calculations are not counted as diferent strategies.

In the second stage, for each problem and method, the 16 extracted strategy summaries are anonymized and randomly ordered. The judge assigns every trajectory to exactly one semantic strategy cluster. During clustering, the judge does not see the source algorithm, checkpoint name, correctness label, or original response order. Correctness labels are applied only after clustering, and the reported diversity metric restricts the resulting cluster assignments to correct trajectories.

Metric. For a fixed problem and method, restrict the judged cluster assignments to correct trajectories and let $p _ { 1 } , \ldots , p _ { K }$ denote the proportions of those trajectories in the resulting semantic strategy clusters. We report correct-only Simpson strategy diversity,

$$
D _ { \mathrm { S i m p s o n } } = 1 - \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } ,\tag{70}
$$

which is the probability that two randomly sampled correct trajectories use diferent semantic solution strategies. A problem is included in the aggregate only when it has at least two correct trajectories.

Scope. LLM-based clustering is inherently partly subjective, even with anonymization and a two-stage protocol. The main-text conclusion is therefore restricted to this AIME24 seed-0 diagnostic: FlowBalance exhibits higher judged semantic strategy diversity among correct trajectories at this checkpoint.

## D.4 Prompt for Semantic Strategy Clustering

The LLM-judged diagnostic uses two prompts: one extracts a strategy summary from each trajectory, and the other clusters anonymized summaries within the same problem and method.

## Prompt 1: Per-trajectory strategy extraction

Given the problem statement and one complete reasoning trajectory, describe the mathematical strategy actually attempted by the trajectory. Do not repair incorrect reasoning, replace the attempted method with a cleaner solution, or infer a method not present in the text.

Extract: (i) the primary solution strategy, (ii) the central mathematical representation, (iii) the main tools or theorems, (iv) a short reasoning outline, (v) a compact strategy signature, and (vi) for incorrect responses, the attempted failure mode. Treat changes in notation, variable names, wording, response length, arithmetic detail, or the order of equivalent calculations as the same strategy. Treat substantively diferent mathematical objects or tools—for example, rectangular-box embedding versus Cayley–Menger determinant, coordinate versus synthetic geometry, generating functions versus direct counting, or roots-of-unity methods versus real polynomial methods—as diferent strategies.

## Prompt 2: Within-problem strategy clustering

Given 16 anonymized strategy summaries for the same problem and method, assign every trajectory to exactly one semantic strategy cluster. The summaries are randomly ordered and do not reveal the source algorithm, checkpoint name, correctness label, or original response order.

Cluster by the core mathematical representation and main tools, not by surface wording, formatting, or response length. Incorrect but coherent attempts should still be clustered by their attempted strategy; correctness labels are applied only after clustering when computing the correct-only diversity metric.

Return the cluster assignment for each anonymous trajectory, a short cluster name, and a brief rationale for each cluster.

## D.5 Why the FlowBalance Traces Are Interesting

The main text shows AIME24 Problem 23 as a compact illustration of the semantic distinctions used by the LLM-judged diversity diagnostic. Here we provide three additional FlowBalance cases from the same seed-0, step-180, 16-sample AIME24 run. Each case contains two representative correct trajectories whose final answers agree but whose central mathematical objects difer; the boxes mark the strategy-defining reasoning actions, following the format of Table 4.

Together with the AIME24 Problem 23 example in the main text, these cases show that FlowBalance’s additional correct trajectories are not merely longer, more verbose, or diferently worded variants of a standard template. We regard a trace as interesting when the boxed steps change the mathematical object or criterion that drives the solution: for example, replacing a local algebraic test with a global envelope argument, replacing a three-dimensional geometry problem with a two-dimensional cross-section reduction, or replacing direct elimination with a parameterized conic argument. In other words, the diversity signal is intended to capture a semantic change in how the problem is solved, not a superficial change in phrasing.

On Problem 5, FlowBalance finds both a global envelope view of the unit-intercept line family and a local multiplicity view. The first identifies the special point as the tangency point on the astroid $x ^ { 2 / 3 } + y ^ { 2 / 3 } = 1$ while the second detects uniqueness by forcing the known line AB to be a double root of a one-variable trigonometric equation. On Problem 15, one FlowBalance trajectory reduces the three-dimensional torus sphere contact to two circle tangencies in a meridian plane. Another keeps the problem as an implicit surface calculation and derives the contact radii from collinear normals in cylindrical coordinates. On Problem 27, FlowBalance again covers two diferent representations of the same constraint: direct coordinate elimination in squared variables, and a secant-tangent hyperbola parameterization that turns the rhombus condition into the trigonometric product sin θ sin ϕ = −5/6. These examples mirror the main-text box-embedding case and help explain why the FlowBalance traces are meaningful: the model is spreading probability mass across genuinely diferent correct derivations, not just stylistic rewrites of one template.

How to read the traces. The boxed lines mark only the reasoning actions that define the route; standard algebra, routine simplifications, and repeated bookkeeping are suppressed with “ · · · ”. The goal is not to claim that every token sequence difers globally, but to show that the high-level derivational backbone difers in a way that would matter to a human reader choosing between strategies.

Table 8 Additional case study on AIME24 Problem 5. Two correct FlowBalance trajectories solve the same unitintercept-segment problem using diferent criteria for uniqueness.
<table><tr><td rowspan=1 colspan=1>Content (boxed = reasoning actions; “. . . &quot; = omitted)</td></tr><tr><td rowspan=1 colspan=1>Question         Let $O = ( 0 , 0 ) , A = ( 1 / 2 , 0 ) ,$ and $B = ( 0 , { \sqrt { 3 } } / 2 )$ . Among all unit-length segments with one endpoint on the positivex-axis and one endpoint on the positive y-axis, there is a unique interior point C of AB that lies on no other suchsegment. 1 $\operatorname { f } O C ^ { 2 } = p / q$ in lowest terms, find $p + q .$ </td></tr><tr><td rowspan=1 colspan=1> $F ( x , y , u ) = { x } / { u } + { y } / { \sqrt { 1 - { u ^ { 2 } } } } - 1$ for the unit-intercept line family.</td></tr><tr><td rowspan=1 colspan=1>Solve $F = 0$ and $F _ { u } = 0$ to obtain the envelope.</td></tr><tr><td rowspan=1 colspan=1> $x = u ^ { 3 } , y = ( 1 - u ^ { 2 } ) ^ { 3 / 2 } ;$ the first-quadrant astroid.</td></tr><tr><td rowspan=1 colspan=1> $A B$ corresponds to $u = 1 / 2 ,$ sO $C = ( 1 / 8 , 3 \sqrt { 3 } / 8 )$ </td></tr><tr><td rowspan=1 colspan=1> $O C ^ { 2 } = 7 / 1 6 ,$ hence $p + q = 2 3 .$ </td></tr><tr><td rowspan=1 colspan=1>Write an interior point as $C ( t ) = ( ( 1 - t ) / 2 , \sqrt { 3 } t / 2 ) .$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { A }$ unit-intercept line through $C ( t )$ satisfies $( 1 - t ) / ( 2 \cos \theta ) + { \sqrt { 3 } } t / ( 2 \sin \theta ) = 1 .$ </td></tr><tr><td rowspan=1 colspan=1>The segment AB is the known solution $\theta = \pi / 3 .$ </td></tr><tr><td rowspan=1 colspan=1>Uniqueness at the boundary forces this solution to be a double root.</td></tr><tr><td rowspan=1 colspan=1> $f ^ { \prime } ( \pi / 3 ) = { \sqrt { 3 } } ( 1 - t ) - { \sqrt { 3 } } t / 3 = 0 , \textnormal { s o } t = 3 / 4 .$ </td></tr><tr><td rowspan=1 colspan=1> $C = ( 1 / 8 , 3 \sqrt { 3 } / 8 )$ and $p + q = 2 3 .$ </td></tr></table>

Table 9 Additional case study on AIME24 Problem 15. Two correct FlowBalance trajectories use either a two-dimensional meridian section or a three-dimensional implicit-normal condition.
<table><tr><td></td><td>Content (boxed = reasoning actions;  $^ { 6 6 } \cdot \cdot \cdot ^ { 9 9 } =$  omitted)</td></tr><tr><td>Question</td><td>A torus is formed by rotating a circle of radius 3 about an axis in its plane at distance 6 from the circle&#x27;s center. A sphere of radius 11 is tangent to the torus in two rotationally symmetric positions, with corresponding contact-circle radii ri and  $r _ { o } .$   ${ \mathrm { I f ~ } } r _ { i } - r _ { o } = m / n$  in lowest terms, find  $m + n .$ </td></tr><tr><td rowspan="7">Route 1</td><td>Meridian-section circle-tangency route.  $^ { 6 6 } \cdot \cdot \cdot ^ { 9 9 }$ </td></tr><tr><td>Take a plane through the rotation axis; the torus cross-section is a circle of radius 3 centered at</td></tr><tr><td>The sphere cross-section is a circle of radius 11 centered at  $( 0 , h ) .$ </td></tr><tr><td>The two tangencies have center distance  $1 1 - 3 = 8$  or  $1 1 + 3 = 1 4 .$ </td></tr><tr><td>The contact-circle radius is the horizontal coordinate of the tangency point.</td></tr><tr><td> $r _ { i } = 1 1 \cdot 6 / 8 = 3 3 / 4 { \mathrm { ~ a n d ~ } } r _ { o } = 1 1 \cdot 6 / 1 4 = 3 3 / 7 .$ </td></tr><tr><td> $r _ { i } - r _ { o } = 9 9 / 2 8 , \mathrm { s o } m + n = 1 2 7 .$ </td></tr><tr><td rowspan="7">Route 2</td><td>Implicit-surface normal route.  $^ { 6 6 } \cdot \cdot \cdot ^ { 9 9 }$ </td></tr><tr><td>Use cylindrical coordinates  $( \rho , z ) .$ </td></tr><tr><td>Torus:  $( \rho - 6 ) ^ { 2 } + z ^ { 2 } = 9 ; { \mathrm { s p h e r e : ~ } } \rho ^ { 2 } + ( z - h ) ^ { 2 } = 1 2 1 .$ </td></tr><tr><td>Tangency means normals are collinear:  $( \rho - 6 , z ) = \lambda ( \rho , z - h ) .$ </td></tr><tr><td>Eliminate  $z , h$  to get  $\rho / | \rho - 6 | = 1 1 / 3 .$ </td></tr><tr><td> $\rho > 6$  gives  $\rho = 3 3 / 4 ; \rho < 6 { \mathrm { ~ g i v e s ~ } } \rho = 3 3 / 7 .$ </td></tr><tr><td> $m + n = 1 2 7 .$ </td></tr></table>

Table 10 Additional case study on AIME24 Problem 27. Two correct FlowBalance trajectories optimize the same diagonal length using algebraic elimination or a trigonometric hyperbola parameterization.
<table><tr><td></td><td>Content (boxed = reasoning actions; “. . . &quot;= omitted)</td></tr><tr><td>Question</td><td>Points A, B, C, D lie on the hyperbola  $x ^ { 2 } / 2 0 - y ^ { 2 } / 2 4 = 1$  and form a rhombus whose diagonals intersect at the origin. Find the greatest real number that is strictly less than  $B D ^ { 2 }$  for every such rhombus.</td></tr><tr><td>Route 1</td><td>Coordinate-elimination route.  $^ { 6 6 } \cdot \cdot \cdot ^ { 9 9 }$ </td></tr><tr><td></td><td>Write opposite vertices as  $( a , b ) , ( - a , - b )$  and  $( c , d ) , ( - c , - d )$ </td></tr><tr><td></td><td>A centered parallelogram is a rhombus iff its diagonals are perpendicular, so ac +  $b d = 0 .$ </td></tr><tr><td>Set</td><td> $x = a ^ { 2 }$  and use  $b ^ { 2 } = 6 x / 5 - 2 4 .$ </td></tr><tr><td></td><td>From perpendicularity,  $c ^ { 2 } = ( b ^ { 2 } / a ^ { 2 } ) d ^ { 2 } .$ </td></tr><tr><td></td><td>The hyperbola equation for  $B { \mathrm { ~ g i v e s ~ } } d ^ { 2 } = 6 0 0 x / ( 1 1 x - 7 2 0 ) ,$ </td></tr><tr><td></td><td> $B D ^ { 2 } = 4 ( c ^ { 2 } + d ^ { 2 } ) = 4 8 0 + 2 8 8 0 0 0 / ( 1 1 x - 7 2 0 ) > 4 8 0 ,$  with limit 480 from above.</td></tr><tr><td>Route 2</td><td>sec-tan parameterization route.  $^ { 6 6 } \cdot \cdot \cdot ^ { 9 9 }$ </td></tr><tr><td></td><td>Parameterize the hyperbola as  $( 2 { \sqrt { 5 } } \sec \theta , 2 { \sqrt { 6 } } \tan \theta ) .$ </td></tr><tr><td></td><td>Take A with parameter θ and B with parameter φ.</td></tr><tr><td></td><td>Perpendicularity gives 20 sec θ sec φ + 24 tan θ tan  $\phi = 0 .$ </td></tr><tr><td></td><td>Equivalently, sin θ sin  $\phi = - 5 / 6 .$ </td></tr><tr><td></td><td>Since | sin θ| &lt; 1, we have | sin φ|  $> 5 / 6 ,$  hence  $\tan ^ { 2 } { \phi } > 2 5 / 1 1 .$ </td></tr><tr><td></td><td> $B D ^ { 2 } = 8 0 + 1 7 6 \tan ^ { 2 } \phi > 4 8 0$  with limit 480 attainable only at infinity</td></tr></table>