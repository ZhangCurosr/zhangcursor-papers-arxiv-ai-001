# Optimizing the Score, Losing Sight of the Task

Reward Hacking Across Weights, Selection, and Prompts

Vansh Wahi

AI Research Engineer

v3wahi@uwaterloo.ca

September 2026

## Abstract

A higher evaluation score does not always mean a better language model system. When optimization exploits an evaluator’s mistakes, measured progress can conceal unchanged or deteriorating task performance. This failure can arise through parameter updates, selection among generated outputs, or revisions to persistent prompts. We develop a comparative framework for reward hacking across these three optimization substrates: weights, selection, and text. Building on the Proxy Compression Hypothesis and research on inference-time and in-context reward hacking, we examine how reachable behavior, optimization budgets, and persistent adaptation shape exposure to proxy error. We formalize a distance-dependent upper bound on evaluator disagreement and a capacity ordering for nested policy classes, then show why distance alone cannot establish a universal ranking of vulnerability. An exact finite-output illustration demonstrates how the location of a scoring defect changes the behavior favored by each method. We also map representative defenses across substrates, identifying which mechanisms transfer directly and which offer only functional analogies. Persistent prompts receive particular attention: their contents are inspectable, but the behavior induced by a small textual change may be difficult to anticipate. The formal analysis, numerical illustration, and published evidence together provide a basis for comparing opti mization methods and identifying the conditions under which their defenses transfer. The resulting framework connects optimization choices to verification requirements: reliable im provement depends on controlling accessible failure modes and preserving evidence of task quality independent of the score being optimized.

## Introduction

A language model system receives a higher score after an update. Has it become better at the task, or better at satisfying the evaluator? That distinction is central to any system that uses measured performance to guide its own improvement. A score makes optimization possible, but it also creates an opportunity: behavior that exploits a weakness in the measurement can be rewarded alongside behavior that solves the task.

This opportunity appears in several familiar forms. Parameter training can increasingly favor outputs that a reward model scores highly even as an independent measure of quality declines (Gao et al., 2023). Best-of-n selection can expose a scorer’s blind spots by searching a larger pool of candidates (Khalaf et al., 2025). Persistent prompt optimization can encode a scoring shortcut into instructions that are then reused across future inputs. In one production example, a prompt mutation raised a judge’s rationale-alignment pass rate from 23.1% to 80.0% by adopting its preferred vocabulary, while defect-identification precision remained unchanged (Wahi, 2026). The apparent improvement came from matching the evaluator’s preferences more closely.

We study these settings together because they share an optimization problem. Each method changes which outputs the system produces or accepts in response to an imperfect reward signal. What differs is how that change is implemented. Weight optimization modifies parameters; selection changes the distribution of returned candidates; prompt optimization revises the instructions, examples, or memory on which the model is conditioned. We call these three implementations optimization substrates.

The shared mechanism has precedents in the literature. The Proxy Compression Hypothesis organizes reward hacking around objective compression, optimization amplification, and evaluator–policy co-adaptation, including connections between training-time and inference-time methods (Wang et al., 2026). Research on in-context feedback and iterative self-refinement demonstrates that reward exploitation can also occur without updating model weights (Pan et al., 2024a, 2024b). Our contribution is an explicit comparison of the three substrates, centered on how their reachable behaviors and available controls differ. Persistent text artifacts are especially useful for this comparison because the intervention can be read directly, even when its behavioral effects remain uncertain.

A common explanation is that more expressive optimization permits more severe reward hacking. This intuition captures something important, but it needs a precise interpretation. A method that reaches farther from its starting distribution may encounter larger evaluator errors. It may also move in directions where the evaluator remains accurate. Conversely, a small change can concentrate probability on an exploit already near the baseline. The severity of hacking therefore depends on the relationship between the reachable policies and the evaluator’s errors, as well as the resources available to search those policies.

We develop this distinction through three contributions.

A common formal setup. We describe weights, selection, and persistent prompts through budgeted policy classes, proxy and task rewards, and behavioral divergence. The setup separates what a method can represent from what a particular optimizer discovers, and distinguishes evaluator disagreement from an actual decline in task performance.

An account of reach and capacity. We establish a conditional upper bound on the available proxy gap and a separate ordering result for nested policy classes. A counterexample shows why maximum distance alone does not order vulnerability. An exact finite-output illustration then demonstrates how different scoring defects interact with flexible distribution optimization, bounded conditioning, and best-of-n selection.

A defense correspondence. We compare methods for improving evaluation, constraining optimization, preserving independent evidence, and controlling acceptance. Some mechanisms transfer readily across substrates, while others require different assumptions or forms of access. For example, both KL penalties and prompt-edit budgets can limit optimization, but only the former directly controls the behavioral divergence used in our bounds. The comparison makes these differences explicit while retaining the practical value of the analogy.

The framework also helps interpret differing results on prompt optimization and reinforcement learning. GEPA reports strong performance with fewer rollouts in its evaluated settings (Agrawal et al., 2025), while other studies find task-dependent advantages for simpler prompt search or parameter adaptation (Chang and Chen, 2026; Liu et al., 2026). We treat sample efficiency, feedback quality, and reachable behavior as distinct explanations to be tested. Performance in one setting does not establish a universal ranking of the substrates.

This is a framework paper supported by published observations, mathematical analysis, and a reproducible numerical illustration. Its contribution is the comparative account of reachable behavior, proxy error, and defenses across the three substrates. The numerical illustration is computed over explicitly specified distributions; no new language-model experiments are reported. The practical aim is to connect the choice of optimization method to the evidence needed to trust its improvements.

## Background and Scope

## The Proxy Compression Hypothesis

Wang et al. (2026) propose the Proxy Compression Hypothesis (PCH) as a framework for understanding reward hacking. Its three organizing dynamics are objective compression, optimization amplification, and evaluator–policy co-adaptation.

Compression occurs when an evaluation procedure represents a rich objective through a restricted signal. Human preferences depend on context and trade-offs; a learned reward model summarizes them in a score. A program specification contains many requirements; a test suite checks a subset. An LLM judge receives a rubric, but its verdict also depends on how it interprets and applies that rubric.

Compression is not, by itself, proof that hacking must occur. A scalar score can perfectly represent the ordering needed for a particular task. The concern is the combination of missing or misvalued information and an optimizer capable of exploiting it. For this paper, proxy error includes both omitted requirements and incorrect assessments of requirements the evaluator attempts to measure.

Amplification occurs when optimization increasingly favors the proxy’s mistakes. An error that is rare under the base model may become common under a selected or adapted policy. The optimizer need not explicitly represent the evaluator’s weakness: preferentially retaining outputs with high scores can be enough to concentrate probability on it.

Co-adaptation concerns repeated changes to both policy and evaluator. A policy exploits the current evaluator; the evaluator or oversight procedure is repaired; subsequent optimization responds to the repair. This differs from adaptation to a fixed evaluator, where the policy changes but the evaluator does not. Both are relevant, and we keep them separate in Shared Mechanisms and Different Dynamics.

PCH already discusses inference-time selection. We build on that connection by treating weights, selection procedures, and persistent text artifacts as alternative ways of constructing the evaluated policy. The emphasis is on differences in feasible behavior, inspection, and control, rather than a claim that the underlying mechanism was previously unknown.

## What the three-substrate abstraction covers

The abstraction fixes a task distribution, a base generator, and an evaluation setup, then asks how feedback changes the distribution of outputs. Weight optimization includes parameter updates such as RLHF and related post-training procedures. Selection includes best-of-n and reranking over sampled candidates. Prompt optimization includes persistent revisions to instructions, examples, or textual memory.

These are useful categories, not an exhaustive inventory of everything that can change in an agent system. Tool access, retrieval policies, code, model routing, and environmental permissions can all matter. Some can be represented within an enlarged policy; others require an explicit change to the environment. A change that grants filesystem access is not adequately characterized as a short textual edit if its main effect is to expose an answer key.

For agentic tasks, the output variable can represent a full interaction trajectory rather than a single response. The environment and available actions must then be held fixed for a clean substrate comparison. When they change, the change should be reported as part of the intervention.

## Persistent prompt optimization and in-context refinement

A single conversation can contain repeated refinement without producing a reusable artifact. Persistent prompt optimization differs because the resulting instructions or memory are applied to later inputs. This distinction affects both the spread of an exploit and the defenses available against it.

A misleading explanation produced in one refinement loop may affect one answer. A misleading instruction accepted into a shared prompt can affect an entire class of future answers. Conversely, a persistent text artifact can be versioned, inspected, compared, and rolled back. These properties motivate our emphasis on prompt-space while connecting it to the earlier incontext reward-hacking literature (Pan et al., 2024a, 2024b).

## Three Substrates, One Formal Setup

## Policies, rewards, and budgets

Let $x \sim \mathcal { D }$ denote a task input and let $\pi _ { 0 } ( y \mid x )$ be a base policy. Let $R ( x , y )$ be the intended task reward and $\hat { R } ( x , y )$ the proxy used for optimization. Define

$$
J ( \pi ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi ( \cdot | x ) } [ R ( x , y ) ] , \qquad \hat { J } ( \pi ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi ( \cdot | x ) } [ \hat { R } ( x , y ) ] .
$$

True reward is an idealization. In practice, R may be approximated through independently authored tests, expert review, or another imperfect evaluation instrument. A study should state what its independent evaluation actually measures rather than silently treating it as an oracle.

Let $\Pi _ { s } ( b )$ denote a specified policy class for substrate s under budget b. The budget may include sample count, training resources, context length, or permitted edits. A representable class is not the same as a set a particular optimizer can reliably discover. We distinguish these whenever discussing theoretical capacity and measured performance.

For policies with finite forward KL, use the common behavioral divergence

$$
d ( \pi , \pi _ { 0 } ) = \operatorname { \mathbb { E } } _ { x \sim \mathcal { D } } [ \mathrm { K L } ( \pi ( \cdot \mid x ) \parallel \pi _ { 0 } ( \cdot \mid x ) ) ] .
$$

An idealized proxy optimizer solves

$$
\pi _ { s , b , \delta } ^ { * } \in \mathop { \mathrm { a r g } } \operatorname* { m a x } _ { \pi \in \Pi _ { s } ( b ) , \ d ( \pi , \pi _ { 0 } ) \leq \delta } \hat { J } ( \pi ) ,
$$

when a maximizer exists. Otherwise the optimization value can be defined as a supremum. For fixed s and $b ,$ the optimal proxy value is non-decreasing as the feasible distance budget expands. This is a property of nested feasible sets. It does not assert that a practical search procedure improves monotonically at every iteration.

## What counts as hacking in the comparison

A trajectory provides an operational signal of overoptimization when proxy performance increases while independently assessed task performance decreases. If t indexes pressure or optimization steps, the characteristic event is

$$
\hat { J } ( \pi _ { t + 1 } ) > \hat { J } ( \pi _ { t } ) , \qquad J ( \pi _ { t + 1 } ) < J ( \pi _ { t } ) .
$$

This is not an exhaustive definition of reward hacking. An exploit can be present from the beginning, or raise the proxy while leaving task quality unchanged. We therefore also examine evaluator disagreement and concrete exploit mechanisms.

When rewards have a declared common scale, define the signed proxy gap

$$
g ( \pi ) = \hat { J } ( \pi ) - J ( \pi ) .
$$

The gap is not invariant to changing the proxy’s units or offset. It should not be used to compare unrelated benchmarks without calibration. To distinguish base miscalibration from amplification, one can additionally report $g ( \pi ) - g ( \pi _ { 0 } )$ . Neither quantity replaces reporting J and J<sup>ˆ</sup> separately.

The largest signed gap in a class, the gap encountered by proxy optimization, and the largest decrease in true reward along a trajectory are different objects. Maximizing J<sup>ˆ</sup> does not generally maximize g, because J also varies. The formal bounds in Reach, Geometry, and Hacking Capacity concern class-wide capacity; the behavior discovered by a concrete algorithm also depends on its search procedure and budget.

## Weight-space

Weight-space policies have the form $\pi _ { \theta }$ and belong to an architecture-dependent class $\Pi _ { W } =$ $\{ \pi _ { \theta } : \theta \in \Theta \}$ . Training searches a portion of this class subject to data, compute, optimizer, and regularization constraints. Parameter updates can make behavior that was extremely unlikely under the base practically accessible and reusable across inputs.

It is tempting to call this a change in support, but literal support is usually the wrong distinction. With a fixed vocabulary, finite logits, and no hard decoding exclusions, a softmax model gives positive probability to every allowed finite sequence. A behavior can be inside that support while remaining essentially inaccessible within a practical sampling budget. If a policy truly assigns positive probability where the reference assigns zero, forward KL is infinite.

Weight optimization therefore offers flexible redistribution of probability, not a general license to declare support expansion or unbounded KL. Whether its class contains the distributions induced by prompting or selection depends on the model, horizon, and representation assumptions. Such inclusion must be established for the comparison being made.

## Selection-space

Best-of-n draws n candidates from the reference policy and returns one with maximum proxy score, breaking ties uniformly among the maximizing candidates. For a maximum sample budget N, define

$$
\Pi _ { S } ( N ) = \{ \mathrm { B o N } _ { n } ( \pi _ { 0 } , \hat { R } ) : 1 \leq n \leq N \} .
$$

Soft selection adds parameters controlling how strongly the scorer influences the choice. These variants should be included explicitly when used rather than treated as automatically equivalent to best-of-n.

For the best-of-n policy, Beirami et al. (2025) establish the bound

$$
d ( \mathrm { B o N } _ { n } , \pi _ { 0 } ) \leq \log n - \frac { n - 1 } { n } \leq \log n .
$$

Thus a finite sample budget gives a behavioral-divergence bound. Letting N grow removes that particular fixed ceiling; a logarithm that grows slowly is not a constant. Best-of-n preserves support because every returned output was sampled from the base. Its practical ability to locate a rare exploit depends on the base probability of that exploit and the number of draws.

The distribution selected by best-of-n need not belong to a restricted parameter family chosen for a comparison. Selection can alter shape and concentration, not merely the mean. This observation will matter when interpreting toy models.

## Prompt-space

Let p be an instruction, demonstration bank, or textual memory. A prompt-conditioned policy is

$$
\pi _ { p } ( y \mid x ) = q _ { 0 } ( y \mid p , x ) , \qquad \Pi _ { P } ( L ) = \{ \pi _ { p } : | p | \leq L \} ,
$$

where $q _ { 0 }$ is the fixed prompt-conditioned generator and the reference policy is $ \pi _ { 0 } ( y \mid x ) =$ $q _ { 0 } ( y \mid p _ { 0 } , x )$ for a fixed baseline prompt $p _ { 0 }$ . A context-length limit gives finitely many possible artifacts over a finite vocabulary. It does not, by itself, supply a useful numerical bound on their behavioral KL, particularly if support conditions are not specified.

Two kinds of constraint are relevant. An artifact constraint limits tokens, edits, examples, or access to evaluation content. A behavioral constraint limits changes in the output distribution or measured task behavior. These are related through the base model but are not interchangeable. A one-line instruction can substantially change behavior, while a long addition may have little effect.

Persistent text has an inspection advantage: its contents are directly available for review. This makes copied examples, external-tool instructions, and changes to output contracts easier to identify. Readability does not make behavior transparent, however. The same instruction can interact differently with different inputs, and a readable patch can still introduce regressions.

Table 1: A comparison under explicitly specified budgets.
<table><tr><td>Property</td><td>Weights</td><td>Selection</td><td>Persistent text</td></tr><tr><td>Intervention</td><td>Update parameters</td><td>Sample and rank candidates</td><td>Revise instructions, examples, or memory</td></tr><tr><td>Durable object</td><td>Model checkpoint</td><td>Scorer and selection procedure; generator may stay fixed</td><td>Versioned text artifact</td></tr><tr><td>Typical budget</td><td>Training compute, data, KL</td><td>Samples per input, ranking cost</td><td>Search compute, tokens, edits, examples</td></tr><tr><td>Behavioral constraint</td><td>KL or related regularization</td><td>Best-of-n bound at fixed sample count</td><td>Induced-policy divergence or behavioral checks</td></tr><tr><td>Direct inspection</td><td>Parameters are available but hard to interpret</td><td>Candidate pool and selection decisions</td><td>Added, removed, and revised text</td></tr><tr><td>Principal caveat</td><td>Representation does not guarantee learnability</td><td>More samples can expose rarer scorer errors</td><td>Small textual changes can have large effects</td></tr></table>

## Shared Mechanisms and Different Dynamics

## Proxy error is available to all three substrates

A scorer that rewards length without adequately checking correctness creates an opportunity regardless of how outputs are produced. Weight updates can increase verbosity, selection can preferentially retain verbose candidates, and prompt optimization can add an instruction to elaborate. The failure is shared even though the intervention differs.

The same reasoning applies to a verifier that checks visible tests but omits resource use, or a judge that mistakes a canonical phrase for valid reasoning. What determines exposure is whether the method can produce and favor the relevant behavior. There is no requirement that the optimizer explicitly understand the evaluator’s defect.

Distribution shift is one source of increasing proxy error, but not the only one. A proxy can already have a large error near the base distribution. Optimization can exploit it by concentrating probability locally. For this reason, the framework treats distance as a possible control on error, not as a universal causal explanation of it.

## Amplification and the shape of the curve

The familiar overoptimization curve shows proxy reward rising while true reward first improves and then deteriorates. Gao et al. (2023) study this phenomenon in reward-model optimization. Khalaf et al. (2025) analyze inference-time mechanisms and characterize different possible regimes under stated conditions. Their account includes monotonic improvement, hacking, im mediate decline, and a decline followed by improvement; proxy imperfection alone does not force one shape.

A simple example explains why. If $\hat { R } = R + c$ for a constant c, the proxy is numerically miscalibrated but ranks outputs exactly as the true reward does. Best-of-n under that proxy cannot produce a systematic decline in expected true reward as sample count grows. The relevant issue is the error’s relationship to the behaviors favored by optimization.

![](images/ce12e8553908d18f757c273a0f0a0b6e219fe6f9611d9952b6c728e6de712164.jpg)  
Figure 1: A schematic of overoptimization: proxy reward continues to rise after true task quality peaks. This is an illustrative pattern, not experimental data or a claim that every proxy produces this trajectory.

Prompt-space evidence supports this separation without requiring a complete curve. Wahi (2026) reports a mutation that raised rationale alignment from 23.1% to 80.0% through lexical mimicry without improving defect-identification precision. A separate corrupted-label incident drove removal of correct compliance rules. These are evidence of exploitation and capability regression, respectively, not a controlled estimate of the full trajectory in Figure 1.

## Fixed-evaluator adaptation versus co-adaptation

Three processes should be distinguished. Selection-induced concentration occurs when a fixed procedure increasingly favors the high-scoring tail of a fixed generator. Fixed-evaluator adaptation occurs when weights or prompts are repeatedly revised against an unchanged scorer. Evaluator–policy co-adaptation occurs when the scorer or oversight process is also revised in response.

A single best-of-n call needs no persistent learning to exploit a scorer. Persistence changes what can accumulate across calls: a tuned selection temperature, a revised judge prompt, or an exploit encoded in shared memory may carry information forward. Therefore, the absence of weight updates does not imply the absence of system-level adaptation.

Prompt optimization makes this distinction concrete. The task prompt may adapt to a fixed judge; a separate calibration process may adapt the judge; and a deployment team may then change both. A defense aimed at one process should not be assumed to control the others.

The shared questions are what the proxy misvalues, which accessible behaviors expose that error, and what information accumulates during optimization. The answers depend on the substrate: adaptations may persist in weights, text, or tuned selection procedures, and evaluator repair may change the problem being optimized.

## Reach, Geometry, and Hacking Capacity

## A distance-dependent upper bound

Fix the input distribution, reward functions, and baseline. For a specified class and budget, define

$$
D _ { s } ( b ) = \operatorname* { s u p } _ { \pi \in \Pi _ { s } ( b ) } d ( \pi , \pi _ { 0 } ) , \qquad G _ { s } ( b ) = \operatorname* { s u p } _ { \pi \in \Pi _ { s } ( b ) } g ( \pi ) .
$$

Here $G _ { s }$ is the largest signed proxy overestimate available in the class. It is a capacity measure, not a claim about what a particular optimizer will find.

Assumption 1 (A common error envelope). On an ambient class containing the policies under comparison, there is a finite, non-decreasing function $\varepsilon : [ 0 , \infty ) \to [ 0 , \infty )$ such that

$$
| g ( \pi ) | \leq \varepsilon ( d ( \pi , \pi _ { 0 } ) ) .
$$

The assumption gives an upper envelope, not a statement that every policy farther from the base has larger error. When rewards are bounded, a constant envelope is always possible; the assumption is practically useful only when the available envelope is informative at the distances being considered. Estimating or validating such an envelope for a real evaluator is a separate empirical problem.

Proposition 1 (Reach bounds the available proxy gap). Under Assumption 1, if $D _ { s } ( b ) < \infty$ , then

$$
G _ { s } ( b ) \leq \varepsilon ( D _ { s } ( b ) ) .
$$

Proof. Every $\pi \in \Pi _ { s } ( b )$ satisfies $d ( \pi , \pi _ { 0 } ) \leq D _ { s } ( b )$ . Therefore $g ( \pi ) \leq | g ( \pi ) | \leq \varepsilon ( d ( \pi , \pi _ { 0 } ) ) \leq$ $\varepsilon ( D _ { s } ( b ) )$ . Taking a supremum over the class proves the claim. □

The proposition requires neither attainment of the supremum nor existence of a maximizing policy. For an unbounded distance class, one can use the extended bound $G _ { s } ( b ) \leq \operatorname* { s u p } _ { r \geq 0 } \varepsilon ( r )$ which may be uninformative. Crucially, the proposition does not order actual capacities from their upper bounds. A larger permitted radius may increase the error allowed by the bound while the reachable policies avoid the regions where that error occurs.

## A concrete bound when proxy error is bounded

A useful special case does not require assuming that error actually grows with distance. Let $e ( x , y ) = { \hat { R } } ( x , y ) - R ( x , y )$ and suppose e lies in a common interval of width M over the input– output space. Then

$$
| g ( \pi ) - g ( \pi _ { 0 } ) | \leq M \sqrt { \frac { d ( \pi , \pi _ { 0 } ) } { 2 } } .
$$

To see this, treat ${ \mathcal { D } } ( x ) \pi ( y \mid x )$ and ${ \mathcal { D } } ( x ) \pi _ { 0 } ( y \mid x )$ as joint distributions. Their KL is $d ( \pi , \pi _ { 0 } )$ The change in the expectation of a function with range width M is at most M times their total variation distance, and Pinsker’s inequality bounds that distance by the displayed square root.

This elementary inequality is not a new substrate-specific guarantee. Its value here is to expose what a behavioral constraint can buy. It controls additional evaluator disagreement relative to the base under common bounded-error assumptions. It does not guarantee task improvement, and it can be loose. If both rewards lie in [0, 1], then $M \leq 2$ , while the gap itself also has the direct bound $| g ( \pi ) | \le 1$

The bound transfers to all three substrates when the same behavioral divergence is actually controlled. An instruction-edit count alone is insufficient to invoke it. This is a precise reason to distinguish artifact budgets from behavioral budgets.

## When a capacity ordering is justified

Proposition 2 (Inclusion orders worst-case gap). For common rewards and input distribution, if $\Pi _ { A } \subseteq \Pi _ { B }$ , then

$$
\operatorname* { s u p } _ { \pi \in \Pi _ { A } } g ( \pi ) \leq \operatorname* { s u p } _ { \pi \in \Pi _ { B } } g ( \pi ) .
$$

Proof. Every policy considered by the supremum on the left is also considered by the supremum on the right. □

This result is deliberately simple. It identifies the condition needed to turn an expressivity comparison into a worst-case capacity comparison. A weight class that contains all policies induced by a specified prompt family has at least that family’s gap capacity. Whether a given architecture, adaptation method, or compute budget supplies this inclusion is not established by the name of the substrate.

Inclusion also does not imply an ordering of discovered exploits under matched compute. A search over a larger class may fail to locate a policy that a more structured method finds quickly. Thus a prompt optimizer can outperform a training procedure on observed reward without contradicting inclusion of their idealized classes.

Counterexample: farther does not mean more exploitable. Consider three outputs, a uniform base $p _ { 0 } = ( 1 / 3 , 1 / 3 , 1 / 3 )$ , true reward $R = ( 0 , 0 , 0 )$ , and proxy reward $\hat { R } = ( 0 , 1 , 0 )$ Let

$$
\Pi _ { A } = \{ p _ { 0 } , ( 0 . 7 , 0 . 1 5 , 0 . 1 5 ) \} , \qquad \Pi _ { B } = \{ p _ { 0 } , ( 1 / 3 , 0 . 4 , 4 / 1 5 ) \} .
$$

The largest KL distances are approximately $D _ { A } = 0 . 2 8 0$ and $D _ { B } = 0 . 0 1 3$ , yet $G _ { A } = 1 / 3 <$ $0 . 4 = G _ { B }$ . The farther class shifts probability away from the output carrying the positive error; the nearer class shifts toward it. Both contain the baseline. This is a counterexample about signed-gap capacity, not a trajectory with declining true reward, since true reward is constant here.

## The interpretation for language models

The two propositions support a conditional account. Behavioral regularization can constrain an error envelope. A richer class can contain more severe exploits if it includes the relevant policies. But substrate labels alone do not determine how a particular proxy’s errors intersect the class.

Selection is restricted to candidates the generator samples, but can concentrate on a narrow exploit without changing the generator. A prompt can move many outputs together through a shared instruction, which may help it discover a useful rule or a systematic scoring shortcut. Parameter updates can redistribute behavior through many degrees of freedom, but the optimizer may need substantial feedback to discover a particular change. These are differences in geometry and search, not an unconditional safety ladder.

The appropriate practical question is therefore: what useful and exploitative behaviors become reachable as this method’s budget increases, and what evidence distinguishes them? Distance helps answer that question when it is informative about the evaluator’s error. It is not a substitute for identifying the error.

## A finite-output numerical illustration

We use a small exact model so that the distinction between a class, a search path, and a budget can be inspected directly. All distributions and expected rewards are computed over five outputs; no LLM calls or Monte Carlo estimates are involved.

The five outputs are labeled a through e. Their base probabilities, true rewards, two proxies, and conditioning feature are

$$
\begin{array} { r l } & { \quad p _ { 0 } = ( 0 . 5 0 0 , 0 . 3 0 0 , 0 . 1 6 0 , 0 . 0 3 5 , 0 . 0 0 5 ) , } \\ & { \quad R = ( 0 , 0 . 5 0 , 0 . 9 0 , 0 . 4 0 , 0 . 1 0 ) , } \\ & { \quad \hat { R } _ { A } = ( 0 , 0 . 4 5 , 0 . 8 0 , 0 . 9 2 , 1 . 0 0 ) , } \\ & { \quad \hat { R } _ { B } = ( 0 . 9 0 , 0 . 5 0 , 0 . 8 5 , 0 . 4 0 , 0 . 1 0 ) , } \\ & { \quad f = ( - 1 , 0 , 1 , 1 . 5 , 2 ) . } \end{array}
$$

These values fully specify the numerical setup.

Proxy A rewards initially rare, low-quality outputs d and e. Proxy B places its largest positive error on the already common output a. Both rewards and proxies are explicitly on the same zero-to-one scale. The base true reward is 0.3085.

Flexible distribution optimization. The ambient class is the full five-output probability simplex. Its entropy-regularized proxy optimum is

$$
q _ { \lambda } ( i ) = \frac { p _ { 0 } ( i ) \exp ( \lambda \hat { R } ( i ) ) } { \sum _ { j } p _ { 0 } ( j ) \exp ( \lambda \hat { R } ( j ) ) } , \qquad \lambda \in [ 0 , 2 0 ] .
$$

For $\lambda > 0$ , this maximizes $\mathbb { E } _ { q } [ \hat { R } ] - \mathrm { K L } ( q \| p _ { 0 } ) / \lambda$ . The limit at zero is $p _ { 0 }$ . It is a tractable surrogate for flexible policy optimization, not a model of the dynamics or compute costs of neural-network training.

Bounded conditioning. The prompt surrogate is $q _ { t } ( i ) \propto p _ { 0 } ( i ) \exp ( t f _ { i } )$ with $| t | \leq a ,$ where the artifact budget a ranges from zero to two. For each budget, we choose the proxy-best member on a declared grid with spacing 0.002. This is an explicit constrained family, not a claim that real prompts implement exponential tilting. The selected parameter is positive for proxy A and negative for proxy B in this setup.

Selection. For each $n \in \{ 1 , \ldots , 4 0 9 6 \}$ , we compute the best-of-n distribution exactly. Order outputs by increasing proxy score and write $F _ { i }$ for their cumulative base probability. Since the scores are distinct,

$$
q _ { n } ( i ) = F _ { i } ^ { n } - F _ { i - 1 } ^ { n } , \qquad F _ { 0 } = 0 .
$$

Observed behavior. Under proxy $\mathrm { A } ,$ flexible distribution optimization initially improves true reward and then overoptimizes: the maximum on the sampled regularization path is approximately 0.666, followed by 0.398 at $\lambda = 2 0$ . Selection similarly reaches approximately 0.680 before concentrating on output e and approaching true reward 0.100. The bounded conditioning family ends at true reward 0.603 and proxy reward 0.792. Its restricted direction and budget prevent it from concentrating entirely on e.

Under proxy B, both the flexible path and selection deteriorate from the base as they increasingly favor output a. Bounded conditioning can also exploit this nearby error: at budget two its selected policy has true reward approximately 0.042 and proxy reward 0.870. A bounded artifact family is therefore not inherently protected against severe disagreement already accessible near the base.

The ambient full-simplex class contains both the conditioning and selection distributions. Its maximum signed gap is 0.9 for either proxy. Selection can approach this value; a finite regularization sweep need not. Thus a larger gap observed at the endpoint of one evaluated method does not refute Proposition 2. It can reflect the different endpoints chosen for the illustration.

This setup also avoids interpreting a logarithmic divergence bound as an asymptotic ceiling independent of the base. Here selection has a finite limiting divergence because the highestscoring output has positive base mass in a finite space. That conclusion follows from this setup, not from the general expression log n. The implementation and per-setting values are provided as ancillary files.

## Description length as an operational constraint

Prompt budgets remain useful even without a KL equivalence. Limiting examples and edit volume can reduce how much evaluation-specific information is inserted in one iteration. Requiring a general principle alongside an example can make a patch easier to review and challenge. Wahi (2026) uses such constraints in production prompt-optimization loops.

Their interpretation should remain operational. An edit cap is not necessarily a character or token cap; one edit can replace a large block. A per-iteration limit is not a cumulative limit; many small accepted edits can eventually encode a large amount of information. And a short rule can still memorize a compact identifier or introduce a broad scoring shortcut.

A prompt-optimization protocol should therefore record at least total artifact length, added examples, cumulative change, and reuse of evaluation content. When possible, it should also measure behavioral drift on an independent probe set. The combination preserves the practical advantages of text constraints without assigning them a guarantee they do not provide.

## A Defense Correspondence Across Substrates

The common setup suggests that defenses can be compared by what they act on. A defense may improve the proxy, restrict the policies considered, detect disagreement, or prevent an unverified change from being deployed. These functions are complementary. A more accurate judge does not replace access control, and a deployment gate does not establish that an accepted policy generalizes.

We distinguish three levels of evidence. Established mechanism means the cited literature or elementary construction supplies the mechanism in that setting; it does not imply a universal efficacy guarantee. Functional analogy means two methods serve a similar purpose without equivalent formal guarantees. Proposed transfer means a concrete adaptation is worth testing but is not presented here as a measured result. The comparison is representative rather than exhaustive.

## Improve or diversify the evaluation signal

Reward-model ensembles have been studied as a mitigation for overoptimization (Eisenstein et al., 2024), and diverse model panels have been studied for LLM evaluation (Verga et al., 2024).

Table 2: Representative defenses across the three substrates.
<table><tr><td>Defense</td><td>Weights</td><td>Selection</td><td>Persistent text</td></tr><tr><td>Diversify evaluation</td><td>Reward-model ensembles</td><td>Aggregate candidate scores</td><td>Judge panels</td></tr><tr><td>Handle uncertainty</td><td>Penalize uncertain rewards</td><td>Use conservative scores</td><td>Review or abstain on uncertain judgments</td></tr><tr><td>Limit optimization</td><td>Behavioral regularization and stopping rules</td><td>Sample caps and softened Search limits, artifact selection</td><td>budgets, and behavioral checks</td></tr><tr><td>Enforce requirements</td><td>Constrained policy optimization</td><td>Filter candidates</td><td>Check patches, contracts, and outputs</td></tr><tr><td>Preserve independent evidence</td><td>Protected evaluation data Held-out inputs and</td><td>evaluators</td><td>Frozen holdouts and restricted feedback</td></tr><tr><td>Monitor integrity</td><td>Memorization and leakage checks</td><td>Audit suspicious selected outputs</td><td>Canaries and artifact inspection</td></tr></table>

These provide established components for a transfer. Their effectiveness inside a particular persistent prompt-search loop remains a question about that loop, not a consequence of using more than one judge.

Uncertainty-aware judging also exists. Sheng et al. (2025) study conformal intervals for LLMbased scores, while Zhang et al. (2026) propose uncertainty-aware auditing of pairwise judgments. The open question relevant here is how reliably such uncertainty estimates identify the outputs that an adaptive optimizer preferentially discovers. A guarantee under an evaluation distribution should not be silently extended to a shifted distribution produced by search.

The distinction between static calibration and optimization robustness is especially important for prompt-space. A judge can agree better with reference labels on ordinary examples while retaining a small, exploitable class of mistakes. That class may be too rare to dominate the calibration metric and yet dominate the optimized policy.

## Constrain optimization pressure and admissible behavior

Optimization constraints act on different quantities. Training can limit behavioral drift or the number of updates; selection can cap candidates or soften their ranking; prompt search can bound iterations, cumulative text, and examples. Constraints can also restrict which changes are admissible through policy constraints, candidate filters, or patch checks. Their guarantees depend on the quantity actually controlled.

A KL penalty and an edit budget are the clearest example of a useful but incomplete analogy. Both discourage unconstrained movement. Only the former directly constrains the behavioral quantity in Reach, Geometry, and Hacking Capacity, assuming the KL is accurately evaluated. The latter controls a human-readable intervention whose behavioral effect must still be tested.

Stopping rules require a similar distinction. A plateau in proxy reward is not the same as a plateau in true reward, and deterioration may begin before the proxy stops improving. Khalaf et al. (2025) develop inference-time parameter tuning with access to an independent reward signal. For prompt optimization, a defensible stopping protocol likewise needs a protected source of evidence about task quality, rather than simply counting iterations that fail to please the optimizing judge.

Hard constraints can sometimes remove an exploit more directly than regularization. An evaluated agent that cannot inspect cached labels cannot use that particular exfiltration route. A prompt patch that breaks the declared parser contract can be rejected before a judge assesses its prose. Such controls constrain the system around the policy and can be applied regardless of whether the policy was trained, selected, or prompted.

## Worked correspondence: ensembles and judge panels

Suppose two candidate outputs receive similar average judge scores but differ sharply in disagreement among evaluators. A conservative selector may prefer the candidate supported consistently across evaluators, rather than the candidate rewarded exceptionally highly by one. A training objective can use a related penalty, and a prompt optimizer can apply it when ranking candidate patches by their downstream outputs.

The transferable idea is to make exploitation of a single evaluator less attractive. The nontransferable part is the guarantee: evaluator diversity is not equivalent to error independence, and a panel can share a systematic preference for verbosity or familiar phrasing. Repeated optimization against the same panel can expose common blind spots. Accordingly, a panel should be evaluated on exploit transfer and task quality, not only on its internal agreement.

A separate choice concerns what the panel sees. Reviewing a prompt patch asks whether the text appears reasonable. Reviewing resulting outputs asks whether the patch changed behavior appropriately. These are distinct assessments. A panel of patch reviewers cannot replace evaluation of the modified system.

## Worked correspondence: holdouts and information boundaries

A protected evaluation set is a general defense against adaptation to observed examples. Its usefulness is not restricted to any substrate. What differs is how information about that set can reach the optimizer.

For parameter training, exposure can occur through training data, checkpoint selection, or repeated analyst decisions. For selection, the reranker or its operating parameters can be tuned against a supposedly independent benchmark. For prompt optimization, cases, expected answers, or detailed critiques can enter the evolving artifact almost verbatim. The artifact’s readability makes some of this leakage easy to inspect, but does not prevent it automatically.

Wahi (2026) describes frozen partitions with restrictions on what optimizer subagents can see. This is an operational implementation of a broader principle: components that propose changes should not receive unrestricted feedback from the data used to establish generalization. Aggregate feedback can still carry information if it repeatedly influences selection or stopping. A final untouched evaluation set is useful when the monitoring set has become part of the development loop.

Independent inputs and independent evaluators address different problems. New inputs evaluated by the same biased judge can reveal case memorization while missing a judge-wide scoring shortcut. A second evaluator can reveal evaluator-specific exploitation while sharing the same input distribution. A careful protocol states which independence it supplies.

## Worked correspondence: canaries and integrity monitoring

Canaries are deliberately constructed items whose treatment reveals an integrity failure. They have precedents in security and in memorization testing for trained models (Carlini et al., 2019). Their role in a reward-optimization loop depends on what signal they are designed to detect.

In Wahi (2026), an impossible-case canary has an intentionally inappropriate reference answer: ordinary task reasoning should not produce the answer that the scoring system marks as correct. A pass triggers investigation for access to hidden labels or another evaluation failure. This differs from placing a secret sequence in training data and measuring whether the model memorizes it. Both are canary mechanisms, but their threat models and interpretations differ.

Such an integrity tripwire can also monitor outputs from a trained policy or a selection pipeline. It is not intrinsically tied to prompts. The proposed transfer is to integrate it into acceptance and audit procedures around optimization, with the response specified in advance.

A canary pass should be an alarm, not automatic proof of intentional cheating. Coincidence, ambiguous construction, or an unrelated evaluator bug can produce the same signal. Conversely, a selective exploit may avoid a canary. Canaries should therefore be evaluated for detection coverage and false alarms and reported separately from ordinary task accuracy. Including intentionally invalid items in the main quality score would contaminate the objective the system is supposed to improve.

## Detection, authority, and acceptance

Monitoring and authority are separate design choices. A judge, held-out test, or canary can detect a concern, but a system still needs a rule for what happens next. Possible responses include rejecting a candidate, rolling back a prompt, stopping a run, or requesting independent review. The action should depend on the type and strength of evidence.

The PROCTOR architecture described by Wahi (2026) illustrates this separation for prompt patches: mechanical checks can reject a proposal that an LLM reviewer approves, and postapplication evaluation can reveal a regression the reviewer missed. This provides a concrete implementation pattern rather than proof that mechanical checks capture every failure.

The distinction generalizes. A deployment gate for a trained model, a pre-ranking filter for sampled outputs, and a pre-apply check for prompt patches all constrain what is accepted. Their value depends on the requirements they actually test. A failed parser check can be decisive about format; a passed parser check says little about semantic correctness. Keeping the scope of each gate explicit prevents an ensemble of imperfect checks from being described as a correctness oracle.

## Choosing and Combining Substrates

## Choose according to the required change

Selection is attractive when the base model already produces useful candidates often enough and the scoring procedure can reliably distinguish them. It adds inference cost but can avoid maintaining a separately trained model or a large evolving artifact. Its risk is concentration on high-scoring errors, especially when those errors are rare under ordinary sampling and therefore underrepresented in evaluator validation.

Prompt optimization is attractive when a reusable instruction, example, or memory can express the desired change. The resulting artifact is inspectable and reversible. This is particularly useful when the task contains explicit domain rules or output requirements. Its risks include encoding evaluation-specific details, inducing broad unintended behavior through a short change, and adapting to the judge’s language rather than the task.

Weight optimization is attractive when reliable behavior is difficult to elicit through the available conditioning interface or when deployment requires the behavior to be internalized across many inputs. Its effectiveness depends on the adaptation method and feedback. Its risks include persistent exploitation that is harder to identify by inspecting the learned artifact and difficulty attributing a behavioral change to a particular training intervention.

These are selection criteria, not a universal risk ranking. A restricted prompt can directly specify an exploit that a training procedure never discovers; a well-constrained training run can be less harmful than aggressive selection against a poor verifier. The evaluation setup and accessible failure modes matter alongside the substrate.

## Interpreting comparisons of prompting and reinforcement learning

Three variables help organize apparently different comparative results: the behaviors a method can express, how efficiently its optimizer finds them, and what information the feedback supplies. GEPA uses reflective feedback and reports strong results relative to GRPO on its evaluated tasks (Agrawal et al., 2025). Naive Prompt Optimization reports competitive results from a simpler prompt-search procedure and task-dependent advantages for GRPO (Chang and Chen, 2026). Liu et al. (2026) study loophole discovery and find advantages for RL over non-parametric baselines in their setting.

These findings are compatible with the framework without proving its proposed explanation. A task may admit an effective short instruction; another may require changes not easily found through the selected prompt interface. A richer critique may improve search independently of the substrate. Differences in model family, optimizer strength, or compute accounting can also contribute.

Comparative studies should distinguish efficient search from expressive reach. Strong performance with few rollouts may reflect richer feedback or a favorable prompt interface; an RL advantage may reflect behaviors the tested prompt search did not discover. Readable artifacts make interventions easier to inspect, but their quality still has to be established through behavior and generalization.

The widely cited 25× gap in Liu et al. appears in their social-media engagement example. It is useful evidence about that comparison, not a universal multiplier for the hacking capacity of reinforcement learning. More generally, reward magnitudes from different environments should not be pooled into a substrate ranking without a common interpretation.

## Hybrid systems and transferred exploits

Practical systems often combine substrates. A prompt is optimized, several responses are sampled under it, a judge selects one, and selected outputs may later become training data. Prompt and weight optimization can also be jointly beneficial, as explored by Soylu et al. (2024).

For a simple prompt-plus-selection pipeline, the effective policy is

$$
\pi _ { p , n } = \operatorname { B o N } _ { n } ( q _ { 0 } ( { \cdot } \mid p , x ) , { \hat { R } } ) .
$$

The best-of-n KL bound applies relative to the prompt-conditioned generator used to sample candidates. It does not automatically bound divergence from the original baseline prompt, and KL has no general triangle inequality that would justify simply adding arbitrary component divergences.

This interaction has an operational consequence. A prompt can make an exploit more common, after which selection amplifies it. Distillation can then make the selected behavior persistent in weights. A defense applied only to the last step may miss how earlier steps changed the candidate distribution.

Each transition therefore needs an explicit account of what information and behavior it transfers. Candidate logs, prompt versions, scorer versions, and training-data provenance can make that account inspectable. The framework recommends validating the combined policy, even when each component has already been evaluated separately.

## Evidence and Relationship to Prior Work

## What the existing evidence supports

The literature supports reward exploitation across several optimization mechanisms. Its studies use different tasks, evaluators, and measures, so the evidence should be combined at the level it can bear: shared failure mechanisms and informative comparisons, rather than a single universal law fitted across incompatible experiments.

The evidence has complementary roles: empirical overoptimization studies establish failures in specific settings; best-of-n analysis supplies distributional bounds; in-context and production studies demonstrate exploitation through feedback and text. Together they support the comparative framework and its account of how proxy errors interact with different optimization methods.

LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails (Wahi, 2026) motivates several examples in this framework. Its findings include a 100% exploited pass rate versus a 68.1% clean baseline on a 47-case contract-analysis suite, judge-phrasing mimicry, and regression from corrupted labels. These are concrete observations of evaluation failure. They were not collected as a controlled comparison of substrates. The code-quality calibration benchmark also combines 15 human-labeled directories with 39 labeled by a humancalibrated model. Its agreement numbers should therefore be described as agreement with reference labels across the full suite, rather than direct human agreement on every item.

The same source reports that rationale-before-score output ordering improved agreement in one calibration campaign, while several rubric refinements did not. We use this as an example of a measured evaluator intervention. It does not imply that rubric changes are generally ineffective or that output ordering universally resolves judge bias.

## Relation to the main research strands

Reward gaming and proxy optimization. Skalse et al. (2022) formalize reward gaming, while Gao et al. (2023) investigate overoptimization empirically. Wang et al. (2026) provide the PCH synthesis. Our formal section is intentionally narrower than a new general theory of reward hacking: it specifies what distance bounds and class inclusion do and do not imply in a substrate comparison.

Inference-time alignment. Beirami et al. (2025) analyze best-of-n, and Khalaf et al. (2025) study hacking and hedging in inference-time mechanisms. These works make selection a useful reference point because parts of its induced distribution can be characterized directly. The present framework uses those results without assuming that prompt editing inherits their guarantees.

Prompt optimization and in-context adaptation. DSPy, TextGrad, and GEPA provide different mechanisms for optimizing language-model programs or prompts (Khattab et al., 2023; Yuksekgonul et al., 2024; Agrawal et al., 2025). Pan et al. (2024a, 2024b) establish relevant reward-hacking behavior in feedback loops. Wahi (2026) contributes production observations and verification practices. We connect this work through the distinction between a readable artifact and the distribution of behavior it induces.

Evaluator robustness and integrity. Ensembles, diverse panels, uncertainty estimates, and canary-based memorization tests offer mechanisms relevant to several substrates (Eisenstein et al., 2024; Verga et al., 2024; Sheng et al., 2025; Carlini et al., 2019). The defense correspondence organizes how they might be integrated into optimization loops. It distinguishes a published component from an untested combination of components.

The specific contribution. The paper brings these strands into an explicit three-way comparison centered on persistent text optimization. Its contribution is the combination of a common formal setup, an account of reach and capacity, an inspectable numerical example, and a defense map that records the limits of transfer. None requires the claim that reward hacking or the connections between its literatures were previously unrecognized.

## Limitations

The framework does not establish a universal safety ranking. Its propositions bound a quantity under specified assumptions and order it under class inclusion. Real architectures, prompt interfaces, and optimizers need not satisfy a simple inclusion relation. Even when they do, optimization efficiency remains a separate issue.

The numerical illustration is deliberately small. Five outputs make exact inspection possible but omit compositional generation, long trajectories, stochastic environments, and learned evaluators. The flexible distribution path is an analytic optimization surrogate. The bounded conditioning family is designed for illustration. Neither is evidence about the frequency or severity of hacking in deployed LLMs.

True reward is difficult to observe. Independent tests and expert labels can be incomplete or wrong. The distinction between proxy and true reward is mathematically useful, but an empirical study must describe the limitations of its operational reference. The label-pipeline failures documented by Wahi (2026) illustrate why that distinction cannot be taken for granted.

Distance may be difficult to estimate. Exact induced-policy KL can require access unavailable in hosted models, and sequence-level estimation can be expensive or noisy. Artifactlevel constraints remain useful in those settings, but should be evaluated as such rather than described as measured behavioral bounds.

The defense map is a synthesis, not a meta-analysis. It compares functions and access requirements across selected methods. It does not establish effect sizes across tasks or prove that a method that works in one substrate will work equally well in another. These distinctions separate established components from proposed transfers.

Scope is limited by the fixed-environment abstraction. Tool additions, retrieval changes, altered permissions, and evaluator tampering can change the effective problem rather than merely move a policy within it. The framework can accommodate these through expanded policy and environment definitions, but the simple three-substrate presentation does not fully analyze them.

## Conclusion

Reward hacking can arise when weights are updated, when outputs are selected, and when persistent text is revised. The shared mechanism is optimization against a signal that fails to represent the task adequately. The substrate matters because it changes which behaviors are reachable, what information persists, and what can be inspected or constrained.

A useful comparison therefore needs more than a measure of how far a policy moves. A distance-dependent error envelope gives an upper bound; class inclusion gives a capacity ordering; the geometry of accessible behaviors and the effectiveness of search determine what a particular system actually finds. Keeping these statements separate makes the framework applicable without assuming that every proxy produces the same curve or that one substrate is always safest.

For practitioners, the defense correspondence offers the most immediate use. Improve the evaluator where possible, constrain the behaviors that can be accepted, preserve independent evidence of task quality, and specify what happens when that evidence disagrees with the optimizing score. Prompt artifacts make some of these controls unusually inspectable, but inspection must be followed by behavioral evaluation.

Treating weights, selection, and text within one framework makes it possible to reuse insights across optimization settings while keeping their guarantees distinct. The central design principle is to judge an improvement by its effect on the task, supported by evidence independent of the score that produced it.

## References

Wahi, V. (2026). LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails. arXiv:2609.02246. https://doi.org/10.48550/arXiv.2609.02246.

Agrawal, L. A., et al. (2025). GEPA: Reflective prompt evolution can outperform reinforcement learning. ICLR 2026. arXiv:2507.19457.

Beirami, A., Agarwal, A., Berant, J., D’Amour, A., Eisenstein, J., Nagpal, C., and Suresh, A. T. (2025). Theoretical guarantees on the best-of-n alignment policy. ICML. arXiv:2401.01879.

Carlini, N., Liu, C., Erlingsson, Ú., Kos, J., and Song, D. (2019). The Secret Sharer: Evaluating and testing unintended memorization in neural networks. USENIX Security. arXiv:1802.08232.

Chang, Y., and Chen, X. (2026). Naive Prompt Optimization: Rethinking the need for complex prompt search. arXiv:2608.27266.

Eisenstein, J., et al. (2024). Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking. CoLM. arXiv:2312.09244.

Gao, L., Schulman, J., and Hilton, J. (2023). Scaling laws for reward model overoptimization. ICML. arXiv:2210.10760.

Khattab, O., et al. (2023). DSPy: Compiling declarative language model calls into self-improving pipelines. arXiv:2310.03714.

Khalaf, H., Verdun, C. M., Oesterling, A., Lakkaraju, H., and Calmon, F. du P. (2025). Inference-time reward hacking in large language models. NeurIPS. arXiv:2506.19248.

Liu, W., Mou, X., Yan, H., Wei, Z., and He, Y. (2026). Large language models hack rewards, and society. arXiv:2606.04075.

Pan, A., Jones, E., Jagadeesan, M., and Steinhardt, J. (2024a). Feedback loops with language models drive in-context reward hacking. ICML. arXiv:2402.06627.

Pan, J., He, H., Bowman, S. R., and Feng, S. (2024b). Spontaneous reward hacking in iterative selfrefinement. arXiv:2407.04549.

Sheng, H., Liu, X., He, H., Zhao, J., and Kang, J. (2025). Analyzing uncertainty of LLM-as-a-judge: Interval evaluations with conformal prediction. EMNLP. arXiv:2509.18658.

Skalse, J., Howe, N. H. R., Krasheninnikov, D., and Krueger, D. (2022). Defining and characterizing reward gaming. NeurIPS. arXiv:2209.13085.

Soylu, D., Potts, C., and Khattab, O. (2024). Fine-tuning and prompt optimization: Two great steps that work better together. EMNLP. arXiv:2407.10930.

Verga, P., et al. (2024). Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. arXiv:2404.18796.

Wang, X., et al. (2026). Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges. arXiv:2604.13602.

Yuksekgonul, M., et al. (2024). TextGrad: Automatic differentiation via text. arXiv:2406.07496.

Zhang, Z., Hung, Y.-T., He, W., Zhang, J., Ding, L., and Yeh, C.-K. (2026). AURA: Adaptive uncertaintyaware refinement for LLM-as-a-judge auditing. arXiv:2606.19714.