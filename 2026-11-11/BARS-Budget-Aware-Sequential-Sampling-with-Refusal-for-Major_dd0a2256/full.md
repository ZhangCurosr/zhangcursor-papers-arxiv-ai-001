# BARS: Budget-Aware Sequential Sampling with Refusal for Majority Identification under Unknown Distributions

Shenxi Liu<sup>a</sup>, Yiwei Li<sup>a</sup>, Mingyang Zhao<sup>b</sup>, Yuhang Tian<sup>a</sup> and Kan Li<sup>a,∗</sup>

<sup>a</sup>Beijing Institute of Technology, Beijing, 100081, China

<sup>b</sup>Department of Computing, The Hong Kong Polytechnic University, 999077, Hong Kong

## A R T I C L E I N F O

Keywords:   
Sequential sampling   
Majority identification   
Selective prediction   
Anytime-valid confidence sequences   
Test-time computation   
Large language models

## A BS T RA C T

Repeated sampling with majority selection is an efective means of enforcing self-consistency in large language models. Recently, a variant has also been introduced that terminates early to save costs when the output distribution is favorable. But it still cannot yield a correct confidence estimate from the answer distribution. For this issue, we formalize this as majority identification under an unknown distribution with a budget and a target error rate, and propose BARS, a sequential rule certifying the top-two gap and returning an early stop, a budgeted majority vote, or a refusal when feasibility cannot be established. A refusal is a statement about the rule and its budget, not a claim that a question is unanswerable, and it tells practitioners when to widen the budget or switch models. A certified mode carries proven bounds on the joint answering error and on refusing an instance the budget could handle, while a practical mode, used in our experiments, replaces them with empirical calibration. On synthetic instances drawn from known distributions over thousands of seeds, the dificulty-driven behavior is recovered and the price of the certified mode is measured. On GSM8K the practical mode draws 22.3% fewer samples than fixed-8 sampling over the same pool, refused questions included, at the cost of declining 18.8% of the questions. On MATH, it greatly improves the accuracy of its output answers into 82.5% at the cost of honestly refusing to answer half of the dificult questions that the model itself is unable to solve.

## 1. Introduction

Large language models (LLMs) are increasingly embedded in expert and decision-support systems. In such systems a single answer is often produced by several stochastic decodes of the same prompt, and the majority of those decodes is taken as the consensus, a procedure known as self-consistency [1]. Self-consistency markedly improves reasoning accuracy on benchmarks such as GSM8K [2] and MATH [3], yet it applies the same number of decodes to every question. This paper addresses the deployment question that follows: how many samples are enough to trust an answer, and when should the system refuse to answer at all?

A fixed number of decodes fails in two opposite directions, and both failures are common in deployment. When the answer distribution is concentrated on one answer, a few samples already give the accuracy obtained with the full budget, so the remaining samples produce no additional evidence. When the distribution is spread over several answers, further samples still separate them only slowly, and the majority vote becomes a guess. A budget-aware system should therefore adapt its sampling strategy to each question and declare when a reliable answer cannot be produced within the budget.

We refer to the two decisions as the stopping decision (whether the current evidence is already suficient) and the refusal decision (whether the budget can ever be suficient for this question). Section 3 formalizes both, and the remainder of the paper is organized so that each result can be traced to the decision it supports.

We reduce the deployment problem to a statistical primitive with an unknown answer-space size and unknown answer probabilities: drawing ballsfrom a bag with replacement, and identifying the most probable color. We call this the bag-of-balls abstraction (Figure 1): each decode draws one ball, and a color is the parsed answer. The abstraction assumes neither the number of colors nor the shape of the distribution, and two consequences follow. First, it applies to any question whose decodes can be mapped to a finite candidate set and to any model that can be sampled repeatedly. Second, classical probability tools cover the rest of the analysis, because the dificulty of the problem is captured by a single quantity: the top-two gap $\Delta = p _ { ( 1 ) } - p _ { ( 2 ) }$

Formally, we study the following problem.

Within a budget of at most �<sup>∗</sup> samples, identify the majority of an unknown distribution with error probability at most $p ^ { * }$ , conditional on answering; when this target cannot be met, the system must report that it is infeasible.

The analysis assumes independent and identically distributed samples and a unique most probable answer $( \Delta \ > \ 0 )$ . It does not prescribe a distributional shape, but two further conditions bind at the boundaries of the results and are stated where they first matter: the certified width mode requires a known upper bound �<sup>̄</sup> on the number of colors (Assumption 1), and every guarantee concerns the majority answer rather than the correct one. The second condition deserves emphasis: if the model’s most probable response is wrong, the rule identifies it with the same confidence it would apply to a correct answer, and nothing in the analysis detects the diference.

![](images/b6f1292fc9556326e2661cb19cd14a0573417a6a20a960ca78c9922dc0a63e99.jpg)  
Figure 1: The bag-of-balls abstraction. (a) Answer distribution obtained by repeatedly decoding the same question at temperature �; the top-two gap $\Delta = p _ { ( 1 ) } - p _ { ( 2 ) }$ measures the separation between the two most probable answers. (b) Each decode is viewed as drawing a ball, with replacement, from a bag whose size and composition are unknown; the color of the ball is the parsed answer.

We propose BARS (Budget-Aware Refusal Sampling), a sequential stopping rule that maintains the empirical toptwo gap $\hat { \Delta } _ { t }$ and its plug-in variance $\hat { \sigma } _ { t } ^ { 2 }$ and certifies distributional dificulty with time-uniform confidence sequences [4–6]. The rule returns one of three outputs. Thefast branch stops after � consecutive same-color samples, where the run length $w$ is the smallest admissible value for which the worst-case error rate is at most $p ^ { * }$ , computed by a Markov dynamic program. The naive branch spends the remaining budget and takes the majority, with a Hoefding bound controlling its error. The refusal branch outputs “infeasible within budget” when the rule’s own criterion cannot establish feasibility; its rate of firing on instances the budget could have handled is bounded by � plus an explicit, empirically calibrated plug-in bias term. We treat refusal as an informative third output rather than a failure, because it lets a deployed system escalate to a human, widen the budget or switch the underlying model.

Two width modes parameterize the same rule and are kept apart throughout the paper. The certified mode uses a time-uniform half-width whose constants are explicit, so that the trigger and the resulting error bounds hold with probability at least $1 - \delta ;$ it is the mode covered by the theorems. The practical mode uses a variance-scaled half width with a fixed multiplier that controls the deviation only at the current sample size; it carries no theorem-level guarantee and is calibrated empirically, and it is the mode used in our language-model experiments. Each of the three outputs is therefore tied to a declared configuration, and every claim in this paper is marked with the configuration under which it applies.

Our contributions are as follows.

1. A budget-aware stopping and refusal rule with explicit configuration. BARS drives its sampling cost by the certified dificulty of the instance and returns either an answer or a refusal, and it separates two width modes: the certified mode, for which the mistrigger probability over the whole exploration process is bounded by �, and the practical mode, which trades that bound for a higher answer rate.

2. Guarantees on answering and on refusing, together with the boundary of the analysis. For the certified mode with the pessimistic chooser, we prove a bound of $p ^ { * } + \delta$ on the joint event that the rule answers and answers wrongly, a bound of $\delta + B _ { \mathrm { p l u g } }$ on the joint event that it refuses an instance the budget could have handled, an exact decomposition of the expected number of samples, and a lower bound showing that no rule can certify the majority of the two-point family with fewer than $c \Delta ^ { - 2 } \ln ( 1 / \delta )$ samples in expectation. We do not claim a matching upper bound, and we prove the lower bound only for the two-point family (� = 2).

3. An empirical account of when the budget buys what. Synthetic instances over thousands of seeds show the dificulty-driven behavior predicted by the certified-mode analysis, and real-LLM runs on GSM8K and MATH (Qwen2.5-7B, a pre-generated pool of sixteen decodes per question) show the same qualitative pattern at a deployment-scale budget, where the certified mode is provably unreachable and the practical mode is what a practitioner would actually run. A discussion of those runs separates what the two settings establish from what they do not.

What we do not claim is as important as what we do. Our guarantees concern the joint events of answering wrongly and of refusing a handleable instance, not the conditional error rate on answered questions alone, which is smaller only when the answer rate is close to one. The certified bounds hold for the certified width with the pessimistic chooser; the practical mode is calibrated empirically, and $B _ { \mathrm { p l u g } }$ has no distribution-free upper bound, because none exists for a plug-in criterion of this kind. Finally, a refusal is a statement about this rule and this budget, not a proof that the question is unanswerable.

The remainder of the paper is organized as follows. Section 2 compares BARS with the three lines of work it sits between. Section 3 formalizes the decision problem, states the two width modes and gives the rule in pseudocode. Section 4 states the guarantees, in the order in which they depend on one another. Section 5 reports the experiments, in two settings that share one design and one competitor set, and then states what the two establish together and what neither establishes. Section 6 concludes.

## 2. Related work

BARS sits between three lines of work: adaptive testtime sampling, selective prediction with abstention, and best-arm identification with abstention. We compare BARS with each line of work and close with a table that separates them along five dimensions, so that the boundary of this paper can be read of directly rather than assembled from the prose. The analysis itself builds on standard tools — Hoefding’s inequality [7], the empirical Bernstein bound [8], timeuniform confidence sequences [4–6] and the classical theory of runs [9] — and we do not present any of them as a new idea. What the paper contributes is how they are assembled into a decision rule with an explicit refusal output, and what that assembly costs.

## 2.1. Test-time scaling and adaptive sampling

Self-consistency [1] is the fixed-budget baseline for LLM majority voting, and its error analysis typically assumes a finite, known answer space [1]. Adaptive variants include early-stopping self-consistency (ESC) [10], dificultyadaptive self-consistency [11], CGES [12] and sequentialevidence DASE [13]. These methods stop on heuristics that yield empirical savings but carry no provable control of the error rate or of the mis-trigger rate, and none of them treats refusal as an output. Two further groups are close in method but not in objective: test-time scaling studies how accuracy grows with the number of samples when the answer space is known exactly [14–16], and pass-� analyses score the sample set rather than selecting a single answer [16, 17]. The prior work closest to ours in problem setting is MARS [18], which stops on parallel test-time scaling checkpoints using a margin-adversarial criterion. MARS difers in three respects that matter for deployment: its guarantee is heuristic rather than derived under a stated confidence level; it has no “infeasible within budget” output; and it targets parallel traces, whereas we replay the pool as a stream and therefore allow a single sampler.

## 2.2. Selective prediction, abstention, and agreement-based selection

Selective prediction lets a model abstain when it is uncertain. The SelectiveNet family [19, 20] trades coverage against risk through a threshold or a training objective, and conformal abstention [21] with its successors, such as SConU [22] and conformal risk control (CRC) [23], controls the risk of the abstention decision itself. In all of these methods the input is a batch of responses that has already been drawn, and the decision is a single use-or-not choice; there is no sequential sampling decision, and hence no notion of a budget that the decision layer itself consumes. CRC makes this boundary sharp: its impossibility result [23] shows that whenever the base risk exceeds a threshold, any distributionfree method under a coverage constraint must abstain on a fixed fraction of the inputs. That result bounds how often a rule must abstain; it does not tell a deployment what to do with an individual question, because the sampling process that produced the batch is outside its scope. Our refusal output is aimed at exactly that gap: the refusal is justified by online evidence about the dificulty of the instance and by a finite budget, and it can therefore be acted on per question.

Closest in intuition is agreement-based selective prediction [24], which reads cross-head agreement through a Lipschitz version space to obtain a certified selective prediction. It shares our consensus-to-certificate move, but it certifies a fixed, one-shot decision, whereas we certify a stopping time under a budget and use time-uniform confidence sequences in place of version spaces. A second boundary is worth stating explicitly, because it is easy to overstate: certified self-consistency [25] also attaches concentration guarantees to majority voting, with bounds that hold uniformly over the sample size. The diference lies in what is certified and in how the budget enters: our rule is asked, for a fixed budget and target error, either to answer, to stop early, or to refuse with a bounded mis-refusal probability, and the certified analysis covers all three outputs jointly, with the budget entering as a feasibility criterion rather than as a stopping time that is free to grow. The boundary between the two formulations is the point at which we would expect the two lines of work to be compared directly, and we flag it as such rather than claiming a separation that our experiments cannot demonstrate.

## 2.3. Bandits with abstention and best-arm identification

Our work is theoretically adjacent to bandits with abstention and to best-arm identification (BAI). Bandits with abstention [26–28] and Bayesian BAI with abstention [29] optimize regret or budget allocation over an action space the learner chooses from, rather than certifying the output of a single decision layer over a fixed pool. Classical BAI [30–34] identifies the best arm in fixed-confidence or fixedbudget regimes with sample-complexity guarantees but no refusal output, since an arm must always be recommended. The distinction from our formulation is the output alphabet: BARS difers in treating refusal as a third output with a bounded mis-refusal probability, a formulation that fixedconfidence best-arm identification does not consider because it has no budget under which answering could be infeasible. The closest formulation on the impossibility side is the conformal impossibility result discussed above [23], which bounds abstention from outside the sampling process; we bound it from inside, conditionally on the observed evidence.

Nearest families along five dimensions. “Hard budget” means that the total number of samples is bounded by a constant fixed before the run; “refusal bounded” means that the probability of refusing an instance the budget could have handled carries a stated bound.
<table><tr><td>Family</td><td>Answer space</td><td>Outputs</td><td>Hard budget</td><td>Guarantee, and on what</td></tr><tr><td>Self-consistency and its adaptive variants [1, 10, 12, 13, 18]</td><td>discrete</td><td>answer</td><td>no (stop rule free to grow)</td><td>empirical savings; heuristic margin criteria</td></tr><tr><td>Conformal abstention [21-23]</td><td>fixed batch</td><td>answer, abstain</td><td>not applicable</td><td>distribution-free risk and abstention bounds for a one-shot decision</td></tr><tr><td>Agreement-based selective prediction [24]</td><td>fixed features</td><td>answer, abstain</td><td>not applicable</td><td>certified selective risk in a version space</td></tr><tr><td>Certified self-consistency [25]</td><td>discrete</td><td>answer, continue</td><td>no</td><td>concentration bound uniform over sample size</td></tr><tr><td>Best-arm identification [30, 31]</td><td>discrete arms</td><td>arm</td><td>fixed-confidence or fixed-budget</td><td>sample complexity; no refusal</td></tr><tr><td>BARS (this paper)</td><td>unknown discrete</td><td>early stop, answer, refuse</td><td>yes (E*)</td><td>joint answering-error and mis-refusal bounds (certified mode), plus a two-point lower bound</td></tr></table>

## 2.4. Positioning along five dimensions

Table 1 places the four nearest families and BARS on the same five axes. The axes are chosen so that each one corresponds to a place where a deployment decision is made: what the answer space is, what the rule is allowed to output, whether the budget is a hard constraint, what the guarantee is about, and whether the refusal rate is bounded.

## 3. Problem formulation and the BARS algorithm

## 3.1. The decision problem, its outputs and its error quantities

We formulate the deployment problem as identifying the most probable color of an unknown distribution under a sampling budget and an error-rate target.

Problem (budgeted majority identification with refusal). We are given an unknown distribution $p$ over an unknown set of colors, and we may draw iid samples from $p ,$ each sample consuming one unit of a total budget $E ^ { * }$ . The rule must return one of three outputs: an answer (a color), an early stop with an answer that uses fewer than $E ^ { * }$ samples, or an explicit refusal. Conditional on answering, the probability of returning a color other than the true majority must be at most $p ^ { * }$ When answering is in fact possible within the budget, the probability of refusing must be at most �.

The budget is a per-path cap rather than an expectedvalue constraint: no single execution of the rule draws more than $E ^ { * }$ samples, and we report the expected sample count because the quantity of interest in deployment is the average cost over a batch of questions. The only assumptions are that a unique majority exists, that is $\Delta = p _ { ( 1 ) } - p _ { ( 2 ) } > 0$ , and that the samples are iid. The number of colors � and the shape of � are otherwise unrestricted, and we write $\hat { K } _ { t }$ for the number of distinct colors observed up to time �.

Assumption 1 (Bounded number of colors). A known upper bound $\bar { K } \ge K$ on the number of colors is available to the rule; the certified width (Section 3.4) and the feasibility criterion ofthe naive branch (Section 3.5) use �<sup>̄</sup> in place of the unknown �.

Two events recur throughout the analysis, and we fix their notation now. Let $N = E ^ { * } - \tau$ denote the remaining budget at the branch-entry time �. The naive-feasibility event  holds when the Hoefding majority rule of the naive branch meets the target with the remaining budget:

$$
\begin{array} { r } { A : = \left\{ \left( K - 1 \right) e ^ { - N \Delta ^ { 2 } / 2 } \leq p ^ { * } \right\} , } \end{array}
$$

evaluated with the true gap Δ and the remaining budget � at the branch entry; it is an analyst-level notion that the algorithm never observes. Because  is defined through the Hoefding rule, the mis-refusal guarantee is relative to that rule and not to the best possible rule. The certification event  holds when the conservative evidence — the plug-in lower endpoint $d _ { \mathrm { u s e } }$ and, in the certified mode, the color-count bound �<sup>̄</sup> — deems the target infeasible:

$$
\begin{array} { r } { I : = \left\{ ( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } > p ^ { * } \right\} . } \end{array}
$$

A mis-refusal is the joint event ∧refusal: refusing although the naive Hoefding rule would have suficed. The rule can refuse only through the naive branch, so that a refusal is the event {enter the naive branch} $\wedge T ;$ instances that enter the fast branch are answered. The label “infeasible within bud-$\mathrm { g e t } ^ { \prime }$ printed by the rule is therefore a rule-specific statement, not a claim that no rule could answer. Keeping them apart is the reason the paper distinguishes the certified and practical modes throughout: the two modes do not difer in the rule, only in which of these three quantities is bounded.

Notation. � denotes the true distribution, with probabilities arranged in descending order; ̂⋅ denotes the online estimate of the corresponding quantity, and ̄⋅ a known bound used by the certified mode.
<table><tr><td>Symbol</td><td>Meaning</td><td>Notes</td></tr><tr><td> $p , \ p _ { ( 1 ) } > p _ { ( 2 ) } \geq \cdots$ </td><td>true distribution (desc.)</td><td> $p _ { ( 1 ) }$  is the majority</td></tr><tr><td> $K$ </td><td>number of colors (unknown)</td><td>estimated online by  $\hat { K } _ { t }$ </td></tr><tr><td> $\bar { K }$ </td><td>known upper bound on K</td><td>required by the certified mode only (Assumption 1)</td></tr><tr><td> $\Delta = p _ { ( 1 ) } - p _ { ( 2 ) }$ </td><td>top-two gap, the difficulty of the instance</td><td>larger is easier</td></tr><tr><td> $\sigma ^ { 2 }$ </td><td>per-sample variance of the gap</td><td> $p _ { ( 1 ) } ( 1 - p _ { ( 1 ) } ) + p _ { ( 2 ) } ( 1 - p _ { ( 2 ) } ) + 2 p _ { ( 1 ) } p _ { ( 2 ) }$ </td></tr><tr><td> $E ^ { * }$ </td><td>sampling budget</td><td>at most this many samples</td></tr><tr><td> $p ^ { * }$ </td><td>target conditional error rate</td><td>bound on the error of an answered question</td></tr><tr><td> $\delta$ </td><td>mis-trigger bound</td><td>nominal level of the certified width</td></tr><tr><td> $B _ { \mathrm { p l u g } }$ </td><td>plug-in bias of the refusal decision</td><td>no distribution-free bound; calibrated empirically</td></tr><tr><td> $\Delta _ { \mathrm { h i g h } }$ </td><td>fast-branch trigger threshold</td><td>lower-bound claim on ∆</td></tr><tr><td> $T _ { \mathrm { { c a p } } }$ </td><td>exploration cap</td><td>bounds the exploration phase</td></tr><tr><td> $\operatorname* { m i n } _ { - } { \mathrm { t } }$ </td><td>minimum sample count before triggering</td><td>safeguard against degenerate triggers</td></tr><tr><td> $\hat { p } _ { t } , ~ \hat { \Delta } _ { t }$ </td><td>empirical frequencies and gap at t</td><td>maintained online</td></tr><tr><td> $\widehat { \sigma } _ { t } ^ { 2 }$ </td><td>plug-in variance of the gap</td><td>see the formula above</td></tr><tr><td> $w _ { t }$ </td><td>confidence half-width at t</td><td>certified or practical mode</td></tr><tr><td> $\tau$ </td><td>samples used at branch entry</td><td>trigger time</td></tr><tr><td> $w$ </td><td>run length of the fast branch</td><td>chosen by the dynamic program</td></tr><tr><td> $N = E ^ { * } - \tau$ </td><td>remaining budget</td><td>shared by the fast and naive branches</td></tr><tr><td> $d _ { \mathrm { u s e } }$ </td><td>conservative lower endpoint of  $\Delta$ </td><td>max  $( 0 , \hat { \Delta } _ { \tau } - w _ { \tau } ) ,$  used in the feasibility criterion</td></tr><tr><td> $p _ { \mathrm { e r r } } [ w ; q ]$ </td><td>error of the branch rule under candidate q</td><td>exact chain value; see Section 3.5</td></tr><tr><td> $\rho _ { N }$ </td><td>folding residual of the pessimistic chain</td><td> $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 8 }$ </td></tr><tr><td> ${ \mathcal { A } } ,$  I</td><td>naive-feasibility and certification events</td><td>analyst level; defined in Section 3.1</td></tr></table>

## 3.2. Notation

All symbols are collected in Table 2. We distinguish the true number of colors �, its online estimate $\hat { K } _ { t }$ and the bound �<sup>̄</sup> required by the certified mode, and we use � for the run length of the fast branch and $w _ { t }$ for the confidence half-width of the exploration phase; the two are never interchanged.

## 3.3. The rule

BARS alternates between exploration and one of three decision branches, and the pseudocode is given in Algorithm 1. Sections 3.4 and 3.5 state the two ingredients it relies on. We describe the rule as a sequence of decisions rather than as three phases, because the three outputs are mutually exclusive at the level of a single question: once the trigger fires, the naive and refusal branches are unreachable for that question.

## 3.4. Exploration and the two width modes

The validity of the trigger $\hat { \Delta } _ { t } - w _ { t } \ge \Delta _ { \mathrm { h i g h } }$ depends on whether $w _ { t }$ is a valid time-uniform lower confidence halfwidth for the true gap Δ. We provide two choices, and the choice is the only place in the rule where a guarantee is traded for a higher answer rate.

• Certified mode. With $\bar { K }$ as in Assumption 1 — in the LLM setting one may take the size of the parsed answer space — set

$$
w _ { t } = 2 \sqrt { \frac { 2 \bar { \sigma } _ { t } ^ { 2 } ( 2 \ln t + \ln \frac { \bar { K } \pi ^ { 2 } } { 3 \delta } ) } { t } } ,
$$

where $\hat { \sigma } _ { t } ^ { 2 }$ is the plug-in variance, floored as described below. The leading factor 2 absorbs the Lipschitz constant of the top-two gap functional, and the ln �<sup>̄</sup> term pays for a union bound over colors; both are made explicit so that $w _ { t }$ is a time-uniform lower confidence half-width for the true $\Delta \colon$ on the good event  ofprobability at least $\begin{array} { r } { 1 - \delta , | \hat { \Delta } _ { t } - \Delta | \leq w _ { t } } \end{array}$ holds simultaneously over the exploration times (Corollary 1). Because the plug-in variance of a batch of identical draws is exactly zero, the variance used in $w _ { t }$ is floored at a positive value $\sigma _ { \operatorname* { m i n } } ^ { 2 } > 0$ that decays with �; without the floor the width would vanish on the first samples and the trigger would fire on no evidence (Section 4.1, Remark 1). The drawbacks are that its constants are conservative and that the certification time

$$
T _ { \mathrm { c e r t } } : = \mathrm { m i n } \big \{ t \geq \mathrm { m i n \_ t } : w _ { t } \leq \hat { \Delta } _ { t } - \Delta _ { \mathrm { h i g h } } \big \} ,
$$

the smallest � at which the certified trigger can fire, scales roughly as $( \Delta \mathrm { ~ - ~ } \Delta _ { \mathrm { h i g h } } ) ^ { - 2 } ;$ small budgets are therefore pushed towards refusal. Section 5.2.4 reports its explicit values.

Algorithm 1 BARS: budget-aware adaptive majority voting with refusal   
Require: budget �<sup>∗</sup>, error $p ^ { * }$ , threshold $\Delta _ { \mathrm { h i g h } } ,$ confidence $\delta ,$ cap $T _ { \mathrm { c a p } } ,$ min\_t, width mode (certified or practical)   
Require: Choose-� mode (pessimistic or optimistic)   
1: Initialize counts $ \{ \} , t  0 ,$ run ← 0, last $\gets \emptyset$ ⊳ �: samples used   
2: repeat   
3: $c \gets \mathbf S _ { i }$ AMPLING() ⊳ drawing a ball from the bag (one LLM decode)   
4: � ← � + 1; counts[�] ← counts[�] + 1 ⊳ update counts   
5: $( \hat { p } _ { t } , \hat { \Delta } _ { t } , \hat { \sigma } _ { t } ^ { 2 } ) \gets \mathrm { T o p } 2 \mathrm { S T A T S } ( \mathrm { c o u n t s } , t )$ ⊳ online estimates   
6: $w _ { t } \gets \dot { \mathrm { W I D T H } } ( \hat { \sigma } _ { t } ^ { 2 } , t )$ ⊳ certified or practical   
7: i $\mathrm { : } t \geq$ min\_t and $\hat { \Delta } _ { t } - w _ { t } \geq \Delta _ { \mathrm { h i g h } }$ then   
8: $\tau \gets t ;$ goto FASTBRANCH ⊳ enter the fast branch   
9: end if   
10: if $t = T _ { \mathrm { c a p } }$ then   
11: $\tau \gets t ;$ goto NAIVEBRANCH   
12: end if   
13: until false   
14: function FASTBRANCH   
15: $N \gets E ^ { * } - \tau$ ⊳ remaining budget   
16: � ← CHOOSE $\mathbf { W } ( \widehat { p } _ { \tau } , N , p ^ { * }$ , mode) ⊳ dynamic program; mode is pessimistic or optimistic   
17: if � ≠ NONE then   
18: repeat   
19: � ← SAMPLING(); � ← � + 1; counts[�] ← counts[�] + 1   
20: run ← (� = last)? run + 1 ∶ 1; last ← �   
21: if run ≥ � then   
22: break   
23: end if   
24: until $t \geq E ^ { * }$   
25: return MAJORITY(counts) ⊳ majority over the full trajectory   
26: else   
27: goto NAIVEBRANCH ⊳ no admissible window within budget: fall back to the feasibility criterion   
28: end if   
29: end function   
30: function NAIVEBRANCH   
31: $N \gets E ^ { * } - \tau$ ⊳ remaining budget; set here as well, because the cap can route directly to this branch   
32: $d _ { \mathrm { u s e } }  \operatorname* { m a x } ( 0 , \ \hat { \Delta } _ { \tau } - w _ { \tau } )$ ⊳ conservative lower endpoint of Δ   
33: $K _ { \mathrm { c r i t } }  \hat { K } _ { \tau }$ ⊳ observed distinct colors   
34: if certified mode then   
35: $K _ { \mathrm { c r i t } }  \bar { K }$ ⊳ use the Assumption 1 bound, as the theorems do   
36: end if   
37: if $( K _ { \mathrm { c r i t } } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } \le p ^ { * }$ then   
38: take � more balls; return MAJORITY(counts)   
39: else   
40: return REFUSE ⊳ infeasible within budget   
41: end if   
42: end function

• Practical mode.

$$
w _ { t } = z \sqrt { \hat { \sigma } _ { t } ^ { 2 } / t } , \mathrm { t y p i c a l l y } z = 2 .
$$

This is a heuristic, variance-scaled width: it controls the deviation only at the current sample size and carries no time-uniform guarantee, so its safety is calibrated empirically rather than proved. Experiments indicate that $z \geq 3$ substantially mitigates mis-triggers on hard instances; we recommend $z = 2$ for interactive deployment, and $z \geq 3$ when stricter mis-trigger control is required (quantified in Section 5.2.3).

The certified mode is used when an auditable certificate is required; the practical mode is the configuration we expect an operator of a small sampling budget to run.

## 3.5. The algorithm stages

Exploration. As samples arrive, the rule updates the empirical frequencies $\hat { p } _ { t } ,$ , the top-two gap $\hat { \Delta } _ { t }$ and the plug-in variance $\hat { \sigma } _ { t } ^ { \bar { 2 } }$ , tests the trigger $\hat { \Delta } _ { t } - w _ { t } \ge \Delta _ { \mathrm { h i g h } }$ and, if the trigger has not fired by $t \ = \ T _ { \mathrm { c a p } }$ , leaves the exploration phase. The trigger has two safeguards: min\_t prevents it from firing before any evidence on the variance exists, and $\Delta _ { \mathrm { h i g h } }$ requires the certified lower endpoint to clear a strictly positive threshold.

Thefast branch. For a fixed run length � and a candidate distribution � over the observed colors, the dynamic program evaluates the exact error probability of the branch’s answering rule,

$$
\begin{array} { r l } & { p _ { \mathrm { e r r } } [ w ; q ] = \mathbb { P } _ { q , N } \mathrm { ( t r a j e c t o r y - m a j o r i t y ~ e r r o r ~ a t ~ t h e ~ s t o p ) } } \\ & { \qquad = \big ( U ( w ; q ) - W ( w ; q ) \big ) + V ( w ; q ) + F ( w ; q ) , } \end{array}
$$

where � is the probability that the first �-run is completed by a non-majority color (run part); � the probability of that event with the majority still leading at the stop, in which case the trajectory-majority answer is correct; � the probability that the majority completes the first �-run but does not lead at the stop (count-gap part, which earlier analyses of runbased stopping omitted); � the probability that the majority completes the run while still leading, so that stopping on a run has total probability $U + V + X$ ; and � the probability that no color completes a run within � samples and the fallback majority is wrong. Since $W \geq 0$ , summing only $U + V + F$ gives a conservative upper bound, whereas the chain computes the exact value $U - W + V + F ;$ the construction of the chain, its state count and the residual that the pessimistic mode incurs are given in Appendix B. The chosen run length is the smallest admissible � for which the worst-case error rate, residual included, is at most $p ^ { * }$ , and the branch stops at the first time the last � samples share a color and answers with the majority of the full trajectory. The pessimistic mode evaluates $p _ { \mathrm { e r r } }$ at the distribution $q ^ { \mathrm { p e s s } }$ that pushes the majority probability down to its confidencelower endpoint $\hat { p } _ { ( 1 ) } - w _ { \tau }$ and merges the residual mass into a single filler color; the optimistic mode evaluates it at the plug-in distribution $\hat { p } _ { \tau } .$ . The theorems of Section 4 hold for the pessimistic mode; the optimistic mode is an empirically validated relaxation.

The naive branch and refusal. With $d _ { \mathrm { u s e } } = \operatorname* { m a x } ( 0 , \hat { \Delta } _ { \tau } -$ $w _ { \tau } )$ as the conservative lower endpoint of Δ (the certified width already absorbs the factor-2 and ln �<sup>̄</sup> constants), the branch checks $( K _ { \mathrm { c r i t } } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } \le p ^ { * }$ , where $K _ { \mathrm { c r i t } } = \hat { K } _ { \tau }$ in the practical mode and $K _ { \mathrm { c r i t } } = \bar { K }$ in the certified mode. If the criterion holds, the branch spends the remaining budget and takes the majority, and its error is controlled by the Hoefding bound of Section 4.3; otherwise it refuses. The naive branch is retained even though our experiments never reach it, because the trigger and the criterion are logically independent: a question that fails the trigger can still be feasible, and a rule that dropped this branch would have no path for such questions.

## 3.6. Cost model

All decision logic runs on the CPU: TOP2STATS sorts in $O ( \hat { K } _ { t }$ log $\hat { K } _ { t } ) _ { : }$ , the width costs �(1), and the dynamic program has $O ( \hat { K } _ { \tau } w )$ states for the run part plus $O ( w N )$ for the count-gap tracking (Appendix B). For language models the dominant cost is the number of forward passes, so the sample count is the cost metric throughout this paper.

## 4. Theory

The results are ordered as they depend on one another, so that each one uses only what precedes it. The chain is: simultaneous coverage of the estimated gap, which certifies the trigger; the transfer of the fast-branch analysis to the worst-case distribution inside the confidence set, which certifies the fast branch; the two bounds that constitute the main deliverable, on answering and on refusing; the cost decomposition, which is what makes the rule budget-aware; and a lower bound that says how much of that cost is unavoidable. Proofs are collected in Appendix A.

## 4.1. Simultaneous coverage and the certified trigger

We build on two classical tools. Hoefding’s inequality controls the deviation of an empirical mean from its expectation [7], and an empirical-Bernstein time-uniform confidence sequence controls the deviation simultaneously over all times [4, 5, 8]; the second is the direct source of the certified width.

Lemma 1 (Time-uniform confidence sequence, with the additive term retained). Let $X _ { t }$ be iid and bounded in [0, 1], with mean � and variance $\sigma ^ { 2 } ,$ , let $\hat { \sigma } _ { t } ^ { 2 }$ be the empirical variance, and let $\begin{array} { r } { c _ { t } : = 2 \ln t + \ln { \frac { \pi ^ { 2 } } { 3 \delta } } } \end{array}$ . For every exploration time $t \geq 2 ,$ , define

$$
w _ { t } = \sqrt { \frac { 2 \hat { \sigma } _ { t } ^ { 2 } c _ { t } } { t } } + \frac { 7 c _ { t } } { 3 t } .
$$

Then

$$
\mathbb { P } \Big ( \exists t \geq 2 : | \bar { X } _ { t } - \mu | \geq w _ { t } \Big ) \ \leq \ \delta .
$$

Remark 1 (Why the additive term is not optional). The width has two terms and both are needed. The square-root term alone vanishes when the first draws coincide, because the plug-in variance of a constant batch is exactly zero; a rule that used it alone would certify a gap from no evidence at all. The additive term $7 c _ { t } / ( 3 t )$ never vanishes, and it is the term that keeps the certified trigger valid on thefirst samples. Two conventions are in use in the literature and they difer here by afactor we quantify, because the diference matters for the reported budgets: the alternative convention absorbs the additive term into the multiplicative constant, which is safe only while the plug-in variance is bounded away from zero. The implementation and the figures of Section 5.2.4 use that second convention. At the exploration times reached there, the additive term is 5.7%, 17.7% and 67.6% of the square-root term for $\Delta ~ = ~ 0 . 2 , ~ 0 . 4$ and 0.8, so retaining it multiplies the certification time by about 1.1, 1.4 and 3.2 respectively, while leaving the $( \dot { \Delta } - \Delta _ { \mathrm { h i g h } } ) ^ { - 2 }$ scaling unchanged. We report both sets of values in Section 5.2.4. The simulations confirm that the degenerate regime is not a corner case: with min $\underline { { \mathfrak { t } } } \ : = \ : 2$ the square-root-only width certifies a gap on instances with $\begin{array} { r l r } { \Delta } & { { } < } & { \Delta _ { \mathrm { h i g h } } } \end{array}$ in $3 2 . 6 -$ 32.9% of runs, against 0 of 200,000 for the two-term width (Section 5.2.4).

The details of the variance endpoint, the constants and the union bound over colors are in Appendix $\mathrm { A . 1 }$ . Because the order statistics $p _ { ( 1 ) } ~ \geq ~ p _ { ( 2 ) } ~ \geq ~ \cdots$ are 1-Lipschitz in the $\ell _ { \infty }$ sense, the same concentration bound transfers to the empirical top-two gap: $| \hat { \Delta } _ { t } - \Delta | \leq 2 \operatorname* { m a x } _ { c } | \hat { p } _ { c } - p _ { c } |$ Combining the two with a union bound over the at most $\bar { K }$ colors yields the certified trigger used by the rule.

Corollary 1 (Certified trigger). Let $\bar { K }$ be the known upper bound on the number ofcolors (Assumption 1), and let

$$
w _ { t } = 2 \left[ \sqrt { \frac { 2 \hat { \sigma } _ { t } ^ { 2 } \bar { c } _ { t } } { t } } + \frac { 7 \bar { c } _ { t } } { 3 t } \right] , \qquad \bar { c } _ { t } : = 2 \ln t + \ln \frac { \bar { K } \pi ^ { 2 } } { 3 \delta } .
$$

Then, on the event  ofprobability at least $1 - \delta ,$

$$
\mathbb { P } \Big ( \exists t \geq 2 : | \hat { \Delta } _ { t } - \Delta | \geq w _ { t } \Big ) \leq \delta ,
$$

and consequently, on ${ \mathcal { G } } ,$ the trigger $\hat { \Delta } _ { t } - w _ { t } \ge \Delta _ { \mathrm { h i g h } }$ certifies $\Delta \geq \Delta _ { \mathrm { h i g h } } ,$ ; the factor 2 pays for the Lipschitz constant and ln �<sup>̄</sup> for the union bound over colors.

## 4.2. The fast branch

Entering the fast branch does not give the rule the true distribution, only a confidence set of radius $w _ { \tau }$ around the empirical distribution. Our analysis therefore evaluates the branch’s error at the most adversarial distribution inside that set, obtained by a pessimistic redistribution in which the majority probability is pushed to its minimum and the residual mass is merged into a single filler color. The next lemma states the dominance relation that makes this reduction legitimate, and it is why the dynamic program can be evaluated at a single distribution.

Lemma 2 (Pessimistic dominance). Fix the confidence configuration $( \hat { p } _ { \tau } , w _ { \tau } )$ at branch-entry time �, let $\begin{array} { r l } { m } & { { } = } \end{array}$ $\hat { p } _ { ( 1 ) } - w _ { \tau }$ be the conservative lower endpoint ofthe majority probability, let $q ^ { \mathrm { p e s s } }$ be the folded two-color distribution whose majority probability is � and whose remaining mass sits on a singlefiller color, and let $N = E ^ { * } - \tau .$

Then for every distribution � with $p _ { ( 1 ) } ~ \geq ~ m -$ in particular every � in the confidence set on $\mathcal { G } - e \nu e r y$ run length $w \ge 1$ and the same initial counts,

$$
\begin{array} { r l } & { \mathbb { P } _ { p , N } \big ( t r a j e c t o r y - m a j o r i t y  { e r r o r } a t  { t h e } b r a n c h  { s t o p } \big ) } \\ & { \qquad \le  { p } _ { \mathrm { e r r } } [ w ; q ^ { \mathrm { p e s s } } ] + \rho _ { N } , } \end{array}
$$

where $p _ { \mathrm { e r r } }$ is the exact chain value ofSection 3.5 and $\rho _ { N } =$ $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 8 }$ is the folding residual of Appendix A.3.

The residual is the price of the folding and it is small for the same reason that the problem is easy: on the paths where the folded process is killed by a filler run while the majority still leads, the unfolded process continues, and it can err only if some minority color overtakes the majority against a negative drift. This is a large-deviation event: on $\mathcal { G }$ the Hoefding inequality combined with the reflection principle bounds its probability by $( \bar { K } \mathrm { ~ - ~ } 1 ) e ^ { - N \Delta ^ { 2 } / 8 } \ \leq \ \rho _ { N }$ , where $d _ { \mathrm { u s e } } ~ \leq ~ \Delta$ is used. On the tested confidence-set grid this residual is numerically below $1 0 ^ { - 5 }$ (Section 5.2.4).

Theorem 1 (Fast-branch error, certified mode and pessimistic chooser). Run the rule with the certified width of Corollary 1 and the pessimistic Choose-�. Conditioned on the good event , ifthe dynamicprogram returns a run length � with $p _ { \mathrm { e r r } } [ w ; q ^ { \mathrm { p e s s } } ] + \rho _ { N } \leq p ^ { \ast }$ , then

<sub>ℙ</sub>(<sub>wrong answer ∣ enter thefast branch,</sub> ${ \mathcal { G } } ) ~ \leq ~ p ^ { * }$

because the error event at the stop is the union of three disjoint events that the chain enumerates: a minority completes the first �-run and leads at the stop, the majority completes it without leading, or no run completes within � samples and the branch falls back to the trajectory majority, the last being controlled by the majority Hoefding bound of Theorem 2. The criterion $p _ { \mathrm { e r r } } + \rho _ { N } \le p ^ { * }$ charges the residual to the same $p ^ { * }$ budget; numerically $\rho _ { N } \leq 1 0 ^ { - 5 }$ , so the chooser’s output is unafected.

The fast branch answers with the majority of the whole trajectory rather than with the color that completed the run, and the diference can be quantified exactly.

Proposition 1 (Trajectory-majority benefit). Consider the fast branch entered with run length � under , and let �, �, �, � be the chain quantities ofSection 3.5 evaluated at $q ^ { \mathrm { p e s s } }$ . Conditional on a run completing, the run-color rule errs on the mass � and the trajectory-majority rule on the mass $( U - W ) + V$ , so the two error probabilities difer by exactly

$$
P _ { \mathrm { e r r } } ^ { \mathrm { r u n } } - P _ { \mathrm { e r r } } ^ { \mathrm { t r a j } } \ = \ \frac { W - V } { U + V + X } ,
$$

where $U + V + X = \mathbb { P } ( t h e$ branch stops at a �-run): � has a minority completing the run and leading at the stop, � has the majority completing the run without leading, and � has the majority completing it while still leading. The sign of $W - V$ is notfixed by the identity: the trajectory majority is better exactly when $W > V$ , and Section 5.2.4 reports that $W - V$ was non-negative at every point of the tested grid, with the gap growing in the remaining budget �.

## 4.3. The naive branch, refusal, and the main guarantees

After � iid samples the majority is not the true majority with probability at most $( \bar { K } - 1 ) e ^ { - N \Delta ^ { 2 } / 2 }$ , by Hoefding’s inequality and a union bound over the at most $\bar { K }$ colors. This bound is what the feasibility criterion of the naive branch evaluates at the conservative endpoint $d _ { \mathrm { u s e } }$ instead of at $\Delta$

Theorem 2 (Joint answering and mis-refusal guarantees, certified mode and pessimistic chooser). Run the rule with the certified width of Corollary 1 and the pessimistic Choose-�. Then,for any true distribution � with $\Delta > 0$

$$
\begin{array} { r } { \underbrace { \mathbb { P } ( w r o n g m a j o r i t y \wedge a n s w e r s ) } _ { a n s w e r i n g e r r o r ( j o i n t ) } \le p ^ { * } + \delta , } \\ { \underbrace { \mathbb { P } ( r e f u s e s \wedge \mathscr { A } ) } _ { m i s r e f u s a l ( j o i n t ) } \le \delta + B _ { \mathrm { p l u g } } , } \end{array}
$$

where  is the naive-feasibility event ofSection 3.1 and

$$
B _ { \mathrm { p l u g } } ~ = ~ { \mathbb P } \big ( r e f u s e s \wedge A \wedge \mathcal { G } \big )
$$

is the plug-in bias: it is nonzero only when the conservative endpoint $d _ { \mathrm { u s e } }$ falls short of the true Δ far enough that the criterion refuses an instance the budget would have handled.

Three remarks delimit what Theorem 2 guarantees and what does not. First, the joint form is the natural statement for guarantees that involve the observed events $\ " \mathrm { a n - }$ swers” and “refuses”; conditioning them out gives $( p ^ { * } +$ �)∕ℙ(answers) and $( \delta + B _ { \mathrm { p l u g } } ) / \mathbb { P } ( \mathcal { A } )$ respectively, which reduce to the joint bounds when the answer rate, or the feasibility rate, is close to one. We therefore quote the joint form here and the conditional numbers in the experiments, and we never convert between them implicitly. Second, the suficient condition for $B _ { \mathrm { p l u g } } = 0$ is $d _ { \mathrm { u s e } } = \Delta$ together with $K \ = \ \bar { K }$ ; when $K ~ < ~ \bar { K }$ the conservatism introduced by the color-count bound is a further source of mis-refusal that the plug-in bias alone does not capture. Third, $B _ { \mathrm { p l u g } }$ has no distribution-free upper bound, and we do not claim one: it is calibrated empirically in Section 5, and the same statement is why the certified mode is not ofered as a guarantee of the practical one. Finally, the guarantee of the practical width mode is empirical rather than a consequence of Theorem 2; the two modes difer in the width alone.

## 4.4. Expected sampling cost

For a Bernoulli sequence with success probability $q ,$ the expected number of trials until the first run of � consecutive successes is $( 1 - q ^ { w } ) / \big ( ( 1 - q ) q ^ { w } \big )$ [9]. This identity is what makes the cost of the fast branch predictable in closed form.

Theorem 3 (Expected-samples decomposition). Let $P _ { \mathrm { F } } ,$ $P _ { \mathrm { N } }$ and $P _ { \mathrm { R } }$ denote the fast, naive and refusal branch probabilities, and let $\tau _ { \mathrm { F } }$ and $\tau _ { \mathrm { R } }$ denote the fast-branch trigger time and the refusal decision time; here $\tau _ { \mathrm { R } } = T _ { \mathrm { c a p } }$ when the branch is enteredfrom the main loop, and $\tau _ { \mathrm { R } } = \tau _ { \mathrm { F } }$ when the fast branchfalls back. Write $T _ { \mathrm { r u n } } ( \hat { p } _ { ( 1 ) } , w )$ for the run waiting time with Bernoulli parameter equal to the plug-in majority probability at branch entry. Then

$$
\begin{array} { r l } & { \mathbb { E } [ t o t a l s a m p l e s ] = P _ { \mathrm { F } } \cdot \left( \mathbb { E } [ \tau _ { \mathrm { F } } ] + \mathbb { E } [ \operatorname* { m i n } ( T _ { \mathrm { r u n } } ( \hat { p } _ { ( 1 ) } , w ) , N ) ] \right) } \\ & { \quad \quad \quad + P _ { \mathrm { N } } \cdot E ^ { * } + P _ { \mathrm { R } } \cdot \mathbb { E } [ \tau _ { \mathrm { R } } ] , } \end{array}
$$

with $N = E ^ { * } - \tau _ { \mathrm { F } } ;$ the truncation accounts for the fact that the fast branch cannot run past the budget, and it is immaterial when � exceeds the typical waiting time. The identity is exact when the two-color waiting time is used; with the true multi-color race the fast-branch term is an upper bound, and Monte Carlo verification matches the dynamic-programming output to $1 0 ^ { - 4 }$

Because the decomposition is closed-form, the amortized cost of a batch can be computed before the batch is run, and a budget can be chosen to match a latency target. We do not use this bound to explain the measured sample counts, because it is conservative by construction: it charges the exploration cap and ignores the samples saved by early refusal.

## 4.5. Why small budgets force refusal: a lower bound

The last result explains why refusal is not an implementation artefact but a consequence of the budget. We consider a two-point family of distributions $\{ P ^ { + } , P ^ { - } \}$ whose majorities are two diferent colors and whose top-two gap is the same Δ. Their single-sample KL divergence is Δ ln $\begin{array} { r l } { \frac { 1 + \Delta } { 1 - \Delta } = } & { { } } \end{array}$ $2 { \Delta } ^ { 2 } + { O } ( { \Delta } ^ { 4 } )$ . By Le Cam’s two-point method together with Stein’s lemma [35] we obtain the following bound.

Theorem 4 (Two-point lower bound). Let $\delta ~ \in ~ ( 0 , \frac { 1 } { 4 } ) .$ Any sequential rule that outputs the correct majority with probability at least 1−� on each distribution ofthe two-point family satisfies, in expectation,

$$
n \geq c \Delta ^ { - 2 } \ln \frac { 1 } { \delta }
$$

for an absolute constant $c > 0 .$

## 5. Experiments

The experiments divide into two settings and: randomseed synthetic instances whose generating distributions are known, where the predictions of the rule can be checked exactly, and a real model at a deployment-scale budget. The two settings share one implementation of the rule and one set of competing rules.

## 5.1. Experiments design

Competing rules. We compare against two families of competitors in Table 1. The first family contains fixedbudget and fixed-confidence best-arm-identification (BAI) rules that are evaluated on the same sampled stream: the fixed-budget majority rule with the full budget (FIXED-�), Successive Rejects [30], Sequential Halving [32], LUCB [32], lil’UCB [33] and the Hoefding race [36]. The second family contains the sequential early-stopping rules designed for LLM majority voting that we reviewed in Section 2.1: early-stopping self-consistency (ESC) [10], its dificultyadaptive ofspring [11], confidence-guided early stopping (CGES) [12], the sequential-evidence rule DASE [13], the margin-adversarial rule MARS [18] and the martingale majority certificate of certified self-consistency [25]. Each competitor is implemented from its published stopping rule, at the parameters its authors report as default, and every rule consumes the same sampled stream, uses the same answer normalisation and is scored against the same reference; the implementations and their exact stopping criteria are listed in Appendix C. Two of the six LLM competitors need information the pool-replay protocol does not carry: CGES scores answers by their decoding confidence and MARS needs partial-trace states. We therefore evaluate CGES in its confidence-free form, where the posterior over candidate answers is the Dirichlet-multinomial posterior implied by the counts, and MARS in its answer-level form, where the margin is required to exceed a conservative bound on future vote movement; both reductions are stated so that they are not mistaken for the full methods.

The FIXED-� rule and the BAI rules always return an answer, since a best arm is always recommended; BARS and the LLM early-stopping rules can refuse or fall back. A comparison must therefore separate two quantities that a single accuracy number would conflate: the error rate on the instances a rule answers, and the rate at which a rule correctly declines instances for which no budget-feasible answer exists. We define the second quantity against an oracle that knows �: an instance isfeasible when the oracle’s criterion $( K - 1 ) e ^ { - N \Delta ^ { 2 } / 2 } \le p ^ { * }$ holds at the full budget, and we report the mis-refusal rate (refusing a feasible instance) and the correct refusal rate (refusing an infeasible one) separately. This is the synthetic counterpart of the selectiveprediction convention of Section 2.2, and it is the only protocol under which a rule that declines to answer is not penalised for a decision that the fixed-budget baselines are structurally unable to make.

Reporting rules. An answer rate is the fraction of questions the rule answers, so that its complement is the refusal rate and the two sum to one, as in selective prediction. Accuracy on answered questions divides by the number of answered questions, whereas the mean sample count divides by all questions of the pool and therefore includes the questions that were refused early; the 22.3% figure below is of the second kind, and we state it with its denominator every time it is quoted. The synthetic cells report conditional error rates averaged over seeds, and the LLM cells report proportions with Wilson 95% intervals, because the latter are binomial proportions over 80 questions and a standard deviation would misstate their uncertainty; the synthetic figures are exact up to the Monte Carlo verification of the dynamic program $( 1 0 ^ { - 4 } )$ and the seed count. Competitor curves are drawn with the same seeds and the same instance batches as the BARS curves throughout, so every point in the synthetic figures is a paired comparison.

## 5.2. Random-seed synthetic experiments

Three properties make synthetic instances the right place to begin. First, they cover the situations a deployed system actually meets: a grid of dificulties in the top-two gap Δ that runs from instances the budget settles easily to instances it cannot settle at all, crossed with budgets from tens to hundreds of samples, so that the rule is not tested only where it happens to be comfortable. Second, the generating distribution is known, because it is the distribution we drew from: the oracle feasibility criterion $( K - 1 ) e ^ { - N \Delta ^ { 2 } / 2 } \le p ^ { * }$ the true top-two gap $\Delta$ and the true majority are all exactly computable, and none of the three is available on a real dataset. Third, and because of the second, the synthetic side is where the rule can be audited rather than merely scored: the certified mode’s predictions can be checked against the quantity they predict, and the rule can be priced separately where it is right and where it is wrong. Section 5.3 can do none of this; it answers a diferent question, namely whether the behavior survives contact with a real model.

## 5.2.1. Instances and the synthetic parameter set

The instances are drawn from a Δ-grid family: the two leading colors split 0.8 of the probability mass with a gap of exactly $\Delta ,$ and the remaining eight colors share the residual 0.2 equally, so that dificulty is the single axis along which the instances difer. Default parameters are $E ^ { * } = 2 0 0 , p ^ { * } =$ 0.05, Δ<sub>high</sub> = 0.1, � = 0.05, $T _ { \mathrm { c a p } } = 1 5 0$ and min $\underline { { \boldsymbol { \mathrm { ~ t ~ } } } } = 8$ , with the practical width $( z = 2 )$ and the optimistic chooser; the main cells use 2000 seeds and the tuning cells 600–1200.

## 5.2.2. Behavioral signature across dificulty

Figure 2 shows how the practical mode behaves as the dificulty of the instance varies, and places it against the competitors on the same instances.

At $\Delta \mathit { \Psi } = \mathit { \Delta } 0 . 4$ the fast branch handles 97.3% of the seeds with a mean of 38.6 samples, which is 19% of the budget, and at $\Delta \ = \ 0 . 8$ every seed is answered with no error in 14 samples on average, a 93% reduction in sample count. The same two points cost the fixed-budget baseline its full 200 samples, and cost LUCB 199.8 and 96.7 samples respectively, so the saving is a property of stopping rather than of the answer rule: all four rules agree on the answer at $\Delta = 0 . 8$ , and the diference is entirely in how long they wait. For $\Delta \leq 0 . 2$ the answer rate of BARS falls to 13–47%, so most hard instances are refused rather than guessed, and a fixed-majority baseline that draws the full $N = 2 0 0$ samples still leaves 21–33% error at $\Delta ~ \leq ~ 0 . 0 5$ . The eliminationstyle competitors do worse, not better, as the budget grows useless: Successive Rejects and Sequential Halving leave 35–44% error at $\Delta \leq 0 . 0 5$ , because a fixed-budget BAI rule must spend its budget eliminating arms it cannot distinguish and then recommend one of them. Exhausting the budget does not rescue these instances, which is the regime the refusal branch exists for.

The refusal decision itself is audited in Figure 3.

The composition in Figure 3 shows a single transition rather than a continuum: larger gaps are handled by the fast branch, smaller ones by refusal, and the boundary moves as the certification cost predicts. The detailed data sweep in numbers rule by rule, is shown in Table 7. On the right panel, the correct refusal rate is 0.53 at $\Delta = 0 . 2 , 0 . 7 6$ at $\Delta = 0 . 1$ and 0.87 at $\Delta = 0 . 0 2$ , while the mis-refusal rate is below 3% everywhere on the grid except at $\Delta = 0 . 3$ , where it is 18%: the rule errs towards answering, not towards refusing, which is the direction the error budget $p ^ { * }$ is stated in. The naïve branch is never reached in this sweep: every instance either triggers the fast branch or is refused at the exploration cap, so the answer rate is carried entirely by the fast branch and the refusal decision rather than by the fallback path, but the branch is retained for the logical completeness argued in Section 3.5.

![](images/0b6906e30ca16e68e408d7c255c4a54ffbcf08d6f2aa6fd83dcbd1391b5b070e.jpg)

![](images/590db2b95829baa88ebe270169b94294916f7cd157683e37b8fbc12255b8a5b1.jpg)  
Figure 2: Results across the Δ sweep under the practical mode, with the competing rules on the same instances. Left: conditional error rate, that is the error over the instances each rule answers (the BAI rules and Fixed-� answer every instance, so their conditional and total error coincide). Right: expected sample count. Every curve is averaged over 2000 seeds per cell; the fixed-budget baseline draws exactly 200 samples on every instance.

![](images/43822a891bbdedcffbdd48b5844b155c4c806c376cd539ed6252c51117b660e5.jpg)

![](images/6cdbeafad24b1f135f710b92b68b026b81188e97f66b5699f3f7acb8d7594d77.jpg)  
Figure 3: Branch composition and refusal quality as functions of Δ. Left: the refusal region marks the instances for which the criterion cannot establish feasibility within the budget; the transition to the fast branch is consistent with the $( \Delta - \Delta _ { \mathrm { h i g h } } ) ^ { - 2 }$ certification cost of Section 3.4. Right: refusal correctness against the oracle feasibility criterion of Section 5.1, that is the fraction of infeasible instances that are refused and of feasible ones that are not.

## 5.2.3. The decision map and the tuning ofthepractical width

The dificulty sweep varies one axis at a time. Crossing it with the budget gives a single decision map, and Figure 4 is that map: the answer rate and the conditional error on the $( \Delta , E ^ { * } )$ plane, with the oracle feasibility criterion of Section 5.1 overlaid as a dashed contour.

Three features of the map matter. First, the answer region is monotone in both axes but with very diferent slopes:

along $\Delta = 0 . 0 5$ the answer rate moves only from 0.05 to 0.15 when the budget grows 40-times, whereas along $\Delta = 0 . 4$ it moves from 0.24 to 0.99, because on the first row no budget in the swept range makes the instance feasible while on the second the boundary sits at $E ^ { * } \approx 5 0$ . Second, the rule’s answer region has the same shape as the oracle’s feasible region but not the same boundary: on the infeasible rows the rule still answers 5–24% of the instances, and on the feasible rows it still refuses part of them, so the two boundaries are within one grid step of each other without coinciding. Third, the errors are concentrated where the rule answers rarely: inside the feasible region the conditional error is at most 1.6% for $\Delta \ \ge \ 0 . 3$ and exactly zero for $\Delta \ \ge \ 0 . 6 $ whereas at $\Delta = 0 . 0 5$ , where the rule answers only a small fraction, the answers it does give carry errors of 19–37%. That asymmetry is the whole argument in one panel: the rule answers most where it is right and least where it is wrong.

![](images/51140162056ca77b5f1ec864b63647e80700ab7e059ff6b862d19aeac322f494.jpg)

![](images/e6e208bd6aa6c015b3624d6441d9fe07230bb2b8f1170c4c6841d83ec2de388f.jpg)  
Figure 4: Decision map of the practical mode on the (Δ, �<sup>∗</sup>) plane, 400 seeds per cell. Left: answer rate. Right: conditional error on the answered instances (a dash marks a cell in which the rule answered nothing). The dashed contour in both panels is the oracle feasibility boundary of Section 5.1: cells above and to the right of it are instances a budget-feasible answer exists for.

![](images/9f15b32761578ac2845afffeed067b1826f0437f5899806e0dd03fc6af705d86.jpg)

![](images/ed85ab794827b116f9ce5f1c2194f9f9f915b00e48448fc989b13eaf819819f6.jpg)

![](images/7a3f4389f7226cca4c70145404a0b41b453dd8c01b915dc4c04ee48eb7c4ffdd.jpg)

![](images/7d048b7a51b78df0eb8a53daed631bf1443e6ec3ed7e7f125898afb1cef1cc32.jpg)

![](images/76abff1fc4f8696d79e32626e34d7a7a5b56f5d4045501b3dc0b1444be5f0034.jpg)

![](images/66cab185c650744448653064588bccbf0135cab80ec9acef0575c09512703aa3.jpg)  
Figure 5: Parameter-tuning heatmaps for the practical width mode, three panels per row (conditional error, answer rate, expected sample count). Top: an easier instance with $\Delta = 0 . 4$ . Bottom: a harder instance with $\Delta = 0 . 0 5$ . Axes are the width multiplier � and the trigger threshold $\Delta _ { \mathrm { h i g h } } ;$ 600 seeds per cell. Cells in which fewer than five instances were answered are drawn but their conditional error is not interpretable.

Tuning the practical mode trades answer rate against mis-trigger control, and the trade-of is visible directly in the $z \times \Delta _ { \mathrm { h i g h } }$ grid of Figure 5, whose numbers are listed in Table 3: on the easier instance the conditional error stays below 0.5% everywhere on the grid, and the answer rate falls monotonically as either � or $\Delta _ { \mathrm { h i g h } }$ grows, from 98.5% at $( z , \Delta _ { \mathrm { h i g h } } ) = ( 2 , 0 . 0 5 )$ to 7.5% at (4, 0.3).

On the harder instance the same trade-of is the only thing the grid records, because the rule refuses most instances throughout: the answer rate falls from 42.7% at $z =$ 1.5 to below 1% at $z \ge 3 .$ , and the conditional error of the few answered cells is not stable enough to be read as a trend. We therefore recommend $\Delta _ { \mathrm { h i g h } } \in [ 0 . 1 , 0 . 2 ]$ with $z = 2$ for interactive deployment and $z \geq 3$ when stricter mis-trigger control is required, which is the same recommendation as in Section 3.4.

The one reading of the practical mode that moves against the rule is the mis-refusal rate at the largest budget: at $\Delta =$ 0.2 and $E ^ { * } = 5 0 0$ , by which point the instance is feasible, the rule answers 57.1% of the instances with a 1.1% conditional error and still mis-refuses 42.9% of the feasible ones. The practical width is the cause, not the criterion: the criterion is evaluated at $d _ { \mathrm { u s e } } = \operatorname* { m a x } ( 0 , \hat { \Delta } _ { \tau } - w _ { \tau } )$ with the $z = 2$ heuristic width, so at moderate budgets the endpoint is conservative enough to move a borderline instance across the boundary. We report this as a measured cost of the practical mode rather than as a defect of the criterion, and raising � is what trades it back against the answer rate, as the grid above shows.

Table 3  
Parameter-tuning grid for the practical width, 600 seeds per cell. $\ " { \mathsf { e a s y } } ^ { \prime \prime }$ is $\Delta = 0 . 4$ and “hard” is $\Delta = 0 . 0 5 ;$ the conditional error in cells with few answered instances is not interpretable and is reported only for completeness.
<table><tr><td rowspan="2">Z</td><td rowspan="2"> $\Delta _ { \mathrm { h i g h } }$ </td><td colspan="3">easy  $\left( \Delta = 0 . 4 \right)$ </td><td colspan="3">hard  $\left( \Delta = 0 . 0 5 \right)$ </td></tr><tr><td>Cond. err.</td><td>Answer</td><td>Mean n</td><td>Cond. err.</td><td>Answer</td><td>Mean n</td></tr><tr><td>1.5</td><td>0.05</td><td>0.0052</td><td>0.970</td><td>30.8</td><td>0.2930</td><td>0.427</td><td>101.8</td></tr><tr><td>1.5</td><td>0.1</td><td>0.0017</td><td>0.987</td><td>31.6</td><td>0.2976</td><td>0.342</td><td>110.8</td></tr><tr><td>1.5</td><td>0.15</td><td>0.0017</td><td>0.977</td><td>35.9</td><td>0.3025</td><td>0.198</td><td>126.6</td></tr><tr><td>1.5</td><td>0.2</td><td>0.0017</td><td>0.967</td><td>40.9</td><td>0.3061</td><td>0.163</td><td>130.5</td></tr><tr><td>1.5</td><td>0.3</td><td>0.0000</td><td>0.815</td><td>64.8</td><td>0.2157</td><td>0.085</td><td>139.0</td></tr><tr><td>2</td><td>0.05</td><td>0.0000</td><td>0.985</td><td>33.5</td><td>0.2647</td><td>0.227</td><td>123.8</td></tr><tr><td>2</td><td>0.1</td><td>0.0017</td><td>0.972</td><td>39.3</td><td>0.2273</td><td>0.147</td><td>132.3</td></tr><tr><td>2</td><td>0.15</td><td>0.0000</td><td>0.968</td><td>45.3</td><td>0.3387</td><td>0.103</td><td>137.9</td></tr><tr><td>2</td><td>0.2</td><td>0.0000</td><td>0.922</td><td>58.5</td><td>0.2895</td><td>0.063</td><td>142.1</td></tr><tr><td>2</td><td>0.3</td><td>0.0000</td><td>0.638</td><td>89.1</td><td>0.2222</td><td>0.015</td><td>148.1</td></tr><tr><td>2.5</td><td>0.05</td><td>0.0000</td><td>0.972</td><td>41.7</td><td>0.1875</td><td>0.107</td><td>137.4</td></tr><tr><td>2.5</td><td>0.1</td><td>0.0000</td><td>0.960</td><td>48.5</td><td>0.3600</td><td>0.083</td><td>140.3</td></tr><tr><td>2.5</td><td>0.15</td><td>0.0000</td><td>0.927</td><td>60.4</td><td>0.2083</td><td>0.040</td><td>145.1</td></tr><tr><td>2.5</td><td>0.2</td><td>0.0000</td><td>0.865</td><td>72.0</td><td>0.1905</td><td>0.035</td><td>145.5</td></tr><tr><td>2.5</td><td>0.3</td><td>0.0000</td><td>0.427</td><td>112.2</td><td>0.2500</td><td>0.007</td><td>149.1</td></tr><tr><td>3</td><td>0.05</td><td>0.0000</td><td>0.960</td><td>51.1</td><td>0.4000</td><td>0.050</td><td>144.4</td></tr><tr><td>3</td><td>0.1</td><td>0.0000</td><td>0.930</td><td>63.5</td><td>0.2222</td><td>0.030</td><td>146.5</td></tr><tr><td>3</td><td>0.15</td><td>0.0000</td><td>0.885</td><td>74.3</td><td>0.2105</td><td>0.032</td><td>146.0</td></tr><tr><td>3</td><td>0.2</td><td>0.0000</td><td>0.712</td><td>92.4</td><td>0.0000</td><td>0.005</td><td>149.4</td></tr><tr><td>3</td><td>0.3</td><td>0.0000</td><td>0.275</td><td>127.3</td><td>0.3333</td><td>0.005</td><td>149.3</td></tr><tr><td>4</td><td>0.05</td><td>0.0000</td><td>0.920</td><td>77.2</td><td>0.2143</td><td>0.023</td><td>147.2</td></tr><tr><td>4</td><td>0.1</td><td>0.0000</td><td>0.810</td><td>92.2</td><td>0.0000</td><td>0.005</td><td>149.4</td></tr><tr><td>4</td><td>0.15</td><td>0.0000</td><td>0.593</td><td>114.0</td><td>0.0000</td><td>0.002</td><td>149.8</td></tr><tr><td>4</td><td>0.2</td><td>0.0000</td><td>0.363</td><td>126.5</td><td>0.0000</td><td>0.003</td><td>149.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>0.3</td><td>0.0000</td><td>0.075</td><td>143.3</td><td></td><td>0.000</td><td>150.0</td></tr></table>

The chooser is the second tuning axis, and it is the one with the larger efect. The pessimistic chooser is not usable at these budgets: it produces a formally valid certificate but judges almost every instance infeasible, collapsing the answer rate, whereas the optimistic chooser reaches a 97.2% answer rate with a 0.17% conditional error and 39.3 samples on average, 2.4 times the oracle cost. This is the empirical form of the trade-of stated in Section 4: the pessimistic chooser is the configuration the theorems cover, and the optimistic chooser is the configuration that runs.

## 5.2.4. Ablation studies

In this part, we try to separate what each configuration contributes in the two experiments following: the first prices the certified mode by running the theorem configuration, and the second prices the fast branch by running it only.

The certified mode against the practical mode.

Figure 6 examines the certified mode directly, and Table 4 gives the underlying closed-form inputs.

The certified mode answers only instances whose gap is large enough to certify within the budget: at $E ^ { * } ~ = ~ 1 0 0 0$ the answer rate is zero for $\Delta \ : \leq \ : 0 . 2 , \ : 5 \%$ at $\Delta \ : = \ : 0 . 4$ and 100% at $\Delta \ : = \ : 0 . 8$ , and raising the budget to $E ^ { * } ~ = ~ 4 0 0 0$ moves the $\Delta = 0 . 4$ point only to 13%. No incorrect answer was observed in any certified cell, which is the qualitative content of Corollary 1 on instances whose certified trigger fires. The asymmetry is the point: the certified width is not a heuristic that can be tightened by spending more samples on a hard instance, because the certification cost grows as $( \Delta - \Delta _ { \mathrm { h i g h } } ) ^ { - 2 }$ and the instance is simply not certifiable at these budgets.

Solving $w _ { t } \leq \hat { \Delta } _ { t } - \Delta _ { \mathrm { h i g h } }$ with the certified width yields the smallest � at which the trigger can fire, that is $T _ { \mathrm { c e r t } } ,$ and with the two-point-family variance $\sigma ^ { 2 } = 1 - \Delta ^ { 2 }$ the efective gap is $\Delta \mathrm { ~ - ~ } \Delta _ { \mathrm { h i g h } }$ rather than Δ, which is why the scaling is not exactly $\Delta ^ { - 2 }$ . Concretely, the square-root convention gives $2 . 0 5 \times 1 0 ^ { 4 } , 1 . 6 0 \times 1 0 ^ { 3 }$ and $9 . 2 0 \times 1 0 ^ { 1 }$ samples for $\Delta = 0 . 2 , 0 . 4$ and 0.8, and retaining the additive term of Lemma 1 raises these to $2 . 1 8 \times 1 0 ^ { 4 } , 1 . 9 6 \times 1 0 ^ { 3 }$ and $1 . 9 9 \times 1 0 ^ { 2 }$ . Both sets of values quantify the same tension: a certified answer on a moderately hard instance costs two to three orders of magnitude more samples than a budgeted one, and the gap is a property of the guarantee rather than of the implementation. Finally, the search for a window in the fast branch has to be distinguished from the trigger: the trigger can fire only after $T _ { \mathrm { c e r t } }$ , and the dynamic program is what decides whether a window then exists. We report both conventions because the choice is a modeling decision about which form of the empirical-Bernstein bound is used, not a numerical detail: the ratio between the two grows monotonically with the gap, from 1.06 at $\Delta = 0 . 2$ to 2.16 at $\Delta = 0 . 8$ , while the scaling in $( \Delta - \Delta _ { \mathrm { h i g h } } ) ^ { - 2 }$ is common to both, and it is under the square-root-only convention that a constant batch of draws would certify a gap with zero width.

![](images/85a266d3943e86acf78e5e7e23946b9ceaf8c893cfa9ef88ff68318eec8eff27.jpg)

![](images/edfdfa97d38953c7159c01c2b7f7541a89362b11be3b315310442b123bae6f62.jpg)  
Figure 6: Budget suficiency in the certified mode. Left: at a given budget the answer rate increases with $\Delta ;$ no incorrect answer was observed in these runs, so no conditional-error panel is drawn. Right: the theoretical certification time $T _ { \mathrm { c e r t } } ,$ computed from $w _ { t } \le \Delta - \Delta _ { \mathrm { h i g h } }$ under the two conventions of Lemma 1.

Certification time $T _ { \mathrm { c e r t } }$ under the square-root convention used by the implementation and the figures, and under the two-term convention of Lemma 1, with $\delta = 0 . 0 5 , \Delta _ { \mathrm { h i g h } } = 0 . 1 , \bar { K } = 1 0$ and $\sigma ^ { 2 } = 1 - \Delta ^ { 2 }$ . All values are closed-form and follow from Section 3.4; no experiment is involved, so there is no sampling uncertainty to report.
<table><tr><td>Δ</td><td>square-root term</td><td>two-term (Lemma 1) ratio</td><td></td></tr><tr><td>0.2</td><td> $2 . 0 5 \times 1 0 ^ { 4 }$ </td><td> $2 . 1 8 \times 1 0 ^ { 4 }$ </td><td>1.06</td></tr><tr><td>0.3</td><td> $4 . 2 7 \times 1 0 ^ { 3 }$ </td><td> $4 . 8 1 \times 1 0 ^ { 3 }$ </td><td>1.13</td></tr><tr><td>0.4</td><td> $1 . 6 0 \times 1 0 ^ { 3 }$ </td><td> $1 . 9 6 \times 1 0 ^ { 3 }$ </td><td>1.22</td></tr><tr><td>0.6</td><td> $3 . 8 2 \times 1 0 ^ { 2 }$ </td><td> $5 . 6 8 \times 1 0 ^ { 2 }$ </td><td>1.49</td></tr><tr><td>0.8</td><td> $9 . 2 0 \times 1 0 ^ { 1 }$ </td><td> $1 . 9 9 \times 1 0 ^ { 2 }$ </td><td>2.16</td></tr></table>

## The configuration the theorems cover.

This second run closes the gap between the runs and the theorems by using the theorem configuration on the same Δ grid as the practical one, with all four configurations sharing one batch of seeds. We use four configurations. C1 is the certified width with the pessimistic chooser and the square-root-only convention; C2 is the certified width with the pessimistic chooser and the two-term width of Lemma 1, which is exactly the configuration of the two theorems; C3 is the certified width with the optimistic chooser and the two-term width; and C4 is the practical width $\mathbf { \Sigma } ( z \ = \ 2 )$ with the optimistic chooser, the configuration already used in Sections 5.2.2 and 5.2.3.

Three readings of Table 5 matter. First, the configuration the theorems cover is feasible: C2 answers every seed at $\Delta =$ 0.8 with zero conditional error, at about 997 samples. But it refuses almost everything at $\Delta \leq 0 . 4$ , where its answer rate is at most 3.3%, so the certified width alone already makes the rule unusable at the budgets of Section 5.2.2. Second, what is expensive is the chooser rather than the width: at the same certified width and two-term width, replacing the pessimistic chooser by the optimistic one (C2→C3) lowers the mean sample count at $\Delta \ : = \ : 0 . 8$ from 997.2 to 89.8, a factor of about eleven, with the conditional error still zero. Third, the additive term is not optional in practice either: the degenerate-trigger counts logged during the sweep are 45, 7 and 2 at $\Delta = 0 . 8$ , 0.4 and 0.2 for C1, and 1, 0 and 0 for C2, so the additive term essentially clears the early degenerate triggers.

## The fast branch in isolation.

Proposition 1 attributes the gain of answering with the trajectory majority, rather than with the color that completed the run, to the chain mass $W \setminus V ;$ we verify the identity in a controlled version of the fast branch that involves no language model, and — because the same experiment compares BARS against the rule it would be if the run color were the answer — it also isolates the contribution of the answering rule from the contribution of the stopping rule. Figure 7 reports the result.

The gain grows with the remaining budget �: at $\Delta = 0 . 2$ with the dynamic program’s typical run length $w = 1 0 \ \mathrm { i t }$ rises from $1 . 3 \times 1 0 ^ { - 3 }$ at $N = 1 0 0$ to $5 . 2 \times 1 0 ^ { - 3 }$ at $N = 2 0 0$ and $1 . 3 \times 1 0 ^ { - 2 } \mathrm { a t } N = 5 0 0 .$ , and it was non-negative at every point of the tested grid, which is what Proposition 1 allows for without proving it for every distribution. Conditional on a run completing, the trajectory majority halves the error of the run-color rule: at $\Delta \ = \ 0 . 2 , w \ = \ 1 0 \ \mathrm { a n d } \ N \ =$ 200 the conditional error falls from 2.5% to 1.2%, and the improvement is the mass of the paths on which a minority completes the run while the majority still leads, an event that becomes more likely as the lead is established over more samples. The same sweep provides a numerical check on Lemma 2: folding random distributions from the confidence set into a single filler color never produced a candidate that outperformed the folded pessimistic bag, with a maximum deficit below $1 0 ^ { - 5 }$ over the tested grid.

The four configurations defined in this subsection, on a shared Δ grid with 300 seeds per cell, $E ^ { * } = 1 0 0 0 , T _ { \mathrm { c a p } } = 2 5 0 , \mathrm { m i n \_ t } = 8 ,$ $p ^ { * } = 0 . 0 5 , \delta = 0 . 0 5 , \Delta _ { \mathrm { h i g h } } = 0 . 1$ and $K = 1 0 ;$ the four share one seed batch. C1: certified width, pessimistic chooser, square-root-only convention. C2: certified width, pessimistic chooser, two-term width (the configuration the theorems cover). C3: certified width, optimistic chooser, two-term width. C4: practical width $( z = 2 )$ , optimistic chooser. “Deg. trig.” counts the seeds whose trigger fired on a batch with zero plug-in variance.
<table><tr><td>Config</td><td>Δ</td><td>Answer</td><td>Refusal</td><td>Mean samples</td><td>Mean trigger</td><td>Cond. err.</td><td>Deg.  $\mathrm { t r i g . }$ </td></tr><tr><td>C1</td><td>0.05</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C1</td><td>0.1</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C1</td><td>0.2</td><td>0.013</td><td>0.987</td><td>253.5</td><td>247.4</td><td>0.0000</td><td>2</td></tr><tr><td>C1</td><td>0.4</td><td>0.743</td><td>0.257</td><td>784.5</td><td>156.6</td><td>0.0000</td><td>7</td></tr><tr><td>C1</td><td>0.8</td><td>1.000</td><td>0.000</td><td>838.1</td><td>18.9</td><td>0.0000</td><td>45</td></tr><tr><td>C2</td><td>0.05</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C2</td><td>0.1</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C2</td><td>0.2</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C2</td><td>0.4</td><td>0.033</td><td>0.967</td><td>275.0</td><td>248.3</td><td>0.0000</td><td>0</td></tr><tr><td>C2</td><td>0.8</td><td>1.000</td><td>0.000</td><td>997.2</td><td>85.0</td><td>0.0000</td><td>1</td></tr><tr><td>C3</td><td>0.05</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C3</td><td>0.1</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C3</td><td>0.2</td><td>0.000</td><td>1.000</td><td>250.0</td><td>250.0</td><td>0.0000</td><td>0</td></tr><tr><td>C3</td><td>0.4</td><td>0.033</td><td>0.967</td><td>248.6</td><td>248.3</td><td>0.0000</td><td>0</td></tr><tr><td>C3</td><td>0.8</td><td>1.000</td><td>0.000</td><td>89.8</td><td>85.0</td><td>0.0000</td><td>1</td></tr><tr><td>C4</td><td>0.05</td><td>0.183</td><td>0.817</td><td>215.2</td><td>210.5</td><td>0.3091</td><td>0</td></tr><tr><td>C4</td><td>0.1</td><td>0.263</td><td>0.737</td><td>211.6</td><td>199.3</td><td>0.0886</td><td>0</td></tr><tr><td>C4</td><td>0.2</td><td>0.733</td><td>0.267</td><td>151.7</td><td>116.5</td><td>0.0091</td><td>2</td></tr><tr><td>C4</td><td>0.4</td><td>1.000</td><td>0.000</td><td>40.4</td><td>27.5</td><td>0.0000</td><td>7</td></tr><tr><td>C4</td><td>0.8</td><td>1.000</td><td>0.000</td><td>13.8</td><td>8.9</td><td>0.0000</td><td>45</td></tr></table>

(a) benefit vs N (solid: w=10; dashed: $w { = } 8 )$  
![](images/7571fe6be93f049b01b8fe2bbe04a312e14e8f586e21e1090c01200e55590757.jpg)

(b) =0.2, w=10 (lines=exact DP, markers=MC)  
![](images/2b5ddb1bcda666628b9eb67445eba1ca39703a85f0de0e5ede4f250ea8f5dc2e.jpg)  
Figure 7: Trajectory-majority benefit in synthetic fast-branch experiments without a language model. Left: the exact gain $W - V$ against the remaining budget � for $\Delta = 0 . 1 , 0 . 2 , 0 . 4$ and run lengths $w = 1 0$ (solid) and $w = 8$ (dashed); markers are Monte Carlo estimates over 40,000 seeds. Right: conditional error rates of the run-color and trajectory-majority rules, given that a run completes $\left( \Delta = 0 . 2 , w = 1 0 \right)$ ; lines are exact chain values and markers are Monte Carlo estimates.

![](images/0bba4c7c219a2c7acf4ba15b04dd02b0f7a9da1b013f7d4e4923616f783783f9.jpg)  
Figure 8: Real-LLM results on GSM8K and MATH in the practical mode, with a pre-generated pool of $E ^ { * } = 1 6$ decodes per question and 80 questions per dataset. (a) GSM8K and (b) MATH: accuracy on the answered questions against the mean number of samples, one marker per rule, with the fixed-� majority joined by a dashed line as a reference and the rule under test drawn as a filled square.

## 5.3. Real-dataset experiments

Section 5.2 measured quantities that only a known distribution supplies: the true gap, the oracle feasibility boundary and the mis-refusal rate, none of which can be computed on real questions. This subsection measures what only a real model supplies. The pool is produced by Qwen2.5-7B on GSM8K and MATH, sixteen decodes per question and eighty questions per dataset, and every rule listed in Section 5.1 consumes the same pool. Two quantities change meaning across this boundary, and both are stated where they change: accuracy is now scored against a reference rather than against a known majority, and the mis-refusal rate is not reported at all, because its definition refers to the true gap; the refusals are instead priced ex post, by counting how many withheld questions the pool majority would have answered correctly.

## 5.3.1. Setup and parsing

In this part, each response is parsed by a priority-based extraction rule and the extracted final answer is normalized to a color key. MATH answers receive light normalization, and the few malformed multi-candidate responses are counted as errors rather than discarded, so that the error figures are not flattered by parsing. The mis-refusal rate cannot be computed on real questions, because its definition refers to the true gap (Section 5.1), so we report no value for it here and instead price the refusal ex post in the results below. This section runs under its own parameter set, namely $E ^ { * } = 1 6 , p ^ { * } = 0 . 0 2 , \Delta _ { \mathrm { h i g h } } = 0 . 3 , \delta = 0 . 0 5$ and the practical width $( z = 2 )$ , so the certified-mode level � is not a gate on the running system here, and the synthetic defaults of Section 5.2 do not transfer to these runs.

## 5.3.2. Results

Figure 8 and Table 6 report the results.

On GSM8K the rule answers 81.2% of the questions after 6.21 samples on average, with an accuracy of 93.8% on those answers, against 88.8% for the fixed-16 majority from the same pool. Because the two rules do not spend the same amount, the matched-cost comparison is the informative one: at 6.21 samples the closest fixed baseline is $N \ = \ 8 .$ which reaches 88.8%, so at the same average cost the rule buys 5.1 points of accuracy on the answered questions, and the currency it pays in is coverage. Over the dataset as a whole — the denominator that includes the refused questions — the rule draws 22.3% fewer samples than fixed-8. On MATH the same comparison is sharper: the rule answers 50.0% of the questions at 7.76 samples with 82.5%, where the matched-cost baseline � = 8 reaches 70.0% and the full-budget � = 16 baseline reaches 73.8% over the whole set. The full-coverage baselines report those numbers only because they answer everything, so their accuracy is a ceiling rather than a choice. The refusal is not free, and the pool lets us price it ex post rather than only in principle: of the 15 questions the rule refuses on GSM8K, 10 would have been answered correctly by the pool majority, and of the 40 it refuses on MATH, 26 would have been. So the rule gives up 18.8% of the GSM8K set and 50.0% of the MATH set, and we report that as the measured price of the output channel rather than as a success of it; the Wilson intervals of the rule and of the full-coverage baselines overlap on both datasets, so the accuracy gain is not statistically established at these sample sizes. Note that this ex post count is not the mis-refusal rate defined in Section 5.1: the latter is stated against the analyst-level feasibility event , which is built from the true gap and therefore cannot be evaluated on real questions, whereas the count here only says that the answers the rule withheld would in hindsight have been right. Scoring each refusal as an error rather than as a withheld answer places the rule at 76.2% on GSM8K and 41.2% on MATH over the whole set, below every full-coverage baseline; we report the figure because it is the same trade-of seen from the other side, and a deployment that must answer every question should not use this rule. Finally, the competitor rows carry the same reading as the synthetic ones: the earlystopping rules match the full-budget accuracy at a fraction of the cost but cannot exceed it, because exceeding it would require declining some questions; on MATH, where the pool is genuinely fragmented, that ceiling is what separates them from the rule under test. The two datasets therefore difer in how much the refusal costs rather than in whether the rule behaves the same way, and we read that as the crossdomain content of the finding rather than as a per-dataset tuning artifact.

Real-LLM results, Qwen2.5-7B, 80 questions per dataset, pre-generated pool $E ^ { * } = 1 6 ,$ all rules on the same pool. “ans.” is the answer rate, “acc.” the accuracy on the answered questions, and � the mean sample count over all questions of the dataset, refused questions included. Accuracy intervals are Wilson 95% intervals and are reported in the text.
<table><tr><td>Rule</td><td colspan="3">GSM8K</td><td colspan="3">MATH</td></tr><tr><td></td><td>ans. (%)</td><td>acc. (%)</td><td>n</td><td>ans. (%)</td><td>acc. (%)</td><td>n</td></tr><tr><td>Fixed-1</td><td>100.0</td><td>81.2</td><td>1.00</td><td>100.0</td><td>48.8</td><td>1.00</td></tr><tr><td>Fixed-2</td><td>100.0</td><td>81.2</td><td>2.00</td><td>100.0</td><td>48.8</td><td>2.00</td></tr><tr><td>Fixed-4</td><td>100.0</td><td>86.2</td><td>4.00</td><td>100.0</td><td>62.5</td><td>4.00</td></tr><tr><td>Fixed-8</td><td>100.0</td><td>88.8</td><td>8.00</td><td>100.0</td><td>70.0</td><td>8.00</td></tr><tr><td>Fixed-16</td><td>100.0</td><td>88.8</td><td>16.00</td><td>100.0</td><td>73.8</td><td>16.00</td></tr><tr><td> $\mathsf { E S C } ( w { = } 2 )$ </td><td>100.0</td><td>88.8</td><td>2.77</td><td>100.0</td><td>70.0</td><td>3.67</td></tr><tr><td> $\mathsf { E S C } ( w { = } 3 )$ </td><td>100.0</td><td>88.8</td><td>4.26</td><td>100.0</td><td>73.8</td><td>6.04</td></tr><tr><td> $\mathsf { E S C } ( w { = } 4 )$ </td><td>100.0</td><td>88.8</td><td>6.34</td><td>100.0</td><td>73.8</td><td>8.66</td></tr><tr><td> ${ \mathsf { C G E S } } ( \gamma = 0 . 9 )$ </td><td>100.0</td><td>88.8</td><td>11.00</td><td>100.0</td><td>73.8</td><td>15.00</td></tr><tr><td> $\mathsf { C G E S } ( \gamma = 0 . 9 5 )$ </td><td>100.0</td><td>88.8</td><td>16.00</td><td>100.0</td><td>73.8</td><td>16.00</td></tr><tr><td>DSC</td><td>100.0</td><td>88.8</td><td>2.83</td><td>100.0</td><td>72.5</td><td>4.01</td></tr><tr><td>DASE</td><td>100.0</td><td>88.8</td><td>2.91</td><td>100.0</td><td>72.5</td><td>4.14</td></tr><tr><td>MMC</td><td>100.0</td><td>88.8</td><td>15.81</td><td>100.0</td><td>73.8</td><td>15.82</td></tr><tr><td>MARS</td><td>100.0</td><td>88.8</td><td>4.26</td><td>100.0</td><td>73.8</td><td>7.66</td></tr><tr><td>BARS-NoRefusal</td><td>100.0</td><td>88.8</td><td>7.71</td><td>100.0</td><td>73.8</td><td>11.76</td></tr><tr><td>BARS</td><td>81.2</td><td>93.8</td><td>6.21</td><td>50.0</td><td>82.5</td><td>7.76</td></tr></table>

Reading the competitor rows rule by rule, each of them lands at or below the full-budget accuracy, because none of them can decline a question: The distinction is not that BARS is cheaper on every cell but that its cheapness and its accuracy are produced by one decision: on MATH the competitors spend their budget on questions whose answer distribution they cannot resolve, whereas BARS declines them and spends its budget on the rest.

## 5.4. Discussion

The synthetic and the deployed runs exhibit the same qualitative behavior: easy instances are answered quickly and hard ones are refused. The competitor comparison adds three findings that the internal audit alone could not produce. First, where no answer is budget-feasible the advantage of BARS is not a matter of degree but of kind: on the smallest gap of the sweep, which the oracle calls infeasible, the rule refuses 87% of the instances, whereas the five fixed-budget rules assert an answer that is wrong in 33–44% of them, and no parameter of theirs could have produced the honest output because none of them has a third output to produce it with. Second, on instances that are answerable, the saving comes from stopping rather than from answering: BARS matches the conditional error of the fixed-budget baseline and of LUCB on the easy cells at 14–39 samples against their 97–200, and the diference is entirely in how long the rules wait rather than in what they answer. Third, the eliminationbased BAI rules are the wrong tool for this problem rather than a weak version of the right one: on the hard cells of the sweep their error (35–44% at $\Delta \le 0 . 0 5 )$ exceeds that of the fixed-budget majority (21–33%), because a fixed-budget arm-identification rule must spend its budget eliminating arms it cannot separate and then recommend one of them.

The fixed-budget baseline cannot recover hard instances even when it spends the whole budget, whereas the adaptive rule gives up on a subset of questions in exchange for higher accuracy on the rest.

## 6. Conclusion and future work

This paper formalizes multiple-sampling majority voting as majority identification under an unknown distribution with a sampling budget and a target error rate, and proposes BARS (Budget-Aware Refusal Sampling), a sequential rule driven by the top-two gap Δ and its plug-in variance. The rule returns one of three outputs — an early stop, a majority vote after the budget is spent, or an explicit refusal — and it can be run in two configurations: the certified mode, for which we prove bounds on the joint answering error and on the mis-refusal probability, and the practical mode, which is empirically calibrated and for which no theorem is claimed. Each certified-mode guarantee is backed by a proof; the practical mode is backed by measurement, and the distance between the two is reported by this paper.

The experiments divide into two settings and one discussion, and the findings follow that division. First, the synthetic sweeps over random seeds show the dificultydriven behavior that the certified-mode analysis predicts: large gaps are settled far below the budget, small gaps are refused, and the boundary moves with the $( \Delta - \Delta _ { \mathrm { h i g h } } ) ^ { - 2 }$ certification cost. Second, certification itself is expensive at these problem sizes: a certified answer costs two to three orders of magnitude more samples than a budgeted one, which is why the certified mode refuses at small budgets instead of answering badly, and why the two-point lower bound is a statement about budgets rather than about algorithms. Third, on a real deployment budget the rule behaves as the abstraction predicts: on GSM8K it draws 22.3% fewer samples over the pool than fixed-8 sampling (denominator: all 80 questions, refused questions included) at the cost of declining 18.8% of the questions, and on MATH it refuses half of the questions while lifting the answered accuracy from 70.0% to 82.5%, so that refusal converts questions the configuration cannot handle from silent errors into an actionable output.

The two theorems’ own configuration is also measured now: at $\Delta \ : = \ : 0 . 8$ the certified width with the pessimistic chooser is feasible but spends essentially the whole sampling budget, and changing only the chooser to the optimistic one reduces the mean sample count to 89.8 (Section 5.2.4).

Five directions follow from the limitations we report, ordered by how much of the remaining uncertainty each would remove. First, the certified mode needs tighter timeuniform widths, for instance self-normalised or bettingbased confidence sequences [4, 5], since its current constant is what rules it out at deployment-scale budgets. Second, an independent reference direction, generated separately from the rollouts the rule consumes, would remove the reference dependence of every accuracy figure reported here. Third, a low-repetition or multi-peaked answer pool would test the abstraction where it is weakest, because the current pool’s concentration is partly repetition rather than semantic certainty. Fourth, decoupling Δ from answer quality, for instance by weighting answers the model judges plausible, would address the structural boundary that no bound in this paper can detect: a model that is confidently and consistently wrong. Fifth, online batching would use the stopping-time distribution as a scheduling signal when many questions share a GPU, and larger models and budgets would test whether the deployed behavior we report survives at scale.

## CRediT authorship contribution statement

Shenxi Liu: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing – original draft. Yiwei Li: Methodology, Validation, Writing – review & editing. Mingyang Zhao: Software, Data curation, Visualization, Writing – review & editing. Yuhang Tian: Investigation, Writing – review & editing. Kan Li: Supervision, Funding acquisition, Project administration, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This research was supported by the Beijing Natural Science Foundation (No. 4262065), and the National Natural Science Fundation of China (No. 62676029, 62171183).

## Data availability

All experimental code related to this study can be accessed at https://github.com/LtColLiuPeiqiang/adaptive-s topping/.

## References

[1] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, D. Zhou, Self-consistency improves chain of thought reasoning in language models, in: International Conference on Learning Representations (ICLR), 2023. arXiv:2203.11171.

[2] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, J. Schulman, Training verifiers to solve math word problems, in: International Conference on Learning Representations (ICLR), 2022. arXiv:2110 .14168.

[3] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, J. Steinhardt, Measuring mathematical problem solving with the MATH dataset, in: Advances in Neural Information Processing Systems (NeurIPS), 2021. arXiv:2103.03874.

[4] S. R. Howard, A. Ramdas, J. McAulife, J. Sekhon, Time-uniform, nonparametric, nonasymptotic confidence sequences, The Annals of Statistics 49 (2) (2021) 1055–1080. arXiv:1810.08240.

[5] I. Waudby-Smith, A. Ramdas, Estimating means of bounded random variables by betting, Journal of the Royal Statistical Society: Series B 85 (1) (2023) 1–27. arXiv:2010.09686.

[6] D. A. Darling, H. Robbins, Iterated logarithm inequalities, The Annals of Mathematical Statistics 38 (1) (1967) 139–148.

[7] W. Hoefding, Probability inequalities for sums of bounded random variables, Journal of the American Statistical Association 58 (301) (1963) 13–30.

[8] A. Maurer, M. Pontil, Empirical bernstein bounds and sample variance penalization, in: Proceedings of the 22nd Conference on Learning Theory (COLT), 2009. arXiv:0907.3740.

[9] W. Feller, An Introduction to Probability Theory and Its Applications, 3rd Edition, Vol. I, Wiley, 1968.

[10] Y. Li, P. Yuan, S. Feng, B. Pan, X. Wang, B. Sun, H. Wang, K. Li, Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning, in: International Conference on Learning Representations (ICLR), 2024. arXiv:2401.10480.

[11] X. Wang, S. Feng, Y. Li, P. Yuan, Y. Zhang, C. Tan, B. Pan, Y. Hu, K. Li, Make every penny count: Dificulty-adaptive self-consistency for cost-eficient reasoning, in: Findings of the Association for Computational Linguistics: NAACL, 2025. arXiv:2408.13457.

[12] E. Aghazadeh, A. Ghasemi, H. Beyhaghi, H. Pishro-Nik, CGES: Confidence-guided early stopping for eficient and accurate selfconsistency, arXiv preprintExtended version; preliminary version at the Eficient Reasoning Workshop, NeurIPS 2025 (2025). arXiv: 2511.02603.

[13] R. E. Medina, Adaptive consensus in LLM ensembles via sequential evidence accumulation (DASE), arXiv preprint (2026). arXiv:2605.0 4236.

[14] B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. R ’e, A. Mirhoseini, Large language monkeys: Scaling inference compute with repeated sampling, arXiv preprint (2024). arXiv:2407.21787.

[15] C. Snell, J. Lee, K. Xu, A. Kumar, Scaling LLM test-time compute optimally can be more efective than scaling model parameters, arXiv preprint (2024). arXiv:2408.03314.

[16] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, W. Zaremba, Evaluating large language models trained on code, arXiv preprint (2021). arXiv:2107.03374.

[17] P. Singh, P. Singh, What fixed-rollout pass@k evaluations can identify, arXiv preprint (2026). arXiv:2609.09245.

[18] W. Chen, P. Li, M. Liu, W. Su, T. Xie, MARS: Margin-adversarial risk-controlled stopping for parallel LLM test-time scaling, arXiv preprint (2026). arXiv:2606.12935.

[19] Y. Geifman, R. El-Yaniv, Selectivenet: A deep neural network with an integrated reject option, in: International Conference on Machine Learning (ICML), 2019, pp. 2151–2159. arXiv:1901.09192.

[20] S. Rabanser, A. Thudi, K. Hamidieh, A. Dziedzic, I. Bahceci, A. B. Sediq, H. Sokun, N. Papernot, Selective prediction via training dynamics, Transactions on Machine Learning Research (TMLR) (2022). arXiv:2205.13532.

[21] Y. A. Yadkori, I. Kuzborskij, D. Stutz, András György, A. Fisch, A. Doucet, I. Beloshapka, W.-H. Weng, Y.-Y. Yang, Csaba Szepesvári, A. T. Cemgil, N. Tomasev, Mitigating LLM hallucinations via conformal abstention, arXiv preprint (2024). arXiv:2405.01563.

[22] Z. Wang, Q. Wang, Y. Zhang, T. Chen, X. Zhu, X. Shi, K. Xu, SConU: Selective conformal uncertainty in large language models, in: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025. arXiv:2504.14154.

[23] V. Kotte, When can conformal risk control certify LLM outputs? bounds, impossibility, and practical achievability, arXiv preprint (2026). arXiv:2606.29054.

[24] M. Khosravani, Selective prediction from agreement: A Lipschitzconsistent version space approach, arXiv preprint (2026). arXiv: 2605.02611.

[25] P. Cordero-Encinar, A. B. Duncan, Certified self-consistency: Statistical guarantees and test-time training for reliable reasoning in LLMs, arXiv preprint (2025). arXiv:2510.17472.

[26] S. Pasteris, A. Rumi, M. Thiessen, S. Saito, A. Miyauchi, F. Vitale, M. Herbster, Bandits with abstention under expert advice, arXiv preprint (2024). arXiv:2402.14585.

[27] J. Yang, T. Jin, V. Y. F. Tan, Asymptotically and minimax optimal regret bounds for multi-armed bandits with abstention, arXiv preprint (2024). arXiv:2402.15127.

[28] S. Liaw, B. Plaut, Learning when not to learn: Risk-sensitive abstention in bandits with unbounded rewards, in: Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS), 2026. arXiv:2510.14884.

[29] Y. Huang, Y. Hou, V. Y. F. Tan, Bayesian best-arm identification with abstention: A polynomial-to-exponential phase transition, arXiv preprint (2026). arXiv:2606.29203.

[30] E. Kaufmann, O. Cappé, A. Garivier, On the complexity of bestarm identification in multi-armed bandit models, Journal of Machine Learning Research 17 (1) (2016) 1–42. arXiv:1407.4443.

[31] A. Garivier, E. Kaufmann, Optimal best arm identification with fixed confidence, in: Proceedings of the 29th Conference on Learning Theory (COLT), 2016, pp. 998–1027. arXiv:1602.04589.

[32] S. Kalyanakrishnan, A. Tewari, P. Auer, P. Stone, Pac subset selection in stochastic multi-armed bandits, in: Proceedings of the 29th International Conference on Machine Learning (ICML), 2012.

[33] K. Jamieson, M. Malloy, R. Nowak, S. Bubeck, lil’ UCB: An optimal exploration algorithm for multi-armed bandits, in: Proceedings of the 27th Conference on Learning Theory (COLT), PMLR 35, 2014, pp. 423–439. arXiv:1312.7308.

[34] S. Bubeck, N. Cesa-Bianchi, Regret analysis of stochastic and nonstochastic multi-armed bandit problems, Foundations and Trends in Machine Learning 5 (1) (2012) 1–122. arXiv:1204.5721.

[35] A. B. Tsybakov, Introduction to Nonparametric Estimation, Springer, 2009.

[36] O. Maron, A. W. Moore, Hoefding races: Accelerating model selection search for classification and function approximation, Advances in Neural Information Processing Systems (NIPS) 6 (1993).

## A. Proofs

The proofs follow the order in which the results are used, so that no proof relies on a statement proved after it. Throughout,  denotes the good event of Lemma 1 and Corollary 1, and constants absorbed into the width are flagged where they are used.

## A.1. Proof of the confidence sequence with the additive term (Lemma 1)

At each time �, the empirical-Bernstein inequality of [8] (Theorem 4, for variables bounded in [−1, 1]) gives, for any $c _ { t } > 0$

$$
\mathbb { P } \bigg ( | \bar { X } _ { t } - \mu | ~ \geq ~ \sqrt { \frac { 2 \hat { \sigma } _ { t } ^ { 2 } c _ { t } } { t } } + \frac { 7 c _ { t } } { 3 t } \bigg ) ~ \leq ~ 2 e ^ { - c _ { t } } .
$$

Substituting $\begin{array} { r } { c _ { t } = 2 \ln t + \ln { \frac { \pi ^ { 2 } } { 3 \delta } } } \end{array}$ turns the right-hand side into

$$
2 e ^ { - 2 \ln t } \cdot e ^ { - \ln \frac { \pi ^ { 2 } } { 3 \delta } } = \frac { 6 \delta } { \pi ^ { 2 } } \cdot \frac { 1 } { t ^ { 2 } } ,
$$

and a union bound over $t \ge 2$ , using $\begin{array} { r } { \sum _ { t \geq 1 } t ^ { - 2 } = \pi ^ { 2 } / 6 . } \end{array}$ yields $ \mathbb { P } ( \exists t \ \geq \ 2 \ : \ | { \bar { X } } _ { t } - \mu | \ \geq \ w _ { t } ) \ \leq \ \delta$ . This is the mechanism of time-uniformity: the per-test mass $\frac { 6 \delta } { \pi ^ { 2 } t ^ { 2 } }$ is a Cauchy sequence in �, so the total over all times is exactly the nominal level. Both terms of $w _ { t }$ are needed: the square-root term shrinks with the plug-in variance and would vanish on a constant batch of draws, whereas the additive term $7 c _ { t } / ( 3 t )$ is a strictly decreasing positive function of � and keeps the bound informative at every time (Remark 1).

## A.2. Proof of the top-two gap control and of the certified trigger (Corollary 1)

The order statistics $p _ { ( k ) }$ are 1-Lipschitz in the $\ell _ { \infty }$ sense: for any pair of distributions $p , \hat { p }$ and any rank $k ,$

$$
\left| \hat { p } _ { ( k ) } - p _ { ( k ) } \right| \le \operatorname* { m a x } _ { c } | \hat { p } _ { c } - p _ { c } | .
$$

Applying this at ranks $k \ = \ 1$ and $k \ = \ 2$ and using the triangle inequality gives $| \hat { \Delta } _ { t } - \Delta | \leq 2 \operatorname* { m a x } _ { c } | \hat { p } _ { c } - p _ { c } |$ . To transfer Lemma 1 to the colors, a union bound over the at most $\bar { K }$ of them allocates the per-test mass $6 \delta / ( \bar { K } \pi ^ { 2 } t ^ { 2 } )$ to each color, so the joint event $\{ \exists t \geq 2 , c : | \hat { p } _ { t , c } - p _ { c } | \geq w _ { t } ^ { 0 } \}$ has probability at most $\delta ,$ where $w _ { t } ^ { 0 }$ is the per-color halfwidth of Lemma 1 with $\bar { K }$ inserted into $c _ { t }$ . Multiplying $w _ { t } ^ { 0 }$ by the Lipschitz factor 2 of the top-two gap gives the width of Corollary 1, and on the resulting event $\mathcal { G }$ every trigger $\hat { \Delta } _ { t } - w _ { t } \ge \Delta _ { \mathrm { h i g h } }$ certifies $\Delta \geq \Delta _ { \mathrm { h i g h } }$

## A.3. Proof of the pessimistic dominance (Lemma 2)

We split the argument into a monotonicity step over the two-color family and a folding step.

Step 1 (two-color monotonicity). For a two-color bag with majority probability $m ,$ the fast branch is a Markov chain on the state $( r _ { 1 } , r _ { 2 } , d )$ , where $r _ { 1 } , r _ { 2 } \in \{ 0 , \ldots , w - 1 \}$ are the current run lengths of the majority color and the filler and $d$ is the count gap $N _ { 1 } - N _ { 2 } ;$ the chain is killed when either run length reaches � or $N$ steps elapse, and the payof is $\mathrm { \ddot { ~ } e r r o r } = d \mathrm { ~ \leq ~ } 0$ at the kill”. Order the states by $( d \uparrow , r _ { 1 } \uparrow , r _ { 2 } \downarrow )$ . Each transition is monotone in this order: a majority draw maps $( d , r _ { 1 } , r _ { 2 } )$ to $( d + 1 , r _ { 1 } + 1 , 0 )$ or kills with payof $d \mathrm { ~ + ~ } 1 \mathrm { ~ \le ~ } 0 .$ , whereas a filler draw maps to $( d - 1 , 0 , r _ { 2 } + 1 )$ or kills with payof $d - 1 \leq 0$ Hence, by the standard comparison theorem for monotone Markov chains (equivalently, by backward induction on the remaining horizon: the value function is non-decreasing in the order, and increasing � shifts probability mass from filler draws to majority draws, which only raises the value), the error probability is non-increasing in �. The hardest twocolor bag with $p _ { \left( 1 \right) } \geq m$ is therefore the one whose majority probability equals �, namely $q ^ { \mathrm { p e s s } }$

Step 2 (folding). Fix a majority probability $m ^ { \prime } \geq m$ and couple any distribution � with $p _ { \mathrm { ( 1 ) } } = m ^ { \prime }$ to the folded two-color bag $q _ { m ^ { \prime } }$ on a common uniform sequence: every minority outcome of $p$ maps to the single filler color of $q _ { m ^ { \prime } }$ At every time � the majority count and the total minority count coincide in both processes, so the gap process $d _ { t } \ =$ $N _ { ( 1 ) } ( t ) -$ (total minority count)(�) is identical; the majority run lengths coincide, and the folded filler run length is at least every individual minority run length. Consequently the unfolded error event at a fixed time, $\{ \exists c ^ { \prime } \geq 2 : N _ { c ^ { \prime } } ( t ) \geq$ $N _ { ( 1 ) } ( t ) \}$ , is contained in the folded error event $\{ d _ { t } ~ \leq ~ 0 \}$ The only remaining diference is the killing time: the folded process is killed by a filler �-run no later than the unfolded one. On the paths where the folded process is killed by a majority run the two processes agree exactly; on the paths where it is killed by a filler run while the majority still leads $( d ~ > ~ 0 )$ , the unfolded process continues, and the folded error event no longer witnesses the unfolded one. On those paths the unfolded process can err only if some minority overtakes the majority against a negative drift $- \Delta _ { c } \leq - \Delta$ The overshoot time $\tau = \operatorname* { m i n } \{ t \geq \tau _ { \mathrm { f o l d } } : \exists c ^ { \prime } \geq 2 , N _ { c ^ { \prime } } ( t ) \geq$ $N _ { ( 1 ) } ( t ) \}$ is an upcrossing of the negative-drift random walk $\begin{array} { r } { S _ { t } ^ { ( c ^ { \prime } ) } = \sum _ { i = \tau _ { \mathrm { f o l d } } + 1 } ^ { t } ( { \bf 1 } \{ X _ { i } = c ^ { \prime } \} - { \bf 1 } \{ X _ { i } = 1 \} ) } \end{array}$ ; by Hoefding’s inequality on its increments (bounded in [−1, 1], drift $- \Delta _ { c } )$ combined with the reflection principle, $\mathbb { P } ( \tau \leq N \mid \mathcal { G } ) \leq$ $2 e ^ { - N \Delta ^ { 2 } / 2 }$ for each minority color, and a union over the at most $\bar { K } - 1$ minority colors gives $2 ( \bar { K } \mathrm { ~ - ~ } 1 ) e ^ { - N \Delta ^ { 2 } / 2 } \leq$ $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 8 } = \rho _ { N }$ , where the last step uses $2 e ^ { - N \Delta ^ { 2 } / 2 } \leq$ $e ^ { - N \Delta ^ { 2 } / 8 }$ and $d _ { \mathrm { u s e } } \leq \Delta$ on $\mathcal { G } .$ Combining with Step 1 gives the statement of the lemma.

## A.4. Proof of the fast-branch error (Theorem 1)

The error event at the branch stop splits into three disjoint events (Section 3.5): (a) the first �-run is completed by a minority color and the minority is the trajectory leader at the stop; (b) the first �-run is completed by the majority color yet the majority is not the trajectory leader at the stop; and (c) no color completes a �-run within the remaining � samples, in which case the branch falls back to the majority of the whole trajectory. The absorbing chain of Appendix B computes $( a )$ and (b) exactly, as $U \mathrm { ~ - ~ } W$ and $V ,$ , for any candidate distribution $q ;$ evaluating it at $q ^ { \mathrm { p e s s } }$ and invoking Lemma 2 transfers the bound to every $p$ in the confidence set on $\mathcal { G } .$ For (c), the trajectory majority at budget exhaustion is the majority over $E ^ { * }$ iid samples; the Hoefding bound of Section 4.3 with $\bar { K }$ in place of � (Assumption 1) and $d _ { \mathrm { u s e } } \leq$ $\Delta$ on $\mathcal { G }$ bounds its probability by $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } = F ;$ which the chooser includes as its third term. Summing the three parts bounds ℙ(wrong ∣ enter the fast branch, ) by $p _ { \mathrm { e r r } } [ w ; q ^ { \mathrm { p e s s } } ] + \rho _ { N }$ , which the criterion of Theorem 1 charges to the $p ^ { * }$ budget.

## A.5. Proof of the majority Hoefding bound (Section 4.3)

Fix $c \geq 2$ and define $Y _ { i } = \mathbf { 1 } \{ X _ { i } = c \} - \mathbf { 1 } \{ X _ { i } = 1$ }; then $\mathbb { E } Y _ { i } ~ = ~ - \Delta _ { c } ~ \leq ~ - \Delta$ and $| Y _ { i } | ~ \le ~ 1$ . The majority is wrong if and only if there exists $c \ge 2$ with $N _ { c } \ \ge \ N _ { 1 }$ that is $\textstyle \sum _ { i } Y _ { i } \ \geq \ 0$ , and Hoefding’s inequality on $[ - 1 , 1 ]$ gives $\mathbb { P } ( \sum _ { i } Y _ { i } \ge 0 ) \le e ^ { - N \Delta _ { c } ^ { 2 } / 2 } \le e ^ { - N \Delta ^ { 2 } / 2 }$ . A union over $c = 2 , \ldots , K$ gives at most $( K - 1 ) e ^ { - N \Delta ^ { 2 } / 2 }$ , and replacing � by its upper bound �<sup>̄</sup> gives the bound used in the text.

## A.6. Proof of the answering-error and mis-refusal bounds (Theorem 2)

Define the good event  with $\mathbb { P } ( \mathcal { G } ) \geq 1 - \delta$ (Lemma 1 and Corollary 1); on $\mathcal { G } , | \hat { \Delta } _ { t } - \Delta | \leq w _ { t }$ for all $t \geq 2$ , hence $d _ { \mathrm { u s e } } = \operatorname* { m a x } ( 0 , \hat { \Delta } _ { \tau } - w _ { \tau } ) \leq \Delta$

Answering error (joint). The rule answers through the fast branch or through the naive branch. On  both paths are safe: Theorem 1 bounds the fast-path error by $p ^ { * }$ , and in the naive branch the criterion $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } \leq p ^ { * }$ together with $d _ { \mathrm { u s e } } \leq \Delta$ and $K \leq { \bar { K } }$ implies $( K - 1 ) e ^ { - N \Delta ^ { 2 } / 2 } \le p ^ { * }$ so the true error is at most $p ^ { * }$ as well. Hence ℙ(wrong ∧ answers $\wedge { \mathcal { G } } ) \leq p ^ { * } \mathbb { P } ( { \mathrm { a n s w e r s } } \wedge { \mathcal { G } } )$ , and of $\mathcal { G }$ only the mass � remains, so

$$
\mathbb { P } ( \mathrm { w r o n g } \wedge \mathrm { a n s w e r s } ) \ \leq \ p ^ { * } \mathbb { P } ( \mathrm { a n s w e r s } \wedge \mathcal { G } ) + \delta \ \leq \ p ^ { * } + \delta .
$$

Mis-refusal. A mis-refusal is the joint event refuse $\wedge A .$ with $\mathcal { A } ~ = ~ \{ ( K - 1 ) e ^ { - N \Delta ^ { 2 } / 2 } ~ \leq ~ p ^ { * } \}$ the naive-feasibility event of Section 3.1. Any mis-refusal either lies outside $\mathcal { G } -$ probability at most $\delta - \mathrm { o r }$ lies inside , where by definition it is counted by the plug-in bias $B _ { \mathrm { p l u g } } \mathrm { : }$ on ${ \mathcal { G } } ,$ a refusal means $( \bar { K } - 1 ) e ^ { - N d _ { \mathrm { u s e } } ^ { 2 } / 2 } > p ^ { * }$ , which is compatible with  only if the conservative endpoint $d _ { \mathrm { u s e } }$ underestimates $\Delta$ enough to flip the criterion, or if the replacement of � by $\bar { K }$ does so. Hence $\mathbb { P } ( \mathrm { r e f u s e } \land \mathcal { A } ) \le \delta + B _ { \mathrm { p l u g } }$ . The suficient condition for $B _ { \mathrm { p l u g } } = 0$ is therefore $d _ { \mathrm { u s e } } = \Delta$ and $K = { \bar { K } } \cdot$ ; the color-count bound contributes a mis-refusal channel of its own when $K < \bar { K }$ , which no distribution-free argument can remove and which we therefore report rather than bound.

## A.7. Proof of the expected-samples decomposition (Theorem 3)

The three terms correspond to the three branches. The fast branch spends $\tau _ { \mathrm { F } }$ in exploration and then an expected min $T _ { \mathrm { r u n } } , N )$ in the run phase, truncated by the remaining budget; the naive branch always spends $E ^ { * }$ in total; the refusal branch spends $\tau _ { \mathrm { R } }$ . Weighting by the branch probabilities and applying linearity of expectation gives the decomposition. Because the majority’s run typically completes first in the multi-color race, the true expectation is slightly below $\mathbb { E } [ T _ { \mathrm { r u n } } ( \hat { p } _ { ( 1 ) } , w ) ]$ , so the stated fast-branch term is an upper bound.

## A.8. Proof of the two-point lower bound (Theorem 4)

We use the Neyman–Pearson log-likelihood-ratio test. Let $\begin{array} { r } { \Lambda _ { n } = \sum _ { i = 1 } ^ { n } \ln \frac { d P ^ { + } } { d P ^ { - } } ( X _ { i } ) } \end{array}$ , a random walk with mean � KL under $P ^ { + }$ and −� KL under $P ^ { - } ;$ decide $P ^ { + }$ if and only if $\Lambda _ { n } ~ \geq ~ 0$ . By Stein’s lemma the sum of the two error probabilities of the optimal test decays as $e ^ { - n \operatorname { K L } ( 1 + o ( 1 ) ) }$ , so requiring per-side error at most � gives � $\begin{array} { r } { \mathrm { K L } \ge \ln \frac { 1 } { 2 \delta } ( 1 + } \end{array}$ $o ( 1 ) )$ . Substituting the two-point family’s divergence $\mathbf { \tilde { K } L = }$ Δ ln $\begin{array} { r } { \frac { 1 + \Delta } { 1 - \Delta } = 2 \Delta ^ { 2 } + O ( \Delta ^ { 4 } ) } \end{array}$ yields $\begin{array} { r } { n \geq c \Delta ^ { - 2 } \ln \frac { 1 } { \delta } } \end{array}$ for an absolute constant �. The sequential version follows by applying the same argument at every time step, a standard technique in bandit lower bounds; the expected stopping time has the same order.

## B. The Choose-� dynamic program, and the oracle comparison

## B.1. The chain behind $p _ { \mathrm { e r r } }$

The run part is computed by an absorbing Markov chain over states $( c , r )$ : the last � draws are all of color � and no �-run has been completed. Let $g ( c , r )$ be the probability that, starting from $( c , r )$ , the first completed �-run is completed by a non-majority color, with absorption $g ( c , w ) =$ �{� is not the majority}. Then for $r = 1 , \ldots , w - 1$

$$
\begin{array} { c } { { g ( c , r ) = q _ { c } g ( c , r + 1 ) + \displaystyle \sum _ { c ^ { \prime } \neq c } q _ { c ^ { \prime } } g ( c ^ { \prime } , 1 ) , } } \\ { { g ( \emptyset ) = \displaystyle \sum _ { c } q _ { c } g ( c , 1 ) . } } \end{array}
$$

This system has $O ( \hat { K } _ { \tau } w )$ states and a rank-one coupling, and it can be solved in $O ( w \hat { K } _ { \tau } )$ time by exploiting its neartriangular structure. The count-gap terms � and � require, in addition, the count gap $d = \hat { N } _ { ( 1 ) } - \operatorname* { m a x } _ { c \neq ( 1 ) } \hat { N } _ { c }$ between the majority color and the strongest minority color; the augmented chain over the run configuration and $d ,$ with $| d | \ \leq \ N$ , has $O ( \hat { K } _ { \tau } w N )$ states in the worst case, which for the folded two-color distributions used by the pessimistic mode reduces to $O ( w N )$ . The fallback term is bounded by the majority Hoefding bound of Section 4.3, which the chooser adds as its third term. One modelling remark is in order, because it delimits what the dynamic program proves: the count gap � is treated as a suficient state together with the run configuration, which is exact for the two-color folds the pessimistic mode evaluates and is an approximation for a general multi-color empirical distribution. All of this runs on the CPU and is negligible relative to the sampling cost (Section 3.6).

## B.2. Comparison with an omniscient oracle

We record the oracle comparison here because it is a supporting computation rather than part of the guarantee chain. Let $C _ { \mathrm { a d a p t } }$ and $C _ { \mathrm { o r a c l e } }$ denote the expected sample counts of the rule and of an oracle that knows $p$ and therefore skips exploration, and let $w ^ { * }$ and $w ^ { \mathrm { p e s s } }$ be the smallest windows satisfying $p _ { \mathrm { e r r } } [ w ^ { * } ; p ] ~ \leq ~ p ^ { * }$ and $p _ { \mathrm { e r r } } [ w ; q ^ { \mathrm { p e s s } } ] + \rho _ { N } \leq p ^ { * }$ respectively.

Theorem 5 (Oracle gap, achievable instances). For instances on which both rules take thefast branch,

$$
\begin{array} { r l } { 0 \le C _ { \mathrm { a d a p t } } - C _ { \mathrm { o r a c l e } } \le T _ { \mathrm { c a p } } + \mathbb { E } [ T _ { \mathrm { r u n } } ( \hat { p } _ { ( 1 ) } , w ^ { \mathrm { p e s s } } ) ] } \\ & { \quad \quad - \mathbb { E } [ T _ { \mathrm { r u n } } ( p _ { ( 1 ) } , w ^ { \ast } ) ] + O ( \delta E ^ { \ast } ) . } \end{array}
$$

The three components are the exploration cap, which the oracle does not pay, the additional waiting time induced by the conservative window, and the expected cost of mistriggers. The proof proceeds in three steps: the pessimistic feasible set is contained in the true one, so $w ^ { \mathrm { p e s s } } \geq w ^ { \ast } ;$ ; on achievable instances the two rules difer term by term along the branch decomposition; and outside ${ \mathcal { G } } ,$ of probability at most $\delta ,$ each path costs at most $E ^ { * }$

Step 1 $( w ^ { \mathrm { p e s s } } \geq w ^ { \ast } )$ . On ${ \mathcal { G } } ,$ Lemma 2 gives $p _ { \mathrm { e r r } } [ w ; p ] \leq$ $p _ { \mathrm { e r r } } [ w ; q ^ { \mathrm { p e s s } } ] + \rho _ { N }$ for every � and every $p$ in the confidence set, so any window accepted by the pessimistic criterion is feasible for $p ;$ the pessimistic feasible set is therefore a subset of the true one.

Step 2 (branch-conditional comparison). Conditional on branch entry at time � with remaining budget $N = E ^ { * } - \tau$ Theorem 3 and the run-waiting formula give

$$
\begin{array} { r l } { C _ { \mathrm { a d a p t } } - C _ { \mathrm { o r a c l e } } } & { \leq T _ { \mathrm { c a p } } + \mathbb { E } [ T _ { \mathrm { r u n } } ( \hat { p } _ { ( 1 ) } , w ^ { \mathrm { p e s s } } ) ] } \\ & { \quad \quad - \mathbb { E } [ T _ { \mathrm { r u n } } ( p _ { ( 1 ) } , w ^ { * } ) ] . } \end{array}
$$

where $T _ { \mathrm { { c a p } } }$ bounds the exploration cost, the truncation of the waiting time by � only lowers the adaptive term, and the expectations use the plug-in majority probability at entry.

Step 3 (mis-triggers). Of  the plug-in estimates may be uninformative and each such path spends at most $E ^ { * }$ contributing $O ( \delta E ^ { * } )$ .

## C. Competing rules: exact stopping criteria

Every competitor is implemented from its published stopping rule and consumes the same sampled stream as BARS: one sample is one draw from the underlying distribution (synthetic setting) or one parsed decode (LLM setting). For a stream $x _ { 1 } , x _ { 2 } , \ldots$ we write $N _ { c } ( t )$ for the count of value � up to time $t , \hat { p } _ { c } ( t ) = N _ { c } ( t ) / t .$ , and $\hat { p } _ { ( 1 ) } ( t ) \geq \hat { p } _ { ( 2 ) } ( t )$ for the top two empirical frequencies.

## C.1. Fixed-budget and fixed-confidence best-arm identification

FIXED-�. Stop at $t = N$ and return arg max $N _ { c } ( N ) ;$ the rule never refuses and always spends the whole budget.

Successive Rejects [30]. With budget � and � arms, set $\begin{array} { r } { \bar { n } = K + \frac { 1 } { 2 } - \sum _ { i = 2 } ^ { K } \frac { 1 } { i } . } \end{array}$ , and in round $k = 1 , \dots , K - 1$ draw $n _ { k } = \lceil ( B \bar { - } K ) / ( \bar { n } ( \dot { K } + 1 - k ) ) \rceil$ times from each surviving arm, then eliminate the arm with the smallest empirical mean.

Sequential Halving [32]. Split the budget uniformly over $\lceil \log _ { 2 } K \rceil$ rounds and, in each round, draw equally from the survivors and keep the better half by empirical mean.

lil’UCB [33]. Round-robin sampling with the radius rad $\begin{array} { r } { \underline { { \mathrm { \Pi } } } _ { a } ( t ) \ = \ \sqrt { 1 . 5 \ln \bigl ( ( \ln n _ { a } ) ^ { 2 } K 4 t ^ { 2 } / \delta \bigr ) / n _ { a } ; } } \end{array}$ stop when $\hat { p } _ { a } -$ $\mathrm { r a d } _ { a } > \mathrm { m a x } _ { b \ne a } ( \hat { p } _ { b } + \mathrm { r a d } _ { b } )$ for the leader �.

LUCB [32]. In each round sample the empirical leader ℎ and runner-up � once, and stop when $\hat { p } _ { h } -  { \mathrm { r a d } _ { h } } > \hat { p } _ { \ell } +  { \mathrm { r a d } _ { \ell } }$ with the Hoefding radius ra $1 _ { a } = \sqrt { \ln ( 4 K t ^ { 2 } / \delta ) / ( 2 n _ { a } ) } .$

Hoefding race [36]. In each round draw once from every surviving arm and eliminate any arm � with $\hat { p } _ { a } +$ $\mathrm { r a d } _ { a } < \hat { p } _ { \mathrm { b e s t } } - \mathrm { r a d } _ { \mathrm { b e s t } }$ , where $\mathrm { r a d } _ { a } = \sqrt { \ln ( 2 K / \delta ) / ( 2 n _ { a } ) }$

All five rules are run under the same per-path budget $E ^ { * }$ and return their leader at expiry; FIXED-� and the elimination rules have no refusal output, which is why Section 5.1 reports their conditional and total error as one number.

## C.2. Sequential early-stopping rules for LLM majority voting

ESC [10]. Given a window � and a cap $L ,$ , stop at the first $t \geq w$ with $x _ { t - w + 1 } = \cdots = x _ { t }$ (a window in which all answers coincide), and answer with the plurality over the prefix; if no such window occurs, answer with the plurality over � samples. The paper’s control scheme fixes � from a warmup window of size $w _ { 0 } = 5 ;$ we report $w \in \{ 2 , 3 , 4 \}$ which brackets the values its figures use.

CGES [12]. The posterior score of candidate $a _ { i }$ after � samples is $\begin{array} { r } { s _ { a _ { i } } = \prod _ { t : \ R _ { t } = a _ { i } } C _ { t } \prod _ { t : \ R _ { t } \neq a _ { i } } \frac { 1 - C _ { t } } { K - 1 } } \end{array}$ , normalised over candidates; the rule stops at the first � with ma $\mathrm { X } _ { i } s _ { a _ { i } } \geq \gamma$ The score depends on a per-sample decoding confidence $C _ { t } ,$ , which the pool-replay protocol does not carry, so we evaluate the confidence-free reduction $s _ { a _ { i } } \propto N _ { a _ { i } } ( t ) + 1$ (the Dirichlet-multinomial posterior mean under a uniform prior, with an explicit unseen-answer bucket so that $K \ge 2$ from the first round, as the paper’s online implementation does) and stop on max $\hat { p } _ { a _ { i } } ( t ) \geq \gamma$ with $\gamma ~ \in ~ \{ 0 . 9 , 0 . 9 5 \}$ . This reduction is stated in Section 5.1 and is the reason the CGES numbers here are a lower bound on its advantage rather than an evaluation of its confidence weighting.

DASE [13]. DASE commits when the evidence is concentrated and otherwise falls back to the global frequency; on a pool without deliberation rounds we instantiate it as committing at the first � with $\hat { p } _ { ( 1 ) } ( t ) \ge 1 / 2 + \tau$ and $\tau = 0 . 1$ and falling back to the plurality at �.

MARS [18]. The margin between the leader and the runner-up must remain safe under a conservative bound on future vote movement; at the answer level we use ma $\mathrm { \ g i n } ( t ) =$ $N _ { ( 1 ) } ( t ) - N _ { ( 2 ) } ( t ) > z \sqrt { 2 \hat { \sigma } ^ { 2 } t }$ with $z = 2$ and $\hat { \sigma } ^ { 2 }$ the plugin variance of Section 3.2, stopping at the first � where this holds. MARS’s real rule also learns per-trace switch probabilities, which need partial-trace states that the pool does not carry.

MMC, the certificate of certified self-consistency [25]. The Martingale Majority Certificate is an anytime-valid stopping rule built on �-processes; at the answer level it is the certified width of Section 3.4 applied without the budget or refusal logic, $\begin{array} { r } { w _ { t } = \sqrt { 2 \hat { \sigma } _ { t } ^ { 2 } ( 2 \ln t + \ln \frac { \pi ^ { 2 } } { 3 \delta } ) / t } . } \end{array}$ , with a stop at the first � where $\hat { \Delta } _ { t } - w _ { t } \ge \Delta _ { \mathrm { h i g h } }$ . The distinction from BARS is not the width but the absence of a refusal output and of a budget argument: MMC answers at � whatever the width says, which is why its conditional error is reported on the full pool.

Behavior across the dificulty sweep, 2000 seeds per cell, all rules on the same instances. “Answer” is the answer rate, “cond. err.” the error on answered instances (for the rules with no refusal output this equals the total error), “miss (feas.)” the rate at which a feasible instance is refused, and “OK (infeas.)” the rate at which an infeasible instance is correctly refused; the last two are defined only for BARS.
<table><tr><td>Δ</td><td>Rule</td><td>Answer</td><td>Cond. err.</td><td>Total err.</td><td>Mean n</td><td>Miss (feas.)</td><td>OK (infeas.)</td></tr><tr><td>0.02</td><td>BARS (practical)</td><td>0.134</td><td>0.3806</td><td>0.9170</td><td>134.0</td><td>0.0000</td><td>0.8660</td></tr><tr><td>0.02</td><td>Fixed-N</td><td>1.000</td><td>0.3345</td><td>0.3345</td><td>200.0</td><td></td><td></td></tr><tr><td>0.02</td><td>SR</td><td>1.000</td><td>0.4445</td><td>0.4445</td><td>200.0</td><td></td><td></td></tr><tr><td>0.02</td><td>SH</td><td>1.000</td><td>0.4160</td><td>0.4160</td><td>150.0</td><td></td><td></td></tr><tr><td>0.02</td><td>LUCB</td><td>1.000</td><td>0.3760</td><td>0.3760</td><td>200.0</td><td></td><td></td></tr><tr><td>0.02</td><td>lil&#x27;UCB</td><td>1.000</td><td>0.3850</td><td>0.3850</td><td>200.0</td><td></td><td></td></tr><tr><td>0.05</td><td>BARS (practical)</td><td>0.161</td><td>0.2617</td><td>0.8815</td><td>131.1</td><td>0.0000</td><td>0.8395</td></tr><tr><td>0.05</td><td>Fixed-N</td><td>1.000</td><td>0.2110</td><td>0.2110</td><td>200.0</td><td></td><td></td></tr><tr><td>0.05</td><td>SR</td><td>1.000 1.000</td><td>0.3720 0.3490</td><td>0.3720</td><td>200.0</td><td></td><td></td></tr><tr><td>0.05 0.05</td><td>SH</td><td>1.000</td><td>0.2185</td><td>0.3490</td><td>150.0</td><td></td><td></td></tr><tr><td>0.05</td><td>LUCB</td><td>1.000</td><td>0.3215</td><td>0.2185 0.3215</td><td>200.0</td><td></td><td></td></tr><tr><td></td><td>lil&#x27;UCB</td><td></td><td></td><td></td><td>200.0</td><td></td><td></td></tr><tr><td>0.1</td><td>BARS (practical)</td><td>0.240</td><td>0.1167</td><td>0.7880</td><td>124.4</td><td>0.0000</td><td>0.7600</td></tr><tr><td>0.1</td><td>Fixed-N</td><td>1.000</td><td>0.0570</td><td>0.0570</td><td>200.0</td><td></td><td></td></tr><tr><td>0.1</td><td>SR</td><td>1.000</td><td>0.2490</td><td>0.2490</td><td>200.0</td><td></td><td></td></tr><tr><td>0.1</td><td>SH</td><td>1.000</td><td>0.1580</td><td>0.1580</td><td>150.0</td><td></td><td></td></tr><tr><td>0.1 0.1</td><td>LUCB</td><td>1.000</td><td>0.0630</td><td>0.0630</td><td>200.0</td><td></td><td></td></tr><tr><td></td><td>lil&#x27;UCB</td><td>1.000</td><td>0.1975</td><td>0.1975</td><td>200.0</td><td></td><td></td></tr><tr><td>0.15</td><td>BARS (practical)</td><td>0.316</td><td>0.0380</td><td>0.6960</td><td>117.8</td><td>0.0000</td><td>0.6840</td></tr><tr><td>0.15</td><td>Fixed-N</td><td>1.000</td><td>0.0115</td><td>0.0115</td><td>200.0</td><td></td><td></td></tr><tr><td>0.15 0.15</td><td>SR SH</td><td>1.000</td><td>0.1865</td><td>0.1865</td><td>200.0</td><td></td><td></td></tr><tr><td>0.15</td><td></td><td>1.000</td><td>0.0830</td><td>0.0830</td><td>150.0</td><td></td><td></td></tr><tr><td>0.15</td><td>LUCB</td><td>1.000</td><td>0.0130</td><td>0.0130</td><td>200.0</td><td></td><td></td></tr><tr><td></td><td>lil&#x27;UCB</td><td>1.000</td><td>0.1330</td><td>0.1330</td><td>200.0</td><td></td><td></td></tr><tr><td>0.2</td><td>BARS (practical)</td><td>0.472</td><td>0.0244</td><td>0.5395</td><td>102.9</td><td>0.0000</td><td>0.5280</td></tr><tr><td>0.2</td><td>Fixed-N</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.2</td><td>SR</td><td>1.000</td><td>0.1370</td><td>0.1370</td><td>200.0</td><td></td><td></td></tr><tr><td>0.2</td><td>SH</td><td>1.000</td><td>0.0360</td><td>0.0360</td><td>150.0</td><td></td><td></td></tr><tr><td>0.2</td><td>LUCB</td><td>1.000</td><td>0.0025</td><td>0.0025</td><td>200.0</td><td></td><td></td></tr><tr><td>0.2</td><td>liI&#x27;UCB</td><td>1.000</td><td>0.0730</td><td>0.0730</td><td>200.0</td><td></td><td></td></tr><tr><td>0.3</td><td>BARS (practical)</td><td>0.821</td><td>0.0043</td><td>0.1825</td><td>64.8</td><td>0.1790</td><td></td></tr><tr><td>0.3</td><td>Fixed-N</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.3</td><td>SR</td><td>1.000</td><td>0.0995</td><td>0.0995</td><td>200.0</td><td></td><td></td></tr><tr><td>0.3</td><td>SH</td><td>1.000</td><td>0.0040</td><td>0.0040</td><td>150.0</td><td></td><td></td></tr><tr><td>0.3</td><td>LUCB</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.3</td><td>lil&#x27;UCB</td><td>1.000</td><td>0.0150</td><td>0.0150</td><td>200.0</td><td></td><td></td></tr><tr><td>0.4</td><td>BARS (practical)</td><td>0.973</td><td>0.0005</td><td>0.0275</td><td>38.6</td><td>0.0270</td><td></td></tr><tr><td>0.4</td><td>Fixed-N</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.4</td><td>SR</td><td>1.000</td><td>0.0600</td><td>0.0600</td><td>200.0</td><td></td><td></td></tr><tr><td>0.4</td><td>SH</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>150.0</td><td></td><td></td></tr><tr><td>0.4</td><td>LUCB</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>199.8</td><td></td><td></td></tr><tr><td>0.4</td><td>lil&#x27;UCB</td><td>1.000</td><td>0.0015</td><td>0.0015</td><td>200.0</td><td></td><td></td></tr><tr><td>0.6</td><td>BARS (practical)</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>18.5</td><td>0.0000</td><td></td></tr><tr><td>0.6</td><td>Fixed-N</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.6</td><td>SR</td><td>1.000</td><td>0.0260</td><td>0.0260</td><td>200.0</td><td></td><td></td></tr><tr><td>0.6</td><td>SH</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>150.0</td><td></td><td></td></tr><tr><td>0.6</td><td>LUCB</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>166.9</td><td></td><td></td></tr><tr><td>0.6</td><td>lil&#x27;UCB</td><td>1.000</td><td>0.0000</td><td>0.0000</td><td>200.0</td><td></td><td></td></tr><tr><td>0.8 0.8</td><td>BARS (practical)</td><td>1.000</td><td>0.0000 0.0000</td><td>0.0000 0.0000 200.0</td><td>14.0</td><td>0.0000</td><td></td></tr><tr></table>