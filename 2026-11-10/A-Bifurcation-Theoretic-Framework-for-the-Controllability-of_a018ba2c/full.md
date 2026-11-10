# A Bifurcation-Theoretic Framework for the Controllability of Emergent Intelligent Systems

Hao Li<sup>∗1</sup> and Shaowen Lu<sup>2</sup>

<sup>1</sup>Aufkenn Technologies Co., Ltd., Ningbo, China <sup>2</sup>Northeastern University, Shenyang, China

November 10, 2026

## Abstract

Real-world intelligent systems must reconcile two demands: the adaptability conferred by emergence and the steerability required for accountable deployment. We show that these demands are governed by two distinct bifurcations on the same object and that, consequently, increased capability does not imply increased controllability. Emergence is a bifurcation of the capability variable S, with criterion Γ := F<sup>′</sup>(0) > 1 equivalent to the existence of a unique positive stable fixed point of a self-consistency equation, following solely from the concavity of F. Controllability is a bifurcation of the permission variable c, whose dynamics include an erosion term and exhibit a saddle-node fold with hysteresis as the gate criterion ρ<sup>∗</sup> crosses unity. Three quantitative results follow: the critical exponent distinguishes families rather than classifying the transition; the gating-cost bound K(M) ∝ log(1 + M) grows with scope, not capability; and the fold threshold is γ<sub>c</sub> ≈ 4µκ. Calibrated in both directions, the criteria reproduce the two-dimensional Ising transition, admit two true-feedback order parameters, and reject nine candidate quantities, supporting the conclusion that the order parameter is geometric rather than capability-related. Since permission cannot be endogenized and no deployed system has been measured to satisfy Γ > 1 and ρ<sup>∗</sup> > 1 simultaneously, increased capability does not imply controllability.

## 1 Introduction

## 1.1 Background

The capabilities of intelligent systems are not encoded item by item, but grow out of the interactions among many local units (Rumelhart, Hinton & McClelland, 1986), which is also what enables them to generalize to unseen tasks (LeCun, Bengio & Hinton, 2015). However, when such systems enter fields such as medicine, power, and manufacturing—settings that demand reviewability and accountability—emergence ceases to be a purely academic question and becomes an engineering constraint: the system must be allowed to emerge, yet one must remain responsible for its consequences.

Current systems use probability and statistics as their fitting mechanism, so their outputs are inherently uncertain; yet the fields of medicine, power, and manufacturing demand the determinacy that comes with reviewability and accountability. Converting the uncertain into the determinate is precisely the content of the goal “emergent and controllable.” This paper does not advocate eliminating uncertainty—it is both a source of capability and something that cannot be wholly erased. Instead, we stratify uncertainty: which uncertainties can be absorbed by criteria into determinate bounds, and which must be explicitly flagged as margins.

We take the brain as a reference. The brain is the only known system proven capable of general intelligence (Hassabis et al., 2017), and its organizing principle is “simple nodes + complex networks”: the nodes, though “simple,” possess internal structure (dendritic computation, London & Häusser, 2005; thermodynamic limits on ion-channel function, Betancourt & Machta, 2026); connectivity admits three readings—structural, functional, and efective (Bullmore & Sporns, 2009; Friston, 1994)—rather than being a mere accumulation of compute (Thompson et al., 2020). Connectomics indicates that the organizing principle of the brain is connectivity (Sporns, Tononi & Kötter, 2005), with connectivity lying at the mesoscopic scale between neurons and cognition. The criteria of this paper are derived from connectivity structure.

## 1.2 The Problem and the Limits of Existing Work

The assumption that “stronger capability naturally means more controllable” pervades the alignment and capability-risk discussion, exemplified by the scaling optimists and the “capability is alignment” narrative (Wei et al., 2022). We argue that this assumption is incorrect.

Existing research answers “why emergence happens,” not “how to achieve controllability.” Since Lewes (1875) and Anderson (1972), explanations have faced three persistent dificulties: the abruptness may be a measurement artifact (Schaefer et al., 2023; Du et al., 2024), the core mechanism remains a black box, and there is tension between performance and interpretability (Ribeiro et al., 2016; Lundberg & Lee, 2017). Engineering practice can only confirm emergence after the fact by means of scale thresholds; this is passive observation, not active design.

The notion of “emergence” used in this paper is dynamical, determined by the fixed-point stability of a self-consistency equation, and difers from information-theoretic or causal notions of emergence (Hoel et al., 2013; Tononi, 2004). Emergence can be engineered (Jansma & Hoel, 2025), but “how to control it” still lacks a measurable coordinate.

## 1.3 Research Question

What measurable, intervenable, and reviewable coordinate should be used to determine whether an intelligent or emergent system has reached a “deployable controlled state,” and to provide an operable trade-of between control and adaptation?

## 1.4 Core Thesis

Increased capability does not imply controllability. The emergence and controllability of intelligence are two bifurcations on the same object, characterized by two criteria that do not imply each other; controlled emergence requires an exogenous gate, which does not arrive automatically as capability grows.

We put forward three theses:

A — Emergence is a bifurcation. The dynamics of capability S lose stability at $\Gamma : =$ $\mathcal { F } ^ { \prime } ( 0 ) > 1 $ : the zero solution loses stability and a unique positive stable fixed point appears. The criterion depends only on the concavity of $\mathcal { F }$ and on $\mathcal { F } ( 0 ) = 0$ , and is independent of network details (§2, §3).

B — Controllability is another bifurcation. The permission variable c is not a given parameter but possesses its own dynamics; these dynamics contain an erosion term, so that once the gate falls below the behavior it must cover, the supervisory apparatus itself begins to lose eficacy, the result being a saddle-node fold and hysteresis. The upper bound on gating cost grows with scope and not with capability (§2, §4).

C — The two bifurcations do not imply each other. The order parameter is a geometric quantity, not a capability quantity; making the system stronger does not make it more controllable. Controllability must rest on an exogenous gate (§5, §7, §8).

This paper provides a criterion system that turns the inverse problem of the forward/inverse design program of Li, Ren & Lu (2026, arXiv:2610.25630) into a decidable form: within the class where (Θ, U) is fixed, arg max J is reduced to a set of decidable inequalities (symbol mapping in §6).

## 1.5 Contributions

The contributions of this paper are fourfold:

1. Two bifurcations: the separation of emergence and controllability. We establish the framework that “emergence and controllability are two independent bifurcations on the same object”: each has a computable criterion (Γ > 1 and $\rho ^ { * } < 1 )$ , and neither implies the other. Consequently, the tacitly accepted assumption that “stronger capability naturally means more controllable” no longer holds, and controllability becomes an object that must be argued for and measured on its own, in parallel with emergence.

2. The dynamics and cost of controllability. We formulate controllability as a dynamical problem of the same form as emergence: the gate is no longer a given parameter but a state variable with its own bifurcation behavior (erosion, saddle-node fold, hysteresis, threshold $\gamma _ { c } \approx 4 \mu \kappa )$ ; the gating cost has an explicit upper bound $K ( M ) \propto \log ( 1 + M )$ that grows with scope and not with capability. “Can the gate hold” thus becomes a computable, budgetable engineering question, rather than an empirical dificulty confirmable only after the fact.

3. The feasibility triple. We turn the three coordinates of Li, Ren & Lu (2026) into an acceptable engineering form: the three coordinates $\alpha / \mathrm { r e p } / \chi$ , together with the three-term decomposition of ε (two hard criteria and one soft gap). Whether “controlled emergence is feasible” thus becomes an object that can be accepted item by item and explicitly rejected, rather than a qualitative judgment.

4. Two-directional calibration and hard boundaries. We calibrate the admission of order parameters in both directions—positive controls and a rejection gallery (G1–G4)—thereby establishing the discriminative principle that “the order parameter is a geometric quantity, not a capability quantity”; we argue that the exogeneity of the gate $( u \perp J )$ is a precondition for safety, and provide falsification conditions F1–F18. The criteria are therefore both verifiable and falsifiable.

## 1.6 Methodological Statement

All claims in this paper fall into three classes, which the reader should treat separately:

• Class I — Identities and algebraic corollaries (coverage closure, the multiplicative bound, the independence of Γ from higher-order terms): these are algebraic identities and are not falsifiable.

• Class II — Normative choices (the weighted sum, thresholds, the position of the gate’s red line): these are value choices, not falsifiable, and must be signed of by the responsible party.

• Class III — Falsifiable assertions $( \Gamma > 1 , \rho ^ { * } < 1$ , the fold threshold, the cost scaling, the ε decomposition): these can be refuted by data, and the F-series at the end of each section states the corresponding refutation conditions (list in Supplementary Material F).

All substantive claims of this paper belong to Class III; Class I is not a finding, and Class II is not a law. Correspondingly, we state explicitly the ways in which the work could be falsified: if the scaling of behavioral diversity is not $\log _ { 2 } ( 1 + M S )$ , then the conclusion that “the cost bound depends only on scope” no longer holds unconditionally (§4).

## 2 Two Bifurcations on the Capability–Permission Pair

The structure of controlled emergence can be summarized as: the gate is multiplicative, the channels are additive, and emergence is an endogenous bifurcation. The gate being multiplicative means that the readout gating is veto-like $( c \to 0 \Rightarrow \Psi \to 0 )$ ; the channels being additive means that redundant channels form a “noisy $\begin{array} { r } { \mathrm { O R } ^ { \prime \prime } \left( Q _ { g } = 1 - \prod _ { i } ( 1 - r _ { i } a _ { i } ) \right. } \end{array}$ , see $\ S 3 )$ so that any single channel passing sufices; emergence being endogenous means that $S$ appears on both sides of the self-consistency equation, so that whether a solution exists is decided by fixed-point stability rather than by a threshold drawn by hand. The rigorous form of these three points is a system of three equations, expressing emergence and controllability as two bifurcations on the same object $( S , c )$ ; controlled emergence holds if and only if $\Gamma > 1$ and $\rho ^ { * } < 1$ and the two inequalities do not imply each other.

## 2.1 The Three Equations

$$
\Psi = c S , \qquad \dot { S } = \mathcal { F } ( S ; c ) - S , \qquad \dot { c } = G ( c , S ) .
$$

• The first equation is the readout: deployment potential = gate × capability; it is an algebraic relation and holds identically.

• The second equation is the capability side: $S$ appears on both sides of the equation, so multiple solutions are possible; emergence appears as a change in the number of solutions at some parameter value.

• The third equation is the permission side: c is no longer a given parameter but has its own dynamics; loss of control appears as the disappearance of its fixed point.

The whole paper treats a single object, namely the “capability–permission” pair $( S , c )$ ; the three equations correspond respectively to its readout, fast equation, and slow equation.

## 2.2 Bilateral Appearance of S in the Capability Equation

If S appeared only on the right-hand side, then $S = \mathcal { F } ( c )$ would be single-valued, with always exactly one solution, so that “whether emergence occurs” could only be decided by a threshold line drawn by hand—precisely the approach this paper rejects. Once S appears on both sides, the self-consistency equation $\mathcal { F } ( S ; c ) = S$ may admit multiple solutions, and the number of solutions changes at $\mathcal { F } ^ { \prime } ( 0 ; c ) = 1$

emergence $\iff \Gamma : = \mathcal { F } ^ { \prime } ( 0 ; c ) > 1$ (zero solution loses stability, positive fixed point appears).

Emergence is thereby transformed from $^ { 6 6 } \mathrm { a }$ threshold drawn by hand” into $^ { 6 6 } \mathrm { a }$ change in the number of solutions”; this is the only substantive divergence between this paper and the explanatory tradition (for the philosophical demarcation of weak/strong emergence see Bedau, $1 9 9 7 ;$ this paper uses the term only at the functional level).

## 2.3 The Permission Variable as a State Variable

If c were treated merely as a parameter on the right-hand side, then all statements about “controllability” would degenerate into conditionals: “if c can hold, then Ψ is bounded,” while saying nothing about whether c can in fact hold. Promoting c to a state variable requires two steps: first, define it (the probability that the gate actually takes efect when needed, determined by three coordinates: resolving power $\mathcal { V } _ { c }$ , independence $u \perp J ,$ , and delay $\tau _ { \mathrm { g a t e } } )$ ; second, write an equation for it (the gate is maintained by investment, decays naturally, and is subject to erosion that accelerates failure once breached). Controllability thereby acquires exactly the same form as emergence:

controllability $\iff \rho ^ { * } < 1$ (the gate’s fixed point exists and is stable).

## 2.4 Controlled Emergence

$$
\Bigl | \mathrm { c o n t r o l l e d ~ e m e r g e n c e } \Bigr | \Longleftrightarrow \Gamma > 1 \land \rho ^ { * } < 1 \Bigr |
$$

The two inequalities do not imply each other: $\Gamma > 1$ with $\rho ^ { * } > 1$ corresponds to “can grow, cannot hold”; $\Gamma < 1$ with $\rho ^ { * } < 1$ corresponds to “can hold, but did not grow.” Both must hold for controlled emergence, which requires simultaneously accomplishing two engineering tasks that are mutually independent.

![](images/b974873db654248c64fa8eafdc9e903b1625dbaadfd57468e42781fefeb8f91f.jpg)  
Figure 1: On a minimal JEPA-style self-supervised loop, the gate criterion $\rho ^ { * }$ increases monotonically with representation capacity N (capability) and crosses $\rho ^ { * } = 1$ (dashed line) near $N \approx 3 ;$ capability $R ^ { 2 }$ remains $\approx 1$ throughout (no degradation). Capability and controllability are two independent knobs: increasing N only pushes $\rho ^ { * }$ higher and does not improve controllability. (Conceptual, not empirical.)

## 2.5 Separability of the Two Subsystems

The coupling between the two halves is one-directional and weak: c entering F is algebraic and strong; the entry of $\dot { S }$ into G consists of only the single term $\begin{array} { r } { \dot { \mathcal { V } } _ { a } = \frac { M \dot { S } } { ( 1 + M S ) \ln { 2 } } } \end{array}$ and is weak. Hence “strong capability with weak $\mathrm { g a t e } ^ { , \dag }$ is a common state, whereas “strong gate with weak capability” is a stable state (a one-way valve). Consequently, one may refute half of the thesis with only a half experiment.

The zero solution of the first half is an unstable point when $\Gamma > 1$ , whereas both fixed points of the second half are attractors: emergence manifests as appearance, and loss of control manifests as a fall (§4).

## 3 Emergence as a Bifurcation of Capability

The emergence criterion lies on the capability side. One picture of emergence is that of “two previously unconnected nodes forming a new connection” (neural reorganization, Kounios & Beeman, 2009; new synapse formation, Trachtenberg et al., 2002). Multiplicative gating corresponds to the readout $\Psi = c S$ , additive channels correspond to coverage closure $S =$ $\Sigma _ { g } w _ { g } Q _ { g }$ , and endogenous emergence corresponds to the $\Gamma > 1$ criterion.

Coverage closure (lemma): Suppose that within the channel set $C _ { g }$ of group g the channels are independent, and that the response of channel i’s competence to its own capability level is $a _ { i } = a _ { i } ( S )$ , with $a _ { i } ( 0 ) = 0$ and, near the origin, $a _ { i } ( S ) = \eta _ { i } S + O ( S ^ { 2 } )$ . Taking $r _ { i } = 1$ and $a _ { i } = 1 - e ^ { - \eta _ { i } S }$ , we obtain

$$
Q _ { g } = 1 - \prod _ { i \in C _ { g } } ( 1 - r _ { i } a _ { i } ) = 1 - e ^ { - \Gamma _ { g } S } , \qquad \Gamma _ { g } : = \sum _ { i \in C _ { g } } \eta _ { i } .
$$

Substituting into the self-consistency equation gives $\begin{array} { r } { S = \mathcal { F } ( S ) : = \sum _ { q } w _ { g } ( 1 - e ^ { - \Gamma _ { g } S } ) } \end{array}$ ). S<sup>∗</sup> depends on the network through $\{ \Gamma _ { g } \}$ , whereas the number of nodes and the intra-group assignment are entirely invisible: the network enters the equation only as a vector Γ (for the empirical observation that post-training weights fall into a low-dimensional subspace see Kaushik et al., 2025; the two are of the same family but in diferent spaces, and cannot be inferred from each other).

Theorem 1 (endogenous criterion of emergence): Suppose $\mathcal { F }$ satisfies $\mathcal { F } ( 0 ) = 0$ , strict concavity, and $\mathcal { F } ^ { \prime } ( 0 ) = \Gamma$ . Then

$\Gamma \leq 1 \iff S ^ { * } = 0$ is the unique fixed point;

$\Gamma > 1 \iff$ there is a unique positive fixed point, and it is stable.

The proof uses only concavity (for normal forms and stability criteria see Strogatz, 2015): $h ( S ) : = \mathcal { F } ( S ) - S$ satisfies $h ( 0 ) = 0 , h ^ { \prime } ( 0 ) = \Gamma - 1 , h ^ { \prime \prime } < 0$ . Adding higher-order terms does not move the criterion; it only changes its depth.

Three quantities are worth noting: the critical channel number $N ^ { * } = 1 / ( r \eta \bar { W } )$ (W<sup>¯</sup> being the mean node weight; below it the emergent state does not exist in the phase diagram); the critical gate $c ^ { * } = 1 / \Gamma ;$ and the critical exponent—the coverage-closure family (asymmetric, with an $S ^ { 2 }$ term) gives $\beta = 1$ , while the odd family (tanh, no $S ^ { 2 }$ term) gives $\beta = 1 / 2$ . The exponent is a criterion for selecting a family, not for classifying the transition (for family invariance under symmetry constraints see Barreiro, Kutz & Shlizerman, 2017; Yang & Choi, 2026): distinguishing first-order from second-order transitions can only be done through jumps and hysteresis.

There is, moreover, a trade-of between reuse and scale: splitting the channels into a shared pool and dedicated channels, let the degree of reuse be m and the reuse ratio be $q : = \eta _ { 2 } / \eta _ { 1 }$ $( \eta _ { 1 }$ the dedicated-channel gain, η<sub>2</sub> the shared-channel gain). The monotonicity of $S ^ { * }$ in m is governed by the relation between $q$ and the critical ratio $q ^ { * } = w _ { A } / ( w _ { A } + w _ { B } )$ , yielding a three-regime structure: for $q \leq q _ { l o }$ no sharing occurs $( m ^ { * } = 0 )$ ; for $q _ { l o } < q < q ^ { * }$ a unique interior optimum exists; for $q \geq q ^ { * }$ full sharing occurs $( m ^ { * } = n )$ ; the closed form of $q _ { l o }$ and the proof of unimodality are given in Supplementary Material E. In other words, if the quality of shared channels is not below the critical ratio, reuse substitutes for scale; below the lower bound, sharing is not worth doing at all.

## 4 Controllability as a Bifurcation of Permission

The controllability criterion lies on the permission side. The gate is not a scalar but a capability over three coordinates: resolving power $\nu _ { c }$ (the number of perturbation classes that can be distinguished, which must cover the behavioral diversity—Ashby’s law of requisite variety, Ashby, 1956), independence (whether the gate can act against the interest of the gated, formalized as $u \perp J ;$ independence can rest only on institutional structure rather than professional ethics, Bazerman, Morgan & Loewenstein, 1997), and delay $\tau _ { \mathrm { g a t e } }$ . The scalar $c$ is the degenerate case in which all three coordinates are saturated. The phenomenon of weak supervisors eliciting strong students (Burns et al., 2023; Lang et al., 2025) appears to violate requisite variety, but the two transmit a gradient signal rather than coverage of the behavioral space; this is “teaching” rather than “guarding,” and does not constitute a counterexample. What the gate must cover is the behavioral diversity actually exhibited by the system, which is bounded:

$$
\begin{array} { r } { \mathcal { V } _ { a } = \log _ { 2 } ( 1 + M S ) \in [ 0 , \bar { V } ] , \qquad \bar { V } : = \log _ { 2 } ( 1 + M ) , } \end{array}
$$

where M is the scale of the task space (scope) and $S \in [ 0 , 1 ]$ is capability (coverage). Had we taken $\log _ { 2 } S$ , it would be negative on $[ 0 , 1 ]$ , so the bounded form must be used. The only dimensionful quantities are $\mathcal { V } _ { a } , \mathcal { V } _ { c }$ (bits); the criteria may contain only the dimensionless ratios $\kappa : = w / \mathcal { V } _ { a }$ (the fractional transition bandwidth) and $d : = D / \nu _ { a }$ (the relative gap).

The dynamics of the coverage gap $D : = \mathcal { V } _ { c } - \mathcal { V } _ { a }$ contain an erosion term:

$$
\dot { D } = \rho u - \left[ \mu + \gamma _ { 0 } \sigma ( - d / \kappa ) \right] \left( \mathcal { V } _ { a } + D \right) - \dot { \mathcal { V } } _ { a } .
$$

The three terms are, in order, input–output $( \rho u )$ , natural decay $( \mu \colon$ staf turnover / standard drift / model refresh), and failure erosion (when $\mathcal { V } _ { c } < \mathcal { V } _ { a }$ the supervisory apparatus itself loses eficacy, so that below that line the system does not lag gradually but collapses at an accelerating rate; empirical corroboration is that narrow-domain fine-tuning can induce broad cross-domain misalignment, Betley et al., 2026).

The good fixed point (gate holds, erosion of):

$$
D ^ { * } = \frac { \rho u - \mu \mathcal { V } _ { a } - \dot { \mathcal { V } } _ { a } } { \mu } > 0 \Longleftrightarrow \rho _ { \mathrm { h o l d } } ^ { * } : = \frac { \mu \mathcal { V } _ { a } + \dot { \mathcal { V } } _ { a } } { \rho u } < 1 .
$$

When the erosion is strong enough and the transition sharp enough, a saddle-node fold appears between the good and bad branches: the positive fixed point collides with an unstable fixed point and vanishes, and c falls to 0. The fold threshold has the explicit form $\gamma _ { c } = \mu / \operatorname* { m a x } _ { \theta } \Phi ( \theta ) \approx 4 \mu \kappa$ $( \kappa : = w / \mathcal { V } _ { a } < 1 / 2 )$ . Loss of control is not an instability but a fall onto a second attractor (both branches are attractors), separated by a hysteresis band $\Delta \rho ^ { * } = \gamma _ { 0 } \gamma _ { a } / ( \rho u )$ . The cost of rebuilding far exceeds that of maintaining; once c falls below the compliance threshold, compliance is no longer the rational choice for the system (Baum, Binkyt˙e & Jahn, 2026), the erosion term opens, and the transition accelerates.

The investment required to maintain $D ^ { * } > 0$ , taking the capability-saturated bound $S \to 1$ is

$$
K ( M ) = \frac { \kappa _ { 0 } \mu } { \rho } \log _ { 2 } ( 1 + M ) , \qquad M _ { \mathrm { m a x } } = 2 ^ { \rho B / ( \kappa _ { 0 } \mu ) } - 1 .
$$

This bound contains no $S { : }$ the cost bound grows with scope M and not with capability $S ;$ when unsaturated, the actual cost $\propto \log _ { 2 } ( 1 + M S )$ increases monotonically in capability, at most logarithmically, approaching the bound. Hence what determines deployment dificulty is scope (a product decision), not capability (a model property); what the budget caps is scope, not the capability ceiling.

## 5 Non-Implication of the Two Criteria and Gate Exogeneity

The two criteria $\Gamma > 1$ and $\rho ^ { * } < 1$ involve entirely diferent quantities: Γ comes from ${ \mathcal { F } } ^ { \prime } ( 0 )$ (capability side), and $\rho ^ { * }$ from the gate’s balance of income and expenditure (permission side); algebraically they do not imply each other. This conclusion can be unpacked in three layers:

1. Constructive separability: fixing the capability side and changing only the mode of inspection sufices to move a system across cells;

2. Algebraic non-implication: the two criteria share no common quantity;

3. Empirical independence remains unproven: no system currently in service has been measured to fall in $( \Gamma > 1 , \rho ^ { * } > 1 )$ simultaneously.

Therefore the hard boundary is: the gate must be exogenous, formalized as $u \perp J$ (the supervisory input is not within the arg max of the gated). If this is violated, c can still take apparently normal values, but it no longer belongs to the responsible party, and all derivations concerning $\rho ^ { * }$ lose their object. The mainstream “alignment” routes (writing human preferences into the objective, Amodei et al., 2016; corrigibility, Soares et al., 2015; interruptibility, Orseau & Armstrong, 2016; control protocols residing in the same optimization process, Greenblatt et al., 2023) all fall inside $u \perp J ;$ they make the gate more likely to be obeyed, rather than independent of the gated.

This “gate must be exogenous” requirement holds equally on the slow time scale: new channels grow additively under local rules and cost constraints (incorporated only when the marginal controlled contribution is positive), but whether a new layer may enter service is still multiplicatively licensed by the external gate. Growth is endogenous, permission is exogenous; when the gate is closed, even the most profitable channel may not be incorporated (complete generative skeleton in §10).

## 6 The Feasibility Triple

The two criteria must also be reduced to acceptable engineering coordinates. We decompose “engineering” into three independently measurable coordinates: controllability $\alpha ,$ reproducibility rep, and formalizability $\chi ;$ these correspond to the $\kappa / \rho / \phi$ of Li, Ren $\&$ Lu (2026), and its feasibility criterion is turned into a three-term decomposition of ε. The symbol mapping is as follows:

Table 1: Symbol mapping to the feasibility triple.
<table><tr><td>Li, Ren &amp; Lu (2026)</td><td>This paper</td><td>Meaning</td></tr><tr><td>κ (controllability)</td><td> $\alpha = 1 - \rho _ { \mathrm { h o l d } } ^ { * }$ </td><td>controllability</td></tr><tr><td> $\rho \ ( \mathrm { r e p r o d u c i b i l i t y } )$ </td><td>rep</td><td>reproducibility</td></tr><tr><td> $\phi \ ( \mathrm { f o r m a l i z a b i l i t y } )$ </td><td>X</td><td>formalizability</td></tr><tr><td> $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ </td><td> $\varepsilon ^ { * } \leq \varepsilon _ { \mathrm { t i p } }$ </td><td>feasibility</td></tr></table>

This paper additionally uses $\kappa : = w / \mathcal { V } _ { a }$ (transition bandwidth) and $\rho$ (supervisory productivity), which are homonymous but distinct from the κ (controllability) and $\rho$ (reproducibility) of Li, Ren & Lu (2026), and must be distinguished.

## 6.1 α (Controllability)

$$
\alpha : = 1 - \rho _ { \mathrm { h o l d } } ^ { * } = 1 - \frac { \mu \mathcal { V } _ { a } + \dot { \mathcal { V } } _ { a } } { \rho u } \in ( - \infty , 1 ] .
$$

$\alpha > 0$ means the gate holds, and α is the share of the total requirement covered by the margin; $\alpha \leq 0$ means the gate must collapse, in which case the feasibility distance is $\varepsilon _ { \alpha } = +$ ∞ (hard infeasibility, not $^ { 6 6 } \mathrm { a }$ little short”).

## 6.2 rep (Reproducibility)

Table 2: Layers at which reproducibility must be measured.
<table><tr><td>Layer</td><td>Procedure</td><td>Counts as repro- ducibility</td></tr><tr><td>(i) same-configuration repetition (same code, bitwise identical same seed)</td><td></td><td>no (identity)</td></tr><tr><td>versions / floating-point order)</td><td>(ii) perturbed repetition (threads / library measurable difference after 40k steps yes</td><td></td></tr><tr><td>(iii) environmental repetition (across devices versions)</td><td></td><td>yes</td></tr></table>

Reproducibility must be measured at layers (ii)(iii); reporting only the layer-(i) “reproducible given the same seed” is an invalid claim. The criterion is rep $: = 1 - { \frac { \mathrm { V a r } _ { \mathrm { p e r t u r b } } } { \mathrm { a l l o w a b l e ~ v a r i a n c e } } }$ , requiring $\mathrm { r e p } \geq 0$ for acceptance. Under zero failures, the one-sided confidence lower bound is given by the Clopper–Pearson interval (Clopper & Pearson, 1934): with n consecutive zero failures, the one-sided confidence lower bound on the success rate is $\zeta ^ { 1 / n }$ ; requiring it to satisfy $\geq \varrho$ yields the required sample size $n _ { \mathrm { m i n } } = \lceil \ln \zeta / \ln \varrho \rceil$

## 6.3 Order-Parameter Admission Criteria (G1–G4)

A quantity $S$ can serve as an order parameter in the sense of this paper if and only if it passes the following four tests:

• G1 Intensive quantity: the value in the ordered phase does not drift with scale $( \geq 4$ scales, CIs pairwise overlapping, slope CI containing 0);

• G2 Self-consistency: can be written as a self-consistency equation free of microscopic parameters (fit coeficients do not vary with N);

• G3 Scaling: of two types, G3a (efective equilibrium free energy, $\beta / \nu = 1 / 4$ , Landau, 1937) and G3b (non-equilibrium dynamical scaling, e.g. $m ( t ) \sim t ^ { - 1 / 2 } )$ ;

• G4 Shape matching: the functional form of the criterion must be consistent with the asymptotic behavior of the object under test; including bounded-domain extrapolation (the fitting family must respect the domain of values at parameter extremes; out-of-range is invalidated outright, regardless of significance).

Two design rules: the order parameter must be of ratio/overlap type, not of norm/count type; the functional form of the criterion must match the asymptotic behavior of the object under test—fitting with the wrong family yields “an interval without meaning” rather than “not significant.”

## 6.4 The Three-Term Decomposition of $\varepsilon$

$$
\varepsilon = \left\{ \begin{array} { l l } { + { \infty } , } & { \alpha \leq 0 ~ \mathrm { o r } ~ \chi ~ \mathrm { f a i l s ~ G 1 - G 4 } \quad \mathrm { ( h a r d ~ i n f e a s i b l e ) } } \\ { \displaystyle \frac { 1 - \mathrm { r e p } _ { \mathrm { m e a s u r e d } } } { \mathrm { r e p } } , } & { \alpha > 0 , ~ \chi ~ \mathrm { p a s s e s } \quad \mathrm { ( s o f t ~ g a p ) } } \end{array} \right.
$$

$$
{ \mathrm { f e a s i b l e } } \iff \varepsilon \leq \varepsilon _ { \mathrm { t i p } } .
$$

Feasibility has three sources, two of which are hard: when controllability or formalizability is unqualified, there is no $^ { 6 6 } \mathrm { a }$ little short”—only “infeasible”; $^ { 6 6 } \mathrm { a }$ little short” can arise only from reproducibility.

## 7 Empirical Calibration of the Criteria

This section calibrates the criteria in two directions: positive controls show that the criteria can recognize a genuine transition, and the rejection gallery shows that they can refute spurious candidates.

## 7.1 Positive Controls

The same finite-size scaling + relative-collapse pipeline run on the 2D Ising positive system reproduces $T _ { c } = 2 . 2 5 6$ (expected exact value 2.269, Onsager, 1944), $1 / \nu = 1 . 0 3 7$ (expected 1.000), $\beta / \nu = 0 . 0 9 4$ (expected 0.125, due to finite-size corrections). The criteria can answer $^ { 6 6 } \mathrm { y e s } ^ { \mathrm { , 9 } }$ to a genuine transition, so the subsequent “no”s are the criteria discriminating, not the pipeline failing.

## 7.2 Two Order Parameters in Two Universality Classes

First, the spectral-model order parameter $m = \operatorname { t a n h } ( g m )$ (odd family): on the coupled network $x _ { t + 1 } = \phi ( J x _ { t } )$ (ϕ = tanh, a true-feedback system), the dominant-mode amplitude m satisfies the mean-field self-consistency equation $m = \operatorname { t a n h } ( g m ) , g = \rho ( J )$ . m passes G1 $( N = 2 0 0 / 8 0 0 / 3 2 0 0$ nearly coincide), G2 (a small-pulse probe at rest independently estimates ${ \hat { g } } ,$ , difering from the true value by $< 1 0 ^ { - 4 } )$ , G3b (dynamical scaling: at $g = 1 , m ( t ) \sim t ^ { - 1 / 2 }$ , measured $- 0 . 4 9 8 5 )$ , and G4 (shape matching $\beta = 0 . 4 9 8 4 , 1 / \delta = 0 . 3 3 3 2 ) \cdot$ —all four pass.

Second, the self-supervised coverage loop $m = 1 - e ^ { - g m }$ (coverage-closure family): N binary units, each step independently covered with probability $p = 1 - e ^ { - g m }$ , with m the group’s own consensus (no external signal). m passes G1 (stationary $m ^ { * } = 0 . 5 8 2 8 1 2 )$ , G2 (inverted $\hat { g }$ difers by $1 0 ^ { - 1 3 } )$ , G3b $( m ( t ) \sim t ^ { - 0 . 9 9 1 5 }$ , theory $t ^ { - 1 } )$ , and G4 $( \beta = 0 . 9 8 1 4$ , theory 1)—all four pass.

The two order parameters belong to two families (coverage closure $\beta = 1$ versus odd family $\beta = 1 / 2 )$ ; that the same G1–G4 set passes in diferent mechanisms and diferent universality classes shows the criteria are not tailored to any particular model (numerics and code in Supplementary Material C).

## 7.3 Rejected Candidate Quantities

Nine candidate quantities (capability quantities, representation-geometry quantities, first-layer spectral quantities) all fail: eight are rejected outright (G1 drift with scale $/$ no scaling $/ \Gamma < 1 )$ , and the ninth, $r _ { M }$ (the largest-eigenvalue fraction), with a bare collapse $Q = 1 . 0 3 8$ , appears to pass but fails the null-baseline test: the mean Q of smooth, non-critical synthetic data is 0.974, and $r _ { M }$ falls at its 65.5th percentile, so the low $Q$ can be explained by smooth data and fit degrees of freedom rather than a critical point.

## 7.4 Criterion Validation Precedes Object Rejection

Candidate quantity $\mathcal { C } _ { 2 }$ (spectral-domain peak concentration) was initially judged to ${ } ^ { 6 6 } \mathrm { f a i l \ G 1 , } ^ { \mathrm { \prime } }$ but the cause was found to be an error in the criterion itself: the original G1 required pairwise CI overlap, so that any quantity with finite-size corrections would necessarily fail—and a criterion that always fails carries no more information than one that always passes. After correcting the criterion (G1→asymptotic version G1<sup>′</sup>; adding G4 bounded-domain extrapolation), C<sub>2</sub> still fails $( P = 3 7$ overshoot, no plateau), at which point “fails” reflects that the object is genuinely unqualified.

The order must not be reversed: suspect the criterion first, then judge the object. Only when the object still fails after the criterion has been corrected by the data does it mean that the object is genuinely unqualified.

## 7.5 Geometric Versus Capability Quantities as Order Parameters

The nine rejected quantities are capability quantities, representation quantities, or first-layer spectral quantities; the two that pass are collective (geometric) quantities—the dominant-mode amplitude of the spectral model and the coverage fraction of the coverage loop—not the capability quantity of any single unit. Making the system stronger does not make it more controllable. A quantity that can pass the criteria requires a true-feedback loop system (self-supervision, context write-back), which a minimal static classifier cannot realize.

## 8 Scope and Boundaries

The domain of applicability of this criterion system is pinned down by “the gate must be exogenous” together with a series of falsification conditions (F1–F18, see Supplementary Material F).

Adjacent works are of the same family as this paper but difer from it: the Vicsek phase diagram has been rewritten at unprecedented scales, with the ordered phase only partially metastable (Duan et al., 2026a, 2026b); the scaling collapse of dual-gated Josephson junction arrays demonstrates the feasibility of the G3a/G3b methods in real physical systems (Fugl et al., 2026); the phase transitions of small Transformers are invisible on the loss curve and appear only under probes (Hong & Hong, 2025); the code scafolding surrounding an agent delimits its action space rather than constituting an exogenous supervisory gate (Lou et al., 2026); the mean-field transition classes and the order-parameter bifurcation of this paper belong to diferent observational layers (Truong, 2026); the stage genealogy of portable intelligence is borrowed here as sampling points for loop length (Zhu et al., 2023).

## 8.1 Gate Exogeneity

All conclusions presuppose the exogeneity of the gate (u ⊥ J of §5). There is a universal requirement in functional-safety practice that is isomorphic to this: the safety function must be physically and functionally independent of the controlled object, and must not reside in the software of the controlled object itself (the Therac-25 interlock failed after being moved into the software of the same machine, Leveson & Turner, 1993; the 737 MAX MCAS relied on a single sensor, Herkert et al., 2020). The “controllability” of this paper is in the sense of permission (allow/deny), not in the sense of dynamical regulation (pushing the system to a certain state), which is a diferent problem. In the autopoietic tradition, organizational closure is responsible for “staying alive,” while the gate is responsible for “being permitted to be alive” (Maturana & Varela, 1980); the gate of this paper is its externalization, not the closure itself.

## 8.2 The State of the Evidence

• Gate side, measured (half-witness): two checkable randomized-controlled records— Schroter et al. (2004) single-pass detection rate 0.237∼0.301, Godlee et al. (1998) 0.213∼0.263. Under a 5% miss-rate target, 8∼13 independent reviews are required; even dual-person review cannot hold $( \hat { \rho } ^ { * } = 4 . 2 \small { \sim } 6 . 3 )$ . The license-free measurements on opensource platforms are consistent: across four major projects, the mean number of independent

PR reviewers is about 1.3, far below what is required $( \hat { \rho } ^ { * } \approx 4 . 5 \sim 9 . 8 $ , see Supplementary Material G).

• Capability side, irremediable (structural conclusion): a qualifying witness requires a self-improvement loop to achieve $\Gamma > 1$ under the condition that “what counts as correct” is produced entirely by the system itself, without relying on any non-system-given source of judgment. Checking the recorded self-improvement loops one by one (self-play, self-training, self-distillation, etc.), none qualifies: self-play retains the rules, self-training retains the verifier, self-distillation retains the question bank. Three structural lower bounds (computational reachability, verification undecidability, non-increase of information, see Supplementary Material) show that this is not a shortage of data but conceptual immeasurability; the safety risk of recursive self-improvement accumulates and propagates across updates (Gong et al., 2026), in the same direction.

• No system currently in service has been measured to fall in $( \Gamma > 1 , \rho ^ { * } > 1 )$ simultaneously, so none is promoted to “witness.” Capability can be endogenous but responsibility cannot (Stanley, 2019; Watson, Levin & Buckley, 2022); this is the other-side basis for the exogeneity of the gate.

## 8.3 Falsification Conditions and Load-Bearing Structure

Each substantive claim carries its own refutation condition, stated in the same section as the claim. Grouped by “judgeable now $/$ requires building a measurement first”: the first group (Γ criterion, exponents, collapse) can be judged immediately by numerical simulation or pure logic; the second group (fold, cost, hysteresis) shares a single precondition, namely a field setting that can simultaneously produce output quantity, inspection quantity, and inspection efort.

The degradation ladder marks the load-bearing parts: even if the entire second half (the gate dynamics) is refuted, the first half remains fully independent $( \Gamma > 1 \iff$ positive fixed point, requiring only the concavity of F), as do the reproducibility layers of rep and the rule that “the order parameter must be of ratio $\mathrm { t y p e } ^ { \mathfrak { P } } ;$ ; the latter two are independent of any theory and are pure methodology.

## 9 Conclusion

1. Emergence and controllability are not two separate things but two bifurcations. The former is the loss of stability of the fixed point of capability $S \ ( \Gamma > 1 )$ ; the latter is a bifurcation on the permission side: the good fixed point of the gate vanishes as $\rho ^ { * }$ crosses 1 $( \rho ^ { * } > 1$ means the gate collapses).

2. The two inequalities cannot be derived from each other. “Can grow” and “can hold” must be argued for, measured, and accounted for separately.

3. Gating has a cost, and that cost is computable. $\textstyle K _ { \operatorname* { m a x } } ( M ) = { \frac { \kappa _ { 0 } \mu } { \rho } } \log _ { 2 } ( 1 + M )$ ; the cost bound grows with scope and not with capability; when the budget is capped, $M _ { \mathrm { m a x } } = 2 ^ { \rho B / ( \kappa _ { 0 } \mu ) } - 1$

Controllability is not the opposite of emergence but its price. What this paper delivers is criteria and forms, not an implementation: every claim can be refuted by data (F1–F18, see Supplementary Material F), but none has yet been verified on a real system. The ceiling of capability is set by the gate $( \Psi = c S$ and $S \leq 1$ give $\Psi \leq c )$ ; it is impossible to let capability grow without bound while not paying an unbounded price for guarding it; and “where to stop” can only be signed of by the responsible party, not computed by the system itself. Hence, increased capability does not imply controllability.

## 10 Future Work

What this paper delivers is a computable form; to turn the criteria from “writable” to “known,” and the “empirical independence of the two inequalities” from “unproven” to “proven,” the following work remains:

1. Proof of empirical independence. At present, “gate-side half-witness + capability-side structural lower bound” supports only the weaker claim that “the two critical ratios are set by diferent mechanisms,” not “empirical independence.” The strengthening path is the license-free data collection identified in §8.2: the number of reviewers and rounds of each merge request on public code-hosting platforms is exactly the denominator of $\hat { \rho } ^ { * } ;$ a field setting that can simultaneously produce output quantity, inspection quantity, and inspection efort would fill both the gate side and the capability side.

2. Parameter calibration on real systems. $\rho , \mu , \gamma _ { 0 } , \kappa _ { 0 }$ have not yet been calibrated on real organizations; turning them from “writable” to “known” requires longitudinal field data, otherwise the “runway length” of §4 remains a computable quantity but not a known one.

3. Strong emergence (layered emergence). This paper treats only the layer in which macroscopic order grows on nodes that already possess capability; the second bifurcation “from nothing to something” (slow variable A = 1) belongs to another time scale, and its complete generative skeleton (community detection / MDL / Rent’s rule / hierarchical JEPA) is left to subsequent work. The pre-registered negative result of hierarchical JEPA (H-JEPA) suggests that layered emergence is not a natural byproduct of increased depth, and requires specific architectural constraints or training objectives to be crossed.

4. Multiple capability dimensions. S is a scalar; with multiple capability dimensions Γ is a matrix, and the destabilizing direction is given by the Perron vector. The signed version of the spectral model $( Q _ { \mathrm { m i n } } = \operatorname* { m i n } ( g - 1 , 1 - s ) )$ ) already provides an entry point for the matrix case.

5. The free function of supervisory productivity $\rho ( S )$ . This paper treats ρ as constant throughout; if AI auditing makes $\rho$ grow with capability S, the race between the capability side and the permission side could be sustained indefinitely—this is precisely the form of “can capability bootstrap itself without bound,” which must be answered on real systems and cannot be extrapolated from the constant-ρ assumption of this paper.

## A Existence, Location, and Criterion of the Fold

Write $\dot { D } = \mathcal { R } - [ \mu + \gamma _ { 0 } \sigma ( - d / \kappa ) ] ( \mathcal { V } _ { a } + D ) , \mathcal { R } : = \rho u - \dot { \mathcal { V } } _ { a } ;$ σ smooth and monotonically increasing, $\sigma ( 0 ) = 1 / 2 ; \theta : = - D / w = - d / \kappa ; \kappa : = w / \mathcal { V } _ { a }$

In the sharp limit $( \sigma  \Theta )$ the two branches are each linear: the good branch $D _ { + } = \mathcal { R } / \mu - \mathcal { V } _ { a }$ $\left( \mathrm { s l o p e } - \mu \right)$ , the bad branch $D _ { - } = \mathcal { R } / ( \mu + \gamma _ { 0 } ) - \mathcal { V } _ { a }$ (slope −(µ + γ<sub>0</sub>)); both branches are stable. The self-consistency condition gives a three-regime criterion:

Table 3: Three-regime criterion of the gate.
<table><tr><td>Regime</td><td>Conclusion</td></tr><tr><td> $\rho _ { \mathrm { b r e a k } } ^ { * } < 1$ </td><td>only the good branch valid, the gate must hold</td></tr><tr><td> $\rho _ { \mathrm { h o l d } } ^ { * } < 1 < \rho _ { \mathrm { b r e a k } } ^ { * }$   $\rho _ { \mathrm { h o l d } } ^ { * } > 1$ </td><td>both branches valid, bistable (hysteresis) only the bad branch valid, the gate must collapse</td></tr></table>

where $\begin{array} { r } { \rho _ { \mathrm { h o l d } } ^ { * } : = \frac { \mu \mathcal { V } _ { a } + \dot { \mathcal { V } } _ { a } } { \rho u } , \rho _ { \mathrm { b r e a k } } ^ { * } : = \frac { ( \mu + \gamma _ { 0 } ) \mathcal { V } _ { a } + \dot { \mathcal { V } } _ { a } } { \rho u } } \end{array}$ , and the hysteresis window is $\Delta \rho ^ { * } = \gamma _ { 0 } \gamma _ { a } / ( \rho u )$

Smooth fold: the fold point must simultaneously satisfy $\dot { D } = 0$ and $\mathrm { d } \dot { D } / \mathrm { d } D = 0$ , giving

$$
\gamma _ { c } = \frac { \mu } { \mathrm { m a x } _ { \theta } \Phi ( \theta ) } , \qquad \Phi ( \theta ) : = \sigma ^ { \prime } ( \theta ) \Big ( \frac { 1 } { \kappa } - \theta \Big ) - \sigma ( \theta ) .
$$

The logistic closed form is $\begin{array} { r } { \Phi ( 0 ) = \frac { 1 - 2 \kappa } { 4 \kappa } , \ : \mathrm { s o } \ \gamma _ { c } \leq \frac { 4 \mu \kappa } { 1 - 2 \kappa } \approx 4 \mu \kappa \left( \kappa < \frac { 1 } { 2 } \right) } \end{array}$ . The fold exists if and only if $\gamma _ { 0 } \geq \gamma _ { c }$

Collapse time scale: near the fold point the equation becomes $\dot { x } = - \varepsilon - x ^ { 2 }$ , with escape time $\tau = \pi / ( 2 \sqrt { \varepsilon } ) \propto \varepsilon ^ { - 1 / 2 }$ ; the closer to the fold point, the slower the collapse (ghost efect); a $\mathrm { ^ { 6 6 } j u m p ^ { \mathrm { 7 } } }$ occurs only in the sharp limit, whereas finite κ gives a continuous rapid transition.

Exponent comparison: the saddle-node (second half) $1 / 2$ has a jump; tanh (odd, first half) $1 / 2$ has none; coverage closure (asymmetric) 1 has none. The exponent is a criterion for selecting a family, not for classifying the transition; first/second order can only be distinguished by jump and hysteresis.

## B Bounded Demand and Path-Dependent Failure

Bounded demand (Theorem D1): $\mathcal { V } _ { a } ( t ) \in [ 0 , \bar { V } ]$ holds identically, $\bar { V } = \log _ { 2 } ( 1 + M ) , \ \dot { \mathcal { V } } _ { a }$ = $\frac { M \dot { S } } { ( 1 + M S ) \ln 2 } \ge 0$ and tends to 0 as $S  1$ . Had one used the unbounded form $\mathcal { V } _ { a } \propto \log _ { 2 } S$ , it would be negative on [0, 1] and unusable as a diversity measure.

The long-run condition reduces to a static inequality (Theorem D2): the good-branch steady state is $D _ { + } = \rho u / \mu - \bar { V }$ , so $\rho _ { \infty } ^ { * } : = \mu \bar { V } / ( \rho u ) < 1 \iff$ the gate must hold in the long run, independent of the rate of capability growth.

Failure is path-dependent (Theorem D3): the worst gap is $\Delta _ { \mathrm { m a x } } = \bar { V } - \mathcal { V } _ { c } ( 0 )$ . Once D falls into the hysteresis band (crosses the fold), it cannot return even if the long-run condition holds. The failure condition is $( \rho _ { \infty } ^ { * } < 1$ satisfied $\Lambda \left( \exists t : \ D ( t ) < \right.$ −bandwidth—a single crossing loses, rather than a gradual lagging behind.

The evidence family for cost versus scope: measured audit costs are $K \propto M ^ { b } \left( b \in ( 0 , 1 )\right.$ , power law) with step-wise jumps, none giving log M. The $\log _ { 2 } ( 1 + M )$ of this paper is a theoretical prediction (coverage requires $\log _ { 2 } ( 1 + M )$ bits) and will underestimate cost at large M. Hence $\rho = \bar { V } / K \propto \log M / M ^ { b } \to 0$ : human audit productivity declines with scope, and supervisory elasticity in the scope direction is negative. What the literature measures is diminishing marginal detection eficiency (audit-cost empirics: Hay et al., 2006; Simunic, 1980; Porter et al., 1997; Raz & Yaung, 1997; Kantorowitz et al., 1997; increasing discovery dificulty: Maillart et al., 2017; Böhme & Falk, 2020; marginal superlinearity: Bifl & Halling, 2003; Vitharana, 2017; limits of AI auditing: McAleese et al., 2024; Kumar, 2026; Tufano et al., 2025; Klishevich et al., 2025; questionable validity of coverage: Mockus et al., 2009; Wei, Meyer & Oriol, 2012; functional-safety cost: Xie et al., 2018); it takes a two-step bridge (marginal diminishing→convex investment $+ \ { \bar { V } } \propto \log M )$ to obtain supervisory elasticity $< 0 ;$ the literature does not directly prove that $\rho$ declines with scope.

## C Coupled-Matrix Spectral Model

Model: $x _ { t + 1 } = \phi ( J x _ { t } ) , \phi =$ tanh; the gate divides nodes into covered $C \left( \left| x _ { C } \right| \leq c \right)$ and uncovered $U ;$ the uncovered deviation is $\delta _ { t + 1 } = J _ { U U } \delta _ { t } + J _ { U C } x _ { C }$ . Notation: $g = \rho ( J ) , s = \rho ( J _ { U U } )$

Key propositions: E1 (capability side) $m = \phi ( g m )$ , a pitchfork bifurcation at $g > 1$ , with $m ^ { 2 } \approx 3 ( g - 1 ) / g ^ { 3 }$ under tanh; E2 (permission-side margin) for $J \geq 0 , s < 1 , \| \delta \| _ { \infty } \leq c \| R J _ { U C } \mathbf { 1 } \| _ { \infty } \quad$ and unbounded for $s \geq 1 ; \mathrm { { E 3 } }$ (monotone independence) $0 \leq s \leq g _ { \cdot }$ , so $g > 1$ implies neither $s < 1$ nor $s > 1$ ; E4 (Q criterion) $Q = ( g - 1 ) ( 1 - s ) > 0 \iff g > 1$ and $s < 1$ , with $S = J _ { C C } + J _ { C U } R J _ { U C }$ the transitional form; E5 (loop threshold) unidirectional coupling of two subcritical modules does not change g, and the loop gain $\Lambda = w ^ { 2 } R _ { A } [ a , a ] R _ { B } [ b , b ] > 1$ of the closed loop is required for $g > 1$ ; E7 (plasticity budget) a budget $< 1 / r _ { \mathrm { m a x } }$ on the total weight of new incoming edges in $U$ is necessary and suficient for “the back door still holds after new connections.”

Four tests (m passes all): G1 $( N = 2 0 0 / 8 0 0 / 3 2 0 0$ nearly coincide), G2 (probe independently estimates ${ \hat { g } } ,$ difering by $< 1 0 ^ { - 4 } )$ ), G3b (at $g = 1 , m ( t ) \sim t ^ { - 1 / 2 }$ , measured −0.4985), G4 $( \beta = 0 . 4 9 8 4 , 1 / \delta = 0 . 3 3 3 2 )$

Signed $( \operatorname { E / I } ) \colon s \leq g$ does not hold; the always-correct criterion is $Q _ { \operatorname* { m i n } } = \operatorname* { m i n } ( g - 1 , 1 - s ) > 0 ;$ $g < 1 < s$ is a new failure mode (the whole network stable but clamping C severs the inhibitory loop of the stable $U _ { : }$ the gate itself causing instability; 10 cases out of 4000).

Three real systems: the ANDES Kundur power grid (19× improvement after fixing probe error, reliable admission at $\times 1 / \times 1 0$ , at $\times 2 0 / \times 4 0$ within measurement error of the critical point); the F-TRACT brain atlas $( s / g = 1 - \kappa _ { C }$ holds within 3%, the window depending only on coverage fraction); a trained RNN (g crosses 1 first, loss drops afterward; $s < 1$ does not guarantee the function is preserved).

The G3b non-equilibrium dynamical scaling has a real-system reference (avalanche cascades in deep networks are in the quasi-critical regime, Ghavasieh et al., 2025) and a theoretical motivation (the critical-slowing tendency of perceptual systems, Friston, Breakspear & Deco, 2012).

Limitations: all validation is under random matrices + mean field + small-amplitude linearization, and does not touch real industrial systems.

## D Structural Lower Bounds for the Capability-Side Witness

Three parallel sub-propositions: A (degree-bounded, computational reachability, Lu, 2026) under a fixed oracle A, all total decision procedures produced by efective internal self-modification halt within $\mathcal { C } ( A ) = \{ D : D \leq _ { T } A \}$ , and a strict ascent $A < _ { T } B$ requires expanding the oracle; B (verification undecidable, Gumbau Mezquita, 2026a, 2026b) for a Turing-complete self-modifying system there is no verifier that is sound, complete, and decidable; C (information non-increase, Finzi et al., 2026) $I ( M ; Y ) \leq I ( M ; D , B )$ : in a closed loop, if the “what counts as correct” signal is produced entirely by the system itself, the mutual information about the external correctness standard does not exceed the initial input.

Joint conclusion: a qualifying witness requires “no oracle expansion + fully sound selfjudgment + information gain,” and the three are respectively blocked by $\mathrm { A / B / C }$ . Hence the $\ S 8 . 2$ statement that “the capability-side irremediability is a structural conclusion” is a structural upper bound, not “capability cannot grow”; the list of recorded self-improvement loops is empirical corroboration, not the whole basis.

## E Reuse Monotonicity and Unimodality

Setup: $F ( S ; m ) = w _ { A } ( 1 - e ^ { - \Gamma _ { A } ( m ) S } ) + w _ { B } ( 1 - e ^ { - \Gamma _ { B } ( m ) S } ) , \Gamma _ { A } ( m ) = ( n - m ) \eta _ { 1 } + m \eta _ { 2 } , \Gamma _ { B } ( m ) = 0$ mη<sub>2</sub>, $q : = \eta _ { 2 } / \eta _ { 1 } , q ^ { * } : = w _ { A } / ( w _ { A } + w _ { B } )$

Theorem 1 $( q \geq q ^ { * }$ reuse to the top): $\partial { \cal F } / \partial m \geq 0$ for all $S > 0$ , so $S ^ { * } ( m )$ is non-decreasing and $m ^ { * } = n$ . Theorem $2 \ ( q < q ^ { * }$ full reuse necessarily non-optimal): at $m = n , \partial F / \partial m < 0$ , so $\textstyle m ^ { * } \neq n ;$ the upper boundary is always $q ^ { * }$ , and does not move with the degree of supercriticality. Theorem 3 (closed form of the lower boundary): at $m = 0 , S ^ { * }$ rises if and only if $q > q _ { l o } : =$ $\begin{array} { r } { \frac { w _ { A } u } { w _ { B } + w _ { A } u } , u : = e ^ { - n \eta _ { 1 } S ^ { * } ( 0 ) } = 1 - Q _ { A } ( S ^ { * } ( 0 ) ) \ ( Q _ { A } } \end{array}$ the coverage of group A, see $\ S 3 )$ . There is a band $( q _ { l o } , q ^ { * } )$ ; as $u  1 \ ( \Gamma  1 ^ { + } )$ the band collapses to a point.

Unimodality (quasiconcavity route, $\kappa \leq 1 )$ : the superlevel set $\{ S ^ { * } \ge s \} = \{ w _ { A } \varphi ( \Gamma _ { A } ) + $ $w _ { B } \varphi ( \Gamma _ { B } ) \leq w _ { A } + w _ { B } - s \}$ , with $\varphi ( x ) = \exp ( - s x - \frac { \kappa } { 2 } s ^ { 2 } x ^ { 2 } )$ convex when $\kappa \leq 1$ , so the superlevel set is convex, and $S ^ { * } ( m )$ is strictly quasiconcave, unimodal, with a unique interior maximum. When $\mathcal { K } > 1 , \varphi$ is concave near the origin, the convexity argument breaks down, and this is the same criticality as the $\ S 4$ bistability. Numerical check: $q _ { l o }$ formula vs numerical bisection (n = 200, 32 grid points), maximum error $9 \times 1 0 ^ { - 4 }$

## F List of Falsification Conditions (F1–F18)

Each substantive claim carries its own refutation condition, grouped by “judgeable now / requires building a measurement first.”

## Group I (judgeable now)

Table 4: Group I falsification conditions (judgeable now).
<table><tr><td>#</td><td></td><td>Location If observed</td><td>Consequence</td></tr><tr><td>F8</td><td>§6.4</td><td> $\alpha > 0 ,$ </td><td>X passes, but de- the ε three-term decomposition is ployment fails and does incomplete (a fourth class is miss- ing)</td></tr><tr><td></td><td>F9 §6.2</td><td>not fall under rep layer (ii) and layer (i)</td><td>rep has equal variance at the reproducibility-layer thesis is wrong</td></tr><tr><td></td><td>F11 §3</td><td>exist or is unstable</td><td>Γ &gt; 1 but S* does not the concavity of F or F(0) = 0 is violated</td></tr><tr><td>F12 §3</td><td></td><td>details while  $\{ \Gamma _ { g } \}$  is un- changed</td><td>S* changes with network coverage closure is refuted</td></tr><tr><td>F13 §3</td><td></td><td>ther 1 nor 1/2</td><td>measured exponent is nei- neither family is correct</td></tr><tr><td>F14 §3</td><td></td><td>same Γ, different net- same as above works, S* differs by &gt; 10%</td><td></td></tr><tr><td>F16 §7</td><td></td><td>plateau value drifts G3 fails monotonically with P by ≥ 0.03</td><td></td></tr><tr><td>F17 §7</td><td></td><td>0.02/decade</td><td>plateau slope CI excludes saturation judgment is wrong</td></tr><tr><td>F18 §7</td><td></td><td>growth-law coefficients G2 fails drift with N</td><td></td></tr></table>

## Group II (requires building a measurement first)

## G License-Free Measurements on the Gate Side

Formal reviews of 60 merged PRs each from four major open-source projects (kubernetes, tensorflow, rust, pytorch): mean number of independent reviewers about 1.3 (0.73–1.87), far below the 8–13 required for a 5% miss rate, $\hat { \rho } ^ { * } \approx 4 . 5 \sim 9 . 8 \gg 1$ . Definition: independent reviewer count = number of unique users submitting a formal review; limitations (detection-rate definition, review efort, bot contamination, heterogeneity) are detailed in the License-Free Gate-Side Measurement Report.

## Acknowledgements

We thank the relevant colleagues for the feedback and suggestions provided during the discussion of the framework. AI tools were used for English translation and language polishing during the

Table 5: Group II falsification conditions (requires building a measurement first).
<table><tr><td>#</td><td>Location If observed</td><td></td><td>Consequence</td></tr><tr><td>F1</td><td>§4</td><td> $\rho _ { \mathrm { h o l d } } ^ { * } > 1$  the long run</td><td>yet c stable in the three-regime criterion fails</td></tr><tr><td>F2 §4</td><td></td><td>collapse without hystere- two branches: sis</td><td> $\gamma _ { 0 } < \gamma _ { c }$  (this paper still holds) or the erosion term does not exist</td></tr><tr><td>F3 §4</td><td></td><td>K significantly inconsis- tent with log(1 + M)</td><td> $M _ { \mathrm { m a x } }$  invalidated</td></tr><tr><td>F4 §4</td><td></td><td>portional to  $\gamma _ { 0 } \mathcal { V } _ { a } / ( \rho u )$ </td><td>hysteresis width not pro- the linear-erosion assumption is wrong</td></tr><tr><td></td><td>F5 Supp. B</td><td>supervisory zero but  $\nu _ { c } / \nu _ { a }$  does not fall</td><td>elasticity Theorem A wrong</td></tr><tr><td></td><td>F6 Supp. B</td><td>supervisory elasticity Theorems differs significantly but does not</td><td> $\mathrm { A } / \mathrm { C }$  invalidated</td></tr><tr><td></td><td></td><td> $M _ { \mathrm { m a x } }$  space does not raise strategy&quot; does not hold</td><td>F7 Supp. B narrowing the action “restricted interface = control</td></tr><tr><td>F10 §6.3</td><td></td><td>measured  $M _ { \mathrm { m a x } }$  verted  $S ^ { * }$  differs from ficient</td><td>passes G1-G4 but in- order-parameter admission is insuf-</td></tr><tr><td>F15 §4</td><td></td><td>measurement by  $> 1 0 \times$  decrease monotonically in M</td><td>K(M)/ log M does not human-audit supervisory elasticity  $< 0$  is wrong</td></tr></table>

## References

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. arXiv:1606.06565.

Anderson, P. W. (1972). More is diferent. Science, 177(4047), 393–396.

Ashby, W. R. (1956). An introduction to cybernetics. Chapman & Hall.

Barreiro, A. K., Kutz, J. N., & Shlizerman, E. (2017). Symmetries constrain dynamics in a family of balanced neural networks. Journal of Mathematical Neuroscience, 7, 10.

Baum, K., Binkyt˙e, R., & Jahn, F. (2026). Norms at a price: Why RL-based alignment can promise conditional compliance at best [Preprint]. arXiv:2609.07627.

Bazerman, M. H., Morgan, K. P., & Loewenstein, G. (1997). The impossibility of auditor independence. Sloan Management Review, 38 (4), 89–94.

Bedau, M. A. (1997). Weak emergence. Philosophical Perspectives, 11, 375–399.

Betancourt, J. M., & Machta, B. B. (2026). Thermal fluctuations set fundamental limits on ion channel function [Preprint]. arXiv:2604.03538.

Betley, J., Warncke, N., Sztyber-Betley, A., Tan, D., Bao, X., Soto, M., Srivastava, M., Labenz, N., & Evans, O. (2026). Training large language models on narrow tasks can lead to broad misalignment. Nature, 649, 584–589. https://doi.org/10.1038/s41586-025-09937-5

Bifl, S., & Halling, M. (2003). Investigating the defect detection efectiveness and cost benefit of nominal inspection teams. IEEE Transactions on Software Engineering, 29 (5), 385–397.

Böhme, M., & Falk, B. (2020). Fuzzing: On the exponential cost of vulnerability discovery. In Proceedings of the 28th ACM Joint Meeting on ESEC/FSE (pp. 713–724). ACM.

Bullmore, E., & Sporns, O. (2009). Complex brain networks: Graph theoretical analysis of structural and functional systems. Nature Reviews Neuroscience, 10(3), 186–198.

Burns, C., Izmailov, P., Kirchner, J. H., et al. (2023). Weak-to-strong generalization: Eliciting strong capabilities with weak supervision [Preprint]. arXiv:2312.09390.

Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4), 404–413.

Duan, Y., Chaté, H., Mahault, B., & Shi, X. (2026a). Coexistence phase of the two-dimensional Vicsek model [Preprint]. arXiv:2610.26561.

Duan, Y., Chaté, H., Mahault, B., & Shi, X. (2026b). Phase Diagram of the Two-Dimensional Vicsek Model [Preprint]. arXiv:2610.26569.

Du, Z., Zeng, A., Dong, Y., & Tang, J. (2024). Understanding emergent abilities of language models from the loss perspective. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://doi.org/10.52202/079017-1683

Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). From entropy to epiplexity: Rethinking information for computationally bounded intelligence [Preprint]. arXiv:2601.03220.

Friston, K. J. (1994). Functional and efective connectivity in neuroimaging: A synthesis. Human Brain Mapping, 2 (1-2), 56–78.

Friston, K., Breakspear, M., & Deco, G. (2012). Perception and self-organized instability. Frontiers in Computational Neuroscience, 6, 44.

Fugl, E., Olsen, L. L., Sasmal, S., Efthymiou-Tsironi, M., Nagda, G., Krizek, F., Casparis, L., & Vaitiek˙enas, S. (2026). Continuous tuning of quantum critical behavior in hybrid Josephson junction arrays [Preprint]. arXiv:2610.40104.

Ghavasieh, A., Vila-Miñana, M., Khurd, A., Beggs, J., Ortiz, G., & Fortunato, S. (2025). Toward a physics of deep learning and brains [Preprint]. arXiv:2509.22649.

Godlee, F., Gale, C. R., & Martyn, C. N. (1998). Efect on the quality of peer review of blinding reviewers and asking them to sign their reports: A randomized controlled trial. JAMA, 280(3), 237–240.

Gong, C., Bi, J., Yao, D., Liang, X., Xiang, C., & Guo, R. (2026). Evolutionary safety of recursive self-improving AI: Taxonomy, risk discovery, and evaluation [Preprint]. arXiv:2609.31186.

Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2023). AI control: Improving safety despite intentional subversion. arXiv:2312.06942.

Gumbau Mezquita, J. P. (2026a). Algorithmic unverifiability of safety for fixed and recursively self-improving systems [Preprint]. arXiv:2606.28639.

Gumbau Mezquita, J. P. (2026b). Rice’s theorem under self-modification: Elevation operators and a normal form [Preprint]. arXiv:2609.11326.

Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017). Neuroscience-inspired artificial intelligence. Neuron, 95(2), 245–258.

Hay, D. C., Knechel, W. R., & Wong, N. (2006). Audit fees: A meta-analysis of the efect of supply and demand attributes. Contemporary Accounting Research, 23 (1), 141–191.

Herkert, J. R., Borenstein, J., & Miller, K. (2020). The Boeing 737 MAX: Lessons for engineering ethics. Science and Engineering Ethics, 26(6), 2957–2974.

Hoel, E. P., Albantakis, L., & Tononi, G. (2013). Quantifying causal emergence shows that macro can beat micro. Proceedings of the National Academy of Sciences, 110 (49), 19790–19795.

Hong, N., & Hong, T. (2025). Evidence of phase transitions in small transformer-based language models [Preprint]. arXiv:2511.12768.

Jansma, A., & Hoel, E. (2025). Engineering emergence. arXiv:2510.02649.

Kantorowitz, E., Guttman, A., & Arzi, L. (1997). The performance of the N-fold requirement inspection method. Requirements Engineering, 2, 152–164.

Kaushik, P., Chaudhari, S., Vaidya, A., Chellappa, R., & Yuille, A. (2025). The universal weight subspace hypothesis [Preprint]. arXiv:2512.05117.

Klishevich, E., Denisov-Blanch, Y., Obstbaum, S., Ciobanu, I., & Kosinski, M. (2025). Measuring determinism in large language models for software code review [Preprint]. arXiv:2502.20747.

Kounios, J., & Beeman, M. (2009). The Aha! moment: The cognitive neuroscience of insight. Current Directions in Psychological Science, 18 (4), 210–216.

Kumar, D. (2026). SWE-PRBench: Benchmarking AI code review quality against pull request feedback [Preprint]. arXiv:2603.26130.

Landau, L. D. (1937). On the theory of phase transitions. Zhurnal Eksperimental’noi i Teoreticheskoi Fiziki, 7, 19–32.

Lang, H., Huang, F., & Li, Y. (2025). Debate helps weak-to-strong generalization. In Proceedings of the AAAI Conference on Artificial Intelligence (pp. 27410–27418). arXiv:2501.13124.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521 (7553), 436–444.

Leveson, N. G., & Turner, C. S. (1993). An investigation of the Therac-25 accidents. Computer, 26 (7), 18–41.

Lewes, G. H. (1875). Problems of life and mind: First series (Vol. 2). Trübner & Co.

Li, H., Ren, M., & Lu, S. (2026). The forward/inverse duality of controlled emergence: A computational design program from explanation to engineering. arXiv:2610.25630.

London, M., & Häusser, M. (2005). Dendritic computation. Annual Review of Neuroscience, 28, 503–532.

Lou, X., Lázaro-Gredilla, M., Dedieu, A., Wendelken, C., Lehrach, W., & Murphy, K. P. (2026). AutoHarness: Improving LLM agents by automatically synthesizing a code harness. In International Conference on Learning Representations (ICLR 2026). arXiv:2603.03329.

Lu, C.-P. (2026). A relative-computability theory of self-improving agents [Preprint]. arXiv:2605.27381.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems 30 (pp. 4765–4774).

Maillart, T., Zhao, M., Grossklags, J., & Chuang, J. (2017). Given enough eyeballs, all bugs are shallow? Revisiting Eric Raymond with bug bounty programs. Journal of Cybersecurity, 3 (2), 81–90. https://doi.org/10.1093/cybsec/tyx008

Maturana, H. R., & Varela, F. J. (1980). Autopoiesis and cognition: The realization of the living. D. Reidel.

McAleese, N., Pokorny, R. M., Uribe, J. F. C., Nitishinskaya, E., Trebacz, M., & Leike, J. (2024). LLM critics help catch LLM bugs [Preprint]. arXiv:2407.00215.

Mockus, A., Nagappan, N., & Dinh-Trong, T. T. (2009). Test coverage and post-verification defects: A multiple case study. In ESEM 2009. IEEE.

Onsager, L. (1944). Crystal statistics. I. A two-dimensional Ising model with an order-disorder transition. Physical Review, 65 (3-4), 117–149.

Orseau, L., & Armstrong, S. (2016). Safely interruptible agents. In Proceedings of the 32nd Conference on Uncertainty in Artificial Intelligence (pp. 557–566). AUAI Press.

Porter, A., Siy, H., Toman, C. A., & Votta, L. G. (1997). An experiment to assess the costbenefits of code inspections in large-scale software development. IEEE Transactions on Software Engineering, 23 (6), 329–346.

Raz, T., & Yaung, A. T. (1997). Factors afecting design inspection efectiveness in software development. Information and Software Technology, 39, 297–305.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135–1144).

Rumelhart, D. E., Hinton, G. E., & McClelland, J. L. (1986). A general framework for parallel distributed processing. In D. E. Rumelhart & J. L. McClelland (Eds.), Parallel Distributed Processing: Explorations in the Microstructure of Cognition (Vol. 1, pp. 45–76). MIT Press.

Schaefer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In Advances in Neural Information Processing Systems 36. arXiv:2304.15004.

Schroter, S., Black, N., Evans, S., Godlee, F., Osorio, L., & Smith, R. (2004). Efects of training on quality of peer review: randomised controlled trial. BMJ, 328 (7441), 673.

Simunic, D. A. (1980). The pricing of audit services: Theory and evidence. Journal of Accounting Research, 18 (1), 161–190.

Soares, N., Fallenstein, B., Yudkowsky, E., & Armstrong, S. (2015). Corrigibility. In AAAI Workshop on AI and Ethics (Technical Report WS-15-02). AAAI Press.

Sporns, O., Tononi, G., & Kötter, R. (2005). The human connectome: A structural description of the human brain. PLoS Computational Biology, 1 (4), e42.

Stanley, K. O. (2019). Why open-endedness matters. Artificial Life, 25 (3), 232–235.

Strogatz, S. H. (2015). Nonlinear dynamics and chaos: With applications to physics, biology, chemistry, and engineering (2nd ed.). Westview Press.

Thompson, N. C., Greenewald, K., Lee, K., & Manso, G. F. (2020). The computational limits of deep learning [Preprint]. arXiv:2007.05558.

Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5, 42.

Trachtenberg, J. T., Chen, B. E., Knott, G. W., Feng, G., Sanes, J. R., Welker, E., & Svoboda, K. (2002). Long-term in vivo imaging of experience-dependent synaptic plasticity in adult cortex. Nature, 420 (6917), 788–794.

Truong, X. K. (2026). Emergence via Phase Transitions: Mechanism Landscapes and Universal Convergence across Complex Systems [Preprint]. SSRN.

Tufano, R., Martin-Lopez, A., Tayeb, A., Dabić, O., Haiduc, S., & Bavota, G. (2025). Deep learning-based code reviews: A paradigm shift or a double-edged sword? In Proceedings of the IEEE/ACM 47th International Conference on Software Engineering (pp. 1640–1652). IEEE/ACM.

Vitharana, P. (2017). Defect propagation at the project-level: Results and a post-hoc analysis on inspection eficiency. Empirical Software Engineering, 22 (1), 57–79.

Watson, R. A., Levin, M., & Buckley, C. L. (2022). Design for an individual: Connectionist approaches to the evolutionary transitions in individuality. Frontiers in Ecology and Evolution, 10, 823588.

Wei, J., Tay, Y., Bommasani, R., Rafel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., & Fedus, W. (2022). Emergent abilities of large language models. Transactions on Machine Learning Research. arXiv:2206.07682.

Wei, Y., Meyer, B., & Oriol, M. (2012). Is branch coverage a good measure of testing efectiveness? In Empirical Software Engineering and Verification (pp. 194–212). Springer.

Xie, G., Chen, Y., Liu, Y., Li, R., & Li, K. (2018). Minimizing development cost with reliability goal for automotive functional safety during design phase. IEEE Transactions on Reliability, 67 (1).

Yang, J., & Choi, H. (2026). Nonlinear dynamics of random neural networks with second-order synaptic motifs [Preprint]. arXiv:2609.14251.

Zhu, S., Yu, T., Xu, T., Chen, H., Dustdar, S., Gigan, S., Gunduz, D., Hossain, E., Jin, Y., Lin, F., Liu, B., Wan, Z., Zhang, J., Zhao, Z., Zhu, W., Chen, Z., Durrani, T. S., Wang, H., Wu, J., Zhang, T., & Pan, Y. (2023). Intelligent computing: The latest advances, challenges, and future. Intelligent Computing, 2, 0006.