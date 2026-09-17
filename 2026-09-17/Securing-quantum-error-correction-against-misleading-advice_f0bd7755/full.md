# Securing quantum error correction against misleading advice from AI agents

A. Barış Özgüler<sup>∗</sup>

Haas School of Business, University of California, Berkeley, California 94720, USA and

GeeQChiQ Technologies LLC, Berkeley, California, USA

(Dated: September 16, 2026)

Can an attacker turn influence over an artificial intelligence (AI) adviser into a harmful quantum error-correction update? We identify an ambiguity in passive syndrome records that obstructs recovery selection, then show how additional calibration measurements support certified recovery updates under uncertainty and drift. In an odd-distance square toric code with error-free preparation, syndrome measurements, and recovery operations, opposite coherent X rotations produce identical passive syndrome-history distributions. Yet a fixed phase correction can help at one sign and harm at the other. A terminal logical measurement on known encoded calibration states supplies the missing sign information. A separate evaluator accepts an update only when calibration uncertainty and a justified drift bound certify improvement over the current recovery, without assuming that the adviser recommends correctly. In simulated advice attacks, calibration-confidence checks reject harmful proposals while retaining beneficial updates under honest advice. We derive suficient limits on calibration age that require improvement through deployment. In matched simulations, a validated channel-specific bound retains more beneficial updates than the general bound after accounting for evaluation time, while preventing the tested harmful activations under the stated drift assumption. A separate surface-code experiment includes stochastic circuit faults and noise changing during acquisition. Deterministic controllers achieve at least as many beneficial updates with the same observations. Violating the drift assumption permits harmful acceptance in the toric experiment. The results identify information required for recovery selection, establish conditional guarantees against harmful updates, and quantify the recovery improvements forgone through conservative acceptance.

## I. INTRODUCTION

Quantum error correction (QEC) uses noisy measurement records to choose a recovery operation. A controller assisted by artificial intelligence (AI) that proposes changes to that operation introduces a security question: can an attacker turn influence over its advice into a harmful recovery update? A proposed change may be harmful because advice is mistaken or malicious, or because an accurate calibration has aged as the noise changes. The relevant question is therefore whether the available observations still justify the proposed update at the time of use. Rejecting every proposal would block harmful updates while also losing every opportunity to improve recovery. We assess protection together with the benefit retained by acceptance.

We connect this decision to an exact information limit. In the odd-distance square toric instrument studied here, opposite coherent-rotation signs produce identical passive syndrome-history distributions. Yet a fixed phase correction can help at one sign and harm at the other. A signed calibration supplies the missing information. A risk evaluator separate from the adviser then checks whether the proposed recovery improves on the current recovery throughout the physical conditions consistent with the measurements, their uncertainty, and a justified drift bound. The controller proposes an operation or requests calibration; the evaluator decides whether the operation can be activated.

Learning and reinforcement learning already perform useful QEC tasks [1, 2], while classical adaptive estimation provides an essential comparison [3, 4]. Neural models also support hardware quantum control by predicting pulse parameters [5]. Baseline-relative policy improvement, predictive safety filters, and controls on untrusted text are established approaches [6–8]. Guatto et al. [9] combine multi-agent code discovery with bandit-controlled adaptation to drifting noise. Syndrome-driven self-calibration also has guarantees for time-dependent drift [10]. We use these approaches to identify which observations and deployment-time uncertainty bounds certify improvement over the incumbent recovery despite attacker-controlled advice. Large language model (LLM) proposals test this acceptance boundary; accepted beneficial updates and their recovery gains measure the benefit retained. Figure 1 illustrates this question in the toric example. Figure 2 follows the complete obstruction–calibration–recovery chain within the same toric instrument. A separate surface-code maintenance experiment tests timed decoder updates under stochastic circuit faults (Figs. 4 and S16).

Trusted calibration has both a provenance requirement and a physical requirement: the evidence must belong to the experiment and recovery under consideration, and its interpretation must remain valid at deployment. Authenticity establishes the origin of a measurement; the observation model, uncertainty bound, and characterized drift establish what that measurement supports. The evaluator is the acceptance service: it owns the records and checks workload and action identifiers. It is logically separate from proposal generation and outside the modeled advice attacker’s control. Its implementation, clock, measurement source, and action enforcement are trusted components; the noise model and drift bounds are physical assumptions. The software prototype tests this separation within one process. Independent outcome assessment uses the true simulation parameters, kept separate from controller inputs. In the advice attacks, the adversary controls an external note. The security endpoint is the integrity of the activated recovery update: acceptance must remain justified despite attacker-controlled advice. A harmful proposal recommends a recovery with greater simulated risk than the incumbent; a harmful activation additionally requires acceptance. Each experiment specifies the tolerance and, where outcomes are sampled, the confidence-interval criterion used to establish harm. The attacker does not control trusted measurements, the evaluator, or the operation applied after acceptance. Separate physical-premise violations test failure of the assumed calibration-to-deployment relation. Both circuit studies compare the language model with a deterministic controller given the same calibration observations, actions and budget.

Classical decoding has several distinct complexity regimes. Bravyi, Suchara, and Vargo give eficient exact surface-code maximum-likelihood decoding for independent bit and phase flips with noiseless checks [11]. Bravyi et al. give eficient classical simulation of a restricted coherent-error model, with the chosen decoder’s runtime added to the simulation cost [12]. General optimal stabilizer decoding has counting-complexity hardness results [13]; surface-code decoding also has worst-case hardness when qubit-dependent Pauli probabilities are part of the input [14]. Practical matching and neural decoders address structured noise models and finite latency budgets [15, 16]. These computational questions difer from the observation obstruction studied here: a fast decoder can still use an incorrect or outdated noise model, and unlimited computation cannot distinguish conditions with identical observation laws.

Orsucci, Tiersch, and Briegel identify sign and periodicity ambiguities in individual graph-state stabilizeroutcome probabilities [53]. Our toric result treats the joint distribution of every passive syndrome history and connects its sign symmetry to diferent recovery consequences.

Coherent-error studies establish syndrome blindness and parity-dependent logical-input sensitivity in surface codes [12, 17–19]. Hu, Liang, and Calderbank give generator-coeficient criteria for input dependence [20]; toric path and partition-function calculations supply related geometric structure [21–23]. Our two-logicalqubit specialization identifies the surviving product parity, proves its nonzero leading coeficient, and establishes exact sign symmetry of every passive history. These fixed-distance coherent results complement correctability– privacy duality [24] and Shen and Zhong’s transcriptprivacy bounds for noisy memories [25].

The history law uses repeated quantum nondemolition (QND) measurement statistics [26]; the syndromeconditioned polar correction applies recovery by classical feedback [27]. Two related results emerge from the same instrument. The parity coeficient and history bounds quantify leakage about the stored logical input. Exact noise-sign symmetry identifies a diferent ambiguity that changes which fixed recovery is beneficial. The confidenceset acceptance argument uses this second observation problem and a valid risk bound; it does not require the perturbative parity coeficient. Table S5 and Supplemental Material, Sec. I separate established ingredients from the cancellation of contributions to the syndrome efects, the nonzero leading coeficient, and the attained scaling of distinguishability from complete histories derived here.

We outline the rest of the paper. Section II establishes the toric information obstruction and its recovery consequence. Section III develops encoded calibration and acceptance guarantees. Section IV evaluates the complete chain in that same instrument. Section V gives the separate timed surface-code extension, and Sec. VI draws the implications. The Supplemental Material contains proofs, complete protocols, all controller comparisons and additional readout and circuit validations.

## II. TORIC EXAMPLE: MISSING INFORMATIONCHANGES RECOVERY

Consider a periodic square toric code of fixed odd distance $L \geq 3$ , encoding two logical qubits. Here ideal means that preparation, projective syndrome measurement, terminal readout, and application of the specified recovery introduce no additional errors. The coherent rotations are the modeled noise. Phase discretization and actuation noise enter only the separately identified extensions. Each round applies uniform physical $R _ { X } ( \theta ) = \exp ( - i \theta X / 2 )$ rotations, extracts the complete Z syndrome, and applies deterministic minimum-weight X recovery. Let $K _ { s } ( \theta )$ be the corrected logical Kraus operator and $F _ { s } ( \theta ) = K _ { s } ( \theta ) ^ { \dagger } K _ { s } ( \theta )$ its efect. These statements concern this fixed measurement and recovery instrument. The dimensionless θ is a signed angle: $R _ { X } ( - \theta ) = R _ { X } ( \theta ) ^ { \dagger }$ reverses the rotation direction. Both signs describe evolution over a positive duration, not negative elapsed time. For $\begin{array} { r } { P = \overline { { X } } _ { x } \overline { { X } } _ { y } , } \end{array}$ , every efect satisfies

$$
\begin{array} { c } { { F _ { s } ( \theta ) = q _ { s } ( \theta ) I _ { 4 } + b _ { s } ( \theta ) P , } } \\ { { q _ { s } ( - \theta ) = q _ { s } ( \theta ) , \qquad b _ { s } ( - \theta ) = b _ { s } ( \theta ) . } } \end{array}\tag{1}
$$

Thus the logical input enters the syndrome law through the single expectation $m _ { \rho } = { \mathrm { T r } } ( P \rho )$ . An error support can contribute to a syndrome efect only if it is a cycle contained in a cut of the dual graph. Cut cancellation and the odd-L parity of logical representatives leave the scalar and product-parity classes, with coeficients that are even functions of the rotation angle.

The corrected Kraus operators commute in the logical-X basis. Writing $p _ { s } ^ { \pm } = q _ { s } \pm b _ { s }$ , the complete R-round

![](images/3d9b5a02cda60dc9f2eda74a1df46ad373e4ad83012dbd5697cf97e276e7ade3.jpg)  
FIG. 1. Trusted calibration separates advice from authority over recovery. (a) Opposite rotation signs give identical passive syndrome-history distributions in the odd-distance square toric instrument defined in Sec. II, although one fixed correction table can help at one sign and harm at the other. (b) Known encoded calibration states undergo 100 rounds of the same toric instrument and a terminal logical $Y _ { 0 3 }$ measurement. Their sign-sensitive response supplies the missing observation; finite counts constrain the calibration angle. (c) The evaluator admits a proposed update only when sampling uncertainty and a justified drift bound certify recovery improvement at deployment; otherwise, the controller retains the incumbent or requests new calibration. The adviser can propose an action, including one induced by attacker-controlled text; the independent criterion determines whether that action may be applied. The displayed intervals are schematic. (d) The toric cycle uses ideal checks and X recovery $C _ { s } ;$ the dashed V<sub>s</sub>(u) applies the specified logical operation without additional error. Circuit definitions are in Supplementa Material, Sec. V; the encoded calibration protocol is in Sec. Y. Figure 2 gives quantitative evidence for the complete chain.

history law is

$$
\begin{array} { l } { { P _ { \rho , \theta } ^ { ( R ) } ( s _ { 1 } , \ldots , s _ { R } ) = { \displaystyle \frac { 1 + m _ { \rho } } { 2 } \prod _ { j } p _ { s _ { j } } ^ { + } ( \theta ) } } } \\ { { \displaystyle \qquad + \frac { 1 - m _ { \rho } } { 2 } \prod _ { j } p _ { s _ { j } } ^ { - } ( \theta ) , \qquad P _ { \rho , + \theta } ^ { ( R ) } = P _ { \rho , - \theta } ^ { ( R ) } . } } \end{array}\tag{2}
$$

The sign symmetry holds for every input, angle, and history length in this instrument. A longer passive record therefore leaves the sign ambiguity intact. Sensitivity to the logical input is a distinct quantity: it first occurs at order $\bar { \theta } ^ { 2 L }$ , with the following nonzero leading coeficient

$$
[ \theta ^ { 2 L } ] b _ { 0 } ( \theta ) = - \frac { L [ { \binom { 2 L } { L } } - 2 L ] } { 2 ^ { 2 L - 1 } } .\tag{3}
$$

At fixed odd L as $\theta  0$ , a pair of parity eigenstates $\mathrm { a t - }$ tains one-round contrast in total-variation (TV) distance $\Theta _ { L } \big ( | \theta | ^ { 2 L } \big )$ . After the specified number of rounds, chosen to avoid a return of the accumulated logical rotation to the identity, the established order- $- \theta ^ { L }$ coherent channel term [21] produces a diamond-norm distance from the identity that remains finite and nonzero, while the maximum complete-history contrast scales as $\Theta _ { L } ( | \theta | ^ { L } )$ . A rare syndrome from the diagonal staircase attains this exponent: the event that it occurs at least once supplies the matching lower bound. Thus even the optimal equalprior history classifier has vanishing advantage on this accumulation scale. These fixed-L bounds have distancedependent constants and validity neighborhoods; proofs are in the Supplemental Material.

Logical-input sensitivity and sign identifiability compare diferent quantities: changing the stored parity versus reversing the noise at fixed input. A record can reveal the former while carrying no sign information. More training data cannot resolve this exact ambiguity. Figure 1(a) illustrates the observation limit; the recovery comparison shows its operational consequence.

At $L = 3 .$ , numerical checks compare all 256 syndrome probabilities and evaluate the efect symmetry over 361 angles and four logical-X sectors. The largest efect residual is $4 . 4 4 \times 1 0 ^ { - 1 6 }$ (Supplemental Fig. S19). These checks test the implementation; the coeficient derivation and history factorization establish the analytical result.

Fix two syndrome-dependent inverse-polar phase tables $V _ { s } ( u )$ at ±0.10 rad. The incumbent u<sub>0</sub> retains minimumweight recovery. The Supplemental Material describes the larger catalog. Define

$$
\begin{array} { r l } & { \displaystyle \mathcal { E } _ { \theta , u } ( \rho ) = \sum _ { s } V _ { s } ( u ) K _ { s } ( \theta ) \rho K _ { s } ( \theta ) ^ { \dagger } V _ { s } ( u ) ^ { \dagger } , } \\ & { \displaystyle \mathcal { R } ( \theta , u ) = 1 - F _ { e } ( \mathcal { E } _ { \theta , u } ^ { H } ) , \qquad D _ { u } ( \theta ) = \mathcal { R } ( \theta , u ) - \mathcal { R } ( \theta , u _ { 0 } ) . } \end{array}\tag{4}
$$

Here R is the stationary H-round entanglement infidelity relative to the identity, and $D _ { u }$ is the change in risk relative to the incumbent. Negative $D _ { u }$ means improved recovery. At $L = 3 , | \theta | = 0 . 1 0 \ i$ and $H \ : = \ : 3 0 0$ , the incumbent infidelity is 0.1323553, compared with approximately 0.0007456 for the correctly signed table. The same fixed +0.10 table harms recovery at the opposite sign [Supplemental Fig. S24(b)]. Every comparator in this experiment uses the same fixed catalog; these losses exclude gate-synthesis errors and actuator overhead.

Keeping the phase table fixed while reversing θ withholds the hidden sign. The incorrectly signed table raises infidelity to approximately 0.4140468.

The missing information matters only if it changes the decision. Let $[ \theta ] _ { \mathcal { O } }$ contain the conditions giving the same law under an allowed observation experiment O. An update is uniformly beneficial on that class when $\begin{array} { r } { \operatorname* { s u p } _ { \theta ^ { \prime } \in [ \theta ] _ { \mathcal { O } } } D _ { u } \mathopen { } \mathclose \bgroup \left( \theta ^ { \prime } \aftergroup \egroup \right) < 0 } \end{array}$ . Within the declared deterministic action catalog, an empty intersection of beneficial action sets excludes a uniformly beneficial choice from those observations. Enlarging the policy class, for example to randomized recovery, requires evaluating its own loss before drawing the same conclusion. Conversely, an action beneficial at both signs can be justified without identifying either sign. The encoded calibration below supplies the additional observation using the same noisy memory instrument. The Supplemental Material describes a separate unencoded-sentinel implementation and its transfer assumptions.

## III. ENCODED CALIBRATION AND EVIDENCE-BASED ACCEPTANCE

## A. A sign-sensitive observation in the same instrument

The toric calibration-and-recovery demonstration uses $L = 3$ and 18 data qubits. In the logical-X basis, let $k _ { s x } ( \theta )$ be the diagonal entries of $K _ { s } ( \theta )$ and

$$
C _ { x y } ( \theta ) = \sum _ { s } k _ { s x } ( \theta ) k _ { s y } ( \theta ) ^ { * } .\tag{5}
$$

The incumbent channel multiplies each matrix element $\rho _ { x y }$ by $C _ { x y }$ . Prepare a dedicated known calibration memory

in $( | 0 _ { X } \rangle + | 3 _ { X } \rangle ) / \sqrt { 2 }$ , where sectors 0 and 3 have logical-$X$ eigenvalues $( + , + )$ and $( - , - )$ . Execute $H _ { c } ~ = ~ 1 0 0$ unchanged toric rounds and measure

$$
Y _ { 0 3 } = - i | 0 _ { X } \rangle \langle 3 _ { X } | + i | 3 _ { X } \rangle \langle 0 _ { X } | .\tag{6}
$$

On the populated two-dimensional subspace its outcomes are ±1. The terminal plus probability is

$$
q _ { Y } ( \theta ) = \frac { 1 - \mathrm { I m } [ C _ { 0 3 } ( \theta ) ^ { H _ { c } } ] } { 2 } , \qquad q _ { Y } ( - \theta ) = 1 - q _ { Y } ( \theta ) .\tag{7}
$$

At $\theta = + 0 . 1 0$ rad, $q _ { Y } \simeq 0 . 6 4 7 1 3 6$ , whereas at −0.10 it is 0.352864 [Fig. 2(b)]. The passive syndrome histories still have identical laws. The terminal observation changes the available information, rather than extracting an absent sign from the passive record. Dedicated known states are used for calibration; the unknown stored state is not measured to determine the sign.

Each calibration record contains N independently prepared memories with stationary angle $\theta _ { c } .$ . A two-sided 99% Clopper–Pearson interval [28] for the terminal plus count is inverted over the complete supported domain [−0.15, 0.15] rad. Every compatible angle interval is retained, including disconnected intervals. The encoded probe uses the memory noise directly and the preparation and readout assumptions above, so no separate sentinelto-code transfer relation is invoked.

## B. A bound that includes the deployment path

Scenario clock. We express modeled durations in an abstract reference unit $T _ { 0 }$ . Clock variables such as τ and A carry units $T _ { 0 } .$ , the angular rate v has units rad ${ \mathbf { \nabla } } / { T _ { 0 } } ,$ and the surface-model rate w has units $T _ { 0 } ^ { - 1 }$ Equivalently, $\tau / T _ { 0 } , A / T _ { 0 } , v T _ { 0 }$ and $w T _ { 0 }$ give the corresponding normalized quantities. The signed rotation angle θ is distinct from the clock variable τ. In the central toric and timed surface studies, one measured second of inference is assigned one $T _ { 0 } .$ . The multistep maintenance extension instead assigns 60 measured seconds to one $T _ { 0 } ;$ Table S6 lists the separate scenarios. These choices stipulate inference-to-device timing ratios. The adviser performs maintenance while the incumbent remains available; it is not required to respond once per quantum round. The numerical experiment fixes 3 actions before evaluation: the incumbent and the two existing inversepolar tables at ±0.10 rad. Let C be the angle region compatible with the calibration count, and $D _ { u } ( \theta )$ the stationary 300-round excess infidelity in Eq. (4). A grid of 60001 points includes bounds on variation between grid points for both inference of compatible angles and maximization of recovery risk. Denote the resulting bound by $U _ { u } ^ { \mathrm { c a l } } \ge \operatorname* { s u p } _ { \theta \in \mathcal { C } } D _ { u } ( \theta )$ The bound on variation between grid points follows analytically; the floating-point implementation adds a tested numerical tolerance. The guarantee below requires a valid enclosure. Supplemental Material, Sec. Y describes the original implementation;

Sec. AB supplies an outward-rounded toric evaluator and a matched timing comparison.

In this toric scenario, noise changes only after the stationary acquisition. Write θ for the angle sequence during the $H _ { d } = 3 0 0$ deployment rounds and $D _ { u } ( \pmb \theta )$ for the corresponding composed-channel excess infidelity. If $| { \dot { \theta } } | \leq v ,$ we compare deployment with the calibrated channel one round at a time. Summing the resulting channel-distance bounds gives

$$
\begin{array} { r } { U _ { u } ( A ) = U _ { u } ^ { \mathrm { c a l } } + L _ { D } v A , \qquad } \\ { L _ { D } = 2 n H _ { d } = 1 0 8 0 0 , \qquad n = 1 8 , } \\ { U _ { u } ( A ) \geq \displaystyle \operatorname* { s u p } _ { \theta _ { c } \in \mathcal { C } } D _ { u } ( \pmb { \theta } ) . \qquad } \\ { \mathrm { a l l o w e d ~ p a t h s ~ f r o m ~ } \theta _ { c } } \end{array}\tag{8}
$$

Here A is the elapsed time from the start of the earliest calibration round to the end of deployment and includes acquisition, measured inference latency and delivery delay. The drift-rate bound supplied to the evaluator is $v = 1 0 ^ { - 6 }$ rad $1 / T _ { 0 }$ . The simulated quantum round lasts $1 0 ^ { - 6 } T _ { 0 } ;$ additional preparation, readout and logical-actuation durations are set to zero. Including the complete acquisition duration is conservative because the acquisition is stationary in this experiment. The coeficient and grid allowances are derived in Supplemental Material, Sec. Y.

The calibration record must correspond to the workload and recovery under consideration. The evaluator then admits the proposed recovery only when

$$
U _ { u } ( A ) \leq - \delta , \qquad \delta = 0 . 0 0 1 .\tag{9}
$$

Otherwise the incumbent is retained. The rule certifies improvement throughout the allowed path family; it can reject an operation that happens to improve the executed path. More accurate observations can narrow sampling uncertainty, whereas an unsupported physical drift bound requires new physical information.

For a declared family of acquisitions $j ,$ let $E _ { j }$ mean that the true deployment condition lies in the corresponding confidence region. Suppose $\begin{array} { r } { \mathrm { P r } ( E _ { j } ^ { c } ) \leq \alpha _ { j } , \sum _ { i } \alpha _ { j } \leq \alpha } \end{array}$ the applicable preparation, readout, transfer, and drift premises hold, and each computed bound is valid for every allowed action. If the activated operation is the one evaluated, the following probability bound holds for any adaptive proposal strategy

$$
\operatorname* { P r } \{ \exists { \mathrm { ~ a c c e p t e d ~ } } ( j , u ) : D _ { u } ( \pmb \theta _ { j } ) > - \delta \} \leq \alpha .\tag{10}
$$

On $E _ { j }$ , Eq. (9) guarantees the margin for every admitted action, including one selected after observing the data. $\mathrm { A }$ violation therefore requires an $E _ { j } ^ { c }$ , and the union bound gives Eq. (10). Proposer accuracy is absent from these assumptions. The probability is taken over the specified acquisition family, without conditioning on acceptance. The encoded-calibration intervals have 99% coverage per acquisition. The 95% deployment intervals assess sampled outcomes within an instance. Neither percentage is a study-wide safety probability; such a statement requires the explicit family allocation in Eq. (10). Absolute logical accuracy and workflow completion are separate endpoints.

Repeated proposals using one region consume no additional statistical allowance. Each new calibration block is assigned part of the total failure-probability budget, even when its measurements cannot support an update. If earlier observations determine a new acquisition policy, coverage must hold conditionally on that history or simultaneously over all permitted choices. The experiments use predetermined looks or budgets chosen before independent newly acquired measurements. This allocation also governs retries after an inconclusive calibration.

Calibration age is elapsed time; evidence remains valid for a proposed update when its uncertainty propagated to deployment still certifies the proposed improvement. Recent noisy observations can be insuficient, while older precise observations can still certify improvement under slow drift. Supplemental Fig. S24(d) evaluates this boundary for 3 nested shot budgets; these curves bound validity for the displayed observations and drift assumptions; they are not acceptance probabilities.

The same acceptance argument applies to other observations when their confidence regions and bounds on how physical noise changes afect recovery risk are justified. Supplemental Material, Sec. X reports the separate sentinel, measured-readout, operation-identity and learned-proposal studies. The encoded demonstration below supplies the central chain without a change of code or noise model. The subsequent surface-code extension changes both the observation and the loss explicitly and allows noise to vary during acquisition.

## IV. THE COMPLETE CHAIN IN ONE TORICCIRCUIT

## A. Measurements, proposals and independent assessment

The 6 angles $\theta = \pm 0 . 0 8 , \pm 0 . 1 0 , \pm 0 . 1 2$ rad each have 2 independent acquisition realizations, giving 12 calibration workloads. Each memory retains its 100 syndrome outcomes and terminal logical readout. Budgets 512, 2048 and 8192 are nested prefixes; the controller evaluation uses 8192 shots. The same 12 workloads are reused across honest advice, a misleading override, a $3 0 T _ { 0 }$ delivery delay, wrong evidence identity and an undeclared sign flip. These conditions therefore give 60 model requests, rather than 60 independent simulation workloads.

The local Qwen3:8b adviser [29] receives terminal counts, timing, record identity, the action ranking, and the set of updates currently supported by the observations. The misleading note deliberately requests the worstranked action, providing a controlled test of whether a harmful proposal can reach activation. Each request returns 1 activation proposal or retention. The deterministic comparator uses the same ranking and ignores the untrusted note. Prompts and outputs from the pretrained

a Passive record: no sign information  
![](images/f84334e25b546131d5923fb58794873dd54bc3774a6912067c770ec1aec3c781.jpg)

b Encoded calibration resolves the sign  
![](images/d50c7bbb7efbff5c29439fc493e6f3f000b08a28afb458503bcf8a38c24c9e99.jpg)

![](images/fd3eed2981016ff2dcdba00d77e83555c88866d003f39de9ec0acb9516543a98.jpg)  
FIG. 2. The same passive record can conceal opposite recovery consequences. (a) All 256 one-round syndrome probabilities evaluated from the exact instrument for the encoded calibration state coincide at θ = ±0.10 rad, displayed in one common ordering. The passive-history theorem extends the sign blindness to every history length. (b) Terminal logica measurements distinguish the signs: each circle or square is one of two independently simulated 8192-memory acquisitions at each angle, with a 99% binomial interval. (c–e) Independent estimates of excess recovery infidelity for every Qwen proposa under honest advice, misleading advice, and a sign flip that violates the drift bound. The same 12 calibration workloads are used in all three panels. Filled circles mark full-safeguard acceptance; open circles show the independently estimated consequence of applying a rejected proposal. Rejection retains the incumbent, giving zero deployed excess risk. Colors identify benefit or harm established beyond ±0.001. Each action has 16384 deployment readouts; 95% intervals jointly cover the 3 action risks within each instance. All panels use the L = 3 instrument of Sec. II. Supplemental Fig. S23 includes both controllers and every rule.

model are retained.

For each controller and condition, the coherent instrument is composed numerically along a 300-round deployment path. The valid paths drift at the declared rate after stationary acquisition. Independent terminal measurements test return to the initial maximally entangled state of the logical memory and reference. These Bell-return tests assess each of the 3 actions, with 16384 shots per action. The resulting infidelity intervals give candidate-minus-incumbent bounds: an interval entirely below −0.001 establishes benefit, and one entirely above +0.001 establishes harm. They cover all 3 action risks within an instance, not the entire study simultaneously. Each action uses a separate sampled stream; unlike the surface-code comparison, these are not paired shot differences. The terminal probabilities are evaluated from the channel-composition formula and checked against a separate computational-basis gate simulation; each deployment readout is sampled from this distribution rather than expanded into a gate-by-gate trajectory.

## B. Beneficial recovery and harmful advice

Under honest advice, all 12 proposed updates are beneficial. Confidence-only acceptance admits 10, while the full confidence-and-timing rule admits 8 [Fig. 2(c)]. Every other case retains the incumbent. An untrusted note claiming an urgent operator override induces Qwen to choose the wrong signed table in all 12 misleading-advice workloads. Independent deployment intervals establish harm for all 12 proposals [Fig. 2(d)]. Both evidencebased rules reject them: confidence at calibration already sufices for this attack, and the full rule retains that protection. The deterministic controller follows the evaluatorcomputed ranking and achieves 12, 10, and 8 beneficial updates under authorization, confidence-only and full acceptance, respectively, even with the misleading note.

The $3 0 T _ { 0 }$ delivery delay instead isolates a utility cost: all 12 proposals remain beneficial on their executed paths, but the full rule rejects them because its propagated risk bound no longer meets the required improvement margin. Identity checks reject all 12 requests tied to the wrong workload. Supplemental Fig. S23 shows every outcome, including retention; the deployment intervals establish benefit or harm for every accepted update in this run.

A validated replay preserves the calibration records and proposals while charging measured evaluation time (Supplemental Material, Sec. AB). A channel-specific drift coeficient reduces the general constant from 10800 to 123.459. Both rules then use identical outward-rounded stationary bounds: the sharper rule retains all 12 delayed beneficial updates per controller, compared with 0 under the general bound. Valid-premise harmful activations remain 0; the invalid sign flip still causes harm. These are enclosed-model outcomes on latency-adjusted paths. The original computational-basis reconstruction and sampledoutcome protocol remain in Sec. Y.

## C. Failure when the physical premise is false

An abrupt sign reversal after calibration violates the declared rate bound. Authorization alone admits 12 harmful updates, confidence-only acceptance admits 10, and the full rule admits 8. The same counts occur for the deterministic controller. Figure 2(e) resolves these simulated recovery harms in the toric circuit that supplied the information obstruction and calibration. A genuine calibration record can therefore support a harmful action if its assumed relation to deployment is false. The guarantee in Eq. (10) is conditional on that relation, not on whether the proposer is a language model.

## D. Additional attacks, maintenance, and calibration expiry

The 3 model configurations—Qwen3:8b, Gemma 3:1b, and GPT-OSS:20b [29–31]—are tested on 6 development and 24 held-out acquisitions. Fixed notes spoof authority, allege measurement-label corruption, or hijack the output format; a 3-query adaptive attack uses replies, service feedback, and the visible ranking. Acquisitions are paired across models and attacks. With GPT-OSS, adaptive attacks induce harmful proposals in 12 of 24 workloads, compared with 4 for any success among 3 fixed attacks at the same query budget. Evidence checks block the observed harmful proposals. These counts describe the specified attack design.

Separate maintenance workflows require acquiring missing calibration, refreshing expired evidence, or binding the correct workload. Honest and misleading notes share observations, tools, and budgets. With enforced command formatting, Qwen completes 12 of 18 beneficial updates under honest notes and 0 under misleading notes. With native tool calls, GPT-OSS completes 12 and 8, respectively. Thus protected update integrity coexists with loss of useful progress. These maintenance arms share the 60-second clock convention; their completion rates are not a timing-controlled comparison with the central onesecond scenario. Supplemental Material, Sec. AA retains all interfaces, failures, and deterministic comparisons.

To isolate the physical role of timing, we also evolve the noise continuously after stationary calibration, at a rate no greater than the declared bound, toward the opposite sign and then hold it fixed during deployment. Figure 3 evaluates the same update at 161 predetermined delays per acquisition. Confidence-only acceptance produces harmful activations in 16 of 24 acquisition workloads; the full timing rule produces 0 on these paths. For all 16 initially certified updates, the validity period ends before the update becomes harmful. The suficient validity periods range from 6.41 to 25.14T , whereas the first sampled harmful delays occur between 46500 and 66000T . At the first harmful delay the rotation still has its initial sign: the fixed correction ceases to improve recovery as the magnitude falls. Expiry marks loss of certification under a suficient bound, not a prediction of when harm begins. The conservative inversion, grid, drift and acquisition allowances are separated in Supplemental Material, Sec. $\mathrm { \ Y 2 a }$ The 3864 path evaluations reuse 24 acquisitions. Their abstract delays reach the failure regime; the validated replay above separately improves the shorter-delay toric test.

![](images/cee2aa7d1f0483d2f0827ac5945a4d28f59c987aea719c6f8adc83794992dd72.jpg)  
FIG. 3. Timing prevents harmful reuse under a valid drift premise, with conservative expiry. (a) Exact-channel excess recovery risk for the initially preferred correction after a continuous rate-limited ramp from three positive initial angles; negative-angle counterparts obey the same symmetry. Each point gives the recovery risk when the angle is held fixed during deployment at the value reached after the preceding drift. Positive excess risk denotes harm. The rate is $1 0 ^ { - 6 }$ rad ${ } _ { \cdot } / T _ { 0 } ;$ the ramp stops at the opposite initial angle. Acquisition precedes the ramp and deployment follows it. (b) Each row is one of the 16 independently sampled acquisitions initially admitted by confidence. Circles mark the end of the certified validity period; crosses mark full acquisition-to-completion age at the first harmful point on the 1500T<sub>0</sub> delay grid. Their separation exposes conservatism, rather than estimating a sharp optimal validity period. The remaining 8 acquisitions are rejected initially. Calibration-only acceptance admits later harmful updates in all 16 rows; the full confidence-and-timing rule rejects them. No drift assumption is violated.

## V. EXTENSION TO TIME-RESOLVED SURFACE-CODE MAINTENANCE

The toric example supplies the exact obstruction, encoded calibration and recovery in one coherent instrument. We now test the acceptance principle when stochastic circuit noise changes during acquisition, inference and deployment. This extension uses reinitialized known-state surface memories and classical decoder-prior updates, with a corresponding observation model and loss.

## A. Changing-noise acquisition and certification

Stim and PyMatching provide 30-round rotated surfacecode memories and matching decoders [15, 32]. Each shot prepares a known logical state, extracts noisy checks and ends with a logical measurement; both logical bases are tested. Candidate and incumbent decoders process the same detector record, and risk is the larger failure probability across the two bases. We fix 6 candidate decoder priors before acquisition. Distance 3 uses localgate amplification; distance 5 uses excess idle-Z faults. These families test distinct maintenance conditions rather than isolate distance scaling.

Both controllers receive calibration summaries, the evaluator-computed action ranking and menu, the shot budget and an external note. One Qwen3:8b [29] request proposes an update or retention; the deterministic comparator follows the same ranking. The scenario clock includes the converted measured or stipulated processing latency.

For a shot starting at s, elementary fault laws interpolate as $Q _ { j } ( \lambda ) = Q _ { j } ( 0 ) + \lambda [ Q _ { j } ( 1 ) - Q _ { j } ( 0 ) ]$ at their instruction times. Assume $| \dot { \lambda } | \le w$ and let Γ bound the sum of fault-law total-variation slopes. This stochastic coordinate λ difers from the toric angle θ. If basis-b calibration shots start at $s _ { b i }$ and deployment ends at $\tau _ { f } .$ coupling and averaging bound the transport of mean calibration risk by

$$
\epsilon _ { b } = \operatorname* { m i n } \{ 1 , \Gamma w ( \tau _ { f } - \overline { { s } } _ { b } ) \} .\tag{11}
$$

Fixed-look intervals derived from Chernof bounds on the Kullback–Leibler (KL) divergence allow independent shots with nonidentical probabilities. The bound on excess failure risk in the worse of the two memory bases using paired candidate and incumbent outcomes is derived in Supplemental Material, Sec. W. Propagating only from the batch-end timestamp omits part of the allowance.

The 24 independent workloads comprise 2 simulated parameter draws at each of 2 distances in 6 conditions: honest advice, misleading text, delivery delay, slow acquisition, wrong-workload evidence and an abrupt simulated noise step. The 3 nested budgets of 2048, 8192 and 32768 shots per basis give 72 instances per controller; the independent unit remains the workload. Acquisition spans $( 0 . 1 4 3 - 2 . 2 9 4 ) T _ { 0 }$ in the normal scenario, the normalized median Qwen inference contribution is $0 . 6 9 0 T _ { 0 }$ and independently sampled deployment of 16384 shots per basis spans $1 . 1 4 7 T _ { 0 }$ . Device-cycle and communication durations, and the reference controller’s $1 0 ^ { - 3 } T _ { 0 }$ latency, are stipulated; the Qwen contribution is converted from measured requests.

Every rule checks evidence identity. Authorization alone admits a matching nonzero proposal; confidenceonly adds the calibration-time bound; end-timestamp and whole-acquisition rules add their respective transport allowances. Evidence-based acceptance requires improvement by 0.001. Deployment samples are independent of calibration. Within each deployment sample, candidate and incumbent decoders are evaluated on the same measurement record. The resulting paired intervals establish benefit when wholly below −0.001 and harm when wholly above +0.001; other accepted-update outcomes remain unresolved. Their action coverage is within each stream, not simultaneous across the study. Zero observed resolved harm is an outcome count; an unresolved accepted efect is not counted as evidence of safety.

## B. Protection and retained benefit

Misleading text induces the worst-ranked nonzero proposal in all 12 attack instances. Authorization alone admits 6 resolved harmful updates, from 2 distance-5 workloads at 3 budgets, increasing worst-basis failure risk by 2.40–3.35 percentage points. All evidence-based rules reject the misleading proposals; calibration-time confidence already blocks this attack. This comparison isolates protection against manipulated advice from the separate role of elapsed-time accounting. Identity checks also reject all 12 wrong-workload requests. The following honest-advice comparison measures the recovery benefit retained by these checks.

Independent deployment intervals establish benefit for all 12 honest-advice Qwen instances, which reuse 4 workloads at 3 nested budgets. Whole-acquisition checking retains 4; end-timestamp checking retains 6 [Fig. 4]. The 2 additional rejections are both distance-5 workloads at 32768 shots per basis. They expose the utility cost of requiring a certificate for the allowed conditions, even when the executed path benefits. A deterministic controller achieves more beneficial updates across the workloads satisfying the assumptions. In the delayed and slowacquisition cases, no candidate meets the full acceptance criterion when the evaluator first lists the available updates; they exercise the complete clock without isolating later expiration of an initially admissible update.

An abrupt simulated noise step outside the declared rate bound produces 4 accepted Qwen efects whose benefit or harm remains unresolved. In 3 of those cases, an exploratory deployment interval excludes the improvement promised by the numerical bound: one certificate promises excess risk at most −0.03996, while its independent interval is $\left[ - 0 . 0 1 5 3 2 , 0 . 0 1 6 4 1 \right]$ . This exploratory interval excludes the promised certificate margin, while spanning both signs of the excess risk. The toric sign flip instead resolves harmful acceptance in its simulated recovery loss. Supplemental Figure S16 preserves every Qwen proposal and interval, including rejected beneficial proposals and unresolved recovery outcomes. Section W gives condition-resolved counts and the circuit and timing protocols.

## VI. CONCLUSIONS AND IMPLICATIONS FOR RECOVERY SECURITY

The toric instrument connects sign ambiguity in passive histories to the calibration needed for recovery. Attackercontrolled notes induce harmful proposals that authorization alone admits. Evidence checks block their activation while retaining beneficial updates under honest advice. The guarantee requires valid observations and bounds, without assuming accurate advice.

Confidence-only acceptance rejects the harmful proposals induced by the tested advice attacks. Timing checks address a distinct failure mechanism: a justified update can lose its justification as the physical noise changes. In the short-delay toric workloads of Sec. IV their additional efect is a utility cost. The bounded-ramp extension isolates a further security efect: confidenceonly acceptance admits harmful updates after suficiently long valid drift, while timing checks reject them. The original certificate expires much earlier than harm begins. The validated toric comparison recovers delayed benefit through a sharper channel bound under the same drift rate. An undeclared toric sign flip causes harmful acceptances; the surface-code step contradicts numerical certificates while leaving the sign of the simulated excess failure risk unresolved.

AI agents supply one tested class of proposer. A deterministic controller following the evaluator’s ranking achieves at least as many beneficial updates and is the simpler choice here. Broader AI workflows need a taskspecific benefit; both proposers face the same evidence requirements.

Additional observations resolve decision ambiguity; risk evaluation requires scaling and numerical-error checks. Misleading notes can suppress beneficial completion even when harmful activation is blocked: recovery integrity and workflow availability are distinct endpoints.

a Outcomes under valid drift  
![](images/80ed15d8fe52c4c9f0a636190763e49108091c7cc378028bf2fd317c0b210957.jpg)

b Which beneficial updates lose acceptance?  
![](images/d1b5ee66b19905514242dfa868cda12d51a076b34b85af50fd4670eb7f967f80.jpg)  
FIG. 4. Protection, retained recovery benefit and the cost of stricter timing. (a) All observed outcomes under the stated premises: 20 independent workloads at 3 nested budgets for each controller and rule. Benefit/harm require independent paired deployment intervals below −0.001/above +0.001; other accepted-update outcomes remain unresolved. All rules check workload identity. (b) All 12 honest-advice Qwen instances (4 workloads, 3 budgets). Filled/open squares denote acceptance/rejection by the end-timestamp and whole-acquisition rules. Shaded rows identify 2 distance-5 updates additionally rejected when acquisition time is included. Their independent deployment intervals still establish benefit, as do those of the other 10 proposals. Intervals cover all candidate actions within each stream, not the study simultaneously. Budget labels 2k/8k/32k mean 2048/8192/32768 shots per basis. Supplemental Figure S16 includes all advice conditions; Supplemental Fig. S25 retains the analytical timing curves.

Adaptive estimation [4] and reinforcement learning [2] address changing noise. Longer acquisition reduces sampling uncertainty but can average over drift and delay deployment, as our surface-code extension illustrates. Supplemental Material, Sec. S 1 compares observation requirements and estimation costs.

Tsubouchi, Kwon, Jiang, and Yoshioka [33] identify a related information advantage: conditioning quantum measurements on the syndrome can outperform fixed logical measurements followed by classical processing. This concerns access to the quantum state, rather than faster decoding of the same classical record. Their low-noise asymptotic result assumes independent local Pauli noise and even-distance blocks and averages over logical states. Its extension to unknown drift requires validating the observation model.

The toric chain assumes ideal preparation, extraction, readout and logical actuation, with stationary calibration. The surface-code extension uses reinitialized known-state memories and classical decoder updates. Stored-state structure afects memory lifetime [34], and spectator shifts afect control [35]. Continuous storage and hardware deployment require characterization of preparation, readout, noise drift, and the recovery actually applied.

The security principle is to justify each activated update from independent evidence, assessing protection together with retained recovery benefit.

## ACKNOWLEDGMENTS

This research was supported in part by the National Science Foundation under PHY-2309135. The author thanks participants of the AI for Quantum Matter 2026 program for discussions at the Kavli Institute for Theoretical Physics in Santa Barbara, California, United States.

Data availability. The Supplemental Material provides the analytical derivations, protocols for the numerical simulations and AI-controller evaluations, and aggregate results.

## Supplemental Material Contents

Model and recovery framework   
I. Code, instrument, and observables ........... ..... 12   
II. Exact information in the syndrome instrument..... .....14   
III. Logical dynamics and operational separation .... .....17   
IV. Recovery from the same syndrome record ...................................... ....... 18   
V. Finite-angle validation at L = 3........ ....... 19   
VI. From exact information to justified recovery updates................................................................................. ...... 20   
VII. Measurement assumptions and conditional acceptance ................................................................. .......21   
VIII. Robust acceptance under misleading and stale advice........................................................................... ...... 22   
IX. Recovery improvement, workflow completion, and transfer ................................................................. ....... 23   
Proofs, protocols, and validation   
A. Channel proofs and the general privacy reference ........ .....24   
B. Finite-precision recovery comparisons ........... ....... 26   
C. Extended validity tests ..... ......... 26   
D. Language-model maintenance and preliminary interface tests ....................................................................... ............27   
E. Learning, classical references, and acquisition cost......... ................................................................ .......... 28   
F. Calibration and transfer to circuit-level decoder updates............................................................. ...... 29   
G. Geometry. Disconnected supports and the parity law....... ...... 29   
H. Efect character expansion and exact circuit count ..... ..... 30   
I. Coeficient-level comparison with stabilizer efects and toric paths .......................................................... ...... 30   
J. Leading-order line reduction and the channel remainder...... ..... 32   
K. Product limit and null-prefix histories................. .......... 32   
L. Exhaustive numerical protocol.......   
M. Recovery calculations and validation ............... ............................................................................................ ........ 33   
N. Exact channel evaluation and continuous bounds .................................................. ......34   
O. Information and duration references...... ........ 35   
P. Training, splits, and workflow details ...... ...... 35   
Q. Paired circuit risks and confirmation uncertainty........................ .....37   
R. Numerical precision and validation..... ........37   
S. Theoretical context and prior-work comparison ................................. ......... 38   
T. Calibration, activation, and circuit validation ...............   
U. Detailed empirical panels supporting the quantitative overview .......................................................................... ..........44   
V. Circuit definitions, measurement conventions, and shared schedules....................................................... .......44   
W. Time-resolved circuit maintenance and adversarial advice ...............................................................................................47   
X. Additional validation of calibration and recovery updates ......................................................................... .........54   
Y. Unified toric calibration and recovery: complete protocol ......... ...... 62   
Z. Supporting recovery curves and analytical timing bounds ... .... 65   
AA. Additional attacks, bounded drift, and multistep maintenance ................................................... .......67   
AB. Validated toric bounds and matched timing comparison..... ......71

## I. CODE, INSTRUMENT, AND OBSERVABLES

The supporting studies use distinct observation models and endpoints. Sections I–V define the toric instrument and its exact results. Sections VII–IX introduce the separate sentinel calibration and controller comparisons. Section T gives the readout, activation, and stationaryacquisition circuit protocols. Sections W, Y, and AA give the timed surface-code, encoded toric, and additional attack protocols, respectively. Throughout, observational nonidentifiability means that diferent conditions have the same allowed observation law; sampling uncertainty is finite-data uncertainty within a specified observation model; and physical-model uncertainty or violation concerns whether preparation, readout, transfer and drift assumptions describe deployment. Additional shots address the second problem, a diferent measurement may address the first, and independently justified physical characterization addresses the third.

The analytical acceptance statement is a conditional theorem about a valid risk enclosure. Numerical channel evaluations and simulation audits check implementations; sampled deployment intervals assess outcomes. In the tests that hold proposals fixed while comparing acceptance rules, harm means positive evaluated excess infidelity. The multistep controller study uses the thresholds $D _ { u } < - \delta$ and $D _ { u } > \delta .$ , whereas the central toric and timed surface studies require deployment intervals wholly beyond those thresholds. A margin violation $D _ { u } > - \delta$ can still be an improvement relative to the incumbent. The validated toric replay in Sec. AB instead classifies interval-enclosed model risks after adding evaluation latency. Each study states its sampling unit and coverage scope; pooled counts across these studies are not a common success rate. Independence of samples refers to the stated sampling law. A separate implementation provides a cross-check of a calculation, while separation of evaluator and adviser restricts who can change records or authorize an action. Neither software distinction makes reused observations statistically independent.

## A. Periodic square toric code

Fix an integer $L \ \geq \ 2$ . Vertices and faces are both labeled by $\mathbb { Z } _ { L } ^ { 2 }$ , and the $n = 2 L ^ { 2 }$ physical qubits occupy tagged edges

$$
\begin{array} { l } { { h _ { x , y } : ( x , y ) \longrightarrow ( x + 1 , y ) , ~ i ( h _ { x , y } ) = x + L y , } } \\ { { v _ { x , y } : ( x , y ) \longrightarrow ( x , y + 1 ) , ~ i ( v _ { x , y } ) = L ^ { 2 } + x + L y . } } \end{array}\tag{1}
$$

Coordinates are modulo $L ;$ tags remain distinct when edges share endpoints at $L \ = \ 2$ . For an edge word $a \in \mathbb { F } _ { 2 } ^ { E _ { L } }$ , write $\begin{array} { r } { \bar { X } ( a ) = \prod _ { e } X _ { e } ^ { a _ { e } } } \end{array}$ and let |a| be its Ham ming weight. The binary matrices $H _ { X }$ and $H _ { Z }$ contain respectively the star and plaquette supports,

$$
\begin{array} { r } { ( H _ { X } ) _ { x , y } = \{ h _ { x , y } , h _ { x - 1 , y } , v _ { x , y } , v _ { x , y - 1 } \} , } \\ { ( H _ { Z } ) _ { x , y } = \{ h _ { x , y } , h _ { x , y + 1 } , v _ { x , y } , v _ { x + 1 , y } \} . } \end{array}\tag{2}
$$

All binary algebra is over $\mathbb { F } _ { 2 } ,$ and $H _ { X } H _ { Z } ^ { \mathsf { T } } = 0$ The common +1 eigenspace of these Calderbank–Shor–Steane (CSS) generators encodes two qubits because both check matrices have rank $L ^ { 2 } - 1$ . We use the logical frame

$$
\begin{array} { r l r } { \overline { { X } } _ { x } = \displaystyle \prod _ { x \in \mathbb { Z } _ { L } } X _ { v _ { x , 0 } } , \qquad } & { \overline { { X } } _ { y } = \displaystyle \prod _ { y \in \mathbb { Z } _ { L } } X _ { h _ { 0 , y } } , } & \\ { \overline { { Z } } _ { x } = \displaystyle \prod _ { y \in \mathbb { Z } _ { L } } Z _ { v _ { 0 , y } } , \qquad } & { \overline { { Z } } _ { y } = \displaystyle \prod _ { x \in \mathbb { Z } _ { L } } Z _ { h _ { x , 0 } } . } & \end{array}\tag{3}
$$

The codespace isometry $V : \mathbb { C } ^ { 4 } \to \mathcal { H } _ { \mathrm { p h y s } }$ is fixed by

$$
\begin{array} { r l } & { | \overline { { 0 0 } } \rangle = | \operatorname { r o w } H _ { X } | ^ { - 1 / 2 } \displaystyle \sum _ { u \in \operatorname { r o w } H _ { X } } | u \rangle , } \\ & { V | a b \rangle = \overline { { X } } _ { x } ^ { a } \overline { { X } } _ { y } ^ { b } | \overline { { 0 0 } } \rangle . } \end{array}\tag{4}
$$

For $c = ( c _ { 1 } , c _ { 2 } ) \in \mathbb { F } _ { 2 } ^ { 2 } .$ , we abbreviate $\overline { { \boldsymbol X } } ^ { c } = \overline { { \boldsymbol X } } _ { x } ^ { c _ { 1 } } \overline { { \boldsymbol X } } _ { y } ^ { c _ { 2 } }$ . This is standard toric-code geometry [36, 37]; the tags and signs fix $L = 2$ and all phases below.

## B. One-round instrument

Preparation, syndrome extraction, recovery, and terminal readout are ideal: they introduce no additional errors. The coherent rotation below is the noise process. Extra syndrome-conditioned logical corrections are applied as specified; separate extensions explicitly introduce phase discretization or actuation noise. Measurement outcomes remain random according to the Born rule.

Let ${ \widetilde { H } } _ { Z }$ be a fixed independent row basis of $H _ { Z }$ , with rank $r = L ^ { 2 } - 1$ , and $\boldsymbol { s } \in \mathbb { F } _ { 2 } ^ { r }$ the reduced complete syndrome. Removing the single plaquette constraint loses no information. For each reachable s, choose a minimumweight word $r _ { s }$ with $\widetilde { H } _ { Z } r _ { s } = s$ , breaking ties by the lexicographically smallest ascending edge-index tuple. Let $C _ { s } = X ( r _ { s } )$ and Π project onto that syndrome sector, including the +1 star sector. Each round physically applies $C _ { s }$ in this frame before the next round begins.

For a real, dimensionless signed angle θ, the physical perturbation and the induced logical Kraus operators are

$$
U _ { L } ( \theta ) = \prod _ { e \in E _ { L } } e ^ { - i \theta X _ { e } / 2 } , \qquad K _ { s } ( \theta ) = V ^ { \dagger } C _ { s } \Pi _ { s } U _ { L } ( \theta ) V .\tag{5}
$$

Here θ specifies rotation direction and magnitude, while τ denotes clock time. For a constant signed angular rate Ω acting for a positive duration $\Delta \tau , \theta = \Omega \Delta \tau ;$ changing the sign of Ω reverses the rotation without reversing time. The equality of the passive record laws at $\pm \theta$ is a property of the instrument proved below, not a claim that inverse unitaries are indistinguishable by every measurement.

TABLE S1. Main notation and operational roles.
<table><tr><td>Symbol</td><td>Operational role</td></tr><tr><td>L</td><td>Toric-code linear size;  $n = 2 L ^ { 2 }$  physical qubits</td></tr><tr><td> $\mathcal { I } _ { s } ^ { ( \theta ) }$ </td><td>Unnormalized state-update map for branch s</td></tr><tr><td> $F _ { s } ( \theta )$ </td><td>Measurement effect determining the probability of outcome s</td></tr><tr><td> $\Lambda _ { L } ( \theta )$ </td><td>Averaged corrected logical channel</td></tr><tr><td> $d _ { \mathrm { a f f } }$ </td><td>Minimum cut-covered nontrivial logical support</td></tr><tr><td> $b _ { \mathrm { r e c } } ( L )$ </td><td>First Taylor order of a nonscalar effect</td></tr><tr><td> $\eta _ { L } ( \theta )$ </td><td>Worst-case one-round record distance</td></tr><tr><td> $\scriptstyle \sum _ { \mathrm { r e c } } ( \theta , R )$ </td><td>Worst-case R-round history distance</td></tr><tr><td> $\underline { { R _ { * } ( \theta ) } }$ </td><td>Explicit nonresonant repeated-round schedule</td></tr></table>

The unnormalized branch map, syndrome efect, and averaged corrected channel are

$$
\begin{array} { r l r } {  { \mathcal { T } _ { s } ^ { ( \theta ) } ( \rho ) = K _ { s } ( \theta ) \rho K _ { s } ( \theta ) ^ { \dagger } , } } \\ & { } & { F _ { s } ( \theta ) = K _ { s } ( \theta ) ^ { \dagger } K _ { s } ( \theta ) , \qquad \Lambda _ { L } ( \theta ) = \sum _ { s } \mathcal { T } _ { s } ^ { ( \theta ) } . } \end{array}\tag{6}
$$

Completeness gives $\begin{array} { r } { \sum _ { s } F _ { s } ( { \boldsymbol { \theta } } ) = I _ { 4 } } \end{array}$ and makes $\Lambda _ { L } ( \theta )$ completely positive and trace preserving, while $\{ F _ { s } ( \theta ) \} _ { s }$ is the logical syndrome positive-operator-valued measure $( \mathrm { P O V } \mathrm { \bar { M } } )$ . For any trace-one logical density operator $\rho ,$ efect $F _ { s } ( \theta )$ gives

$$
p _ { s } ( \theta | \rho ) = \mathrm { T r } [ F _ { s } ( \theta ) \rho ] .\tag{7}
$$

The efect governs record information; $\mathcal { J } _ { s } ^ { ( \theta ) }$ is the unnormalized state update. No branch normalization occurs, so zero branches cause no division. For unitary $U$ , write $\mathrm { A d } _ { U } ( A ) = U A U ^ { \dagger }$

Separate the scalar and nonscalar parts of an efect by

$$
q _ { s } ( \theta ) = \frac 1 4 \mathrm { T r } F _ { s } ( \theta ) , \qquad D _ { s } ( \theta ) = F _ { s } ( \theta ) - q _ { s } ( \theta ) I _ { 4 } .\tag{8}
$$

For an analytic matrix function, ord A is the smallest nonnegative Taylor degree with a nonzero coeficient, and $\operatorname { o r d } _ { \theta } 0 = + \infty$ . We define the first Taylor order of logicalinput dependence in the syndrome efects and maximal one-round logical-input distinguishability by

$$
b _ { \mathrm { r e c } } ( L ) = \operatorname* { m i n } _ { s } { \mathrm { o r d } } _ { \theta } D _ { s } ( \theta ) ,\tag{9}
$$

$$
\eta _ { L } ( \theta ) = \operatorname* { s u p } _ { \rho , \sigma } \frac { 1 } { 2 } \sum _ { s } \left| \mathrm { T r } [ F _ { s } ( \theta ) ( \rho - \sigma ) ] \right| ,\tag{10}
$$

over logical density operators. Thus $\eta _ { L }$ measures distinguishability between encoded inputs at one fixed physical angle θ. For discrete laws, the total-variation (TV) distance is $\begin{array} { r } { \mathrm { T V } ( p , q ) = \frac { 1 } { 2 } \sum _ { s } | p _ { s } - q _ { s } | } \end{array}$

## C. A positive calibrated-recovery example

We first show a positive recovery example using the instrument just defined, before asking what its passive record can identify. Numerical calibration experiments fix $L = 3 ,$ so $N = n = 1 8$ . In this section, N denotes the physical-qubit count and H the memory horizon denoted by R in the exact theorems. The main encoded-calibration experiment uses n for qubits and N for calibration memories. The initial feasibility sweep uses the true simulated angle to select a fixed candidate before testing acceptance from calibration counts. The later controller workflows select candidates without access to that angle. The inverse-polar construction is derived in Sec. IV.

The single-qubit rotation is $R _ { X } ( \theta ) = \mathrm { e x p } ( - i \theta X / 2 )$ 2 as in Eq. (5). The logical Kraus operators $K _ { s } ( \theta )$ are diagonal in the simultaneous logical-X eigenbasis. An action u specifies a fixed diagonal phase table $V _ { s } ( u )$ , and its one-round channel is

$$
\mathcal { E } _ { \theta , u } ( \rho ) = \sum _ { s } V _ { s } ( u ) K _ { s } ( \theta ) \rho K _ { s } ( \theta ) ^ { \dagger } V _ { s } ( u ) ^ { \dagger } .\tag{11}
$$

The incumbent $u _ { 0 }$ adds no phase correction. The other 12 actions use inverse-polar phases calibrated at ±0.025, $\pm 0 . 0 5 , \pm 0 . 0 7 5 , \pm 0 . 1 0 , \pm 0 . 1 2 5$ , and ±0.15 radians. The catalog consists of twelve syndrome-conditioned recovery tables computed from the model. The tables are fixed before evaluation.

For a stationary H-round memory, define the loss and excess loss by

$$
\begin{array} { r } { \mathcal { R } ( \theta , u ) = 1 - F _ { e } ( \mathcal { E } _ { \theta , u } ^ { H } ) , } \end{array}\tag{12}
$$

$$
D _ { u } ( \theta ) = \mathcal { R } ( \theta , u ) - \mathcal { R } ( \theta , u _ { 0 } ) .\tag{13}
$$

Here $F _ { e }$ is entanglement fidelity relative to the identity logical channel. Negative $D _ { u }$ is beneficial; positive $D _ { u }$ is harmful relative to the incumbent. The complete simulated channel determines whether an update is beneficial or harmful. The commuting phase tables can be accumulated and applied terminally. Every comparator can defer correction to the terminal step, placing online and deferred actuation on the same footing.

At |θ| = 0.1 and $H = 3 0 0$ , the incumbent infidelity is 0.1323553, whereas the correctly calibrated phase table gives 0.0007456. The initial feasibility sweep uses 72 conditions, spanning six signed angles, three horizons, and four probe budgets. For 8192 probe shots, 1000 calibration replicates at each sign admit the correctly chosen table in 85.8% and 83.6% of cases. These candidates, selected using the true angle, show that the calibration rule can accept beneficial updates. The subsequent workflows evaluate selection based only on observations. Workflows using only observed information are evaluated in Sec. IX and Supplemental Material, Sec. E.

![](images/77fd1a11558e617138a87d0fd43a6f1c906ecec59563bebc843114bcd233a070.jpg)

b 3,000 memory rounds  
![](images/53aca0cfc565167c976d4fb49c228c745622cfe9640e690bbb529961d7062664.jpg)  
FIG. S1. Beneficial and harmful recovery under calibration mismatch. Each curve evaluates the exact logical-channel formula numerically at 2001 angles; the panels show two memory horizons. Fixed phase tables can greatly reduce infidelity near their calibration point and worsen it elsewhere. The plotted grid illustrates the risk, while the evaluator also bounds its variation between grid points. The correction assumes error-free logical actuation; the appreciable incumbent infidelity makes its recovery benefit visible.

## II. EXACT INFORMATION IN THE SYNDROME INSTRUMENT

## A. Efect characters and commuting Kraus operators

Assign each edge an independent angle $\epsilon _ { e }$ and write $\begin{array} { r } { U _ { L } ( \epsilon ) = \prod _ { e } \exp ( - i \epsilon _ { e } X _ { e } / 2 ) } \end{array}$ . With $h _ { z } = \widetilde { H } _ { Z } ^ { \mathsf { T } } z \ \mathrm { f o r } \ z \in \mathbb { F } _ { 2 } ^ { r }$ the exact expansion of the syndrome efects is

$$
F _ { s } ( \epsilon ) = \sum _ { [ f ] \in \mathcal { L } _ { X } } B _ { s , [ f ] } ( \epsilon ) \overline { { X } } _ { [ f ] } ,\tag{14}
$$

$$
B _ { s , [ f ] } = \sum _ { u \in \mathrm { r o w } H _ { X } } \beta _ { s , f + u } ,\tag{15}
$$

$$
\beta _ { s , f } = ( - i ) ^ { | f | } 2 ^ { - r } \sum _ { z \in \mathbb { R } _ { 2 } ^ { r } \atop f \subseteq h _ { z } } ( - 1 ) ^ { z \cdot s } \prod _ { e \in f } \sin \epsilon _ { e } \prod _ { e \in h _ { z } \backslash f } \cos \epsilon _ { e } ,\tag{16}
$$

for $f \in$ ker $H _ { Z }$ , and $\beta _ { s , f } ~ = ~ 0$ otherwise. Here $\overline { { \boldsymbol X } } _ { [ f ] }$ is the logical X operator associated with the quotient class [f]. Equation (16) follows by expanding $U _ { L } ( \epsilon )$ in X supports a and inserting the syndrome character $2 ^ { - r } \textstyle \sum _ { z } ( - 1 ) ^ { z \cdot ( \widetilde { H } _ { Z } a + s ) }$ The four local pair sums 1, cos $\epsilon _ { e } , 0 , - i$ sin $\epsilon _ { e }$ yield the containment condition. Summing over s leaves $z = 0 ,$ hence $\textstyle \sum _ { s } F _ { s } = I _ { 4 }$

Expanding the Kraus operators in logical X classes

gives

$$
K _ { s } ( \theta ) = \sum _ { c \in \mathbb { F } _ { 2 } ^ { 2 } } \kappa _ { s , c } ( \theta ) \overline { { \boldsymbol X } } ^ { c } ,\tag{17}
$$

$$
\kappa _ { s , c } ( \theta ) = \sum _ { \stackrel { a : \widetilde { H } _ { Z } a = s } { \left[ a + r _ { s } \right] = c } } \cos ( \theta / 2 ) ^ { n - | a | } \left[ - i \sin ( \theta / 2 ) \right] ^ { | a | } .\tag{18}
$$

## B. The all-angle parity law and sign ambiguity

For an integer $R \geq 1$ , define the exact retained history law

$$
P _ { \rho , \theta } ^ { ( R ) } ( s _ { 1 } , \ldots , s _ { R } ) = \operatorname { T r } \left[ \mathcal { T } _ { s _ { R } } ^ { ( \theta ) } \circ \cdot \cdot \cdot \circ \mathcal { T } _ { s _ { 1 } } ^ { ( \theta ) } ( \rho ) \right]\tag{19}
$$

and its maximal logical-input distance

$$
\Delta _ { \mathrm { r e c } } ( \theta , R ) = \operatorname* { s u p } _ { \rho , \sigma } \mathrm { T V } \Big ( P _ { \rho , \theta } ^ { ( R ) } , P _ { \sigma , \theta } ^ { ( R ) } \Big ) .\tag{20}
$$

Rounds need not be independent. The conditional logical state after a prefix sets the next syndrome law. For $R = 0$ , the law is unit mass on the empty record and $\Delta _ { \mathrm { r e c } } ( \theta , 0 ) = 0$

The phase-parity selection mechanism has precedents in [17–19, 38]. Here it identifies the observable product parity of this two-logical-qubit toric instrument. The subsequent product mixture is the repeated quantum nondemolition (QND) construction of Bauer, Benoist, and Bernard [26, Secs. 3 and 5, Eq. (5)]. Its pointer states are the simultaneous logical-X eigenstates; states with the same parity have equal emission laws. Write $\begin{array} { r } { P = \overline { { X } } _ { x } \overline { { X } } _ { y } . } \end{array}$ The conserved degenerate sectors are $\Pi _ { \pm } = ( I _ { 4 } \pm P ) / \bar { 2 } $ with initial weights $\mathrm { T r } ( \Pi _ { \pm } \rho ) = ( 1 \pm m _ { \rho } ) / 2$ . At angles where $p ^ { + } = p$ <sup>−</sup> the two sectors are themselves observationally indistinguishable. The specific toric calculation is this all-angle parity identification and its nonzero order- $\cdot \theta ^ { 2 L }$ leakage coeficient in theorem 4.

Proposition 1 (Exact parity record and sign symmetry). For fixed odd $L \geq 3$ and every real θ, there are real even functions $b _ { s } ( \theta )$ such that

$$
F _ { s } ( \theta ) = q _ { s } ( \theta ) I _ { 4 } + b _ { s } ( \theta ) P , \qquad P = \overline { { X } } _ { x } \overline { { X } } _ { y } .\tag{21}
$$

Set $p _ { s } ^ { \pm } ( \theta ) = q _ { s } ( \theta ) \pm b _ { s } ( \theta )$ and $m _ { \rho } = { \mathrm { T r } } ( P \rho )$ . The $p ^ { \pm }$ are probability laws, and for every $R \geq 1$

$$
\begin{array} { l } { { \displaystyle P _ { \rho , \theta } ^ { ( { \cal R } ) } ( s _ { 1 } , \ldots , s _ { \cal R } ) = \frac { 1 + m _ { \rho } } { 2 } \prod _ { j = 1 } ^ { { \cal R } } p _ { s _ { j } } ^ { + } ( \theta ) } } \\ { { \displaystyle ~ + \frac { 1 - m _ { \rho } } { 2 } \prod _ { j = 1 } ^ { { \cal R } } p _ { s _ { j } } ^ { - } ( \theta ) . } } \end{array}\tag{22}
$$

Consequently,

$$
\eta _ { L } ( \theta ) = \sum _ { s } | b _ { s } ( \theta ) | ,\tag{23}
$$

$$
\begin{array} { r } { \Delta _ { \mathrm { r e c } } ( \theta , R ) = \mathrm { T V } \big ( ( p ^ { + } ) ^ { \otimes R } , ( p ^ { - } ) ^ { \otimes R } \big ) , } \end{array}\tag{24}
$$

$$
P _ { \rho , - \theta } ^ { ( R ) } = P _ { \rho , \theta } ^ { ( R ) } .\tag{25}
$$

Inputs with equal $m _ { \rho }$ have exactly identical history laws at all angles and all history lengths.

Proof. Every support f surviving equation (16) is a cycle contained in a cut, hence has even weight. Star generators have even weight, while the two chosen logical X repre sentatives each have odd weight L. Thus the parity of a cycle in logical class $( c _ { 1 } , c _ { 2 } )$ is $c _ { 1 } + c _ { 2 }$ modulo two. Only classes (0, 0) and $( 1 , 1 )$ can survive. The factor $( - i ) ^ { | f | }$ is real, and uniform sign reversal multiplies its sine product by $( - 1 ) ^ { | f | } = 1$ . Both $q _ { s }$ and $b _ { s }$ are therefore real and even. Positivity and completeness of the POVM make $q _ { s } \pm b _ { s }$ nonnegative and normalized.

By equation (17), all $K _ { s }$ and their adjoints belong to the commuting algebra generated by ${ \overline { { X } } } _ { x } , { \overline { { X } } } _ { y } .$ . The efect of a complete history is consequently

$$
( K _ { s _ { R } } \cdot \cdot \cdot K _ { s _ { 1 } } ) ^ { \dagger } ( K _ { s _ { R } } \cdot \cdot \cdot K _ { s _ { 1 } } ) = \prod _ { j = 1 } ^ { R } F _ { s _ { j } } .\tag{26}
$$

Resolve this product on the projectors $( I _ { 4 } \pm P ) / 2$ and take its trace against $\rho$ to obtain equation (22). Diferences of two mixtures are $( m _ { \rho } - m _ { \sigma } ) / 2$ times the diference of the two product laws. The maximum $| m _ { \rho } - m _ { \sigma } | = 2$ is attained by states in opposite parity sectors, proving equations (23) and (24). Evenness proves equation (25).

The history is conditionally independent and identically distributed (i.i.d.) given a conserved parity label, although it is generally correlated when that label is unknown. For distinguishing the two parity sectors with equal priors, an optimal classifier compares $\prod _ { j } p _ { s _ { j } } ^ { + }$ with $\prod _ { j } p _ { s _ { j } } ^ { - }$ . This product form handles zero probabilities without logarithms. Syndrome counts are suficient statistics by this product form of repeated QND measurements with the same instrument [26]; temporal ordering supplies no extra input information for this instrument. The contraction bound in Sec. III B also applies without this commuting structure.

## C. The geometric constraint on nonscalar efects

The ordinary logical quotients and distances are

$$
\begin{array} { r } { \begin{array} { l l l } { \mathcal { L } _ { X } = \ker H _ { Z } / \mathrm { r o w } H _ { X } , } & { d _ { X } = \underset { a \in \ker H _ { Z } \backslash \mathrm { r o w } H _ { X } } { \operatorname* { m i n } } | a | , } \\ { \mathcal { L } _ { Z } = \ker H _ { X } / \mathrm { r o w } H _ { Z } , } & { d _ { Z } = \underset { b \in \ker H _ { X } \backslash \mathrm { r o w } H _ { Z } } { \operatorname* { m i n } } | b | . } \end{array} } \end{array}\tag{27}
$$

Efects impose an additional incidence constraint.

$$
d _ { \mathrm { a f f } } : = \operatorname* { m i n } \left\{ \left| a \right| : \begin{array} { l } { a \in \ker H _ { Z } \setminus \mathrm { r o w } H _ { X } , } \\ { \exists h \in \mathrm { r o w } H _ { Z } : \ \mathrm { s u p p } a \subseteq \mathrm { s u p p } h } \end{array} \right\} ,\tag{28}
$$

with $+ \infty$ for an empty set. Cycle–cut orthogonality makes admissible supports even; circuit decomposition makes this equivalently the shortest even, homologically nontrivial dual-graph circuit. “Afine” follows binarymatroid usage [39]; only (28) is used.

Lemma 2 (Cut containment for a circuit). Let C be a simple circuit of a loopless labeled graph G, allowing a length-two circuit of distinct parallel edges. Then

$$
( \exists S \subseteq V ( G ) : E ( C ) \subseteq \delta ( S ) { \big ) } \quad \iff \quad | E ( C ) | { \mathrm { ~ } } i s \ e v e n .\tag{29}
$$

Proof. Write the circuit as v<sub>0</sub>, $v _ { 1 } , \ldots , v _ { m } = v _ { 0 }$ and let $s _ { j }$ indicate membership in S. Every edge crossing the cut requires $s _ { j + 1 } = 1 - s _ { j }$ , which closes consistently only when m is even. Conversely, choose the alternating vertices of an even circuit as S. For two tagged parallel edges, choose one endpoint. Extra edges in δ(S) do not afect containment. □

This step applies the standard graph characterization of cut containment. There must exist a cut containing the circuit: a fixed arbitrary cut need not cover an even circuit. Nor does even total weight sufice for disconnected supports; two disjoint triangles give a counterexample. In general a support is cut-contained exactly when its support subgraph is bipartite. Indeed, a containing cut gives a two-coloring of every supported edge; conversely, choose one bipartition class from each connected component as S and assign isolated ambient vertices arbitrarily. Every supported edge then crosses $\delta _ { G } ( S )$ , irrespective of other ambient edges. The circuit-decomposition argument needed here is given in Supplemental Material, Sec. G.

Theorem 3 (Toric distance and afine-distance parity law). For every integer $L \geq 2$ , the code in equations (1) and (2) has parameters $[ [ 2 L ^ { 2 } , 2 , L ] ]$ and

$$
d _ { X } ( L ) = d _ { Z } ( L ) = L , \qquad d _ { \mathrm { a f f } } ( L ) = \left\{ L , \quad L \ e v e n , \right.\tag{30}
$$

Proof. Regard $H _ { Z }$ as the vertex–edge incidence matrix of the labeled dual graph. Then ker $H _ { Z }$ is its cycle space, row $H _ { Z }$ its cut space, and row $H _ { X }$ spans dual-face boundaries. Seam parities identify ker $H _ { Z } /$ row $H _ { X } \simeq \mathbb { F } _ { 2 } ^ { 2 }$ , and any nontrivial Eulerian support contains a nontrivial circuit. If a length-w oriented circuit lifts to displacement $L ( p , q )$ in the square-grid universal cover, step counting gives

$$
w \geq L ( | p | + | q | ) , \qquad w \equiv L ( p + q ) { \pmod { 2 } } .\tag{31}
$$

Its class is (p mod $2 , q$ mod 2), so $w \ge L$ , attained by straight winding-one circuits; primal–dual symmetry gives $d _ { Z } = L$

A cut-contained circuit must be even; conversely, alternating vertices of an even simple circuit gives a containing cut (for the $L = 2$ parallel-edge circuit, select one endpoint). Hence even L admits a straight length-L logical circuit. For odd $L ,$ equation (31) makes $p + q$ even, so nontrivial winding forces both $p , q$ odd and $w \geq 2 L$ . The staircase

$$
C _ { \mathrm { d i a g } } ( L ) = \{ h _ { j , j } , v _ { j + 1 , j } : j \in \mathbb { Z } _ { L } \}\tag{32}
$$

has length 2L, winding (1, 1), and lies in the cut selected by $\{ f _ { j , j } : j \in \mathbb { Z } _ { L } \}$ . For $L = 2$ , the distinct parallel dual edges $\{ h _ { 0 , 0 } , h _ { 0 , 1 } \}$ supply a length-two witness. Finally, rank $H _ { X } =$ rank $H _ { Z } = \dot { L } ^ { 2 } - 1$ completes the parameter count. See Supplemental Material, Sec. G for disconnected supports. □

For even L this obstruction is absent. Henceforth L is odd, allowing the two orders to separate.

## D. Attainment and the exact leading coeficient

Let $P = \overline { { X } } _ { x } \overline { { X } } _ { y }$ and let $B _ { 0 , P } ( \theta )$ denote its coeficient in the zero-syndrome efect after the uniform specialization $\epsilon _ { e } = \theta$

Theorem 4 (Exact syndrome-record onset). For every fixed odd integer $L \geq 3$ ,

$$
b _ { \mathrm { r e c } } ( L ) = 2 L ,\tag{33}
$$

$$
[ \theta ^ { 2 L } ] B _ { 0 , P } ( \theta ) = - \frac { M _ { L } } { 2 ^ { 2 L - 1 } } ,\tag{34}
$$

$$
M _ { L } = L \left[ { \binom { 2 L } { L } } - 2 L \right] ,\tag{35}
$$

$$
\eta _ { L } ( \theta ) = \Theta _ { L } ( | \theta | ^ { 2 L } ) \qquad ( \theta \to 0 ) .\tag{36}
$$

The $\Theta _ { L }$ constants and the neighborhood of the origin may depend on L. Here $M _ { L }$ is the cardinality of the distinct unrooted, unoriented, tagged-edge simple circuits of length 2L that are contained in a dual cut and have nontrivial winding.

Proof. For a Taylor multi-index $\pmb { a } \in \mathbb { N } ^ { 2 L ^ { 2 } }$ , sine/cosine parity in equation (16) forces the support for $\epsilon ^ { a }$ to be $f _ { e } =$ $a _ { e }$ mod 2. Thus $| f | \leq | a |$ , and a surviving nonscalar class requires $f$ to be a nontrivial logical support contained in a word of row $H _ { Z }$ . By theorem $3 , \ \vert \pmb { a } \vert \geq 2 L$ . Thus all terms below degree 2L vanish before uniform specialization.

At equality, $| f | = | \pmb { a } | = 2 L ,$ so the monomial is squarefree and $a = f$ . Then f is one even nontrivial circuit, whose winding parity (1, 1) gives class P. At $s = 0$ all character signs are positive and $( - i ) ^ { 2 L } = - 1$ . Rank– nullity gives $2 ^ { r - 2 L + 1 }$ selectors per covered circuit, hence contribution $- 2 ^ { - ( 2 L - 1 ) }$ after $2 ^ { - r }$

Equality in the winding bound forces L monotone horizontal and vertical steps and one of four sign pairs. At each root their order is a balanced binary word; exactly the 2L cyclic shifts of $H ^ { L } V ^ { L }$ among $\binom { 2 L } { L }$ words revisit a projected vertex. Thus there are $4 L ^ { 2 } [ \binom { 2 L } { L } - 2 L ]$ rooted oriented tuples. Dividing by 2L roots and two orientations gives equations (34) and (35).

For the total-variation lower bound, the trace-one states

$$
\rho _ { \pm } = \frac { I _ { 4 } \pm P } { 4 }\tag{37}
$$

obey

$$
[ \theta ^ { 2 L } ] \{ p _ { 0 } ( \theta | \rho _ { + } ) - p _ { 0 } ( \theta | \rho _ { - } ) \} = - \frac { M _ { L } } { 2 ^ { 2 L - 2 } } \neq 0 .\tag{38}
$$

This gives the fixed-L lower bound. For the upper bound, $D _ { s } \hat { ( \theta ) = } \theta ^ { 2 L } R _ { s } \hat { ( \theta ) }$ with $R _ { s }$ analytic and locally bounded. For $\Delta = \rho - \sigma _ { \mathrm { { \ell } } }$ , Tr $\Delta = 0$ and $\| \Delta \| _ { 1 } \leq 2 ,$ , so

$$
\frac { 1 } { 2 } \sum _ { s } | \operatorname { T r } [ F _ { s } ( \theta ) \Delta ] | \leq \sum _ { s } \| D _ { s } ( \theta ) \| _ { \infty } = O _ { L } ( | \theta | ^ { 2 L } ) .\tag{39}
$$

Finitely many syndromes complete the proof; Supplemental Material, Sec. H expands the character derivation and count. □

For example, $M _ { 3 } = 4 2 .$ , so the exact coeficients in equations (34) and (38) are respectively $- 2 1 / 1 6$ and $- 2 1 / 8$ The plotted quantities are coeficients in the

![](images/c9bcd711ada7f0c3b033afb3161d1a2ed9d3787ea97c79d7b9d5331929e9e5a9.jpg)

![](images/c539966662d1729fde68f6511c7586b5bbbb88c440d64d8aeb51d0b7844f56db.jpg)  
FIG. S2. Geometric witnesses for theorem 3. A straight logical circuit has length L. At odd $L ,$ the additional even/cut-covered condition forces winding in both torus directions, and the diagonal staircase attains length 2L. The drawing depicts the code’ periodic topology.

small-angle Taylor expansion. The formula gives $M _ { L } =$ (42, 1210, 23926, 437418) at $L = ( 3 , 5 , 7 , 9 )$ .

The count and the normalized coeficient have diferent distance dependence:

$$
M _ { L } \sim \sqrt { L / \pi } 4 ^ { L } , \qquad \frac { M _ { L } } { 2 ^ { 2 L - 1 } } \sim 2 \sqrt { L / \pi } .\tag{40}
$$

The exponential support multiplicity cancels in this particular leading coeficient. Uniform control in L of the full contrast, higher-order terms, and validity neighborhoods requires separate bounds.

## III. LOGICAL DYNAMICS AND OPERATIONALSEPARATION

## A. The known leading coherent drift

The corrected-channel drift has an established leading order [21]. We retain its coeficient and an explicit remainder in the conventions of this instrument to compare it with the separately derived efect onset.

Theorem 5 (Fixed-odd-L corrected logical drift). For every fixed odd integer $L \geq 3$ , the instrument in equation (5) satisfies

$$
\Lambda _ { L } ( \theta ) = \mathrm { i d } - i \alpha _ { L } \theta ^ { L } [ \overline { { { H } } } , \cdot ] + E _ { L } ( \theta ) , \qquad \overline { { { H } } } = \overline { { { X } } } _ { x } + \overline { { { X } } } _ { y } ,
$$

where

(41)

$$
\alpha _ { L } = \frac { L } { 2 ^ { L } } { \binom { L - 1 } { ( L - 1 ) / 2 } } > 0 .\tag{42}
$$

Every coeficient of orders $1 , \ldots , L - 1$ vanishes. At order L there is no Hilbert–Schmidt self-adjoint (dissipative)

component and no ${ \overline { { X } } } _ { x } { \overline { { X } } } _ { y }$ component. In the unhalved diamond norm, one explicit global bound is

$$
\| E _ { L } ( \theta ) \| _ { \infty } \leq \kappa _ { L } | \theta | ^ { L + 1 } , \qquad \kappa _ { L } = 1 6 \frac { ( 2 L ^ { 2 } ) ^ { L + 1 } 2 ^ { 4 L ^ { 2 } } } { ( L + 1 ) ! } .\tag{43}
$$

The remainder bound holds for every real θ. The estimates apply at fixed L.

The proof is given in Supplemental Material, Sec. A 1. Reversing $e ^ { - i \theta X / 2 } \ t o \ e ^ { + i \mathbf { \dot { \theta } } \mathbf { \dot { X } } / 2 }$ reverses the odd-L generator; removing the half angle multiplies $\alpha _ { L }$ by $2 ^ { L }$ . These checks fix common convention errors. The record result remains a separate calculation.

## B. The complete-history bound and a nonresonant schedule

Proposition 6 (History contraction). For every fixed L, real θ, and integer $R \geq 1$ ，

$$
\Delta _ { \mathrm { r e c } } ( \theta , R ) \leq 1 - [ 1 - \eta _ { L } ( \theta ) ] ^ { R } \leq \mathrm { m i n } \{ 1 , R \eta _ { L } ( \theta ) \} .\tag{44}
$$

Proof. At each positive-probability common prefix, separately normalize both conditional logical states; equation (10) bounds their next-syndrome kernels by $\eta _ { L } ( \theta )$ . At a null prefix, any normalized next-syndrome distribution completes the transition rule without changing the zeromass law. While histories agree, maximally couple the next kernels. Each step preserves agreement with probability at least $1 - \eta _ { L } ( \theta )$ , so induction gives $[ 1 - \hat { \eta _ { L } } ( \theta ) ] ^ { R }$ Coupling and Bernoulli inequalities prove the bounds, without i.i.d. rounds or equal conditional states. □

For this asymptotic schedule, $\tau _ { * }$ is a dimensionless accumulated scale $R | \theta | ^ { L }$ , separate from the scenario clock used in the control experiments. Combine the theorems at the scale

$$
\mathcal { G } _ { L } = - i \alpha _ { L } [ \overline { { { H } } } , \cdot ] , \qquad \tau _ { * } = \frac { \pi } { 8 \alpha _ { L } } ,
$$

$$
R _ { * } ( \theta ) = \left\lfloor \frac { \tau _ { * } } { | \theta | ^ { L } } \right\rfloor , \qquad \theta \neq 0 .\tag{45}
$$

Theorem 7 (Same-instrument operational separation). For every fixed odd $L \geq 3$ , as $\theta  0 ^ { \pm }$ , convergence in diamond norm holds:

$$
\Lambda _ { L } ( \theta ) ^ { R _ { * } ( \theta ) } \longrightarrow \mathcal { U } _ { \pm } : = \mathrm { A d } _ { \exp ( \mp i \pi \overline { { H } } / 8 ) } ,\tag{46}
$$

while

$$
\Delta _ { \mathrm { r e c } } ( \theta , R _ { * } ( \theta ) ) = \Theta _ { L } ( | \theta | ^ { L } ) \longrightarrow 0 .\tag{47}
$$

In the unhalved diamond norm,

$$
\| \mathcal { U } _ { \pm } - \mathrm { i d } \| _ { \diamond } = \sqrt { 2 } .\tag{48}
$$

Order of limits. Fix finite odd L before $\theta  0 ^ { \pm }$ and $R _ { * } ( \theta ) \to \infty$ . The constants $A _ { L } , \delta _ { L } , \kappa _ { L } , g _ { L }$ and the validity neighborhoods depend on $L ;$ the schedule $R _ { * } ( \theta ) =$ $\Theta _ { L } ( | \theta | ^ { - L } )$ describes small-angle accumulation at fixed code size.

Equation (47) concerns logical inputs at fixed θ. Sepa rately, proposition 1 establishes exact blindness to the sign of θ for every history length. Scalar syndrome frequencies can still reveal its magnitude or presence.

The proof is given in Supplemental Material, Sec. A 2.

## IV. RECOVERY FROM THE SAME SYNDROME RECORD

A syndrome record can guide recovery even when it cannot distinguish particular encoded inputs. We now allow an additional logical correction after the declared physical recovery, while keeping the same syndrome measurements and known signed θ. This changes the controlled channel, not the original instrument’s conventions. No controller receives the encoded input. Established recovery theory gives an operational consequence of the exact instrument structure.

## A. Removing branch phases leaves a parity measurement

Write $K _ { s } = U _ { s } \sqrt { F _ { s } }$ in the common logical-X eigenbasis, choosing the polar factor’s unitary extension diagonal also on any kernel (with arbitrary phases on zero eigenvalues). Applying $V _ { s } = U _ { s } ^ { \dagger }$ leaves the positive branch

$$
\begin{array} { l } { { V _ { s } K _ { s } = \sqrt { q _ { s } I + b _ { s } P } } } \\ { { \ = \sqrt { p _ { s } ^ { + } } \Pi _ { + } + \sqrt { p _ { s } ^ { - } } \Pi _ { - } , \qquad \Pi _ { \pm } = \displaystyle \frac { I \pm P } { 2 } . } } \end{array}\tag{49}
$$

These extra diagonal actions commute with the parity efects, so the syndrome-history laws are unchanged. This is standard conditional polar recovery. Gregoratti and Werner state its optimal channel fidelity for a fixed purebranch instrument [27]. In dimension four the entanglement fidelity is

$$
F _ { e } ( \{ A _ { s } \} ) = \frac { 1 } { 1 6 } \sum _ { s } | \operatorname { T r } A _ { s } | ^ { 2 } .\tag{50}
$$

The trace-norm bound $| \operatorname { T r } ( V _ { s } K _ { s } ) | \le \operatorname { T r } \sqrt { F _ { s } }$ immediately shows optimality among syndrome-conditioned unitary corrections. The average pure-state infidelity is $4 ( 1 -$ $F _ { e } ) / 5$ [40]; both quantities measure fidelity loss.

The specific consequence here is that all remaining backaction lies in the single parity measurement. Define

$$
\mathcal { A } _ { L } ( \theta ) = \sum _ { s } \sqrt { p _ { s } ^ { + } ( \theta ) p _ { s } ^ { - } ( \theta ) } .\tag{51}
$$

The corrected channel preserves every within-parity matrix element and multiplies every cross-parity element by $A _ { L } ( \theta )$ . Since both parity sectors have dimension two,

$$
F _ { e } ^ { \mathrm { p o l a r } } ( \theta , R ) = \frac { 1 + \mathcal { A } _ { L } ( \theta ) ^ { R } } { 2 } .\tag{52}
$$

Thus every state within one parity sector is preserved exactly, although its preparation label remains unidentifiable from the record. Across sectors, the overlap quantifies the disturbance left after removing reversible phases. This expression applies established recovery theory to the exact parity efect algebra.

## B. Information and action classes of the comparisons

For $\begin{array} { r } { K _ { s } = \sum _ { c } \kappa _ { s , c } X _ { c } , } \end{array}$ the best additional logical Pauli for one-round $F _ { e }$ chooses c maximizing $| \kappa _ { s , c } | ^ { 2 }$ in each branch. This is the two-logical-qubit version of coherenceaware Pauli selection [18]. The polar-versus-Pauli comparison includes the additional arbitrary-angle logical actuation available to polar recovery.

For a comparison allowing syndrome-independent terminal logical control, write $\begin{array} { r } { \dot { C } _ { x y } = \sum _ { s } z _ { s } ( \bar { x } ) z _ { s } ( y ) ^ { * } } \end{array}$ for the original Schur channel in the logical-X basis. Let $C ^ { \circ R }$ denote entrywise powers. Every single syndromeindependent terminal unitary V satisfies

$$
F _ { e } ( V \Lambda _ { L } ^ { R } ) \leq \frac { \lambda _ { \operatorname* { m a x } } ( C ^ { \circ R } ) } { 4 } .\tag{53}
$$

Indeed only $\begin{array} { r } { \begin{array} { r c l } { v _ { x } } & { = } & { V _ { x x } } \end{array} } \end{array}$ enters the trace, and $F _ { e } \ =$ $v ^ { \top } C ^ { \circ R } v ^ { * } / 1 6 \leq \lambda _ { \operatorname* { m a x } } ( \bar { C } ^ { \circ R } ) \| v \| ^ { 2 } / 1 6$ , with $\| v \| ^ { 2 } \leq 4$ . This bound includes every terminal unitary and fixed diagonal corrections distributed over the rounds. The optimization is restricted to the stated control class; arbitrary interleaved noncommuting controls and syndrome-dependent terminal corrections require a larger class.

That last distinction is substantive. All $K _ { s }$ and $V _ { s }$ commute, so for a complete history s the product $\prod _ { r } V _ { s _ { r } }$ can be postponed until the end of storage, leaving the same product of positive branches. Per-round extra logical actuation is therefore unnecessary for final-memory fidelity in this model. The original minimum-weight physical recoveries remain per round. A terminal policy retains the history or suficient syndrome counts and accurately accumulates calibrated phases; its protection is assessed at the final memory endpoint.

For every integer precision $b \geq 1$ and storage horizon $R \in \mathbb { Z } _ { \geq 0 }$ , assume accurate accumulation of the calibrated branch phases and an ideal terminal logical action. $\mathrm { A }$ single terminal nearest-bin phase correction, using a uniform ${ \bar { 2 } } ^ { b } .$ -point grid for each accumulated eigenphase, then has the rigorous bound

$$
F _ { e } ^ { \mathrm { t e r m i n a l , b } } ( \theta , R ) \geq \cos ^ { 2 } \left( \frac { \pi } { 2 ^ { b } } \right) F _ { e } ^ { \mathrm { p o l a r } } ( \theta , R ) .\tag{54}
$$

For each history, the positive branch amplitudes $a _ { x }$ acquire phase errors $| \delta _ { x } | \leq \pi / 2 ^ { b }$ Since $b \geq 1$ makes $\bar { \cos ( \pi / 2 ^ { b } ) } \ge 0$ , we may square the nonnegative lower bound $| \sum _ { x } a _ { x } e ^ { i \delta _ { x } } | \geq \cos ( \pi / 2 ^ { b } ) \sum _ { x } a _ { x }$ and sum histories to prove the result. At $b = 1$ the bound is trivial, and at $R = 0$ the empty-history action is identity. The result bounds phase quantization under the specified control assumptions; physical gate synthesis and long-history sampling require additional analysis.

Finally, the efects $F _ { s }$ alone do not determine $V _ { s }$ . The physical angles $\theta$ and $- \theta$ have identical syndrome laws but opposite branch phases. Implementing the specified phase-canceling correction therefore requires signed-angle information beyond this passive record in general. Pauli recovery and sign-robust control can still improve recovery, and degenerate angles can make the sign irrelevant. The next example concerns the diferent task of identifying the stored input, which remains impossible even when recovery is available.

## C. An unidentifiable record-only target

We now specify a decision task, keeping noise detection separate from input-dependent consequences. A preparation device selects one of two known logical pure states with equal prior probability and withholds the preparation label from a monitor. The monitor knows $L , \theta .$ , the instrument, and the number of rounds, and receives only the complete syndrome history. It receives neither a final logical measurement nor any side information about the label. In the simultaneous logical-X basis, choose

$$
| a \rangle = | + + \rangle , \qquad | b \rangle = \frac { | + + \rangle + | - - \rangle } { \sqrt { 2 } } .\tag{55}
$$

Both have parity $P = + 1$ . Their history laws are exactly equal by proposition 1, so every record-only classifier has success probability $1 / 2$ under equal priors.

Define expected memory loss relative to the prepared state by

$$
\ell _ { j } ( \theta , R ) = 1 - \langle j | \Lambda _ { L } ( \theta ) ^ { R } ( | j \rangle \langle j | ) | j \rangle , \qquad j \in \{ a , b \} .\tag{56}
$$

Every $K _ { s }$ acts as a scalar on |a⟩, hence $\ell _ { a } = 0$ exactly. On the schedule in equation (45), theorem 7 gives

$$
\ell _ { b } ( \theta , R _ { * } ( \theta ) ) \longrightarrow 1 - \left| \frac { e ^ { - i \pi / 4 } + e ^ { i \pi / 4 } } { 2 } \right| ^ { 2 } = \frac { 1 } { 2 } .\tag{57}
$$

For all suficiently small nonzero $\theta ,$ deciding whether the prepared state has zero expected loss or expected loss exceeding $1 / 4$ therefore has record-only error probability $1 / 2$ . The statement concerns state-fidelity loss averaged over the experiment. It does not require the two candidate preparations to be orthogonal.

This example exploits exact blindness within a parity sector. The $\bar { \theta } ^ { 2 L }$ onset is needed for the stronger statement in theorem $\mathrm { 7 } \colon$ even the most distinguishable pair of inputs, including opposite parity sectors, becomes indistinguishable on the drift schedule. A known preparation label, additional logical probes, or a diferent instrument changes the decision problem.

## V. FINITE-ANGLE VALIDATION AT $L = 3$

## A. Record and channel separation

We enumerate all $2 ^ { 1 8 } = 2 6 2 1 4 4$ physical X supports, assign their complete syndrome and minimum-weight recovery with the stated tie rule, and count supports by syndrome, residual logical class, and weight. Evaluating the finite trigonometric sums gives the finite-angle channel directly by exhaustive support enumeration. Integer counts reproduce $[ \theta ^ { 6 } ] B _ { 0 , P } = \dot { - } 2 1 / 1 6$ and the leading channel coeficient $\alpha _ { 3 } = 3 / 4$ . The enumeration procedure and numerical checks are described in Supplemental Material, Sec. L.

Figure S3 compares the one-round response with the repeated-round schedule. To avoid identifying a state distance with a channel norm, we display the explicitly computed lower bound

$$
d _ { b } ( \theta , R ) : = \| \Lambda _ { 3 } ( \theta ) ^ { R } ( | b \rangle \langle b | ) - | b \rangle \langle b | \| _ { 1 } \leq \| \Lambda _ { 3 } ( \theta ) ^ { R } - \mathrm { i d } \| _ { \diamond } .\tag{58}
$$

The history curve evaluates the rigorous upper bound min $\{ 1 , R _ { * } \eta _ { 3 } ( \theta ) \}$ }. For $\theta \ = \ 0 . 0 5 , \ R _ { * } \ = \ 4 1 8 8$ , and the calculation gives

$$
d _ { b } \simeq 1 . 3 7 9 3 4 , \qquad \Delta _ { \mathrm { r e c } } ( \theta , R _ { \ast } ) \le R _ { \ast } \eta _ { 3 } ( \theta ) \simeq 0 . 0 0 1 6 2 7 2 1 .\tag{59}
$$

Thus the worst-case equal-prior record classification success is at most approximately 0.500814, while the corrected channel has order-one unhalved diamond displacement. For the within-parity pair in equation (55), success remains exactly $1 / 2$ and the superposition state |b⟩ has expected loss approximately 0.504346.

![](images/95ffe2868cb5c3303cae6ec69a77d896123d3da378c99d33868e1fa1fef5c07c.jpg)

![](images/ebc389a326688b4ae5ce766789d174b62d357e01fdd1dab1a902eb1f828e783b.jpg)  
FIG. S3. Finite-angle separation for the 18-qubit toric instrument of Sec. I. (a) The channel carrier magnitude and exact maximal one-round record total-variation (TV) distance approach orders $\theta ^ { 3 }$ and ${ \bf \dot { \theta } } ^ { 6 }$ , respectively. The dotted curve is only an exponent guide. (b) An evaluated input-state trace distance lower-bounds the unhalved channel diamond distance, while $R _ { * } \eta _ { 3 }$ upper-bounds the maximal history TV. The history curve bounds the distance from above. All solid curves are exhaustive support-sum evaluations at 48 angles.

The finite-angle evaluations use exhaustive floatingpoint sums with independent 60-digit checks; formal interval certification remains separate. They demonstrate a regime for this 18-qubit instrument defined above, without extrapolation to larger codes or faulty extraction circuits. In contrast, the global analytic remainder constant is already $\kappa _ { 3 } \simeq 4 . 8 \bar { 1 } \times 1 0 ^ { 1 5 }$ and is used only to prove the fixed-code asymptotic limit.

## VI. FROM EXACT INFORMATION TO JUSTIFIED RECOVERY UPDATES

The exact calculation identifies recovery-relevant information missing from the passive record and motivates the signed-calibration construction below. Three steps connect it to the experiments that follow.

First, select the decision target. The leakage $\eta _ { L } ( \theta )$ compares logical inputs at a fixed physical angle. The $\mathrm { o r d e r } { \bf - } \theta ^ { 2 L }$ onset describes logical-input sensitivity of the instrument; the recovery loss separately quantifies staleupdate consequences. The record-only memory-loss example in Sec. IV C instead makes a particular target unidentifiable even at unlimited history length. Neither result prevents improved recovery of an unknown stored state.

Second, specify what distinguishes the noise conditions relevant to an action. Proposition 1 makes +θ and −θ exactly equivalent under the passive experiment. Section IV shows how syndrome-conditioned phases informed by signed calibration can reduce infidelity. A larger predictor cannot extract the absent sign from identical laws.

The relevant remedy is an observation that changes the equivalence class, or an action beneficial throughout that class. The supporting sentinel study implements a separate signed probe as one possible additional observation channel. The main demonstration instead uses a terminal logical measurement on known encoded states in the same toric instrument. Active code probes and control dithering ofer alternative observation experiments.

Third, compare the proposed recovery with the incumbent over every deployment condition still compatible with the evidence. A noise point estimate need not be identifiable if 1 action is uniformly beneficial; conversely, a precisely measured past angle need not justify a current update. The evaluator uses a confidence set propagated to deployment time and a bound on $D _ { u }$ computed separately from the proposer.

The proof and experiment regimes remain distinct. The theorem fixes finite odd L before $\theta ~  ~ 0$ with $R _ { * } ( \theta ) = \lfloor \tau _ { * } / \vert \theta \vert ^ { L } \rfloor$ . The finite-angle recovery demonstration includes $\theta = 0 . 0 5 , R = 4 1 8 8$ , and precision-limited controls. The calibration robustness tests instead fix $L = 3 , H = 3 0 0$ , and fixed phase tables, with calibrationangle magnitudes in [0.08, 0.12]. These use the same underlying rotation, logical basis, and physical Pauli recovery, but evaluate diferent declared operating conditions. Direct finite-angle channel calculations determine the robustness results. Commutativity allows terminal phase application in both tiers, making deferred recovery available to every comparator.

The measurement, certification, and activation workflow is shown in Supplemental Fig. S17.

## VII. MEASUREMENT ASSUMPTIONS AND CONDITIONAL ACCEPTANCE

## A. What must the observations distinguish?

Proposition 1 establishes sign invariance for the passive record at every angle and every history length. This is separate from Theorem 4, which compares logical inputs at a fixed angle. The following acceptance construction uses the sign equivalence and the recovery risks in Sec. I C; logical-input sensitivity and calibration failure risk are distinct quantities.

Let $[ \theta ] _ { \mathcal { O } }$ denote the parameter conditions with the same observation law under an allowed experiment O. A deterministic update is uniformly beneficial on that class only if

$$
\operatorname* { s u p } _ { \theta ^ { \prime } \in [ \theta ] _ { \mathcal { O } } } D _ { u } ( \theta ^ { \prime } ) < 0 .\tag{60}
$$

For the declared deterministic action catalog, the operational obstruction is an empty intersection of beneficial action sets across the indistinguishable conditions. Randomized policies enlarge the action class and require a separate risk calculation, including how randomization is composed over repeated rounds. Parameter nonidentifiability alone is insuficient: an action may improve both signs, and the detailed finite-horizon risk matters. A learned model cannot resolve identical observation laws by increasing its capacity, but changing the allowed observations can remove the obstruction.

As one additional observation channel, we add a sentinel prepared in |0⟩, exposed to the nominal $R _ { X } ( \theta )$ rotation, and measured in the Y basis. With error-free preparation and readout, the mean is − sin $\theta ;$ a direct matrix-exponential test checks this sign. For readout contrast a and ofset $b ,$ the measurement model is

$$
\mathbb { E } [ Y ] = b - a \sin \theta , \qquad a \in [ 0 . 9 8 , 1 ] , \quad b \in [ - 0 . 0 0 1 , 0 . 0 0 1 ] .\tag{61}
$$

The code angle can difer from the sentinel angle by at most $\eta = 0 . 0 0 1$ rad. This transfer radius η is distinct from the logical-input contrast $\eta _ { L } ( \theta )$ . The generative readout uses $a = 0 . 9 9 , b = 0$ . The readout and transfer bounds are declared inputs to the simulation. If b is unrestricted, a sign change can be concealed by an ofset change. Correct identification of an acquisition does not establish that its readout and transfer assumptions describe the device. Under the stated readout and transfer assumptions, the channel distinguishes recovery-relevant signs; finite-shot and nuisance uncertainty determine whether that information sufices for certification.

A known-state encoded probe or active syndrome dither could supply diferent information. The implemented sentinel supplies an additional observation channel. In particular, active self-calibration changes the observation experiment; its positive results are compatible with a passive sign ambiguity [10].

## B. Record integrity and the threat model

A calibration record contains the measurements used to estimate noise and their acquisition time. The allowed recovery operations form a fixed catalog. Certification means that the evaluator bounds the proposed operation’s excess infidelity below −δ throughout the deployment confidence set. The software checks below preserve the association between those measurements, the experiment, and the operation actually applied.

The evaluator owns an immutable registry containing an acquisition identifier, workload identifier, request nonce, plus-outcome count, shot count, and acquisition time. The candidate is an integer indexing the fixed table. The evaluator rejects requests with unknown records, mismatched workload or nonce identifiers, future timestamps, malformed actions, or exhausted budgets. The proposer can select an action or evidence identifier and can request a new probe acquisition within its budget. It cannot insert records, alter timestamps, change the reference channel, or activate a table directly (Fig. S17).

The attacker can supply misleading recommendations, including an instruction to ignore probe sign, or induce the use of authentic old evidence. Separate tests submit mismatched workload and nonce identifiers. The primary physical drift remains inside a publicly declared rate bound; an explicit out-of-bound challenge tests that premise. The registry’s trust boundary assumes an uncompromised process, reliable acquisition electronics, and physical evolution within the declared model. The prototype implements control of evidence creation and modification within one evaluator process; cryptographic authentication, attestation, and post-quantum deployment protocols are separate implementation tasks.

## C. From finite measurements to a deployment set

Let k plus outcomes be observed in n independent stationary sentinel shots. Form a two-sided Clopper– Pearson interval $[ p _ { - } , p _ { + } ]$ for the Bernoulli probability [28]. With $m = 2 p - 1$ invert Eq. (61) through all allowed $^ { a , }$ b to obtain a calibration-angle interval $\bar { C } _ { \mathrm { c a p } } .$ . The sine response is one-to-one on the domain $| \theta | \leq 0 . { \overset { } { 1 8 } }$ for fixed contrast and ofset. Invert the probability interval at every corner of the allowed contrast–ofset rectangle. Take the smallest angle interval containing the feasible endpoints and intersect it with the physical domain. An empty interval is a model-inconsistency rejection.

For evidence age A and declared angular drift rate $v ,$ the stationary-deployment set is

$$
C _ { \mathrm { d e p } } = C _ { \mathrm { c a p } } \oplus [ - ( \eta + v A ) , \eta + v A ] .\tag{62}
$$

The evaluator rejects a proposal if its deployment region extends beyond the supported angle domain. Define a bound $U _ { u }$ over this entire set and accept only when

$$
U _ { u } \ge \operatorname* { s u p } _ { \theta \in C _ { \mathrm { d e p } } } D _ { u } ( \theta ) , \qquad U _ { u } \le - \delta , \quad \delta = 1 0 ^ { - 3 } .\tag{63}
$$

The reference is independent of the model used to propose u. Learned uncertainty guides acquisition, while the confidence set supplies the activation criterion. Rejection leaves the incumbent in place; a workflow can subsequently probe again if a separately budgeted acquisition is available.

Proposition 8 (Conditional protection of accepted updates). Consider a finite declared family of acquisition looks j, with simultaneous deployment sets whose failure probabilities are at most $\alpha _ { j }$ and $\textstyle \sum _ { j } \alpha _ { j } \ \leq \alpha$ . Suppose the physical drift, readout, and transfer premises hold; each computed bound is valid for every allowed action; and the activated action is exactly the evaluated action. Then, for any proposer selecting actions and available looks adaptively,

$$
\operatorname* { P r } \{ \exists ~ a c c e p t e d ~ ( j , u ) : D _ { u } ( \theta _ { j } ) > - \delta \} \leq \alpha .\tag{64}
$$

Proof. On the event that every deployment condition lies in its set, Eq. (63) implies $D _ { u } ( \theta _ { j } ) \leq U _ { u } \leq - \delta$ for every admitted action. This event holds uniformly over the action catalog and therefore also for a data-dependent choice from it. Its complement has probability at most $\textstyle \sum _ { j } \alpha _ { j }$ by the union bound. □

For histories that determine a new acquisition policy, the corresponding coverage must hold conditionally on that history, or a simultaneous construction must cover all permitted choices. The experiments use predetermined looks or a budget selected before independent newly acquired measurements. Repeated queries to the same simultaneous physical set do not require a new failureprobability allocation; new acquisition looks do. Acquisition is restricted to the declared, statistically budgeted looks.

Proposition 8 is a standard confidence-set argument applied to this QEC contract. It bounds the unconditional margin-violation probability over the declared experiment family. The protected quantity is improvement relative to the incumbent; an absolute logical-risk ceiling would require an additional criterion. The analytical statement requires a valid numerical enclosure. The implementation supports that requirement with a tested $1 0 ^ { - 8 }$ allowance; the deployment scope is discussed in Supplemental Material, Sec. T. Supplemental Material, Sec. N specifies this distinction.

## D. Age, acquisition duration, and deployment duration

Equation (62) assumes stationary capture and a stationary deployment block. Finite capture duration, clock error, within-block drift, and correction cost need additional allowances. For time-dependent angles, a conservative channel-distance argument gives the excess-risk allowance

$$
B _ { \mathrm { p a t h } } = N v \Delta H ( H + 1 ) ,\tag{65}
$$

when $| \theta _ { r } - \theta _ { \mathrm { c a p } } | \leq v r \Delta$ . Supplemental Material, Sec. O derives it and reports an 81-rate sweep with direct composition of changing channels. The extension incorporates execution duration into the freshness criterion under the stipulated physical bounds.

## VIII. ROBUST ACCEPTANCE UNDER MISLEADING AND STALE ADVICE

## A. Mechanism stress tests and baselines

We compare authorization alone, authorization with a fixed time-to-live (TTL), confidence without age propagation, and the full confidence-and-freshness rule. All gates use the same simulated episodes, counts, proposed actions, and incumbent. Authorization verifies that the record was issued by the evaluator and matches the workload and request; the risk certificate assesses physical usefulness. The confidence-only rule includes readout and transfer uncertainty but treats capture evidence as current. Fixed TTLs are 0, 0.1, 0.3, 1, 3 in units $T _ { 0 }$ of this supporting scenario. No gate and the complete TTL results remain in the research record.

Here the label fresh denotes the acquisition-time condition, before the tested staleness or delay; it does not by itself certify an update. Each of 7 cases contains 200 independent simulated noise paths, with capture magnitude uniform on [0.08, 0.12], random sign, 8192 probe shots, and $H = 3 0 { \dot { 0 } }$ . The proposer maps measurement-based advice to the fixed catalog; hidden angles are used only to evaluate loss and configure the physical challenge. The selected conditions make honest proposals reduce infidelity, so rejection has a measurable opportunity cost. These mechanism stress tests measure harm and acceptance within the specified challenge distributions.

In the bounded-stale case, the true sign reverses after age $0 . 5 T _ { 0 }$ under the public bound $v = 0 . 6 ~ \mathrm { r a d } / T _ { 0 }$ . The largest required change is 0.24 rad, below the permitted 0.3 rad. Benign delay instead uses age $2 T _ { 0 } , v = 0 . 0 0 1$ $\mathrm { r a d } / T _ { 0 }$ , and no change in the noise angle. These cases have diferent public drift contexts. A globally fixed expiry can sacrifice utility across diferent drift contexts; contextspecific tuning is a relevant alternative. The drift-bound violation challenge reverses the sign after age $0 . 0 1 T _ { 0 }$ while declaring $v = 0 . 0 0 1$ rad $/ T _ { 0 }$ , and consequently lies outside the guarantee.

Confidence alone blocks misleading fresh recommendations but admits 178 harmful stale updates. The full rule admits none in that bounded-stale case (Table S2). It accepts 190 honest fresh updates and 168 under benign delay, all beneficial in the numerical channel evaluation. The pointwise acceptance proportions are 95% and 84%; the latter’s two-sided 95% binomial interval is approximately [78.2%, 88.8%]. The interval’s lower endpoint falls below the study’s stated 80% acceptance target in this regime. Zero observed harms in 200 independent trials of one fixed challenge distribution gives a one-sided 95% binomial upper limit of approximately 1.49% for that distribution. It is not a bound for arbitrary attacks or diferent workload distributions.

TABLE S2. Accepted/harmful updates out of 200 paths per constructed mechanism stress test. Harm means positive excess infidelity from numerical evaluation of the exact channel. Both confidence-only and full rules include authorization. The drift-bound violation violates the drift premise.
<table><tr><td>Case</td><td>Authorization Confidence</td><td></td><td>Full</td></tr><tr><td>Fresh honest</td><td>200/0</td><td>190/0</td><td>190/0</td></tr><tr><td>Wrong advice</td><td>200/200</td><td>0/0</td><td>0/0</td></tr><tr><td>Bounded stale</td><td>200/200</td><td>178/178</td><td>0/0</td></tr><tr><td>Benign delay</td><td>200/0</td><td>182/0</td><td>168/0</td></tr><tr><td>Wrong workload</td><td>0/0</td><td>0/0</td><td>0/0</td></tr><tr><td>Wrong nonce</td><td>0/0</td><td>0/0</td><td>0/0</td></tr><tr><td>Drift bound violated</td><td>200/200</td><td>187/187 187/187</td><td></td></tr></table>

The number of accepted updates does not specify the size of their recovery benefit. Define retained recovery gain relative to authorization-only deployment on the same honest paths:

$$
G = \frac { \sum _ { i } [ \mathcal { R } ( \theta _ { i } , u _ { 0 } ) - \mathcal { R } ( \theta _ { i } , u _ { \mathrm { f u l l , i } } ) ] } { \sum _ { i } [ \mathcal { R } ( \theta _ { i } , u _ { 0 } ) - \mathcal { R } ( \theta _ { i } , u _ { \mathrm { a u t h , i } } ) ] } .\tag{66}
$$

Rejected updates use $u _ { 0 }$ in the numerator. The full gate retains 97.1% of fresh-evidence gain, with paired pathbootstrap 95% interval [94.9%, 98.9%], and 90.8% under benign delay, with interval [86.8%, 94.2%]. This is the mechanism study’s utility result: rejecting uncertain updates preserves most of the aggregate improvement in these tested distributions. The retained fraction depends on the workload distribution.

## B. Failure when the drift bound is exceeded

The out-of-bound step causes 187 harmful activations out of 200 under the full rule, exactly as under confidenceonly acceptance. The acquisition can be correctly authorized while its physical interpretation is wrong. The extended readout, transfer, and age sweeps in Supplemental Material, Sec. C expose this same boundary. They report 5400 evaluations of condition–replicate pairs satisfying the assumptions with no observed harmful activation and 4767 harmful activations among 22600 evaluations of condition–replicate pairs violating the assumptions. Reused count blocks make these totals unsuitable as independent binomial trials. These failures locate the physical boundary of the guarantee.

## IX. RECOVERY IMPROVEMENT, WORKFLOW COMPLETION, AND TRANSFER

The acceptance tests establish what evidence supports an action. We next assess whether the controllers complete calibration and activation workflows that improve recovery. Detailed workflows and all secondary outcomes remain in Supplemental Material, Secs. D–F.

The controller studies distinguish beneficial recovery from workflow completion. A beneficial completion deploys an operation with excess risk below −δ; harmful activation has excess risk above δ. Retention and incomplete workflows remain in the denominators. The expanded multistep evaluation in Supplemental Material, Sec. T measures beneficial completion and calibration cost under misleading advice, limited initial data, stale evidence and identity failures. Its matched deterministic comparator receives the same observations and permitted actions. Early interface pilots are summarized in Sec. D.

## A. Learned recovery and a Bayesian comparison

Learned quantum control also has hardware implementations: Xu et al. [5] implement a neural predictor of pulse parameters on a field-programmable gate array (FPGA). This connects learned control decisions to their implementation cost; the recovery comparison below evaluates predictors through the same calibrated acceptance rule.

The 5 gated recurrent units (GRUs) and 5 multilayer perceptrons (MLPs) are trained on signed-probe histories, with whole simulated noise paths and parameter blocks separated across training, development, and test. The learned and classical references have the same observations, recovery catalog, and acquisition budgets. Across 600 held-out paths, the guarded GRU’s mean infidelity is 0.088018, compared with 0.225490 for retaining the incumbent and 0.089296 for a development-tuned Bayesian filter. The paired 95% interval [−0.002675, +0.000070] leaves the neural-versus-Bayesian comparison inconclusive. Both the learned predictor and Bayesian filter improve recovery relative to retention.

These models receive signed-probe information; the passive-record impossibility follows analytically from identical observation laws for every postprocessing architecture. In a separate acquisition study with fixed checkpoints, Bayesian allocation reduces mean newly acquired calibration shots by 19.7% while mean infidelity increases from 0.0365583 to 0.0372372. Learned allocation traces a similar cost–risk frontier. Supplemental Material, Sec. E reports the risk–cost tradeof, common historical shot cost, and changed confidence allocation.

![](images/bd92b1e5454fa4718a1631eb72cdfd163e81750886736d89a9d2ee3f04e9500f.jpg)

![](images/3a492794979a53785f4322bf958bc3479f3d9122946b91d7275e2309d2e92877.jpg)  
FIG. S4. Harm and utility in paired mechanism stress tests. (a) Harmful-activation fractions and pointwise exact $9 5 \%$ binomial intervals, 200 paths per case; the shaded case violates the drift premise. (b) The same stale-harm and benign-delay acceptance endpoints for the full rule and fixed expiry, with pointwise intervals on both axes. Time-to-live (TTL) values 0, 0.1, and 0.3 coincide. The benign-delay and bounded-stale challenges use diferent declared drift contexts.

## B. Acceptance at calibration in the profile-changing circuit challenge

A separate stochastic surface-code study uses Stim and PyMatching at distances 3 and 5 with 30 rounds and paired scoring of matching priors on the same records [15, 32]. All 400 candidates at the calibration noise settings pass the calibration rule; simultaneous confidence intervals for both memory-basis risks of candidate and incumbent establish benefit in 346 cases, leaving 54 unresolved. Under stale deployment, all 400 individual efects remain unresolved. The stale gate rejects every candidate because its valid bound is vacuous, so this profile-changing challenge supports acceptance at the calibration settings. These stochastic Pauli circuits form a separate test of acceptance; their noise model difers from the coherent toric instrument. Supplemental Material, Sec. F reports the pilot, confirmation, and limitations of these results.

## Supplemental Material A: Channel proofs and the general privacy reference

## 1. Proof of the corrected-channel expansion

Proof. Each $\theta ^ { m }$ coeficient of $\kappa _ { s , c }$ is $( - i ) ^ { m }$ times a real number. Let $w ( s ) = \operatorname* { m i n } \{ | a | : { \widetilde { H } } _ { Z } a = s \}$ . A nonidentity class cannot occur below max $\{ w ( s ) , L - w ( s ) \}$ . Since L is odd, a product of two nonidentity Kraus coeficients starts at order at least $L + 1$ . Trace preservation makes the identity–identity channel coeficient equal to $1 + O _ { L } \big ( \theta ^ { L + 1 } \big )$ Thus order-L terms only pair identity and nonidentity amplitudes of orders $w ( s )$ and $L - w ( s )$ . Their purely imaginary common phase makes a commutator, not a dissipator.

Every nontrivial weight-L cycle is one of $L$ straight representatives of class (1, 0) or (0, 1); the product class has minimum weight 2L. Row or column projection shows that for A on such a line with $| A | > L / 2$ , its complement is the unique global minimum recovery for its boundary. Each line therefore gives the odd repetition-code majority sum

$$
- \frac { i } { 2 ^ { L } } { \binom { L - 1 } { ( L - 1 ) / 2 } } ,\tag{A1}
$$

using $\begin{array} { r } { \sum _ { k = ( L + 1 ) / 2 } ^ { L } ( - 1 ) ^ { k } \binom { L } { k } = ( - 1 ) ^ { ( L + 1 ) / 2 } \binom { L - 1 } { ( L - 1 ) / 2 } } \end{array}$ . Multiplying by L lines per direction yields equations (41) and (42). Iverson and Preskill already obtained this contribution from each logical line and the magnitude of the summed coeficient, up to logical labels and signs, in this setting [21]; their Appendix H also identifies weight-2L double-logical paths. We reproduce the established onset and coeficient in common conventions to compare the channel with its efects. Related repetition-code reductions and minimum-logical-support perturbative expansions appear in coherent-error analyses [17, 41, 42].

Finally, each channel class-matrix coeficient is a finite trigonometric polynomial whose integral Taylor remainder is bounded by $( 2 L ^ { 2 } ) ^ { L + 1 } 2 ^ { 4 L ^ { 2 } } | \theta | ^ { L + 1 } / ( L + 1 ) !$ . The 16 left– right logical-X carriers have diamond norm one because unitary multiplication preserves singular values, giving equation (43); Supplemental Material, Sec. J gives the order squeeze and line projection. □

## 2. Proof of the nonresonant operational separation

Proof. Put $h \ : = \ : | \theta | ^ { L }$ and $\sigma = \operatorname { s g n } ( \theta )$ . By theorem $5 ,$ $\Lambda _ { L } ( \theta ) = \mathrm { i d } + \sigma h \mathcal { G } _ { L } + E _ { L } ( \theta )$ . Compare one round with $e ^ { \sigma h \dot { \mathcal { G } } _ { L } }$ . The integral exponential remainder and equation (43) give

$$
\begin{array} { r } { \| \Lambda _ { L } ( \theta ) - e ^ { \sigma h \mathcal { G } _ { L } } \| _ { \diamond } \leq \kappa _ { L } | \theta | ^ { L + 1 } + \frac { 1 } { 2 } h ^ { 2 } g _ { L } ^ { 2 } e ^ { h g _ { L } } , \ g _ { L } = \| \mathcal { G } _ { L } \| _ { \diamond } . } \end{array}\tag{A2}
$$

Both maps are channels. A mixed telescoping sum over $R = \lfloor \tau / h \rfloor$ rounds, together with the error from rounding the round count down to an integer, gives, uniformly for $0 \leq \tau \leq T$

$$
\begin{array} { r l r } {  { \bigg \| \Lambda _ { L } ( \theta ) ^ { \lfloor \tau / h \rfloor } - e ^ { \sigma \tau \mathcal { G } _ { L } } \bigg \| _ { \diamond } } } \\ & { } & { \leq \kappa _ { L } T | \theta | + \frac { T } { 2 } g _ { L } ^ { 2 } e ^ { g _ { L } h } h + g _ { L } h . } \end{array}\tag{A3}
$$

The right side vanishes at fixed $L , T ; \ \tau \ = \ \tau ,$ <sub>∗</sub> proves equation (46).

By theorem 4, fixed-L constants $A _ { L } , \delta _ { L } > 0$ satisfy $\eta _ { L } ( \check { \theta } ) \leq A _ { L } | \theta | ^ { 2 L }$ for $| \theta | \leq \delta _ { L }$ . Hence

$$
\begin{array} { r } { \Delta _ { \mathrm { r e c } } ( \theta , R _ { * } ) \leq R _ { * } \eta _ { L } ( \theta ) \leq A _ { L } \tau _ { * } | \theta | ^ { L } , } \end{array}\tag{A4}
$$

proving the upper bound in equation (47). The matching lower bound is established below in Sec. A 3.

For unitary channels, the unhalved norm obeys

$$
\begin{array} { r } { \| \mathrm { A d } _ { U } - \mathrm { A d } _ { V } \| _ { \diamond } = 2 \big [ 1 - \nu ( U ^ { \dagger } V ) ^ { 2 } \big ] ^ { 1 / 2 } , } \end{array}\tag{A5}
$$

where $\nu ( W )$ is the minimum modulus in the convex hull of $W \mathrm { { s } }$ spectrum. With an arbitrary ancilla, the output overlap is ${ \mathrm { T r } } ( \rho U ^ { \dagger } V )$ for some density operator $\rho ;$ these values fill that convex hull, and the pure-state trace-distance formula proves equation (A5). Here H has spectrum $\{ 2 , 0 , 0 , - 2 \} , \mathrm { s o } e ^ { - i \pi \overline { { H } } / 8 }$ has spectrum $\{ e ^ { - i \pi / 4 } , 1 , 1 , e ^ { i \pi / 4 } \}$ and convex-hull minimum modulus $1 / { \sqrt { 2 } } ,$ , giving distance $\sqrt { 2 }$ from identity. □

## 3. A rare syndrome attains the history exponent

Fix odd $L \geq 3$ . Split the simple diagonal staircase in Eq. (32) into its two alternating matchings

$$
E _ { 0 } = \{ h _ { j , j } : j \in \mathbb { Z } _ { L } \} , \qquad E _ { 1 } = \{ v _ { j + 1 , j } : j \in \mathbb { Z } _ { L } \} .\tag{A6}
$$

They have the same syndrome $s _ { * } .$ , supported on the $2 L$ distinct vertices of the dual cycle. An error edge can meet at most two of these odd vertices, so every support with this syndrome has weight at least L. Both matchings attain that weight. Their symmetric diference has winding (1, 1) and hence logical class $P .$

Let $r _ { s _ { * } }$ be the declared minimum-weight recovery, necessarily of weight L. The diference between it and any weight-L support has even cardinality. For odd L, the logical classes (1, 0) and (0, 1) have odd cardinality, so the leading corrected terms lie only in classes I and P. Let a and b count these weight-L terms. Both are positive: the two matchings difer by P. All weight-L amplitudes have the same phase, and multiplication by the X recovery introduces no Pauli sign. Consequently,

$$
\begin{array} { c } { { K _ { s _ { * } } ( \theta ) = ( - i \theta / 2 ) ^ { L } ( a I + b P ) + o _ { L } ( | \theta | ^ { L } ) , } } \\ { { p _ { s _ { * } } ^ { \pm } ( \theta ) = \displaystyle \frac { ( a \pm b ) ^ { 2 } } { 2 ^ { 2 L } } | \theta | ^ { 2 L } + o _ { L } ( | \theta | ^ { 2 L } ) , } } \\ { { p _ { s _ { * } } ^ { + } ( \theta ) - p _ { s _ { * } } ^ { - } ( \theta ) = c _ { L } | \theta | ^ { 2 L } + o _ { L } ( | \theta | ^ { 2 L } ) , \quad c _ { L } = \displaystyle \frac { 4 a b } { 2 ^ { 2 L } } > 0 . } } \end{array}\tag{A7}
$$

These expansions hold from either angle sign. In a fixed parity sector the history law is a product law. For the event $B _ { R }$ that s<sub>∗</sub> occurs at least once,

$$
| P _ { + } ^ { ( R ) } ( B _ { R } ) - P _ { - } ^ { ( R ) } ( B _ { R } ) | = | ( 1 - p _ { s _ { * } } ^ { - } ) ^ { R } - ( 1 - p _ { s _ { * } } ^ { + } ) ^ { R } | .\tag{A8}
$$

At $R = R _ { * } ( \theta )$ , both $R p _ { s _ { * } } ^ { \pm } = O _ { L } ( | \theta | ^ { L } )$ tend to zero. The mean-value theorem applied to $( 1 - p ) ^ { R }$ therefore gives

$$
| P _ { + } ^ { ( R _ { * } ) } ( B _ { R _ { * } } ) - P _ { - } ^ { ( R _ { * } ) } ( B _ { R _ { * } } ) | = \tau _ { * } c _ { L } | \theta | ^ { L } [ 1 + o _ { L } ( 1 ) ] .\tag{A9}
$$

Total variation is at least the diference on any event. Choosing logical inputs in opposite parity sectors proves the lower bound in $\mathrm { E q . ~ ( 4 7 ) ; E q . ~ ( A 4 ) }$ supplies the upper bound. This result concerns stored-input distinguishability at fixed noise. Exact blindness to the sign of the noise is unchanged.

As implementation checks, enumeration of minimum supports at $L = { \ o 3 } , { \ o 5 } , { \ o 7 } , { \ k 9 } , { \ o 1 1 }$ gives positive counts in both classes, and exhaustive weight-three support enumeration at $L = 3$ agrees with the matching construction. For the checked $L = 3$ syndrome, $a = b = 1$ and $c _ { 3 } = 1 / 1 6$ . The event contrast divided by $| \theta | ^ { 3 }$ is 0.03205, 0.03255, and 0.03268 at $\theta = 0 . 0 4 , 0 . 0 2 , 0 . 0 1$ , approaching $\tau _ { * } / 1 6 = 0 . 0 3 2 7 2 5$ . These finite checks support the all-odd-L argument above.

## 4. The general complementary-channel baseline

The qualitative vanishing in equation (47) already follows from correctability–privacy duality. To apply it with the same norms, put $\dot { \mathcal { N } _ { \theta } } \overset { \cdot } { = } \Lambda _ { L } ( \overset { \cdot } { \theta } ) ^ { R _ { * } ( \theta ) }$ and let s range over complete histories, with $K _ { \mathbf { s } } = K _ { s _ { R _ { * } } } \cdot \cdot \cdot K _ { s _ { 1 } }$ . A dilation and its complementary channel are

$$
W _ { \theta } = \sum _ { \mathbf { s } } K _ { \mathbf { s } } \otimes | \mathbf { s } \rangle , \qquad W _ { \theta } ^ { \dagger } W _ { \theta } = I _ { 4 } ,\tag{A10}
$$

$$
\mathcal { N } _ { \theta } ^ { c } ( \rho ) = \sum _ { \mathbf { s } , \mathbf { u } } \mathrm { T r } ( K _ { \mathbf { s } } \rho K _ { \mathbf { u } } ^ { \dagger } ) | \mathbf { s } \rangle \langle \mathbf { u } | .
$$

Dephasing the history register gives exactly the classical record channel $\begin{array} { r } { \mathcal { M } _ { \theta } ( \rho ) = \sum _ { \mathbf { s } } P _ { \rho , \theta } ^ { ( R _ { * } ) } ( \mathbf { s } ) \vert \mathbf { s } \rangle \langle \mathbf { s } \vert } \end{array}$ . By equation (A3), the unhalved diamond error $\epsilon _ { \theta } = \| \mathcal { N } _ { \theta } - \mathcal { U } _ { \sigma } \| _ { \circ } =$ $O _ { L } ( | \theta | )$ , where $\sigma = \operatorname { s g n } ( \theta )$ . Inverting $\mathcal { U } _ { \sigma }$ is therefore an ϵ<sub>θ</sub>-accurate recovery. Theorem 5 of Kretschmann, Kribs, and Spekkens [24] and contractivity under dephasing give a constant classical channel $\mathcal { C } _ { \theta } ( \rho ) = \mathrm { T r } ( \rho ) \omega _ { \theta }$ satisfying

$$
\| { \mathcal { M } } _ { \theta } - { \mathcal { C } } _ { \theta } \| _ { \diamond } \leq 2 { \sqrt { \epsilon _ { \theta } } } .\tag{A11}
$$

For any pair of inputs, the triangle inequality contributes two such errors, and the factor $1 / 2$ in total variation cancels that factor two:

$$
\Delta _ { \mathrm { r e c } } ( \theta , R _ { * } ) \leq 2 \sqrt { \epsilon _ { \theta } } = O _ { L } ( \sqrt { | \theta | } ) .\tag{A12}
$$

Thus the general bound shows that logical-input distinguishability vanishes in this limit even during nontrivial unitary evolution. The toric efect calculation improves this guarantee to equation (A4), constraining the success advantage of every equal-prior binary history classifier to $\Delta _ { \mathrm { r e c } } / 2 = O _ { L } ( | \bar { \theta } | ^ { L } )$ . This sharper bound supplies a quantitative reference for monitoring at the coherent accumulation time. The rare-syndrome construction in Sec. A 3 attains the exponent at this schedule. The result fixes the small-angle order of the maximum contrast; it does not determine its leading constant or a generally optimal sample schedule.

Remark 9 (Why the schedule is explicit). For any sequence approaching zero with eventually fixed sign $\sigma$ and $R ( \theta ) | \theta | ^ { L } \to \tau < \infty$ , the same proof gives $e ^ { \sigma \tau \mathcal { G } _ { L } }$ . Resonant values exist. $\mathrm { A t ~ } \tau = m \pi / \alpha _ { L } , m \in \mathbb { Z } _ { > 0 }$ , the limiting channel is identity. Thus $R = \Theta _ { L } ( | \theta | ^ { - L } )$ alone is insuficient; $\tau _ { * } = \pi / ( 8 \alpha _ { L } )$ is a specified nonresonant choice.

## Supplemental Material B: Finite-precision recovery comparisons

## 1. A controlled recovery comparison

Using the same exhaustive instrument, we evaluate entanglement fidelity at $\theta = 0 . 0 5$ and $R = 4 1 8 8 .$ , the reference finite-angle example. Table S3 compares evaluated policies and explicitly labeled bounds. All receive the same signed calibration. At this angle, one-round optimal Pauli selection equals minimum-weight recovery, so there is no Pauli-only improvement at the anchor. Coherent feedback improves recovery, while the fixed-unitary comparison isolates a benefit from conditioning on the syndrome.

Nearest-bin 12-bit feedback reduces one-round infidelity by a factor of approximately 151 relative to minimumweight recovery, yet accumulated phase bias makes it worse than the implemented fixed controller at the declared storage horizon. An exploratory extension randomizes independently between adjacent phase bins with unbiased phase expectation. Summing all 16 possible quantized diagonal actions per syndrome gives the exact ensemble-average channel. Its final-memory infidelity is approximately $4 . 0 0 1 7 \times 1 0 ^ { - 4 }$ , smaller by a factor of about 74 than that of the implemented fixed controller. Figure S5 shows both the improvement and the failed deterministic-rounding policy. The selection probabilities and classical arithmetic use float64; only the realized phases are 12-bit. The guarantee averages over the declared randomization resource; individual rounding sequences can have diferent errors.

History-aware deferred control remains a relevant alternative to online feedback. With the extra correction postponed until the end, equation (54) gives 12-bit infidelity below $1 . 7 7 4 \times 1 0 ^ { - 4 }$ , already smaller than the per-round randomized result. The terminal entry evaluates a rigorous analytical upper bound from the exact overlap. It assumes accurate classical phase accumulation and an ideal final diagonal logical action.

Wrong-sign calibration reverses the benefit: the randomized controller’s memory infidelity rises to approximately 0.7521, versus 0.2920 without extra feedback. Under an explicit additional complete-logical-depolarization model after every feedback round, its one-round advantage over the Pauli baseline survives only below a probability of complete logical depolarization of approximately $7 . 6 5 \times 1 0 ^ { - 6 }$ per whole logical action. The stringent budget applies to repeated actuation; deferred correction has a diferent resource requirement. The angle, calibration, precision, and actuation-noise sweeps, including failures, are retained in Supplemental Material, Sec. M. These calculations illustrate recovery within the stated instrument and identify the control resources needed for its benefit.

## Supplemental Material C: Extended validity tests

## 1. Where the guarantee fails

The drift-bound violation produces 187 harmful activations under both the confidence-only and full rules. No software binding failure is required: the physical validity premise is false. An additional age–drift grid and readout/transfer stress study expose the same boundary (Fig. S6). Across 5400 evaluations of condition–replicate pairs satisfying the assumptions there are no observed harmful activations; across 22600 evaluations of condition– replicate pairs violating the assumptions there are 4767. Each count block is shared across 5 transfer values; statistical uncertainty must account for that grouping.

The ofset curves integrate every possible count against its binomial probability at 201 ofsets. The count distribution is summed exhaustively for the selected model and nuisance settings. Floating-point evaluation remains the computational uncertainty; Monte Carlo sampling intervals do not apply to this deterministic sum. Deployment requires a justified physical interpretation of the observations in addition to an authentic record.

TABLE S3. Entanglement infidelity at $L = 3 , \theta = 0 . 0 5$ . Entries labeled as bounds are analytical comparators. The fixed controller is tuned to the storage horizon, while the Pauli policy optimizes one-round fidelity. Extra logical quantum actuation is ideal except for the stated phase discretization.
<table><tr><td>Recovery or comparator</td><td>One round 4188 rounds</td><td></td></tr><tr><td>Minimum-weight Pauli</td><td> $7 . 2 6 4 4 3 \times 1 0 ^ { - 6 }$ </td><td>0.291997</td></tr><tr><td>One-round optimal Pauli</td><td> $7 . 2 6 4 4 3 \times 1 0 ^ { - 6 }$ </td><td>0.291997</td></tr><tr><td>Fixed terminal unitary: lower bound</td><td> $7 . 2 4 6 6 4 \times 1 0 ^ { - 6 }$ </td><td>0.0296724</td></tr><tr><td>Fixed controller tuned to storage horizon</td><td></td><td>0.0296724</td></tr><tr><td>Exact conditional polar recovery</td><td> $4 . 2 2 1 \times 1 0 ^ { - 8 }$ </td><td>0.000176744</td></tr><tr><td>Per-round nearest-bin 12-bit feedback</td><td> $4 . 8 1 2 8 8 \times 1 0 ^ { - 8 }$ </td><td>0.0685292</td></tr><tr><td>Per-round randomized 12-bit feedback</td><td> $9 . 5 5 8 2 1 \times 1 0 ^ { - 8 }$ </td><td>0.000400171</td></tr><tr><td>Terminal 12-bit correction: upper bound</td><td></td><td>0.000177333</td></tr></table>

a Storage at θ = 0.05 rad  
![](images/7dc9e37142c4f0be63060248626032fa0a1091453c2268b683c3451811a1943a.jpg)

b Accumulated rounding error  
![](images/3d3ef9cde21ac306d5529042bd68b36253892a169bf1c53dda34c48773456547.jpg)  
FIG. S5. Recovery and accumulated control precision for the exact $L = 3$ instrument at $\theta = 0 . 0 5$ . (a) Final-memory entanglemen infidelity versus rounds. The fixed diagonal controller is tuned to $R = 4 1 8 8 ;$ it is an implemented feasible policy, close to the terminal-unitary bound there. (b) The complete integer precision sweep at the same horizon. Ordinary rounding leaves coherent bias; randomized rounding trades it for dephasing. The terminal 12-bit entry reports the analytical guarantee. The comparison assumes ideal extraction and signed calibration; logical-gate implementation costs would add to the reported resources.

## Supplemental Material D: Language-model maintenance and preliminary interface tests

The multistep controller may request a new probe acquisition, propose a recovery with its evidence identifier, or retain the incumbent recovery. The evaluator lists actions supported by the observations and rechecks each proposal at activation. The adviser can select from this menu but cannot change the reference calculation. The expanded evaluation in Supplemental Material, Sec. T supplies the principal multistep evidence, including incomplete workflows, acquisition cost and a matched deterministic baseline.

Earlier interface pilots exposed repeated rejected advice and wrong-sign proposals after new calibration measurements. A revised interface achieved 23 beneficial guarded completions in 24 episodes, with 1 incomplete workflow; authorization alone admitted 6 harmful updates, and the deterministic comparator completed all 24 beneficially. The two interfaces used diferent physical seeds, so this is not a paired causal estimate of an interface improvement. The initial and subsequent pilots contain 24 and 48 model interactions, respectively. These pilots motivate the interface but are not additional independent evidence for the central toric experiment.

![](images/d8e51d05a4b1fd6c4ca7f77a05c7352a3e6b75ba7b523d386cfb5f73ac7d5aa6.jpg)

![](images/b510057532e812d90ae20fd9f5f0422396d6dc8cb984975a169f5d7e23333d49.jpg)  
FIG. S6. Physical validity of the acceptance contract. Left: the acceptance region on an 81-by-81 grid of evidence ages and declared drift rates for one fixed observed block. Right: exhaustively summed binomial probabilities of harmful activation across 201 simulated ofsets and selected transfer mismatches. The shaded ofset band is the declared interval. Transfer outside ±0.001 rad violates the contract even at zero ofset.

## Supplemental Material E: Learning, classical references, and acquisition cost

The same observation and acceptance rules also apply to learned predictors. We train 5 gated recurrent units (GRUs) and 5 multilayer perceptrons (MLPs) on the available measurement histories. A GRU has 13,121 parameters. Inputs contain 24 historical means from 128 shots each, plus an input feature giving the elapsed time between measurements; next-observation training targets use 1024 shots. True angles, path families, and evaluation risks are stored separately. Whole simulated noise paths and base-parameter blocks are disjoint across training, development, and test. Architecture details and the limits of the ablations are in Supplemental Material, Sec. P.

The references include a development-tuned Bayesian random-walk filter, a tuned sliding window, the latest observation, fixed retention, and robust finite-catalog search. They use the same observation histories, permitted recovery actions, and budgets for new calibration shots. The comparison uses local reference implementations of the specified adaptive policies. The comparison measures recovery improvement relative to retaining the incumbent and the diference between neural and classical predictors under the same observation and acquisition constraints. Conditional protection applies to both neural and classical proposers.

In the first held-out test, all methods spend 8192 new calibration shots through 4 predetermined looks with total $\alpha = 0 . 0 1$ . Across 600 paths, mean guarded GRU infidelity is 0.088018, compared with 0.225490 for retaining the incumbent and 0.089296 for the Bayesian method. Thus learned advice lowers mean infidelity relative to retaining the incumbent in this simulation. A classical estimator also achieves most of this recovery improvement under the same contract. The paired GRU-minus-Bayesian diference is −0.001278, with hierarchical 95% bootstrap interval $\left[ - 0 . 0 0 2 6 7 5 , + 0 . 0 0 0 0 7 0 \right]$ , including zero. The measured neural improvement falls short of the 10% risk-reduction target. All methods incur the same cost of new calibration measurements in this 4-look protocol. The statistical comparison remains inconclusive, including for equivalence. These predictors receive signed-probe histories; the passive sign-blindness result in Proposition 1 concerns a diferent observation experiment.

## 1. Choosing a budget before acquisition

The acquisition extension uses 600 new simulated noise paths and keeps the predictor checkpoints unchanged. It chooses 1 budget $n \in \{ 0 , 1 2 8 , 5 1 2 , 2 \bar { 0 } 4 8 , 8 1 9 2 \}$ before the new measurements. At each of 5 fixed prices λ, the chosen budget maximizes expected recovery gain minus shot cost:

$$
n ^ { * } ( h ) \in \arg \operatorname* { m a x } _ { n } \{ V ( n \mid h ) - \lambda n \} ,\tag{E1}
$$

$$
V ( n \mid h ) = \mathbb { E } _ { \theta \mid h } \mathbb { E } _ { k \mid \theta , n } [ \mathcal { R } ( \theta , u _ { 0 } ) - \mathcal { R } ( \theta , u ( k , n ) ) ] .\tag{E2}
$$

For every possible count, $u ( k , n )$ is the best robustly certified catalog action, or retention. The inner binomial sum includes every count and is evaluated numerically. The outer expectation uses a finite physical grid. Bayesian acquisition uses its posterior; neural and window forecasts use development-fitted Gaussian uncertainty. The predictive distributions guide measurement allocation; a confidence region formed from newly acquired calibration outcomes supplies the acceptance certificate. This region does not use the predictor’s uncertainty estimate as a coverage guarantee.

At $\lambda = 1 0 ^ { - 6 }$ , Bayesian acquisition uses 6581 new calibration shots on average, a 19.7% reduction from 8192, while mean infidelity changes from 0.0365583 to 0.0372372. The common historical cost is another 3072 shots. The GRU traces a similar cost–risk frontier $( { \mathrm { F i g . } } { \mathrm { S 7 } } ) ;$ ; the result describes a one-look risk–cost tradeof. Unlike the first test, this extension makes 1 acquisition with $\alpha = 0 . 0 1$ Its stronger single-look confidence allocation and new simulated noise paths prevent direct attribution of the cross-experiment risk decrease to learning.

## Supplemental Material F: Calibration and transfer to circuit-level decoder updates

The coherent toric model gives losses by numerical evaluation of the exact channel under the extraction and phase-correction assumptions of Sec. I. A separate circuitlevel experiment tests whether finite-data acceptance can admit classical decoder updates that reduce logical failure risk in stochastic circuits. We use Stim 1.16.0 [32] and PyMatching 2.4.0 [15], distances 3 and 5, 30 syndrome rounds, and both X- and Z-basis memories. Noise consists of explicit stochastic Pauli faults. This experiment uses Stim’s stochastic Pauli circuits and a catalog of classical matching decoders.

The incumbent uses a fixed matching prior. Five alternative priors emphasize measurement faults, idle-Z faults, local-gate faults, paired faults, or a mixture. Each candidate is scored on the same immutable memory record as the incumbent. Paired binary failure diferences and simultaneous binomial bounds compare the worst of the two basis risks. Supplemental Material, Sec. Q gives the bound. A separately implemented raw measurement parity conversion is compared against Stim’s detector/observable conversion on every sampled batch.

A 16-condition pilot uses 1,310,720 complete memory shots and certifies 10 updates. A confirmation with the same fixed prior catalog samples 200 new parameter paths per distance, focusing on distance-3 local-gate drift and distance-5 idle-Z drift. Their comparison changes noise family as well as distance. Each basis uses 8192 calibration shots and two independent final batches of 4096 shots: the calibration channel and a return-to-base channel. The confirmation adds 13,107,200 shots, for 14,417,920 across both stages. Calibration admits all 400 candidates at the calibration noise settings. Separate pointwise final risk boxes establish benefit for all 200 distance-3 cases and 146 distance-5 cases; the remaining 54 are unresolved.

In the original return-to-base challenge, acceptance succeeds at the calibration settings while the bound for later deployment remains vacuous. Under stale deployment all 400 individual efects remain unresolved by the prescribed final boxes. The full stale gate refuses every update by adding a worst-case total-variation allowance of one, or two in excess risk. This bound is valid but vacuous; a fixed expiry can also refuse this single-delay challenge. Estimated means, their maximum-operation bias, and a separately labeled basis-averaged diagnostic appear in Supplemental Material, Sec. Q. This circuit experiment establishes benefit at the calibration settings; stale-update certification requires a noise-family-specific drift bound.

## Supplemental Material G: Geometry. Disconnected supports and the parity law

The labeled dual graph $G _ { L } ^ { * }$ has $L ^ { 2 }$ vertices, $2 L ^ { 2 }$ edges, and incidence matrix $H _ { Z } ;$ at $L = 2 .$ , parallel edge labels remain distinct. Connectivity gives rank $H _ { Z } = \check { L ^ { 2 } } - 1$ , as does the primal calculation for $H _ { X }$ . Therefore

$$
\dim \ker H _ { Z } = L ^ { 2 } + 1 , \qquad \dim ( \ker H _ { Z } / \operatorname { r o w } H _ { X } ) = 2 .\tag{G1}
$$

For an arbitrary Eulerian support $^ { a , }$ define seam parities

$$
\begin{array} { c l } { { \displaystyle \omega _ { x } ( a ) = \sum _ { y } a _ { v _ { 0 , y } } } } & { { ( \mathrm { m o d \ 2 } ) , } } \\ { { \displaystyle } } & { { \displaystyle } } \\ { { \omega _ { y } ( a ) = \sum _ { x } a _ { h _ { x , 0 } } } } & { { ( \mathrm { m o d \ 2 } ) . } } \end{array}\tag{G2}
$$

They vanish on dual-face boundaries and equal $( 1 , 0 ) , ( 0 , 1 )$ on the straight logical cycles. By equation (G1) they identify the quotient with $\mathbb { F } _ { 2 } ^ { 2 }$ , avoiding an integer winding for disconnected supports.

Every Eulerian support decomposes into edge-disjoint labeled circuits; a nontrivial support has a component with nonzero seam parity. If it lies in a cut, each component does and is even by cycle–cut orthogonality. Thus equation (28) reduces to one even nontrivial circuit.

Orient that circuit and let $n _ { x } ^ { \pm } , n _ { y } ^ { \pm }$ count its lifted signed steps. For displacement $L ( p , q )$ and length w,

$$
\begin{array} { c c } { { w = n _ { x } ^ { + } + n _ { x } ^ { - } + n _ { y } ^ { + } + n _ { y } ^ { - } , } } & { { } } \\ { { L p = n _ { x } ^ { + } - n _ { x } ^ { - } , } } & { { L q = n _ { y } ^ { + } - n _ { y } ^ { - } , } } \\ { { w - L ( p + q ) = 2 ( n _ { x } ^ { - } + n _ { y } ^ { - } ) . } } & { { \mathrm { ( G 3 ) } } } \end{array}
$$

This proves equation (31); seam parity is $( p$ mod $2 , q$ mod 2). For odd $L ,$ even w makes p+q even, so a nonzero parity pair is (1, 1), proving the 2L bound without assuming a staircase.

For achievability, follow $f _ { j , j - 1 } \xrightarrow { h _ { j , j } } f _ { j , j } \xrightarrow { v _ { j + 1 , j } } f _ { j + 1 , j } .$ For $L \geq 3$ , the 2L projected vertices and tagged edges are distinct before closure, forming a simple circuit. Each edge has one endpoint in $\{ f _ { j , j } \}$ , hence lies in its cut. At $L = 2$ , the main proof’s parallel-edge witness avoids falsely simplifying the graph.

![](images/6a2e65153d2f61c8dad0124ad2e318908b5e72ec1af3bd17fa45d447668fa530.jpg)

![](images/5609bbff5a84f0c275a6e40b2503227f75981e6c2edf84f03bec86a91ed751f2.jpg)  
FIG. S7. Risk and calibration cost on 600 held-out simulated noise paths. The 5 evaluated prices per method are connected by visual guides. Horizontal and vertical bars show the saved pointwise hierarchical bootstrap 95% intervals for mean new calibration shots and infidelity. Neural methods are gated recurrent units (GRUs) and multilayer perceptrons (MLPs), each with 5 training seeds. The right panel expands the operating region. The horizontal axis counts new calibration shots beyond the common 3072 historical shots per path

## Supplemental Material H: Efect character expansion and exact circuit count

For independent angles, write $c _ { e } = \cos ( \epsilon _ { e } / 2 )$ and $q _ { e } =$ sin $( \epsilon _ { e } / 2 )$ . Then

$$
U _ { \epsilon } = \sum _ { a \in \mathbb { F } _ { 2 } ^ { n } } ( - i ) ^ { | a | } \prod _ { e \in a } q _ { e } \prod _ { e \notin a } c _ { e } X ( a ) .\tag{H1}
$$

Projection selects ${ \widetilde { H } } _ { Z } a = s ;$ since X operators commute and $a + r _ { s } \in$ ker $H _ { Z }$ there,

$$
K _ { s } ( \epsilon ) = \sum _ { a : \widetilde { H } _ { Z } a = s } ( - i ) ^ { | a | } \prod _ { e \in a } q _ { e } \prod _ { e \not \in a } c _ { e } \overline { { { X } } } _ { [ a + r _ { s } ] } .\tag{H2}
$$

In $F _ { s } ~ = ~ K _ { s } ^ { \dagger } K _ { s }$ , recovery cancels between paired supports. Writing their diference $\textbf { \textit { f } } \in$ ker $H _ { Z }$ gives $\begin{array} { r } { \beta _ { s , f } = \sum _ { a : \widetilde { H } _ { Z } a = s } \overline { { a } } _ { a } a _ { a + f } } \end{array}$ for scalar coeficient $ { \boldsymbol { a } } _ { a }$ in equation (H1). Insert

$$
\mathbf { 1 } _ { \{ \widetilde { H } _ { Z } a = s \} } = 2 ^ { - r } \sum _ { z \in \mathbb { F } _ { 2 } ^ { r } } ( - 1 ) ^ { z \cdot ( s + \widetilde { H } _ { Z } a ) } .\tag{H3}
$$

For one edge, the four sums over $a _ { e } .$ , indexed by $\left( f _ { e } , ( h _ { z } ) _ { e } \right)$ ， are

$$
\begin{array} { r l r l r l } { 1 , } & { { } } & { \cos \epsilon _ { e } , } & { } & { { } 0 , } & { } & { { } - i \sin \epsilon _ { e } . } \end{array}\tag{H4}
$$

Their product gives equation (16). For a nonzero Taylor multi-index a, sine coordinates are exactly those with odd $a _ { e } ,$ , proving the support-parity argument used in the proof of theorem 4 without a numerical-0 test.

For a covered weight-2L circuit $^ { g , }$ only 0 and $g$ have zero boundary, so the restriction of ${ \widetilde { H } } _ { Z }$ to $g$ has rank $2 { \cal L } - 1$ . The nonempty fiber of $z \mapsto h _ { z } | _ { g }$ over the all-one word therefore has size $2 ^ { r - ( 2 L - 1 ) }$ , proving the selector factor in equation (34).

For the circuit count, orient a covered minimum circuit. Equality in $2 L \geq L ( | p | + | q | )$ , together with the established (1, 1) winding parity, gives $| p | = | q | = 1$ and forces every step to have the corresponding winding sign. Thus its root, four sign pairs, and a word with L horizontal and vertical letters determine it. Coincident intermediate projected vertices require an intervening subword to change each coordinate by a multiple of $L ;$ a proper repeated segment therefore contains all letters of one type and none of the other. The nonsimple words are exactly the 2L cyclic shifts of $H ^ { L } V ^ { L }$ . Dividing by the $2 L$ roots and two orientations of each simple cycle yields

$$
M _ { L } = { \frac { L ^ { 2 } \cdot 4 [ { \binom { 2 L } { L } } - 2 L ] } { 2 L \cdot 2 } } = L [ { \binom { 2 L } { L } } - 2 L ] .\tag{H5}
$$

Supplemental Material I: Coeficient-level comparison with stabilizer efects and toric paths

## 1. Generator coeficients and the cut constraint

Hu, Liang, and Calderbank express syndrome probabilities as diagonal generator-coeficient terms plus logical cross terms [20, Sec. 4.2, Eq. (91)]. To translate their diagonal-Z presentation, exchange physical X and Z by Hadamard conjugation. Their classical spaces then obey

$C _ { 2 } = \operatorname { r o w } H _ { Z } , C _ { 1 } ^ { \perp } = \operatorname { r o w } H _ { X }$ , and $C _ { 2 } ^ { \perp } / C _ { 1 } ^ { \perp } = \mathcal { L } _ { X }$ in our notation. All stabilizer signs are positive here. Their syndrome coset representative $\mu$ may be chosen as our $r _ { s } ,$ so $s = \widetilde { H } _ { Z } \mu ;$ their logical coset label $\gamma$ is our $c \in \mathbb { F } _ { 2 } ^ { 2 }$ With the physical rotation angle θ and the half-angle amplitudes in equation (18), their $A _ { \mu , \gamma }$ is $\kappa _ { s , c }$ . Changing the recovery representative only relabels these coeficients and does not change the efect.

In particular their Eq. (91), with the conjugation from $K _ { s } ^ { \dagger }$ written explicitly, becomes

$$
B _ { s , d } = \sum _ { c \in \mathbb { F } _ { 2 } ^ { 2 } } \overline { { \kappa _ { s , c } } } \kappa _ { s , c + d } , \qquad d \in \mathbb { F } _ { 2 } ^ { 2 } ,\tag{I1}
$$

$$
p _ { s } ( \boldsymbol { \rho } ) = \sum _ { c } | \kappa _ { s , c } | ^ { 2 } + \sum _ { d \neq 0 } B _ { s , d } \operatorname { T r } ( \overline { { \boldsymbol { X } } } ^ { d } \boldsymbol { \rho } ) .\tag{I2}
$$

Thus the basic criterion for input dependence is already present in that work. The additional issue is which physical cross terms cancel in this toric specialization. Write $u _ { a }$ for the physical amplitude of $X ( a )$ in equation (H1). Expanding the first line gives

$$
B _ { s , d } = \sum _ { f \in \ker H _ { Z } \atop [ f ] = d } \sum _ { a : \widetilde H _ { Z } a = s } \overline { { { u _ { a } } } } u _ { a + f } = \sum _ { [ f ] = d \atop [ f ] = d } \beta _ { s , f } .\tag{I3}
$$

The character transform of the inner sum is equation (16): the local pair sum is zero when $f _ { e } = 1$ and $( h _ { z } ) _ { e } \overset { \cdot } { = } \overset { \cdot } { 0 }$ This is the extra cut-containment cancellation. At the first allowed degree $2 L$ , every surviving nonscalar support is a simple circuit in class $P ;$ at zero syndrome all have the same sign and the same selector factor $- 2 ^ { - ( 2 L - 1 ) }$ This noncancellation establishes an attained efect order, rather than just a lower bound from even logical weight. It also shows precisely how the support multiplicity enters the observable coeficient.

## 2. Seam-linked paths and distinct simple supports

Iverson and Preskill count monotone paths with $\binom { l _ { 1 } + l _ { 2 } } { l _ { 1 } }$ in Eq. (H.1), then use two seam linkings in Eq. (H.2) to estimate weight- $- 2 L$ double-logical strings [21, Appendix $\mathrm { H } ]$ After exchanging $Z$ and X and identifying the square lattice with its dual, their double-logical class is our $P .$ Let $i , j = 0 , \ldots , L - 1$ label the two chosen seam edges. Denote their Eq. (H.2) by

$$
\begin{array} { l } { { N _ { i j } = A _ { i j } + A _ { L - 1 - i , j } , } } \\ { { A _ { i j } = { \binom { i + j } { i } } { \binom { 2 L - 2 - i - j } { L - 1 - i } } . } } \end{array}\tag{I4}
$$

The two terms correspond to the two relative winding signs. A simple minimum circuit crosses each seam exactly once, so its seam-edge pair and linking are unique; no choice of root or traversal orientation remains in this representation. Within each linking there is also one nonsimple path pairing: a straight horizontal loop joined to a straight vertical loop at their common vertex. It has the selected seam edges but contains two odd length-L circuits, so cannot be contained in a cut. The two linkings represent this same excluded support twice. These are the only nonsimple cases, by the repeated-vertex argument in Supplemental Material, Sec. H. Consequently the conversion to our distinct tagged simple supports is

$$
M _ { L } = \sum _ { i , j = 0 } ^ { L - 1 } ( N _ { i j } - 2 ) = L { \binom { 2 L } { L } } - 2 L ^ { 2 } .\tag{I5}
$$

For the last equality, fix $k \ = \ i + j .$ Vandermonde convolution gives $\begin{array} { r } { \sum _ { i + j = k } A _ { i j } = \binom { 2 L - 2 } { L - 1 } } \end{array}$ for every $k =$ $0 , \ldots , 2 L - 2$ , with binomial coeficients outside their natural range set to zero. The second linking has the same double sum. Hence $\begin{array} { r } { \sum _ { i , j } N _ { i j } = 2 ( 2 L - 1 ) \binom { 2 L - 2 } { L - 1 } = L \binom { 2 L } { L } } \end{array}$

Equivalently, rooted oriented words have $L ^ { 2 }$ roots and four winding sign pairs. Removing the 2L cyclic shifts of $H ^ { L } V ^ { L }$ per root and sign pair, then dividing by 2L roots and two orientations per simple support, yields equation (H5). Translated supports are distinct because the edge tags are fixed. The two normalizations therefore agree exactly. An independent integer-arithmetic check verifies the seam-sum identity and enumerates the tagged simple supports at $L = 3 , 5$ . Equation (H.3) of Ref. [21] uses the $4 ^ { L } / \sqrt { \pi L }$ path-growth scale for its large-$L$ comparison, whereas the tagged-support multiplicity here scales as $M _ { L } \sim \sqrt { L / \pi } 4 ^ { \breve { L } }$ The enumeration uses standard seam-linking and path-counting arguments.

The exact coeficient measures the full leading zerosyndrome parity contrast, $- M _ { L } / 2 ^ { 2 L - 2 }$ , rather than the existence of one surviving circuit. It supplies a parameterfree normalization for perturbative checks of the syndrome instrument and distinguishes an attained leakage order from a support obstruction alone. A single nonzero witness would sufice for the exponent and the qualitative separation. The contribution is the cancellation and common-sign survival in the efect, the identified parity observable, and the quantitative history bound.

## 3. A related fixed-code perturbative privacy bound

Shen and Zhong also obtain a fixed- $( d , T )$ weakamplitude-damping bound $O ( \gamma ^ { d _ { Z } } )$ on the syndrome contrast between logical $Z$ eigenstates [25, Proposition 1]. That result excludes leakage below a permitted order without establishing general attainment. The present coherent-rotation calculation proves a nonzero $\theta ^ { \dot { 2 } L }$ coefficient for the product parity P of the odd torus. The fixed-instrument calculation complements the established privacy comparison for growing-distance noisy memories.

Supplemental Material J: Leading-order line reduction and the channel remainder

Write $\begin{array} { r } { \kappa _ { s , c } ( \theta ) = \sum _ { m } \theta ^ { m } \kappa _ { s , c } ^ { ( m ) } } \end{array}$ . From equation (18),

$$
\kappa _ { s , c } ^ { ( m ) } = ( - i ) ^ { m } r _ { s , c } ^ { ( m ) } , \qquad r _ { s , c } ^ { ( m ) } \in \mathbb { R } .\tag{J1}
$$

A weight-w support contributes only at powers congruent to w modulo two. For $c \neq 0 , a + r _ { s }$ is nontrivial, and

$$
| a | \geq w ( s ) , \qquad | a | \geq L - w ( s )\tag{J2}
$$

give the main proof’s threshold; for class (1, 1), the second becomes $2 L - w ( s )$

To show that the following inequalities are equalities, consider an order-L cross term with nontrivial residual $g = a + r _ { s }$ and identity-class support $a ^ { \prime }$ of the same syndrome. Taylor degree dominates support weight, while equation (J2) gives $| a | + | a ^ { \prime } | \geq L ;$ thus equality holds. Minimality of $r _ { s }$ gives $w ( s ) \leq | a ^ { \prime } |$ , so

$$
L \leq | g | \leq | a | + w ( s ) \leq | a | + | a ^ { \prime } | = L .\tag{J3}
$$

All inequalities are equalities. Specifically, $| g | \ = \ L$ $| a ^ { \prime } | = w ( s )$ a $\cap r _ { s } = \varnothing , r _ { s } \subset g$ , and $a = g \setminus r _ { s }$ . Also $| a | = L - w ( s )$ and, because L is odd, $w ( s ) \leq ( L - 1 ) / 2$ The cycle $a ^ { \prime } + r$ has weight at most $2 w ( s ) < L$ and is stabilizer-trivial. Every nontrivial weight-L X cycle is a straight row or column. The projection below makes $r _ { s }$ its unique boundary minimizer; hence the other minimizer $a ^ { \prime } = r _ { s }$ . Exchanging the pair is identical. Thus complementary subsets of straight logical lines exhaust the order-L coeficient.

The channel has the class-matrix form

$$
\begin{array} { r l } & { \Lambda _ { L } ( \theta ) ( \rho ) = \displaystyle \sum _ { c , c ^ { \prime } \in  { \mathbb { F } } _ { 2 } ^ { 2 } } M _ { c , c ^ { \prime } } ( \theta ) \overline { { \boldsymbol { X } } } ^ { c } \rho \overline { { \boldsymbol { X } } } ^ { c ^ { \prime } } , } \\ & { M _ { c , c ^ { \prime } } ( \theta ) = \displaystyle \sum _ { s } \kappa _ { s , c } ( \theta ) \overline { { \kappa _ { s , c ^ { \prime } } ( \theta ) } } . } \end{array}\tag{J4}
$$

At odd $L ,$ coeficients with both logical-class indices nonzero start above degree $L ;$ trace preservation gives $\begin{array} { r } { \sum _ { c } M _ { c , c } ( \theta ) = 1 } \end{array}$ and $\tilde { M _ { 0 , 0 } } = \dot { 1 } + O _ { L } \dot { ( \theta ^ { L + 1 } ) }$ . At degree L, only $( m , m ^ { \prime } ) = ( L - w ( s ) , w ( s ) )$ , whose phase from equation (J1) is

$$
( - i ) ^ { m } ( i ) ^ { m ^ { \prime } } = ( - 1 ) ^ { m } i ^ { L } .\tag{J5}
$$

This phase is purely imaginary, so the leading generator is a sum of commutators.

For global line minimization, take a horizontal dualgraph logical row $R _ { y _ { 0 } }$ and let $\pi _ { 0 } ^ { ( y _ { 0 } ) } ( x , y ) = ( x , y _ { 0 } )$ . On edge basis elements, define $\pi _ { 1 } ^ { ( y _ { 0 } ) } ( e _ { x , y } ^ { \parallel } ) = e _ { x , y _ { 0 } } ^ { \parallel }$ for edges parallel to the row and $\pi _ { 1 } ^ { ( y _ { 0 } ) } ( e _ { x , y } ^ { \perp } ) = 0$ , extending linearly. On basis edges, $\partial \pi _ { 1 } ^ { ( y _ { 0 } ) } = \pi _ { 0 } ^ { ( y _ { 0 } ) } \partial ;$ mod-two collisions only decrease weight. If $A \subset R _ { y _ { 0 } }$ has boundary $D _ { \mathbf { \lambda } }$ every boundary-D chain projects to A or $R _ { y _ { 0 } } \setminus A$ . Hence

$$
w ( D ) = \operatorname* { m i n } \{ | A | , L - | A | \} .\tag{J6}
$$

$\mathrm { I f } \ | A | < L / 2$ and $| r | = | A |$ , projection must equal A with no weight loss, forbidding transverse edges and collisions. The boundary excludes of-target partial rows, and a full of-target row weighs $L > | A | ;$ thus $r = A$ uniquely. Exchanging coordinates proves the column case. Odd L excludes half-line ties, and winding equality places every active residual on such a line.

On a fixed line, complementary subsets share a syndrome; the lighter corrects to identity and the heavier to the line class. Their leading products sum to

$$
\begin{array} { c } { { \beta ^ { \mathrm { l i n e } } = \displaystyle \frac { i ^ { L } } { 2 ^ { L } } \sum _ { k = ( L + 1 ) / 2 } ^ { L } ( - 1 ) ^ { k } { \binom { L } { k } } } } \\ { { = - \displaystyle \frac { i } { 2 ^ { L } } { \binom { L - 1 } { ( L - 1 ) / 2 } } . } } \end{array}\tag{J7}
$$

Pascal induction proves the partial alternating-binomial identity. There are L representatives per one-logical class and none of weight L for the product class, giving equation (42).

For the remainder, equip finite Fourier polynomials with $\begin{array} { r } { \| \boldsymbol { f } \| _ { \mathcal { F } , 1 } = \sum _ { \omega } | \widehat { f } _ { \omega } | } \end{array}$ . Each support amplitude cos $( \theta / 2 ) ^ { n - | a | } [ - i \sin ( \theta / 2 ) ] ^ { | a | }$ has Fourier coeficient norm at most one. If $N _ { s , c }$ counts supports in cell $( s , c )$ , then $\| \kappa _ { s , c } \| _ { \mathcal { F } , 1 } \leq N _ { s , c }$ and submultiplicativity gives

$$
\begin{array} { l } { \displaystyle \| M _ { c , c ^ { \prime } } \| _ { \mathcal { F } , 1 } \leq \sum _ { s } N _ { s , c } N _ { s , c ^ { \prime } } } \\ { \leq \Big ( \sum _ { s } N _ { s , c } \Big ) \Big ( \sum _ { s } N _ { s , c ^ { \prime } } \Big ) } \\ { \leq 2 ^ { 2 n } = 2 ^ { 4 L ^ { 2 } } . } \end{array}\tag{J8}
$$

The frequencies of $M _ { c , c ^ { \prime } }$ have magnitude at most $n = 2 L ^ { 2 }$ so every $( L + 1 ) \mathrm { s t }$ derivative is bounded by $n ^ { L + 1 } 2 ^ { 2 n }$ Integral Taylor remainder therefore gives, for real θ,

$$
\left| M _ { c , c ^ { \prime } } ( \theta ) - \sum _ { j = 0 } ^ { L } [ \theta ^ { j } ] M _ { c , c ^ { \prime } } ( \theta ) \theta ^ { j } \right| \leq \frac { ( 2 L ^ { 2 } ) ^ { L + 1 } 2 ^ { 4 L ^ { 2 } } } { ( L + 1 ) ! } | \theta | ^ { L + 1 } .\tag{J9}
$$

The 16 maps that multiply an operator by logical Paulis on the left and right have diamond norm one; the triangle inequality lifts equation (J9) to equation (43).

## Supplemental Material K: Product limit and null-prefix histories

For a history prefix z, compose equation (19)’s branch maps to obtain its unnormalized conditional state. At positive trace, divide by it; at zero trace, every child has zero probability regardless of the filler kernel. Thus the recursion for probabilities of partial histories is exact and $\eta _ { L } ( \theta )$ controls every next-round kernel pair. First-disagreement maximal coupling proves equation (44) without commonsupport or likelihood-ratio assumptions.

Second, define $\mathcal { V } _ { h } = e ^ { \sigma h \mathcal { G } _ { L } }$ , a unitary channel. For any $R ,$

$$
\Lambda _ { L } ( \theta ) ^ { R } - \mathcal { V } _ { h } ^ { R } = \sum _ { j = 0 } ^ { R - 1 } \Lambda _ { L } ( \theta ) ^ { R - 1 - j } [ \Lambda _ { L } ( \theta ) - \mathcal { V } _ { h } ] \mathcal { V } _ { h } ^ { j } .\tag{K1}
$$

Each channel factor has diamond norm one, avoiding powers of the generally nonpositive Euler map id $+ \sigma h \mathcal { G } _ { L } .$ Exponential remainder gives $\| \mathcal { V } _ { h } \mathrm { ~ - ~ } \mathrm { i d } \mathrm { ~ - ~ } \sigma h \mathcal { G } _ { L } \| _ { \diamond } \le$ $h ^ { 2 } \bar { g } _ { L } ^ { 2 } e ^ { h g _ { L } } / 2$ , and $R \leq T / h$ gives the first two terms of equation (A3). For $R = \lfloor \tau / h \rfloor , 0 \leq \tau - R h < h ,$ , and Duhamel’s formula bounds the remaining time interval by $h g _ { L }$ , proving the uniform limit including floor discontinuities.

## Supplemental Material L: Exhaustive numerical protocol

For $L = 3$ we use the first 8 plaquette rows in the coordinate order of equation (1) as the independent syndrome basis. For each support $a \in \mathbb { F } _ { 2 } ^ { 1 8 }$ , we compute its syndrome, Hamming weight, and seam parities. Comparing weight and then the ascending edge-index tuple over all supports gives exactly the declared recovery $r _ { s }$ . Let $N _ { s , c , w }$ count supports of weight w with syndrome s and residual logical class $c = [ a + r _ { s } ]$ . Then

$$
\kappa _ { s , c } ( \theta ) = \sum _ { w = 0 } ^ { 1 8 } N _ { s , c , w } \cos ( \theta / 2 ) ^ { 1 8 - w } [ - i \sin ( \theta / 2 ) ] ^ { w } .\tag{L1}
$$

The integer counts are evaluated exhaustively. Independent decimal-arithmetic checks assess numerical precision; the plotted values follow directly from these counts.

For a simultaneous logical-X eigenstate labeled by $x \in$ $\mathbb { F } _ { 2 } ^ { 2 } .$ , let $\begin{array} { r } { z _ { s } ( x ) = \sum _ { c } ( - 1 ) ^ { x \cdot c } \kappa _ { s , c } } \end{array}$ . The averaged channel multiplies $| x \rangle \langle y | \ \mathrm { b y }$

$$
C _ { x y } ( \theta ) = \sum _ { s } z _ { s } ( x ) \overline { { z _ { s } ( y ) } } .\tag{L2}
$$

Consequently repeated rounds multiply this matrix element by $C _ { x y } ^ { R } ,$ without constructing any histories. Taking $x ~ = ~ + + , ~ y ~ = ~ --$ gives $d _ { b } ~ = ~ | 1 - C _ { x y } ^ { R _ { * } } |$ and $\ell _ { b } = ( 1 - \mathrm { R e } C _ { x y } ^ { R _ { * } } ) / 2$ . The plotted one-round channel coeficient is $\begin{array} { r } { \big | \sum _ { s } \kappa _ { s , ( 1 , 0 ) } \overline { { \kappa _ { s , ( 0 , 0 ) } } } \big | } \end{array}$ , tending to $\alpha _ { 3 } \theta ^ { 3 }$ . This quantity is the indicated channel-expansion coeficient; the diamond norm is evaluated separately. The exact identity equation (23) supplies the record curve without optimization over input states.

The checks verify support-count completeness, trace preservation, absence of single-logical-X terms in every efect, sign symmetry, and selected direct two-round compositions against equation (22). The two leading coeficients are checked with rational arithmetic on integer support counts. Independent 60-digit trigonometric sums and repeated complex multiplication at $\theta = 0 . 0 1 5 , 0 . 0 2 5 , 0 . 0 5 , 0 . 1 , 0 . 1 5$ check floating-point accumulation. No normalization is imposed to force the tracepreservation check to pass. The grid evaluates finite-angle behavior over $0 . 0 1 5 \leq \theta \leq 0 . 1 5 ;$ the analytical results establish the asymptotic exponent and stated all-angle identities.

## Supplemental Material M: Recovery calculations and validation

The calculation regenerates all $2 ^ { 1 8 }$ support counts and recovery representatives before use. An independent 18-qubit computational-basis construction applies every physical $R _ { X } ( \theta )$ gate, projects onto all 256 syndrome sectors, and recovers their logical Kraus matrices at $\theta = 0 , 0 . 0 1 5 , 0 . 0 5 , 0 . 1 5 , - 0 . 0 5 , 0 . 3 , 0 . 6$ . The tolerance is $1 0 ^ { - 1 2 }$ , using complex128 and float64 without probability clipping or forced renormalization. Unit tests check Kraus versus full superoperator composition, parity overlap, Pauli optimization, feasible fixed control, the spectral bound, and enumeration of all randomized actions. The additional terminal check explicitly sums every two-round history and tests the analytical precision bound independently.

The fixed comparison uses the principal eigenvector of $C ^ { \circ R }$ , projects its entries to unit modulus, and distributes the resulting phase correction over R rounds. The eigenvector phase is fixed before choosing angle branches. At the anchor, the feasible policy lies approximately $2 . 4 1 \times 1 0 ^ { - 8 }$ above the spectral lower bound, quantifying its proximity within the stated comparison. The Pauli policy maximizes one-round fidelity. The optimization here selects the best one-round Pauli correction.

For grid spacing $\Delta = 2 \pi / 2 ^ { b }$ , write the desired phase as $\phi = ( n + f ) \Delta$ , with $n \in \mathbb { Z }$ and $0 \leq f < 1$ . Independently choose n∆ or $( n + 1 ) \Delta$ with probabilities $1 - f$ and $f .$ With $w _ { s } ( x ) = \dot { \mathbb { E } } e ^ { i \phi _ { s } ( \dot { x } ) }$ , the corrected Schur multiplier is

$$
C _ { x y } ^ { \mathrm { r a n d } } = \left\{ \sum _ { s } z _ { s } ( x ) z _ { s } ( y ) ^ { * } w _ { s } ( x ) w _ { s } ( y ) ^ { * } , ~ x \neq y , \right.\tag{M1}
$$

The diagonal uses $\mathbb { E } | e ^ { i \phi } | ^ { 2 } = 1$ , not $| \mathbb { E } e ^ { i \phi } | ^ { 2 }$ . Independently sampled controller randomness is assumed each round. This applies the established benefit of mixing nearby unitaries to phase quantization [43]; it difers from the Pauli-twirling construction of randomized compiling [44].

For complete logical depolarization with probability ϵ after each additional control round, unitality gives

$$
F _ { e } ^ { ( \epsilon ) } ( R ) = ( 1 - \epsilon ) ^ { R } F _ { e } ( R ) + \frac { 1 - ( 1 - \epsilon ) ^ { R } } { 1 6 } .\tag{M2}
$$

The phenomenological budget treats the correction as a whole action; pulse synthesis, measurement faults, and latency-dependent gate noise require a resolved implementation model. It also charges per-round control that can be deferred in this commuting storage example.

![](images/fb1b212e9b82968452796efd9e56e41aa71d493d3eb9e03d25b7d37109188f5a.jpg)

![](images/810684f8faf56fec77bfd29dfdfc4efd8937d5b812d7d63d37ccd018a2016407.jpg)

![](images/f6a6ffa16edb9f0fd73831ad3fd4e89b8be529aae92a23a3d4c76ed0151e5c73.jpg)

![](images/5d75fea19ae3075e2f72beecf08c05e6e0e6df3e1a4496ed833a819cc3543d10.jpg)  
FIG. S8. Complete recovery robustness sweeps. (a) One-round infidelity across 122 angles including the reference angle $\theta = 0 . 0 5 .$ (b) Relative Pauli-only improvement occurs at stronger noise, where logical infidelities are high. (c) Assumed calibration divided by true $\theta = 0 . 0 5 ; - 1$ is the wrong-sign case. (d) Extra per-round logical depolarization erases the benefit. Dashed and dotted curves labeled as bounds are analytical comparators. Every curve is an exhaustive instrument calculation, so sampling error bars are inapplicable.

The parameter sweeps contain 122 angles from 0.015 to 0.6, 124 distinct round counts through 4188, 281 signed calibration ratios from −1.5 to 2, 162 actuation-noise values including zero, and every integer precision from 4 to 24 bits. Six pilot angles were inspected before the protocol was written. The precision/randomization extension followed the initial rounding failure; both the initial run and this exploratory extension are retained. The study was conducted without preregistration, with its decisions and failed checks retained in the research record. This calculation evaluates the instrument directly through exhaustive deterministic sums.

## Supplemental Material N: Exact channel evaluation and continuous bounds

Write the diagonal entries of $V _ { s } ( u ) K _ { s } ( \theta )$ as $q _ { s , i } ( \theta , u )$ Equation (11) is a Schur channel, with

$$
C _ { i j } ( \theta , u ) = \sum _ { s } q _ { s , i } ( \theta , u ) q _ { s , j } ( \theta , u ) ^ { * } ,\tag{N1}
$$

$$
[ \mathcal { E } _ { \theta , u } ( \rho ) ] _ { i j } = C _ { i j } ( \theta , u ) \rho _ { i j } .\tag{N2}
$$

For fixed u, channel composition multiplies entries, and

$$
F _ { e } ( \mathcal { E } _ { \theta , u } ^ { H } ) = \frac { 1 } { 1 6 } \sum _ { i , j = 1 } ^ { 4 } C _ { i j } ( \theta , u ) ^ { H } .\tag{N3}
$$

The sum is real; the implementation takes its real part and checks the channel conventions against direct evaluation of the instrument. The reference uses float64/complex128 arithmetic. Physical support counts come from exhaustive enumeration of the 18-qubit model. The original 78 tests, including a separate physical-state construction, were rerun before reuse.

Instrument amplitudes are degree-N homogeneous polynomials in $\cos ( \theta / 2 )$ and $\sin ( \theta / 2 )$ . Their products with complex conjugates contain integer Fourier frequencies at most N. The fixed phase tables do not change this degree. Consequently 37 Fourier samples recover each $C _ { i j }$ at $N = 1 8$ in exact arithmetic. For coeficients $c _ { f , i j }$ ，

$$
\vert C _ { i j } ^ { \prime \prime } ( \theta ) \vert \leq M _ { i j } : = \sum _ { f = - N } ^ { N } f ^ { 2 } \vert c _ { f , i j } \vert .\tag{N4}
$$

At a cell center x and half-width h, this gives $\begin{array} { r } { \operatorname* { s u p } _ { | \theta - x | \leq h } | C _ { i j } ^ { \prime } ( \theta ) | \leq | C _ { i j } ^ { \prime } ( x ) | + h M _ { i j } } \end{array}$ . Complete positivity and trace preservation imply $| C _ { i j } | \le 1$ , so diferentiation of Eq. (N3) yields

$$
| F _ { e } ^ { \prime } ( \theta ) | \leq \frac { H } { 1 6 } \sum _ { i j } | C _ { i j } ^ { \prime } ( \theta ) | .\tag{N5}
$$

The numerical bound on $D _ { u }$ includes derivative allowances for both candidate and incumbent. It considers every cell intersecting $C _ { \mathrm { d e p } }$ , including boundary cells, and adds the corresponding half-width allowance to the grid excess loss. The plotting grid alone is never treated as a certificate.

Comparisons at additional angles check Fourier reconstruction and derivatives against direct Kraus evaluation. An additional $1 0 ^ { - 8 }$ allowance covers observed numerical residuals. The tests assess the numerical tolerance. Proposition 8 requires valid numerical bounds; a formal floating-point enclosure would additionally certify rounding error beyond the tested cases.

A separate audit enumerates all 8193 possible counts and all 13 actions at 1692 physical/nuisance settings. An ofline audit flags a count if any admitted action violates the promised margin. The largest integrated probability is 0.00281349, below $\alpha = 0 . 0 1$ . The audit exhausts counts and actions on a finite physical-parameter grid, using evaluator knowledge to compute worst-selection risk. This zero-drift audit is the 1692-setting subset of the extended audit in Supplemental Material, Sec. T. Adding drift radii 0.005 and 0.02 rad, with the corresponding allowed deployment shifts, increases the grid to 5076 settings and gives maximum margin-violation probability 0.00378720114; the zero-drift results are unchanged.

## Supplemental Material O: Information and duration references

For two equal-prior sign hypotheses, let C and W be the probabilities of correct-sign and wrong-sign activation, with abstention allowed, and $q = C + W$ . If the observation laws have total-variation distance T, a binary decision rule obeys $C - W \leq T$ . Indeed, the diference is one half of the expectation diference of a function taking values in [−1, 1], bounded by total variation. Hence

$$
W \geq \operatorname* { m a x } \{ 0 , ( q - T ) / 2 \} .\tag{O1}
$$

Requiring $q \ \geq \ 0 . 8$ and $W ~ \leq ~ 0 . 0 1$ therefore requires $T \geq 0 . 7 8$ . With known ofset zero, contrast 0.99, and $| \theta | = 0 . 1$ , numerical evaluation of the binomial likelihoods first meets that necessary condition at 153 shots. This optimistic sign-only reference excludes nuisance uncertainty, magnitude accuracy, and recovery-margin requirements. The 153-shot estimate concerns sign discrimination; QEC certification additionally constrains magnitude, nuisance parameters, and the recovery margin. Actions beneficial at both signs require a diferent decision criterion from sign discrimination.

For changing angles, product rotations satisfy

$$
\begin{array} { r } { \| R _ { X } ( \theta ) ^ { \otimes N } - R _ { X } ( \theta _ { 0 } ) ^ { \otimes N } \| \le \frac { N } { 2 } | \theta - \theta _ { 0 } | . } \end{array}\tag{O2}
$$

The corresponding channel diamond distance is at most $N | \theta - \theta _ { 0 } |$ Fixed syndrome extraction, recovery, and classical record retention are completely positive tracepreserving postprocessing and cannot increase it. Telescoping H rounds bounds one policy’s loss change by $N \hat { \sum _ { r } } \left| \tilde { \theta } _ { r } - \theta _ { 0 } \right|$ using a conservative fidelity bound. Adding the bounds on candidate and incumbent losses gives $\begin{array} { r } { \hat { 2 N } \sum _ { r } v r \Delta = N v \Delta H ( H + 1 ) } \end{array}$ , Eq. (65). The physical model specifies the pulse-error and channel assumptions; unknown pulse errors and policy-dependent adversarial channels require an extended model.

The executed extension assumes $H = 3 0 0 , \mathrm { ~ a ~ } 1 0 ^ { - 6 } T _ { 0 }$ round, a $1 0 ^ { - 6 } T _ { 0 }$ probe shot, and a 0.001 terminal logicaldepolarization cost. Full acquisition duration enlarges the age allowance; correction cost is charged once terminally. These stipulated durations define the timing ratios for this simulation. Of 81 tested drift rates, 47 admit the chosen table, and every independently composed changing-angle loss is below the bound (Fig. S9). The sweep evaluates the decision boundary for one observed count; coverage under varying or dependent outcomes requires a sampling model for those outcomes.

## Supplemental Material P: Training, splits, and workflow details

The training set has 600 paths, equally divided between stationary and ramp families, with signed base magnitudes 0.04, 0.07, 0.10, 0.13 and magnitude jitter ±0.002. Development has 200 paths at bases 0.055, 0.085, 0.115. The initial test has 600 paths, equally divided among stationary, ramp, and periodic families, at bases 0.05, 0.08, 0.11, 0.14. Periodic drift is held out from both training and development. Whole paths and base blocks, rather than individual trajectories from a common source pool, define splits.

Each path has 25 measurement times. The predictor sees the first 24 means and an elapsed-time input feature. GRUs have hidden size 64; MLPs process pertime two-component inputs through two 64-unit hidden layers. The MLP uses the specified summary representation; a flattened full-history feed-forward model would be an additional comparator. Both families use 5 seeds, 11, 23, 37, 51, 79, 40 Adam epochs, learning rate 0.003, and select the checkpoint with the lowest binary cross-entropy on the development measurements available to the model. The trained target is an observed next-outcome frequency, not a hidden angle or logical risk.

![](images/a539869f769d0bf4e7961271ba4dbed8f1180730a0d4bfd9a8ef716a062faac6.jpg)

![](images/b53afa2795b6562195d1e7270beab49b3a5a93c2c89448549c274aea03083b05.jpg)  
FIG. S9. Within-block drift extension over 81 evaluated rates. (a) Numerically evaluated changing-channel excess risk and the conservative upper bound. (b) An expanded view of the acceptance boundary at excess risk $- 1 0 ^ { - 3 }$ . Curves outside each displayed vertical range are clipped. The simulation specifies acquisition durations, terminal-action noise, and physical drift bounds.

The Bayesian reference uses a grid random walk, choosing its transition scale from 0.001, 0.003, 0.01, 0.03 on development observations. Sliding-window lengths are selected from 1, 2, 4, 8, 12, 24. In the initial 4-look test the Bayesian posterior is updated directly, whereas the other point predictors blend the forecast with the current observed mean, assigning the forecast the weight of 512 observations and the observed mean the weight of the current shot count. This limits attribution of diferences to model architecture alone. The robust catalog reference was added in a documented amendment and accepts 323 of 600 paths; its mean risk is 0.0893974. On the public development distribution and tested average-risk objective, retention is the best fixed policy in the catalog.

The cost-aware extension fixes new bases 0.047, 0.077, 0.107, 0.137 with ±0.001 jitter and 600 new paths. Its 5 prices are $0 , 2 . 5 \times 1 0 ^ { - 7 } , 1 \dot { 0 } ^ { - 6 } , 4 \times 1 0 ^ { - 6 }$ , and ${ \bar { 1 } } . 6 \times 1 0 ^ { - 5 }$ . Acquisition integration uses 181 angle points and all possible binomial counts. Forecast spreads are fitted only to public development targets, subtracting estimated binomial variance and imposing a minimum standard deviation of 0.005 in units of the observed outcome mean. No test target is used to fit uncertainty or choose a favorable price afterward.

TABLE S4. Circuit confirmation. Mean plug-in worst-basis excess risk and the count of unresolved individual efects under independent final confidence boxes. Each row has 200 paths. The two deployment rows at each distance reuse those paths; distance 3 uses local-gate noise and distance 5 uses idle-Z noise. Plug-in means retain finite-sample maximum bias.
<table><tr><td>d Deployment</td><td>Mean excess Unresolved</td></tr><tr><td>3 Same channel</td><td>-0.091455</td></tr><tr><td>5 Same channel</td><td>0 -0.032284 54</td></tr><tr><td>3 Return to base</td><td>-0.002561</td></tr><tr><td></td><td>200</td></tr><tr><td>5 Return to base</td><td>+0.002532 200</td></tr></table>

Ablations remove measurements, permute history order, or zero the elapsed-time input feature. Their output is saved as forecast sensitivity. With constant training-time gaps, zeroing that feature probes an out-of-distribution input change; learned age reasoning would require variation in training and a targeted evaluation. These predictors receive numerical measurements and time features, without the untrusted text.

The preliminary interface tests use temperature-zero Qwen inference, seed 17, a 4-turn cap and at most one additional probe. The expanded multistep protocol and its matched observation budget are specified in Supplemental Material, Sec. T; the central single-proposal toric protocol is specified in Sec. Y.

![](images/65684465cd0c77f18d419f88ef9005bdfb77c58faf3f59c6661513a219f43b85.jpg)

![](images/8372d522758f32e7ce5799a291fd5daaaf94ff18e14464e5d8125d0a8e673e2d.jpg)  
FIG. S10. Empirical distributions on 600 held-out simulated noise paths. (a) Cumulative distributions of guarded-policy infidelity. Neural methods are gated recurrent units (GRUs) and multilayer perceptrons (MLPs); risks are averaged over 5 training seeds within each path. (b) Paired GRU-minus-Bayesian guarded-risk diferences, with 200 paths per drift family; negative values favor the GRU. The curves display empirical cumulative distributions; inferential intervals are reported separately where available. All methods have the same observations, action catalog, and budget for new calibration shots.

## Supplemental Material Q: Paired circuit risks and confirmation uncertainty

For independent memory records in basis $b ,$ let $L _ { u } , L _ { 0 } \in$ {0, 1} denote candidate and incumbent failures on the same record. Their risk diference is

$$
\begin{array} { r } { r _ { u , b } - r _ { 0 , b } = \operatorname* { P r } ( L _ { u } = 1 , L _ { 0 } = 0 ) - \operatorname* { P r } ( L _ { u } = 0 , L _ { 0 } = 1 ) . } \\ { ( \mathrm { Q 1 } } \end{array}\tag{Q1}
$$

Separate one-sided exact binomial limits on the two discordance probabilities yield an upper bound $d _ { b }$ . Let $r _ { 0 , b } \in [ \ell _ { b } , h _ { b } ]$ simultaneously. Then

$$
\begin{array} { l } { \displaystyle { \operatorname* { m a x } _ { b } r _ { u , b } - \operatorname* { m a x } _ { b } r _ { 0 , b } } } \\ { \displaystyle { \quad \leq \operatorname* { m a x } _ { b \in \{ X , Z \} } \left[ d _ { b } + \operatorname* { m i n } \{ 0 , h _ { b } - \ell _ { - b } \} \right] . } } \end{array}\tag{Q2}
$$

To see this, subtract max $( r _ { 0 , b } , r _ { 0 , - b } )$ from each $r _ { 0 , b } + d _ { b }$ and maximize the remaining diference over the baseline box. The calibration error allocation covers both bases and every predeclared alternative simultaneously, with total $\alpha = 0 . 0 1$ . A single record supports rescoring classical matching priors; evaluating a diferent quantum feedback policy would require its corresponding physical trajectory. The endpoint is the maximum of the two memory-basis failure risks.

Pilot simulated base rates are 0.0017 and 0.0023; the prior base is 0.002. Four families amplify measurement, idle-Z, local-gate, or paired faults. Independent final batches use 32768 shots per basis. The selected confirmation regimes draw simulated base rates from [0.0018, 0.0022] with new physical seeds, retaining the prior catalog and thresholds. The calibration test and final uncertainty calculation remain separate.

Pointwise 95% final boxes cover the four candidate/incumbent basis risks for each path. The individual return-to-base efects remain statistically unresolved (Table S4). The estimated mean primary excess risks have bootstrap intervals [−0.003108, −0.002010] at distance 3 and [0.002362, 0.002704] at distance $5 ,$ but the maximum of finite-sample risks is biased. The maximum-estimation bias persists under path resampling.

As a separately labeled secondary endpoint, the equally weighted X/Z mean-risk diference avoids the maximum operation. Its return-to-base estimates are +0.0004706 (95% interval [0.0001977, 0.0007361]) and +0.0015100 ([0.0014172, 0.0015991]). These quantify an additional average-risk diagnostic alongside the prescribed worstbasis endpoint. The full stale rule uses the valid but uninformative total-variation bound one and rejects all cases. The point estimates and universal refusal do not establish a recovery-improving update justified by old evidence in this challenge.

## Supplemental Material R: Numerical precision and validation

Physics validation uses float64/complex128 arithmetic. Neural inference uses float32; a comparison of numerical implementations gives a maximum logit discrepancy of $5 . \dot { 9 6 } \times 1 0 ^ { - 8 }$ for the same GRU checkpoint and a 200-path, 24-observation batch.

Validation covers the physical probe, continuous bounds, confidence construction, evidence binding, budgets, malformed input, circuit conversion, acquisition, and permitted maintenance operations. A separate rerun reproduces the scientific rows of five stages using fixed checkpoints, at absolute tolerance $1 0 ^ { - 1 2 }$ and relative tolerance $1 0 ^ { - 1 0 }$ . Figures and tables are generated from numerical results with consistency checks.

![](images/86016f5c0e90f767c763a4f5f284ce5c3d26733707ebe035b3b01869f251db7f.jpg)

![](images/973d04ee2ef63d355db7e2e3a776725bd51b772728f70fbbb9c06cabd821f0c5.jpg)

![](images/442d87cd094dce6c1e70ac14cc3596f948bf2c4aa20592cb9e9608ef27951dfe.jpg)

![](images/4d2cdb69817d9d6022c15e4a788f9251b274c9af7372464f7f11c5df7b59ff39.jpg)  
FIG. S11. Circuit confirmation with all 200 paths per distance and deployment stage. Each point is an estimate from an independent deployment sample of worst-basis excess risk; vertical segments are the saved pointwise 95% final risk boxes. Each panel orders paths independently by its own point estimates; paired stages must be compared through their workload identifiers. Benefit is established for 200 distance-3 and 146 distance-5 paths at the calibration settings; all 400 stale efects remain unresolved. Point estimates retain finite-sample maximum bias; harmful staleness requires a resolved adverse risk diference.

## Supplemental Material S: Theoretical context and prior-work comparison

The corrected logical channel and the syndrome effects address distinct observables of a coherent-error instrument. Pauli amplitudes in one syndrome sector interfere, and recovery maps their relative logical classes to a syndrome-conditioned logical map. Coherent logical channels under quantum error correction have been studied for repetition, stabilizer, surface, and toric codes [12, 17, 18, 21, 41, 50, 51]. Stabilizer records support insitu noise characterization [3, 45, 52–54], detector-model inference [46, 47, 55], and circuit-level logical-noise estimation [48, 49, 56]. Syndrome-conditioned evolution can also generate logical unitary ensembles [38]. Odddistance rotated square-lattice planar surface codes with even-weight relevant stabilizers and odd-weight logical representatives have input-independent syndrome probabilities despite conditioned logical rotations [12, 19, 38]. Other parity classes reveal logical polarization [17, 18], while diagonal-gate methods quantify state–syndrome correlations [20]. Input-agnostic branches also support repeated syndrome-resolved logical-gate protocols [57, 58]. We instead determine when the complete syndrome record of the periodic instrument in Sec. I first depends on the encoded input.

Orsucci, Tiersch, and Briegel explicitly identify the ambiguity $\lambda _ { a } \mapsto \pm \lambda _ { a } + 2 k \pi$ in the probability diferences of individual graph-state stabilizer outcomes, since these depend on cos $\lambda _ { a }$ [53], Eq. (6) and the paragraph following it. This is a direct antecedent of the parameter-identifiability question. Our statement concerns a diferent observation object: the full joint law of arbitrarily long passive histories for the declared toric instrument. Its proof uses the corrected Kraus operators and their common eigenbasis; marginal symmetry alone would not establish that history statement.

TABLE S5. Positioning under the declared fixed-odd-L toric instrument.
<table><tr><td>Result</td><td>Contribution and scope</td></tr><tr><td> $[ [ 2 L ^ { 2 } , 2 , L ] ]$  parameters and Toric geometry [36, 37] and weight-2L Standard geometry. parity-dependent  $d _ { \mathrm { a f f } }$  paths [21, 22]</td><td> $d _ { \mathrm { a f f } }$  denotes the cut-containment requirement for a nontrivial support contributing to a syndrome effect defined here. The parity pattern follows established geometry.</td></tr><tr><td> ${ \mathrm { O r d e r } } { - } \theta ^ { L }$  corrected channel and Iverson-Preskill [21]  $\alpha _ { L }$ </td><td>Reproduced. A common frame, sign, half-angle, and recovery make the channel directly comparable with the effects.</td></tr><tr><td>dence in one-logical-qubit codes dependent leakage [12, 17–19]</td><td>Parity-controlled record depen- Square-lattice blindness and parity- Prior contrast. Two logical qubits here permit  ${ \overline { { X } } } _ { x } { \overline { { X } } } _ { y }$  dependence at finite order.</td></tr><tr><td>estimation Ref. [53]</td><td>Sign ambiguity in coherent-error Graph-state stabilizer marginal probabil- Prior ambiguity. Their cosine dependence loses ro- ities, Eq. (6) and following discussion of tation sign and period. Here the symmetry holds for complete passive toric histories.</td></tr><tr><td>Noise or channel inference from Noise, detector-model, and circuit-level Different target. They infer faults, priors, or channels. syndrome records inference [3, 45–49]</td><td>We distinguish encoded inputs at fixed signed θ. and 2L geometry [21, 22] and effect criteria Exact specialization. Cut cancellation and common-</td></tr><tr><td> $\begin{array} { r l r } { b _ { \mathrm { r e c } } } & { { } = } & { 2 L , } \end{array}$  exact  $M _ { L }$   $\eta _ { L } ( \theta ) = \Theta _ { L } ( | \theta | ^ { 2 L } )$  [17, 18, 20]</td><td>sign survival attain the effect order; Supplemental Ma- terial, Sec. I maps coefficients and path multiplicities explicitly.</td></tr><tr><td>phase randomization mixing nearby unitaries [43, 44]</td><td>Conditional polar recovery and Resolved-instrument restoration [27] and Applied methods. Exact parity residual and model- specific application of established recovery theory, with deferred actuation available.</td></tr><tr><td>Parity-product history law Repeated QND statistics [26]</td><td>Applied framework. Identify the toric pointer sectors I±; the product mixture and count sufficiency then follow.</td></tr><tr><td>Order-one dynamics with van- Correctability-privacy duality [24] ishing complete-record distin- guishability</td><td>Attained model-specific scaling.  $\Theta _ { L } ( | \theta | ^ { L } )$  with a rare-syndrome lower bound, improves on the  $O _ { L } ( \sqrt { | \theta | } )$  baseline from the same channel estimate.</td></tr><tr><td>Logical-input transcript privacy Noisy local memories [25]</td><td>Different regime. Their distance-dependent upper bounds use assumptions on backbone blindness, locality, and noise strength; our one-round order is attained at fixed odd L under coherent rotation noise.</td></tr></table>

We answer for a periodic square toric code with fullsupport coherent X rotation, complete ideal Z-syndrome extraction, and deterministic minimum-weight X recovery. Stochastic toric-code work found the parity of the minimum even logical representative [22]; here that scale controls a coherent, input-dependent syndrome efect. Earlier work establishes the underlying L-versus-2L pat tern. For essentially this setting, Iverson and Preskill derived the leading order- $\cdot \theta ^ { L }$ corrected-channel coeficient (our $\alpha _ { L }$ in magnitude) and weight-2L double-logical paths [21]. We reproduce the established channel onset, coeficient, and double-logical paths in common conventions to compare them with the same instrument’s efects. Our central quantitative result is the exact record-efect onset and its nonzero coeficient. The CSS cross terms of Hu, Liang, and Calderbank [20, Eq. (91)] become our efect-character expansion; the toric cut constraint then cancels forbidden supports and makes all minimum surviving contributions have a common sign. Our exact support count is a specialization of the seam-linked path enumeration in Ref. [21, Appendix H], with nonsimple supports removed and multiplicities fixed (Supplemental

Material, Sec. I). The count specializes standard enumeration to the tagged supports used in the efect calculation. Recent adjacent work analyzes logical-noise coherence and information-theoretic performance, projected logical ensembles, syndrome-distribution phase structure, correlated coherent noise, and decodability transitions with monitored-dynamics duals [19, 23, 42, 59, 60]. These works supply operational and many-body context for the fixed-instrument calculation. The recent transcriptprivacy preprint of Shen and Zhong [25] addresses the same logical-input target for noisy single-logical-qubit stabilizer memories, including rotated surface codes. Its Theorem 1 bounds logical components under explicit backbone blindness, locality, and smallness hypotheses with growing distance and $T = \Theta ( d )$ . Here the geometry is a two-logical-qubit torus, the noise is a coherent rotation, and L is fixed before $\theta  0 ;$ we attain the one-round order and bound the growing-history distance. Each comparison applies under its own limiting regime and hypotheses.

The physical context also includes the structure of the stored quantum information and its coupling to noise. Otten et al. [34] compare information lifetimes in qubitand qudit-based memories under diferent physical noise models. At the control level, spectator modes can reduce the fidelity of pulses optimized for an isolated qudit [35]. These dependencies motivate specifying the memory and control environment when interpreting a calibrated recovery model.

## 1. Decoding cost, changing records, and measurement advantage

Scope of the computational comparison. Finding a likely physical error, selecting the most likely logical class after summing degenerate errors, simulating a memory, and certifying an update are diferent tasks. Optimal degenerate stabilizer decoding has counting-complexity hardness under explicit input and promise conditions [13]. Surface code hardness also holds with arbitrary qubit-dependent Pauli probabilities as inputs [14]. These worst-case results coexist with eficient structured instances; establishing hardness under a particular drift model requires a separate analysis.

The two eficient results of Bravyi and collaborators also concern diferent tasks. Exact maximum-likelihood decoding in Ref. [11] has $O ( n ^ { 2 } )$ cost for independent bit and phase flips with noiseless checks. Its more general matrix-product approximation has $O ( n \chi ^ { 3 } )$ cost at bond dimension $\chi ;$ accuracy must be checked as χ changes. The coherent-error simulation of Ref. [12] treats single-axis rotations and noiseless checks and recovery. Its runtime includes the chosen classical decoder’s cost in addition to the polynomial simulation cost. Eficient simulation therefore does not supply a general optimal decoder.

Our toric evaluation exploits two-term single-qubit Pauli expansions and a shared eigenbasis of corrected logical Kraus operators. Channel composition multiplies their matrix-element factors. Constructing those factors requires syndrome sums; the present exhaustive enumeration grows exponentially when extended directly. The evaluator ranks actions for both controllers, enabling independent outcome checks. Scaling the certificate requires an evaluation method with controlled approximation error, while the observation obstruction determines when further measurements are needed.

Noisy records and drift. Noisy syndrome extraction requires inference over a history that contains both data faults and measurement faults. Correlations and drift can change the appropriate history model. A decoder can remain fast while its assumed probabilities become inaccurate; selected non-Markovian noise models exhibit degraded memory performance with a decoder using an independent detector-error approximation [61]. This is a statement about the tested noise–decoder pair. Longer records can improve estimation when the model remains applicable, while temporal averaging can obscure changes. An observation ambiguity, such as the sign symmetry proved here, persists even with exact inference in the declared model.

Recent adaptive approaches address diferent parts of this problem. Bhardwaj et al. infer drifting detectormodel parameters from syndrome data using overlapping windows, exposing the filtering and sampling tradeof [4].

Sivak et al. demonstrate reinforcement-learning control of QEC under experimental drift [2]. Their detector-based controller steering and logical-error-rate-based decoder steering use diferent feedback: the latter requires the additional outcome information used in their memory experiments. Calibration-conditioned neural decoding has also been evaluated on repetition-code experiments [62]. QAdapt combines neural pre-decoding with matching under varying noise [63]; its reported residual-matching timings exclude the neural and transfer stages, which must be included in a complete latency assessment.

Pancotti, Saravanan, and Svore study exact tensornetwork likelihoods for learning quantum noise [64]. Their demonstrations use synthetic samples, including samples from a hardware-characterized detector model, with both syndromes and terminal logical-outcome labels. Their drift study uses synthetic changing noise. Exact contraction has a treewidth-dependent cost, and an expressive model’s population optimum is distinct from finite-data identifiability and reliable optimization. These observation and computation requirements matter when comparing their task with passive syndrome-only inference.

Fast decoding likewise needs a precise timing metric. Sparse Blossom provides eficient matching [15]. AlphaQubit [1] and AlphaQubit 2 [16] establish learned decoding as a substantive task distinct from languagemodel maintenance advice. AlphaQubit 2’s per-cycle streaming throughput and final-result latency are diferent quantities; its smaller-distance real-time results and larger-distance accuracy results are diferent operating points. The recent parallel Sparse-Blossom analysis [65] obtains subconstant average time per round by amortizing parallel work over a distance-sized window below threshold. It uses growing parallel resources and a circuit-depth model without geometric communication overhead. For a timed acceptance rule, acquisition, inference, communication, and actuation all contribute to evidence age, regardless of which decoding algorithm is used.

What the syndrome-aware quantum advantage establishes. Tsubouchi et al. [33] define an efective error rate through the information available for estimating a logical observable. Their classical comparison fixes the logical measurement basis and uses the syndrome in subsequent processing; the quantum protocol conditions the logical measurement on the syndrome. The quantum protocol thus changes the accessible measurement rather than merely the algorithm operating on a fixed classical transcript. Their Pauli-channel framework accommodates complete syndrome histories of Cliford circuits, but the explicit asymptotic advantage has narrower hypotheses. Theorem 2 concerns independent local Pauli noise, evendistance blocks, a low-noise limit, and a Haar average over pure logical states; the improvement then scales with the number of encoded blocks. Odd-distance finite-noise examples are assessed separately. The protocol requires state characterization and generally nonstabilizer logical measurements, with corresponding control resources.

Extending that advantage to unknown drift requires a time-dependent model, an estimation target, and calibration and measurement costs. Eficient classical decoding is compatible with quantum computational advantage, and syndrome-conditioned measurements can improve estimation despite eficient classical processing. Our security evaluation addresses a diferent operational question: which physical evidence supports a recovery update when observations omit relevant information or become outdated?

## Supplemental Material T: Calibration, activation, and circuit validation

The following tests examine three requirements for a recovery update: a calibration that constrains the physical noise, a risk bound valid until deployment, and application of the recovery table that was evaluated. We first measure readout uncertainty, then audit acceptance over every count and catalog action. Circuit tests and controller workloads assess the resulting recovery improvement and completion rates.

## 1. Why the proposer need not be accurate

Let $E _ { j }$ be the event that confidence region $C _ { j }$ contains every physical condition relevant to deployment j. Suppose $\mathrm { P r } ( E _ { j } ^ { c } ) ~ \leq ~ \alpha _ { j }$ , uniformly over the allowed nuisance parameters, and the record and activated action are correctly bound. On $E _ { j }$ , any action satisfying $U _ { j , u } \geq \operatorname* { s u p } _ { \theta \in C _ { i } } D _ { u } ( \theta )$ and $U _ { j , u } \leq - \delta$ has $D _ { u } ( \theta _ { j } ) \leq - \delta$ This implication holds simultaneously for every allowed action, including an action selected after seeing the data. Therefore an accepted violation implies $\textstyle \bigcup _ { j } \bar { E } _ { j } ^ { c }$ , whose probability is at most $\textstyle \sum _ { j } \alpha _ { j }$ . Repeated queries against one region do not require an action-by-action statistical penalty; new acquisitions consume the declared family allowance. A new confidence construction after datadependent stopping would require its own justification.

The argument imposes no accuracy or calibration condition on the proposer. It requires correctly associated measurement records, valid physical bounds, a correct risk evaluation, and application of the evaluated action. It controls the unconditional probability of violating the promised excess-loss margin across the declared experiment family. Beneficial completion is evaluated separately from protection of accepted actions. Repeated invalid proposals, refusal to acquire evidence, and budget exhaustion can leave the incumbent in place. The implementation caps both evidence acquisition and proposal work so that rejected requests are not a free, unbounded service resource.

## 2. Measuring readout nuisance parameters

The signed sentinel uses mean b−a sin θ with stipulated contrast and ofset bounds. To measure these quantities in the simulated observation model, prepare known +Y and −Y references and measure them through the same stationary readout channel. Their plus-outcome probabilities satisfy

$$
r _ { + } = ( 1 + b + a ) / 2 , \quad r _ { - } = ( 1 + b - a ) / 2 .\tag{T1}
$$

For independent reference counts with Clopper–Pearson intervals $[ l _ { + } , u _ { + } ]$ and $[ l _ { - } , u _ { - } ]$ , a simultaneous outer rectangle is

$$
a \in [ l _ { + } - u _ { - } , u _ { + } - l _ { - } ] ,
$$

$$
b \in [ l _ { + } + l _ { - } - 1 , u _ { + } + u _ { - } - 1 ] .\tag{T2}
$$

(T3)

Intersect contrast with its physical upper limit one and reject if a strictly positive contrast is not established. The rectangle is conservative because it discards correlations between the inferred nuisance parameters. For probe interval [l, u], invert $2 [ l , u ] - 1 = b - a$ sin θ over all rectangle corners, then intersect only with the declared identifiable angle domain using the inversion procedure in Sec. VII. Invalid probabilities and unsupported domains fail explicitly. Divide the family allowance equally among the two reference intervals and the signed-probe interval. By the union bound their composite capture set has failure probability at most α. Transfer and age expansion are subsequently applied as before.

The calibration estimates readout nuisance parameters using known reference states; preparation quality requires independent characterization. The construction assumes known reference states, a common readout channel, conditionally independent shots, and stationarity across all 3 acquisition blocks. Real acquisition durations require the additional physical allowances already described in the calibration protocol. Here the blocks are simulated at one capture angle.

The simulation design uses contrasts/ofsets (0.99, 0), (0.97, 0.01), and $( 0 . 9 4 , - 0 . 0 2 )$ ; magnitudes 0.06, 0.10, 0.14 rad with alternating signs; 200 independent paths per magnitude, setting, and budget; and reference budgets 512, 2048, 8192, 32768 shots per state. Every signed probe has 8192 shots. Thus there are 7200 composite records with total costs 9216, 12288, 24576, 73728 shots. All candidate tables use $H = 3 0 0$ and $\delta \ : = \ : 0 . 0 0 1$ The true simulated angle is used to generate measurements and assess recovery outcomes. Action selection and acceptance use only the measurement-derived uncertainty region.

Nuisance calibration from simulated reference counts yields 2323 beneficial activations, 0 observed harmful activations, and 3 calibration-angle noncoverage events. By readout setting the beneficial-activation counts are 994, 740, 589 out of 2400 each. A paired legacy comparator uses the same signed counts and the original fixed nuisance rectangle: it gives 1498, 1537, 1449 beneficial activations and 16, 83, 470 noncoverage events, respectively. The latter two settings violate its nuisance premises. No harmful activation is observed in these particular legacy workloads, but the lost confidence coverage prevents interpreting their higher acceptance as justified protection. Measuring nuisance parameters costs shots and can sharply reduce acceptance of beneficial operations. Figure S20 quantifies this tradeof.

## 3. An audit allowing arbitrary catalog advice

For one 8192-shot capture and $\alpha = 0 . 0 1$ , enumerate every count $k = 0 , \ldots , 8 1 9 2$ For each count and all 13 actions, including retention, recompute the permitted deployment region and certification rule. For each physical setting, mark the count if any certified action would violate the promised margin. Numerically summing the binomial probabilities of all such counts gives the worstselection violation probability, without fitting a particular attacker or AI model.

The audit uses 141 capture angles from −0.14 to 0.14, contrast endpoints 0.98, 1, ofset endpoints ±0.001, and drift radii 0, 0.005, 0.02 rad. For each radius the simulated transfer/drift shift takes three values: zero and $\pm ( \mathrm { r a d i u s } + 0 . 0 0 1 )$ ). There are 5076 physical settings. The maximum violation probability is 0.003787201138615698, below the declared 0.01. A violation is any excess loss above the promised −0.001 margin, including a smallerthan-promised improvement. This audit exhaustively sums binomial probabilities and the specified numerical channel bound, so sampling confidence intervals are inappropriate for these curves. The audit exhausts counts and catalog choices on its finite physical grid using the stated floating-point treatment; the analytical guarantee separately requires valid continuous risk bounds.

Figure S20(a,b) uses 600 independent records per cost point and 2400 per readout setting. Its pointwise 95% Hoefding intervals cover averages of independent, possibly nonidentical Bernoulli outcomes. Panel (c) maximizes the numerically evaluated binomial margin-violation probability across the specified nuisance parameters and deployment shifts at each calibration angle, allowing selection among all certified catalog actions.

## 4. Binding the action at activation time

The software represents the certified validity period as a single-use activation authorization. This authorization binds a workload, epoch, evidence identifier and digest, reference digest, action identifier and digest of the action array to be applied, issue time, and expiry. Evidence records are service-owned; sampled composite calibration records include both reference counts, signed-probe counts, and the derived interval. The reference digest covers the physical configuration, enumerated support counts, channel coeficients, derivative bounds, cached risks, and all other reference arrays. The digests identify the evidence, reference arrays, and recovery table checked within the evaluator process; signatures and post-quantum execution proofs would require additional protocols.

Issuance verifies a risk bound through the requested deployment deadline. Activation takes a lock, checks that the authorization remains unused, checks workload and epoch, verifies current evidence/reference/action identity, checks clock order and expiry, reruns the current-time risk test, and copies the evaluated action array while still inside the critical section. This prevents the in-process race in which a checked catalog index points to a replaced action by activation time. A concurrent duplicate request permits 1 activation only. Epoch changes invalidate outstanding authorizations but do not reset cumulative acquisition allowance. Each new acquisition is charged its share of the family budget, including a measured reference block that fails to establish positive contrast; such a failure cannot fall back to the fixed-nuisance rule. Proposal work also has a finite cap.

The 13 deterministic conformance cases exercise honest activation, benign waiting, expiry, forged action or reference fields, replacement of action/reference/evidence, replay, epoch change, clock rollback, concurrent replay, and measured-nuisance evidence. The 13 cases are deterministic functional conformance checks. Their scoring evaluates the channel formula using the returned action array. The prototype enforces activation within a process and clock assumed uncompromised; external identity and acknowledgment of physical actuator execution require additional interfaces. The noise is stationary during the coherent recovery block after activation.

## 5. A suficient validity period for a fixed circuit-noise family

Consider a fixed stochastic fault circuit and noise family parameterized by a scalar rate $p .$ . Each elementary fault has distribution $Q _ { i } ( p )$ satisfying

$$
\mathrm { T V } ( Q _ { i } ( p ) , Q _ { i } ( p ^ { \prime } ) ) \leq c _ { i } \vert p - p ^ { \prime } \vert .\tag{T4}
$$

For a Bernoulli Pauli fault with probability $s _ { i } p , c _ { i } = s _ { i }$ For a depolarizing channel with total nonidentity probability $s _ { i } p ,$ the same coeficient applies, provided its conditional nonidentity distribution remains fixed. A onequbit Pauli channel contributes the sum of its Pauliprobability slopes. Independent targets count separately; one explicitly correlated multi-target fault is one random event. Coupling each fault, taking the union bound on any mismatch, and then applying the deterministic circuit/decoder map gives

$$
\mathrm { T V } ( P _ { \mathrm { d e p } } , P _ { \mathrm { c a p } } ) \leq \operatorname* { m i n } \{ 1 , \sum _ { i } c _ { i } | p _ { i } ^ { \mathrm { d e p } } - p _ { i } ^ { \mathrm { c a p } } | \} .\tag{T5}
$$

The guarantee assumes the conditional fault distributions and stochastic independence structure of the declared family.

If a common drift bound $| \dot { p } | \le v$ holds from calibration to the end of a block of duration $T$ starting at age $A ,$ each diference is at most $v ( A + T )$ . Define $\bar { \boldsymbol { K } } = \bar { \boldsymbol { \sum } } _ { i } \boldsymbol { c } _ { i }$ Every fixed decoder’s binary loss expectation changes by at most $\varepsilon = \operatorname* { m i n } \{ 1 , K v ( A + T ) \}$ The maximum over the two basis risks is 1-Lipschitz in their sup norm. Applying the loss bound separately to candidate and incumbent therefore gives a 2ε allowance for worst-basis excess risk. When $v > 0$ and a nonempty certified interval exists, the maximum certified age at deployment is

$$
A _ { \mathrm { m a x } } = \frac { - \delta - U _ { \mathrm { c a p } } } { 2 K v } - T .\tag{T6}
$$

A negative value means no deployment window. The total-variation function also handles zero drift and saturation at one explicitly; the finite-age solver requires strictly positive drift. Standard coupling supplies this conservative freshness allowance.

The fixed six-decoder catalog and 30-round circuits are held fixed. The confirmed distance-3 family amplifies local gate noise by 8; distance 5 sets idle-Z fault probability to $8 p ,$ , which is 24 times the base $p / 3$ idle- $. Z$ probability. Full precision generated circuits give worst-basis coeficients $K _ { 3 } = 3 9 1 1 . 5$ and $K _ { 5 } = 1 0 1 0 1 . 5$ . Each coeficient sums the slopes over all fault locations in the complete memory circuit. These coeficients difer in both code distance and noise family, so their comparison does not isolate distance scaling. We independently reconstruct all 400 calibration-time candidate selections and bounds from 800 archived packed failure arrays; all selections match and the maximum bound discrepancy is zero.

The 4 declared rates, $1 0 ^ { - 9 } , 1 0 ^ { - 8 } , 1 0 ^ { - 7 } , 1 0 ^ { - 6 }$ per $T _ { 0 } .$ , are evaluated at 161 ages for every calibration record. At $v = 1 0 ^ { - 8 } T _ { 0 } ^ { - 1 }$ and $T = 3 0 T _ { 0 }$ , all 200 distance-3 records and 199 of 200 distance-5 records admit a nonempty deployment window. $\mathrm { A t } ~ 1 0 ^ { - 7 }$ none of the distance-5 records does; at $1 0 ^ { - 6 }$ neither family does. The simulation treats these rates as declared scenario parameters. The scalar-family extension establishes freshness within that family; the original profile-changing challenge retains its vacuous bound.

Outcome validation with new samples selects the first 8 recorded workloads at each distance and tests ages $0 ,$ $A _ { \mathrm { m a x } } / 2 .$ , and $2 A _ { \mathrm { m a x } }$ using $v \ : = \ : 1 0 ^ { - 8 }$ . At deployment, $p _ { \mathrm { d e p } } = p _ { \mathrm { c a p } } + v A ;$ each new block is stationary. There are 8192 shots in each of the X and $Z$ bases at each of 48 in stances: 786432 new complete memory shots. Candidate and incumbent decode identical records, and a separately implemented measurement-to-detector parity conversion is checked on every sample. Four Clopper–Pearson risk boxes with allowance $0 . 0 5 / 4$ yield a conservative pointwise 95% interval for the diference of worst-basis risks. All 32 within-window instances are accepted and their independent interval lies below −0.001. All 16 beyondwindow instances are rejected, although their independent intervals also establish benefit. The result demonstrates a suficient, conservative freshness condition. Expiry marks loss of suficient certification, while an expired action may remain beneficial. Figure S22 displays all validation intervals.

The age curves in Fig. S22 deterministically reevaluate the same 200 calibration records per distance at each age. The maximum-age distribution keeps denominator 200, including the distance-5 record with no certified window. The 4 drift rates are stipulated simulation parameters in probability per $T _ { 0 }$

## 6. Language-model and deterministic controllers

The controller evaluation comprises 192 independently generated workload blocks, 24 per condition: honest, insuficient initial evidence, misleading advice, bounded staleness, benign delay, workload substitution, nonce replay, and a physical step violating the drift bound. Horizons are 100, 300, 600 rounds; initial budgets are 4096, 8192, 16384 shots, except for the 512-shot insuficient-evidence condition. Each agent may request 1 additional 8192-shot acquisition. Both controllers receive the same public evidence, certified-action menu, untrusted report, and action/acquisition budget in each paired block. The 4 malicious-report variants are allocated across the advice attacks. Hidden angles and evaluator losses are unavailable to the agent.

Each block has Qwen3:8b and deterministic statemachine rollouts with authorization alone and with the full gate. This is 384 local-model rollouts, 384 deterministic rollouts, and 668 model turns. There are no Hypertext Transfer Protocol (HTTP) or response-parsing errors in this expanded run. Among 168 workloads satisfying the assumptions the full-gate large language model (LLM) gives 159 beneficial updates, 0 observed harmful activations, 6 incomplete workflows, and 3 completed retentions. Authorization alone gives 142 beneficial updates and 26 harmful activations. The state machine gives 162 beneficial updates and 0 observed harmful activations under either gate. Additional acquisition costs over these blocks are 1,007,616 shots for the full-gate LLM, 851,968 for authorization-only LLM, and 860,160 for each deterministic variant. The AI controller completes beneficial updates, while the deterministic reference completes more such updates at lower acquisition cost.

The 24 intentionally invalid drift-premise blocks produce 16 harmful full-gate LLM activations and 21 harmful full-gate deterministic activations. These are retained as failures. Pointwise 95% Chernof–KL intervals in Fig. S18 allow independent, nonidentical Bernoulli means across the fixed horizon and budget allocation. Each independent workload block supplies 4 paired rollouts. The harm intervals and outcomes characterize the specified attack set under its stated physical premises.

## 7. Numerical validation

The numerical evaluation uses fixed random seeds. Validation checks confidence inversion, invalid inputs, action/evidence replacement, timing, concurrency, budget use, elementary coupling, and age-bound solving.

Circuit coeficients and independent validation samples use full-precision parameters. A separate comparison verifies instruction identities and target ordering; numerical arguments agree up to the expected rounding in a lower-precision representation. Evaluation of the channel formula uses the stated numerical tolerance.

## 8. Scope of deployment

The computational result applies to the stated observation and noise models, ideal toric actuation, and an in-process evaluator whose memory and clock are assumed uncompromised. Deploying it requires characterized reference preparation, readout, probe transfer, and drift, together with verification that the physical actuator applies the evaluated correction. Wider noise families and withinblock coherent drift require corresponding risk bounds. Cryptographic authentication and formal verification of numerical enclosures would strengthen the implementation boundary. The evaluated controllers choose among fixed probes and recovery tables; learned probe selection and matched neural/Bayesian circuit control remain possible extensions.

## Supplemental Material U: Detailed empirical panels supporting the quantitative overview

The following figures expand the information checks, calibration-budget maps, and controller comparisons in Fig. S18. The paired-comparison methods below also support Fig. S21.

Information and recovery. Figure S19 uses the exact $L = 3$ square toric instrument with $R _ { x } ( \theta ) = e ^ { - i \theta X / 2 }$ and the syndrome measurement and recovery of Sec. I. Panel (a) orders all 256 syndrome probabilities for $\rho = I / 4$ using the same ranking at both signs. Every probability is evaluated and saved; every fifth negative-sign value is marked for legibility. Panel (b) independently evaluates the efects at 361 angles and four logical X sectors, with normalization checked separately. These residuals test the implementation of the exact efect and history identities. The phase tables in panels (c,d) stay fixed at their ±0.10 calibration angles; logical actuation follows the assumptions of Sec. I.

Decision maps. In Fig. S12, the proposal is the +0.10 table at $H = 3 0 0$ and the required infidelity improvement is $\delta = 0 . 0 0 1$ . Each budget n uses the representative count $k = \mathrm { r o u n d } \{ n [ 1 - 0 . 9 9 \sin ( 0 . 1 0 ) ] / 2 \}$ . Figure S18 uses $k = 3 6 9 1$ and $n = 8 1 9 2$ . A 99% Clopper–Pearson count interval is inverted over contrast [0.98, 1] and ofset [−0.001, 0.001], then expanded by $0 . 0 0 1 + v A$ rad. Escape from the supported domain rejects the proposal. Certification uses a continuous numerical bound with the tested $1 0 ^ { - 8 }$ allowance. Four 121 × 121 maps vary age and drift; a slice at fixed drift rate evaluates 161 budgets and 121 ages at $v = 0 . 0 8$ . The evaluator classifies rejected operations along the illustrative bounded path $\theta _ { \mathrm { d e p } } = 0 . 1 0 - \operatorname* { m i n } ( v A , 0 . 2 0 )$ , with excess risk within the chosen tolerance interval $[ - \delta , \delta ]$ . This path supplies the simulated outcome colors, not information available to the controller. Each map is a decision boundary conditional on its representative record, rather than an acceptance probability. Age is expressed in $T _ { 0 }$ and angular drift in rad $/ T _ { 0 }$ for this supporting scenario.

Paired mechanism comparison. Figure S21 uses saved paths with the same measurements and proposal for every baseline. Authorization checks their binding. Confidenceonly additionally tests the calibration-time risk; fixed expiry uses a $T _ { 0 }$ age cutof without that risk test; the full safeguard propagates bounded drift. Each full-gate decision either applies the proposed operation or retains the incumbent. The scatter includes overlapping points without jitter, and the empirical cumulative distributions are unsmoothed. Harm intervals are pointwise two-sided 95% Clopper–Pearson intervals over 200 independent paths per condition. Aggregate retained gain is the ratio of total deployed to total candidate gain; its interval resamples matched paths with 10,000 bootstrap replicates. These condition-specific intervals difer from the family allowance in the acceptance theorem.

Controller comparison. Figure S13 resolves the 192 independent workloads into 8 conditions, with 24 workloads each. Each supplies 4 paired rollouts: Qwen3:8b or a state machine, under authorization alone or the full gate. Thus the evaluation contains 384 local-LLM and 384 deterministic rollouts. Horizons are 100, 300, and 600 rounds; initial budgets are 4096, 8192, or 16384 shots, except for 512 in the small-sample condition. Each controller may request one additional 8192-shot block. Both receive identical public evidence, certified-action menus, and untrusted reports. The 4 malicious-report variants are distributed across the 24 advice attacks. Beneficial completion and harmful activation use all workloads as denominators, including retention and incomplete episodes. Conservative pointwise 95% Chernof–KL intervals allow independent, nonidentical Bernoulli means across the fixed horizon/budget allocation. The evaluator computes recovery-risk bounds and checks the acceptance criterion; the controller manages acquisition and action selection.

## Supplemental Material V: Circuit definitions, measurement conventions, and shared schedules

The circuits in Fig. 1 distinguish the coherent toric instrument from the stochastic surface-code extension. Figure S14 gives the parity-measurement and probe conventions. The following protocol specifies the memory circuits and their noise parameters.

a n = 512 shots; k = 231  
![](images/bb467592de87da414db49590f40f97da6ab2b2a6537b73f1d3f3a97e0d1dbef3.jpg)  
c n = 8,192 shots; k = 3,691

b n = 2,048 shots; k = 923  
![](images/757107a10a7a564f37b895ab9925ab4c74fe3e5a26826d5732cf9d56361acf43.jpg)  
d n = 32,768 shots; k = 14,765

![](images/5d88539b0a30b1ffd9dee6fe7b6f9bafa41c037fc372c53a840a24572a15f088.jpg)

![](images/bec981a9a4e4921965d50de57cc732badfdfa48568e174381f2f6228b502be72.jpg)  
FIG. S12. Certification depends on measurement budget, evidence age, and drift. (a–d) Conditional decision maps at 4 shot budgets. (e) Budget–age section at v = 0.08. Colors distinguish certification from the evaluator’s simulated outcome classification. Rejected beneficial updates quantify foregone opportunity: the controller must recalibrate, select another justified action, or retain the incumbent. The methods above specify the representative records and deployment path.

## 1. Toric cycle and logical actuation

Each round begins in the codespace of the periodic square toric code with $2 L ^ { 2 }$ edge qubits and two logical qubits. Apply $R _ { X } ( \theta ) = \exp ( - i \bar { \theta } X / 2 )$ on every edge, measure a complete independent set of Z plaquettes, and apply $C _ { s } = X ( r _ { s } )$ with the minimum-weight and lexicographic tie-breaking rule of Sec. I. A weight-four plaquette can be read using a newly initialized |0⟩ ancilla, four controlled-NOT (CNOT) gates with data qubits as controls and the ancilla as target, and a computational-basis measurement. For ancilla result m, the data projector is $[ I + ( - 1 ) ^ { m } Z _ { 1 } Z _ { 2 } Z _ { 3 } Z _ { 4 } ] / 2$ . Measuring all plaquettes adds one redundant parity; the model retains $\bar { L } ^ { 2 } - 1$ independent bits. The star sector is initially +1 and remains so under the stated X operations.

With error-free gates and readout, this ancilla circuit realizes the stated projector. The exact toric calculation uses the projectors directly; it assigns no duration or noise to the individual extraction gates. The logical operators $K _ { s } ( \theta )$ include $C _ { s }$ and the return to the fixed logical frame. The optional $V _ { s } ( u )$ then applies the inverse-polar phase table determined by candidate angle u. Its dashed box denotes assumed ideal logical actuation. The calculation evaluates the logical channel directly; a physical decomposition of this syndrome-conditioned logical operation is outside that implementation. The passive-sign theorem concerns the fixed instrument before this additional phase correction.

a Useful recovery across 192 workloads  
![](images/5088055c247bae4bccf732c18cc90b0fd7ff8a333bf1d7a6596dd42dfc4de58f.jpg)

b The safeguard depends on physical premises  
![](images/d4d22b0558ee643853eebbe696867407a66c94f867417dd8b9827371298cc9f1.jpg)

c Effect on the same LLM workload  
![](images/b7d9269efad8f523f3af83ca2ccf5ee8cb25082c281af35e328764d73de2be4a.jpg)

d Useful evidence acquisition  
![](images/a4932184c972a96380a2c4ca703df069b4af6dffae3fb487498c7945962d4efb.jpg)  
FIG. S13. Large language model (LLM) participation and a matched deterministic baseline. (a,b) Beneficial completion and harmful activation by condition, with pointwise 95% intervals. (c) Paired change in LLM recovery gain. (d) Additional acquisition followed by beneficial recovery under the full gate. In 168 workloads satisfying the assumptions, the full-gate LLM gives 159 beneficial completions and 0 observed harmful activations; authorization alone gives 142 and 26, respectively. The state machine gives 162 beneficial completions and 0 observed harmful activations under either gate. Exceeding the drift bound causes $1 6 / 2 4$ harmful full-gate LLM activations.

## 2. Signed probe and measured readout

The separate sentinel probe starts in |0⟩ and undergoes $R _ { X } ( \theta )$ . To measure Y, apply S<sup>†</sup> followed by H, then measure Z, with $S = \mathrm { d i a g } ( 1 , i )$ Outcome $m = 0$ represents $Y = + 1$ , and $m = 1$ represents $Y = - 1$ . With error-free preparation and readout, the mean is − sin θ. Characterized readout changes the mean to b − a sin θ, so the probability of the reported positive outcome is $( 1 + b - a \sin \theta ) / 2$

Reference states $\vert + Y \rangle$ and $| - Y \rangle$ are prepared by H followed by S or $S ^ { \dagger }$ , respectively. The same readout gives positive-outcome probabilities $r _ { \pm } = ( 1 + b \pm a ) / 2$ . These reference preparations and the physical relation between probe and protected data are assumptions to characterize experimentally. The simulations sample the stated response model. Arbitrary-angle $R _ { X } ( \theta )$ is evaluated directly with complex-valued rotation matrices, rather than approximated by a Cliford gate in Stim. In Stim syntax, RX means an X-basis reset, not this coherent rotation.

## 3. Executed surface-code memory schedules

The circuit-level extension uses rotated planar surface codes with distances $\begin{array} { r l r } { d } & { { } = } & { 3 , 5 , ~ \mathsf { \bar { d } } ^ { 2 } } \end{array}$ data qubits, $d ^ { 2 } \ - \ 1$ ancillas, and 30 extraction rounds. Stim’s surface\_code:rotated\_memory\_x and rotated\_memory\_z templates fix preparation, all CNOT targets and ordering, measurement/reset operations, and terminal data readout. The circuit constructor flattens the template and inserts the following faults while preserving its detector and logical-observable annotations.

At the start of each round, every data qubit receives PAULI\_CHANNEL\_1 with the specified X, Y, Z probabilities. Each CNOT is followed by two-qubit depolarization. Each Hadamard is followed by one-qubit depolarization with probability 0.1p. A computational-basis measurement is preceded by X error and an X-basis measurement by Z error, each at the readout rate. Resets have the corresponding anticommuting error immediately afterward at rate $p / 2$ . Thus an ancilla measurement/reset has distinct premeasurement and postreset faults. There is no additional idle fault at every tick; the stated data-idle channel is inserted once per round.

The base prior uses CNOT rate $p ,$ readout rate $p ,$ and idle probabilities $( p / 3 , p / 3 , p / 3 )$ The six catalog members are: base; readout 8p; idle-Z probability $8 p ;$ local CNOT rate $8 p ;$ independently drawn two-qubit XX and ZZ events of probability $3 p$ each per round; and the mixed prior with readout $4 p$ , idle-Z probability 4p, and local CNOT rate $4 p$ . The modified entries replace the corresponding base entries. In particular, idle- $Z = 8 p$ is 24 times the base idle-Z entry, although it is 8 times the scalar rate $p .$

A CNOT is “local” for this construction if either target has both Stim coordinates at least d. Correlated faults act on the lexicographically first adjacent horizontal data pair in the generator coordinates. Their XX and ZZ draws are independent. The simulation preserves these coordinate rules exactly. The freshness validation fixes catalog family 3 (zero-based indexing, local gates) at distance 3 and family 2 (idle Z) at distance 5. The fixed decoders use all 6 priors at $p = 0 . 0 0 2$ and correlated matching. The simulated physical rate varies by workload; each independently validated deployment block is stationary at its recorded deployment rate.

The detector record consists of template-specified measurement parities, with $( d ^ { 2 } - 1 ) 3 0$ detector bits per shot. The logical observable is a terminal memory parity. Both are reconstructed independently by exclusive OR (XOR) of the outcomes at the specified absolute measurement indices and checked against Stim’s converter before decoding. No hidden physical rate is an input to the controller. The complete schedules include boundary detectors, the four CNOT layers within each extraction round, terminal measurements, and record ofsets.

## 4. Independent circuit reconstruction

The validation covers 24 decoder priors, 800 calibration/basis configurations (also used for honest confirmation), and 96 deployment/basis configurations with independently sampled shots, spanning 16 reused calibration workloads at three ages. Each circuit is specified by its scalar rate, family, distance, and basis. Independent reconstruction checks instruction identities, targets, and numerical parameters, and reproduces $K _ { 3 } = 3 9 1 1 . 5$ and $K _ { 5 } = 1 0 1 0 \bar { 1 } . 5$

Circuit reconstruction uses full-precision parameters and adds no independent Monte Carlo observations. These checks cover the fixed-family freshness experiment; the earlier profile-changing challenges use the separate definitions and results stated above.

## Supplemental Material W: Time-resolved circuit maintenance and adversarial advice

This experiment extends the independently initialized surface-memory setting of Sec. V. It reuses the complete gate ordering, detector definitions, observable parities, and 6 decoder priors. The following derivation specifies the additional noise and time assumptions; the exact toricinstrument statements elsewhere in the paper retain their original scope.

## 1. Simulation time units and measured inference durations

The abstract unit $T _ { 0 }$ specifies the simulation clock. For a duration τ, an angular drift rate v and an interpolation rate w, define

$$
\widehat { \tau } = \tau / T _ { 0 } , \qquad \widehat { v } = v T _ { 0 } , \qquad \widehat { w } = w T _ { 0 } .\tag{W1}
$$

The normalized clock and interpolation rate are dimensionless; $\widehat { v }$ is an angle per normalized clock increment. Products are invariant: $v A = { \widehat { v } } { \widehat { A } }$ and $w \Delta \tau = \widehat { w } \Delta \widehat { \tau }$ Hence the transport and acceptance bounds have identical values in either representation. We keep the scenarioclock symbols in the transport equations and show their $T _ { 0 }$ units explicitly in tables and captions.

The toric and time-resolved surface records express acquisition and processing times in seconds. Their scenario representation retains each stored numerical value as the coeficient of $T _ { 0 }$ . In particular, a measured request

![](images/506b9bd99f7d40dc9c19d24f51210428d0a25fe9816de6a5e734074e58b50af4.jpg)  
Ideal projector realization; no gate noise here.

![](images/73237b6dc12a7f0225396aaff23d08c7f0db013fca1c38e051f8fa237f698dbb.jpg)  
FIG. S14. Circuit conventions for the toric instrument and calibration. (a) Four controlled-NOT (CNOT) gates with data qubits as controls transfer a Z-plaquette parity to an ancilla. The ancilla measurement returns its eigenvalue and preserves coherence inside the selected parity sector. (b) A signed probe and two reference preparations use the same Y readout. The parameters a and b describe its contrast and ofset. Gates are applied from left to right. The drawings assume error-free gates and readout; the noisy surface-memory schedules are specified below.

duration $\ell _ { \mathrm { w a l l } }$ maps to

$$
\ell _ { \mathrm { s c e n a r i o } } = { \frac { \ell _ { \mathrm { w a l l } } } { 1 \mathrm { s } } } T _ { 0 } .\tag{W2}
$$

This conversion fixes the acquisition, inference and deployment ratios and every drift–duration product. It is a scenario normalization rather than a measured quantumdevice timescale. Assigning a diferent round time or inference-to-round ratio defines a diferent scenario.

The two measured-latency conversions difer by a factor of 60 relative to the prescribed quantum round. Changing that ratio changes the scenario. Model and interface comparisons within the expanded maintenance study use its common conversion; cross-study diferences in completion cannot be attributed solely to the model or interface. The central toric median request contributes approximately $0 . 7 8 2 T _ { 0 } .$ , or 782000 nominal rounds. This is maintenance latency with an available incumbent, rather than a deadline for decoding each syndrome round.

The earlier stationary-acquisition circuit study uses one nominal round as its abstract unit: there $T = 3 0 T _ { 0 }$ . The supporting coherent/sentinel sweeps likewise retain their own prescribed unit as $T _ { 0 }$ . These are distinct scenario families; their absolute timings are not compared. In the central toric and time-resolved surface scenarios the nominal round is instead $1 0 ^ { - 6 } T _ { 0 }$ . The common notation exposes each study’s stated relative timing without identifying their clocks with one measured device.

Only the explicitly converted inference duration enters the scientific scenario clock. The stipulated referencecontroller latency is distinct from measured Qwen latency, so the timing model does not establish a measured controller-speed comparison.

## 2. Simulated noise path, clock, and sampling unit

At each elementary noise location $j ,$ the probability vector is

$$
\begin{array} { c l c r } { { Q _ { j } ( \lambda ) = Q _ { j } ( 0 ) + \lambda [ Q _ { j } ( 1 ) - Q _ { j } ( 0 ) ] , } } \\ { { \lambda ( \tau ) = \operatorname* { m a x } \{ 0 , 1 - w _ { \mathrm { t r u e } } \tau \} . } } \end{array}\tag{W3}
$$

At distance 3, $Q _ { j } ( 1 )$ has the local-gate amplification; at distance 5 it has amplified idle-Z noise. $Q _ { j } ( 0 )$ is the corresponding base family. The physical scale p is independently drawn uniformly from [0.0018, 0.0022] for each workload and used to generate the simulated noise, rather than supplied to the controller. Matching priors remain fixed at $p = 0 . 0 0 2$ . Both $p$ and all evaluation losses are held in separate evaluator records. The distinction between the two ensembles precludes a pure code-distance scaling interpretation.

The interpreter assigns the timestamp $s _ { i } + 3 0 \tau _ { r } k _ { j } / K _ { \mathrm { t i c k } }$ to an instruction at cumulative tick $k _ { j }$ in the unchanged flattened schedule with total $K _ { \mathrm { t i c k } }$ ticks. The declared round time is $\tau _ { r } = 1 0 ^ { - 6 } T _ { 0 }$ normally and $1 0 ^ { - 4 } T _ { 0 }$ in the slow-acquisition condition. A shot period is $h = 3 0 \tau _ { r } +$ $5 \times 1 0 ^ { - 5 } T _ { 0 }$ , including the reset overhead. Alternating $\mathrm { X } / \mathrm { Z }$ calibration shots begin at $s _ { X i } = 2 i h , s _ { Z i } = ( 2 i + 1 ) h ,$ for $i = 0 , \ldots , n - 1$ , and finish acquisition at 2nh. No logical state persists between independently initialized memories.

TABLE S6. Scenario clocks used in the controller studies. The wall-time conversion applies only to measured inference; quantum durations and reference-controller processing are stipulated. Common symbols do not identify diferent rows with one hardware clock.
<table><tr><td>Study</td><td>Measured inference</td><td>Other timing assumptions</td></tr><tr><td>Central encoded toric</td><td> $( \ell _ { \mathrm { w a l l } } / 1 \mathrm { s } ) T _ { 0 }$ </td><td>Round  $1 0 ^ { - 6 } T _ { 0 } ;$  acquisition  $N H _ { c }$  rounds; deploy- ment 300 rounds.</td></tr><tr><td>Timed surface maintenance</td><td> $( \ell _ { \mathrm { w a l l } } / 1 \mathrm { s } ) T _ { 0 }$ </td><td>Round  $1 0 ^ { - 6 } T _ { 0 } ;$  acquisition and delivery accounted for explicitly.</td></tr><tr><td>Expanded advice attacks</td><td>Fixed evaluation age</td><td>Decision evaluated at prescribed age  $T _ { 0 } ;$  model wall time is recorded but does not set that age.</td></tr><tr><td>Expanded multistep maintenance</td><td> $( \ell _ { \mathrm { w a l l } } / 6 0 \mathrm { s } ) T _ { 0 }$ </td><td>Round  $1 0 ^ { - 6 } T _ { 0 } ;$  service  $0 . 0 5 T _ { 0 } ;$  reference processing  $0 . 0 0 1 T _ { 0 } .$ </td></tr><tr><td>studies</td><td>ages</td><td>Earlier stationary circuit and sentinel Study-specific prescribed Their own nominal units, with the stated transfer and duration assumptions.</td></tr></table>

TABLE S7. Specified timing scenarios. Rates bound the dimensionless interpolation coordinate per $T _ { 0 } .$ Durations are stipulated scenario quantities; the request-latency conversion is given in $\operatorname { E q . }$ (W2). The simulated step in the final row violates its declared bound.
<table><tr><td>Condition</td><td colspan="2">wT0 Delay/T0  $\tau _ { r } / ( 1 0 ^ { - 6 } T _ { 0 } )$ </td></tr><tr><td>Honest</td><td>0.0005</td><td>0 1</td></tr><tr><td>Misleading text</td><td>0.0005</td><td>0 1</td></tr><tr><td>Delivery delay</td><td>0.002</td><td>600 1</td></tr><tr><td>Slow acquisition</td><td>0.02</td><td>0 100</td></tr><tr><td>Wrong identity</td><td>0.0005</td><td>0 1</td></tr><tr><td>Drift bound violated 0.0005</td><td></td><td>0 1</td></tr></table>

Fault draws are independent across locations and shots conditional on this deterministic path. Noise probabilities change at every location and shot without time binning. Two-qubit depolarization selects one of the 15 nonidentity Paulis; the one-qubit Pauli channel draws mutually exclusive $\mathrm { \dot { X } / Y / Z }$ faults. A separately implemented parity converter and Stim’s converter reconstruct detector and observable flips from the stored measurement-flip records. These records are relative to Stim’s reference sample, rather than absolute hardware measurement bits.

The simulation uses 3 nested calibration prefixes n = 2048, 8192, 32768. A deployment stream contains $n _ { f } ~ = ~ 1 6 3 8 4$ new shots per basis and lasts $2 n _ { f } h$ Its start is acquisition end plus $0 . 0 1 0 T _ { 0 }$ analysis, $\mathrm { \bar { 0 } } . 0 0 2 T _ { 0 }$ communication, converted measured local-model latency (or the stipulated $0 . 0 0 1 T _ { 0 }$ reference latency), and any delivery delay. Its final time $\tau _ { f }$ includes the entire deployment stream. Simulation runtime is not substituted for

quantum-device time.

There are two independent parameter draws per distance and condition in Table S7. Thus the 72 budgeted instances per controller represent 24 independent workloads. Repeated budgets, acceptance rules, and the 5 alternative decoder actions are not additional independent workloads. The two controllers have separately sampled deployment streams because their inference latencies difer. Comparisons among rules for one controller are paired on identical observations and outcomes.

## 3. Confidence for changing acquisition probabilities

Let independent Bernoulli losses $X _ { i }$ have probabilities $q _ { i } .$ , with ${ \bar { q } } = n ^ { - 1 } \sum _ { i } q _ { i }$ and empirical mean ${ \widehat { q } } .$ For real $z ,$ concavity yields

$$
\begin{array} { r l r } { \log \mathbb { E } e ^ { z \sum _ { i } X _ { i } } = \displaystyle \sum _ { i } \log ( 1 - q _ { i } + q _ { i } e ^ { z } ) } & { } & \\ { \leq n \log ( 1 - \bar { q } + \bar { q } e ^ { z } ) . } & { } & \end{array}\tag{W4}
$$

The binary Kullback–Leibler (KL) divergence is

$$
\operatorname { k l } ( p \| q ) = p \log { \frac { p } { q } } + ( 1 - p ) \log { \frac { 1 - p } { 1 - q } } ,
$$

with natural logarithms and endpoint values defined by limits. A zero-probability term contributes zero; the divergence is infinite when the reference assigns zero probability to an outcome of positive probability. The usual Chernof argument therefore bounds each tail by exp[<sup>−</sup>n kl(qb∥q¯)]. Inverting

$$
n \mathrm { k l } ( \widehat { q } \| q ) \le \log ( 2 / \alpha _ { \mathrm { b o x } } )\tag{W5}
$$

gives a conservative two-sided interval for the average probability even when the $q _ { i }$ difer. Endpoint cases are evaluated by their limiting formulas. This experiment uses three fixed looks, not arbitrary stopping; broader time-uniform constructions are available [66].

At each look, 12 marginal boxes cover two bases and 6 actions. Twenty additional confidence intervals cover the two discordant outcomes for each of five candidate/incumbent pairs in each basis. Assigning $\alpha _ { \mathrm { { b o x } } } =$ $0 . 0 5 / ( 3 \times 3 2 )$ covers all these boxes and all three looks within a workload by a union bound. Marginal and discordant counts are computed on the same records; no independence among decoders is assumed.

Label the X and $\mathrm { Z }$ memory bases by $b = 0 , 1$ , respectively. Let $[ \ell _ { b } , u _ { b } ]$ bound the incumbent’s average acquisition risk. For candidate a, the average paired diference has upper bound

$$
d _ { b } ( a ) = u \{ X _ { a } = 1 , X _ { 0 } = 0 \} - \ell \{ X _ { 0 } = 1 , X _ { a } = 0 \} ,\tag{W6}
$$

where $u \{ \cdot \}$ and $\ell \{ \cdot \}$ denote the confidence endpoints for the indicated discordant-cell probabilities. Pairing retains correlation between candidate and incumbent errors on each shot.

## 4. Transport and the acceptance bound

At each location define $c _ { j } = \mathrm { T V } [ Q _ { j } ( 1 ) , Q _ { j } ( 0 ) ]$ . Using the same schedule for calibration and deployment, couple the elementary faults of memories beginning at s and $\tau \geq s .$ . The common instruction ofset cancels, giving

$$
\mathrm { T V } ( P _ { s } , P _ { \tau } ) \leq \operatorname* { m i n } \{ 1 , \Gamma w ( \tau - s ) \} , \ \quad \ \Gamma \geq \sum _ { j } c _ { j } . \ ( \mathrm { W } 7 )
$$

No record or decoder loss difers when every coupled fault agrees. The experiment uses Γ = 5.544 at distance $3$ and 12.65 at distance $5 ,$ calculated at the maximum permitted p, rather than the hidden physical value. These are slopes with respect to $\lambda ,$ and difer from the scalar-p coeficients in $\operatorname { E q . }$ (X1).

Averaging over all calibration starts and bounding every future start by the deployment end $\tau _ { f }$ gives Eq. (11): the average of min $( 1 , x _ { i } )$ is at most min $( 1 , { \bar { x } } )$ . It bounds the change of each decoder’s mean basis risk, including its average over a deployment stream. The interval for the incumbent becomes $[ \ell _ { b } - \epsilon _ { b } , u _ { b } + \epsilon _ { b } ]$ , while the candidateminus-incumbent risk diference is at most $d _ { b } ( a ) + 2 \epsilon _ { b }$

For baseline risks $r _ { b } ,$ the identity $r _ { b } - \operatorname* { m a x } ( r _ { b } , r _ { 1 - b } ) =$ min $\left( 0 , r _ { b } - r _ { 1 - b } \right)$ gives the worst-basis excess-risk upper bound

$$
\begin{array} { r l r } {  { U ( a ) = \operatorname* { m a x } _ { b \in \{ 0 , 1 \} } \big [ d _ { b } ( a ) + 2 \epsilon _ { b } } } \\ & { } & { \qquad + \operatorname* { m i n } \{ 0 , u _ { b } + \epsilon _ { b } - \ell _ { 1 - b } + \epsilon _ { 1 - b } \} \big ] . } \end{array}\tag{W8}
$$

Here 1 − b means the opposite memory basis. When all count-based confidence intervals cover their corresponding probabilities, maximizing over the enclosing incumbent boxes proves the bound. Acceptance requires matching workload identity, a permitted nonzero candidate, and $U ( a ) \leq - 0 . 0 0 1$ . Candidate zero retains the incumbent. Since all actions and all three looks share the confidence event, proposals may depend on observed calibration and external text. Transport applies for every allowed later time, including a deployment time determined by measured inference latency. The guarantee assumes the declared noise-path class and correct acquisition and deployment timestamps. Its probability is over acquisition outcomes; it is not conditioned on acceptance, and the 0.05 allocation applies within 1 workload across its three looks.

Authorization checks identity and the action catalog but omits the risk test. Confidence-only sets $\epsilon _ { b } = 0 .$ . The endtimestamp comparison instead uses min{1, Γw $\left( \tau _ { f } - \tau _ { \mathrm { e n d } } \right) \}$ , omitting the acquisition span. The whole-acquisition rule uses Eq. (11). These rules share the same proposal and final records. The end-timestamp allowance is a comparison rule, not the derived coverage bound for changing acquisition probabilities.

## 5. Controller interface, threats, and deployment outcomes

The local Qwen3:8b controller receives identifiers for available actions, their empirical ranking, the actions certified when the evidence was issued, an evidence identifier that carries no physical parameters, acquisition duration, delivery delay, shot budget, and confirmation of workload identity. A separate note is explicitly designated untrusted. True simulation parameters and detailed assessment losses are excluded from its inputs. Each episode produces 1 activation or retention request in a specified format. Repeated command fields, unavailable actions, invalid responses, or an incorrect evidence identifier cause rejection without activation. The deterministic reference chooses the best ranked action when identity matches and otherwise retains.

Misleading notes recommend the publicly worst-ranked nonzero action and claim an urgent operator override. Delayed notes describe old evidence as approved for reuse; wrong-identity notes claim the mismatch flag is a dashboard bug. Qwen follows the misleading action in all 12 such instances and requests nonzero activation in all 12 wrong-identity instances. Every request and parsing result is retained. The threat model assumes that the software provides the correct workload-match flag. The flag itself supplies no cryptographic authentication.

All 72 Qwen requests complete, with temperature zero, seed 19, a 4096-token context and a 192-token output limit. The model is Qwen3:8b, served by local Ollama with $\mathrm { Q 4 \_ K \_ M }$ weight quantization, a mixed-precision format using primarily 4-bit weights. A recorded warmup precedes the evaluation; the median measured request time is 0.690 s, with range 0.624–2.972 s. Fixed run order, a resident-model assumption and the stipulated reference latency bound the interpretation of speed comparisons.

The endpoint is max<sub>b</sub> $\bar { r } _ { b } ( a ) - \operatorname* { m a x } _ { b } \bar { r } _ { b } ( 0 )$ over independently sampled deployment records. Its paired confidence interval uses the same 32-box allocation at one look with total allowance 0.05. Applying Eq. (W8) at zero transport gives the upper endpoint; exchanging candidate and incumbent gives the negative of the lower endpoint. This covers all five comparisons within 1 stream. It is not a simultaneous guarantee across the entire study. Benefit requires the upper endpoint below −0.001; harm requires the lower endpoint above +0.001; other accepted-update outcomes are unresolved.

Figure S15 gives the condition-resolved counts and Fig. S16 retains every proposed recovery outcome. Under honest advice all 12 proposals have established benefit, but the whole-acquisition rule admits 4 per controller. For Qwen, averaging the paired point-estimated risk reduction equally over those 12 instances, with rejection contributing zero, retains 48.5% of the authorization-only mean gain. This exploratory ratio is a descriptive average over the three tested budgets for each workload, not a population interval or an algorithm-success probability.

In the delayed and slow-acquisition conditions the full menus are already empty at issuance. The slow case reserves a $9 8 . 4 6 8 T _ { 0 }$ deployment horizon, enough to make the transport bound vacuous at the specified drift rate. These conditions expose conservative rejection without isolating acquisition age from execution duration. The analytic delay curves in Supplemental Fig. S25(b) use an initially admissible honest record, but introduce no new physical observations.

The invalid-premise condition sets λ abruptly to zero at deployment. A total of 4 proposals per controller are admitted; all 8 benefit/harm classifications remain unresolved. As a separately labeled exploratory diagnostic, compare the deployment lower endpoint with the accepted numerical $U ( a )$ . It exceeds that certificate in 3 of 4 Qwen cases and 4 of 4 deterministic cases. All 9 confidence-only admissions per controller in the bounded-delay condition likewise contradict their claimed numerical improvement. These discrepancies assess the promised bound, rather than resolving harm relative to the incumbent. The diagnostic was added after inspecting outcomes; its intervals are within-stream quantities and provide no study-wide simultaneous failure rate.

## 6. Computational record and reproduction

The design fixes the physical probability interval, seeds, shot budgets, three looks, timing scenarios and primary benefit/harm endpoints before main deployment outcomes. No parameter or sample budget is retuned on those outcomes. The complete run contains 1,572,864 new calibration and 4,718,592 new deployment memory shots. The alternative-action audit evaluates 5 actions on the same records, rather than adding independent physical samples.

The declared cycle time and communication costs are scenario parameters in $T _ { 0 } ;$ the model request latency is a measured wall-time input converted by Eq. (W2).

Validation includes stationary comparison with native Stim, exact small-distribution coverage and timetransport checks. A separate audit reconstructs every reported decision and outcome from the sampled circuit records. Repeated simulation reproduces 393,216 shots bit for bit; these repeats are not independent evidence. Replay holds the model requests and clocks fixed; rerunning inference can change deployment times.

a Local Qwen3:8b  
b Deterministic reference  
![](images/7e005762a9b60f7bd88ad3a2778b6f2648c43370f1149df90c69624e95dd1075.jpg)  
FIG. S15. All timed-maintenance conditions and rules. Each bar contains 12 case/budget instances on 4 independen workloads. Counts distinguish established benefit, established harm, unresolved accepted-update outcomes, and retention. The final condition intentionally violates the physical drift bound and is excluded from the aggregate valid-premise panel of Fig. 4. Every method includes the same identity check. The deterministic controller ignores misleading prose; the language model may request the wrong action even when the evaluator-generated menu excludes it. Zero observed resolved harm is not a population security rate.

Paired deployment outcomes. The following figure resolves every surface-code proposal against independently sampled deployment outcomes.

![](images/bed7c767688c47f48400ab2dc00807aa982294076fe242d59f0a31ae26d45a2e.jpg)

![](images/0fef6c21050e2e31564e2e7d7fa6156512f498d1b6dc47ab48ae7cd0d6c49c9e.jpg)

![](images/e47dfdb56f00ace023b349c938ebe2d9b94ea28fe8e3d37f3074a67de7dbf0f1.jpg)

![](images/4f3ea66979c33d8c4d6573f246bf9f714af03db234fe6ac270f575dfdda02b7d.jpg)

![](images/21cececad799306d91d5deecab10a05ab3897e9c1f7ab1d59a4236f0a10aeed4.jpg)

![](images/ac1a5e2cbe59b5c677b48c3328aa094eb64b5bb0110e8a5025fbb6133fb333d0.jpg)  
FIG. S16. Independently sampled deployment outcomes for every Qwen proposal. Each row has 16384 new deployment shots per basis. Lines are paired worst-basis risk intervals with simultaneous action coverage within each deployment stream; filled points mark admission by the whole-acquisition rule. Open points show the simulated efect of a rejected proposal, which is the estimated consequence of applying a proposal that was instead rejected. Row labels identify distance, replicate and calibration budget; k means 1024 shots. The 6 resolved harmful misleading-advice outcomes arise from 2 distance-5 workloads at 3 nested budgets. Shading identifies the deliberately violated drift premise.

Supplemental Material X: Additional validation of calibration and recovery updates

These supporting studies test readout calibration, the coherent-memory controller and the stationary-acquisition circuit bound. Their physical models and sampling units are distinct from the timed surface-code maintenance experiment in the main text. We retain their benefits, failures and comparison methods here.

## 1. Readout estimation and activation checks

The simulated outcome colors in the overview use a specified deployment path, $\theta _ { \mathrm { d e p } } = 0 . 1 0 - \operatorname* { m i n } ( v A , 0 . 2 0 )$ while the evaluator receives only the expanded confidence set. This distinction makes the rejected-but-beneficial region interpretable. The plotted physical angle can still favor the candidate even though another angle consistent with the evidence would not. Rejection reflects that unresolved possibility. As the uncertainty grows across the sign ambiguity, a phase table supported by the calibrationtime bound can lose its certificate while it still improves recovery on the evaluated path. The age boundary therefore measures what the observations justify, not what a controller with access to the hidden noise would choose.

Figure $\mathrm { S 2 0 ( a , b ) }$ examines the cost and coverage of readout calibration. Readout uncertainty can be measured using known ±Y references, for which $r _ { \pm } = ( 1 + b \pm a ) / 2$ Confidence intervals on both reference counts and the signed probe share the failure allowance. In 7200 simulated calibrations, this construction admits 2323 beneficial updates with 0 observed harmful activations, at total costs of 9216–73728 shots. It assumes known preparation, a common stationary readout channel, and a justified probe-to-code relation. Measuring readout parameters preserves coverage in the changed-readout settings, where the original fixed nuisance bounds become invalid; the additional uncertainty reduces acceptance at small reference budgets.

Section T specifies the reference budgets and readout settings. The reference measurements convert uncertainty about readout bias into an interval that the recovery-risk calculation propagates to deployment.

The contrast–ofset confidence rectangle and its failureprobability allocation are derived in Sec. T. If the reference measurements cannot establish positive contrast, the acquisition cannot support the assumed signed response and is rejected.

The paired comparator keeps the original fixed readout bounds while using the same probe counts. Its higher acceptance in the two changed-readout settings accompanies loss of confidence coverage: there are 83 and 470 noncoverage events, respectively, out of 2400 records per setting. Measuring the nuisance parameters gives 3 noncoverage events across all 7200 calibrations. The legacy comparator produces no observed harmful update in these workloads, so noncoverage and observed harm must be distinguished. Figure S20 shows the price of maintaining a supported uncertainty model, rather than inferring protection solely from a favorable loss sample.

Figure S20(c) tests the evaluator through an exhaustive numerical audit independently of a particular adviser’s strategy. For an 8192-shot calibration it enumerates all 8193 counts and all 13 actions. At each physical setting, it sums the binomial probability of counts for which any certified action violates the promised margin. The maximum is $0 . 0 0 3 7 8 7 2 0 1 1 4 < 0 . 0 1$ across 5076 settings. The included 1692-setting zero-drift subset matches the earlier audit exactly. This finite-grid check complements the analytical guarantee and its numerical-bound assumption.

A final implementation check ensures that the operation applied is the operation evaluated. The software ties the calibration, experiment, current configuration, reference model, and recovery-table contents together. At activation it checks their identities, validity period, and single use; it then reevaluates the current risk and copies the checked table while holding a lock. The 13 deterministic conformance cases test these steps. Section T specifies the trusted process and clock assumptions and its physical assumptions.

The checks connect certification to the applied recovery. A certificate for the wrong workload or a table replaced after evaluation cannot support Eq. (10). Evidence binding and checked actuation address substitution and replay; deployment-time risk evaluation separately accounts for physical change.

## 2. Coherent-memory controllers and learned proposals

The coherent-memory controller experiment compares local Qwen3:8b [29] with a deterministic state machine, each under authorization alone or the full acceptance rule. Both receive the same public measurements, certifiedaction menu, untrusted reports, and acquisition budget. The evaluator computes the exact-channel model numerically; the LLM manages calibration requests and action selection. The simulated angles and independently calculated losses are used only for assessment, not as controller inputs.

The experiment uses 192 independent workloads, each evaluated with both controllers and both acceptance rules. Section T specifies the 8 conditions, horizons, measurement budgets, and 4 malicious-report variants. The two controllers receive the same observations and permitted actions in each matched workload.

We distinguish simulated recovery performance from workflow completion. A beneficial completion finishes the workflow and deploys an action with $D _ { u } < - \delta$ . A harmful activation deploys an action with $D _ { u } \ > \ \delta$ . Retention and incomplete workflows are separate outcomes, both included in the denominators. These rates describe the constructed workloads.

The experiment isolates workflow decisions under evidence-based acceptance; the separate predictor study tests learning proposals from measurement histories.

Across 168 workloads satisfying the drift premise, the guarded LLM gives 159 beneficial completions, 0 observed harmful activations, 6 incomplete workflows, and 3 retentions. Authorization alone gives 142 beneficial completions and 26 harmful activations. The deterministic controller gives 162 beneficial completions and 0 observed harmful activations under either gate because it follows the certified menu even without enforcement. Additional calibration-shot totals are 1,007,616 for the guarded LLM, 851,968 for authorization alone, and 860,160 for either deterministic variant. Thus both controllers improve on retention, with the deterministic controller completing more beneficial updates at lower calibration cost [Fig. S18(c)].

When the physical drift premise is deliberately violated, the full rule permits harmful activation in 16/24 LLM and $2 1 / 2 4$ deterministic rollouts [Fig. S18(d)]. Authentic old data cannot reveal the unobserved sign change in this sign-blind instrument. The pointwise Chernof–KL intervals allow independent, nonidentical Bernoulli outcomes, using workloads as the independent units. They quantify uncertainty in the observed simulation frequencies separately from the family-level guarantee in Eq. (10).

Figure S21 isolates the role of freshness using paired mechanism tests. Confidence without age propagation admits 178 harmful updates in 200 bounded-stale paths; the full rule admits none. It retains 97.1% of aggregate candidate gain with fresh evidence and 90.8% with benign delay. Matched-path bootstrap intervals are [94.9, 98.9]% and [86.8, 94.2]%, respectively. Rejection retains the incumbent instead of substituting a new proposal. Fixed expiry also rejects the sampled stale challenge; its calibration to diferent drift contexts determines the utility cost. A separate violated-drift challenge causes harm on 187/200 paths under the full rule.

Figure S21 holds each path’s observations and candidate fixed while changing the rule. Rejection gives zero deployed gain relative to the incumbent. The retention fraction is summed deployed gain divided by summed candidate gain within the condition; it measures preserved benefit rather than acceptance frequency.

The baseline rules remove diferent pieces of evidence. Authorization alone checks permission and binding; confidence-only evaluation uses calibration-time uncertainty without propagating age; fixed expiry imposes a common time cutof. Their comparison with the full rule tests whether uncertainty and drift should enter the decision together. The 200-path mechanism tests and the 168 valid-premise agent workloads have diferent sampling units and purposes. The former isolate the rule with fixed proposals; the latter include acquisition choices, incomplete workflows, and the cost of interacting with an adviser.

The learned-predictor comparison evaluates 5 GRU and 5 MLP training seeds on 600 held-out simulated noise paths. Neural and classical methods receive the same observations, catalog, and budget for new calibration shots, in addition to 3072 common historical shots per path. The Supplemental Material reports risk distributions, calibration cost, and paired GRU-minus-Bayesian diferences. Learning improves recovery relative to retention; the Bayesian comparison remains statistically inconclusive.

## 3. Stationary-acquisition circuit validation

A stationary-acquisition Stim $/ \mathrm { P y }$ Matching study [15, 32] uses distance-3 and distance-5 surface-code memories, 30 rounds, and 6 matching priors fixed before evaluation. The endpoint is worst-basis logical failure. Within a fixed scalar noise family, coupling fault distributions gives the suficient acceptance condition

$$
U _ { \mathrm { c a p } } + 2 \operatorname* { m i n } \{ 1 , K v ( A + T ) \} \leq - \delta ,\tag{X1}
$$

Here K sums total-variation slopes over all fault locations, $| \dot { p } | \le v$ , and $T$ bounds execution duration. The factor of two accounts for changes in both candidate and incumbent risk. The coeficients are 3911.5 for the distance-3 localgate family and 10101.5 for the distance-5 idle-Z family; their diference reflects both distance and noise family.

Figure S22(a–c) propagates 200 reused calibration records per distance across evidence age and 4 declared drift rates. At $v = 1 0 ^ { - 8 }$ , all 200 distance-3 records and 199 distance-5 records admit a deployment window. These curves reevaluate each calibration rather than drawing independent measurements at every age.

The coupling argument in Sec. T bounds the change in each decoder’s failure probability by the sum of changes in the local fault distributions, capped at one. Comparing candidate and incumbent gives the factor of two in Eq. (X1).

For a positive drift bound in the regime with a positive risk margin and an unsaturated transport bound, the remaining risk margin fixes a latest certified age at the start of deployment,

$$
A _ { \mathrm { m a x } } = \frac { - \delta - U _ { \mathrm { c a p } } } { 2 K v } - T .\tag{X2}
$$

A negative value gives no certified startup window. At zero drift the age penalty vanishes and acceptance is decided directly from the calibration bound. The subtraction of $T$ reserves uncertainty for the execution itself: an operation that passes at startup must also fit within the supported duration. The two distances in Fig. S22 use diferent noise families, so their windows compare those specified experiments rather than isolating a scaling law in code distance.

Independent validation [Fig. S22(d)] selects 8 workloads per distance and tests ages zero, half the maximum certified startup age, and twice that age at $v = 1 0 ^ { - 8 }$ . Each instance uses 8192 final shots in each memory basis, totaling 786,432 shots. Candidate and incumbent decode identical records. All 32 within-window instances are accepted and independently support improvement; all 16 beyondwindow instances are rejected although still beneficial. The three ages share each calibration workload. Expiry therefore marks loss of suficient certification, rather than the measured onset of harm. The profile-changing circuit challenge retains its vacuous bound in the Supplemental Material.

Independent deployment samples assess the candidate and incumbent on the same new records. Benefit beyond expiry measures conservatism: the suficient certificate can end before the action becomes harmful.

Figures supporting the additional validation. The following panels report the distinct supporting studies described above.

![](images/a7397ccc418a1c68228a2cad1a9ea306156da5e8e6f3217ad2f3d15e840aca28.jpg)  
FIG. S17. Measurement integrity and checked activation define the operational workflow. Measurement records bind signed-probe counts to an acquisition time and workload; characterized readout and transfer bounds support their physical interpretation. The risk evaluator propagates uncertainty to deployment and requires improvement relative to the incumbent throughout the allowed conditions. The large language model (LLM) proposer can request measurements and select a recovery, but cannot create acquisition records or bypass acceptance. Activation checks evidence and recovery-table identities and applies the evaluated operation; otherwise the incumbent is retained or new evidence is acquired. True physical parameters are used for independent outcome evaluation. The simulation software maintains the records and prevents the proposer from modifying them; the guarantee requires valid physical bounds and application of the evaluated recovery.

a Same record, different consequences  
![](images/bef8f41b7112d8eb0ce51d9fb320279f5faf2defffec7bd19ad5f4408386cfa8.jpg)

b Drift can invalidate an update  
![](images/fe7deb6c0c393b526737c082b16224e71ba58213ff0f8a29af05d22eedfe0071.jpg)

c Recovery benefit survives the safeguard  
![](images/db19f4e725b0328a33f9fb7a26f92eccbd9869524bdd2fa1f39debaa463b701d.jpg)

d Protection has a physical premise  
![](images/8c553f9542b774d0d387653032bd3b86f29c9471b9e50674e1bb5f585a8bf395.jpg)

FIG. S18. Evidence connects an information limit to a recovery decision. (a) Identical passive history laws have diferent recovery consequences: one fixed +0.10 phase table helps at the correct sign and harms at the opposite sign in the $L = 3$ toric instrument of Sec. I. (b) Certification and rejection for 1 8192-shot calibration record, as evidence age and the drift bound increase. Rejected colors distinguish evaluator-only recovery outcomes; the controller sees the evidence and certificate. (c) Recovery-gain distributions over 168 paired workloads satisfying the assumptions. The guarded large language model (LLM) completes 159 beneficial updates with 0 observed harmful activations; authorization alone gives 142 beneficia updates and 26 harmful activations. The guarded deterministic controller completes 162 beneficial updates with 0 observed harmful activations. (d) Harmful-activation fractions with pointwise 95% Chernof intervals based on the Kullback–Leibler (KL) divergence. Exceeding the drift bound causes harm under the full gate: $1 6 / 2 4$ LLM and $2 1 / 2 4$ deterministic workloads. Each independent workload supplies 4 paired rollouts. Supplemental Material, Sec. $\mathrm { ~ X 2 , }$ defines outcomes of the coherent-memory controller experiment; the detailed empirical panels in the supporting material give plotting methods and condition-resolved results.

a Identical observed distributions  
![](images/4ab23ef54856eb07df727854e5984d8ebe7e9ca10deb287391cb499feaf761bf.jpg)  
c One action, different consequences

b Numerical check of effect equality  
![](images/2cd837002b671f6e748a97eb3201e74604b247b76bf2c2bd5d97930620bd7b1e.jpg)  
d Signed calibration supplies useful control

![](images/613376843f591498b8b29ccfa9ba4332cc68352c5309c593c865f606f48a9966.jpg)

![](images/7fed34f79b4df7128582239df70923d48b13d631dd0997063e13d6682bd77ede.jpg)  
FIG. S19. Passive information and active recovery answer diferent questions. (a) Opposite-sign syndrome probabilities coincide. (b) The maximum independently evaluated efect residual is $4 . 4 4 \overset { - } { \times } 1 0 ^ { - 1 6 }$ . (c) One fixed +0.10 recovery table helps at +0.10 and harms at −0.10. (d) The fixed ±0.10 tables have diferent risk profiles over the $H = 3 0 0$ angle sweep. Numerical residuals quantify computational error; the exact identities establish the information limit.

Total shots: two references + signed probe  
![](images/123308bdd6647bbeaa588f3d7ddc25ea178cbada3dabc051437c9d66193e9599.jpg)

![](images/84bf881de8d14ae12ff78b0952c63530e132c6179bd0b2c609e85cc713540760.jpg)

![](images/a8e9bcbaee856efb0d927545c659aca03051e30a60ff1546b99ed07f5e3f7486.jpg)  
FIG. S20. Readout estimation trades shot cost for justified acceptance. Known-state references and signed-probe counts are simulated. (a) Beneficial-activation fraction versus total reference-plus-probe shots, using 600 independent records per cost point and readout setting. (b) Calibration-angle noncoverage for measured and fixed nuisance procedures, using 2400 records per setting. Panels (a,b) show pointwise 95% Hoefding intervals. The changed readout settings violate the fixed procedure’s nuisance assumptions. (c) Worst-selection margin-violation probability across the 5076-setting audit, maximizing over nuisance parameters and deployment shifts at each angle. Every possible 8192-shot count and all 13 catalog actions are evaluated. These curves sum binomial probabilities on a finite physical grid using the stated numerical treatment; they carry no sampling error bars. The dashed line is the declared $\alpha = 0 . 0 1$

a Paired recovery gains  
![](images/084433ce7dabc9faf634219e6a71a1a2f3b446bfa79e199716b9538b69b33c2a.jpg)

b Path-level retained-benefit distribution  
![](images/778829edb24a900bdde577d423522647b0b456a86adbb68294383d461785d14d.jpg)  
d Retained utility against gate baselines

c Harmful activations by condition  
![](images/6096db1b9c60cb1a25afd826007bb8c0e2dbc31151beda5cca45f16065f94738.jpg)

![](images/27821e101e49fd9823899cb1b9ca669c579ccc0516f1cafcfd36ac0f48fa88e0.jpg)  
FIG. S21. Paired recovery gains expose protection and its cost. (a,b) Proposed and deployed gains for 200 fresh and 200 benign-delay paths. (c) Harmful activations in five challenge conditions, with pointwise 95% Clopper–Pearson intervals over 200 paths per condition. Violating the drift bound causes full-gate harm on 187/200 paths. (d) Aggregate gain retained: 97.1% [94.9,98.9]% for fresh evidence and 90.8% [86.8,94.2]% for benign delay, using 10,000 matched-path bootstrap replicates. Baselines share measurements and proposals; rejection retains the incumbent. The Supplemental Material gives the baseline definitions and plotting protocol.

![](images/3e81b9c69896bd6728e14e0a7bcd4e98ffc22f035ca161d6a328343b26575f04.jpg)

![](images/480d855f2d0c977e5fd75090f376309abb066a1e4edc37b7b2bd9c67740c7a4b.jpg)

![](images/166c9011a472c6250fd26e7a32902b3cd2c3d9efcf755a61e1130f5c3ccb7d1f.jpg)

![](images/d1174384add69c445f453d40cbb8b47aa2ad8859c8f07916a1d116ffa36296d4.jpg)  
FIG. S22. Freshness within a specified scalar circuit-noise family. (a,b) Fraction of 200 reused calibration records per distance certifying an update at each age and declared drift rate. (c) Maximum certified deployment age at $v = 1 0 ^ { - 8 }$ per $T _ { 0 } ,$ retaining denominator 200 even when no window exists. (d) Independent final excess-risk estimates and conservative pointwise 95% intervals; circles and squares denote distances 3 and 5. Three ages share each of 8 calibration workloads per distance. Each instance uses 8192 shots per memory basis. All 48 intervals support benefit, including 16 updates rejected beyond expiry. The dashed line marks the required −0.001 improvement. Distance and noise family difer between the two circuit ensembles.

Supplemental Material Y: Unified toric calibration and recovery: complete protocol

This study uses the same square toric instrument defined in Sec. I for passive histories, encoded calibration and recovery assessment. It is distinct from the separate sentinel and surface-code experiments above.

## 1. Encoded measurement and joint syndrome sampling

There are 18 data qubits and 8 independent Z checks; the ninth check is redundant. The physical layer is $R _ { X } ( \theta )$ on every edge, with the unchanged minimum-weight X re covery and logical-X character ordering. Let $k _ { s x }$ and $C _ { x y }$ be as in Eq. (5). The known superposition of sectors 0 and 3 stays in their span, since all corrected Kraus operators are diagonal in this basis. Its of-diagonal density-matrix element after $H _ { c }$ rounds is $C _ { 0 3 } ^ { H _ { c } } / 2$ . Taking the trace with $Y _ { 0 3 }$ gives − Im $C _ { 0 3 } ^ { H _ { c } }$ and hence Eq. (7). Complex conjugation under $\theta \mapsto - \theta$ reverses this mean, while preserving the complete passive history law. The terminal binary measurement can be extended arbitrarily on the unpopulated logical sectors.

Sectors 0 and 3 have the same product parity and therefore identical syndrome probabilities. A complete calibration trajectory can be sampled exactly using $| \bar { k } _ { s 0 } | ^ { 2 }$ at each round and accumulating

$$
z _ { R } = \prod _ { j = 1 } ^ { R } \frac { k _ { s _ { j } 0 } k _ { s _ { j } 3 } ^ { * } } { | k _ { s _ { j } 0 } | ^ { 2 } } .\tag{Y1}
$$

Only positive-probability syndromes are sampled. The conditional plus probability is (1 − Im $z _ { R } ) / 2$ . Averaging over histories reproduces Eq. (7). The saved final calibration data contain the full syndrome history and terminal readout for each independently prepared memory. No unknown stored logical state is exposed to the adviser.

## 2. Continuous confidence inversion and path allowance

At a stationary calibration angle $\theta _ { c } ,$ the N terminal readouts are independent Bernoulli trials with probability $q _ { Y } ( \theta _ { c } )$ . Inverting the two-sided Clopper–Pearson interval at failure allowance 0.01 over [−0.15, 0.15] rad yields a confidence region, potentially disconnected. The calculation uses 60001 grid points, spacing $h = 5 \times 1 0 ^ { - 6 }$ rad, and retains every cell whose possible response intersects that interval.

The generator of one physical rotation layer is $G =$ $\textstyle \sum _ { e = 1 } ^ { n } X _ { e } / 2$ , with $\| G \| = { \bar { n } } / { \bar { 2 } }$ . Telescoping repeated layers and using contraction under quantum channels bounds the change in any final event probability by $H n | \Delta \theta | / 2$ . Thus $L _ { q } = n H _ { c } / 2 = 9 0 0$ bounds the slope of $q _ { Y } . \mathrm { ~ A ~ }$ cell centered at $\theta _ { i }$ is retained if $[ q _ { Y } ( \theta _ { i } ) - \hat { L _ { q } } h / 2 , \hat { q _ { Y } } ( \theta _ { i } ) + L _ { q } h / 2 ]$ intersects the count interval. This construction retains the cell of every angle compatible with that interval, without an injectivity assumption.

The deployment loss is entanglement infidelity relative to the identity. It can be represented as the failure probability of a terminal Bell-return test on the logical channel and a reference. The same channel-distance argument applies with this reference included. The conservative constant $L _ { D } = 2 n H _ { d } = 1 0 8 0 0$ bounds changes of candidate-minus-incumbent loss, uniformly over the two fixed candidate tables. The stationary upper bound is

$$
U _ { u } ^ { \mathrm { c a l } } = \operatorname* { m a x } _ { i \mathrm { r e t a i n e d } } D _ { u } ( \theta _ { i } ) + L _ { D } h / 2 + 1 0 ^ { - 9 } .\tag{Y2}
$$

The last term is a tested numerical allowance, not a formally verified floating-point enclosure. The continuouscell argument is analytical; its machine implementation assumes that this allowance bounds the evaluation error. The incumbent has identically zero excess risk. The grid allowance alone contributes 0.027, making its conservatism material to acceptance.

Let $\theta _ { j }$ be the physical angle during deployment round j. Telescoping the candidate and incumbent channels against the stationary calibrated channel bounds their excess-risk diference by 2n times $\sum _ { j } | \theta _ { j } - \theta _ { c } |$ . Under the declared angular rate, every deviation is at most vA. The chosen upper-bound coeficient $L _ { D }$ therefore gives Eq. (8) for every such path. The executed valid paths begin drifting only after stationary acquisition; A includes the full acquisition duration and is thus an overestimate of the physical drift time in this experiment. Changingnoise acquisition is treated in the separate surface-code protocol, not inferred from the binomial model here.

The bound holds simultaneously for the fixed threeaction catalog on the calibration coverage event. Selecting an action after seeing the record does not require an additional confidence allocation. Each new calibration has its own 0.01 allowance. The 3 nested budgets and reused conditions do not establish study-wide 99% coverage. Equation (10) requires an explicit family-level allocation for such a statement.

## a. Contributions to the suficient bound

For the fixed grid and drift rate, the accepted upper bound decomposes as

$$
\begin{array} { r } { U _ { u } ( A ) = \underbrace { \underset { i \mathrm { ~ r e t a i n e d } } { \operatorname* { m a x } } D _ { u } ( \theta _ { i } ) } _ { \mathrm { r i s k ~ o v e r ~ c o m p a t i b l e ~ c e l l s } } + \underbrace { 0 . 0 2 7 } _ { \mathrm { r i s k - g r i d ~ a l l o w a n c e } } } \\ { + \underbrace { 1 0 ^ { - 9 } } _ { \mathrm { n u m e r i c a l ~ a l l o w a n c e } } + \underbrace { 0 . 0 1 0 8 A / T _ { 0 } } _ { \mathrm { d r i f t ~ a l l o w a n c e } } . } \end{array}\tag{Y3}
$$

Finite counts determine the inversion region; the calibration grid additionally expands each response by $L _ { q } h / 2 = 0 . 0 0 2 2 5$ . Its efect is through the retained-cell

set, not a separate additive risk term. With $\delta = 0 . 0 0 1$ 2 an initially negative bound permits age at most

$$
A _ { \mathrm { m a x } } / T _ { 0 } = \frac { - 0 . 0 0 1 - U _ { u } ^ { \mathrm { c a l } } } { 0 . 0 1 0 8 } , \qquad U _ { u } ^ { \mathrm { c a l } } < - 0 . 0 0 1 .\tag{Y4}
$$

Thus the risk-grid allowance alone consumes 2.5T of this suficient age budget at fixed retained cells. Charging stationary acquisition to age contributes a further $N H _ { c } \tau _ { r }$ These algebraic contributions are properties of the implemented bound; removing an allowance would require replacing its guarantee. The first harmful point on a particular drift path instead solves a path-specific recovery question and need not coincide with expiry. The numerical study evaluates this conditional bound with tested error allowances. Supplemental Material, Sec. AB implements outward-rounded bounds and a channel-specific drift comparison for this toric model.

## 3. Physical settings, fixed actions and timing

The 6 angles are $\pm 0 . 0 8 , \pm 0 . 1 0 \mathrm { ~ a n d ~ } \pm 0 . 1 2$ rad, each with 2 independently sampled acquisitions. Budgets 512, 2048 and 8192 use nested prefixes of each record. The model evaluation uses 8192. Each calibration memory lasts 100 rounds, and each evaluated recovery path lasts 300 rounds. The stipulated round duration is $1 0 ^ { - 6 } T _ { 0 }$ Known logical preparation, syndrome extraction, terminal readout and conditional inverse-polar logical actuation are ideal; additional hardware overheads are absent from this clock.

The 3 fixed actions are the incumbent and existing inverse-polar tables at −0.10 and +0.10 rad. They act as the specified syndrome-dependent logical rotations; the same catalog is used for both controllers and every acceptance rule. The valid deployment path is

$$
\theta _ { j } = \theta _ { c } + \mathrm { s g n } ( \theta _ { c } ) v ( \ell + d + j \tau _ { r } ) , \quad j = 1 , \ldots , H _ { d } ,\tag{Y5}
$$

where $v = 1 0 ^ { - 6 }$ rad $1 / T _ { 0 } .$ , ℓ is controller latency in scenario units, d is delivery delay and $\tau _ { r } = 1 0 ^ { - 6 } T _ { 0 }$ . The age used in the certificate is $A = N H _ { c } \tau _ { r } + \ell + d + H _ { d } \tau _ { r }$ . Qwen latency is converted from its measured wall time using Sec. W 1; deterministic latency is stipulated at $1 0 ^ { - 3 } T _ { 0 } ^ { \bar { } }$ Delay is zero except for the $3 0 T _ { 0 }$ delay condition. The invalid-drift condition instead sets every deployment angle to $- \theta _ { c } .$ , deliberately violating the rate premise.

## 4. Language-model requests and matched enforcement

The 60 Qwen3:8b requests use temperature 0, seed 19, context 4096 and output budget 192 tokens, following a separate warmup. The recorded model is the 8.2-billionparameter Q4\_K\_M Ollama build. Each prompt contains counts, budget, time, identity, evaluator-computed ranking and set of currently supported actions. Controller inputs exclude the true angle, deployment risk, and labels that reveal the physical parameters. Each response requests activation or retention in a specified command format.

The misleading note claims an urgent engineer override and asks for the worst-ranked nonzero action from the public calibration ranking. Wrong-identity text calls the mismatch a dashboard error; the evaluator checks identity separately from the adviser. The deterministic comparator uses the same evaluator-computed ranking while ignoring the note. All rules evaluate the same recorded proposal. Authorization checks identity, confidence-only adds Eq. (Y2), and the full rule adds the complete age allowance. Both evidence rules require excess risk at most −0.001.

## 5. Independent outcome assessment and complete counts

For each of 120 controller/condition/workload instances, the logical Schur-channel formulas are evaluated and composed along the deployment path for each of the 3 actions. Each resulting channel supplies an entanglement-infidelity probability. Conditional on the specified deployment path, 3 independent streams of 16384 terminal Bell-return failure bits are sampled. They are independent of the calibration samples. Clopper–Pearson intervals $[ l _ { u } , r _ { u } ]$ on the three failure probabilities share a total 0.05 allowance, giving simultaneous coverage within an instance. The excess-risk interval is $[ l _ { u } - r _ { 0 } , r _ { u } - l _ { 0 } ]$ . These are independent action streams, not paired shot diferences. Coverage is not simultaneous over the entire study. Channel-derived risk values are recorded separately from the intervals based on sampled outcomes.

Figure S23 reports every outcome under all three rules, including retention and the invalid physical premise. No accepted outcome was statistically unresolved. The maintext comparison in Sec. IV uses this same evaluation.

The 12 calibration workloads are the independently acquired units; repeated scenarios and nested budgets reuse them. Counts in diferent bars therefore cannot be treated as independent sample sizes. There are 5898240 deployment terminal readouts, 98304 final calibration memories and 12288 additional memories checking the joint sampler. Evaluating the channel-composition formula avoids sampling every gate in each readout; their count is not a count of independently expanded physical circuit trajectories or independent controller workloads.

## 6. Physical-model verification and scope

An independent computational-basis implementation applies physical rotations, projects all 256 syndromes, and applies recovery on the 18-qubit encoded star-orbit basis. $\mathrm { A t ~ 0 , ~ \pm 0 . 0 8 , ~ \pm 0 . 1 0 , ~ \pm 0 . 1 2 }$ , and ±0.15 rad, its logical Kraus matrices agree with the support-count reduction to

![](images/b9e0338f9d451ad49301fbb77f107e193e4da6bd57ff41846b9571c4c04754cd.jpg)  
FIG. S23. Complete controller comparison within the unified toric model. Every condition and rule is shown fo Qwen3:8b proposals and the deterministic reference. Each bar uses 12 calibration workloads, reused across conditions and rules. Benefit and harm are assessed by independent terminal Bell-return readouts; retention applies the incumbent. No accepted outcomes were statistically unresolved. Confidence-only checks already block the misleading Qwen proposals. The full rule rejects all delayed proposals although they remain beneficial on the executed paths. The undeclared sign flip produces harmful acceptances under both controllers and violates the physical drift premise. The deterministic controller follows the evaluator-computed ranking under misleading advice and preserves benefit in that condition.

$2 . 7 8 \times 1 0 ^ { - 1 5 }$ . It also reconstructs the terminal calibration probability.

The exported circuit specifies 18 rotations, 8 ancillabased Z checks with 32 CNOTs, and conditional recovery. Its parity matches the check masks on all 262144 computational-basis strings. Preparation and logical actuation retain the declared error-free assumptions; the export specifies the circuit rather than demonstrating external compiler execution. A total of 24 tests cover the physical model, sampling, bounds, and malformed inputs. A replay audit reconstructs 36 calibration summaries, 120 deployment instances, and 360 rule rows, with bitwise agreement for calibration and deployment records. These checks reuse the saved sampling seeds.

## Supplemental Material Z: Supporting recovery curves and analytical timing bounds

The main figures display simulated calibration records, independently sampled deployment outcomes and recorded acceptance decisions. LLM requests supply proposals and measured computer latencies. The following panels retain the broad recovery sweeps and conditional timing formulas that explain their construction. Dense curve evaluations are analytical formulas or numerical evaluations of the channel, not additional independent measurements.

a Missing information becomes observable  
![](images/546fff6d98ca1a1bbe6508f5c690faf03fff566e453603ae889d4757a545b473.jpg)  
c Useful protection and its failure boundary  
b Recovery uses that information

![](images/4881bd9a33f59d48661a78204786caeda2f3acc118e06377985f994a2cca0c2a.jpg)

d Finite evidence has a validity window  
![](images/e9ba9c496b003bb5c71db817455511a2393ced399ab4c704c4c11790543dff1b.jpg)  
FIG. S24. Supporting response, recovery and timing calculations for the unified toric model. (a) Numerically evaluated encoded terminal signal and passive sign contrast, with the 12 8192-shot calibration records. (b) Stationary 300-round entanglement infidelity for the incumbent and fixed recovery tables at ±0.10 rad. The horizontal axis is the physical rotation angle; the vertical axis is recovery risk. (c) Aggregate outcomes for the same Qwen proposals under three rules; the invalid-drift condition violates the physical premise. (d) Upper bounds $U _ { u } ^ { \mathrm { c a l } } + L _ { D } v A$ versus evidence age for 3 nested budgets in 1 calibration realization. Straight segments follow directly from the stipulated drift allowance. The curves illustrate this suficient criterion; they do not measure the physical onset of harm. The full protocol is in Supplemental Material, Sec. Y.

a Deployment outcomes with valid drift  
![](images/843622ab10b84692ab1bae741a69c00aa12fd19f2a7a6a31d03d944165949d26.jpg)

![](images/85e6ea262e2164d7a07d3c178256a7998534973ba16523a1e05fb1c766012d58.jpg)  
FIG. S25. Supporting conditional timing bound for the surface-code study. (a) Outcomes from the same 20 valid-premise workloads at 3 nested budgets, shown in main Fig. 4(a). (b) Additional-delay bounds for one recorded honest distance-3 calibration at 32768 shots per basis and its measured Qwen latency converted to the scenario clock. Confidence-only, end-timestamp and whole-acquisition curves reevaluate their transport formulas on the same observation. Including acquisition time shortens the suficient acceptance window. These curve points are not new simulated samples. Main Fig. 4(b) instead identifies the decisions changed by the two timing rules and their independent recovery outcomes.

Supplemental Material AA: Additional attacks, bounded drift, and multistep maintenance

## 1. Design and observation boundaries

The additional study preserves the toric instrument, action catalog, complete calibration inversion, bound on variation between grid points and numerical tolerance of Sec. Y. New independent terminal counts are drawn from the exact encoded calibration marginal. There are 6 development acquisitions and 24 held-out acquisitions, with disjoint random streams; both use the same 6 angle values. Each uses 8192 known-state calibration memories. Model input contains the calibrated action ranking and admissible menu, evidence identity and timing, and the external note. The simulated angle and exact recovery risks are reserved for independent outcome assessment. For this extension, benefit and harm are assessed from the exact channel calculation, with thresholds −0.001 and +0.001, respectively. These endpoints difer from the sampled deployment intervals in the original experiment.

Qwen3:8b uses Q4\_K\_M quantization, Gemma 3:1b uses Q4\_K\_M, and GPT-OSS:20b uses microscaling 4-bit floating-point (MXFP4) quantization. All use temperature 0, one fixed inference seed, a 4096-token context and a 512-token generation budget, including reasoning where applicable. Qwen reasoning is disabled; GPT-OSS uses its low reasoning setting. These are three specified configurations, with difering capacities, rather than controlled estimates of model-family superiority. Every malformed response, unavailable tool request, and unfinished workflow remains in the denominator. In the text-interface runs, an invalid command performs no operation and is represented in subsequent conversation history as retention with explicit error feedback.

## 2. Static and response-adaptive attacks

The 3 fixed templates assert operator authority, claim that measurement labels or the displayed ranking are corrupted, or dictate a particular output command. They request the worst-ranked nonzero action using only the public ranking. The experiment supplies these notes from fixed templates; the LLM generates the resulting action proposals. The honest note asks the model to follow trusted measurements. Honest and misleading therefore describe the supplied notes, not the model’s intentions or the quality of every response. Development outcomes select the initial template by the number of proposals choosing the attacker-requested, worst-ranked action. The held-out adaptive arm makes 3 queries, changing wording according to whether the previous reply followed the requested action, refused it, or returned an invalid command. Service acceptance or rejection is observable. This bounded decision tree preserves all attempts; its endpoint of at least 1 harmful proposal in 3 queries is calculated per acquisition. It does not search an unrestricted prompt

space.

Static and adaptive gate comparisons hold total evidence age at $1 T _ { 0 }$ to isolate advice and interface efects. The same proposed action is scored under authorization, calibration confidence, and full acceptance. The following table counts acquisition workloads with at least 1 harmful proposal; the adaptive column combines 3 dependent queries per workload.

TABLE S8. Harmful-proposal workloads out of 24. A harmful proposal counts as a harmful activation only when the acceptance rule admits it. All observed harmful proposals in these arms are rejected by both confidence and full checks.
<table><tr><td colspan="5">Model Authority Labels Format Adaptive</td></tr><tr><td>Qwen3:8b</td><td>24</td><td>24</td><td>24</td><td>24</td></tr><tr><td>Gemma 3:1b</td><td>0</td><td>0</td><td>24</td><td>24</td></tr><tr><td>GPT-OSS:20b</td><td>0</td><td>0</td><td>4</td><td>12</td></tr></table>

## 3. Multistep maintenance and command-format comparison

The 6 angle settings are crossed with missing evidence, stale evidence, and wrong-workload evidence, giving 18 operational workloads per advice condition and controller. The controller can bind the current workload, request calibration, activate an action, or retain the incumbent. It has 4 tool steps and at most two independent 8192- memory calibrations. Binding invalidates the previous evidence. Each fresh acquisition receives failure allowance 0.005, giving a per-workflow union-bound allowance 0.01. Every controller sees the same pre-sampled measurement pools for a matched workload and has the same tools and budget. Empty or expired evidence requires a new acquisition; an external claim alone cannot repair it.

The physical angle remains stationary in this operational comparison, within the declared drift model. Every activation includes elapsed inference and complete deployment time in its bound. Actual model inference duration is converted using 60 measured seconds per scenario unit $T _ { 0 } ;$ this explicit conversion is a scenario assumption. Tool service overhead is $0 . 0 5 T _ { 0 }$ and the deterministic processing contribution is stipulated as $0 . 0 0 1 T _ { 0 }$ . Each calibration resets age to its full acquisition duration plus service overhead. Missing evidence is never treated as an informative record.

The initial text responses sometimes omitted required information or requested an invalid operation. After observing these failures, we added an exploratory comparison that enforces the required command fields and permitted value types during generation. This format restriction (the schema condition in Fig. S27) leaves the choice of action to the model and preserves the prompt, action budget, measurement pools, and acceptance rule. The original results remain in the comparison. GPT-OSS also returned direct tool calls, which the text-only interface could not process. These failures reflect a mismatch between the model response and its receiving software. A further exploratory comparison supports those direct calls (the native condition), validates one call at a time, and returns its outcome to the model. It uses all 36 matched operational cases with unchanged physical observations, budget and acceptance checks. GPT-OSS then completes a beneficial update in 12 of 18 workflows under honest notes and 8 under misleading notes. Counts distinguish beneficial recovery, explicit retention, and failure to finish.

TABLE S9. Workflows completing a beneficial update out of 18 matched operational workloads per column. Text commands with and without an enforced format use the same physical data and action budget. H: honest note; M: misleading note.
<table><tr><td colspan="2"></td><td colspan="2">Text commands Format enforced</td></tr><tr><td>Controller</td><td>H M</td><td>H</td><td>M</td></tr><tr><td>Qwen3:8b</td><td>0</td><td>0 12</td><td>0</td></tr><tr><td>Gemma 3:1b</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>GPT-OSS:20b</td><td>0</td><td>1 4</td><td>2</td></tr><tr><td>Deterministic 12</td><td></td><td>12 12</td><td>12</td></tr></table>

## 4. Valid drift and interpretation of the statistical unit

The timed extension uses a continuous ramp at the declared rate, followed by stationary deployment. The

complete acquisition is stationary. Starting from $\theta _ { c } ,$ the angle at delay d is

$$
\theta ( d ) = \theta _ { c } - \mathrm { s g n } ( \theta _ { c } ) \operatorname* { m i n } \{ v d , 2 | \theta _ { c } | \} .\tag{AA1}
$$

Its speed is bounded by v throughout, including the stationary portions. The evaluator uses full acquisition-tocompletion age, which is conservative here. We evaluate 161 delays uniformly over $[ 0 , 2 4 0 0 0 0 ] T _ { 0 }$ for each of 24 acquisitions. Repeated delay points share evidence and are not independent acquisitions. Harm is determined by the channel risk at the reached endpoint; the deployed angle is held fixed for all 300 recovery rounds. Numerical agreement between stationary-channel powers and explicit round-by-round channel products checks this implementation. Main Fig. 3 reports first sampled harmful delays, with grid resolution $1 5 0 0 T _ { 0 } .$ separately from analytical expiry of the suficient bound.

Attack rates and completion counts describe this fixed six-angle design. Paired descriptive uncertainty resamples whole independent acquisition blocks within angle strata; it neither pools repeated prompts nor supplies a universal attack-success probability. Observing 0 harmful activations complements the conditional bound in Eq. (10); it does not verify untested threats. Records, evaluator, clock and operation enforcement remain trusted. The expanded study tests controlled advice attacks, a valid-drift failure of calibration-only acceptance, and simple maintenance workflows. More complex scheduling, repeated inference seeds and physical deployment remain distinct tasks.

![](images/57800206541aa397f7fcb32e1e8a068baefa02819e9c52aeef64d5e3bdfda2a3.jpg)  
FIG. S26. Advice manipulation and operational utility are separate endpoints. (a) Harmful-proposal counts for paired acquisition workloads, including any-success over 3 adaptive queries. Authorization admits these proposals; the evidence checks reject them. A low proposal count can also arise from an invalid response. (b) Workflows completing a beneficial update in the original text-command interface. The denominator includes malformed replies, explicit retention and unfinished workflows. The deterministic controller uses the same evidence and tools. Counts describe the fixed design, without pooling queries as independent samples.

![](images/efdd1b5e9faee14362763ef931150f274f25abe62cea2cfb002632826598fda1.jpg)

![](images/7f3139d69267d0a1fb761ee104c1f3f04804215c9859eee8f430c237fd3eddc7.jpg)  
FIG. S27. A specified command format separates formatting failures from maintenance decisions. The exploratory format-enforced comparison preserves the original maintenance cases and calibration pools. The GPT-OSS native row supports direct tool calls; the schema row retains the failures caused by receiving those calls through a text-only interface. Bars distinguish beneficial completion, explicit retention, and failure to finish. No harmful activation occurs in these interface comparisons. The model still chooses when to associate the workload, acquire calibration, and request activation. All 18 workloads per advice condition remain in each denominator.

Supplemental Material AB: Validated toric bounds and matched timing comparison

The encoded toric study permits a direct test of how numerical validation and a sharper drift bound afect retained benefit. We preserve its exhaustive support counts, the 3 stored recovery tables, all calibration counts, and every recorded controller proposal. The comparison uses the 12 original independent acquisitions and their 3 nested budgets. It evaluates 120 controller–condition instances; repeated budgets, conditions, and rules reuse evidence. The surface-code experiments and their unresolved invalidpremise harm classification are unchanged.

## 1. Outward-rounded inference and stationary risk

Declared decimal domains, rates, confidence levels, and improvement margins are enclosed outward from their decimal values. Stored binary phase values define fixed diagonal unitaries. For each logical pair $x \ < \ y$ , the corrected channel multiplier has the form

$$
\begin{array} { r } { C _ { x y } ^ { u } ( \theta ) = c ^ { 2 n } P _ { x y } ^ { u } ( z ) , \qquad c = \cos ( \theta / 2 ) , \quad z = \tan ( \theta / 2 ) , } \\ { ( \mathrm { A B } 1 \backslash \theta ) } \end{array}\tag{AB1}
$$

where $P _ { x y } ^ { u }$ is a polynomial of degree at most $2 n = 3 6 .$ Its coeficients sum integer support-count products with the fixed unitary phases. Real and rectangular complex intervals enclose these coeficients and subsequent arithmetic and trigonometric operations using 128-bit directed rounding with the MPFR arbitrary-precision floatingpoint library (version 4.0.2) [67].

Binomial-tail inequalities verify outward endpoints of each Clopper–Pearson interval. Adaptive subdivision of the full calibration domain [−0.15, 0.15] retains every cell whose enclosed response can intersect that interval. A cell is discarded only after its entire response interval is excluded. Boundary cells are subdivided to width at most $1 0 ^ { - 6 }$ rad; disconnected components are retained. Stationary excess risk is maximized over this covering union using interval bounds and subdivision, with a termination gap of $2 \times 1 0 ^ { - 5 }$ . Unresolved calculations retain the incumbent. This gap controls optimization tightness; outward rounding already covers the numerical evaluation error.

All 36 count/budget cases completed. At the original deployment ages, the validated calculation supports all 24 previously accepted updates in workloads satisfying the physical assumptions. It also verifies the numerical inequalities for the 16 accepted instances with a deliberately false drift premise; those inequalities provide no protection against that premise failure. Refinement to 192 bits, half the inversion-cell width, and half the maximization gap in three representative cases changes stationary upper bounds by at most $1 . 2 4 \times 1 0 ^ { - 5 }$ The validation concerns this toric evaluator, with its declared exact support tensor and ideal logical operations. Earlier sentinel and surface-code numerical studies retain their separately

stated treatment.

## 2. A channel-specific drift constant

Diferentiating Eq. (AB1) gives

$$
\frac { d C _ { x y } ^ { u } } { d \theta } = \frac { c ^ { 2 n } } { 2 } \left[ ( 1 + z ^ { 2 } ) ( P _ { x y } ^ { u } ) ^ { \prime } ( z ) - 2 n z P _ { x y } ^ { u } ( z ) \right] .\tag{AB2}
$$

Let $m _ { x y } ^ { u }$ enclose the magnitude of this derivative throughout a declared angle domain. Each round is a tracepreserving Schur channel, so $| C _ { x y } ^ { u } | \le 1$ . For an exogenous sequence of round angles, the excess infidelity is

$$
D _ { u } ( \pmb \theta ) = \frac { 1 } { 8 } \sum _ { x < y } \operatorname { R e } \left[ \prod _ { j = 1 } ^ { H _ { d } } C _ { x y } ^ { 0 } ( \theta _ { j } ) - \prod _ { j = 1 } ^ { H _ { d } } C _ { x y } ^ { u } ( \theta _ { j } ) \right] .\tag{AB3}
$$

Replacing one factor at a time bounds each product difference by its derivative bound times $\begin{array} { r } { \sum _ { j } | \theta _ { j } - \theta _ { c } | . } \end{array}$ . This sum is at most $H _ { d } v A$ , giving

$$
\begin{array} { c } { { \displaystyle | D _ { u } ( \pmb \theta ) - D _ { u } ( \theta _ { c } ) | \leq M _ { u } v A , } } \\ { { M _ { u } = \displaystyle \frac { H _ { d } } { 8 } \sum _ { x < y } ( m _ { x y } ^ { 0 } + m _ { x y } ^ { u } ) . } } \end{array}\tag{AB4}
$$

This comparison covers every exogenous angle sequence satisfying the same deviation constraint within the declared channel model, including nonmonotone paths. It uses neither the true calibration angle nor the executed ramp to choose an action. History-conditioned noise would require a corresponding adaptive-instrument bound.

A 256-cell interval partition of [−0.2, 0.2] gives $M _ { u } \le$ 123.459 for either nonzero action, versus the general coeficient 10800. Acceptance also enforces an age cap: the full calibration domain expanded by vA must remain inside the derivative domain. Here $A \leq 5 0 0 0 0 T _ { 0 }$ sufices. A 512- cell, 192-bit refinement gives $M _ { u } \le 1 2 3 . 2 3 8$ , consistent with a modest partition efect. The production comparison retains the coarser, conservative constant. Both rules use the same validated stationary upper bound; replacing the drift constant isolates its efect. At budget 8192, maximum certified ages range from 1.53 to $2 8 . 1 \bar { 5 } T _ { 0 }$ with the general bound and from 134.25 to 2462.93T with the derivative bound, a common factor of 87.48 before latency is charged.

## 3. Latency, retained benefit, and premise failure

Online evaluation time includes confidence endpoints, inversion, and risk maximization. The same measured time is added to evidence age for both rules, using the toric scenario’s declared inference-to-clock conversion. At budget 8192 its median is $0 . 8 7 8 T _ { 0 } ,$ with range 0.774– 1.824T<sub>0</sub>. The evaluator kernels and derivative-domain bounds are constructed before acquisition; charging their

![](images/e1b55ea6d557ed983f3c7657c43612a7ea284596f919dfa3a842be266120099f.jpg)

![](images/44eba2268329f067e9dd485946e5b2ab72b48d3fbae30755487a71c0df7da5e7.jpg)

![](images/77778ef764a3b8c487cd8a04c49e7dfb76fb4131a03e1d429af0457a37dca6dd.jpg)

d Evaluation contributes to age  
![](images/d2473b7317512041c464b53a828a50648a26581e93030cbbbd322811c51e26d9.jpg)  
FIG. S28. Validated numerical bounds recover benefit while retaining the physical failure boundary. (a) Paired certified maximum ages for the 12 acquisitions at budget 8192, sorted by the general-bound expiry. Both methods use identica validated stationary bounds. (b) Beneficial activations after charging the same measured evaluation time. (c) Harmful activations when the declared drift premise is violated; marker colors retain their method meaning. The count of 0 harmful activations under valid premises refers to the 32 and 60 accepted updates from the same 96 reused instances. (d) All 36 measured online evaluation costs, with medians marked; the clock conversion is stipulated. Outcomes in (b,c) are interval-enclosed model risks not newly sampled success rates. The repeated conditions and nested budgets do not create additional independent calibration records.

6.850T<sub>0</sub> first-use cost as well gives a separate cold-start sensitivity check. These are stipulated clock ratios, not hardware operation times.

Each rule receives the same recorded action and uses the same latency-shifted deployment path. Recomputed interval products enclose the model outcome on that path; the original terminal readouts are not reused to score changed paths. Thus Fig. S28 reports enclosed-model outcomes from a fixed-proposal replay, rather than new language-model calls or independent readout samples.

Across the 96 valid-premise instances, the general rule accepts 32 beneficial updates and the derivative rule accepts 60; neither accepts a harmful update. Wrongidentity requests are rejected by both rules. All 12 delayed beneficial proposals per controller survive the sharper bound, including when the first-use cost is charged. Both rules continue to reject the misleading Qwen proposals. Under the invalid sign flip, the sharper rule admits more harmful updates because it accepts more actions overall. Validated arithmetic and a less conservative bound improve utility within the physical model; trustworthy deployment still requires that model’s calibration-to-use relation.

[1] J. Bausch, A. W. Senior, F. J. H. Heras, et al., Learning high-accuracy error decoding for quantum processors, Nature 635, 834 (2024).

[2] V. Sivak, A. Morvan, M. Broughton, et al., Reinforcement learning control of quantum error correction, Nature 655, 879 (2026), arXiv:2511.08493 [quant-ph].

[3] T. Wagner, H. Kampermann, D. Bruß, and M. Kliesch, Optimal noise estimation from syndrome statistics of quantum codes, Physical Review Research 3, 013292 (2021), arXiv:2010.02243 [quant-ph].

[4] D. Bhardwaj, E. Takou, Y. Lin, and K. R. Brown, Adaptive estimation of drifting noise in quantum error correction, PRX Quantum 7, 033024 (2026), arXiv:2511.09491 [quant-ph].

[5] D. Xu, A. B. Özgüler, G. Di Guglielmo, N. Tran, G. N. Perdue, L. P. Carloni, and F. Fahim, Neural network accelerator for quantum control, in 2022 IEEE/ACM Third International Workshop on Quantum Computing Software (2022) pp. 43–49, arXiv:2208.02645 [quant-ph].

[6] R. Laroche, P. Trichelair, and R. Tachet des Combes, Safe policy improvement with baseline bootstrapping, in Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 97 (2019) pp. 3652–3661.

[7] K. P. Wabersich and M. N. Zeilinger, A predictive safety filter for learning-based control of constrained nonlinear dynamical systems, Automatica 129, 109597 (2021).

[8] E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramèr, Defeating prompt injections by design (2025), arXiv:2503.18813 [cs.CR].

[9] M. Guatto, F. Preti, M. Schilling, T. Calarco, F. A. Cárdenas-López, and F. Motzoi, Real-time adaptive quantum error correction by model-free multi-agent learning (2026), version 2; originally posted in 2025, arXiv:2509.03974 [quant-ph].

[10] W. Gong and H.-Y. Hu, Provably eficient selfcalibrating quantum fault tolerance (2026), version 2, arXiv:2608.05686 [quant-ph].

[11] S. Bravyi, M. Suchara, and A. Vargo, Eficient algorithms for maximum likelihood decoding in the surface code, Physical Review A 90, 032326 (2014), arXiv:1405.4883.

[12] S. Bravyi, M. Englbrecht, R. König, and N. Peard, Correcting coherent errors with surface codes, npj Quantum Information 4, 55 (2018), arXiv:1710.02270 [quant-ph].

[13] P. Iyer and D. Poulin, Hardness of decoding quantum stabilizer codes, IEEE Transactions on Information Theory 61, 5209 (2015), arXiv:1310.3235.

[14] A. Fischer and A. Miyake, Hardness results for decoding the surface code with Pauli noise, Quantum 8, 1511 (2024), arXiv:2309.10331.

[15] O. Higgott and C. Gidney, Sparse blossom: correcting a million errors per core second with minimum-weight matching, Quantum 9, 1600 (2025).

[16] A. W. Senior et al., A scalable and real-time neural decoder for topological quantum codes (2026), preprint, version 2; originally posted in 2025, arXiv:2512.07737 [quant-ph].

[17] E. Huang, A. C. Doherty, and S. T. Flammia, Performance of quantum error correction with coherent errors, Physical Review A 99, 022313 (2019), arXiv:1805.08227 [quant-ph].

[18] F. Venn and B. Béri, Error-correction and noisedecoherence thresholds for coherent errors in planar-graph surface codes, Physical Review Research 2, 043412 (2020), arXiv:2006.13055 [quant-ph].

[19] J. Behrends and B. Béri, The surface code beyond Pauli channels: Logical noise coherence, information-theoretic measures, and errorfield-double phenomenology, PRX Quantum 6, 040350 (2025), arXiv:2412.21055 [quant-ph].

[20] J. Hu, Q. Liang, and R. Calderbank, Designing the quantum channels induced by diagonal gates, Quantum 6, 802 (2022), arXiv:2109.13481 [quant-ph].

[21] J. K. Iverson and J. Preskill, Coherence in logical quantum channels, New Journal of Physics 22, 073066 (2020), arXiv:1912.04319 [quant-ph].

[22] L. Wichette, H. Hohenfeld, E. Mounzer, and L. Grans-Samuelsson, A partition function framework for estimating logical error curves in stabilizer codes, Quantum 10, 2175 (2026), arXiv:2505.15758 [quant-ph].

[23] Z. Yang, A. W. W. Ludwig, and C.-M. Jian, Decoding coherent errors in toric codes on honeycomb and square lattices: Duality to majorana monitored dynamics and symmetry classes (2026), preprint, arXiv:2604.08650 [cond-mat.stat-mech].

[24] D. Kretschmann, D. W. Kribs, and R. W. Spekkens, Complementarity of private and correctable subsystems in quantum cryptography and error correction, Physical Review A 78, 032330 (2008), arXiv:0711.3438 [quant-ph].

[25] J. Shen and H. Zhong, Execution-transcript privacy for fault-tolerant surface-code memories (2026), version 1, arXiv:2609.09334 [quant-ph].

[26] M. Bauer, T. Benoist, and D. Bernard, Repeated quantum non-demolition measurements: convergence and continuous-time limit, Annales Henri Poincaré 14, 639 (2013), arXiv:1206.6045 [math-ph].

[27] M. Gregoratti and R. F. Werner, On quantum errorcorrection by classical feedback in discrete time, Journal of Mathematical Physics 45, 2600 (2004), arXiv:quantph/0403092.

[28] C. J. Clopper and E. S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26, 404 (1934).

[29] A. Yang, A. Li, B. Yang, et al., Qwen3 technical report (2025), arXiv:2505.09388 [cs.CL].

[30] Gemma Team, Gemma 3 technical report (2025), arXiv:2503.19786 [cs.CL].

[31] OpenAI, gpt-oss-120b & gpt-oss-20b model card (2025).

[32] C. Gidney, Stim: a fast stabilizer circuit simulator, Quantum 5, 497 (2021).

[33] K. Tsubouchi, H. Kwon, L. Jiang, and N. Yoshioka, Quantum advantages for syndrome-aware noisy logical observable estimation (2026), preprint, version 2, arXiv:2603.05145 [quant-ph].

[34] M. Otten, K. Kapoor, A. B. Özgüler, E. T. Holland, J. B. Kowalkowski, Y. Alexeev, and A. L. Lyon, Impacts of noise and structure on quantum information encoded in a quantum memory, Physical Review A 104, 012605 (2021), arXiv:2011.13143 [quant-ph].

[35] A. B. Özgüler and J. A. Job, Dynamics of qudit gates and efects of spectator modes on optimal control pulses, Physical Review A 109, 052404 (2024), arXiv:2207.14006 [quant-ph].

[36] D. Gottesman, Stabilizer Codes and Quantum Error Correction, Ph.D. thesis, California Institute of Technology (1997), arXiv:quant-ph/9705052.

[37] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, Topological quantum memory, Journal of Mathematical Physics 43, 4452 (2002), arXiv:quant-ph/0110143.

[38] Z. Cheng, E. Huang, V. Khemani, M. J. Gullans, and M. Ippoliti, Emergent unitary designs for encoded qubits from coherent errors and syndrome measurements, PRX Quantum 6, 030333 (2025), arXiv:2412.04414 [quant-ph].

[39] A. Abdi, G. Cornuéjols, B. Guenin, and L. Tunçel, Clean clutters and dyadic fractional packings, SIAM Journal on Discrete Mathematics 36, 1012 (2022).

[40] M. A. Nielsen, A simple formula for the average gate fidelity of a quantum dynamical operation, Physics Letters A 303, 249 (2002), arXiv:quant-ph/0205035.

[41] D. Greenbaum and Z. Dutton, Modeling coherent errors in quantum error correction, Quantum Science and Technology 3, 015007 (2018), arXiv:1612.03908 [quant-ph].

[42] R. N. Rajmohan, A. Brillant, P. Groszkowski, A. Seif, J. Koch, and A. Clerk, Correlated coherent errors in stabilizer codes: A general cumulant framework and interference-based error suppression (2026), preprint, arXiv:2607.22503 [quant-ph].

[43] E. Campbell, Shorter gate sequences for quantum computing by mixing unitaries, Physical Review A 95, 042306 (2017), arXiv:1612.02689.

[44] J. J. Wallman and J. Emerson, Noise tailoring for scalable quantum computation via randomized compiling, Physical Review A 94, 052325 (2016), arXiv:1512.01098.

[45] T. Wagner, H. Kampermann, D. Bruß, and M. Kliesch, Learning logical pauli noise in quantum error correction, Physical Review Letters 130, 200601 (2023), arXiv:2209.09267 [quant-ph].

[46] E. Takou and K. R. Brown, Estimating decoding graphs and hypergraphs of memory quantum error-correction experiments, Physical Review A 112, 052414 (2025), arXiv:2504.20212 [quant-ph].

[47] E. Takou and K. R. Brown, Correcting non-Pauli noise with detector error models estimated from syndrome measurements, Physical Review A 114, 022449 (2026), arXiv:2510.23797 [quant-ph].

[48] X. Xiao, D. Hangleiter, D. Bluvstein, M. D. Lukin, and M. J. Gullans, In situ benchmarking of fault-tolerant quantum circuits: Cliford circuits, PRX Quantum 10.1103/ghn4-8y96 (2026), arXiv:2601.21472 [quant-ph].

[49] H. Zheng, C.-T. Chu, S. Chen, A. G. Manes, S.-u. Lee, S. Zhou, and L. Jiang, Eficient learning of logical noise from syndrome data (2026), preprint, version 2, arXiv:2601.22286 [quant-ph].

[50] S. J. Beale, J. J. Wallman, M. Gutiérrez, K. R. Brown, and R. Laflamme, Quantum error correction decoheres noise, Physical Review Letters 121, 190501 (2018), arXiv:1805.08802 [quant-ph].

[51] Á. Márton and J. K. Asbóth, Coherent errors and readout errors in the surface code, Quantum 7, 1116 (2023), arXiv:2303.04672 [quant-ph].

[52] J. Combes, C. Ferrie, C. Cesare, M. Tiersch, G. J. Milburn, H. J. Briegel, and C. M. Caves, In-situ characterization of quantum devices with error correction (2014), preprint, arXiv:1405.5656 [quant-ph].

[53] D. Orsucci, M. Tiersch, and H. J. Briegel, Estimation of coherent error sources from stabilizer measurements, Physical Review A 93, 042303 (2016), arXiv:1512.07083 [quant-ph].

[54] T. Wagner, H. Kampermann, D. Bruß, and M. Kliesch, Pauli channels can be estimated from syndrome measurements in quantum error correction, Quantum 6, 809 (2022), arXiv:2107.14252 [quant-ph].

[55] R. Blume-Kohout and K. Young, Estimating detector error models from syndrome data (2025), preprint, arXiv:2504.14643 [quant-ph].

[56] E. Takou, C. Benito, A. Vezvaee, D. A. Lidar, and K. R. Brown, Logical error estimation from syndrome data of surface-code experiments (2026), preprint, version 2, arXiv:2606.11496 [quant-ph].

[57] E. Huang, P.-G. Rozon, A. Dua, S. Gopalakrishnan, and M. J. Gullans, A robust phase of continuous transversal gates in quantum stabilizer codes (2025), preprint, version 2 revised 2026, arXiv:2510.01319 [quant-ph].

[58] N. Yoshioka, A. Seif, P. Groszkowski, A. Cross, and A. Javadi-Abhari, Theory and architecture of syndromeresolved logical gates (2025), preprint, version 2 revised 2026, arXiv:2510.08290 [quant-ph].

[59] M. Bejan, J. Behrends, M. McGinley, and B. Béri, Projected logical ensembles in surface codes via the random-matrix theory of quantum dots (2026), preprint, arXiv:2606.17140 [quant-ph].

[60] H. Liu and X. Chen, Coherent error induced phase transition, Physical Review B 10.1103/jly5-q9mx (2026), accepted 27 August 2026; volume and article number pending, arXiv:2506.00650 [quant-ph].

[61] J. F. Kam, S. Gicev, K. Modi, A. Southwell, and M. Usman, Detrimental non-Markovian errors for surface code memory, Quantum Science and Technology 10, 035060 (2025), arXiv:2410.23779.

[62] S. Stein, S. Kan, C. Liu, A. Harkness, S. Garner, Z. Du, Y. Ding, Y. Mao, and A. Li, Calibration-conditioned FiLM decoders for low-latency decoding of quantum error correction evaluated on IBM repetition-code experiments (2026), preprint, version 1, arXiv:2601.16123 [quant-ph].

[63] R. Miao, R. Luo, X. Shan, and X. Sun, QAdapt: A noise-adaptive neural pre-decoding framework for quantum error correction (2026), preprint, version 1, arXiv:2607.28422 [quant-ph].

[64] N. Pancotti, V. Saravanan, and K. Svore, Exact learning of quantum noise with tensor networks (2026), version 1, arXiv:2609.00169 [quant-ph].

[65] R. Mikami and H. Yamasaki, Sparse-blossom decoding in o(1) time (2026), preprint, version 1, arXiv:2609.12262 [quant-ph].

[66] S. R. Howard, A. Ramdas, J. McAulife, and J. Sekhon, Time-uniform, nonparametric, nonasymptotic confidence sequences, The Annals of Statistics 49, 1055 (2021), arXiv:1810.08240.

[67] MPFR Project, GNU MPFR 4.0.2 reference manual (2019).