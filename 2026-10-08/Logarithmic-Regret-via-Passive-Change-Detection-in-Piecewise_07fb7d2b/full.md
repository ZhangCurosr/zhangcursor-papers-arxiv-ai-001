# Logarithmic Regret via Passive Change Detection in Piecewise-Stationary Self-Tuning Regulation

A.Ch. Madhusudanarao Department of Computer Science and Automation Indian Institute of Science, Bengaluru, India madhusudanar@iisc.ac.in

Rahul Singh Laboratoire de Recherche de l’EPITA, Paris, France rahulsingh0188@gmail.com

## Abstract

We study minimum-variance control of an unknown autoregressive system with exogenous inputs and coeficients that change at unknown times. Under bounded independent disturbances, fixed detection gaps, stability and feasibility conditions, and suficient time between changes, we prove O((C + 1) log((T + 1)/δ)) regret with probability at least 1 − δ, where T is the horizon and C the number of changes. Unlike switching bandits, where unselected arms can change unobserved, admissible plant changes provide information during exploitation: the correct feasible controller leaves only the disturbance in the output, whereas a detectable change raises output energy under the old controller. PIECE-CD explores initially and after alarms, then uses gated recursive least squares for control. Its energy test compares windowed output power with a threshold above the noise floor; the extension to unstable controller mismatches also monitors the reference controller’s input proposal. We control false alarms across the horizon and prove logarithmic detection delay. Inputs are clipped to prescribed bounds. Logarithmic regret also holds under an explicit condition ensuring that clipping becomes inactive after a finite burn-in. Under the stated feasibility conditions, the extended detector covers destabilizing changes with detectable excess energy over a fixed window.

## 1 Introduction

Self-tuning regulation (STR) seeks minimum-variance control of an unknown autoregressive system with exogenous inputs (ARX). The ideal controller cancels the predictable part of the next output, leaving only the disturbance. Classical work established stability and asymptotic eficiency (Goodwin et al., 1981; Becker et al., 1985; Kumar and Praly, 1987; Lai, 1986; Lai and Wei, 1987); Singh et al. (2024) obtained finite-time logarithmic regret for a stationary plant. We ask whether this rate survives unknown changes in the plant coeficients. A controller must detect these changes and identify the new plant, while avoiding repeated exploration caused by false alarms.

The key observation is that minimum-variance control can supply evidence of a change during exploitation. With the correct feasible controller, $y _ { t + 1 } = w _ { t + 1 }$ , so output power equals the disturbance variance $\sigma _ { w } ^ { 2 } .$ . A change that raises output power under the old controller can therefore be detected from ordinary control observations. This difers from switching bandits, where an unselected arm can change unobserved and even a two-breakpoint class has an Ω( T) minimax lower bound (Garivier and Moulines, 2011). The distinction is informational: our guarantee requires a fixed detectable energy gap and suficient time between changes, and does not apply to arbitrary nonstationarity.

Our regret measures the predictable output energy left uncancelled by the learner; its expectation equals the expected excess output energy above the disturbance level ((3)–(4)).

Our algorithm, PIECE-CD, extends Probing Inputs for Exploration in Certainty Equivalence (PIECE). It explores initially and after an alarm, then controls with gated recursive least squares (RLS) around an estimated reference gain. Its principal version combines an output-energy test with a boundary alarm on the reference controller’s requested input. The latter controls clipping errors when a changed plant is driven by the old gain. Outputs remain dependent, and alarm times depend on the observed history; both features matter for the guarantees.

## Contributions.

1. A common detector and logarithmic regret. Theorem 5 proves no false alarms, logarithmic detection delay, and $\mathcal { R } _ { T } = O ( ( C + 1 ) \log ( ( T + 1 ) / \delta ) )$ with probability at least 1 − δ, where C is the number of changes. Under own-regime feasibility, the theorem covers stable mismatches with a stationary power gap and potentially unstable mismatches with a finite-window energy gap, without identifying which condition applies. For the compact minimum-phase ARX class, non-Schur mismatches imply a uniform finite-window gap. The rate holds for fixed model, feasibility, and detectability bounds under the stated dwell-time requirements.

2. Identification and control after random restarts. We establish excitation and estimation bounds for complete exploration blocks conditional on the history at each restart. An RLS potential bound controls both accepted RLS actions and reference fallback actions. Chronological failure-event bounds combine these results with detection without assuming independent episodes or conditioning on future alarms.

3. Supporting guarantees under stable mismatches. An energy-only baseline isolates the stationary-gap argument (Theorem 4). A separate extension replaces uniform clipping feasibility by a steady-state slack condition and a finite burn-in, preserving logarithmic regret (Theorem 6).

## 1.1 Related Work

Our closest predecessor is stationary PIECE (Singh et al., 2024). Its covariance analysis treats sparse, recurring exploration episodes. Here the exploration phase is a contiguous block after a data-dependent restart. This requires conditional excitation guarantees and a control analysis that remains valid before the next change or alarm, together with a detector for the dependent observations generated by feedback.

Detect-and-restart methods are well established in piecewise-stationary bandits, including CUSUM-based CD-UCB (Liu et al., 2018), M-UCB (Cao et al., 2019), GLR-klUCB (Besson et al., 2022), and the modular DAB framework (Huang et al., 2026). Their exploration schedules address changes hidden by partial feedback. Additional structural information can permit logarithmic gap-dependent bounds, as when all arms change simultaneously (Mukherjee and Maillard, 2019). PIECE-CD similarly uses a restricted detectable class; its monitoring signal comes from outputs observed during exploitation. Sequential detection provides the statistical background (Veeravalli and Banerjee, 2014; Xie et al., 2021; Tartakovsky et al., 2014).

Fully unknown online linear-quadratic regulation has square-root local minimax regret (Simchowitz and Foster, 2020). STR instead has no positive input penalty, and its ideal controller leaves only the innovation in the output. This property makes the noise floor an observable reference for detecting the changes admitted here.

## 2 MODEL AND SCOPE

We consider a scalar ARX system of known orders $p , q \geq 1$ whose coeficients change C times over a horizon $T .$ . Let $0 = \tau ^ { ( 0 ) } < \tau ^ { ( 1 ) } < \cdots < \tau ^ { ( C ) } < \tau ^ { ( C + 1 ) } = T$ denote the deterministic segment boundaries, with the $C$ interior boundaries unknown. On segment $c \in \{ 0 , \ldots , C \}$ ,

$$
\begin{array} { c } { { y _ { t + 1 } = \displaystyle \sum _ { i = 1 } ^ { p } a _ { i } ^ { ( c ) } y _ { t - i + 1 } + \displaystyle \sum _ { j = 1 } ^ { q } b _ { j } ^ { ( c ) } u _ { t - j + 1 } + w _ { t + 1 } , } } \\ { { \tau ^ { ( c ) } \leq t < \tau ^ { ( c + 1 ) } . } } \end{array}\tag{1}
$$

Here $y _ { t }$ is the output, $u _ { t }$ the control input, and $w _ { t }$ the disturbance. The unknown parameter vector is constant within each segment:

$$
\boldsymbol { \theta } ^ { ( c ) } = ( a _ { 1 } ^ { ( c ) } , \ldots , a _ { p } ^ { ( c ) } , b _ { 1 } ^ { ( c ) } , \ldots , b _ { q } ^ { ( c ) } ) ^ { \mathsf { T } } .
$$

Write $\theta _ { t } = \theta ^ { ( c ) }$ for $\tau ^ { ( c ) } < t < \tau ^ { ( c + 1 ) }$

We use Euclidean vector norms, induced spectral matrix norms, and $\| x \| _ { M } ^ { 2 } = x ^ { \mathsf { T } } M x$ . We use two regressors:

$$
\begin{array} { r l } & { \psi _ { t } = ( y _ { t } , \ldots , y _ { t - p + 1 } , u _ { t - 1 } , \ldots , u _ { t - q + 1 } ) ^ { \mathsf { T } } , } \\ & { \phi _ { t } = ( y _ { t } , \ldots , y _ { t - p + 1 } , u _ { t } , \ldots , u _ { t - q + 1 } ) ^ { \mathsf { T } } . } \end{array}
$$

The control regressor $\psi _ { t } \in \mathbb { R } ^ { d }$ , with $d = p { + } q - 1$ , contains quantities available before $u _ { t }$ is chosen. The identification regressor $\phi _ { t } \in \mathbb { R } ^ { p + q }$ additionally includes $u _ { t } .$ , so that $y _ { t + 1 } = ( \theta ^ { ( c ) } ) ^ { \mathsf { T } } \phi _ { t } + w _ { t + 1 }$ The lagged-input block in $\psi _ { t }$ is empty when $q = 1$

For a generic parameter vector $\theta = ( a _ { 1 } , \ldots , a _ { p } , b _ { 1 } , \ldots , b _ { q } ) ^ { \mathsf { T } }$ with $b _ { 1 } \neq 0$ , define the minimumvariance gain

$$
\lambda ( \theta ) : = - \frac { 1 } { b _ { 1 } } ( a _ { 1 } , \ldots , a _ { p } , b _ { 2 } , \ldots , b _ { q } ) ^ { \mathsf { T } } .\tag{2}
$$

On segment $^ { c , }$

$$
\begin{array} { r } { y _ { t + 1 } = b _ { 1 } ^ { ( c ) } \{ u _ { t } - \lambda ( \theta ^ { ( c ) } ) ^ { \mathsf { T } } \psi _ { t } \} + w _ { t + 1 } . } \end{array}
$$

When feasible, the ideal action $\boldsymbol { u } _ { t } ^ { \star } = \lambda ( \theta ^ { ( c ) } ) ^ { \mathsf { T } } \psi _ { t }$ therefore leaves $y _ { t + 1 } = w _ { t + 1 }$ . We measure learning by the predictable output energy

$$
\mathcal { R } _ { T } : = \sum _ { t = 0 } ^ { T - 1 } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } .\tag{3}
$$

Let $\mathcal { F } _ { t }$ include the observations and randomization available through the choice of $u _ { t }$ . Then $v _ { t + 1 } : = y _ { t + 1 } - w _ { t + 1 }$ is $\mathcal { F } _ { t } .$ -measurable and $\mathbb { E } [ v _ { t + 1 } w _ { t + 1 } \mid \mathcal { F } _ { t } ] = 0$ . Hence

$$
\mathbb { E } \mathcal { R } _ { T } = \mathbb { E } \sum _ { t = 0 } ^ { T - 1 } ( y _ { t + 1 } ^ { 2 } - w _ { t + 1 } ^ { 2 } ) .\tag{4}
$$

This is an expectation identity; the realized excess output cost also contains $2 \textstyle \sum _ { t } v _ { t + 1 } w _ { t + 1 }$ . Our high-probability bounds concern $\mathcal { R } _ { T }$

We first state the conditions for the stable-mismatch baseline. The extension in Section 5 replaces its mismatch and feasibility conditions; the model, noise, and open-loop switching conditions remain in force. Appendices A–B give the corresponding constants and exploration conditions.

Assumption 1 (Baseline conditions). The controller knows a compact class $\Theta \subset \mathbb { R } ^ { p + q }$ containing every $\theta ^ { ( c ) }$ . Each $\theta \in \Theta$ has a Schur-stable AR polynomial, a minimum-phase input polynomial, and $0 < b _ { \mathrm { m i n } } \leq b _ { 1 } \leq \bar { b }$ . The autonomous output dynamics are uniformly exponentially stable under admissible switching. The disturbances are $i . i . d .$ , independent of the available past, and satisfy $\mathbb { E } w _ { t } = 0 , \mathbb { E } w _ { t } ^ { 2 } = \sigma _ { w } ^ { 2 } > 0$ , and $| w _ { t } | \le B _ { w }$ , where $\sigma _ { w } ^ { 2 }$ and $B _ { w }$ are known.

Uniform clipping feasibility holds: $| \lambda ( \theta ) ^ { \mathsf { T } } \psi | \leq B _ { u }$ for every $\theta \in \Theta$ and every regressor $\psi$ reachable under admissible switching, bounded disturbances, and inputs in $[ - B _ { u } , B _ { u } ]$

For each change $c \in \{ 1 , \ldots , C \}$ , let $\theta ^ { - } = \theta ^ { ( c - 1 ) }$ and $\theta ^ { + } = \theta ^ { ( c ) }$ . The unclipped loop obtained by applying the old gain $\lambda ( \theta ^ { - } )$ to the new plant $\theta ^ { + }$ has matrix $A _ { \Delta } : = A ( \theta ^ { - } , \theta ^ { + } )$ , where $A ( \theta ^ { - } , \theta ^ { + } ) : = A _ { \mathrm { c l } } ( \lambda ( \theta ^ { - } ) , \theta ^ { + } )$ , satisfying $\left\| A _ { \Delta } ^ { k } \right\| \le C _ { \Delta } \rho _ { \Delta } ^ { k }$ for all $k \geq 0$ , with known common $C _ { \Delta } < \infty$ and $\rho _ { \Delta } \in ( 0 , 1 )$ . Its stationary output power satisfies

$$
J _ { y } ( \lambda ( \theta ^ { - } ) ; \theta ^ { + } ) \ge \sigma _ { w } ^ { 2 } + \Delta\tag{5}
$$

for a known fixed $\Delta > 0$ . Here $\begin{array} { r } { J _ { y } = \sigma _ { w } ^ { 2 } \sum _ { k \geq 0 } ( e _ { 1 } ^ { \mathsf { T } } A _ { \Delta } ^ { k } e _ { 1 } ) ^ { 2 } } \end{array}$ , where $e _ { 1 }$ is the first coordinate vector in $\mathbb { R } ^ { d }$

For exploration length H and delay bound D defined below, the dwell conditions, when $C \geq 1$ are

$$
\begin{array} { r } { \tau ^ { ( 1 ) } \geq H , \qquad T - \tau ^ { ( C ) } \geq D , } \\ { \tau ^ { ( c + 1 ) } - \tau ^ { ( c ) } \geq D + H \quad ( 1 \leq c < C ) . } \end{array}\tag{6}
$$

The final exploration may be truncated at $T$

Stability and detectability serve diferent purposes. Stability controls the discrepancy between the learner and the exact old-gain loop; the power gap supplies the detection signal. A parameter change preserving the minimum-variance gain can remain at the noise floor and is excluded by the gap condition. The algorithm uses known model, noise, excitation, and gap bounds for tuning, while the segment parameters and change times are unknown. The alternative feasibility conditions are stated with their respective results in Sections 5 and 6.

Switching stability and clipped inputs give known deterministic bounds $| y _ { t } | ~ \le ~ B _ { Y }$ and $\lVert \psi _ { t } \rVert \leq B _ { \psi } \mathrm { ; }$ ; their explicit values are given in (40).

## 3 PIECE-CD

We first specify the energy-only baseline; Section 5 adds the boundary alarm for potentially unstable mismatches. At initialization and after each alarm, PIECE-CD estimates a fixed reference gain from H exploration rounds. During exploitation, a gate keeps RLS actions close to that reference while the detector monitors output energy. Use $\delta _ { 0 } = \delta / [ 3 2 ( T + 1 ) ]$ for estimation and RLS and $\delta _ { d } = \delta / 2$ for detection.

Exploration and the reference controller. Let r denote the start time of the current exploration phase, with $r = 0$ at initialization. After an alarm upon observing $y _ { t + 1 }$ , the next exploration phase starts at $r = t + 1$ . For $t = r , \ldots , r + H - 1$ , draw i.i.d. mean-zero inputs from a prescribed distribution with positive variance and support in $[ - B _ { u } , B _ { u } ]$ , independently of the disturbances. Let $r _ { 0 } = \operatorname* { m a x } \{ p , q \}$ , choose $H > r _ { 0 }$ , and retain the rows $\mathcal { T } _ { r } = \{ r + r _ { 0 } , \ldots , r + H { - } 1 \}$ , whose lagged measurements all follow the restart.

Fix a ridge parameter $\nu > 0$ . Write $I _ { k }$ for the $k \times k$ identity matrix and $\Pi _ { \Theta }$ for a Euclidean

nearest-point projection onto Θ. After completing exploration, compute

$$
\begin{array} { l } { \displaystyle V _ { r } ^ { \theta } = \nu I _ { p + q } + \sum _ { t \in \mathcal { T } _ { r } } \phi _ { t } \phi _ { t } ^ { \mathsf { T } } , } \\ { \displaystyle \widetilde { \theta } _ { r } = ( V _ { r } ^ { \theta } ) ^ { - 1 } \sum _ { t \in \mathcal { T } _ { r } } \phi _ { t } y _ { t + 1 } , } \\ { \displaystyle \widehat { \theta } _ { r } = \Pi _ { \Theta } ( \widetilde { \theta } _ { r } ) , \qquad \widehat { b } _ { r } = b _ { 1 } ( \widehat { \theta } _ { r } ) . } \end{array}\tag{7}
$$

The reference gain $\overline { { \lambda } } _ { r } = \lambda ( \widehat { \theta } _ { r } )$ and $\widehat { b } _ { r }$ remain fixed until the next restart.

The computable parameter and reference-gain error bounds $E _ { \theta } ( H , \delta _ { 0 } )$ and $E _ { \lambda } ( H , \delta _ { 0 } )$ are defined in (46)–(47). For the prescribed choice of H and a complete exploration block within segment c, with probability at least $1 - 2 \delta _ { 0 }$ conditional on the restart history,

$$
\begin{array} { r l } & { \left\| \widehat { \theta } _ { r } - \theta ^ { ( c ) } \right\| \leq E _ { \theta } ( H , \delta _ { 0 } ) , } \\ & { ~ E _ { \theta } ( H , \delta _ { 0 } ) = O \left( \sqrt { \frac { \log \left( ( T + 1 ) / \delta \right) } { H } } \right) . } \end{array}\tag{8}
$$

This reference accuracy sets the gate tolerance: $g _ { H } = 4 E _ { \lambda } ( H , \delta _ { 0 } )$ . Write $\varepsilon _ { u } = 5 E _ { \lambda } ( H , \delta _ { 0 } ) B _ { \psi }$ for the resulting reference-tracking tolerance used in the analysis.

Gated control and recursive estimation. Let $\widehat { \lambda } _ { t }$ denote the current RLS gain estimate and $V _ { t }$ its regularized $d \times d$ Gram matrix. Initialize both from the exploration data:

$$
V _ { r + H } = \nu I _ { d } + \sum _ { t \in \mathcal { Z } _ { r } } \psi _ { t } \psi _ { t } ^ { \mathsf { T } } , \qquad \widehat { \lambda } _ { r + H } = \overline { { \lambda } } _ { r } .
$$

At each subsequent control round, form the input proposal $z _ { t }$ and clip it to the prescribed bounds:

$$
\begin{array} { r l } & { z _ { t } = \Big \{ \widehat \lambda _ { t } ^ { \mathsf { T } } \psi _ { t } , \quad | ( \widehat \lambda _ { t } - \overline { \lambda } _ { r } ) ^ { \mathsf { T } } \psi _ { t } | \le g _ { H } \left\| \psi _ { t } \right\| , } \\ & { } \\ & { u _ { t } = \operatorname* { m i n } \{ B _ { u } , \operatorname* { m a x } \{ - B _ { u } , z _ { t } \} \} . } \end{array}\tag{9}
$$

The gate limits departure from the reference action used for detection; clipping enforces the prescribed input bound.

After observing $y _ { t + 1 }$ , construct the response $x _ { t + 1 } ^ { \lambda }$ and update RLS:

$$
\begin{array} { r l } & { x _ { t + 1 } ^ { \lambda } = u _ { t } - \frac { y _ { t + 1 } } { \widehat { b } _ { r } } , } \\ & { \widehat { \lambda } _ { t + 1 } = \widehat { \lambda } _ { t } + V _ { t + 1 } ^ { - 1 } \psi _ { t } \cdot ( x _ { t + 1 } ^ { \lambda } - \widehat { \lambda } _ { t } ^ { \top } \psi _ { t } ) . } \end{array}
$$

$$
V _ { t + 1 } = V _ { t } + \psi _ { t } \psi _ { t } ^ { \mathsf { T } } ,\tag{10}
$$

RLS uses every exploitation observation, including rounds with a reference fallback or an active clip. The analysis accounts for bias from estimating $b _ { 1 } ^ { ( c ) }$

Monitoring and restarting. Let the positive integers h and m denote the sampling interval and detection-window size, respectively, chosen below. After exploration, fix the recording times $s _ { j } = r + H + j h , j = 1 , 2 , . . . .$ At each $s _ { j }$ , store $y _ { s _ { j } } ^ { 2 }$ and retain the most recent m values. Once $j \geq m$ , raise an alarm if

$$
\widehat { J } ( s _ { j } ) : = \frac { 1 } { m } \sum _ { i = j - m + 1 } ^ { j } y _ { s _ { i } } ^ { 2 } \geq \sigma _ { w } ^ { 2 } + \frac { \Delta } { 2 } .\tag{11}
$$

Control and RLS updates continue every round; only the energy test is subsampled. An alarm at $s _ { j }$ clears the estimates and stored values, sets $\boldsymbol { r } = \boldsymbol { s } _ { j }$ , and starts a new exploration phase.

Choosing the exploration and detection lengths. Choose the smallest integer $H > r _ { 0 }$ satisfying (93), (51), and (52) in Appendix B. These conditions ensure informative exploration and suficiently accurate reference estimates for control and detection. Choose the smallest $h \geq 1$ satisfying

$$
\frac { \sigma _ { w } ^ { 2 } C _ { \Delta } ^ { 2 } \rho _ { \Delta } ^ { 2 h } } { 1 - \rho _ { \Delta } ^ { 2 } } \leq \frac { \Delta } { 8 } .
$$

Recorded outputs remain dependent; their spacing permits concentration around lag-h conditional second moments.

Let $B _ { o }$ be the known common bound in (50) on disturbances, actual outputs, and outputs of the new plant driven by the exact old gain from the state at the change. Set the window size m and associated detection-delay bound D to

$$
m = \left\lceil \frac { 5 1 2 B _ { o } ^ { 4 } } { \Delta ^ { 2 } } \log \frac { 4 T ^ { 2 } } { \delta _ { d } } \right\rceil , \qquad D = ( m + 1 ) h .\tag{12}
$$

This window size controls fluctuations across the horizon’s repeated tests. For fixed problem constants, $h = O ( 1 )$ and $H , m , D = O ( \log ( ( T + 1 ) / \delta ) )$ .

Algorithm 1 PIECE-CD: stable-mismatch baseline   
1. Initialize ${ \overline { { r = 0 } } } .$ . Stop at T, including during an incomplete phase.   
2. Explore for H rounds; compute (7), the reference gain, and the RLS initialization. Clear the detector   
bufer and fix the sampling times $s _ { j } = r + H + j h . $   
3. At every exploitation round, apply (9), observe $y _ { t + 1 } .$ , and update (10). At sampling times, update the   
bufer and test (11) once it is full.   
4. If an alarm occurs on observing $y _ { t + 1 }$ , set $r = t + 1$ and return to step 2; otherwise continue step 3.

## 4 FINITE-TIME ANALYSIS

The proof bounds detection delays and false alarms, then controls prediction error on each unchanged exploitation interval. These bounds yield the stable-mismatch baseline in Theorem 4. Section 5 extends the detector analysis to potentially unstable mismatches under alternative input feasibility.

Detection guarantee. Let ${ \mathcal E } _ { \mathrm { r e f } }$ be the event that every reached, complete, unchanged exploration block whose restart precedes the first detector failure produces an accurate reference gain. Here a detector failure is a false alarm or a change not detected by its D-round deadline; the precise reference-event definition is given in (74). Let $\mathcal { E } _ { \mathrm { d e t } }$ be the event of no false alarms and exactly one alarm per change. Writing $\widehat { \tau } ^ { \left( c \right) }$ for the output time of the alarm corresponding to change c, this event also requires

$$
\tau ^ { ( c ) } < \widehat { \tau } ^ { ( c ) } \leq \tau ^ { ( c ) } + D , \qquad c = 1 , \ldots , C .\tag{13}
$$

Theorem 2 (Alarm correspondence). Under the complete assumptions in Appendices A and B and the tuning above,

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { r e f } } \cap \mathcal { E } _ { \mathrm { d e t } } ^ { c } ) \leq \delta _ { d } .\tag{14}
$$

Proof. See Theorem 18 and Appendix H.

This joint bound controls detector failure on accurate-reference histories; referenceestimation failures enter Theorem 4 separately.

Why the detector works. Let $\mathcal { F } _ { t }$ denote the information available when $u _ { t }$ is selected, including the controller’s randomization and the selected input. Suppose the current reference is accurate and the plant has not changed since its exploration. The gate and input feasibility then give, at each recorded output time $s ,$

$$
y _ { s } = w _ { s } + v _ { s } , \qquad | v _ { s } | \leq \varepsilon _ { y } ,\tag{15}
$$

where $v _ { s } : = y _ { s } - w _ { s }$ is $\mathcal { F } _ { s - 1 }$ -measurable and $\varepsilon _ { y }$ is the deterministic output-tracking bound in (49). Thus $w _ { s }$ is fresh noise relative to $v _ { s }$ . The tuning makes $v _ { s } ^ { 2 }$ small enough that concentration of $w _ { s } ^ { 2 }$ and $2 w _ { s } v _ { s }$ keeps unchanged windows below the threshold with high probability.

After a change from $\theta ^ { - }$ <sup>−</sup> to $\theta ^ { + }$ at control time τ, we compare the learner with an auxiliary process that applies the exact old gain $\lambda ( \theta ^ { - } )$ to the new plant. It starts from the learner’s regressor $\psi _ { \tau }$ and uses the same future disturbances:

$$
\begin{array} { r l } & { \psi _ { \tau } ^ { \mathrm { a u x } } = \psi _ { \tau } , } \\ & { \psi _ { t + 1 } ^ { \mathrm { a u x } } = A _ { \Delta } \psi _ { t } ^ { \mathrm { a u x } } + e _ { 1 } w _ { t + 1 } , \qquad t \geq \tau . } \end{array}
$$

Here $A _ { \Delta }$ is the closed-loop matrix of the new plant under the old gain, $e _ { 1 } \in \mathbb { R } ^ { d }$ is the first coordinate vector, and $y _ { s } ^ { \mathrm { a u x } } = e _ { 1 } ^ { \mathsf { T } } \psi _ { s } ^ { \mathrm { a u x } }$ is the auxiliary output; see (68). For a fixed calendar time s with $s - h \geq \tau$ , the fresh disturbances over the preceding h steps give

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } | \mathcal { F } _ { s - h } ] = ( e _ { 1 } ^ { \top } A _ { \Delta } ^ { h } \psi _ { s - h } ^ { \mathrm { a u x } } ) ^ { 2 } + \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { h - 1 } ( e _ { 1 } ^ { \top } A _ { \Delta } ^ { j } e _ { 1 } ) ^ { 2 } } } \\ & { } & \\ & { } & { \geq \sigma _ { w } ^ { 2 } + \frac { 7 \Delta } { 8 } . } \end{array}\tag{16}
$$

The initial-state term is nonnegative, and the choice of h loses at most $\Delta / 8$ from the stationary power gap. On an accurate-reference history, the gate and stability also ensure $| y _ { s } ^ { 2 } - ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } | \leq$ $\Delta / 8$ until the next alarm or plant change. With high probability, the average of m fully postchange auxiliary output squares lies at most $\Delta / 8$ below its average conditional power. Unless an alarm occurs earlier, the actual average is therefore at least $\sigma _ { w } ^ { 2 } + 5 \Delta / 8$ , above the threshold. The sampling calendar supplies such a window by $\tau + D$ ; the dwell-time condition prevents another change before this deadline.

Restart times depend on past observations. For each potential restart, the proof uses a continuation with the same control and recording rules but suppresses subsequent restarts. It agrees with the implemented trajectory through the first alarm. Joint failure bounds over reached histories give the global guarantee without conditioning on future detector success; see Appendices H and I.

With restarts and detection delays controlled, we next bound the RLS regret accumulated during unchanged exploitation.

Theorem 3 (Unchanged exploitation prefixes). Suppose a full exploration block $[ r , r + H )$ lies in one regime and its Gram and estimation events hold. Under the baseline conditions and tuning, conditional on the history at $r + H ,$ , with probability at least $1 - \delta _ { 0 } ,$ , simultaneously for every exploitation prefix ending no later than the next plant change, restart, or horizon,

$$
\sum _ { t = r + H } ^ { r + H + n - 1 } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } \leq R _ { E } ( n , \delta _ { 0 } ) .\tag{17}
$$

Here $R _ { E }$ is the explicit bound in (78). For fixed model, noise, and exploration constants and the prescribed choice of H, it satisfies $R _ { E } ( T , \delta _ { 0 } ) = O ( \log ( ( T + 1 ) / \delta ) )$ .

Theorem 19 and Appendix G give the conditional, time-uniform bound. Its main mechanism is as follows.

RLS argument. Consider an unchanged exploitation interval in segment c after successful exploration, and write $\lambda = \lambda ( \theta ^ { ( c ) } )$ and $b _ { \star } = b _ { 1 } ^ { ( c ) }$ . Define the RLS proposal error and the applied action error by

$$
\begin{array} { r } { a _ { t } = ( \widehat { \lambda } _ { t } - \lambda ) ^ { \mathsf { T } } \psi _ { t } , \qquad e _ { t } = u _ { t } - \lambda ^ { \mathsf { T } } \psi _ { t } . } \end{array}
$$

Reference accuracy, the gate, and feasibility of the ideal action imply $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$ . Thus it sufices to control the cumulative RLS proposal error, even on rounds that use the reference input or clipping.

The response used by RLS satisfies

$$
\begin{array} { r l } & { x _ { t + 1 } ^ { \lambda } = \lambda ^ { \top } \psi _ { t } + \alpha e _ { t } + \eta _ { t + 1 } , } \\ & { ~ \alpha : = 1 - \frac { b _ { \star } } { \widehat { b } _ { r } } , \qquad \eta _ { t + 1 } : = - \frac { w _ { t + 1 } } { \widehat { b } _ { r } } . } \end{array}
$$

Here $\alpha e _ { t }$ is the predictable bias caused by estimating $b _ { \star }$ , whereas $\eta _ { t + 1 }$ is bounded and conditionally mean zero. To track estimation progress, define

$$
\begin{array} { r l } & { Q _ { t } = ( \widehat { \lambda } _ { t } - \lambda ) ^ { \mathsf { T } } V _ { t } ( \widehat { \lambda } _ { t } - \lambda ) , } \\ & { \ell _ { t } = \psi _ { t } ^ { \mathsf { T } } V _ { t + 1 } ^ { - 1 } \psi _ { t } . } \end{array}
$$

Exploration controls both |α| and $\ell _ { t } .$ . Consequently, the exact RLS identity gives

$$
Q _ { t + 1 } - Q _ { t } \leq - c _ { 0 } a _ { t } ^ { 2 } + 2 a _ { t } \eta _ { t + 1 } + 2 \ell _ { t } \eta _ { t + 1 } ^ { 2 } ,\tag{18}
$$

with $c _ { 0 } > 0$ defined in (77).

With probability at least $1 - \delta _ { 0 }$ , a time-uniform martingale bound controls the cumulative mixed term by a fraction of $c _ { 0 } \textstyle \sum _ { t } a _ { t } ^ { 2 }$ plus a confidence term. Bounded noise and the logdeterminant bound on $\textstyle \sum _ { t } \ell _ { t }$ control the final term logarithmically. Summing (18), using $Q _ { t } \geq 0$ and $Q _ { r + H } \leq \overline { { Q } } _ { H }$ , therefore bounds $\textstyle \sum _ { t } a _ { t } ^ { 2 }$ simultaneously over every exploitation prefix. Finally,

$$
( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } = b _ { \star } ^ { 2 } e _ { t } ^ { 2 } \leq \bar { b } ^ { 2 } a _ { t } ^ { 2 } ,
$$

which yields Theorem 3. Appendix G gives the identities and concentration details.

Theorem 4 (Stable-mismatch regret guarantee). Under Assumption 1, the exploration setup in Section 3, and the precise model and exploration bounds in Appendices $A { \mathrm { - } } B ,$ choose the smallest feasible H and the detector tuning (12). Then, with probability at least $1 - \delta$ , there are no false alarms, every change has exactly one alarm satisfying (13), and

$$
\mathcal { R } _ { T } \leq C K _ { D } D + ( C + 1 ) K _ { X } H + ( C + 1 ) R _ { E } ( T , \delta _ { 0 } ) ,\tag{19}
$$

where the fixed per-step delay and exploration bounds $K _ { D } , K _ { X }$ are defined in (84)–(85). Consequently, when all problem constants (model orders, model-class and noise bounds, initial state, exploration and ridge settings, stability constants, clipping threshold, and power gap) are held fixed as T grows,

$$
\mathcal { R } _ { T } = O \bigg ( ( C + 1 ) \log \frac { T + 1 } { \delta } \bigg ) .\tag{20}
$$

Proof. See Theorem 20 and Appendices I–J.

Proof sketch of Theorem 4. The regret splits into three parts: each change costs at most $K _ { D } D$ before it is detected (per-step cost $K _ { D }$ times delay D), each of the C+1 exploration blocks costs at most $K _ { X } H$ , and Theorem 3 bounds the regret on the remaining unchanged exploitation periods. For the probability, each restart can fail in three ways (insuficient excitation, inaccurate estimate, or RLS failure), each with probability at most $\delta _ { 0 }$ , and at most $T + 1$ restarts are possible. Adding the detector failure probability $\delta _ { d }$ gives a total of $3 ( T + 1 ) \delta _ { 0 } + \delta _ { d } < \delta$ . Because restart times are random, failures are counted in time order, each only if its restart is actually reached; Appendix I gives the details.

## 5 DETECTION UNDER STABLE AND UNSTABLE MIS-MATCHES

The baseline theorem in Section 4 assumes that the new plant remains stable under the old controller. We now allow this mismatched closed loop to be unstable. Its stationary power may then be unavailable, and clipping can alter its output-energy signal. We address these issues by converting both detectability conditions below into a common finite-window gap and combining the energy test with an alarm on the reference gain’s input proposal. Fixed-window comparisons replace the comparison over the full detection delay used in the stable analysis.

Two sources of detectable energy. Let $\mathcal { P } \subseteq \Theta ^ { 2 }$ be the prescribed class of admissible adjacent parameter pairs. For an actual change $z _ { c } = ( \theta ^ { ( c - 1 ) } , \theta ^ { ( c ) } ) , c = 1 , \dots , C$ , write

$$
A _ { c } : = A _ { z _ { c } } : = A ( \theta ^ { ( c - 1 ) } , \theta ^ { ( c ) } ) ,
$$

the closed-loop matrix of the new plant under the old gain. We cover $\mathcal { P }$ by two possibly overlapping subfamilies. In the stable subfamily ${ \mathcal { P } } _ { \mathsf { S } }$ , known common constants $C _ { 5 } < \infty , \rho _ { 5 } \in$ (0, 1), and $\underline { { \Delta } } _ { \mathsf { S } } > 0$ satisfy

$$
\begin{array} { c } { { \left\| A _ { c } ^ { k } \right\| \leq C _ { 5 } \rho _ { 5 } ^ { k } \quad ( k \geq 0 ) , } } \\ { { J _ { y } ( \lambda ( \theta ^ { ( c - 1 ) } ) ; \theta ^ { ( c ) } ) \geq \sigma _ { w } ^ { 2 } + \underline { { { \Delta } } } _ { 5 } . } } \end{array}\tag{21}
$$

In the finite-visible subfamily $\mathcal { P } _ { \mathsf { F } }$ , a known common $\underline { { \Delta } } _ { \mathsf { F } } > 0$ instead satisfies

$$
\mathcal { V } ( z _ { c } ) : = \sigma _ { w } ^ { 2 } \sum _ { j = 1 } ^ { d } ( e _ { 1 } ^ { \mathsf { T } } A _ { c } ^ { j } e _ { 1 } ) ^ { 2 } \geq \underline { { \Delta } } _ { \mathsf { F } } ,\tag{22}
$$

where $e _ { 1 } \in \mathbb { R } ^ { d }$ is the first coordinate vector. The coeficients $e _ { 1 } ^ { \mathsf { T } } A _ { c } ^ { j } e _ { 1 }$ describe a disturbance’s efect on later outputs, so $\mathcal { V } ( z _ { c } )$ measures its excess energy over the first d lags. This condition permits unstable mismatches. Every actual pair belongs to ${ \mathcal { P } } \subseteq { \mathcal { P } } _ { \mathsf { S } } \cup { \mathcal { P } } _ { \mathsf { F } }$ ; the algorithm uses the class-wide bounds for tuning but never determines which subfamily covers a particular change.

A common finite-window gap. For an integer $h \geq 1$ and a $d \times d$ matrix A, define

$$
G _ { h } ( A ) : = \sigma _ { w } ^ { 2 } \sum _ { j = 1 } ^ { h - 1 } ( e _ { 1 } ^ { \top } A ^ { j } e _ { 1 } ) ^ { 2 } .
$$

It measures the excess variance supplied by the preceding $h - 1$ disturbances. For stable pairs, the omitted tail is bounded by $r _ { \mathsf { S } } ( h )$ below. When both subfamilies are nonempty, choose

$$
\begin{array} { r l r } {  { r _ { 5 } ( h ) : = \frac { \sigma _ { w } ^ { 2 } C _ { 5 } ^ { 2 } \rho _ { 5 } ^ { 2 h } } { 1 - \rho _ { 5 } ^ { 2 } } , } } \\ & { } & { h _ { 5 } : = \operatorname* { m i n } \lbrace h \geq 1 : r _ { 5 } ( h ) \leq \underline { { \Delta _ { 5 } } } / 8 \rbrace , } \\ & { } & { h _ { \mathsf { U } } : = \operatorname* { m a x } \lbrace d + 1 , h _ { 5 } \rbrace , \quad \quad \underline { { \Delta } } _ { \mathsf { U } } : = \operatorname* { m i n } \lbrace 7 \underline { { \Delta } } _ { 5 } / 8 , \underline { { \Delta } } _ { \mathsf { F } } \rbrace . } \end{array}\tag{23}
$$

Appendix K gives the convention when a subfamily is empty. Since $h _ { \mathsf { U } } \geq h _ { \mathsf { S } }$ bounds the stable tail and $h _ { \mathsf { U } } \geq d + 1$ includes every term in (22),

$$
G _ { h _ { \mathsf { U } } } ( A _ { c } ) \geq \left\{ \begin{array} { l l } { 7 \underline { { \Delta } } _ { \mathsf { S } } / 8 , } & { z _ { c } \in \mathcal { P } _ { \mathsf { S } } , } \\ { \underline { { \Delta } } _ { \mathsf { F } } , } & { z _ { c } \in \mathcal { P } _ { \mathsf { F } } , } \end{array} \geq \underline { { \Delta } } _ { \mathsf { U } } . \right.\tag{24}
$$

Stable pairs therefore need not satisfy a separate first-d visibility condition: their stationary gap supplies the required finite-window gap.

The detector and its tuning. Algorithm 3 retains PIECE-CD’s exploration, estimation, gated RLS updates, and restart rule. Its energy test records one output square every $h _ { \mathsf { U } }$ rounds and raises an alarm when the average of the latest m recorded squares reaches $\sigma _ { w } ^ { 2 } + \underline { { \Delta } } _ { \mathsf { U } } / 2$ Let $B _ { \mathsf { U } }$ be the common bound on disturbances, actual outputs, and auxiliary old-gain outputs over an $h _ { \mathsf { U } }$ -step comparison, defined in (141). The auxiliary bound is finite even for an unstable mismatch because each comparison lasts only $h _ { \mathsf { U } }$ steps. Set the number of recorded outputs and the detection-delay bound to

$$
m _ { \mathsf { U } } : = \left\lceil \frac { 5 1 2 B _ { \mathsf { U } } ^ { 4 } } { \underline { { \Delta } } _ { \mathsf { U } } ^ { 2 } } \log \frac { 4 T ^ { 2 } } { \delta _ { d } } \right\rceil , \qquad D _ { \mathsf { U } } : = ( m _ { \mathsf { U } } + 1 ) h _ { \mathsf { U } } .\tag{25}
$$

To handle clipping, replace uniform feasibility by closed own-regime feasibility (Assumption 24). Each ideal gain must respect the input bound on every regressor reached after at least $r _ { 0 }$ controls under its own plant, from any initial regressor of norm at most $B _ { \psi }$ , with bounded disturbances and arbitrary inputs in $[ - B _ { u } , B _ { u } ]$ . Equality is allowed. This is an alternative condition: it need not follow from baseline feasibility because it also allows initially unreachable regressors. The old gain need not remain feasible after a change. The additional test monitors the reference gain’s input proposal. After observing $y _ { t + 1 }$ and forming $\psi _ { t + 1 }$ , it raises an alarm if

$$
\vert \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t + 1 } \vert > B _ { u } + \zeta _ { H } , \qquad \zeta _ { H } : = E _ { \lambda } ( H , \delta _ { 0 } ) B _ { \psi } .\tag{26}
$$

With an accurate reference, own-regime feasibility prevents this alarm while the plant is unchanged. After a change, before this alarm fires, the old ideal input exceeds the allowed interval by at most $2 \zeta _ { H }$ , and the applied input difers from it by at most $7 \zeta _ { H }$ . These bounds control the efect of clipping in the finite-step comparison.

Choose H to satisfy the exploration and RLS conditions together with (142), which controls finite-window tracking errors. Smaller gaps require longer exploration and detection windows, increasing the required dwell time and regret constant. Either test triggers a restart, with simultaneous crossings counted as one alarm. The following theorem gives the detection and logarithmic-regret guarantees.

Theorem 5 (Detection and regret for the union class). Retain the compact-model, noise, and open-loop switching-stability conditions of Assumption 1, and impose the own-regime feasibility above. Under the precise union-class conditions in Assumption 26, suppose each actual adjacent pair satisfies at least one of (21) and (22), without revealing which one. Choose H as the smallest feasible integer satisfying (93), (51), and (142); choose m , $D _ { \mathsf { U } }$ by (25), and impose (6) with $D _ { \mathsf { U } }$ in place of D. Then Algorithm 3, with probability at least $1 - \delta$ , has no false alarms, and each change has exactly one alarm ${ \widehat { \tau } } ^ { \left( c \right) }$ satisfying

$$
\tau ^ { ( c ) } < { \widehat { \tau } } ^ { ( c ) } \leq \tau ^ { ( c ) } + D _ { \mathsf { U } } ,\tag{27}
$$

and

$$
\mathcal { R } _ { T } \leq C K _ { D } D \mathsf { u } + ( C + 1 ) K _ { X } H + ( C + 1 ) R _ { E } ( T , \delta _ { 0 } ) .\tag{28}
$$

For fixed model and union-class bounds, including every applicable positive gap bound for the nonempty cover components, this is $O ( ( C + 1 ) \log ( ( T + 1 ) / \delta ) )$ . The algorithm never determines which subfamily covers any particular change.

Proof. See Theorem 31 in Appendix L.

Detection argument for Theorem 5. The proof starts a separate $h _ { \mathsf { U } }$ -step auxiliary process before each stored output. If the boundary test has not already crossed, the actual and auxiliary output squares remain close over these finitely many steps. For a pair in ${ \mathcal { P } } _ { \mathsf { S } }$ , its stationary gap and stable-tail bound give conditional auxiliary energy at least $\sigma _ { w } ^ { 2 } + 7 \underline { { \Delta } } _ { \sf S } / 8 $ for a pair in $\mathcal { P } _ { \mathsf { F } }$ finite visibility gives at least $\sigma _ { w } ^ { 2 } + \underline { { \Delta } } _ { \sf F }$ . Equation (24) therefore supplies the same $\underline { { \Delta } } _ { \mathsf { U } }$ to the concentration argument in both cases.

Finite-order visibility of unstable ARX mismatches. Spectral instability alone does not imply visibility for arbitrary matrices. For this compact minimum-phase ARX class, (131) derives a uniform finite-order gap from $\rho ( A ( \theta ^ { - } , \theta ^ { + } ) ) \ge 1$ , so no separate visibility assumption is needed on that subclass.

Why input feasibility is needed. Some restriction on saturation is unavoidable for the regret benchmark in (3). For every clipped policy,

$$
\mathcal { R } _ { T } \geq b _ { \operatorname* { m i n } } ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } ( | \lambda ( \theta _ { t } ) ^ { \mathsf { T } } \psi _ { t } | - B _ { u } ) _ { + } ^ { 2 } .\tag{29}
$$

Appendix K also gives an exploitation bound with this saturation energy as an additive term.   
Persistent saturation can therefore force linear regret, even when the parameter is known.

## 6 EVENTUAL CLIPPING INACTIVITY

For stable detectable mismatches, a separate variant replaces uniform clipping feasibility by a steady-state slack condition. Define

$$
P _ { \infty } ( B _ { u } ) : = \sqrt { \left( \frac { C _ { y } } { 1 - \rho _ { y } } ( B _ { w } + B _ { b } B _ { u } ) \right) ^ { 2 } + ( q - 1 ) B _ { u } ^ { 2 } } ,
$$

where $C _ { y } , \rho _ { y }$ are the uniform open-loop switching bounds and $B _ { b }$ bounds the sum of absolute input coeficients. Let $\Lambda = \operatorname* { s u p } _ { \theta \in \Theta } \| \lambda ( \theta ) \|$ . The quantity $P _ { \infty }$ bounds the forced-response contribution to the regressor norm; the slack absorbs the decaying initial-state term.

Theorem 6 (Regret under eventual clipping inactivity). Retain the baseline model, noise, switching-stability, stable power-gap, and exploration conditions, but replace uniform feasibility by

$$
\mu _ { 0 } : = B _ { u } - \Lambda P _ { \infty } ( B _ { u } ) > 0 .\tag{30}
$$

Choose the smallest baseline-feasible H also satisfying $\varepsilon _ { u } \le \mu _ { 0 } / 2$ . After each full exploration, use the clipped reference controller for $\tau _ { \mathrm { n c } }$ rounds, with RLS and detection disabled, where $\tau _ { \mathrm { n c } }$ is given by (119) with $\mu _ { \mathrm { n c } } = \mu _ { 0 } - \varepsilon _ { u } . \ A t \ t _ { 0 } = r + H + \tau _ { \mathrm { n c } }$ , initialize RLS from the same exploration data as in Section 3, and record outputs at $t _ { 0 } + j h , j \ge 1$ . Use (6) with $H + \tau _ { \mathrm { n c } }$ in place of H. With probability at least $1 - \delta$ , there are no false alarms, every change is detected within D rounds, and

$$
\mathcal { R } _ { T } \leq C K _ { D } D + ( C + 1 ) K _ { X } H + ( C + 1 ) \{ K _ { D } \tau _ { \mathrm { n c } } + R _ { E } ( T , \delta _ { 0 } ) \} .\tag{31}
$$

For fixed problem constants and positive slack, $\tau _ { \mathrm { n c } } ~ = ~ O ( 1 )$ , so the regret remains $O ( ( C +$ $1 ) \log ( ( T + 1 ) / \delta ) )$

Proposition 23 and the burn-in argument in Appendix K.2 establish feasibility after the burn-in, including the continuations needed for detection. Thus only the additional bounded cost per restart enters the regret decomposition.

## 7 LIMITATIONS

The logarithmic horizon dependence holds for fixed change count, confidence level, and problem constants. The bounds retain the factor $C + 1$ when changes become more frequent. A uniform energy gap and suficient dwell time are essential: hidden or arbitrarily weak changes are outside the guarantees. The algorithm also requires supplied bounds on noise, detectability, excitation, and the relevant stability and finite-window sensitivities. Smaller gaps increase the delay and required dwell time. Estimating these bounds, removing knowledge of the horizon, and accommodating heavy-tailed disturbances remain open extensions. The feasibility conditions restrict the nonlinear efects of saturation; they do not cover arbitrary clipped dynamics.

## 8 CONCLUSION

We establish high-probability regret $O ( ( C + 1 ) \log ( ( T + 1 ) / \delta ) )$ for minimum-variance control of unknown piecewise-stationary ARX systems under fixed detectability gaps and the stated feasibility and dwell-time conditions. PIECE-CD uses output observations during exploitation to detect changes, explores at initialization and after each alarm, and uses gated RLS to control regret between changes. A common detection rule covers both stable controller mismatches with a stationary power gap and potentially unstable mismatches whose disturbance response produces detectable excess energy within a fixed number of steps.

## AI Use Statement

Generative AI tools assisted with editing and reorganizing the exposition, checking consistency between main-text statements and the appendices, and LaTeX formatting and compilation checks. The authors take responsibility for the final text, mathematical claims, proofs, and references.

## References

Abbasi-Yadkori, Y., Pál, D., and Szepesvári, C. (2011). Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, volume 24.

Audibert, J.-Y. and Bubeck, S. (2009). Minimax policies for adversarial and stochastic bandits. In Proceedings of the 22nd Annual Conference on Learning Theory.

Azuma, K. (1967). Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, 19(3):357–367.

Becker, A., Kumar, P. R., and Wei, C.-Z. (1985). Adaptive control with the stochastic approximation algorithm: Geometry and convergence. IEEE Transactions on Automatic Control, 30(4):330–338.

Besbes, O., Gur, Y., and Zeevi, A. (2014). Stochastic multi-armed-bandit problem with nonstationary rewards. In Advances in Neural Information Processing Systems, volume 27, pages 199–207.

Besson, L., Kaufmann, E., Maillard, O.-A., and Seznec, J. (2022). Eficient change-point detection for tackling piecewise-stationary bandits. Journal of Machine Learning Research, 23(77):1–40.

Cao, Y., Wen, Z., Kveton, B., and Xie, Y. (2019). Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 418–427. PMLR.

Garivier, A. and Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. In Proceedings of the 22nd International Conference on Algorithmic Learning Theory, pages 174–188.

Goodwin, G. C., Ramadge, P. J., and Caines, P. E. (1981). Discrete time stochastic adaptive control. SIAM Journal on Control and Optimization, 19(6):829–853.

Hoefding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30.

Horn, R. A. and Johnson, C. R. (2013). Matrix Analysis. Cambridge University Press, Cambridge, 2 edition.

Howard, S. R., Ramdas, A., McAulife, J., and Sekhon, J. (2020). Time-uniform Chernof bounds via nonnegative supermartingales. Probability Surveys, 17:257–317.

Huang, Y.-H., Gerogiannis, A., Bose, S., and Veeravalli, V. V. (2026). Detection augmented bandit procedures for piecewise stationary MABs: A modular approach. IEEE Transactions on Information Theory.

Kumar, P. R. and Praly, L. (1987). Self-tuning trackers. SIAM Journal on Control and Optimization, 25(4):1053–1071.

Lai, T. L. (1986). Asymptotically eficient adaptive control in stochastic regression models. Advances in Applied Mathematics, 7(1):23–45.

Lai, T. L. and Robbins, H. (1985). Asymptotically eficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22.

Lai, T. L. and Wei, C.-Z. (1987). Asymptotically eficient self-tuning regulators. SIAM Journal on Control and Optimization, 25(2):466–481.

Liu, F., Lee, J., and Shrof, N. B. (2018). A change-detection based framework for piecewisestationary multi-armed bandit problem. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, pages 3651–3658.

Mukherjee, S. and Maillard, O.-A. (2019). Distribution-dependent and time-uniform bounds for piecewise i.i.d. bandits. arXiv preprint arXiv:1905.13159.

Simchowitz, M. and Foster, D. (2020). Naive exploration is optimal for online LQR. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 8937–8948. PMLR.

Singh, R., Mete, A., Kar, A., and Kumar, P. R. (2024). Finite time logarithmic regret bounds for self-tuning regulation. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 45571–45636. PMLR.

Tartakovsky, A., Nikiforov, I., and Basseville, M. (2014). Sequential Analysis: Hypothesis Testing and Changepoint Detection. CRC Press.

Tropp, J. A. (2011). Freedman’s inequality for matrix martingales. Electronic Communications in Probability, 16:262–270.

Veeravalli, V. V. and Banerjee, T. (2014). Quickest change detection. Academic Press Library in Signal Processing, 3:209–255.

Ville, J. (1939). Étude critique de la notion de collectif. Gauthier-Villars, Paris.

Xie, L., Zou, S., Xie, Y., and Veeravalli, V. V. (2021). Sequential (quickest) change detection: Classical results and new directions. IEEE Journal on Selected Areas in Information Theory, 2(2):494–514.

Zhou, H., Wang, L., Varshney, L. R., and Lim, E.-P. (2020). A near-optimal change-detection based algorithm for piecewise-stationary combinatorial semi-bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 6933–6940.

## APPENDIX CONTENTS

A MODEL AND ASSUMPTIONS. . 17   
B THE PIECE-CD ALGORITHM 19   
C CHANGE-DETECTION GUARANTEES.. . 22   
D REGRET ANALYSIS. .26   
E DETERMINISTIC BOUNDS AND RESTART CONVENTIONS.. . 29   
F EXPLORATION, EXCITATION, AND THE RESTARTED ESTIMATE. . 30   
G THE RLS POTENTIAL ARGUMENT. .34   
H DETAILED DETECTOR PROOFS. .37   
I GLOBAL EVENT AND REGRET DECOMPOSITION. .39   
J TUNING FEASIBILITY AND ORDERS. .41   
K RELAXATIONS OF CLIPPING AND MISMATCH STABILITY .. . 42   
L A UNIFIED DETECTOR FOR THE UNION CLASS. .52

## ROADMAP OF THE APPENDICES

A. Specifies the piecewise-stationary ARX model, regret criterion, and assumptions on stability, noise, clipping feasibility, and detectability of adjacent changes.

B. Gives the exploration estimator, gated RLS controller, output-energy detector, tuning and dwell-time requirements, and complete PIECE-CD algorithm.

C. Proves action tracking, sampled-square concentration, false-alarm control, detection delay, and correspondence between alarms and true changes.

D. Derives exploitation regret on unchanged segments and combines it with exploration and detection-delay costs to obtain the total regret guarantee.

E. Establishes deterministic output and regressor bounds, clipping projection inequalities, auxiliary-process bounds, and conditioning conventions at random restarts.

F. Proves conditional excitation, exploration Gram-matrix growth, and projected ridge estimation bounds for a complete unchanged exploration block.

G. Develops the RLS potential identity, time-uniform noise control, and elliptical-potential bound underlying the segmentwise exploitation guarantee.

H. Supplies detailed auxiliary-process power, sampling, coupling, and calendar arguments supporting the detection results in Appendix C.

I. Constructs a common success event and partitions the horizon to complete the proof of the global regret bound.

J. Verifies feasibility of the tuning requirements and derives the logarithmic orders of exploration length, detection delay, and regret.

K. Develops the clipping-feasibility relaxations, eventual nonclipping, boundary alarms, finiteorder visibility and union-class detection, and regret bounds with saturation.

L. Presents the unified union-class algorithm and common tuning, then proves detection and regret guarantees without identifying the realized mismatch class.

## A MODEL AND ASSUMPTIONS

Fix an integer horizon $T \geq 1$ , confidence level $\delta \in ( 0 , 1 )$ , and integer model orders $p , q \geq 1$ . The deterministic but unknown segment boundaries are

$$
0 = \tau ^ { ( 0 ) } < \tau ^ { ( 1 ) } < \cdot \cdot \cdot < \tau ^ { ( C ) } < \tau ^ { ( C + 1 ) } = T .
$$

On segment c, the parameter

$$
\theta ^ { ( c ) } : = ( a _ { 1 } ^ { ( c ) } , \ldots , a _ { p } ^ { ( c ) } , b _ { 1 } ^ { ( c ) } , \ldots , b _ { q } ^ { ( c ) } ) ^ { \prime }
$$

is active at control times $\tau ^ { ( c ) } \leq t < \tau ^ { ( c + 1 ) }$ , and the system obeys

$$
y _ { t + 1 } = \sum _ { i = 1 } ^ { p } a _ { i } ^ { ( c ) } y _ { t - i + 1 } + \sum _ { j = 1 } ^ { q } b _ { j } ^ { ( c ) } u _ { t - j + 1 } + w _ { t + 1 } , \qquad \tau ^ { ( c ) } \leq t < \tau ^ { ( c + 1 ) } .\tag{32}
$$

Write $\theta _ { t } = \theta ^ { ( c ) }$ on this segment and $Y _ { t } = ( y _ { t } , \dots , y _ { t - p + 1 } ) ^ { \prime }$ . The initial history is fixed, with $\| Y _ { 0 } \| < \infty$ and $| u _ { t } | \le B _ { u }$ for $t = - q + 1 , \ldots , - 1$ . The control and identification regressors are

$$
\psi _ { t } : = ( y _ { t } , \ldots , y _ { t - p + 1 } , u _ { t - 1 } , \ldots , u _ { t - q + 1 } ) ^ { \prime } ,\tag{33}
$$

$$
\begin{array} { r } { \phi _ { t } : = ( y _ { t } , \ldots , y _ { t - p + 1 } , u _ { t } , u _ { t - 1 } , \ldots , u _ { t - q + 1 } ) ^ { \prime } , \qquad d : = p + q - 1 . } \end{array}\tag{34}
$$

For $\theta = ( a _ { 1 } , \ldots , a _ { p } , b _ { 1 } , \ldots , b _ { q } )$ with $b _ { 1 } \neq 0$ , define

$$
\lambda ( \theta ) : = - \frac { 1 } { b _ { 1 } } ( a _ { 1 } , \ldots , a _ { p } , b _ { 2 } , \ldots , b _ { q } ) ^ { \prime } .\tag{35}
$$

The corresponding gain representation is

$$
y _ { t + 1 } = b _ { 1 } ^ { ( c ) } \{ u _ { t } - \lambda ( \theta ^ { ( c ) } ) ^ { \prime } \psi _ { t } \} + w _ { t + 1 } .\tag{36}
$$

The ideal input $u _ { t } = \lambda ( \theta ^ { ( c ) } ) ^ { \prime } \psi _ { t }$ leaves only the disturbance in the output. Our sample-path learning regret is

$$
\mathcal { R } _ { T } : = \sum _ { t = 0 } ^ { T - 1 } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } .
$$

Let $\mathcal { F } _ { t }$ contain the initial history, the controller randomization revealed through time $t ,$ and all inputs and outputs available when $u _ { t }$ is selected. In particular, $u _ { t }$ is $\mathcal { F } _ { t } .$ -measurable.

Assumption $\mathbf { 7 }$ (Known model class). The controller knows a compact set $\Theta \subset \mathbb { R } ^ { p + q }$ containing every $\theta ^ { ( c ) }$ . For each $\theta \in \Theta$ , the autoregressive polynomial is Schur stable, the input polynomial is minimum phase, and

$$
0 < b _ { \mathrm { m i n } } \leq b _ { 1 } ( \theta ) \leq \bar { b } < \infty .\tag{37}
$$

The gain map in (35) is therefore Lipschitz on $\Theta ; \hbar x$ a strictly positive known Lipschitz bound $L _ { \lambda } > 0$ satisfying

$$
\begin{array} { r } { \| \lambda ( \theta ) - \lambda ( \vartheta ) \| \le L _ { \lambda } \| \theta - \vartheta \| , \qquad \theta , \vartheta \in \Theta . } \end{array}\tag{38}
$$

Write $\Lambda : = \operatorname* { s u p } _ { \theta \in \Theta } \| \lambda ( \theta ) \|$ and $\begin{array} { r } { B _ { b } : = \operatorname* { s u p } _ { \theta \in \Theta } \sum _ { j = 1 } ^ { q } | b _ { j } ( \theta ) | } \end{array}$

Let $A _ { y } ( \theta )$ be the companion matrix in (88). A parameter sequence is admissible if it takes values in Θ and is piecewise constant at the specified segment boundaries. Individual Schur stability does not ensure stability under switching, so we impose the following uniform product bound.

Assumption 8 (Uniform switching stability). There are $1 \le C _ { y } < \infty$ and $\rho _ { y } \in ( 0 , 1 )$ such that, $f o r$ every admissible parameter sequence and every $k \geq 0$

$$
\| A _ { y } ( \theta _ { t + k - 1 } ) \cdot \cdot \cdot A _ { y } ( \theta _ { t } ) \| \leq C _ { y } \rho _ { y } ^ { k } .\tag{39}
$$

Assumption 9 (Noise). Each $w _ { t }$ is independent of $\mathcal { F } _ { t - 1 }$ , and the disturbances are identically distributed. They satisfy

$$
\mathbb { E } w _ { t } = 0 , \qquad \mathbb { E } w _ { t } ^ { 2 } = \sigma _ { w } ^ { 2 } > 0 , \qquad | w _ { t } | \le B _ { w } \quad a l m o s t s u r e l y .
$$

The detector knows $\sigma _ { w } ^ { 2 }$ and $B _ { w }$

The algorithm always clips its input to $[ - B _ { u } , B _ { u } ]$ . Consequently, Assumptions $_ { 7 - 9 }$ imply the deterministic bounds

$$
\begin{array} { l } { \displaystyle B _ { Y } : = C _ { y } \| Y _ { 0 } \| + \frac { C _ { y } } { 1 - \rho _ { y } } ( B _ { w } + B _ { b } B _ { u } ) , \qquad | y _ { t } | \le B _ { Y } , } \\ { \displaystyle B _ { \psi } : = \{ p B _ { Y } ^ { 2 } + ( q - 1 ) B _ { u } ^ { 2 } \} ^ { 1 / 2 } , \qquad \| \psi _ { t } \| \le B _ { \psi } . } \end{array}\tag{40}
$$

A proof is given in Appendix E.

For the baseline analysis, the ideal action must also be feasible at every reachable regressor. Here, reachability is from the fixed initial history under admissible parameter sequences, disturbances bounded by $B _ { w }$ , and inputs in $[ - B _ { u } , B _ { u } ]$

Assumption 10 (Uniform clipping feasibility). The ideal action belongs to the closed clipping interval:

$| \lambda ( \theta ) ^ { \prime } \psi | \le B _ { u }$ for every θ ∈ Θ and every reachable ψ with $\lVert \psi \rVert \leq B _ { \psi }$

The stronger condition $\Lambda B _ { \psi } \leq B _ { u }$ is suficient. Equality is allowed; projection nonexpansiveness requires no positive margin.

For a gain $\ell \in \mathbb { R } ^ { d }$ and parameter θ, let $A _ { \mathrm { c l } } ( \ell , \theta )$ be the matrix satisfying

$$
\psi _ { t + 1 } = A _ { \mathrm { c l } } ( \ell , \theta ) \psi _ { t } + e _ { 1 } w _ { t + 1 }\tag{41}
$$

when the unclipped input $u _ { t } = \ell ^ { \prime } \psi _ { t }$ is applied. Here $e _ { 1 }$ is the first coordinate vector. If the input is $u _ { t } = \ell ^ { \prime } \psi _ { t } + e _ { t }$ , the regressor dynamics become

$$
\psi _ { t + 1 } = A _ { \mathrm { c l } } ( \ell , \theta ) \psi _ { t } + g _ { u } ( \theta ) e _ { t } + e _ { 1 } w _ { t + 1 } ,
$$

where the input-channel vector and its uniform bound are defined by

$$
g _ { u } ( \theta ) : = \left\{ \begin{array} { l l } { b _ { 1 } ( \theta ) e _ { 1 } , } & { q = 1 , } \\ { ( b _ { 1 } ( \theta ) , 0 _ { p - 1 } ^ { \prime } , 1 , 0 _ { q - 2 } ^ { \prime } ) ^ { \prime } , } & { q \geq 2 , } \end{array} \right. \quad \quad G _ { u } : = \operatorname* { s u p } _ { \theta \in \Theta } \| g _ { u } ( \theta ) \| \leq \sqrt { \bar { b } ^ { 2 } + 1 } .\tag{42}
$$

Here $0 _ { k }$ denotes the zero vector in $\mathbb { R } ^ { k }$ ; the final block is absent when $q = 2$ . When $A _ { \mathrm { c l } } ( \ell , \theta )$ is Schur stable, define its stationary output power

$$
J _ { y } ( \ell ; \theta ) : = \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { \infty } \{ e _ { 1 } ^ { \prime } A _ { \mathrm { c l } } ( \ell , \theta ) ^ { j } e _ { 1 } \} ^ { 2 } .\tag{43}
$$

To justify this terminology, freeze $( \ell , \theta )$ and put $A = A _ { \mathrm { c l } } ( \ell , \theta )$ . The zero-mean stationary solution of $\psi _ { t + 1 } = A \psi _ { t } + e _ { 1 } w _ { t + 1 }$ is

$$
\psi _ { t } ^ { \mathrm { s t a t } } = \sum _ { j = 0 } ^ { \infty } A ^ { j } e _ { 1 } w _ { t - j } .
$$

The series converges in mean square because A is Schur. Its first coordinate is the stationary output. Independence, zero mean, and $\mathbb { E } w _ { t } ^ { 2 } = \sigma _ { w } ^ { 2 }$ eliminate all cross terms, so

$$
\begin{array} { r l } & { \mathbb { E } [ ( y _ { t } ^ { \mathrm { s t a t } } ) ^ { 2 } ] = \displaystyle \sum _ { j , k \geq 0 } ( e _ { 1 } ^ { \prime } A ^ { j } e _ { 1 } ) ( e _ { 1 } ^ { \prime } A ^ { k } e _ { 1 } ) \mathbb { E } [ w _ { t - j } w _ { t - k } ] } \\ & { \quad \quad = \displaystyle \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = 0 } ^ { \infty } ( e _ { 1 } ^ { \prime } A ^ { j } e _ { 1 } ) ^ { 2 } = J _ { y } ( \ell ; \theta ) . } \end{array}
$$

Thus (43) is the stationary second moment (and, because the mean is zero, the variance) of the unclipped frozen-gain output process. It is not asserted directly for the implemented clipped trajectory; Lemmas 13 and 16 provide that comparison. In particular, $J _ { y } ( \lambda ( \theta ) ; \theta ) = \sigma _ { w } ^ { 2 }$

Assumption 11 (Stable and detectable adjacent changes). There are $C _ { \Delta } < \infty , \rho _ { \Delta } \in ( 0 , 1 )$ , and $\Delta > 0$ such that, for every actual adjacent pair $( \theta ^ { ( c - 1 ) } , \theta ^ { ( c ) } )$ ,

$$
\begin{array} { r } { \left\| A _ { \mathrm { c l } } ( \lambda ( \theta ^ { ( c - 1 ) } ) , \theta ^ { ( c ) } ) ^ { k } \right\| \leq C _ { \Delta } \rho _ { \Delta } ^ { k } , \qquad k \geq 0 , } \end{array}\tag{44}
$$

$$
J _ { y } ( \lambda ( \theta ^ { ( c - 1 ) } ) ; \theta ^ { ( c ) } ) \ge \sigma _ { w } ^ { 2 } + \Delta .\tag{45}
$$

The power-gap condition excludes changes that leave the old controller’s output power unchanged. Appendix K gives two extensions: a burn-in variant under eventual nonclipping, and a boundary-alarm variant covering both stable power gaps and finite-order visibility.

Tuning requires valid bounds for $C _ { \Delta } , \rho _ { \Delta }$ , and $\Delta ,$ , and the model and exploration bounds defining $B _ { \psi } , B _ { o } ,$ and $\kappa _ { G }$ in (40), (50), and (92). Conservative bounds sufice; the analysis does not provide a data-driven calibration rule.

## B THE PIECE-CD ALGORITHM

Each PIECE-CD cycle consists of H exploration rounds followed by gated RLS control and output-energy monitoring. An alarm starts a new cycle.

## B.1 Exploration and Estimation

Let $s : = p + q , r _ { 0 } : = \operatorname* { m a x } \{ p , q \}$ , and $n _ { H } : = H - r _ { 0 }$ . Fix a ridge parameter $\nu > 0$ . Exploration uses fresh i.i.d. mean-zero inputs, bounded by $B _ { e } \leq B _ { u }$ and of variance $\sigma _ { e } ^ { 2 } > 0$ , independent of the disturbance sequence. Conditional on $\mathcal { F } _ { r } , u _ { r }$ is fixed and the subsequent exploration inputs remain independent with this distribution. For a full unchanged exploration block satisfying (93), Proposition 21 gives

$$
\lambda _ { \operatorname* { m i n } } \left( \sum _ { { t = r + r _ { 0 } } } ^ { r + H - 1 } \phi _ { t } \phi _ { t } ^ { \prime } \right) \ge \kappa _ { G } n _ { H }
$$

with conditional probability at least $1 - \delta _ { 0 }$ given $\mathcal { F } _ { r }$ . The constant $\kappa _ { G } > 0$ in (92) is uniform over Θ and the bounded restart states.

Put $B _ { \phi } : = ( B _ { \psi } ^ { 2 } + B _ { u } ^ { 2 } ) ^ { 1 / 2 }$ and $S _ { \Theta } : = \operatorname* { s u p } _ { \theta \in \Theta } \| \theta \|$ . For a confidence level $\delta _ { 0 }$ , define the deterministic estimation bound

$$
\mathcal { L } _ { H } ( \delta _ { 0 } ) : = 2 \log ( 1 / \delta _ { 0 } ) + s \log \left( 1 + \frac { H B _ { \phi } ^ { 2 } } { s \nu } \right) ,
$$

$$
E _ { \theta } ( H , \delta _ { 0 } ) : = \frac { 2 \{ B _ { w } \sqrt { \mathcal { L } _ { H } ( \delta _ { 0 } ) } + \sqrt { \nu } S _ { \Theta } \} } { \sqrt { \nu + \kappa _ { G } n _ { H } } } ,\tag{46}
$$

$$
E _ { \lambda } ( H , \delta _ { 0 } ) : = L _ { \lambda } E _ { \theta } ( H , \delta _ { 0 } ) ,
$$

$$
g _ { H } : = 4 E _ { \lambda } ( H , \delta _ { 0 } ) .\tag{47}
$$

For the estimator and reference gain in (55)–(56), Proposition 22 establishes $\left\| { \widehat { \theta } } _ { r } - \theta \right\| \leq E _ { \theta }$ and $\left\| \overline { { \lambda } } _ { r } - \lambda ( \theta ) \right\| \leq E _ { \lambda }$ on the joint Gram and estimation event.

For the horizon-wide theorem we use

$$
\delta _ { d } : = \delta / 2 , \qquad \delta _ { 0 } : = \frac { \delta } { 3 2 ( T + 1 ) } .\tag{48}
$$

This allocation remains valid without knowing C, because no more than $T + 1$ restart locations are possible.

## B.2 Tuning and Dwell Time

On the successful reference event, Lemma 13 bounds the action and unchanged-regime output errors by

$$
\varepsilon _ { u } : = \frac { 5 } { 4 } g _ { H } B _ { \psi } , \qquad \varepsilon _ { y } : = \bar { b } \varepsilon _ { u } .\tag{49}
$$

For a post-change fixed-old-gain auxiliary trajectory, set

$$
\begin{array} { r l } & { ~ B _ { \mathrm { a u x } } : = C _ { \Delta } B _ { \psi } + \displaystyle \frac { C _ { \Delta } B _ { w } } { 1 - \rho _ { \Delta } } , \qquad B _ { o } : = \operatorname* { m a x } \{ B _ { Y } , B _ { \mathrm { a u x } } , B _ { w } \} , } \\ & { ~ R _ { \Delta } ( H ) : = \displaystyle \frac { C _ { \Delta } G _ { u } \varepsilon _ { u } } { 1 - \rho _ { \Delta } } . } \end{array}\tag{50}
$$

Choose $H > r _ { 0 }$ as the smallest integer for which the sample-size condition (93) and the following inequalities hold:

$$
\frac { E _ { \theta } ( H , \delta _ { 0 } ) } { b _ { \mathrm { { m i n } } } } \leq \frac { 1 } { 1 6 } ,
$$

$$
\frac { B _ { \psi } ^ { 2 } } { \nu + \kappa _ { G } n _ { H } } \leq \frac { 1 } { 1 6 } ,\tag{51}
$$

$$
\varepsilon _ { y } ^ { 2 } \le { \frac { \Delta } { 8 } } , \qquad \varepsilon _ { y } \le B _ { o } ,
$$

$$
2 B _ { o } R _ { \Delta } ( H ) \leq \frac { \Delta } { 8 } .\tag{52}
$$

Condition (51) controls the RLS bias and leverage; (52) controls the null bias and coupling error.

Choose the integer sampling interval $h ,$ window size $m ,$ and delay bound D by

$$
h : = \operatorname* { m i n } \left\{ k \geq 1 : \frac { \sigma _ { w } ^ { 2 } C _ { \Delta } ^ { 2 } \rho _ { \Delta } ^ { 2 k } } { 1 - \rho _ { \Delta } ^ { 2 } } \leq \frac { \Delta } { 8 } \right\} ,\tag{53}
$$

$$
m : = \left\lceil \frac { 5 1 2 B _ { o } ^ { 4 } } { \Delta ^ { 2 } } \log \frac { 4 T ^ { 2 } } { \delta _ { d } } \right\rceil ,
$$

$$
D : = ( m + 1 ) h .\tag{54}
$$

The spacing truncates the stationary-power tail at $\Delta / 8$ . The window size controls both null and post-change deviations uniformly over potential restart locations and test endpoints.

Assumption 12 (Minimum dwell time). The first change satisfies $\tau ^ { ( 1 ) } \geq H$ when $C \geq 1$ Every pair of successive changes satisfies

$$
\tau ^ { ( c + 1 ) } - \tau ^ { ( c ) } \geq D + H , \qquad c = 1 , \dots , C - 1 ,
$$

and the terminal segment satisfies $T - \tau ^ { \left( C \right) } \geq D$ when $C \geq 1$ . When each change is detected within D rounds, these conditions leave a full unchanged exploration block before the next change. The final exploration block may be truncated at T.

## B.3 Algorithm

For a vector $x ,$ let $\Pi _ { \Theta } ( x )$ be a measurable Euclidean projection onto Θ. If Θ is not convex, any measurable nearest-point selection can be used; the only property required below is $\| \Pi _ { \Theta } ( x ) - \theta \| \leq 2 \| x - \theta \|$ for $\theta \in \Theta$ . Indeed, since θ itself is an admissible point in the nearestpoint minimization,

$$
\lVert \Pi _ { \Theta } ( x ) - x \rVert \leq \lVert \theta - x \rVert .
$$

The triangle inequality therefore gives $\Vert \Pi _ { \Theta } ( x ) - \theta \Vert \leq \Vert \Pi _ { \Theta } ( x ) - x \Vert + \Vert x - \theta \Vert \leq 2 \Vert x - \theta \Vert$ . This factor transfers the unprojected ridge error to the feasible estimate in Proposition 22. If Θ is closed and convex, projection is nonexpansive and the factor two can be replaced by one.

## Algorithm 2: The PIECE-CD algorithm

Input: the known orders and model class; $T , \delta ;$ all tuning quantities defined above; and the exploration distribution.

Initialize the restart time $r \gets 0$ and clear the detector bufer.

Loop. While $r < T .$ , execute the following cycle.

(a) Explore. For $t = r , \ldots , \operatorname* { m i n } \{ r + H - 1 , T - 1 \}$ , draw the independent exploratory input, apply it, and observe $y _ { t + 1 }$ . If the horizon ends, stop.

(b) Estimate. Using the post-boundary rows $\mathcal { T } _ { r } : = \{ r + r _ { 0 } , \ldots , r + H - 1 \}$ , form

$$
V _ { r } ^ { \theta } : = \nu I _ { s } + \sum _ { t \in \mathcal { Z } _ { r } } \phi _ { t } \phi _ { t } ^ { \prime } , \qquad \widetilde { \theta } _ { r } : = ( V _ { r } ^ { \theta } ) ^ { - 1 } \sum _ { t \in \mathcal { Z } _ { r } } \phi _ { t } y _ { t + 1 } ,\tag{55}
$$

$$
\widehat { \theta } _ { r } : = \Pi _ { \Theta } ( \widetilde { \theta } _ { r } ) ,
$$

$$
{ \widehat { b } } _ { r } : = b _ { 1 } ( { \widehat { \theta } } _ { r } ) ,
$$

$$
{ \overline { { \lambda } } } _ { r } : = \lambda ( { \widehat { \theta } } _ { r } ) .\tag{56}
$$

Initialize

$$
V _ { r + H } : = \nu I _ { d } + \sum _ { t \in \mathcal { Z } _ { r } } \psi _ { t } \psi _ { t } ^ { \prime } , \qquad \widehat { \lambda } _ { r + H } : = \overline { { \lambda } } _ { r } .\tag{57}
$$

Clear the detector bufer and declare the output-time sampling grid

$$
S _ { r } : = \{ r + H + h + j h : j = 0 , 1 , \ldots \} .
$$

(c) Exploit and monitor. For $t = r + H , r + H + 1 , \ldots , T - 1$ , set

$$
z _ { t } : = \left\{ \begin{array} { l l } { \widehat { \lambda } _ { t } ^ { \prime } \psi _ { t } , } & { \left| ( \widehat { \lambda } _ { t } - \overline { { \lambda } } _ { r } ) ^ { \prime } \psi _ { t } \right| \leq g _ { H } \left. \psi _ { t } \right. , } \\ { \overline { { \lambda } } _ { r } ^ { \prime } \psi _ { t } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad u _ { t } : = \mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( z _ { t } ) .\tag{58}
$$

Before observing $y _ { t + 1 }$ , declare it detector-eligible exactly when $t + 1 \in S _ { r }$ . Observe $y _ { t + 1 }$ and make the standard RLS update with pseudo-response

$$
x _ { t + 1 } ^ { \lambda } : = u _ { t } - \frac { y _ { t + 1 } } { \widehat { b } _ { r } } ,
$$

$$
V _ { t + 1 } : = V _ { t } + \psi _ { t } \psi _ { t } ^ { \prime } ,\tag{59}
$$

$$
\widehat { \lambda } _ { t + 1 } : = \widehat { \lambda } _ { t } + V _ { t + 1 } ^ { - 1 } \psi _ { t } ( x _ { t + 1 } ^ { \lambda } - \widehat { \lambda } _ { t } ^ { \prime } \psi _ { t } ) .\tag{60}
$$

If $y _ { t + 1 }$ was declared eligible, append $y _ { t + 1 } ^ { 2 }$ to the bufer. Retain only the most recent m stored values. When the bufer is full, compute

$$
\widehat J ( t + 1 ) : = \frac { 1 } { m } \sum _ { z \in \mathrm { b u f f e r } } z .
$$

If $\widehat { J } ( t + 1 ) \geq \sigma _ { w } ^ { 2 } + \Delta / 2$ , raise an alarm, set $r \gets t + 1$ , clear the bufer, and begin a new cycle. If no alarm is raised and $t = T - 1 , \mathrm { s t o p }$

## C CHANGE-DETECTION GUARANTEES

Throughout this appendix, the assumptions and tuning of Appendices A and B are in force. Fix a restart r with a full uncontaminated exploration block: the parameter remains equal to $\theta ^ { - }$ <sup>−</sup> throughout $[ r , r + H )$ . Write $\lambda ^ { - } : = \lambda ( \theta ^ { - } )$ and define the reference event

$$
\mathcal { G } _ { r } ^ { \mathrm { e s t } } : = \left\{ \left. \overline { { \lambda } } _ { r } - \lambda ^ { - } \right. \leq g _ { H } / 4 \right\}
$$

whose failure probability is controlled by Proposition 22.

Lemma 13 (Gate and action tracking). On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ , at every exploitation time before the next alarm, including times after an undetected change,

$$
\left| u _ { t } - ( \lambda ^ { - } ) ^ { \prime } \psi _ { t } \right| \leq \frac { 5 } { 4 } g _ { H } \left\| \psi _ { t } \right\| \leq \varepsilon _ { u } .\tag{61}
$$

If no true change has occurred and $a _ { t } : = ( \widehat { \lambda } _ { t } - \lambda ^ { - } ) ^ { \prime } \psi _ { t }$ , then the applied action error $\begin{array} { r l } { e _ { t } } & { { } : = } \end{array}$ $u _ { t } - ( \lambda ^ { - } ) ^ { \prime } \psi _ { t }$ also satisfies

$$
e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 } .\tag{62}
$$

Proof. If the recursive branch of (58) is selected, then

$$
\left| z _ { t } - ( \lambda ^ { - } ) ^ { \prime } \psi _ { t } \right| \leq g _ { H } \| \psi _ { t } \| + \left( g _ { H } / 4 \right) \| \psi _ { t } \| .
$$

If the reference branch is selected, the same left-hand side is at most $\left( g _ { H } / 4 \right) \| \psi _ { t } \|$ . Since the ideal action $( \lambda ^ { - } ) ^ { \prime } \psi _ { t }$ lies in $[ - B _ { u } , B _ { u } ]$ by Assumption 10, the metric projection onto this interval is nonexpansive and (61) follows after clipping.

For (62), equality holds in the recursive branch before clipping, and nonexpansiveness can only reduce the error. In the reference branch,

$$
\begin{array} { r l } & { | a _ { t } | \geq \left| ( \widehat { \lambda } _ { t } - \overline { { \lambda } } _ { r } ) ^ { \prime } \psi _ { t } \right| - \left| ( \overline { { \lambda } } _ { r } - \lambda ^ { - } ) ^ { \prime } \psi _ { t } \right| } \\ & { ~ > \frac { 3 } { 4 } g _ { H } \left\| \psi _ { t } \right\| , } \end{array}
$$

whereas the reference action error is at most $\left( g _ { H } / 4 \right) \left\| \psi _ { t } \right\|$ . Clipping again cannot increase it.

## C.1 Sampled-Square Concentration

The sampling interval h allows concentration around conditional moments given the history h steps earlier.

Lemma 14 (Lagged sampled-square concentration). Let $h \leq s _ { 1 } < \cdots < s _ { m }$ be finite lag-h predictable output times, meaning $\{ s _ { k } = t \} \in \mathcal { F } _ { t - h }$ for every t, and suppose $s _ { k + 1 } - s _ { k } \geq h$ . If $X _ { s _ { k } }$ is $\mathcal { F } _ { s _ { k } }$ -measurable and $| X _ { s _ { k } } | \le B _ { o }$ , then, for every $a > 0$ ，

$$
\mathbb { P } \left( \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \{ X _ { s _ { k } } ^ { 2 } - \mathbb { E } [ X _ { s _ { k } } ^ { 2 } \mid \mathcal { F } _ { s _ { k } - h } ] \} \le - a \right) \le \exp \left( { - \frac { m a ^ { 2 } } { 2 B _ { o } ^ { 4 } } } \right) .\tag{63}
$$

The analogous upper-tail bound also holds.

Proof. Set $Z _ { k } = X _ { s _ { k } } ^ { 2 } - \mathbb { E } [ X _ { s _ { k } } ^ { 2 } \mid { \mathcal { F } } _ { s _ { k } - h } ]$ . With $\mathcal { H } _ { k - 1 } = \mathcal { F } _ { s _ { k } - h }$ and, for $k < m , \mathcal { H } _ { k } = \mathcal { F } _ { s _ { k + 1 } - h } .$ , the spacing condition implies that $Z _ { k }$ is $\mathcal { H } _ { k }$ -measurable and has conditional mean zero. Moreover, $\mathcal { H } _ { m } : = \mathcal { F } _ { s _ { m } }$ contains $Z _ { m }$ . Lag predictability makes the random-indexed sigma-fields well defined as stopped sigma-fields, and $\mathcal { F } _ { s _ { k } } \subseteq \mathcal { F } _ { s _ { k + 1 } - h }$ . Moreover, $| Z _ { k } | \le B _ { o } ^ { 2 }$ . The Azuma–Hoefding inequality (Azuma, 1967; Hoefding, 1963) gives (63); the upper tail follows by applying it to $Z _ { k }$ □

## C.2 False Alarms

For each potential restart $r ,$ define an alarm-suppressed continuation by retaining the control, RLS, grid, and bufer recursions while recording threshold crossings without restarting. It coincides with the implemented trajectory through the first alarm.

A block is null for restart r if every stored output is generated under the reference parameter $\theta ^ { - }$ <sup>−</sup>. If that parameter remains active until control time τ, a block ending at output time $\tau$ is null: $y _ { \tau }$ is generated by $u _ { \tau - 1 }$ . On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ , (36) and Lemma 13 give, for each stored output in a null block,

$$
y _ { s } = w _ { s } + v _ { s } , \qquad v _ { s } : = b _ { 1 } ( \theta ^ { - } ) \{ u _ { s - 1 } - ( \lambda ^ { - } ) ^ { \prime } \psi _ { s - 1 } \} , \qquad | v _ { s } | \leq \varepsilon _ { y } .\tag{64}
$$

The variable $v _ { s }$ is $\mathcal { F } _ { s - 1 ^ { - } } \mathrm { m e a s u r a b l e } ,$ while $w _ { s }$ is independent of $\mathcal { F } _ { s - 1 }$ . Hence, for any null detector block S of m stored samples,

$$
\frac { 1 } { m } \sum _ { s \in S } ( y _ { s } ^ { 2 } - \sigma _ { w } ^ { 2 } ) = \frac { 1 } { m } \sum _ { s \in S } ( w _ { s } ^ { 2 } - \sigma _ { w } ^ { 2 } ) + \frac { 2 } { m } \sum _ { s \in S } w _ { s } v _ { s } + \frac { 1 } { m } \sum _ { s \in S } v _ { s } ^ { 2 } .\tag{65}
$$

Lemma 15 (One-block false-alarm bound). Suppose (52) holds. Conditional on any $\mathcal { F } _ { r + H }$ history on which $\mathcal G _ { r } ^ { \mathrm { e s t } }$ holds, any fixed potential block generated by the predictable sampling grid that is null for restart r crosses the threshold with probability at most $\delta _ { d } / ( 2 T ^ { 2 } )$

Proof. By (64), every summand in the last term of (65) satisfies $v _ { s } ^ { 2 } \le \varepsilon _ { y } ^ { 2 } .$ . Consequently,

$$
\frac { 1 } { m } \sum _ { s \in S } v _ { s } ^ { 2 } \leq \varepsilon _ { y } ^ { 2 } \leq \frac { \Delta } { 8 } ,
$$

where the final inequality is exactly the first condition in (52). The standard bounded-increment exponential supermartingale for the predictably sampled disturbance squares (Hoefding, 1963; Howard et al., 2020), applied along the fixed predictable block through its predetermined endpoint, gives

$$
\mathbb { P } \left( \frac { 1 } { m } \sum _ { s \in S } ( w _ { s } ^ { 2 } - \sigma _ { w } ^ { 2 } ) > \frac { \Delta } { 8 } \right) \leq \exp \left( - \frac { m \Delta ^ { 2 } } { 3 2 B _ { w } ^ { 4 } } \right) .\tag{66}
$$

The products $w _ { s } v _ { s }$ form bounded martingale diferences, so the same Azuma–Hoefding construction (Azuma, 1967; Hoefding, 1963) applies. If ${ { \varepsilon } _ { y } } = 0$ , this cross term vanishes identically. Otherwise, the analogous stopped exponential bound gives

$$
\mathbb { P } \bigg ( \frac { 2 } { m } \sum _ { s \in S } w _ { s } v _ { s } > \frac { \Delta } { 8 } \bigg ) \leq \exp \bigg ( - \frac { m \Delta ^ { 2 } } { 5 1 2 B _ { w } ^ { 2 } \varepsilon _ { y } ^ { 2 } } \bigg ) .\tag{67}
$$

With m from (54) and $B _ { o } \geq \operatorname* { m a x } \{ B _ { w } , \varepsilon _ { y } \}$ , each right-hand side is at most $\delta _ { d } / ( 4 T ^ { 2 } )$ . Outside these two deviations, the left side of (65) is at most $3 \Delta / 8 < \Delta / 2$ □

## C.3 Detection after a Change

Suppose the current cycle has not alarmed when the parameter changes from $\theta ^ { - }$ to $\theta ^ { + }$ at control time $\tau .$ . Starting from $\psi _ { \tau } ^ { \mathrm { a u x } } = \psi _ { \tau }$ , define the auxiliary process driven by the same future disturbances and the exact old gain:

$$
\psi _ { t + 1 } ^ { \mathrm { a u x } } = A _ { \Delta } \psi _ { t } ^ { \mathrm { a u x } } + e _ { 1 } w _ { t + 1 } ,
$$

$$
A _ { \Delta } : = A ( \theta ^ { - } , \theta ^ { + } ) .\tag{68}
$$

Let $y _ { t } ^ { \mathrm { a u x } }$ be its first coordinate. By (44), $| y _ { t } ^ { \mathrm { a u x } } | \le B _ { \mathrm { a u x } }$

Lemma 16 (Actual-to-auxiliary coupling). On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ , until the first of the next alarm and the next true change after $\tau _ { \mathrm { { ; } } }$

$$
\begin{array} { r } { \| \psi _ { t } - \psi _ { t } ^ { \mathrm { a u x } } \| \leq R _ { \Delta } ( H ) , } \end{array}\tag{69}
$$

$$
\left| y _ { t } ^ { 2 } - ( y _ { t } ^ { \mathrm { a u x } } ) ^ { 2 } \right| \leq 2 B _ { o } R _ { \Delta } ( H ) \leq \Delta / 8 .\tag{70}
$$

Proof. Writing the actual input as $u _ { t } = \lambda ( \theta ^ { - } ) ^ { \prime } \psi _ { t } + e _ { t }$ and subtracting (68) yields

$$
\psi _ { t + 1 } - \psi _ { t + 1 } ^ { \mathrm { a u x } } = A _ { \Delta } ( \psi _ { t } - \psi _ { t } ^ { \mathrm { a u x } } ) + g _ { u } ( \theta ^ { + } ) e _ { t } .
$$

Put $\delta _ { t } : = \psi _ { t } - \psi _ { t } ^ { \mathrm { a u x } }$ . Because the two processes start from the same state, $\delta _ { \tau } = 0$ . Iterating the displayed recursion gives, for every $t > \tau$ before the stopping endpoint,

$$
\delta _ { t } = \sum _ { k = \tau } ^ { t - 1 } A _ { \Delta } ^ { t - 1 - k } g _ { u } ( \theta ^ { + } ) e _ { k } .
$$

Lemma 13 gives $| e _ { k } | \le \varepsilon _ { u }$ , while (42) and (44) give $\lVert g _ { u } ( { \theta } ^ { + } ) \rVert \leq G _ { u }$ and $\left\| A _ { \Delta } ^ { j } \right\| \le C _ { \Delta } \rho _ { \Delta } ^ { j }$ Therefore

$$
\begin{array} { r l } {  { \| \delta _ { t } \| \le C _ { \Delta } G _ { u } \varepsilon _ { u } \sum _ { k = \tau } ^ { t - 1 } \rho _ { \Delta } ^ { t - 1 - k } } } \\ & { \le \frac { C _ { \Delta } G _ { u } \varepsilon _ { u } } { 1 - \rho _ { \Delta } } = R _ { \Delta } ( H ) , } \end{array}
$$

which is (69). Since output is the first state coordinate, $| y _ { t } - y _ { t } ^ { \mathrm { a u x } } | \leq \| \delta _ { t } \| \leq R _ { \Delta } ( H )$ . Both outputs are bounded by $B _ { o }$ , and hence

$$
\begin{array} { r l } & { | y _ { t } ^ { 2 } - ( y _ { t } ^ { \mathrm { a u x } } ) ^ { 2 } | = | y _ { t } - y _ { t } ^ { \mathrm { a u x } } | | y _ { t } + y _ { t } ^ { \mathrm { a u x } } | } \\ & { ~ \leq 2 B _ { o } R _ { \Delta } ( H ) \leq \displaystyle \frac { \Delta } { 8 } , } \end{array}
$$

where the last inequality is the second condition in (52). This proves (70).

For every auxiliary output time s whose preceding h controls are all post-change, independence gives the exact finite-horizon identity

$$
\mathbb { E } [ ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } \mid \mathcal { F } _ { s - h } ] = ( e _ { 1 } ^ { \prime } A _ { \Delta } ^ { h } \psi _ { s - h } ^ { \mathrm { a u x } } ) ^ { 2 } + \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { h - 1 } ( e _ { 1 } ^ { \prime } A _ { \Delta } ^ { j } e _ { 1 } ) ^ { 2 } .\tag{71}
$$

Indeed, iterating (68) for h steps and taking the first coordinate $\mathrm { g i }$ ves

$$
y _ { s } ^ { \mathrm { a u x } } = e _ { 1 } ^ { \prime } A _ { \Delta } ^ { h } \psi _ { s - h } ^ { \mathrm { a u x } } + \sum _ { j = 0 } ^ { h - 1 } e _ { 1 } ^ { \prime } A _ { \Delta } ^ { j } e _ { 1 } w _ { s - j } .
$$

Conditional on $\mathcal { F } _ { s - h }$ , the first term is fixed. Every future disturbance is centered and independent of that field and of the other future disturbances. Thus the state–noise cross terms and the cross terms with distinct disturbance indices have conditional expectation zero, while each diagonal disturbance term contributes its coeficient squared times $\sigma _ { w } ^ { 2 }$ . This proves (71) at deterministic s. The same identity holds at a finite lag-h predictable random output time: on each event $\{ s = t \} \in \mathcal { F } _ { t - h }$ it is the deterministic-time identity, and summing over t gives the equality conditional on the stopped field $\mathcal { F } _ { s - h }$ . The first term is nonnegative. Equations (45) and (53) imply the remaining inequality as follows. With $A _ { \Delta } = A _ { \mathrm { c l } } ( \lambda ( \theta ^ { - } ) , \theta ^ { + } )$ ,

$$
\begin{array} { r l } & { \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = 0 } ^ { h - 1 } ( e _ { 1 } ^ { \prime } A _ { \bigtriangleup } ^ { j } e _ { 1 } ) ^ { 2 } = J _ { y } ( \lambda ( \theta ^ { - } ) ; \theta ^ { + } ) - \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = h } ^ { \infty } ( e _ { 1 } ^ { \prime } A _ { \bigtriangleup } ^ { j } e _ { 1 } ) ^ { 2 } } \\ & { \qquad \quad \geq \sigma _ { w } ^ { 2 } + \Delta - \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = h } ^ { \infty } \left\| A _ { \bigtriangleup } ^ { j } \right\| ^ { 2 } } \\ & { \qquad \quad \geq \sigma _ { w } ^ { 2 } + \Delta - \frac { \sigma _ { w } ^ { 2 } C _ { \Delta } ^ { 2 } \rho _ { \Delta } ^ { 2 h } } { 1 - \rho _ { \Delta } ^ { 2 } } } \\ & { \qquad \quad \geq \sigma _ { w } ^ { 2 } + \frac { 7 \Delta } { 8 } . } \end{array}
$$

Combining this bound with the nonnegative conditional-state term in (71) proves

$$
\mathbb { E } [ ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } \mid \mathcal { F } _ { s - h } ] \ge \sigma _ { w } ^ { 2 } + \frac { 7 \Delta } { 8 } .\tag{72}
$$

Lemma 17 (One-change detection). Fix a potential restart r and a change time τ with $r { + } H \leq \tau$ and $\tau + D \le T$ . Let $\mathcal { A } _ { r , \tau } \in \mathcal { F } _ { \tau }$ be the event that the cycle at r is reached, $\theta ^ { - }$ is active throughout $[ r , \tau ) , \mathcal { G } _ { r } ^ { \mathrm { e s t } } \ h o l d s$ , and the cycle remains in exploitation after observing $y _ { \tau }$ . Suppose $\theta ^ { + }$ is active throughout $[ \tau , \tau + D )$ . Let $\widehat { \tau } _ { r }$ be this cycle’s first alarm output time strictly after $\tau ,$ with value +∞ if none occurs. Then, almost surely on $A _ { r , \tau }$

$$
\mathbb { P } ( \widehat { \tau } _ { r } > \tau + D \mid \mathcal { F } _ { \tau } ) \leq \frac { \delta _ { d } } { 4 T ^ { 2 } } .\tag{73}
$$

Proof. Condition on an arbitrary $\mathcal { F } _ { \tau }$ history in $A _ { r , \tau }$ . In the continuation in which alarms are suppressed, the restart-relative sampling grid, the auxiliary initial state $\psi _ { \tau }$ , and the old reference are now fixed, while all future disturbances remain fresh. Within fewer than 2h steps, the grid has an output time $s _ { 1 }$ with $s _ { 1 } - h \geq \tau$ . The next $m - 1$ grid times are h steps apart, and the last satisfies

$$
s _ { m } - \tau < 2 h + ( m - 1 ) h = ( m + 1 ) h = D .
$$

Thus the complete block exists within the finite horizon. If any earlier block alarms, the claim is already true. Otherwise the actual and alarm-suppressed paths agree through this complete post-change block.

The conditional version of Lemma 14 is valid because the future innovations are fresh after $\mathcal { F } _ { \tau }$ and $\mathcal { F } _ { \tau } \subseteq \mathcal { F } _ { s _ { 1 } }$ <sub>−</sub> by construction. It, together with (72) and $a = \Delta / 8$ , gives

$$
\mathbb { P } \bigg ( \frac { 1 } { m } \sum _ { k = 1 } ^ { m } ( y _ { s _ { k } } ^ { \mathrm { a u x } } ) ^ { 2 } < \sigma _ { w } ^ { 2 } + \frac { 3 \Delta } { 4 } \bigg ) \leq \exp \bigg ( { - \frac { m \Delta ^ { 2 } } { 1 2 8 B _ { o } ^ { 4 } } } \bigg ) \leq \frac { \delta _ { d } } { 4 T ^ { 2 } } .
$$

Outside this deviation, Lemma 16 applies because no second change occurs before the tested block and gives an actual average at least $\sigma _ { w } ^ { 2 } + 5 \Delta / 8$ , which crosses the detector threshold. This proves (73). Equivalently, for every $A \in { \mathcal { F } } _ { \tau }$ with $A \subseteq A _ { r , \tau }$

$$
\mathbb { P } ( A \cap \{ \widehat { \tau } _ { r } > \tau + D \} ) \leq \frac { \delta _ { d } } { 4 T ^ { 2 } } \mathbb { P } ( A ) ,
$$

which is the eventwise form used below.

## C.4 Alarm Correspondence

Theorem 18 (Alarm correspondence and delay). Suppose Assumptions 7–12 hold. Let $\sigma _ { \mathrm { d e t } }$ be the first output time at which a false alarm occurs or the first deadline at which a true change remains unmatched, with inf $\varnothing = + \infty$ . For each potential r, let $\mathcal { E } _ { r } ^ { \mathrm { r e a c h } }$ be the event that r is reached, $r < \sigma _ { \mathrm { d e t } }$ , and $[ r , r + H )$ is a full, unchanged exploration block. Define

$$
\mathcal { E } _ { \mathrm { r e f } } : = \bigcap _ { r = 0 } ^ { T } \{ ( \mathcal { E } _ { r } ^ { \mathrm { r e a c h } } ) ^ { c } \cup \mathcal { G } _ { r } ^ { \mathrm { e s t } } \} .\tag{74}
$$

Let $\mathcal { E } _ { \mathrm { d e t } }$ be the event that PIECE-CD has no false alarms and raises exactly one alarm $\widehat { \tau } ^ { \left( c \right) }$ for each true change, with

$$
\tau ^ { ( c ) } < \widehat { \tau } ^ { ( c ) } \leq \tau ^ { ( c ) } + D , \qquad c = 1 , \ldots , C .\tag{75}
$$

Then

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { r e f } } \cap \mathcal { E } _ { \mathrm { d e t } } ^ { c } ) \leq \delta _ { d } .\tag{76}
$$

Proof. There are at most $T$ potential restart locations and at most T potential test endpoints after each location. Fix such a pair. Intersect its crossing event with the events that the restart is reached, its reference event holds, no earlier detector failure has occurred, and every control generating an output in the block uses the same parameter as the reference exploration; that is, the block is null for this restart. Conditional on the history at the end of exploration, apply Lemma 15 to the fixed predictable block in the continuation with alarms suppressed. If the actual test is reached, the continuation and actual trajectory agree through that test. Thus the probability of this joint gated event, rather than a probability conditional on survival to the test, is at most $\delta _ { d } / ( 2 T ^ { 2 } )$ . A union bound over the restart–endpoint pairs gives a total false-alarm contribution at most $\delta _ { d } / 2$

For a true change $\tau ^ { \left( c \right) }$ , intersect missed detection with the successful history up to that change. For each potential $r ,$ the event that the past detector history is successful, the cycle at r is active at $\tau ^ { \left( c \right) }$ , and its reference succeeds is an $\mathcal { F } _ { \tau ^ { ( c ) } }$ -measurable subset of $\boldsymbol { \mathcal { A } } _ { \boldsymbol { r } , \tau ^ { ( c ) } }$ from Lemma 17. These events are disjoint over r. The dwell conditions give $r + H \leq \tau ^ { ( c ) } , \tau ^ { ( c ) } + D \leq T$ and no second change in $[ \tau ^ { ( c ) } , \bar { \tau ^ { ( c ) } } + D )$ . Multiplying (73) by each active-cycle indicator, taking expectations, and summing over the disjoint partition gives $\delta _ { d } / ( 4 T ^ { 2 } )$ for that change, without an additional restart factor. There are at most $C \leq T - 1$ changes, so the total missed-delay contribution is at most $C \delta _ { d } / ( 4 T ^ { 2 } ) \leq ( T - 1 ) \delta _ { d } / ( 4 T ^ { 2 } ) < \delta _ { d } / 4$ . A block containing outputs from both sides of a true change is not a null block; if it crosses, it is a valid earlier detection. On the complement of these gated deviations, induction over the change index gives one-to-one alarm correspondence. The deterministic calendar calculation in Appendix H shows that the complete post-change block ends strictly before $\tau ^ { ( c ) } + D ;$ no probability is spent on boundary or integer-rounding conventions. The dwell condition ensures that each alarm occurs before the next change and, when another change remains, leaves an uncontaminated H-step exploration block for the next cycle. The false-alarm contribution is at most $\delta _ { d } / 2$ and the missed-delay contribution is less than $\delta _ { d } / 4$ , so their sum is below $\delta _ { d } .$ . This proves (76). □

## D REGRET ANALYSIS

On the detection event of Theorem 18, the horizon splits into exploration, detection-delay, and unchanged exploitation intervals. The first two have bounded per-round cost; the RLS potential argument controls the cumulative cost of the third.

## D.1 Segmentwise Exploitation Regret

Consider an unchanged exploitation interval following a restart at r. Suppress the restart subscript and write $\Breve { b _ { \star } } = \tilde { b _ { 1 } ( \theta ) } , \lambda = \lambda ( \theta ) , \widehat { b } = \widehat { b } _ { r }$ , and $\overline { { \lambda } } = \overline { { \lambda } } _ { r }$ . Define

$$
\alpha : = 1 - \frac { b _ { \star } } { \widehat { b } } ,
$$

$$
\overline { { \alpha } } : = \frac { E _ { \theta } ( H , \delta _ { 0 } ) } { b _ { \operatorname* { m i n } } } ,
$$

$$
\underline { { { v } } } _ { H } : = \nu + \kappa _ { G } n _ { H } ,
$$

$$
\overline { { \ell } } : = \frac { B _ { \psi } ^ { 2 } } { \underline { { v } } _ { H } } ,
$$

$$
c _ { 0 } : = 1 - 2 \overline { { \alpha } } - 2 \overline { { \ell } } ( 1 + \overline { { \alpha } } ) ^ { 2 } ,
$$

$$
B _ { \eta } : = B _ { w } / b _ { \mathrm { m i n } } ,
$$

$$
\overline { { { Q } } } _ { H } : = E _ { \lambda } ( H , \delta _ { 0 } ) ^ { 2 } ( \nu + n _ { H } B _ { \psi } ^ { 2 } ) ,
$$

$$
\Gamma ( n ) : = d \log \left( 1 + \frac { n B _ { \psi } ^ { 2 } } { d \underline { { { v } } } _ { H } } \right) .\tag{77}
$$

The tuning condition (51) ensures $c _ { 0 } > 0 . 7$

Theorem 19 (Logarithmic exploitation regret). Suppose a single parameter θ is active throughout the full exploration block $[ r , r + H )$ and throughout an exploitation interval $[ r + H , r + H + N )$ of that same cycle, ending no later than its next alarm. Suppose the Gram and estimation events in Propositions 21 and 22 hold for this same θ. Conditional on the history at the end of exploration, with probability at least $1 - \delta _ { 0 }$ , simultaneously for every prefix length $0 \leq n \leq N$

$$
\begin{array} { c } { \displaystyle \sum _ { t = r + H } ^ { r + H + n - 1 } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } \leq R _ { E } ( n , \delta _ { 0 } ) , } \\ { \displaystyle R _ { E } ( n , \delta _ { 0 } ) : = \bar { b } ^ { 2 } \left[ \frac { 2 \overline { { Q } } _ { H } } { c _ { 0 } } + \frac { 8 B _ { \eta } ^ { 2 } } { c _ { 0 } ^ { 2 } } \log \frac { 1 } { \delta _ { 0 } } + \frac { 4 B _ { \eta } ^ { 2 } } { c _ { 0 } } \Gamma ( n ) \right] . } \end{array}\tag{78}
$$

Proof. We give the core argument; Appendix G records the matrix identities and conditioning details. Let $d _ { t } = \widehat { \lambda } _ { t } - \lambda , a _ { t } = d _ { t } ^ { \prime } \psi _ { t }$ , and $e _ { t } = u _ { t } - \lambda ^ { \prime } \psi _ { t }$ . Lemma 13 gives $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$ . From (36), the pseudo-response in (59) obeys

$$
x _ { t + 1 } ^ { \lambda } = \lambda ^ { \prime } \psi _ { t } + \alpha e _ { t } + \eta _ { t + 1 } ,
$$

$$
\eta _ { t + 1 } : = - w _ { t + 1 } / \widehat { b } .
$$

Here $\eta _ { t + 1 }$ is conditionally mean zero and $B _ { \eta ^ { - } \mathrm { s u b - G a u s s i a n } }$

Let $Q _ { t } = d _ { t } ^ { \prime } V _ { t } d _ { t }$ and $\ell _ { t } = \psi _ { t } ^ { \prime } V _ { t + 1 } ^ { - 1 } \psi _ { t }$ . To display the algebra, set $\omega _ { t } : = \alpha e _ { t } + \eta _ { t + 1 } - a _ { t }$ . The RLS update and $V _ { t + 1 } = V _ { t } + \psi _ { t } \psi _ { t } ^ { \prime } \ \mathrm { g i }$ ve

$$
\begin{array} { r l } & { d _ { t + 1 } = d _ { t } + V _ { t + 1 } ^ { - 1 } \psi _ { t } \omega _ { t } , } \\ & { Q _ { t + 1 } = d _ { t } ^ { \prime } V _ { t + 1 } d _ { t } + 2 \omega _ { t } d _ { t } ^ { \prime } \psi _ { t } + \omega _ { t } ^ { 2 } \psi _ { t } ^ { \prime } V _ { t + 1 } ^ { - 1 } \psi _ { t } } \\ & { ~ = Q _ { t } + a _ { t } ^ { 2 } + 2 a _ { t } \omega _ { t } + \ell _ { t } \omega _ { t } ^ { 2 } . } \end{array}
$$

Substituting the definition of $\omega _ { t }$ and collecting $a _ { t } ^ { 2 } - 2 a _ { t } ^ { 2 } = - a _ { t } ^ { 2 }$ gives the exact identity

$$
Q _ { t + 1 } - Q _ { t } = - a _ { t } ^ { 2 } + 2 a _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } ) + \ell _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } - a _ { t } ) ^ { 2 } .\tag{79}
$$

On the exploration event, $| \alpha | \leq \overline { { \alpha } }$ and $\ell _ { t } \leq \overline { { \ell } } .$ . Using $\left| e _ { t } \right| \le \left| a _ { t } \right|$ in (79) yields

$$
Q _ { t + 1 } - Q _ { t } \leq - c _ { 0 } a _ { t } ^ { 2 } + 2 a _ { t } \eta _ { t + 1 } + 2 \ell _ { t } \eta _ { t + 1 } ^ { 2 } .\tag{80}
$$

For completeness, conditional $B _ { \eta ^ { - } \mathrm { s u b - G } }$ aussianity and predictability of $a _ { t }$ imply that, for every fixed $\xi > 0$

$$
M _ { n } : = \exp \left\{ \xi \sum _ { t = r + H } ^ { r + H + n - 1 } a _ { t } \eta _ { t + 1 } - \frac { \xi ^ { 2 } B _ { \eta } ^ { 2 } } { 2 } \sum _ { t = r + H } ^ { r + H + n - 1 } a _ { t } ^ { 2 } \right\}
$$

is a nonnegative supermartingale starting at one. Ville’s inequality (Ville, 1939; Howard et al., 2020) therefore gives, with probability at least $1 - \delta _ { 0 }$ , simultaneously over every prefix,

$$
\sum _ { t } a _ { t } \eta _ { t + 1 } \leq \frac { \xi B _ { \eta } ^ { 2 } } { 2 } \sum _ { t } a _ { t } ^ { 2 } + \frac { 1 } { \xi } \log { \frac { 1 } { \delta _ { 0 } } } .
$$

Taking the deterministic value $\xi = c _ { 0 } / ( 2 B _ { \eta } ^ { 2 } )$ yields

$$
\sum _ { t } a _ { t } \eta _ { t + 1 } \leq \frac { c _ { 0 } } { 4 } \sum _ { t } a _ { t } ^ { 2 } + \frac { 2 B _ { \eta } ^ { 2 } } { c _ { 0 } } \log \frac { 1 } { \delta _ { 0 } } .\tag{81}
$$

Appendix G gives the same argument directly from the one-step conditional moment-generatingfunction inequality. Moreover, the matrix determinant lemma gives

$$
\sum _ { t } \ell _ { t } \leq \log { \frac { \operatorname* { d e t } V _ { r + H + n } } { \operatorname* { d e t } V _ { r + H } } } \leq \Gamma ( n ) .\tag{82}
$$

Summing (80), using $Q _ { t } \geq 0 , \eta _ { t + 1 } ^ { 2 } \leq B _ { \eta } ^ { 2 }$ , and (81)– (82), gives

$$
\sum _ { t } a _ { t } ^ { 2 } \leq \frac { 2 Q _ { r + H } } { c _ { 0 } } + \frac { 8 B _ { \eta } ^ { 2 } } { c _ { 0 } ^ { 2 } } \log \frac { 1 } { \delta _ { 0 } } + \frac { 4 B _ { \eta } ^ { 2 } } { c _ { 0 } } \Gamma ( n ) .\tag{83}
$$

Finally, $Q _ { r + H } \leq \overline { { Q } } _ { H }$ and, by (36), $( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } = b _ { \star } ^ { 2 } e _ { t } ^ { 2 } \leq \bar { b } ^ { 2 } a _ { t } ^ { 2 }$ . This proves (78).

## D.2 Delay and Exploration Costs

The deterministic state bound gives the per-step delay cost

$$
K _ { D } : = ( B _ { Y } + B _ { w } ) ^ { 2 } .\tag{84}
$$

During exploration, $| u _ { t } | \le B _ { e }$ and $\lVert \psi _ { t } \rVert \leq B _ { \psi }$ , so (36) gives the per-step bound

$$
K _ { X } : = \bar { b } ^ { 2 } ( B _ { e } + \Lambda B _ { \psi } ) ^ { 2 } .\tag{85}
$$

Thus a delay interval of length at most D costs at most $K _ { D } D$ , and an exploration block costs at most $K _ { X } H$

## D.3 Total Regret

Theorem 20 (Finite-time regret of PIECE-CD). Let Assumptions 7–12 hold. Let H be the smallest integer greater than $r _ { 0 }$ that satisfies (93) and (51)–(52); choose h, m, D by (53)–(54), and run Algorithm 2. Then, with probability at least 1 − δ, there are no false alarms, every change has exactly one alarm satisfying (75), and

$$
\mathcal { R } _ { T } \leq C K _ { D } D + ( C + 1 ) K _ { X } H + ( C + 1 ) R _ { E } ( T , \delta _ { 0 } ) .\tag{86}
$$

Consequently, for fixed orders and fixed model, noise, initialization, exploration, ridge, stability, clipping, and gap bounds,

$$
\mathcal { R } _ { T } = O \bigg ( ( C + 1 ) \log \frac { T + 1 } { \delta } \bigg ) .\tag{87}
$$

Proof. If $T \leq H$ , Assumption 12 forces $C = 0$ . The algorithm then performs only truncated exploration, and the deterministic exploration bound gives $\mathcal { R } _ { T } \leq K _ { X } T \leq K _ { X } H$ , which is bounded by the right-hand side of (86). Hence assume $T > H$ below.

At every potential full, unchanged restart block, gate the estimation and RLS failure events by the event that this restart is reached with a successful history, and then condition on the past. Propositions 21 and 22 cost at most $2 \delta _ { 0 }$ , and the time-uniform RLS event in Theorem 19 costs another $\delta _ { 0 }$ . The tower property and a union bound over at most $T + 1$ locations give $3 ( T + 1 ) \delta _ { 0 } = 3 \delta / 3 2$ . Gating the false-alarm and delay events in the same chronological way, Theorem 18 costs less than $\delta _ { d } = \delta / 2$ . Their intersection therefore has probability greater than $1 - \delta .$ . Appendix I gives the full induction; in particular, it never conditions an early detector event on future estimator success.

On this intersection, each of the C true changes has a pre-alarm interval of length at most $D ,$ giving $C K _ { D } D$ . There is one initial exploration block and at most one block after each alarm, giving at most $( C + 1 ) K _ { X } H$ . Removing these intervals leaves at most $C + 1$ unchanged exploitation prefixes. Apply Theorem 19 to each and upper-bound its length by $T ;$ this gives the last term in (86).

Appendix J constructs a feasible $H = O ( \log ( ( T + 1 ) / \delta ) )$ ; the smallest feasible value is therefore no larger. Equations $( 5 3 ) \mathrm { - } ( 5 4 )$ give fixed h and $D = O ( \log ( ( T + 1 ) / \delta ) )$ . Finally, $\overline { { Q } } _ { H }$ $\Gamma ( T )$ , and $\log ( 1 / \delta _ { 0 } )$ are all $O ( \log ( ( T + 1 ) / \delta ) )$ ). Substitution proves (87). When $C = 0$ , the delay term is absent, leaving the stationary logarithmic bound rather than the meaningless expression $O ( C \log T )$ . □

## E DETERMINISTIC BOUNDS AND RESTART CONVEN-TIONS

Bounded applied inputs and Assumptions 7–9 yield the pathwise bounds (40).

## E.1 Output and Regressor Bounds

Let $Y _ { t } = ( y _ { t } , \dots , y _ { t - p + 1 } ) ^ { \prime }$ . Equation (32) has the state form

$$
Y _ { t + 1 } = A _ { y } ( \theta _ { t } ) Y _ { t } + e _ { 1 } \zeta _ { t + 1 } , \qquad \zeta _ { t + 1 } : = w _ { t + 1 } + \sum _ { j = 1 } ^ { q } b _ { j } ( \theta _ { t } ) u _ { t - j + 1 } .\tag{88}
$$

Since $| u _ { t } | \le B _ { u }$ by construction,

$$
| \zeta _ { t + 1 } | \le B _ { w } + B _ { b } B _ { u } .
$$

Unrolling (88) and applying (39) gives

$$
\begin{array} { r l r } {  { \| \boldsymbol { Y } _ { t } \| \le C _ { y } \rho _ { y } ^ { t } \| \boldsymbol { Y } _ { 0 } \| + C _ { y } ( B _ { w } + B _ { b } B _ { u } ) \sum _ { j = 0 } ^ { t - 1 } \rho _ { y } ^ { j } } } \\ & { } & { \leq C _ { y } \| \boldsymbol { Y } _ { 0 } \| + \frac { C _ { y } } { 1 - \rho _ { y } } ( B _ { w } + B _ { b } B _ { u } ) = B _ { Y } . } \end{array}
$$

Every output coordinate is therefore bounded by $B _ { Y }$ . The definition (33) and clipping give

$$
\begin{array} { r } { \| \psi _ { t } \| ^ { 2 } \leq p B _ { Y } ^ { 2 } + ( q - 1 ) B _ { u } ^ { 2 } = B _ { \psi } ^ { 2 } , } \end{array}
$$

and (34) gives $\lVert \phi _ { t } \rVert \leq B _ { \phi }$

## E.2 Projection onto the Clipping Interval

Let clip<sub>[−</sub> $\mathbf { \partial } _ { \cdot B , B ] } ( x ) = ( - B ) \lor x \land B$ . As the Euclidean projection onto a closed convex set, it is nonexpansive:

$$
| \exp _ { [ - B , B ] } ( x ) - \exp _ { [ - B , B ] } ( z ) | \leq | x - z | .
$$

If $z \in [ - B , B ]$ , this becomes

$$
| \exp _ { [ - B , B ] } ( x ) - z | \leq | x - z | .\tag{89}
$$

Assumption 10 permits (89) with $z = \lambda ( \theta ) ^ { \prime } \psi _ { t }$ for $\theta \in \Theta$ . For an arbitrary target $z ,$ the bound includes its distance from the clipping interval:

$$
| \mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( x ) - z | \leq | x - z | + ( | z | - B _ { u } ) _ { + } .
$$

## E.3 Auxiliary-Process Bounds

For the fixed mismatch matrix $A _ { \Delta }$ in (68), (44) gives, for $t \geq \tau$

$$
\begin{array} { r l } & { \| \psi _ { t } ^ { \mathrm { a u x } } \| \le C _ { \Delta } \rho _ { \Delta } ^ { t - \tau } \| \psi _ { \tau } \| + C _ { \Delta } B _ { w } \displaystyle \sum _ { j = 0 } ^ { t - \tau - 1 } \rho _ { \Delta } ^ { j } } \\ & { \qquad \le C _ { \Delta } B _ { \psi } + \displaystyle \frac { C _ { \Delta } B _ { w } } { 1 - \rho _ { \Delta } } = B _ { \mathrm { a u x } } . } \end{array}
$$

Thus both actual and auxiliary outputs are bounded in absolute value by $B _ { o }$

## E.4 Conditioning at a Random Restart

An alarm time r is an $\{ \mathcal { F } _ { t } \}$ -stopping time. Conditional on ${ \mathcal { F } } _ { r } ,$ the state $\psi _ { r }$ is fixed with $\| \psi _ { r } \| \leq$ $B _ { \psi } ,$ , and future disturbances retain the independence, moments, and bound in Assumption 9. The input $u _ { r }$ is already $\mathcal { F } _ { r }$ -measurable; subsequent exploratory inputs are fresh and independent of the disturbances.

Consider a conditional concentration bound that holds uniformly over these restart histories, with failure probability at most $\varepsilon _ { \mathrm { f a i l } }$ . For any eligibility event $\mathcal { C } _ { r } \in \mathcal { F } _ { r }$ , the tower property gives

$$
\begin{array} { r l } & { \mathbb { P } ( \mathcal { C } _ { r } \cap \{ \mathrm { f a i l u r e ~ a t ~ } r \} ) = \mathbb { E } [ \mathbf { 1 } _ { \mathcal { C } _ { r } } \mathbb { P } ( \mathrm { f a i l u r e ~ a t ~ } r \mid \mathcal { F } _ { r } ) ] } \\ & { \qquad \leq \varepsilon _ { \mathrm { f a i l } } \mathbb { P } ( \mathcal { C } _ { r } ) \leq \varepsilon _ { \mathrm { f a i l } } . } \end{array}
$$

A union bound over at most $T + 1$ possible restart locations then controls these failures across the horizon. This is the basis for the allocation (48).

## F EXPLORATION, EXCITATION, AND THE RESTARTED ESTIMATE

The covariance analysis of stationary PIECE in Singh et al. (2024) concerns sparse, recurring exploration episodes. Here we establish linear Gram growth and estimation bounds for a contiguous exploration block after a random restart.

## F.1 Conditional Excitation

Fix a complete exploration block $[ r , r + H )$ during which the parameter remains equal to $\theta .$ Put $s : = p + q , L : = r _ { 0 } = \operatorname* { m a x } \{ p , q \}$ , and $\underline { { \sigma } } ^ { 2 } : = \operatorname* { m i n } \{ \sigma _ { w } ^ { 2 } , \sigma _ { e } ^ { 2 } \}$ . At any exploration time $t \geq r + L$ collect the most recent innovations in

$$
Z _ { t } : = ( w _ { t - p + 1 } , \ldots , w _ { t } , u _ { t - q + 1 } , \ldots , u _ { t } ) ^ { \prime } .
$$

Because the parameter is fixed within the exploration block, repeated substitution in the ARX recursion gives

$$
\phi _ { t } = M ( \theta ) Z _ { t } + \xi _ { t } .\tag{90}
$$

To make the remainder explicit, let $\overline { { Z } } _ { t }$ collect the members of $( w _ { t - L + 1 } , \dots , w _ { t } , u _ { t - L + 1 } , \dots , u _ { t } )$ that do not already occur in $Z _ { t }$ . Repeated substitution gives $\xi _ { t } = c _ { t } + \overline { { M } } ( \theta ) \overline { { Z } } _ { t } .$ , where $c _ { t }$ is F -measurable. Conditional on $\mathcal { F } _ { t - L }$ , the vector $Z _ { t }$ is centered, is independent of $\overline { { Z } } _ { t }$ , and has covariance

$$
\operatorname { C o v } ( Z _ { t } \mid \mathcal { F } _ { t - L } ) = \operatorname { d i a g } ( \sigma _ { w } ^ { 2 } I _ { p } , \sigma _ { e } ^ { 2 } I _ { q } ) .
$$

Thus $Z _ { t }$ and $\xi _ { t }$ are conditionally independent. The remainder depends on the $\mathcal { F } _ { t - L }$ -measurable history and the omitted innovations.

To see the triangular structure explicitly, order the output coordinates and their disturbances chronologically as $( y _ { t - p + 1 } , \ldots , y _ { t } )$ and $( w _ { t - p + 1 } , \ldots , w _ { t } )$ . The equation for $y _ { j }$ contains $w _ { j }$ with coeficient one and only disturbances dated at or before $j$ . Thus the output–disturbance block is triangular with unit diagonal. Ordering the current exploration inputs as $( u _ { t - q + 1 } , \ldots , u _ { t } )$ makes the input-coordinate block the identity. All remaining post-F<sub>t−L</sub> innovations are precisely those collected in $\overline { { Z } } _ { t } ;$ older terms belong to $c _ { t }$ . Hence, up to these fixed row and column permutations, $M ( \theta )$ has the block-triangular form

$$
M ( \theta ) = \left[ \begin{array} { c c } { { T _ { w } ( \theta ) } } & { { T _ { u } ( \theta ) } } \\ { { 0 } } & { { I _ { q } } } \end{array} \right] ,
$$

where $T _ { w } ( \theta )$ is triangular with unit diagonal: each $y _ { j }$ contains its contemporaneous disturbance $w _ { j }$ with coeficient one. Hence $M ( \theta )$ is nonsingular. Continuity and compactness give

$$
\chi : = \operatorname* { i n f } _ { \theta \in \Theta } \sigma _ { \operatorname* { m i n } } ( M ( \theta ) ) > 0 , \qquad \kappa _ { \mathrm { i n n } } : = \underline { { { \sigma } } } ^ { 2 } \chi ^ { 2 } > 0 .
$$

For completeness, let $\Sigma _ { Z } : = \mathrm { C o v } ( Z _ { t } \mid \mathcal { F } _ { t - L } )$ . Conditional independence removes both crosscovariance terms, and hence

$$
\begin{array} { r l } & { \mathrm { C o v } ( \phi _ { t } \mid \mathcal { F } _ { t - L } ) = M ( \theta ) \Sigma _ { Z } M ( \theta ) ^ { \prime } + \mathrm { C o v } ( \xi _ { t } \mid \mathcal { F } _ { t - L } ) } \\ & { \qquad \quad \succeq \underline { { \sigma } } ^ { 2 } M ( \theta ) M ( \theta ) ^ { \prime } \succeq \underline { { \sigma } } ^ { 2 } \sigma _ { \mathrm { m i n } } ( M ( \theta ) ) ^ { 2 } I _ { s } . } \end{array}
$$

Also, $\mathbb { E } [ \phi _ { t } \phi _ { t } ^ { \prime } \mid \mathcal { F } _ { t - L } ]$ equals this covariance plus the positive semidefinite outer product of the conditional mean. Therefore

$$
\begin{array} { r } { \mathbb E [ \phi _ { t } \phi _ { t } ^ { \prime } \mid \mathcal F _ { t - L } ] \succeq \mathrm { C o v } ( \phi _ { t } \mid \mathcal F _ { t - L } ) \succeq \kappa _ { \mathrm { i n n } } I _ { s } . } \end{array}\tag{91}
$$

The bound is uniform over the history in $\mathcal { F } _ { t - L }$

## F.2 Exploration Gram Bound

Fix a restart r and the trimmed set $\mathcal { T } _ { r } = \{ r + r _ { 0 } , \ldots , r + H - 1 \}$ . Let

$$
\Phi _ { r } : = \left[ \begin{array} { c } { \phi _ { r + r _ { 0 } } ^ { \prime } } \\ { \vdots } \\ { \phi _ { r + H - 1 } ^ { \prime } } \end{array} \right] , \qquad G _ { r } : = \Phi _ { r } ^ { \prime } \Phi _ { r } = \sum _ { t \in \mathcal { T } _ { r } } \phi _ { t } \phi _ { t } ^ { \prime } .
$$

Set

$$
N _ { H } : = 1 + \left\lfloor \frac { n _ { H } - 1 } { L } \right\rfloor , \qquad \kappa _ { G } : = \frac { \kappa _ { \mathrm { i n n } } } { 2 L } .\tag{92}
$$

Consider only the L-spaced rows $t _ { k } = r + L + ( k - 1 ) L , k = 1 , \dots , N _ { H }$ . They are contained in $\mathcal { T } _ { r }$ , and $N _ { H } \ge n _ { H } / L$

The conditioning argument in Appendix E applies at a stopping-time restart r. Although $u _ { r }$ is already F -measurable, the trimming starts at $t = r + L$ , so every innovation appearing in $Z _ { t }$ has index at least $r + 1$ . Consequently, (90)–(96) apply without treating r as deterministic.

Proposition 21 (Exploration Gram bound). Let r be a stopping time with $r + H \leq T$ , and suppose a single $\theta \in \Theta$ is active at every control time $t = r , \ldots , r + H - 1$ . If

$$
N _ { H } \ge { \frac { 1 0 B _ { \phi } ^ { 4 } } { \kappa _ { \mathrm { i n n } } ^ { 2 } } } \log { \frac { s } { \delta _ { 0 } } } ,\tag{93}
$$

then, conditional on $\mathcal { F } _ { r }$

$$
\begin{array} { r } { \mathbb { P } \Big ( \lambda _ { \operatorname* { m i n } } ( G _ { r } ) \geq \kappa _ { G } n _ { H } , \quad \mathrm { t r } ( G _ { r } ) \leq n _ { H } B _ { \phi } ^ { 2 } \ \Big | \ \mathcal { F } _ { r } \Big ) \geq 1 - \delta _ { 0 } . } \end{array}\tag{94}
$$

The same event implies

$$
\lambda _ { \operatorname* { m i n } } \left( \sum _ { t \in \mathcal { T } _ { r } } \psi _ { t } \psi _ { t } ^ { \prime } \right) \geq \kappa _ { G } n _ { H } .\tag{95}
$$

Proof. Condition on $\mathcal { F } _ { r }$ . For $X _ { k } : = \phi _ { t _ { k } } \phi _ { t _ { k } } ^ { \prime }$ , use the stopped filtration $\mathcal { H } _ { 0 } = \mathcal { F } _ { \iota }$ and $\mathcal { H } _ { k } = \mathcal { F } _ { t _ { k } } =$ $\mathcal { F } _ { r + k L }$ . Here $t _ { 1 } - L = r .$ , and $t _ { k } - L = t _ { k - 1 }$ for $k \geq 2$ , so (91) gives

$$
A _ { k } : = \mathbb { E } [ X _ { k } \mid { \mathcal { H } } _ { k - 1 } ] \succeq \kappa _ { \mathrm { i n n } } I _ { s } .
$$

Also $0 \preceq X _ { k } \preceq B _ { \phi } ^ { 2 } I _ { s }$ . Thus $D _ { k } : = X _ { k } - A _ { k }$ is a self-adjoint matrix-martingale diference with $\| D _ { k } \| \le B _ { \phi } ^ { 2 }$ and predictable quadratic variation bounded by $N _ { H } B _ { \phi } ^ { 4 } I _ { s }$ . Moreover, $\kappa _ { \mathrm { i n n } } \leq B _ { \phi } ^ { 2 }$ because $A _ { k } \ \preceq \ B _ { \phi } ^ { 2 } I _ { s }$ . The lower-tail form of the matrix Freedman inequality (Tropp, 2011), applied to $- \sum k { \dot { D } } _ { k }$ , gives

$$
\begin{array} { r l r } {  { \mathbb { P } ( \lambda _ { \operatorname* { m i n } } ( \sum _ { k = 1 } ^ { N _ { H } } X _ { k } ) \leq \frac { \kappa _ { \operatorname* { i n n } } N _ { H } } { 2 }  \begin{array} { l } { \mathcal { F } _ { r } } \\ {  } \end{array} \} \le s \exp \{ - \frac { \kappa _ { \operatorname* { i n n } } ^ { 2 } N _ { H } } { 8 B _ { \phi } ^ { 4 } + ( 4 / 3 ) B _ { \phi } ^ { 2 } \kappa _ { \operatorname* { i n n } } } \} } } \\ & { } & { \leq s \exp ( - \frac { \kappa _ { \operatorname* { i n n } } ^ { 2 } N _ { H } } { 1 0 B _ { \phi } ^ { 4 } } ) . } \end{array}\tag{96}
$$

Condition (93) makes the right-hand side at most $\delta _ { 0 }$ . Since $G _ { r } \succeq \sum _ { k } X _ { k }$ and $N _ { H } \ge n _ { H } / L$ , the claimed eigenvalue bound follows. The trace bound is deterministic because $\lVert \phi _ { t } \rVert \leq B _ { \phi }$

Finally, $\textstyle \sum _ { t \in \mathbb { Z } _ { r } } \psi _ { t } \psi _ { t } ^ { \prime }$ is the principal block of $G _ { r }$ obtained by deleting the current-input row and column. Eigenvalue interlacing, or direct restriction of the Rayleigh quotient, proves (95).

## F.3 Projected Ridge Estimation

Proposition 22 (Restarted parameter and gain estimate). Let r be a stopping time with $r { + } H \leq$ $T _ { i }$ , and suppose a single $\theta \in \Theta$ is active at every control time $t = r , \ldots , r + H - 1$ . Given $\mathcal { F } _ { r }$

event (100) has probability at least $1 - \delta _ { 0 }$ . On its intersection with the event in Proposition 21, the estimator defined in (55)–(56) satisfies

$$
\begin{array} { r } { \left\| \widehat { \theta } _ { r } - \theta \right\| \leq E _ { \theta } ( H , \delta _ { 0 } ) , \quad \quad } \\ { \left\| \overline { { \lambda } } _ { r } - \lambda ( \theta ) \right\| \leq E _ { \lambda } ( H , \delta _ { 0 } ) = g _ { H } / 4 . } \end{array}\tag{97}
$$

Moreover, the RLS initialization satisfies

$$
\smash { \ v { v } _ { H } \boldsymbol { I } _ { d } \preceq { V } _ { r + H } \preceq ( \nu + n _ { H } \boldsymbol { B } _ { \psi } ^ { 2 } ) \boldsymbol { I } _ { d } }\tag{98}
$$

in the sense of its smallest- and largest-eigenvalue bounds. If (93) holds, the Gram event and the two estimates above therefore hold jointly with conditional probability at least $1 - 2 \delta _ { 0 }$

Proof. Let $V = V _ { r } ^ { \theta }$ and $\begin{array} { r } { S _ { r } = \sum _ { t \in \mathcal { T } _ { \mathrm { i } } } } \end{array}$ ϕ<sub>t</sub> $w _ { t + 1 }$ . Since $y _ { t + 1 } = \phi _ { t } ^ { \prime } \theta + w _ { t + 1 }$ during the uncontaminated block, write $\begin{array} { r } { G _ { r } = \sum _ { t \in \mathbb { Z } _ { r } } \phi _ { t } \phi _ { t } ^ { \prime } } \end{array}$ and calculate

$$
\begin{array} { r l } { \displaystyle \sum _ { t \in \mathbb { Z } _ { r } } \phi _ { t } y _ { t + 1 } = \displaystyle \sum _ { t \in \mathbb { Z } _ { r } } \phi _ { t } ( \phi _ { t } ^ { \prime } \theta + w _ { t + 1 } ) } & { } \\ { \displaystyle } & { = G _ { r } \theta + S _ { r } = ( V - \nu I _ { s } ) \theta + S _ { r } . } \end{array}
$$

Multiplication by $V ^ { - 1 }$ gives

$$
\widetilde { \theta } _ { r } = \theta + { V } ^ { - 1 } ( S _ { r } - \nu \theta ) ,
$$

and therefore

$$
\widetilde { \theta } _ { r } - \theta = V ^ { - 1 } ( S _ { r } - \nu \theta ) .\tag{99}
$$

Assumption 9 and Hoefding’s lemma make $w _ { t + 1 }$ conditionally $B _ { w } \mathrm { - s u b \mathrm { - } G a u s s i a n }$ . The fixedridge self-normalized inequality (Abbasi-Yadkori et al., 2011, Theorem 1) gives, with conditional probability at least $1 - \delta _ { 0 }$ 2

$$
\| S _ { r } \| _ { V ^ { - 1 } } \leq B _ { w } \sqrt { \log \frac { \operatorname* { d e t } V } { \nu ^ { s } \delta _ { 0 } ^ { 2 } } } .\tag{100}
$$

Here again $s = p + q ,$ and $\nu ^ { s } = \operatorname* { d e t } ( \nu I _ { s } )$ is the determinant of the initial ridge matrix. Also $\| \nu \theta \| _ { V ^ { - 1 } } \leq \sqrt { \nu } S _ { \Theta }$ . By the arithmetic-geometric mean inequality and the trace bound, the determinant step can be written explicitly. Let $\gamma _ { 1 } , \ldots , \gamma _ { s } \geq 0$ be the eigenvalues of $G _ { r }$ . Since $V = \nu I _ { s } + G _ { r }$

$$
\begin{array} { r l } & { \displaystyle \frac { \operatorname* { d e t } V } { \nu ^ { s } } = \prod _ { i = 1 } ^ { s } \left( 1 + \frac { \gamma _ { i } } { \nu } \right) } \\ & { \qquad \le \left( \displaystyle \frac { 1 } { s } \sum _ { i = 1 } ^ { s } \left( 1 + \frac { \gamma _ { i } } { \nu } \right) \right) ^ { s } = \left( 1 + \frac { \mathrm { t r } ( G _ { r } ) } { s \nu } \right) ^ { s } } \\ & { \qquad \le \left( 1 + \frac { n _ { H } B _ { \phi } ^ { 2 } } { s \nu } \right) ^ { s } . } \end{array}
$$

The last inequality uses $\begin{array} { r } { \mathrm { t r } ( G _ { r } ) = \sum _ { t \in \mathbb { Z } _ { r } } \| \phi _ { t } \| ^ { 2 } \leq n _ { H } B _ { \phi } ^ { 2 } } \end{array}$ . Taking logarithms and using $n _ { H } \leq H$ yields

$$
\log \frac { \operatorname * { d e t } V } { \nu ^ { s } } \leq s \log \left( 1 + \frac { n _ { H } B _ { \phi } ^ { 2 } } { s \nu } \right) \leq s \log \left( 1 + \frac { H B _ { \phi } ^ { 2 } } { s \nu } \right) .\tag{101}
$$

Equations (94), (99)–(101) imply the unprojected Euclidean error bound equal to one half of (46). A nearest-point projection onto a compact set containing $\theta$ increases Euclidean error by at most a factor two, proving (97). If Θ is closed and convex, projection is nonexpansive and the factor two may be removed.

The gain bound follows from (38). The lower eigenvalue in (98) follows from (95); the upper eigenvalue follows from $\sum { \psi _ { t } \psi _ { t } ^ { \prime } } \preceq ( \sum \| \psi _ { t } \| ^ { 2 } ) I _ { d }$ □

The linear Gram bound (94) is the excitation estimate used in Appendix J to obtain $H =$ $O ( \log ( { ( T + 1 ) / \delta } ) )$ ).

## G THE RLS POTENTIAL ARGUMENT

We establish the potential estimates used in Theorem 19. Consider its unchanged exploration block and exploitation interval, and condition on an $\mathcal { F } _ { r + H }$ history on which the Gram and estimation events hold. We use the constants in (77), set $t _ { 0 } = r + H$ , and let $t _ { 1 } \geq t _ { 0 }$ denote the final control time of an exploitation prefix.

## G.1 Control Error and Prediction Error

Let λ be the true gain on the unchanged segment and let $\bar { \lambda }$ be the exploration reference. For a gate of radius $g > 0$ , the recursive proposal is accepted when $| ( \widehat { \lambda } _ { t } - \overline { { \lambda } } ) ^ { \prime } \psi _ { t } | \leq g \| \psi _ { t } \|$ and the reference proposal is used otherwise. Suppose

$$
\left\| \overline { { \lambda } } - \lambda \right\| \leq \epsilon , \qquad g > \epsilon .
$$

For $a _ { t } = ( \widehat { \lambda } _ { t } - \lambda ) ^ { \prime } \psi _ { t }$ , the unclipped gated action error $\widetilde { e } _ { t } = z _ { t } - \lambda ^ { \prime } \psi _ { t }$ obeys

$$
\tilde { e } _ { t } ^ { 2 } \leq K _ { g } a _ { t } ^ { 2 } , \qquad K _ { g } : = \operatorname* { m a x } \left\{ 1 , \left( \frac { \epsilon } { g - \epsilon } \right) ^ { 2 } \right\} .\tag{102}
$$

If $\psi _ { t } = 0$ , then $a _ { t } = \widetilde { e } _ { t } = 0$ and the claim is immediate. If $\psi _ { t } \neq 0$ , then on the accepted branch $\widetilde { e } _ { t } = a _ { t }$ . On the fallback branch,

$$
\begin{array} { r l } & { | a _ { t } | \geq | ( \widehat \lambda _ { t } - \overline { \lambda } ) ^ { \prime } \psi _ { t } | - | ( \overline { \lambda } - \lambda ) ^ { \prime } \psi _ { t } | } \\ & { ~ > ( g - \epsilon ) \left. \psi _ { t } \right. , } \end{array}
$$

while $| \tilde { e } _ { t } | \leq \epsilon \| \psi _ { t } \|$ . Since $g - \epsilon > 0$ , division gives

$$
| \widetilde e _ { t } | \leq \frac { \epsilon } { g - \epsilon } | a _ { t } | .
$$

Squaring this inequality and taking the larger constant over the two branches proves (102). If the ideal action lies in the clipping interval, nonexpansiveness preserves (102). In the algorithm, $g = g _ { H }$ and $\epsilon = E _ { \lambda } = g _ { H } / 4$ . Therefore $\epsilon / ( g - \epsilon ) = 1 / 3$ and $K _ { g } = 1$

## G.2 Exact Regression and Potential Identity

On the segment,

$$
y _ { t + 1 } = b _ { \star } ( u _ { t } - \lambda ^ { \prime } \psi _ { t } ) + w _ { t + 1 } .\tag{103}
$$

Let $\gamma = b _ { \star } / \widehat { b } , \alpha = 1 - \gamma$ , and $e _ { t } = u _ { t } - \lambda ^ { \prime } \psi _ { t }$ . The pseudo-response is exactly

$$
\begin{array} { r l } & { x _ { t + 1 } ^ { \lambda } = u _ { t } - y _ { t + 1 } / \widehat { b } } \\ & { \qquad = \lambda ^ { \prime } \psi _ { t } + \alpha e _ { t } + \eta _ { t + 1 } , } \end{array}
$$

$$
\eta _ { t + 1 } = - w _ { t + 1 } / \widehat { b } .
$$

Because $\widehat { \theta } _ { r } \in \Theta$ by projection and hence $\widehat { b } \geq b _ { \mathrm { m i n } }$ by (37), $| \eta _ { t + 1 } | \le B _ { \eta } = B _ { w } / b _ { \mathrm { m i n } }$ and Hoefding’s lemma gives

$$
\mathbb { E } [ \exp ( \xi \eta _ { t + 1 } ) \mid \mathcal { F } _ { t } ] \leq \exp ( \xi ^ { 2 } B _ { \eta } ^ { 2 } / 2 ) , \qquad \xi \in \mathbb { R } .\tag{104}
$$

Set $d _ { t } = \widehat { \lambda } _ { t } - \lambda , a _ { t } = d _ { t } ^ { \prime } \psi _ { t } , Q _ { t } = d _ { t } ^ { \prime } V _ { t } d _ { t }$ , and

$$
\ell _ { t } : = \psi _ { t } ^ { \prime } V _ { t + 1 } ^ { - 1 } \psi _ { t } .
$$

The update (60) becomes

$$
d _ { t + 1 } = d _ { t } + V _ { t + 1 } ^ { - 1 } \psi _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } - a _ { t } ) .\tag{105}
$$

Using $V _ { t + 1 } = V _ { t } + \psi _ { t } \psi _ { t } ^ { \prime }$ in (105) and expanding gives

$$
Q _ { t + 1 } - Q _ { t } = - a _ { t } ^ { 2 } + 2 a _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } ) + \ell _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } - a _ { t } ) ^ { 2 } .\tag{106}
$$

On the exploration event,

$$
| \alpha | = \frac { | \widehat { b } - b _ { \star } | } { \widehat { b } } \leq \frac { E _ { \theta } } { b _ { \mathrm { m i n } } } = \overline { { \alpha } } .
$$

Also $V _ { t + 1 } \succeq V _ { r + H } \succeq \underline { v } _ { H } I$ , so

$$
0 \leq \ell _ { t } \leq \frac { \| \psi _ { t } \| ^ { 2 } } { \underline { { v } } _ { H } } \leq \bar { \ell } .
$$

By $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$ and $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$

$$
\begin{array} { c } { 2 \alpha a _ { t } e _ { t } \leq 2 \overline { { \alpha } } a _ { t } ^ { 2 } , } \\ { \ell _ { t } ( \alpha e _ { t } + \eta _ { t + 1 } - a _ { t } ) ^ { 2 } \leq 2 \ell _ { t } ( 1 + \overline { { \alpha } } ) ^ { 2 } a _ { t } ^ { 2 } + 2 \ell _ { t } \eta _ { t + 1 } ^ { 2 } . } \end{array}
$$

Substitution into (106) proves (80).

## G.3 Time-Uniform Noise Bound

For any fixed $\xi > 0 ,$ , (104) and predictability of $a _ { t }$ imply that the prefix-indexed process

$$
\exp \left\{ \xi \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } a _ { t } \eta _ { t + 1 } - \frac { \xi ^ { 2 } B _ { \eta } ^ { 2 } } { 2 } \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } a _ { t } ^ { 2 } \right\}
$$

is a nonnegative supermartingale starting at one, conditionally on $\mathcal { F } _ { t _ { 0 } }$ . Its one-step multiplicative increment satisfies

$$
\mathbb { E } \left[ \exp \left\{ \xi a _ { t } \eta _ { t + 1 } - \frac { \xi ^ { 2 } B _ { \eta } ^ { 2 } a _ { t } ^ { 2 } } { 2 } \right\} \Bigg | \mathcal { F } _ { t } \right] \leq 1 ,
$$

because $a _ { t }$ is $\mathcal { F } _ { t }$ -measurable and (104) applies with argument $\xi a _ { t }$

Ville’s inequality (Ville, 1939; Howard et al., 2020) says that, with probability at least $1 - \delta _ { 0 }$ this process never exceeds $1 / \delta _ { 0 }$ . Taking logarithms on that event and dividing by the fixed $\xi > 0$ gives, simultaneously for every exploitation prefix,

$$
\sum _ { t } a _ { t } \eta _ { t + 1 } \leq \frac { \xi B _ { \eta } ^ { 2 } } { 2 } \sum _ { t } a _ { t } ^ { 2 } + \frac { 1 } { \xi } \log \frac { 1 } { \delta _ { 0 } } .
$$

Choosing the deterministic value $\xi = c _ { 0 } / ( 2 B _ { \eta } ^ { 2 } )$ gives

$$
\sum _ { t } a _ { t } \eta _ { t + 1 } \leq \frac { c _ { 0 } } { 4 } \sum _ { t } a _ { t } ^ { 2 } + \frac { 2 B _ { \eta } ^ { 2 } } { c _ { 0 } } \log ( 1 / \delta _ { 0 } ) .\tag{107}
$$

## G.4 Elliptical Potential

For $x _ { t } = \psi _ { t } ^ { \prime } V _ { t } ^ { - 1 } \psi _ { t }$ , the Sherman–Morrison identity $V _ { t + 1 } ^ { - 1 } = V _ { t } ^ { - 1 } - V _ { t } ^ { - 1 } \psi _ { t } \psi _ { t } ^ { \prime } V _ { t } ^ { - 1 } / ( 1 + x _ { t } )$ and the matrix determinant lemma give

$$
\ell _ { t } = \frac { x _ { t } } { 1 + x _ { t } } , \qquad \mathrm { ~ } \qquad \mathrm { ~ } \frac { \operatorname* { d e t } V _ { t + 1 } } { \operatorname* { d e t } V _ { t } } = 1 + x _ { t } .
$$

Since $x / ( 1 + x ) \leq \log ( 1 + x )$ for $x \geq 0$

$$
\sum _ { t = t _ { 0 } } ^ { t _ { 1 } } \ell _ { t } \leq \log { \frac { \operatorname* { d e t } V _ { t _ { 1 } + 1 } } { \operatorname* { d e t } V _ { t _ { 0 } } } } .
$$

The determinants are positive since $V _ { t _ { 0 } } \succeq { v _ { H } } { I } \succ 0$ . Set

$$
S _ { n } : = \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } \psi _ { t } \psi _ { t } ^ { \prime } , \qquad A _ { n } : = V _ { t _ { 0 } } ^ { - 1 / 2 } S _ { n } V _ { t _ { 0 } } ^ { - 1 / 2 } , \qquad n : = t _ { 1 } - t _ { 0 } + 1 .
$$

Then $A _ { n } \succeq 0$ and

$$
\begin{array} { l } { \displaystyle \log \frac { \operatorname* { d e t } V _ { t _ { 1 } + 1 } } { \operatorname* { d e t } V _ { t _ { 0 } } } = \log \operatorname* { d e t } ( I + A _ { n } ) } \\ { \displaystyle \qquad \leq d \log \biggl ( 1 + \frac { \mathrm { t r } ( A _ { n } ) } { d } \biggr ) } \\ { \displaystyle \qquad \leq d \log \biggl ( 1 + \frac { n B _ { \psi } ^ { 2 } } { d \underline { { v } } _ { H } } \biggr ) = \Gamma ( n ) . } \end{array}\tag{108}
$$

The first inequality follows by applying Jensen’s inequality to log(1 + x) over the nonnegative eigenvalues of $A _ { n }$ . For the second,

$$
\mathrm { t r } ( A _ { n } ) = \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } \psi _ { t } ^ { \prime } V _ { t _ { 0 } } ^ { - 1 } \psi _ { t } \leq \frac { 1 } { \underline { { v } } _ { H } } \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } \| \psi _ { t } \| ^ { 2 } \leq \frac { n B _ { \psi } ^ { 2 } } { \underline { { v } } _ { H } } .
$$

For the standard elliptical-potential argument, see Abbasi-Yadkori et al. (2011, Lemma 11).

## G.5 Completion of the Segment Bound

Sum (80). Since $Q _ { t _ { 1 } + 1 } \geq 0 , \eta _ { t + 1 } ^ { 2 } \leq B _ { \eta } ^ { 2 }$ , and (107)–(108) hold,

$$
\begin{array} { r l r } {  { c _ { 0 } \sum _ { t } a _ { t } ^ { 2 } \le Q _ { t _ { 0 } } + 2 \sum _ { t } a _ { t } \eta _ { t + 1 } + 2 B _ { \eta } ^ { 2 } \Gamma ( n ) } } \\ & { } & \\ & { } & { \le Q _ { t _ { 0 } } + \frac { c _ { 0 } } { 2 } \sum _ { t } a _ { t } ^ { 2 } + \frac { 4 B _ { \eta } ^ { 2 } } { c _ { 0 } } \log ( 1 / \delta _ { 0 } ) + 2 B _ { \eta } ^ { 2 } \Gamma ( n ) . } \end{array}
$$

Rearranging yields (83). At initialization,

$$
\begin{array} { r l } & { Q _ { t _ { 0 } } = ( \overline { { \lambda } } - \lambda ) ^ { \prime } V _ { t _ { 0 } } ( \overline { { \lambda } } - \lambda ) } \\ & { \qquad \leq E _ { \lambda } ^ { 2 } \lambda _ { \operatorname* { m a x } } ( V _ { t _ { 0 } } ) \leq E _ { \lambda } ^ { 2 } ( \nu + n _ { H } B _ { \psi } ^ { 2 } ) = \overline { { Q } } _ { H } . } \end{array}
$$

Finally, (103) and the gate charging inequality give

$$
\sum _ { t } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } = b _ { \star } ^ { 2 } \sum _ { t } e _ { t } ^ { 2 } \leq \bar { b } ^ { 2 } \sum _ { t } a _ { t } ^ { 2 } ,
$$

which proves Theorem 19.

## H DETAILED DETECTOR PROOFS

This appendix supplies the conditional power estimates, concentration arguments, and calendar bounds used in Appendix C. The assumptions and tuning of Appendices A and B remain in force.

## H.1 Finite-Horizon Power of the Auxiliary Process

Fix a change at control time τ and write $A = A _ { \Delta }$ . For $s - h \ge \tau$ , iterating (68) for h steps gives

$$
\psi _ { s } ^ { \mathrm { a u x } } = A ^ { h } \psi _ { s - h } ^ { \mathrm { a u x } } + \sum _ { j = 0 } ^ { h - 1 } A ^ { j } e _ { 1 } w _ { s - j } .
$$

Taking the first coordinate and squaring, all cross terms involving distinct disturbances have conditional expectation zero. Thus

$$
\mathbb { E } [ ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } \mid \mathcal { F } _ { s - h } ] = ( e _ { 1 } ^ { \prime } A ^ { h } \psi _ { s - h } ^ { \mathrm { a u x } } ) ^ { 2 } + \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { h - 1 } ( e _ { 1 } ^ { \prime } A ^ { j } e _ { 1 } ) ^ { 2 } .\tag{109}
$$

The stationary power is

$$
J _ { y } ( \lambda ^ { - } ; \theta ^ { + } ) = \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { \infty } ( e _ { 1 } ^ { \prime } A ^ { j } e _ { 1 } ) ^ { 2 } .
$$

Since the first term of (109) is nonnegative,

$$
\begin{array} { r l } { \mathbb { E } [ ( y _ { s } ^ { \mathrm { a u x } } ) ^ { 2 } \mid \mathcal { F } _ { s - h } ] \ge J _ { y } ( \lambda ^ { - } ; \theta ^ { + } ) - \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = h } ^ { \infty } \left\| A ^ { j } \right\| ^ { 2 } } & { } \\ { \ge J _ { y } ( \lambda ^ { - } ; \theta ^ { + } ) - \displaystyle \frac { \sigma _ { w } ^ { 2 } C _ { \Delta } ^ { 2 } \rho _ { \Delta } ^ { 2 h } } { 1 - \rho _ { \Delta } ^ { 2 } } . } \end{array}
$$

Equations (45) and (53) yield (72) for every initial state at time $s - h$

The calculation also holds when s is a finite lag-h predictable random time. Indeed, $\{ s =$ $t \} \in \mathcal { F } _ { t - h } ;$ multiplying the deterministic-time identity by $\mathbf { 1 } \{ s = t \}$ and summing over t gives (109) with s and the stopped sigma-field $\mathcal { F } _ { s - h }$

## H.2 Lag-Predictable Sampling

Let $h \leq s _ { 1 } < \cdots < s _ { m }$ be finite lag-h predictable output times with $s _ { k + 1 } - s _ { k } \geq h$ . Define

$$
\mathcal { H } _ { k - 1 } : = \mathcal { F } _ { s _ { k } - h } , \qquad k = 1 , \ldots , m , \qquad \mathcal { H } _ { m } : = \mathcal { F } _ { s _ { m } } .
$$

Lag predictability makes these stopped sigma-fields well defined. The spacing condition $\mathrm { g i } \mathfrak { r }$ ves $\mathcal { F } _ { s _ { k } } \subseteq \mathcal { H } _ { k }$ for $k < m$ , and the inclusion also holds for $k = m$ by definition. Hence $y _ { s _ { k } } ^ { 2 } - \mathbb { E } [ y _ { s _ { k } } ^ { 2 } \ |$ $\mathcal { F } _ { s _ { k } - h } ]$ is H<sub>k</sub>-measurable and has conditional mean zero given $\mathcal { H } _ { k - 1 }$ . This is the martingale structure used in Lemma 14; it need not hold for consecutive outputs centered at lag h, or for samples selected after observing their values.

## H.3 Null Windows

Fix a potential restart r with a full uncontaminated exploration block and a potential test endpoint e. In the alarm-suppressed continuation from r, let $S _ { r , e }$ be the corresponding block of m calendar samples. The block is null for r when each sampled output is generated under the reference parameter. If that parameter remains active throughout $[ r , \tau )$ , a block ending at output time $\tau$ is null, since $y _ { \tau }$ is generated at control time $\tau - 1$

Condition on an $\mathcal { F } _ { r + H }$ history on which $\mathcal G _ { r } ^ { \mathrm { e s t } }$ holds. The calendar fixes the block before its outputs are observed, and (64) applies to every stored output of a null block. Each $w _ { s } ^ { 2 } - \sigma _ { w } ^ { 2 }$ is centered and lies in an interval of length $B _ { w } ^ { 2 }$ . The Hoefding exponential process (Hoefding, 1963; Howard et al., 2020) therefore gives, for $x > 0$ ，

$$
\mathbb { P } \left( \sum _ { s \in S _ { r , e } } ( w _ { s } ^ { 2 } - \sigma _ { w } ^ { 2 } ) \geq m x \middle | \mathcal { F } _ { r + H } \right) \leq \exp \left( - \frac { 2 m x ^ { 2 } } { B _ { w } ^ { 4 } } \right) .\tag{110}
$$

For the cross term, $v _ { s }$ is predictable and $| w _ { s } v _ { s } | \le B _ { w } \varepsilon _ { y }$ . If $\varepsilon _ { y } = 0$ , the cross term is identically zero. Otherwise, the corresponding Azuma–Hoefding exponential process (Azuma, 1967; Hoefding, 1963) gives

$$
\mathbb { P } \left( \sum _ { s \in S _ { r , e } } w _ { s } v _ { s } \geq m x \ \middle | \ \mathcal { F } _ { r + H } \right) \leq \exp \left( - \frac { m x ^ { 2 } } { 2 B _ { w } ^ { 2 } \varepsilon _ { y } ^ { 2 } } \right) .\tag{111}
$$

Taking $x = \Delta / 8$ in (110) and $x = \Delta / 1 6$ in (111) gives (66) and (67). The deterministic term in (65) is at most $\varepsilon _ { y } ^ { 2 } \leq \Delta / 8$ . Outside the two deviation events, the excess empirical power is therefore at most $3 \Delta / 8$ , below the threshold $\Delta / 2$

If the actual test at e is reached, the alarm-suppressed continuation agrees with the implemented trajectory through that test. The conditional bound thus controls the joint event that the reference succeeds, the test is reached, its block is null for that reference, and the threshold is crossed. There are at most $T ^ { 2 }$ potential restart–endpoint pairs; applying the $\delta _ { d } / ( 2 T ^ { 2 } )$ oneblock bound separately to each pair and taking a union bound gives a horizon-wide false-alarm contribution of at most $\delta _ { d } / 2$ . Overlapping windows require no independence, and the argument does not condition on survival to a test.

## H.4 Trajectory Coupling

Let $\delta _ { t } = \psi _ { t } - \psi _ { t } ^ { \mathrm { a u x } }$ and use the input-channel bound $G _ { u }$ from (42). On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ , for control times after the change at $\tau$ and before the next alarm or true change, Lemma 13 gives

$$
u _ { t } = ( \lambda ^ { - } ) ^ { \prime } \psi _ { t } + e _ { t } , \qquad \left\| g _ { u } ( \theta ^ { + } ) e _ { t } \right\| \le G _ { u } \varepsilon _ { u } .
$$

Consequently,

$$
\delta _ { t + 1 } = A \delta _ { t } + g _ { u } ( \theta ^ { + } ) e _ { t } , \qquad \delta _ { \tau } = 0 .
$$

Unrolling and using (44),

$$
\left\| \delta _ { t } \right\| \leq \sum _ { j = 0 } ^ { t - \tau - 1 } \left\| A ^ { j } \right\| G _ { u } \varepsilon _ { u } \leq \frac { C _ { \Delta } G _ { u } \varepsilon _ { u } } { 1 - \rho _ { \Delta } } = R _ { \Delta } ( H ) .
$$

The gate bound uses feasibility of the old-gain action $( \lambda ^ { - } ) ^ { \prime } \psi _ { t }$ at the actual state. The auxiliary action is unclipped and need not be feasible. The comparison holds up to the first alarm or next true change, while the actual plant parameter remains the fixed $\theta ^ { + }$ used in A.

## H.5 Calendar and Delay

Consider a change at control time τ under the hypotheses of Lemma 17: the cycle is still in exploitation after observing $y _ { \tau } , r { + } H \leq \tau$ , and $\tau { + } D \leq T$ . In its alarm-suppressed continuation,

stored output times form an arithmetic grid with gap h. Let $s _ { 1 }$ be the first grid output with $s _ { 1 } - h \geq \tau$ . The distance from $\tau + h$ to the next grid point is less than $h ,$ so

$$
s _ { 1 } - \tau < 2 h .
$$

The mth grid output is $s _ { m } = s _ { 1 } + ( m - 1 ) h$ , hence

$$
s _ { m } - \tau < ( m + 1 ) h = D .
$$

Every interval $[ s _ { k } - h , s _ { k } )$ lies after the change, so the auxiliary outputs at these times satisfy (72). The post-change parameter remains active throughout the block by the hypotheses of Lemma 17. If any earlier block alarms, the change has already been detected. Otherwise, the actual and alarm-suppressed trajectories agree through the complete post-change test at $s _ { m }$ Since times are integer valued, $s _ { m } \leq \tau + D - 1$ ; the test is therefore available within the horizon.

For the induction in Theorem 18, the dwell condition places the first change after the initial exploration. If change $c < C$ is detected by $\tau ^ { \left( c \right) } + D$ , the next exploration ends by $\tau ^ { ( c ) } + D + H \leq \tau ^ { ( c + 1 ) }$ . Thus each subsequent change is preceded by a full uncontaminated reference block on the successful detection history.

## I GLOBAL EVENT AND REGRET DECOMPOSITION

We combine the restart and detector guarantees on a common event and partition the control times to prove Theorem 20.

## I.1 Global Success Event

Assume $T > H ;$ the case $T \leq H$ is handled deterministically in the proof of Theorem 20. Record each failure at its first adapted decision time: at the end of a reached full, unchanged exploration block if its Gram or self-normalized event fails; at the first same-parameter exploitation prefix violating its time-uniform RLS event; at a false detector crossing; or at the deadline of a missed change. Ties at an exploration endpoint are ordered as Gram, self-normalized, and then RLS. Since the true parameter sequence is deterministic in the analysis, these decisions are measurable with respect to the history at their decision times. Thus

$$
\mathcal E { \leq } _ { \leq t } ^ { \mathrm { g o o d } } : = \{ \mathrm { n o ~ f a i l u r e ~ h a s ~ b e e n ~ r e c o r d e d ~ b y ~ t i m e ~ } t \} \in \mathcal F _ { t } .
$$

For each potential restart $r \in \{ 0 , \ldots , T \}$ , define

$$
\mathcal { C } _ { r } : = \{ r \ \mathrm { i s ~ r e a c h e d } \} \cap \mathcal { E } _ { \leq r } ^ { \mathrm { g o o d } } \cap \{ [ r , r + H ) \ \mathrm { i s ~ f u l l ~ a n d ~ h a s ~ o n e ~ p a r a m e t e r } \} .
$$

The first two events are $\mathcal { F } _ { r }$ -measurable, and the last is fixed by the change sequence, so $\mathcal { C } _ { r } \in \mathcal { F } _ { r }$ Let $\mathcal { E } _ { r } ^ { G }$ and $\mathcal { E } _ { r } ^ { \theta }$ denote the Gram and self-normalized events. Let $\mathcal { E } _ { r } ^ { R }$ be the time-uniform RLS event on the subsequent exploitation prefix with the same parameter as in $[ r , r + H )$ , ending at the first of the next true change, the next alarm output time, and the horizon. Set $\mathcal E _ { r } ^ { R } = \Omega$ when this prefix is empty. The controller-failure events are

$$
\begin{array} { r l } & { \mathcal { B } _ { r } ^ { G } : = \mathcal { C } _ { r } \cap ( \mathcal { E } _ { r } ^ { G } ) ^ { c } , } \\ & { \mathcal { B } _ { r } ^ { \theta } : = \mathcal { C } _ { r } \cap \mathcal { E } _ { r } ^ { G } \cap ( \mathcal { E } _ { r } ^ { \theta } ) ^ { c } , } \\ & { \mathcal { B } _ { r } ^ { R } : = \mathcal { C } _ { r } \cap \mathcal { E } _ { r } ^ { G } \cap \mathcal { E } _ { r } ^ { \theta } \cap ( \mathcal { E } _ { r } ^ { R } ) ^ { c } . } \end{array}
$$

Conditional on ${ \mathcal { F } } _ { r } ,$ Propositions 21 and 22 bound the Gram and self-normalized failure probabilities by $\delta _ { 0 }$ each. On their joint success event, Theorem 19 bounds the conditional RLS failure

probability given $\mathcal { F } _ { r + H }$ by $\delta _ { 0 }$ . Multiplying by the corresponding history indicators and applying the tower property gives

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { B } _ { r } ^ { G } ) \leq \delta _ { 0 } , \qquad \qquad \mathbb { P } ( \mathcal { B } _ { r } ^ { \theta } ) \leq \delta _ { 0 } , \qquad \qquad \mathbb { P } ( \mathcal { B } _ { r } ^ { R } ) \leq \delta _ { 0 } . } \end{array}
$$

Consequently, for $\begin{array} { r } { B _ { \mathrm { c t r l } } : = \bigcup _ { r = 0 } ^ { T } ( \mathcal { B } _ { r } ^ { G } \cup \mathcal { B } _ { r } ^ { \theta } \cup \mathcal { B } _ { r } ^ { R } ) } \end{array}$ 2

$$
\mathbb { P } ( \mathcal { B } _ { \mathrm { c t r l } } ) \le 3 ( T + 1 ) \delta _ { 0 } = \frac { 3 \delta } { 3 2 } .\tag{112}
$$

On $B _ { \mathrm { c t r l } } ^ { c }$ , every reached full, unchanged reference whose restart precedes the first detector failure is accurate: otherwise the first controller failure would belong to one of the events above. Hence $B _ { \mathrm { c t r l } } ^ { c } \cap { \mathcal { E } } _ { \mathrm { d e t } } ^ { c } \subseteq { \mathcal { E } } _ { \mathrm { r e f } } \cap { \mathcal { E } } _ { \mathrm { d e t } } ^ { c }$ , and Theorem 18 gives

$$
\mathbb { P } ( B _ { \mathrm { c t r l } } ^ { c } \cap \mathcal { E } _ { \mathrm { d e t } } ^ { c } ) \leq \delta _ { d } .
$$

This bounds a joint event and does not condition an early detector test on the success of a later estimator.

Define $\mathcal { E } _ { \mathrm { g o o d } } : = \mathcal { B } _ { \mathrm { c t r l } } ^ { c } \cap \mathcal { E } _ { \mathrm { d e t } }$ . Then

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { g o o d } } ) \geq 1 - \frac { 3 \delta } { 3 2 } - \frac { \delta } { 2 } = 1 - \frac { 1 9 \delta } { 3 2 } > 1 - \delta .
$$

On this event, the initial reference is valid, every change is detected, and the dwell condition ensures that each nonterminal post-alarm exploration block is full and uncontaminated.

## I.2 Regret Decomposition

On ${ \mathcal E } _ { \mathrm { g o o d } }$ , let ${ \widehat { \tau } } ^ { \left( c \right) }$ be the alarm corresponding to change c. Its delay interval is

$$
\mathcal D _ { c } : = \{ \tau ^ { ( c ) } , \ldots , \widehat { \tau } ^ { ( c ) } - 1 \} , \qquad | \mathcal D _ { c } | \leq D , \qquad c = 1 , \ldots , C .
$$

The initial and post-alarm exploration intervals are

$$
\begin{array} { r l } & { \mathcal { X } _ { 0 } : = \{ 0 , \dots , \operatorname* { m i n } \{ H , T \} - 1 \} , } \\ & { \mathcal { X } _ { c } : = \{ \widehat { \tau } ^ { ( c ) } , \dots , \operatorname* { m i n } \{ \widehat { \tau } ^ { ( c ) } + H , T \} - 1 \} , \qquad c = 1 , \dots , C . } \end{array}
$$

An interval is empty when its lower endpoint exceeds its upper endpoint. The dwell condition makes the delay and exploration intervals disjoint, and $| \mathcal { X } _ { c } | \le H$ . Their complement consists of at most $C + 1$ unchanged exploitation prefixes ${ \mathcal P _ { c } } ,$ one in each true segment, with empty prefixes allowed. Therefore

$$
\{ 0 , \dots , T - 1 \} = \left( \bigcup _ { c = 1 } ^ { C } \mathcal { D } _ { c } \right) \dot { \cup } \left( \bigcup _ { c = 0 } ^ { C } \mathcal { X } _ { c } \right) \dot { \cup } \left( \bigcup _ { c = 0 } ^ { C } \mathcal { P } _ { c } \right) .
$$

For $t \in \mathcal { D } _ { c } .$ , the bound $| y _ { t + 1 } - w _ { t + 1 } | \leq B _ { Y } + B _ { w }$ gives cost at most $K _ { D }$ . For $t \in \mathcal { X } _ { c } , ( 3 6 )$ gives cost at most $K _ { X }$ . Set $n _ { c } : = | \mathcal { P } _ { c } |$ . Every nonempty $\mathcal { P } _ { c }$ follows a full, unchanged exploration block, so Theorem 19 bounds its regret by $R _ { E } ( n _ { c } , \delta _ { 0 } )$ , uniformly over the resulting prefix length. An empty prefix has zero regret, also bounded by $R _ { E } ( 0 , \delta _ { 0 } )$ . Consequently,

$$
\begin{array} { r l r } {  { \mathcal { R } _ { T } \leq \sum _ { c = 1 } ^ { C } K _ { D } | \mathcal { D } _ { c } | + \sum _ { c = 0 } ^ { C } K _ { X } | \mathcal { X } _ { c } | + \sum _ { c = 0 } ^ { C } R _ { E } ( n _ { c } , \delta _ { 0 } ) } } \\ & { } & { \leq C K _ { D } D + ( C + 1 ) K _ { X } H + ( C + 1 ) R _ { E } ( T , \delta _ { 0 } ) , } \end{array}
$$

which is (86).

## J TUNING FEASIBILITY AND ORDERS

We verify that the exploration and detector requirements in Appendix B admit the logarithmic choices used in Theorem 20. The asymptotic statements keep the model orders and the model, noise, initialization, exploration, ridge, stability, clipping, and gap bounds fixed.

## J.1 Exploration Length

Equation (49) gives

$$
\varepsilon _ { y } = 5 \bar { b } L _ { \lambda } B _ { \psi } E _ { \theta } ( H , \delta _ { 0 } ) .
$$

It is suficient to require

$$
E _ { \theta } ( H , \delta _ { 0 } ) \leq \epsilon _ { \star } ,
$$

where

$$
\epsilon _ { \star } : = \operatorname* { m i n } \bigg \{ 1 , \frac { b _ { \operatorname* { m i n } } } { 1 6 } , \frac { \sqrt { \Delta / 8 } } { 5 \bar { b } L _ { \lambda } B _ { \psi } } , \frac { \Delta ( 1 - \rho _ { \Delta } ) } { 8 0 B _ { o } C _ { \Delta } G _ { u } L _ { \lambda } B _ { \psi } } , \frac { B _ { o } } { 5 \bar { b } L _ { \lambda } B _ { \psi } } \bigg \} .
$$

The four accuracy terms enforce the leading-coeficient bound, the null-bias margin, the coupling margin, and $\varepsilon _ { y } \leq B _ { o } .$ respectively. The cap at one ensures $\log ( 1 / \epsilon _ { \star } ) \geq 0$ . All denominators are positive under the standing assumptions, including $L _ { \lambda } > 0$ from Assumption 7. The deterministic bound (46) satisfies

$$
E _ { \theta } ( H , \delta _ { 0 } ) \leq \frac { K _ { 0 } \{ 1 + \sqrt { \log ( 1 / \delta _ { 0 } ) + s \log ( 1 + K _ { 1 } H ) } \} } { \sqrt { H - r _ { 0 } } }\tag{113}
$$

for constants $K _ { 0 } , K _ { 1 }$ independent of $H , T , \delta$ . Thus

$$
\begin{array} { r l } { \mathcal { H } : = \left\{ H \in { \mathbb { N } } : H > r _ { 0 } , } & { N _ { H } \geq \displaystyle \frac { 1 0 B _ { \phi } ^ { 4 } } { \kappa _ { \mathrm { i n n } } ^ { 2 } } \log \displaystyle \frac { s } { \delta _ { 0 } } , \quad E _ { \theta } ( H , \delta _ { 0 } ) \leq \epsilon _ { \star } , \right. } \\ & { \displaystyle \left. \frac { B _ { \psi } ^ { 2 } } { \nu + \kappa _ { G } n _ { H } } \leq \displaystyle \frac { 1 } { 1 6 } \right\} } \end{array}
$$

is nonempty: as $H  \infty , \ N _ { H }$ grows linearly, while $E _ { \theta } ( H , \delta _ { 0 } )$ and the leverage bound tend to zero. Every member satisfies (93) and (51)–(52). The smallest feasible integer selected in Appendix B is therefore at most min H.

For $a , b \geq 1$ , the condition $x \geq 2 a \log ( 2 a b )$ implies

$$
x \geq a \log ( b x ) .\tag{114}
$$

Indeed, setting $z = x / ( 2 a )$ gives $\log ( b x ) = \log ( 2 a b ) +$ log $z \le \log ( 2 a b ) + z \le 2 z = x / a$ . Squaring (113), restricting to $H \geq 2 r _ { 0 }$ , and applying (114) shows that the accuracy condition is satisfied with

$$
H = O \Bigl ( \epsilon _ { \star } ^ { - 2 } \{ \log ( 1 / \delta _ { 0 } ) + s \log ( 1 / \epsilon _ { \star } ) + 1 \} \Bigr ) .
$$

The Gram-size and leverage conditions require only $O ( \log ( 1 / \delta _ { 0 } ) + 1 )$ additional lower bounds on H and are absorbed in this order. Since $\delta _ { 0 } = \delta / [ 3 2 ( T + 1 ) ]$ ，

$$
H = O ( \log ( ( T + 1 ) / \delta ) ) .\tag{115}
$$

## J.2 Detector Spacing and Delay

Since $\rho _ { \Delta } \in ( 0 , 1 )$ , the spacing in (53) is finite. Writing log ${ \bf \Phi } _ { + } ( x ) : = \operatorname* { m a x } \{ 0 , \log x \}$ , its explicit form is

$$
h = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { \log _ { + } \left( \frac { 8 \sigma _ { w } ^ { 2 } C _ { \Delta } ^ { 2 } } { ( 1 - \rho _ { \Delta } ^ { 2 } ) \Delta } \right) } { 2 \log ( 1 / \rho _ { \Delta } ) } \right\rceil \right\} .
$$

Thus h is independent of T. Equation (54) gives

$$
m = O \left( \frac { B _ { o } ^ { 4 } } { \Delta ^ { 2 } } \log \frac { T + 1 } { \delta } \right) , \qquad D = O \left( \frac { h B _ { o } ^ { 4 } } { \Delta ^ { 2 } } \log \frac { T + 1 } { \delta } \right) .\tag{116}
$$

## J.3 Regret Constants

Under (51),

$$
c _ { 0 } \geq 1 - \frac { 1 } { 8 } - \frac { 1 } { 8 } \left( \frac { 1 7 } { 1 6 } \right) ^ { 2 } > 0 . 7 3 .
$$

Equation (46) also yields

$$
\begin{array} { l } { { \overline { { { Q } } } _ { H } = E _ { \lambda } ^ { 2 } ( \nu + n _ { H } B _ { \psi } ^ { 2 } ) = O ( \log ( 1 / \delta _ { 0 } ) + s \log ( 1 + H ) ) } } \\ { { \qquad = O ( \log ( ( T + 1 ) / \delta ) ) , } } \end{array}
$$

while

$$
\Gamma ( T ) = d \log \left( 1 + \frac { T B _ { \psi } ^ { 2 } } { d \underline { { { v } } } _ { H } } \right) = O ( d \log T ) .
$$

Therefore $R _ { E } ( T , \delta _ { 0 } ) = O ( \log ( ( T + 1 ) / \delta ) )$ . Substituting this bound together with (115) and (116) into (86) proves (87).

## K RELAXATIONS OF CLIPPING AND MISMATCH STA-BILITY

This appendix establishes the feasibility and detection results used in Sections 6 and 5. The extensions use either eventual nonclipping or own-regime feasibility with a boundary alarm.

## K.1 Closed Clipping Feasibility

Let $P ( x ) = \mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( x )$ . Projection onto a closed interval is nonexpansive. If $x \in [ - B _ { u } , B _ { u } ]$ 2 including $x = \pm B _ { u } .$ , then

$$
| P ( z ) - x | = | P ( z ) - P ( x ) | \leq | z - x | .\tag{117}
$$

Equation (117) is the only consequence of the clipping condition used in Lemma 13. That lemma and every later detector, RLS, and global-regret argument therefore hold under the closed feasibility condition in Assumption 10; no positive margin enters a proof constant. For the stable-mismatch theorem, Assumption 10 can be replaced by feasibility of the true reference gain along every unchanged exploitation prefix and every complete alarm-suppressed continuation used in the detector analysis.

## K.2 Eventual Inactivity of Clipping

A quantitative steady-state slack condition yields eventual inactivity of clipping and a burn-in time uniform over restarts. This gives the variant of PIECE-CD in Theorem 6. Define the forced-response output bound and its corresponding regressor bound by

$$
\begin{array} { c } { { Y _ { \infty } ( B _ { u } ) : = \displaystyle \frac { C _ { y } } { 1 - \rho _ { y } } ( B _ { w } + B _ { b } B _ { u } ) , } } \\ { { { } } } \\ { { P _ { \infty } ( B _ { u } ) : = \displaystyle \left\{ Y _ { \infty } ( B _ { u } ) ^ { 2 } + ( q - 1 ) B _ { u } ^ { 2 } \right\} ^ { 1 / 2 } , } } \\ { { \mu _ { \mathrm { n c } } : = B _ { u } - \varepsilon _ { u } - \Lambda P _ { \infty } ( B _ { u } ) . } } \end{array}\tag{118}
$$

Here $\begin{array} { r } { \varepsilon _ { u } = \frac { 5 } { 4 } g _ { H } B _ { \psi } } \end{array}$ , defined in (49), bounds the raw-proposal error on $\mathcal G _ { r } ^ { \mathrm { e s t } }$ . Suppose $\mu _ { \mathrm { n c } } > 0$ With the convention $\tau _ { \mathrm { n c } } = 0$ when $\Lambda C _ { y } B _ { Y } = 0$ , put otherwise

$$
\tau _ { \mathrm { n c } } : = \operatorname* { m a x } \left\{ 0 , \left\lceil \frac { \log ( \Lambda C _ { y } B _ { Y } / \mu _ { \mathrm { n c } } ) } { \log ( 1 / \rho _ { y } ) } - H \right\rceil \right\} .\tag{119}
$$

Proposition 23 (Uniform eventual inactivity of clipping). Fix a reached restart r whose full exploration block has one parameter $\theta ^ { - }$ , and suppose $\mathcal G _ { r } ^ { \mathrm { e s t } }$ holds. At every exploitation time before the next restart satisfying

$$
t \geq r + H + \tau _ { \mathrm { n c } } ,
$$

the raw gated proposal belongs to the clipping interval:

$$
| z _ { t } | \le B _ { u } , \qquad u _ { t } = z _ { t } .\tag{120}
$$

The conclusion remains valid after a plant change and does not require the old-gain/new-plant mismatch matrix to be Schur.

Proof. The implemented input is clipped at every time, so $| u _ { t } | \le B _ { u }$ regardless of whether the conclusion has already become true. Starting the state recursion (88) at $r ,$ uniform switching stability therefore gives, for every $t \geq r$ and for any admissible changes after $r ,$

$$
\begin{array} { r } { \| Y _ { t } \| \leq C _ { y } \rho _ { y } ^ { t - r } \| Y _ { r } \| + Y _ { \infty } ( B _ { u } ) } \\ { \leq C _ { y } \rho _ { y } ^ { t - r } B _ { Y } + Y _ { \infty } ( B _ { u } ) . } \end{array}
$$

The $q - 1$ stored inputs in $\psi _ { t }$ each have magnitude at most $B _ { u }$ . Consequently,

$$
\begin{array} { r l } & { \| \psi _ { t } \| \leq \left[ \{ Y _ { \infty } ( B _ { u } ) + C _ { y } \rho _ { y } ^ { t - r } B _ { Y } \} ^ { 2 } + ( q - 1 ) B _ { u } ^ { 2 } \right] ^ { 1 / 2 } } \\ & { \qquad \leq P _ { \infty } ( B _ { u } ) + C _ { y } \rho _ { y } ^ { t - r } B _ { Y } . } \end{array}
$$

On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ , the raw (pre-clipping) part of the gate calculation in Lemma 13 does not use feasibility. On the recursive branch its distance from $( \lambda ^ { - } ) ^ { \top } \psi _ { t }$ is at most $\left( g _ { H } + g _ { H } / 4 \right) \| \psi _ { t } \|$ , and on the reference branch it is at most $\left( g _ { H } / 4 \right) \left\| \psi _ { t } \right\|$ . Hence, on either branch and also after a change,

$$
| z _ { t } - ( \lambda ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | \leq \varepsilon _ { u } .
$$

Since $\| \lambda ^ { - } \| \leq \Lambda$ , the preceding two displays imply

$$
\begin{array} { r l } & { | z _ { t } | \leq \varepsilon _ { u } + \Lambda P _ { \infty } ( B _ { u } ) + \Lambda C _ { y } B _ { Y } \rho _ { y } ^ { t - r } } \\ & { \qquad = B _ { u } - \mu _ { \mathrm { n c } } + \Lambda C _ { y } B _ { Y } \rho _ { y } ^ { t - r } . } \end{array}
$$

Definition (119) makes the last term at most $\mu _ { \mathrm { n c } }$ whenever $t - r \geq H + \tau _ { \mathrm { n c } }$ , proving (120). It also gives $| ( \lambda ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | \leq B _ { u } - \varepsilon _ { u }$ . The calculation is pathwise on the reference event and uniform over admissible switching sequences. □

Proposition 22 and the restart union in (74) give $\mathbb { P } ( \mathcal { E } _ { \mathrm { r e f } } ) \geq 1 - 2 ( T + 1 ) \delta _ { 0 } = 1 - \delta / 1 6$ . On this event, the conclusion holds simultaneously for every reached full, unchanged exploration block whose restart precedes the first detector failure.

Both $P _ { \infty }$ and $\varepsilon _ { u }$ depend on $B _ { u } ,$ so a positive $\mu _ { \mathrm { n c } }$ is not guaranteed by increasing the clipping threshold. An explicit suficient condition is obtained by setting

$$
a _ { 0 } : = \frac { C _ { y } B _ { w } } { 1 - \rho _ { y } } , \qquad a _ { 1 } : = \frac { C _ { y } B _ { b } } { 1 - \rho _ { y } } , \qquad \kappa _ { u } : = 1 - \Lambda \{ a _ { 1 } + \sqrt { q - 1 } \} .
$$

If $\kappa _ { u } > 0$ , choose

$$
B _ { u } > \frac { 2 \Lambda a _ { 0 } } { \kappa _ { u } } ~ \mathrm { a n d ~ t h e n ~ c h o o s e } ~ H ~ \mathrm { s o ~ t h a t } ~ \varepsilon _ { u } \leq \frac { \kappa _ { u } B _ { u } } { 2 } .\tag{121}
$$

Indeed, $P _ { \infty } ( B _ { u } ) \leq a _ { 0 } + ( a _ { 1 } + \sqrt { q - 1 } ) B _ { u }$ , so (121) implies $\mu _ { \mathrm { n c } } > 0$ . For fixed $B _ { u }$ the required inequality for $\varepsilon _ { u }$ is achieved by a suficiently large feasible H (when the horizon permits) under the exploration bound in (46). A direct suficient check for immediate nonclipping is $( \Lambda +$ $g _ { H } ) B _ { \psi } \leq B _ { u }$ : since $\left\| \overline { { \lambda } } _ { r } \right\| \leq \overset { } { \Lambda }$ , every raw exploitation proposal then lies in the clipping interval and no burn-in is needed.

Bounded noise alone does not ensure eventual nonclipping. Consider the known scalar system

$$
\begin{array} { r } { y _ { t + 1 } = \frac 1 2 y _ { t } + u _ { t } + w _ { t + 1 } , \qquad B _ { u } = \frac 1 4 , \qquad { \mathbb P } ( w _ { t } = 1 ) = { \mathbb P } ( w _ { t } = - 1 ) = \frac 1 2 . } \end{array}
$$

The exact minimum-variance gain is $\lambda = - 1 / 2$ . If the exact minimum-variance law were unclipped from some finite time onward, then $y _ { t + 1 } = w _ { t + 1 }$ from that time onward, and every following raw action would have magnitude $1 / 2 > B _ { u }$ , a contradiction. Thus no finite $\tau _ { \mathrm { n c } }$ exists for this example under that law.

Burn-in variant. Before the burn-in ends, clipping may invalidate gate charging and the detector estimates. After each full exploration block, apply reference control for $\tau _ { \mathrm { n c } }$ rounds with detection and RLS updates disabled and the detector bufer empty. Use $z _ { t } = \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t }$ and $u _ { t } = \mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( z _ { t } )$ . At $t _ { 0 } = r + H + \tau _ { \mathrm { n c } }$ set $\begin{array} { r } { V _ { t _ { 0 } } = \nu I _ { d } + \sum _ { t \in \mathbb { Z } _ { r } } \psi _ { t } \psi _ { t } ^ { \mathsf { T } } } \end{array}$ and $\widehat { \lambda } _ { t _ { 0 } } = \overline { { \lambda } } _ { r }$ , then use the shifted output grid $S _ { r } ^ { \mathrm { n c } } = \{ t _ { 0 } + h + j h : j \ge 0 \}$ . Truncate exploration or burn-in at $T .$ Require the first change no earlier than $H + \tau _ { \mathrm { n c } } ,$ successive changes to be separated by at least $D + H + \tau _ { \mathrm { n c } } ,$ and the terminal segment to have length at least D. From $t _ { 0 }$ onward clipping is inactive by (120), so the baseline gate, false-alarm, coupling, and RLS proofs apply without Assumption 10. The regret bound gains only the deterministic term $( C + 1 ) K _ { D } \tau _ { \mathrm { n c } }$ , and the logarithmic order is unchanged whenever $\tau _ { \mathrm { n c } } = O ( \log ( ( T + 1 ) / \delta ) )$ (in particular, when it is a fixed problem-dependent constant).

## K.3 Own-Regime Feasibility and a Boundary Alarm

A boundary alarm controls the post-change action error under feasibility of each gain only on regressors generated by its own plant.

Call a regressor an own-regime successor for θ if it can occur after at least $r _ { 0 }$ consecutive controls generated by plant $\theta ,$ starting from any regressor with norm at most $B _ { \psi } ,$ under disturbances bounded by $B _ { w }$ and arbitrary inputs in $[ - B _ { u } , B _ { u } ]$ . This definition includes the successors produced during exploration and the terminal state at a segment boundary; it is not restricted to states generated by the ideal policy.

Assumption 24 (Closed own-regime clipping feasibility). For every $\theta \in \Theta$ and every ownregime successor ψ for θ,

$$
\begin{array} { r } { | \lambda ( \theta ) ^ { \mathsf { T } } \psi | \leq B _ { u } . } \end{array}\tag{122}
$$

Equality is allowed, and no positive clipping margin is required.

Assumption 24 is an alternative feasibility condition: it checks each gain only on its own-regime successors and does not require the old gain to remain feasible on states produced by the new regime. Because the successor definition permits any norm-bounded starting regressor, including unreachable states, this condition is not asserted to follow from Assumption 10. c

Define the deterministic reference uncertainty and safe tracking bounds

$$
\begin{array} { r l } & { \zeta _ { H } : = E _ { \lambda } ( H , \delta _ { 0 } ) B _ { \psi } , } \\ & { { \bar { \varepsilon } _ { u } } : = \varepsilon _ { u } + 2 \zeta _ { H } = 7 L _ { \lambda } B _ { \psi } E _ { \theta } ( H , \delta _ { 0 } ) , } \\ & { { \bar { \varepsilon } _ { y } } : = \bar { b } \bar { \varepsilon } _ { u } . } \end{array}\tag{123}
$$

The branch-free union-class version of PIECE-CD adds the following test. After an exploitation round t, once $y _ { t + 1 }$ has been observed and $\psi _ { t + 1 }$ has been formed, raise one alarm and restart at control time $t + 1$ if

$$
| \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t + 1 } | > B _ { u } + \zeta _ { H } .\tag{124}
$$

The strict inequality is part of the test: under the null, equality with the upper bound is possible for discrete disturbances. If both the boundary and energy tests cross at the same output, they count as one alarm. Algorithm 3 states the complete composite procedure.

Lemma 25 (Boundary-alarm dichotomy). Under Assumption ${ \it 2 4 } ,$ fix a reached cycle whose full exploration block has parameter $\theta ^ { - }$ , and suppose $\mathcal G _ { r } ^ { \mathrm { e s t } }$ holds. Then the boundary test cannot cross while the reference parameter remains active. On such an unchanged exploitation interval, with $a _ { t } : = ( \widehat { \lambda } _ { t } - \lambda ( \theta ^ { - } ) ) ^ { \mathsf { T } } \psi _ { t }$ and $e _ { t } : = u _ { t } - \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } .$ , one also has $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$ . After a change within this exploitation cycle, at every control time before the cycle’s next alarm,

$$
\left( | \lambda ( { \boldsymbol { \theta } } ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | - B _ { u } \right) _ { + } \leq 2 \zeta _ { H } ,\tag{125}
$$

$$
\left| u _ { t } - \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } \right| \leq \overline { { \varepsilon } } _ { u } .\tag{126}
$$

These inequalities also hold in the alarm-suppressed continuation at control states satisfying $| \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t } | \leq B _ { u } + \zeta _ { H }$

Proof. While the reference parameter $\theta ^ { - }$ remains active, every exploitation regressor is an ownregime successor. On $\mathcal { G } _ { r } ^ { \mathrm { e s t } } , \left\| \overline { { \lambda } } _ { r } - \lambda ( \theta ^ { - } ) \right\| \leq E _ { \lambda } ( H , \delta _ { 0 } )$ , so (122) gives

$$
\begin{array} { r l } & { | \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t } | \leq | \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | + \left\| \overline { { \lambda } } _ { r } - \lambda ( \theta ^ { - } ) \right\| \left\| \psi _ { t } \right\| } \\ & { \qquad \leq B _ { u } + \zeta _ { H } . } \end{array}
$$

The strict test in (124) therefore cannot give a boundary false alarm. Since the old ideal action is feasible on the unchanged interval, the gate-charging and projection argument of Lemma 13 applies verbatim and gives $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$

For a change occurring during the exploitation cycle in the lemma, at the first post-change control the regressor is still an own-regime successor for $\theta ^ { \cdot }$ <sup>−</sup>, so its saturation deficit is zero. At every later post-change control time with no previous boundary alarm,

$$
\begin{array} { r } { | \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | \leq | \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t } | + E _ { \lambda } ( H , \delta _ { 0 } ) B _ { \psi } \leq B _ { u } + 2 \zeta _ { H } , } \end{array}
$$

which proves (125). The raw gated proposal satisfies $| z _ { t } - \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } | \ \leq \ \varepsilon _ { u }$ . For $P ( x ) =$ $\mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( x )$ and any $x , z \in \mathbb { R }$

$$
\begin{array} { c l c r } { { | P ( z ) - x | \le | P ( z ) - P ( x ) | + | P ( x ) - x | } } \\ { { } } & { { } } \\ { { } } & { { \leq | z - x | + ( | x | - B _ { u } ) _ { + } . } } \end{array}
$$

Combining this inequality with (125) gives (126). The same calculation applies pointwise in the alarm-suppressed continuation whenever $| \overline { { \lambda } } _ { r } ^ { \top } \psi _ { t } | \leq B _ { u } + \zeta _ { H }$ □

## K.4 Detection under the Union Condition

The boundary tracking bound (126) permits a common finite-window detector for stable, directly visible, and non-Schur ARX mismatches.

For an admissible adjacent pair $z = ( \theta ^ { - } , \theta ^ { + } )$ , write

$$
A _ { z } : = A ( \theta ^ { - } , \theta ^ { + } ) , \qquad G _ { h } ( A _ { z } ) : = \sigma _ { w } ^ { 2 } \sum _ { j = 1 } ^ { h - 1 } ( e _ { 1 } ^ { \top } A _ { z } ^ { j } e _ { 1 } ) ^ { 2 } .
$$

The supplied admissible adjacent-pair class $\mathcal { P } \subset \Theta ^ { 2 }$ has a cover ${ \mathcal { P } } \subseteq { \mathcal { P } } _ { \mathsf { S } } \cup { \mathcal { P } } _ { \mathsf { F } }$ . The finite-window component may itself be covered by ${ \mathcal { P } } _ { \mathsf { F } } \subseteq { \mathcal { P } } _ { \mathsf { D } } \cup { \mathcal { P } } _ { \mathsf { A } }$ . The set $\mathcal { P } _ { \mathsf { D } }$ uses a directly supplied finiteorder gap, whereas ${ \mathcal { P } } _ { \mathsf { A } }$ uses only the ARX non-Schur condition $\rho ( A _ { z } ) \geq 1$ . These sets may overlap. These classes specify uniform tuning bounds.

Define the finite-order innovation visibility

$$
\mathcal { V } ( z ) : = \sigma _ { w } ^ { 2 } \sum _ { j = 1 } ^ { d } \{ e _ { 1 } ^ { \top } A _ { z } ^ { j } e _ { 1 } \} ^ { 2 } .\tag{127}
$$

For $\theta \in \Theta$ , define the monic old-input polynomial

$$
\overline { { B } } _ { \theta } ( \xi ) : = \xi ^ { q - 1 } + \frac { b _ { 2 } ( \theta ) } { b _ { 1 } ( \theta ) } \xi ^ { q - 2 } + \cdots + \frac { b _ { q } ( \theta ) } { b _ { 1 } ( \theta ) } ,
$$

with $\overline { { B } } _ { \theta } \equiv 1$ when $q = 1$ . Under the minimum-phase convention in Assumption $^ { 7 , }$ every root of $\overline { { B } } _ { \theta }$ lies in the open unit disk. Continuity of polynomial roots and compactness therefore give a safe uniform number

$$
r _ { B } \ge \operatorname* { s u p } _ { \theta \in \Theta } \operatorname* { m a x } \{ | \beta | : \overline { { B } } _ { \theta } ( \beta ) = 0 \} , \qquad 0 \le r _ { B } < 1 ,\tag{128}
$$

where the maximum over the empty set is zero. Also let

$$
R _ { A } \geq \operatorname* { m a x } \left\{ 1 , \operatorname* { s u p } _ { ( \theta ^ { - } , \theta ^ { + } ) \in \Theta ^ { 2 } } \left\| A _ { \mathrm { c l } } ( \lambda ( \theta ^ { - } ) , \theta ^ { + } ) \right\| _ { 2 } \right\} < \infty , \qquad K _ { A } : = \sum _ { k = 1 } ^ { d } R _ { A } ^ { d - k } \sum _ { j = 0 } ^ { k - 1 } { \binom { d } { j } } R _ { A } ^ { j } .\tag{129}
$$

These are ofline model-class bounds. The uniform root bound follows from continuity of polynomial zeros and compactness (Horn and Johnson, 2013, Appendix D, Theorem D1, and $\mathrm { A p \mathrm { - } }$ pendix E).

Finite-order visibility of non-Schur mismatches. Define

$$
\underline { { \Delta } } _ { \mathsf { A } } : = \sigma _ { w } ^ { 2 } \left\{ \frac { ( 1 - r _ { B } ) ^ { q - 1 } } { K _ { A } } \right\} ^ { 2 } > 0 .\tag{130}
$$

Then every ARX mismatch matrix $A _ { z }$ satisfying $\rho ( A _ { z } ) \geq 1$ obeys

$$
\begin{array} { r } { \mathcal { V } ( z ) \geq \underline { { \Delta } } _ { \mathsf { A } } . } \end{array}\tag{131}
$$

This implication uses the ARX and minimum-phase structure; it is false for a general square matrix.

To prove it, put $h _ { j } = e _ { 1 } ^ { \mathsf { T } } A _ { z } ^ { j } e _ { 1 }$ and

$$
D _ { z } ( \xi ) : = \operatorname * { d e t } ( \xi I - A _ { z } ) = \xi ^ { d } + \sum _ { k = 1 } ^ { d } d _ { k } \xi ^ { d - k } ,
$$

$$
Q _ { - } ( \xi ) : = \xi ^ { p } \overline { { { B } } } _ { \theta ^ { - } } ( \xi ) = \xi ^ { d } + \sum _ { k = 1 } ^ { d } q _ { k } \xi ^ { d - k } .
$$

A cofactor expansion using the shift rows of $A _ { z }$ gives

$$
e _ { 1 } ^ { \mathsf { T } } \mathrm { a d j } ( \xi I - A _ { z } ) e _ { 1 } = \xi ^ { p - 1 } \overline { { B } } _ { \theta ^ { - } } ( \xi ) .\tag{132}
$$

Indeed, deleting the first row and column leaves a block-lower-triangular minor: its outputshift block has determinant $\xi ^ { p - 1 }$ and its old-input companion block has determinant $\overline { { B } } _ { \theta ^ { - } } ( \xi )$ The adjugate formula and the Laurent expansion at infinity then give the exact disturbance-tooutput identity, for $| \xi | > \rho ( A _ { z } )$ (equivalently, as a formal Laurent identity at infinity),

$$
\xi e _ { 1 } ^ { \mathsf { T } } ( \xi I - A _ { z } ) ^ { - 1 } e _ { 1 } = 1 + \sum _ { j \geq 1 } h _ { j } \xi ^ { - j } = { \frac { Q _ { - } ( \xi ) } { D _ { z } ( \xi ) } } .\tag{133}
$$

Comparing the coeficient of $\xi ^ { d - k }$ in $\begin{array} { r } { D _ { z } ( \xi ) ( 1 + \sum _ { j \ge 1 } h _ { j } \xi ^ { - j } ) = Q . } \end{array}$ <sub>−</sub>(ξ) gives, with $d _ { 0 } = 1$ 2

$$
q _ { k } - d _ { k } = \sum _ { i = 1 } ^ { k } d _ { k - i } h _ { i } , \qquad k = 1 , \dots , d .\tag{134}
$$

Let $\eta : = \operatorname* { m a x } _ { 1 \leq j \leq d } | h _ { j } |$ and choose an eigenvalue $\lambda _ { \star }$ with $| \lambda _ { \star } | = \rho ( A _ { z } ) \geq 1$ . Since $| d _ { j } | \leq { \binom { d } { j } } R _ { A } ^ { j }$ and $| \lambda _ { \star } | \le R _ { A }$ (the spectral radius is bounded by every induced matrix norm (Horn and Johnson, 2013, Theorem 5.6.9)), equations (134) and $D _ { z } ( \lambda _ { \star } ) = 0$ imply

$$
| Q _ { - } ( \lambda _ { \star } ) | \leq \eta K _ { A } .
$$

On the other hand, $Q _ { - }$ has $p$ zeros at zero and its other $q - 1$ zeros have modulus at most $r _ { B }$ so

$$
| Q _ { - } ( \lambda _ { \star } ) | \geq | \lambda _ { \star } | ^ { p } ( | \lambda _ { \star } | - r _ { B } ) ^ { q - 1 } \geq ( 1 - r _ { B } ) ^ { q - 1 } .
$$

Combining the last two displays and using $\mathcal { V } ( z ) \geq \sigma _ { w } ^ { 2 } \eta ^ { 2 }$ proves (131). If a stronger class-wide margin $\rho ( A _ { z } ) \ \geq \ c _ { a } > 1$ is available, the same proof gives the sharper constant $\underline { { \Delta } } _ { \mathsf { A } } ( c _ { a } ) : = \qquad $ $\sigma _ { w } ^ { 2 } \{ c _ { a } ^ { p } ( c _ { a } - r _ { B } ) ^ { q - 1 } / K _ { A } \} ^ { 2 }$ , which may replace $\Delta _ { \mathsf { A } }$ in the tuning. Such a margin is not required for the theorem below.

Assumption 26 (Union stable/direct/non-Schur detectability). Every realized adjacent pair $z _ { c } = ( \theta ^ { ( c - 1 ) } , \theta ^ { ( c ) } ) , \ c = 1 , \dots , C$ , belongs to P. For a nonempty stable component there are supplied constants $C _ { 5 } \geq 1 , \rho _ { 5 } \in ( 0 , 1 )$ , and $\underline { { \Delta } } _ { \mathsf { S } } > 0 .$ ; for a nonempty direct-visibility component there is a supplied $\Delta _ { \mathsf { D } } > 0$ ; and the possibly empty ARX non-Schur component satisfies

$$
\begin{array} { r l r } & { z \in \mathcal { P } _ { \mathsf { S } } \Longrightarrow \left\| A _ { z } ^ { k } \right\| \leq C _ { \mathsf { S } } \rho _ { \mathsf { S } } ^ { k } \quad ( k \geq 0 ) , \quad } & { J _ { y } ( \lambda ( \theta ^ { - } ) ; \theta ^ { + } ) - \sigma _ { w } ^ { 2 } \geq \Delta _ { \mathsf { S } } , } \\ & { z \in \mathcal { P } _ { \mathsf { D } } \Longrightarrow \mathcal { V } ( z ) \geq \Delta _ { \mathsf { D } } , } \\ & { z \in \mathcal { P } _ { \mathsf { A } } \Longrightarrow \rho ( A _ { z } ) \geq 1 . } \end{array}\tag{135}
$$

Each admitted pair satisfies at least one component condition. A stable pair requires no separate finite-order gap; for a non-Schur pair, that gap is supplied by (130).

Set $\Gamma _ { \mathsf { D } } = \underline { { \Delta } } _ { \mathsf { D } }$ when $\mathcal { P } _ { \mathsf { D } } \neq \emptyset$ and $\Gamma _ { \mathsf { D } } = + \infty$ otherwise. Set $\Gamma _ { \mathsf { A } } = \underline { { \Delta } } _ { \mathsf { A } }$ when $\mathcal { P } _ { \mathsf { A } } \neq \emptyset$ and $\Gamma _ { \mathsf { A } } = + \infty$ otherwise. If $\mathcal { P } _ { \mathsf { F } } \neq \emptyset$ , define

$$
\underline { { \Delta } } _ { \mathsf { F } } : = \operatorname* { m i n } \{ { \Gamma _ { \mathsf { D } } } , { \Gamma _ { \mathsf { A } } } \} > 0 .\tag{136}
$$

Equations (127) and (131) then give the derived common bound

$$
z \in { \mathcal { P } } _ { \mathsf { F } } \Longrightarrow { \mathcal { V } } ( z ) \geq \Delta _ { \mathsf { F } } .\tag{137}
$$

This is the finite-visible condition used in Section 5.

For a nonempty stable component, define

$$
h _ { 5 } : = \operatorname* { m i n } \left\{ k \ge 1 : \frac { \sigma _ { w } ^ { 2 } C _ { S } ^ { 2 } \rho _ { S } ^ { 2 } } { 1 - \rho _ { 5 } ^ { 2 } } \le \frac { \Delta _ { S } } { 8 } \right\} , \quad \quad \quad \quad \Gamma _ { 5 } : = \frac { 7 \underline { { \Delta } } _ { 5 } } { 8 } .
$$

If $\mathcal { P } _ { \mathsf { S } } = \emptyset$ , set $( h _ { 5 } , \Gamma _ { 5 } ) \ = \ ( 1 , + \infty )$ . If $\mathcal { P } _ { \mathsf { F } } \neq \emptyset$ , set $( h _ { \mathsf { F } } , \Gamma _ { \mathsf { F } } ) = ( d + 1 , \underline { { \Delta } } _ { \mathsf { F } } )$ ; otherwise set $( h _ { \mathsf { F } } , \Gamma _ { \mathsf { F } } ) = ( 1 , + \infty )$ $\operatorname { I f } \mathcal { P } = \varnothing$ , set $h _ { \mathsf { U } } = 1$ and choose any fixed finite $\underline { { \Delta } } _ { \mathsf { U } } > 0$ for null calibration; every missed-change statement is then vacuous. Otherwise at least one cover component is nonempty, and set

$$
h _ { \mathsf { U } } : = \operatorname* { m a x } \{ h _ { { \mathsf { S } } } , h _ { \mathsf { F } } \} ,
$$

$$
\underline { { \Delta } } _ { \mathsf { U } } : = \operatorname* { m i n } \{ \Gamma _ { 5 } , \Gamma _ { \mathsf { F } } \} > 0 .\tag{138}
$$

The resulting common finite-horizon power separation is

$$
G _ { h _ { \mathsf { U } } } ( A _ { z } ) \geq \underline { { \Delta } } _ { \mathsf { U } } , \qquad z \in \mathcal { P } .\tag{139}
$$

Indeed, $\mathrm { i f } ~ z \in \mathcal { P } _ { \mathsf { S } }$ , then

$$
\begin{array} { r l } & { G _ { h _ { \mathsf { U } } } ( A _ { z } ) = J _ { y } ( \lambda ( \theta ^ { - } ) ; \theta ^ { + } ) - \sigma _ { w } ^ { 2 } - \sigma _ { w } ^ { 2 } \displaystyle \sum _ { j = h _ { \mathsf { U } } } ^ { \infty } ( e _ { 1 } ^ { \top } A _ { z } ^ { j } e _ { 1 } ) ^ { 2 } } \\ & { \qquad \quad \ge \displaystyle \underline { { \Delta } } _ { \mathsf { S } } - \frac { \sigma _ { w } ^ { 2 } C _ { \mathsf { S } } ^ { 2 } \rho _ { \mathsf { S } } ^ { 2 h _ { \mathsf { U } } } } { 1 - \rho _ { \mathsf { S } } ^ { 2 } } \ge \frac { 7 \underline { { \Delta } } _ { \mathsf { S } } } { 8 } \ge \underline { { \Delta } } _ { \mathsf { U } } . } \end{array}
$$

If $z \in { \mathcal { P } } _ { \mathsf { F } }$ , then $h _ { \mathsf { U } } \geq d + 1$ and the nonnegative summands give $G _ { h _ { \mathsf { U } } } ( A _ { z } ) \geq \mathcal { V } ( z ) \geq \underline { { \Delta } } _ { \mathsf { F } } \geq \underline { { \Delta } } _ { \mathsf { U } }$ This proves (139) without identifying which route holds for the realized change.

The first d impulse coeficients exhaust finite-order visibility by Cayley–Hamilton (Horn and Johnson, 2013, Theorem 2.4.3.2). Let $h _ { j } = e _ { 1 } ^ { \mathsf { T } } A ^ { j } e _ { 1 }$ , so $h _ { 0 } = 1 .$ , and write the characteristic polynomial as $p _ { A } ( z ) = z ^ { d } + \kappa _ { d - 1 } z ^ { d - 1 } + \cdot \cdot \cdot + \kappa _ { 0 }$ . If $h _ { 1 } = \cdots = h _ { d } = 0$ , sandwiching $p _ { A } ( A ) = 0$ between $e _ { 1 } ^ { \mathsf { T } }$ and $e _ { 1 }$ gives $\kappa _ { 0 } = 0$ . Sandwiching $A ^ { k } p _ { A } ( A ) = 0$ for $k \geq 1$ and using strong induction then gives $h _ { j } = 0$ for every $j \geq 1$ . Thus every change visible through a noise-driven impulse coeficient is visible by order d.

If $\mathcal { P } = \varnothing$ , set $L _ { 0 , \mathsf { U } } = L _ { w , \mathsf { U } } = L _ { u , \mathsf { U } } = 0$ . Otherwise, the same model-class bound $R _ { A }$ gives the explicit finite-horizon choices

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { 0 , \mathsf { U } } : = { \cal R } _ { A } ^ { h _ { \mathsf { U } } } , } } \\ { { \displaystyle { \cal L } _ { w , \mathsf { U } } : = \sum _ { j = 0 } ^ { h _ { \mathsf { U } } - 1 } { \cal R } _ { A } ^ { j } } , } \end{array}
$$

$$
L _ { u , \mathsf { U } } : = G _ { u } \sum _ { j = 0 } ^ { h \mathsf { U } - 1 } R _ { A } ^ { j } .\tag{140}
$$

Indeed, $| x ^ { \mathsf { T } } A ^ { j } y | \leq \| x \| _ { 2 } \| A \| _ { 2 } ^ { j } \| y \| _ { 2 }$ . These bounds are finite because $h _ { \mathsf { U } }$ is finite, even when some matrices in ${ \mathcal { P } } _ { \mathsf { A } }$ are non-Schur. Sharper supplied bounds may replace them without changing the proof.

Define

$$
\begin{array} { r } { B _ { \mathrm { a u x , \mathsf { U } } } : = L _ { 0 , \mathsf { U } } B _ { \psi } + L _ { w , \mathsf { U } } B _ { w } , \qquad } \\ { B _ { \mathsf { U } } : = \operatorname* { m a x } \{ B _ { Y } , B _ { \mathrm { a u x } , \mathsf { U } } , B _ { w } \} , } \end{array}
$$

$$
\begin{array} { r } { R _ { \mathsf { U } } ( H ) : = L _ { u , \mathsf { U } } \overline { { \varepsilon } } _ { u } . } \end{array}\tag{141}
$$

In addition to the exploration sample-size condition and (51), choose the smallest H satisfying

$$
\overline { { \varepsilon } } _ { y } ^ { 2 } \leq \frac { \Delta _ { \mathsf { U } } } { 8 } ,
$$

$$
\begin{array} { r } { \overline { { \varepsilon } } _ { y } \leq B _ { \mathsf { U } } , } \end{array}
$$

$$
2 B _ { \mathsf { U } } R _ { \mathsf { U } } ( H ) \leq \frac { \Delta _ { \mathsf { U } } } { 8 } .\tag{142}
$$

Use spacing $h _ { \mathsf { U } }$ , the energy threshold $\sigma _ { w } ^ { 2 } + \underline { { \Delta } } _ { \mathsf { U } } / 2$ , and set

$$
m _ { \mathsf { U } } : = \left\lceil \frac { 5 1 2 B _ { \mathsf { U } } ^ { 4 } } { \underline { { \Delta } } _ { \mathsf { U } } ^ { 2 } } \log \frac { 4 T ^ { 2 } } { \delta _ { d } } \right\rceil ,
$$

$$
D _ { \mathsf { U } } : = ( m _ { \mathsf { U } } + 1 ) h _ { \mathsf { U } } .\tag{143}
$$

Assumption 12 is read with $D _ { \mathsf { U } }$ in place of $D$

Fix $z = ( \theta ^ { - } , \theta ^ { + } ) \in \mathcal { P }$ and a potential restart r whose exploration reference estimates $\lambda ( \theta ^ { - } )$ Continue the clipped controller on the same restart-relative calendar with both alarms suppressed. Its regressor remains bounded by $B _ { \psi }$ on every path. For a stored output time s with $s - h _ { \mathsf { U } } \geq r + H$ , define the $h _ { \mathsf { U } }$ -step auxiliary

$$
\begin{array} { r l } & { \widetilde \psi _ { s - h _ { \mathsf { U } } } ^ { ( s ) } : = \psi _ { s - h _ { \mathsf { U } } } , } \\ & { \quad \widetilde \psi _ { t + 1 } ^ { ( s ) } = A _ { z } \widetilde \psi _ { t } ^ { ( s ) } + e _ { 1 } w _ { t + 1 } , \qquad s - h _ { \mathsf { U } } \leq t < s . } \end{array}\tag{144}
$$

Let $\widetilde { y } _ { s } ^ { \left( s \right) }$ be its first coordinate. This auxiliary is defined on every continuation path, including paths on which the boundary test would cross.

Lemma 27 (Sample-local union power and comparison). For the pair $z \in { \mathcal { P } }$ and continuation defined above, suppose $\theta ^ { + }$ is active throughout $[ s - h _ { \mathsf { U } } , s )$ . Then

$$
| \mathcal { \widetilde { y } } _ { s } ^ { ( s ) } | \le B _ { \mathrm { a u x , \mathsf { U } } } ,\tag{145}
$$

$$
\begin{array} { r } { \mathbb { E } [ ( \widetilde { y } _ { s } ^ { ( s ) } ) ^ { 2 } \mid \mathcal { F } _ { s - h \mathsf { U } } ] \ge \sigma _ { w } ^ { 2 } + \underline { { \Delta \mathsf { u } } } . } \end{array}\tag{146}
$$

On $\mathcal G _ { r } ^ { \mathrm { e s t } }$ and on the event that the boundary test has not crossed at the states used for the controls in $[ s - h _ { \mathsf { U } } , s )$ ，

$$
\begin{array} { r l r } & { \displaystyle | y _ { s } - \widetilde { y } _ { s } ^ { ( s ) } | \le R _ { \mathsf { U } } ( H ) , } & \\ & { \displaystyle | y _ { s } ^ { 2 } - ( \widetilde { y } _ { s } ^ { ( s ) } ) ^ { 2 } | \le 2 B _ { \mathsf { U } } R _ { \mathsf { U } } ( H ) \le \frac { \Delta _ { \mathsf { U } } } { 8 } . } & \end{array}\tag{147}
$$

The conditional statements remain valid at finite lag-h<sub>U</sub> predictable output times.

Proof. Iterating (144) gives

$$
\widetilde \psi _ { s } ^ { ( s ) } = A _ { z } ^ { h _ { \mathsf { U } } } \psi _ { s - h _ { \mathsf { U } } } + \sum _ { j = 0 } ^ { h _ { \mathsf { U } } - 1 } A _ { z } ^ { j } e _ { 1 } w _ { s - j } .
$$

The definitions of ${ \cal L } _ { 0 , \mathrm { U } }$ and $\mathit { L } _ { w , \mathrm { U } }$ give (145). Conditional on $\mathcal { F } _ { s - h _ { \mathsf { U } } } { : }$ the starting state is fixed and $w _ { s - h _ { \mathsf { U } } + 1 } , \hdots , w _ { s }$ are fresh, independent, and centered. Hence

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( \widetilde { y } _ { s } ^ { ( s ) } ) ^ { 2 } \mid \mathcal { F } _ { s - h _ { \mathsf { U } } } ] = ( e _ { 1 } ^ { \mathsf { T } } A _ { z } ^ { h _ { \mathsf { U } } } \psi _ { s - h _ { \mathsf { U } } } ) ^ { 2 } + \sigma _ { w } ^ { 2 } \sum _ { j = 0 } ^ { h _ { \mathsf { U } } - 1 } ( e _ { 1 } ^ { \mathsf { T } } A _ { z } ^ { j } e _ { 1 } ) ^ { 2 } } } \\ & { } & { \geq \sigma _ { w } ^ { 2 } + G _ { h _ { \mathsf { U } } } ( A _ { z } ) \geq \sigma _ { w } ^ { 2 } + \Delta _ { \mathsf { U } } , } \end{array}
$$

which proves (146).

On the no-boundary event in the statement, Lemma 25 gives $u _ { t } = \lambda ( \theta ^ { - } ) ^ { \mathsf { T } } \psi _ { t } + e _ { t }$ with $| e _ { t } | \le \overline { { \varepsilon } } _ { u }$ throughout $[ s - h _ { \mathsf { U } } , s )$ . Subtracting the two state recursions and unrolling from their common state at $s - h _ { \mathsf { U } }$ gives

$$
\vert y _ { s } - \widetilde { y } _ { s } ^ { ( s ) } \vert \leq \overline { { \varepsilon } } _ { u } \sum _ { j = 0 } ^ { h _ { \mathsf { U } } - 1 } \vert e _ { 1 } ^ { \mathsf { T } } A _ { z } ^ { j } g _ { u } ( \theta ^ { + } ) \vert \leq L _ { u , \mathsf { U } } \overline { { \varepsilon } } _ { u } .
$$

Both output magnitudes are at most $B _ { \mathsf { U } }$ , so $| x ^ { 2 } - z ^ { 2 } | \leq | x - z | ( | x | + | z | )$ proves (147). For a finite lag-h<sub>U</sub> predictable time, multiply the deterministic-time identities by $\mathbf { 1 } \{ s = t \} \in$ $\mathcal { F } _ { t - h _ { \mathsf { U } } }$ and sum over t. This gives the same relations with the stopped field $\mathcal { F } _ { s - h _ { \mathsf { U } } }$ □

Lemma 28 (One-change detection under the union condition). Adopt the stopping-time, activecycle, and horizon hypotheses of Lemma $^ { 1 7 , }$ use Algorithm 3, replace D by $D _ { \mathsf { U } }$ , and suppose Assumptions $\it { 2 4 }$ and 26 hold. Almost surely on the corresponding $\boldsymbol { \mathcal { A } } _ { \boldsymbol { r } , \tau }$

$$
\mathbb { P } ( \widehat { \tau } _ { r } > \tau + D _ { \mathsf { U } } \mid \mathcal { F } _ { \tau } ) \leq \frac { \delta _ { d } } { 4 T ^ { 2 } } .\tag{148}
$$

Proof. Condition on an arbitrary $\mathcal { F } _ { \tau }$ history in $\mathcal { A } _ { \boldsymbol { r } , \tau }$ and use the alarm-suppressed continuation just defined. The deterministic calendar argument in Appendix H gives stored output times $s _ { 1 } , \ldots , s _ { m _ { \mathsf { U } } }$ such that $s _ { 1 } - h _ { \mathsf { U } } \geq \tau , s _ { k + 1 } - s _ { k } = h _ { \mathsf { U } }$ , and $s _ { m _ { \mathsf { U } } } - \tau < D _ { \mathsf { U } }$ . Conditionally on $\mathcal { F } _ { \tau }$ the restart-relative grid is lag-h<sub>U</sub> predictable, $\mathcal { F } _ { \tau } \subseteq \mathcal { F } _ { s _ { 1 } - h _ { \mathsf { U } } }$ , and the disturbances used by the sample-local auxiliaries remain fresh.

Apply the conditional form of Lemma 14 given $\mathcal { F } _ { \tau }$ to the sample-local variables $\widetilde { y } _ { s _ { k } } ^ { ( s _ { k } ) }$ , using the lagged filtration $\mathcal { H } _ { k - 1 } : = \mathcal { F } _ { s _ { k } - h _ { \mathsf { U } } }$ . Lemma 27 gives the required conditional means, and $| \widetilde { y } _ { s _ { k } } ^ { ( s _ { k } ) } | \le B _ { \mathsf { U } }$ . Therefore

$$
\begin{array} { r l r } & { } & { \mathbb { P } \bigg ( \frac { 1 } { m _ { \mathsf { U } } } \displaystyle \sum _ { k = 1 } ^ { m _ { \mathsf { U } } } ( \widetilde { y } _ { s _ { k } } ^ { ( s _ { k } ) } ) ^ { 2 } < \sigma _ { w } ^ { 2 } + \frac { 7 \Delta _ { \mathsf { U } } } { 8 } \biggm | \mathcal { F } _ { \tau } \bigg ) \leq \exp \bigg ( - \frac { m _ { \mathsf { U } } \Delta _ { \mathsf { U } } ^ { 2 } } { 1 2 8 B _ { \mathsf { U } } ^ { 4 } } \biggm ) } \\ & { } & { \leq \frac { \delta _ { d } } { 4 T ^ { 2 } } . \qquad } \end{array}\tag{149}
$$

This concentration bound is not conditioned on survival of the boundary test.

If either test raises an alarm at or before $s _ { m _ { \mathsf { U } } }$ , the change has already been detected. Otherwise, the actual and alarm-suppressed trajectories agree through the complete block, and (147) holds at every sample. Outside the deviation in (149), the actual empirical energy is at least $\sigma _ { w } ^ { 2 } +$ $3 \underline { { \Delta } } _ { \mathsf { U } } / 4$ , which crosses the threshold $\sigma _ { w } ^ { 2 } + \underline { { \Delta } } _ { \mathsf { U } } / 2$ . Consequently,

$$
\{ { \hat { \tau } } _ { r } > \tau + D \mathsf { u } \} \subseteq \left\{ \frac { 1 } { m \mathsf { u } } \sum _ { k = 1 } ^ { m _ { \mathsf { U } } } ( \widetilde { y } _ { s _ { k } } ^ { ( s _ { k } ) } ) ^ { 2 } < \sigma _ { w } ^ { 2 } + \frac { 7 \Delta \mathsf { u } } { 8 } \right\} .
$$

Taking probabilities proves (148).

A smaller $\underline { { \Delta } } _ { \mathsf { U } }$ increases the exploration and window requirements in (142)–(143). Uniform finite delay requires every admitted Schur pair to satisfy either the stable-gap or direct-visibility condition.

A scalar non-Schur mismatch. Take $p \ = \ q \ = \ 1 , \ B _ { u } \ = \ 1 0 .$ , zero initial history, and Rademacher noise with $B _ { w } = \sigma _ { w } = 1$ . Let $\Theta = \{ \theta ^ { - } , \theta ^ { + } \}$ and consider the one-change sequence $\theta ^ { - }  \theta ^ { + }$ , where

$$
\theta ^ { - } = ( - 0 . 2 , 0 . 2 ) , \qquad \theta ^ { + } = ( 0 . 2 , 1 ) .
$$

Both open-loop coeficients have magnitude 0.2, and the deterministic output bound is $B _ { Y } =$ $1 1 / ( 1 - 0 . 2 ) = 1 3 . 7 5$ . The interval $[ - B _ { Y } , B _ { Y } ]$ is invariant under either plant with $| u _ { t } | \le B _ { u }$ and $| w _ { t } | \leq 1$ . The two ideal gains are $\lambda ^ { - } = 1$ and $\lambda ^ { + } = - 0 . 2$ . Every one-step own-regime successor satisfies

$$
\begin{array} { r l } & { | \lambda ^ { - } y ^ { + } | \le 0 . 2 B _ { Y } + 0 . 2 B _ { u } + B _ { w } = 5 . 7 5 , } \\ & { | \lambda ^ { + } y ^ { + } | \le 0 . 2 B _ { Y } = 2 . 7 5 . } \end{array}
$$

Thus Assumption 24 holds, while the old gain need not be feasible on every state produced by the new plant. Nevertheless, $A ( \theta ^ { - } , \theta ^ { + } ) = 0 . 2 + 1 \cdot 1 = 1 . 2$ . Here $d = 1$ , so $h _ { \mathsf F } = 2$ and $\mathcal { V } ( \theta ^ { - } , \theta ^ { + } ) = 1 . 2 ^ { 2 } = 1 . 4 4$ . Since $r _ { B } = 0$ here, the non-Schur formula itself supplies the valid positive bound $\Delta _ { \mathsf { A } } = 1 ;$ no separate visibility assumption is needed.

## K.5 Regret with Saturation

Without clipping feasibility, exploitation regret can be controlled by the realized saturation energy. For a true gain λ, define

$$
d _ { t } ^ { \mathrm { s a t } } : = \left( | \lambda ^ { \mathsf { T } } \psi _ { t } | - B _ { u } \right) _ { + } .\tag{150}
$$

On an unchanged exploitation segment with a successful reference and $g _ { H } = 4 E _ { \lambda }$ , the unclipped gate-charging inequality from Appendix G and the projection distance inequality give

$$
\begin{array} { r } { | u _ { t } - \lambda ^ { \mathsf { T } } \psi _ { t } | \leq | a _ { t } | + d _ { t } ^ { \mathrm { s a t } } , \qquad a _ { t } = ( \widehat { \lambda } _ { t } - \lambda ) ^ { \mathsf { T } } \psi _ { t } . } \end{array}\tag{151}
$$

Let $\beta _ { s } : = { 1 } / { 8 }$ and define

$$
\begin{array} { r l } & { c _ { s } : = 1 - 2 \overline { { \alpha } } - \beta _ { s } - 4 \overline { { \ell } } ( 1 + \overline { { \alpha } } ) ^ { 2 } , } \\ & { k _ { s } : = \overline { { \alpha } } ^ { 2 } ( \beta _ { s } ^ { - 1 } + 4 \overline { { \ell } } ) . } \end{array}\tag{152}
$$

Under (51), $c _ { s } > 0 . 4 6 7$

Proposition 29 (Exploitation regret with saturation energy). Under the estimation, Gram, and unchanged-segment hypotheses of Theorem 19, but without clipping feasibility, with conditional probability at least $1 - \delta _ { 0 }$ , simultaneously for every prefix of length n,

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { t } ( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } \le 2 \overline { { b } } ^ { 2 } \bigg [ \frac { 2 \overline { { Q } } _ { H } } { c _ { s } } + \frac { 8 B _ { \eta } ^ { 2 } } { c _ { s } ^ { 2 } } \log \frac { 1 } { \delta _ { 0 } } + \frac { 4 B _ { \eta } ^ { 2 } } { c _ { s } } \Gamma ( n ) } \\ & { } & { \quad \quad \quad \quad \quad + \left( 1 + \frac { 2 k _ { s } } { c _ { s } } \right) \sum _ { t } ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } \bigg ] . } \end{array}\tag{153}
$$

Proof. Start from the exact identity (79). From (151), $| e _ { t } | \leq | a _ { t } | + d _ { t } ^ { \mathrm { s a t } }$ . Young’s inequality gives

$$
\begin{array} { c } { { \displaystyle 2 \alpha a _ { t } e _ { t } \le ( 2 \overline { { { \alpha } } } + \beta _ { s } ) a _ { t } ^ { 2 } + \displaystyle \frac { \overline { { { \alpha } } } ^ { 2 } } { \beta _ { s } } ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } , } } \\ { { \displaystyle ( \alpha e _ { t } + \eta _ { t + 1 } - a _ { t } ) ^ { 2 } \le 4 ( 1 + \overline { { { \alpha } } } ) ^ { 2 } a _ { t } ^ { 2 } + 4 \overline { { { \alpha } } } ^ { 2 } ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } + 2 \eta _ { t + 1 } ^ { 2 } . } } \end{array}
$$

Using $\ell _ { t } \leq \overline { { \ell } }$ yields

$$
Q _ { t + 1 } - Q _ { t } \leq - c _ { s } a _ { t } ^ { 2 } + 2 a _ { t } \eta _ { t + 1 } + 2 \ell _ { t } \eta _ { t + 1 } ^ { 2 } + k _ { s } \big ( d _ { t } ^ { \mathrm { s a t } } \big ) ^ { 2 } .
$$

The same time-uniform exponential supermartingale as in (81), now evaluated with $c _ { s } ,$ , and the same elliptical-potential bound give

$$
\sum _ { t } a _ { t } ^ { 2 } \leq \frac { 2 \overline { { Q } } _ { H } } { c _ { s } } + \frac { 8 B _ { \eta } ^ { 2 } } { c _ { s } ^ { 2 } } \log \frac { 1 } { \delta _ { 0 } } + \frac { 4 B _ { \eta } ^ { 2 } } { c _ { s } } \Gamma ( n ) + \frac { 2 k _ { s } } { c _ { s } } \sum _ { t } ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } .
$$

Finally, $( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } \leq 2 \bar { b } ^ { 2 } ( a _ { t } ^ { 2 } + ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } )$ , which proves (153).

Proposition 29 gives logarithmic unchanged-segment regret when $\begin{array} { r } { \sum _ { t } ( d _ { t } ^ { \mathrm { s a t } } ) ^ { 2 } = O ( \log ( ( T + 1 ) / \delta ) ) } \end{array}$ A global detection guarantee additionally requires a blockwise power gap for the clipped loop or control of its saturation residual on every potential detector block. The non-Schur gap (130) applies to the unclipped auxiliary; Lemma 27 transfers it to the actual output using the boundary test.

Proposition 30 (Saturation lower bound). For every policy with $| u _ { t } | \le B _ { u }$ 2

$$
\mathcal { R } _ { T } \geq b _ { \operatorname* { m i n } } ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } ( \vert \lambda ( \theta _ { t } ) ^ { \mathsf { T } } \psi _ { t } \vert - B _ { u } ) _ { + } ^ { 2 } .\tag{154}
$$

Consequently, if the realized saturation energy grows faster than log $T$ , logarithmic regret under (3) is impossible, even when the parameter sequence is known.

Proof. For $x _ { t } = \lambda ( \theta _ { t } ) ^ { \mathsf { T } } \psi _ { t }$ and every feasible action,

$$
| u _ { t } - x _ { t } | \geq \mathrm { d i s t } ( x _ { t } , [ - B _ { u } , B _ { u } ] ) = ( | x _ { t } | - B _ { u } ) _ { + } .
$$

The gain form (36) gives $( y _ { t + 1 } - w _ { t + 1 } ) ^ { 2 } = b _ { 1 } ( \theta _ { t } ) ^ { 2 } ( u _ { t } - x _ { t } ) ^ { 2 }$ . Use $b _ { 1 } ( \theta _ { t } ) \geq b _ { \mathrm { m i n } }$ and sum over t. □

## L A UNIFIED DETECTOR FOR THE UNION CLASS

Algorithm 3 implements the union-class detector introduced in Section 5. Its two tests use constants supplied ofline; the algorithm neither receives the realized pair’s class nor computes its spectral radius. The terms Schur and non-Schur refer to the old-gain/new-plant matrix $A _ { \mathrm { c l } } ( \lambda ( \theta ^ { - } ) , \theta ^ { + } )$ . The plant retains Assumptions 7, 8, and 9; bounded applied inputs therefore preserve the deterministic state bound.

## L.1 Complete Algorithm

Algorithm 3: Branch-free union-class PIECE-CD

Input: $p , q , \Theta , T , \delta , B _ { u } , \nu , \sigma _ { w } ^ { 2 } , B _ { w } ;$ the exploration distribution and the safe model/excitation bounds used in $B _ { \psi } , \kappa _ { G } , E _ { \theta } ;$ H satisfying (93), (51), and (142); the ofline union-class constants defining h<sub>U</sub> and $\underline { { \Delta } } _ { \mathsf { U } }$ in (138); the model-class bounds $r _ { B } , R _ { A }$ in (128)–(129) when the non-Schur route is used; the finite-horizon constants in (140); and $m _ { \mathsf { U } } , D _ { \mathsf { U } }$ from (143). Set $\zeta _ { H } = E _ { \lambda } ( H , \delta _ { 0 } ) B _ { \psi }$

Initialize the restart time $r \gets 0$ and clear the energy bufer.

Loop. While $r < T$ , execute one cycle.

(a) Explore. For $t = r , \ldots$ , min $\{ r + H - 1 , T - 1 \}$ , draw a fresh exploratory input, apply it, and observe $y _ { t + 1 }$ . If the horizon is reached during this step, stop.

(b) Estimate and initialize. From $\mathcal { T } _ { r } = \{ r + r _ { 0 } , \ldots , r + H - 1 \}$ , compute

$$
\begin{array} { r l r l r l } & { V _ { r } ^ { \theta } : = \nu I _ { s } + \displaystyle \sum _ { t \in \mathcal { T } _ { r } } \phi _ { t } \phi _ { t } ^ { \mathsf { T } } , \qquad } & & { \widetilde { \theta } _ { r } : = ( V _ { r } ^ { \theta } ) ^ { - 1 } \displaystyle \sum _ { t \in \mathcal { T } _ { r } } \phi _ { t } y _ { t + 1 } , } \\ & { \widehat { \theta } _ { r } : = \Pi _ { \Theta } ( \widetilde { \theta } _ { r } ) , \qquad } & & { \widehat { b } _ { r } : = b _ { 1 } ( \widehat { \theta } _ { r } ) , } & & { \overline { { \lambda } } _ { r } : = \lambda ( \widehat { \theta } _ { r } ) . } \end{array}
$$

Initialize

$$
V _ { r + H } : = \nu I _ { d } + \sum _ { t \in \mathcal { Z } _ { r } } \psi _ { t } { \psi _ { t } ^ { \mathsf { T } } } , \qquad \widehat { \lambda } _ { r + H } : = \overline { { \lambda } } _ { r } .
$$

Clear the energy bufer and declare the predictable output-time grid

$$
S _ { r } : = \{ r + H + h _ { \mathsf { U } } + j h _ { \mathsf { U } } : j = 0 , 1 , . . . \} .
$$

(c) Exploit and monitor. For $t = r + H , r + H + 1 , \ldots , T - 1$ , form

$$
\begin{array} { r } { z _ { t } : = \left\{ \begin{array} { l l } { \widehat { \lambda } _ { t } ^ { \mathsf { T } } \psi _ { t } , } & { | ( \widehat { \lambda } _ { t } - \overline { { \lambda } } _ { r } ) ^ { \mathsf { T } } \psi _ { t } | \leq g _ { H } \left. \psi _ { t } \right. , } \\ { \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad u _ { t } : = \mathrm { c l i p } _ { [ - B _ { u } , B _ { u } ] } ( z _ { t } ) . } \end{array}
$$

Before observing $y _ { t + 1 }$ , mark that output as energy-eligible exactly when $t + 1 \in S _ { r }$ Apply $u _ { t } .$ , observe $y _ { t + 1 }$ , and form $\psi _ { t + 1 }$ . Make the RLS update

$$
\begin{array} { l l } { { x _ { t + 1 } ^ { \lambda } : = u _ { t } - \frac { y _ { t + 1 } } { \widehat { b } _ { r } } , \qquad } } & { { V _ { t + 1 } : = V _ { t } + \psi _ { t } \psi _ { t } ^ { \mathsf { T } } , } } \\ { { \widehat { \lambda } _ { t + 1 } : = \widehat { \lambda } _ { t } + V _ { t + 1 } ^ { - 1 } \psi _ { t } ( x _ { t + 1 } ^ { \lambda } - \widehat { \lambda } _ { t } ^ { \mathsf { T } } \psi _ { t } ) . } } & { { } } \end{array}
$$

Set the boundary flag

$$
B _ { t + 1 } : = \mathbf { 1 } \Big \{ | \overline { { \lambda } } _ { r } ^ { \mathsf { T } } \psi _ { t + 1 } | > B _ { u } + \zeta _ { H } \Big \} .
$$

The strict inequality is part of the algorithm. If $y _ { t + 1 }$ was marked eligible, append $y _ { t + 1 } ^ { 2 }$ to the energy bufer and retain only the most recent m<sub>U</sub> stored values. Set

$$
E _ { t + 1 } : = \left\{ \begin{array} { l l } { \displaystyle \mathbf { 1 } \left\{ | \mathrm { b u f f e r } | = m _ { \mathsf { U } } , ~ \frac { 1 } { m _ { \mathsf { U } } } \sum _ { z \in \mathrm { b u f f e r } } z \geq \sigma _ { w } ^ { 2 } + \frac { \Delta _ { \mathsf { U } } } { 2 } \right\} , } & { t + 1 \in \mathcal { S } _ { r } , } \\ { 0 , } & { t + 1 \not \in \mathcal { S } _ { r } . } \end{array} \right.
$$

If $B _ { t + 1 } = 1$ or $E _ { t + 1 } = 1$ , raise one alarm at output time $t + 1$ , discard the current estimates, set $r \gets t + 1$ , clear the bufer, and return to step (a). If both flags equal one, still record only one alarm. If neither flag crosses and $t = T - 1$ , stop; otherwise continue exploitation.

Eligibility is declared before $y _ { t + 1 }$ is observed, so the sampling grid is predictable. The boundary flag is checked after every exploitation output, not only at energy-sampling times. It is therefore available before the next control is chosen.

## L.2 Common Tuning

Under Assumption 26, (138) specifies a common lag $h _ { \mathsf { U } }$ and separation $\underline { { \Delta } } _ { \mathsf { U } }$ . For the excess innovation power $G _ { h }$ defined in Appendix K, (139) gives

$$
G _ { h _ { \mathsf { U } } } ( A ) \geq \underline { { \Delta } } _ { \mathsf { U } }
$$

for every admitted mismatch $A = { \cal A } _ { \mathrm { c l } } ( \lambda ( \theta ^ { - } ) , \theta ^ { + } )$ . The finite-horizon bounds in (140)– (141), the accuracy conditions (142), and the window in (143) therefore provide one calibration for all three detection conditions.

## L.3 Detection and Regret Guarantee

Theorem 31 (Branch-free union-class detector and regret guarantee). Retain Assumptions 7, 8, and 9. Replace global cross-regime clipping feasibility and stable mismatch by closed own-regime feasibility (Assumption $\it { 2 4 } )$ and the union stable/direct/non-Schur condition (Assumption 26), with the common separation (139) and finite-horizon bounds (140). Choose H as the smallest feasible integer satisfying (93), (51), and (142); choose m<sub>U</sub>, $D _ { \mathsf { U } }$ by (143); and read Assumption 12 with $D _ { \mathsf { U } }$ in place of D.

Then Algorithm 3 satisfies, with probability at least $1 - \delta ,$ , no false alarms and exactly one alarm ${ \widehat { \tau } } ^ { \left( c \right) }$ for every true change, with

$$
\tau ^ { ( c ) } < { \widehat { \tau } } ^ { ( c ) } \leq \tau ^ { ( c ) } + D _ { \mathsf { U } } ,\tag{155}
$$

and

$$
\mathcal { R } _ { T } \leq C K _ { D } D \mathsf { u } + ( C + 1 ) K _ { X } H + ( C + 1 ) R _ { E } ( T , \delta _ { 0 } ) .\tag{156}
$$

For fixed problem constants and fixed C, $\mathcal { R } _ { T } = O ( \log ( ( T + 1 ) / \delta ) )$ . More generally,

$$
\mathcal { R } _ { T } = O \bigg ( ( C + 1 ) \log \frac { T + 1 } { \delta } \bigg ) .\tag{157}
$$

Proof. If $T \leq H$ , the dwell condition forces $C = 0$ . Algorithm 3 performs only truncated exploration and stops, so $\mathcal { R } _ { T } \leq K _ { X } T \leq K _ { X } H$ , which is bounded by the right-hand side of (156). Hence assume $T > H$ below.

Read the restart, reference, and detector events from Theorem 18 and Appendix I using the first composite crossing and $D _ { \mathsf { U } }$ in place of D. Fix a potential restart r and test endpoint e for which the reference parameter remains active throughout $[ r , e )$ . Condition on an $\mathcal { F } _ { r + H }$ history with a full exploration block and $\mathcal G _ { r } ^ { \mathrm { e s t } }$ . Since $H > r _ { 0 } .$ , the action and boundary-test regressors are own-regime successors throughout the unchanged exploitation interval. Lemma 25 therefore excludes boundary crossings and gives $e _ { t } ^ { 2 } \leq a _ { t } ^ { 2 }$ , as required by Theorem 19. This includes a test at output time $e = \tau$ when the first subsequent change occurs at control time τ .

For each fixed null energy block, the proof of Lemma 15 applies with $( \Delta , B _ { o } , m )$ replaced by $( \underline { { \Delta } } _ { \mathsf { U } } , B _ { \mathsf { U } } , m _ { \mathsf { U } } )$ . Indeed, $| v _ { s } | \leq \varepsilon _ { y } \leq \overline { { \varepsilon } } _ { y } , \overline { { \varepsilon } } _ { y } ^ { 2 } \leq \underline { { \Delta } } _ { \mathsf { U } } / 8$ , and $B _ { \mathsf { U } } \geq \operatorname* { m a x } \{ B _ { w } , \overline { { \varepsilon } } _ { y } \}$ . The conditional crossing probability is bounded by

$$
\exp \left( - \frac { m _ { \mathsf { U } } \underline { { \Delta } } _ { \mathsf { U } } ^ { 2 } } { 3 2 B _ { w } ^ { 4 } } \right) + \exp \left( - \frac { m _ { \mathsf { U } } \underline { { \Delta } } _ { \mathsf { U } } ^ { 2 } } { 5 1 2 B _ { w } ^ { 2 } \overline { { \varepsilon } } _ { y } ^ { 2 } } \right) \leq \frac { \delta _ { d } } { 2 T ^ { 2 } } ,
$$

where the second term is omitted when $\overline { { \varepsilon } } _ { y } = 0$ . If the actual test is reached, its trajectory agrees with the alarm-suppressed continuation through that test. The bound therefore controls the joint event that the reference succeeds, the test is reached with the stated null history, and the threshold is crossed. It does not condition on survival to the test. Before the first detector failure, every false-alarm candidate has such an unchanged null history.

After a change, any earlier crossing of either test completes detection. Otherwise, the absence of a boundary crossing bounds the old ideal action’s saturation deficit by $2 \zeta _ { H }$ and the appliedaction error by $\overline { { \varepsilon } } _ { u }$ . Lemma 27 then compares each output in the complete post-change block with its h<sub>U</sub>-step auxiliary. These auxiliaries are defined on the full alarm-suppressed continuation, so Lemma 14 applies without conditioning on survival of either test. The event inclusion in Lemma 28 gives a composite alarm by $\tau + D _ { \mathsf { U } }$ except with conditional probability $\delta _ { d } / ( 4 T ^ { 2 } )$ on each eligible $\mathcal { F } _ { \tau }$ history.

The null bounds are unioned over at most $T ^ { 2 }$ restart–endpoint pairs. At a fixed true change, the successful-history active-cycle events are disjoint $\mathcal { F } _ { \tau } { \mathrm { - m e a s u r a b l e } }$ subsets of the corresponding $A _ { r , \tau }$ . Multiplying the conditional missed-detection bound by these indicators and summing over r costs at most $\delta _ { d } / ( 4 T ^ { 2 } )$ for that change. The chronological induction in Theorem 18 and Appendix I therefore applies with total detector failure probability at most $\delta _ { d } ;$ boundary crossings add no null failure probability on successful-reference histories. Controller failures cost at most $3 ( T + 1 ) \delta _ { 0 }$ , so the allocation in (48) is below δ. The delay intervals cost at most $C K _ { D } D _ { \mathsf { U } }$ , exploration costs at most $( C + 1 ) K _ { X } H$ , and unchanged exploitation prefixes cost at most $( C + 1 ) R _ { E } ( T , \delta _ { 0 } )$ . This proves (155)–(156).

It remains to check the order. Because $\bar { \varepsilon } _ { u } = 7 L _ { \lambda } B _ { \psi } E _ { \theta } ( H , \delta _ { 0 } )$ and $\overline { { \varepsilon } } _ { y } = 7 \overline { { b } } L _ { \lambda } B _ { \psi } E _ { \theta } ( H , \delta _ { 0 } )$ , the detector conditions follow from

$$
\begin{array} { r } { E _ { \theta } ( H , \delta _ { 0 } ) \le \epsilon _ { \cup } ^ { \mathrm { u n i } } : = \operatorname* { m i n } \Bigg \{ \frac { b _ { \operatorname* { m i n } } } { 1 6 } , \frac { \sqrt { \underline { { \Delta } } _ { \mathsf { U } } / 8 } } { 7 \overline { { b } } L _ { \lambda } B _ { \psi } } , \frac { B _ { \mathsf { U } } } { 7 \overline { { b } } L _ { \lambda } B _ { \psi } } , } \\ { \frac { \displaystyle \underline { { \Delta } } _ { \mathsf { U } } } { 1 1 2 B _ { \mathsf { U } } L _ { u , \mathsf { U } } L _ { \lambda } B _ { \psi } } \Bigg \} , } \end{array}\tag{158}
$$

Omit the last entry when $L _ { u , \cup } = 0$ . Appendix J and (143) then prove (157).