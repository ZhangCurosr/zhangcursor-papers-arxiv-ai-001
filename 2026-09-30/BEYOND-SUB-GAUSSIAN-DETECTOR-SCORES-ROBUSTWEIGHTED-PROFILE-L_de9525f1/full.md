# BEYOND SUB-GAUSSIAN DETECTOR SCORES: ROBUSTWEIGHTED PROFILE-LOSS CHANGE POINT DETECTIONFOR HUMAN-LLM TEXT SEGMENTATION

Wan Tian<sup>1\*</sup> Zhongyi Li<sup>2\*</sup> Yawen Li<sup>3</sup> Rui Zhang<sup>4</sup> Yijie Peng<sup>5†</sup> Fuzhen Zhuang<sup>2†</sup>

<sup>1</sup>Peking University <sup>2</sup>Beihang University <sup>3</sup>Capital Normal University

<sup>4</sup>Renmin University of China <sup>5</sup>Nanjing University

<sup>\*</sup>These authors contributed equally to this work. <sup>†</sup>Corresponding authors.

Correspondence: pengyijie@nju.edu.cn; zhuangfuzhen@buaa.edu.cn

## ABSTRACT

Mixed human-LLM documents require locating authorship transitions from detector scores whose reliability varies across text units. Existing weighted mean contrasts are vulnerable to extreme scores, while directly replacing means with robust centers obscures how a misplaced boundary changes the population objective. We propose Robust Weighted Profile-Loss Change Point Detection (RWCP), which combines capped reliability weights, Huber profile gains, and narrowestover-threshold search in reliability coordinates. Our key analysis expresses the population gap between a true and a displaced split as a merge cost, avoiding a closed-form solution for the nonlinear center of a mixed segment. Under explicit curvature, spacing, and dependence conditions, core RWCP recovers the number of changes and localizes their boundaries; its quadratic-loss limit recovers squared weighted CUSUM. We also study RWCP-R, a separately evaluated decoder that shares source centers across nonadjacent passages. Across five retrospective cached-score benchmark families, core RWCP reduces family-macro WindowDiff by 17.6% relative to weighted change-point detection, and RWCP-R lowers it further. Boundary recovery improves most clearly for isolated changes, while both fixed configurations miss changes in collaborative and densely alternating text.

## 1 INTRODUCTION

Mixed human–LLM writing calls for locating authorship changes rather than labeling an entire document. Human passages can alternate with generated continuations or model-assisted edits, so the output is a sequence of source-homogeneous segments and boundaries. Mapping each sentence or short unit $X _ { i }$ to a detector score $Y _ { i } = { \overline { { \phi } } } ( X _ { i } )$ makes this an ordered change-point problem. WCP uses the unequal reliability of these scores (Li et al., 2026), but weighting alone does not control extreme observations or context-induced dependence.

Rare tokens, code, equations, paraphrases, and short units can yield extreme scores, while underestimated variance proxies can give a single unit excessive weight. Huber scores bound their influence, but substituting Huber centers into mean-based CUSUM does not preserve its population decomposition: an incorrect split mixes two source locations, whose optimum is generally not their weighted arithmetic mean. The analysis must compare fitted segments without assuming a formula for that mixed center.

The profile gain is the reduction in optimized Huber loss obtained by splitting an interval. With one change, its population difference between the true and a displaced boundary equals a merge cost; local risk curvature makes this gap proportional to misplaced reliability mass and squared jump size. Capped weights control leverage, bounded Huber scores support dependence-aware concentration without sub-Gaussian raw scores, and reliability-coordinate intervals isolate changes by information mass. Guarded recursion converts these properties into count and localization guarantees.

![](images/bd456ba7dcf6a7a86dc6d6960d0e2dfb1803af7a8a431c6251a3a1aa9b9b5738.jpg)  
Figure 1: Core RWCP: capped reliability weights, Huber profile gains, and weighted-coordinate narrowest-over-threshold search. The recovery theorem covers this procedure. RWCP-R separately decodes recurrent source states using (6).

Core RWCP combines robust weighted profile gains with weighted-coordinate narrowest-overthreshold search; in the unconstrained quadratic limit, its gain is exactly squared weighted CUSUM. We prove finite-sample exact count recovery and weighted localization under explicit curvature, central-mass, spacing, and dependence conditions, including the logarithmic cost of geometric mixing. The merge-cost analysis distinguishes this result from a direct robust-center substitution in an existing contrast.

Recurring human and LLM passages offer a second opportunity: nonadjacent passages can share a source state. We study RWCP-R, a distinct decoder with shared centers and a fitted residual scale. We now evaluate an analytic-threshold implementation of core RWCP on the same cached scores as WCP before assessing RWCP-R separately. Core RWCP reduces window error most clearly on controlled single-change texts, but its fixed threshold often misses changes in real collaborative and dense-change documents; RWCP-R improves average window error further without removing this count-recovery limitation. The theorem covers core RWCP under its stated assumptions, not the empirical RWCP-R decoder or the claim that either fixed configuration satisfies those assumptions on detector scores.

Huber profile costs are established in penalized segmentation (Fearnhead & Rigaill, 2019), narrowestover-threshold search supplies interval selection (Baranowski et al., 2019), and WCP supplies the reliability-weighted detector-score setting (Li et al., 2026). Our technical step is to analyze the weighted difference of optimized profile gains through a merge-cost margin and carry it through dependence-aware control, reliability-coordinate isolation, and guarded recursion. Related work, full protocols, and proofs appear in Appendices A–C.

## 2 ROBUST WEIGHTED PROFILE SEGMENTATION

We first define a robust gain for testing whether one location or two better describe an interval. We then use that gain in reliability-coordinate interval search; a separate recurrent objective shares centers across passages assigned to the same source state.

## 2.1 ROBUST LOCATIONS, RELIABILITY, AND PROFILE GAINS

Let $0 = \tau _ { 0 } < \tau _ { 1 } < \cdots < \tau _ { K } < \tau _ { K + 1 } = N$ denote unknown boundaries. Observation i has a deterministic reliability weight $w _ { i } > 0$ , scale $s _ { i } = w _ { i } ^ { - 1 / 2 }$ , and segment index z(i). For cutoff $c > 0$ the Huber loss and score are

$$
\begin{array} { r } { \rho _ { c } ( u ) = \left\{ \begin{array} { l l } { u ^ { 2 } / 2 , } & { | u | \leq c , } \\ { c | u | - c ^ { 2 } / 2 , } & { | u | > c , } \end{array} \right. \quad \psi _ { c } ( u ) = \mathrm { s i g n } ( u ) \operatorname* { m i n } \{ | u | , c \} . } \end{array}
$$

Definition 2.1 (Piecewise common Huber location). There are locations $\theta _ { 1 } , \ldots , \theta _ { K + 1 }$ in a compact interval Θ such that, whenever $\tau _ { j - 1 } < i \leq \tau _ { j } , \theta _ { j }$ minimizes $Q _ { i } ( \theta ) = \mathbb { E } \rho _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta ) \}$ }. For interior unique minimizers this is equivalent to

$$
\mathbb { E } [ \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta _ { j } ) \} ] = 0 .\tag{1}
$$

Adjacent jump sizes are $\kappa _ { j } = | \theta _ { j + 1 } - \theta _ { j } |$

The location is defined by the detector-score risk rather than by an ordinary expectation. A symmetric location family $Y _ { i } =  { \theta _ { z ( i ) } } + \sigma _ { i } \xi _ { i }$ is sufficient, but asymmetric distributions are allowed whenever the centering condition in (1) holds. Given a variance proxy $\sigma _ { i } ^ { 2 }$ , the capped oracle weight is

$$
w _ { i } ^ { \circ } = \Pi _ { [ w _ { \operatorname* { m i n } } , w _ { \operatorname* { m a x } } ] } ( \sigma _ { i } ^ { - 2 } ) , \qquad \Pi _ { [ a , b ] } ( u ) = \operatorname* { m i n } \{ \operatorname* { m a x } ( u , a ) , b \} .\tag{2}
$$

Practical weights replace $\sigma _ { i } ^ { 2 }$ by an externally estimated proxy and add a small denominator ridge. Capping in (2) prevents a single underestimated variance from receiving unbounded leverage. Write $\begin{array} { r } { S _ { a : b } ^ { w } = \sum _ { i = a } ^ { b } w _ { i } \mathrm { ~ a n d ~ } d _ { w } ( u , v ) = \sum _ { \operatorname* { m i n } ( u , v ) < i \leq \operatorname* { m a x } ( u , v ) } w _ { i } . } \end{array}$

For an interval $A ,$ define $\begin{array} { r } { \widehat { \mathcal { L } } _ { A } ( \theta ) = \sum _ { i \in A } \rho _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta ) \} } \end{array}$ and let $\widehat { \theta } _ { A }$ minimize it over $\Theta .$ For $s \leq b < e ,$ RWCP uses the profile improvement

$$
\widehat { \Gamma } _ { s , e } ( b ) = 2 \Big \{ \widehat { \mathcal { L } } _ { s : e } ( \widehat { \theta } _ { s : e } ) - \widehat { \mathcal { L } } _ { s : b } ( \widehat { \theta } _ { s : b } ) - \widehat { \mathcal { L } } _ { b + 1 : e } ( \widehat { \theta } _ { b + 1 : e } ) \Big \} .\tag{3}
$$

The gain in (3) is nonnegative because the one-location model is nested in the split model. Its population analogue makes a displaced split pay for merging two distinct source locations, the mechanism quantified in Theorem 3.8.

Proposition 2.2 (Quadratic-loss reduction). $I f \rho _ { c }$ is replaced by $\rho _ { \infty } ( u ) = u ^ { 2 } / 2$ and the three fits are unconstrained, or their weighted means all lie in Θ, then

$$
\widehat { \Gamma } _ { s , e } ( b ) = \frac { S _ { s : b } ^ { w } S _ { b + 1 : e } ^ { w } } { S _ { s : e } ^ { w } } \left( \bar { Y } _ { s : b } ^ { w } - \bar { Y } _ { b + 1 : e } ^ { w } \right) ^ { 2 } = \{ W _ { s , e } ^ { Y } ( b ) \} ^ { 2 } ,\tag{4}
$$

where $\begin{array} { r } { \bar { Y } _ { a : b } ^ { w } = \sum _ { i = a } ^ { b } w _ { i } Y _ { i } / S _ { a : b } ^ { w } } \end{array}$ and $W _ { s , e } ^ { Y } ( b )$ is the weighted CUSUM.

Active parameter constraints introduce projection terms (Appendix C.3.2). Thus RWCP retains the familiar WCP contrast when its loss is quadratic, while the finite-cutoff analysis uses bounded Huber scores in place of a raw-score tail condition.

## 2.2 WEIGHTED-COORDINATE SEARCH AND RECURRENT STATES

The reliability mass used by the profile gain also defines the search geometry. RWCP samples intervals uniformly in the cumulative coordinate $\begin{array} { r } { W ( t ) = \sum _ { i \leq t } w _ { i } . } \end{array}$ , rather than uniformly in sentence index. Within $I = [ s , e ]$ , it scans splits whose two sides each have mass at least $m _ { N }$ , computes $T ( I ) = \operatorname* { m a x } _ { b } \widehat { \Gamma } _ { s , e } ( b )$ , and retains intervals with $T ( I ) > r _ { N }$ . Among them it selects the interval with the least reliability mass, reports its maximizing split $b ^ { \star }$ , and recurses. This is the narrowestover-threshold principle (Baranowski et al., 2019) in the information geometry supplied by the detector.

After accepting a boundary, a guard of mass $g _ { N }$ prevents duplicate discoveries while preserving well-separated neighboring changes. For a current search range $[ a , b ]$ , define the nearest mass crossings

$$
\begin{array} { l l } { { t _ { L } = \operatorname* { m a x } \{ t \in [ a , b ^ { \star } ] : S _ { t : b ^ { \star } } ^ { w } \geq g _ { N } \} , } } & { { \ell _ { g } = t _ { L } - 1 , } } \\ { { t _ { R } = \operatorname* { m i n } \{ t \in [ b ^ { \star } + 1 , b ] : S _ { b ^ { \star } + 1 : t } ^ { w } \geq g _ { N } \} , } } & { { r _ { g } = t _ { R } + 1 . } } \end{array}\tag{5}
$$

If the left or right crossing does not exist, set $\ell _ { g } = a - 1 \mathrm { o r } r _ { g } = b + 1$ , respectively. Recursion continues on the nonempty ranges $[ a , \ell _ { g } ]$ and $[ r _ { g } , b ]$ . Each removed side has mass below $g _ { N } + w _ { \mathrm { m a x } }$ The theorem uses the analytic threshold $r _ { N } \doteq \mathbf { \dot { C } } _ { r } \mathbf { \dot { M } } _ { N } ( \delta )$ . In practice, a block-multiplier alternative forms clipped influence residuals from a conservative pilot, multiplies overlapping residual blocks by Gaussian weights, and takes the conditional quantile of the maximum over the same fixed candidate collection. This calibration is not part of the recovery theorem.

When source states recur, nonadjacent passages can inform a common center. RWCP-R uses this structure through a separate recurrent-state objective; it replaces the core interval search and does not invoke its threshold or guard. With robustly standardized scores $\widetilde { Y } _ { i }$ , compressed capped weights $\widetilde { w } _ { i }$ $S$ source states, centers $\mu ,$ and scale $\sigma \geq \sigma _ { \operatorname* { m i n } }$ , it seeks a local minimum of

$$
\mathcal { I } ( z , \mu , \sigma ) = \sum _ { i = 1 } ^ { N } \rho _ { c } \left( \frac { \sqrt { \widetilde { w } _ { i } } ( \widetilde { Y } _ { i } - \mu _ { z _ { i } } ) } { \sigma } \right) + N \log { \sigma } + \lambda \sum _ { i = 2 } ^ { N } 1 \{ z _ { i } \neq z _ { i - 1 } \} .\tag{6}
$$

For fixed centers and scale, Viterbi decoding solves the state update in $O ( N S ^ { 2 } )$ ; for fixed states, weighted Huber fits update the centers and a monotone score equation updates the scale. Exact conditional updates cannot increase (6); the joint problem is nonconvex and the finite-iteration solver has no global-optimality guarantee. Here $\sigma$ is common to all states. With residuals $r _ { i } =$ $\sqrt { \widetilde { w } _ { i } } ( \widetilde { Y } _ { i } - \mu _ { z _ { i } } ) / \sigma ,$ an interior scale update solves $\begin{array} { r } { \sum _ { i } \psi _ { c } ( r _ { i } ) r _ { i } = N ; } \end{array}$ ; otherwise the constrained optimum is $\sigma _ { \mathrm { m i n } } .$ . We declare $S = 3$ for CoAuthor and ${ \dot { S } } = 2$ for the controlled tasks. Boundaries are state transitions; S specifies source classes, not their number of occurrences. Appendix B.3 gives the finite-iteration algorithm. Section 3 analyzes core RWCP; Section 4 evaluates the recurrent decoder as a separate empirical extension.

## 3 THEORETICAL GUARANTEES

The recovery argument follows the algorithm’s four steps: profile-loss geometry supplies a population margin; bounded Huber scores and empirical curvature preserve it in the observed data; reliabilitycoordinate intervals isolate each change; and a guard prevents duplicate detections during recursion. We state the assumptions and main result here, with verification results, lower bounds, constants, and proofs in Appendix C.

Assumption 3.1 (Robust locations). Definition 2.1 and (1) hold with finite risks. There is a fixed radius $r _ { 0 } > 0$ such that dist $( \theta _ { j } , \partial \Theta ) \ge r _ { 0 }$ for every true center.

Assumption 3.2 (Risk curvature). There are fixed $0 < m _ { 0 } \le M _ { 0 } < \infty$ such that, for every i in segment j and every $| \theta - \theta _ { j } | \leq r _ { 0 }$

$$
\frac { m _ { 0 } w _ { i } } { 2 } ( \theta - \theta _ { j } ) ^ { 2 } \leq Q _ { i } ( \theta ) - Q _ { i } ( \theta _ { j } ) \leq \frac { M _ { 0 } w _ { i } } { 2 } ( \theta - \theta _ { j } ) ^ { 2 } .\tag{7}
$$

Assumption 3.3 (Reliability). The deterministic weights satisfy $0 < w _ { \mathrm { m i n } } \le w _ { i } \le w _ { \mathrm { m a x } } < \infty .$ , and the minimum segment mass $\begin{array} { r } { \Delta _ { w } = \operatorname* { m i n } _ { j } S _ { \tau _ { j - 1 } + 1 : \tau _ { j } } ^ { w } } \end{array}$ obeys $w _ { \mathrm { m a x } } \leq \Delta _ { w } / 2 4$

Assumption 3.4 (Dependence). Every centered bounded transform $V _ { i } = f _ { i } ( Y _ { i } )$ used for scores or central-mass indicators obeys, on every interval A,

$$
\mathbb { P } \Bigg ( \Bigg | \sum _ { i \in A } V _ { i } \Bigg | > C _ { \mathrm { d e p } } \| V \| _ { \infty } \{ \sqrt { | A | x } + q _ { N } x \} \Bigg ) \leq 2 e ^ { - x } , \qquad x \geq 1 .
$$

For a two-sided extension, absolute regularity is measured by

$$
\beta ( k ) = \operatorname* { s u p } _ { t } \mathbb { E } \left[ \operatorname* { s u p } _ { B \in \mathcal { F } _ { t + k } ^ { \infty } } \left| \mathbb { P } ( B \mid \mathcal { F } _ { - \infty } ^ { t } ) - \mathbb { P } ( B ) \right| \right] .\tag{8}
$$

Independent and fixed-order dependent observations permit $q _ { N } = O ( 1 )$ ; the conservative geometricmixing specialization $\beta ( k ) \lesssim e ^ { - c k }$ uses $q _ { N } = O \{ \log ^ { 2 } ( e N ) \}$

Assumption 3.5 (Central mass). There is a fixed $p _ { 0 } \in ( 0 , 1 ]$ ] such that every observation i in segment $j$ and every $| u | \leq r _ { 0 }$ satisfy $\mathbb { P } \big ( | \sqrt { w _ { i } } ( Y _ { i } - \theta _ { j } - u ) | \le c / 2 \big ) \ : \ge p _ { 0 }$ . For $K \geq 1$ , we impose the local-jump restriction $\kappa _ { \mathrm { m a x } } \leq p _ { 0 } r _ { 0 } / 1 6 ;$ ; for $K = 0$ , set $\kappa _ { \operatorname* { m a x } } = 0$

Let $\mathcal { A } _ { N }$ contain all nonempty index intervals and let $L _ { N } = 1 + N ^ { 2 } + N ^ { 3 }$ bound their number together with all split triples. Write $\dot { x } _ { N } ( \delta ) = \log ( 6 4 L _ { N } / \delta )$ and define

$$
\Lambda _ { N } ( \delta ) = q _ { N } ^ { 2 } \log \biggl ( \frac { 6 4 L _ { N } } { \delta } \biggr ) .
$$

Set $m _ { \mathrm { c u r v } } = C _ { \mathrm { c u r v } } \Lambda _ { N } ( \delta )$

Assumption 3.6 (Spacing and signal). For $K \geq 1$ , set $\kappa _ { \mathrm { m i n } } = \operatorname* { m i n } _ { j } \kappa _ { j } > 0$ and $\kappa _ { \mathrm { m a x } } = \operatorname* { m a x } _ { j } \kappa _ { j }$ For constants $c _ { m , 1 } , C _ { g } , C _ { \mathrm { s n r } } > 0$ and $0 < c _ { m , 2 } \leq 1 / 1 2$ , the admissible side mass $m _ { N }$ , guard mass $g _ { N }$ , and signal obey

$$
\begin{array} { r l } { m _ { \mathrm { c u r v } } \leq m _ { N } \leq c _ { m , 1 } \displaystyle \frac { \Lambda _ { N } ( \delta ) } { \kappa _ { \operatorname* { m a x } } ^ { 2 } } , \quad } & { C _ { g } \displaystyle \frac { \Lambda _ { N } ( \delta ) } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } + w _ { \operatorname* { m a x } } \leq g _ { N } \leq c _ { m , 2 } \Delta _ { w } , } \\ { m _ { N } \leq \displaystyle \frac { \Delta _ { w } } { 1 2 } , \quad } & { \kappa _ { \operatorname* { m i n } } ^ { 2 } \Delta _ { w } \geq C _ { \mathrm { s n r } } \Lambda _ { N } ( \delta ) . } \end{array}
$$

For adjacent pure intervals $A , B ,$ let $D ^ { \star } ( A , B )$ be the excess population loss caused by fitting their union with one location instead of two

Lemma 3.7 (Mixed-segment excess risk). IfA and B have locations $\theta _ { A } , \theta _ { B }$ in the common curvature region, then

$$
D ^ { \star } ( A , B ) \geq \frac { m _ { 0 } } { 2 } \frac { S _ { A } ^ { w } S _ { B } ^ { w } } { S _ { A } ^ { w } + S _ { B } ^ { w } } ( \theta _ { A } - \theta _ { B } ) ^ { 2 } ,\tag{9}
$$

$$
D ^ { \star } ( A , B ) \leq \frac { M _ { 0 } } { 2 } \frac { S _ { A } ^ { w } S _ { B } ^ { w } } { S _ { A } ^ { w } + S _ { B } ^ { w } } ( \theta _ { A } - \theta _ { B } ) ^ { 2 } .\tag{10}
$$

Theorem 3.8 (Single-change population margin). Suppose $[ s , e ]$ contains one change τ of size κ. The population counterpart of(3) is maximized at τ. For $b < \tau$

$$
\Gamma _ { s , e } ^ { \star } ( \tau ) - \Gamma _ { s , e } ^ { \star } ( b ) = 2 D ^ { \star } ( [ b + 1 , \tau ] , [ \tau + 1 , e ] ) \ge m _ { 0 } \kappa ^ { 2 } \frac { S _ { b + 1 : \tau } ^ { w } S _ { \tau + 1 : e } ^ { w } } { S _ { b + 1 : \tau } ^ { w } + S _ { \tau + 1 : e } ^ { w } } ,\tag{11}
$$

with the symmetric boundfor $b > \tau$ . When the misclassified mass is no larger than the opposite pure-side mass, the gap is at least $m _ { 0 } \kappa ^ { 2 } d _ { w } ( b , \tau ) / 2$

The identity in (11) converts a difficult mixed-center calculation into an excess-risk comparison. Bounded Huber scores keep empirical same-location merge gains at $O \{ \Lambda _ { N } ( \delta ) \}$ , while differentlocation merges retain the curvature term in Lemma 3.7.

Lemma 3.9 (Uniform empirical merge costs). For adjacent intervals $A , B$ with $S _ { A } ^ { w } , S _ { B } ^ { w } \geq m _ { \mathrm { c u r v } } ,$ define

$$
\widehat { D } ( A , B ) = \operatorname* { i n f } _ { \theta } \{ \widehat { \mathcal { L } } _ { A } ( \theta ) + \widehat { \mathcal { L } } _ { B } ( \theta ) \} - \operatorname* { i n f } _ { \theta } \widehat { \mathcal { L } } _ { A } ( \theta ) - \operatorname* { i n f } _ { \theta } \widehat { \mathcal { L } } _ { B } ( \theta ) .
$$

On an event ofprobability at least $1 - \delta / 2 ,$ , uniformly over the master collection, intervalsfrom the same true segment with masses at least $m _ { \mathrm { c u r v } }$ satisfy

$$
0 \le \widehat { \cal D } ( A , B ) \le C _ { 0 } \Lambda _ { N } ( \delta ) .\tag{12}
$$

If instead A, B lie in adjacent segments separated by $\kappa ,$ then

$$
\widehat { D } ( A , B ) \geq c _ { 0 } \kappa ^ { 2 } \frac { S _ { A } ^ { w } S _ { B } ^ { w } } { S _ { A } ^ { w } + S _ { B } ^ { w } } - C _ { 1 } \Lambda _ { N } ( \delta ) ,\tag{13}
$$

$$
\widehat { D } ( A , B ) \leq C _ { 2 } \kappa ^ { 2 } \frac { S _ { A } ^ { w } S _ { B } ^ { w } } { S _ { A } ^ { w } + S _ { B } ^ { w } } + C _ { 3 } \Lambda _ { N } ( \delta ) .\tag{14}
$$

Thus the same gain separates homogeneous intervals from intervals containing a sufficiently strong change: bounded scores control the null merge cost, and empirical curvature preserves the $\kappa ^ { 2 } .$ weighted alternative up to a uniform fluctuation.

Let $W _ { N } = W ( N )$ denote the total reliability mass. Weighted-coordinate sampling supplies the algorithmic counterpart of the population margin.

Lemma 3.10 (Weighted-coordinate isolation). Under Assumption 3.3, suppose max<sub>i</sub> $w _ { i } \leq \Delta _ { w } / 2 4$ Ifthe number ofsampled intervals satisfies

$$
M \geq C _ { I } \left( \frac { W _ { N } } { \Delta _ { w } } \right) ^ { 2 } \log \left\{ \frac { 4 ( K \vee 1 ) } { \delta } \right\} ,\tag{15}
$$

then, with probability at least $1 - \delta / 4 ,$ , every change $\tau _ { j }$ is the only change in some sampled interval $[ s , e ]$ for which

$$
\frac { \Delta _ { w } } { 6 } \leq S _ { s : \tau _ { j } } ^ { w } , S _ { \tau _ { j } + 1 : e } ^ { w } \leq \frac { \Delta _ { w } } { 3 } , \qquad S _ { s : e } ^ { w } \leq \frac { 2 \Delta _ { w } } { 3 } .\tag{16}
$$

Theorem 3.11 (Finite-sample count recovery and localization). Under Assumptions 3.1–3.6, sample M weighted-coordinate intervals with M satisfying (15), and choose $r _ { N } = C _ { r } \Lambda _ { N } ( \delta )$ with the constants specified in Appendix C.2.3. Then, with probability at least $1 - \delta ,$ , core RWCP returns exactly K changes and, after ordering,

$$
d _ { w } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq C _ { \mathrm { l o c } } \frac { \Lambda _ { N } ( \delta ) } { \kappa _ { j } ^ { 2 } } , \qquad j \in [ K ] .\tag{17}
$$

For $K = 0 ,$ Assumptions 3.1–3.5, $m _ { N } \geq m _ { \mathrm { c u r v } } ,$ , and $r _ { N } > 2 C _ { 0 } \Lambda _ { N }$ sufficefor nofalse boundary;   
the jump-dependent spacing and interval-count conditions are unnecessary.

The theorem describes locally separated changes; feasible side mass requires $C _ { \mathrm { c u r v } } \kappa _ { \mathrm { m a x } } ^ { 2 } \leq c _ { m , 1 }$ , and $g _ { N } , M$ use signal/spacing bounds rather than fully adaptive tuning. In the proof, uniform merge-cost bounds suppress homogeneous intervals, while weighted isolation supplies a short over-threshold interval for each change. The narrowest such interval contains one change; the profile margin localizes it, and the guard removes it without losing neighboring isolating intervals. Induction yields exact K and simultaneous localization.

Corollary 3.12 (Estimated-weight transfer). Suppose a weight estimator obeys $c _ { w } w _ { i } ^ { \circ } \leq \widehat { w } _ { i } \leq C _ { w } w _ { i } ^ { \circ }$ uniformly with probability at least $1 - \delta _ { w }$ , and conditional score assumptions hold uniformly on this event. Then RWCP computed with $\widehat { w } _ { i }$ satisfies, with probability at least $1 - \delta - \delta _ { w }$

$$
\widehat { K } = K , \qquad d _ { w ^ { \circ } } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq c _ { w } ^ { - 1 } d _ { \widehat { w } } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq C \frac { \Lambda _ { N } ( \delta ) } { \kappa _ { j } ^ { 2 } } .
$$

The spacing, no-dominant-atom, and signal conditions change only by constants depending on $( c _ { w } , C _ { w } )$ .

Independent and fixed-order dependent scores attain weighted localization order $O \{ \log ( N / \delta ) / \kappa _ { j } ^ { 2 } \}$ ; geometric mixing incurs the explicit $q _ { N } ^ { 2 }$ factor. External proxies or sample splitting can justify Corollary 3.12; target-score-derived weights require separate joint analysis.

Under independent Gaussian scores and inactive caps, the two-point lower bound matches the log $N / \kappa ^ { 2 }$ order (Appendix C.3.1); it does not address dependence or active caps.

The weighted metric translates to text units: bounded weights make $d _ { w } ( u , v )$ comparable to the number of misplaced sentences, while $\sigma _ { i } ^ { 2 } \asymp n _ { i } ^ { - 1 }$ with inactive caps makes it comparable to misplaced tokens. Formal metric corollaries and the lower-bound argument appear in Appendix C.3.1.

## 4 EXPERIMENTS

## 4.1 PROTOCOL AND OVERALL COMPARISON

We compare an analytic-threshold implementation of core RWCP with WCP, then evaluate whether the separate recurrent decoder improves window segmentation and how both methods handle dense changes. Five cached-score families cover single changes in WikiQA, News, and Story; real CoAuthor sessions; multiple-change Story documents; text attacks; and varying LLM-written proportions. The suite has 25 settings and 2,690 instances, including all WikiQA records under the common singleton convention (Appendix B.1). Comparators are TextTiling, direct and sentence-level prediction, local voting, VCP, and WCP where outputs are available (Hearst, 1997; Zhang et al., 2024; Li et al., 2026). Core RWCP, RWCP-R, and WCP share cached scores and reliability proxies on nontrivial inputs; single-change experiments use fine-tuned scores and the other families use normalized scores. Core RWCP uses one fixed analytic-threshold configuration, specified in Appendix B.2; its profile search is the subject of Theorem 3.11, although these numerical settings are not certified to satisfy the theorem’s unknown constants and assumptions. RWCP-R uses a separately frozen configuration: Huber cutoff $c = 1 . 5$ , transition penalty $\lambda = 4$ , weight exponent $\eta = 0 . 5 ,$ , scale floor 0.25, and six deterministic starts. Appendix B.4 documents its selection on benchmarks that informed earlier development, making the comparisons retrospective.

We use standard raw-boundary WindowDiff (WD, lower is better) to measure window-level disagreement and raw count error $\mathrm { C E } _ { d } = K _ { d } - \widehat { K } _ { d }$ to show its direction. Positive CE indicates missed changes and negative CE indicates excess changes; count MAE, matched-boundary F1, and a noboundary reference complete the recovery assessment in Appendices B.2 and B.7.2. The inherited final-label window error $\mathrm { W D } _ { \mathrm { s u p } }$ is reported separately because it can exceed one and measures a different output.

Table 1: Raw-boundary results averaged over documents, then equally over settings within each family (300, 290, 1,000, 600, and 500 instances). Core RWCP uses the fixed analytic threshold; the historical RWCP-analytic/block variants in Appendix B.7 use additional practical refinements. Bold marks the lowest unrounded WD, not significance; CE is signed count bias. Dashes denote unavailable complete-family results.
<table><tr><td rowspan="3">Method</td><td colspan="2">Single change</td><td colspan="2">CoAuthor</td><td colspan="2">Multiple changes</td><td colspan="2">Text attacks</td><td colspan="2">LLM proportion</td></tr><tr><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td></tr><tr><td>TextTiling</td><td>0.6449</td><td>-2.487</td><td>0.5445</td><td>7.314</td><td>0.6251</td><td>-4.585</td><td>0.6421</td><td>-2.478</td><td>0.7611</td><td>-3.730</td></tr><tr><td>LLMPred</td><td>0.4446</td><td>-0.807</td><td>0.4996</td><td>8.303</td><td>0.4967</td><td>-0.141</td><td>0.4508</td><td>-1.013</td><td>0.4074</td><td>-1.356</td></tr><tr><td>SenPred</td><td>0.6297</td><td>-6.717</td><td>0.5485</td><td>-4.155</td><td>0.9032</td><td>-23.498</td><td>0.6791</td><td>-7.390</td><td>0.9511</td><td>-14.500</td></tr><tr><td>Voting</td><td>0.2725</td><td>-0.903</td><td>0.5487</td><td>3.286</td><td>0.4797</td><td>-4.626</td><td>0.3514</td><td>-1.338</td><td>0.4939</td><td>-3.164</td></tr><tr><td>VCP</td><td>0.3086</td><td>-0.603</td><td>0.4913</td><td>9.369</td><td>0.4613</td><td>-0.579</td><td></td><td></td><td></td><td></td></tr><tr><td>WCP</td><td>0.3961</td><td>-1.403</td><td>0.4878</td><td>7.717</td><td>0.4363</td><td>0.772</td><td>0.3609</td><td>-0.358</td><td>0.3538</td><td>-0.804</td></tr><tr><td>Core RWCP</td><td>0.2315</td><td>0.220</td><td>0.4897</td><td>10.152</td><td>0.4108</td><td>3.716</td><td>0.3369</td><td>0.940</td><td>0.2085</td><td>0.920</td></tr><tr><td>RWCP-R</td><td>0.1895</td><td>0.280</td><td>0.4869</td><td>10.166</td><td>0.3993</td><td>3.181</td><td>0.2676</td><td>0.475</td><td>0.2119</td><td>0.600</td></tr></table>

Core RWCP lowers family-macro WD from 0.4070 for WCP to 0.3355 (17.6% relative) and improves WD in four of five families, but these window-level gains do not imply reliable boundary recovery. Its family-macro count MAE rises from 2.808 to 3.229 and matched-boundary F1 falls from 0.287 to 0.129. Single-change texts are the family-level exception: count MAE improves from 1.637 to 0.393 and F1 from 0.435 to 0.487, although F1 does not improve in every domain. On CoAuthor, multiplechange Story, text attacks, and LLM-proportion documents, core RWCP reports no boundary in 82.4%, 93.1%, 94.7%, and 92.8% of instances, respectively. Appendix Table 6 reports these diagnostics and conditional localization errors; no CoAuthor document has the correct positive change count under this fixed configuration. These core comparisons are descriptive; the paired intervals below concern RWCP-R.

RWCP-R has lower mean WD than WCP in all five families and lower WD than the fixed core implementation in four (Table 1). With equal family weights, WD falls from 0.4070 for WCP to 0.3110, a 23.6% relative reduction; legacy $\mathrm { W D _ { s u p } }$ falls from 0.3249 to 0.2990 (8.0%). The frozen RWCP-R configuration improves standard WD in 23 of 25 settings against WCP and in 20 against every available non-RWCP baseline. Nineteen pointwise paired 95% intervals for the WD difference against WCP lie below zero, one lies above zero, and five include or touch zero. The joint sourcecomponent bootstrap gives a family-macro difference of −0.0959 with interval [−0.1074, −0.0844] (Appendix Table 19); these exploratory intervals condition on the saved configurations and available source linkage.

Window-level agreement and recovery of individual changes answer different questions, so we interpret the two together. RWCP-R improves matched-boundary F1 over WCP on single-change and text-attack families, from 0.435 to 0.544 and from 0.272 to 0.300, respectively. On CoAuthor and dense-change Story, the fitted transition penalty yields fewer boundaries and lower F1 despite the WD gains; Appendix Table 16 reports the corresponding count MAE, F1, and no-boundary rates. This distinction motivates the density analysis below and identifies the regimes where the recurrent model’s window-level advantage translates most directly into localized boundaries.

## 4.2 CONTROLLED LOCALIZATION AND TEXT PERTURBATIONS

The single-change and attack settings test whether the same configuration transfers across three domains and two text perturbations (Table 2). RWCP-R attains the lowest available standard WD in all three unperturbed domains, all three decoherence settings, and two of three paraphrasing settings; Voting is better on paraphrased News. All nine paired WD intervals against WCP exclude zero in favor of RWCP-R. Count bias adds a useful qualification: RWCP-R is closer to zero than WCP on unperturbed News and Story, while it is more conservative on WikiQA and the decoherence settings.

Table 2: Controlled Claude Haiku 4.5 results: 100 records per domain and condition under the common singleton convention. Bold marks the lowest unrounded WD; CE is signed count bias. Dashes indicate unavailable outputs. Legacy metrics appear in Tables 9 and 13.
<table><tr><td rowspan="2">Method</td><td colspan="2">WikiQA</td><td colspan="2">News</td><td colspan="2">Story</td></tr><tr><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td><td>WD↓</td><td>CE</td></tr><tr><td>Single change</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TextTiling</td><td>0.427</td><td>-0.09</td><td>0.796</td><td>-4.18</td><td>0.712</td><td>-3.19</td></tr><tr><td>LLMPred</td><td>0.428</td><td>0.41</td><td>0.420</td><td>-0.81</td><td>0.486</td><td>-2.02</td></tr><tr><td>PaLD</td><td>0.431</td><td>-1.28</td><td></td><td></td><td></td><td></td></tr><tr><td>SenPred</td><td>0.257</td><td>-0.90</td><td>0.707</td><td>-6.74</td><td>0.924</td><td>-12.51</td></tr><tr><td>Voting</td><td>0.286</td><td>0.00</td><td>0.107</td><td>-0.32</td><td>0.424</td><td>-2.39</td></tr><tr><td>VCP</td><td>0.267</td><td>0.20</td><td>0.265</td><td>-0.99</td><td>0.394</td><td>-1.02</td></tr><tr><td>WCP</td><td>0.284</td><td>0.07</td><td>0.370</td><td>-1.73</td><td>0.534</td><td>-2.55</td></tr><tr><td>RWCP-R</td><td>0.223</td><td>0.47</td><td>0.102</td><td>0.03</td><td>0.243</td><td>0.34</td></tr><tr><td>Decoherence</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TextTiling</td><td>0.427</td><td>-0.09</td><td>0.797</td><td>-4.18</td><td>0.708</td><td>-3.17</td></tr><tr><td>LLMPred</td><td>0.416</td><td>0.37</td><td>0.437</td><td>-1.22</td><td>0.542</td><td>-2.96</td></tr><tr><td>SenPred</td><td>0.362</td><td>-1.30</td><td>0.885</td><td>-10.58</td><td>0.953</td><td>-12.89</td></tr><tr><td>Voting</td><td>0.359</td><td>-0.03</td><td>0.363</td><td>-1.75</td><td>0.639</td><td>-3.82</td></tr><tr><td>WCP</td><td>0.340</td><td>0.44</td><td>0.407</td><td>-0.59</td><td>0.398</td><td>-0.45</td></tr><tr><td>RWCP-R</td><td>0.291</td><td>0.65</td><td>0.335</td><td>0.60</td><td>0.335</td><td>0.59</td></tr><tr><td>Paraphrasing</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TextTiling</td><td>0.431</td><td>-0.06</td><td>0.785</td><td>-4.16</td><td>0.704</td><td>-3.21</td></tr><tr><td>LLMPred</td><td>0.429</td><td>0.30</td><td>0.395</td><td>-0.44</td><td>0.486</td><td>-2.13</td></tr><tr><td>SenPred</td><td>0.273</td><td>-0.96</td><td>0.696</td><td>-6.53</td><td>0.905</td><td>-12.08</td></tr><tr><td>Voting</td><td>0.267</td><td>0.04</td><td>0.091</td><td>-0.24</td><td>0.389</td><td>-2.23</td></tr><tr><td>WCP</td><td>0.266</td><td>0.25</td><td>0.326</td><td>-1.05</td><td>0.428</td><td>-0.75</td></tr><tr><td>RWCP-R</td><td>0.189</td><td>0.37</td><td>0.146</td><td>0.12</td><td>0.310</td><td>0.52</td></tr></table>

Together with the higher family-level boundary F1, these results identify single changes and attacked text as the clearest empirical successes of the recurrent decoder. They compare complete frozen pipelines; the component analysis below examines individual design choices at that configuration. The GPT-5-mini single-change aggregates remain a separate supplemental panel (Table 10), outside the primary means and paired analysis.

## 4.3 CHANGE DENSITY AND REAL COLLABORATIVE TEXT

Table 3: Story change-density results: 100 documents per generator and K. True K is used only for evaluation. WD is standard raw-boundary WindowDiff; CE is signed count bias. Bold compares WCP and RWCP-R within each generator.
<table><tr><td rowspan="2">Generator</td><td rowspan="2">K</td><td colspan="2">WCP</td><td colspan="2">RWCP-R</td></tr><tr><td>WD</td><td>CE</td><td>WD</td><td>CE</td></tr><tr><td rowspan="4">Claude Haiku 4.5</td><td>1</td><td>0.400</td><td>-0.52</td><td>0.288</td><td>0.58</td></tr><tr><td>2</td><td>0.404</td><td>-0.19</td><td>0.374</td><td>1.44</td></tr><tr><td>3</td><td>0.425</td><td>-0.11</td><td>0.378</td><td>2.10</td></tr><tr><td>5 8</td><td>0.425 0.430</td><td>0.39 2.85</td><td>0.429 0.448</td><td>4.26 7.21</td></tr><tr><td rowspan="5">GPT-5-mini</td><td>1</td><td>0.434</td><td>-0.38</td><td>0.339</td><td>0.79</td></tr><tr><td>2</td><td>0.430</td><td>0.01</td><td>0.408</td><td>1.51</td></tr><tr><td>3</td><td>0.470</td><td>0.31</td><td>0.418</td><td>2.42</td></tr><tr><td>5</td><td>0.475</td><td>1.26</td><td>0.447</td><td>4.28</td></tr><tr><td>8</td><td>0.471</td><td>4.10</td><td>0.464</td><td>7.22</td></tr></table>

The multiple-change settings probe the transition from isolated changes to short alternating passages (Table 3). Relative to WCP, standard WD improves for Claude at $\check { K } \in \{ 1 , 2 , 3 \}$ and for GPT-5-mini

at every evaluated K. The advantage narrows at high density: at $K = 8$ , the WD difference is +0.018 for Claude and −0.006 for GPT-5-mini. CE exceeds seven in both cases, corresponding to fewer than one detected boundary on average despite eight reference changes. The trend is consistent with the practical difficulty of separating short passages under a fixed transition penalty and shows where the window-level advantage becomes small.

On CoAuthor, WD changes only from 0.4878 to 0.4869, while mean CE increases from 7.717 to 10.166. The no-boundary predictor is close on CoAuthor and the LLM-proportion family: its WD is 0.4893 and 0.2141, compared with 0.4869 and 0.2119 for RWCP-R. Thus the strongest evidence for localized boundary improvement comes from the single-change and perturbation families, whereas these two families emphasize the distinction between WD and exact boundary recovery. Appendix Tables 11, 14, and 16 give the full baseline, proportion, and recovery diagnostics.

## 4.4 COMPONENT CONTRIBUTIONS

Table 4: Frozen component comparisons, averaged over settings and then families. Fixed scale uses $\sigma = 1$ , uniform weights exponent zero, and quadratic loss $c = 1 0 ^ { 6 }$ . No variant is retuned. Bold marks the lowest window error; signed CE is not ranked.

<table><tr><td rowspan="2">Variant</td><td colspan="2">Raw boundaries</td><td>Final labels</td></tr><tr><td>WD↓</td><td>Mean CE</td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td></tr><tr><td>RWCP-analytic</td><td>0.3466</td><td>2.915</td><td>0.3233</td></tr><tr><td>RWCP-block</td><td>0.3371</td><td>3.050</td><td>0.3158</td></tr><tr><td>RWCP-R</td><td>0.3110</td><td>2.940</td><td>0.2990</td></tr><tr><td>R: fixed scale</td><td>0.3170</td><td>2.984</td><td>0.3038</td></tr><tr><td>R: uniform weights</td><td>0.3118</td><td>2.756</td><td>0.3058</td></tr><tr><td>R: quadratic approximation</td><td>0.3179</td><td>2.782</td><td>0.3130</td></tr></table>

Table 4 probes three design choices at the selected recurrent configuration. Fitting the scale and retaining the finite Huber cutoff each improve both window errors relative to their fixed-configuration alternatives. Uniform weights nearly match standard WD: the paired difference for full RWCP-R minus uniform weights is −0.0008, with interval [−0.0052, 0.0038]. This contrast makes the empirical role of weighting metric-specific: the full model has lower legacy window error, while uniform weights have lower count bias and stronger boundary matching in Appendix Table 17.

RWCP-R also reduces family-macro standard WD by 10.3% and 7.7% relative to the historical analytic and block-calibrated variants. These are different from the fixed analytic-threshold core evaluation in Table 1. Because those historical variants differ in search, scale, and other settings together, we interpret this comparison as a pipeline result; the fixed component variants provide the more focused evidence about scale, loss, and weighting. Family-wise comparisons, auxiliary metrics, and the post-freeze stability check are reported in Appendix B.7.

## 5 DISCUSSION AND CONCLUSION

RWCP turns robust change-point detection into a comparison of optimized segment risks. The merge-cost identity supplies a population margin without solving for nonlinear mixed-segment Huber centers, and the quadratic limit recovers weighted CUSUM. With bounded-score concentration, reliability-coordinate isolation, and guarded recursion, this yields exact count recovery and weighted localization for core RWCP under the stated conditions. The new fixed-configuration core evaluation directly measures the corresponding profile search on cached detector scores, but it does not establish that those scores or numerical thresholds meet the theorem’s assumptions.

The core evaluation lowers average WindowDiff relative to WCP, with a simultaneous count and boundary-F1 gain only on the single-change family. Its high no-boundary rates on CoAuthor and dense or altered texts demonstrate the limits of a fixed analytic threshold in this retrospective score suite. The recurrent extension shares information across nonadjacent passages and lowers familymacro WindowDiff further, with the clearest boundary-recovery gain on single changes; its count/F1 tradeoff persists in other families. These results motivate count-aware calibration on independent detector streams.

The two contributions have complementary scopes: Theorem 3.11 covers analytic-threshold core RWCP under specified conditions, while RWCP-R evaluates a distinct recurrent objective on retrospective cached scores. The empirical results support window-error reductions, not general exact-count recovery on this benchmark suite; gradual editing, within-sentence mixing, reversed score semantics, long-memory dependence, and target-dependent weight estimation remain outside the present score model.

## ETHICS STATEMENT

RWCP is a research tool for provenance segmentation, not automatic evidence of misconduct. Detector errors can vary across language, genre, dialect, and demographic writing style; deployment therefore requires domain-specific validation, uncertainty reporting, human review, and respect for dataset and model-provider terms.

## REPRODUCIBILITY STATEMENT

The appendices document assumptions, proofs, frozen settings, and per-setting results. The accompanying local records preserve cached inputs and outputs, metric-recomputation scripts, and the complete-record correction; they do not constitute a fresh independent test or establish exact regeneration from model names.

## AI USAGE STATEMENT

Generative AI assisted with manuscript editing, LAT<sub>E</sub>X, code, and consistency checks, including the reporting correction. Numerical comparisons are derived from saved predictions and explicit evaluation conventions. The authors remain responsible for validating the mathematics, implementation, citations, and empirical claims.

## REFERENCES

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In International Conference on Learning Representations, 2024.

Rafal Baranowski, Yining Chen, and Piotr Fryzlewicz. Narrowest-over-threshold detection of multiple change points and change-point-like features. Journal of the Royal Statistical Society: Series B, 81 (3):649–672, 2019.

Haeran Cho and Claudia Kirch. Two-stage data segmentation permitting multiscale change points, heavy tails and dependence. Annals of the Institute of Statistical Mathematics, 74(4):653–684, 2022.

Paul Doukhan. Mixing: Properties and Examples. Springer, 1994.

Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pp. 889–898, 2018. doi: 10.18653/v1/P18-1082.

Paul Fearnhead and Guillem Rigaill. Changepoint detection in the presence of outliers. Journal ofthe American Statistical Association, 114(525):169–183, 2019. doi: 10.1080/01621459.2017.1385466.

Piotr Fryzlewicz. Wild binary segmentation for multiple change-point detection. The Annals of Statistics, 42(6):2243–2281, 2014.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander M. Rush. GLTR: Statistical detection and visualization of generated text. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pp. 111–116, 2019.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Spotting LLMs with binoculars: Zero-shot detection of machine-generated text. In International Conference on Machine Learning, 2024.

Marti A. Hearst. TextTiling: Segmenting text into multi-paragraph subtopic passages. Computational Linguistics, 23(1):33–64, 1997.

Peter J. Huber. Robust estimation of a location parameter. The Annals ofMathematical Statistics, 35 (1):73–101, 1964.

Peter J. Huber and Elvezio M. Ronchetti. Robust Statistics. Wiley, 2 edition, 2009.

Hans R. Kunsch. The jackknife and the bootstrap for general stationary observations.¨ The Annals of Statistics, 17(3):1217–1241, 1989.

Mina Lee, Percy Liang, and Qian Yang. CoAuthor: Designing a human–AI collaborative writing dataset for exploring language model capabilities. In Proceedings ofthe 2022 CHI Conference on Human Factors in Computing Systems, pp. 1–19, 2022.

Enzo Lei, Hsiang Hsu, and Ching-Feng Chen. PaLD: Detection of text partially written by large language models. In International Conference on Learning Representations, 2025.

Mengchu Li and Yi Yu. Adversarially robust change point detection. In Advances in Neural Information Processing Systems, volume 34, pp. 22955–22967, 2021.

Mengchu Li, Jin Zhu, Jinglai Li, and Chengchun Shi. Segmenting human–LLM co-authored text via change point detection. arXiv preprint arXiv:2605.03723, 2026.

Florence Merlevede, Magda Peligrad, and Emmanuel Rio. Bernstein inequality and moderate \` deviations under strong mixing conditions. In High Dimensional Probability V: The Luminy Volume, pp. 273–292. Institute of Mathematical Statistics, 2009.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. DetectGPT: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pp. 24950–24962, 2023.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1797–1807, 2018. doi: 10.18653/v1/D18-1206.

Lev Pevzner and Marti A. Hearst. A critique and improvement of an evaluation metric for text segmentation. Computational Linguistics, 28(1):19–36, 2002.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, 2016. doi: 10.18653/v1/D16-1264.

Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, and Irina Piontkovskaya. Intrinsic dimension estimation for robust detection of AI-generated texts. In Advances in Neural Information Processing Systems, volume 36, pp. 39257–39276, 2023.

Daren Wang, Yi Yu, and Alessandro Rinaldo. Univariate mean change point detection: Penalization, CUSUM and optimality. Electronic Journal ofStatistics, 14(1):1917–1961, 2020.

Yi Yu. A review on minimax rates in change point detection and localisation. arXiv preprint arXiv:2011.01857, 2020.

Zijie Zeng, Shuyu Liu, Lei Sha, Zhuang Li, Kaixuan Yang, Sheng Liu, Dragan Gaseviˇ c, and Guanliang´ Chen. Detecting AI-generated sentences in human–AI collaborative hybrid texts: Challenges, strategies, and insights. In International Joint Conference on Artificial Intelligence, 2024.

Zhenyu Zhang, Wen Qin, and Bryan A. Plummer. Machine-generated text localization. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 8357–8371, 2024.

## A RELATED WORK

This section positions RWCP at the intersection of machine-text detection, structured authorship localization, robust change point analysis, and dependence-aware calibration. The comparison clarifies which ingredients are inherited from detector scoring and weighted segmentation, and which are introduced by the profile-loss formulation.

Detection and mixed-authorship localization. Document-level machine-text detection has been studied through supervised classifiers, likelihood and rank statistics, perturbation tests, rewriting dis tances, intrinsic-dimension summaries, and watermark-based mechanisms. Representative zero-shot procedures include GLTR-style token-rank diagnostics, DetectGPT and Fast-DetectGPT probabilitycurvature statistics, Binoculars, and intrinsic-dimension or divergent n-gram tests (Gehrmann et al., 2019; Mitchell et al., 2023; Bao et al., 2024; Hans et al., 2024; Tulchinskii et al., 2023). Supervised approaches train classifiers on paired human and machine corpora or adapt detection rules to a target generator. These methods differ in their assumptions and transfer behavior, but most produce a scalar score or probability that can be used as an input to the present framework. RWCP is detector-agnostic: it changes how ordered scores are segmented, not how a single text unit is scored.

Mixed human–AI writing has motivated sentence-level prediction, token-to-sentence aggregation, window voting, supervised sequence segmentation, and partial-LLM detectors. The CoAuthor data record detailed human interactions with model suggestions and provide a particularly important real-world test bed (Lee et al., 2022). Existing localization procedures often classify each unit independently, aggregate token predictions, or smooth local predictions over a fixed window (Zeng et al., 2024; Zhang et al., 2024; Lei et al., 2025). Change point formulations impose a stronger structural prior: consecutive units share a source until a boundary occurs. This structure reduces isolated label flips and aligns the optimization objective with boundary localization.

The foundational WCP formulation makes a further advance by incorporating heterogeneous score variability. Its upper and lower bounds show that inverse-variance information, rather than raw sentence count, determines localization difficulty. Our work preserves this geometry but replaces the mean-based split statistic by a robust profile-loss gain and extends the stochastic analysis to locally dependent bounded-influence scores.

Change points and robust profile methods. CUSUM statistics are central to univariate meanchange detection (Wang et al., 2020; Yu, 2020). Binary segmentation, wild binary segmentation, and narrowest-over-threshold (NOT) algorithms use multiscale or random intervals to isolate individual changes (Fryzlewicz, 2014; Baranowski et al., 2019). NOT selects the shortest interval whose maximal contrast exceeds a threshold and then recurses, thereby reducing cancellation among multiple changes. Non-asymptotic localization theory typically combines three ingredients: a deterministic contrast margin, a uniform noise bound, and an interval-isolation event.

Our proof follows that architecture in reliability coordinates. The deterministic object is a profile gain rather than a linear contrast; the noise control is based on bounded Huber scores under an explicit dependence-aware Bernstein condition; and random intervals are sampled in cumulative reliability mass. The recursive proof also has to control boundary leakage because an estimated split need not coincide exactly with the true change.

Huber loss interpolates between quadratic and absolute loss and yields bounded score influence (Huber, 1964; Huber & Ronchetti, 2009). Robust mean and location estimators have been developed under finite moments, contamination, and dependence. Median-of-means and Catoni-type estimators offer alternative guarantees, while Huber estimators are especially convenient here because their optimized objective defines a profile likelihood-ratio analogue. Robust change point methods include median and rank contrasts, robustified CUSUMs, M-estimation losses, and procedures designed for heavy-tailed or contaminated sequences (Li & Yu, 2021; Cho & Kirch, 2022).

Profile costs of the form min $\textstyle \sum _ { i } \rho ( Y _ { i } - \theta )$ , including Huber losses, are established in robust penalized segmentation (Fearnhead & Rigaill, 2019). Bounded influence alone does not provide immunity to arbitrary outliers; in particular, an unbounded Huber loss and a bounded robust loss have different contamination behavior. Our guarantees require finite risk, central mass, bounded weights, local jumps, and the stated concentration condition. The specific distinction analyzed here is between a robust center-difference contrast and a reliability-weighted profile-loss gain. The former requires understanding a nonlinear mixed-distribution center at every candidate split. The latter compares optimized risks and converts an incorrect split into a merge-cost problem. This distinction is not cosmetic; it is the main reason a transparent deterministic margin can be proved.

Dependence and calibration. Text scores may be dependent because adjacent sentences share topic, vocabulary, discourse structure, and detector context. Change point theory under mixing and other weak-dependence conditions replaces independent concentration by blocking, coupling, martingale, or spectral arguments (Doukhan, 1994; Merlevede et al., 2009). Block bootstrap and dependent \` multiplier methods are widely used to approximate distributions of statistics under serial dependence (Kunsch, 1989). In our main theory, dependence enters through an explicit Bernstein condition.¨ Independent and fixed-order dependent sequences have the usual logarithmic complexity, while the geometric beta-mixing specialization retains the published logarithmic penalty. The bootstrap is kept secondary: it is a practical calibration mechanism, while exact recovery is established using an analytic threshold.

## B EXPERIMENTAL PROTOCOL AND ADDITIONAL RESULTS

This appendix documents the cached-score inventory, fixed RWCP-R configuration, selection history, and evaluation conventions before presenting the full benchmark comparisons. It then reports recovery and component diagnostics, post-freeze stability, implementation details, and paired uncertainty estimates. All 25 primary settings and 2,690 instances use the same reporting conventions as Section 4.

## B.1 DATA, COMPARATORS, AND EVALUATION SCOPE

The controlled domains are denoted WikiQA, News, and Story following the reference segmentation pipeline (Li et al., 2026); their source corpora are SQuAD, XSum, and WritingPrompts, respectively (Rajpurkar et al., 2016; Narayan et al., 2018; Fan et al., 2018). In the primary suite, single-change and attack data use cached Claude Haiku 4.5 texts (claude-haiku-4-5); multiple-change Story also uses gpt-5-mini. CoAuthor retains its existing 868/289/290 training/validation/test session partition and three source classes: human, collaborative, and AI (Lee et al., 2022). The evaluated split has 290 sessions and 7,042 sentences.

Table 5: Completed benchmark inventory. Counts are evaluated document instances; transformed copies can share a source. FT is the repository’s fine-tuned score and NFT its normalized score. The retained benchmarks use the unknown-count protocol for segmentation.
<table><tr><td>Experiment</td><td>Settings</td><td>Instances</td><td>Score</td><td>Count protocol</td></tr><tr><td>Single change</td><td>3</td><td>300</td><td>FT</td><td>Unknown K</td></tr><tr><td>CoAuthor</td><td>1</td><td>290</td><td>NFT</td><td>Unknown K</td></tr><tr><td>Multiple changes</td><td>10</td><td>1,000</td><td>NFT</td><td>Unknown K</td></tr><tr><td>Text attacks</td><td>6</td><td>600</td><td>NFT</td><td>Unknown K</td></tr><tr><td>LLM proportion</td><td>5</td><td>500</td><td>NFT</td><td>Unknown K</td></tr></table>

Every controlled setting contains 100 records, including WikiQA, News, and Story. In the three Claude WikiQA settings (single change and two attacks), records 80 and 86 each contain one sentence and no reference boundary. We retain these records: the only admissible boundary set is empty, both window errors and count MAE are zero, and empty–empty boundary F1 is one. This is a boundary-only convention, not a claim of correct source classification. It is applied equally to every method. The other 98 records per setting retain their saved predictions. Thus the primary suite contains 2,690 instances; six trivial instances enter reporting only and do not alter historical parameter selection. The multiple-change settings use $K \in \{ 1 , 2 , 3 , 5 , 8 \}$ for each generator; the proportion settings use target fractions of 5%, 10%, 20%, 40%, and 80%. We reuse the saved hybrid texts, predictions, and detector scores. Generation revisions, dates, and decoding metadata are incomplete; reproducibility is from these caches, rather than exact regeneration from a model name.

Comparators are TextTiling (Hearst, 1997), direct LLM prediction (LLMPred), independent sentence prediction (SenPred), local majority voting (Voting), VCP, and WCP. VCP/WCP use the reference change-point implementation (Li et al., 2026); sentence prediction and voting follow the localization pipeline (Zhang et al., 2024). PaLD (Lei et al., 2025) has a complete local output only on WikiQA in the single-change benchmark. SegFormer outputs on CoAuthor and VCP outputs for the attack/proportion experiments are unavailable. We retain saved baseline predictions and recompute metrics consistently; RWCP-R is the rerun method. The new core RWCP comparison is also recomputed from the same caches, but uses its own fixed analytic threshold rather than the historical practical RWCP-analytic/block configurations.

For every nontrivial input in the primary suite, core RWCP and RWCP-R consume exactly the WCP cached score sequence and reliability proxies. FT and NFT are distinct score pipelines, and other baselines need not share their representations. Historical WCP re-scores concatenated predicted segments with the language model, whereas RWCP-R labels its states from fitted score centers. Thus final-label metrics compare complete pipelines. Standard WD on raw boundaries and the component ablations provide complementary evidence about segmentation.

## B.2 FIXED ANALYTIC-THRESHOLD CORE RWCP EVALUATION

To evaluate the method analyzed in Theorem 3.11, we run the weighted Huber profile gain, reliabilitycoordinate interval search, analytic threshold, and guarded recursion without the scale estimation, additional multiscale intervals, global penalized fitting, or state sharing of the practical variants. The fixed numerical configuration is $c = 1 . 3 4 5$ , capped weights in [0.2, 5], minimum side and guard masses both 1, 500 sampled intervals, $\delta = 0 . 0 5$ , threshold multiplier $\dot { C } _ { r } = 1$ , dependence multiplier 1, and seed 42. Cached WCP reliability proxies are clipped directly without per-document renormalization. The implementation additionally includes the full document as a deterministic candidate interval and minimizes each Huber loss over the observed score span; these details are outside the theorem’s literal fixed-Θ, sampled-interval specification. The run does not verify the unknown signal, spacing, curvature, or dependence constants required by the theorem, and should not be read as a test of its probability guarantee.

Table 6: Fixed analytic-threshold core RWCP versus saved WCP boundaries on the same cached scores. Family means average documents within settings and then settings equally. F1 matches boundaries within one sentence. Zero CP is the fraction with no detected boundary. Loc. coverage is the fraction of documents with at least one true change whose estimated change count is exactly correct. Weighted loc. MAE averages ordered boundary distances $d _ { w }$ only over those eligible documents; – means none is eligible. The conditional localization mean must be interpreted together with coverage. These are descriptive results without newly estimated paired intervals.
<table><tr><td></td><td colspan="3">WCP</td><td colspan="6">Core RWCP</td></tr><tr><td>Family</td><td>WD</td><td>MAE</td><td>F1</td><td>WD</td><td>MAE</td><td>F1</td><td>Zero CP</td><td>Loc. cov.</td><td>Weighted loc. MAE</td></tr><tr><td>Single change</td><td>0.3961</td><td>1.637</td><td>0.435</td><td>0.2315</td><td>0.393</td><td>0.487</td><td>31.3%</td><td>61.4%</td><td>1.636</td></tr><tr><td>CoAuthor</td><td>0.4878</td><td>7.917</td><td>0.302</td><td>0.4897</td><td>10.159</td><td>0.057</td><td>82.4%</td><td>0.0%</td><td></td></tr><tr><td>Multiple changes</td><td>0.4363</td><td>2.130</td><td>0.205</td><td>0.4108</td><td>3.716</td><td>0.026</td><td>93.1%</td><td>1.0%</td><td>0.892</td></tr><tr><td>Text attacks</td><td>0.3609</td><td>0.998</td><td>0.272</td><td>0.3369</td><td>0.940</td><td>0.038</td><td>94.7%</td><td>5.4%</td><td>2.788</td></tr><tr><td>LLM proportion</td><td>0.3538</td><td>1.360</td><td>0.220</td><td>0.2085</td><td>0.936</td><td>0.037</td><td>92.8%</td><td>6.4%</td><td>4.395</td></tr></table>

Across the 25 settings, the core implementation has lower family-macro WD than WCP, but its count MAE and F1 are worse overall. On single-change texts, the 61.4% localization coverage is higher than WCP’s 30.9%, yet the conditional weighted localization error is 1.636 for core RWCP versus 0.821 for WCP; these means use different eligible subsets and cannot establish a per-boundary improvement. Only 1.0% of multiple-change documents have the correct positive change count under core RWCP. The apparent conditional weighted error of 0.892 in that family rests on just 10 eligible documents. The zero-boundary rates and low coverage limit any empirical claim of exact recovery on these cached detector scores.

Table 7: Per-setting standard WD for WCP and fixed analytic-threshold core RWCP, with core count MAE, boundary F1 at one-sentence tolerance, and no-boundary fraction. Each setting contains 100 documents except CoAuthor (290). The six one-sentence WikiQA records follow the common empty-boundary convention.
<table><tr><td>Family</td><td>Setting</td><td>WCP WD</td><td>Core WD</td><td>Core MAE</td><td>Core F1</td><td>Zero CP</td></tr><tr><td>Single</td><td>WikiQA</td><td>0.284</td><td>0.279</td><td>0.580</td><td>0.380</td><td>60%</td></tr><tr><td>Single</td><td>News</td><td>0.370</td><td>0.119</td><td>0.130</td><td>0.760</td><td>8%</td></tr><tr><td>Single</td><td>Story</td><td>0.534</td><td>0.296</td><td>0.470</td><td>0.320</td><td>26%</td></tr><tr><td>CoAuthor</td><td>Sessions</td><td>0.488</td><td>0.490</td><td>10.159</td><td>0.057</td><td>82%</td></tr><tr><td>Multiple</td><td>Claude K = 1</td><td>0.400</td><td>0.319</td><td>0.920</td><td>0.070</td><td>92%</td></tr><tr><td>Multiple</td><td>Claude K = 2</td><td>0.404</td><td>0.393</td><td>1.900</td><td>0.047</td><td>91%</td></tr><tr><td>Multiple</td><td>Claude K = 3</td><td>0.425</td><td>0.419</td><td>2.840</td><td>0.051</td><td>88%</td></tr><tr><td>Multiple</td><td>Claude K = 5</td><td>0.425</td><td>0.447</td><td>4.830</td><td>0.035</td><td>87%</td></tr><tr><td>Multiple</td><td>Claude K = 8</td><td>0.430</td><td>0.460</td><td>7.820</td><td>0.022</td><td>86%</td></tr><tr><td>Multiple</td><td>GPT-5-mini K = 1</td><td>0.434</td><td>0.333</td><td>1.000</td><td>0.000</td><td>100%</td></tr><tr><td>Multiple</td><td>GPT-5-mini K = 2</td><td>0.430</td><td>0.395</td><td>1.970</td><td>0.020</td><td>97%</td></tr><tr><td>Multiple</td><td>GPT-5-mini K = 3</td><td>0.470</td><td>0.424</td><td>2.970</td><td>0.009</td><td>98%</td></tr><tr><td>Multiple</td><td>GPT-5-mini K = 5</td><td>0.475</td><td>0.453</td><td>4.960</td><td>0.006</td><td>97%</td></tr><tr><td>Multiple</td><td>GPT-5-mini K = 8</td><td>0.471</td><td>0.464</td><td>7.950</td><td>0.002</td><td>95%</td></tr><tr><td>Attack</td><td>WikiQA decoh.</td><td>0.340</td><td>0.365</td><td>0.980</td><td>0.020</td><td>100%</td></tr><tr><td>Attack</td><td>News decoh.</td><td>0.407</td><td>0.336</td><td>1.000</td><td>0.000</td><td>100%</td></tr><tr><td>Attack</td><td>Story decoh.</td><td>0.398</td><td>0.321</td><td>0.860</td><td>0.060</td><td>86%</td></tr><tr><td>Attack</td><td>WikiQA para.</td><td>0.266</td><td>0.361</td><td>0.970</td><td>0.030</td><td>99%</td></tr><tr><td>Attack</td><td>News para.</td><td>0.326</td><td>0.314</td><td>0.930</td><td>0.070</td><td>93%</td></tr><tr><td>Attack</td><td>Story para.</td><td>0.428</td><td>0.325</td><td>0.900</td><td>0.050</td><td>90%</td></tr><tr><td>Proportion</td><td>5%</td><td>0.321</td><td>0.076</td><td>0.980</td><td>0.007</td><td>97%</td></tr><tr><td>Proportion</td><td>10%</td><td>0.338</td><td>0.137</td><td>0.970</td><td>0.017</td><td>96%</td></tr><tr><td>Proportion</td><td>20%</td><td>0.375</td><td>0.259</td><td>0.920</td><td>0.040</td><td>91%</td></tr><tr><td>Proportion</td><td>40%</td><td>0.404</td><td>0.301 0.269</td><td>0.870 0.940</td><td>0.090</td><td>86%</td></tr><tr><td>Proportion</td><td>80%</td><td>0.330</td><td></td><td></td><td>0.030</td><td>94%</td></tr></table>

## B.3 PRACTICAL RWCP CONFIGURATION

RWCP-R implements Section 2.2, with three declared source states on CoAuthor and two elsewhere. Given positive cached reliability proxies $r _ { i } .$ , we compute

$$
\widetilde w _ { i } = \mathrm { c l i p } _ { [ 0 . 0 5 , 1 0 ] } \left( \frac { r _ { i } ^ { \eta } } { N ^ { - 1 } \sum _ { j } r _ { j } ^ { \eta } } \right) , \qquad \eta = 0 . 5 ,
$$

and do not renormalize after clipping. The cache proxy is $n _ { i } ^ { 2 } / v _ { i } \colon n _ { i }$ is the scoring tokenizer’s unit token count and $v _ { i }$ the NFT score-variance proxy; FT uses $v _ { i } = 1$ . Compression tempers extreme reliability ratios without discarding their ordering. All settings use the same exponent.

Scores are centered by their median and divided by an initial robust scale. For adjacent differences, let

$$
{ \cal D } _ { i } = \frac { Y _ { i + 1 } - Y _ { i } } { \sqrt { 1 / \widetilde w _ { i } + 1 / \widetilde w _ { i + 1 } } } , \qquad s _ { D } = \frac { \mathrm { m e d i a n } _ { i } | D _ { i } - \mathrm { m e d i a n } ( D ) | } { 0 . 6 7 4 4 8 9 7 5 } .
$$

The initial scale is the maximum of $s _ { D } , 0 . 0 5$ times the Gaussian-scaled MAD of score levels, and $1 0 ^ { - 6 }$ times the score range; a constant sequence uses scale one. The fitted dimensionless scale in (6) is constrained by $\sigma \geq 0 . 2 5$

The frozen main configuration uses Huber cutoff $c = 1 . 5$ and transition penalty $\lambda = 4$ . Initial centers are equally spaced quantiles between $( q , 1 - q )$ , for $q \in \{ 0 . 0 5 , 0 . 2 0 , \bar { 0 } . 3 5 \}$ , combined with initial scales one and 0.25, giving six deterministic starts. Each start runs at most 50 alternating updates, stopping when the objective change is below $1 0 ^ { - 8 } ( 1 + | \mathcal { I } | )$ . Center and scale bisection use tolerance $1 0 ^ { \dot { - } \dot { 8 } }$ . We retain the smallest attained objective, breaking exact ties by fewer transitions. Empty states retain their previous centers. Fitted centers are ranked in ascending order to assign source-state labels; this assumes score ordering agrees with source semantics. The smallest allowed state run is one observation; no reference boundary, target LLM proportion, generator identity, or reference change count enters an unknown-count prediction.

Algorithm 1 Recurrent profile decoding (RWCP-R)   
Require: Scores, reliability proxies, declared state count $S ,$ and frozen $c , \eta , \lambda , \sigma _ { \mathrm { m i n } } .$   
1: If $N = 1$ , return no boundary (the boundary-only evaluation convention).   
2: Standardize scores and construct capped weights as specified above.   
3: for each of the six quantile/scale initializations do   
4: for $t = 1 , \ldots , 5 0$ do   
5: Decode z by Viterbi with Huber emissions and penalty $\lambda .$   
6: Fit each occupied center by weighted Huber bisection; retain empty centers.   
7: Let $\begin{array} { r } { F ( \sigma ) = \bar { \sum } _ { i } \psi _ { c } ( r _ { i } ( \sigma ) ) \dot { r } _ { i } ( \sigma ) } \end{array}$   
8: Set $\sigma = \sigma _ { \mathrm { m i n } } \mathrm { i f } F ( \sigma _ { \mathrm { m i n } } ) \leq N ;$ otherwise solve $F ( \sigma ) = N .$   
9: Stop if the objective change is below $1 0 ^ { - 8 } ( 1 + | \mathcal { T } | ) .$   
10: end for   
11: end for   
12: Keep the smallest attained objective; break exact ties by fewer transitions.   
13: Rank centers to label states and return $\{ i : z _ { i } \neq z _ { i + 1 } \}$

The implementation uses at most 64 bisection iterations per conditional fit with absolute tolerance $1 0 ^ { - 8 }$ . For B bisection steps and $T$ alternating iterations, six starts cost $O ( 6 T N ( S ^ { 2 } + S B ) )$ ) time and $O ( N S )$ decoding memory. The scale ablation fixes $\sigma = 1$ and consequently uses only the three center initializations. The dynamic program is exact conditional on centers and scale; the overall fit is a deterministic multistart approximation.

For continuity, RWCP-analytic and RWCP-block denote the preceding analytic block v3 implementation, using $c = 3 ,$ clipped mean-one weights in [0.02, 10], three refinement passes, and common threshold multiplier $0 . 8 5 .$ These practical core variants also use implementation refinements beyond the unmodified analytic-threshold theorem. Their existing results are retained in Tables 15 and 17. The five retained comparisons use RWCP-R throughout. The historical core configurations fix 500 sampled intervals (with exhaustive intervals for $N \leq 2 4 )$ , minimum side mass and guard mass one, nominal level 0.05, dependence factor one, seed 42, and optimization tolerance $1 0 ^ { - 6 }$ . Block calibration uses length four and 499 multiplier draws; its threshold mixes the bootstrap value with the analytic reference at weight 0.35 on the latter and is capped at 1.5 times that reference. Both variants use curvature floor 0.05, curvature shrinkage 0.5, and the common threshold multiplier 0.85. These empirical values are not asserted to satisfy the theorem’s unknown signal/spacing requirements.

## B.4 DEVELOPMENT PATH AND LEAKAGE CONTROLS

Configuration selection minimized the family-balanced development mean of

$$
\frac { 1 } { 2 } \mathrm { W D } _ { d } + \frac { 1 } { 2 } \operatorname* { m i n } \{ { \mathrm { W D } _ { \mathrm { s u p } } ( d ) , 1 } \} + \frac { 1 } { 4 } \frac { | K _ { d } - \widehat { K } _ { d } | } { K _ { d } + 1 } + \frac { 1 } { 4 } \{ 1 - \mathrm { F } 1 _ { d , h = 1 } \} .
$$

The search was shared across all reported settings rather than repeated separately for each dataset. Table 8 records the adaptive sequence. The final recurrent sweep was the Cartesian product $c \in$ $\{ 1 . 5 , 3 \} , \eta \in \{ 0 . 2 5 , 0 . 5 , 1 \} , \lambda \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ , and $\sigma _ { \mathrm { m i n } } \in \{ 0 . 2 5 , 0 . 5 \}$ , giving 60 candidates. The selected point is the frozen configuration used in the primary results and the reference point for the component ablations.

The historical nontrivial subset contains 643 linked source components, with 716 instances assigned to development and 1,968 to the internal check by a fixed hash. Six singleton instances restored for complete 100-record reporting do not enter either split or parameter selection. RWCP-R parameters and all component ablations were frozen before that check. No reference boundary, reference change count, target LLM proportion, or generator identity enters an unknown-count prediction, and paired intervals resample source components rather than individual transformed records. The search was adaptive rather than preregistered, and the internal check still uses benchmark sources that influenced earlier development; it therefore supports a post-freeze stability check, not an independent confirmatory claim.

Table 8: Adaptive development path under the original broader protocol. The objective is the criterion above averaged first within families and then across families; lower is better. These are developmentselection values, not test-set performance estimates.
<table><tr><td>Stage</td><td>Candidates</td><td>Main change</td><td>Best dev. objective</td></tr><tr><td>Existing RWCP-block</td><td>fixed</td><td>NOT with block calibration</td><td>0.62712</td></tr><tr><td>Global profile study</td><td>45</td><td>global Huber profile and state pruning</td><td>0.61382</td></tr><tr><td>Recurrent RWCP-R</td><td>60</td><td>recurrent centers and fitted scale</td><td>0.58506</td></tr></table>

## B.5 METRICS, SELECTION, AND PROVENANCE

Standard boundary WindowDiff (Pevzner & Hearst, 2002), defined in (21), compares raw detected boundary counts in windows of width

$$
k _ { d } = \operatorname* { m a x } \Biggl \{ 1 , \mathrm { r o u n d } \left( \frac { N _ { d } } { 2 ( K _ { d } + 1 ) } \right) \Biggl \} ,\tag{18}
$$

with ties-to-even rounding. It lies in [0, 1]. For comparison with the inherited tables, $\mathrm { W D _ { s u p } }$ instead averages the absolute discrepancy between reference and final-label boundary counts, using width max $\{ \bar { 1 } , \lfloor N _ { d } / [ 2 ( K _ { d } + 1 ) ] \rfloor \}$ . It can exceed one and is not standard WD. Old aggregate fields are not mixed with new RWCP values.

Raw count error is $\mathrm { C E } _ { d } = K _ { d } - \widehat { K } _ { d } .$ , positive for missed changes, and count MAE averages $| K _ { d } - \widehat { K } _ { d } |$ over documents. CoAuthor also retains its historical final-label error $\mathrm { C E _ { l a b e l } }$ . Boundary F1 uses one-to-one matching of estimated and reference boundaries within tolerance $h = 1$ observation; $h = 0$ is an exact-match diagnostic. Two empty sets receive F1 one and exactly one empty set receives zero. For singleton documents, which have no evaluable window, both window errors are defined as zero. The zero-detection fraction includes the retained singletons; outside these known no-change records, a high value diagnoses missed boundaries. Table boldface is computed before rounding and does not denote significance.

The reported suite focuses on sentence-level segmentation with unknown change counts. This scope was narrowed after inspecting the broader development results, so its comparisons are retrospective. The selected configuration and predictions were retained without retuning for this subset.

The development objective, adaptive search path, source-group split, freezing decisions, and leakage controls are summarized in Appendix B.4; this appendix retains the complete metric definitions and benchmark-specific results. Source IDs are incomplete, so component linkage reduces obvious reuse but does not establish independence.

Pointwise paired 95% intervals against WCP use 2,000 bootstrap repetitions, resampling source components within each setting. We report them in Appendix B.8.3, without multiplicity correction. Family summaries average settings equally; the component summary then averages the five retained families equally.

## B.6 FIVE BENCHMARK COMPARISONS

Single-change localization. In the primary Claude evaluation, RWCP-R has the lowest $\mathrm { W D } _ { \mathrm { s u p } }$ on all three domains (Table 9): 0.184 on WikiQA, 0.097 on News, and 0.235 on Story. Their strongest available baseline values are 0.200, 0.123, and 0.306. Standard WD averaged over the domains falls from 0.2441 for RWCP-block to 0.1895 for RWCP-R, and count MAE falls from 0.490 to 0.340. Paired standard-WD intervals against WCP exclude zero on all three domains, showing a consistent pipeline-level gain in the controlled Claude single-change setting. This comparison alone does not isolate Huber loss from state sharing, scale fitting, or penalty selection.

The GPT-5-mini aggregate extension is shown separately in Table 10. RWCP-R has lower legacy window error on News and Story, while VCP is lower on WikiQA. These supplied aggregates do not enter family means, ablations, win counts, or paired intervals for the primary study.

Table 9: Primary Claude Haiku 4.5 single-change results, 100 records each in WikiQA, News, and Story. Values are document means ± sample standard deviations, recomputed with the singleton boundary convention. Bold marks the lowest unrounded $\mathrm { W D _ { s u p } }$ or smallest absolute mean CE, not significance. PaLD has outputs only for WikiQA.
<table><tr><td rowspan="2">Method</td><td colspan="2">WikiQA</td><td colspan="2">News</td><td colspan="2">Story</td></tr><tr><td> $\mathrm { W D _ { s u p } \ \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \ \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td></tr><tr><td>TextTiling</td><td> $0 . 3 8 8 { \scriptstyle \pm 0 . 2 6 7 }$ </td><td> $- 0 . 0 9 { \pm } 0 . 8 4$ </td><td> $1 . 0 6 6 { \scriptstyle \pm 0 . 2 4 4 }$ </td><td> $- 4 . 1 8 { \pm } 0 . 9 9$ </td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 2 3 3 }$ </td><td> $- 3 . 1 9 2 0 . 9 4$ </td></tr><tr><td>LLMPred</td><td> $0 . 3 6 2 { \scriptstyle \pm 0 . 2 0 4 }$ </td><td> $0 . 4 1 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $0 . 5 0 7 { \scriptstyle \pm 0 . 4 5 6 }$ </td><td> $- 0 . 8 1 \pm 1 . 9 4$ </td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 7 1 3 }$ </td><td> $- 2 . 0 2 \pm 3 . 2 6$ </td></tr><tr><td>PaLD</td><td> $0 . 4 4 6 { \scriptstyle \pm 0 . 3 3 0 }$ </td><td> $- 1 . 2 8 { \pm } 1 . 2 3$ </td><td></td><td></td><td></td><td></td></tr><tr><td>SenPred</td><td> $0 . 2 7 1 { \scriptstyle \pm 0 . 2 9 1 }$ </td><td> $- 0 . 9 0 { \pm } 1 . 1 1$ </td><td> $1 . 6 6 3 { \scriptstyle \pm 0 . 8 2 9 }$ </td><td> $- 6 . 7 4 \pm 3 . 1 8$ </td><td> $2 . 9 7 2 { \scriptstyle \pm 1 . 0 7 1 }$ </td><td> $- 1 2 . 5 1 { \pm } 4 . 2 9$ </td></tr><tr><td>Voting</td><td> $0 . 2 8 2 { \scriptstyle \pm 0 . 2 7 7 }$ </td><td> $\mathbf { 0 . 0 0 } \pm \mathbf { 0 . 4 0 }$ </td><td> $0 . 1 3 1 { \scriptstyle \pm 0 . 2 1 3 }$ </td><td> $- 0 . 3 2 { \pm } 0 . 8 0$ </td><td> $0 . 6 3 5 { \scriptstyle \pm 0 . 5 4 9 }$ </td><td> $- 2 . 3 9 \pm 2 . 2 4$ </td></tr><tr><td>VCP</td><td> $0 . 2 0 0 { \scriptstyle \pm 0 . 1 8 7 }$ </td><td> $0 . 2 0 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $0 . 1 2 4 { \scriptstyle \pm 0 . 1 8 6 }$ </td><td> $- 0 . 9 9 \pm 1 . 4 3$ </td><td> $0 . 3 0 6 { \scriptstyle \pm 0 . 2 6 9 }$ </td><td> $- 1 . 0 2 { \pm } 1 . 6 3$ </td></tr><tr><td>WCP</td><td> $0 . 2 0 0 { \scriptstyle \pm 0 . 1 9 0 }$ </td><td> $0 . 0 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $0 . 1 2 3 { \scriptstyle \pm 0 . 2 1 9 }$ </td><td> $- 1 . 7 3 { \pm } 1 . 8 7$ </td><td> $0 . 3 7 2 { \scriptstyle \pm 0 . 3 0 7 }$ </td><td> $- 2 . 5 5 { \pm 2 . 1 1 }$ </td></tr><tr><td>RWCP-R</td><td> $\mathbf { 0 . 1 8 4 \pm 0 . 1 7 0 }$ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $\mathbf { 0 . 0 9 7 { \scriptstyle \pm 0 . 1 4 2 } }$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 3 0 }$ </td><td> $\mathbf { 0 . 2 3 5 \bot 0 . 1 6 3 }$ </td><td> $\mathbf { 0 . 3 4 \pm 0 . 5 9 }$ </td></tr></table>

Table 10: Supplemental GPT-5-mini single-change panel: 100 records per domain, as specified by the dataset inventory. Entries preserve the supplied aggregate means ± sample standard deviations. This panel is outside the primary 25-setting evaluation and its paired comparisons; document-level outputs and matching selection provenance are not included in that evaluation bundle.
<table><tr><td rowspan="2">Method</td><td colspan="2">WikiQA</td><td colspan="2">News</td><td colspan="2">Story</td></tr><tr><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \ \downarrow }$ </td><td> $\mathrm { C E }  0$ </td></tr><tr><td>TextTiling</td><td> $0 . 3 6 { \pm } 0 . 2 4$ </td><td> $\mathbf { - 0 . 0 2 \pm 0 . 8 1 }$ </td><td> $1 . 3 2 { \pm } 0 . 3 0$ </td><td> $- 5 . 1 9 { \pm } 0 . 9 7$ </td><td> $1 . 1 1 { \pm } 0 . 3 0$ </td><td> $- 4 . 3 9 \pm 1 . 0 6$ </td></tr><tr><td>LLMPred</td><td> $0 . 4 4 \pm 0 . 2 0$ </td><td> $0 . 0 9 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $\mathbf { - 0 . 5 9 { \scriptstyle \pm 1 . 4 6 } }$ </td><td> $0 . 5 2 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $- 0 . 8 4 \pm 2 . 1 6$ </td></tr><tr><td>PaLD</td><td> $0 . 5 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $- 1 . 4 1 \pm 1 . 4 6$ </td><td> $1 . 7 6 { \pm } 0 . 7 5 $ </td><td> $- 6 . 7 5 { \pm } 3 . 3 7$ </td><td> $2 . 1 9 { \pm } 0 . 8 9$ </td><td> $- 8 . 4 7 \pm 3 . 8 2$ </td></tr><tr><td>SenPred</td><td> $0 . 5 2 { \pm } 0 . 3 8$ </td><td> $- 1 . 6 2 { \pm } 1 . 5 1$ </td><td> $3 . 3 2 \pm 0 . 8 5$ </td><td> $- 1 3 . 3 6 \pm 3 . 0 9$ </td><td> $3 . 6 1 { \pm } 1 . 0 5$ </td><td> $- 1 4 . 4 9 \pm 3 . 8 9$ </td></tr><tr><td>Voting</td><td> $0 . 3 9 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $- 0 . 1 0 { \pm } 0 . 7 7$ </td><td> $0 . 8 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $- 3 . 1 7 \pm 2 . 1 5$ </td><td> $1 . 1 2 { \pm } 0 . 4 6$ </td><td> $- 4 . 1 4 \pm 2 . 0 0$ </td></tr><tr><td>VCP</td><td> $\mathbf { 0 . 2 5 \pm 0 . 1 8 }$ </td><td> $0 . 3 9 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $0 . 4 1 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $- 0 . 8 3 { \pm } 1 . 7 7$ </td><td> $0 . 5 2 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $- 0 . 6 9 { \pm } 1 . 8 8$ </td></tr><tr><td>WCP</td><td> $0 . 2 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $0 . 2 6 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $0 . 4 4 \pm 0 . 2 6$ </td><td> $- 1 . 1 6 \pm 1 . 8 7$ </td><td> $0 . 5 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $- 1 . 8 5 { \pm } 1 . 9 9$ </td></tr><tr><td>RWCP-R</td><td> $0 . 2 6 { \pm } 0 . 1 7$ </td><td> $0 . 6 4 \pm 0 . 5 4$ </td><td> $\mathbf { 0 . 3 2 \pm 0 . 1 2 }$ </td><td> $0 . 6 2 \pm 0 . 5 5$ </td><td> $\mathbf { 0 . 3 5 \pm 0 . 1 2 }$ </td><td> $\mathbf { 0 . 6 4 \pm 0 . 5 9 }$ </td></tr></table>

Real co-authoring. On CoAuthor (Table 11), RWCP-R has the smallest $\mathrm { W D _ { s u p } }$ mean, 0.4780 versus 0.4804 for VCP. Standard WD is 0.4869 versus 0.4878 for WCP. The table reports count and boundary diagnostics alongside the window-error gains, allowing the recurrent objective to be compared with the historical labeling pipeline.

Table 11: CoAuthor: 290 sessions and 7,042 sentences. $\mathrm { C E _ { l a b e l } }$ is the historical final-label count error; raw CE, count MAE, and Zero CP use detected boundaries. RWCP-R has the smallest $\mathrm { W D _ { s u p } }$ mean. Zero CP is descriptive and is not ranked. SegFormer has no local output.
<table><tr><td rowspan="2">Method</td><td colspan="2">Final labels</td><td colspan="3">Raw boundaries</td></tr><tr><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E } _ { \mathrm { l a b e l } }  0$ </td><td> $\mathrm { C E }  0$ </td><td>MAE↓</td><td> $\mathrm { Z e r o } \mathrm { C P }$ </td></tr><tr><td>TextTiling</td><td>0.551</td><td>7.31</td><td>7.31</td><td>7.62</td><td>0.0%</td></tr><tr><td>LLMPred</td><td>0.516</td><td>8.30</td><td>8.30</td><td>8.51</td><td>19.0%</td></tr><tr><td>SenPred</td><td>0.809</td><td>-4.16</td><td>-4.16</td><td>5.58</td><td>0.0%</td></tr><tr><td>Voting</td><td>0.647</td><td>3.29</td><td>3.29</td><td>5.27</td><td>0.0%</td></tr><tr><td>VCP</td><td>0.480</td><td>9.42</td><td>9.37</td><td>9.40</td><td>61.7%</td></tr><tr><td>WCP</td><td>0.483</td><td>7.98</td><td>7.72</td><td>7.92</td><td>15.9%</td></tr><tr><td>RWCP-R</td><td>0.478</td><td>10.17</td><td>10.17</td><td>10.19</td><td>84.1%</td></tr></table>

Multiple changes. RWCP-R leads in five of ten $\mathrm { W D _ { s u p } }$ settings (Table 12): Claude $K = 1$ and GPT-5-mini $K = 1 , 3 , 5 , 8 .$ . VCP leads at Claude $K = 2 , 3 , 5$ , Voting at Claude $K = 8 ,$ , and WCP at GPT-5-mini K = 2. Against the previous block configuration, mean standard WD improves from

0.4112 to 0.3993 and count MAE from 3.628 to 3.233. The full recurrent pipeline improves these averages over the historical block configuration, but this comparison does not isolate the effect of shared states. WCP still has lower count MAE and higher F1 in this family.

Table 12: Multiple-change Story results: 100 documents per generator and true K. The displayed count is not provided to unknown-count detectors. Entries are mean $\mathrm { W D } _ { \mathrm { s u p } }$ and raw CE. RWCP-R leads in five of ten WD settings.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $K = 1$ </td><td colspan="2"> $K = 2$ </td><td colspan="2"> $K = 3$ </td><td colspan="2"> $K = 5$ </td><td colspan="2"> $K = 8$ </td></tr><tr><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \ \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td>CE → 0</td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \ \downarrow }$ </td><td> $\mathrm { C E }  0$ </td></tr><tr><td colspan="9">Claude Haiku 4.5</td><td></td></tr><tr><td>TextTiling</td><td>0.898</td><td>-3.35</td><td>0.702</td><td>-3.57</td><td>0.711</td><td>-4.49</td><td>0.588</td><td>-4.66</td><td>0.483</td><td>-2.90</td></tr><tr><td>LLMPred</td><td>0.714</td><td>-1.99</td><td>0.689</td><td>-1.71</td><td>0.733</td><td>-1.83</td><td>0.624</td><td>0.42</td><td>0.504</td><td>4.84</td></tr><tr><td>SenPred</td><td>2.711</td><td>-11.28</td><td>2.712</td><td>-16.79</td><td>2.619</td><td>-22.17</td><td>2.183</td><td>-27.81</td><td>1.583</td><td>-29.68</td></tr><tr><td>Voting</td><td>0.476</td><td>-1.85</td><td>0.464</td><td>-2.15</td><td>0.475</td><td>-3.22</td><td>0.459</td><td>-4.29</td><td>0.348</td><td>-3.30</td></tr><tr><td>VCP</td><td>0.293</td><td>-1.13</td><td>0.319</td><td>-1.02</td><td>0.315</td><td>-1.32</td><td>0.351</td><td>-1.53</td><td>0.376</td><td>0.35</td></tr><tr><td>WCP</td><td>0.313</td><td>-0.52</td><td>0.337</td><td>-0.19</td><td>0.339</td><td>-0.11</td><td>0.369</td><td>0.39</td><td>0.383</td><td>2.85</td></tr><tr><td>RWCP-R</td><td>0.273</td><td>0.58</td><td>0.360</td><td>1.44</td><td>0.354</td><td>2.10</td><td>0.397</td><td>4.26</td><td>0.409</td><td>7.21</td></tr><tr><td colspan="9">GPT-5-mini</td><td></td></tr><tr><td>TextTiling</td><td>1.120</td><td>-4.39</td><td>0.905</td><td>-4.66</td><td>0.882</td><td>-6.12</td><td>0.734</td><td>-7.01</td><td>0.536</td><td>-4.70</td></tr><tr><td>LLMPred</td><td>0.645</td><td>-1.52</td><td>0.778</td><td>-2.29</td><td>0.696</td><td>-1.16</td><td>0.681</td><td>-0.16</td><td>0.542</td><td>3.99</td></tr><tr><td>SenPred</td><td>3.444</td><td>-13.56</td><td>3.275</td><td>-19.45</td><td>3.225</td><td>-25.81</td><td>2.685</td><td>-33.02</td><td>1.894</td><td>-35.41</td></tr><tr><td>Voting</td><td>1.128</td><td>-4.09</td><td>0.997</td><td>-5.25</td><td>1.018</td><td>-6.78</td><td>0.874</td><td>-8.29</td><td>0.675</td><td>-7.04</td></tr><tr><td>VCP</td><td>0.445</td><td>-0.95</td><td>0.458</td><td>-1.05</td><td>0.451</td><td>-0.82</td><td>0.462</td><td>-0.35</td><td>0.454</td><td>2.03</td></tr><tr><td>WCP</td><td>0.423</td><td>-0.38</td><td>0.404</td><td>0.01</td><td>0.450</td><td>0.31</td><td>0.438</td><td>1.26</td><td>0.438</td><td>4.10</td></tr><tr><td>RWCP-R</td><td>0.342</td><td>0.79</td><td>0.419</td><td>1.51</td><td>0.419</td><td>2.42</td><td>0.427</td><td>4.28</td><td>0.429</td><td>7.22</td></tr></table>

Text perturbations. RWCP-R has the lowest $\mathrm { W D } _ { \mathrm { s u p } }$ in five of six attack settings (Table 13); Voting remains best on paraphrased News. Standard WD falls from 0.3273 for RWCP-block to 0.2676, and count MAE from 0.785 to 0.602. Unlike the earlier block configuration, RWCP-R improves standard WD against WCP in both attacked WikiQA conditions; all six pointwise paired intervals against WCP are negative across the cached perturbations.

Table 13: Text-attack results: 100 records in each of WikiQA, News, and Story per attack. Predictions for nontrivial inputs are unchanged; singleton boundary conventions match Table 9. RWCP-R has the lowest $\mathrm { W D } _ { \mathrm { s u p } }$ in five of six settings; Voting leads on paraphrased News.
<table><tr><td rowspan="2">Method</td><td>WikiQA</td><td></td><td>News</td><td></td><td>Story</td></tr><tr><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td> $\mathrm { C E }  0$ </td></tr><tr><td colspan="7">Decoherence</td></tr><tr><td>TextTiling</td><td>0.388</td><td>-0.09</td><td>1.066</td><td>-4.18</td><td>0.844</td><td>-3.17</td></tr><tr><td>LLMPred</td><td>0.352</td><td>0.37</td><td>0.604</td><td>-1.22</td><td>0.957</td><td>-2.96</td></tr><tr><td>SenPred</td><td>0.383</td><td>-1.30</td><td>2.573</td><td>-10.58</td><td>3.119</td><td>-12.89</td></tr><tr><td>Voting</td><td>0.337</td><td>-0.03</td><td>0.491</td><td>-1.75</td><td>0.993</td><td>-3.82</td></tr><tr><td>WCP RWCP-R</td><td>0.254</td><td>0.44</td><td>0.328</td><td>-0.59 0.60</td><td>0.364 0.328</td><td>-0.45 0.59</td></tr><tr><td>Paraphrasing</td><td>0.226</td><td>0.65</td><td>0.324</td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>TextTiling</td><td>0.389</td><td>-0.06</td><td>1.049</td><td>-4.16</td><td>0.839</td><td>-3.21</td></tr><tr><td>LLMPred</td><td>0.368</td><td>0.30</td><td>0.445</td><td>-0.44</td><td>0.737</td><td>-2.13</td></tr><tr><td>SenPred</td><td>0.294</td><td>-0.96</td><td>1.590</td><td>-6.53</td><td>2.857</td><td>-12.08</td></tr><tr><td>Voting</td><td>0.264</td><td>0.04</td><td>0.107</td><td>-0.24</td><td>0.581</td><td>-2.23</td></tr><tr><td>WCP</td><td>0.187</td><td>0.25</td><td>0.167</td><td>-1.05</td><td>0.336</td><td>-0.75</td></tr><tr><td>RWCP-R</td><td>0.148</td><td>0.37</td><td>0.141</td><td>0.12</td><td>0.303</td><td>0.52</td></tr></table>

LLM-written proportion. RWCP-R leads in $\mathrm { W D } _ { \mathrm { s u p } }$ at 5%, 10%, 40%, and 80%, with WCP leading at 20% (Table 14). The setting-average standard WD is 0.2119, slightly worse than RWCPblock’s 0.2084, while count MAE improves from 0.874 for RWCP-block to 0.736. The consistent gains across four target proportions indicate that the recurrent profile remains effective as the LLMwritten fraction varies.

Table 14: Story with varying target LLM-written fractions, 100 documents per setting. RWCP-R leads in $\mathrm { W D _ { s u p } }$ at 5%, 10%, 40%, and 80%; WCP leads at 20%. Neither the target fraction nor the true boundary is supplied to RWCP-R.
<table><tr><td rowspan="2">Method</td><td colspan="2">5%</td><td colspan="2">10%</td><td colspan="2">20%</td><td colspan="2">40%</td><td colspan="2">80%</td></tr><tr><td> $\mathrm { W D _ { s u p } }$  →</td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td>CE → 0</td><td> $\mathrm { W D _ { s u p } \ . }$ </td><td>CE → 0</td><td> $\mathrm { W D _ { s u p } }$  →</td><td> $\mathrm { C E }  0$ </td><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td>CE → 0</td></tr><tr><td>TextTiling</td><td>1.166</td><td>-3.92</td><td>1.155</td><td>-4.15</td><td>1.048</td><td>-4.15</td><td>1.005</td><td>-3.68</td><td>0.766</td><td>-2.75</td></tr><tr><td>LLMPred</td><td>0.620</td><td>-1.50</td><td>0.700</td><td>-1.70</td><td>0.698</td><td>-1.26</td><td>0.692</td><td>-1.26</td><td>0.525</td><td>-1.06</td></tr><tr><td>SenPred</td><td>4.500</td><td>-16.48</td><td>4.264</td><td>-15.64</td><td>3.880</td><td>-14.77</td><td>3.233</td><td>-13.18</td><td>3.134</td><td>-12.43</td></tr><tr><td>Voting</td><td>1.503</td><td>-5.04</td><td>0.768</td><td>-2.64</td><td>0.671</td><td>-2.40</td><td>0.629</td><td>-2.46</td><td>0.943</td><td>-3.28</td></tr><tr><td>WCP</td><td>0.213</td><td>-1.02</td><td>0.235</td><td>-0.95</td><td>0.228</td><td>-1.13</td><td>0.289</td><td>-0.82</td><td>0.275</td><td>-0.10</td></tr><tr><td>RWCP-R</td><td>0.111</td><td>0.79</td><td>0.185</td><td>0.49</td><td>0.243</td><td>0.48</td><td>0.277</td><td>0.52</td><td>0.271</td><td>0.72</td></tr></table>

## B.7 RECOVERY DIAGNOSTICS AND COMPONENT ABLATIONS

## B.7.1 FAMILY-WISE COMPARISON OF RWCP VARIANTS

Table 15 reports the family-wise results for the preceding analytic and block-calibrated variants alongside RWCP-R. RWCP-R has the lowest WD in four families, while the block variant is lower in the LLM-proportion family. The comparisons retain historical configurations and do not isolate state sharing.

Table 15: Standard raw-boundary WD, equally averaged over settings within families. Best baseline requires complete family coverage. Analytic and Block are historical fixed configurations; their differences from RWCP-R include search, scale, and other settings. This comparison is not an isolated ablation of recurrence.
<table><tr><td rowspan="2">Family</td><td colspan="2">Best baseline</td><td>Analytic</td><td>Block</td><td>RWCP-R</td></tr><tr><td>Method</td><td>WD</td><td>WD</td><td>WD</td><td>WD</td></tr><tr><td>Single change</td><td>Voting</td><td>0.2725</td><td>0.2853</td><td>0.2441</td><td>0.1895</td></tr><tr><td>CoAuthor</td><td>WCP</td><td>0.4878</td><td>0.4922</td><td>0.4945</td><td>0.4869</td></tr><tr><td>Multiple changes</td><td>WCP</td><td>0.4363</td><td>0.4094</td><td>0.4112</td><td>0.3993</td></tr><tr><td>Text attacks</td><td>Voting</td><td>0.3514</td><td>0.3307</td><td>0.3273</td><td>0.2676</td></tr><tr><td>LLM proportion</td><td>WCP</td><td>0.3538</td><td>0.2155</td><td>0.2084</td><td>0.2119</td></tr></table>

## B.7.2 RECOVERY STRESS TESTS

The main text emphasizes WD and signed CE. Table 16 additionally reports absolute count error, matched-boundary F1, the no-boundary fraction, and the WD of a predictor that returns no boundary. These diagnostics distinguish a lower window error from successful boundary recovery and expose count errors that signed averaging can cancel.

Single changes and text attacks improve on the no-boundary predictor by 0.1552 and 0.0772 WD and improve boundary F1 relative to WCP by 0.110 and 0.028. CoAuthor and LLM proportion are within 0.0024 WD of the no-boundary predictor and return no boundary in 84.1% and 66.8% of instances. Their F1 differences against WCP are −0.244 and −0.069; multiple-change data improve modestly on the null WD but lose 0.108 F1. Across the five families, count MAE is 3.019 for RWCP-R versus 2.808 for WCP, and F1 is 0.230 versus 0.287. Thus the primary window-error improvement is accompanied by worse aggregate count and boundary diagnostics. Table 19 gives the paired intervals.

## B.7.3 COMPLETE COMPONENT DIAGNOSTICS

The full component comparison in Table 17 supplements the WD/CE presentation in Table 4. Scale adaptation improves both window errors, count MAE, and boundary F1 at these configurations.

Table 16: Raw-boundary recovery diagnostics, equally averaged over settings. F1 matches boundaries within one observation; zero CP is the no-boundary fraction and Null WD is the no-boundary predictor. The complete-record convention applies throughout.
<table><tr><td rowspan="2">Family</td><td colspan="2">WCP</td><td colspan="3">RWCP-R</td><td rowspan="2">No boundary WD</td></tr><tr><td>MAE</td><td>F1</td><td>MAE</td><td>F1</td><td>Zero CP</td></tr><tr><td>Single change</td><td>1.637</td><td>0.435</td><td>0.340</td><td>0.544</td><td>31.7%</td><td>0.3448</td></tr><tr><td>CoAuthor</td><td>7.917</td><td>0.302</td><td>10.186</td><td>0.058</td><td>84.1%</td><td>0.4893</td></tr><tr><td>Multiple changes</td><td>2.130</td><td>0.205</td><td>3.233</td><td>0.097</td><td>63.9%</td><td>0.4181</td></tr><tr><td>Text attacks</td><td>0.998</td><td>0.272</td><td>0.602</td><td>0.300</td><td>54.5%</td><td>0.3448</td></tr><tr><td>LLM proportion</td><td>1.360</td><td>0.220</td><td>0.736</td><td>0.151</td><td>66.8%</td><td>0.2141</td></tr></table>

Uniform weights improve count MAE and both F1 variants while nearly matching standard WD; the quadratic approximation also improves count MAE and F1 relative to the full method. These fixed-configuration comparisons support metric-specific conclusions rather than a claim that every component improves localization.

Table 17: Frozen component ablations, averaged equally over settings and then five families. Fixed scale uses $\sigma = 1 ;$ ; uniform weights uses exponent zero; the quadratic approximation uses $c = 1 0 ^ { 6 }$ . No variant is retuned. Bold marks the best unrounded value per column; it does not indicate significance.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Window error</td><td rowspan="2">Count error</td><td colspan="2">Boundary F1↑</td></tr><tr><td> $\mathrm { W D _ { s u p } \downarrow }$ </td><td>WD↓</td><td>MAE↓ h = 0</td><td> $h = 1$ </td></tr><tr><td>RWCP-analytic</td><td>0.3233</td><td>0.3466</td><td>3.155</td><td>0.115</td><td>0.176</td></tr><tr><td>RWCP-block</td><td>0.3158</td><td>0.3371</td><td>3.200</td><td>0.111</td><td>0.169</td></tr><tr><td>RWCP-R</td><td>0.2990</td><td>0.3110</td><td>3.019</td><td>0.163</td><td>0.230</td></tr><tr><td>R: fixed scale</td><td>0.3038</td><td>0.3170</td><td>3.071</td><td>0.148</td><td>0.211</td></tr><tr><td>R: uniform weights</td><td>0.3058</td><td>0.3118</td><td>2.917</td><td>0.189</td><td>0.261</td></tr><tr><td>R: quadratic loss</td><td>0.3130</td><td>0.3179</td><td>2.945</td><td>0.166</td><td>0.239</td></tr></table>

## B.7.4 POST-FREEZE INTERNAL STABILITY

Table 18 reports the historical source-group development and internal-check split. Against the block variant, family-macro WD changes from 0.3374 to 0.2990 on development and from 0.3380 to 0.3165 on the internal check. RWCP-R improves four of five check families; LLM proportion is the exception. These sources had already influenced development and do not constitute a fresh independent test. Code checks verify Huber interval fits, fixed-center decoding against exhaustive enumeration, nonincreasing alternating objectives, score/weight unit invariance, known-count behavior, and boundary/label consistency. Production reruns reproduce the frozen predictions across all five families. Runs use CPU with optional Numba acceleration and cached scores; timing excludes model inference and data generation. Fresh-source evaluation, no-change error calibration on actual detector scores, and matched whole-segment labeling remain necessary for stronger generalization and mechanism claims.

Table 18: Post-freeze stability by the existing source-group split. Development (716 nontrivial instances) supplies parameter selection; internal check (1,968 nontrivial instances) follows configuration freezing. Six reporting-only singletons are outside these historical splits. Both subsets come from benchmarks previously used in research development, and the latter is not a new independent test set. Entries are standard WD.
<table><tr><td rowspan="2">Experiment</td><td colspan="2">RWCP-block</td><td colspan="2">RWCP-R</td></tr><tr><td>Dev. WD</td><td>Check WD</td><td>Dev. WD</td><td>Check WD</td></tr><tr><td>Single change</td><td>0.2501</td><td>0.2445</td><td>0.1959</td><td>0.1895</td></tr><tr><td>CoAuthor</td><td>0.4803</td><td>0.4997</td><td>0.4729</td><td>0.4921</td></tr><tr><td>Multiple changes</td><td>0.4075</td><td>0.4125</td><td>0.3861</td><td>0.4036</td></tr><tr><td>Text attacks</td><td>0.3352</td><td>0.3269</td><td>0.2573</td><td>0.2749</td></tr><tr><td>LLM proportion</td><td>0.2138</td><td>0.2064</td><td>0.1827</td><td>0.2224</td></tr><tr><td>Family macro</td><td>0.3374</td><td>0.3380</td><td>0.2990</td><td>0.3165</td></tr></table>

## B.8 ALGORITHMS AND EXPERIMENTAL DETAILS

This appendix describes the core Huber solver and block calibration, followed by the metrics used in the completed cached-score evaluation. Algorithm 1 specifies the distinct recurrent solver used for the main empirical results.

One-dimensional Huber optimization. For interval A, define the monotone score

$$
G _ { A } ( \theta ) = \sum _ { i \in { \cal A } } \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta ) \} .
$$

Because $\psi _ { c }$ is nondecreasing in its argument and $Y _ { i } - \theta$ decreases with $\theta , G _ { A }$ is nonincreasing. An interior minimizer satisfies $G _ { A } ( \theta ) = 0 . { \mathrm { ~ I f ~ } } G _ { A } ( \theta ) > 0$ throughout Θ, the minimizer is the right endpoint; if it is negative throughout, the minimizer is the left endpoint.

Proposition B.1 (Optimization accuracy). Suppose the empirical loss on A is α $S _ { A } ^ { w }$ -strongly convex in a neighborhood containing the exact minimizer $\widehat { \theta } _ { A }$ and numerical solution $ { \widetilde { \theta } } _ { A }$ , and suppose $\widehat { \theta } _ { A } \in \mathrm { i n t } ( \Theta ) . \ I f \vert \widetilde { \theta } _ { A } - \widehat { \theta } _ { A } \vert \leq \epsilon _ { \mathrm { o p t } }$ , then

$$
0 \leq \widehat { \mathcal { L } } _ { A } ( \widetilde { \theta } _ { A } ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \theta } _ { A } ) \leq \frac { 1 } { 2 } S _ { A } ^ { w } \epsilon _ { \mathrm { o p t } } ^ { 2 } .
$$

For a profile gain, the total numerical error is at most

$$
2 S _ { s : e } ^ { w } \epsilon _ { \mathrm { o p t } } ^ { 2 } .\tag{19}
$$

Proof. The Huber empirical loss has gradient Lipschitz constant $S _ { A } ^ { w }$ by Lemma C.10. Since the exact minimizer has zero subgradient in the interior, smoothness gives the first upper bound. The profile gain contains one parent and two child optimized losses. Their masses sum to $2 S _ { s : e } ^ { w }$ after applying the factor two in (3), which yields the conservative bound (19). □

If the constrained optimum is a boundary point, Algorithm 2 returns that endpoint exactly from the score-sign check; the quadratic numerical-error proposition is used only on the interior branch.

To keep numerical error below the statistical threshold, it is enough to choose

$$
\epsilon _ { \mathrm { o p t } } \leq c _ { \mathrm { o p t } } \sqrt { \frac { \Lambda _ { N } } { W _ { N } } } .
$$

In practice a fixed tolerance such as $1 0 ^ { - 6 }$ is usually much smaller.

Warm starts and cached evaluations. For neighboring candidate splits b and b + 1, the left and right intervals differ by one observation. Their Huber minimizers typically change smoothly. Use $\widehat { \theta } _ { s : b }$ to initialize the solve for $[ s , b + 1 ]$ and $\widehat { \theta } _ { b + 1 : e }$ for $[ b + 2 , e ]$ . A safeguarded Newton step is

$$
\theta ^ { ( r + 1 ) } = \Pi _ { \Theta } \left[ \theta ^ { ( r ) } + \frac { G _ { A } ( \theta ^ { ( r ) } ) } { H _ { A } ( \theta ^ { ( r ) } ) \vee h _ { \operatorname* { m i n } } } \right] ,
$$

Algorithm 2 Bisection for an interval Huber minimizer   
Require: Interval $A ,$ scores, weights, c, parameter range $[ L , U ] .$ , tolerance $\epsilon _ { \mathrm { o p t } }$   
1: if $G _ { A } ( L ) \leq 0$ then   
2: return L.   
3: else if $G _ { A } ( U ) \geq 0$ then   
4: return $U .$   
5: end if   
6: while $U - L > \epsilon _ { \mathrm { o p t } }$ do   
7: $M  ( L + U ) / 2 .$   
8: if $G _ { A } ( \dot { M } ) > 0$ then   
9: $L \gets M .$   
10: else   
11: $U  M .$   
12: end if   
13: end while   
14: return $( L + U ) / 2 .$

where $H _ { A }$ is defined in (26). If the proposed step does not reduce the loss, revert to bisection. This hybrid retains deterministic convergence while exploiting local quadratic behavior.

For a fixed iterate θ, the score and Hessian can be evaluated from prefix sums of

$$
a _ { i } ( \theta ) = \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta ) \} , \qquad h _ { i } ( \theta ) = w _ { i } { \bf 1 } \{ | \sqrt { w _ { i } } ( Y _ { i } - \theta ) | < c \} .
$$

Because these quantities depend on $\theta ,$ caching is most effective within batches of nearby intervals sharing warm starts.

Complexity accounting. Let $L _ { m } = e _ { m } - s _ { m } + 1$ be the length of random interval m and let J be the average number of score evaluations required by the one-dimensional solver. A direct scan has cost

$$
O \left( J \sum _ { m = 1 } ^ { M } L _ { m } ^ { 2 } \right)
$$

because each of $O ( L _ { m } )$ splits requires losses over $O ( L _ { m } )$ observations. With cached interval data and warm-started score evaluations, the practical cost is closer to

$$
O \left( J _ { \mathrm { w a r m } } \sum _ { m = 1 } ^ { M } L _ { m } + \sum _ { m = 1 } ^ { M } L _ { m } \log L _ { m } \right) ,
$$

although this is an implementation-dependent empirical complexity rather than a proved worst-case bound. Memory is ${ \bar { O } } ( N + M )$ excluding detector inference. Block calibration multiplies the segmentation-statistic cost by approximately Q, but repetitions are embarrassingly parallel.

Weighted interval generation. Generate cumulative weights once and use binary search in $W ( \tilde { 1 } ) , \ldots , W ( N )$ to map a uniform coordinate to an index. One interval costs ${ \cal O } ( \log N )$ time, so pre-sampling M intervals costs $O ( M \log N )$ . Deduplicate repeated endpoint pairs before profile evaluation. The random seed and the final endpoint list should be stored to make every comparison deterministic across segmentation methods.

One-step multiplier profile approximation. For a homogeneous pilot interval A, the first-order Huber estimator expansion is

$$
\widehat { \theta } _ { A } - \theta _ { A } \approx \frac { \sum _ { i \in A } \zeta _ { i } } { H _ { A } ( \theta _ { A } ) } ,
$$

where $\zeta _ { i }$ is the centered score. In a bootstrap repetition, replace the score sum by $\textstyle \sum _ { i \in A } \zeta _ { i } ^ { \star }$ and curvature by its pilot estimate. The approximate quadratic profile gain is then

$$
\Gamma _ { s , e } ^ { \star , \mathrm { 1 s t e p } } ( b ) = \frac { \Bigl ( H _ { s : b } ^ { - 1 } \sum _ { i = s } ^ { b } \zeta _ { i } ^ { \star } - H _ { b + 1 : e } ^ { - 1 } \sum _ { i = b + 1 } ^ { e } \zeta _ { i } ^ { \star } \Bigr ) ^ { 2 } } { H _ { s : b } ^ { - 1 } + H _ { b + 1 : e } ^ { - 1 } } ,\tag{20}
$$

where $H _ { a : b }$ denotes the pilot Hessian sum. Formula (20) is the robust analogue of a squared standardized difference. It is used only for calibration, not in the main estimator.

## B.8.1 METRIC DEFINITIONS AND REPRODUCIBILITY

WindowDiff and boundary metrics. Let $C _ { i } ^ { ( r ) }$ and $C _ { i } ^ { ( h ) }$ be the true and estimated numbers of boundaries in window $[ i , i + k ]$ . WindowDiff is

$$
\mathrm { W D } = \frac { 1 } { N - k } \sum _ { i = 1 } ^ { N - k } { \bf 1 } \{ C _ { i } ^ { ( r ) } \ne C _ { i } ^ { ( h ) } \} .\tag{21}
$$

For document $d ,$ use the pre-specified width $k = k _ { d }$ in (18); this keeps the implementation identical across methods while adapting to reference segment length. Count Error is

$$
\mathrm { C E } = K - { \widehat K } .
$$

Match an estimate to a true boundary when $| { \widehat { \tau } } - \tau | \leq h$ , using maximum bipartite matching to avoid multiple credit. The completed RWCP-R evaluation reports F1 for $h = 1$ observation and the exact-match case $h = 0$ . Precision, recall, and F1 follow from matched, unmatched estimated, and unmatched true boundaries.

The empirical evidence consists of saved score streams, predictions, source-group assignments, selected configurations, and metric recomputation. The reporting correction includes every controlled record and explicitly records the singleton boundary convention. It preserves the historical development/check memberships and all nontrivial predictions. Generation revisions, complete original-source identifiers, and end-to-end detector timings are not fully available; the tables therefore establish cache-level reproducibility. No fitted detector-score tail law, empirical mixing verification, or calibrated no-change false-positive rate is claimed here. Those missing measurements delimit the connection between the score-model assumptions and these applications.

## B.8.2 UNCERTAINTY IN AGGREGATE AND COMPONENT DIFFERENCES

The per-setting intervals condition on one setting at a time. To assess the family-macro contrasts, we additionally resample the available source components jointly across the entire primary suite. Each draw assigns one multiplicity to a component and reuses it in every setting where that component occurs. We then recompute document means within settings, equally average settings within each family, and equally average the five families. This preserves known cross-setting reuse. The corrected reporting inventory contains 647 metadata-linked components: 643 historical components plus four from the singleton records under the same exact-source/surviving-human-text linkage rule. Missing source identifiers still preclude a claim of full independence.

Table 19: Exploratory family-macro differences: RWCP-R minus comparator, with pointwise paired 95% source-component bootstrap intervals (2,000 draws, seed 121209). Lower WD/MAE and higher F1 are better. Configurations are fixed, linkage is incomplete, and there is no multiplicity correction.
<table><tr><td rowspan="2">Comparator</td><td colspan="3">RWCP-R minus comparator: paired 95% intervals</td></tr><tr><td>∆WD [95% CI]</td><td>∆MAE [95% CI]</td><td>∆F1 [95% CI]</td></tr><tr><td>WCP</td><td>-0.0959 [-0.1074, -0.0844]</td><td>+0.211 [+0.126, +0.293]</td><td>-0.057 [-0.073, -0.040]</td></tr><tr><td>R: fixed scale</td><td>-0.0060 [-0.0099, -0.0021]</td><td>-0.052 [-0.071, -0.033]</td><td>+0.019 [+0.010, +0.027]</td></tr><tr><td>R: uniform weights</td><td>-0.0008 [-0.0052, +0.0038]</td><td>+0.103 [+0.083, +0.124]</td><td>-0.031 [-0.040, -0.023]</td></tr><tr><td>R: quadratic approximation</td><td>-0.0068 [-0.0108, -0.0033]</td><td>+0.074 [+0.052, +0.098]</td><td>-0.009 [-0.016, -0.001]</td></tr></table>

Table 19 sharpens the scope of the benefit. Against WCP, the window-error improvement coexists with higher count MAE and lower boundary F1. Against uniform weights, the WD interval includes zero while the count and F1 intervals favor uniform weights. Against fixed scale, all three intervals favor scale adaptation at the selected configurations. The large-cutoff quadratic approximation trades worse WD for better count and F1. These are conditional comparisons of the saved configurations, not evidence from a newly sampled test corpus or equally retuned alternative models.

## B.8.3 PER-SETTING PAIRED RESULTS

The complete standard-WD paired table is Table 20; the preceding appendix tables retain the fuller comparator-specific and legacy-metric views.

Table 20: Primary 25-setting study: standard raw-boundary WD and RWCP-R F1 (tolerance one). Every controlled setting includes 100 records; CoAuthor has 290. Intervals are pointwise paired 95% source-component bootstrap intervals for RWCP-R minus WCP (2,000 draws); negative values favor RWCP-R. Singleton boundary conventions are applied to all methods.
<table><tr><td rowspan="2">Family</td><td rowspan="2">Setting</td><td rowspan="2">WCP</td><td colspan="2">RWCP-R</td><td>RWCP-R-WCP</td></tr><tr><td>WD WD</td><td>F1 (h = 1)</td><td>∆WD [95% CI]</td></tr><tr><td>Single</td><td>WikiQA</td><td>0.284</td><td>0.223</td><td>0.490</td><td>-0.060 [-0.107, -0.013]</td></tr><tr><td>Single</td><td>News</td><td>0.370</td><td>0.102</td><td>0.790</td><td>-0.268 [-0.327, -0.210]</td></tr><tr><td>Single</td><td>Story</td><td>0.534</td><td>0.243</td><td>0.353</td><td>-0.291 [-0.342, -0.238]</td></tr><tr><td>CoAuthor</td><td>290 sessions</td><td>0.488</td><td>0.487</td><td>0.058</td><td>-0.001 [-0.019, 0.015]</td></tr><tr><td>Multiple</td><td>Claude K = 1</td><td>0.400</td><td>0.288</td><td>0.180</td><td>-0.112 [-0.156, -0.072]</td></tr><tr><td>Multiple</td><td>Claude  $K = 2$ </td><td>0.404</td><td>0.374</td><td>0.140</td><td>-0.031 [-0.063, 0.000]</td></tr><tr><td>Multiple</td><td>Claude K = 3</td><td>0.425</td><td>0.378</td><td>0.178</td><td>-0.047 [-0.075, -0.019]</td></tr><tr><td>Multiple</td><td>Claude K = 5</td><td>0.425</td><td>0.429</td><td>0.107</td><td>0.004 [-0.018, 0.024]</td></tr><tr><td>Multiple</td><td>Claude K = 8</td><td>0.430</td><td>0.448</td><td>0.079</td><td>0.018 [0.002, 0.036]</td></tr><tr><td>Multiple</td><td>GPT K = 1</td><td>0.434</td><td>0.339</td><td>0.047</td><td>-0.095 [-0.131, -0.060]</td></tr><tr><td>Multiple</td><td>GPT K = 2</td><td>0.430</td><td>0.408</td><td>0.055</td><td>-0.022 [-0.047, 0.003]</td></tr><tr><td>Multiple</td><td>GPT K = 3</td><td>0.470</td><td>0.418</td><td>0.071</td><td>-0.052 [-0.071, -0.032]</td></tr><tr><td>Multiple</td><td>GPT K = 5</td><td>0.475</td><td>0.447</td><td>0.066</td><td>-0.028 [-0.045, -0.012]</td></tr><tr><td>Multiple</td><td>GPT K = 8</td><td>0.471</td><td>0.464</td><td>0.044</td><td>-0.006 [-0.019, 0.007]</td></tr><tr><td>Attack</td><td>WikiQA decoh.</td><td>0.340</td><td>0.291</td><td>0.277</td><td>-0.049 [-0.086, -0.011]</td></tr><tr><td>Attack</td><td>News decoh.</td><td>0.407</td><td>0.335</td><td>0.067</td><td>-0.072 [-0.117, -0.030]</td></tr><tr><td>Attack</td><td>Story decoh.</td><td>0.398</td><td>0.335</td><td>0.067</td><td>-0.063 [-0.105, -0.021]</td></tr><tr><td>Attack</td><td>WikiQA para.</td><td>0.266</td><td>0.189</td><td>0.563</td><td>-0.077 [-0.124, -0.034]</td></tr><tr><td>Attack</td><td>News para.</td><td>0.326</td><td>0.146</td><td>0.648</td><td>-0.180 [-0.236, -0.126]</td></tr><tr><td>Attack</td><td>Story para.</td><td>0.428</td><td>0.310</td><td>0.175</td><td>-0.118 [-0.157, -0.079]</td></tr><tr><td>Proportion</td><td>5%</td><td>0.321</td><td>0.104</td><td>0.037</td><td>-0.217 [-0.270, -0.169]</td></tr><tr><td>Proportion</td><td>10%</td><td>0.338</td><td>0.172</td><td>0.183</td><td>-0.166 [-0.219, -0.118]</td></tr><tr><td>Proportion</td><td>20%</td><td>0.375</td><td>0.235</td><td>0.210</td><td>-0.140 [-0.190, -0.097]</td></tr><tr><td>Proportion</td><td>40%</td><td>0.404</td><td>0.278</td><td>0.203</td><td>-0.126 [-0.174, -0.079]</td></tr><tr><td></td><td>80%</td><td>0.330</td><td>0.270</td><td></td><td></td></tr><tr><td>Proportion</td><td></td><td></td><td></td><td>0.122</td><td>-0.060 [-0.097, -0.024]</td></tr></table>

Table 21: Primary 25-setting legacy final-label window error $\mathrm { W D _ { s u p } }$ and RWCP-R raw count error. Best base is the smallest available non-RWCP mean in each setting. Bold marks the smaller legacy error between it and RWCP-R before rounding. These are distinct from raw-boundary WD and matched-boundary F1.
<table><tr><td rowspan="2">Family</td><td rowspan="2">Setting</td><td colspan="2">Best baseline</td><td>WCP</td><td colspan="2">RWCP-R</td></tr><tr><td>Method</td><td> $\mathrm { W D _ { s u p } }$ </td><td> $\mathrm { W D _ { s u p } }$ </td><td> $\mathrm { W D _ { s u p } }$ </td><td>CE</td></tr><tr><td>Single</td><td>WikiQA</td><td>VCP</td><td>0.200</td><td>0.200</td><td>0.184</td><td>0.47</td></tr><tr><td>Single</td><td>News</td><td>WCP</td><td>0.123</td><td>0.123</td><td>0.097</td><td>0.03</td></tr><tr><td>Single</td><td>Story</td><td>VCP</td><td>0.306</td><td>0.372</td><td>0.235</td><td>0.34</td></tr><tr><td>CoAuthor</td><td>290 sessions</td><td>VCP</td><td>0.480</td><td>0.483</td><td>0.478</td><td>10.17</td></tr><tr><td>Multiple</td><td>Claude  $K = 1$ </td><td>VCP</td><td>0.293</td><td>0.313</td><td>0.273</td><td>0.58</td></tr><tr><td>Multiple</td><td>Claude  $K = 2$ </td><td>VCP</td><td>0.319</td><td>0.337</td><td>0.360</td><td>1.44</td></tr><tr><td>Multiple</td><td>Claude K = 3</td><td>VCP</td><td>0.315</td><td>0.339</td><td>0.354</td><td>2.10</td></tr><tr><td>Multiple</td><td>Claude K = 5</td><td>VCP</td><td>0.351</td><td>0.369</td><td>0.397</td><td>4.26</td></tr><tr><td>Multiple</td><td>Claude K = 8</td><td>Voting</td><td>0.348</td><td>0.383</td><td>0.409</td><td>7.21</td></tr><tr><td>Multiple</td><td>GPT K = 1</td><td>WCP</td><td>0.423</td><td>0.423</td><td>0.342</td><td>0.79</td></tr><tr><td>Multiple</td><td>GPT K = 2</td><td>WCP</td><td>0.404</td><td>0.404</td><td>0.419</td><td>1.51</td></tr><tr><td>Multiple</td><td>GPT K = 3</td><td>WCP</td><td>0.450</td><td>0.450</td><td>0.419</td><td>2.42</td></tr><tr><td>Multiple</td><td>GPT K = 5</td><td>WCP</td><td>0.438</td><td>0.438</td><td>0.427</td><td>4.28</td></tr><tr><td>Multiple</td><td>GPT K = 8</td><td>WCP</td><td>0.438</td><td>0.438</td><td>0.429</td><td>7.22</td></tr><tr><td>Attack</td><td>WikiQA decoh.</td><td>WCP</td><td>0.254</td><td>0.254</td><td>0.226</td><td>0.65</td></tr><tr><td>Attack</td><td>News decoh.</td><td>WCP</td><td>0.328</td><td>0.328</td><td>0.324</td><td>0.60</td></tr><tr><td>Attack</td><td>Story decoh.</td><td>WCP</td><td>0.364</td><td>0.364</td><td>0.328</td><td>0.59</td></tr><tr><td>Attack</td><td>WikiQA para.</td><td>WCP</td><td>0.187</td><td>0.187</td><td>0.148</td><td>0.37</td></tr><tr><td>Attack</td><td>News para.</td><td>Voting</td><td>0.107</td><td>0.167</td><td>0.141</td><td>0.12</td></tr><tr><td>Attack</td><td>Story para.</td><td>WCP</td><td>0.336</td><td>0.336</td><td>0.303</td><td>0.52</td></tr><tr><td>Proportion</td><td>5%</td><td>WCP</td><td>0.213</td><td>0.213</td><td>0.111</td><td>0.79</td></tr><tr><td>Proportion</td><td>10%</td><td>WCP</td><td>0.235</td><td>0.235</td><td>0.185</td><td>0.49</td></tr><tr><td>Proportion</td><td>20%</td><td>WCP</td><td>0.228</td><td>0.228</td><td>0.243</td><td>0.48</td></tr><tr><td>Proportion</td><td>40%</td><td>WCP</td><td>0.289</td><td>0.289</td><td>0.277</td><td>0.52</td></tr><tr><td>Proportion</td><td>80%</td><td>WCP</td><td>0.275</td><td>0.275</td><td>0.271</td><td>0.72</td></tr></table>

## C THEORETICAL PROOFS

## C.1 TECHNICAL PRELIMINARIES AND POPULATION GEOMETRY

The master collection is $\mathcal { A } _ { N } = \{ [ s , e ] : 1 \leq s \leq e \leq N \}$ ; subintervals arising in any recursive call therefore belong to it. For $I = [ s , e ]$ , the admissible set is

$$
{ \mathcal { B } } ( I ) = \{ b \in \{ s , \ldots , e - 1 \} : S _ { s : b } ^ { w } \geq m _ { N } , \ S _ { b + 1 : e } ^ { w } \geq m _ { N } \} .
$$

Set $T ( I ) = - \infty$ if this set is empty. Draw two independent uniform coordinates on $( 0 , W _ { N } ]$ , map each to min $\{ i : W ( i ) \geq u \}$ , and sort the resulting endpoints. Draw the candidate collection once, independently of the scores conditional on deterministic weights; recursive calls retain only intervals fully contained in their search range. Ties are resolved deterministically by endpoints and then split index. The uniform events cover the full master collection, not only the sampled intervals.

This appendix collects the deterministic notation and population-risk arguments underlying the profile margin in Section 3. It first records merge identities, then establishes the behavior of Huber minimizers, and finally proves the single-change population result.

For an interval $A = [ a , b ]$ , write $S _ { A } ^ { w } = S _ { a : b } ^ { w }$ and

$$
{ \widehat { L } } _ { A } ^ { \star } = \operatorname* { i n f } _ { \theta \in \Theta } { \widehat { \mathcal { L } } } _ { A } ( \theta ) , \qquad L _ { A } ^ { \star } = \operatorname* { i n f } _ { \theta \in \Theta } { \mathcal { L } } _ { A } ( \theta ) .
$$

For adjacent intervals A, B, empirical and population merge costs are

$$
\begin{array} { r } { \widehat { D } ( A , B ) = \widehat { L } _ { A \cup B } ^ { \star } - \widehat { L } _ { A } ^ { \star } - \widehat { L } _ { B } ^ { \star } , } \\ { D ^ { \star } ( A , B ) = { L } _ { A \cup B } ^ { \star } - { L } _ { A } ^ { \star } - { L } _ { B } ^ { \star } . } \end{array}
$$

Both are nonnegative because a common parameter is more restrictive than separate parameters.

Lemma C.1 (Profile-gain difference identity). Suppose $s \leq b < \tau <$ e and set

$$
A = [ s , b ] , \qquad B = [ b + 1 , \tau ] , \qquad C = [ \tau + 1 , e ] .
$$

Then

$$
\begin{array} { r l } & { \Gamma _ { s , e } ^ { \star } ( \tau ) - \Gamma _ { s , e } ^ { \star } ( b ) = 2 \{ D ^ { \star } ( B , C ) - D ^ { \star } ( A , B ) \} , } \\ & { \widehat { \Gamma } _ { s , e } ( \tau ) - \widehat { \Gamma } _ { s , e } ( b ) = 2 \{ \widehat { D } ( B , C ) - \widehat { D } ( A , B ) \} . } \end{array}
$$

IfA and B belong to the same true segment, then $D ^ { \star } ( A , B ) = 0 .$

Proof. By definition,

$$
\frac { 1 } { 2 } \Gamma _ { s , e } ^ { \star } ( \tau ) = L _ { A \cup B \cup C } ^ { \star } - L _ { A \cup B } ^ { \star } - L _ { C } ^ { \star }
$$

and

$$
\frac { 1 } { 2 } \Gamma _ { s , e } ^ { \star } ( b ) = L _ { A \cup B \cup C } ^ { \star } - L _ { A } ^ { \star } - L _ { B \cup C } ^ { \star } .
$$

Subtracting cancels the parent term and gives

$$
\frac { 1 } { 2 } \{ \Gamma _ { s , e } ^ { \star } ( \tau ) - \Gamma _ { s , e } ^ { \star } ( b ) \} = L _ { A } ^ { \star } + L _ { B \cup C } ^ { \star } - L _ { A \cup B } ^ { \star } - L _ { C } ^ { \star } .
$$

Add and subtract $L _ { B } ^ { \star }$ to obtain

$$
\{ L _ { B \cup C } ^ { \star } - L _ { B } ^ { \star } - L _ { C } ^ { \star } \} - \{ L _ { A \cup B } ^ { \star } - L _ { A } ^ { \star } - L _ { B } ^ { \star } \} ,
$$

which is $D ^ { \star } ( B , C ) - D ^ { \star } ( A , B )$ . The empirical identity is identical with hats. If A and B share minimizer $\theta _ { - }$ , then

$$
L _ { A \cup B } ^ { \star } \le \mathcal { L } _ { A } ( \theta _ { - } ) + \mathcal { L } _ { B } ( \theta _ { - } ) = L _ { A } ^ { \star } + L _ { B } ^ { \star } .
$$

The reverse inequality follows from separate minimization, so equality holds and $D ^ { \star } ( A , B ) = 0$ □

Lemma C.2 (Harmonic-mass inequalities). For x, $y > 0$

$$
{ \frac { 1 } { 2 } } \operatorname* { m i n } ( x , y ) \leq { \frac { x y } { x + y } } \leq \operatorname* { m i n } ( x , y ) .\tag{22}
$$

Moreover,for all $a , b , c \geq 0$

$$
( a - b - c ) _ { + } ^ { 2 } \geq \frac 1 2 a ^ { 2 } - 2 b ^ { 2 } - 2 c ^ { 2 } .\tag{23}
$$

Proof. Assume $x \ \leq \ y .$ . Then $x + y \leq 2$ y gives $x y / ( x + y ) \ge x / 2$ , while $x + y \ge y$ gives $x y / ( x + y ) \leq x .$ . For the second inequality, $\operatorname { f } a \leq b + c ,$ , the right side is at most $a ^ { 2 } / 2 - \overset { \vartriangle } { \left( b + c \right) } ^ { 2 } \le 0$ I $\therefore a > b + c ,$ use $( a - d ) ^ { 2 } = a ^ { 2 } - { \dot { 2 } } a d { \dot { + } } d ^ { 2 } \geq a ^ { 2 } / 2 - d ^ { 2 }$ with $d = b + c ,$ , followed by $( b + c ) ^ { \overline { { 2 } } } \leqq$ $2 b ^ { 2 } + 2 c ^ { 2 }$

## C.1.1 POPULATION GEOMETRY PROOFS

## Symmetric-location proposition.

Proof. Fix i in segment $j$ and write $a _ { i } = \sigma _ { i } / s _ { i } > 0$ . For $u = \theta - \theta _ { j }$

$$
\begin{array} { r } { Q _ { i } ( \theta _ { j } + u ) = \mathbb { E } \rho _ { c } ( a _ { i } \xi _ { i } - u / s _ { i } ) . } \end{array}
$$

The function is convex in u. At $u = 0$ , an interior subgradient is

$$
- \frac { 1 } { s _ { i } } \mathbb { E } \psi _ { c } ( a _ { i } \xi _ { i } ) .
$$

The Huber score is odd and the law of $\xi _ { i }$ is symmetric, hence $\mathbb { E } { \psi } _ { c } ( a _ { i } \xi _ { i } ) = 0$ . Therefore zero belongs to the subdifferential and $u = 0$ minimizes the convex risk. If $\mathbb { E } | \xi _ { i } | < \infty$ and $\mathbb { E } \xi _ { i } = 0$ , then

$$
\mathbb { E } Y _ { i } = \theta _ { j } + \sigma _ { i } \mathbb { E } \xi _ { i } = \theta _ { j } .
$$

This last conclusion concerns the ordinary mean and uses integrability; it is not implied by Huber centering alone. □

## Existence and uniqueness of interval minimizers.

Lemma C.3 (Existence). For every nonempty interval A, the empirical loss $\widehat { \mathcal { L } } _ { A }$ and population loss $\mathcal { L } _ { A }$ attain minima on compact Θ.

Proof. The Huber loss is continuous. Therefore $\widehat { \mathcal { L } } _ { A }$ , a finite sum of continuous functions, is continuous on compact Θ and attains its minimum. Since $\rho _ { c }$ is nonnegative and Lipschitz with constant $c ,$ dominated convergence on the compact parameter set gives continuity of each $Q _ { i }$ whenever $\mathbb { E } \rho _ { c } ( \sqrt { w _ { i } } ( Y _ { i } - \theta _ { 0 } ) ) < \infty$ for one $\theta _ { 0 } \in \Theta$ . Thus $\mathcal { L } _ { A }$ is continuous and also attains its minimum.

Lemma C.4 (Uniqueness under curvature). If A is contained in one segment and Assumption 3.2 holds, then $\theta _ { A } ^ { \star }$ is unique and equals that segment’s $\theta _ { j }$

Proof. Summing (7) over $i \in A$ yields

$$
\mathcal { L } _ { A } ( \theta ) - \mathcal { L } _ { A } ( \theta _ { j } ) \geq \frac { m _ { 0 } } { 2 } S _ { A } ^ { w } ( \theta - \theta _ { j } ) ^ { 2 }
$$

for $| \theta - \theta _ { j } | \leq r _ { 0 }$ . If a distinct global minimizer existed, convexity would force the aggregate risk to be constant on the entire line segment joining it to $\theta _ { j }$ . That segment contains points arbitrarily close to $\theta _ { j } ,$ , contradicting the strictly positive local quadratic lower bound above. Hence $\theta _ { j }$ is the unique interval minimizer. □

## Mixed-segment excess risk.

Proof. Let $x = S _ { A } ^ { w } , y = S _ { B } ^ { w } , a = \theta _ { A }$ , and $b = \theta _ { B }$ . By curvature,

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { A } ( \theta ) - L _ { A } ^ { \star } \geq \frac { m _ { 0 } x } { 2 } ( \theta - a ) ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { B } ( \theta ) - L _ { B } ^ { \star } \geq \frac { m _ { 0 } y } { 2 } ( \theta - b ) ^ { 2 } } \end{array}
$$

for every $\theta$ on the segment joining a and b. A minimizer of the sum lies between a and b because the first derivative is nonpositive to the left of both minimizers and nonnegative to the right. Therefore

$$
\begin{array} { l } { { D ^ { \star } ( A , B ) = \operatorname* { i n f } _ { \theta } \{ \mathcal { L } _ { A } ( \theta ) - L _ { A } ^ { \star } + \mathcal { L } _ { B } ( \theta ) - L _ { B } ^ { \star } \} } } \\ { { \displaystyle \geq \frac { m _ { 0 } } { 2 } \operatorname* { i n f } _ { \theta } \{ x ( \theta - a ) ^ { 2 } + y ( \theta - b ) ^ { 2 } \} . } } \end{array}
$$

The quadratic minimizer is $( x a + y b ) / ( x + y )$ and the minimum is

$$
{ \frac { x y } { x + y } } ( a - b ) ^ { 2 } .
$$

This proves (9). The upper curvature bound evaluated at the same weighted average gives (10).

## Single-change population margin.

Proof. Consider $b < \tau$ and use the intervals $A , B , C$ from Lemma C.1. Since A and B belong to the left segment, $D ^ { \star } ( A , B ) = 0$ . Therefore

$$
\Gamma _ { s , e } ^ { \star } ( \tau ) - \Gamma _ { s , e } ^ { \star } ( b ) = 2 D ^ { \star } ( B , C ) .
$$

Lemma 3.7 gives

$$
2 D ^ { \star } ( B , C ) \geq m _ { 0 } \kappa ^ { 2 } \frac { B _ { w } C _ { w } } { B _ { w } + C _ { w } } ,
$$

which proves (11). The case $b > \tau$ is symmetric after exchanging left and right. The difference is nonnegative for every b and is strictly positive when the candidate differs from τ, so the true split is the unique maximizer among admissible splits. Finally, if $B _ { w } \le C _ { w }$ , Lemma C.2 yields

$$
\frac { B _ { w } C _ { w } } { B _ { w } + C _ { w } } \geq \frac { 1 } { 2 } B _ { w } = \frac { 1 } { 2 } d _ { w } ( b , \tau ) .
$$

The other cases are identical.

## Quadratic-loss reduction.

Proof. Under $\rho _ { \infty } ( u ) = u ^ { 2 } / 2$

$$
\widehat { \mathcal { L } } _ { a : b } ( \boldsymbol { \theta } ) = \frac { 1 } { 2 } \sum _ { i = a } ^ { b } w _ { i } ( Y _ { i } - \boldsymbol { \theta } ) ^ { 2 }
$$

and the minimizer is $\bar { Y } _ { a : b } ^ { w }$ . The weighted analysis-of-variance identity gives

$$
\begin{array} { l } { { \displaystyle \sum _ { i = s } ^ { e } w _ { i } ( Y _ { i } - \bar { Y } _ { s : e } ^ { w } ) ^ { 2 } - \sum _ { i = s } ^ { b } w _ { i } ( Y _ { i } - \bar { Y } _ { s : b } ^ { w } ) ^ { 2 } - \sum _ { i = b + 1 } ^ { e } w _ { i } ( Y _ { i } - \bar { Y } _ { b + 1 : e } ^ { w } ) ^ { 2 } } } \\ { { \displaystyle \quad \quad = \frac { S _ { s : b } ^ { w } S _ { b + 1 : e } ^ { w } } { S _ { s : e } ^ { w } } ( \bar { Y } _ { s : b } ^ { w } - \bar { Y } _ { b + 1 : e } ^ { w } ) ^ { 2 } . } } \end{array}
$$

Multiplication by the factor two in (3) cancels the one-half in the loss, proving (4).

## C.2 STOCHASTIC CONTROL AND EXACT-RECOVERY PROOF

This appendix supplies the stochastic part of the recovery proof. We state a valid dependenceaware concentration bound, transfer it to interval fits and merge costs, and combine these bounds with weighted interval isolation and boundary-fragment control to prove exact recovery. For a homogeneous interval A with true robust location $\theta _ { A }$ , write

$$
U _ { A } = \sum _ { i \in { \cal A } } \sqrt { w _ { i } } \psi _ { c } ( \sqrt { w _ { i } } ( Y _ { i } - \theta _ { A } ) ) .
$$

Lemma C.5 (Uniform Huber-score concentration). Under Assumptions 3.1, 3.3, and 3.4, with probability at least $1 - \delta / 8$

$$
\vert U _ { A } \vert \leq C _ { \mathrm { d e p } } \left\{ \sqrt { S _ { A } ^ { w } x _ { N } ( \delta ) } + q _ { N } x _ { N } ( \delta ) \right\}
$$

simultaneously for every homogeneous $A \in \mathcal A _ { N }$ , without a minimum-mass restriction. Moreover, whenever $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$

$$
\vert U _ { A } \vert \leq C _ { \mathrm { d e p } } ^ { \prime } \sqrt { S _ { A } ^ { w } \Lambda _ { N } ( \delta ) } .\tag{24}
$$

Lemma C.6 (Uniform empirical curvature). Under Assumptions $3 . 3 \mathrm { - } 3 . 5 ,$ , with probability at least $1 - \delta / 8 ,$

$$
\widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \theta _ { A } ) \geq - U _ { A } ( \theta - \theta _ { A } ) + \frac { p _ { 0 } } { 8 } S _ { A } ^ { w } ( \theta - \theta _ { A } ) ^ { 2 }\tag{25}
$$

for every homogeneous $A \in \mathcal A _ { N }$ with $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ and every $| \theta - \theta _ { A } | \leq r _ { 0 } / 2$

We first establish a blocking inequality used for Huber scores and empirical curvature indicators. Constants in this appendix are explicit functions of the mixing parameters but are denoted by C to avoid obscuring the rates.

## Bernstein input.

Lemma C.7 (Dependence-aware Bernstein inequality). Let $V _ { 1 } , \ldots , V _ { n }$ be centered, coordinate-wise measurable variables with $| V _ { i } | \le B$ . Under Assumption 3.4, for every $x \ge 1$

$$
\mathbb { P } \bigg ( \bigg | \sum _ { i = 1 } ^ { n } V _ { i } \bigg | > C _ { \mathrm { d e p } } B \{ \sqrt { n x } + q _ { N } x \} \bigg ) \le 2 e ^ { - x } .
$$

For independent orfixed-order dependent sequences, $q _ { N } = O ( 1 )$ . For geometrically beta-mixing bounded sequences in the sense of (8), one may take the conservative $q _ { N } = O \{ \log ^ { 2 } ( e N ) \}$ specialization of the published strong-mixing Bernstein inequality, with constants depending on $( C _ { \beta } , c _ { \beta } )$ .

Proof. The first claim is Assumption 3.4. The independent and fixed-order cases follow from ordinary Bernstein blocking. Geometric beta mixing implies geometric alpha mixing; specializing the bounded-variable inequality of Merlevede et al. (2009) gives a linear term of order\` $x \log ^ { 2 } ( e n )$ We retain that factor through $q _ { N }$ and do not invoke a pairwise-covariance-to-cumulant shortcut.

## Uniform Huber-score concentration.

Proof. For a homogeneous interval $A = [ a , b ]$ , the summands in $U _ { A }$ are centered by (1) and bounded by

$$
B _ { Z } = c \sqrt { w _ { \mathrm { m a x } } } .
$$

Apply Lemma $\mathrm { { C . 7 } }$ to the subsequence indexed by $A ,$ noting that restriction to a contiguous interval preserves the mixing bound. Since $| A | \leq S _ { A } ^ { w } / w _ { \mathrm { m i n } } ,$ Lemma C.7 gives

$$
\begin{array} { c } { \displaystyle { \left| U _ { A } \right| \le C B _ { Z } \left[ \sqrt { \frac { S _ { A } ^ { w } } { w _ { \mathrm { m i n } } } x } + q _ { N } x \right] } } \\ { \displaystyle { \le C _ { \mathrm { d e p } } \left[ \sqrt { S _ { A } ^ { w } x } + q _ { N } x \right] . } } \end{array}
$$

Use $x = x _ { N } ( \delta )$ and take a union bound over at most $L _ { N }$ intervals. Because $2 L _ { N } e ^ { - x _ { N } ( \delta ) } \leq \delta / 3 2$ the simultaneous failure probability is at most $\delta / 8$ . Finally, $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } } \gtrsim q _ { N } ^ { 2 } x _ { N }$ turns the displayed bound into inequality (24). □

Uniform empirical curvature. For any interval A, define the empirical Hessian where it exists,

$$
H _ { A } ( \theta ) = \sum _ { i \in { \cal A } } w _ { i } { \bf 1 } \left\{ | \sqrt { w _ { i } } ( Y _ { i } - \theta ) | < c \right\} .\tag{26}
$$

Lemma C.8 (Uniform inlier mass with adjacent references). Under Assumptions 3.3, 3.4, and 3.5, with probability at least $1 - \delta / 8 ,$ , the following holds uniformly. Let $A \in { \mathcal { A } } _ { N }$ be homogeneous or intersect at most two adjacent true segments. Let $\theta _ { 0 }$ either be a true location represented in A or, when A is homogeneous, the location of a true segment immediately adjacent to the segment represented in A. Suppose $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ . Then

$$
H _ { A } ( \theta ) \geq \frac { p _ { 0 } } { 2 } S _ { A } ^ { w }\tag{27}
$$

for every $| \theta - \theta _ { 0 } | \leq r _ { 0 } / 2$ , provided $C _ { \mathrm { c u r v } }$ is sufficiently large.

Proof. For each eligible pair $( A , \theta _ { 0 } )$ , construct a deterministic grid $\mathcal { G }$ on $[ - r _ { 0 } / 2 , r _ { 0 } / 2 ]$ with mesh

$$
\eta = \frac { c } { 8 \sqrt { w _ { \mathrm { m a x } } } } .
$$

For $u \in \mathcal G$ , define

$$
I _ { i } ( u ) = \mathbf { 1 } \left\{ | \sqrt { w _ { i } } \{ Y _ { i } - ( \theta _ { 0 } + u ) \} | \leq \frac { 3 c } { 4 } \right\} .
$$

For every eligible pair and every $i \in A .$ , the locations $\theta _ { 0 }$ and $\theta _ { z ( i ) }$ are identical or belong to adjacent true segments. Hence

$$
| \theta _ { 0 } + u - \theta _ { z ( i ) } | \leq r _ { 0 } / 2 + \kappa _ { \mathrm { m a x } } \leq 3 r _ { 0 } / 4 < r _ { 0 } ,
$$

where $\kappa _ { \operatorname* { m a x } } = 0$ in the no-change case. Assumption 3.5 therefore gives $\mathbb { E } I _ { i } ( u ) \geq p _ { 0 } ;$ the larger $3 c / 4$ window contains the $c / 2$ event used in that assumption. The centered variables

$$
V _ { i } ( u ) = w _ { i } \{ I _ { i } ( u ) - \mathbb { E } I _ { i } ( u ) \}
$$

are bounded by $w _ { \mathrm { m a x } } .$ . Apply Lemma C.7 at $x = x _ { N } ( \delta ) + \log ( 1 6 | \mathcal { G } | )$ and take a union bound over the at most three eligible reference locations per interval and the fixed grid. The additive constant in x is absorbed into ${ \bar { C } } ,$ giving simultaneously

$$
\left| \sum _ { i \in A } V _ { i } ( u ) \right| \leq C \left\{ \sqrt { S _ { A } ^ { w } x _ { N } } + q _ { N } x _ { N } \right\} .
$$

The condition $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ and a sufficiently large $C _ { \mathrm { c u r v } }$ make the right side at most $( p _ { 0 } / 2 ) S _ { A } ^ { w }$ Hence

$$
\sum _ { i \in A } w _ { i } I _ { i } ( u ) \geq \frac { p _ { 0 } } { 2 } S _ { A } ^ { w } .
$$

For arbitrary $| \boldsymbol { v } | \leq r _ { 0 } / 2 .$ , choose $u \in \mathcal G$ with $| u - v | \leq \eta . \operatorname { I f } I _ { i } ( u ) = 1$ , then

$$
| \sqrt { w _ { i } } \{ Y _ { i } - ( \theta _ { 0 } + v ) \} | \le \frac { 3 c } { 4 } + \sqrt { w _ { i } } | u - v | < c .
$$

Thus the grid inlier set is contained in the Hessian inlier set at v, which proves (27) for every eligible interval–reference pair. □

Proof of Lemma C.6. Fix a homogeneous A and write $d = \theta - \theta _ { A }$ . The fundamental theorem for convex functions gives

$$
\widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \theta _ { A } ) = - U _ { A } d + \int _ { 0 } ^ { d } ( d - u ) H _ { A } ( \theta _ { A } + u ) \mathrm { d } u
$$

when $d \geq 0$ , with the analogous oriented integral for $d < 0$ . Lemma C.8 lower bounds $H _ { A }$ by $( p _ { 0 } / 2 ) S _ { A } ^ { w }$ along the path. Hence the integral is at least

$$
\frac { p _ { 0 } } { 4 } S _ { A } ^ { w } d ^ { 2 } .
$$

The claimed coefficient $p _ { 0 } / 8$ is weaker and therefore holds uniformly.

## C.2.1 UNIFORM HUBER FITS AND EMPIRICAL MERGE COSTS

Lemma C.9 (Homogeneous interval fit). On the intersection of the score and curvature events, every homogeneous $A \in \mathcal { A } _ { N }$ with $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ satisfies

$$
| \widehat { \theta } _ { A } - \theta _ { A } | \leq C _ { \theta } \sqrt { \frac { \Lambda _ { N } ( \delta ) } { S _ { A } ^ { w } } } \leq r _ { 0 } / 4\tag{28}
$$

and

$$
0 \leq \widehat { \mathcal { L } } _ { A } ( \theta _ { A } ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \theta } _ { A } ) \leq C _ { L } \Lambda _ { N } ( \delta ) .\tag{29}
$$

The final $r _ { 0 } / 4$ bound is enforced by the constant in $m _ { \mathrm { c u r v } }$

Lemma 3.9 is stated in Section 3; the following arguments prove its three cases.

## Homogeneous interval fit.

Proof. Work on the score and curvature events. Let $S = S _ { A } ^ { w } , U = U _ { A } , \theta _ { 0 } = \theta _ { A }$ , and $d = \theta - \theta _ { 0 }$ By (25),

$$
\widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) \geq - | U | | d | + \alpha S d ^ { 2 } , \qquad \alpha = p _ { 0 } / 8 .\tag{30}
$$

Lemma C.5 and $S \geq m _ { \mathrm { c u r v } } = C _ { \mathrm { c u r v } } \Lambda _ { N }$ give

$$
\frac { | U | } { S } \le C \sqrt { \frac { \Lambda _ { N } } { S } } \le \alpha r _ { 0 } / 4
$$

after increasing $C _ { \mathrm { c u r v } }$ . Evaluating (30) at $| d | = r _ { 0 } / 2$ then gives a strictly positive loss difference. Convexity rules out a minimizer outside these two boundary points, so the full path to every minimizer lies in the curvature region and

$$
| \widehat { \theta } _ { A } - \theta _ { 0 } | \leq \frac { | U | } { \alpha S } \leq C \sqrt { \frac { \Lambda _ { N } } { S } } \leq r _ { 0 } / 4 .
$$

Because dist $( \theta _ { 0 } , \partial \Theta ) \ge r _ { 0 }$ , the fitted location is interior and has zero subgradient. This proves (28). For the loss improvement, minimization and (30) imply

$$
\begin{array} { r l } & { 0 \leq \widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \theta } _ { A } ) } \\ & { \quad \leq \underset { d \in \mathbb { R } } { \operatorname* { s u p } } \{ | U | | d | - \alpha S d ^ { 2 } \} = \frac { U ^ { 2 } } { 4 \alpha S } \leq C _ { L } \Lambda _ { N } , } \end{array}
$$

which is (29).

## Empirical strong convexity around fitted locations.

Lemma C.10 (Local empirical strong convexity). On the event of Lemma C.8, let $A \in \mathcal A _ { N }$ be homogeneous with $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ , and suppose $\theta _ { A }$ and θ both lie within $r _ { 0 } / 2$ of the true location. Then

$$
\widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \theta } _ { A } ) \geq \frac { p _ { 0 } } { 4 } S _ { A } ^ { w } ( \theta - \widehat { \theta } _ { A } ) ^ { 2 } .
$$

Furthermore, for all $\theta , \theta ^ { \prime } { } _ { ; }$

$$
\widehat { \mathcal { L } } _ { A } ( \theta ) \leq \widehat { \mathcal { L } } _ { A } ( \theta ^ { \prime } ) + g _ { A } ( \theta ^ { \prime } ) ( \theta - \theta ^ { \prime } ) + \frac { 1 } { 2 } S _ { A } ^ { w } ( \theta - \theta ^ { \prime } ) ^ { 2 } ,\tag{31}
$$

where $g _ { A } ( \theta ^ { \prime } )$ is any derivative at a differentiability point or subgradient otherwise.

Proof. Integrating the Hessian lower bound between $\widehat { \theta } _ { A }$ and θ gives strong convexity with coefficient $( p _ { 0 } / 2 ) S _ { A } ^ { w }$ in the conventional definition, hence the factor $p _ { 0 } / 4$ in the function inequality. For smoothness, the Huber score is one-Lipschitz, so each map $\theta \stackrel { \cdot } { \mapsto } \rho _ { c } ( \sqrt { w _ { i } } ( Y _ { i } - \theta ) )$ ) has derivative Lipschitz constant at most $w _ { i }$ . Summing gives Lipschitz gradient constant $S _ { A } ^ { w }$ , which yields (31).

## Empirical merge-cost bounds.

Proof. We work on the uniform score, estimator, and curvature events.

Same-location upper bound. Suppose A and B share true location $\theta _ { 0 }$ . Since the merged optimum is no worse than evaluating the common parameter $\theta _ { 0 }$

$$
\begin{array} { r l } & { \widehat { D } ( A , B ) = \widehat { L } _ { A \cup B } ^ { \star } - \widehat { L } _ { A } ^ { \star } - \widehat { L } _ { B } ^ { \star } } \\ & { \qquad \leq \widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) + \widehat { \mathcal { L } } _ { B } ( \theta _ { 0 } ) - \widehat { L } _ { A } ^ { \star } - \widehat { L } _ { B } ^ { \star } } \\ & { \qquad = \{ \widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \theta } _ { A } ) \} + \{ \widehat { \mathcal { L } } _ { B } ( \theta _ { 0 } ) - \widehat { \mathcal { L } } _ { B } ( \widehat { \theta } _ { B } ) \} } \\ & { \qquad \leq 2 C _ { L } \Lambda _ { N } . } \end{array}
$$

Nonnegativity follows from nesting. This proves (12).

Different-location lower bound. Let true locations be a and $b ,$ jump $\kappa = | a - b |$ , masses $x = S _ { A } ^ { w }$ and $y = S _ { B } ^ { w }$ , and estimates ${ \widehat { a } } , { \widehat { b } } .$ Lemma C.9 gives $| { \widehat { a } } - a | , | { \widehat { b } } - b | \leq r _ { 0 } / 4 ;$ ; combined with $\kappa \leq r _ { 0 } / 4$ every point between the two estimates lies within $r _ { 0 } / 2$ of each relevant true location. Lemma C.10 therefore applies and gives

$$
\begin{array} { l } { \widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \widehat { a } ) \geq \displaystyle \frac { p _ { 0 } x } { 4 } ( \theta - \widehat { a } ) ^ { 2 } , } \\ { \widehat { \mathcal { L } } _ { B } ( \theta ) - \widehat { \mathcal { L } } _ { B } ( \widehat { b } ) \geq \displaystyle \frac { p _ { 0 } y } { 4 } ( \theta - \widehat { b } ) ^ { 2 } . } \end{array}
$$

The minimizer of the merged convex loss lies between $\widehat { a }$ and ${ \widehat { b } } .$ Therefore

$$
{ \widehat { D } } ( A , B ) \geq { \frac { p _ { 0 } } { 4 } } { \frac { x y } { x + y } } ( { \widehat { a } } - { \widehat { b } } ) ^ { 2 } .
$$

Estimator localization gives

$$
| \widehat { a } - \widehat { b } | \geq \kappa - C _ { \theta } \sqrt { \Lambda _ { N } / x } - C _ { \theta } \sqrt { \Lambda _ { N } / y } .
$$

Apply (23) and multiply by $x y / ( x + y )$ . Since

$$
{ \frac { x y } { x + y } } { \frac { 1 } { x } } \leq 1 , \qquad { \frac { x y } { x + y } } { \frac { 1 } { y } } \leq 1 ,
$$

we obtain

$$
\widehat { D } ( A , B ) \geq \frac { p _ { 0 } } { 8 } \kappa ^ { 2 } \frac { x y } { x + y } - C \Lambda _ { N } ,
$$

which is (13).

Different-location upper bound. The interior margin and Lemma C.9 ensure that both separate minimizers are interior, so their subgradients vanish. Global smoothness therefore gives

$$
\begin{array} { l } { \displaystyle \widehat { \mathcal { L } } _ { A } ( \theta ) - \widehat { \mathcal { L } } _ { A } ( \widehat { \boldsymbol { a } } ) \leq \frac { x } { 2 } ( \theta - \widehat { \boldsymbol { a } } ) ^ { 2 } , } \\ { \displaystyle \widehat { \mathcal { L } } _ { B } ( \theta ) - \widehat { \mathcal { L } } _ { B } ( \widehat { \boldsymbol { b } } ) \leq \frac { y } { 2 } ( \theta - \widehat { \boldsymbol { b } } ) ^ { 2 } . } \end{array}
$$

Evaluate the merged objective at $\theta = ( x \widehat { a } + y \widehat { b } ) / ( x + y )$ to get

$$
\widehat { D } ( A , B ) \leq \frac { 1 } { 2 } \frac { x y } { x + y } ( \widehat { a } - \widehat { b } ) ^ { 2 } .
$$

Using $( u + v + w ) ^ { 2 } \leq 3 ( u ^ { 2 } + v ^ { 2 } + w ^ { 2 } )$ and the estimator bounds gives

$$
\widehat { D } ( A , B ) \leq C \kappa ^ { 2 } \frac { x y } { x + y } + C \Lambda _ { N } ,
$$

which proves (14).

## C.2.2 WEIGHTED INTERVAL ISOLATION

For the proof of Lemma 3.10, for each change $\tau _ { j }$ , define weighted endpoint neighborhoods

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { j } = \left\{ s \leq \tau _ { j } : \frac { \Delta _ { w } } { 6 } \leq S _ { s : \tau _ { j } } ^ { w } \leq \frac { \Delta _ { w } } { 3 } \right\} , } \\ { \displaystyle \mathcal { R } _ { j } = \left\{ e > \tau _ { j } : \frac { \Delta _ { w } } { 6 } \leq S _ { \tau _ { j } + 1 : e } ^ { w } \leq \frac { \Delta _ { w } } { 3 } \right\} . } \end{array}
$$

ProofofLemma 3.10. Fix change $\tau _ { j }$ . The segment to its left has mass at least $\Delta _ { w } .$ . Move left from $\tau _ { j }$ until the accumulated mass first reaches $\Delta _ { w } / 6$ . Because one atom is at most $\Delta _ { w } / 2 4$ , the attained mass is at most $5 \Delta _ { w } / 2 4 < \Delta _ { w } / 3$ . Continuing until just before mass exceeds $\Delta _ { w } / 3$ shows that the set of weighted coordinates corresponding to valid left endpoints has length at least

$$
\frac { \Delta _ { w } } { 3 } - \frac { \Delta _ { w } } { 6 } - 2 \operatorname* { m a x } _ { i } w _ { i } \ge \frac { \Delta _ { w } } { 1 2 } .
$$

The same argument holds on the right.

Under weighted-coordinate sampling, the probability that one ordered pair of endpoints falls in the left and right neighborhoods is at least

$$
p _ { j } \ge 2 \left( \frac { \Delta _ { w } / 1 2 } { W _ { N } } \right) ^ { 2 } = \frac { \Delta _ { w } ^ { 2 } } { 7 2 W _ { N } ^ { 2 } } ,
$$

where the factor two accounts for the two orders before sorting. Hence the probability that none of M intervals isolates $\tau _ { j }$ is at most

$$
( 1 - p _ { j } ) ^ { M } \leq \exp ( - M p _ { j } ) .
$$

With (15), this is at most $\delta / [ 4 ( K \vee 1 ) ]$ . A union bound over K changes gives failure probability at most $\delta / 4$ and proves simultaneous isolation; for $K = 0$ the statement is vacuous. The endpoint construction keeps s and e inside the adjacent segments, so the interval contains no other change. The mass bounds follow directly from the definitions of $\mathcal { L } _ { j }$ and $\mathcal { R } _ { j }$ □

## C.2.3 BOUNDARY FRAGMENTS AND EXACT RECOVERY

The constants can be selected in dependency order. First choose $C _ { \mathrm { c u r v } }$ large enough for the score and inlier-mass bounds under the fixed $c , p _ { 0 } , r _ { 0 } , w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } , C _ { \mathrm { d e p } }$ . Fix a jump range and $c _ { m , 1 } \geq$ $C _ { \mathrm { c u r v } } \kappa _ { \mathrm { m a x } } ^ { 2 }$ . Next choose $c _ { \mathrm { c e r t } } \ >$ max $\{ c _ { m , 1 } , 2 ( C _ { 0 } + C _ { 1 } ) / c _ { 0 } \}$ , then $C _ { r } > \operatorname* { m a x } \{ 2 C _ { 0 } , C _ { \mathrm { f r a g } } ( 1 +$ $c _ { \mathrm { c e r t } } ) \}$ . Choose $C _ { \mathrm { l o c } } > \mathrm { m a x } \{ C _ { \mathrm { c u r v } } \kappa _ { \mathrm { m a x } } ^ { 2 } , 2 ( C _ { 0 } + C _ { 1 } ) / c _ { 0 } \}$ , and $C _ { g } > C _ { \mathrm { l o c } }$ . Finally enlarge $C _ { \mathrm { s n r } }$ to make isolating gains exceed $C _ { r } \Lambda _ { N }$ , localization at most $\bar { \Delta } _ { w } / 1 2$ , and the guard/side-mass intervals nonempty, including atom overshoot. These constants depend on the stated fixed model parameters, not on the document-specific score realization.

We now prove Theorem 3.11. The proof uses an additional deterministic consequence of bounded Huber influence: a true segment fragment with small reliability mass cannot by itself create an arbitrarily large profile gain, even if the raw scores in that fragment are extreme.

## A small-fragment gain bound.

Lemma C.11 (Score transfer between adjacent locations). Let A lie in a segment with location $\theta _ { 1 }$ and let $\theta _ { 2 }$ be an adjacent location with $\kappa = | \theta _ { 1 } - \theta _ { 2 } |$ . Define

$$
U _ { A } ( \theta ) = \sum _ { i \in A } \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta ) \} .
$$

Then

$$
| U _ { A } ( \boldsymbol { \theta } _ { 2 } ) | \leq | U _ { A } ( \boldsymbol { \theta } _ { 1 } ) | + \kappa S _ { A } ^ { w } .
$$

Proof. The Huber score is one-Lipschitz. Hence, term by term,

$$
\begin{array} { r l } & { | \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta _ { 2 } ) \} - \sqrt { w _ { i } } \psi _ { c } \{ \sqrt { w _ { i } } ( Y _ { i } - \theta _ { 1 } ) \} | } \\ & { \qquad \leq w _ { i } | \theta _ { 2 } - \theta _ { 1 } | = w _ { i } \kappa . } \end{array}
$$

Summing and applying the triangle inequality proves the claim.

Lemma C.12 (Fit improvement relative to a reference location). Let A intersect at most two adjacent true segments. Let $\theta _ { 0 }$ either be represented in A or, when A is homogeneous, be the location of a true segment immediately adjacent to the segment represented in A. Define $\begin{array} { r } { r _ { A } = \sum _ { i \in A : \theta _ { z ( i ) } \neq \theta _ { 0 } } w _ { i } } \end{array}$ and $\kappa = \operatorname* { m a x } _ { i \in A } | \theta _ { z ( i ) } - \theta _ { 0 } |$ . Suppose $S _ { A } ^ { w } \geq m _ { \mathrm { c u r v } }$ . On the uniform score and empirical-curvature event,

$$
0 \leq \widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) - \operatorname* { i n f } _ { \theta \in \Theta } \widehat { \mathcal { L } } _ { A } ( \theta ) \leq C \left\{ \Lambda _ { N } + \kappa ^ { 2 } \frac { r _ { A } ^ { 2 } } { S _ { A } ^ { w } } \right\} .
$$

Proof. Decompose $A = A _ { 0 } \cup A _ { 1 }$ , where $A _ { 0 }$ contains the observations whose location is $\theta _ { 0 }$ and $A _ { 1 } = A \setminus A _ { 0 }$ . By eligibility, all observations in a nonempty $A _ { 1 }$ share one location adjacent to $\theta _ { 0 }$ and $S _ { A _ { 1 } } ^ { w } = r _ { A } ;$ either set may be empty. The score at $\theta _ { 0 }$ is

$$
U _ { A } ( \theta _ { 0 } ) = U _ { A _ { 0 } } ( \theta _ { 0 } ) + U _ { A _ { 1 } } ( \theta _ { 0 } ) .
$$

Uniform score concentration on the nonempty homogeneous fragments and Lemma C.11 give

$$
\begin{array} { r } { | U _ { A } ( \theta _ { 0 } ) | \le C \sqrt { S _ { A } ^ { w } \Lambda _ { N } } + \kappa r _ { A } . } \end{array}\tag{32}
$$

By the represented-or-adjacent reference clause of Lemma C.8, the empirical Hessian on A is at least $\alpha S _ { A } ^ { w }$ for every $| \theta - \theta _ { 0 } | \overset { \cdot } { \le } r _ { 0 } / 2$ , with fixed $\alpha = p _ { 0 } / 2$ . Because $S _ { A } ^ { w } \geq C _ { \mathrm { c u r v } } \Lambda _ { N }$ and $\kappa \leq p _ { 0 } r _ { 0 } / 1 6$ $( 3 2 )$ is at most $\alpha S _ { A } ^ { w } r _ { 0 } / 4$ after choosing the constants. Evaluating the quadratic lower bound at $\theta _ { 0 } \pm r _ { 0 } / 2$ gives a positive loss difference; convexity therefore places every mixed-interval minimizer inside that interval. Only after this containment step do we integrate the empirical curvature along the path, obtaining

$$
\widehat { \mathcal { L } } _ { A } ( \theta _ { 0 } ) - \operatorname* { i n f } _ { \theta } \widehat { \mathcal { L } } _ { A } ( \theta ) \leq C \frac { U _ { A } ^ { 2 } ( \theta _ { 0 } ) } { S _ { A } ^ { w } } .
$$

Finally, (32) and $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ give

$$
\frac { U _ { A } ^ { 2 } ( \theta _ { 0 } ) } { S _ { A } ^ { w } } \leq C \left\{ \Lambda _ { N } + \kappa ^ { 2 } \frac { r _ { A } ^ { 2 } } { S _ { A } ^ { w } } \right\} .
$$

Lemma C.13 (Small boundary fragment). Let $I = [ s , e ]$ contain exactly one change with jump κ, and let

$$
r ( I ) = \operatorname* { m i n } \{ S _ { s : \tau } ^ { w } , S _ { \tau + 1 : e } ^ { w } \} .
$$

Ifevery admissible child has mass at least $m _ { N } \geq m _ { \mathrm { { c u r v } } } ;$ , then on the uniform event,

$$
\operatorname* { m a x } _ { b \in \mathcal { B } ( I ) } \widehat { \Gamma } _ { s , e } ( b ) \leq C _ { \mathrm { f r a g } } \{ \Lambda _ { N } + \kappa ^ { 2 } r ( I ) \} .\tag{33}
$$

Proof. Assume without loss of generality that the left true segment is the smaller one, with location θ and mass $r = r ( I )$ , while the right segment has location $\theta _ { + }$ . For an admissible candidate $b ,$ let $L = [ s , b ]$ and $R = \dot { [ { b + 1 } , { e } ] }$ . Since the one-location parent fit is no larger than evaluation at $\theta _ { + }$

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \widehat \Gamma _ { s , e } ( b ) = \widehat L _ { L \cup R } ^ { \star } - \widehat L _ { L } ^ { \star } - \widehat L _ { R } ^ { \star } } } \\ & { } & { \leq \{ \widehat { \mathcal L } _ { L } ( \theta _ { + } ) - \widehat L _ { L } ^ { \star } \} + \{ \widehat { \mathcal L } _ { R } ( \theta _ { + } ) - \widehat L _ { R } ^ { \star } \} . } \end{array}
$$

Let $r _ { L } , r _ { R }$ be the left-segment masses contained in $L , R ,$ so $r _ { L } + r _ { R } = r$ . Lemma C.12 gives

$$
\begin{array} { c l l } { \displaystyle \widehat { \Gamma } _ { s , e } ( b ) \leq C \left. \Lambda _ { N } + \kappa ^ { 2 } \frac { r _ { L } ^ { 2 } } { S _ { L } ^ { w } } + \Lambda _ { N } + \kappa ^ { 2 } \frac { r _ { R } ^ { 2 } } { S _ { R } ^ { w } } \right. } \\ { \leq C \{ \Lambda _ { N } + \kappa ^ { 2 } ( r _ { L } + r _ { R } ) \} , } \end{array}
$$

where $r _ { L } ^ { 2 } / S _ { L } ^ { w } \leq r _ { L }$ and similarly for R. This is (33).

## No false positives and isolated signal.

Lemma C.14 (No false positive). $\mathit { I f I } = [ s , e ]$ contains no true change, then on the event of Lemma 3.9,

$$
\begin{array} { r } { T ( I ) \le 2 C _ { 0 } \Lambda _ { N } . } \end{array}
$$

Thus $T ( I ) \le r _ { N }$ whenever $C _ { r } > 2 C _ { 0 }$

Proof. For every admissible $b ,$ the two child intervals belong to the same segment. By (3),

$$
\widehat { \Gamma } _ { s , e } ( b ) = 2 \widehat { D } ( [ s , b ] , [ b + 1 , e ] ) .
$$

Apply (12) and maximize over b.

Lemma C.15 (Signal on an isolating interval). Let $I = [ s , e ]$ isolate change $\tau _ { j }$ and satisfy (16). If m $\ d _ { l N } \leq \Delta _ { w } / 1 2$ , then $\tau _ { j }$ is admissible and

$$
\widehat { \Gamma } _ { s , e } ( \tau _ { j } ) \geq c _ { \mathrm { s i g } } \kappa _ { j } ^ { 2 } \Delta _ { w } - C _ { \mathrm { s i g } } \Lambda _ { N } .
$$

Consequently $T ( I ) > r _ { N }$ under the signal condition with a sufficiently large $C _ { \mathrm { s n r } }$

Proof. The two true-side masses are at least $\Delta _ { w } / 6 > m _ { N }$ , so the split is admissible. Moreover,

$$
\begin{array} { r } { \widehat { \Gamma } _ { s , e } ( \tau _ { j } ) = 2 \widehat { D } ( [ s , \tau _ { j } ] , [ \tau _ { j } + 1 , e ] ) . } \end{array}
$$

Lemma 3.9 and Lemma C.2 yield

$$
\begin{array} { c l l } { { \widehat \Gamma _ { s , e } ( \tau _ { j } ) \geq 2 c _ { 0 } \kappa _ { j } ^ { 2 } \displaystyle \frac { S _ { s : \tau _ { j } } ^ { w } S _ { \tau _ { j } + 1 : e } ^ { w } } { S _ { s : e } ^ { w } } - 2 C _ { 1 } \Lambda _ { N } } } \\ { { \geq c _ { \mathrm { s i g } } \kappa _ { j } ^ { 2 } \Delta _ { w } - C _ { \mathrm { s i g } } \Lambda _ { N } . } } \end{array}
$$

The signal condition makes this larger than $C _ { r } \Lambda _ { N }$

## The selected interval contains one well-supported change.

Lemma C.16 (Narrowest interval contains one change). On the no-false-positive and isolation events, every recursive search range containing an unresolved change has a nonempty over-threshold set. The selected narrowest interval contains exactly one unresolved true change.

Proof. Let $\tau _ { j }$ be unresolved in the current search range. The induction argument in Lemma C.21 below shows that one of its isolating intervals remains fully inside the range. Lemma C.15 places that interval in the over-threshold set, so the set is nonempty. Its weighted length is at most $2 \Delta _ { w } / 3$ Hence the selected interval has weighted length at most $2 \bar { \Delta _ { w } } / 3$

An interval containing no change cannot be over threshold by Lemma C.14. An interval containing two or more consecutive changes must contain the complete segment between two adjacent changes, whose mass is at least $\Delta _ { w }$ . Such an interval has weighted length at least $\Delta _ { w } > 2 \Delta _ { w } ^ { - } / 3$ . Therefore the selected interval contains exactly one change. □

Lemma C.17 (True-side mass certification). Choose $C _ { r } > C _ { \mathrm { f r a g } } ( 1 + c _ { \mathrm { c e r t } } )$ for a constant $c _ { \mathrm { c e r t } } >$ max $\{ c _ { m , 1 } , 2 ( \dot { C } _ { 0 } + C _ { 1 } ) / c _ { 0 } \}$ . Every selected one-change interval I with jump $\kappa _ { j }$ satisfies

$$
r ( I ) \geq c _ { \mathrm { c e r t } } \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } \geq m _ { N } .
$$

In particular, the true split is admissible.

Proof. Selection implies $T ( I ) > C _ { r } \Lambda _ { N }$ . Lemma C.13 gives

$$
C _ { r } \Lambda _ { N } < T ( I ) \le C _ { \mathrm { f r a g } } \{ \Lambda _ { N } + \kappa _ { j } ^ { 2 } r ( I ) \} .
$$

Rearranging yields

$$
r ( I ) > \left( \frac { C _ { r } } { C _ { \mathrm { f r a g } } } - 1 \right) \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } \geq c _ { \mathrm { c e r t } } \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } .
$$

Because $\kappa _ { j } ~ \leq ~ \kappa _ { \mathrm { m a x } }$ and $m _ { N } \le c _ { m , 1 } \Lambda _ { N } / \kappa _ { \mathrm { m a x } } ^ { 2 }$ , the final quantity is at least $m _ { N }$ when $c _ { \mathrm { c e r t } } \ >$ $c _ { m , 1 }$ □

## Localization on the selected interval.

Lemma C.18 (Single-interval localization). Let $C _ { \mathrm { l o c } , 0 }$ exceed max $\{ C _ { \mathrm { c u r v } } \kappa _ { \mathrm { m a x } } ^ { 2 } , 2 ( C _ { 0 } + C _ { 1 } ) / c _ { 0 } \}$ and choose $C _ { \mathrm { l o c } } > C _ { \mathrm { l o c } , 0 }$ . Let $I = [ s , e ]$ be a selected interval containing exactly one change $\tau _ { j }$ , and suppose its true split is admissible. Then every maximizer $\widehat { \tau } _ { j } o f \widehat { \Gamma } _ { s , e }$ satisfies

$$
d _ { w } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq C _ { \mathrm { l o c } } \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } .
$$

Proof. Consider a candidate $b < \tau _ { j }$ and set $A = [ s , b ] , B = [ b + 1 , \tau _ { j } ]$ , and $C = [ \tau _ { j } + 1 , e ]$ . By Lemma C.1,

$$
\widehat { \Gamma } _ { s , e } ( \tau _ { j } ) - \widehat { \Gamma } _ { s , e } ( b ) = 2 \{ \widehat { D } ( B , C ) - \widehat { D } ( A , B ) \} .
$$

The intervals $A , B$ share one location, while $B , C$ have jump $\kappa _ { j }$ . The candidate admissibility gives $S _ { A } ^ { w } \ge m _ { N } \ge m _ { \mathrm { c u r v } } ,$ and Lemma C.17 gives $S _ { C } ^ { w } \geq m _ { N } .$ . If

$$
S _ { B } ^ { w } = d _ { w } ( b , \tau _ { j } ) \ge C _ { \mathrm { l o c } } \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } ,
$$

then, because $\kappa _ { j } \leq \kappa _ { \mathrm { m a x } }$ and $C _ { \mathrm { l o c } }$ is large, $S _ { B } ^ { w } \geq m _ { \mathrm { c u r v } }$ . Lemma 3.9 and (22) give

$$
\widehat { D } ( B , C ) - \widehat { D } ( A , B ) \geq c _ { 0 } \kappa _ { j } ^ { 2 } \frac { S _ { B } ^ { w } S _ { C } ^ { w } } { S _ { B } ^ { w } + S _ { C } ^ { w } } - C _ { 1 } \Lambda _ { N } - C _ { 0 } \Lambda _ { N } .
$$

If $S _ { B } ^ { w } \ \leq \ S _ { C } ^ { w }$ , the harmonic term is at least $S _ { B } ^ { w } / 2$ and the right side is positive for sufficiently large $C _ { \mathrm { l o c } } . \ \breve { \mathrm { I f } } \ S _ { B } ^ { w } > S _ { C } ^ { w }$ , Lemma C.17 gives $S _ { C } ^ { w } \geq c _ { \mathrm { c e r t } } \Lambda _ { N } / \bar { \kappa } _ { i } ^ { 2 }$ , so the harmonic term is at least $c _ { \mathrm { c e r t } } \Lambda _ { N } / ( 2 \kappa _ { j } ^ { 2 } )$ . Choosing $c _ { \mathrm { c e r t } }$ larger than a constant multiple of $( C _ { 0 } + C _ { 1 } ) / c _ { 0 }$ makes the right side positive. Therefore every candidate farther than the displayed radius has strictly smaller gain than the true split. The case $b > \tau _ { j }$ is symmetric. □

Remark C.19. The last case in the proof is where the minimum-side certification matters. A completely jump-adaptive constant can be obtained by replacing the fixed admissibility mass by a short geometric grid of masses and maximizing over the grid with a corresponding union-bound penalty. We retain one $m _ { N }$ to keep the algorithm and notation readable; the stated theorem constants depend on the fixed jump range $\left[ \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } } \right]$

## Guard containment and preservation.

Lemma C.20 (Guard containment). ${ \cal I } f g _ { N } \ge C _ { g } \Lambda _ { N } / \kappa _ { \mathrm { m i n } } ^ { 2 } + w _ { \mathrm { m a x } } w i t h C _ { g } > C _ { \mathrm { l o c } } ,$ , then the guard around every accepted estimate contains the corresponding true change.

Proof. By Lemma C.18,

$$
d _ { w } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq C _ { \mathrm { l o c } } \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } } \leq C _ { \mathrm { l o c } } \frac { \Lambda _ { N } } { \kappa _ { \mathrm { m i n } } ^ { 2 } } .
$$

The discrete guard constructed in (5) covers at least $g _ { N } - w _ { \mathrm { m a x } }$ mass on each available side of $\widehat { \tau } _ { j } .$ The choice of $C _ { g }$ therefore places $\tau _ { j }$ inside the removed guard.

Lemma C.21 (Preservation of neighboring isolation intervals). Suppose $g _ { N } \leq c _ { m , 2 } \Delta _ { w }$ with $c _ { m , 2 } \leq$ $1 / 1 2$ and the localization radius is at most $\Delta _ { w } / 1 2$ . Removing the guard around a detected change does not remove any other true change or the weighted isolation neighborhoods of an adjacent unresolved change.

Proof. Adjacent true changes are separated by a complete segment of mass at least $\Delta _ { w }$ . The accepted estimate differs from its true change by at most $\bar { \Delta _ { w } } / 1 2$ . The removed guard extends by at most $g _ { N } + w _ { \operatorname* { m a x } } \le \Delta _ { w } / 1 2 + \Delta _ { w } / 2 4$ beyond the estimate on either side. Thus its total reach from the detected true change is below $\Delta _ { w } / \dot { 4 }$ . The nearest endpoint neighborhood used to isolate the next change lies within $\Delta _ { w } / 3$ of that next change and therefore at least $2 \Delta _ { w } / 3$ from the detected change before discretization. The guard and that neighborhood are disjoint. The same reasoning applies on the left. □

## Proof of Theorem 3.11.

Proof. Let E be the intersection of the uniform score event, empirical-curvature event, empirical merge-cost event, and weighted isolation event. The preceding lemmas and their union bounds give

$$
\mathbb { P } ( \mathcal { E } ) \geq 1 - \delta
$$

after distributing the failure budget and adjusting numerical constants.

We argue by induction over recursive calls on E. Initially, every true change is unresolved and has an isolating interval. If a search range contains no unresolved change, every retained interval is homogeneous after previously detected guards have been removed, so Lemma C.14 makes the over-threshold set empty and recursion stops.

If the range contains at least one unresolved change, Lemma C.21 ensures that an isolating interval for such a change remains inside the range. Lemma C.15 makes the over-threshold set nonempty. Lemma C.16 shows that the selected interval contains exactly one unresolved change, and Lemma C.17 makes its true split admissible. Lemma C.18 yields (17) for the accepted estimate.

Lemma C.20 removes the corresponding true change from both child search ranges, so it cannot be detected twice. Lemma C.21 shows that no other change or isolating interval is removed. Therefore the two child calls partition the remaining unresolved changes without loss. Each accepted split reduces their number by one. After exactly K acceptances, all child ranges contain no unresolved change and stop by the no-false-positive lemma. Hence ${ \widehat { K } } = K$ , and every accepted boundary satisfies the stated localization bound. □

## C.3 EXTENSIONS, LOWER BOUNDS, AND MODEL VERIFICATION

This appendix develops results that complement the main recovery theorem: stability to estimated weights, an explicitly delimited information lower bound, reduction identities, and concrete conditions under which the curvature assumptions hold for common score distributions and contamination models. Let $w _ { i } ^ { \circ }$ be the oracle capped weights in (2) and define

$$
\mathcal { E } _ { w } = \{ c _ { w } w _ { i } ^ { \circ } \leq \widehat { w } _ { i } \leq C _ { w } w _ { i } ^ { \circ } \mathrm { ~ f o r ~ a l l ~ } i \in [ N ] \} ,
$$

where $0 < c _ { w } \le C _ { w } < \infty$

Corollary 3.12 is stated in Section $_ { 3 ; }$ we now prove it.

Proof of Corollary 3.12. Condition on the sigma-field that generates $\widehat { w } _ { 1 : N }$ and on $\mathcal { E } _ { w }$ . For every interval A,

$$
c _ { w } S _ { A } ^ { w } { } ^ { \circ } \leq S _ { A } ^ { \widehat { w } } \leq C _ { w } S _ { A } ^ { w } { } ^ { \circ } .
$$

Similarly, for every pair u, v,

$$
c _ { w } d _ { w ^ { \circ } } ( u , v ) \leq d _ { \widehat { w } } ( u , v ) \leq C _ { w } d _ { w ^ { \circ } } ( u , v ) .\tag{34}
$$

Thus the estimated-weight spacing is at least $c _ { w } \Delta _ { w ^ { \circ } }$ , and the no-dominant-atom ratio changes by at most $C _ { w } / c _ { w }$ . The weighted-coordinate measure of every isolation neighborhood changes by the same constant factors, so the interval count in Lemma 3.10 is multiplied by at most a constant depending on $( c _ { w } , C _ { w } )$

By the conditional assumptions, population Huber risks computed with $\widehat { w } _ { i }$ satisfy the same centering and curvature inequalities with constants bounded away from zero and infinity uniformly on $\mathcal { E } _ { w }$ Theorem 3.11 therefore applies conditionally and gives

$$
d _ { \widehat { w } } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq C \frac { \Lambda _ { N } } { \kappa _ { j } ^ { 2 } }
$$

with conditional probability at least $1 - \delta .$ . The first inequality in (34) implies

$$
d _ { w ^ { \circ } } ( \widehat { \tau } _ { j } , \tau _ { j } ) \leq c _ { w } ^ { - 1 } d _ { \widehat { w } } ( \widehat { \tau } _ { j } , \tau _ { j } ) .
$$

Finally,

$$
\mathbb { P } ( \mathrm { f a i l u r e } ) \le \mathbb { P } ( \mathcal { E } _ { w } ^ { c } ) + \mathbb { P } ( \mathrm { f a i l u r e } \mid \mathcal { E } _ { w } ) \le \delta _ { w } + \delta .
$$

A sample-splitting construction. The comparability event can be verified in simple calibration designs. Suppose an auxiliary corpus supplies R independent score replicates for each length bin $g .$ Let $v _ { g }$ be the bin variance proxy and define a robust scale estimate $\widehat { v } _ { g }$ from the auxiliary data. If

$$
\mathbb { P } \left( \operatorname* { m a x } _ { g \in [ G ] } \left| \frac { \widehat { v } _ { g } } { v _ { g } } - 1 \right| > \eta \right) \le \delta _ { w } , \qquad 0 < \eta < 1 ,
$$

then, before capping,

$$
\frac { 1 } { 1 + \eta } v _ { g } ^ { - 1 } \leq \widehat { v } _ { g } ^ { - 1 } \leq \frac { 1 } { 1 - \eta } v _ { g } ^ { - 1 } .\tag{35}
$$

Projection onto a common interval $[ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ]$ is monotone and nonexpansive, so capped weights satisfy a constant-factor version of (35). Because the auxiliary corpus is independent of the target sequence, conditioning is immediate.

## C.3.1 INFORMATION LOWER BOUND

To calibrate this upper rate, consider the independent one-change Gaussian submodel $Y _ { i } \sim N ( 0 , \sigma _ { i } ^ { 2 } )$ before the boundary and $Y _ { i } ~ \sim ~ N ( \kappa , \sigma _ { i } ^ { 2 } )$ after it, with information distance $d _ { \mathrm { i n f o } } ( u , v ) \ =$ $\textstyle \sum _ { i \in ( u , v ] \cup ( v , u ] } \sigma _ { i } ^ { - 2 }$ . Let $Q _ { \mathrm { i n f o } } ( \delta , { \widehat { \tau } } , P _ { \tau } )$ be the smallest radius containing $\widehat { \tau }$ with probability at least $1 - \delta$

Theorem C.22 (Two-point information lower bound). Fix $\delta \in ( 0 , 1 / 8 )$ and $c _ { H } \in ( 0 , 1 / 2 ]$ . If the admissible class contains $\tau _ { 0 } < \tau _ { 1 }$ with $H = d _ { \mathrm { i n f o } } ( \tau _ { 0 } , \tau _ { 1 } )$ satisfying

$$
c _ { H } \frac { \log ( 1 / \delta ) } { \kappa ^ { 2 } } \leq H \leq \frac { 1 } { 2 } \frac { \log ( 1 / \delta ) } { \kappa ^ { 2 } } ,\tag{36}
$$

then

$$
\operatorname* { i n f } _ { \widehat { \tau } } \operatorname* { s u p } _ { \tau } Q _ { \mathrm { i n f o } } ( \delta , \widehat { \tau } , P _ { \tau } ) \geq \frac { c _ { H } } { 3 } \frac { \log ( 1 / \delta ) } { \kappa ^ { 2 } } .
$$

Corollary C.23 (Independent-regime rate comparison). If caps are inactive, $w _ { i } \asymp \sigma _ { i } ^ { - 2 } , L _ { N }$ is polynomial in $N ,$ and $\delta = N ^ { - a }$ , the RWCP upper bound and Theorem C.22 are both of order log $N / \kappa ^ { 2 }$ , up to constants. This comparison is restricted to the admissible independent Gaussian submodel and is not asserted under dependence or active caps.

For a fixed variance sequence $\sigma _ { 1 : N } ^ { 2 }$ and an admissible set of locations $\mathcal { T } _ { N }$ , let $P _ { \tau }$ denote the independent Gaussian model

$$
Y _ { i } \sim \left\{ { \begin{array} { c c } { N ( 0 , \sigma _ { i } ^ { 2 } ) , } & { i \leq \tau , } \\ { N ( \kappa , \sigma _ { i } ^ { 2 } ) , } & { i > \tau . } \end{array} } \right.
$$

For an estimator ${ \widehat { \tau } } ,$ , define its $( 1 - \delta )$ information-radius by

$$
Q _ { \mathrm { i n f o } } ( \delta , \hat { \tau } , P _ { \tau } ) = \operatorname* { i n f } \left\{ r \geq 0 : P _ { \tau } ( d _ { \mathrm { i n f o } } ( \hat { \tau } , \tau ) \leq r ) \geq 1 - \delta \right\} .
$$

Theorem C.22 and Corollary C.23 are stated in Section $_ { 3 ; }$ the following testing reduction proves them.

## A testing reduction.

Lemma C.24 (Separated localization implies testing). Let $P _ { 0 } , P _ { 1 }$ have change points τ , τ separated by

$$
H = d _ { \mathrm { i n f o } } ( \tau _ { 0 } , \tau _ { 1 } ) .
$$

Ifan estimator satisfies

$$
P _ { j } \left( d _ { \mathrm { i n f o } } ( \widehat { \tau } , \tau _ { j } ) < \frac { H } { 3 } \right) \geq 1 - \delta , \qquad j = 0 , 1 ,
$$

then there exists a test between $P _ { 0 }$ and $P _ { 1 }$ whose sum of type-I and type-II errors is at most $2 \delta .$

Proof. The information distance is a path metric on ordered indices. Therefore the open balls

$$
B _ { j } = \{ t : d _ { \mathrm { i n f o } } ( t , \tau _ { j } ) < H / 3 \}
$$

are disjoint: if t belonged to both, the triangle inequality would give $H < 2 H / 3$ . Define the test $\varphi = 1$ when $\widehat { \tau } \in B _ { 1 }$ and $\varphi = 0$ otherwise. Under $P _ { 0 } ,$ , a type-I error implies $\widehat { \tau } \notin B _ { 0 }$ , so its probability is at most δ. Under P , a type-II error implies $\widehat { \tau } \notin B _ { 1 }$ , also with probability at most δ. □

Lemma C.25 (High-probability two-point bound). For any two distributions $P _ { 0 } , P _ { 1 }$ and any test $\varphi ,$

$$
P _ { 0 } ( \varphi = 1 ) + P _ { 1 } ( \varphi = 0 ) \geq \frac { 1 } { 2 } \exp \{ - D _ { \mathrm { K L } } ( P _ { 0 } \| P _ { 1 } ) \} .
$$

Proof. Let $L = \mathrm { d } P _ { 0 } / \mathrm { d } P _ { 1 }$ on the absolutely continuous part. The sum of testing errors is at least

$$
\int \operatorname* { m i n } ( \mathrm { d } P _ { 0 } , \mathrm { d } P _ { 1 } ) .
$$

The Bretagnolle–Huber inequality states

$$
\int \mathrm { m i n } ( \mathrm { d } P _ { 0 } , \mathrm { d } P _ { 1 } ) \geq \frac { 1 } { 2 } e ^ { - D _ { \mathrm { K L } } ( P _ { 0 } \| P _ { 1 } ) } .
$$

For completeness, write $A = \{ L \geq 1 \}$ . Jensen’s inequality under $P _ { 0 }$ on $A ^ { c }$ and under $P _ { 1 }$ on A bounds the overlap from below by the displayed exponential quantity; equivalently the result follows from the variational representation of KL divergence applied to the binary partition $( A , A ^ { c } )$ □

## Proof of Theorem C.22.

Proof. Let $P _ { 0 } = P _ { \tau _ { 0 } }$ and $P _ { 1 } = P _ { \tau _ { 1 } }$ for the two admissible locations in (36). The two distributions differ only on $( \tau _ { 0 } , \tau _ { 1 } ]$ ], and Gaussian additivity gives

$$
D _ { \mathrm { K L } } ( P _ { 0 } \| P _ { 1 } ) = \frac { \kappa ^ { 2 } } { 2 } \sum _ { i = \tau _ { 0 } + 1 } ^ { \tau _ { 1 } } \sigma _ { i } ^ { - 2 } = \frac { \kappa ^ { 2 } H } { 2 } \le \frac { 1 } { 4 } \log ( 1 / \delta ) .\tag{37}
$$

If one estimator had information-radius strictly below $H / 3$ under both models, Lemma C.24 would yield a test with total error at most 2δ. Lemma C.25 and (37) instead give total error at least ${ \frac { 1 } { 2 } } \delta ^ { 1 / 4 } > 2 \delta$ for $\delta < 1 / 8$ , a contradiction. Hence one of the two models has radius at least $H / 3 ,$ and the lower bound in (36) completes the proof. □

## Index and token interpretations.

Corollary C.26 (Index-distance lower bound). $I f 0 < \underline { { \sigma } } ^ { 2 } \le \sigma _ { i } ^ { 2 } \le \overline { { \sigma } } ^ { 2 } < \infty ,$ , then

$$
| u - v | / \overline { { \sigma } } ^ { 2 } \leq d _ { \mathrm { i n f o } } ( u , v ) \leq | u - v | / \underline { { \sigma } } ^ { 2 } .
$$

Consequently Theorem C.22 implies an index localization lower bound of order $\underline { { \sigma } } ^ { 2 } \log ( 1 / \delta ) / \kappa ^ { 2 }$

Proof. The path between u and v contains $\left| u - v \right|$ atoms, each between $1 / \overline { { \sigma } } ^ { 2 }$ and $1 / \underline { { \sigma } } ^ { 2 }$ . Sum these inequalities and apply Theorem C.22. □

Corollary C.27 (Token-distance interpretation). $I f c _ { 1 } n _ { i } ^ { - 1 } \leq \sigma _ { i } ^ { 2 } \leq c _ { 2 } n _ { i } ^ { - 1 }$ , then

$$
\frac { 1 } { c _ { 2 } } \sum _ { i \in ( u , v ] \cup ( v , u ] } n _ { i } \leq d _ { \mathrm { i n f o } } ( u , v ) \leq \frac { 1 } { c _ { 1 } } \sum _ { i \in ( u , v ] \cup ( v , u ] } n _ { i } .
$$

Thus the minimax information lower bound is equivalent, up to constants, to a lower bound on the number ofmisplaced tokens.

## C.3.2 REDUCTION RESULTS

ProofofProposition 2.2. Write $x = S _ { s : b } ^ { w } , y = S _ { b + 1 : e } ^ { w }$ , and let ${ \bar { Y } } _ { L } , { \bar { Y } } _ { R } , { \bar { Y } } _ { P }$ be the weighted means of the left, right, and parent intervals. Completing the square gives the unconstrained gain $x y ( { \bar { Y } } _ { L } -$ $\bar { Y } _ { R } ) ^ { 2 } / ( x + y )$ . If the fit is constrained to Θ, the gain instead equals

$$
\{ W _ { s , e } ^ { Y } ( b ) \} ^ { 2 } + ( x + y ) d ( \bar { Y } _ { P } , \Theta ) ^ { 2 } - x d ( \bar { Y } _ { L } , \Theta ) ^ { 2 } - y d ( \bar { Y } _ { R } , \Theta ) ^ { 2 } .
$$

Thus the stated identity holds whenever the projection terms vanish. For example, $Y = ( 2 , 3 )$ , unit weights, and $\Theta = [ - 1 , 1 ]$ give constrained gain zero but squared weighted CUSUM $1 / 2$ □

Proposition C.28 (Equal-weight reduction). $H w _ { i } \equiv w _ { ; }$ , weighted-coordinate sampling reduces to index-uniform sampling up to endpoint discretization, $d _ { w } ( u , v ) = w | u - v |$ , and RWCP becomes a robust profile-loss version ofunweighted change point detection.

Proposition C.29 (Additive-score WCP–GCP equivalence). Suppose a segment-level detector obeys

$$
\phi ( X _ { a : b } ) = \frac { \sum _ { i = a } ^ { b } \nu _ { i } \phi ( X _ { i } ) } { \sum _ { i = a } ^ { b } \nu _ { i } }\tag{38}
$$

for known $\nu _ { i } > 0$ . Then the generalized CUSUM based on $\phi ( X _ { s : t } )$ and $\phi ( X _ { t + 1 : e } )$ equals WCP with weights $w _ { i } = \nu _ { i }$ . Without (38), exact equivalence need not hold, particularly for context-sensitive detectors applied to concatenated text.

ProofofProposition C.28. If $w _ { i } \equiv w$ , then $W ( t ) = w t$ and weighted-coordinate uniform sampling maps to uniform sampling over the index axis up to the discretization created by $W ^ { - 1 }$ . Cumulative masses satisfy $S _ { a : b } ^ { w } = w ( \bar { b } - a + 1 )$ and $d _ { w } ( u , v ) = w | u - v |$ . The Huber loss becomes

$$
\sum _ { i \in A } \rho _ { c } \{ { \sqrt { w } } ( Y _ { i } - \theta ) \} ,
$$

which differs from an unweighted robust profile loss only by the common scale $\sqrt { w }$ and the corresponding tuning convention. Hence the method is a robust profile-loss analogue of VCP. □

Proof of Proposition C.29. Under (38),

$$
\phi ( X _ { s : b } ) = \bar { Y } _ { s : b } ^ { \nu } , \qquad \phi ( X _ { b + 1 : e } ) = \bar { Y } _ { b + 1 : e } ^ { \nu } .
$$

Substitution into the generalized segment contrast gives

$$
\left( \frac { S _ { s : b } ^ { \nu } S _ { b + 1 : e } ^ { \nu } } { S _ { s : e } ^ { \nu } } \right) ^ { 1 / 2 } | \bar { Y } _ { s : b } ^ { \nu } - \bar { Y } _ { b + 1 : e } ^ { \nu } | ,
$$

which is precisely WCP with $w _ { i } = \nu _ { i }$ . If the detector evaluates concatenated text with cross-sentence context, (38) may fail, and no algebraic identity follows. □

## C.3.3 ASSUMPTION VERIFICATION

Proposition C.30 (Curvature from central density). Suppose $Y _ { i } = \theta _ { j } + \sigma _ { i } \xi _ { i } , a _ { i } = \sqrt { w _ { i } } \sigma _ { i } \in [ a _ { - } , a _ { + } ] ,$ $\mathbb { E } \rho _ { c } \bar { ( } a _ { i } \xi _ { i } ) < \infty , \mathbb { E } \psi _ { c } ( a _ { i } \xi _ { i } ) = 0$ (for example by symmetry), and $\xi _ { i }$ has density $f _ { i }$ satisfying

$$
f _ { i } ( x ) \geq f _ { 0 } > 0 f o r | x | \leq R .
$$

$I f c / a _ { + } + r _ { 0 } / ( \sigma _ { i } ) \leq R$ uniformly, then Assumption 3.2 holds with

$$
m _ { 0 } \geq 2 f _ { 0 } \frac { c } { a _ { + } }
$$

up to truncation at one, and $M _ { 0 } \leq 1$

Proof. At differentiability points,

$$
\begin{array} { c } { Q _ { i } ^ { \prime \prime } ( \theta ) = w _ { i } \mathbb { P } \left( | \sqrt { w _ { i } } ( Y _ { i } - \theta ) | < c \right) } \\ { = w _ { i } \mathbb { P } \left( | a _ { i } \xi _ { i } + \sqrt { w _ { i } } ( \theta _ { j } - \theta ) | < c \right) . } \end{array}
$$

For $| \theta - \theta _ { j } | \leq r _ { 0 } .$ , the event contains an interval in $\xi _ { i }$ of length at least $2 c / a _ { + }$ <sub>+</sub> lying inside $[ - R , R ]$ Its probability is at least $2 f _ { 0 } c / a _ { + }$ . The upper bound follows because a probability is at most one. The score-centering condition makes $\theta _ { j }$ a minimizer; only then does integrating the second-derivative bounds twice around $\theta _ { j }$ give (7). □

Proposition C.31 (Active caps change the metric). Let $w _ { i } ^ { \circ } = \Pi _ { \left[ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } \right] } ( \sigma _ { i } ^ { - 2 } )$ . Then

$$
\sum _ { i \in ( u , v ] } w _ { i } ^ { \circ } \leq d _ { \mathrm { i n f o } } ( u , v )\tag{39}
$$

need not hold ifthe lower cap is active, and the reverse inequality need not hold ifthe upper cap is active. Therefore a guarantee in $d _ { w ^ { \circ } }$ cannot generally be called inverse-variance minimax optimal.

Proof. I $\mathrm { ~ f ~ } \sigma _ { i } ^ { - 2 } < w _ { \operatorname* { m i n } }$ , then $w _ { i } ^ { \circ } > w _ { i } ^ { \mathrm { i n f o } }$ , violating (39). If $\sigma _ { i } ^ { - 2 } > w _ { \mathrm { m a x } } ,$ , then $w _ { i } ^ { \circ } < w _ { i } ^ { \mathrm { i n f o } }$ , violating the reverse comparison. Only uniform comparability or inactive caps identify the two metrics up to constants. □

## C.3.4 DISTRIBUTIONAL AND CONTAMINATION CALCULATIONS

This section makes the abstract curvature constants in the main theorem concrete. The calculations are not needed for the logical validity of Theorem 3.11, but they clarify which score distributions satisfy its assumptions and how the Huber parameter affects the constants. Throughout this section, let

$$
q _ { F } ( u ) = \mathbb { E } _ { F } \rho _ { c } ( Z - u ) , \qquad Z \sim F ,
$$

where $F$ is a standardized detector-noise distribution. Define the central-mass modulus

$$
m _ { F } ( c , r ) = \operatorname* { i n f } _ { | u | \leq r } \mathbb { P } _ { F } ( | Z - u | < c ) .
$$

The next proposition converts this probability into both a risk-curvature bound and a two-population separation bound.

Proposition C.32 (Exact standardized Huber curvature). Suppose F is symmetric around zero and $\mathbb { E } _ { F } | \overline { { Z } } | < \infty$ . Then $u = 0$ minimizes $q _ { F } .$ . Moreover, at every differentiability point,

$$
q _ { F } ^ { \prime } ( u ) = - \mathbb { E } _ { F } \psi _ { c } ( Z - u ) , \qquad q _ { F } ^ { \prime \prime } ( u ) = \mathbb { P } _ { F } ( | Z - u | < c ) .
$$

Consequently, for every $| u | \leq r ,$

$$
\frac { m _ { F } ( c , r ) } { 2 } u ^ { 2 } \leq q _ { F } ( u ) - q _ { F } ( 0 ) \leq \frac { 1 } { 2 } u ^ { 2 } .\tag{40}
$$

Let $B , C > 0$ and $| \kappa | \leq r .$ Define the two-population profile separation

$$
\mathfrak { D } _ { F } ( B , C ; \kappa ) = 2 \operatorname* { i n f } _ { t \in \mathbb { R } } \left[ B \{ q _ { F } ( t ) - q _ { F } ( 0 ) \} + C \{ q _ { F } ( t - \kappa ) - q _ { F } ( 0 ) \} \right] .\tag{41}
$$

Then

$$
m _ { F } ( c , r ) \frac { B C } { B + C } \kappa ^ { 2 } \leq \mathfrak { D } _ { F } ( B , C ; \kappa ) \leq \frac { B C } { B + C } \kappa ^ { 2 } .\tag{42}
$$

Proof. The Huber loss is convex and differentiable except at two points. Dominated differentiation is valid for the first derivative because $| \psi _ { c } | \leq c .$ . Symmetry of $F$ and oddness of $\psi _ { c }$ imply

$$
q _ { F } ^ { \prime } ( 0 ) = - \mathbb { E } _ { F } \psi _ { c } ( Z ) = 0 .
$$

Convexity therefore makes zero a global minimizer. At points where $F$ has no atom at $u \pm c ,$ differentiating once more gives

$$
q _ { F } ^ { \prime \prime } ( u ) = \mathbb { E } _ { F } { \mathbf { 1 } } \{ | Z - u | < c \} = \mathbb { P } _ { F } ( | Z - u | < c ) .
$$

The same identity holds in the distributional sense when boundary atoms are present. For $u \geq 0$ , the fundamental theorem of calculus and $q _ { F } ^ { \prime } ( 0 ) = 0$ yield

$$
q _ { F } ( u ) - q _ { F } ( 0 ) = \int _ { 0 } ^ { u } q _ { F } ^ { \prime } ( v ) \mathrm { d } v = \int _ { 0 } ^ { u } \int _ { 0 } ^ { v } q _ { F } ^ { \prime \prime } ( z ) \mathrm { d } z \mathrm { d } v .
$$

On $[ 0 , r ]$ , the inner integrand lies between $m _ { F } ( c , r )$ and one. This proves (40); the case $u < 0$ follows by the same argument or by symmetry.

Because $q _ { F }$ is convex with unique minimizer zero whenever $m _ { F } ( c , r ) > 0$ , a minimizer of the expression in (41) lies between 0 and κ when $\kappa > 0 .$ , and between κ and 0 when $\kappa < 0$ . Hence both t and $t - \kappa$ have absolute value at most r. Applying the lower bound in (40) gives

$$
\begin{array} { l } { { \mathfrak { D } _ { F } } ( B , C ; \kappa ) \geq m _ { F } ( c , r ) \displaystyle \operatorname* { i n f } _ { t } \{ B t ^ { 2 } + C ( t - \kappa ) ^ { 2 } \} } \\ { = m _ { F } ( c , r ) \displaystyle \frac { B C } { B + C } \kappa ^ { 2 } , } \end{array}
$$

where the minimizing quadratic argument is ${ t = C \kappa / ( B + C ) }$ . Evaluating the original objective at this same argument and using the upper bound in (40) proves the upper inequality. □

Remark C.33 (Why the profile formulation is useful). Proposition C.32 controls the loss paid by forcing two populations to share one location. It does not require an explicit formula for the minimizer of the mixed risk

$$
t \mapsto B q _ { F } ( t ) + C q _ { F } ( t - \kappa ) .
$$

For a nonquadratic Huber loss, that minimizer is generally not $C \kappa / ( B + C )$ . The quadratic point is used only as a feasible argument for the upper bound, while strong convexity supplies the lower bound. This is precisely the distinction that makes profile-risk geometry more tractable than a direct difference of mixed-segment Huber centers.

Gaussian and Student examples. The central-mass modulus has a closed form for several standard heavy- and light-tailed reference families.

Corollary C.34 (Gaussian curvature constant). $I f Z \sim N ( 0 , 1 )$ and Φ denotes the standard normal distributionfunction, then

$$
m _ { \mathrm { G } } ( c , r ) = \Phi ( c - r ) + \Phi ( c + r ) - 1 .
$$

Therefore, for every $| \kappa | \leq r ,$

$$
\left\{ \Phi ( c - r ) + \Phi ( c + r ) - 1 \right\} \frac { B C } { B + C } \kappa ^ { 2 } \leq \mathfrak { D } _ { \mathtt { G } } ( B , C ; \kappa ) \leq \frac { B C } { B + C } \kappa ^ { 2 } .
$$

Proof. For $u \geq 0$

$$
\begin{array} { r } { \mathbb { P } ( | Z - u | < c ) = \Phi ( c + u ) - \Phi ( - c + u ) \qquad } \\ { = \Phi ( c + u ) + \Phi ( c - u ) - 1 . } \end{array}\tag{43}
$$

Its derivative is $\varphi ( c + u ) - \varphi ( c - u ) \leq 0$ , because $| c - u | \leq c + u$ and the Gaussian density $\varphi$ decreases with absolute value. The probability is even in u, so its minimum on $[ - r , r ]$ is attained at $| u | = r$ . Substitute the resulting modulus into Proposition C.32. □

Corollary C.35 (Student curvature constant). Let $Z$ have a centered Student distribution with $\nu > 1$ degrees offreedom and distributionfunction $T _ { \nu }$ . Then

$$
m _ { t _ { \nu } } ( c , r ) = T _ { \nu } ( c - r ) + T _ { \nu } ( c + r ) - 1 ,\tag{44}
$$

and the separation sandwich (42) holds with this constant.

Proof. The Student density is symmetric and nonincreasing in absolute value. Repeating (43) with $T _ { \nu }$ in place of Φ shows that the probability of a length-2c window is minimized at the largest displacement $| u | = r$ . Since $\nu > 1$ , the first moment required in Proposition C.32 is finite. □

The preceding corollary illustrates why the raw detector score need not be sub-Gaussian. A Student variable with small ν has polynomial tails, while the Huber score remains bounded and the local population risk still has positive curvature whenever (44) is positive.

Symmetric and asymmetric contamination. We next quantify two distinct effects of contamination. Symmetric contamination preserves the Huber target but reduces curvature. Asymmetric contamination can move the target, although bounded Huber influence controls that displacement.

Proposition C.36 (Symmetric contamination preserves the target). Let

$$
F _ { \epsilon } = ( 1 - \epsilon ) F _ { 0 } + \epsilon H , \qquad 0 \le \epsilon < 1 ,\tag{45}
$$

where both $F _ { 0 }$ and H are symmetric around zero and havefinitefirst moments. Then zero minimizes $q _ { F _ { \epsilon } }$ and

$$
m _ { F _ { \epsilon } } ( c , r ) \geq ( 1 - \epsilon ) m _ { F _ { 0 } } ( c , r ) .\tag{46}
$$

Consequently,

$$
\mathfrak { D } _ { F _ { \epsilon } } ( B , C ; \kappa ) \ge ( 1 - \epsilon ) m _ { F _ { 0 } } ( c , r ) \frac { B C } { B + C } \kappa ^ { 2 }
$$

for every $| \kappa | \leq r .$

Proof. A mixture of symmetric distributions is symmetric, so the first conclusion follows from Proposition C.32. For every $u ,$

$$
\begin{array} { r l } & { { \mathbb P } _ { F _ { \epsilon } } ( | Z - u | < c ) = ( 1 - \epsilon ) { \mathbb P } _ { F _ { 0 } } ( | Z - u | < c ) + \epsilon { \mathbb P } _ { H } ( | Z - u | < c ) } \\ & { \qquad \geq ( 1 - \epsilon ) { \mathbb P } _ { F _ { 0 } } ( | Z - u | < c ) . } \end{array}
$$

Taking the infimum over $| u | \leq r$ proves (46); the separation result follows from (42).

Proposition C.37 (Displacement under asymmetric contamination). Let $F _ { 0 }$ be symmetric around zero and suppose

$$
q _ { F _ { 0 } } ( u ) - q _ { F _ { 0 } } ( 0 ) \geq \frac { m _ { 0 } } { 2 } u ^ { 2 } , \qquad | u | \leq r ,
$$

for $m _ { 0 } > 0$ . Let H be an arbitrary distribution withfinitefirst moment and define $F _ { \epsilon }$ by (45). Let

$$
u _ { \epsilon } \in \underset { | u | \leq r } { \arg \operatorname* { m i n } } q _ { F _ { \epsilon } } ( u ) .
$$

Then

$$
\left. u _ { \epsilon } \right. \leq \frac { 2 \epsilon c } { ( 1 - \epsilon ) m _ { 0 } } .
$$

If the right-hand side is strictly smaller than $^ { r , }$ every constrained minimizer is interior and hence is also an unconstrained local minimizer.

Proof. The Huber loss is c-Lipschitz, so for every $u ,$

$$
\begin{array} { r } { | q _ { H } ( u ) - q _ { H } ( 0 ) | \leq c | u | . } \end{array}
$$

Because $u _ { \epsilon }$ minimizes the contaminated risk over $[ - r , r ]$

$$
\begin{array} { r l } & { 0 \geq q _ { F _ { \epsilon } } ( u _ { \epsilon } ) - q _ { F _ { \epsilon } } ( 0 ) } \\ & { = ( 1 - \epsilon ) \{ q _ { F _ { 0 } } ( u _ { \epsilon } ) - q _ { F _ { 0 } } ( 0 ) \} + \epsilon \{ q _ { H } ( u _ { \epsilon } ) - q _ { H } ( 0 ) \} } \\ & { \geq \frac { ( 1 - \epsilon ) m _ { 0 } } { 2 } u _ { \epsilon } ^ { 2 } - \epsilon c \lvert u _ { \epsilon } \rvert . } \end{array}
$$

If $u _ { \epsilon } = 0$ , the conclusion is immediate. Otherwise divide by $| u _ { \epsilon } |$ and rearrange. Strict interiority follows when the bound is less than $r .$ □

Remark C.38 (Interpretation of asymmetric contamination). Theorem 3.11 is formulated in terms of the segment-wise population Huber locations actually identified by the score distributions. Proposition C.37 therefore does not add an unmodeled bias term to that theorem. Instead, it quantifies how far those robust targets may move from a symmetric clean center. If two adjacent segments have clean centers separated by $\kappa _ { 0 }$ and contamination levels $\epsilon _ { - }$ and $\epsilon _ { + }$ , then their contaminated Huber-location jump obeys the deterministic lower bound

$$
\kappa \geq \left[ \kappa _ { 0 } - \frac { 2 \epsilon _ { - } c } { ( 1 - \epsilon _ { - } ) m _ { - } } - \frac { 2 \epsilon _ { + } c } { ( 1 - \epsilon _ { + } ) m _ { + } } \right] _ { + } .
$$

Thus contamination affects localization through both curvature and effective separation.

Finite-sample replacement sensitivity. The next result is deterministic. It does not claim robustness to an arbitrary fraction of adversarially replaced sentences; rather, it quantifies the local influence of a finite replacement set on one interval-level Huber fit.

Proposition C.39 (Deterministic replacement sensitivity). Fix an interval A and observations $\left\{ y _ { i } : i \in A \right\}$ . Let

$$
L _ { A } ( \theta ) = \sum _ { i \in A } \rho _ { c } ( \sqrt { w _ { i } } ( y _ { i } - \theta ) )
$$

have minimizer $\widehat { \theta } _ { A }$ . Replace the observations on a set $C \subseteq A$ by arbitrary values to obtain $\widetilde { L } _ { A }$ and minimizer $\widetilde { \theta } _ { A }$ . Suppose both minimizers lie in an interval I on which $L _ { A }$ is $\alpha S _ { A } ^ { w }$ -strongly convex. Then

$$
| \widetilde { \theta } _ { A } - \widehat { \theta } _ { A } | \leq \frac { 2 c } { \alpha S _ { A } ^ { w } } \sum _ { i \in C } \sqrt { w _ { i } } \leq \frac { 2 c \sqrt { w _ { \operatorname* { m a x } } } | C | } { \alpha S _ { A } ^ { w } } .\tag{47}
$$

Proof. Choose zero subgradients $g _ { A } ( \widehat { \theta } _ { A } ) = 0 \mathrm { a n d } \widetilde { g } _ { A } ( \widetilde { \theta } _ { A } ) = 0$ . Replacing one observation changes its score contribution

$$
\sqrt { w _ { i } } \psi _ { c } ( \sqrt { w _ { i } } ( y _ { i } - \theta ) )
$$

by at most $2 c \sqrt { w _ { i } }$ , uniformly in θ. Hence

$$
| g _ { A } ( \theta ) - \widetilde { g } _ { A } ( \theta ) | \leq 2 c \sum _ { i \in C } \sqrt { w _ { i } } \qquad { \mathrm { f o r ~ a l l ~ } } \theta .
$$

Strong convexity implies strong monotonicity of the subgradient map on $I { \boldsymbol { : } }$

$$
| g _ { A } ( u ) - g _ { A } ( v ) | \geq \alpha S _ { A } ^ { w } | u - v | , \qquad u , v \in I .
$$

Apply this inequality with $u = \widetilde { \theta } _ { A }$ and $v = \widehat { \theta } _ { A }$ . Since $g _ { A } ( \widehat { \theta } _ { A } ) = 0$ and $\widetilde { g } _ { A } ( \widetilde { \theta } _ { A } ) = 0 .$

$$
\begin{array} { r l r } {  { \alpha S _ { A } ^ { w } | \widetilde { \theta } _ { A } - \widehat { \theta } _ { A } | \le | g _ { A } ( \widetilde { \theta } _ { A } ) | } } \\ & { } & { = | g _ { A } ( \widetilde { \theta } _ { A } ) - \widetilde { g } _ { A } ( \widetilde { \theta } _ { A } ) | } \\ & { } & { \le 2 c \displaystyle \sum _ { i \in C } \sqrt { w _ { i } } . } \end{array}
$$

The second inequality in (47) follows from $w _ { i } \leq w _ { \mathrm { m a x } }$

Remark C.40 (Scope of Proposition C.39). The bound concerns the fitted location on a fixed interval and requires empirical curvature along the path between the two fits. It does not by itself establish exact change point recovery under adversarial replacements, because an adversary may also alter which intervals cross the threshold. A full adversarial theorem would need a simultaneous bound on every affected profile gain and a contamination-aware signal condition. We therefore use (47) as a sensitivity diagnostic rather than as an unsupported global robustness claim.

Constants and the Huber tuning parameter. The main theorem suppresses fixed constants to keep its statement readable. Tracking the proof reveals the two roles of $c .$ The score concentration scale is proportional to the uniform score bound

$$
b _ { c } = c \sqrt { w _ { \mathrm { m a x } } } ,
$$

whereas empirical and population curvature are controlled by central probabilities such as $m _ { F } ( c , r )$ and $p _ { 0 } ( c , r )$ . The resulting localization constant therefore has the schematic form

$$
d _ { w } ( \hat { \tau } _ { j } , \tau _ { j } ) \leq C _ { \mathrm { d e p } , w } \hat { \mathsf { R } } ( c , r , p _ { 0 } ) \frac { \Lambda _ { N } ( \delta ) } { \kappa _ { j } ^ { 2 } } ,\tag{48}
$$

where $C _ { \mathrm { d e p } , w }$ depends on fixed dependence and weight bounds, and $\textstyle \mathcal { R } ( c , r , p _ { 0 } )$ collects the bounded score and inverse-curvature constants. We do not claim this schematic factor is the sharp optimized constant. Crucially, all multiplicity and dependence growth remains in $\Lambda _ { N } ( \delta )$ , including the $q _ { N } ^ { 2 }$ penalty. Equation (48) records the roles of the tuning quantities without replacing the rate in Theorem 3.11.

For a reference family $F ,$ a transparent tuning proxy is

$$
\mathfrak { C } _ { F } ( c ; r ) = \frac { c ^ { 2 } } { m _ { F } ( c , r ) ^ { 2 } } .
$$

A small value of c limits the influence of extreme scores but may reduce the mass in the locally quadratic region. A large value increases curvature toward its quadratic-loss limit but also increases the bounded-score constant. The proxy can be minimized on a predetermined compact grid without using ground-truth boundaries. In practice, the grid should be chosen using a training or calibration corpus so that selection does not invalidate the target-document guarantee.

Proposition C.41 (Continuity of the tuning proxy). Suppose F has a continuous strictly positive density on every compact interval. For fixed $r < \infty ,$ , thefunction $c \mapsto m _ { F } ( c , r )$ is continuous and strictly positivefor $c > 0 .$ . Hence ${ \mathfrak { C } } _ { F } ( c ; r )$ is continuous on every compact interval $[ c _ { - } , c _ { + } ] \subset ( 0 , \infty )$ and attains a minimum there.

Proof. For each fixed $u ,$ the map

$$
c \mapsto \mathbb { P } _ { F } ( | Z - u | < c ) = F ( u + c ) - F ( u - c )
$$

is continuous. Joint continuity in $( c , u )$ follows from continuity of the distribution function, and the infimum over the compact set $| u | \leq r$ is therefore continuous by the maximum theorem. Strict positivity follows because every interval $( u - c , u + c )$ has positive probability under a strictly positive density. The remaining claims follow from elementary continuity and compactness. □

Remark C.42 (Quadratic-loss limit). The algebraic identity in Proposition 2.2 holds in the quadratic limit when the unconstrained means are feasible. The heavy-tail concentration proof, however, does not pass to this limit because the score bound $b _ { c }$ diverges. Recovering the classical WCP probability bound at c = ∞ requires a separate tail assumption on the raw scores, such as the sub-Gaussian condition used in the foundational analysis. Algebraic reduction and probabilistic reduction are therefore distinct statements.