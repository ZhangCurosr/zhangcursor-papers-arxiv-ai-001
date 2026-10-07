# Early Memory Selection for Balanced Adam

Alberto Fernández-Hernández   
Universitat Politècnica de València Valencia, Spain a.fernandez@upv.es

Cristian Pérez-Corral Universitat Politècnica de València Valencia, Spain cpercor@upv.es

Manuel F. Dolz Universitat Jaume I Castelló de la Plana, Spain dolzm@uji.es

Jose I. Mestre sitat Politècnica de València Valencia, Spain jimesmir@upv.es

Enrique S. Quintana-Ortí Universitat Politècnica de València Valencia, Spain quintana@disca.upv.es

## Abstract

We propose a method for choosing the shared memory parameter $\beta _ { 1 } ~ = ~ \beta _ { 2 } ~ = ~ \beta$ in Adam from a short pilot training. The selected $\beta$ remains fixed during the subsequent full training. A local model of Adam’s normalized direction balances sampling variability against the delay introduced by averaging past gradients. This balance gives a cubic memory rule, whose two coeficients are estimated from gradient probes at a few pilot checkpoints. The estimator uses the numerator and denominator jointly, preserving their covariance. With a 200-update pilot and sixteen probe gradients at each of four checkpoints, a seed-matched retrospective evaluation on eleven vision and language workloads reduces mean relative validation gap by 40.7% and worst-quarter mean gap by 44.3% against the grid representative of shared $\beta$ 0.95. The mean gap is also 32.3% lower than that of the best constant $\beta$ chosen across all eleven workloads.

## 1 INTRODUCTION

Adam’s memory is a statistical timescale: it determines which past gradients still influence the current update (Kingma and Ba, 2015). A short memory responds quickly to changes and retains more sampling variability. A long memory averages that variability more efectively and follows a changing gradient with a delay. The useful timescale therefore depends on the dynamics of the task. A single fixed default makes a compromise across those dynamics; early measurement ofers a way to choose the timescale for the workload at hand.

Recent language-model experiments show that setting the two parameters equal can retain strong training performance (Orvieto and Gower, 2025). This family is called balanced Adam: $\beta _ { 1 } = \beta _ { 2 } = \beta$ . It reduces tuning to one memory while preserving coordinatewise normalization. A shared value around 0.95 is a useful empirical reference. The remaining choice still matters: one memory per training is diferent from one memory for every training. This raises a concrete question: can a short observation of gradient dynamics choose a useful fixed memory for each workload?

Adam normalizes an averaged gradient by an averaged squared gradient, so the errors of its two moments act jointly. We retain their covariance and analyze how averaging reduces fluctuations while delaying the response to a changing direction. A first-order local model separates these two efects and gives a memory rule in units $H = ( 1 - \beta ) ^ { - 1 }$ . Grouped gradient probes and checkpoint diferences estimate its coefficients. Completed fixed-memory trainings evaluate the resulting choices.

The contributions are:

• A joint-moment tracking objective that separates retained noise from delay and yields a cubic memory law. Its sensitivity formula explains why useful calibration tolerates coeficient error.

• An explicit early finite-probe rule and a finiteresolution lag interpretation, with a quantified population-model allowance for endpoint uncertainty.

• An eleven-workload evaluation against every fixed grid memory, with matched training recipes, tasklevel calibration, dense timing sensitivity, and gradient-equivalent cost accounting.

Diferent fixed $\beta$ values serve diferent tasks well. Our selector uses gradients to choose a memory for each run. On the evaluated suite it reduces mean, maximum, and tail gap compared with every fixed grid value, including the best constant selected from all validation outcomes. Code and reproducible evaluation material are available at https://github.com/ AlbertoFdezHdez/Adam\_beta\_rule\_cubic.

## 2 RELATED WORK

We distinguish analyses of Adam’s moments, theoretical relationships between its $\beta$ values, and methods that learn optimizer parameters during training. Our question is the early choice of one fixed shared memory.

Adam’s statistical structure and shared memories. Balles and Hennig (2018) analyze the interaction between sign, magnitude, and variance in Adam. Orvieto and Gower (2025) provide extensive empirical support for equal memory parameters and a statistical perspective on Adam. These results motivate the shared-memory family. A complementary budgetbased rule relates its memory to a useful training horizon (Fernández-Hernández et al., 2026). The present rule obtains its selection signal from early normalizeddirection variability and temporal change.

β relationships and optimization guarantees. Ahn et al. (2024) interpret Adam through online learning of updates and follow-the-regularized-leader. Nguyen (2026) analyze both sides of $\beta _ { 1 } ~ = ~ \sqrt { \beta _ { 2 } }$ whose optimality depends on the adversary model. Their criterion relates the memories under online regret; ours chooses a shared memory from an observed gradient process under local tracking risk. Stochastic-diferential-equation analyses explain scaling with batch size (Malladi et al., 2022), whereas we hold the training recipe fixed and measure its early dynamics.

Learning optimizer parameters. Chandra et al. (2022) diferentiate through optimizers to learn their hyperparameters, including both Adam memories. MADA jointly learns memory and optimizerinterpolation coeficients through online hypergradient descent; its fixed-state variant reuses the optimizer learned at the end of another training (Ozkara et al.,

2024). Our rule instead uses an early noise–lag measurement to choose one shared $\beta ,$ fixed for full training.

Adaptive momentum and moment estimation. Topollai and Choromanska (2026) derive online momentum adaptation from a two-plane objective model. Their AdamW variant adapts the first-moment memory while keeping the second-moment memory fixed, with layerwise adaptation in language models. YellowFin estimates curvature and gradient statistics to tune SGD’s momentum and learning rate (Zhang and Mitliagkas, 2019). Kalman-Adam estimates first and second moments using separate Bayesian filters (Ali and Bhatt, 2026). The present method retains Adam’s exponential averages and calibrates their common timescale early, using fluctuations and change of their joint normalized target. Online memory adaptation and early fixed shared-memory selection provide complementary ways to use information from the training problem.

## 3 WHAT SHOULD THE MEMORY TRACK?

Choosing a memory starts with identifying what Adam’s numerator and denominator jointly estimate. This statistical reference gives a directly analyzable objective: the tracking error of the normalized update direction.

## 3.1 Adam and its population reference

Let ${ \theta } _ { t } ~ \in ~ \mathbb { R } ^ { d }$ be the parameters and $g _ { t } \in \mathbb { R } ^ { d }$ an efective-batch gradient, including accumulation and the recipe’s gradient processing. From zero initial moments Adam computes

$$
\begin{array} { r l } & { m _ { t } = \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) g _ { t } , } \\ & { ~ v _ { t } = \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { \odot 2 } , } \\ & { u _ { t } = \cfrac { m _ { t } / ( 1 - \beta _ { 1 } ^ { t } ) } { \sqrt { v _ { t } / ( 1 - \beta _ { 2 } ^ { t } ) } + \varepsilon } . } \end{array}\tag{1}
$$

The superscript ⊙2 means coordinatewise squaring. Square roots and division are coordinatewise, with ε > 0 outside the root. The gradient update is −η<sub>t</sub>u<sub>t</sub>; AdamW additionally applies decoupled weight decay (Loshchilov and Hutter, 2019).

At a fixed reference time, the expectation is over a fresh efective-batch gradient at the current parameters. Write $\mu = \mathbb { E } g , q = \mathbb { E } g ^ { \odot 2 }$ , and define

$$
r = \phi ( \mu , q ) , \phi ( x , y ) = \frac { x } { \sqrt { y } + \varepsilon } .\tag{2}
$$

This normalized population direction includes gradient variability in its denominator. In one coordinate, $\mu = 1$ and $q = 4$ give $r \approx 1 / 2$ for small epsilon; a deterministic gradient of one gives $r \approx 1$ . Thus the reference combines the gradient’s sign with a magnitude determined by its mean and second moment. If clipping is used, $\mu$ is the mean clipped gradient. The reference describes what Adam’s processed moments track.

## 3.2 The joint efect of moment errors

In one coordinate, first-order perturbation gives

$$
\delta r \approx \frac { \delta m } { \sqrt { q } + \varepsilon } - \frac { \mu \delta v } { 2 \sqrt { q } ( \sqrt { q } + \varepsilon ) ^ { 2 } } .\tag{3}
$$

An overestimated numerator increases the update, whereas an overestimated denominator decreases it. Their errors can therefore partly compensate. Because both moments use the same gradient samples, their covariance matters. The following diagonal matrices give the coordinatewise sensitivities of the normalized direction to each moment:

$$
A = \mathrm { d i a g } \frac { 1 } { \sqrt { q } + \varepsilon } , \quad B = \mathrm { d i a g } \frac { - \mu } { 2 \sqrt { q } ( \sqrt { q } + \varepsilon ) ^ { 2 } } .\tag{4}
$$

Diferentiation assumes $q _ { i } > 0 .$ Practical probes use a separately specified numerical floor.

## 4 A LOCAL NOISE–LAG BALANCE

The next question is how averaging afects a direction that moves steadily over a short window. The quantity to minimize is the expected squared diference between the first-order Adam direction and the current normalized population direction. This tracking risk separates smoothing and delay; it supplies a statistical objective for choosing memory, which is subsequently evaluated through training losses.

## 4.1 Explicit local assumptions

The assumptions describe a local approximation. A diferentiable moment curve has a constant term and a linear term in its first-order expansion; these motivate afine means over the window. Holding the covari ance fixed is the corresponding local approximation for noise. Independence across ages isolates the efect of temporal averaging. This last condition is an additional modeling assumption, since adaptive training can introduce temporal dependence. Dependence between numerator and denominator errors at the same age is retained throughout. Appendix B explains how deviations enter the approximation residual.

Assumption 1 (Local first-order moment model).   
Fix an update $t \geq 1$ and a history of t observations.

Their ages are $k = 0 , \ldots , t - 1 \colon$ age zero is the current observation, and age one is one update old. The current moment means $\mu , q$ and their changes per update $\dot { \mu } , \dot { q }$ are deterministic, with $q _ { i } > 0$ . The model’s moment observations at age k are $\mu - k \dot { \mu } + \xi _ { k }$ and $q - k \dot { q } + \chi _ { k }$ . The random vectors $\xi _ { k } , \chi _ { k } \in \mathbb { R } ^ { d }$ are centered deviations from these means. Their pairs are independent across ages and share one finite covariance; the two vectors in a pair may be dependent.

For example, $\dot { \mu } _ { i } = 0 . 0 1$ says that the expected gradi ent in coordinate i increases by 0.01 per update under this model. The model describes fluctuations of both moments around their local means. Actual gradient observations and their squares impose additional joint structure; the stated covariance can retain it. The theorem is exact for the specified linearized model, and its use for nonlinear Adam is a local approximation.

Define the direction innovation and slope by

$$
\zeta _ { k } = A \xi _ { k } + B \chi _ { k } , \qquad d _ { r } = A \dot { \mu } + B \dot { q } .
$$

Here $\zeta _ { k }$ is the direction error caused by one noisy moment observation, and $d _ { r }$ is the first-order change of the normalized direction in one update. Their size is summarized by

$$
V _ { \mathrm { p o p } } = \mathbb { E } \left. \zeta _ { k } \right. ^ { 2 } , \qquad D _ { \mathrm { p o p } } = \left. d _ { r } \right. ^ { 2 } .\tag{5}
$$

$V _ { \mathrm { p o p } }$ measures squared normalized-direction fluctua tion per observation and includes the cross contribu tion $2 \mathbb { E } \langle A \xi _ { k } , B \chi _ { k } \rangle . \ D _ { \mathrm { p o p } }$ measures squared direction change per update. Random fluctuations accumulate through squared averaging weights; a systematic slope accumulates through the mean age of the observations.

For balanced memory $\beta ,$ bias correction gives weights

$$
w _ { \beta , t } ( k ) = \frac { ( 1 - \beta ) \beta ^ { k } } { 1 - \beta ^ { t } } , \quad 0 \le k < t .\tag{6}
$$

The weights sum to one. Define their mean age $\begin{array} { r } { { a } _ { \beta , t } = \sum _ { k = 0 } ^ { t - 1 } w _ { \beta , t } ( k ) k } \end{array}$ and squared-weight sum ${ c } _ { \beta , t } =$ $\scriptstyle \sum _ { k = 0 } ^ { t - 1 } w _ { \beta , t } ( k ) ^ { 2 }$ . The first measures delay in updates; the second measures the fraction of independent observation noise retained by the average. For an ordinary average of $N$ observations, this squared-weight sum is $1 / N$

Theorem 1 (Finite-history balanced tracking risk). Under Assumption 1, the first-order error is

$$
e _ { \beta , t } = \sum _ { k } w _ { \beta , t } ( k ) \zeta _ { k } - a _ { \beta , t } d _ { r } ,
$$

and its mean squared norm equals

$$
\begin{array} { r } { \mathbb { E } \left\| e _ { \beta , t } \right\| ^ { 2 } = V _ { \mathrm { p o p } } c _ { \beta , t } + D _ { \mathrm { p o p } } a _ { \beta , t } ^ { 2 } . } \end{array}\tag{7}
$$

For $0 < \beta < 1$

$$
c _ { \beta , t } = \frac { 1 - \beta } { 1 + \beta } \frac { 1 + \beta ^ { t } } { 1 - \beta ^ { t } } ,\tag{8}
$$

$$
a _ { \beta , t } = \frac { \beta } { 1 - \beta } - \frac { t \beta ^ { t } } { 1 - \beta ^ { t } } .
$$

$$
A t \ \beta = 0 , c _ { 0 , t } = 1 \ a n d \ a _ { 0 , t } = 0 .
$$

Proof. The averaged first-moment deviation is $\begin{array} { r } { - a _ { \beta , t } \dot { \mu } + \sum _ { k } w _ { \beta , t } ( k ) \xi _ { k } ; } \end{array}$ the second-moment deviation has the same form with $\dot { q } , \chi _ { k }$ Applying the linear sensitivities A, B gives the displayed error. The innovation sum is centered, so expanding the squared error leaves its variance plus the squared lag $a _ { \beta , t } ^ { 2 } \left. d _ { r } \right. ^ { 2 }$ . For diferent ages, independence and centering give $\mathbb { E } \langle \zeta _ { k } , \zeta _ { \ell } \rangle \ : = \ : 0$ Thus the variance is $\begin{array} { r } { \sum _ { k } w _ { \beta , t } ( k ) ^ { 2 } V _ { \mathrm { p o p } } } \end{array}$ , proving the risk identity. The weight formulas follow from the geometric sums worked out in Appendix A. □

Equation (7) is the mechanism: longer memory suppresses noise and increases lag. Scalar weight formulas make this risk directly computable. For any fixed linear map $P ,$ , the theorem also holds in its measurement geometry: replace $\zeta _ { k } , d _ { r }$ by $P \zeta _ { k } , P d _ { r }$ in (5).

## 4.2 From tracking risk to a memory value

In memory units $H = ( 1 - \beta ) ^ { - 1 } \geq 1 , \beta 0 . 9$ corresponds to $H = 1 0$ , and $\beta ~ 0 . 9 9$ to $H = 1 0 0$ . First consider a history long relative to the memory, so that $\beta ^ { t }$ is small. Then (7) approaches

$$
f ( H ) = \frac { V _ { \mathrm { p o p } } } { 2 H - 1 } + D _ { \mathrm { p o p } } ( H - 1 ) ^ { 2 } .\tag{9}
$$

For positive coeficients its unique minimizer solves

$$
D _ { \mathrm { p o p } } ( H - 1 ) ( 2 H - 1 ) ^ { 2 } = V _ { \mathrm { p o p } } .\tag{10}
$$

The derivative vanishes there, and the left-hand side increases strictly from zero to infinity on $H \geq 1$ . This equation is readily solved in one dimension. For a closed-form approximation, when H is large relative to one replace $2 H - 1$ by 2H and H − 1 by H. This second simplification gives

$$
f _ { 0 } ( H ) = \frac { V _ { \mathrm { p o p } } } { 2 H } + D _ { \mathrm { p o p } } H ^ { 2 } , \quad H _ { \star } = \left( \frac { V _ { \mathrm { p o p } } } { 4 D _ { \mathrm { p o p } } } \right) ^ { 1 / 3 } .\tag{11}
$$

Indeed, $f _ { 0 } ^ { \prime } ( H ) \ = \ - V _ { \mathrm { p o p } } / ( 2 H ^ { 2 } ) + 2 D _ { \mathrm { p o p } } H$ vanishes when $4 D _ { \mathrm { p o p } } H ^ { 3 } = V _ { \mathrm { p o p } } .$ More noise favors a longer average; faster direction change favors a shorter one. For $\bar { V } _ { \mathrm { p o p } } ~ = ~ 0 . 1 8$ and $D _ { \mathrm { p o p } } ~ = ~ 4 \times 1 0 ^ { - 6 }$ this gives $H _ { \star } \approx 2 2 . 4 1$ , hence $\beta _ { \star } \approx 0 . 9 5 5 4$ . These values illustrate the local model. Appendix A gives the constrained and zero-coeficient cases.

## 4.3 Sensitivity to coeficient errors

Finite measurements introduce coeficient uncertainty. The next result quantifies its efect on the continuous memory and the resulting model risk.

Proposition 1 (Continuous memory sensitivity). For the interior leading model, suppose positive estimates satisfy $| \log ( \widetilde { V } / V _ { \mathrm { p o p } } ) | \le \epsilon _ { V }$ and $| \log ( \widetilde { D } / D _ { \mathrm { p o p } } ) | \le \epsilon _ { D }$ Their minimizer $\bar { \tilde { H } } = ( \tilde { V } / ( 4 \widetilde { D } ) ) ^ { 1 / 3 }$ satisfies

$$
| \log ( \widetilde { H } / H _ { \star } ) | \le ( \epsilon _ { V } + \epsilon _ { D } ) / 3 .
$$

Writin $g \ z = \widetilde { H } / H _ { \star }$ , its relative model risk is

$$
\frac { f _ { 0 } ( \widetilde { H } ) } { f _ { 0 } ( H _ { \star } ) } = \frac { 2 / z + z ^ { 2 } } { 3 } .\tag{12}
$$

Proof. Dividing the two cubic formulas and taking logs gives $\begin{array} { r } { \log ( \tilde { H } / \bar { H _ { \star } } ) = [ \log ( \tilde { V } / V _ { \mathrm { p o p } } ) - \log ( \tilde { D } / D _ { \mathrm { p o p } } ) ] / 3 . } \end{array}$ The triangle inequality gives the bound. To compare risks, use the optimality identity $V _ { \mathrm { p o p } } / ( 2 H _ { \star } ) =$ $2 D _ { \mathrm { p o p } } H _ { \star } ^ { 2 }$ . Substituting ${ \cal \tilde { H } } \ = \ z { \cal H } ,$ makes the noise term $2 D _ { \mathrm { p o p } } H _ { \star } ^ { 2 } / z$ and the lag term $z ^ { 2 } D _ { \mathrm { p o p } } H _ { \star } ^ { 2 }$ . Their sum divided by $3 D _ { \mathrm { p o p } } H _ { \star } ^ { 2 }$ proves (12). □

The numbers $\epsilon _ { V } , \epsilon _ { D }$ describe multiplicative coeficient accuracy on a log scale; a factor-two error has magnitude log 2. The cube root attenuates such errors. More importantly, excess relative risk equals $( z - 1 ) ^ { 2 } ( z +$ $2 ) / ( 3 z )$ : near the optimum, risk changes quadratically with memory error. Doubling $V _ { \mathrm { p o p } }$ with accurate $D _ { \mathrm { p o p } }$ changes memory by $2 ^ { 1 / 3 }$ and increases leading risk by about 5.8%. Thus useful memory calibration can tolerate appreciable coeficient error. The practical rule additionally has sampling and grid-rounding efects.

## 5 ESTIMATING THE COEFFICIENTS FROM A PILOT

The memory rule in (11) requires two coeficients: the variability of the normalized direction at fixed parameters, and its change as training proceeds. We estimate the first by comparing gradient groups at one checkpoint and the second by comparing checkpoints. This section specifies the measurements and derives the scaling of both estimates; Section 6 then turns them into a $\beta .$

## 5.1 Measurement design and moment estimates

Let $T _ { \mathrm { p } }$ be an integer pilot length in optimizer updates and $\bar { \beta } _ { \mathrm { p i l o t } } \in [ 0 , 1 )$ its shared memory. Choose distinct integer lags $1 \leq n _ { 1 } , \ldots , n _ { K } \leq T _ { \mathrm { p } } ,$ with $K \geq 2 .$ , and measure at $T _ { \mathrm { p } }$ and $T _ { \mathrm { p } } - n _ { \ell }$ . Write $L = \operatorname* { m a x } _ { \ell } n _ { \ell }$ for the longest lag. At each checkpoint s, freeze the parameters and collect $m \geq 2$ fresh gradients $g _ { s } ^ { ( 1 ) } , \ldots , g _ { s } ^ { ( m ) }$ Each uses the same efective batch and gradient processing as a training update, including accumulation and clipping when prescribed. The pilot length, pilot $\beta ,$ lags, and probe allocation are measurement hyperparameters.

The sample means estimate the two population moments together:

$$
\widehat { \mu } _ { s } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } g _ { s } ^ { ( i ) } , \qquad \widehat { q } _ { s } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( g _ { s } ^ { ( i ) } ) ^ { \odot 2 } .
$$

Apply the same normalization as in (2):

$$
\widehat { r } _ { s } = \phi ( \widehat { \mu } _ { s } , \widehat { q } _ { s } ) = \frac { \widehat { \mu } _ { s } } { \sqrt { \widehat { q } _ { s } } + \varepsilon } .\tag{13}
$$

The optimizer’s $\varepsilon > 0$ makes this defined even when a sample second moment is zero. Estimating both moments from the same gradients preserves their joint fluctuations.

Split the m gradients into $G \ge 2$ equal groups, where G divides m. For group $j ,$ compute its moment means and normalized direction $\widehat { r } _ { s } ^ { ( j ) }$ by the same formulas restricted to its $m / G$ gradients. A fixed linear map $P$ specifies the measurement geometry; it may be the identity or a storage-saving projection. Define the pooled vector $z _ { s } = P \widehat { r } _ { s }$ and group vectors $z _ { s } ^ { ( j ) } = P \widehat { r } _ { s } ^ { ( j ) }$ The same P is used at every checkpoint. Theorem 1 applies in this geometry with noise coeficient $V _ { P } = \mathbb { E } \| P \zeta _ { k } \| ^ { 2 }$ and slope coeficient $D _ { P } = \left. P d _ { r } \right. ^ { 2 }$ All norms below refer to these measured vectors.

## 5.2 Estimating variability at fixed parameters

At the final pilot checkpoint, use

$$
\widehat { V } _ { \mathrm { e m p } } = \frac { m } { G - 1 } \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \left. z _ { T _ { \mathrm { p } } } ^ { ( j ) } - z _ { T _ { \mathrm { p } } } \right. ^ { 2 } .\tag{14}
$$

The multiplier corrects for averaging inside each group. In the local linearized model, a group averages $m / G$ independent direction innovations, so its variance trace is $G V _ { P } / m$ . Centering G independent groups around their mean retains the fraction $( G - 1 ) / G$ of this variance. The expected mean squared group deviation is therefore $( G - 1 ) V _ { P } / m ;$ multiplying by $m / ( G - 1 )$ recovers $V _ { P }$ . This is the same degrees-of-freedom correction as in sample variance, together with the correction for group size.

Nonlinear normalization adds a second contribution: normalizing pooled moments can difer from averaging separately normalized groups. The following identity quantifies that contribution without an approximation.

Proposition 2 (Dispersion decomposition). For group vectors $z ^ { ( j ) }$ , their mean $\begin{array} { r } { \bar { z } = G ^ { - 1 } \sum _ { j } z ^ { ( j ) } } \end{array}$ , and any pooled vector z,

$$
\begin{array} { c } { { { \displaystyle \frac { 1 } { G } \sum _ { j } \left\| z ^ { ( j ) } - z \right\| ^ { 2 } = \frac { 1 } { G } \sum _ { j } \left\| z ^ { ( j ) } - \bar { z } \right\| ^ { 2 } } } } \\ { { + \left\| \bar { z } - z \right\| ^ { 2 } . } } \end{array}\tag{15}
$$

Proof. Write $z ^ { ( j ) } - z = ( z ^ { ( j ) } - \bar { z } ) + ( \bar { z } - z )$ and expand the square. The cross terms average to zero because $\begin{array} { r } { \sum _ { j } ( z ^ { ( j ) } - \bar { z } ) = 0 } \end{array}$ . The remaining terms give (15).

Thus (14) includes both group scatter and pooling displacement. In a linear observation model the displacement vanishes and the estimate is unbiased for $V _ { P }$ . In the nonlinear implementation the identity makes the pooling displacement measurable. Appendix E reports that contribution for every workload.

## 5.3 Estimating change between checkpoints

For each observed lag n, measure the squared secant

$$
y _ { n } = \left\| z _ { T _ { \mathrm { p } } } - z _ { T _ { \mathrm { p } } - n } \right\| ^ { 2 } / n ^ { 2 } .
$$

The division by $n ^ { 2 }$ puts diferent time separations on a common per-update scale. Consider the local observation model $z _ { s } = P r _ { s } + \epsilon _ { s } ,$ where $r _ { s } = \phi ( \mathbb { E } g _ { s } , \mathbb { E } g _ { s } ^ { \odot 2 } )$ is the population direction at checkpoint s and $\epsilon _ { s }$ is its measurement error. Assume an afine projected direction, $P ( r _ { T _ { \mathrm { p } } } - r _ { T _ { \mathrm { p } } - n } ) = n P d _ { r } $ , and centered independent errors with a common finite expected squared norm. Define $C = 2 \mathbb { E } \left. \epsilon _ { T _ { \mathrm { p } } } \right. ^ { 2 }$ , the combined error energy of two endpoints. Then

$$
\mathbb { E } y _ { n } = D _ { P } + C / n ^ { 2 } .\tag{16}
$$

Indeed, the secant is $P d _ { r } + ( \epsilon _ { T _ { \mathrm { p } } } - \epsilon _ { T _ { \mathrm { p } } - n } ) / n$ . Its cross term has zero expectation, and the two endpoint error energies add. A persistent slope gives the intercept $D _ { P } ;$ endpoint uncertainty gives the inverse-square term.

This relation motivates a two-coeficient least-squares fit:

$$
\left( I _ { \mathrm { f i t } } , C _ { \mathrm { f i t } } \right) = \underset { I , C \in \mathbb { R } } { \arg \operatorname* { m i n } } \sum _ { \ell = 1 } ^ { K } \left( y _ { n _ { \ell } } - I - C / n _ { \ell } ^ { 2 } \right) ^ { 2 } .\tag{17}
$$

Distinct lags give a full-rank design. Under (16), the unconstrained fit is unbiased for $( D _ { P } , C ) $ : write the response as the design times these coeficients plus a mean-zero residual, then apply the linear least-squares map. This argument permits correlated responses, as occurs when secants share the final checkpoint. Least squares estimates the mean lag relation; its use here rests on that relation rather than a Gaussian likelihood assumption.

Enforce the two components’ nonnegative interpretation by setting $\widehat { I } = \operatorname* { m a x } ( 0 , I _ { \mathrm { f i t } } )$ and $\widehat { C } = \operatorname* { m a x } ( 0 , C _ { \mathrm { f i t } } )$ The implemented estimate is the fitted change at the longest observed lag:

$$
\widehat { D } _ { \mathrm { e m p } } = \widehat { I } + \widehat { C } / L ^ { 2 } .\tag{18}
$$

It estimates change at resolution $L ,$ combining persistent change with endpoint uncertainty. The subscript “emp” distinguishes this finite-probe quantity from the population slope $D _ { P }$ . The clipping is a specified post-processing step; the unbiasedness calculation above concerns the unconstrained fit. The next section explains why the rule retains both components.

## 6 FROM MEASURED CHANGE TO A FIXED MEMORY

With finitely many probes, a small population slope and uncertain endpoints can produce comparable measured changes. We use the total change observable at lag L in the memory objective. The first result identifies the resulting allowance relative to the ideal local tracking risk; the rule then follows by the same minimization as in Section 4.

## 6.1 A tracking objective at measurement resolution

Under the observation model of Section 5, define $D _ { L } =$ $\mathbb { E } y _ { L } = D _ { P } + C / L ^ { 2 }$ . This is an expected squared measured slope, rather than a new population derivative. Replacing $D _ { P }$ by $D _ { L }$ charges delay at the resolution of the pilot measurements.

Proposition 3 (Finite-resolution lag allowance). Under Assumption 1 and the centered, independent, equal-variance afine endpoint model in (16), $\mathit { f i x } 1 \leq$ $L \leq t$ . For each $\beta$ and history length t, define

$$
U _ { L , t } ( \beta ) = V _ { P } c _ { \beta , t } + D _ { L } a _ { \beta , t } ^ { 2 } .
$$

This measurement-resolution objective satisfies

$$
U _ { L , t } ( \beta ) = \mathbb { E } \left. P e _ { \beta , t } \right. ^ { 2 } + \frac { C } { L ^ { 2 } } a _ { \beta , t } ^ { 2 } .\tag{19}
$$

Proof. The endpoint calculation gives $D _ { L } \ = \ D _ { P } +$ $C / L ^ { 2 }$ . Substituting this into the definition of $U _ { L , t }$ gives $V _ { P } c _ { \beta , t } + D _ { P } a _ { \beta , t } ^ { 2 } + ( C / L ^ { 2 } ) a _ { \beta , t } ^ { 2 } ,$ . The first two terms are the tracking risk in Theorem 1, measured through $P .$ □

The additional term is the endpoint uncertainty multiplied by squared delay. Longer lags reduce it as $L ^ { - 2 }$ ; more accurate endpoint measurements reduce C. The identity concerns expected coeficients under the stated local model. Applying the stationary and largememory approximations yields $V _ { P } / ( 2 H ) + \dot { D } _ { L } H ^ { 2 }$ . The measured coeficients from Section 5 supply a plug-in version of this objective. Appendix B.2 quantifies the efect of finite coeficient accuracy on its memory decision.

## 6.2 Selecting β and specifying the algorithm

Given the two estimates, minimize

$$
\widehat { f } ( H ) = \frac { \widehat { V } _ { \mathrm { e m p } } } { 2 H } + \widehat { D } _ { \mathrm { e m p } } H ^ { 2 } .\tag{20}
$$

For positive coeficients its unconstrained minimizer is

$$
\widehat { H } _ { \mathrm { e m p } } = \left( \frac { \widehat { V } _ { \mathrm { e m p } } } { 4 \widehat { D } _ { \mathrm { e m p } } } \right) ^ { 1 / 3 } .\tag{21}
$$

The cubic exponent and factor four come from diferentiating (20), exactly as in (11). The measurement design determines the coeficients; minimization determines the memory.

Choose a finite nonempty $\beta$ grid ${ \mathcal { G } } \subset [ 0 , 1 )$ , with endpoints $\beta _ { \mathrm { m i n } }$ and $\beta _ { \mathrm { m a x } }$ . Restrict memory to $[ 1 / ( 1 -$ $\beta _ { \mathrm { m i n } } ) , 1 / ( 1 - \beta _ { \mathrm { m a x } } ) ]$ ], convert it using $\beta = 1 - 1 / H$ , and round to the nearest grid value in $\beta$ distance, breaking ties toward the smaller $\beta .$ . This is the discretization used in the evaluation. If an estimated coeficient is zero, use the minimizing boundary specified in $\mathrm { A p \mathrm { - } }$ pendix C.

Table 1 gives the general procedure. Its measurement hyperparameters are $T _ { \mathrm { p } } , \beta _ { \mathrm { p i l o t } }$ , the lags, m, G, and $P ;$ the candidate range is set by ${ \mathcal { G } } .$ . Section 7 specifies one common instance for all workloads and evaluates the resulting fixed-β trainings.

## 7 EMPIRICAL EVALUATION

## 7.1 Workloads and comparison protocol

We evaluate early memory selection on eleven vision and language workloads: two Llama-style causal models, two GPT-2-style causal models, two ResNets, two ${ \mathrm { V i T s } } ,$ Swin-T, EficientNet-B0, and T5-small. Tasks are image classification, causal language modeling, and span-corruption denoising. Table 2 reports each selected $\beta$ and its gap to the workload-specific grid reference. Appendix D specifies architectures, data, optimizer settings, schedules, and validation.

Table 1: Early selection of a fixed shared memory. Measurements are made during the pilot; the full train ing uses the selected $\beta$ throughout.  
```latex
1 Run $T _ { \mathrm { p } }$ pilot updates with shared $\beta _ { \mathrm { p i l o t } }$ $\mathrm { A t }$
$T _ { \mathrm { p } }$ and $T _ { \mathrm { p } } - n _ { \ell } ,$ , collect m probe gradients
with parameters fixed.
2 Compute pooled and $G$ group moment esti
mates, normalize them using (13), and store
their images under $P .$
3 Estimate variability using (14); fit (17) and
compute change using (18).
4 Apply (21), restrict memory to the candidate
range, convert to $\beta ,$ and round to ${ \mathcal { G } } .$
5 Restart from the pilot’s initialization and use
this fixed shared $\beta$ with the prescribed full
training recipe.
```

The common measurement design is $T _ { \mathrm { p } } = 2 0 0 , \beta _ { \mathrm { p i l o t } } =$ 0.95, $m = 1 6 , G = 4$ , and lags {20, 50, 100}, so $L =$ 100. This gives checkpoints 100, 150, 180, and 200. The fixed map P is a 1,024-coordinate signed-bucket projection with tensor-size balancing; Appendix C defines it exactly. Epsilon is $1 0 ^ { - 8 }$ , as in the training optimizer. The four-group allocation, decision time, and measurement spacing are examined in Appendix E.

The shared-memory grid is

$$
\begin{array} { r l } & { \mathcal { G } = \{ 0 . 6 8 3 7 7 , 0 . 8 2 2 1 7 , 0 . 9 0 0 0 0 , 0 . 9 4 3 7 7 , } \\ & { \quad \quad 0 . 9 6 8 3 8 , 0 . 9 8 2 2 2 , 0 . 9 9 0 0 0 , 0 . 9 9 4 3 8 , } \\ & { \quad \quad 0 . 9 9 6 8 4 , 0 . 9 9 8 2 2 , 0 . 9 9 9 0 0 \} . } \end{array}
$$

It is approximately logarithmic in memory units. Shared β 0.94377 is the grid representative nearest the empirical reference 0.95; $\beta$ 0.96838 is the best benchmark-wide constant by mean gap after examining all eleven fixed values. These serve distinct roles: a near-0.95 reference and a stronger hindsight comparator. Table 3 also includes β 0.9, and Appendix E.3 reports the complete grid. For each workload, candidates use the same prescribed learning-rate recipe, and the selector chooses their shared $\beta .$

For each workload, the pilot and every grid candidate start from the same initialization, generated with seed 1 (the same pretrained checkpoint for T5). This pairing holds the initial weights fixed when comparing memories. Let $L _ { e } ( \beta )$ be workload e’s smallest recorded validation loss, and $L _ { e } ^ { \star } = \operatorname* { m i n } _ { \beta \in \mathcal { G } } L _ { e } ( \beta )$ its finite-grid reference. Define

$$
\mathrm { g a p } _ { e } ( \beta ) = 1 0 0 \cdot \frac { L _ { e } ( \beta ) - L _ { e } ^ { \star } } { L _ { e } ^ { \star } } .\tag{22}
$$

Early gradient measurements determine $\beta ;$ candidate validation losses then evaluate that decision. The finite-grid reference gives the evaluation scale. All workloads use the same 16-probe budget, checkpoints, objective, and rounding. Reported metrics are equalworkload mean, maximum, and $\mathrm { C V a R _ { 2 5 } }$ , the average of the worst $\lceil 1 1 / 4 \rceil = 3$ gaps. These are descriptive statistics from a retrospective fixed $- \beta$ evaluation. Appendix E.13 reports common-seed comparisons (Table 17); Appendix E.8 analyzes decision timing, Appendix E.9 measurement spacing, and Appendix E.12 validation summaries.

## 7.2 Improving the fixed-memory compromise

The goal is a useful task-specific memory: reduce large calibration errors while keeping well-served tasks close to their grid reference. Table 2 shows the oracle $\beta ,$ the selected $\beta ,$ and the resulting gap for every task. The rule selects 0.9 for the four causal language-model tasks, 0.96838 for EficientNet, both ResNets, and T5, and 0.94377 for Swin and both ViTs. Thus one common calculation produces diferent memory horizons from diferent measured dynamics.

Table 3 makes the fixed-memory compromise visible. With $\beta \_ 0 . 9$ the largest gap occurs on ViT/TinyImageNet; with 0.94377 it occurs on ResNet/Food; with 0.96838 it occurs on Swin/Caltech. Changing the constant changes which task carries the largest calibration cost. The measured selector reduces mean gap by 40.7% against the near-0.95 reference and by 32.3% against the best universal constant. Its maximum gap is 39.0% below the smallest maximum attained by any fixed grid memory, and its CVaR is also smaller than every fixed value. Workload-specific calibration therefore improves both the average compromise and its largest residual error on this suite.

The tasks contributing the gains change with the comparator, as expected when correcting a universal compromise. Against 0.94377, ResNet losses improve by 1.3895% on Food-101 and 2.2517% on ImageNet100, while Swin and both ViTs retain the reference configuration. Against 0.96838, the ResNet choices coincide and the gains move to Swin and ViT/CIFAR-100, whose losses improve by 2.1395% and 0.4020%. The four causal tasks have gaps below 0.259%. The result is useful calibration across tasks rather than another universal constant. Appendix E.4 provides every signed comparison, including the smaller increases ofset by these corrections.

## 7.3 Stability of calibration

We examine coeficient sensitivity, grid decisions, and validation gaps separately. Proposition 1 quantifies coeficient sensitivity under the local score. A 10% perturbation of each observed variability-to-change ratio preserves every selected $\beta .$ . The stationary root, leading-grid score, stationary-grid score, and exact finite-history grid score at t = 200 also recover the same eleven choices (Appendix E).

Table 2: Early memory choices and their distance to the workload-specific grid reference. Pilot and candidate trainings use the same initialization (seed 1). Oracle $\beta$ minimizes the smallest recorded validation loss over the eleven grid candidates and is used only for evaluation. The selector uses early gradient measurements. Gaps are percentages.
<table><tr><td>Network</td><td>Dataset</td><td>Oracle  $\beta$ </td><td>Selected  $\beta$ </td><td>Relative gap (%)</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>0.82217</td><td>0.96838</td><td>1.0124</td></tr><tr><td>Llama60M</td><td>C4</td><td>0.94377</td><td>0.90000</td><td>0.2582</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>0.94377</td><td>0.90000</td><td>0.1175</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>0.94377</td><td>0.90000</td><td>0.1157</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td>0.94377</td><td>0.90000</td><td>0.1613</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>0.99684</td><td>0.96838</td><td>1.0613</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>0.96838</td><td>0.96838</td><td>0.0000</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td>0.94377</td><td>0.94377</td><td>0.0000</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>0.98222</td><td>0.96838</td><td>0.0039</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td>0.94377</td><td>0.94377</td><td>0.0000</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>0.99000</td><td>0.94377</td><td>1.3339</td></tr></table>

Table 3: Workload-specific calibration versus universal fixed memories. Gaps are percentages. Best fixed minimizes mean gap over all eleven fixed grid choices using their validation outcomes. Lower is better.
<table><tr><td></td><td>Mean</td><td>Maximum</td><td> $\mathrm { C V a R _ { 2 5 } }$ </td></tr><tr><td>Fixed 0.90000</td><td>1.4292</td><td>3.8770</td><td>3.3543</td></tr><tr><td>Fixed 0.94377</td><td>0.6236</td><td>2.4853</td><td>2.0409</td></tr><tr><td>Best fixed 0.96838</td><td>0.5461</td><td>2.1862</td><td>1.4304</td></tr><tr><td>Tracking rule</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr></table>

Across 351 decision updates from 150 to 500 on the same pilots, mean gap improves over 0.94377 at 339 updates (96.6%) and CVaR improves at all 351. Mean gap improves over the best global fixed value 0.96838 at 288 updates (82.1%). Appendix E.8 gives the dense audit. Using the final loss or the mean of the last three losses retains mean-gap reductions of 20.5% and 11.0% against their respective best fixed memories (Appendix E.12). These are sensitivity checks under matched initialization; Appendix E.13 separately examines reuse across initializations.

## 7.4 Cost and required information

A 200-update pilot supplies four measured checkpoints, requiring 64 additional efective-batch gradients. Moment aggregation and fixed projection cost $O ( d )$ operations and $O ( d )$ additional storage for a fixed probe budget. Each checkpoint stores one pooled and four group directions of 1,024 coordinates; the scalar fit and memory conversion have constant cost.

After restarting, a full training of $T$ updates has gradient-equivalent surplus $1 0 0 \cdot ( T _ { \mathrm { p } } + m ( K + 1 ) ) / T \%$ Our design gives $1 0 0 \cdot ( 2 0 0 + 6 4 ) / T \%$ , or 2.64% for $T = 1 0 , 0 0 0 .$ . This counts one probe as one training gradient evaluation; loading, transfers, sketches, and restarting add wall-clock costs. Appendix C describes the dense measurement record and compact collection.

## 8 DISCUSSION AND CONCLUSION

Early gradient measurements choose one shared $\beta$ for full training. A joint-moment model explains its noise–lag objective; grouped probes and a temporal fit make the coeficients measurable. On the matched benchmark, the rule reduces mean, maximum, and tail gap compared with every fixed grid memory. Diferent tasks supply the gains against diferent constants, showing how measured memory selection improves the compromise made by a universal default.

## ACKNOWLEDGEMENTS

This research was funded by the projects PID2023- 146569NB-C21 and PID2023-146569NB-C22 supported by MICIU/AEI/10.13039/501100011033 and ERDF/UE. Alberto Fernández-Hernández was supported by the predoctoral grant PREP2023-001826 supported by MICIU/AEI/10.13039/501100011033 and ESF+. Cristian Pérez-Corral received support from the Conselleria de Educación, Cultura, Universidades $y$ Empleo (reference CIACIF/2024/412) through the European Social Fund Plus 2021–2027 (FSE+) program of the Comunitat Valenciana.

Jose I. Mestre was supported by the predoctoral grant ACIF/2021/281 of the Generalitat Valenciana. Manuel F. Dolz was supported by grant CNS2025- 165098 funded by MICIU/AEI/10.13039/501100011033 and by the Plan Gen–T grant CIDEXG/2022/013 of the Generalitat Valenciana.

## AI USE STATEMENT

The authors take full responsibility for the scientific content and final manuscript, including AI-assisted material. Generative AI supported theoretical devel opment, data analysis, literature search, writing, and internal review.

## References

Ahn, K., Zhang, Z., Kook, Y., and Dai, Y. (2024). Understanding Adam optimizer via online learning of updates: Adam is FTRL in disguise. In International Conference on Machine Learning.

Ali, M. and Bhatt, R. (2026). Kalman-Adam: Optimal bayesian moment estimation for memory-eficient and generalizable deep learning. Knowledge-Based Systems, 342:115907.

Balles, L. and Hennig, P. (2018). Dissecting adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning.

Bossard, L., Guillaumin, M., and Van Gool, L. (2014). Food-101: Mining discriminative components with random forests. In European Conference on Computer Vision.

Chandra, K., Xie, A., Ragan-Kelley, J., and Meijer, E. (2022). Gradient descent: The ultimate optimizer. In Advances in Neural Information Processing Systems, volume 35.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations.

Fernández-Hernández, A., Pérez-Corral, C., Mestre, J. I., Dolz, M. F., and Quintana-Ortí, E. S. (2026). Refresh-scaling the memory of balanced Adam. arXiv:2605.10119.

Gokaslan, A. and Cohen, V. (2019). Openwebtext corpus. Open-source reproduction of the WebText corpus.

Grifin, G., Holub, A., and Perona, P. (2007). Caltech-256 object category dataset. Technical report, California Institute of Technology.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In International Conference on Learning Representations.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. (2013). Collecting a large-scale dataset of fine-grained cars. In ICCV Workshop on Fine-Grained Visual Categorization.

Krizhevsky, A. (2009). Learning multiple layers of fea tures from tiny images. Technical report, University of Toronto.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations.

Malladi, S., Lyu, K., Panigrahi, A., and Arora, S. (2022). On the sdes and scaling rules for adaptive gradient algorithms. In Advances in Neural Information Processing Systems.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2017). Pointer sentinel mixture models. In International Conference on Learning Representations.

Nguyen, Q. M. (2026). How to set $\beta _ { 1 } , \beta _ { 2 }$ in Adam: An online learning perspective. In Algorithmic Learning Theory, volume 313, pages 1–16. PMLR.

Orvieto, A. and Gower, R. M. (2025). In search of adam’s secret sauce. In Advances in Neural Information Processing Systems.

Ozkara, K., Karakus, C., Raman, P., Hong, M., Sabach, S., Kveton, B., and Cevher, V. (2024). MADA: Meta-adaptive optimizers through hypergradient descent. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 38983–39008. PMLR.

Rafel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J., Hestness, J., and Dey, N. (2023). Slimpajama: A 627b token cleaned and deduplicated version of red pajama. Dataset release.

Tan, M. and Le, Q. V. (2019). Eficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning.

Topollai, K. and Choromanska, A. E. (2026). Adaptive memory momentum via a model-based framework for deep learning optimization. In Artificial Intelligence and Statistics, volume 300, pages 4060–4068. PMLR.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. (2023). Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971.

Zhang, J. and Mitliagkas, I. (2019). YellowFin and the art of momentum tuning. In Conference on Machine Learning and Systems.

Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., and Fidler, S. (2015). Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE International Conference on Computer Vision.

## REPRODUCIBILITY CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes: Sections 3–6 and Appendix D.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. Yes: Section 7.4.

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. Yes.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. Yes: the local assumption and conditions accompanying results.

(b) Complete proofs of all theoretical results. Yes: the main text and Appendices A–B.

(c) Clear explanations of any assumptions. Yes: alongside the derivations and approximation scope.

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Yes for the selection evaluation. Appendices C–D specify the measurement procedures, training recipes, and evaluation calculations.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). Yes for the documented experimental protocol: Appendix D specifies architectures, activations, optimizer settings, batches, schedules, splits, seed matching, and evaluation. It sep arately identifies archive-specific corpus manifests and environment revisions needed for identical raw-stream replay.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Yes: descriptive cross-workload metrics, seed matching, and common-seed training outcomes are specified.

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). Yes: a cluster with eight NVIDIA A100-SXM4 80GB GPUs; CUDA and pilot bfloat16 execution are described in Appendix D.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. Yes: model and dataset sources are cited.

(b) The license information of the assets, if applicable. Yes: Appendix D.8 records software/checkpoint licenses and datasetprovider access sources, identifying the documented reuse provenance.

(c) New assets either in the supplemental material or as a URL, if applicable. Yes: measured sketches and validation-loss summaries.

(d) Information about consent from data providers/curators. N/A: the study uses existing benchmark assets under their providers’ access terms.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Yes: Appendix D discusses inherited corpus content and the scope of the supplied data.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. N/A: existing benchmark evaluation.

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. N/A: existing benchmark evaluation.

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. N/A: existing benchmark evaluation.

# Early Memory Selection for Balanced Adam: Supplementary materials

## A DETAILED DERIVATIONS

This appendix derives the averaging formulas step by step. Each calculation concerns the local model stated in the main text and expresses the selection mechanism through explicit statistical quantities.

## A.1 Obtaining the balanced first-order error

The first derivation shows why shared weights let the two moment channels become one direction innovation and one delay term. Bias correction normalizes the exponential weights to sum to one. The first-moment average in Assumption 1 difers from its current mean by

$$
\sum _ { k } w _ { \beta , t } ( k ) ( - k \dot { \mu } + \xi _ { k } ) = - a _ { \beta , t } \dot { \mu } + \sum _ { k } w _ { \beta , t } ( k ) \xi _ { k } .
$$

The second-moment average similarly difers by $\begin{array} { r } { - a _ { \beta , t } \dot { q } + \sum _ { k } w _ { \beta , t } ( k ) \chi _ { k } } \end{array}$ . Applying (4) gives

$$
e _ { \beta , t } = - a _ { \beta , t } ( A \dot { \mu } + B \dot { q } ) + \sum _ { k } w _ { \beta , t } ( k ) ( A \xi _ { k } + B \chi _ { k } ) .
$$

This is exact for the first-order model. The innovation sum is centered; independence across ages yields variance $c _ { \beta , t } V _ { \mathrm { p o p } }$ . Its squared mean is $a _ { \beta , t } ^ { 2 } D _ { \mathrm { p o p } }$ , proving (7). The assumptions pair independent ages with dependent moment channels at each age.

## A.2 Finite-history weight formulas

The retained noise and the delay depend on two sums of weights. Computing them explicitly makes the finitehistory risk a scalar function of $\beta$ and history length. For $0 < \beta < 1$ , geometric summation gives

$$
c _ { \beta , t } = { \frac { ( 1 - \beta ) ^ { 2 } } { ( 1 - \beta ^ { t } ) ^ { 2 } } } { \frac { 1 - \beta ^ { 2 t } } { 1 - \beta ^ { 2 } } } = { \frac { 1 - \beta } { 1 + \beta } } { \frac { 1 + \beta ^ { t } } { 1 - \beta ^ { t } } } .
$$

Moreover, $\begin{array} { r } { \sum _ { k = 0 } ^ { t - 1 } k \beta ^ { k } = \beta \frac { d } { d \beta } [ ( 1 - \beta ^ { t } ) / ( 1 - \beta ) ] } \end{array}$ . Multiplication by the weight normalizer yields $a _ { \beta , t } = \beta / ( 1 - \beta ) -$ $t \beta ^ { t } / ( 1 - \beta ^ { t } )$ . At β zero only the current observation has weight. At $t = 1 , a _ { \beta , 1 } = 0$ and $c _ { \beta , 1 } = 1$ for every $\beta .$ Thus the first Adam update is independent of both memories with zero initial moments and standard bias correction.

## A.3 Stationary and leading minimizers

There are two simplifications: removing finite-history correction when $\beta ^ { t }$ is small, then expanding for large H. This subsection derives each separately and states the boundary cases. For fixed $\beta < 1 , \beta ^ { t } \to 0$ , so $a _ { \beta , t } \to H - 1$ and $c _ { \beta , t }  ( 2 H - 1 ) ^ { - 1 }$ . For positive coeficients,

$$
f ^ { \prime } ( H ) = - \frac { 2 V _ { \mathrm { p o p } } } { ( 2 H - 1 ) ^ { 2 } } + 2 D _ { \mathrm { p o p } } ( H - 1 ) , \qquad f ^ { \prime \prime } ( H ) = \frac { 8 V _ { \mathrm { p o p } } } { ( 2 H - 1 ) ^ { 3 } } + 2 D _ { \mathrm { p o p } } > 0 .
$$

The unique minimum solves (10). Replacing $2 H - 1$ by 2H and $H - 1$ by H gives the large-memory leading expression, whose stationary point is $H ^ { 3 } = V _ { \mathrm { p o p } } / ( 4 D _ { \mathrm { p o p } } )$ . If that leading minimum is below one, its constrained minimum is at one. If $D _ { \mathrm { p o p } } = 0 < V _ { \mathrm { p o p } }$ , longer memory reduces stationary-model risk; if $V _ { \mathrm { p o p } } = 0 < D _ { \mathrm { p o p } }$ $H = 1$ is optimal. If both are zero, every memory has zero model risk.

## A.4 Sensitivity

Comparing an estimated memory with the population-model minimizer shows how coeficient errors propagate through the cube root. The logarithmic ratio of memory estimates is

$$
\log \frac { \widetilde { H } } { H _ { \star } } = \frac { 1 } { 3 } \left( \log \frac { \widetilde { V } } { V _ { \mathrm { p o p } } } - \log \frac { \widetilde { D } } { D _ { \mathrm { p o p } } } \right) .
$$

The triangle inequality proves the bound. At the optimum the variance term is twice the lag term, so division of risks gives $( 2 / z + z ^ { 2 } ) / 3$ . With $z = 2 ^ { 1 / 3 }$ this is approximately 1.0583. For the worked coeficients 0.18 and $4 \times 1 0 ^ { - 6 } , H _ { \star } = 2 2 . 4 0 7$ and $\beta _ { \star } = 0 . 9 5 5 3 7$ . The memory value follows directly from the statistical coeficients.

## A.5 The two-memory expression

The two-memory expression explains why sharing weights gives a particularly simple selection problem: the two moment channels combine into one innovation and one lag coeficient. Retain weights $w _ { 1 } = w _ { \beta _ { 1 } , t } , w _ { 2 } = w _ { \beta _ { 2 } , t } .$ Let $\begin{array} { r } { a _ { i } = \sum _ { k } k w _ { i } ( k ) , c _ { i } = \sum _ { k } w _ { i } ( k ) ^ { 2 } } \end{array}$ , and $\begin{array} { r } { c _ { 1 2 } = \sum _ { k } w _ { 1 } ( k ) w _ { 2 } ( k ) } \end{array}$ . Their linearized direction error is $e _ { \beta _ { 1 } , \beta _ { 2 } , t } =$ $\begin{array} { r } { \sum _ { k } [ w _ { 1 } ( k ) A \xi _ { k } + w _ { 2 } ( k ) B \chi _ { k } ] - a _ { 1 } A \dot { \mu } - a _ { 2 } B \dot { q } } \end{array}$ . Under the same assumptions,

$$
\begin{array} { r l } & { \mathbb { E } \left\| e _ { \beta _ { 1 } , \beta _ { 2 } , t } \right\| ^ { 2 } = c _ { 1 } \mathbb { E } \left\| A \xi _ { k } \right\| ^ { 2 } + c _ { 2 } \mathbb { E } \left\| B \chi _ { k } \right\| ^ { 2 } + 2 c _ { 1 2 } \mathbb { E } \langle A \xi _ { k } , B \chi _ { k } \rangle } \\ & { \qquad + \left\| a _ { 1 } A \dot { \mu } + a _ { 2 } B \dot { q } \right\| ^ { 2 } . } \end{array}
$$

Centering removes lag–innovation cross terms; independence removes cross-age terms; expansion retains sameage covariance. Equal memories recover (7), combining the channels into two scalar coeficients. This reduction provides an analytical basis for selecting a memory from the shared-memory family.

## B SCOPE AND APPROXIMATION

The tracking identity is exact under its first-order model. This section explains the additional quantities required to connect that identity to nonlinear updates and a smooth loss.

## B.1 Residual terms

The Taylor approximation and violations of the local assumptions can be collected into a direction residual. Its size controls how much the model’s risk can difer from the actual direction-error risk. Write the actual direction error as $e _ { \beta , t } + R _ { \beta , t }$ , where $e _ { \beta , t }$ is the theorem’s linearized error and $R _ { \beta , t }$ is their diference. Let $\rho$ be any nonnegative bound on its root-mean-square size: $\mathbb { E } \left. R _ { \beta , t } \right. ^ { 2 } \leq \rho ^ { 2 }$ . Expanding the actual squared norm gives

$$
\begin{array} { r } { \mathbb { E } \left\| e _ { \beta , t } + R _ { \beta , t } \right\| ^ { 2 } - \mathbb { E } \left\| e _ { \beta , t } \right\| ^ { 2 } = 2 \mathbb { E } \langle e _ { \beta , t } , R _ { \beta , t } \rangle + \mathbb { E } \left\| R _ { \beta , t } \right\| ^ { 2 } . } \end{array}
$$

The absolute inner product is bounded by the product of root-mean-square norms, by Cauchy–Schwarz. Therefore

$$
\left| \mathbb { E } \left\| e _ { \beta , t } + R _ { \beta , t } \right\| ^ { 2 } - \mathbb { E } \left\| e _ { \beta , t } \right\| ^ { 2 } \right| \leq 2 \rho \sqrt { \mathbb { E } \left\| e _ { \beta , t } \right\| ^ { 2 } } + \rho ^ { 2 } .
$$

The residual represents nonlinear normalization and deviations from the local statistical model. The inequality quantifies how a stated residual bound transfers the model risk to the actual direction-error risk. Applying it along an adaptive trajectory requires a bound on the residual at the corresponding updates. The workload measurements audit the practical proxies and selected configurations.

## B.2 Comparing memories at finite measurement resolution

Proposition 3 quantifies the allowance introduced by estimating a slope from noisy endpoints. This subsection measures its efect on the memory decision under the same local model. Fix a finite candidate set $B \subset [ 0 , 1 )$ and the update t. Define $R _ { t } ( \beta ) = \mathbb { E } \left\| P e _ { \beta , t } \right\| ^ { 2 }$ , and let $\beta _ { R }$ minimize $R _ { t }$ and $\beta _ { U }$ minimize $U _ { L , t }$ on B. The coeficients in both functions are the population and expected-measurement coeficients specified in the proposition. Then

$$
0 \leq R _ { t } ( \beta _ { U } ) - R _ { t } ( \beta _ { R } ) \leq \frac { C } { L ^ { 2 } } \big ( a _ { \beta _ { R } , t } ^ { 2 } - a _ { \beta _ { U } , t } ^ { 2 } \big ) \leq \frac { C } { L ^ { 2 } } a _ { \beta _ { R } , t } ^ { 2 } .
$$

To prove this, start with $U _ { L , t } ( \beta _ { U } ) \leq U _ { L , t } ( \beta _ { R } )$ and substitute (19) on both sides. Rearranging gives the middle inequality; optimality of $\beta _ { R }$ gives the lower bound. Since mean ages are nonnegative, discarding the subtracted age term gives the last bound. The result compares two memory decisions through their local tracking risk and makes the cost of measurement resolution explicit.

Finite coeficient accuracy can be incorporated in the same comparison. Suppose a measured score $\widehat { U } _ { L , t }$ satisfies $| \widehat { U } _ { L , t } ( \beta ) - U _ { L , t } ( \beta ) | \le \delta$ simultaneously for every $\beta \in B .$ and let $\widehat { \beta }$ minimize this measured score. Inserting the two uniform error bounds into $\widehat { U } _ { L , t } ( \widehat { \beta } ) \leq \widehat { U } _ { L , t } ( \beta _ { U } )$ yields $U _ { L , t } ( \widehat { \beta } ) \leq U _ { L , t } ( \beta _ { U } ) + 2 \delta$ . Hence

$$
R _ { t } ( \widehat { \beta } ) - R _ { t } ( \beta _ { R } ) \leq \frac { C } { L ^ { 2 } } a _ { \beta _ { R } , t } ^ { 2 } + 2 \delta .
$$

Here δ is an assumed uniform coeficient-score accuracy, and the bound is conditional on it. For example, bounds $| \widehat { V } _ { \mathrm { e m p } } - V _ { P } | \leq e _ { V }$ and $| \widehat { D } _ { \mathrm { e m p } } - D _ { L } | \leq e _ { D }$ give $\begin{array} { r } { \delta = \operatorname* { m a x } _ { \beta \in B } ( e _ { V } c _ { \beta , t } + e _ { D } a _ { \beta , t } ^ { 2 } ) } \end{array}$ . For a cubic-and-rounded candidate $b ,$ define its measured optimization discrepancy as $\begin{array} { r } { \widehat { \eta } = \widehat { U } _ { L , t } ( b ) - \operatorname* { m i n } _ { \beta \in \mathcal { B } } \widehat { U } _ { L , t } ( \beta ) } \end{array}$ . The same argument gives the bound $R _ { t } ( b ) - R _ { t } ( \beta _ { R } ) \leq ( C / L ^ { 2 } ) a _ { \beta _ { R } , t } ^ { 2 } + 2 \delta + \widehat { \eta } \mathrm { ~ . ~ }$ insert this discrepancy before applying the two uniform error bounds. Thus coeficient accuracy and optimization accuracy have separate roles. The exact finite-history audit finds $\widehat { \eta } = 0$ for all eleven observed choices at $t = 2 0 0$ . Timing and decision-margin audits describe their additional empirical sensitivity.

## B.3 A local connection to the loss

Tracking error matters because it perturbs a reference update. The following one-step inequality separates a systematic directional shift from the curvature cost of fluctuations. If F has an L-Lipschitz gradient, $\theta , r$ are fixed, and $u = r + e$ has finite second moment, the smoothness inequality at $\theta - \eta r$ gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } F ( \theta - \eta u ) \leq F ( \theta - \eta r ) - \eta \langle \nabla F ( \theta - \eta r ) , \mathbb { E } e \rangle + \frac { L \eta ^ { 2 } } { 2 } \mathbb { E } \left. e \right. ^ { 2 } } \\ { \displaystyle \qquad \leq F ( \theta - \eta r ) + \eta \left. \nabla F ( \theta - \eta r ) \right. \left. \mathbb { E } e \right. + \frac { L \eta ^ { 2 } } { 2 } \mathbb { E } \left. e \right. ^ { 2 } . } \end{array}
$$

The term involving Ee measures a systematic shift from the reference direction; the term involving E $\| e \| ^ { 2 }$ is the curvature penalty. This one-step inequality connects direction-error moments to the loss under the stated smoothness assumption. Its parameter-space norm makes a norm-transfer bound necessary when the error is measured through a sketch. The empirical comparisons evaluate the selected configurations through their fulltraining validation losses.

## B.4 Numerical checks of the identities

These checks test the algebra and the specified innovation model, independently of the saved neural-network outcomes. With NumPy seed 20261005, 1,000 random-vector cases verify (15) with maximum absolute residual $7 . 1 1 \times 1 0 ^ { - 1 5 }$ , and 1,000 positive-coeficient cases verify (12) with residual $1 . 3 3 \times 1 0 ^ { - 1 5 }$ . An afine-model check uses 30,000 independent histories of length 128, β 0.95, innovation standard deviation 0.3, and slope 0.002:

$$
e = \sum _ { k = 0 } ^ { 1 2 7 } w _ { 0 . 9 5 , 1 2 8 } ( k ) \xi _ { k } - 0 . 0 0 2 \sum _ { k = 0 } ^ { 1 2 7 } k w _ { 0 . 9 5 , 1 2 8 } ( k ) , \qquad \xi _ { k } \sim \mathcal { N } ( 0 , 0 . 3 ^ { 2 } ) .
$$

Its measured mean squared error is 0.00373524, versus exact 0.00373090; the Monte Carlo standard error is $2 . 8 4 \times 1 0 ^ { - 5 }$ . These checks test the statistical formulas, separately from network training.

## C MEASUREMENT REPRODUCIBILITY

To reproduce a selection, one needs four direction measurements and the fixed geometry used to compare them. The following specification covers gradient processing, state preservation, projection, and scalar calculation in that order.

## C.1 Gradient samples and probe state

The pilot seed is 1 and shared β 0.95. Models remain in training mode during probes, including dropout and augmentation. For accumulation count a, each efective-batch gradient averages a physical-batch losses. Norm clipping follows accumulation. Coupled Adam decay is added after clipping; AdamW’s decoupled decay belongs to the parameter update. Sixteen fresh efective batches are drawn per required checkpoint with parameters held fixed. Partition their ordered samples into four groups of four; each group computes its own first and second moments before normalization. The pooled direction uses all sixteen samples. Thus the dispersion calculation compares four group directions, each formed from four samples, with one pooled direction.

Python, NumPy, CPU/CUDA random states, module training flags, and mutable bufers are restored after probing, preserving the pilot’s training state. The dedicated probe iterator and generator advance through the sampled examples. Gradients are cleared afterward. Dense archives contain checkpoints through update 500 and additional probe budgets; the rule replays checkpoints 100, 150, 180, and 200 with sixteen samples each. Matching the original probe examples in a sparse collection requires reproducing the dense stream’s advancement. The supplied measured sketches provide exact reconstruction of the reported decisions. The 64-gradient operation count describes a prospective compact four-checkpoint deployment; the experiments reconstruct its decision from the dense archive’s observations.

Pilot vision loaders use zero worker processes and a separate probe generator seeded by training seed plus 10000019. GPT pilot training and probe generators use seed plus 101 and seed plus 10000101, respectively. Llama’s training shufle uses seed 42 and the probe shufle uses seed plus 10000003.

## C.2 The exact fixed sketch

The sketch makes measurements storable while giving each parameter tensor comparable weight. It defines a tensor-balanced measurement norm. For example, multiplying one tensor’s dimension by four reduces each of its coordinate weights by two. The same fixed linear map applies to the theorem’s innovations and slope, giving the tracking identity in this measurement geometry. Transferring a sketch-space error bound to the original parameter-space norm requires an accompanying norm-transfer bound. Use dimension 1,024 and seed 12,345. For trainable tensor p with $n _ { p }$ entries, assign coordinate i to bucket $b _ { p , i }$ and sign $s _ { p , i } \in \{ - 1 , 1 \}$ . The projection is

$$
( P r ) _ { b } = \sum _ { p } { \frac { 1 } { \sqrt { n _ { p } } } } \sum _ { i : b _ { p , i } = b } s _ { p , i } r _ { p , i } .
$$

To obtain each tensor’s bucket/sign generator seed, join the decimal projection seed, parameter name, and sufix idx or sgn with |. Hash its UTF-8 bytes with BLAKE2b, eight-byte digest; interpret little-endian unsigned and retain the low 31 bits. Seed independent Torch CPU generators for bucket and sign draws. Buckets are uniform integers 0–1023; signs are generated by uniform integers 0 or 1 mapped to −1 or +1. Flatten tensors and accumulate in float64. Names are the model’s named trainable parameters. Global sketches sum tensor sketches and supply the selection inputs. The archive also contains layerwise 32-coordinate diagnostic sketches.

## C.3 Statistical basis of the coeficient estimates

The main estimators use two standard operations: variance estimation from independent groups and linear regression on the lag responses. This subsection supplies the calculations and states precisely where their unbiasedness applies.

Grouped variability. At a fixed checkpoint, let $Z _ { i } ~ = ~ P \zeta _ { i }$ be independent, centered linearized direction innovations with $\mathbb { E } \left. \dot { Z _ { i } } \right. ^ { 2 } = V _ { P }$ . Each of G disjoint groups contains m/G observations. Its innovation average is $\begin{array} { r } { Z ^ { ( j ) } = ( G / m ) \sum _ { i \mathrm { ~ i n ~ g r o u p ~ } j } Z _ { i } } \end{array}$ , and their common mean is $\bar { Z } = G ^ { - 1 } \sum _ { i } ^ { ^ { \prime } } Z ^ { ( j ) }$ . Independence and centering give

$$
\mathbb { E } \left\| Z ^ { ( j ) } \right\| ^ { 2 } = \frac { G } { m } V _ { P } , \qquad \mathbb { E } \left\| \bar { Z } \right\| ^ { 2 } = \frac { 1 } { m } V _ { P } .
$$

The identity $G ^ { - 1 } \sum _ { j } \left\| Z ^ { ( j ) } - \bar { Z } \right\| ^ { 2 } = G ^ { - 1 } \sum _ { j } \left\| Z ^ { ( j ) } \right\| ^ { 2 } - \left\| \bar { Z } \right\| ^ { 2 }$ therefore yields

$$
\mathbb { E } \left[ \frac { 1 } { G } \sum _ { j } \left. Z ^ { ( j ) } - \bar { Z } \right. ^ { 2 } \right] = \frac { G - 1 } { m } V _ { P } .
$$

In the linearized model, subtracting the common population direction from pooled and group estimates leaves these innovation averages. Equation (14) thus estimates $V _ { P }$ without bias under this model. For nonlinear normalized moment estimates, Proposition 2 still holds exactly, but the group vectors also contain higher-order terms. Its pooling displacement describes one observable contribution; the residual analysis in Appendix B addresses the additional approximation error.

Temporal regression. Condition on the pilot parameter trajectory in the observation model of Section 5. The lags and projected population directions are then fixed. Write $\boldsymbol { Y } = ( y _ { n _ { 1 } } , \dots , y _ { n _ { K } } ) ^ { \intercal }$ and let X be the K-by-two matrix whose row ℓ is $( 1 , n _ { \ell } ^ { - 2 } )$ . Equation (16) gives $Y = X ( D _ { P } , C ) ^ { \top } + \nu ;$ , where ν is the regression residual vector and $\mathbb { E } \nu = 0$ . Distinct lags make the two columns independent. The ordinary least-squares coeficients satisfy

$$
\binom { I _ { \mathrm { f i t } } } { C _ { \mathrm { f i t } } } = ( X ^ { \top } X ) ^ { - 1 } X ^ { \top } Y = \binom { D _ { P } } { C } + ( X ^ { \top } X ) ^ { - 1 } X ^ { \top } \nu .
$$

Taking expectations proves unbiasedness of the unconstrained coeficients. The errors in $Y$ may be correlated and have diferent variances; these features afect uncertainty and eficiency, while the expectation calculation only uses their zero means. The implementation’s least-squares choice matches the afine mean relation and requires no distributional likelihood assumption. Componentwise clipping subsequently enforces the nonnegative score coeficients and can alter their sampling expectations, particularly near zero.

## C.4 Coeficient calculation and conversion

The calculation uses fresh probe moments: pooled directions for checkpoint changes and individual group directions for fluctuations at a fixed checkpoint. For $\begin{array} { r } { m = 1 6 , G = 4 , \widehat { V } _ { \mathrm { e m p } } = ( 4 / 3 ) \sum _ { j = 1 } ^ { 4 } \left\| z _ { 2 0 0 } ^ { ( j ) } - z _ { 2 0 0 } \right\| ^ { 2 } } \end{array}$ . Each group normalizes moments from four efective-batch gradients; the pooled direction uses moments from all sixteen. The archived collector also uses a square-root floor $\tau = 1 0 ^ { - 3 0 }$ in auxiliary derivative calculations. Its guarded direction is numerically identical to (13) in the collector’s float64 arithmetic with $\varepsilon = 1 0 ^ { - 8 }$ : the floor changes a denominator only when $\sqrt { \widehat { q } } < \tau$ , and both sums with epsilon then round to epsilon. For exact arithmetic the coordinatewise diference is at most $\tau ^ { 2 } / ( 4 \varepsilon ^ { 2 } ) = 2 . 5 \times 1 0 ^ { - 4 5 }$ , since $| { \widehat { \mu } } | \leq { \sqrt { \widehat { q } } } .$ Thus the direction estimator can be written in the standard Adam form without changing the archived selection inputs.

Build a three-by-two least-squares design with rows $( 1 , n ^ { - 2 } )$ for $n = 2 0 , 5 0 , 1 0 0$ , and response $y _ { n } .$ . Solve for $( I , C )$ , apply componentwise clipping $\widehat { I } = \operatorname* { m a x } ( 0 , I )$ and $\widehat { C } = \operatorname* { m a x } ( 0 , C )$ , and evaluate at lag 100. The algorithm therefore consists of one ordinary least-squares fit followed by clipping. Raw checkpoint sketches supply the lag observations directly. The resulting $\widehat { D } _ { \mathrm { e m p } }$ is the fitted nonnegative change on the stated observation scale, including its finite-lag contribution.

For positive coeficients, calculate the cubic memory in (21). The admissible interval is $H _ { \mathrm { { m i n } } } = 1 / ( 1 - 0 . 6 8 3 7 7 )$ through $H _ { \mathrm { m a x } } = 1 / ( 1 - 0 . 9 9 9 )$ ; restrict memory to this interval and convert it to $\beta$ . Equivalently, this minimizes the operational leading score over the continuous interval before nearest $- \beta$ rounding. If $\widehat { D } _ { \mathrm { e m p } } = 0 < \widehat { V } _ { \mathrm { e m p } }$ , the score decreases with memory, so use $H _ { \mathrm { m a x } }$ . If $\widehat { V } _ { \mathrm { e m p } } = 0 < \widehat { D } _ { \mathrm { e m p } }$ , use $H _ { \mathrm { m i n } }$ . If both vanish, all scores tie and the smaller $\beta$ resolves the tie. These cases follow from minimizing the stated objective. All main measurements have positive coeficients. Rounding is in $\beta$ distance, with ties toward the smaller grid value.

Appendix C.3 derives the variance correction; Proposition 2 separates group scatter and nonlinear pooling displacement.

## D TRAINING REPRODUCIBILITY

The selected configurations are evaluated using fixed-memory training runs. This appendix specifies the common protocol, then the architecture, data processing, and validation for each family. Ofline reconstruction from the supplied measurements is described separately from regenerating the original training streams.

## D.1 Shared settings and schedules

All sweep runs use equal $\beta$ values and epsilon $1 0 ^ { - 8 }$ outside the root. For each workload, candidates share batch size, architecture, data, learning-rate recipe, and evaluation schedule. The main comparison uses seed 1. Initialization seeds $\mathrm { P y }$ thon, NumPy, Torch CPU, and all CUDA devices; cuDNN uses deterministic mode

with benchmarking set to False. Models use random initialization, except T5-small, which uses its pretrained checkpoint. Pilots use CUDA and bfloat16 autocast; fused optimizers are used when available.

Full-training budgets are 10,000 updates for language models and T5; 50 epochs for Food-101 and both ViTs; 80 epochs for ImageNet100; 100 epochs for Cars; 60 epochs for Caltech-256. Vision full-training schedule horizons equal epochs times training batches per epoch. Their rates warm up linearly, then decay as

$$
\eta _ { s } = \eta _ { \mathrm { m i n } } + \frac { \eta _ { \mathrm { m a x } } - \eta _ { \mathrm { m i n } } } { 2 } \left[ 1 + \cos \left( \pi \frac { s - W } { T - W } \right) \right] , \qquad s > W ,
$$

with $\eta _ { s } = \eta _ { \mathrm { m a x } } s / W$ during warmup. ResNet warmup is 5% of the horizon; ViT uses 1,000 updates; EficientNet and Swin use the larger of 100 updates and 5% of the horizon. Language warmups are 1,000 for Llama and T5, and 2,000 for GPT-2-style models.

Pilot schedule horizons are explicitly configured as listed in Table 4, including ImageNet100 and Caltech-256. Pilot warmup lengths are 1,000 for Llama and T5; 2,000 for GPT-2-style models; 1,477 for Food; 4,060 for ImageNet100; 1,000 for ViTs; 640 for Cars; and 1,128 for Swin. Thus all four compact measurements occur during warmup.

Table 4: Pilot schedule and optimizer settings. Steps is the configured schedule horizon; batch is physical batch size. Accumulation defines the efective batch. Clip zero denotes an unmodified gradient norm.
<table><tr><td>Network</td><td>Dataset</td><td>Steps</td><td>Batch</td><td>Accum.</td><td>LR max</td><td>Decay</td><td>Clip</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>12800</td><td>64</td><td>1</td><td>0.0008</td><td>5e-05</td><td>1</td></tr><tr><td>Llama60M</td><td>C4</td><td>10000</td><td>64</td><td>8</td><td>0.001</td><td>1e-05</td><td>0</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>10000</td><td>64</td><td>8</td><td>0.001</td><td>1e-05</td><td>0</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>10000</td><td>8</td><td>128</td><td>0.0006</td><td>0.01</td><td>1</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td>10000</td><td>8</td><td>128</td><td>0.0006</td><td>0.01</td><td>1</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>29550</td><td>128</td><td>1</td><td>0.0003</td><td>0.01</td><td>0</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>81200</td><td>128</td><td>1</td><td>0.0003</td><td>0.01</td><td>0</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td>22560</td><td>64</td><td>1</td><td>0.0004</td><td>0.05</td><td>1</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>10000</td><td>8</td><td>8</td><td>0.0001</td><td>0.01</td><td>1</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td>19500</td><td>128</td><td>1</td><td>0.0005</td><td>0.1</td><td>1</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>39050</td><td>128</td><td>1</td><td>0.0005</td><td>0.1</td><td>1</td></tr></table>

AdamW excludes one-dimensional parameters, biases, and normalization parameters from decay according to the recipe: batch-normalization names for ResNet; normalization names for GPT, ViT, Swin, and T5. Llama uses coupled Adam decay on all parameters; EficientNet also uses coupled Adam decay. Clipping is on the full accumulated gradient norm before the optimizer update.

## D.2 Causal language modeling

GPT-2-style models. Both have six decoder blocks, width 384, six attention heads, feedforward width 1,536, vocabulary 50,304, and context 256. They use learned position embeddings, tied input/output embeddings, layer normalization epsilon 10<sup>−5</sup>, GPT-2 approximate GELU, initialization standard deviation 0.02, and dropout 0.2 in embeddings, attention, and residuals. Each efective batch averages 128 physical batches of eight sequences (262,144 input tokens). AdamW uses maximum rate $6 \times 1 0 ^ { - 4 }$ , minimum $6 \times 1 0 ^ { - 6 }$ , decay 0.01, and clip 1.

WikiText-103 uses its raw training/validation splits and GPT-2 tokenization (Merity et al., 2017). OpenWeb-Text uses separately packed training/validation streams (Gokaslan and Cohen, 2019). Batches sample uniform contiguous start positions, giving length-256 inputs and one-token-shifted targets. Evaluation every 500 updates averages next-token cross-entropy over 50 validation batches. Its validation generator uses seed 123456789; trainset evaluation uses 987654321. The packed OpenWebText streams define the actual split; their corpus revision and packing manifest are required to match the exact archived samples.

Llama-style models. Both have eight decoder blocks, width 512, feedforward width 1,376, eight attention heads, SiLU-gated feedforward layers, rotary positional encoding, RMS normalization epsilon $1 0 ^ { - 6 }$ , and ini tialization standard deviation 0.02 (Touvron et al., 2023). The configured vocabulary begins at 32,000 and is expanded for the T5-base tokenizer when required. Sequences have length 256; eight physical batches of 64 sequences form each efective batch (131,072 token positions). Adam uses maximum rate $1 0 ^ { - 3 }$ , minimum $1 0 ^ { - 4 }$ coupled decay $1 0 ^ { - 5 }$ , and gradients with their original norms.

C4 uses the English configuration (Rafel et al., 2020); the second task uses the cached DKYoon/SlimPajama-6B subset (Soboleva et al., 2023). Texts are shufled with seed 42, tokenized with T5-base, truncated/padded to 256, and evaluated with valid-token masking. Validation uses batch size 64 and a ten-million-valid-token target. Evaluation is recorded at update 1 and every 1,000 updates, aggregating losses by valid target count.

## D.3 Image classification models

ResNet50. Bottleneck stages have depths (3, 4, 6, 3) and output widths (256, 512, 1024, 2048), using ReLU, batch normalization, a $7 \times 7$ stride-2 stem and max pooling, and global average pooling (He et al., 2016). Dropout 0.2 precedes the linear 101-class Food or 100-class ImageNet head. AdamW uses maximum rate $3 \times 1 0 ^ { - 4 }$ minimum $3 \times 1 0 ^ { - 5 }$ , decay 0.01, gradients with their original norms, and batch 128.

ViT-B/16. Inputs are 224 × 224 RGB images with $1 6 \times 1 6$ patches. The encoder has 12 blocks, width 768, 12 heads, feedforward width 3,072, GELU, layer-normalization epsilon $1 0 ^ { - 6 }$ , and the standard torchvision zero encoder/attention-dropout defaults (Dosovitskiy et al., 2021). Heads have 100 CIFAR or 200 TinyImageNet outputs. AdamW uses maximum rate $5 \times 1 0 ^ { - 4 }$ , minimum $5 \times 1 0 ^ { - 6 }$ , decay 0.1, clip 1, and batch 128.

Swin-T. Patch size is 4, window size 7, base width 96, stage depths (2, 2, 6, 2), head counts (3, 6, 12, 24), and feedforward expansion 4 (Liu et al., 2021). Shifted-window attention, GELU, layer normalization, and stochastic depth 0.2 are used, with a 256-class head. AdamW uses maximum rate $4 \times 1 0 ^ { - 4 }$ , minimum $4 \times 1 0 ^ { - 6 }$ , decay 0.05, clip 1, and batch 64.

EficientNet-B0. The 32-channel stem is followed by seven MBConv stages. Their (expansion, kernel, first stride, output channels, repetitions) are (1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3), (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), and (6, 3, 1, 320, 1) (Tan and Le, 2019). Squeeze-and-excitation, SiLU, stochastic depth 0.2, a final width of 1,280, global pooling, dropout 0.2, and a 196-class head complete the model. Adam uses maximum rate $8 \times 1 0 ^ { - 4 }$ , minimum $1 0 ^ { - 5 }$ , coupled decay $5 \times 1 0 ^ { - 5 }$ , clip 1, and batch 64.

## D.4 Vision datasets, preprocessing, and evaluation

Food-101 uses 75,750 training and 25,250 validation images (Bossard et al., 2014). Stanford Cars uses 8,144/8,041 images, with its test partition used for validation (Krause et al., 2013). CIFAR-100 uses 50,000/10,000 images (Krizhevsky, 2009). TinyImageNet uses 100,000 training and 10,000 validation images across 200 classes. ImageNet100 is a 100-class subset (Deng et al., 2009); the pilot loads 126,689 training images and remaps labels consistently across four training partitions and validation. Exact subset membership is determined by its synset manifest.

Caltech-256 excludes clutter (Grifin et al., 2007). Sort category names and image paths lexicographically, create one Python random generator with seed 123, visit classes in order and shufle each class’s paths, then take the first 60 images for training and the remainder for validation. The generator continues across classes. This gives 15,360 training images.

All inputs are converted to RGB. ResNet uses random resized crop 224, area scale (0.6, 1), aspect ratio $( 3 / 4 , 4 / 3 )$ horizontal flip probability 0.5, and brightness/contrast/saturation/hue jitter (0.2, 0.2, 0.2, 0.1). ViT and Swin use crop area scale (0.5, 1), the same ratio/flip, RandAugment with two operations and magnitude 12, and random erasing probability 0.25, area (0.02, 0.2), ratio (0.3, 3.3), random fill.

EficientNet uses crop area (0.7, 1), ratio (0.85, 1.15), flip 0.5, color jitter (0.2, 0.2, 0.15, 0.03), RandAugment with one operation and magnitude 7, and erasing probability 0.1, area (0.02, 0.12), ratio (0.5, 2), random fill. Erasing follows tensor conversion and normalization. Evaluation resizes the shorter side to 256 and center crops to 224. Normalization uses means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225); CIFAR uses (0.5071, 0.4867, 0.4408) and (0.2675, 0.2565, 0.2761). Training uses cross-entropy with label smoothing 0.1, except EficientNet with 0.05. Validation uses plain cross-entropy with label smoothing set to zero. Validation averages batch-mean losses, weighting the final partial batch equally at batch level. Evaluation occurs every epoch. Training shufles and drops partial batches except EficientNet; validation preserves dataset order and includes every batch.

## D.5 T5 denoising

T5-small begins from the pretrained checkpoint and tokenizer (Rafel et al., 2020). Both encoder and decoder have six blocks, width 512, feedforward width 2,048, eight heads with key/value width 64, ReLU, normalization epsilon $1 0 ^ { - 6 }$ , and dropout 0.1. BookCorpus (Zhu et al., 2015) uses span corruption with maximum input length 256, noise density 0.15, mean span length 3, and sentinel tokens. Padded targets are masked with −100.

Corruption is seeded by training seed plus example index; validation adds 10,000,000. Full training reserves the first 20,000 corpus records for validation and uses the remaining 73,984,228 for training. Here a record is one indexed text row. The pilot reserves the first 1,000 records. These are distinct pilot/full partitions. Eight physica batches of eight form each efective batch. AdamW uses maximum rate $1 0 ^ { - \bar { 4 } }$ , minimum $1 0 ^ { - 5 }$ , decay 0.01, and clip 1. The 10,000-update training evaluates every 1,000 updates with batch size eight and valid-target-token weighting. The ten-workload comparison excluding T5 gives mean gaps 0.6850% for fixed $\beta$ and 0.4060% for the rule.

## D.6 Metrics and implementation provenance

Take each run’s minimum recorded validation loss. For the primary table, the smallest of the eleven seed-1 grid losses defines (22)’s reference. For the common-seed comparison, first take each run’s minimum and then average those minima over identical seed sets.

The implementation uses PyTorch, torchvision, Transformers, NumPy, and dataset-loading utilities. Training was executed on a cluster equipped with eight NVIDIA A100-SXM4 GPUs with 80 GB memory each; this identifies available cluster hardware. The retained execution records identify CUDA and pilot bfloat16 autocast. Exact regeneration of the original training streams additionally requires pinned package versions, the ImageNet100 synset list, packed OpenWebText split manifest, and corpus/checkpoint revisions. The training recipes and matched-seed evaluation are specified here; those additional identifiers determine the exact environment and sample streams.

## D.7 Reconstructing the selection evaluation

For each workload, use the pooled and group directions at its four measurement checkpoints. Calculate $\widehat { V } _ { \mathrm { e m p } }$ by (14), fit the three squared secants by (17), calculate $\widehat { D } _ { \mathrm { e m p } }$ by (18), and convert their ratio to a grid $\beta$ by Section 6. Then evaluate that $\beta \mathrm { ^ { * } s }$ recorded validation loss and compute its relative gap by (22). Take the arithmetic mean of the eleven gaps, their maximum, and the mean of their three largest values. Applying the identical aggregation to a fixed $\beta$ gives the comparator. The resulting mean gaps are 0.6235665% for β 0.94377 and 0.3694690% for the rule. Timing and probe-budget analyses repeat these steps with their stated measurement designs, holding the candidate trainings fixed.

Existing corpus content. The language tasks use existing text corpora, which can contain ofensive text and personally identifying information inherited from their sources. The study uses these benchmark assets for aggregate training and validation measurements. Released measurements consist of gradient-direction sketches and aggregated validation losses. Use of the original datasets and checkpoints remains subject to their providers access conditions and licenses.

## D.8 Existing assets and access conditions

Software and checkpoint licenses and dataset-provider terms govern access to the original assets. Table 5 identifies their providers and documented license information. Released measurements consist of gradient-direction sketches and aggregated losses.

## E ABLATIONS AND ADDITIONAL MEASUREMENTS

The following analyses address separate questions: whether a nearby decision time retains the benefit, how many probe batches are needed, how measured coeficients determine $\beta ,$ and how the selected configurations behave over common training seeds. Each analysis reuses saved candidate losses. Measurements select the configuration; the loss lookup evaluates it afterward.

Table 5: Software and model license records, and dataset access sources. The providers’ records govern access and reuse of the original assets.
<table><tr><td>Asset</td><td>License information or provider record</td></tr><tr><td>PyTorch /torchvision</td><td>BSD-style / BSD-3-Clause: PyTorch license and torchvision license.</td></tr><tr><td>NumPy / Transformers T5-small checkpoint</td><td>BSD-3-Clause / Apache-2.0: NumPy license and Transformers license. Apache-2.0, as stated in the provider model card. The other evaluated models</td></tr><tr><td>C4 / WikiText-103</td><td>start from random weights. C4 card: ODC-By. WikiText card: CC-BY-SA-3.0 and GFDL.</td></tr><tr><td>SlimPajama-6B</td><td>The subset card points to the original SlimPajama record for source and license</td></tr><tr><td>OpenWebText / Book-</td><td>information. This original provider record supplies the access provenance used here. Provider records: OpenWebText and BookCorpus. Rights in underlying web</td></tr><tr><td>Corpus ImageNet/ TinyIma-</td><td>text and books remain with their respective sources.</td></tr><tr><td>geNet</td><td>ImageNet access terms specify noncommercial research and educational use. The TinyImageNet release supplies its distribution provenance; consult the provider&#x27;s terms for reuse.</td></tr><tr><td>Caltech-256 Food-101 / Cars / CI-</td><td>CaltechDATA provider record, including its rights metadata. Original access sources: Food-101, Stanford Cars, and CIFAR-100. These</td></tr></table>

## E.1 Decision timing and probe budget

First hold the probe budget at sixteen and change the decision update. Table 6 reports updates 100, 150, 200, and 250. For a decision at update s, measure at $s - 1 0 0 , s - 5 0 , s - 2 0$ , s and use exactly the same fit, cubic rule, and rounding. For instance, a decision at update 150 uses observations at 50, 100, 130, and 150. Shifting this window changes the observed dynamics while preserving the measurement scales and training recipes.

Table 6: Decision-time sensitivity with sixteen probes per checkpoint and matched seed 1. Gaps are percentages. The near-0.95 comparator, β 0.94377, has mean 0.6236, maximum 2.4853, and CVaR 2.0409.
<table><tr><td>Decision update</td><td>Mean gap</td><td>Maximum gap</td><td> $\mathrm { C V a R _ { 2 5 } }$ </td></tr><tr><td>100</td><td>0.6345</td><td>2.1862</td><td>1.5271</td></tr><tr><td>150</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr><tr><td>200</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr><tr><td>250</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr></table>

Updates 150, 200, and 250 give the same aggregate result. Update 100 has a slightly larger mean gap and a smaller maximum and CVaR than the near-0.95 reference. These results locate a useful early observation region. Appendix E.8 extends the analysis to every integer update from 100 to 500 on the same pilot trajectories and fixed candidate trainings.

## E.2 Probe allocation and budget sensitivity

The common sixteen-probe design assigns four gradients to each of four groups. Averaging the gradients of each group supplies its first and second moment estimates before normalization; comparing groups supplies a dispersion measurement. The pooled direction uses all sixteen gradients. Thus the budget supports two statistical roles, rather than sixteen independent votes for a $\beta .$ Powers of two provide equal groups throughout the measured allocation $m \in \{ 2 , 4 , 8 , 1 6 \}$ , with $G = \operatorname* { m i n } ( m , 4 )$

The averaging role has a simple theoretical scale. If independent linear direction observations have covariance $\Sigma ,$ their m-sample mean has covariance $\Sigma / m \colon$ expand the covariance of the sum and use independence to eliminate cross terms. Under independent, equal-variance endpoints, the squared secant has an expected noise contribution $2 \operatorname { t r } ( \Sigma ) / ( m n ^ { 2 } )$ at lag n. This explains why the observation budget afects the finite-resolution term in the temporal fit. For the implemented nonlinear normalization, Proposition 2 separately accounts for pooling order. The choice $m = 1 6 , G = 4$ is a compact common allocation; these sampling identities explain its roles and scaling, while the measured budget audit evaluates its selections.

Table 7 holds update 200, the coeficient formula, and the rounding convention fixed, and reports every examined budget. The variability multiplier remains $m / ( G - 1 )$ throughout. Four and eight probes produce the same discrete choices, while sixteen changes the selections for EficientNet, both ResNets, and T5. The table therefore exposes how increased averaging and changed endpoint resolution translate into the practical memory decision. The common allocation is used unchanged across tasks. Workload-specific information enters through measured

Table 7: Probe-budget sensitivity at update 200. Relative gaps are percentages; the fixed comparator has mean 0.6236%, maximum 2.4853%, and $\mathrm { C V a R _ { 2 5 } }$ 2.0409%. Bold marks the smallest entry in each metric column.
<table><tr><td>Effective batches</td><td>Groups</td><td>Mean gap</td><td>Maximum gap</td><td> $\mathrm { C V a R _ { 2 5 } }$ </td></tr><tr><td>2</td><td>2</td><td>0.9185</td><td>2.5913</td><td>2.4600</td></tr><tr><td>4</td><td>4</td><td>0.6829</td><td>2.4853</td><td>2.0409</td></tr><tr><td>8</td><td>4</td><td>0.6829</td><td>2.4853</td><td>2.0409</td></tr><tr><td>16</td><td>4</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr></table>

variability and temporal change. The coeficient sensitivity and rounding margins quantify how these inputs determine the continuous memory and the final grid choice.

Separation of measurement and evaluation. For each protocol, gradient measurements determine the coeficients and the $\beta$ before the candidate-loss lookup. Candidate losses then evaluate the resulting configuration. The same equations and rounding apply to every workload. Protocol comparisons are exploratory sensitivity analyses on this benchmark suite, describing how the decisions respond to the implementation choices.

## E.3 Comparison with every fixed grid value

Table 8 evaluates all eleven fixed $\beta$ values using the same seed-1 losses and aggregation. Each constant applies to the entire workload suite. The rule has smaller mean, maximum, and CVaR than every constant, reducing mean gap by 32.3% against the retrospectively best fixed value, 0.96838. This evaluates the same gradient-based decisions under a stronger comparator than the near-0.95 reference.

Table 8: All fixed shared memories on the evaluated grid, and the tracking rule. Gaps are percentages. Bold marks the smallest value in each metric column.
<table><tr><td>Fixed shared  $\beta$  or method</td><td>Mean gap</td><td>Maximum gap</td><td> $\mathrm { C V a R _ { 2 5 } }$ </td></tr><tr><td>0.68377</td><td>4.8152</td><td>11.6651</td><td>9.9106</td></tr><tr><td>0.82217</td><td>2.0501</td><td>6.4732</td><td>4.5653</td></tr><tr><td>0.90000</td><td>1.4292</td><td>3.8770</td><td>3.3543</td></tr><tr><td>0.94377</td><td>0.6236</td><td>2.4853</td><td>2.0409</td></tr><tr><td>0.96838</td><td>0.5461</td><td>2.1862</td><td>1.4304</td></tr><tr><td>0.98222</td><td>1.8128</td><td>7.2826</td><td>4.5072</td></tr><tr><td>0.99000</td><td>3.1592</td><td>9.4700</td><td>8.2164</td></tr><tr><td>0.99438</td><td>7.9687</td><td>24.6070</td><td>18.4218</td></tr><tr><td>0.99684</td><td>13.6732</td><td>34.5576</td><td>28.0746</td></tr><tr><td>0.99822</td><td>27.6722</td><td>63.7899</td><td>55.7021</td></tr><tr><td>0.99900</td><td>42.7583</td><td>81.3312</td><td>73.2353</td></tr><tr><td>Tracking rule</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr></table>

## E.4 Task-level calibration and workload composition

The two strongest fixed choices expose diferent compromises. Shared β 0.94377 minimizes the four causallanguage-model losses and the vision-transformer family mean in the recorded grid. Shared β 0.96838 minimizes the ResNet family mean and the whole-suite fixed mean. The measured rule combines these task-specific choices with its shorter causal-model memory.

For a comparator b, define gap benefit as $\mathrm { g a p } _ { e } ( b ) - \mathrm { g a p } _ { e } ( \widehat { \beta } _ { e } )$ , in percentage points, and direct loss improvement as $1 0 0 [ L _ { e } ( b ) - L _ { e } ( \widehat { \beta } _ { e } ) ] / L _ { e } ( b )$ . Positive values favor the rule. Table 9 separates those metrics for both fixed comparators. Relative gap reduction is the reduction of distance to the grid reference; direct loss improvement gives the scale in the actual objective. Against 0.94377, positive task-level gap changes sum to 3.7332 points and

Table 9: Signed task-level benefits against the near-0.95 fixed value and the best global fixed value. Gap benefits are percentage points; direct loss improvements are percentages. Positive entries favor the rule.
<table><tr><td colspan="4"></td><td colspan="2">Gap benefit (pp)</td><td colspan="2">Loss improvement (%)</td></tr><tr><td>Network</td><td>Dataset</td><td> $\widehat { \beta }$ </td><td>Rule gap</td><td>.94377</td><td>.96838</td><td>.94377</td><td>.96838</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>0.96838</td><td>1.0124</td><td>-0.2853</td><td>+0.0000</td><td>-0.2833</td><td>+0.0000</td></tr><tr><td>Llama60M</td><td>C4</td><td>0.90000</td><td>0.2582</td><td>-0.2582</td><td>-0.0700</td><td>-0.2582</td><td>-0.0699</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>0.90000</td><td>0.1175</td><td>-0.1175</td><td>-0.0924</td><td>-0.1175</td><td>-0.0924</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>0.90000</td><td>0.1157</td><td>-0.1157</td><td>-0.1137</td><td>-0.1157</td><td>-0.1137</td></tr><tr><td>NanoGPT</td><td>WikiText</td><td>0.90000</td><td>0.1613</td><td>-0.1613</td><td>-0.0800</td><td>-0.1613</td><td>-0.0800</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>0.96838</td><td>1.0613</td><td>+1.4240</td><td>+0.0000</td><td>+1.3895</td><td>+0.0000</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>0.96838</td><td>0.0000</td><td>+2.3036</td><td>+0.0000</td><td>+2.2517</td><td>+0.0000</td></tr><tr><td>Swin-T</td><td>Caltech</td><td>0.94377</td><td>0.0000</td><td>+0.0000</td><td>+2.1862</td><td>+0.0000</td><td>+2.1395</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>0.96838</td><td>0.0039</td><td>+0.0056</td><td>+0.0000</td><td>+0.0056</td><td>+0.0000</td></tr><tr><td>ViT-B/16</td><td>CIFAR</td><td>0.94377</td><td>0.0000</td><td>+0.0000</td><td>+0.4036</td><td>+0.0000</td><td>+0.4020</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>0.94377</td><td>1.3339</td><td>+0.0000</td><td>-0.2903</td><td>+0.0000</td><td>-0.2873</td></tr></table>

adverse changes to 0.9381 points. Against 0.96838, these sums are 2.5899 and 0.6465 points. Both comparisons express larger corrections ofset by smaller calibration costs. The ResNet contribution is 3.7276 points against the first comparator and exactly zero against the second, where both configurations coincide. Swin and ViT/CIFAR-100 provide the positive changes against the best global fixed value. These complete decompositions locate the gains of the benchmark-wide comparison.

For a workload-composition audit, remove each task in turn, retain every original rule decision, and recompute both the aggregate and the best fixed value on the remaining ten tasks. The rule has smaller mean gap in ten of the eleven subsets. Table 10 gives all subsets and also keeps the original fixed comparators visible. In the subset excluding Swin, the rule’s mean gap is 0.4064% and the best fixed mean is 0.3821%. This is a descriptive composition audit of the saved suite: the decisions remain those of the original early measurements.

Table 10: Workload-removal sensitivity. Each row evaluates the remaining ten tasks. Reduction columns give relative mean-gap reductions; positive values favor the rule. The last comparator is reselected on that ten-task subset.
<table><tr><td colspan="2">Excluded task</td><td colspan="3">Mean-gap reduction (%)</td><td colspan="2">Best fixed on ten</td></tr><tr><td>Network</td><td>Dataset</td><td>Rule mean</td><td>vs .94377</td><td>vs .96838</td><td> $\beta$ </td><td>Reduction (%)</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>0.3052</td><td>+50.23</td><td>+38.91</td><td>0.96838</td><td>+38.91</td></tr><tr><td>Llama60M</td><td>C4</td><td>0.3806</td><td>+44.51</td><td>+34.60</td><td>0.96838</td><td>+34.60</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>0.3947</td><td>+42.46</td><td>+34.03</td><td>0.96838</td><td>+34.03</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>0.3948</td><td>+42.44</td><td>+34.25</td><td>0.96838</td><td>+34.25</td></tr><tr><td>NanoGPT</td><td>WikiText</td><td>0.3903</td><td>+43.10</td><td>+34.14</td><td>0.96838</td><td>+34.14</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>0.3003</td><td>+31.35</td><td>+39.29</td><td>0.94377</td><td>+31.35</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>0.4064</td><td>+10.79</td><td>+32.35</td><td>0.94377</td><td>+10.79</td></tr><tr><td>Swin-T</td><td>Caltech</td><td>0.4064</td><td>+40.75</td><td>-6.36</td><td>0.96838</td><td>-6.36</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>0.4060</td><td>+40.72</td><td>+32.37</td><td>0.96838</td><td>+32.37</td></tr><tr><td>ViT-B/16</td><td>CIFAR</td><td>0.4064</td><td>+40.75</td><td>+27.48</td><td>0.96838</td><td>+27.48</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>0.2730</td><td>+50.59</td><td>+45.00</td><td>0.96838</td><td>+45.00</td></tr></table>

## E.5 From measured coeficients to a selected β

Table 11 exposes every scalar entering the decision. Divide the variability proxy by four times the temporalchange proxy, take the cube root, convert memory to $\beta ,$ then apply the stated grid rounding. Together with the supplied sketches, these scalar values make every selection reproducible. For example, Food-101 gives $\widehat { V } _ { \mathrm { e m p } }$ ≈

Table 11: Measured coeficients and continuous-to-grid conversion. Values are rounded for display; decisions use full precision.
<table><tr><td>Network</td><td>Dataset</td><td> $\widehat { V } _ { \mathrm { e m p } }$ </td><td> $\widehat { D } _ { \mathrm { e m p } }$ </td><td> $\widehat { H } _ { \mathrm { e m p } }$ </td><td>Continuous  $\beta$ </td><td>Grid  $\beta$ </td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td> $1 . 9 7 8 \times 1 0 ^ { 2 }$ </td><td> $2 . 3 7 9 \times 1 0 ^ { - 3 }$ </td><td>27.494</td><td>0.963628</td><td>0.96838</td></tr><tr><td>Llama60M</td><td>C4</td><td> $2 . 8 7 5 \times 1 0 ^ { 1 }$ </td><td> $1 . 0 4 7 \times 1 0 ^ { - 2 }$ </td><td>8.823</td><td>0.886654</td><td>0.90000</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td> $2 . 1 1 1 \times 1 0 ^ { 1 }$ </td><td> $9 . 0 3 0 \times 1 0 ^ { - 3 }$ </td><td>8.361</td><td>0.880399</td><td>0.90000</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td> $2 . 1 4 2 \times 1 0 ^ { 1 }$ </td><td> $1 . 1 8 3 \times 1 0 ^ { - 2 }$ </td><td>7.678</td><td>0.869751</td><td>0.90000</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td> $2 . 2 9 2 \times 1 0 ^ { 1 }$ </td><td> $8 . 2 0 8 \times 1 0 ^ { - 3 }$ </td><td>8.871</td><td>0.887272</td><td>0.90000</td></tr><tr><td>ResNet50</td><td>Food-101</td><td> $1 . 6 3 2 \times 1 0 ^ { 2 }$ </td><td> $2 . 2 0 3 \times 1 0 ^ { - 3 }$ </td><td>26.458</td><td>0.962205</td><td>0.96838</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td> $1 . 6 9 5 \times 1 0 ^ { 2 }$ </td><td> $2 . 1 7 3 \times 1 0 ^ { - 3 }$ </td><td>26.917</td><td>0.962849</td><td>0.96838</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td> $1 . 7 7 5 \times 1 0 ^ { 2 }$ </td><td> $4 . 3 6 9 \times 1 0 ^ { - 3 }$ </td><td>21.656</td><td>0.953824</td><td>0.94377</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td> $1 . 2 7 1 \times 1 0 ^ { 2 }$ </td><td> $1 . 6 1 0 \times 1 0 ^ { - 3 }$ </td><td>27.019</td><td>0.962989</td><td>0.96838</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td> $1 . 1 4 2 \times 1 0 ^ { 2 }$ </td><td> $8 . 1 4 7 \times 1 0 ^ { - 3 }$ </td><td>15.190</td><td>0.934168</td><td>0.94377</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td> $1 . 4 1 8 \times 1 0 ^ { 2 }$ </td><td> $4 . 3 9 6 \times 1 0 ^ { - 3 }$ </td><td>20.055</td><td>0.950137</td><td>0.94377</td></tr></table>

163.23 and $\widehat { D } _ { \mathrm { e m p } } \approx 0 . 0 0 2 2 0 3 2 4 .$ . Thus $\widehat { H } _ { \mathrm { e m p } } = [ 1 6 3 . 2 3 / ( 4 \times 0 . 0 0 2 2 0 3 2 4 ) ] ^ { 1 / 3 } \approx 2 6 . 4 5 8 .$ , giving continuous $\beta$ 0.962205. The neighboring grid values are 0.94377 and 0.96838, whose midpoint is 0.956075. The continuous value lies above that midpoint, so nearest- $- \beta$ rounding selects 0.96838. Every workload uses this arithmetic at full precision.

## E.6 Stationary approximation and discretization

The main rule uses a closed-form approximation and nearest-β rounding. Four alternatives use exactly the same measured coeficients: the stationary root followed by nearest-β rounding, direct grid minimization of (20), direct grid minimization of (9), and direct grid minimization of $\widehat { V } _ { \mathrm { e m p } } c _ { \beta , 2 0 0 } + \widehat { D } _ { \mathrm { e m p } } a _ { \beta , 2 0 0 } ^ { 2 }$ . All four recover the same eleven selected $\beta$ values. The last uses the exact finite-history weights, including bias correction at history length 200. The choices thus agree across the exact finite-history score and its stationary and leading versions.

## E.7 Sensitivity of grid decisions

The cube-root sensitivity concerns continuous memory. A discrete decision additionally depends on its distance to a rounding boundary, the midpoint of two neighboring $\beta$ values. For a boundary b and continuous memory ${ \widehat { H } } _ { \mathrm { e m p } } ,$ multiplying the measured ratio $\widehat { V } _ { \mathrm { e m p } } / \widehat { D } _ { \mathrm { e m p } }$ by $[ 1 / ( ( 1 - b ) \widehat { H } _ { \mathrm { e m p } } ) ] ^ { 3 }$ moves the continuous $\beta$ exactly to that boundary. Table 12 reports this deterministic arithmetic margin for the nearest boundary; its sign indicates an increase or decrease of the ratio. The table shows how far each measured decision is from changing its grid value. A small boundary margin requires more precise coeficient estimation than a large one. Continuous coeficient attenuation and stability of a discrete decision are therefore complementary diagnostics: the cube root controls the former, while the measured arithmetic margins describe the latter for the observed coeficients.

Across all eleven workloads, the simultaneous safe interval for a multiplicative change in each observed ratio $\widehat { V } _ { \mathrm { e m p } } / \widehat { D } _ { \mathrm { e m p } }$ is (0.824286, 1.161718). A ratio perturbation of ±10% lies inside this interval and preserves every decision. Independently perturbing each coeficient by ±7% gives ratio multipliers between 0.93/1.07 and 1.07/0.93, also inside the interval. These are deterministic properties of the observed coeficients and the rounding map. Table 13 pairs each nearest boundary with the validation-loss consequence of the adjacent grid choice, connecting arithmetic sensitivity to performance sensitivity.

## E.8 Complete timing profile

For each integer decision time $s = 1 0 0 , \ldots , 5 0 0$ , use checkpoint lags {20, 50, 100}, calculate the two coeficients, and evaluate the selected $\beta$ by (22). This gives 401 decisions on each pilot trajectory. Over the 351 times from 150 to 500, mean gap ranges from 0.3435% to 0.6692%; it improves over the near-0.95 reference at 339

Table 12: Rounding-boundary audit from the same measured coeficients. Ratio change is the signed relative change needed to reach the nearest $\beta$ midpoint.
<table><tr><td>Network</td><td>Dataset</td><td>Continuous  $\beta$ </td><td>Nearest boundary</td><td> $\beta$  distance</td><td>Ratio change (%)</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>0.963628</td><td>0.956075</td><td>0.007553</td><td>-43.224</td></tr><tr><td>Llama60M</td><td>C4</td><td>0.886654</td><td>0.861085</td><td>0.025569</td><td>-45.679</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>0.880399</td><td>0.861085</td><td>0.019314</td><td>-36.180</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>0.869751</td><td>0.861085</td><td>0.008666</td><td>-17.571</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td>0.887272</td><td>0.861085</td><td>0.026187</td><td>-46.563</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>0.962205</td><td>0.956075</td><td>0.006130</td><td>-36.294</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>0.962849</td><td>0.956075</td><td>0.006774</td><td>-39.495</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td>0.953824</td><td>0.956075</td><td>0.002251</td><td>+16.172</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>0.962989</td><td>0.956075</td><td>0.006914</td><td>-40.181</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td>0.934168</td><td>0.921885</td><td>0.012283</td><td>-40.144</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>0.950137</td><td>0.956075</td><td>0.005938</td><td>+46.289</td></tr></table>

Table 13: Nearest decision boundary and its recorded loss consequence. Ratio change moves the observed coeficient ratio to that boundary. Gap and loss changes compare the adjacent choice with the current choice; positive changes indicate a larger loss.
<table><tr><td>Network</td><td>Dataset</td><td> $\widehat { \beta }$ </td><td>Adjacent  $\beta$ </td><td> $\Delta$  ratio (%)</td><td> $\Delta$  gap (pp)</td><td> $\Delta$  loss (%)</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>0.96838</td><td>0.94377</td><td>-43.22</td><td>-0.2853</td><td>-0.2825</td></tr><tr><td>Llama60M</td><td>C4</td><td>0.90000</td><td>0.82217</td><td>-45.68</td><td>+0.6477</td><td>+0.6460</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>0.90000</td><td>0.82217</td><td>-36.18</td><td>+0.7058</td><td>+0.7050</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>0.90000</td><td>0.82217</td><td>-17.57</td><td>+0.4311</td><td>+0.4306</td></tr><tr><td>NanoGPT</td><td>WikiText</td><td>0.90000</td><td>0.82217</td><td>-46.56</td><td>+0.2984</td><td>+0.2979</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>0.96838</td><td>0.94377</td><td>-36.29</td><td>+1.4240</td><td>+1.4091</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>0.96838</td><td>0.94377</td><td>-39.50</td><td>+2.3036</td><td>+2.3036</td></tr><tr><td>Swin-T</td><td>Caltech</td><td>0.94377</td><td>0.96838</td><td>+16.17</td><td>+2.1862</td><td>+2.1862</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>0.96838</td><td>0.94377</td><td>-40.18</td><td>+0.0056</td><td>+0.0056</td></tr><tr><td>ViT-B/16</td><td>CIFAR</td><td>0.94377</td><td>0.90000</td><td>-40.14</td><td>+2.5913</td><td>+2.5913</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>0.94377</td><td>0.96838</td><td>+46.29</td><td>-0.2903</td><td>-0.2865</td></tr></table>

times and over the best global fixed value at 288. CVaR improves over the near-0.95 reference at every time. Table 6 displays four times. The candidate trainings remain fixed, isolating sensitivity to the placement of the observation window.

## E.9 How measurement spacing changes a discrete choice

Measurement density and observation scale are separate design choices. To isolate density, hold the decision update at 200, keep the shortest and longest lags at 20 and 100, and continue evaluating the fitted change at lag 100. Replace the original lags {20, 50, 100} by a regular mesh on [20, 100] with spacing 1, 2, 5, 10, 20, 40, or 80. Including the final endpoint, these meshes use respectively 82, 42, 18, 10, 6, 4, or 3 checkpoints. Every checkpoint uses sixteen probes, and the OLS fit and rounding are otherwise unchanged.

All seven regular meshes preserve ten original $\beta$ choices and select 0.96838 for Swin instead of 0.94377. Their coeficients difer, but their rounded choices coincide. Consequently their looked-up validation losses and gaps coincide as well: mean 0.5682%, maximum 2.1862%, and CVaR 1.5271%. The mean-gap benefit over the near-0.95 reference is 8.9% for every mesh; the original four-checkpoint layout gives 40.7%. This diagnostic distinguishes broad utility relative to the empirical default from sensitivity of the additional gain supplied by Swin. The repeated aggregate numbers arise from discrete $\beta$ selection, rather than repeated or smoothed measurements. All lag responses share the final endpoint, so denser meshes also retain dependent measurement errors.

## E.10 Probe decomposition

Pooling share is $\left\| \bar { z } - z \right\| ^ { 2 }$ divided by (15)’s left side. It tells us how much measured dispersion comes from normalizing the pooled and group moments diferently. Finite-lag share is $( \widehat { C } / 1 0 0 ^ { 2 } ) / \widehat { D } _ { \mathrm { e m p } }$ . It tells us how much fitted change remains in the inverse-square component. These descriptive shares quantify the composition of the measured proxies. In five cases the fitted intercept clips to zero, leaving all of the temporal-change proxy in its finite-lag component. Equation (16) explains how endpoint measurement variance can influence the temporal fit.

Table 14: Sixteen-probe measurement decomposition at update 200.
<table><tr><td>Network</td><td>Dataset</td><td>Pooling contribution (%)</td><td>Finite-lag contribution (%)</td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>3.12</td><td>100.00</td></tr><tr><td>Llama60M</td><td>C4</td><td>4.45</td><td>54.57</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>4.63</td><td>91.79</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>3.17</td><td>100.00</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td>3.34</td><td>100.00</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>3.09</td><td>100.00</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>2.64</td><td>91.24</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td>3.46</td><td>55.84</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>3.33</td><td>94.47</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td>2.84</td><td>53.98</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>3.10</td><td>100.00</td></tr></table>

The component may also include roughness of the target. The probe budget is therefore part of the algorithm’s specification.

For intuition, consider the illustrative linear measurement model with a constant population direction. An observation with variance V averaged over m independent samples has variance $V / m .$ Independent endpoints give $C = 2 V / m$ and zero drift intercept, while the linear scatter proxy has expectation V. Inserting these expected coeficients into the operational formula yields $H = [ m 1 0 \bar { 0 ^ { 2 } } / 8 ] ^ { \bar { 1 } / 3 }$ . For $m = 1 6 .$ , this plug-in value is $H \approx 2 7 . 1 4$ , or β approximately 0.96316. The mean of the random memory estimate additionally depends on the joint distribution of its coeficients through the nonlinear cube root. The plug-in calculation isolates how endpoint fluctuations contribute a finite memory at constant population direction. Thus the finite-lag coeficient represents efective measured change, combining the gradient dynamics and the observation scale specified by the algorithm.

## E.11 Separating the two temporal contributions

The constant-direction calculation above supplies a simple diagnostic: inserting the expected coeficients of that illustrative linear model for every workload gives the same continuous β 0.96316, rounding to 0.96838. Table 15 evaluates this common choice against the measured rule. It also repeats the selection calculation after replacing the temporal proxy by either ${ \widetilde { C } } / 1 0 0 ^ { 2 }$ alone or Ib alone. When the intercept vanishes, the latter diagnostic uses the largest admissible memory, as specified by the objective’s boundary rule. These retrospective coeficient diagnostics reuse the measured sketches and completed training outcomes to isolate the information supplied by each component. The full measured rule has smaller aggregate gaps than each diagnostic. Its decisions

Table 15: Contribution diagnostics using the same sketches, rounding, and seed-1 loss lookup. Gaps are percentages; bold marks the smallest value in each column.
<table><tr><td>Coefficient diagnostic</td><td>Mean gap</td><td>Maximum gap</td><td> $\mathrm { C V a R _ { 2 5 } }$ </td></tr><tr><td>Constant-target calculation</td><td>0.5461</td><td>2.1862</td><td>1.4304</td></tr><tr><td>Endpoint contribution only</td><td>0.5682</td><td>2.1862</td><td>1.5271</td></tr><tr><td>Intercept contribution only</td><td>17.9403</td><td>72.4969</td><td>51.6408</td></tr><tr><td>Full measured rule</td><td>0.3695</td><td>1.3339</td><td>1.1359</td></tr></table>

therefore use information beyond a budget-only common memory. The full and endpoint-only rules share ten choices; adding the intercept changes Swin from 0.96838 to 0.94377, accounting for their aggregate diference. This identifies the specific decision through which the persistent-change component contributes in this audit. Together, these checks assess the finite-measurement noise–change score on the saved workload suite, with the population tracking identity supplying the analytical form of its balance.

## E.12 Evaluation summaries of the same trajectories

The primary evaluation uses each run’s minimum recorded validation loss. Two additional summaries test whether the comparison persists at the end of training: the last recorded loss and the arithmetic mean of the last three recorded losses. The selected $\beta$ values remain exactly those from the primary pilot. For each summary, recompute the per-workload grid reference from all eleven candidates, all fixed-memory comparisons, and the best fixed value by mean gap.

All 121 seed-1 grid curves and 155 curves including the common-seed pairs are evaluated. Candidates for each workload have matching validation counts and ordered schedules. Language losses are indexed by update and vision losses by epoch, according to Appendix D.

Table 16: Complete sensitivity to three evaluation summaries. Mean gaps and direct loss improvements are percentages. Best fixed is recomputed for each summary. Direct improvements use the near-0.95 comparator 0.94377; positive values favor the rule. Common-seed values evaluate reuse of the seed-1 calibration across initializations.
<table><tr><td></td><td colspan="3">Mean relative gap (%)</td><td></td><td colspan="2">Direct improvement (%)</td></tr><tr><td>Summary</td><td>Rule</td><td>Fixed .94377</td><td>Best fixed</td><td>Fixed  $\beta$ </td><td>Seed 1</td><td>Common seeds</td></tr><tr><td>Minimum recorded</td><td>0.3695</td><td>0.6236</td><td>0.5461</td><td>0.96838</td><td>+0.2464</td><td>-0.1049</td></tr><tr><td>Last recorded</td><td>0.4537</td><td>0.6242</td><td>0.5707</td><td>0.96838</td><td>+0.1629</td><td>-0.1093</td></tr><tr><td>Mean of last three</td><td>0.4244</td><td>0.4767</td><td>0.4767</td><td>0.94377</td><td>+0.0492</td><td>-0.1586</td></tr></table>

Under each summary, the rule has smaller mean, maximum, and CVaR gap than every fixed grid value. Meangap reductions against the respective best fixed are 32.35%, 20.51%, and 10.96%. Table 16 also reports direct loss improvements, separating evaluation-summary sensitivity from transfer across initializations.

Apply (22) and Section 7’s aggregation to each summary. Common-seed improvements average the separately summarized runs; memory choices remain fixed from the original pilot.

## E.13 All common training seeds

Average each configuration’s minimum validation loss over identical seed sets, obtaining $\bar { L } _ { \mathrm { f i x e d } }$ and $\bar { L } _ { \mathrm { r u l e } }$ , and report $1 0 0 ( \bar { L } _ { \mathrm { f i x e d } } - \bar { L } _ { \mathrm { r u l e } } ) / \bar { L } _ { \mathrm { f i x e d } }$ . Positive values favor the rule. Table 17 uses seeds 1–3, except Food-101 with seed 1. The pilot chooses $\beta$ once with seed 1 and retains it for the other seeds. The primary comparison tests matched calibration; this table tests variability across initializations. Its equal-workload mean is −0.1049%.

Table 17: Direct validation-loss improvement over the same common seeds against β 0.94377. Zero means the selected and fixed configurations coincide.
<table><tr><td>Network</td><td>Dataset</td><td>Common seeds</td><td>Improvement (%)</td><td>Rule  $\widehat { \beta }$ </td></tr><tr><td>EfficientNet-B0</td><td>Cars</td><td>1,2,3</td><td>-2.1064</td><td>0.96838</td></tr><tr><td>Llama60M</td><td>C4</td><td>1,2,3</td><td>-0.2015</td><td>0.90000</td></tr><tr><td>Llama60M</td><td>SlimPajama</td><td>1,2,3</td><td>-0.2985</td><td>0.90000</td></tr><tr><td>NanoGPT</td><td>OpenWebText</td><td>1,2,3</td><td>-0.2046</td><td>0.90000</td></tr><tr><td>NanoGPT</td><td>WikiText-103</td><td>1,2,3</td><td>-0.1235</td><td>0.90000</td></tr><tr><td>ResNet50</td><td>Food-101</td><td>1</td><td>1.3895</td><td>0.96838</td></tr><tr><td>ResNet50</td><td>ImageNet100</td><td>1,2,3</td><td>0.3719</td><td>0.96838</td></tr><tr><td>Swin-T</td><td>Caltech-256</td><td>1,2,3</td><td>0.0000</td><td>0.94377</td></tr><tr><td>T5-small</td><td>BookCorpus</td><td>1,2,3</td><td>0.0190</td><td>0.96838</td></tr><tr><td>ViT-B/16</td><td>CIFAR-100</td><td>1,2,3</td><td>0.0000</td><td>0.94377</td></tr><tr><td>ViT-B/16</td><td>TinyImageNet</td><td>1,2,3</td><td>0.0000</td><td>0.94377</td></tr></table>