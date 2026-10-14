# A Rate Transfer Method: Convergence Rates in Stochastic Gradient Descent via Stochastic Calculus

Neil Walton<sup>\*</sup> Durham University

## Abstract

We develop a rate-transfer method that converts finite-time Lyapunov estimates for an associated stochastic differential equation (SDE) into finite-time convergence bounds for stochastic gradient algorithms. Under generic moment and smoothness conditions, the convergence analysis of the stochastic gradient method reduces to a calculus argument verifying a Lyapunov generator inequality for the SDE. The method recovers established rates for classical stochastic gradient methods and provides a way to establish convergence rates for new optimization algorithms through calculus arguments applied to their associated SDEs.

Keywords: Stochastic Approximation, Stochastic Differential Equations

## 1 Introduction

Stochastic gradient descent (SGD) methods evolve through discrete noisy updates. Their associated continuous-time models can be easier to analyse. In particular, stochastic differential equations have a well-developed calculus for studying convergence. We ask when an SDE convergence bound can be transferred to the stochastic gradient algorithm.

Consider the stochastic gradient recursion

$$
{ \cal X } _ { n + 1 } = { \cal X } _ { n } + \alpha _ { n } \{ b ( { \cal X } _ { n } ) + \varepsilon _ { n + 1 } \} .
$$

Here $b (  { \boldsymbol { X } } _ { n } )$ is the mean update, $\alpha _ { n } > 0$ is the stepsize, and the error term $\varepsilon _ { n + 1 }$ , conditional on the past, has mean zero and covariance $\sigma ( X _ { n } ) \sigma ( X _ { n } ) ^ { \top }$ . The associated SDE is

$$
d \pmb { Z } _ { t } = \pmb { b } ( \pmb { Z } _ { t } ) d t + \sqrt { \alpha ( t ) } \sigma ( \pmb { Z } _ { t } ) d \pmb { B } _ { t } ,
$$

Let $\mathbf { } _ { \mathbf { } \mathbf { } \mathbf { } B _ { t } }$ be a standard Brownian motion. The function α(t) expresses the stepsize sequence on the corresponding diffusion-time scale, defined in Section 2.2.

We can analyze convergence of the SDE through a Lyapunov argument. Let V be a nonnegative, twice-differentiable function measuring the error of interest. The generator of the SDE above is a differential operator $\mathcal { L } _ { t } V ( \pmb { x } ) = \pmb { b } ( \pmb { x } )$ $\begin{array} { r } { \nabla V ( \pmb { x } ) + \frac { \alpha ( t ) } { 2 } \mathrm { t r } \big ( \sigma ( \pmb { x } ) \sigma ( \pmb { x } ) ^ { \top } \nabla ^ { 2 } V ( \pmb { x } ) \big ) } \end{array}$ . We call V a Lyapunov function when there are real functions λ and g such that

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

If the SDE starts at state $X _ { 0 } ,$ , the Lyapunov inequality gives the following finite-time bound for the SDE.

SDE bound:

$$
\mathbb { E } V ( Z _ { t } ) \leq e ^ { - \int _ { 0 } ^ { t } \lambda ( r ) d r } \mathbb { E } V ( X _ { 0 } ) + \int _ { 0 } ^ { t } e ^ { - \int _ { s } ^ { t } \lambda ( r ) d r } g ( s ) d s .
$$

Under the smoothness and moment conditions of Assumption 1, and assuming that the required moments of the stochastic gradient iterates remain bounded, Theorem 2 transfers this SDE estimate to the stochastic gradient method:

SGD bound:

$$
\mathbb { E } V ( X _ { n } ) \leq e ^ { - \int _ { 0 } ^ { t _ { n } } \lambda ( r ) d r } \mathbb { E } V ( X _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } e ^ { - \int _ { s } ^ { t _ { n } } \lambda ( r ) d r } \{ g ( s ) + C \alpha ( s ) \} d s .
$$

Here $t _ { 0 } = 0$ and $\begin{array} { r } { t _ { n } = \sum _ { k = 0 } ^ { n - 1 } \alpha _ { k } } \end{array}$ for $n \geq 1$ , so $t _ { n + 1 } - t _ { n } = \alpha _ { n }$

Once we have established the finite-time bound for the SDE, it indicates the likely form of the SGD bound. The stochastic gradient method inherits the SDE contraction and forcing, while the comparison contributes an additional forcing term: in particular, $g ( s ) \longmapsto g ( s ) + C \alpha ( s )$ . Theorem 2 then establishes the finite-time SGD bound.

We use two examples to illustrate what the rate-transfer method provides. For strongly convex stochastic gradient descent, the transferred SDE estimate recovers the classical convergence rates obtained by direct discrete-time analyses such as Moulines and Bach [2011]. For stochastic Heavy Ball, based on the momentum dynamics introduced by Polyak [1964], the same argument recovers the familiar geometric transient together with an O(h) neighbourhood. In both cases, the purpose is not to improve the known rates, but to obtain them from the same continuous-time Lyapunov calculation and finite-window transfer argument.

The basic idea presented here also extends to other forms of Lyapunov analysis. More broadly, the benefit of a transfer theorem is that it removes much of the detailed discrete-time analysis from the convergence proof. A new stochastic gradient method can be analysed by identifying an associated SDE and proving an appropriate Lyapunov inequality. Convergence rates can therefore often be established through a comparatively short calculus argument, rather than a bespoke discrete-time proof.

## 1.1 Related work.

Our motivation is stochastic gradient descent methods. However, like prior work, we formulate the transfer theory in the broader language of stochastic approximation, of which SGD is a special case.

Continuous time methods have a long history in stochastic approximation. Stochastic approximation is a noisy Euler discretization of its mean dynamics. Ljung [1977] introduced the ODE viewpoint for recursive stochastic algorithms; contemporaneous work of Kushner [1977] developed closely related convergence methods, and Kushner and Clark [1978] substantially extended and systematized the differential-equation approach. This programme was developed further by Metivier and Priouret [1984], Benveniste et al. [1990], and Bena´ ¨ım [1996]; see also the monographs of Kushner and Yin [2003] and Borkar [2008]. The central principle is that a simpler continuous model can reveal the stability and asymptotic behaviour of a stochastic recursion. Transfer principles also appear in other areas. For example, Dai [1995] shows that, when every trajectory of the corresponding fluid limit model reaches zero and remains there, the associated queueing network is positive Harris recurrent. This provides an analogue of the SDE-to-SGD transfer principle considered here.

Diffusion approximations also appear early in this literature. Kushner and Huang [1979, 1981] used weak convergence of appropriately normalized stochastic-approximation errors to diffusion processes. Later work developed strong diffusion approximations for more general recursive stochastic algorithms [Pezeshki-Esfahani and Heunis, 1997], while diffusion methods continue to be used to describe rates and fluctuations; see, Kushner and Yin [2003]. Here the focus is on asymptotic scaling under which the normalized error converges weakly to a nondegenerate Gaussian or diffusion limit. This is distinct from the finite-time convergence bounds studied here, which control the error at each iteration.

A methodological precursor is Borkar and Mitter [1999], who use finite-time diffusion approximations on successive fixed intervals to obtain long-time information for a constant-stepsize stochastic recursion. More recently, SDEs have been used as quantitative continuous-time models of stochastic gradient algorithms. Stochastic modified equations and weak diffusion approximations have been developed for SGD and related methods [Li et al., 2017, 2019, Hu et al., 2019], while continuous-time convergence arguments have also been used to guide corresponding discrete analyses [Orvieto and Lucchi, 2019]. Uniform-in-time weak approximations are available in strongly convex settings [Feng et al., 2020, Li and Wang, 2023]. Particularly close to our decreasing-step setting, Fontaine et al. [2021] establish finite horizon approximations of SGD by a time-inhomogeneous SDE and derive, in parallel, corresponding continuousand discrete-time convergence rates. From the numerical-analysis viewpoint, Gaussian stochastic approximation is an Euler–Maruyama discretisation of the associated SDE, whose finite-horizon approximation theory is classical [Kloeden and Platen, 1992]. A central difficulty is that trajectory-error estimates generally deteriorate as the time horizon grows, a familiar issue in the numerical analysis of dynamical systems [Stuart, 1994]; under additional dissipativity or stability conditions, time-uniform control may sometimes be recovered [Fang and Giles, 2020].

The main challenge addressed by this paper is how to retain a useful SDE estimate over arbitrarily long times when the underlying diffusion approximation is only accurate on finite horizons. Strong approximation approaches seek to keep the stochastic recursion and the diffusion close as processes; however, approximation errors accumulate as the time horizon grows. We instead use only the weaker distributional comparison applied to the Lyapunov function. To establish this comparison over finite times, we use stochastic flow theory of Kunita [1990], in the forms given by Varadhan [2006] and Matsumoto and Taniguchi [2016]. On each fixed diffusion-time epoch, the SDE is restarted from the current iterate and the local comparison error enters the Lyapunov estimate as an additional forcing term. The key point is that the Lyapunov function is now allowed to attenuate the errors introduced on earlier epochs, without requiring a single diffusion approximation to remain accurate over a growing time horizon. Iterating this argument propagates the SDE Lyapunov estimate over arbitrarily many epochs and yields a quantitative bound at every discrete iteration.

## 2 The finite window SDE transfer framework

Our motivation is stochastic gradient descent and related methods. We formulate the transfer theory for stochastic approximation, which includes SGD and also covers discrete stochastic recursions that need not arise from a gradient. We use stochastic approximation for the general recursion and retain SGD terminology when discussing optimization applications.

## 2.1 Notation

For a vector-valued map $b : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , Db denotes its Jacobian and $D ^ { j } \pmb { b }$ its higher derivatives, viewed as multilinear maps. For $\sigma : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times q } , \sigma _ { j }$ denotes its jth column. The symbols $C , c , C _ { T } , . . .$ . denote finite positive constants whose values may change from line to line, with subscripts indicating relevant dependence.

$$
\langle \langle \pmb { x } \rangle \rangle : = ( 1 + \| \pmb { x } \| _ { 2 } ^ { 2 } ) ^ { 1 / 2 } .
$$

We use $\langle \langle { \pmb x } \rangle \rangle ^ { p }$ for the polynomial weight of order $p .$

## 2.2 Stochastic approximation

We now make this broader stochastic-approximation setup precise. Let $( \Omega , \mathcal { F } , \{ \mathcal { F } _ { n } \} _ { n \geq 0 } , \mathbb { P } )$ be a filtered probability space, and let $\boldsymbol { X } _ { n } \in \mathbb { R } ^ { d }$ be adapted. The stepsizes $\alpha _ { n } > 0$ are deterministic, and the total increment $H _ { n + 1 } \operatorname { i s } \mathcal { F } _ { n + 1 }$ -measurable. Write

$$
X _ { n + 1 } = X _ { n } + \alpha _ { n } H _ { n + 1 } , \qquad H _ { n + 1 } = b ( X _ { n } ) + \varepsilon _ { n + 1 } .\tag{1}
$$

Let B be a q-dimensional standard Brownian motion. For functions $\pmb { b } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d } , A : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d }$ , and $\sigma : \mathbb { R } ^ { d } $ $\mathbb { R } ^ { d \times q }$ , the noise satisfies $\mathbb { E } [ \pmb { \varepsilon } _ { n + 1 } \ | \ \mathcal { F } _ { n } ] = 0$ and $\mathbb { E } [ \pmb { \varepsilon } _ { n + 1 } \pmb { \varepsilon } _ { n + 1 } ^ { \top } \ \vert \ \mathcal { F } _ { n } ] = A ( \pmb { X } _ { n } )$ . The covariance factorization is $A ( \pmb { x } ) = \sigma ( \pmb { x } ) \sigma ( \pmb { x } ) ^ { \top }$ . The drift b and covariance factor σ specify the associated diffusion. The total increment ${ \cal { H } } _ { n + 1 }$ remains the quantity used in the moment assumptions below. We also need a clock relating algorithmic iterations to diffusion time.

We introduce the diffusion times $t _ { 0 } = 0$ , and

$$
t _ { n } = \sum _ { k = 0 } ^ { n - 1 } \alpha _ { k } ,
$$

and refer to $t _ { n }$ as the diffusion time associated with iteration n. Throughout the paper we assume $t _ { n } \to \infty \mathrm { a s } n \to \infty ,$ or equivalently $\textstyle \sum _ { n } \alpha _ { n } = \infty$ . We define the piecewise-constant stepsize:

$$
\alpha ( t ) = \alpha _ { n } , \quad t \in [ t _ { n } , t _ { n + 1 } ) , \qquad \alpha _ { \operatorname* { m a x } } : = \operatorname* { s u p } _ { k \geq 0 } \alpha _ { k } < \infty .
$$

Throughout the paper, $T > 0$ denotes a fixed diffusion time horizon, independent of the iteration indices.

We define, for every m,

$$
N _ { T } ( m ) = \operatorname* { m i n } \{ n > m : t _ { n } - t _ { m } \geq T \} .\tag{2}
$$

Thus $N _ { T } ( m )$ is the next iteration after m at which T units of diffusion time have passed.

By construction, $T \leq t _ { N _ { T } ( m ) } - t _ { m } < T + \alpha _ { \operatorname* { m a x } } .$

We now use these quantities to define the associated SDE.

## 2.3 The associated SDE

The associated diffusion evolves on the diffusion time clock defined above. Its SDE is

$$
d \pmb { Z } _ { t } = \pmb { b } ( \pmb { Z } _ { t } ) d t + \sqrt { \alpha ( t ) } \sigma ( \pmb { Z } _ { t } ) d \pmb { B } _ { t } ,\tag{3}
$$

The factor $\sqrt { \alpha ( t ) }$ is chosen so that, over $[ t _ { n } , t _ { n + 1 } )$ , the local drift and covariance scales are $\alpha _ { n } \pmb { b } ( \pmb { x } )$ and $\alpha _ { n } ^ { 2 } A ( { \pmb x } )$ matching the first two conditional moment scales of the stochastic approximation increment.

For any diffusion time $s \geq 0$ and state x, let $Z ^ { s , { \pmb x } } = ( Z _ { t } ^ { s , { \pmb x } } : t \geq s )$ denote the solution to (3) with ${ Z _ { s } ^ { s , x } } = x$ . We refer to (3) with this initial condition as the restarted SDE. The restart construction is important to the finite window argument. At iteration $m ,$ , we compare the stochastic approximation over the next fixed diffusion time window with the continuous model restarted from $\left( { t _ { m } , { X _ { m } } } \right)$ . At the next block boundary the comparison is restarted again.

Thus the method requires controlled SA–SDE approximation only on fixed diffusion time windows, rather than over an ever-growing time interval. We assume that the SDE is well posed for every restart $( s , { \pmb x } )$ and is non-explosive on finite intervals. The corresponding restarted solutions define the evolution operators $P _ { s , t }$ below, with the usual composition property.

To describe the evolution of functions under the restarted diffusion, let V be any function for which the expectation below is finite and define

$$
\begin{array} { r } { P _ { s , t } V ( \pmb { x } ) = \mathbb { E } [ V ( \pmb { Z } _ { t } ^ { s , \pmb { x } } ) ] , \qquad s \leq t . } \end{array}
$$

Thus $P _ { s , t } V ( { \pmb x } )$ is the expected value of V at diffusion time t when the diffusion starts from x at diffusion time s. Later, V will typically be the Lyapunov function used to measure the error of the stochastic approximation. The family $\{ P _ { s , t } \} _ { s \leq t }$ is the evolution family associated with the time-inhomogeneous SDE. It satisfies $P _ { s , s } = I$ and

$$
P _ { s , t } = P _ { s , u } P _ { u , t } , \quad s \leq u \leq t .\tag{4}
$$

We refer to (4) as the composition property. For a fixed terminal index n, define

$$
u _ { k , n } ( { \pmb x } ) = P _ { t _ { k } , t _ { n } } V ( { \pmb x } ) = { \mathbb E } \big [ V \big ( { \pmb Z } _ { t _ { n } } ^ { t _ { k } , { \pmb x } } \big ) \big ] , \qquad 0 \leq k \leq n .\tag{5}
$$

We refer to $u _ { k , n } ( { \pmb x } )$ as the backward function. Thus $u _ { k , n } ( { \pmb x } )$ is the expected terminal value of V when the diffusion is started from x at diffusion time $t _ { k }$ and run until $t _ { n }$ . Since $P _ { t _ { n } , t _ { n } } = I $ , we have $u _ { n , n } = V$ . By the composition property,

$$
u _ { k , n } = P _ { t _ { k } , t _ { k + 1 } } u _ { k + 1 , n } , \qquad 0 \leq k < n .\tag{6}
$$

For $\phi \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ , the time-dependent generator of the SDE (3) is

$$
\mathcal { L } _ { t } \phi ( \pmb { x } ) = \pmb { b } ( \pmb { x } ) ^ { \top } \nabla \phi ( \pmb { x } ) + \frac { \alpha ( t ) } { 2 } \pmb { A } ( \pmb { x } ) : \nabla ^ { 2 } \phi ( \pmb { x } ) ,
$$

Here $\boldsymbol { A } : \boldsymbol { B } = \mathrm { t r } ( \boldsymbol { A } ^ { \top } \boldsymbol { B } )$ denotes the Frobenius inner product.

## 2.4 Lyapunov bounds for the associated SDE

A key feature of our analysis is that we do not need to solve the associated SDE explicitly. Instead, we seek a non-negative Lyapunov function V for which the generator satisfies

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

Here, as with standard Lyapunov analysis, V measures the deviation, energy, or error relevant to the application; $\lambda ( t )$ describes the contraction generated by the continuous dynamics; and $g ( t )$ represents the residual forcing. We take λ and g to be deterministic, nonnegative, and locally integrable.

We can analyze the above Lyapunov bound informally as follows. The formal argument for this is given in Lemma 1 below. For the restarted diffusion, Ito’s formula givesˆ

$$
d V ( Z _ { t } ^ { s , \pmb { x } } ) = \mathcal { L } _ { t } V ( Z _ { t } ^ { s , \pmb { x } } ) d t + d M _ { t } ,
$$

where M is a local martingale. Thus, using the Lyapunov condition,

$$
d V ( Z _ { t } ^ { s , x } ) \leq [ - \lambda ( t ) V ( Z _ { t } ^ { s , x } ) + g ( t ) ] d t + d M _ { t } .
$$

We also define $R ( s , t )$

$$
R ( s , t ) = \exp \left\{ - \int _ { s } ^ { t } \lambda ( r ) d r \right\} .
$$

The scalar factor R satisfies the following composition property.

$$
R ( s , t ) = R ( s , u ) R ( u , t ) , \qquad s \leq u \leq t .\tag{7}
$$

Using an integrating-factor and Gronwall argument, we obtain¨

$$
P _ { s , t } V ( \pmb { x } ) \leq R ( s , t ) V ( \pmb { x } ) + \int _ { s } ^ { t } R ( r , t ) g ( r ) d r .\tag{8}
$$

Here $R ( s , t )$ is the contraction inherited from the SDE, while the integral term is the accumulated forcing. This finite time bound is the continuous input used later by the transfer theorem. The important point is that everything to this point is relatively standard calculus.

## 2.5 The finite-window transfer mechanism

The continuous analysis gives the finite-time bound (8). We now explain how this bound is transferred to the stochastic approximation. The formal result is Theorem 2; Proposition 12 supplies its finite-window comparison. Local weak comparison.

The associated SDE is constructed so that, over one stochastic-approximation step, its drift and covariance scales match those of the discrete update. Consequently, Taylor expansion of the SA step and Dynkin expansion of the SDE have the same leading term, and their one-step weak difference is of order $\alpha _ { k } ^ { 2 }$ . The formal finite-window comparison below is justified by Proposition 12.

$$
\left| \mathbb { E } [ u ( X _ { k + 1 } ) \mid { \mathcal { F } } _ { k } ] - P _ { t _ { k } , t _ { k + 1 } } u ( X _ { k } ) \right| \leq C _ { T } \alpha _ { k } ^ { 2 } .
$$

For a terminal Lyapunov value at iteration n, we use the backward functions defined in (5). Then $u _ { n , n } = V$ , and the one-step backward identity (6) gives $P _ { t _ { k } , t _ { k + 1 } } u _ { k + 1 , n } = u _ { k , n } .$ Lemma 8 supplies the derivative bounds needed to apply these one-step estimates to $u _ { k + 1 , n } .$ , and Proposition 12 makes the resulting finite-window telescoping rigorous. Here, we recall that $\alpha ( t ) = \alpha _ { k }$ for $t \in [ t _ { k } , t _ { k + 1 } )$ , with $t _ { k + 1 } - t _ { k } = \alpha _ { k }$ . For m $\leq n \leq N _ { T } ( m )$ , we telescope the resulting one-step bounds and take expectations to obtain

$$
\left. \mathbb { E } V ( \pmb { X } _ { n } ) - \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \pmb { X } _ { m } ) ] \right. \leq C _ { T } \sum _ { k = m } ^ { n - 1 } \alpha _ { k } ^ { 2 } = C _ { T } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .\tag{9}
$$

Here $C _ { T }$ may depend on the fixed window length T, but is uniform in the absolute restart time $t _ { m }$ One-window transfer.

Substituting the finite-time SDE estimate (8) into the preceding inequality gives

$$
\mathbb { E } V ( X _ { n } ) \leq R ( t _ { m } , t _ { n } ) \mathbb { E } V ( X _ { m } ) + \int _ { t _ { m } } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s + C _ { T } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .\tag{10}
$$

The first two terms are exactly the contraction and forcing from the SDE estimate; the final integral is the approximation cost. This is not yet a long-time result: the comparison is controlled only on a fixed diffusion-time window of length T. Restarting over successive windows.

The restart mechanism combines successive fixed-window comparisons. Recall from (2) that $N _ { T } ( m )$ is the first iteration after m at which at least T units of diffusion time have elapsed. Set $m _ { 0 } = 0$ and $m _ { j + 1 } = N _ { T } ( m _ { j } )$

At each boundary, only the SDE comparator is restarted, from $( t _ { m _ { j } } , X _ { m _ { j } } ) ;$ the stochastic approximation itself continues normally. The crucial point is that the comparison is made directly at the Lyapunov observable. The terminal quantity $\mathbb { E } V ( X _ { m _ { j + 1 } } )$ from one window is exactly the quantity needed to start the next one-window Lyapunov estimate. Since the comparison constants are uniform in the absolute restart time, the same fixed-window comparison applies again. The composition rule (7) propagates the SDE contraction across the windows, and approximation errors from earlier windows are multiplied by the contraction over subsequent windows. Thus no single SDE approximation needs to remain accurate over a growing time horizon.

Iterating the preceding window estimate (10) over these blocks, then appending one final partial window, gives the following bound for every $m _ { J } < n \leq m _ { J + 1 }$

$$
\mathbb E V ( \mathbf X _ { n } ) \le R ( 0 , t _ { n } ) \mathbb E V ( \mathbf X _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s + C _ { T } \sum _ { j = 0 } ^ { J } R ( t _ { m _ { j + 1 } \wedge n } , t _ { n } ) \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } \wedge n } } \alpha ( s ) d s .
$$

This is the desired all-iteration estimate: the SDE contraction and forcing are propagated from time zero, while each local $O ( \alpha _ { k } ^ { 2 } )$ weak error appears as an α(t)-scale cost attenuated by the contraction over later windows. The formal theorem below replaces this schematic calculation by the precise weighted assumptions needed on an unbounded state space. $W h y$ the weak comparison is enough.

As discussed in Section 1.1, diffusion approximations have also been developed through stronger process-level comparisons. Such approximations provide more information, but require correspondingly stronger control. For rate transfer here, it is enough to compare the Lyapunov observable V weakly over each fixed window. This is exactly the quantity that closes under the restart mechanism.

Thus a fixed-window weak comparison suffices to propagate the SDE Lyapunov estimate over all iterations. Under the bounded-contraction condition in the main theorem, the transported approximation costs can be written as the additional forcing $C \alpha ( s )$ , giving the replacement $g ( s ) \mapsto g ( s ) + C \alpha ( s )$ . The precise statement is Theorem 2; its proof is in Appendix B.

## 3 Rate transfer under primitive assumptions

Section 3.1 states the primitive regularity and increment conditions used for the finite-window comparison. Section 3.2 gives the main rate-transfer theorem; its all-iteration conclusion also requires the global moment stability condition $\begin{array} { r } { M _ { p } : = \operatorname* { s u p } _ { n > 0 } \mathbb { E } \langle \langle X _ { n } \rangle \rangle ^ { p } < \infty } \end{array}$ . Section 3.3 gives a sufficient condition for this stability directly from a Foster–Lyapunov inequality for the associated SDE.

## 3.1 Primitive assumptions

Assumption 1. For the main primitive result, let $b \in C ^ { 3 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ and $\sigma \in C ^ { 3 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d \times q } )$ . Here $\| \cdot \| _ { F }$ denotes the Frobenius norm. Assume that,for all x, $\pmb { y } \in \mathbb { R } ^ { d }$

$$
\| b ( { \pmb x } ) - b ( { \pmb y } ) \| \leq L _ { b } \| { \pmb x } - { \pmb y } \| , \qquad \| { \pmb \sigma } ( { \pmb x } ) - { \pmb \sigma } ( { \pmb y } ) \| _ { F } \leq L _ { \sigma } \| { \pmb x } - { \pmb y } \| .\tag{11}
$$

Since the coefficients are $C ^ { 1 }$ , this condition implies

$$
\operatorname* { s u p } _ { \pmb { x } } \| D \pmb { b } ( \pmb { x } ) \| < \infty , \qquad \operatorname* { s u p } _ { \pmb { x } } \sum _ { j } \| D \sigma _ { j } ( \pmb { x } ) \| ^ { 2 } < \infty .
$$

Assume also, for some $\kappa \geq 0 ,$

$$
\| D ^ { 2 } \pmb { b } ( \pmb { x } ) \| + \sum _ { j } \| D ^ { 2 } \sigma _ { j } ( \pmb { x } ) \| \le C \langle \langle \pmb { x } \rangle \rangle ^ { \kappa } .\tag{12}
$$

Let $V \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ with $V \geq 0$ and suppose, for some $v \geq 0 ,$ , that

$$
\| \nabla V ( \pmb { x } ) \| + \| \nabla ^ { 2 } V ( \pmb { x } ) \| \leq C \langle \pmb { \chi } _ { } \rangle ^ { \vee } .\tag{13}
$$

For some $p \ge v + \kappa + 2$ , we assume<sup>1</sup>

$$
\mathbb { E } [ \| \boldsymbol { H } _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] \le M \langle \langle X _ { k } \rangle \rangle ^ { p } .\tag{14}
$$

The smoothness and moment conditions in Assumption 1 are comparable to conditions commonly used in the classical stochastic-approximation literature; see, for example, Kushner and Yin [2003] and Benveniste et al. [1990].

These are the primitive conditions used to obtain the finite-window comparison. They do not include the global moment bound $M _ { p } < \infty$ , which enters only in the all-iteration conclusion of the theorem. Section 3.3 gives an SDE-based sufficient condition for this bound.

## 3.2 Main rate-transfer theorem

The following is the main result of the paper.

Theorem 2 (Rate Transfer Theorem). Under Assumption 1, for each fixed $T > 0$ and indices $m \leq n \leq N _ { T } ( m )$ satisfying $\mathbb { E } \langle \langle X _ { m } \rangle \rangle ^ { p } < \infty ,$

$$
| \mathbb { E } V ( X _ { n } ) - \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( X _ { m } ) ] | \leq C _ { T } \mathbb { E } \langle X _ { m } \rangle ^ { p } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .\tag{15}
$$

If, in addition, $\begin{array} { r } { M _ { p } : = \operatorname* { s u p } _ { k > 0 } \mathbb { E } \langle \langle X _ { k } \rangle \rangle ^ { p } < \infty } \end{array}$ (or equivalently su $\mathrm { p } _ { k > 0 } \mathbb { E } \| X _ { k } \| ^ { p } < \infty )$ and the associated SDE satisfies, for deterministic nonnegative locally integrablefunctions λ and g with $0 \leq \lambda ( t ) \leq \bar { \lambda } < \infty$

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

With $\begin{array} { r } { R ( s , t ) = \exp \{ - \int _ { s } ^ { t } \lambda ( r ) d r \} } \end{array}$ , for every $n \geq 0$

$$
\mathbb { E } V ( X _ { n } ) \leq R ( 0 , t _ { n } ) \mathbb { E } V ( X _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } R ( s , t _ { n } ) \{ g ( s ) + C _ { T } \alpha ( s ) \} d s ,\tag{16}
$$

where $C _ { T } < \infty$ is independent of n and may depend on the fixed-window and moment constants, including $M _ { p }$

The proof combines the primitive fixed-window comparison with the restart argument; it is given in Appendix B. The additional stability hypothesis $M _ { p } < \infty$ is addressed in Section 3.3.

## 3.3 Stability from the associated SDE

In addition to the primitive finite-window assumptions and the SDE Lyapunov inequality, Theorem 2 requires the global moment bound

$$
M _ { p } : = \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \langle ( X _ { n } ) \rangle ^ { p } < \infty .
$$

This is a stability property of the stochastic approximation. We now show how it can be established using a Foster– Lyapunov inequality for the associated SDE.

As with the Transfer Theorem, we follow the same principle here: we establish a Foster–Lyapunov inequality for the associated SDE using generator calculus, then transfer this stability to the stochastic-approximation iterates. The following proposition formalizes this step.

Proposition 3 (Foster–Lyapunov Stability Transfer). Suppose the stochastic approximation and associated SDE satisfy Assumption 1 with moment exponent $p \geq 2 .$ . Let $\Phi \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ satisfy $\Phi \geq 1$ and,for a constant $C _ { \Phi } < \infty$

$$
\begin{array} { r } { \langle x \rangle ^ { p } \leq C _ { \Phi } \Phi ( { \pmb x } ) , \qquad \| \nabla \Phi ( { \pmb x } ) \| \leq C _ { \Phi } \langle \langle { \pmb x } \rangle ^ { p - 1 } , \qquad \| \nabla ^ { 2 } \Phi ( { \pmb x } ) \| \leq C _ { \Phi } \langle \langle { \pmb x } \rangle ^ { p - 2 } . } \end{array}
$$

Assume also that $\mathbb { E } \Phi ( X _ { 0 } ) < \infty$ and that, for some $\gamma > 0 , K \geq 0$ , and finite index $N _ { 0 } ,$ the following bound holds uniformly for $t \geq t _ { N _ { 0 } }$ and x $\in \mathbb { R } ^ { d }$

$$
\mathcal L _ { t } \Phi ( { \boldsymbol x } ) \le - \gamma \Phi ( { \boldsymbol x } ) + K .
$$

There is a sufficiently small $\bar { \alpha } > 0 ,$ , depending only on the constants in these assumptions, such that ifthere is afinite index $N \geq N _ { 0 } f o r$ which $\alpha _ { n } < \bar { \alpha } f o r$ every $n \geq N$ , then

$$
\operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \Phi ( X _ { n } ) < \infty , \qquad M _ { p } : = \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \langle X _ { n } \rangle ^ { p } < \infty .
$$

The stability function Φ need not coincide with the rate Lyapunov function V in Theorem 2.

Remark 4 (Power lifting trick). In applications, the natural SDE Lyapunov function often controls a lower-order moment than the one required by Theorem 2. For example, if W is quadratic, so that $1 + W ( { \pmb x } ) \asymp \langle \langle { \pmb x } \rangle \rangle ^ { 2 }$ , but a p-th moment is required, we use the power-lifting trick and set

$$
\Phi ( { \pmb x } ) = ( 1 + W ( { \pmb x } ) ) ^ { p / 2 } .
$$

More generally, $\mathrm { i f ~ } 1 + W ( \pmb { x } ) \asymp \langle \langle \pmb { x } \rangle \rangle ^ { p _ { 0 } }$ , we take $\Phi ( { \pmb x } ) = ( 1 + W ( { \pmb x } ) ) ^ { p / p _ { 0 } }$ . Under the standing regularity assumptions, the generator chain rule shows that a Foster–Lyapunov bound for W gives a corresponding bound for Φ once the stepsizes are sufficiently small. Since $\Phi ( { \pmb x } ) \asymp \langle \langle { \pmb x } \rangle \rangle ^ { p }$ , Proposition 3 then gives the required p-th moment bound. A short verification of this power-lifting step is given in Lemma 18.

If W controls a lower-order moment of order $p _ { 0 } < p ,$ , while Theorem 2 requires order $p ,$ the power-lifting trick produces a suitable auxiliary function Φ. Proposition 3 then establishes the required uniform p-th moment bound:

$$
M _ { p } : = \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \langle ( X _ { n } ) \rangle ^ { p } < \infty .
$$

## 4 Applying the rate-transfer method

This section gives two applications of the rate-transfer theorem: strongly convex stochastic gradient descent and stochastic Heavy Ball.

## 4.1 How to apply the method

Theorem 2 reduces an application of the rate-transfer method to three model-specific steps.

1. Construct the associated SDE. Identify the conditional drift b and covariance $A = \sigma \sigma ^ { \top }$ of the stochasticapproximation increment. The associated SDE is

$$
d \pmb { Z } _ { t } = \pmb { b } ( \pmb { Z } _ { t } ) d t + \sqrt { \alpha ( t ) } \sigma ( \pmb { Z } _ { t } ) d \pmb { B } _ { t } .
$$

Over a time interval of length $\alpha _ { n } ,$ its drift and covariance match those of one stochastic-approximation step to leading order.

2. Establish the SDE Lyapunov bound. Choose a nonnegative Lyapunov function $V$ that measures the error or energy of interest. Its generator is

$$
\mathcal { L } _ { t } V ( \pmb { x } ) = \pmb { b } ( \pmb { x } ) ^ { \top } \nabla V ( \pmb { x } ) + \frac { \alpha ( t ) } { 2 } A ( \pmb { x } ) : \nabla ^ { 2 } V ( \pmb { x } ) .
$$

Establish, for nonnegative functions λ and $^ { g , }$ the bound

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

Verify the primitive regularity and moment assumptions of Theorem 2. If the required uniform moment $M _ { p } <$ ∞ is not immediate, obtain it from a Foster–Lyapunov inequality for the associated SDE (Proposition 3), using the power-lifting trick when a higher moment is needed. The primitive assumptions and this moment bound give the finite-window comparison needed for the rate theorem.

3. Apply the rate-transfer theorem. Theorem 2 gives

$$
\mathbb { E } V ( X _ { n } ) \leq e ^ { - \int _ { 0 } ^ { t _ { n } } \lambda ( r ) d r } \mathbb { E } V ( X _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } e ^ { - \int _ { s } ^ { t _ { n } } \lambda ( r ) d r } \{ g ( s ) + C _ { T } \alpha ( s ) \} d s .
$$

The SDE supplies the contraction through the exponential factors and the forcing through $^ { g ; }$ the stochastic approximation contributes the additional $C _ { T } \alpha ( s )$ forcing.

The following examples illustrate these three steps.

## 4.2 Strongly convex stochastic gradient descent

To illustrate Theorem 2, we consider stochastic gradient descent (SGD) for a strongly convex function. Let $\scriptstyle { \pmb x } ^ { \star }$ denote the unique minimizer of f. Viewed as a stochastic approximation, the SGD update is:

$$
{ \cal X } _ { k + 1 } = { \cal X } _ { k } - \alpha _ { k } \{ \nabla f ( { \cal X } _ { k } ) + \xi _ { k + 1 } \} .
$$

We assume $f \in C ^ { 4 } (  { \mathbb { R } } ^ { d } )$ is µ-strongly convex with L-Lipschitz gradient, and that there exist $\kappa \geq 0$ and $C <$ ∞ such that $\| D ^ { 3 } f ( { \pmb x } ) \| \leq C \langle \langle { \pmb x } \rangle \rangle ^ { \kappa }$ for $\pmb { x } \in \mathbb { R } ^ { d } .$ Set $p : = \kappa + 3$ . The noise satisfies $\mathbb { E } [ \pmb { \xi } _ { k + 1 } \ | \ \mathcal { F } _ { k } ] = 0$ and $\mathbb { E } [ \pmb { \xi } _ { k + 1 } \pmb { \xi } _ { k + 1 } ^ { \top } \mid \mathcal { F } _ { k } ] = \Sigma .$ together with $\mathbb { E } [ \| \pmb { \xi } _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] \le \nu _ { p }$ . These assumptions are standard in nonasymptotic analyses of strongly convex SGD; see, for example, Moulines and Bach [2011]. We also assume $\mathbb { E } \langle \langle X _ { 0 } \rangle \rangle ^ { p } < \infty$ and nonincreasing stepsizes with $\alpha _ { n }  0$ . Put

$$
b ( \pmb { x } ) = - \nabla f ( \pmb { x } ) , \qquad \sigma = \Sigma ^ { 1 / 2 } , \qquad V ( \pmb { x } ) = \| \pmb { x } - \pmb { x } ^ { \star } \| ^ { 2 } .
$$

This gives the SDE

$$
d \pmb { Z } _ { t } = - \nabla f ( \pmb { Z } _ { t } ) d t + \sqrt { \alpha ( t ) } \Sigma ^ { 1 / 2 } d \pmb { B } _ { t } .
$$

Generator estimate. The generator calculation is

$$
\begin{array} { r l } & { \qquad \mathscr { L } _ { t } V ( { \pmb x } ) = - 2 ( { \pmb x } - { \pmb x } ^ { \star } ) ^ { \top } \nabla f ( { \pmb x } ) + \alpha ( t ) \mathrm { t r } ( { \pmb \Sigma } ) } \\ & { \qquad \le - 2 \mu V ( { \pmb x } ) + \mathrm { t r } ( { \pmb \Sigma } ) \alpha ( t ) . } \end{array}
$$

Strong convexity and $\nabla f ( { \pmb x } ^ { \star } ) = 0$ give the inequality. Thus $\lambda = 2 \mu$ and $g ( t ) = \mathrm { t r } ( \Sigma ) \alpha ( t )$ Theorem assumptions. Since $f \in C ^ { 4 }$ and $\nabla f$ is L-Lipschitz, $\pmb { b } = - \nabla f$ $C ^ { 3 }$ and globally Lipschitz; σ is constant, and $D ^ { 2 } \pmb { b } = - \bar { D ^ { 3 } } f$ has polynomial growth of order κ. For quadratic $V ,$ , the derivative-growth bound holds with $v = 1$ Writing $H _ { k + 1 } = - \nabla f ( X _ { k } ) - \xi _ { k + 1 }$ , the Lipschitz-gradient bound and the conditional noise moment give

$$
\begin{array} { r l } & { \mathbb { E } [ \| H _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] \leq C _ { p } \left( \| \nabla f ( \mathbf { \mathbf { X } } _ { k } ) \| ^ { p } + \mathbb { E } [ \| \pmb { \xi } _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] \right) } \\ & { \qquad \leq C \langle \langle \mathbf { X } _ { k } \rangle \rangle ^ { p } . } \end{array}
$$

Thus the primitive oracle moment condition holds. Since $p = \kappa + 3 = v + \kappa + 2 .$ , Assumption 1 is verified.

Since $\mathbf { \Delta } \mathbf { x } ^ { \star }$ is fixed, $1 + V ( \pmb { x } ) \asymp \langle \pmb { x } \rangle \rangle ^ { 2 }$ . The quadratic function V satisfies the derivative-growth hypotheses of Lemma 18 with $p _ { 0 } = 2$ . Moreover, nonincreasing stepsizes give $\alpha ( t ) \leq \alpha _ { 0 }$ , so the generator estimate yields

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - 2 \mu V ( \pmb { x } ) + \mathrm { t r } ( \Sigma ) \alpha _ { 0 } , } \end{array}
$$

a lower-order Foster–Lyapunov inequality. Set $\Phi ( { \pmb x } ) = ( 1 + V ( { \pmb x } ) ) ^ { p / 2 }$ . Lemma 18, applied with $W = V$ and $p _ { 0 } = 2$ gives the p-order SDE drift inequality when $\alpha ( t )$ is sufficiently small. Since $\alpha _ { n }  0$ , choose N so that this smallness and the stepsize threshold in Proposition 3 hold on the tail. The comparison $\Phi ( { \pmb x } ) \asymp \langle \langle { \pmb x } \rangle \rangle ^ { p }$ shows that the assumed

initial moment implies $\mathbb { E } \Phi ( X _ { 0 } ) < \infty$ . Proposition 3, whose primitive assumptions were verified above, then yields the required global moment bound

$$
M _ { p } : = \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \langle ( X _ { k } ) \rangle ^ { p } < \infty .
$$

Transfer. Theorem 2 gives

$$
\mathbb { E } \| X _ { n } - x ^ { \star } \| ^ { 2 } \leq e ^ { - 2 \mu t _ { n } } \mathbb { E } \| X _ { 0 } - x ^ { \star } \| ^ { 2 } + C \int _ { 0 } ^ { t _ { n } } e ^ { - 2 \mu ( t _ { n } - s ) } \alpha ( s ) d s .
$$

Here C absorbs the SDE noise forcing and the finite-window comparison constant.

Rate calculationfor power-law stepsizes. For $\alpha _ { k } = a / ( k + n _ { 0 } ) ^ { \vartheta }$ with $0 < \vartheta < 1$ , apply Corollary 17 with $\lambda = 2 \mu$ and $\delta = 1$ . It gives

$$
\operatorname { \mathbb { E } } \| X _ { n } - x ^ { \star } \| ^ { 2 } = O { \Big ( } e ^ { - c ( n + n _ { 0 } ) ^ { 1 - \vartheta } } + ( n + n _ { 0 } ) ^ { - \vartheta } { \Big ) } = O ( n ^ { - \vartheta } ) , \qquad c = { \frac { 2 \mu a } { 1 - \vartheta } } .
$$

At $\vartheta = 1$ , the schedule corollary gives the usual three harmonic regimes according as $2 \mu a$ is greater than, equal to, or less than 1.

Discussion. The rate agrees with classical strongly convex SGD analyses such as Moulines and Bach [Mouline and Bach, 2011]; Fontaine et al. [Fontaine et al., 2021] give a related continuous-time treatment. Here it follows by transferring the SDE Lyapunov calculation.

## 4.3 Stochastic Heavy Ball

We consider the stochastic Heavy Ball recursion

$$
{ \bf X } _ { n + 1 } = { \bf X } _ { n } + h { \bf V } _ { n } , \qquad { \bf V } _ { n + 1 } = { \bf V } _ { n } - h \{ \eta { \bf V } _ { n } + \nabla f ( { \bf X } _ { n } ) + \zeta _ { n + 1 } \} .
$$

Let $f \in C ^ { 4 } (  { \mathbb { R } } ^ { d } )$ be µ-strongly convex with L-Lipschitz gradient and bounded third derivative, and set $\eta = \sqrt { \mu }$ . Assume that $\zeta _ { n + 1 }$ is conditionally centred with covariance $\sigma ( X _ { n } , V _ { n } ) \sigma ( X _ { n } , V _ { n } ) ^ { \top }$ , where $\sigma : \mathbb { R } ^ { 2 d }  \mathbb { R } ^ { d \times q }$ is $C ^ { 3 }$ with bounded first and second derivatives and $\mathrm { t r } ( \sigma \sigma ^ { \top } ) \leq \varsigma ^ { 2 }$ . Suppose

$$
\begin{array} { r } { \mathbb { E } [ \| \zeta _ { n + 1 } \| ^ { 3 } \mid \mathcal { F } _ { n } ] \le C \{ 1 + \| { \cal { X } } _ { n } - { \pmb x } ^ { \star } \| ^ { 3 } + \| { \cal { V } } _ { n } \| ^ { 3 } \} , } \end{array}
$$

and take h sufficiently small.

Write $e = x - x ^ { \star }$ and define

$$
\begin{array} { r } { W ( \pmb { x } , \pmb { v } ) = f ( \pmb { x } ) - f ( \pmb { x } ^ { \star } ) + \frac { 1 } { 2 } \| \pmb { v } \| ^ { 2 } + \frac { \eta } { 2 } \pmb { e } ^ { \top } \pmb { v } + \frac { \mu } { 4 } \| \pmb { e } \| ^ { 2 } . } \end{array}
$$

Completing the square and using strong convexity and smoothness give

$$
\begin{array} { r l } & { W ( { \pmb x } , { \pmb v } ) = f ( { \pmb x } ) - f ( { \pmb x } ^ { \star } ) + \frac { 1 } { 2 } \| { \pmb v } + \frac { \eta } { 2 } { \pmb e } \| ^ { 2 } + \frac { \mu } { 8 } \| { \pmb e } \| ^ { 2 } } \\ & { \qquad \geq \frac { \mu } { 2 } \| { \pmb e } \| ^ { 2 } + \frac { 1 } { 4 } \| { \pmb v } \| ^ { 2 } . } \end{array}
$$

Consequently,

$$
1 + W ( { \pmb x } , { \pmb v } ) \asymp \langle ( e , { \pmb v } ) \rangle ^ { 2 } , \qquad \| { \nabla W ( { \pmb x } , { \pmb v } ) } \| \leq C \langle ( e , { \pmb v } ) \rangle , \qquad \| D ^ { 2 } W ( { \pmb x } , { \pmb v } ) \| \leq C .
$$

Assume also the initial moment condition $\mathbb { E } ( 1 + W ( X _ { 0 } , V _ { 0 } ) ) ^ { 3 / 2 } < \infty .$

With $\alpha ( t ) = h$ , this gives the associated SDE

$$
d X _ { t } = V _ { t } d t , \qquad d V _ { t } = - \{ \eta V _ { t } + \nabla f ( X _ { t } ) \} d t + \sqrt { \alpha ( t ) } \sigma ( X _ { t } , V _ { t } ) d B _ { t } .
$$

Generator estimate. For $b _ { \mathrm { H B } } ( { \pmb x } , { \pmb v } ) = ( { \pmb v } , - \eta { \pmb v } - \nabla f ( { \pmb x } ) )$ ,

$$
\begin{array} { r } { \nabla W ( \pmb { x } , \pmb { v } ) ^ { \top } b _ { \mathrm { H B } } ( \pmb { x } , \pmb { v } ) = - \frac { \eta } { 2 } \| \pmb { v } \| ^ { 2 } - \frac { \eta } { 2 } \pmb { e } ^ { \top } \nabla f ( \pmb { x } ) . } \end{array}
$$

Since $\eta ^ { 2 } = \mu .$ , the generator calculation gives

$$
\begin{array} { r } { \mathcal { L } _ { t } W + \frac { \eta } { 2 } W \le - \frac { \eta } { 8 } \{ \| v \| ^ { 2 } + \| v - \eta e \| ^ { 2 } \} + \frac { \alpha ( t ) } { 2 } \operatorname { t r } ( \sigma \sigma ^ { \top } ) . } \end{array}
$$

Thus

$$
\begin{array} { l } { \displaystyle { \mathcal { L } _ { t } W \leq - \lambda _ { \mathrm { H B } } W + \frac { 1 } { 2 } \varsigma ^ { 2 } \alpha ( t ) , \qquad \lambda _ { \mathrm { H B } } = \frac { \sqrt { \mu } } { 2 } . } } \end{array}
$$

Theorem assumptions. Set $Y _ { n } = \left( X _ { n } - \pmb { x } ^ { \star } , V _ { n } \right)$ . In these coordinates, the drift and diffusion factor are

$$
\widetilde { b } _ { \mathrm { H B } } ( e , v ) = ( v , - \eta v - \nabla f ( x ^ { \star } + e ) ) , \qquad \sigma _ { \mathrm { H B } } ( e , v ) = \binom { 0 _ { d \times q } } { \sigma ( x ^ { \star } + e , v ) } .
$$

The drift is $C ^ { 3 }$ and globally Lipschitz. Bounded $D ^ { 3 } f$ makes its second derivative bounded, so $\kappa = 0$ . The diffusion factor is $C ^ { 3 }$ , globally Lipschitz, and has bounded second derivatives. The bounds on ∇W and $D ^ { 2 } W$ above give the Lyapunov growth exponent $v = 1$

Let $\pmb { \varepsilon } _ { n + 1 } = ( \mathbf { 0 } , - \pmb { \zeta } _ { n + 1 } ) ;$ ; it is conditionally centred with covariance ${ \pmb \sigma } _ { \mathrm { H B } } ( { \pmb Y } _ { n } ) { \pmb \sigma } _ { \mathrm { H B } } ( { \pmb Y } _ { n } ) ^ { \top }$ . Set ${ \pmb H } _ { n + 1 } = \widetilde { \pmb b } _ { \mathrm { H B } } ( { \pmb Y } _ { n } ) +$ $\pmb { \varepsilon } _ { n + 1 } , \mathbf { s o } \mathbf { Y } _ { n + 1 } = \mathbf { Y } _ { n } + h \pmb { H } _ { n + 1 }$ . Linear growth of the drift and the conditional third-moment bound give

$$
\begin{array} { r l } & { \mathbb { E } [ \| \pmb { H } _ { n + 1 } \| ^ { 3 } \mid \mathcal { F } _ { n } ] \leq C \left( \| \widetilde { \pmb { b } } _ { \mathrm { H B } } ( \pmb { Y } _ { n } ) \| ^ { 3 } + \mathbb { E } [ \| \pmb { \zeta } _ { n + 1 } \| ^ { 3 } \mid \mathcal { F } _ { n } ] \right) } \\ & { \qquad \leq C \langle \langle \pmb { Y } _ { n } \rangle \rangle ^ { 3 } . } \end{array}
$$

Thus $v = 1 , \kappa = 0$ , and $p = 3 = v + \kappa + 2 .$ , so Assumption 1 holds.

The lower bound on W gives $1 + W \times \langle Y \rangle \rangle ^ { 2 }$ , and the generator estimate gives $\mathcal { L } _ { t } W \le - \lambda _ { \mathrm { H B } } W + \varsigma ^ { 2 } h _ { 0 } / 2$ uniformly for $0 < h \leq h _ { 0 }$ . The derivative bounds above meet the hypotheses of Lemma 18 with $p _ { 0 } = 2$ and $p = 3$ . Set $\Phi = ( 1 + W ) ^ { 3 / 2 }$ . For sufficiently small $h _ { 0 }$ , that lemma gives $\mathcal { L } _ { t } \Phi \le - c \Phi + C$ uniformly over $0 < h \leq h _ { 0 }$ . The initial moment assumption gives $\mathbb { E } \Phi ( Y _ { 0 } ) < \infty$ . Since the primitive conditions have already been verified, choose $h _ { 0 }$ below the stability-transfer threshold and apply Proposition 3 with $N = 0$ . It follows that

$$
\operatorname* { s u p } _ { n \geq 0 } \mathbb { E } ( 1 + W ( X _ { n } , V _ { n } ) ) ^ { 3 / 2 } < \infty , \qquad M _ { 3 } : = \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \langle \pmb { Y } _ { n } \rangle ^ { 3 } < \infty .
$$

Transfer. For constant h, we have $\alpha ( t ) = h , t _ { n } = n h , \lambda ( t ) = \lambda _ { \mathrm { H B } }$ , and $g ( t ) = \varsigma ^ { 2 } h / 2$ . Theorem 2 gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } W ( X _ { n } , V _ { n } ) \leq e ^ { - \lambda _ { \mathrm { H B } } h n } \mathbb { E } W ( X _ { 0 } , V _ { 0 } ) } \\ { \displaystyle \qquad + \int _ { 0 } ^ { h n } e ^ { - \lambda _ { \mathrm { H B } } ( h n - s ) } \{ \varsigma ^ { 2 } h / 2 + C _ { T } h \} d s } \\ { \displaystyle \qquad \leq e ^ { - \lambda _ { \mathrm { H B } } h n } \mathbb { E } W ( X _ { 0 } , V _ { 0 } ) + C h . } \end{array}
$$

The constant $C$ is independent of n and can be chosen uniformly over $0 < h \leq h _ { 0 }$ , since the primitive constants and the bound $M _ { 3 }$ are uniform over this range. The same energy bounds give

$$
\mathbb { E } [ f ( X _ { n } ) - f ( { \pmb x } ^ { \star } ) ] \le \mathbb { E } W ( X _ { n } , V _ { n } ) , \qquad \mathbb { E } \| X _ { n } - { \pmb x } ^ { \star } \| ^ { 2 } \le \frac { 2 } { \mu } \mathbb { E } W ( X _ { n } , V _ { n } ) .
$$

Thus the transient is geometric in iteration count, and persistent noise leaves an $O ( h )$ neighbourhood.

## 5 Conclusion and extensions

We have shown how a finite-time convergence estimate for an associated SDE yields a convergence estimate for a stochastic approximation. The method uses weak comparison only on fixed diffusion-time windows. Restarting the continuous comparator carries its contraction across successive windows, so no growing-horizon SDE approximation i needed. The finite-window error enters the transferred estimate as an additional forcing term.

The two applications do not exhaust the scope of the theorem. The same proof template can be applied, once its moment and regularity conditions are verified, $\mathrm { e . g . }$ to monotone Robbins–Monro schemes, contractive fixed-point recursions, stable linear stochastic approximation, and other optimization recursions with a suitable finite-dimensional SDE Lyapunov function. There are a host of other applications beyond these. We omit these further examples to keep the paper focused on the transfer mechanism.

Other directions require extensions of the theorem rather than only new applications. These include more general Lyapunov dissipation inequalities, higher-order stochastic modified equations, path space and hitting time quantities, Markovian noise, adaptive clocks, and two-timescale recursions. These questions require additional arguments and are outside the first-order rate transfer result studied here. The main proof steps are the same however. Again, the focus of this paper is on the transfer mechanism itself, since it contains the core ideas, and we leave these extensions to future work.

Acknowledgment. AI tools were used to assist in the development and checking of mathematical arguments in this paper. The author takes full responsibility for the correctness, originality and presentation of the work.

## References

Michel Bena¨ım. A dynamical system approach to stochastic approximations. SIAM Journal on Control and Optimization, 34(2):437–472, 1996. doi: 10.1137/S0363012993253534.

Albert Benveniste, Michel Metivier, and Pierre Priouret.´ Adaptive Algorithms and Stochastic Approximations, volume 22 of Stochastic Modelling and Applied Probability. Springer, Berlin, 1990. doi: 10.1007/978-3-642-75894-2.

V. S. Borkar and S. K. Mitter. A strong approximation theorem for stochastic recursive algorithms. Journal of Optimization Theory and Applications, 100(3):499–513, 1999. doi: 10.1023/A:1022630321574.

Vivek S. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint, volume 48 of Texts and Readings in Mathematics. Hindustan Book Agency, New Delhi, 2008. doi: 10.1007/978-93-86279-38-5.

J. G. Dai. On positive Harris recurrence of multiclass queueing networks: A unified approach via fluid limit models. The Annals of Applied Probability, 5(1):49–77, 1995. doi: 10.1214/aoap/1177004828.

Wei Fang and Michael B. Giles. Adaptive euler–maruyama method for SDEs with nonglobally lipschitz drift. The Annals ofApplied Probability, 30(2):526–560, 2020. doi: 10.1214/19-AAP1507.

Yuanyuan Feng, Tingran Gao, Lei Li, Jian-Guo Liu, and Yulong Lu. Uniform-in-time weak error analysis for stochastic gradient descent algorithms via diffusion approximation. Communications in Mathematical Sciences, 18(1):163–188, 2020. doi: 10.4310/CMS.2020.v18.n1.a7.

Xavier Fontaine, Valentin De Bortoli, and Alain Durmus. Convergence rates and approximation results for SGD and its continuous-time counterpart. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings ofMachine Learning Research, pages 1965–2058. PMLR, 2021. URL https://proceedings.mlr.press/v134/fontaine21a.html.

Wenqing Hu, Chris Junchi Li, Lei Li, and Jian-Guo Liu. On the diffusion approximation of nonconvex stochastic gradient descent. Annals ofMathematical Sciences and Applications, 4(1):3–32, 2019. doi: 10.4310/amsa.2019.v4.n1.a1.

Peter E. Kloeden and Eckhard Platen. Numerical Solution ofStochastic Differential Equations, volume 23 of Stochastic Modelling and Applied Probability. Springer, Berlin, Heidelberg, 1992. doi: 10.1007/978-3-662-12616-5.

Hiroshi Kunita. Stochastic Flows and Stochastic Differential Equations, volume 24 of Cambridge Studies in Advanced Mathematics. Cambridge University Press, 1990.

Harold J. Kushner. General convergence results for stochastic approximations via weak convergence theory. Journal of Mathematical Analysis and Applications, 61(2):490–503, 1977. doi: 10.1016/0022-247X(77)90133-0.

Harold J. Kushner and Dean S. Clark. Stochastic Approximation Methods for Constrained and Unconstrained Systems, volume 26 of Applied Mathematical Sciences. Springer, New York, 1978. doi: 10.1007/978-1-4684-9352-8.

Harold J. Kushner and Hai Huang. Rates of convergence for stochastic approximation type algorithms. SIAM Journal on Control and Optimization, 17(5):607–617, 1979. doi: 10.1137/0317043.

Harold J. Kushner and Hai Huang. On the weak convergence of a sequence of general stochastic difference equations to a diffusion. SIAM Journal on Applied Mathematics, 40(3):528–541, 1981. doi: 10.1137/0140044.

Harold J. Kushner and G. George Yin. Stochastic Approximation and Recursive Algorithms and Applications, volume 35 of Stochastic Modelling and Applied Probability. Springer, New York, 2 edition, 2003. doi: 10.1007/b97441.

Lei Li and Yuliang Wang. On uniform-in-time diffusion approximation for stochastic gradient descent. Methods and Applications ofAnalysis, 30(3):95–112, 2023. doi: 10.4310/maa.2023.v30.n3.a1.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and adaptive stochastic gradient algorithms. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2101–2110. PMLR, 2017. URL https://proceedings.mlr.press/v70/li17f. html.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and dynamics of stochastic gradient algorithms i: Mathematical foundations. Journal of Machine Learning Research, 20(40):1–47, 2019.

Lennart Ljung. Analysis of recursive stochastic algorithms. IEEE Transactions on Automatic Control, 22(4):551–575, 1977. doi: 10.1109/TAC.1977.1101561.

Hiroyuki Matsumoto and Setsuo Taniguchi. Stochastic Analysis: Ito and Malliavin Calculus in Tandem ˆ , volume 159 of Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2016. doi: 10.1017/9781316492888.

Michel Metivier and Pierre Priouret. Applications of a kushner and clark lemma to general classes of stochastic´ algorithms. IEEE Transactions on Information Theory, 30(2):140–151, 1984. doi: 10.1109/TIT.1984.1056894.

Eric Moulines and Francis R. Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems 24, pages 451–459. Curran Associates, Inc., 2011.

Bernt Øksendal. Stochastic Differential Equations: An Introduction with Applications. Universitext. Springer, Berlin, 6 edition, 2003. doi: 10.1007/978-3-642-14394-6.

Antonio Orvieto and Aurelien Lucchi. Continuous-time models for stochastic optimization algorithms. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://papers.neurips.cc/paper\_files/paper/2019/hash/ 9cd78264cf2cd821ba651485c111a29a-Abstract.html.

H. Pezeshki-Esfahani and A. J. Heunis. Strong diffusion approximations for recursive stochastic algorithms. IEEE Transactions on Information Theory, 43(2):512–523, 1997. doi: 10.1109/18.556109.

B. T. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, 1964. doi: 10.1016/0041-5553(64)90137-5.

Andrew M. Stuart. Numerical analysis of dynamical systems. Acta Numerica, 3:467–572, 1994. doi: 10.1017/ S0962492900002488.

S. R. S. Varadhan. Stochastic differential equations. Chapter 3 of lecture notes for Stochastic Analysis, Courant Institute of Mathematical Sciences, New York University, 2006. URL https://math.nyu.edu/ varadhan/ fall06/fall06.3.pdf.

## A Finite-window comparison

This appendix establishes the primitive finite-window comparison from the restarted-SDE estimates and the one-step weak comparison.

## A.1 Additional notation and conventions

We retain the notation introduced in Section 2.1 and record here several additional conventions used in the proofs.

More generally, if $A : ( \mathbb { R } ^ { d } ) ^ { k }  \mathbb { R } ^ { m }$ is a k-linear map, we write $A [ \pmb { v } _ { 1 } , \dots , \pmb { v } _ { k } ]$ for its evaluation at $( \pmb { v } _ { 1 } , \dots , \pmb { v } _ { k } )$ For nonnegative quantities f and g, we write $f \asymp _ { p } g$ if there exist constants $0 < c _ { p } \le C _ { p } < \infty .$ , depending only on $p ,$ such that $c _ { p } g \le f \le C _ { p } g$ . For $p \geq 0$ , the polynomial weight satisfies

$$
\begin{array} { r } { \langle \langle \pmb { x } \rangle \rangle ^ { p } \asymp _ { p } 1 + \| \pmb { x } \| _ { 2 } ^ { p } . } \end{array}
$$

We repeatedly use

$$
\begin{array} { r } { \langle \langle { \pmb x } \rangle ) ^ { r } \langle \langle { \pmb x } \rangle \rangle ^ { s } = \langle \langle { \pmb x } \rangle \rangle ^ { r + s } , \qquad r , s \geq 0 , } \end{array}
$$

and

$$
\begin{array} { r } { \langle \langle { \pmb x } + { \pmb y } \rangle \rangle ^ { r } \leq C _ { r } \left\{ \langle \langle { \pmb x } \rangle \rangle ^ { r } + \| { \pmb y } \| _ { 2 } ^ { r } \right\} , \qquad r \geq 0 . } \end{array}\tag{17}
$$

Since $\langle \langle \pmb { x } \rangle \rangle \geq 1$ , if $0 \leq r \leq s$ then $\langle \langle { \pmb x } \rangle \rangle ^ { r } \leq \langle \langle { \pmb x } \rangle \rangle ^ { s } .$

Correspondingly, $f = O _ { T } ( g )$ means that $| f | \le C _ { T } g$ with this same convention.

The symbols $\gamma , \gamma _ { 1 } , \gamma _ { 2 } , . . .$ . denote finite nonnegative exponents whose values are local to the statement or argument in which they are introduced. Their reuse elsewhere in the paper does not imply that they have the same value.

Here and below, an empty product is understood to equal 1.

## A.2 First-order local-to-window approximation

The transfer theorem, Theorem 13, takes two components: the Lyapunov bound (71) and the finite window approximation (72), and transfers them to a finite time bound for the stochastic approximation.

The Lyapunov bound is a generator inequality for the associated SDE. When an appropriate Lyapunov function is available, it can typically be checked directly by standard calculus from the drift and diffusion coefficients. In contrast, the finite window approximation is a more abstract condition describing the relationship between the stochastic approximation and the SDE over a fixed diffusion time window, and, at first glance, it may not appear as immediately checkable from the model specification. Here we show how the finite window approximation can be verified from basic regularity and moment conditions on the stochastic approximation and SDE. The precise conditions are given below. Throughout this subsection we retain the stochastic approximation and associated-SDE setup of Sections 2.2 and 2.3, together with the polynomial weight notation of Section 2.1.

## A.2.1 Primitive finite-window verification

Remark 5 (On the global Lipschitz assumption). The global Lipschitz condition, together with $\alpha _ { \mathrm { m a x } } < \infty$ , also ensures that the associated SDE is well posed for every restart. Indeed, global Lipschitzness gives linear growth, and the time-inhomogeneous diffusion coefficient $\sqrt { \alpha ( t ) } \sigma ( { \pmb x } )$ is uniformly Lipschitz in x and has linear growth. The standard existence and uniqueness theorem for SDEs therefore gives a unique nonexplosive strong solution on every finite interval; see Øksendal [2003, Theorem 5.2.1].

The global Lipschitz condition is a convenient sufficient condition rather than a minimal one. It gives uniformly bounded linear coefficients in the first variational equation and places the later stochastic-flow estimates in their standard setting. We do not pursue weaker local conditions here. Likewise, the linear growth used in the one-step comparison below is convenient rather than essential: polynomially growing coefficients can be handled under nonexplosion and sufficiently strong fixed-window polynomial moment bounds.

## A.2.2 Primitive finite-window comparison

The body theorem gives the finite-window comparison directly from the primitive assumptions. This appendix proves that comparison and then applies the abstract restart result to obtain the global conclusion. The proof of the primitive comparison is given in Section A.2.3.

The proof is given at the end of Section A.2.3. This proposition supplies the finite-window component of Theorem 2; its global conclusion follows by applying the abstract SDE transfer theorem under the stated Lyapunov bound.

The proof of this result is also given in Section A.2.3.

## A.2.3 Proof of the finite window approximation

The objective of this subsection is to prove Proposition 12, which establishes the finite-window approximation (72) required by Theorem 13 under the standing assumptions.

The proof proceeds through a sequence of results with distinct roles. The Taylor lemma 9 gives the stochasticapproximation one-step remainder for

$$
u _ { k , n } ( \pmb { x } ) = P _ { t _ { k } , t _ { n } } V ( \pmb { x } ) = \mathbb { E } \big [ V \big ( \pmb { Z } _ { t _ { n } } ^ { t _ { k } , \pmb { x } } \big ) \big ] ,
$$

provided that the backward function has polynomial gradient and Hessian bounds. The target bounds are

$$
\| \nabla P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \{ { \pmb x } \} ^ { v } , \qquad \| \nabla ^ { 2 } P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \{ { \pmb x } \} ^ { v + \kappa } .
$$

Thus the Taylor lemma will be applied with $\gamma _ { 1 } = v$ and $\gamma _ { 2 } = v + \kappa .$

These derivative bounds require sensitivity of the restarted SDE to its initial state. The stochastic-flow derivatives J and K satisfy linear stochastic variational equations, and Lemma 6 is the general estimate used to control those equations. The state moments are separated into their own lemma because they have three distinct roles. First, the second-variation forcing contains $D ^ { 2 } b ( Z _ { r } )$ and $D ^ { 2 } \sigma _ { j } ( Z _ { r } )$ , whose growth contributes the factor $\langle \langle Z _ { r } \rangle \rangle ^ { \kappa }$ . Second, differentiating the semigroup introduces $\nabla V ( Z _ { t } )$ and $ { \bar { \nabla } } ^ { 2 } V ( Z _ { t } )$ , whose growth contributes $\langle \langle Z _ { t } \rangle \rangle ^ { v }$ . The state moments also control the displacement and polynomial weights in Lemma 11.

The state-moment lemma supplies the SDE moment input, and the spatial-variation lemma gives the J and K moments and hence the bounds on $\nabla P _ { s , t } V$ and $\nabla ^ { 2 } P _ { s , t } V$ . Lemma 10 justifies Dynkin’s formula for functions with polynomially growing derivatives. The state moments and Dynkin formula give Lemma 11, while the Taylor consequence (40) provides the stochastic-approximation estimate under the oracle moment assumption. Proposition 12 applies both one-step approximations to the backward function. Comparing their common first-order expression gives an $O ( \alpha _ { k } ^ { 2 } )$ ) defect, and summing these conditional defects over the diffusion-time window gives the required finite-window comparison.

The stochastic-flow argument is carried out in the standard setting in which b and σ are globally Lipschitz and $C ^ { 3 }$ with polynomial growth allowed for their second derivatives. The extra derivative gives a $C ^ { 2 }$ solution flow, while the quantitative estimates below use only first- and second-derivative bounds. More general local or one-sided conditions are possible, but concern the regularity theory of the associated SDE rather than the finite-window comparison mechanism considered here. We do not pursue such extensions.

The stochastic-flow references used below are Kunita’s standard treatment, Kunita [1990, Section 4.6], and Varadhan’s parameter differentiation argument, Varadhan [2006, Chapter 3].

The p-moment consequence (40) supplies the stochastic-approximation one-step estimate once the first two derivatives of the backward function are controlled. We now develop the corresponding SDE bounds. The spatial derivatives of the restarted SDE satisfy linear stochastic variation equations, and the following energy estimate is the tool used to control them.

Lemma 6 (An inhomogeneous linear energy estimate). Consider an adapted $\mathbb { R } ^ { d } .$ -valued process Y satisfying

$$
d \pmb { Y _ { t } } = \left( A _ { t } \pmb { Y _ { t } } + \pmb { f _ { t } } \right) d t + \sum _ { j } ( \Sigma _ { j , t } \pmb { Y _ { t } } + \pmb { g _ { j , t } } ) d B _ { t } ^ { j } ,\tag{18}
$$

where $A _ { t } , \Sigma _ { j , t } \in \mathbb { R } ^ { d \times d }$ and $f _ { t } , g _ { j , t } \in \mathbb { R } ^ { d }$ are progressively measurable coefficient processes, and $B ^ { j }$ are real-valued Brownian motions. For matrices, $\| \cdot \|$ denotes the Euclidean operator norm. Suppose, almost surely,

$$
\langle \pmb { y } , A _ { t } \pmb { y } \rangle \leq L \| \pmb { y } \| ^ { 2 } f o r \pmb { y } \in \mathbb { R } ^ { d } , \qquad \sum _ { j } \| \Sigma _ { j , t } \| ^ { 2 } \leq L _ { 1 } ^ { 2 } .\tag{19}
$$

Then, for every $p \geq 2$ and everyfixed $S > 0 ,$ , whenever $t - s \leq S$

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \| Y _ { r } \| ^ { p } \leq C _ { p , S } ^ { Y } \left\{ \mathbb { E } \| Y _ { s } \| ^ { p } + \int _ { s } ^ { t } \mathbb { E } \left[ \| f _ { r } \| ^ { p } + \left( \sum _ { j } \| g _ { j , r } \| ^ { 2 } \right) ^ { p / 2 } \right] d r \right\} .\tag{20}
$$

Here $C _ { p , S } ^ { Y }$ may depend on the fixed constants L and $L _ { 1 } ,$ , but not on an operator-norm bound for $A _ { t } .$

Proof. The argument here is a relatively standard stochastic calculus argument. To prove the lemma, we first introduce a state-localizing sequence and apply Ito’s formula to the stopped process. We then bound the finite-variation andˆ quadratic-variation terms; the latter shows that the stopped stochastic integral is a true martingale. We use the Burkholder– Davis–Gundy inequality and Gronwall’s inequality, and finally remove the stopping by Fatou’s lemma.

We define

$$
G _ { r } : = \Big ( \sum _ { j } \| g _ { j , r } \| ^ { 2 } \Big ) ^ { 1 / 2 } , \qquad Z _ { j , r } : = \Sigma _ { j , r } Y _ { r } + \pmb { g } _ { j , r } .
$$

Thus $f _ { r }$ is the part of the drift independent of ${ Y _ { r } }$ , while ${ \mathbf { } } _ { { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf } _ { } { \mathbf { } } _ { } { \mathbf } _ { } { \mathbf { } } _ { } { \mathbf } _ { } { \mathbf } _ { } { \mathbf } _ { } { \mathbf } _ { } { \mathbf } _ { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ } _ { } _ { } _ { } _ { \mathbf } { \mathbf } _ { } _ { } _ { \mathbf } _ { } _ { } _ { \mathbf } _ { } _ { } \mathbf _ { } _ { } _ { } \mathbf _ { } _ { } _ \mathbf { } _ \mathbf { } _ { } _ \mathbf { } _ \mathbf { } _ \mathbf  $ are the corresponding parts of the diffusion coefficients, and $G _ { r }$ records their combined Euclidean size. We can assume that

$\begin{array} { r } { \mathbb { E } \| \mathbf { Y } _ { s } \| ^ { p } < \infty , \int _ { s } ^ { t } \mathbb { E } \| \mathbf { f } _ { r } \| ^ { p } d r < \infty } \end{array}$ , and $\textstyle \int _ { s } ^ { t } \operatorname { \mathbb { E } } G _ { r } ^ { p } d r < \infty$ . If any of these quantities were infinite, the required bound, (20), would be immediate. Define, for $n \geq 1$

$$
\tau _ { n } : = \operatorname* { i n f } \{ u \in [ s , t ] : \| Y _ { u } \| \geq n \} \wedge t .
$$

We use this state-localizing sequence to justify the martingale estimates below. Since Y has continuous paths on the compact interval $[ s , t ] , \operatorname { s u p } _ { s \leq r \leq t } \| Y _ { r } \| <$ ∞ almost surely, so $\tau _ { n } = t$ for all sufficiently large n almost surely, and hence $\tau _ { n } \uparrow t$ almost surely.

For $u \in [ s , t ]$ , Ito’s formula givesˆ

$$
\begin{array} { l } { \displaystyle \| Y _ { u \wedge \tau _ { n } } \| ^ { p } = \| Y _ { s } \| ^ { p } } \\ { \displaystyle \quad \quad + p \int _ { s } ^ { u \wedge \tau _ { n } } \| Y _ { r } \| ^ { p - 2 } \langle Y _ { r } , A _ { r } , Y _ { r } + f _ { r } \rangle d r } \\ { \displaystyle \quad \quad + \frac { p } { 2 } \int _ { s } ^ { u \wedge \tau _ { n } } \| Y _ { r } \| ^ { p - 2 } \sum _ { j } \| Z _ { j , r } \| ^ { 2 } d r } \\ { \displaystyle \quad \quad + \frac { p ( p - 2 ) } { 2 } \int _ { s } ^ { u \wedge \tau _ { n } } \| Y _ { r } \| ^ { p - 4 } \sum _ { j } \langle Y _ { r } , Z _ { j , r } \rangle ^ { 2 } d r } \\ { \displaystyle \quad \quad \quad + p \sum _ { j } \int _ { s } ^ { u \wedge \tau _ { n } } \| Y _ { r } \| ^ { p - 2 } \langle Y _ { r } , Z _ { j , r } \rangle d B _ { r } ^ { j } , } \end{array}\tag{21}
$$

For $u \in [ s , t ]$ , write $M _ { u }$ for the same stochastic integral with upper limit u. Thus the final term above is $M _ { u \wedge \tau _ { n } }$ , and M is the local-martingale term produced by Ito’s formula. Forˆ $p = 2$ , the term multiplied by $p - 2$ is interpreted as zero.

We now bound the finite variation and the martingale terms above. The finite variation integrand in (21) satisfies

$$
\begin{array} { r l } & { p \| \mathbf { Y } _ { r } \| ^ { p - 2 } \langle \pmb { Y } _ { r } , A _ { r } \pmb { Y } _ { r } + \pmb { f } _ { r } \rangle + \displaystyle \frac { p } { 2 } \| \pmb { Y } _ { r } \| ^ { p - 2 } \displaystyle \sum _ { j } \| \pmb { Z } _ { j , r } \| ^ { 2 } + \displaystyle \frac { p ( p - 2 ) } { 2 } \| \pmb { Y } _ { r } \| ^ { p - 4 } \displaystyle \sum _ { j } \langle \pmb { Y } _ { r } , \pmb { Z } _ { j , r } \rangle ^ { 2 } } \\ & { \le p L \| \pmb { Y } _ { r } \| ^ { p } + p \| \pmb { Y } _ { r } \| ^ { p - 1 } \| \pmb { f } _ { r } \| + \displaystyle \frac { p ( p - 1 ) } { 2 } \| \pmb { Y } _ { r } \| ^ { p - 2 } \displaystyle \sum _ { j } \| \pmb { Z } _ { j , r } \| ^ { 2 } } \\ & { \le p L \| \pmb { Y } _ { r } \| ^ { p } + ( p - 1 ) \| \pmb { Y } _ { r } \| ^ { p } + \| \pmb { f } _ { r } \| ^ { p } + \displaystyle \frac { p ( p - 1 ) } { 2 } \| \pmb { Y } _ { r } \| ^ { p - 2 } \left( 2 L _ { 1 } ^ { 2 } \| \pmb { Y } _ { r } \| ^ { 2 } + 2 G _ { r } ^ { 2 } \right) } \\ & { \le C _ { 1 } \left\{ \| \pmb { Y } _ { r } \| ^ { p } + \| \pmb { f } _ { r } \| ^ { p } + G _ { r } ^ { p } \right\} . } \end{array}
$$

Above, we apply the coefficient bounds in (19) and the Cauchy–Schwarz inequality. We also use the inequality:

$$
\sum _ { j } \| Z _ { j , r } \| ^ { 2 } \leq 2 \sum _ { j } \| \Sigma _ { j , r } { \bf Y } _ { r } \| ^ { 2 } + 2 \sum _ { j } \| { \bf g } _ { j , r } \| ^ { 2 } \leq 2 L _ { 1 } ^ { 2 } \| { \bf Y } _ { r } \| ^ { 2 } + 2 G _ { r } ^ { 2 } .\tag{22}
$$

In the last line, we use Young’s inequality as follows:

$$
p \| { \boldsymbol { Y } } _ { r } \| ^ { p - 1 } \| { \boldsymbol { f } } _ { r } \| \leq ( p - 1 ) \| { \boldsymbol { Y } } _ { r } \| ^ { p } + \| { \boldsymbol { f } } _ { r } \| ^ { p } , \quad { \mathrm { a n d } } \quad \| { \boldsymbol { Y } } _ { r } \| ^ { p - 2 } G _ { r } ^ { 2 } \leq { \frac { p - 2 } { p } } \| { \boldsymbol { Y } } _ { r } \| ^ { p } + { \frac { 2 } { p } } G _ { r } ^ { p } .
$$

This gives the above finite-variation bound.

The resulting finite-variation estimate following from (21) is

$$
\| \mathbf { Y } _ { u \wedge \tau _ { n } } \| ^ { p } \leq \| \mathbf { Y } _ { s } \| ^ { p } + C _ { 1 } \int _ { s } ^ { u \wedge \tau _ { n } } \{ \| \mathbf { Y } _ { r } \| ^ { p } + \| \mathbf { f } _ { r } \| ^ { p } + G _ { r } ^ { p } \} d r + M _ { u \wedge \tau _ { n } } ,\tag{23}
$$

where $C _ { 1 }$ depends only on $p , L , L _ { 1 }$

It remains to justify the martingale term. We first record a quadratic-variation bound that will be used twice: first to show that the stopped local martingale is square-integrable, and later in the BDG estimate.

Define

$$
Y _ { n } ^ { * } ( r ) : = \operatorname* { s u p } _ { s \leq v \leq r } \| Y _ { v \wedge \tau _ { n } } \| ^ { p } .
$$

By continuity and the definition of $\tau _ { n } , Y _ { n } ^ { * } ( r ) \leq n ^ { p } \vee \| Y _ { s } \| ^ { p }$ , and hence $\mathbb { E } Y _ { n } ^ { * } ( r ) < \infty$ . For $r \in [ s , t ]$ , the pathwise quadratic variation satisfies

$$
\begin{array} { l } { \langle M \rangle _ { r \wedge \tau _ { n } } = p ^ { 2 } \int _ { s } ^ { r \wedge \tau _ { n } } \| \mathbf { Y } _ { q } \| ^ { 2 p - 4 } \sum _ { j } \langle \mathbf { Y } _ { q } , Z _ { j , q } \rangle ^ { 2 } d q } \\ { \leq p ^ { 2 } \int _ { s } ^ { r \wedge \tau _ { n } } \| \mathbf { Y } _ { q } \| ^ { 2 p - 2 } \sum _ { j } \| Z _ { j , q } \| ^ { 2 } d q } \\ { \leq C _ { 2 } \int _ { s } ^ { r \wedge \tau _ { n } } \left\{ \| \mathbf { Y } _ { q } \| ^ { 2 p } + \| \mathbf { Y } _ { q } \| ^ { 2 p - 2 } G _ { q } ^ { 2 } \right\} d q } \\ { \leq C _ { 2 } \big ( Y _ { n } ^ { * } ( r ) \wedge n ^ { p } \big ) \int _ { s } ^ { r \wedge \tau _ { n } } \left\{ \| \mathbf { Y } _ { q } \| ^ { p } + \| \mathbf { Y } _ { q } \| ^ { p - 2 } G _ { q } ^ { 2 } \right\} d q . } \end{array}\tag{24}
$$

The first inequality is Cauchy–Schwarz and the second uses (22). For $q < \tau _ { n } , \| Y _ { q } \| ^ { p } \leq n ^ { p }$ , while $\| Y _ { q } \| ^ { p } \leq Y _ { n } ^ { * } ( r )$ which gives the final line. The hitting-time endpoint is irrelevant to the time integral.

For the localization, we now use the crude bounds supplied by $\tau _ { n }$

$$
\begin{array} { r l } & { \mathbb { E } \langle M \rangle _ { t \wedge \tau _ { n } } \leq C _ { 2 } \mathbb { E } \left[ \left( Y _ { n } ^ { * } ( t ) \wedge n ^ { p } \right) \int _ { s } ^ { t \wedge \tau _ { n } } \left\{ \| Y _ { r } | | ^ { p } + \| Y _ { r } \| ^ { p - 2 } G _ { r } ^ { 2 } \right\} d r \right] } \\ & { \qquad \leq C _ { 2 } \left\{ S n ^ { 2 p } + n ^ { 2 p - 2 } \mathbb { E } \int _ { s } ^ { t } G _ { r } ^ { 2 } d r \right\} } \\ & { \qquad \leq C _ { 2 } \left\{ S n ^ { 2 p } + n ^ { 2 p - 2 } S ^ { 1 - 2 / p } \mathbb { E } \left[ \left( \int _ { s } ^ { t } G _ { r } ^ { p } d r \right) ^ { 2 / p } \right] \right\} } \\ & { \qquad \leq C _ { 2 } \left\{ S n ^ { 2 p } + n ^ { 2 p - 2 } S ^ { 1 - 2 / p } \left( \int _ { s } ^ { t } \mathbb { E } G _ { r } ^ { p } d r \right) ^ { 2 / p } \right\} } \\ & { \qquad \leq C _ { 2 } \left\{ S n ^ { 2 p } + n ^ { 2 p - 2 } S ^ { 1 - 2 / p } \left( \int _ { s } ^ { t } \mathbb { E } G _ { r } ^ { p } d r \right) ^ { 2 / p } \right\} } \\ & { \qquad \ll \infty . } \end{array}
$$

The second line uses the localization bounds $Y _ { n } ^ { * } ( t ) \wedge n ^ { p } \leq n ^ { p }$ and $\| \mathbf { Y } _ { r } \| \leq n$ before $\tau _ { n }$ . The next inequality uses Holder’s inequality in time together with¨ $t - s \leq S$ , and the following one uses Jensen’s inequality in probability and Tonelli’s theorem. The finite-right-hand-side assumption gives $\textstyle \int _ { s } ^ { t } \operatorname { \mathbb { E } } G _ { r } ^ { p } d r < \infty$ . Therefore $M . _ { \wedge \tau _ { n } }$ is a square-integrable martingale, hence a true martingale.

By the Burkholder–Davis–Gundy (BDG) inequality and Young’s inequality,

the complete estimate is

$$
\begin{array} { r l } { \underset { s \leq u \leq r } { \mathbb { E } } \ | M _ { u \wedge \tau _ { n } } | \leq C _ { p } \mathbb { E } \langle M \rangle _ { r \wedge \tau _ { n } } ^ { 1 / 2 } } & { } \\ & { \leq C _ { 3 } \mathbb { E } \left[ \left( Y _ { n } ^ { * } ( r ) \wedge n ^ { p } \right) ^ { 1 / 2 } \left( \int _ { s } ^ { r \wedge \tau _ { n } } \left\{ \| Y _ { q } | ^ { p } + \| | Y _ { q } | | ^ { p - 2 } G _ { q } ^ { 2 } \right\} d q \right) ^ { 1 / 2 } \right] } \\ & { \leq C _ { 3 } \mathbb { E } \left[ \left( Y _ { n } ^ { * } ( r ) \right) ^ { 1 / 2 } \left( \int _ { s } ^ { r \wedge \tau _ { n } } \left\{ \| Y _ { q } | ^ { p } + \| | Y _ { q } | | ^ { p - 2 } G _ { q } ^ { 2 } \right\} d q \right) ^ { 1 / 2 } \right] } \\ & { \leq \frac { 1 } { 2 } \mathbb { E } Y _ { n } ^ { * } ( r ) + C _ { 4 } \int _ { s } ^ { r } \left\{ \mathbb { E } Y _ { n } ^ { * } ( q ) + \mathbb { E } G _ { q } ^ { p } \right\} d q . } \end{array}\tag{25}
$$

The first line is the Burkholder–Davis–Gundy inequality, and the next uses (24). We then use $Y _ { n } ^ { * } ( r ) \wedge n ^ { p } \leq Y _ { n } ^ { * } ( r )$ . The final line follows from Young’s inequality in the form $\dot { C } _ { 3 } a ^ { 1 / 2 } b ^ { 1 / 2 } \leq \frac { 1 } { 2 } a + C _ { 4 } b$ for a new constant $C _ { 4 }$ , together with the previously displayed bound on $\| \boldsymbol { Y } _ { q } \| ^ { p - 2 } G _ { q } ^ { 2 }$

Finally, set

$$
\Phi _ { n } ( r ) : = \mathbb { E } Y _ { n } ^ { * } ( r ) = \mathbb { E } \operatorname* { s u p } _ { s \leq u \leq r } \| Y _ { u \wedge \tau _ { n } } \| ^ { p } .
$$

For $q \leq r \wedge \tau _ { n } , \| Y _ { q } \| ^ { p } \leq Y _ { n } ^ { * } ( q )$ , while the remaining integrands are nonnegative. Hence, taking the supremum over $u \leq r \mathrm { i n } \left( 2 3 \right)$ , extending the integrals to $[ s , r ]$ , and taking expectations gives

$$
\begin{array} { r l } & { \Phi _ { n } ( r ) \le \mathbb E \| Y _ { s } \| ^ { p } + C _ { 1 } \int _ { s } ^ { r } \mathbb E \| f _ { q } \| ^ { p } + G _ { \eta _ { l } } ^ { p } \| d q + C _ { 1 } \int _ { s } ^ { r } \Phi _ { n } ( q ) d q } \\ & { \qquad + \mathbb E \underbrace { \textup { s u p } | { \cal M } _ { u , v , n } | } } \\ & { \le \mathrm { s t a g } _ { s } s \cdots { \cal N } _ { u , v , n } } \\ & { \le \mathbb E \| Y _ { s } \| ^ { p } + C _ { 1 } \int _ { s } ^ { r } \mathbb E \| f _ { q } \| ^ { p } + C _ { q _ { l } } ^ { p } \| d q + C _ { 1 } \int _ { s } ^ { r } \Phi _ { n } ( q ) d q } \\ & { \qquad + \frac 1 2 \Phi _ { n } ( r ) + C _ { 1 } \int _ { s } ^ { r } \{ \Phi _ { n } ( q ) + \mathbb E G _ { q _ { l } } ^ { p } \} d q } \\ & { \le \frac 1 2 \Phi _ { n } ( r ) + C _ { 5 } \mathbb E \| Y _ { s } \| ^ { p } + C _ { 5 } \int _ { s } ^ { r } \left\{ \mathbb E \| f _ { q } \| ^ { p } + \mathbb E G _ { q } ^ { p } + \Phi _ { n } ( q ) \right\} d q } \\ & { \le C _ { 6 } \left\{ \mathbb E \| Y _ { s } \| ^ { p } + \int _ { s } ^ { r } \mathbb E \| f _ { q } \| ^ { p } + G _ { q } ^ { p } \| d q \right\} + C _ { 6 } \int _ { s } ^ { r } \Phi _ { n } ( q ) d q . } \end{array}
$$

The second inequality uses (25) together with $\Phi _ { n } ( q ) = \mathbb { E } Y _ { n } ^ { * } ( q )$ . The next line collects the $C _ { 1 }$ and $C _ { 4 }$ terms into a new constant $C _ { 5 }$ . Absorbing $\textstyle { \frac { 1 } { 2 } } \Phi _ { n } ( r )$ into the left-hand side gives the Gronwall inequality with a new constant $C _ { 6 }$ . Since $r - s \leq S$ , Gronwall’s inequality gives, uniformly in n,

$$
\Phi _ { n } ( t ) \leq C _ { p , S } ^ { Y } \left\{ \mathbb { E } \Vert \mathbf { Y } _ { s } \Vert ^ { p } + \int _ { s } ^ { t } \mathbb { E } [ \Vert \mathbf { f } _ { q } \Vert ^ { p } + G _ { q } ^ { p } ] d q \right\} .
$$

The window-length assumption is used here to absorb the exponential Gronwall factor into the constant $C _ { p , S } ^ { Y } .$

The constants $C _ { 1 } , \ldots , C _ { 6 }$ depend only on $p , L , L _ { 1 }$ and are independent of $n , s , t .$ . After Gronwall on an interval of length at most $S ,$ these constants and the exponential Gronwall factor are absorbed into $C _ { p , S } ^ { Y }$

It remains to remove the stopping. Since $\tau _ { n } \uparrow t$ almost surely and Y is continuous,

$$
\operatorname* { s u p } _ { s \leq r \leq t } \| Y _ { r \wedge \tau _ { n } } \| ^ { p } \longrightarrow \operatorname* { s u p } _ { s \leq r \leq t } \| Y _ { r } \| ^ { p } \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Fatou’s lemma and the uniform bound above yield

$$
\begin{array} { r l r } {  { \mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \| \pmb { Y } _ { r } \| ^ { p } \leq \operatorname* { l i m i n f } _ { n \to \infty } \mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \| \pmb { Y } _ { r \wedge \tau _ { n } } \| ^ { p } } } \\ & { } & { \leq C _ { p , S } ^ { Y } \{ \mathbb { E } \| \pmb { Y } _ { s } \| ^ { p } + \int _ { s } ^ { t } \mathbb { E } [ \| \pmb { f } _ { r } \| ^ { p } + G _ { r } ^ { p } ] d r \} . } \end{array}
$$

The preceding display is exactly (20). Since $A _ { t }$ enters only through the first inequality in (19), no operator-norm bound on $A _ { t }$ is required. □

We use that continuous adapted processes, and continuous functions of them, are progressively measurable; multipli cation by deterministic Borel functions such as $\sqrt { \alpha ( t ) }$ preserves this property.

The linear energy estimate will first be applied to the restarted state itself. Uniform control over the entire fixed window is needed for three reasons. Differentiating $P _ { s , t } V$ introduces polynomial factors involving $\langle \langle Z _ { t } ^ { s , x } \rangle \rangle$ for arbitrary $t \leq s + S$ . The second spatial-variation equation contains the random factor $\langle \langle Z _ { r } \rangle \rangle ^ { \kappa }$ at varying integration times $^ { r , }$ and the later applications of Holder’s inequality require higher polynomial moment orders. The path-supremum estimate in¨ the following lemma supplies all of these fixed-time controls at once.

The same linear estimate is then applied to the first and second spatial variations. Their moment bounds, together with the state moments, yield

$$
\| \nabla P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \{ { \pmb x } \} ^ { v } , \qquad \| \nabla ^ { 2 } P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \{ { \pmb x } \} ^ { v + \kappa } .
$$

Since $\kappa \geq 0$ , these are precisely the derivative-growth hypotheses needed in Lemma 9, with $\gamma _ { 1 } = v$ and $\gamma _ { 2 } = v + \kappa ,$

Lemma 7 (Fixed-window moments of the restarted SDE). Suppose $b \in C ^ { 1 } (  { \mathbb { R } } ^ { d } ;  { \mathbb { R } } ^ { d } )$ and $\sigma \in C ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d \times q } )$ satisfy the global Lipschitz condition (11), and suppose $\alpha _ { \mathrm { m a x } } < \infty$ . Then,for everyfixed $S > 0$ and everyfinite $p > 0$

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \langle \langle Z _ { r } ^ { s , { \pmb x } } \rangle \rangle ^ { p } \leq C _ { p , S } ^ { Z } \langle \langle { \pmb x } \rangle \rangle ^ { p } .\tag{26}
$$

Here $C _ { p , S } ^ { Z }$ may depend on $( p , S )$ and the fixed coefficient data, but not on s or x.

Proof. We rewrite the restarted SDE in the affine-linear form required by Lemma 6 and verify its coefficient and forcing conditions. The linear energy estimate then gives the required moment bound for $p \geq 2 ,$ , while Jensen’s inequality gives the remaining range $0 < p < 2$

The well-posedness observation in Remark 5 uses only global Lipschitzness and boundedness of $\alpha ,$ , so it applies under the present assumptions. The $C ^ { 1 }$ regularity is used below to express the coefficients in the affine-linear form required by Lemma $^ { 6 . }$

Reduction to the linearform. Write ${ Z _ { r } } = { Z _ { r } ^ { s , x } }$ , which we recall satisfies

$$
d { \pmb Z } _ { r } = b ( { \pmb Z } _ { r } ) d r + \sqrt { \alpha ( r ) \sum _ { j } \sigma _ { j } ( { \pmb Z } _ { r } ) d B _ { r } ^ { j } } , \qquad { \pmb Z } _ { s } = { \pmb x } .
$$

We now express the drift and each diffusion column in affine-linear form. For each r and $j ,$ , set $\begin{array} { r } { A _ { r } : = \int _ { 0 } ^ { 1 } D \pmb { b } ( \theta Z _ { r } ) d \theta } \end{array}$ $\begin{array} { r } { \Sigma _ { j , r } : = \sqrt { \alpha ( r ) } \int _ { 0 } ^ { 1 } D \sigma _ { j } ( \theta Z _ { r } ) d \theta } \end{array}$ , and $\mathbf { \delta } _ { { \mathbf { g } } _ { j , r } } : = \sqrt { \alpha ( r ) } \sigma _ { j } ( \mathbf { 0 } )$ . By the fundamental theorem of calculus,

$$
\begin{array} { l } { { \displaystyle { \pmb b } ( { \pmb Z } _ { r } ) = \int _ { 0 } ^ { 1 } \frac { d } { d \theta } { \pmb b } ( \theta { \pmb Z } _ { r } ) d \theta + { \pmb b } ( { \bf 0 } ) } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { 1 } D { \pmb b } ( \theta { \pmb Z } _ { r } ) { \pmb Z } _ { r } d \theta + { \pmb b } ( { \bf 0 } ) = A _ { r } { \pmb Z } _ { r } + { \pmb b } ( { \bf 0 } ) } . } \end{array}
$$

Similarly, for each diffusion column $\sigma _ { j }$

$$
\sqrt { \alpha ( r ) } \sigma _ { j } ( Z _ { r } ) = \Sigma _ { j , r } Z _ { r } + g _ { j , r } .
$$

Thus the SDE becomes

$$
d { \pmb Z } _ { r } = ( A _ { r } { \pmb Z } _ { r } + { \pmb b } ( { \pmb 0 } ) ) d r + \sum _ { j } ( \Sigma _ { j , r } { \pmb Z } _ { r } + { \pmb g } _ { j , r } ) d B _ { r } ^ { j } , \qquad { \pmb Z } _ { s } = { \pmb x } .
$$

This is the form of Lemma $^ { 6 , }$ with $Y _ { r } = Z _ { r } , f _ { r } = b ( 0 )$ , and $A _ { r } , \Sigma _ { j , r } ,$ and ${ \mathbf { } } g _ { \mathcal { I } , \mathcal { I } }$ as defined above.

The integral maps defining $A _ { r }$ and $\Sigma _ { j , \prime }$ are continuous in $Z _ { r }$ , so $A _ { r } , \Sigma _ { j , r } , \pmb { g } _ { j , r }$ , and $f _ { r }$ are progressively measurable by the observation following Lemma 6.

Coefficient bounds.

The global Lipschitz and $C ^ { 1 }$ assumptions give finite constants

$$
K _ { b } : = \operatorname* { s u p } _ { z } \| D b ( z ) \| , \qquad K _ { \sigma } ^ { 2 } : = \operatorname* { s u p } _ { z } \sum _ { j } \| D \sigma _ { j } ( z ) \| ^ { 2 } .
$$

Hence

$$
\begin{array} { r } { \langle \pmb { y } , A _ { r } \pmb { y } \rangle \leq \| A _ { r } \| \| \pmb { y } \| ^ { 2 } \leq K _ { b } \| \pmb { y } \| ^ { 2 } , } \end{array}
$$

so Lemma 6 may take $L = K _ { b }$ . Jensen’s inequality in θ gives

$$
\begin{array} { r } { \displaystyle \sum _ { j } \| \Sigma _ { j , r } \| ^ { 2 } = \alpha ( r ) \displaystyle \sum _ { j } \left\| \displaystyle \int _ { 0 } ^ { 1 } D \sigma _ { j } ( \theta Z _ { r } ) d \theta \right\| ^ { 2 } } \\ { \displaystyle \leq \alpha ( r ) \displaystyle \sum _ { j } \int _ { 0 } ^ { 1 } \| D \sigma _ { j } ( \theta Z _ { r } ) \| ^ { 2 } d \theta } \\ { \displaystyle \leq \alpha _ { \operatorname* { m a x } } K _ { \sigma } ^ { 2 } , } \end{array}
$$

so one may take $L _ { 1 } ^ { 2 } = \alpha _ { \mathrm { { m a x } } } K _ { \sigma } ^ { 2 } .$

For $f _ { r }$ , we have $\| f _ { r } \| ^ { p } = \| b ( 0 ) \| ^ { p }$ . For $\mathbf { \delta } _ { \mathbf { \delta } _ { g _ { j , r } } }$ ,

$$
\left( \sum _ { j } \| g _ { j , r } \| ^ { 2 } \right) ^ { p / 2 } = \alpha ( r ) ^ { p / 2 } \| \sigma ( \mathbf { 0 } ) \| _ { F } ^ { p } \leq \alpha _ { \operatorname* { m a x } } ^ { p / 2 } \| \sigma ( \mathbf { 0 } ) \| _ { F } ^ { p } .
$$

Together with the progressive measurability noted above, these estimates verify all the hypotheses of Lemma $^ { 6 , }$ with $L = K _ { b }$ and $L _ { 1 } ^ { 2 } = \alpha _ { \mathrm { { m a x } } } K _ { \sigma } ^ { 2 }$

Application ofthe linear estimate. For $p \geq 2$ , Lemma 6 gives

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| Z _ { r } \| ^ { p } \leq C _ { p , S } ^ { Y } \left\{ \| x \| ^ { p } + \int _ { s } ^ { s + S } \left[ \| b ( \mathbf { 0 } ) \| ^ { p } + \alpha ( r ) ^ { p / 2 } \| \sigma ( \mathbf { 0 } ) \| _ { F } ^ { p } \right] d r \right\}
$$

Since $\alpha ( r ) \leq \alpha _ { \mathrm { m a x } }$ and the interval has length S, the integral is bounded by a finite constant independent of x. Absorbing this constant into a new constant $C _ { p , S } ^ { Z }$ gives

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| Z _ { r } \| ^ { p } \leq C _ { p , S } ^ { Z } \big ( 1 + \| \pmb { x } \| ^ { p } \big ) .
$$

Since $\langle \langle z \rangle \rangle ^ { p } \asymp _ { p } 1 + \| z \| ^ { p }$ , this implies

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \langle [ \pmb { Z } _ { r } ^ { s , \pmb { x } } \rangle \rangle ^ { p } \leq C _ { p , S } ^ { Z } \langle \pmb { \mathscr { x } } \rangle \rangle ^ { p } , \qquad p \geq 2 .
$$

For $0 < p < 2$ , Jensen’s inequality applied to the $p = 2$ estimate gives

$$
\begin{array} { r l } & { \mathbb { E } \underset { s \leq r \leq s + S } { \operatorname* { s u p } } \langle Z _ { r } ^ { s , x } \rangle \rangle ^ { p } = \mathbb { E } \left[ \underset { s \leq r \leq s + S } { \operatorname* { s u p } } \langle Z _ { r } ^ { s , x } \rangle \rangle ^ { 2 } \right] ^ { p / 2 } } \\ & { \qquad \leq \left( \mathbb { E } \underset { s \leq r \leq s + S } { \operatorname* { s u p } } \langle Z _ { r } ^ { s , x } \rangle \rangle ^ { 2 } \right) ^ { p / 2 } } \\ & { \qquad \leq \left( C _ { 2 , S } ^ { Z } \right) ^ { p / 2 } \langle x \rangle ^ { p } . } \end{array}
$$

Absorbing $\left( C _ { 2 , S } ^ { Z } \right) ^ { p / 2 }$ into a constant denoted by $C _ { p , S } ^ { Z }$ gives the stated bound for $0 < p < 2$ . This proves the fixed-window state-moment bound for every finite $p > 0$ , and completes the proof.

To obtain gradient and Hessian bounds for $P _ { s , t } V$ , we control the first and second spatial variations of the restarted SDE flow. The process $Z _ { t } ^ { s , \pmb { x } }$ can be differentiated with respect to its initial condition. The resulting spatial derivatives are described by the classical theory of stochastic flows, developed systematically by [Kunita, 1990]. Here we use Matsumoto and Taniguchi [2016, Theorem 4.10.8], applied after localization to the resulting $C ^ { 3 }$ vector fields.

Lemma 8 (Polynomial control of semigroup derivatives). Under (11), (12), and (13) of Assumption 1, for every fixed $S > 0$ and $0 \leq \bar { t } - s \leq S , P _ { s , t } V \in C ^ { 2 } \bar { ( } \mathbb { R } ^ { d } )$ , with

$$
\begin{array} { r l } & { \| \nabla P _ { s , t } V ( \pmb { x } ) \| \le C _ { S } \{ \langle \pmb { x } \rangle \} ^ { v } , } \\ & { \| \nabla ^ { 2 } P _ { s , t } V ( \pmb { x } ) \| \le C _ { S } \{ \langle \pmb { x } \rangle \} ^ { v + \kappa } . } \end{array}\tag{27}
$$

The constant $C _ { S }$ may depend on the fixed coefficient data and on $S ,$ but not on $s , t ,$ or x subject to $0 \leq t - s \leq S$

Proof. The proof proceeds by calculating the first and second derivatives of the restarted flow $Z _ { r } ^ { s , { \pmb x } }$ with respect to its initial state ${ \pmb x } .$ To do this we apply Matsumoto and Taniguchi [2016], Theorem 4.10.8. We then apply Lemma 6 to bound the maximum moments of the flow.

First spatial derivative of the flow. Applying Theorem 4.10.8 of Matsumoto and Taniguchi [2016] to bounded $C ^ { 3 }$ vector fields shows that the Jacobian $J _ { r } : = D _ { x } Z _ { r } ^ { s , x }$ exists with $J _ { s } = I$ and satisfies the first-variation equation:

$$
d J _ { r } = D b ( Z _ { r } ) J _ { r } d r + \sqrt { \alpha ( r ) } \sum _ { j } D \sigma _ { j } ( Z _ { r } ) J _ { r } d B _ { r } ^ { j } .\tag{28}
$$

There are two technical points in applying Matsumoto and Taniguchi [2016, Theorem 4.10.8]. First, that theorem is applied to bounded $C ^ { 3 }$ vector fields with bounded derivatives through order three, whereas our coefficients are $C ^ { 3 }$ and their higher derivatives need only be locally bounded. For each $R > 0$ , a smooth cutoff gives globally bounded $C ^ { 3 }$ extensions $\bar { \pmb { b } } ^ { ( R ) }$ and $\sigma ^ { ( R ) }$ that agree with b and σ on the bal $B ( \mathbf { 0 } , R )$ and have bounded derivatives through order three. Applying the theorem to these modified coefficients gives a $C ^ { 2 }$ stochastic flow. By pathwise uniqueness, the modified and original restarted diffusions, and hence their local flow maps, agree up to the exit time from $B ( \mathbf { 0 } , R )$ . Since the original restarted SDE is nonexplosive, these exit times tend to infinity almost surely as $R \to \infty$ , yielding the local $C ^ { 2 }$ flow for the original equation; this is the standard localization argument, see also Varadhan [2006, Remark 3.1].

Second, $\alpha ( r )$ is piecewise constant. On each interval $[ t _ { k } , t _ { k + 1 } )$ , it equals $\alpha _ { k }$ , so the spatial vector fields are b and $\sqrt { \alpha _ { k } } \sigma _ { j }$ . The corresponding cutoff vector fields are bounded $C ^ { 3 }$ fields, so Theorem 4.10.8 applies on that interval. A fixed diffusion-time window intersects only finitely many such intervals, and composing the corresponding $C ^ { 2 }$ solution maps gives a $C ^ { 2 }$ restarted flow throughout the window. Consequently the first variation satisfies (28). The same localization-and-composition argument gives the second spatial derivative and hence the second-variation equation used below. The $C ^ { 3 }$ assumption is used here to obtain the $C ^ { 2 }$ flow; no global bound on third derivatives is assumed or used in the quantitative estimates below, which rely only on the stated bounds on the first and second derivatives.

For a fixed direction a, set $\pmb { Y _ { r } } : = \pmb { J _ { r } } \pmb { a }$ , with $\mathbf { Y } _ { s } = \mathbf { a }$ . Then $\mathbf { Y }$ satisfies

$$
d \pmb { Y } _ { r } = A _ { r } \pmb { Y } _ { r } d r + \sum _ { j } \Sigma _ { j , r } \pmb { Y } _ { r } d B _ { r } ^ { j } ,
$$

where $A _ { r } = D b ( Z _ { r } ) , \Sigma _ { j , r } = \sqrt { \alpha ( r ) } D \sigma _ { j } ( Z _ { r } )$ . This is precisely the homogeneous case of (18) in Lemma $^ { 6 , }$ with $f _ { r } = 0$ and $\mathbf { \mathscr { g } } _ { j , r } = 0$ . We next verify the coefficient bounds required to apply that lemma. Differentiating the global Lipschitz bounds in Assumption 1 gives $\| D b ( \pmb { x } ) \| \le L _ { b }$ and $\| D \sigma _ { j } ( { \pmb x } ) \| \le L _ { \sigma }$ for $j = 1 , \dotsc , q$ . By the measurability observation following Lemma $^ { 6 , }$ the coefficient processes $A _ { r }$ and $\Sigma _ { j , r }$ are progressively measurable. Their coefficient bounds give

$$
\langle { \pmb y } , A _ { r } { \pmb y } \rangle \leq L _ { b } \| { \pmb y } \| ^ { 2 } , \qquad \sum _ { j } \| \Sigma _ { j , r } \| ^ { 2 } = \alpha ( r ) \sum _ { j = 1 } ^ { q } \| D \sigma _ { j } ( Z _ { r } ) \| ^ { 2 } \leq \alpha _ { \mathrm { m a x } } q L _ { \sigma } ^ { 2 } .
$$

Together with progressive measurability and $\mathbf { \boldsymbol { f } } _ { r } = \mathbf { \boldsymbol { g } } _ { j , r } = \mathrm { ~ 0 ~ }$ , these bounds verify all hypotheses of Lemma $^ { 6 , }$ with $L = L _ { b }$ and $L _ { 1 } ^ { 2 } = \alpha _ { \mathrm { m a x } } q L _ { \sigma } ^ { 2 }$ . Thus, by Lemma 6, we have

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| J _ { r } \pmb { a } \| ^ { p } \leq C _ { p , S } ^ { Y } \| \pmb { a } \| ^ { p } , \qquad p \geq 2 .\tag{29}
$$

For the standard basis $e _ { 1 } , \ldots , e _ { d }$ , finite dimensionality gives $\begin{array} { r } { \| J _ { r } \| \leq \| J _ { r } \| _ { F } \leq \sum _ { i = 1 } ^ { d } \| J _ { r } \boldsymbol { e } _ { i } \| } \end{array}$ . Therefore,

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| J _ { r } \| ^ { p } \leq C _ { p , d } \sum _ { i = 1 } ^ { d } \mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| J _ { r } e _ { i } \| ^ { p } \leq C _ { p , S } ^ { J } .\tag{30}
$$

This proves the required first spatial derivative moment bound. We now use it, together with the state-moment lemma, to bound the $f _ { r }$ and $\mathbf { \delta } _ { \mathbf { \eta } _ { 3 , r } }$ terms in the second spatial derivative equation.

Second spatial derivative of the flow. The second spatial derivative is handled similarly to the first. Since the restarted flow is $C ^ { 2 }$ in its initial state, for fixed directions a, c we may define $K _ { r } [ \mathbf { a } , \mathbf { c } ] : = D _ { x } ^ { 2 } \bar { Z } _ { r } ^ { s , x } [ \mathbf { a } , \mathbf { c } ]$ with $K _ { s } [ { \bf a } , { \bf c } ] = 0 . { \bf \nabla } ^ { 2 }$ Differentiating the Jacobian equation in direction c, as in the proof of Matsumoto and Taniguchi [2016, Theorem 4.10.8], gives the second-variation equation. The same two technical points as for the first variation apply: localization gives the required $C ^ { 2 }$ dependence on the initial state, while the piecewise-constant α is handled interval by interval and the resulting $C ^ { 2 }$ flow maps are composed across the finitely many boundaries.

$$
\begin{array} { l } { { d K _ { r } [ a , c ] = \left\{ D b ( { \pmb Z } _ { r } ) K _ { r } [ a , c ] + D ^ { 2 } b ( { \pmb Z } _ { r } ) [ J _ { r } c , J _ { r } a ] \right\} d r } } \\ { { \displaystyle ~ + \sqrt { \alpha ( r ) } \sum _ { j } \left\{ D \sigma _ { j } ( { \pmb Z } _ { r } ) K _ { r } [ a , c ] + D ^ { 2 } \sigma _ { j } ( { \pmb Z } _ { r } ) [ J _ { r } c , J _ { r } a ] \right\} d B _ { r } ^ { j } . } } \end{array}\tag{31}
$$

Now set $Y _ { r } = K _ { r } [ \pmb { a } , \pmb { c } ]$ , with $Y _ { s } = 0$ , and take $A _ { r } = D b ( Z _ { r } ) , \Sigma _ { j , r } = \sqrt { \alpha ( r ) } D \sigma _ { j } ( Z _ { r } ) , f _ { r } = D ^ { 2 } b ( Z _ { r } ) [ J _ { r } c , J _ { r } a ] .$ and ${ \pmb g } _ { j , r } = \sqrt { \alpha ( r ) } D ^ { 2 } \sigma _ { j } ( { \pmb Z } _ { r } ) [ J _ { r } { \pmb c } , J _ { r } { \pmb a } ]$ . The coefficient hypotheses for $A _ { r }$ and $\Sigma _ { j , r }$ were verified in the first spatial derivative argument. By the measurability observation following Lemma 6 and continuity of the second derivatives, $f _ { r }$ and ${ \mathbf { } } g _ { \mathcal { I } , \mathfrak { i } }$ <sub>r</sub> are progressively measurable. Define $\begin{array} { r } { G _ { r } = \left( \sum _ { j } \| g _ { j , r } \| ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ . Now apply Lemma $^ { 6 , }$ using $Y _ { s } = 0 \mathrm { : }$

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq s + S } \| K _ { r } [ \pmb { a } , \pmb { c } ] \| ^ { p } \leq C _ { p , S } ^ { Y } \int _ { s } ^ { s + S } \mathbb { E } [ \| \pmb { f } _ { r } \| ^ { p } + G _ { r } ^ { p } ] d r .\tag{32}
$$

We now look to bound the $f _ { r }$ and $G _ { r }$ terms in the integrand above.

By (12) in Assumption 1,

$$
\begin{array} { r l r } & { } & { \Vert \pmb { f _ { r } } \Vert \leq \Vert D ^ { 2 } \pmb { b } ( \pmb { Z _ { r } } ) \Vert \Vert J _ { r } \pmb { c } \Vert \Vert J _ { r } \pmb { a } \Vert } \\ & { } & { \leq C \langle \langle \pmb { Z _ { r } } \rangle \rangle ^ { \kappa } \Vert J _ { r } \Vert ^ { 2 } \Vert \pmb { a } \Vert \Vert \pmb { c } \Vert , } \end{array}
$$

and, using $\left( \textstyle \sum _ { j } a _ { j } ^ { 2 } \right) ^ { 1 / 2 } \leq \sum _ { j } a _ { j }$

$$
\begin{array} { r l } & { G _ { r } \le \sqrt { \alpha ( r ) } \| J _ { r } \| ^ { 2 } \| { \pmb a } \| \| { \pmb c } \| \left( \displaystyle \sum _ { j } \| D ^ { 2 } \sigma _ { j } ( { \pmb Z } _ { r } ) \| ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \quad \quad \le \sqrt { \alpha ( r ) } \| J _ { r } \| ^ { 2 } \| { \pmb a } \| \| { \pmb c } \| \displaystyle \sum _ { j } \| D ^ { 2 } \sigma _ { j } ( { \pmb Z } _ { r } ) \| } \\ & { \quad \quad \le C \langle { \pmb Z } _ { r } \rangle ^ { \kappa } \| J _ { r } \| ^ { 2 } \| { \pmb a } \| \| { \pmb c } \| , } \end{array}
$$

The last inequality uses the second-derivative growth condition (12) from Assumption 1, with $\sqrt { \alpha _ { \mathrm { m a x } } }$ absorbed into $C .$ Taking pth powers in the preceding estimates and applying Holder’s inequality gives¨

$$
\begin{array} { r } { \mathbb { E } [ \langle \pmb { \zeta } _ { r } \rangle ^ { \kappa p } \| J _ { r } \| ^ { 2 p } ] \leq \big \{ \mathbb { E } \langle \pmb { Z } _ { r } \rangle \big \} ^ { 2 \kappa p } \big \} ^ { 1 / 2 } \big \{ \mathbb { E } \| J _ { r } \| ^ { 4 p } \big \} ^ { 1 / 2 } . } \end{array}
$$

Together with the state and Jacobian moment bounds, these estimates and Holder’s inequality give¨

$$
\begin{array} { r } { { \mathbb { E } } [ \| \pmb { f } _ { r } \| ^ { p } + G _ { r } ^ { p } ] \leq C _ { p , S } ^ { K } \langle \langle \pmb { x } \rangle \rangle ^ { \kappa p } \| \pmb { a } \| ^ { p } \| \pmb { c } \| ^ { p } . } \end{array}\tag{33}
$$

If $\kappa = 0$ , the first factor in the Holder estimate is simply one. Substituting ¨ (33) into (32) and absorbing the constant $C _ { p , S } ^ { Y }$ into $C _ { p , S } ^ { K }$ gives

$$
\mathbb { E } \operatorname* { s u p } _ { \substack { s \leq r \leq s + S } } \| K _ { r } [ \pmb { a } , \pmb { c } ] \| ^ { p } \leq C _ { p , S } ^ { K } \langle \langle \pmb { x } \rangle \rangle ^ { \kappa p } \| \pmb { a } \| ^ { p } \| \pmb { c } \| ^ { p } .\tag{34}
$$

(By finite dimensionality, applying the preceding estimate to the finitely many coordinate pairs also gives the correspond ing moment bound for the bilinear operator norm of $K _ { t \cdot } )$

Differentiation ofthe semigroup. It remains to translate the estimates (26), (30), and (34) into the semigroup function $P _ { s , t } V$ . (For the stochastic-approximation argument, this is required to analyse the backward function $u _ { k , n } = P _ { t _ { k } , t _ { n } } V$ defined in (5).)

$$
\| \nabla P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \langle \langle { \pmb x } \rangle ^ { v } , \qquad \| \nabla ^ { 2 } P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \langle \langle { \pmb x } \rangle ^ { v + \kappa } .
$$

Given our stochastic flow analysis the pathwise derivatives of $V ( Z _ { t } ^ { s , \pmb { x } } )$ follow directly by the chain rule. (The state, Jacobian, and second-variation moment estimates (26), (30), and (34), in addition to providing regularity bounds, imply uniform integrability. This allows us to justify passing these derivatives through the expectation defining $P _ { s , t } V . )$

To display the dependence on the initial state in this part of the proof, write $J _ { t } ^ { y }$ and $K _ { t } ^ { y }$ for the variations defined above along $\pmb { Z } _ { t } ^ { s , y }$

For fixed $s , t ,$ define $\Phi ( \pmb { y } ) : = V ( Z _ { t } ^ { s , \pmb { y } } )$ . The restarted flow is $C ^ { 2 }$ in its initial state $\mathbf { \pmb { y } } .$ For directions $a , c ,$ the ordinary pathwise chain rule gives

$$
D \Phi ( { \pmb y } ) [ { \pmb a } ] = \nabla V ( \pmb { Z } _ { t } ^ { s , { \pmb y } } ) ^ { \top } J _ { t } ^ { { \pmb y } } { \pmb a } .
$$

The second pathwise derivative is

$$
D ^ { 2 } \Phi ( { \pmb y } ) [ { \pmb a } , { \pmb c } ] = \nabla ^ { 2 } V ( Z _ { t } ^ { s , { \pmb y } } ) [ J _ { t } ^ { y } { \pmb c } , J _ { t } ^ { y } { \pmb a } ] + \nabla V ( Z _ { t } ^ { s , { \pmb y } } ) ^ { \top } K _ { t } ^ { y } [ { \pmb a } , { \pmb c } ] .
$$

Consequently,

$$
\begin{array} { r l r l } & { \| D \Phi ( \pmb { y } ) \| \leq C \langle \langle Z _ { t } ^ { s , y } \rangle ^ { v } \| J _ { t } ^ { y } \| , } & & { \| D ^ { 2 } \Phi ( \pmb { y } ) \| \leq C \langle \langle Z _ { t } ^ { s , y } \rangle ^ { v } \left( \| J _ { t } ^ { y } \| ^ { 2 } + \| K _ { t } ^ { y } \| \right) . } \end{array}
$$

On every bounded neighbourhood of $^ { \mathbf { \delta x } , }$ the state-moment estimate in Lemma 7, together with the Jacobian moment bound (30) and the second-variation bound (34), gives uniform moments of $Z , J ,$ and K. The state moments control the polynomial factor in the displayed derivative bounds; the J moments control the first-derivative and $J ^ { 2 }$ terms, while the K moments control the remaining second-derivative term. By Assumption 1, specifically (13), ∇V and $\nabla ^ { 2 } V$ have polynomial growth. For V itself, the fundamental theorem of calculus gives

$$
| V ( z ) | \leq | V ( \mathbf { 0 } ) | + \| z \| \int _ { 0 } ^ { 1 } \| \nabla V ( \theta z ) \| d \theta \leq C \langle \langle z \rangle \rangle ^ { v + 1 } .
$$

Thus the state-moment estimate also ensures that $V ( Z _ { t } ^ { s , \pmb { y } } )$ is integrable. Holder’s inequality and the local moment¨ bounds imply locally uniform $L ^ { 2 }$ bounds for DΦ and $D ^ { 2 } \Phi$ . These bounds imply uniform integrability on bounded sets of initial conditions. Since the pathwise derivatives are continuous in y, the standard differentiation-under-expectation argument applies to $P _ { s , t } V ( \pmb { x } ) \bar { = } \mathbb { E } [ V ( Z _ { t } ^ { s , \pmb { x } } ) ]$ ]. It yields $P _ { s , t } V \in C ^ { 2 } ( \mathbb { R } ^ { d } )$ and the derivative formulas below.

$$
\nabla P _ { s , t } V ( \pmb { x } ) ^ { \top } \pmb { a } = \mathbb { E } \big [ \nabla V ( Z _ { t } ^ { s , \pmb { x } } ) ^ { \top } J _ { t } \pmb { a } \big ] .\tag{35}
$$

Analogously, the second derivative is

$$
\begin{array} { r } { \nabla ^ { 2 } P _ { s , t } V ( \boldsymbol { x } ) [ a , c ] = \mathbb { E } \left[ \nabla ^ { 2 } V ( Z _ { t } ^ { s , x } ) [ J _ { t } c , J _ { t } a ] + \nabla V ( Z _ { t } ^ { s , x } ) ^ { \top } K _ { t } [ a , c ] \right] . } \end{array}\tag{36}
$$

Equations (35) and (36) give the required polynomial bounds. Specifically, from (35), (13), and the state and Jacobian moment estimates (26) and (30), Cauchy–Schwarz gives

$$
\begin{array} { r } { | \nabla P _ { s , t } V ( { \boldsymbol x } ) ^ { \top } { \boldsymbol a } | \leq C \left\{ \mathbb { E } \langle \langle \boldsymbol { Z } _ { t } ^ { s , x } \rangle \rangle ^ { 2 v } \right\} ^ { 1 / 2 } \left\{ \mathbb { E } \| J _ { t } \| ^ { 2 } \right\} ^ { 1 / 2 } \| { \boldsymbol a } \| \leq C _ { S } \langle \langle \boldsymbol { x } \rangle \rangle ^ { v } \| { \boldsymbol a } \| . } \end{array}
$$

From the first term in (36), (13), and the state and Jacobian moment estimates (26) and (30),

$$
\Bigl | \mathbb { E } \nabla ^ { 2 } V ( Z _ { t } ^ { s , x } ) [ J _ { t } \pmb { c } , J _ { t } \pmb { a } ] \Bigr | \le C _ { S } \langle \pmb { \langle x \rangle } \rangle ^ { v } \| \pmb { a } \| \| \pmb { c } \| .
$$

For the second term in (36), the same growth assumption, together with the state and second-variation moment estimates (26) and (34), gives

$$
\big | \mathbb { E } \nabla V ( Z _ { t } ^ { s , { \boldsymbol { \mathbf { x } } } } ) ^ { \top } K _ { t } [ { \boldsymbol { \mathbf { a } } } , { \boldsymbol { \mathbf { c } } } ] \big | \leq C _ { S } \langle \langle { \boldsymbol { \mathbf { x } } } \rangle \rangle ^ { v + \kappa } \| { \boldsymbol { \mathbf { a } } } \| \| { \boldsymbol { \mathbf { c } } } \| .
$$

Since $\kappa \geq 0$ and $\left. \left. \pmb { x } \right. \right. \geq 1$ , these estimates give

$$
\| \nabla P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \langle \langle { \pmb x } \rangle ^ { v } , \qquad \| \nabla ^ { 2 } P _ { s , t } V ( { \pmb x } ) \| \le C _ { S } \langle \langle { \pmb x } \rangle ^ { v + \kappa } .
$$

We have therefore proved the $C ^ { 2 }$ regularity and the gradient and Hessian bounds stated in Lemma 8. In particular, $P _ { s , t } V$ satisfies the hypotheses of Lemma 9 with $\gamma _ { 1 } = v$ and $\gamma _ { 2 } = v + \kappa$ □

The preceding lemma gives the derivative bounds needed for the one-step comparison. The following lemma records the standard Dynkin formula in the form needed here; see, for example, Øksendal [2003, Theorem 7.4.1]. We include the short argument to verify its applicability under our polynomial growth bounds. (The argument is standard and can be skipped on first reading.)

Lemma 9 (Taylor remainder using both derivative bounds). Suppose $u \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ and, for $0 \leq \gamma _ { 1 } \leq \gamma _ { 2 }$

$$
\begin{array} { r } { \| \nabla u ( { \pmb x } ) \| \le C _ { 1 } \langle \pmb x \rangle \rangle ^ { \gamma _ { 1 } } , \qquad \| \nabla ^ { 2 } u ( { \pmb x } ) \| \le C _ { 2 } \langle \pmb x \rangle \rangle ^ { \gamma _ { 2 } } . } \end{array}\tag{37}
$$

Set $\gamma _ { 3 } = \operatorname* { m a x } \{ 2 , \gamma _ { 1 } + 1 \}$ . Then, $f o r \ z \in \mathbb R ^ { d } ;$

$$
\begin{array} { r } { | u ( { \pmb x } + { \pmb z } ) - u ( { \pmb x } ) - \nabla u ( { \pmb x } ) ^ { \top } { \pmb z } | \leq C _ { 3 } \{ \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 2 } } \| { \pmb z } \| ^ { 2 } + \| { \pmb z } \| ^ { \gamma _ { 3 } } \} . } \end{array}\tag{38}
$$

Consequently, $i f \mathbb { E } [ \| H \| ^ { \gamma _ { 3 } } \mid \mathcal { F } ] \leq M \langle \langle \pmb { x } \rangle \rangle ^ { \gamma _ { 4 } }$ , and x is ${ \mathcal { F } } .$ -measurable, then

$$
\begin{array} { r } { \left| \mathbb { E } [ u ( { \pmb x } + h { \pmb H } ) \ | \ { \mathcal F } ] - u ( { \pmb x } ) - h \nabla u ( { \pmb x } ) ^ { \top } \mathbb { E } [ { \pmb H } \ | \ { \mathcal F } ] \right| \leq C _ { 4 } h ^ { 2 } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 5 } } , } \end{array}\tag{39}
$$

for $0 < h \leq \alpha _ { \mathrm { m a x } }$ and where $\gamma _ { 5 } = \operatorname* { m a x } \{ \gamma _ { 2 } + 2 \gamma _ { 4 } / \gamma _ { 3 } , \gamma _ { 4 } \}$

In particular, if x is ${ \mathcal { F } } .$ -measurable and, for some $p \geq \gamma _ { 2 } + 2 ,$

$$
\mathbb { E } [ \| \pmb { H } \| ^ { p } \mid \mathcal { F } ] \le M _ { p } \langle \langle \pmb { x } \rangle \rangle ^ { p } ,
$$

then, for $0 < h \le \alpha _ { \mathrm { m a x } }$

$$
\begin{array} { r l } & { \Big | \mathbb { E } [ u ( { \pmb x } + h { \pmb H } ) \ | \ { \mathcal F } ] - u ( { \pmb x } ) } \\ & { \qquad - h \nabla u ( { \pmb x } ) ^ { \top } \mathbb { E } [ { \pmb H } \ | \ { \mathcal F } ] \Big | \leq C _ { 5 } h ^ { 2 } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 2 } + 2 } . } \end{array}\tag{40}
$$

Here $C _ { 5 }$ depends only on $C _ { 1 } , C _ { 2 }$ , the derivative exponents γ<sub>1</sub>, γ<sub>2</sub>, p, $M _ { p } ,$ , and $\alpha _ { \mathrm { m a x } }$

Proof. For $\| z \| \leq 1$ , Taylor’s formula with integral remainder gives

$$
\begin{array} { r l r } {  {  u ( \pmb { x } + \mathfrak { z } ) - u ( \pmb { x } ) - \nabla u ( \pmb { x } ) ^ { \top } \pmb { z }  =  \int _ { 0 } ^ { 1 } ( 1 - \theta ) \ : \pmb { z } ^ { \top } \nabla ^ { 2 } u ( \pmb { x } + \theta \pmb { z } ) \ d \mathscr { z } d \theta  } } \\ & { } & { \leq C _ { 2 } \Vert \pmb { z } \Vert ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - \theta ) \langle \pmb { x } + \theta \pmb { z } \rangle ^ { \gamma _ { 2 } } d \theta } \\ & { } & { \leq 2 ^ { \gamma _ { 2 } } C _ { 2 } \langle \pmb { x } \rangle ^ { \gamma _ { 2 } } \Vert \pmb { z } \Vert ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - \theta ) d \theta } \\ & { } & { = 2 ^ { \gamma _ { 2 } - 1 } C _ { 2 } \langle \pmb { x } \rangle ^ { \gamma _ { 2 } } \Vert \pmb { z } \Vert ^ { 2 } . } \end{array}\tag{41}
$$

(42)

In the inequality (41), we apply the assumed bound (37). For the inequality (42), note that the regularized Euclidean bracket satisfies $\left. \left. \pmb { x } + \pmb { y } \right. \right. \leq \left. \left. \pmb { x } \right. \right. + \left\| \pmb { y } \right\|$ . Consequently, we can apply the inequality

$$
\begin{array} { r } { \langle ( { \pmb x } + { \pmb \theta } { \pmb z } \rangle ) ^ { \gamma _ { 2 } } \leq ( \langle ( { \pmb x } \rangle ) + { \pmb \theta } \| { \pmb z } \| ) ^ { \gamma _ { 2 } } \leq ( \langle ( { \pmb x } \rangle ) + 1 ) ^ { \gamma _ { 2 } } \leq 2 ^ { \gamma _ { 2 } } \langle \langle { \pmb x } \rangle ) ^ { \gamma _ { 2 } } , } \end{array}
$$

which is the required bound when $\| z \| \leq 1$

For $\| z \| > 1$ , using the Hessian bound globally would introduce the higher power $\| z \| ^ { \gamma _ { 2 } + 2 }$ . Instead, we use the gradient bound along the segment from x to $x + z { : }$

$$
\left| u ( \pmb { x } + z ) - u ( \pmb { x } ) - \nabla u ( \pmb { x } ) ^ { \top } z \right|
$$

$$
= \left| \int _ { 0 } ^ { 1 } \nabla u ( { \pmb x } + \theta z ) ^ { \top } { \boldsymbol z } d \theta - \nabla u ( { \pmb x } ) ^ { \top } { \boldsymbol z } \right|\tag{43}
$$

$$
\leq \| z \| \int _ { 0 } ^ { 1 } \| \nabla u ( { \pmb x } + \theta z ) \| d \theta + \| z \| \| \nabla u ( { \pmb x } ) \|\tag{44}
$$

$$
\leq C _ { 1 } \| z \| \left\{ \int _ { 0 } ^ { 1 } \langle \langle x + \theta z \rangle \rangle ^ { \gamma _ { 1 } } d \theta + \langle \langle x \rangle \rangle ^ { \gamma _ { 1 } } \right\}\tag{45}
$$

$$
\leq C _ { 1 } ( 1 + C _ { \gamma _ { 1 } } ) \| z \| \left\{ \langle \langle x \rangle \rangle ^ { \gamma _ { 1 } } + \| z \| ^ { \gamma _ { 1 } } \right\}\tag{46}
$$

$$
\leq C _ { 1 } ( 1 + C _ { \gamma _ { 1 } } ) \left\{ \langle \langle \pmb { x } \rangle \rangle ^ { \gamma _ { 2 } } \| z \| ^ { 2 } + \| z \| ^ { \gamma _ { 3 } } \right\} .\tag{47}
$$

In (43), we use the fundamental theorem of calculus along the segment from x to $x + z$ . In (44), we use the triangle inequality and the Euclidean inner-product bound. In (45), we apply the gradient-growth assumption from (37). In (46), we use the translation inequality for $\langle \left. { { \pmb x } + { \pmb y } } \right. \rangle \left( 1 7 \right)$ with $\begin{array} { r } { \mathbf { \nabla } \pmb { y } = \theta \pmb { z } } \end{array}$ , together with $0 \leq \theta \leq 1$ . For (47), since $\gamma _ { 1 } \leq \gamma _ { 2 }$ $\langle \langle { \pmb x } \rangle \rangle \geq 1$ , and $\| z \| > 1$ , we have $\begin{array} { r } { \langle \langle \pmb { x } \rangle \rangle ^ { \gamma _ { 1 } } \| \pmb { z } \| \leq \langle \langle \pmb { x } \rangle \rangle ^ { \gamma _ { 2 } } \| \pmb { z } \| ^ { 2 } } \end{array}$ ; moreover, $\begin{array} { r } { \| z \| ^ { \gamma _ { 1 } + 1 } \leq \| z \| ^ { \gamma _ { 3 } } } \end{array}$ because $\gamma _ { 3 } = \operatorname* { m a x } \{ 2 , \gamma _ { 1 } + 1 \}$ Choose

$$
C _ { 3 } = \operatorname* { m a x } \left\{ 2 ^ { \gamma _ { 2 } - 1 } C _ { 2 } , C _ { 1 } ( 1 + C _ { \gamma _ { 1 } } ) \right\} .
$$

Then the two cases give the deterministic Taylor-remainder bound (38).

$$
\left| \mathbb { E } [ u ( { \pmb x } + h { \pmb H } ) \mid { \mathcal { F } } ] - u ( { \pmb x } ) - h \nabla u ( { \pmb x } ) ^ { \top } \mathbb { E } [ { \pmb H } \mid { \mathcal { F } } ] \right|
$$

$$
\leq \mathbb { E } \big [ | u ( { \pmb x } + h { \pmb H } ) - u ( { \pmb x } ) - h \nabla u ( { \pmb x } ) ^ { \top } { \pmb H } | ~ | ~ \mathcal { F } \big ]
$$

$$
\leq C _ { 3 } h ^ { 2 } \langle \langle \pmb { x } \rangle \rangle ^ { \gamma _ { 2 } } \mathbb { E } [ \| \pmb { H } \| ^ { 2 } \ | \ \mathcal { F } ] + C _ { 3 } h ^ { \gamma _ { 3 } } \mathbb { E } [ \| \pmb { H } \| ^ { \gamma _ { 3 } } \ | \ \mathcal { F } ]\tag{48}
$$

$$
\leq C _ { 3 } M ^ { 2 / \gamma _ { 3 } } h ^ { 2 } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 2 } + 2 \gamma _ { 4 } / \gamma _ { 3 } } + C _ { 3 } M h ^ { \gamma _ { 3 } } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 4 } }\tag{49}
$$

$$
\leq C _ { 3 } h ^ { 2 } \left\{ M ^ { 2 / \gamma _ { 3 } } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 2 } + 2 \gamma _ { 4 } / \gamma _ { 3 } } + M \alpha _ { \operatorname* { m a x } } ^ { \gamma _ { 3 } - 2 } \langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 4 } } \right\}\tag{50}
$$

$$
\leq C _ { 4 } h ^ { 2 } \langle \langle x \rangle \rangle ^ { \gamma _ { 5 } } .\tag{51}
$$

In (48), we apply the deterministic Taylor-remainder estimate (38) with $z = h H$ . For (49), the assumed conditional moment assumption gives the $\gamma _ { 3 }$ -moment term directly. Also (conditional Jensen) and $\gamma _ { 3 } \geq 2$ give that $\mathbb { E } [ \| H \| ^ { 2 } \mid { \mathcal { F } } ] \leq$ $\left( \mathbb { E } [ \| \pmb { H } \| ^ { \gamma _ { 3 } } \mid \mathcal { F } ] \right) ^ { 2 / \gamma _ { 3 } } \le M ^ { 2 / \gamma _ { 3 } } \langle \langle \pmb { x } \rangle \rangle ^ { 2 \gamma _ { 4 } / \gamma _ { 3 } }$ . For (50), since $\gamma _ { 3 } \geq 2$ and $0 < h \le \alpha _ { \mathrm { m a x } }$ , we use $h ^ { \gamma _ { 3 } } \leq \alpha _ { \operatorname* { m a x } } ^ { \gamma _ { 3 } - 2 } h ^ { 2 }$ . Finally, for (51), the definition of γ and $\langle \langle \pmb { x } \rangle \rangle \geq 1$ imply that both preceding polynomial weights are bounded by $\langle \langle { \pmb x } \rangle \rangle ^ { \gamma _ { 5 } }$ , while the constant $C _ { 4 } = C _ { 3 } \left( 1 + M ^ { 2 / \gamma _ { 3 } } + M \alpha _ { \mathrm { m a x } } ^ { \gamma _ { 3 } - 2 } \right)$ . dominates the two coefficients. This proves (39).

For the p-moment specialization, $\gamma _ { 3 } = \operatorname* { m a x } \{ 2 , \gamma _ { 1 } + 1 \} \leq \gamma _ { 2 } + 2 \leq p .$ Conditional Jensen’s inequality gives

$$
\begin{array} { r l } & { \mathbb { E } [ \| \pmb { H } \| ^ { \gamma _ { 3 } } \mid \mathcal { F } ] \le ( \mathbb { E } [ \| \pmb { H } \| ^ { p } \mid \mathcal { F } ] ) ^ { \gamma _ { 3 } / p } } \\ & { \qquad \le M _ { p } ^ { \gamma _ { 3 } / p } \langle \pmb { \chi } \rangle \rangle ^ { \gamma _ { 3 } } . } \end{array}
$$

Thus (39) applies with $\gamma _ { 4 } = \gamma _ { 3 }$ , and its exponent becomes

$$
\gamma _ { 5 } = \operatorname* { m a x } \bigg \{ \gamma _ { 2 } + \frac { 2 \gamma _ { 4 } } { \gamma _ { 3 } } , \gamma _ { 4 } \bigg \} = \operatorname* { m a x } \{ \gamma _ { 2 } + 2 , \gamma _ { 3 } \} = \gamma _ { 2 } + 2 .
$$

This proves (40).

Lemma 10 (Dynkin formula on a fixed window). Under the assumptions ofLemma $7 , \mathrm { ~ } \hbar x \ S > 0 .$ . Let $u \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ and suppose that,for some $a , b \geq 0 ,$

$$
\| \nabla u ( \pmb { y } ) \| \leq C _ { u } \langle \langle \pmb { y } \rangle \rangle ^ { a } , \qquad \| \nabla ^ { 2 } u ( \pmb { y } ) \| \leq C _ { u } \langle \langle \pmb { y } \rangle \rangle ^ { b } .\tag{52}
$$

Then, $f o r 0 \leq t - s \leq S ,$

$$
\begin{array} { r l } & { P _ { s , t } u ( { \pmb x } ) = u ( { \pmb x } ) + \displaystyle \int _ { s } ^ { t } \mathbb { E } \big [ { \pmb b } ( { \pmb Z } _ { r } ^ { s , { \pmb x } } ) ^ { \top } \nabla u ( { \pmb Z } _ { r } ^ { s , { \pmb x } } ) \big ] d r } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \int _ { s } ^ { t } \alpha ( r ) \mathbb { E } \big [ ( \sigma \sigma ^ { \top } ) ( { \pmb Z } _ { r } ^ { s , { \pmb x } } ) : \nabla ^ { 2 } u ( { \pmb Z } _ { r } ^ { s , { \pmb x } } ) \big ] d r . } \end{array}\tag{53}
$$

Proof. For $R > \| { \pmb x } \|$ , define

$$
\tau _ { R } = \operatorname* { i n f } \{ r \geq s : { Z _ { r } ^ { s , { \bf { x } } } } \notin B ( \mathbf { 0 } , R ) \} .
$$

On the closed ball ${ \overline { { B } } } ( \mathbf { 0 } , R )$ , the coefficients and the first two derivatives of u are bounded, since they are continuous on compact sets. Applying Ito’s formula up toˆ $t \wedge \tau _ { R }$ gives

$$
\begin{array} { r l } & { u ( Z _ { t \wedge \tau _ { R } } ^ { s , \boldsymbol { x } } ) = u ( \boldsymbol { x } ) + \displaystyle \int _ { s } ^ { t \wedge \tau _ { R } } b ( Z _ { r } ^ { s , \boldsymbol { x } } ) ^ { \top } \nabla u ( Z _ { r } ^ { s , \boldsymbol { x } } ) d r } \\ & { \quad \quad \quad \quad + \displaystyle \frac { 1 } { 2 } \int _ { s } ^ { t \wedge \tau _ { R } } \alpha ( r ) ( \sigma \sigma ^ { \top } ) ( Z _ { r } ^ { s , \boldsymbol { x } } ) : \nabla ^ { 2 } u ( Z _ { r } ^ { s , \boldsymbol { x } } ) d r } \\ & { \quad \quad \quad \quad + \displaystyle \int _ { s } ^ { t \wedge \tau _ { R } } \sqrt { \alpha ( r ) } \nabla u ( Z _ { r } ^ { s , \boldsymbol { x } } ) ^ { \top } \sigma ( Z _ { r } ^ { s , \boldsymbol { x } } ) d B _ { r } . } \end{array}
$$

On $B ( \mathbf { 0 } , R )$ , the stochastic-integral integrand is bounded. Since $\alpha ( r ) \leq \alpha _ { \mathrm { m a x } }$ and $t - s < \infty$ , the stopped stochastic integral is square-integrable and has expectation zero. Taking expectations in the stopped identity and applying Fubini’s theorem, we obtain therefore gives

$$
\begin{array} { r l } & { \mathbb { E } u ( Z _ { t \wedge \tau _ { R } } ^ { s , x } ) = u ( x ) + \displaystyle \int _ { s } ^ { t } \mathbb { E } \big [ \mathbf { 1 } _ { \{ r \leq \tau _ { R } \} } b ( Z _ { r } ^ { s , x } ) ^ { \top } \nabla u ( Z _ { r } ^ { s , x } ) \big ] d r } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \int _ { s } ^ { t } \alpha ( r ) \mathbb { E } \big [ \mathbf { 1 } _ { \{ r \leq \tau _ { R } \} } ( \sigma \sigma ^ { \top } ) ( Z _ { r } ^ { s , x } ) : \nabla ^ { 2 } u ( Z _ { r } ^ { s , x } ) \big ] d r . } \end{array}
$$

It remains to remove that the stopping term can be safely removed by letting $R \to \infty$ . We use the polynomial derivative bounds and Lemma 7 to control the stopped terminal term and the two time integrals. The fundamental theorem of calculus and the gradient bound give

$$
| u ( { \pmb y } ) | \leq | u ( { \pmb 0 } ) | + \| { \pmb y } \| \int _ { 0 } ^ { 1 } \| \nabla u ( \theta { \pmb y } ) \| d \theta \leq C \langle \langle { \pmb y } \rangle \rangle ^ { a + 1 } .\tag{54}
$$

Linear growth of b and σ gives

$$
| b ( \pmb { y } ) ^ { \top } \nabla u ( \pmb { y } ) | \leq C \langle \langle \pmb { y } \rangle ^ { a + 1 } , \qquad | ( \sigma \sigma ^ { \top } ) ( \pmb { y } ) : \nabla ^ { 2 } u ( \pmb { y } ) | \leq C \langle \langle \pmb { y } \rangle \rangle ^ { b + 2 } .\tag{55}
$$

These bounds control, respectively, the stopped terminal term, the drift integrand, and the second-order integrand. Since $t - s \leq S$ , Lemma 7, applied at the orders $a + 1$ and $b + 2 .$ , gives

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \langle [ \pmb { Z } _ { r } ^ { s , \pmb { x } } \rangle \rangle ^ { a + 1 } < \infty , \qquad \mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \langle [ \pmb { Z } _ { r } ^ { s , \pmb { x } } \rangle \rangle ^ { b + 2 } < \infty .
$$

The assumptions of Lemma 7 ensure nonexplosion on finite intervals, so $\tau _ { R } \to \infty$ almost surely and $Z _ { t \wedge \tau _ { R } } ^ { s , x } \to Z _ { t } ^ { s , x }$ almost surely. By (54), since $t \wedge \tau _ { R } \in [ s , t ]$

$$
| u ( { \pmb Z } _ { t \wedge \tau _ { R } } ^ { s , { \pmb x } } ) | \leq C \langle \langle { \pmb Z } _ { t \wedge \tau _ { R } } ^ { s , { \pmb x } } \rangle | ^ { a + 1 } \leq C \operatorname* { s u p } _ { s \leq r \leq t } \langle \langle { \pmb Z } _ { r } ^ { s , { \pmb x } } \rangle \rangle ^ { a + 1 } .
$$

The final random variable is integrable by Lemma $^ { 7 , }$ as stated in (26). Therefore the dominated convergence theorem gives

$$
\mathbb { E } u ( Z _ { t \wedge \tau _ { R } } ^ { s , { \boldsymbol { x } } } ) \longrightarrow \mathbb { E } u ( Z _ { t } ^ { s , { \boldsymbol { x } } } ) = P _ { s , t } u ( { \boldsymbol { x } } ) .
$$

The same argument applies to the two time-integral terms. By (55), their integrands are bounded respectively by

$$
C \langle \langle Z _ { r } ^ { s , x } \rangle \rangle ^ { a + 1 } \qquad \mathrm { a n d } \qquad C \langle \langle Z _ { r } ^ { s , x } \rangle \rangle ^ { b + 2 } .
$$

The corresponding path suprema are integrable by (26). Since $\mathbf { 1 } _ { \{ r \leq \tau _ { R } \} }  1$ almost surely, dominated convergence therefore removes the stopping from both time integrals. Letting $R \to \infty$ in the stopped expectation identity gives (53). □

The $p \mathrm { - }$ moment consequence of Lemma 9 already provides the stochastic-approximation estimate needed below. It remains to record the corresponding first-order approximation over one SDE interval. Proposition 12 will apply these two estimates to the same backward function.

Lemma 11 (One-step SDE approximation). Under the assumptions ofLemma 7,fix $v , \kappa \geq 0$ and let $u \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ satisfy, for some $C _ { u } < \infty$

$$
\| \nabla u ( \pmb { y } ) \| \leq C _ { u } \langle \pmb { y } \rangle \rangle ^ { v } , \qquad \| \nabla ^ { 2 } u ( \pmb { y } ) \| \leq C _ { u } \langle \pmb { \langle y \rangle } \rangle ^ { v + \kappa } .\tag{56}
$$

Set $q : = v + \kappa + 2 .$ . Then, uniformly in k and x,

$$
\left| P _ { t _ { k } , t _ { k + 1 } } u ( { \pmb x } ) - u ( { \pmb x } ) - \alpha _ { k } { \pmb b } ( { \pmb x } ) ^ { \top } \nabla u ( { \pmb x } ) \right| \le C \alpha _ { k } ^ { 2 } \langle \langle { \pmb x } \rangle ^ { q } ,\tag{57}
$$

where C may depend on the fixed coefficient data, $v , \kappa , C _ { u } ,$ and $\alpha _ { \mathrm { m a x } }$ , but not on k $o r \mathbf { \Omega } _ { \mathbf { 4 } }$

Proof. Set $b _ { u } ( { \pmb y } ) : = { \pmb b } ( { \pmb y } ) ^ { \top } \nabla u ( { \pmb y } )$ . The derivative hypotheses satisfy Lemma 10 with $a = v$ and $b = v + \kappa$ . Since $\alpha _ { k } \leq \alpha _ { \mathrm { m a x } }$ and $\alpha ( t ) = \alpha _ { k }$ on $[ t _ { k } , t _ { k + 1 } )$ , Dynkin’s formula gives

$$
\begin{array} { r l } & { P _ { t _ { k } , t _ { k + 1 } } u ( { \pmb x } ) = u ( { \pmb x } ) + \displaystyle \int _ { 0 } ^ { \alpha _ { k } } \mathbb { E } [ b _ { u } ( { \pmb Z } _ { t _ { k + r } } ^ { t _ { k } , { \pmb x } } ) ] d r } \\ & { \qquad + \displaystyle \frac { \alpha _ { k } } { 2 } \displaystyle \int _ { 0 } ^ { \alpha _ { k } } \mathbb { E } \left[ ( \sigma \sigma ^ { \top } ) ( { \pmb Z } _ { t _ { k } + r } ^ { t _ { k } , { \pmb x } } ) : \nabla ^ { 2 } u ( { \pmb Z } _ { t _ { k } + r } ^ { t _ { k } , { \pmb x } } ) \right] d r . } \end{array}\tag{58}
$$

Since $\begin{array} { r } { \alpha _ { k } b _ { u } ( { \pmb x } ) = \int _ { 0 } ^ { \alpha _ { k } } b _ { u } ( { \pmb x } ) \ d t } \end{array}$ dr, subtracting $u ( { \pmb x } ) + \alpha _ { k } b _ { u } ( { \pmb x } )$ gives the exact identity

$$
\begin{array} { r l } & { P _ { t _ { k } , t _ { k + 1 } } u ( { \pmb x } ) - u ( { \pmb x } ) - \alpha _ { k } b _ { u } ( { \pmb x } ) } \\ & { \quad = \displaystyle \int _ { 0 } ^ { \alpha _ { k } } \mathbb { E } \left[ b _ { u } ( \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , { \pmb x } } ) - b _ { u } ( { \pmb x } ) \right] d r } \\ & { \quad \quad + \displaystyle \frac { \alpha _ { k } } { 2 } \int _ { 0 } ^ { \alpha _ { k } } \mathbb { E } \left[ ( \sigma \sigma ^ { \top } ) ( \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , { \pmb x } } ) : \nabla ^ { 2 } u ( \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , { \pmb x } } ) \right] d r . } \end{array}\tag{59}
$$

We bound the two terms on the right-hand side in turn. To control the first, we use the displacement of the restarted diffusion. The coefficient assumptions give

$$
\| \pmb { b } ( \pmb { y } ) \| + \| \pmb { \sigma } ( \pmb { y } ) \| _ { F } \leq C \langle \pmb { y } \rangle , \qquad \| D \pmb { b } ( \pmb { y } ) \| \leq C .\tag{60}
$$

All moment estimates below use (26) with $S = \alpha _ { \mathrm { m a x } }$ . For $0 \leq r \leq \alpha _ { k } , ( 3 )$ gives

$$
Z _ { t _ { k } + r } ^ { t _ { k } , x } - x = \int _ { 0 } ^ { r } b ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) d s + \sqrt { \alpha _ { k } } \int _ { 0 } ^ { r } \sigma ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) d B _ { t _ { k } + s } .
$$

The Cauchy–Schwarz inequality, Ito isometry, linear growth of the coefficients, and the restarted-SDE moment boundˆ (26) yield

$$
\begin{array} { r l } & { \mathbb { E } \| Z _ { t _ { k } + r } ^ { t _ { k } , x } - x \| ^ { 2 } \leq 2 \mathbb { E } \left\| \displaystyle \int _ { 0 } ^ { r } b ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) d s \right\| ^ { 2 } + 2 \alpha _ { k } \mathbb { E } \left\| \displaystyle \int _ { 0 } ^ { r } \sigma ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) d B _ { t _ { k } + s } \right\| ^ { 2 } } \\ & { \qquad \leq 2 r \displaystyle \int _ { 0 } ^ { r } \mathbb { E } \| b ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) \| ^ { 2 } d s + 2 \alpha _ { k } \displaystyle \int _ { 0 } ^ { r } \mathbb { E } \| \sigma ( Z _ { t _ { k } + s } ^ { t _ { k } , x } ) \| _ { F } ^ { 2 } d s } \\ & { \qquad \leq C r \displaystyle \int _ { 0 } ^ { r } \mathbb { E } \langle Z _ { t _ { k } + s } ^ { t _ { k } , x } \rangle ^ { 2 } d s + C \alpha _ { k } \displaystyle \int _ { 0 } ^ { r } \mathbb { E } \langle Z _ { t _ { k } + s } ^ { t _ { k } , x } \rangle ^ { 2 } d s } \\ & { \qquad \leq C ( r ^ { 2 } + \alpha _ { k } r ) \langle x \rangle ^ { 2 } . } \end{array}\tag{61}
$$

The product rule gives

$$
D b _ { u } ( { \pmb y } ) = D { \pmb b } ( { \pmb y } ) ^ { \top } \nabla u ( { \pmb y } ) + \nabla ^ { 2 } u ( { \pmb y } ) { \pmb b } ( { \pmb y } ) .
$$

Using (56) and (60),

$$
\begin{array} { l } { \| D b _ { u } ( { \pmb y } ) \| \le \| D { \pmb b } ( { \pmb y } ) \| \| \nabla u ( { \pmb y } ) \| + \| \nabla ^ { 2 } u ( { \pmb y } ) \| \| { \pmb b } ( { \pmb y } ) \| } \\ { \le C \langle \langle { \pmb y } \rangle ^ { v } + C \langle \langle { \pmb y } \rangle ^ { v + \kappa + 1 } } \\ { \le C \langle \langle { \pmb y } \rangle ^ { v + \kappa + 1 } , } \end{array}\tag{62}
$$

where the last inequality uses $\kappa \geq 0$ and $\langle \langle \pmb { y } \rangle \rangle \geq 1$ . The fundamental theorem of calculus along the segment from x to y gives

$$
b _ { u } ( { \pmb y } ) - b _ { u } ( { \pmb x } ) = \int _ { 0 } ^ { 1 } D b _ { u } ( { \pmb x } + \theta ( { \pmb y } - { \pmb x } ) ) ^ { \top } ( { \pmb y } - { \pmb x } ) d \theta .
$$

Together with the polynomial translation inequality (17), this implies

$$
\begin{array} { r } { | b _ { u } ( \pmb { y } ) - b _ { u } ( \pmb { x } ) | \leq C \| \pmb { y } - \pmb { x } \| \{ \langle \pmb { x } \rangle \} ^ { v + \kappa + 1 } + \langle \pmb { \langle y \rangle } \rangle ^ { v + \kappa + 1 } \} . } \end{array}
$$

Set $\pmb { y } = \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , \pmb { x } }$ . Cauchy–Schwarz gives

$$
\begin{array} { r l } & { \mathbb { E } \lvert b _ { u } ( \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , x } ) - b _ { u } ( \pmb { x } ) \rvert } \\ & { \quad \le C \left( \mathbb { E } \Vert \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , x } - \pmb { x } \Vert ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad \times \left\{ \langle \pmb { \mathscr { x } } \rangle ^ { v + \kappa + 1 } + \left( \mathbb { E } \langle \pmb { Z } _ { t _ { k } + r } ^ { t _ { k } , x } \rangle \rangle ^ { 2 ( v + \kappa + 1 ) } \right) ^ { 1 / 2 } \right\} . } \end{array}\tag{63}
$$

By (61), the first factor is at most $C ( r + \sqrt { \alpha _ { k } r } ) \langle \langle \pmb { x } \rangle \rangle$ , using $\sqrt { r ^ { 2 } + \alpha _ { k } r } \leq r + \sqrt { \alpha _ { k } r }$ . By (26) with moment order $2 ( v + \kappa + 1 )$ , the second factor is at most $\dot { C } \langle \langle { \pmb x } \rangle \rangle ^ { v + \kappa + 1 }$ . Thus

$$
\mathbb { E } | b _ { u } ( Z _ { t _ { k } + r } ^ { t _ { k } , x } ) - b _ { u } ( \pmb { x } ) | \leq C ( r + \sqrt { \alpha _ { k } r } ) \langle \pmb { x } \rangle ) ^ { v + \kappa + 2 } .\tag{64}
$$

For the second term in (59), linear growth of σ and the Hessian bound give

$$
| ( \sigma \sigma ^ { \top } ) ( \pmb { y } ) : \nabla ^ { 2 } u ( \pmb { y } ) | \leq C \langle \langle \pmb { y } \rangle \rangle ^ { v + \kappa + 2 } .
$$

Using (26) with moment order q yields

$$
\mathbb { E } \left| ( \sigma \sigma ^ { \top } ) ( \boldsymbol { Z } _ { t _ { k } + r } ^ { t _ { k } , \boldsymbol { x } } ) : \nabla ^ { 2 } u ( \boldsymbol { Z } _ { t _ { k } + r } ^ { t _ { k } , \boldsymbol { x } } ) \right| \leq C \langle \langle \boldsymbol { x } \rangle \rangle ^ { q } .\tag{65}
$$

Substituting (64) and (65) into (59) gives

$$
\begin{array} { r l r } {  { | P _ { t _ { k } , t _ { k + 1 } } u ( \pmb { x } ) - u ( \pmb { x } ) - \alpha _ { k } b _ { u } ( \pmb { x } ) | } } \\ & { } & { \le C \langle \langle \pmb { x } \rangle \rangle ^ { q } \{ \int _ { 0 } ^ { \alpha _ { k } } ( r + \sqrt { \alpha _ { k } r } ) d r + \alpha _ { k } \int _ { 0 } ^ { \alpha _ { k } } d r \} . } \end{array}
$$

Here the factor $1 / 2$ has been absorbed into C, and

$$
\int _ { 0 } ^ { \alpha _ { k } } \left( r + { \sqrt { \alpha _ { k } r } } \right) d r = { \frac { 7 } { 6 } } \alpha _ { k } ^ { 2 } , \qquad \alpha _ { k } \int _ { 0 } ^ { \alpha _ { k } } d r = \alpha _ { k } ^ { 2 } .
$$

Since $b _ { u } ( { \pmb x } ) = { \pmb b } ( { \pmb x } ) ^ { \top } \nabla u ( { \pmb x } )$ , absorbing the numerical constants into C proves (57).

Proposition 12 (Primitive finite window verification). Under Assumption $^ { l , }$ for $m \leq n \leq N _ { T } ( m )$ satisfying $\mathbb { E } \langle \langle X _ { m } \rangle ) ^ { p } <$ ∞,

$$
| \mathbb { E } V ( X _ { n } ) - \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( X _ { m } ) ] | \leq C _ { T } \mathbb { E } \langle X _ { m } \rangle ^ { p } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .\tag{66}
$$

$H ,$ in addition, the stepsizes are nonincreasing, then

$$
\begin{array} { r } { \left| \mathbb { E } V ( \pmb { X } _ { n } ) - \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \pmb { X } _ { m } ) ] \right| \leq C _ { T } \alpha _ { m } \mathbb { E } \langle \pmb { \langle X } _ { m } \rangle \rangle ^ { p } . } \end{array}
$$

If, in addition, $\begin{array} { r } { M _ { p } : = \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \langle \langle X _ { k } \rangle \rangle ^ { p } < \infty , } \end{array}$ , then

$$
\mathbb { E } V ( \mathbf { X } _ { n } ) \leq \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \mathbf { X } _ { m } ) ] + C _ { T } M _ { p } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .\tag{67}
$$

Therefore thefinite window approximation (72) holds with $\epsilon _ { m } = 0$ and $\rho ( s ) = C _ { T } M _ { p } \alpha ( s )$

Proof. of Proposition 12 We first establish the discrete fixed-window p-moment bound. We then apply the one-step SDE approximation and the p-moment specialization of Lemma 9 to the same backward function. Both approximate the same first-order expression. Finally, we telescope the resulting one-step defect over the finite window to obtain (66).

For the remainder of the proof se $S _ { T } : = T + \alpha _ { \mathrm { m a x } } .$ . If m $\leq k < n \leq N _ { T } ( m )$ , all restarted-SDE and backward semigroup intervals used below have length at most $S _ { T }$ . Hence Lemmas 7 and 8 apply uniformly with $S = S _ { T }$

Discrete window stability. From $\pmb { X } _ { k + 1 } = \pmb { X } _ { k } + \alpha _ { k } \pmb { H } _ { k + 1 }$ and weighted convexity,

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \pmb { X } _ { k + 1 } \rangle ^ { p } \mid \mathcal { F } _ { k } ] \leq ( 1 + \alpha _ { k } ) ^ { p - 1 } \lbrace \langle \pmb { X } _ { k } \rangle ^ { p } + \alpha _ { k } \mathbb { E } [ \| \pmb { H } _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] \rbrace } \\ & { \qquad \leq ( 1 + \alpha _ { k } ) ^ { p - 1 } ( 1 + M \alpha _ { k } ) \langle \pmb { X } _ { k } \rangle \rbrace ^ { p } } \\ & { \qquad \leq \exp \lbrace ( p - 1 + M ) \alpha _ { k } \rbrace \langle \pmb { X } _ { k } \rangle \rbrace ^ { p } . } \end{array}
$$

Iterating and using $t _ { k } - t _ { m } \leq T + \alpha _ { \operatorname* { m a x } }$ gives

$$
\mathbb { E } [ \langle X _ { k } \rangle ) ^ { p } \mid \mathcal { F } _ { m } ] \leq C _ { T } \langle \langle X _ { m } \rangle ) ^ { p } , \qquad m \leq k \leq N _ { T } ( m ) ,\tag{68}
$$

where one may take $C _ { T } = \exp \{ ( p - 1 + M ) ( T + \alpha _ { \mathrm { m a x } } ) \}$

One-step SDE–stochastic-approximation comparison. Fix $m \le k < n \le N _ { T } ( m )$ and write $u = u _ { k + 1 , n } ~ =$ $P _ { t _ { k + 1 } , t _ { n } } V$ , with $q : = v + \kappa + 2 \leq p$ . By (27), the hypotheses of both the SDE approximation and the conditional Taylor bound hold uniformly on the window, with derivative constant $C _ { T }$

Applying Lemma 11 at $\pmb { x } = \pmb { X } _ { k }$ gives

$$
\left| P _ { t _ { k } , t _ { k + 1 } } u ( \boldsymbol { X } _ { k } ) - u ( \boldsymbol { X } _ { k } ) - \alpha _ { k } b ( \boldsymbol { X } _ { k } ) ^ { \top } \nabla u ( \boldsymbol { X } _ { k } ) \right| \leq C _ { T } \alpha _ { k } ^ { 2 } \langle \boldsymbol { X } _ { k } \rangle ^ { q } .
$$

For the stochastic-approximation estimate, apply (40) with $\gamma _ { 1 } = v , \gamma _ { 2 } = v + \kappa , \pmb { x } = \pmb { X } _ { k } , h = \alpha _ { k } , \pmb { H } = \pmb { H } _ { k + 1 } .$ , and $\mathcal { F } = \mathcal { F } _ { k }$ . The oracle bound (14) supplies the required p-moment, since $p \ge q = \gamma _ { 2 } + 2 .$ . Using $\pmb { X } _ { k + 1 } = \pmb { X } _ { k } + \alpha _ { k } \pmb { H } _ { k + 1 }$ and $\mathbb { E } [ H _ { k + 1 } \mid { \mathcal { F } } _ { k } ] = b ( X _ { k } )$ gives

$$
\begin{array} { r } { \left| \mathbb { E } [ u ( { X } _ { k + 1 } ) \mid \mathcal { F } _ { k } ] - u ( { X } _ { k } ) - \alpha _ { k } b ( { X } _ { k } ) ^ { \top } \nabla u ( { X } _ { k } ) \right| \le C _ { T } \alpha _ { k } ^ { 2 } \langle \langle { X } _ { k } \rangle ^ { q } . } \end{array}
$$

Both quantities are therefore within $C _ { T } \alpha _ { k } ^ { 2 } \langle \langle X _ { k } \rangle \rangle ^ { p }$ of the same first-order expression

$$
u ( { X } _ { k } ) + \alpha _ { k } \pmb { b } ( { X } _ { k } ) ^ { \top } \nabla u ( { X } _ { k } ) ,
$$

where we used $q \leq p$ and enlarged $C _ { T }$ if needed. The triangle inequality gives, after enlarging $C _ { T }$

$$
\left| \mathbb { E } [ u ( X _ { k + 1 } ) \mid { \mathcal { F } } _ { k } ] - P _ { t _ { k } , t _ { k + 1 } } u ( X _ { k } ) \right| \leq C _ { T } \alpha _ { k } ^ { 2 } \langle \langle X _ { k } \rangle \rangle ^ { p } .
$$

Using $\boldsymbol { u } = \boldsymbol { u } _ { k + 1 , n }$ and $P _ { t _ { k } , t _ { k + 1 } } u _ { k + 1 , n } = u _ { k , n }$ gives the one-step defect:

$$
\begin{array} { r } { | \mathbb { E } [ u _ { k + 1 , n } ( X _ { k + 1 } ) \mid \mathcal { F } _ { k } ] - u _ { k , n } ( X _ { k } ) | \leq C _ { T } \alpha _ { k } ^ { 2 } \langle \langle X _ { k } \rangle \rangle ^ { p } . } \end{array}\tag{69}
$$

Passage to the finite window. Since $u _ { n , n } = V$ and $u _ { k , n } = P _ { t _ { k } , t _ { k + 1 } } u _ { k + 1 , n }$ , telescoping and the tower property give

$$
\begin{array} { r l } {  { \mathbb { E } [ V ( \mathbf { X } _ { n } ) \mid \mathcal { F } _ { m } ] - P _ { t _ { m } , t _ { n } } V ( \mathbf { X } _ { m } ) } } \\ & { = \displaystyle \sum _ { k = m } ^ { n - 1 } \mathbb { E } [ \mathbb { E } [ u _ { k + 1 , n } ( \mathbf { X } _ { k + 1 } ) \mid \mathcal { F } _ { k } ] - u _ { k , n } ( \mathbf { X } _ { k } ) \mid \mathcal { F } _ { m } ] . } \end{array}
$$

Using (69) and (68) therefore gives

$$
\begin{array} { r l } { \displaystyle \big | \mathbb { E } \big [ V ( \pmb { X } _ { n } ) \mid \mathcal { F } _ { m } \big ] - P _ { t _ { m } , t _ { n } } V ( \pmb { X } _ { m } ) \big | \leq C _ { T } \langle \pmb { X } _ { m } \rangle \rangle ^ { p } \sum _ { k = m } ^ { n - 1 } \alpha _ { k } ^ { 2 } } & { } \\ { = C _ { T } \langle \pmb { X } _ { m } \rangle ^ { p } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s . } & { } \end{array}\tag{70}
$$

where the equality uses $t _ { k + 1 } - t _ { k } = \alpha _ { k }$ and the piecewise-constant definition of $\alpha ( s )$ . Taking expectations and using $\left| \mathbb { E } Y \right| \leq \mathbb { E } \left| Y \right| { \mathrm { g i v e s ~ } } ( 6 6 )$ . If the stepsizes are nonincreasing, then

$$
\int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s \leq \alpha _ { m } ( t _ { n } - t _ { m } ) \leq ( T + \alpha _ { \operatorname* { m a x } } ) \alpha _ { m } ,
$$

so the preceding estimate gives the stated $C _ { T } \alpha _ { m } \langle \langle X _ { m } \rangle \rangle ^ { p }$ bound after enlarging $C _ { T }$ . Taking expectations gives the endpoint assertion of the proposition.

If $\begin{array} { r } { M _ { p } = \operatorname* { s u p } _ { k } \mathbb { E } \langle \langle X _ { k } \rangle \rangle ^ { p } < \infty } \end{array}$ , the comparison also gives

$$
\mathbb { E } V ( \mathbf { X } _ { n } ) \leq \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \mathbf { X } _ { m } ) ] + C _ { T } M _ { p } \int _ { t _ { m } } ^ { t _ { n } } \alpha ( s ) d s .
$$

Thus (72) holds with $\epsilon _ { m } = 0$ and $\rho ( s ) = C _ { T } M _ { p } \alpha ( s )$ , as required. This proves all claims of the proposition. □

## B Proof of the Rate Transfer Theorem

This appendix combines the finite-window comparison with the SDE Lyapunov envelope and restart argument to prove the Rate Transfer Theorem.

The proof preserves the modular development of the transfer argument. The technical lemmas establish the primitive finite-window comparison; that comparison supplies the input to the abstract restart theorem; the final proof below applies the abstract theorem under the Lyapunov hypothesis in Theorem 2. We give the abstract transfer result, the primitive verification, and their proofs in separate modules.

The main result proved in this appendix section is Theorem 2. Its proof combines the primitive finite-window verification with the abstract SDE transfer result.

The SDE Transfer Theorem has two main inputs: a Lyapunov bound for the associated SDE and a finite window approximation comparing the stochastic approximation with that SDE. In applications, these are the two main conditions that must be verified.

We now state one endpoint transfer theorem for the stochastic approximation over fixed diffusion time windows. Its two inputs are a Lyapunov generator inequality for the restarted SDE and a mean finite window approximation for the stochastic approximation. For the endpoint transfer machinery assume

$$
\alpha _ { n } > 0 , \qquad \alpha _ { \mathrm { m a x } } : = \operatorname* { s u p } _ { n } \alpha _ { n } < \infty , \qquad \sum _ { n = 0 } ^ { \infty } \alpha _ { n } = \infty .
$$

Recall the restarted process $\boldsymbol { Z ^ { s , x } }$ and its evolution family $P _ { s , t }$ defined in Section 2.3. Its composition property (4) will be used repeatedly. The scalar Lyapunov factor is

$$
R ( s , t ) = \exp \left\{ - \int _ { s } ^ { t } \lambda ( r ) d r \right\} .
$$

Its composition property is given by (7).

The endpoint transfer route combines a continuous estimate with a finite window comparison:

$$
\mathrm { c o n t i n u o u s ~ e s t i m a t e } + \mathrm { f i n i t e ~ w i n d o w ~ c o m p a r i s o n } \implies \mathrm { s t o c h a s t i c ~ a p p r o x i m a t i o n ~ e s t i m a t e } .
$$

We first give the abstract restart theorem, then verify its comparison hypothesis from primitive model conditions.

## B.1 Lyapunov-envelope and endpoint transfer

The main result of this subsection is the SDE transfer theorem given below. It combines a Lyapunov bound for the associated SDE with a finite window approximation between the stochastic approximation and that SDE. We state the result first and then develop the two ingredients used in its proof.

Theorem 13 (SDE Transfer Theorem). Assume $V \in C ^ { 2 } ( \mathbb { R } ^ { d } ) , V \geq 0 ,$ , and $\mathbb { E } V ( X _ { 0 } ) < \infty .$ . Assume positive stepsizes with $\alpha _ { \mathrm { m a x } } : = \operatorname* { s u p } _ { n } \alpha _ { n } <$ ∞ and $\textstyle \sum _ { n } \alpha _ { n } = \infty$ , and suppose that the restarted SDE is nonexplosive on finite intervals and that $I t { \hat { o } } ^ { \prime } s .$ formula applies locally.

Suppose, for deterministic nonnegative $\lambda , g ,$ and $\rho$ that are locally integrable,

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}\tag{71}
$$

Assume that, at the required restart starts, the finite window approximation

$$
\mathbb { E } V ( \pmb { X } _ { n } ) \le \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \pmb { X } _ { m } ) ] + \epsilon _ { m } \mathbb { E } V ( \pmb { X } _ { m } ) + \int _ { t _ { m } } ^ { t _ { n } } \rho ( s ) d s , \qquad m \le n \le N _ { T } ( m ) ,\tag{72}
$$

holds, where $\epsilon _ { m } \geq 0$ is deterministic.

Set $m _ { 0 } = 0$ and m $\nu _ { j + 1 } = N _ { T } ( m _ { j } )$ , and put $R _ { j } = R ( t _ { m _ { j } } , t _ { m _ { j + 1 } } )$ . Suppose that $\sum _ { j \ge 0 } \epsilon _ { m _ { j } } / R _ { j } < \infty$ . With

$$
C _ { \epsilon } = \exp \left\{ \sum _ { j \geq 0 } \frac { \epsilon _ { m _ { j } } } { R _ { j } } \right\} ,
$$

assume in addition that $0 \leq \lambda ( t ) \leq \bar { \lambda } <$ ∞ and set $K _ { T } = \exp \{ \bar { \lambda } ( T + \alpha _ { \mathrm { m a x } } ) \}$ . Then, for every $n \geq 0$

$$
\mathbb { E } V ( X _ { n } ) \leq C _ { \epsilon } \Bigg [ R ( 0 , t _ { n } ) \mathbb { E } V ( X _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } R ( s , t _ { n } ) \{ g ( s ) + K _ { T } \rho ( s ) \} d s \Bigg ] .\tag{73}
$$

The proof uses two results. First, Lemma 14 converts the generator Lyapunov bound into a finite time estimate for the restarted SDE. This lemma formalizes the argument that we outlined in Section 2.4. Next, Proposition 15 shows how this continuous estimate and a general finite window residual can be composed over successive restart windows. This proposition is slightly more general than Theorem 13, but it does not make explicit the underlying connection with stochastic calculus. After establishing these two results, we return to prove the theorem by specializing the residual to the finite window approximation above.

The following lemma isolates the main Lyapunov bound supplied by the stochastic differential equation. It converts the generator inequality into the finite time evolution-operator estimate used in the transfer argument.

Lemma 14 (Finite-time SDE Lyapunov envelope). Let $V \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ with $V \geq 0 .$ . Suppose that the restarted SDE is nonexplosive onfinite intervals and that Ito’sformula applies locally. Suppose thatˆ λ and g are deterministic, nonnegative, locally integrablefunctions satisfying

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

Then, $f o r s \leq t ,$

$$
P _ { s , t } V ( \pmb { x } ) \leq R ( s , t ) V ( \pmb { x } ) + \int _ { s } ^ { t } R ( r , t ) g ( r ) d r .\tag{74}
$$

Proof. Write

$$
A _ { r } = \int _ { s } ^ { r } \lambda ( q ) d q .
$$

Since $d e ^ { A _ { r } } = e ^ { A _ { r } } \lambda ( r )$ dr, Ito’s formula givesˆ

$$
d V ( Z _ { r } ^ { s , { \pmb x } } ) = \mathcal { L } _ { r } V ( Z _ { r } ^ { s , { \pmb x } } ) d r + d M _ { r } ,
$$

where M is a local martingale. Since $e ^ { A _ { r } }$ is deterministic and locally bounded on every finite interval, define

$$
d \widetilde { \cal M } _ { r } : = e ^ { \cal A _ { r } } d { \cal M } _ { r } .
$$

Then $\widetilde { M }$ is again a local martingale.

Since the integrating factor has finite variation, its quadratic covariation with $V ( Z _ { r } ^ { s , { \pmb x } } )$ is zero. The product rule and the Lyapunov inequality therefore give

$$
\begin{array} { r l } & { d \left\{ e ^ { A _ { r } } V ( Z _ { r } ^ { s , x } ) \right\} = e ^ { A _ { r } } d V ( Z _ { r } ^ { s , x } ) + V ( Z _ { r } ^ { s , x } ) d e ^ { A _ { r } } } \\ & { \phantom { d } = e ^ { A _ { r } } \left\{ \mathcal { L } _ { r } V ( Z _ { r } ^ { s , x } ) + \lambda ( r ) V ( Z _ { r } ^ { s , x } ) \right\} d r + d \widetilde { M } _ { r } } \\ & { \phantom { d } \leq e ^ { A _ { r } } g ( r ) d r + d \widetilde { M } _ { r } . } \end{array}
$$

□

Indeed, the last inequality is precisely $\begin{array} { r } { \mathcal { L } _ { r } V \leq - \lambda ( r ) V + g ( r ) } \end{array}$ . Let $\tau _ { R }$ be the exit time of $\boldsymbol { Z ^ { s , x } }$ from the ball of radius $R ,$ and let $\left( \sigma _ { N } \right)$ be a localizing sequence such that $\bar { M } ^ { \sigma _ { N } }$ is a martingale. Set η<sub>R,</sub> $_ { , N } = \tau _ { R } \wedge \sigma _ { N }$

Stopping, taking expectations, and using $g \geq 0$ gives

$$
\mathbb { E } \Big [ e ^ { A _ { t \wedge \eta _ { R , N } } } V ( \pmb { Z } _ { t \wedge \eta _ { R , N } } ^ { s , { \boldsymbol { x } } } ) \Big ] \leq V ( { \pmb { x } } ) + \int _ { s } ^ { t } e ^ { A _ { r } } g ( r ) d r .
$$

Since $V \geq 0$ , Fatou’s lemma as $N \to \infty$ gives

$$
\mathbb { E } \big [ e ^ { A _ { t \wedge \tau _ { R } } } V ( { \pmb Z } _ { t \wedge \tau _ { R } } ^ { s , { \pmb x } } ) \big ] \leq V ( { \pmb x } ) + \int _ { s } ^ { t } e ^ { A _ { r } } g ( r ) d r .
$$

By finite time nonexplosion, $\tau _ { R } \to \infty$ almost surely as $R \to \infty$ . A second application of Fatou therefore gives

$$
\mathbb { E } \big [ e ^ { A _ { t } } V ( Z _ { t } ^ { s , x } ) \big ] = e ^ { A _ { t } } \mathbb { E } [ V ( Z _ { t } ^ { s , x } ) ] = e ^ { A _ { t } } P _ { s , t } V ( \pmb { x } ) \leq V ( \pmb { x } ) + \int _ { s } ^ { t } e ^ { A _ { r } } g ( r ) d r .
$$

Thus

$$
P _ { s , t } V ( \pmb { x } ) \leq R ( s , t ) V ( \pmb { x } ) + \int _ { s } ^ { t } R ( r , t ) g ( r ) d r .\tag{75}
$$

Lemma 14 supplies the continuous time estimate used in the blockwise transfer argument.

The next proposition gives a general blockwise transfer result. Given a Lyapunov bound for the restarted SDE, it shows how a one-sided finite window comparison over successive restart windows gives a global bound for the stochastic approximation.

Proposition 15 (General block residual finite window transfer). Assume $V \in C ^ { 2 } ( \mathbb { R } ^ { d } ) , V \geq 0 ,$ , and $\mathbb { E } V ( X _ { m _ { 0 } } ) < \infty$ Assume positive stepsizes with $\alpha _ { \mathrm { m a x } } : = \operatorname* { s u p } _ { n } \alpha _ { n } < \infty$ and $\textstyle \sum _ { n } \alpha _ { n } = \infty ,$ , and suppose that the restarted SDE is nonexplosive onfinite intervals and that $I t { \hat { o } } ^ { \prime } s f { \hat { \boldsymbol { \alpha } } }$ rmula applies locally.

Suppose,for deterministic nonnegative λ and g that are locally integrable,

$$
\begin{array} { r } { \mathcal { L } _ { t } V ( \pmb { x } ) \leq - \lambda ( t ) V ( \pmb { x } ) + g ( t ) . } \end{array}
$$

Starting from $m _ { 0 } ,$ set $m _ { j + 1 } = N _ { T } ( m _ { j } )$ . Assume that, at each required restart start $m , f o r$ every m $\leq n \leq N _ { T } ( m )$

$$
\begin{array} { r } { \mathbb { E } V ( \pmb { X } _ { n } ) \leq \mathbb { E } [ P _ { t _ { m } , t _ { n } } V ( \pmb { X } _ { m } ) ] + \epsilon _ { m } \mathbb { E } V ( \pmb { X } _ { m } ) + r _ { m , n } , \qquad m \leq n \leq N _ { T } ( m ) , } \end{array}\tag{76}
$$

where $\epsilon _ { m } , r _ { m , n } \geq 0$ are deterministic. Put $R _ { j } = R ( t _ { m _ { j } } , t _ { m _ { j + 1 } } )$ and suppose $\sum _ { j \ge 0 } \epsilon _ { m _ { j } } / R _ { j } < \infty$ . With

$$
C _ { \epsilon } = \exp \left\{ \sum _ { j \geq 0 } \frac { \epsilon _ { m _ { j } } } { R _ { j } } \right\} ,
$$

for $n > m _ { 0 } ,$ , choose J so that $m _ { J } < n \leq m _ { J + 1 }$ . Then

$$
\mathbb { E } V ( \mathbf { X } _ { n } ) \leq C _ { \epsilon } \left[ R ( t _ { m _ { 0 } } , t _ { n } ) \mathbb { E } V ( \mathbf { X } _ { m _ { 0 } } ) + \int _ { t _ { m _ { 0 } } } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s + \sum _ { j = 0 } ^ { j - 1 } R ( t _ { m _ { j + 1 } } , t _ { n } ) r _ { m _ { j } , m _ { j + 1 } } + r _ { m _ { J } , n } \right] .
$$

Proof. By Lemma 14, we have the SDE bound (75). Writing out (75) over the block from $m _ { j }$ to $m _ { j + 1 }$ and taking expectations gives

$$
\mathbb { E } \Big [ P _ { { t _ { m _ { j } } } , { t _ { m _ { j + 1 } } } } V ( X _ { m _ { j } } ) \Big ] \leq R ( { t _ { m _ { j } } } , { t _ { m _ { j + 1 } } } ) \mathbb { E } V ( X _ { m _ { j } } ) + \int _ { { t _ { m _ { j } } } } ^ { { t _ { m _ { j + 1 } } } } R ( s , { t _ { m _ { j + 1 } } } ) g ( s ) d s .
$$

Applying this to the finite window approximation (76) with $m = m _ { j }$ and $n = m _ { j + 1 }$ gives

$$
\mathbb { E } V ( X _ { m _ { j + 1 } } ) \le \bigl \{ R ( t _ { m _ { j } } , t _ { m _ { j + 1 } } ) + \epsilon _ { m _ { j } } \bigr \} \mathbb { E } V ( X _ { m _ { j } } ) + \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } R ( s , t _ { m _ { j + 1 } } ) g ( s ) d s + r _ { m _ { j } , m _ { j + 1 } } .
$$

To compress the notation a bit, for $m _ { J } < n \leq m _ { J + 1 } $ , we define $u _ { j } = \mathbb { E } V ( X _ { m _ { i } } ) , 0 \leq j \leq J$ , and $u _ { J + 1 } = \mathbb { E } V ( X _ { n } )$ For this fixed terminal index, set $R _ { j } ^ { \prime } = R _ { j } = R ( t _ { m _ { j } } , t _ { m _ { j + 1 } } ) , 0 \leq \dot { j } < J ,$ and $\bar { R _ { J } ^ { \prime } } = R ( t _ { m _ { J } } , t _ { n } )$ . Define the residual used in the recursion by $r _ { j } ^ { \prime } = r _ { m _ { j } , m _ { j + 1 } } , 0 \leq j < J ,$ and $r _ { J } ^ { \prime } = r _ { m _ { J } , n } .$ Finally,

$$
G _ { j } = \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } R ( s , t _ { m _ { j + 1 } } ) g ( s ) d s , \quad 0 \leq j < J , \qquad { \mathrm { a n d ~ } } \qquad G _ { J } = \int _ { t _ { m _ { J } } } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s .
$$

Applying the finite window approximation (76) and the SDE envelope bound above gives, for every $0 \leq j \leq J ,$

$$
u _ { j + 1 } \le ( R _ { j } ^ { \prime } + \epsilon _ { m _ { j } } ) u _ { j } + G _ { j } + r _ { j } ^ { \prime } .
$$

Iterating this recursion gives the standard discrete Gronwall expansion¨

$$
u _ { J + 1 } \leq \left\{ \prod _ { j = 0 } ^ { J } ( R _ { j } ^ { \prime } + \epsilon _ { m _ { j } } ) \right\} u _ { 0 } + \sum _ { \ell = 0 } ^ { J } \left\{ \prod _ { j = \ell + 1 } ^ { J } ( R _ { j } ^ { \prime } + \epsilon _ { m _ { j } } ) \right\} \{ G _ { \ell } + r _ { \ell } ^ { \prime } \} .
$$

The inequality $R _ { J } ^ { \prime } = R ( t _ { m _ { J } } , t _ { n } ) \geq R ( t _ { m _ { J } } , t _ { m _ { J + 1 } } ) = R _ { J }$ implies that $\epsilon _ { m _ { J } } / R _ { J } ^ { \prime } \leq \epsilon _ { m _ { J } } / R _ { J }$ and also we know $\begin{array} { r } { \prod _ { j = \ell } ^ { J } R _ { j } ^ { \prime } = R ( t _ { m _ { \ell } } , t _ { n } ) } \end{array}$ . Thus,

$$
\begin{array} { r } { \displaystyle \prod _ { j = \ell } ^ { J } ( R _ { j } ^ { \prime } + \epsilon _ { m _ { j } } ) = \left( \prod _ { j = \ell } ^ { J } R _ { j } ^ { \prime } \right) \prod _ { j = \ell } ^ { J } \left( 1 + \frac { \epsilon _ { m _ { j } } } { R _ { j } ^ { \prime } } \right) \le R ( t _ { m _ { \ell } } , t _ { n } ) \exp \left\{ \sum _ { j = \ell } ^ { J } \frac { \epsilon _ { m _ { j } } } { R _ { j } } \right\} } \\ { \le C _ { \epsilon } R ( t _ { m _ { \ell } } , t _ { n } ) . \qquad } \end{array}
$$

Applying these product bounds to the Gronwall expansion gives¨

$$
u _ { J + 1 } \leq C _ { \epsilon } \left[ R ( t _ { m _ { 0 } } , t _ { n } ) u _ { 0 } + \sum _ { \ell = 0 } ^ { J - 1 } R ( t _ { m _ { \ell + 1 } } , t _ { n } ) \{ G _ { \ell } + r _ { m _ { \ell } , m _ { \ell + 1 } } \} + G _ { J } + r _ { m _ { J } , n } \right] .
$$

By the definition of $G _ { \ell }$ and the composition property,

$$
\sum _ { \ell = 0 } ^ { J - 1 } R ( t _ { m _ { \ell + 1 } } , t _ { n } ) G _ { \ell } + G _ { J } = \int _ { t _ { m _ { 0 } } } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s .
$$

Since $u _ { 0 } = \mathbb { E } V ( X _ { m _ { 0 } } )$ and $u _ { J + 1 } = \mathbb { E } V ( X _ { n } )$ , this is the displayed bound.

The preceding proposition is the general block residual statement. We now apply it with the residual induced by the finite window approximation in the theorem above.

Proof. of Theorem 13 For $n = 0$ , the result is immediate. Fix $n > 0$ and choose J so that $m _ { J } < n \leq m _ { J + 1 }$ . Applying Proposition 15 with residual $\begin{array} { r } { r _ { m , n } = \int _ { t _ { m } } ^ { t _ { n } } \rho ( s ) \ d s \qquad } \end{array}$ ds gives

$$
\begin{array} { r l } & { \mathbb { E } V ( \mathbf { X } _ { n } ) \leq C _ { \epsilon } \Bigg [ R ( 0 , t _ { n } ) \mathbb { E } V ( \mathbf { X } _ { 0 } ) + \int _ { 0 } ^ { t _ { n } } R ( s , t _ { n } ) g ( s ) d s } \\ & { \qquad + \displaystyle \sum _ { j = 0 } ^ { J - 1 } R ( t _ { m _ { j + 1 } } , t _ { n } ) \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } \rho ( s ) d s + \int _ { t _ { m _ { J } } } ^ { t _ { n } } \rho ( s ) d s \Bigg ] . } \end{array}\tag{77}
$$

The fixed window construction gives $t _ { m _ { j + 1 } } - t _ { m _ { j } } < T + \alpha _ { \mathrm { m a x } }$ . For each $0 \le j < J$ and $s \in [ t _ { m _ { j } } , t _ { m _ { j + 1 } } ] ,$

$$
R ( s , t _ { m _ { j + 1 } } ) \geq e ^ { - \bar { \lambda } ( T + \alpha _ { \operatorname* { m a x } } ) } = K _ { T } ^ { - 1 } .
$$

For the final partial interval $s \in [ t _ { m _ { J } } , t _ { n } ]$ , similarly $R ( s , t _ { n } ) \geq R ( s , t _ { m _ { J + 1 } } ) \geq K _ { T } ^ { - 1 }$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \sum _ { j = 0 } ^ { J - 1 } R ( l _ { m _ { j + 1 } } , l _ { n } ) \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } \rho ( s ) d s + \int _ { t _ { m _ { J } } } ^ { t _ { n } } \rho ( s ) d s } \\ & { \quad = \displaystyle \sum _ { j = 0 } ^ { J - 1 } \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } \frac { R ( s , t _ { n } ) } { R ( s , t _ { m _ { j + 1 } } ) } \rho ( s ) d s + \int _ { t _ { m _ { J } } } ^ { t _ { n } } \frac { R ( s , t _ { n } ) } { R ( s , t _ { n } ) } \rho ( s ) d s } \\ & { \quad \le K _ { T } \displaystyle \sum _ { j = 0 } ^ { J - 1 } \int _ { t _ { m _ { j } } } ^ { t _ { m _ { j + 1 } } } R ( s , t _ { n } ) \rho ( s ) d s + K _ { T } \displaystyle \int _ { t _ { m _ { J } } } ^ { t _ { n } } R ( s , t _ { n } ) \rho ( s ) d s } \\ & { \quad = K _ { T } \displaystyle \int _ { 0 } ^ { t _ { n } } R ( s , t _ { n } ) \rho ( s ) d s . } \end{array}\tag{78}
$$

Substitution of (78) into (77) proves the theorem.

The general bound (73) transfers the SDE Lyapunov estimate with a multiplicative factor C<sub>ϵ</sub> and an additional forcing term $K _ { T } \rho ( s )$ . In the primitive comparison of Proposition $1 2 , \epsilon _ { m } = 0$ and $\rho ( s ) = C _ { T } M _ { p } \alpha ( s )$ . Thus $C _ { \epsilon } = 1$ , and the additional forcing takes the form $C \alpha ( s )$ in the specialised bound (16).

Remark 16 (Direct SDE envelopes). The standard applications verify the generator drift condition, which is why it is the continuous input to Theorem 13. If an application already has the integrated estimate

$$
P _ { s , t } V \leq R ( s , t ) V + \int _ { s } ^ { t } R ( u , t ) g ( u ) d u ,
$$

the generator hypothesis in Theorem 13 may be replaced by this integrated SDE envelope.

Proof. of Theorem 2 By (13) and the fundamental theorem of calculus along the segment from 0 to x, $| V ( \pmb { x } ) | \leq$ $C \left. \left. \mathbf { x } \right. \right. ^ { v + 1 }$ . Since $p \ge v + \kappa + 2 > v + 1$ and $M _ { p } < \infty$ , we have $\mathbb { E } V ( X _ { 0 } ) < \infty$ . Proposition 12 and the global $p { \cdot } \mathrm { t h }$ polynomial moment closure give the finite window approximation required by

Theorem 13, with $\epsilon _ { m } = 0$ and $\rho ( s ) = C _ { T } M _ { p } \alpha ( s )$ . Hence $C _ { \epsilon } = 1$ . Applying the abstract transfer theorem using the explicitly

stated Lyapunov bound, and absorbing $K _ { T } , M _ { p }$ , and the fixed window constant into $C _ { T }$ , gives the displayed bound. This completes the proof. □

Corollary 17 (Rate inheritance for common stepsize schedules). Suppose that, at the diffusion times $t _ { n } ,$ a nonnegative sequence satisfies

$$
E _ { n } \leq e ^ { - \lambda t _ { n } } E _ { 0 } + C \int _ { 0 } ^ { t _ { n } } e ^ { - \lambda ( t _ { n } - s ) } \alpha ( s ) ^ { \delta } d s , \quad \quad \lambda , \delta > 0 .\tag{79}
$$

For constant stepsizes $\alpha _ { n } = h > 0$ , this gives

$$
E _ { n } \leq e ^ { - \lambda h n } E _ { 0 } + \frac { C } { \lambda } h ^ { \delta } ( 1 - e ^ { - \lambda h n } ) \leq e ^ { - \lambda h n } E _ { 0 } + \frac { C } { \lambda } h ^ { \delta } .
$$

Thus a fixed step gives convergence to an $O ( h ^ { \delta } )$ neighbourhood, not convergence to zero. For $\alpha _ { n } = a / ( n + n _ { 0 } ) ^ { \vartheta }$ , where $a , n _ { 0 } > 0$ and $0 < \vartheta < 1$ , integral comparison yields

$$
t _ { n } = \frac { a } { 1 - \vartheta } \big \{ ( n + n _ { 0 } ) ^ { 1 - \vartheta } - n _ { 0 } ^ { 1 - \vartheta } \big \} + O ( 1 ) ,
$$

so $e ^ { - \lambda t _ { n } } \leq C _ { 0 } e ^ { - c ( n + n _ { 0 } ) ^ { 1 - \vartheta } }$ with $c = \lambda a / ( 1 - \vartheta )$ . Moreover,

$$
\int _ { 0 } ^ { t _ { n } } e ^ { - \lambda ( t _ { n } - s ) } \alpha ( s ) ^ { \delta } d s = O ( \alpha _ { n } ^ { \delta } ) = O ( ( n + n _ { 0 } ) ^ { - \vartheta \delta } ) .
$$

Indeed, on $s \geq t _ { n } / 2$ the piecewise-constant stepsize is $O ( \alpha _ { n } )$ , whereas the contribution from $s < t _ { n } / 2$ is exponentially small. Consequently,

$$
E _ { n } = O \left( e ^ { - c ( n + n _ { 0 } ) ^ { 1 - \vartheta } } + ( n + n _ { 0 } ) ^ { - \vartheta \delta } \right) .
$$

At the harmonic boundary, $\alpha _ { n } = a / ( n + n _ { 0 } )$ and $t _ { n } = a \log ( n { + } n _ { 0 } ) { + } O ( 1 )$ . Comparing the forcing with $\alpha ( s ) ^ { \delta } \asymp e ^ { - \delta s / a }$ gives

$$
E _ { n } = \left\{ { \begin{array} { l l } { O ( n ^ { - \delta } ) , } & { \lambda a > \delta , } \\ { O ( n ^ { - \delta } \log n ) , } & { \lambda a = \delta , } \\ { O ( n ^ { - \lambda a } ) , } & { \lambda a < \delta . } \end{array} } \right.
$$

The threshold is specific to this harmonic boundary: $f o r 0 < \vartheta < 1$ , contraction is stretched exponential in iteration time and eventually dominates polynomialforcing, while in the harmonic case contraction andforcing decay polynomially and their exponents compete.

The constant-step conclusion of Corollary 17 is used here only in the first-order setting with $\epsilon _ { m } = 0 ;$ it makes no claim about a constant-step result when the comparison has a persistent multiplicative defect.

The applications illustrate different schedules. Strongly convex SGD uses a decreasing subharmonic power law, while stochastic Heavy Ball uses a constant step and contracts to an $O ( h )$ neighbourhood. The transfer theorem first gives a finite-time bound in diffusion time, and the learning-rate schedule determines how that bound appears in iteration time.

## C Stability from the associated SDE

This section proves Proposition 3 and records the power-lifting calculation used in Remark 4.

Proof. We first control the finite initial segment. Integrating the gradient of Φ along the line from 0 to x gives

$$
\begin{array} { r l } & { \displaystyle \Phi ( { \pmb x } ) = \Phi ( \mathbf { 0 } ) + \int _ { 0 } ^ { 1 } \nabla \Phi ( \theta { \pmb x } ) ^ { \top } { \pmb x } d \theta } \\ & { \qquad \le \Phi ( \mathbf { 0 } ) + C _ { \Phi } \| { \pmb x } \| \displaystyle \int _ { 0 } ^ { 1 } \langle \langle \theta { \pmb x } \rangle \rangle ^ { p - 1 } d \theta \le C \langle \langle { \pmb x } \rangle \rangle ^ { p } . } \end{array}
$$

The polynomial translation inequality and the primitive oracle bound then give, for every $k ,$

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \langle X _ { k + 1 } \rangle ] ^ { p } \mid \mathcal { F } _ { k } ] \leq C _ { p } ( \langle \langle X _ { k } \rangle ) ^ { p } + \alpha _ { k } ^ { p } \mathbb { E } [ \| H _ { k + 1 } \| ^ { p } \mid \mathcal { F } _ { k } ] ) } \\ & { \quad \quad \leq C _ { p } ( 1 + M \alpha _ { k } ^ { p } ) \langle X _ { k } \rangle \rangle ^ { p } . } \end{array}
$$

The coercive bound in the proposition and $\mathbb { E } \Phi ( X _ { 0 } ) < \infty$ first give ${ \mathbb { E } } \langle \langle X _ { 0 } \rangle \rangle ^ { p } < \infty$ . Since every $\alpha _ { k }$ is finite, induction in the conditional estimate therefore gives ${ \mathbb { E } } \langle \langle X _ { k } \rangle \rangle ^ { p } < \infty$ for every $0 \leq k \leq N$ . The polynomial upper bound on Φ gives EΦ $( X _ { k } ) < \infty$ on this finite prefix, in particular at $k = N$ . Thus no smallness condition is needed for the first N steps.

We now start the Taylor argument at an index $n \geq N$ and write $h = \alpha _ { n }$ . Taylor’s formula with integral remainder gives

$$
\Phi ( { \pmb X } _ { n } + h { \pmb H } _ { n + 1 } ) = \Phi ( { \pmb X } _ { n } ) + h \nabla \Phi ( { \pmb X } _ { n } ) ^ { \top } { \pmb H } _ { n + 1 } + h ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - \theta ) { \pmb H } _ { n + 1 } ^ { \top } \nabla ^ { 2 } \Phi ( { \pmb X } _ { n } + \theta h { \pmb H } _ { n + 1 } ) { \pmb H } _ { n + 1 } d \theta .
$$

The polynomial translation inequality $\langle \langle { \pmb x } + { \pmb y } \rangle \rangle ^ { r } \le C _ { r } ( \langle \langle { \pmb x } \rangle \rangle ^ { r } + \| { \pmb y } \| ^ { r } )$ , for $r \geq 0$ , and the Hessian bound imply that the conditional expectation of the absolute remainder is at most

$$
C h ^ { 2 } [ \langle X _ { n } \rangle ] ^ { p - 2 } { \mathbb { E } } ( \| H _ { n + 1 } \| ^ { 2 } \mid { \mathcal { F } } _ { n } ) + h ^ { p - 2 } { \mathbb { E } } ( \| H _ { n + 1 } \| ^ { p } \mid { \mathcal { F } } _ { n } ) ] .
$$

Conditional Jensen’s inequality and the primitive pth-moment bound give

$$
\mathbb { E } ( \| \boldsymbol { H _ { n + 1 } } \| ^ { 2 } \mid \mathcal { F } _ { n } ) \leq ( \mathbb { E } ( \| \boldsymbol { H _ { n + 1 } } \| ^ { p } \mid \mathcal { F } _ { n } ) ) ^ { 2 / p } \leq C \langle \langle X _ { n } \rangle \rangle ^ { 2 } .
$$

Choose $\bar { \alpha } \leq 1$ . Since $h < \bar { \alpha }$ , the remainder is therefore bounded by $C h ^ { 2 } \langle \langle X _ { n } \rangle \rangle ^ { p }$ . Using $\mathbb { E } [ H _ { n + 1 } \mid { \mathcal { F } } _ { n } ] = b ( X _ { n } )$ and $\langle \langle { \pmb x } \rangle \rangle ^ { p } \leq C _ { \Phi } \Phi ( { \pmb x } )$ , we obtain

$$
\begin{array} { r } { \mathbb { E } [ \Phi ( { \pmb X } _ { n + 1 } ) \ | \ { \mathcal F } _ { n } ] \le \Phi ( { \pmb X } _ { n } ) + h b ( { \pmb X } _ { n } ) ^ { \top } \nabla \Phi ( { \pmb X } _ { n } ) + C h ^ { 2 } \Phi ( { \pmb X } _ { n } ) . } \end{array}
$$

The global Lipschitz bound on σ implies $\| \sigma ( \pmb { x } ) \| _ { F } \leq C \langle \langle \pmb { x } \rangle \rangle$ . Hence

$$
| A ( { \pmb x } ) : \nabla ^ { 2 } \Phi ( { \pmb x } ) | \leq C \langle \langle { \pmb x } \rangle \rangle ^ { 2 } \langle \langle { \pmb x } \rangle \rangle ^ { p - 2 } \leq C \Phi ( { \pmb x } ) .
$$

For $n \geq N$ , we have $t _ { n } \geq t _ { N } \geq t _ { N _ { 0 } }$ and $\alpha ( t _ { n } ) = \alpha _ { n } = h$ . The tail generator inequality therefore yields

$$
b ( { \pmb x } ) ^ { \top } \nabla \Phi ( { \pmb x } ) = \mathcal { L } _ { t _ { n } } \Phi ( { \pmb x } ) - \frac { h } { 2 } A ( { \pmb x } ) : \nabla ^ { 2 } \Phi ( { \pmb x } ) \leq - \gamma \Phi ( { \pmb x } ) + K + C h \Phi ( { \pmb x } ) .
$$

Substitution into the Taylor estimate gives

$$
\mathbb { E } [ \Phi ( X _ { n + 1 } ) \mid { \mathcal { F } } _ { n } ] \leq { \big ( } 1 - \gamma h + C h ^ { 2 } { \big ) } \Phi ( X _ { n } ) + K h .
$$

Decrease α¯ if necessary so that $C \bar { \alpha } \leq \gamma / 2$ and $\bar { \alpha } \leq 2 / \gamma$ . Then for $n \geq N$

$$
\mathbb { E } [ \Phi ( { \mathbf { X } } _ { n + 1 } ) \mid { \mathcal { F } } _ { n } ] \leq \left( 1 - { \frac { \gamma h } { 2 } } \right) \Phi ( { \mathbf { X } } _ { n } ) + K h , \qquad 0 \leq 1 - { \frac { \gamma h } { 2 } } \leq 1 .
$$

Taking expectations, set $u _ { n } = \mathbb { E } \Phi ( X _ { n } )$ and $L = 2 K / \gamma$ . For $n \geq N$

$$
u _ { n + 1 } \leq \left( 1 - { \frac { \gamma \alpha _ { n } } { 2 } } \right) u _ { n } + { \frac { \gamma \alpha _ { n } } { 2 } } L .
$$

The coefficient lies in [0, 1], so induction from N gives $u _ { n } \le \operatorname* { m a x } \{ u _ { N } , L \}$ for every $n \geq N$ . Combining this tail bound with the finite prefix yields

$$
\operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \Phi ( X _ { n } ) \leq \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { 0 \leq k \leq N } \mathbb { E } \Phi ( X _ { k } ) , \frac { 2 K } { \gamma } \right\} < \infty .
$$

Finally, the one-sided coercive bound gives $\begin{array} { r } { M _ { p } : = \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \langle \langle X _ { n } \rangle ^ { p } \leq C _ { \Phi } \operatorname* { s u p } _ { n \geq 0 } \mathbb { E } \Phi ( X _ { n } ) < \infty . } \end{array}$

Lemma 18 (Power lifting). Let $W \in C ^ { 2 } ( \mathbb { R } ^ { d } )$ satisfy $W \geq 0 ,$ , and let $2 \leq p _ { 0 } \leq p .$ Suppose, for positive finite constants $c _ { - } , c _ { + } , a n d C _ { W }$ , that

$$
c _ { - } \langle x \rangle ^ { p _ { 0 } } \leq 1 + W ( { \pmb x } ) \leq c _ { + } \langle { \pmb x } \rangle ^ { p _ { 0 } } , \qquad \| \nabla W ( { \pmb x } ) \| \leq C _ { W } \langle { \pmb x } \rangle ^ { p _ { 0 } - 1 } , \qquad \| \nabla ^ { 2 } W ( { \pmb x } ) \| \leq C _ { W } \langle { \pmb x } \rangle ^ { p _ { 0 } - 2 } .
$$

Assume that σ is globally Lipschitz, as in Assumption 1, and that on a tail

$$
\mathcal { L } _ { t } W ( \pmb { x } ) \leq - \lambda _ { 0 } W ( \pmb { x } ) + K _ { 0 }
$$

for some $\lambda _ { 0 } > 0$ and finite $K _ { 0 } .$ . Set $r = p / p _ { 0 }$ and $\Phi = ( 1 + W ) ^ { r }$ . Then:

(a) $\Phi ( { \pmb x } ) \asymp \langle \langle { \pmb x } \rangle ^ { p } ;$

(b) for afinite constant C,

$$
\| \nabla \Phi ( { \pmb x } ) \| \le C \langle \langle { \pmb x } \rangle ^ { p - 1 } , \qquad \| \nabla ^ { 2 } \Phi ( { \pmb x } ) \| \le C \langle \langle { \pmb x } \rangle ^ { p - 2 } ;
$$

(c) there are $c , C , \bar { \alpha } > 0$ such that, wherever the lower-order Foster–Lyapunov inequality holds and $\alpha ( t ) \leq \bar { \alpha } ,$

$$
\mathcal { L } _ { t } \Phi ( { \boldsymbol { \mathbf { \mathit { x } } } } ) \leq - c \Phi ( { \boldsymbol { \mathbf { \mathit { x } } } } ) + C .
$$

Proof. The two-sided growth assumption gives $\Phi \asymp \langle \langle { \pmb x } \rangle \rangle ^ { p }$ . Differentiation yields

$$
\nabla \Phi = r ( 1 + W ) ^ { r - 1 } \nabla W , \qquad \nabla ^ { 2 } \Phi = r ( 1 + W ) ^ { r - 1 } \nabla ^ { 2 } W + r ( r - 1 ) ( 1 + W ) ^ { r - 2 } \nabla W \nabla W ^ { \top } .
$$

The growth assumptions give the bounds in (b). For $1 \leq r < 2$ , the negative exponent $r - 2$ in the Hessian’s second term is bounded using the lower coercivity bound on $1 + W$ . The generator chain rule gives

$$
\mathcal { L } _ { t } \Phi = r ( 1 + W ) ^ { r - 1 } \mathcal { L } _ { t } W + \frac { \alpha ( t ) } { 2 } r ( r - 1 ) ( 1 + W ) ^ { r - 2 } \| \sigma ^ { \top } \nabla W \| ^ { 2 } .
$$

Writing $z = 1 + W$ , the assumed drift bound gives $r z ^ { r - 1 } \mathcal { L } _ { t } W \leq - r \lambda _ { 0 } z ^ { r } + C z ^ { r - 1 } \leq - c _ { 0 } z ^ { r } + C$ by Young’s inequality. Global Lipschitzness gives $\| \sigma ( \pmb { x } ) \| _ { F } \leq C \langle \langle \pmb { x } \rangle \|$ ⟩. Combining this with the gradient bound and the lower coercivity bound shows

$$
\begin{array} { r } { ( 1 + W ) ^ { r - 2 } \| \sigma ^ { \top } \nabla W \| ^ { 2 } \leq C ( 1 + W ) ^ { r } = C \Phi . } \end{array}
$$

Consequently, $\mathcal { L } _ { t } \Phi \le - c _ { 0 } \Phi + C + C \alpha ( t ) \Phi$ . Choosing α¯ small enough to absorb the last term proves (c). □