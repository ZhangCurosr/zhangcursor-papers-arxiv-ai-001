# A Function-Space Approach to the Statistical Mechanics of Learning Dynamics

Yizhou Zhang<sup>1</sup>, Weichen Wu<sup>2</sup>, Lun Du<sup>3</sup>, Zhengjie Miao<sup>4</sup>

<sup>1</sup> Variational AI, <sup>2</sup> The Voleon Group, <sup>3</sup> Independent Researcher, <sup>4</sup> Simon Fraser University {zyizhou96,weichen.wu.1996,dulun2834}@gmail.com, zhengjie@sfu.ca

## Abstract

Deep neural networks exhibit surprisingly regular macroscopic behavior despite highly nonlinear dynamics in a vast parameter space. We develop a statistical-mechanical description of learning directly in function space, treating parameter configurations as microscopic re alizations and functions together with their dynamical operators as macroscopic variables. For mean-squared loss, the exact error dynamics are governed by the learning operator $M \ = \ J J ^ { * }$ The bare conditional stochastic dynamics supplies a dynamical Boltzmann weight, while parameter-space multiplicity contributes a function-space density of states whose local curvature defines a statistical operator B. Conditioning on a current error macrostate and integrating over the resulting local fluctuation ensemble yields the condi tional free-energy contribution $\begin{array} { r } { \Phi _ { \mathrm { f l u c } } ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } } \end{array}$ log det $( M ^ { - 1 } + B )$ + const on the active sector. At fixed spectrum, this contribution is rotationally stationary when $[ M , B ] = 0 ,$ is minimized by pairing large eigenvalues of M with small eigenvalues of B, and supplies a local restoring force contribution against rotational mismatch. For ReLU-type function spaces under mild stable statistical conditions, B takes the form $B = \sigma _ { \xi } ^ { 2 } L ^ { * } \mathcal { K } L$ , with L measuring coarse-grained second-order structure. Its low-B sector therefore corresponds, up to the bounded anisotropy of K, to directions of low structural curvature. Within this ReLU specialization, the fluctuation-induced contribution therefore supplies a preference for pairing faster relaxation with this low-curvature, data-adaptive sector. These results suggest function space as a natural macroscopic level for studying stable collective organization in learning, with a concrete neural-network model entering as a realization of the macroscopic statistical theory rather than defining its form from the outset.

## 1 Introduction

Deep neural networks pose an unusual problem for theory. Their training dynamics arise from strongly nonlinear interactions among an enormous number of parameters, yet their macroscopic behavior is often strikingly regular. Across several learning domains, performance obeys smooth scaling relations over large changes in model size, data, and compute, while appropriate parameterizations can make optimization hyperparameters transferable across orders of magnitude in width Hestness et al. (2017); Kaplan et al. (2020); Yang et al. (2021). Regular organization also appears within the learned dynamics and representations themselves: neural-network training repeatedly develops characteristic spectral structure Cohen et al. (2021), and learned functions exhibit systematic preferences for smooth, data-dependent directions Rahaman et al. (2019); Kadkhodaie et al. (2024). These observations are striking precisely because such regularity is not obvious from the microscopic equations of training. They raise a basic mechanistic question:

(1)

A particularly successful step in this direction has been to move from parameter space to function space. In the neural tangent kernel regime, a highly nonlinear parameterized model reduces to an approximately closed linear dynamics governed by a kernel that remains nearly fixed throughout training Jacot et al. (2018); Chizat et al. (2019). This reveals that a complicated microscopic system can admit a much simpler macroscopic description. The price of this closure, however, is that the geometry governing learning is efectively frozen. Feature-learning theories relax this restriction and allow representations, kernels, and their spectra to evolve Woodworth et al. (2020); Lauditi et al. (2025; 2026). The two regimes therefore expose a useful tension: fixing the dynamical geometry yields a simple closed description, whereas allowing the geometry itself to adapt restores an essential part of learning but also reintroduces nonlinear, self-consistent, and often modeldependent dynamics. This motivates asking whether the evolving macroscopic organization of learning can be characterized without resolving its full microscopic trajectory.

The combination of enormous microscopic dimension, strong nonlinearity, and reproducible macroscopic structure makes statistical mechanics a natural framework for this question. Statistical-mechanical ideas have long been used to study neural networks through Gibbs ensembles, high-dimensional landscapes, stochasticgradient difusions, mean-field descriptions, and collective order parameters Bahri et al. (2020); Mandt et al. (2016; 2017); Chaudhari & Soatto (2018). These approaches have established that useful macroscopic laws can emerge after coarse graining over microscopic degrees of freedom. In many existing formulations, however, the statistical description is constructed either directly in parameter space or through a set of macroscopic variables chosen for a specific model or limit. A comparatively unexplored possibility is to formulate the statistical mechanics of learning directly at the level where the learned object itself evolves: function space.

This is the perspective developed in the present work. A central methodological distinction is that a neuralnetwork model is treated as a microscopic realization of the statistical theory rather than as its starting point. We first formulate the macroscopic variables and conditional statistical law in function space; a concrete architecture then determines which dynamical operators and microstate geometries are realizable, and therefore how the macroscopic law is instantiated. This does not make architecture irrelevant: both the instantaneous learning operator and the multiplicity of microscopic realizations remain model-dependent. Rather, it separates the form of the organizing principle from its model-specific realization. This distinction is particularly natural for macroscopic regularities that persist across architectures: their concrete realization may vary, while the form of their organizing mechanism need not originate from any one microscopic model.

Parameter configurations are treated as microscopic realizations, while functions and the operators governing their evolution provide the macroscopic variables. Scalar quantities such as the loss remain important observables, but they retain only the magnitude of the prediction error and discard much of its directional and spectral structure. Function space lies between these two extremes: it coarse-grains over redundant parameterizations while preserving the geometry of learning. For mean-squared loss, the exact functionspace gradient dynamics take the simple form

$$
\dot { e } = - M e , \qquad M = J J ^ { * } ,\tag{2}
$$

so the spectrum and eigendirections of M directly determine the relaxation geometry of the error. The parameterization enters this description through the multiplicity of microscopic realizations of a given functionspace state. Denoting the corresponding density of states by Ω(e), its local curvature defines

$$
\boxed { B ( r ) = - \sigma _ { \xi } ^ { 2 } \nabla _ { e } ^ { 2 } \log \Omega ( e ) \big \vert _ { e = r } . }\tag{3}
$$

Thus M describes the dynamical geometry of learning, while B describes the statistical geometry induced by the underlying parameter microstates. Only the compression of B to the finite-dimensional active sector of M enters the determinants and commutators below.

Conditioning on a current error macrostate and integrating over the local function-space fluctuation ensemble yields a conditional free-energy contribution that depends on the dynamical operator,

$$
\sqrt { \Phi _ { \mathrm { f l u c } } ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } ( M ^ { - 1 } + B ) + \mathrm { c o n s t . } }\tag{4}
$$

Its rotational structure produces a sharp conditional preference. At fixed spectrum,

$$
\begin{array} { r } { \boxed { \delta _ { \mathrm { r o t } } \Phi _ { \mathrm { f l u c } } = 0 \quad \Longleftrightarrow \quad \left[ M , B \right] = 0 , } } \end{array}\tag{5}
$$

while the free-energy minimum pairs the spectra as

$$
\boxed { m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } . }\tag{6}
$$

The matched state further has positive rotational curvature in every nondegenerate pairwise direction. Accordingly, this conditional free-energy contribution supplies a local restoring thermodynamic force against operator mismatch. We do not assume that this term exhausts the full slow dynamics of $M ;$ rather, it identifies one definite thermodynamic bias contributed by the local fluctuation sector. The complete local free energy also contains a distinct macrostate term, $\begin{array} { r } { \frac { 1 } { 2 } \langle r _ { a } , M ^ { - 1 } r _ { a } \rangle _ { p } , } \end{array}$ , which supplies an error-directed orientational preference; Sec. 4.1 separates these two contributions explicitly.

We then examine the physical meaning of this abstract statistical geometry in a concrete class of function spaces. ReLU networks represent continuous piecewise-afine functions whose second-order structure is concentrated on activation-cell boundaries. After coarse graining, this structure can be represented by a linear operator $L \simeq \mathcal { C } D ^ { 2 }$ . Under mild statistical boundary conditions on the local microstate ensemble, the curvature operator become

$$
\boxed { B = \sigma _ { \xi } ^ { 2 } L ^ { * } \mathcal { K } L , }\tag{7}
$$

where $\kappa$ is a positive structural entropy metric. Hence, for an eigenmode $B \phi _ { i } = b _ { i } \phi _ { i }$

$$
b _ { i } \asymp \| L \phi _ { i } \| ^ { 2 } ,\tag{8}
$$

and the low-B sector corresponds, up to the bounded anisotropy of $\kappa .$ to directions of small coarse-grained structural curvature. Combining this identification with the thermodynamic matching condition gives

$$
\Big | m _ { \mathrm { l a r g e } } \longleftrightarrow \mathrm { l o w s t r u c t u r a l - c u r v a t u r e d a t a - a d a p t i v e s e c t o r . } \Big |\tag{9}
$$

Within this class of function spaces, the conditional thermodynamic contribution therefore favors pairing faster relaxation with the low-curvature sector of the data-weighted functional geometry.

Taken together, these results suggest that function space provides a natural macroscopic level for the statistical mechanics of learning. At this level, the learning dynamics can be separated from the statistica constraints supplied by the underlying parameterization: the former determines how function-space states evolve, while the latter determines which such states admit many microscopic realizations and how those realizations are organized. In the present work, this separation reveals a thermodynamic bias in the direction of feature evolution. More broadly, the same viewpoint may provide a route for studying other stable collective structures of learning—including representation geometry, spectral organization, and stability—without requiring a complete description of the microscopic parameter trajectory.

Contributions. Our main contributions are:

• We develop a conditional statistical-mechanical formulation of neural-network training directly in function space. The macroscopic statistical law is formulated before choosing a particular architecture, with concrete neural networks entering as microscopic realizations of that law.

• We show how parameter-space microstate multiplicity induces a function-space density of states and a local statistical curvature operator B, thereby separating the dynamical geometry M from the statistical geometry supplied by the parameterization.

• We derive the conditional fluctuation contribution to the operator free energy and prove that its rotational stationary points satisfy $[ M , B ] = 0$ On a fixed-spectrum orbit this contribution is globally minimized by reverse spectral pairing, with larger eigenvalues of M matched to smaller eigenvalues of B, and its gradient supplies a local restoring force contribution around the matched state.

• For ReLU-type function spaces under mild stable statistical boundary conditions, we show that $B = \sigma _ { \varepsilon } ^ { 2 } L ^ { * } \mathcal { K } L$ and that its spectrum measures coarse-grained structural curvature up to the bounded anisotropy of the structural entropy metric. Combining this result with the conditional operator preference yields a thermodynamic bias toward pairing faster relaxation with the low-curvature data-adaptive sector.

The remainder of the paper develops these results in three steps. Section 3 constructs the conditional statistical mechanics of error fluctuations. Section 4 analyzes the conditional operator free-energy contribution and its matching geometry. Section 5 connects the resulting microstate curvature to the structural smoothness of ReLU function spaces.

## 2 Related Work

Kernel limits and feature learning. A large body of work characterizes neural-network training through function-space kernels. In the infinite-width neural tangent kernel (NTK) limit, gradient descent is governed by an approximately fixed kernel, and diferent function-space modes are learned at rates set by the corresponding kernel eigenvalues Jacot et al. (2018). This fixed-kernel description is closely related to the lazy-training regime, in which the network remains near its initial linearization Chizat et al. (2019). Subsequent work has emphasized the distinction between such kernel regimes and richer training regimes in which the representation itself evolves Woodworth et al. (2020). More recent analyses explicitly derive adaptive kernels and evolving spectral structure in feature-learning limits Lauditi et al. (2025; 2026).

Our work concerns this latter regime, but reverses the usual order of construction. Rather than beginning from a specified neural-network model and deriving model-specific macroscopic variables, we formulate the conditional statistical mechanics at the function-space level and then ask how a concrete model realizes it. Within this formulation, the fluctuation contribution depends on the relative geometry of the dynamical operator and the parameterization-induced microstate geometry. Its rotational extrema require M to commute with a microstate-curvature operator B, and its minima pair large dynamical eigenvalues with small microstate-curvature eigenvalues. The ReLU specialization in Sec. 5 is therefore a concrete realization of the general operator-level construction rather than its starting point.

Stochastic gradient dynamics and statistical mechanics. Difusion and statistical-mechanical descriptions of stochastic optimization have a long history in machine learning. Constant-step stochastic gradient dynamics can be approximated locally by difusion or Ornstein–Uhlenbeck processes, leading to efective stationary distributions and Bayesian interpretations Mandt et al. (2016; 2017). Other work has emphasized that realistic stochastic-gradient noise is generally anisotropic and can generate genuinely nonequilibrium behavior Chaudhari & Soatto (2018). More broadly, statistical-mechanical tools such as efective energies, free energies, and high-dimensional random systems have played an important role in theoretical studies of deep learning Bahri et al. (2020). The Langevin and Fokker–Planck machinery used in our bare conditional dynamics follows the standard theory of reversible difusion processes Risken (1989); Pavliotis (2014); Jordan et al. (1998).

The statistical-mechanical object considered here difers from the usual parameter-space loss landscape. We first express mean-squared gradient flow directly in error space,

$$
{ \dot { e } } = - M e ,\tag{10}
$$

and show that isotropic stochastic forcing in the error source induces the mobility $K = M ^ { 2 }$ . This allows the bare conditional dynamics to be represented as an overdamped Langevin process with the quadratic energy

$$
U _ { M } ( e ) = \frac { 1 } { 2 } \langle e , M ^ { - 1 } e \rangle _ { p } .\tag{11}
$$

We then combine this dynamical weight with a parameter-space reference measure and push the resulting microstate ensemble forward to function space. The induced density of states $\Omega ( e )$ counts the multiplicity of

parameter microstates associated with the same coarse-grained function-space state, producing the efective contribution

$$
- \sigma _ { \xi } ^ { 2 } \log \Omega ( e )\tag{12}
$$

to the conditional free energy. Thus the entropy in our formulation is associated with parameter-space multiplicity at fixed function-space macrostate, rather than solely with the local volume of a loss minimum.

Spectral bias and smooth feature learning. Neural networks are known to exhibit a spectral bias toward learning simpler or lower-frequency components of a target function earlier in training Rahaman et al. (2019). Such behavior is commonly characterized using Fourier modes, kernel eigenfunctions, or other externally chosen spectral decompositions, and the resulting bias can depend strongly on the geometry of the data distribution. In the present work, smoothness instead emerges from the same operator geometry that enters feature learning. We derive a microstate-curvature operator

$$
\begin{array} { r } { B _ { p } ( r ) = \sigma _ { \xi } ^ { 2 } L _ { p } ^ { * } \mathcal { K } _ { p } ( r ) L , } \end{array}\tag{13}
$$

whose eigendirections are defined directly in the data-weighted function space $L ^ { 2 } ( \mathcal { X } , p )$ . Under regular structural statistics, its eigenvalues satisfy

$$
b _ { i } \asymp \| L \phi _ { i } \| ^ { 2 } ,\tag{14}
$$

so that smal $\left] - b _ { i } \right.$ identifies low structural curvature up to the bounded anisotropy of the structural metric. Combining this with the conditional operator preference gives

$$
m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } \longleftrightarrow \mathrm { l o w ~ s t r u c t u r a l - c u r v a t u r e ~ d a t a - a d a p t i v e ~ s e c t o r } .\tag{15}
$$

The resulting structural bias is therefore not imposed through a fixed Fourier basis or a fixed kernel spectrum;   
it is defined intrinsically by the data-weighted microstate geometry of the learned function space.

Piecewise-linear geometry of ReLU networks. ReLU networks represent continuous piecewise-afine functions whose input space is partitioned into activation regions. This viewpoint has been developed through spline and piecewise-linear descriptions of deep networks Balestriero & Baraniuk (2018), as well as variational and representer theorems connecting ReLU networks to spline-like function spaces and secondorder regularity Unser (2019). We use this geometric structure as the architectural input to our statistical theory. Within each activation cell the Hessian vanishes, while second-order structure is concentrated on cel boundaries through jumps of the gradient. After coarse graining, this motivates the structural field

$$
h = L c , \qquad L \simeq \mathcal { C } D ^ { 2 } .\tag{16}
$$

Under mild statistical boundary conditions on the entropy of these structural states, the corresponding function-space microstate curvature is

$$
B = \sigma _ { \xi } ^ { 2 } L ^ { * } \mathcal { K } L .\tag{17}
$$

The role of ReLU geometry in our framework is therefore to provide the structural operator whose entropy curvature enters the conditional thermodynamic preference analyzed in Sec. 4.

## 3 Conditional Statistical Mechanics of Error Fluctuations

## 3.1 Exact function-space gradient dynamics

We begin by expressing gradient descent directly in function space. Let $p ( x )$ denote the data distribution and define

$$
\mathcal { H } = L ^ { 2 } ( \mathcal { X } , p ) ,\tag{18}
$$

with inner product

$$
\langle f , g \rangle _ { p } = \int _ { \mathcal { X } } f ( x ) g ( x ) p ( x ) d x .\tag{19}
$$

For a model $f _ { \theta }$ and target function y, define the error field

$$
e _ { \theta } = f _ { \theta } - y\tag{20}
$$

and the mean-squared loss

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \| e _ { \theta } \| _ { p } ^ { 2 } .\tag{21}
$$

Let

$$
J _ { \theta } = D _ { \theta } f _ { \theta }\tag{22}
$$

denote the Jacobian mapping infinitesimal parameter perturbations to function-space perturbations. Its adjoint $J _ { \theta } ^ { * }$ is defined with respect to the parameter-space inner product and $\langle \cdot , \cdot \rangle _ { p }$ . Then

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } = J _ { \boldsymbol { \theta } } ^ { * } e _ { \boldsymbol { \theta } } ,\tag{23}
$$

and continuous-time gradient flow gives

$$
\dot { \theta } = - J _ { \theta } ^ { * } e _ { \theta } .\tag{24}
$$

Applying the chain rule,

$$
\dot { e } _ { \theta } = J _ { \theta } \dot { \theta } = - J _ { \theta } J _ { \theta } ^ { * } e _ { \theta } .\tag{25}
$$

This motivates the function-space dynamical operator

$$
\left| { \cal M } _ { \theta } \equiv J _ { \theta } J _ { \theta } ^ { * } , \right|\tag{26}
$$

which is self-adjoint and positive semidefinite. The exact error dynamics is therefore

$$
\boxed { \dot { e } _ { \theta } = - M _ { \theta } e _ { \theta } . }\tag{27}
$$

Importantly, Eq. equation 27 does not require linearizing the network around a fixed parameter configuration: M<sub>θ</sub> may evolve along the training trajectory.

For a finite-parameter model, M has finite rank. We work throughout on a fixed finite-dimensional active sector

$$
\mathcal { H } _ { a } \equiv \overline { { \mathrm { R a n } M } } , \qquad n \equiv \dim \mathcal { H } _ { a } < \infty ,\tag{28}
$$

and write $P _ { a }$ for the orthogonal projection onto ${ \mathcal { H } } _ { a }$ . The restriction of M to ${ \mathcal { H } } _ { a }$ is taken to be positive definite. All inverses, traces, determinants, and orthogonal rotations in the following sections are understood on $\mathcal { H } _ { a }$ Directions in ker M do not relax under Eq. equation 27 and are excluded from the conditional fluctuation sector.

## 3.2 Conditional Langevin dynamics and dynamical weight

We next construct the dynamical statistical weight associated with the local error dynamics. We condition on a macroscopic configuration for which J and

$$
M = J J ^ { * }\tag{29}
$$

are treated as fixed parameters of the conditional problem. This conditioning defines a family of local ensembles; by itself it does not require a dynamical separation of time scales between M and the error fluctuations.

Consider the standard constant-mobility overdamped Langevin equation

$$
d x _ { t } = - K \nabla U ( x _ { t } ) d t + \sqrt { 2 T K } d W _ { t } ,\tag{30}
$$

where U is an energy, K is a positive mobility operator, and $T$ sets the stochastic scale. Such dynamics admit the standard Itô Fokker–Planck and equilibrium structure Risken (1989); Pavliotis (2014).

Stochastic mobility. We model the stochastic source in error space as isotropic,

$$
\mathbb { E } [ d W _ { t } ] = 0 , \qquad \mathbb { E } [ d W _ { t } d W _ { t } ^ { * } ] = I d t .\tag{31}
$$

In the infinite-dimensional notation, $W _ { t }$ may be understood as a cylindrical Wiener process; because M has finite rank, $M d W _ { t }$ is well defined on $\mathcal { H } _ { a }$ . We assume that this source enters parameter space through the same Jacobian channel as the deterministic gradient,

$$
d \theta _ { \mathrm { n o i s e } } = \sqrt { 2 } \sigma _ { \xi } J ^ { * } d W _ { t } .\tag{32}
$$

With J fixed in the conditional construction,

$$
\begin{array} { r l } & { d e _ { \mathrm { n o i s e } } = J d \theta _ { \mathrm { n o i s e } } } \\ & { \qquad = \sqrt { 2 } \sigma _ { \xi } J J ^ { * } d W _ { t } = \sqrt { 2 } \sigma _ { \xi } M d W _ { t } . } \end{array}\tag{33}
$$

Combining this with Eq. equation 27 gives the bare conditional process

$$
\boxed { d e = - M e d t + \sqrt { 2 } \sigma _ { \xi } M d W _ { t } . }\tag{34}
$$

Its stochastic increment has covariance

$$
\mathbb { E } [ d e _ { \mathrm { n o i s e } } d e _ { \mathrm { n o i s e } } ^ { * } ] = 2 \sigma _ { \xi } ^ { 2 } M ^ { 2 } d t ,\tag{35}
$$

so comparison with Eq. equation 30 identifies

$$
{ \boxed { K = M ^ { 2 } } } , \qquad T = \sigma _ { \xi } ^ { 2 } .\tag{36}
$$

Dynamical energy. To represent the deterministic drift in Langevin form, we seek $U _ { M } ( e )$ such that

$$
K \nabla _ { e } U _ { M } ( e ) = M e .\tag{37}
$$

On ${ \mathcal { H } } _ { a }$ , M is invertible, and with $K = M ^ { 2 }$ this gives

$$
\nabla _ { e } U _ { M } = M ^ { - 1 } e .\tag{38}
$$

Hence

$$
\boxed { U _ { M } ( e ) = \frac { 1 } { 2 } \langle e , M ^ { - 1 } e \rangle _ { p } . }\tag{39}
$$

Equation equation 34 can therefore be written as

$$
d e = - M ^ { 2 } \nabla _ { e } U _ { M } ( e ) d t + \sqrt { 2 \sigma _ { \xi } ^ { 2 } M ^ { 2 } } d W _ { t } .\tag{40}
$$

Theorem 3.1 (Bare conditional Langevin equilibrium). For the conditional process $E q .$ equation $\it { 4 0 }$ on $\mathcal { H } _ { a }$ the density $p ( e , t \mid M )$ satisfies

$$
\begin{array} { r } { \boxed { \partial _ { t } p = \nabla _ { e } \cdot \left[ M ^ { 2 } \left( p \nabla _ { e } U _ { M } + \sigma _ { \xi } ^ { 2 } \nabla _ { e } p \right) \right] . } } \end{array}\tag{41}
$$

Its reversible stationary density with respect to the reference volume de on $\mathcal { H } _ { a }$ is

$$
\boxed { p _ { 0 } ( e \mid M ) = \cfrac { 1 } { Z _ { 0 } } \exp \left[ - \cfrac { U _ { M } ( e ) } { \sigma _ { \xi } ^ { 2 } } \right] . }\tag{42}
$$

Moreover,

$$
\boxed { \mathcal { F } _ { 0 } [ p ] = \int p ( e ) U _ { M } ( e ) d e + \sigma _ { \xi } ^ { 2 } \int p ( e ) \log p ( e ) d e }\tag{43}
$$

is non-increasing along the conditional Fokker–Planck dynamics:

$$
\left| \frac { d \mathcal { F } _ { 0 } } { d t } = - \int p ( e ) \left. \nabla _ { e } \frac { \delta \mathcal { F } _ { 0 } } { \delta p } , M ^ { 2 } \nabla _ { e } \frac { \delta \mathcal { F } _ { 0 } } { \delta p } \right. _ { p } d e \leq 0 . \right|\tag{44}
$$

Proof. The Itô forward equation for Eq. equation 40 is Eq. equation 41, with probability current

$$
j = - M ^ { 2 } \left( p \nabla _ { e } U _ { M } + \sigma _ { \xi } ^ { 2 } \nabla _ { e } p \right) .\tag{45}
$$

For $\operatorname { E q . }$ equation 42, $\nabla _ { e } p _ { 0 } = - ( p _ { 0 } / \sigma _ { \xi } ^ { 2 } ) \nabla _ { e } U _ { M }$ , so $j _ { 0 } = 0$ . The variational derivative of $\operatorname { E q . }$ equation 43 is

$$
\frac { \delta \mathcal { F } _ { 0 } } { \delta p } = U _ { M } + \sigma _ { \xi } ^ { 2 } \log p + \mathrm { c o n s t } ,\tag{46}
$$

and an integration by parts gives Eq. equation 44.

Theorem 3.1 supplies the dynamical Boltzmann weight relative to the reference volume in function space. It does not count how many parameter configurations realize a given error state. In particular, the bare noise process above is not assumed to explore parameter-fiber directions. Parameter multiplicity enters independently through the reference microstate measure introduced next.

## 3.3 Parameter microstates and the induced conditional ensemble

We take the background state z to include the conditioned dynamical operator M together with the remaining constraints supplied by the parameterization, architecture, and data, and let $\nu _ { z } ( d \theta )$ denote the corresponding reference measure over parameter microstates. To define a regular density of states without invoking an infinite-dimensional volume element, introduce a finite-dimensional retained function-space sector

$$
\mathcal { H } _ { a } \subseteq \mathcal { H } _ { R } \subseteq \mathcal { H } , \qquad N _ { R } \equiv \dim \mathcal { H } _ { R } < \infty ,\tag{47}
$$

with orthogonal projector $P _ { R }$ . The sector $\mathcal { H } _ { R }$ contains the active learning sector $\mathcal { H } _ { a } = \mathrm { R a n }$ M but may also retain additional coarse-grained function coordinates needed to define the microstate statistics. We define

$$
\Psi _ { R , z } : \theta \mapsto e _ { R } \equiv P _ { R } ( f _ { \theta } - y ) \in \mathcal { H } _ { R }\tag{48}
$$

and the corresponding pushforward reference measure

$$
\boxed { \mu _ { z } = ( \Psi _ { R , z } ) _ { \# } \nu _ { z } . }\tag{49}
$$

Here $d e _ { R }$ denotes ordinary Lebesgue volume in an orthonormal coordinate system on $\mathcal { H } _ { R }$ . We assume that the coarse-grained pushforward is regular enough to admit a density,

$$
\boxed { \mu _ { z } ( d e _ { R } ) = \Omega _ { z } ( e _ { R } ) d e _ { R } . }\tag{50}
$$

Thus $\Omega _ { z }$ is a density of parameter microstates on the retained coarse-grained function sector $\mathcal { H } _ { R } .$ . No density with respect to an infinite-dimensional “function-space volume” is assumed. The operator calculations below use only the compression of its local curvature to the active subspace ${ \mathcal { H } } _ { a }$

The following factorization is the central statistical assumption that combines the two ingredients above.

A1. Factorization of dynamical weight and microstate multiplicity. At fixed background state z and dynamical operator M, we assume that the microscopic statistical weight factorizes into a dynamical factor that depends on a parameter configuration only through its induced error state and a reference microstate measure that supplies the multiplicity of such realizations. Equivalently,

$$
\boxed { \Pi _ { * } ( d \theta \mid M , z ) \propto \exp \left[ - \frac { U _ { M } ( P _ { a } \Psi _ { R , z } ( \theta ) ) } { \sigma _ { \xi } ^ { 2 } } \right] \nu _ { z } ( d \theta ) . }\tag{51}
$$

The reference measure $\nu _ { z }$ is not assumed to be dynamically sampled by the bare J<sup>∗</sup>-channel noise. Rather, it encodes the conditional multiplicity supplied by the parameterization and other degrees of freedom included in z. Assumption A1 states that this multiplicity can be combined with the bare dynamical weighting without

an additional fiber-dependent energetic factor. Possible dependence of the reference microstate statistics on slower variables, including changes induced by an actual motion of M, is outside the partial conditional comparison performed below.

Under A1, the bare conditional dynamics supplies the Boltzmann factor ex $\mathrm { p } [ - U _ { M } ( P _ { a } e _ { R } ) / \sigma _ { \xi } ^ { 2 } ]$ on the active component, while the pushforward of $\nu _ { z }$ supplies the density of states on $\mathcal { H } _ { R }$ . Their product defines the conditional canonical ensemble

$$
\boxed { P _ { * } ( d e _ { R } \mid M , z ) = \cfrac { 1 } { Z } \exp \left[ - \cfrac { U _ { M } ( P _ { a } e _ { R } ) } { \sigma _ { \xi } ^ { 2 } } \right] \mu _ { z } ( d e _ { R } ) . }\tag{52}
$$

By Assumption A1, Eq. equation 51 is the corresponding parameter-space ensemble, and its pushforward under $\Psi _ { R , z }$ is Eq. equation 52. If Eq. equation 50 holds, then

$$
\boxed { p _ { * } ( e _ { R } \mid M , z ) = \cfrac { 1 } { Z } \Omega _ { z } ( e _ { R } ) \exp \left[ - \cfrac { U _ { M } ( P _ { a } e _ { R } ) } { \sigma _ { \xi } ^ { 2 } } \right] . }\tag{53}
$$

The two factors in Eq. equation 53 have distinct origins. The exponential factor is the dynamical weight derived from the bare conditional Langevin process, whereas $\Omega _ { z } ( e _ { R } )$ is a static density-of-states factor supplied by the parameterization. Their product is therefore an ensemble construction; Eq. equation 53 is not claimed to be the stationary law generated by Eq. equation 34 alone.

Defining

$$
\boxed { F _ { M } ( e _ { R } ; z ) = U _ { M } ( P _ { a } e _ { R } ) - \sigma _ { \xi } ^ { 2 } \log \Omega _ { z } ( e _ { R } ) , }\tag{54}
$$

the conditional ensemble takes the Gibbs form

$$
\boxed { p _ { * } ( e _ { R } \mid M , z ) = \cfrac { 1 } { Z } \exp \left[ - \cfrac { F _ { M } ( e _ { R } ; z ) } { \sigma _ { \xi } ^ { 2 } } \right] . }\tag{55}
$$

Proposition 3.2 (Microstate-weighted conditional ensemble). Let $\nu _ { z }$ be a parameter-space reference measure and let $\mu _ { z } = ( \Psi _ { R , z } ) _ { \# } \nu _ { z } . \ I f \mu _ { z } ( d e _ { R } ) = \Omega _ { z } ( e _ { R } ) d e _ { R } $ , then the factorized canonical weighting Eq. equation 51 induces the function-space ensemble $E q .$ equation 53, equivalently the Gibbs form $E q .$ equation 55 generated by $F _ { M } ( e _ { R } ; z )$

Proof. For any measurable set A,

$$
\begin{array} { l } { \Pi _ { * } ( \Psi _ { R , z } ^ { - 1 } ( A ) \mid M , z ) \propto \displaystyle \int _ { \Psi _ { R , z } ^ { - 1 } ( A ) } e ^ { - U _ { M } ( P _ { a } \Psi _ { R , z } ( \theta ) ) / \sigma _ { \xi } ^ { 2 } } \nu _ { z } ( d \theta ) } \\ { = \displaystyle \int _ { A } e ^ { - U _ { M } ( P _ { a } e _ { R } ) / \sigma _ { \xi } ^ { 2 } } \mu _ { z } ( d e _ { R } ) . } \end{array}\tag{56}
$$

Substituting $\mu _ { z } ( d e _ { R } ) = \Omega _ { z } ( e _ { R } )$ de<sub>R</sub> gives Eq. equation 53.

The parameter-space pushforward therefore contributes the entropic term $- \sigma _ { \xi } ^ { 2 } \log \Omega _ { z } ( e _ { R } )$ to the efective retained-sector free energy, independently of the bare stochastic channel used to identify $U _ { M }$

## 3.4 Local microstate curvature and the conditional fluctuation ensemble

We now condition on a current retained error macrostate $r \in \mathcal { H } _ { R }$ and characterize fluctuations along the active learning sector,

$$
e _ { R } = r + \delta e , \qquad \delta e \in \mathcal { H } _ { a } .\tag{57}
$$

Write $r _ { a } = P _ { a } r$ for the active component of the conditioned macrostate. The point r is a conditioning variable and is not assumed to minimize $F _ { M }$ . The standard constrained-ensemble construction introduces a linear source conjugate to the macrostate. In a full maximum-entropy formulation, the source is the Lagrange multiplier enforcing the chosen mean macrostate. At the local Laplace level used here, conditioning on r amounts to choosing the source so that r is a stationary point of the tilted potential,

$$
\begin{array} { r } { \boxed { \lambda _ { r } \equiv P _ { a } \nabla _ { e _ { R } } F _ { M } ( e _ { R } ; z ) \big | _ { e _ { R } = r } . } } \end{array}\tag{58}
$$

It is useful to separate the zero-order macrostate cost from the fluctuation cost and define the local excess potential

$$
\begin{array} { r } { \boxed { \Delta \widetilde { F } _ { M , r } ( \delta e ; z ) \equiv F _ { M } ( r + \delta e ; z ) - F _ { M } ( r ; z ) - \langle \lambda _ { r } , \delta e \rangle _ { p } . } } \end{array}\tag{59}
$$

Then $\Delta \widetilde { F } _ { M , r } ( 0 ; z ) = 0$ and its first variation vanishes at $\delta e = 0 ;$ , while its Hessian is exactly the Hessian of $F _ { M }$ at r. We do not require r to be the unconstrained mean or mode of the full non-Gaussian Gibbs measure; the construction is the quadratic local form of the usual Legendre/Lagrange constrained ensemble.

Define the retained-sector microstate-curvature form by

$$
\boxed { B ( r ) \equiv - \sigma _ { \xi } ^ { 2 } \nabla _ { e _ { R } } ^ { 2 } \log \Omega _ { z } ( e _ { R } ) \big | _ { e _ { R } = r } . }\tag{60}
$$

The operator entering the finite-dimensional fluctuation sector is its compression

$$
\boxed { B _ { a } ( r ) \equiv P _ { a } B ( r ) P _ { a } \vert _ { \mathcal { H } _ { a } } . }\tag{61}
$$

Since the dynamical term $U _ { M } ( P _ { a } e _ { R } )$ has active-sector Hessian

$$
\nabla _ { \mathcal { H } _ { a } } ^ { 2 } U _ { M } = M ^ { - 1 }\tag{62}
$$

, a second-order expansion of Eq. equation 59 gives

$$
\Delta \widetilde { F } _ { M , r } ( \delta e ; z ) = \frac { 1 } { 2 } \left. \delta e , H _ { a } ( r ) \delta e \right. _ { p } + o ( \| \delta e \| ^ { 2 } ) ,\tag{63}
$$

where

$$
\boxed { H _ { a } ( r ) \equiv M ^ { - 1 } + B _ { a } ( r ) . }\tag{64}
$$

Proposition 3.3 (Local conditional fluctuation ensemble). Suppose that

$$
H _ { a } ( r ) = M ^ { - 1 } + B _ { a } ( r ) \succ 0\tag{65}
$$

on ${ \mathcal { H } } _ { a }$ . Then, to quadratic order around the conditioned macrostate $r ,$ the local fluctuation ensemble is Gaussian:

$$
\boxed { p _ { \mathrm { l o c } } ( \delta e \mid r , M , z ) = \frac { 1 } { Z _ { \mathrm { l o c } } } \exp \left[ - \frac { 1 } { 2 \sigma _ { \xi } ^ { 2 } } \left. \delta e , H _ { a } ( r ) \delta e \right. _ { p } \right] , }\tag{66}
$$

with covariance

$$
\boxed { C _ { * } ( r , M ) = \sigma _ { \xi } ^ { 2 } \left[ M ^ { - 1 } + B _ { a } ( r ) \right] ^ { - 1 } . }\tag{67}
$$

Proof. Equation equation 63 is quadratic with positive-definite Hessian $H _ { a } ( r )$ . The normalized Gaussian measure therefore has precision $H _ { a } ( r ) / \sigma _ { \xi } ^ { 2 }$ and covariance $\sigma _ { \xi } ^ { 2 } H _ { a } ( r ) ^ { - 1 }$ □

Proposition 3.3 is a conditional ensemble statement. No separation between the relaxation time of $\delta e$ and the evolution time of r or M is required for the algebraic results below. Interpreting the same ensemble as an adiabatically realized quasi-equilibrium along an actual training trajectory would require an additional local-equilibration assumption, which we do not use here.

The conditional Gaussian sector is the starting point for the next section. The local Laplace expansion separates the macrostate cost from the fluctuation cost. To quadratic order,

$$
\begin{array} { r } { \mathcal { G } _ { \mathrm { l o c } } ( r , M ; z ) = F _ { M } ( r ; z ) + \Phi _ { \mathrm { f l u c } } ( M ; B _ { a } ( r ) ) + \mathrm { c o n s t } , } \end{array}\tag{68}
$$

where $\Phi _ { \mathrm { H u c } }$ is obtained by integrating the excess fluctuations. The next section isolates this fluctuationinduced contribution and asks what orientational preference it supplies.

## 4 Operator Matching under Thermodynamic Stability

Section 3.4 showed that, at a conditioned macrostate r, the local fluctuation ensemble on the active sector ${ \mathcal { H } } _ { a }$ is governed by

$$
H _ { a } ( r ) = M ^ { - 1 } + B _ { a } ( r ) ,\tag{69}
$$

with

$$
\begin{array} { r } { B _ { a } ( r ) = P _ { a } B ( r ) P _ { a } \big | _ { \mathcal { H } _ { a } } . } \end{array}\tag{70}
$$

We now ask what orientational preference is contributed by this conditional fluctuation sector.

Throughout this section, r and the compressed statistical geometry $B _ { a } ( r )$ are held fixed, and we write

$$
\boldsymbol { B } \equiv B _ { a } ( \boldsymbol { r } )\tag{71}
$$

for brevity. This is a partial, conditional comparison: if an actual parameter-space motion that rotates M also changes r or $B ,$ those responses contribute additional terms to the full slow dynamics and are not included in the derivative computed here. The comparison is restricted to orthogonal rotations of M within the fixed active sector ${ \mathcal { H } } _ { a } ,$ so its eigenvalues, rank, and active subspace remain unchanged.

We focus on the thermodynamically stable sector

$$
\boxed { B \succeq 0 . }\tag{72}
$$

Equivalently, the compressed local entropy curvature is concave on $\mathcal { H } _ { a }$ . Since $M ^ { - 1 } \succ 0$ , this condition guarantees

$$
O M ^ { - 1 } O ^ { * } + B \succ 0\tag{73}
$$

for every orthogonal O acting on ${ \mathcal { H } } _ { a }$ . Thus the local Gaussian ensemble remains normalizable over the entire fixed-spectrum orbit.

Condition equation 72 is a stability restriction, not a consequence of ReLU geometry alone. In Sec. 5, we identify a regular ReLU statistical sector, defined in part by a positive structural entropy curvature, in which this condition is realized.

## 4.1 Conditional operator free-energy contribution

From Proposition 3.3, the local conditional distribution at fixed (r, M, B) is

$$
p _ { \mathrm { l o c } } ( \delta e \mid r , M , B ) = \frac { 1 } { Z _ { \mathrm { l o c } } ( M \mid B ) } \exp \left[ - \frac { 1 } { 2 \sigma _ { \xi } ^ { 2 } } \left. \delta e , ( M ^ { - 1 } + B ) \delta e \right. _ { p } \right] .\tag{74}
$$

The Gaussian partition function on the n-dimensional active sector is

$$
\boxed { Z _ { \mathrm { l o c } } ( M \mid B ) = ( 2 \pi \sigma _ { \xi } ^ { 2 } ) ^ { n / 2 } \underset { \mathcal { H } _ { a } } { \operatorname * { d e t } } ( M ^ { - 1 } + B ) ^ { - 1 / 2 } . }\tag{75}
$$

Integrating over the conditional fluctuations therefore contributes

$$
\begin{array} { r l } & { \Phi _ { \mathrm { f l u c } } ( M ; B ) \equiv - \sigma _ { \xi } ^ { 2 } \log Z _ { \mathrm { l o c } } ( M \mid B ) } \\ & { \qquad = \left\lceil \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname* { d e t } ( M ^ { - 1 } + B ) \right\rceil + \mathrm { c o n s t . } } \end{array}\tag{76}
$$

The omitted constant depends on n and $\sigma _ { \xi } ^ { 2 }$ but is constant on the fixed-rank, fixed-spectrum orbit studied below; it must be restored when comparing sectors of diferent rank or diferent active subspaces. For brevity, we write $\Phi \equiv \Phi _ { \mathrm { H u c } }$ throughout the remainder of this section.

Equation equation 68 makes clear that $\Phi _ { \mathrm { H u c } }$ is only one contribution to the total local free energy. In particular,

$$
F _ { M } ( r ; z ) = \frac { 1 } { 2 } \langle r _ { a } , M ^ { - 1 } r _ { a } \rangle _ { p } - \sigma _ { \xi } ^ { 2 } \log \Omega _ { z } ( r )\tag{77}
$$

contains its own orientational dependence through

$$
\frac { 1 } { 2 } \langle r _ { a } , M ^ { - 1 } r _ { a } \rangle _ { p } = \frac { 1 } { 2 } \mathrm { T r } \big ( M ^ { - 1 } r _ { a } r _ { a } ^ { * } \big ) .\tag{78}
$$

At fixed spectrum, this rank-one macrostate term is minimized when the current active residual direction is aligned with the largest eigenvalue of M, i.e. with the fastest relaxation direction. This error-directed preference is distinct from the fluctuation-induced matching preference studied below. Their relative magnitude depends on the conditioned state, spectrum, parameterization, and training conditions, and we make no universal ordering between them. The present analysis therefore isolates the fluctuation contribution rather than treating it as a proxy for the total orientational free energy.

Our theorems characterize the thermodynamic preference and generalized force supplied specifically by $\Phi _ { \mathrm { { f l u c } } } ;$ they do not claim to minimize the complete local free energy or to determine the full slow dynamics of M. Appendix A shows that, for residual-preserving rotations, the macrostate term is exactly constant, so the fluctuation contribution can also be isolated as a strict orientational statement on that restricted orbit.

Using

$$
M ^ { - 1 } + B = M ^ { - 1 } ( I + M B ) ,\tag{79}
$$

we may write

$$
\boxed { \Phi ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \left[ - \log \operatorname * { d e t } M + \log \operatorname * { d e t } ( I + M B ) \right] + \mathrm { c o n s t } , }\tag{80}
$$

where all determinants are on ${ \mathcal { H } } _ { a }$ . Along a fixed-spectrum orbit, det M is constant, so the orientational contribution is

$$
\Biggl | \Phi _ { \mathrm { o r i e n t } } ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } ( I + M B ) . \Biggr |\tag{81}
$$

The matching theorem below is exact throughout the stable sector, but the magnitude of the orientational bias depends on the dimensionless coupling between M and B. In particular, on a subspace where $B \succ 0$ and all eigenvalues of $M ^ { 1 / 2 } B M ^ { 1 / 2 }$ are asymptotically large,

$$
\log \operatorname * { d e t } ( I + M B ) = \log \operatorname * { d e t } M + \log \operatorname * { d e t } B + o ( 1 ) ,\tag{82}
$$

so the leading orientational dependence disappears. Thus the conditional matching contribution is most pronounced outside this deep strong-coupling limit; the result remains valid there, but its orientational force becomes parametrically weak.

## 4.2 Rotational stationarity and operator matching

Let the spectrum of M be fixed and consider an infinitesimal orthogonal rotation within $\mathcal { H } _ { a }$

$$
M ( t ) = O ( t ) M O ( t ) ^ { * } , \qquad O ( t ) = e ^ { t \Xi } , \qquad \Xi ^ { * } = - \Xi .\tag{83}
$$

Then

$$
\dot { M } = [ \Xi , M ] , \qquad \frac { d } { d t } M ^ { - 1 } = [ \Xi , M ^ { - 1 } ] .\tag{84}
$$

Theorem 4.1 (Operator-matching condition for the conditional contribution). Let $M \succ 0$ and $B = B ^ { * }$ on ${ \mathcal { H } } _ { a }$ , with $M ^ { - 1 } + B \succ 0$ . The conditional free-energy contribution

$$
\Phi ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } ( M ^ { - 1 } + B )\tag{85}
$$

is rotationally stationary on the fixed-spectrum orbit of M if and only if

$$
\boxed { [ M , B ] = 0 . }\tag{86}
$$

Proof. Let $H = M ^ { - 1 } + B$ . Along Eq. equation 83,

$$
\begin{array} { c } { { \displaystyle { \frac { d \Phi } { d t } = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \mathrm { T r } \left( H ^ { - 1 } [ \Xi , M ^ { - 1 } ] \right) } } } \\ { { = \displaystyle { \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \mathrm { T r } \left( [ M ^ { - 1 } , H ^ { - 1 } ] \Xi \right) . } } } \end{array}\tag{87}
$$

Because $M ^ { - 1 }$ and $H ^ { - 1 }$ are self-adjoint, $[ M ^ { - 1 } , H ^ { - 1 } ]$ is anti-self-adjoint. Stationarity for all anti-self-adjoint $\Xi$ is therefore equivalent to

$$
[ M ^ { - 1 } , H ^ { - 1 } ] = 0 .\tag{88}
$$

Since H is invertible, this is equivalent to $[ M ^ { - 1 } , H ] = 0$ , hence to $[ M ^ { - 1 } , B ] = 0$ , and therefore to $[ M , B ] =$ 0. □

Theorem 4.1 characterizes the stationary orientations preferred by the conditional fluctuation contribution.   
It does not assert that the complete parameter dynamics necessarily drives M to such a point.

## 4.3 Reverse spectral pairing

At a rotationally stationary point, M and B share an eigenbasis. Let

$$
m _ { 1 } \geq m _ { 2 } \geq \cdot \cdot \cdot \geq m _ { n } > 0\tag{89}
$$

and

$$
0 \leq b _ { 1 } \leq b _ { 2 } \leq \cdot \cdot \cdot \leq b _ { n } .\tag{90}
$$

At a commuting configuration specified by a permutation π,

$$
M = \mathrm { d i a g } ( m _ { 1 } , \ldots , m _ { n } ) , \qquad B = \mathrm { d i a g } ( b _ { \pi ( 1 ) } , \ldots , b _ { \pi ( n ) } ) ,\tag{91}
$$

and

$$
\Phi _ { \mathrm { o r i e n t } } = { \frac { \sigma _ { \xi } ^ { 2 } } { 2 } } \sum _ { i = 1 } ^ { n } \log ( 1 + m _ { i } b _ { \pi ( i ) } ) .\tag{92}
$$

Theorem 4.2 (Reverse spectral pairing). Within the stable sector $B \succeq 0$ , the conditional orientational free-energy contribution

$$
\Phi _ { \mathrm { o r i e n t } } ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } ( I + M B )\tag{93}
$$

is globally minimized when the spectra of M and B are oppositely ordered:

$$
\boxed { m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } . }\tag{94}
$$

With the ordering above, a minimizing configuration is

$$
M = \mathrm { d i a g } ( m _ { 1 } , \ldots , m _ { n } ) , \qquad B = \mathrm { d i a g } ( b _ { 1 } , \ldots , b _ { n } ) .\tag{95}
$$

Same-order pairing gives the global maximum. If both spectra are nondegenerate, all other commuting permutations are saddles.

Proof. The fixed-spectrum orthogonal orbit is compact, and Eq. equation 73 ensures that $\Phi _ { \mathrm { o r i e n t } }$ is continuous on the full orbit. Any global extremum is therefore stationary and, by Theorem 4.1, commuting.

Consider $m _ { i } > m _ { j }$ and $b _ { a } > b _ { b }$ . The diference between same-order and reversed pairings is

$$
\begin{array} { c } { { ( 1 + m _ { i } b _ { a } ) ( 1 + m _ { j } b _ { b } ) - ( 1 + m _ { i } b _ { b } ) ( 1 + m _ { j } b _ { a } ) } } \\ { { { } } } \\ { { = ( m _ { i } - m _ { j } ) ( b _ { a } - b _ { b } ) > 0 . } } \end{array}\tag{96}
$$

Since log is increasing, replacing an in-order pair by a reversed pair lowers $\Phi _ { \mathrm { o r i e n t } }$ . Repeated exchanges give the global reverse ordering; reversing the argument gives the global maximum.

For completeness, at a commuting configuration the tangent space of the orthogonal orbit is spanned by independent pairwise generators $E _ { i j }$ . Because both M and B are diagonal there,

$$
\dot { H } _ { i j } = \Xi _ { i j } \left( m _ { j } ^ { - 1 } - m _ { i } ^ { - 1 } \right) , \qquad i \neq j ,\tag{97}
$$

so $\dot { H }$ has only the corresponding of-diagonal pair and both $\operatorname { T r } ( H ^ { - 1 } { \ddot { H } } )$ and $\operatorname { T r } \left( H ^ { - 1 } { \dot { H } } H ^ { - 1 } { \dot { H } } \right)$ contain only $\Xi _ { i j } ^ { 2 }$ terms, with no cross-pair contributions. The second variation is therefore diagonal in this basis, with pairwise curvature

$$
\kappa _ { i j } = - \sigma _ { \xi } ^ { 2 } \frac { ( m _ { i } - m _ { j } ) ( b _ { \pi ( i ) } - b _ { \pi ( j ) } ) } { ( 1 + m _ { i } b _ { \pi ( i ) } ) ( 1 + m _ { j } b _ { \pi ( j ) } ) } .\tag{98}
$$

Any nonextremal permutation contains both an in-order and a reversed pair, and therefore has both positiveand negative-curvature tangent directions. It is thus a saddle. □

The theorem describes the orientation favored by the conditional fluctuation contribution: directions of weaker microstate curvature lower this contribution when paired with larger dynamical eigenvalues.

## 4.4 Local rotational restoring contribution

We next characterize the generalized force supplied by the conditional free energy near a reverse-paired configuration. Consider

$$
M _ { * } = \left( \begin{array} { c c } { { m _ { i } } } & { { 0 } } \\ { { 0 } } & { { m _ { j } } } \end{array} \right) , \qquad B = \left( \begin{array} { c c } { { b _ { i } } } & { { 0 } } \\ { { 0 } } & { { b _ { j } } } \end{array} \right) ,\tag{99}
$$

and rotate M<sub>∗</sub> by

$$
M ( \theta ) = R ( \theta ) M _ { * } R ( \theta ) ^ { * } .\tag{100}
$$

Writing

$$
D _ { i j } ( \theta ) \equiv \mathrm { d e t } [ I + M ( \theta ) B ] ,\tag{101}
$$

a direct calculation gives

$$
\Big | D _ { i j } ( \theta ) = D _ { i j } ( 0 ) - ( m _ { i } - m _ { j } ) ( b _ { i } - b _ { j } ) \sin ^ { 2 } \theta , \Big |\tag{102}
$$

where

$$
D _ { i j } ( 0 ) = ( 1 + m _ { i } b _ { i } ) ( 1 + m _ { j } b _ { j } ) .\tag{103}
$$

Hence

$$
\Phi _ { i j } ( \theta ) = \Phi _ { i j } ( 0 ) + \frac { 1 } { 2 } \kappa _ { i j } \theta ^ { 2 } + O ( \theta ^ { 4 } ) ,\tag{104}
$$

with

$$
\boxed { \kappa _ { i j } = - \sigma _ { \xi } ^ { 2 } \frac { ( m _ { i } - m _ { j } ) ( b _ { i } - b _ { j } ) } { ( 1 + m _ { i } b _ { i } ) ( 1 + m _ { j } b _ { j } ) } . }\tag{105}
$$

At the reverse-paired minimum, $m _ { i } > m _ { j }$ implies $b _ { i } < b _ { j }$ , so $\kappa _ { i j } > 0$

Corollary 4.3 (Local rotational restoring contribution). For every nondegenerate pair at a reverse-paired commuting configuration, the conditional orientational free energy has strictly positive quadratic curvature. Defining the generalized force contributed by this sector as

$$
f _ { i j } ^ { \mathrm { c o n d } } \equiv - \frac { \partial \Phi } { \partial \theta _ { i j } } ,\tag{106}
$$

one obtains

$$
\boxed { f _ { i j } ^ { \mathrm { c o n d } } = - \kappa _ { i j } \theta _ { i j } + O ( \theta _ { i j } ^ { 3 } ) , \qquad \kappa _ { i j } > 0 . }\tag{107}
$$

Thus the conditional fluctuation sector contributes a local restoring thermodynamic force against rotational mismatch.

If $m _ { i } = m _ { j } ~ { \mathrm { o r } } ~ b _ { i } = b _ { j }$ , then $\kappa _ { i j } = 0$ and the corresponding rotation is a flat direction of this contribution.

Corollary 4.3 is deliberately an operator-space statement about one term in the efective force on M. The total slow dynamics may contain additional contributions, and whether a given parameterization can realize the corresponding operator rotation depends on the map $\theta \mapsto M ( \theta )$ .

## 4.5 Commutator interpretation

Define the active-sector mismatch

$$
\boxed { \mathcal { C } ( M , B ) \equiv \frac { 1 } { 2 } \| [ M , B ] \| _ { F } ^ { 2 } . }\tag{108}
$$

Near a reverse-paired commuting configuration,

$$
\mathcal { C } ( M , B ) = \sum _ { i < j } ( m _ { i } - m _ { j } ) ^ { 2 } ( b _ { i } - b _ { j } ) ^ { 2 } \theta _ { i j } ^ { 2 } + O ( \Vert \theta \Vert ^ { 3 } ) .\tag{109}
$$

Combining this with Corollary 4.3, every nondegenerate pair satisfies locally

$$
\boxed { \langle - \nabla _ { \mathrm { { r o t } } } \Phi , \nabla _ { \mathrm { { r o t } } } \mathcal { C } \rangle < 0 . }\tag{110}
$$

Thus descent of the conditional free-energy contribution locally reduces operator noncommutativity. This identifies the direction of the thermodynamic bias supplied by the fluctuation sector without assuming that the full parameter dynamics follows this descent exactly.

## 5 ReLU Cell Geometry and Structural Smoothness Preference

Section 4 showed that the conditional fluctuation free-energy contribution is minimized, on a fixed-spectrum orbit, when large eigenvalues of M are paired with small eigenvalues of the compressed microstate-curvature operator $B _ { a }$ . We now examine the structure of this statistical geometry for ReLU networks.

## 5.1 ReLU cell geometry and the structural field

Let $c ( x )$ denote a function represented by a ReLU network. The input space is partitioned into activation cells within which the activation pattern is fixed. Restricted to any such cell, c is afine, and therefore

$$
\boxed { D ^ { 2 } c ( x ) = 0 } \qquad \mathrm { i n s i d e ~ e a c h ~ a c t i v a t i o n ~ c e l l } .\tag{111}
$$

The second-order structure is consequently localized on the boundaries between adjacent cells.

Consider a codimension-one facet F separating two neighboring activation cells. Continuity implies that tangential derivatives agree across the facet, whereas the normal component of the gradient may jump. Thus

$$
\boxed { [ \nabla c ] _ { F } = \alpha _ { F } n _ { F } , }\tag{112}
$$

where $n _ { F }$ is a unit normal and $\alpha _ { F }$ is the jump amplitude. Accordingly, the Hessian is naturally understood distributionally:

$$
\boxed { D ^ { 2 } c = \sum _ { F } \alpha _ { F } n _ { F } \otimes n _ { F } \delta _ { F } . }\tag{113}
$$

Thus the second-order content of a ReLU function is carried by its activation boundaries and associated gradient jumps.

We introduce a fixed coarse-graining operator $\mathcal { C }$ and define the structural field

$$
\boxed { h = L c , \qquad L \equiv \mathcal { C } D ^ { 2 } . }\tag{114}
$$

The field h summarizes coarse-grained departures from local afinity. We choose the retained function sector $\mathcal { H } _ { R }$ introduced in Sec. 3.3 so that these coarse-grained structural coordinates are well defined on the retained directions. The map L sends this retained function sector into a structural-field space $\mathcal { H } _ { h } ;$ ; the finite-dimensional operator entering Sec. 4 is then obtained by compression from $\mathcal { H } _ { R }$ to $\mathcal { H } _ { a }$

## 5.2 Statistical boundary conditions for ReLU microstates

Let $c _ { r }$ denote the current coarse-grained function configuration and $h _ { r } = L c _ { r }$ . We impose three statistical boundary conditions on the local microstate ensemble.

R1. Structural suficiency. Within the retained coarse-grained sector, the relevant local variation of the microstate entropy is determined by $h = L c \mathrm { : }$

$$
\boxed { S _ { \mathrm { m i c r o } } [ c ] = S _ { h } [ L c ] }\tag{115}
$$

in a neighborhood of $c _ { r }$

R2. Local entropy regularity. We assume that $S _ { h }$ is twice Fréchet diferentiable near $h _ { r }$ and define

$$
\boxed { \mathcal { K } ( \boldsymbol { r } ) \equiv - D _ { h } ^ { 2 } S _ { h } [ h _ { r } ] . }\tag{116}
$$

The first variation need not vanish; it afects the local center, whereas the second variation controls the fluctuation geometry. This regularity assumption is imposed on the entropy of the coarse-grained structural field, not on the microscopic facet configuration itself. The role of C is to suppress facet-scale singular structure before this local statistical description is applied; R2 does not claim that the raw activationboundary ensemble is diferentiable.

R3. Mild stable structural statistics. On the structural subspace relevant to the local fluctuations, we assume

$$
\boxed { 0 < k _ { - } I \preceq K ( r ) \preceq k _ { + } I , }\tag{117}
$$

for finite $0 < k _ { - } \le k _ { + } < \infty$

Condition R3 is the stability input of the ReLU specialization: it assumes local concavity of the structural microstate entropy on the retained sector. ReLU cell geometry by itself does not imply this statistical property. The role of R1–R3 is instead to identify a regular ReLU microstate class in which the stable sector used in Sec. 4 is realized.

## 5.3 Pullback of the microstate curvature

On the retained function sector, write $c _ { R } = P _ { R } c , y _ { R } = P _ { R } y$ , and $e _ { R } = c _ { R } - y _ { R }$ . Up to an additive constant, the microstate entropy introduced in Sec. 3.3 is

$$
S _ { \mathrm { m i c r o } } [ c _ { R } ] = \log \Omega _ { z } ( e _ { R } ) , \qquad e _ { R } = c _ { R } - y _ { R } .\tag{118}
$$

Since $y _ { R }$ is fixed, variations in $c _ { R }$ and $e _ { R }$ coincide. Under R1,

$$
D _ { c _ { R } } ^ { 2 } S _ { \mathrm { m i c r o } } [ c _ { r } ] ( u , v ) = D _ { h } ^ { 2 } S _ { h } [ h _ { r } ] ( L u , L v ) ,\tag{119}
$$

and therefore

$$
- D _ { c _ { R } } ^ { 2 } S _ { \mathrm { m i c r o } } [ c _ { r } ] ( u , v ) = \langle L u , K ( r ) L v \rangle _ { h } .\tag{120}
$$

Let $L ^ { * }$ be the adjoint defined by

$$
\langle L u , h \rangle _ { h } = \langle u , L ^ { * } h \rangle _ { p } .\tag{121}
$$

Then

$$
- D _ { c _ { R } } ^ { 2 } S _ { \mathrm { m i c r o } } [ c _ { r } ] ( u , v ) = \langle u , L ^ { * } { \mathcal K } ( r ) L v \rangle _ { p } .\tag{122}
$$

Theorem 5.1 (ReLU microstate curvature). Under R1–R3, the retained-sector microstate-curvature form $i s$

$$
\Big | B ( r ) = \sigma _ { \xi } ^ { 2 } L ^ { * } \mathcal { K } ( r ) L . \Big |\tag{123}
$$

For every retained perturbation u ${ \mathrm { , } } \in { \mathcal { H } } _ { R }$ for which L is defined,

$$
\begin{array} { r } { \boxed { \sigma _ { \xi } ^ { 2 } k _ { - } \| L u \| _ { h } ^ { 2 } \leq \langle u , B ( r ) u \rangle _ { p } \leq \sigma _ { \xi } ^ { 2 } k _ { + } \| L u \| _ { h } ^ { 2 } . } } \end{array}\tag{124}
$$

Hence $B ( r ) \succeq 0$ as a quadratic form on $\mathcal { H } _ { R }$ . The operator entering Sec. 4 is the compression

$$
\boxed { B _ { a } ( r ) = P _ { a } B ( r ) P _ { a } \vert _ { \mathcal { H } _ { a } } , }\tag{125}
$$

which is therefore positive semidefinite on $\mathcal { H } _ { a }$ . Under the strict coercivity in $R \mathcal { 3 } ,$ ker B = ker L on the retained function sector.

Proof. Equation equation 123 follows from Eq. equation 122 and the retained-sector translation $\begin{array} { r } { e _ { R } = c _ { R } - y _ { R } . } \end{array}$ For any $u ,$

$$
\langle u , B ( r ) u \rangle _ { p } = \sigma _ { \xi } ^ { 2 } \langle L u , K ( r ) L u \rangle _ { h } .\tag{126}
$$

Applying R3 yields $\operatorname { E q } .$ . equation 124. Compression preserves positive semidefiniteness. Finally, strict coercivity implies $\langle \boldsymbol { u } , B \boldsymbol { u } \rangle _ { p } = 0$ if and only if $L u = 0$ □

The theorem should therefore be read as a pullback result: R3 supplies the stable structural entropy metric, while the ReLU structural map L determines how that metric is represented in function space.

## 5.4 Microstate curvature and structural smoothness

The spectral matching in Sec. 4 involves the finite-dimensional compressed operator $B _ { a } ( r )$ . Let

$$
B _ { a } ( r ) \phi _ { i } = b _ { i } ( r ) \phi _ { i } , \qquad \phi _ { i } \in \mathcal { H } _ { a } , \qquad \| \phi _ { i } \| _ { p } = 1 .\tag{127}
$$

Because $P _ { a } \phi _ { i } = \phi _ { i }$

$$
b _ { i } ( r ) = \langle \phi _ { i } , B ( r ) \phi _ { i } \rangle _ { p } .\tag{128}
$$

Applying Eq. equation 124,

$$
\begin{array} { r } { \boxed { \sigma _ { \xi } ^ { 2 } k _ { - } \| L \phi _ { i } \| _ { h } ^ { 2 } \leq b _ { i } ( r ) \leq \sigma _ { \xi } ^ { 2 } k _ { + } \| L \phi _ { i } \| _ { h } ^ { 2 } . } } \end{array}\tag{129}
$$

Hence

$$
\begin{array} { r } { \boxed { b _ { i } ( r ) \asymp \| L \phi _ { i } \| _ { h } ^ { 2 } . } } \end{array}\tag{130}
$$

For ReLU functions, $L = \mathcal { C } D ^ { 2 } , \mathrm { s o } \ \| L \phi _ { i } \| _ { h }$ measures coarse-grained second-order content, including activationboundary and gradient-jump structure. We call directions with small $\| L \phi \| _ { h }$ structurally smooth; throughout this paper, “smooth” in the ReLU specialization refers to low coarse-grained structural curvature in this sense, rather than to a fixed Fourier frequency notion.

Corollary 5.2 (Smoothness interpretation). Within the active sector and under R1–R3, the spectrum of $B _ { a }$ controls coarse-grained structural curvature up to the bounded condition number

$$
\kappa _ { \bf { \mathscr { K } } } \equiv \frac { k _ { + } } { k _ { - } } .\tag{131}
$$

In particular,

$$
\kappa _ { \mathcal { K } } ^ { - 1 } \frac { \| L \phi _ { i } \| _ { h } ^ { 2 } } { \| L \phi _ { j } \| _ { h } ^ { 2 } } \leq \frac { b _ { i } } { b _ { j } } \leq \kappa \kappa \frac { \| L \phi _ { i } \| _ { h } ^ { 2 } } { \| L \phi _ { j } \| _ { h } ^ { 2 } } .\tag{132}
$$

Thus low- $B _ { a }$ spectral sectors correspond, within the finite distortion set by $\kappa _ { \kappa } ,$ , to sectors of low coarse-grained structural curvature. Exact pairwise ordering of $\| L \phi _ { i } \| _ { h }$ is not asserted when the structural metric is strongly anisotropic. This limitation afects only the translation from the $B _ { a }$ spectrum to structural smoothness; the reverse-pairing statement of Theorem $4 . 2 ,$ which is formulated directly in terms of the $B _ { a }$ eigenvalues, remains exact.

## 5.5 Data-adaptive structural smoothness preference

The smoothness spectrum is defined in the data-weighted function space

$$
\mathcal { H } = L ^ { 2 } ( \mathcal { X } , p ) ,\tag{133}
$$

with

$$
\langle f , g \rangle _ { p } = \int _ { \mathcal { X } } f ( x ) g ( x ) p ( x ) d x .\tag{134}
$$

Consequently, the adjoint $L ^ { * }$ , orthogonality of modes, and the active-sector compression all depend on the geometry induced by $p ( x )$ . Making this dependence explicit,

$$
\begin{array} { r } { B _ { p , a } ( r ) = P _ { a } \left[ \sigma _ { \xi } ^ { 2 } L _ { p } ^ { * } \mathcal { K } _ { p } ( r ) L \right] P _ { a } \big | _ { \mathcal { H } _ { a } } . } \end{array}\tag{135}
$$

Thus the low-curvature sectors identified through $B _ { p , a }$ are intrinsic to the data-weighted functional geometry. Combining this interpretation with Theorem 4.2, the conditional fluctuation free-energy contribution pairs large dynamical eigenvalues with the low- $B _ { a }$ sector, which corresponds up to the bounded distortion $\kappa _ { \mathcal { K } }$ to low-curvature data-adaptive function-space directions:

$$
| m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } \longleftrightarrow \mathrm { l o w ~ s t r u c t u r a l - c u r v a t u r e ~ s e c t o r } .\tag{136}
$$

Corollary 5.3 (Data-adaptive structural smoothness preference). Under R1–R3 and within the stable active sector, the conditional orientational free-energy contribution favors pairing larger eigenvalues of M with the $l o w \mathrm { - } B _ { a }$ data-adaptive sector. Through Eq. equation 129, this is a bias toward directions of lower coarsegrained structural curvature up to the finite anisotropy factor $\kappa _ { \mathcal { K } }$ . Since $m _ { i }$ sets the gradient-flow relaxation rate along the corresponding eigendirection, the fluctuation contribution therefore favors faster relaxation in this low-curvature sector.

The result is a statement about the geometry preferred by the conditional fluctuation contribution. Whether the full training dynamics realizes this preference depends on the remaining slow operator dynamics and on how parameter motion can realize changes in M.

## 6 Discussion and Future Work

The function-space thermodynamic picture developed here is broadly consistent with several empirical regularities reported in neural networks. Classical observations of spectral bias indicate that smoother or lowerfrequency components are often learned earlier Rahaman et al. (2019). More recently, difusion denoisers have been found to develop geometry-adaptive harmonic representations: their learned input–output Jacobians organize into data-dependent eigendirections whose ordering is closely related to smoothness Kadkhodaie et al. (2024). The operator measured in that work is not the learning operator $M = J _ { \theta } J _ { \theta } ^ { * }$ considered here, so this is not a direct test of our theory. Nevertheless, the observed organization is qualitatively consistent with the geometry favored by our conditional fluctuation contribution,

$$
m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } \longleftrightarrow \mathrm { l o w ~ s t r u c t u r a l - c u r v a t u r e ~ d a t a - a d a p t i v e ~ s e c t o r } .\tag{137}
$$

The same macroscopic language can also accommodate qualitatively diferent learning regimes. If M remains efectively fixed, the description reduces to a kernel-like regime. If additional slow dynamics allow M to respond to the conditional thermodynamic force derived here, the same framework supplies a bias toward regular operator matching. Conversely, apparently sharp macroscopic behavior need not have a unique origin: it may reflect competition between distinct macroscopic states, or it may arise from a broad hierarchy of relaxation times even when the underlying dynamics remain continuous. The latter possibility is conceptually related to quantized models of neural scaling, in which smooth aggregate scaling can coexist with the sudden appearance of individual capabilities Michaud et al. (2023).

Grokking provides a suggestive example of the former possibility. Previous work has connected delayed generalization to structured representations Liu et al. (2022), escape from an early kernel-like regime in modular addition Mohamadi et al. (2024), and first-order phase transitions between representation phases in two-layer teacher–student models Rubin et al. (2024). Our framework highlights an endogenous route by which the relative statistical preference of macroscopic states can change during training. Because the density of states $\Omega ( e )$ and its local curvature $B ( r )$ depend on the current error state, the statistical geometry sampled by the theory changes as $r = r ( t )$ evolves. If two macroscopic branches coexist, their total conditional free energies, denoted schematically by $\mathcal { G } _ { A } ( \boldsymbol { r } )$ and $\mathcal { G } _ { G } ( r )$ , may therefore cross. These symbols refer to full branch free energies, including macrostate, spectral, rank-dependent, and fluctuation contributions; such a term-by-term global branch theory is not constructed in the present work. Schematically,

$$
\mathcal { G } _ { A } ( r _ { * } ) = \mathcal { G } _ { G } ( r _ { * } ) , \qquad \mathcal { G } _ { A } ( r ) - \mathcal { G } _ { G } ( r ) \mathrm { ~ c h a n g e s ~ s i g n ~ a c r o s s ~ } r _ { * } .\tag{138}
$$

In this picture, the training state itself can act as an endogenous control variable. Establishing the relevant competing branches, barriers, and transition dynamics lies beyond the local conditional theory developed here and is left for future work.

Several limitations define natural extensions. First, the present construction assumes an isotropic stochastic source in function space. More generally, the noise may possess its own covariance geometry $Q ,$ , leading schematically to

$$
D \propto M Q M .\tag{139}
$$

The resulting conditional statistical mechanics need not remain reversible, and the operator preference may difer from the one derived here; nonequilibrium extensions are therefore an important direction. Second, $\Phi _ { \mathrm { f l u c } } ( M ; B )$ is the free-energy contribution of the local conditional fluctuation sector, not a derivation of the complete slow dynamics of M. The macrostate term $\frac { 1 } { 2 } \langle r _ { a } , M ^ { - 1 } r _ { a } \rangle _ { p }$ already supplies a distinct errordirected orientational preference, and further slow terms may also be present. Interpreting the fluctuation gradient as an actual component of training dynamics requires that these other contributions do not cancel or overwhelm it. Third, our conditional ensemble does not require a time-scale separation, but interpreting it as a quasi-static distribution realized along a training trajectory would require additional local-equilibration assumptions. Finally, all operator matching results are formulated on a fixed finite-dimensional active sector, and the ability of parameter dynamics to realize the corresponding operator rotations remains architecture dependent.

## 7 Conclusion

We developed a statistical-mechanical description of neural-network learning directly in function space. Parameter configurations provide the microscopic realizations, while functions and the operators governing their evolution provide macroscopic variables. This separation makes it possible to distinguish the dynamica geometry of learning from statistical constraints induced by the underlying parameterization.

For mean-squared loss, the exact error dynamics determines a bare conditional dynamical weight. Combining this weight with parameter-space microstate multiplicity produces a conditional function-space ensemble whose local statistical curvature is described by B. On the finite-dimensional active sector, integrating over local error fluctuations yields a conditional free-energy contribution

$$
\Phi _ { \mathrm { f l u c } } ( M ; B ) = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } ( M ^ { - 1 } + B ) + \mathrm { c o n s t . }\tag{140}
$$

At fixed spectrum, this contribution is stationary when $[ M , B ] = 0$ , is minimized by the reverse pairing

$$
m _ { \mathrm { l a r g e } } \longleftrightarrow b _ { \mathrm { s m a l l } } ,\tag{141}
$$

and supplies a local restoring force contribution against rotational mismatch.

For ReLU-type function spaces, stable structural statistics satisfying R1–R3 give

$$
B = \sigma _ { \xi } ^ { 2 } L ^ { * } \mathcal { K } L ,\tag{142}
$$

whose active-sector spectrum measures coarse-grained structural curvature up to the bounded anisotropy of the structural metric. The conditional thermodynamic contribution therefore favors pairing faster relaxation with the low-curvature, data-adaptive sector.

These results suggest that function space provides a natural macroscopic level for the statistical mechanics of learning. In this organization of the theory, a concrete neural network is a microscopic realization rather than the starting point: the function-space organizing principle is formulated first, while architecture and parameterization determine which operators and microstate geometries realize it. The present theory isolates one thermodynamic contribution within this framework, providing a basis for studying more general slow dynamics and nonequilibrium extensions.

## References

Yasaman Bahri, Jonathan Kadmon, Jefrey Pennington, Samuel S. Schoenholz, Jascha Sohl-Dickstein, and Surya Ganguli. Statistical mechanics of deep learning. Annual Review of Condensed Matter Physics, 11: 501–528, 2020. doi: 10.1146/annurev-conmatphys-031119-050745.

Randall Balestriero and Richard Baraniuk. A spline theory of deep learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 374–383. PMLR, 2018. URL https://proceedings.mlr.press/v80/balestriero18b.html.

Pratik Chaudhari and Stefano Soatto. Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=HyWrIgW0W.

Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in diferentiable programming. In Advances in Neural Information Processing Systems, volume 32, pp. 2933–2943, 2019. URL https:// proceedings.neurips.cc/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html.

Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=jh-rTtvkGeM.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory F. Diamos, Heewoo Jun, Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically, 2017. URL https://arxiv.org/abs/1712.00409.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, pp. 8571–8580, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html.

Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the fokker–planck equation. SIAM Journal on Mathematical Analysis, 29(1):1–17, 1998. doi: 10.1137/S0036141096303359.

Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, and Stéphane Mallat. Generalization in difusion models arises from geometry-adaptive harmonic representations. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=ANvmVS2Yr0.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361.

Clarissa Lauditi, Blake Bordelon, and Cengiz Pehlevan. Adaptive kernel predictors from feature-learning infinite limits of neural networks. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 32617–32648. PMLR, 2025. URL https://proceedings.mlr.press/v267/lauditi25a.html.

Clarissa Lauditi, Cengiz Pehlevan, and Blake Bordelon. Spectral dynamics in deep networks: Feature learning, outlier escape, and learning rate transfer, 2026. URL https://arxiv.org/abs/2605.07870.

Ziming Liu, Ouail Kitouni, Niklas S. Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An efective theory of representation learning. In Advances in Neural Information Processing Systems, volume 35, 2022. doi: 10.52202/068431-2511. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html.

Stephan Mandt, Matthew D. Hofman, and David M. Blei. A variational analysis of stochastic gradient algorithms. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 354–363. PMLR, 2016. URL https://proceedings.mlr. press/v48/mandt16.html.

Stephan Mandt, Matthew D. Hofman, and David M. Blei. Stochastic gradient descent as approximate bayesian inference. Journal of Machine Learning Research, 18(134):1–35, 2017. URL https://jmlr.org/ papers/v18/17-214.html.

Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. In Advances in Neural Information Processing Systems, volume 36, pp. 28699–28722, 2023. doi: 10.52202/075280-1248. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html.

Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, and Danica J. Sutherland. Why do you grok? A theoretical analysis on grokking modular addition. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 35934–35967. PMLR, 2024. URL https://proceedings.mlr.press/v235/mohamadi24a.html.

Grigorios A. Pavliotis. Stochastic Processes and Applications: Difusion Processes, the Fokker–Planck and Langevin Equations, volume 60 of Texts in Applied Mathematics. Springer, New York, 2014. doi: 10.1007/ 978-1-4939-1323-7.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5301–5310. PMLR, 2019. URL https://proceedings.mlr.press/v97/rahaman19a.html.

Hannes Risken. The Fokker–Planck Equation: Methods of Solution and Applications, volume 18 of Springer Series in Synergetics. Springer-Verlag, Berlin, 2 edition, 1989. ISBN 9780387504988.

Noa Rubin, Inbar Seroussi, and Zohar Ringel. Grokking as a first order phase transition in two layer networks. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 682f87a8c306098ec8be29019bd76aa4-Abstract-Conference.html.

Michael Unser. A representer theorem for deep neural networks. Journal of Machine Learning Research, 20 (110):1–30, 2019. URL https://jmlr.org/papers/v20/18-418.html.

Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Danie Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In Proceedings of the Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pp. 3635–3673. PMLR, 2020. URL https://proceedings.mlr.press/v125/woodworth20a.html.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html.

## A Residual-Preserving Rotations of the Local Conditional Free Energy

The main text isolates the fluctuation-induced contribution $\Phi _ { \mathrm { H u c } }$ to the local conditional free energy. Here we record a restricted setting in which this contribution is also the complete orientational variation of the quadratic local free energy within the conditional construction of Sec. 3.4.

Let $r _ { a } = P _ { a } r \in \mathcal { H } _ { a }$ be the active component of the conditioned macrostate. For notational simplicity within this appendix, write $r \equiv r _ { a }$ and assume $r \neq 0$ . Define its stabilizer subgroup

$$
G _ { r } \equiv \{ O \in S O ( \mathcal { H } _ { a } ) : O r = r \} .\tag{143}
$$

Its infinitesimal generators satisfy

$$
\Xi ^ { * } = - \Xi , \qquad \Xi r = 0 .\tag{144}
$$

Consider the fixed-spectrum orbit

$$
M ( O ) = O M O ^ { * } , \qquad O \in G _ { r } .\tag{145}
$$

Proposition A.1 (Residual-preserving isolation of the fluctuation term). For every $O \in G _ { r }$ 2

$$
\frac { 1 } { 2 } \left. r , M ( O ) ^ { - 1 } r \right. _ { p } = \frac { 1 } { 2 } \langle r , M ^ { - 1 } r \rangle _ { p } .\tag{146}
$$

Hence, at fixed conditioned macrostate $r$ and fixed density-of-states geometry $B _ { a } ( r )$ , the orientational variation of the quadratic local free energy

$$
\mathcal { G } _ { \mathrm { l o c } } = F _ { M } ( r ; z ) + \Phi _ { \mathrm { f u c } } ( M ; B _ { a } ( r ) ) + \mathrm { c o n s t }\tag{147}
$$

along $G _ { r }$ is exactly the orientational variation of $\Phi _ { \mathrm { H u c } }$

Proof. Since $M ( O ) ^ { - 1 } = O M ^ { - 1 } O ^ { * }$ and $O ^ { * } r = r .$

$$
\langle r , M ( O ) ^ { - 1 } r \rangle _ { p } = \langle O ^ { * } r , M ^ { - 1 } O ^ { * } r \rangle _ { p } = \langle r , M ^ { - 1 } r \rangle _ { p } .\tag{148}
$$

At fixed $r ,$ the density-of-states term $- \sigma _ { \xi } ^ { 2 } \log \Omega _ { z } ( r )$ is also independent of the rotation. Therefore only $\Phi _ { \mathrm { H u c } }$ varies along the residual-preserving orbit. □

For the remainder of the appendix, write

$$
B \equiv B _ { a } ( r ) , \qquad H = M ^ { - 1 } + B .\tag{149}
$$

Let

$$
P _ { \perp } \equiv I - \frac { r r ^ { \ast } } { \Vert r \Vert _ { p } ^ { 2 } }\tag{150}
$$

denote the orthogonal projector onto $r ^ { \perp } \cap \mathcal { H } _ { a }$ . For a general residual-preserving infinitesimal rotation, the first variation from Eq. equation $^ { 8 7 }$ becomes

$$
\delta \Phi _ { \mathrm { f l u c } } = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \mathrm { T r } \left( [ M ^ { - 1 } , H ^ { - 1 } ] \Xi \right) , \qquad \Xi { r } = 0 ,\tag{151}
$$

Therefore stationarity with respect to all such rotations is equivalent to

$$
\boxed { P _ { \perp } [ M ^ { - 1 } , H ^ { - 1 } ] P _ { \perp } = 0 . }\tag{152}
$$

This is the exact restricted stationarity condition without any additional invariant-subspace assumption.

A particularly transparent case is obtained when the residual direction is a common invariant mode of both operators,

$$
M r = m _ { r } r , \qquad B r = b _ { r } r .\tag{153}
$$

Then span{r} and $r ^ { \perp }$ are invariant under both M and B, and we may write

$$
M = m _ { r } P _ { r } \oplus M _ { \perp } , \qquad B = b _ { r } P _ { r } \oplus B _ { \perp } ,\tag{154}
$$

where $P _ { r } = r r ^ { * } / \| r \| _ { p } ^ { 2 }$ . Residual-preserving rotations act only on the $( n - 1 )$ )-dimensional orthogonal block. The fluctuation orientational contribution factorizes as

$$
\Phi _ { \mathrm { f l u c } } = \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log ( 1 + m _ { r } b _ { r } ) + \frac { \sigma _ { \xi } ^ { 2 } } { 2 } \log \operatorname * { d e t } _ { r ^ { \perp } } ( I _ { \perp } + M _ { \perp } B _ { \perp } ) + \mathrm { c o n s t . }\tag{155}
$$

The first term is fixed on $G _ { r }$ . Applying Theorems 4.1 and 4.2 to the orthogonal block gives

$$
[ M _ { \bot } , B _ { \bot } ] = 0\tag{156}
$$

at restricted stationary orientations, and the restricted minimum pairs the eigenvalues in reverse order,

$$
( m _ { \perp } ) _ { \mathrm { l a r g e } } \longleftrightarrow ( b _ { \perp } ) _ { \mathrm { s m a l l } } .\tag{157}
$$

Thus, whenever the current residual direction forms a common invariant mode, the reverse-pairing result is an exact statement about the total quadratic local free energy on the residual-preserving orbit. The main text does not require this additional condition; it characterizes the fluctuation-induced contribution on the full fixed-spectrum orbit.

If $r = 0$ , the stabilizer is the full orthogonal group and the distinction disappears: the macrostate term vanishes, so the fluctuation contribution is the complete quadratic orientational dependence.