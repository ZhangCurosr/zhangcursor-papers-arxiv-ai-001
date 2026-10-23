# Hyperreduction accelerates finite element model updating for the discovery of neural materials

Hagen Holthusen<sup>a,∗</sup>, Matthias Knipper<sup>a</sup>, Kevin Linka<sup>a</sup>

<sup>a</sup>Computational Mechanics in Medicine, Institute of Applied Medical Engineering, RWTH Aachen University, Pauwelsstraße 20, 52074 Aachen, Germany

## Abstract

Finite element model updating is among the most general and robust approaches to discover constitutive models from structural experiments: since the displacement field is computed rather than measured, it accommodates arbitrary geometries, boundary conditions, and sparse or noisy data. However, every evaluation of the loss requires a nonlinear<sub>2</sub> finite element solve per load step, and the training of neural materials with a large number of weights and biases requires many such evaluations. Intrusive model order reduction is well suited to lower this cost, since it reduces the finite<sup>c</sup> <sub>element model rather than replacing it and thus retains its boundary conditions and variational structure. This raises</sub>O the question: How can we construct a reduced model that remains accurate while the weights of the network change<sub>2</sub> at every step of the training? Here, we combine proper orthogonal decomposition and energy-conserving sampling and2 weighting with adjoint sensitivities of the reduced model, and adapt both the trial space and the sampled elements along the optimization path.

The constitutive law is represented by a polyconvex neural network. To this end, we introduce a logarithmic normalizationC term which, together with a zero-anchored input convex architecture and a growth term, renders the energy polyconvex,<sub>s</sub><sup>.</sup> coercive, and non-negative by construction, independent of the trained weights.c

We demonstrate the approach on two experimental datasets and one synthetic three-dimensional example. For a twophase specimen, the training time decreases from nine hours to 41 minutes compared with a full-order identification, without loss of accuracy. For the three-dimensional example, six modes replace 48 672 degrees of freedom, 116 of 15 625 elements sufice to evaluate the internal forces, and the entire training takes 13 minutes. In all cases, the identified<sub>2</sub> material reproduces the measured reaction force and predicts the displacement field from the force alone. Our results3 show that hyperreduction makes the discovery of neural materials from structural data afordable, while retaining the<sup>1</sup> advantages of finite element model updating: it accommodates arbitrary geometries and boundary conditions as well as sparse, noisy, or force-only data.<sup>0</sup>

Keywords: unsupervised, model discovery, finite element model updating, hyperreduction, model order reduction,6 neural network<sup>2</sup>

## 1. Introduction<sub>a</sub><sup>r</sup>

A finite element simulation is only as predictive as its constitutive model. Neural materials, that is, constitutive models in which neural networks parameterize the energy and the dissipation potential [1], have changed how we obtain these models: constitutive artificial neural networks [2, 3], physics-augmented and input-convex architectures [4, 5], and their extensions to inelasticity [6, 7, 8, 9, 10] and multiphysics [11, 12, 13, 14] discover energies and dissipation potentials that satisfy thermodynamic principles by construction. The data to train them, however, are rarely pairs of stress and strain. What we measure in a structural experiment are displacement fields and reaction forces. This raises the question: How do we discover a neural material from measurements of a structure, and can we do so at a cost that is practical?

Discovery from structural data. Full-field measurements probe a rich set of deformation states within a single specimen, but stresses remain inaccessible, such that the material law has to be inferred from indirect observables [1, 15]. The available methods difer in whether they solve the forward problem [15, 16, 17, 18]. Methods that do not, such as the equilibrium gap [15, 19] and the virtual fields method [20, 21], evaluate balance of linear momentum directly on the measured displacement field. This idea underlies the unsupervised EUCLID framework [22], which has been extended to neural network models [23], constitutive artificial neural networks [24], experimental [25] and three-dimensional data [26], heterogeneous specimens [27, 28], sparse and noisy data [29], and optimized specimen design [30]. The equilibrium gap method has recently been used to calibrate physics-augmented neural networks for viscoelasticity [17] and finitestrain hyperelasticity [18], and the modified constitutive relation error ofers a related route for thermodynamically consistent networks [31]. Material Fingerprinting [32, 33] and neural operators [34] go one step further and replace the optimization by a database lookup or by a single forward pass of a network trained on many specimens. These methods are fast, yet they rely on a dense displacement field, from which strains are computed, and on assumptions such as plane stress. It was shown that an under-resolved surface measurement and a non-negligible specimen thickness can afect the calibration [18], and that the coverage of deformation states governs how well the identified model generalizes. Simulation-based approaches instead compare measured and simulated responses in the spirit of finite element model updating [35]. Since the displacement field is the solution of a boundary value problem rather than an input, these approaches accommodate sparse and noisy data and, in principle, any specimen geometry, load path, and boundary condition that the finite element model resolves. They have been used to calibrate spline-based models [36, 37], physicsaugmented neural networks [38, 39, 40], constitutive artificial neural networks [40], neural elastoplasticity [41, 42], and cohesive zone laws [43]. The same route extends to coupled problems, in which displacements, temperature, and reaction forces enter a single loss and the adjoint of the coupled residual provides the gradient [44]. Diferentiable finite element solvers make this route practical, with sensitivities obtained by automatic diferentiation or adjoints [45, 40, 42, 46], and beyond finite elements for image-based observations with particle methods [47]. In the terminology of [16], these are reduced approaches, in which the state is eliminated by solving the boundary value problem in every iteration.

Cost of simulation-based identification. The robustness of this route comes at a computational price. Each evaluation of the loss requires a nonlinear finite element solution for every load step, and each gradient additionally requires an adjoint solve with the tangent stifness. For neural materials, whose parameters are weights and biases, this procedure is repeated over many optimization steps. Since the generalization of an identified model improves with the combination of specimens and load cases [18], the number of such solves per loss evaluation grows with the richness of the data. GPU acceleration and automatic diferentiation reduce the cost of a single solve [45, 40], but not the number of degrees of freedom that enter it. This motivates us to ask: Can we reduce the cost of the forward model itself, while retaining the finite element model, its boundary conditions, and the physical structure of the constitutive network?

Model order reduction and hyperreduction. Projection-based model order reduction addresses this question in two steps [48]. First, a reduced basis, typically obtained by proper orthogonal decomposition (POD) of solution snapshots, restricts the displacement field to a low-dimensional subspace. Second, because the nonlinear internal forces must still be assembled over all elements, hyperreduction approximates their evaluation on a small subset of the mesh. Hyperreduction follows two lines. Interpolation-based methods, such as the discrete empirical interpolation method [49] and the gappy-POD-based GNAT [50], approximate the nonlinear term by a few of its entries. Cubature-based methods select and weight a subset of elements. They start from the pioneering works of hyperreduction and cubature [51, 52], and lead to energy-conserving sampling and weighting (ECSW) [53] and the empirical cubature method (ECM) [54]. A recen variant does not select a subset of the integration points but places new ones in strain space, by clustering and subsequent empirical correction (E3C) [55]. In solid mechanics, POD with empirical interpolation was applied to hyperelasticity [56], and recent studies extend reduced models to finite-strain homogenization [57, 58, 59], nearly incompressible materials [60], viscoelasticity [61], and plasticity and damage [62, 63, 64]. Comparisons of hyperreduction schemes on nonlinea elasticity [65, 66] and component-wise reduced models coupled by mortar tying [67] address accuracy and reusability. In parallel, nonlinear manifolds replace the linear subspace [68, 69, 70], and non-intrusive surrogates learn the reduced model from data, with autoencoders [71] or physics-augmented neural networks [72]. The latter report a rapid divergence outside the training range, which shows that learned reduced models are reliable only within the range they were trained on.

Reduced models in inverse problems. Reduced models have a long record in inverse problems and PDE-constrained optimization [73]. Hyperreduced models have been embedded in gradient-based design optimization [74], and the basis can be progressively constructed along the optimization path to keep it valid where the optimizer goes [75]. In solid mechanics, POD- or PGD-based models identify the parameters of hydrogels [76], cardiac tissue [77], elastoplastic materials [78], and history-dependent laws [79]. While these works identify a few parameters of an explicit constitutive law, it was noted that the reduced model is valid only within the range of parameters it was trained on [76]. A diferent route replaces the boundary value problem by a learned surrogate, for example by parametric physics-informed neural networks for calibration from full-field data [80]. With this at hand, a neural constitutive model poses a harder question. Its parameters are a large number of weights and biases that change at every optimization step, so that a basis and a set of elements built at a finite number of parameter vectors must remain reliable along the entire optimization path. This raises the question: Can we combine POD and ECSW with the adjoint-based discovery of a neural constitutive model, and does the reduced model remain accurate while the parameters evolve?

Research gap. Reduced-order models accelerate the identification of parametric constitutive laws [76, 77, 78, 79], and hyperreduction is established for forward simulations of hyperelastic and inelastic solids [56, 63, 64]. In contrast, the discovery of neural constitutive models from structural data relies on full-order solvers [40, 39, 38, 41, 42, 46], on local equilibrium evaluations of the measured field [23, 24, 17, 18], or on surrogates that replace the boundary value problem [80, 34]. To the best of our knowledge, the combination of projection-based reduction with hyperreduction and gradient-based constitutive discovery has not been studied.

Aim of the study. We aim to make inverse material discovery from structural data afordable by reducing the finite element model rather than replacing it. For this purpose, we combine three ingredients. First, we construct the reduced model by projecting the equilibrium onto a POD subspace, which for hyperelastic materials corresponds to minimizing the potential on this subspace. Second, we apply ECSW, which approximates the internal energy by a weighted sum over a small subset of elements. Third, we derive adjoint sensitivities of the reduced model with respect to the network parameters. The reduced basis is built from snapshots at diferent load steps and sampled parameter vectors, and the loss compares measured and predicted reaction forces and, if desired, displacements.

Outline. The remainder of this manuscript is organized as follows. In Section 2, we formulate the inverse material discovery problem with its adjoint, the reduction techniques employed, and the neural elasticity model, which we aim to discover. In Section 3, we demonstrate the approach on two experimental and one synthetic examples, while Section 4 discusses our findings. Finally, we conclude our work in Section 5.

## 2. Methods

In this section, we develop a framework to identify the constitutive behavior of a hyperelastic material from structural measurements, i.e., from reaction forces and, optionally, displacement fields. We first formulate the discovery as a bilevel optimization problem and derive its gradient by the adjoint method in Section 2.1. Since every evaluation of the loss requires the solution of a nonlinear finite element problem, we then employ a subspace projection and a hyperreduction in Section 2.2, which lower the computational cost. With this at hand, we specify the constitutive model in Section 2.3, which is represented by a physics-augmented-like neural network, i.e. polyconvex neural material, and whose weights and biases are the unknowns of the inverse problem. We conclude with the workflow in Section 2.4, which adapts the reduction to the evolving weights during training and decides when the reduced model can be trusted.

## 2.1. Inverse material discovery

On the structural level, material parameters cannot be identified from stress–strain pairs directly. Instead, we observe the response of a specimen or component, typically in terms of reaction forces and, if available, displacement fields. For hyperelastic materials, this inverse problem can be elegantly written as a bilevel optimization problem, i.e., as two nested minimization problems,

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } \big ( u ^ { * } ( \theta ) , \theta \big ) \qquad \mathrm { w i t h } \qquad u ^ { * } ( \theta ) = \arg \operatorname* { m i n } _ { u \in \mathcal { U } } \Pi ( u , \theta ) ,\tag{1}
$$

where θ collects the material parameters, u denotes the displacement field, U the set of kinematically admissible displacements, Π the total potential energy of the structure, and L a loss function that quantifies the discrepancy between the predicted and the observed structural response. The inner problem determines the equilibrium state for a given set of material parameters, while the outer problem adjusts the parameters such that the predicted response matches the observations. Note that the loss depends on the parameters in two ways: implicitly, through the equilibrium displacement $\boldsymbol { u } ^ { * } ( \boldsymbol { \theta } )$ , and explicitly, for instance through the reaction forces. In the context of neural networks, θ comprises the weights and biases, and solving the outer problem corresponds to training.

In the following, we first state the inner problem and its discretization. We then define the loss function and conclude with the gradient of the loss, which is required to solve the outer problem by means of gradient-based optimization.

Inner problem. For hyperelastic materials, the constitutive behavior is fully characterized by the strain energy density $\psi ( F ; \pmb \theta )$ , where the semicolon indicates that the material parameters act solely as parameters of the potential and $\pmb { F } = \pmb { I } + \operatorname { G r a d } ( \pmb { u } )$ denotes the deformation gradient. The total potential energy of the structure reads

$$
\Pi ( \pmb { u } , \pmb { \theta } ) = \underbrace { \int _ { \mathcal { B } } \psi \big ( \pmb { F } ( \pmb { u } ) ; \pmb { \theta } \big ) \mathrm { d } V } _ { = : \Pi _ { \mathrm { i n t } } ( \pmb { u } , \pmb { \theta } ) } - \Pi _ { \mathrm { e x t } } ( \pmb { u } ) ,\tag{2}
$$

where $\Pi _ { \mathrm { i n t } }$ denotes the internal and $\Pi _ { \mathrm { e x t } }$ the external potential. We assume dead loads, such that the external potential is linear in the displacement and independent of the material parameters, e.g., $\begin{array} { r } { \Pi _ { \mathrm { e x t } } = \int _ { \mathcal { B } } \pmb { f } \cdot \pmb { u } \mathrm { d } V + \int _ { \partial _ { t } \mathcal { B } } \pmb { \bar { t } } } \end{array}$ · u dA with body force f and prescribed traction t<sup>¯</sup>. The boundary conditions read

$$
{ \pmb u } = \bar { \pmb u } \mathrm { o n } \partial _ { \pmb u } \mathcal { B } , \qquad { \pmb P } \cdot { \pmb n } = \bar { \pmb t } \mathrm { o n } \partial _ { \pmb t } \mathcal { B } ,\tag{3}
$$

where ${ \cal P } = \partial \psi / \partial F$ is the Piola stress and n the outward unit normal in the reference configuration. The essential boundary conditions define the admissible set $\mathcal { U } : = \{ \pmb { u } | \pmb { u } = \bar { \pmb { u } } \mathrm { o n } \partial _ { \pmb { u } } B \}$ , while the natural boundary conditions follow from the stationarity of Π.<sup>1</sup> $\Pi .$

In the present work, we prescribe only Dirichlet boundary conditions and neglect body forces. Consequently, the external potential vanishes, and the structure is loaded exclusively through the prescribed displacements u¯. The forces acting on the structure then emerge as reaction forces on $\partial _ { u } B$ , i.e., as the quantities work-conjugate to the prescribed displacements. These reaction forces are precisely what we measure in the experiment.

Spatial and temporal discretization. In time, we discretize the loading path into $N _ { \mathrm { t } }$ load steps $t _ { n }$ with prescribed displacements $\bar { \mathbf { u } } _ { n }$ . Since the potential depends neither on rates nor on internal variables, the equilibrium state at $t _ { n }$ is fully determined by $\bar { \pmb u } _ { n }$ . Hence, for quasi-static hyperelasticity, the inner problem decouples into $N _ { \mathrm { t } }$ independent problems, and time merely parameterizes the loading path.

In space, we adopt an isoparametric finite element discretization. Within each element e, the displacement field and the reference geometry are approximated as

$$
\begin{array} { r } { \boldsymbol { \mathsf { \pmb { u } } } ^ { h } = \mathbf { N } \mathbf { d } ^ { e } , \qquad \mathbf { X } ^ { h } = \mathbf { N } \mathbf { X } ^ { e } , } \end{array}\tag{4}
$$

where N denotes the shape function matrix, and $\mathbf { d } ^ { e }$ and $\mathbf { X } ^ { e }$ refer to the nodal displacements and reference coordinates of element $e ,$ which are extracted from their global counterparts d and X. Inserting this approximation into Eq. (2) yields the discrete potential $\Pi ^ { h } ( \mathbf { d } , \pmb \theta )$ , whose stationarity condition defines the residual

$$
\mathbf { r } ( \mathbf { d } , \theta ) : = \frac { \partial \Pi ^ { h } } { \partial \mathbf { d } } = \underset { e = 1 } { \overset { n _ { \mathrm { e l } } } { \underset { \mathrm { \tiny { e l } } } { \mathrm { \tiny { A } } } } } \left[ \mathbf { r } _ { \mathrm { i n t } } ^ { e } - \mathbf { r } _ { \mathrm { e x t } } ^ { e } \right] , \qquad \mathbf { r } _ { \mathrm { i n t } } ^ { e } : = \int _ { \mathcal { B } ^ { e } } \frac { \partial \psi \big ( \pmb { F } ( \pmb { u } ^ { h } ) ; \pmb { \theta } \big ) } { \partial \mathbf { d } ^ { e } } \mathrm { d } V ,\tag{5}
$$

where A denotes the assembly operator over all $n _ { \mathrm { e l } }$ elements. By the chain rule, the integrand equals $P : \partial F / \partial { \bf d } ^ { e }$ , such that $\mathbf { r } _ { \mathrm { i n t } } ^ { e }$ represents the discrete virtual work of the stresses. In practice, it is evaluated directly from ψ by algorithmic diferentiation.

To account for the boundary conditions, we partition the global vector of nodal displacements into free and prescribed components, $\mathbf { l } = [ \mathbf { d } _ { \mathrm { f } } , \mathbf { d } _ { \mathrm { D } } ]$ , and the residual accordingly. Equilibrium requires only the free components to vanish,

$$
\mathbf { r } _ { \mathrm { f } } ( \mathbf { d } _ { \mathrm { f } } , \mathbf { d } _ { \mathrm { D } } , \pmb { \theta } ) = \mathbf { 0 } ,\tag{6}
$$

which we solve for $\mathbf { d } _ { \mathrm { f } } ^ { * }$ by a Newton–Raphson scheme with the tangent stifness ${ \bf K } _ { \mathrm { f f } } : = \partial { \bf r } _ { \mathrm { f } } / \partial { \bf d } _ { \mathrm { f } } = \partial ^ { 2 } \Pi ^ { h } / \partial { \bf d } _ { \mathrm { f } } \partial { \bf d } _ { \mathrm { f } }$ . The remaining components do not vanish; instead, they are exactly the nodal reaction forces required to maintain the prescribed displacements,

$$
\mathbf { f } _ { \mathrm { D } } : = \mathbf { r } _ { \mathrm { D } } ( \mathbf { d } _ { \mathrm { f } } ^ { * } , \mathbf { d } _ { \mathrm { D } } , \pmb { \theta } ) .\tag{7}
$$

Hence, the discrete potential provides both the equilibrium equations and the observable forces, which will play a central role in the definition of the loss.

Outer problem. The outer problem defines the discrepancy between the observed and the predicted structural response. Experimentally, we typically record the total reaction force on a set of boundary nodes ${ \mathcal { F } } , \mathrm { e . g . }$ , the loaded surface, in the loading direction e. Its model counterpart at load step $t _ { n }$ follows from Eq. (7) as

$$
\mathcal { H } ( \mathbf { d } _ { n } , \pmb { \theta } ) : = \pmb { e } \cdot \sum _ { i \in \mathcal { F } } \mathbf { f } _ { \mathrm { D } , n } ^ { i } ,\tag{8}
$$

where $\mathbf { f } _ { \mathrm { D } , n } ^ { i }$ denotes the reaction force vector at node i. If, in addition, displacements are measured at $N _ { \mathrm { o b s } }$ observation points $X _ { k } ,$ e.g., by digital image correlation, we compare them to the interpolated finite element displacements $u _ { n } ^ { h } ( X _ { k } )$ To give an idea at this point, the loss then may reads

$$
\mathcal { L } = \frac { 1 } { { { N } _ { \mathrm { t } } } } \sum _ { n = 1 } ^ { { N } _ { \mathrm { t } } } { \left[ { \left[ \mathcal { H } ( { \bf { d } } _ { n } , \theta ) - f _ { n } ^ { \mathrm { e x p } } \right] } ^ { 2 } + \frac { \alpha } { { { N } _ { \mathrm { o b s } } } } \sum _ { k = 1 } ^ { { N } _ { \mathrm { o b s } } } { \left\| { { u } _ { n } ^ { h } ( { \bf { X } } _ { k } ) - u _ { n } ^ { \mathrm { e x p } } ( { \bf { X } } _ { k } ) } \right\| } ^ { 2 } \right] } ,\tag{9}
$$

where $f _ { n } ^ { \mathrm { e x p } }$ and ${ \pmb u } _ { n } ^ { \mathrm { e x p } }$ denote the measured force and displacements at $t _ { n }$ . The weighting factor $\alpha \geq 0$ balances the diferent units and magnitudes of both contributions; for $\alpha = 0$ , we recover a purely force-based identification.

Gradient of the loss. Training is typically carried out by gradient-based optimization, which requires the derivative of the loss with respect to the parameters. Since the equilibrium state $\mathbf { d } _ { \mathrm { f } } ^ { * } ( \pmb { \theta } )$ is only implicitly defined through Eq. (6), we diferentiate the equilibrium condition with respect to θ. By the implicit function theorem, which holds as long as $\mathbf { K } _ { \mathrm { f f } }$ is regular, this yields

$$
{ \bf K } _ { \mathrm { f f } } { \frac { \mathrm { d } { \bf d } _ { \mathrm { f } } ^ { * } } { \mathrm { d } \theta } } + { \frac { \partial { \bf r } _ { \mathrm { f } } } { \partial \theta } } = { \bf 0 } \qquad \Rightarrow \qquad { \frac { \mathrm { d } { \bf d } _ { \mathrm { f } } ^ { * } } { \mathrm { d } \theta } } = - { \bf K } _ { \mathrm { f } } ^ { - 1 } { \frac { \partial { \bf r } _ { \mathrm { f } } } { \partial \theta } } ,\tag{10}
$$

with the mixed derivative $\partial \mathbf { r } _ { \mathrm { f } } / \partial \pmb { \theta } = \partial ^ { 2 } \Pi ^ { h } / \partial \mathbf { d } _ { \mathrm { f } } \partial \pmb { \theta }$ . At a strict local minimum of $\Pi ^ { h } , \mathbf { K } _ { \mathrm { f f } }$ is positive definite, and regularity is guaranteed. Applying the chain rule, the directional derivative of the loss in direction v reads

$$
{ \frac { \mathrm { d } } { \mathrm { d } \varepsilon } } { \mathcal { L } } \left( \mathbf { d } _ { \mathrm { f } } ^ { * } ( \pmb { \theta } + \varepsilon \mathbf { v } ) , \pmb { \theta } + \varepsilon \mathbf { v } \right) { \Bigg | } _ { \varepsilon = 0 } = \underbrace { \left[ { \frac { \partial { \mathcal { L } } } { \partial \pmb { \theta } } } - { \frac { \partial { \mathcal { L } } } { \partial \mathbf { d } _ { \mathrm { f } } } } \mathbf { K } _ { \mathrm { f } } ^ { - 1 } { \frac { \partial \mathbf { r } _ { \mathrm { f } } } { \partial \pmb { \theta } } } \right] } _ { = : \mathrm { d } { \mathcal { L } } / \mathrm { d } \pmb { \theta } } \cdot \mathbf { v } ,\tag{11}
$$

where, for the sake of readability, we omit the sum over the load steps. Here, the explicit derivative $\partial \mathcal { L } / \partial \pmb { \theta }$ stems from the reaction forces in Eq. (7), which depend on the material parameters directly.

The inverse of the tangent stifness is, of course, never formed explicitly. Evaluating Eq. (10) directly would require one linear solve per parameter, which becomes prohibitive for neural networks with many parameters. Instead, we solve a single adjoint problem per load step,

$$
\mathbf { K } _ { \mathrm { f } } ^ { T } \mathbf { \lambda } \lambda = \left[ { \frac { \partial { \mathcal { L } } } { \partial \mathbf { d } _ { \mathrm { f } } } } \right] ^ { T } , \qquad { \frac { \mathrm { d } { \mathcal { L } } } { \mathrm { d } \theta } } = { \frac { \partial { \mathcal { L } } } { \partial \theta } } - { \lambda } ^ { T } { \frac { \partial \mathbf { r } _ { \mathrm { f } } } { \partial \theta } } ,\tag{12}
$$

where λ denotes the adjoint variable. Inserting $\lambda = \mathbf { K } _ { \mathrm { f f } } ^ { - T } \left[ \partial \mathcal { L } / \partial \mathbf { d } _ { \mathrm { f } } \right] ^ { T }$ confirms that Eq. (12) is equivalent to Eq. (11), while its cost is independent of the number of parameters.<sup>2</sup> This procedure is known as the adjoint method in PDE-constrained optimization.

The present formulation belongs to the class of finite element model updating approaches [15]. In contrast to residualbased approaches such as the Equilibrium Gap Method [19] or unsupervised discovery [22], in which the measured displacements are inserted into the discretized weak form and its residual serves as the loss, we solve the discrete equilibrium problem in every forward evaluation. Residual-based approaches thus require full-field displacements at all nodes, complemented by global reaction forces to fix the scale of the material response. In the present formulation, by contrast, global reaction forces are in principle suficient, while displacement data, ranging from a few observation points to full fields, enter only optionally. This reduced data requirement comes at the cost of solving the inner problem, which motivates the reduction techniques discussed in the following.

## 2.2. Reduction techniques

The inverse material discovery outlined above is general in its application, yet computationally demanding: in every epoch, the inner problem has to be solved for all $N _ { \mathrm { t } }$ load steps, each requiring several Newton iterations on the full finite element mesh. Model order reduction techniques aim to lower exactly this cost. However, a reduced model is usually constructed for a specific problem, i.e., for a given geometry, set of boundary conditions, and loading path, and it remains accurate only in the vicinity of the material parameters used to construct it. In general, it does not generalize to arbitrary material parameters.

During training, precisely this limitation becomes critical, since the outer problem continuously updates the material parameters. In the early epochs, the parameters may change substantially, and any reduced model built beforehand is likely to become inaccurate. As training proceeds, however, the parameter updates decrease, and the inner problem is repeatedly solved for similar parameters at identical load steps.

In the following, we introduce two complementary strategies to reduce the cost of solving the inner problem. First, a subspace projection reduces the size of the linear system to be solved in each Newton iteration. While this lowers the cost of the linear solve, the residual and the tangent stifness still have to be assembled over all elements. Second, we therefore employ a hyperreduction method that reduces the assembly cost itself.

Subspace projection. Within one epoch, we solve $\operatorname { E q . }$ (6) for the free nodal displacements $\mathbf { d } _ { \mathrm { f } } \in \mathbb { R } ^ { N _ { \mathrm { f } } }$ at all $N _ { \mathrm { t } }$ load steps, whose solutions difer due to the diferent prescribed displacements. Across epochs, we solve the same problem at identical load steps, whose solutions difer due to the updated material parameters. We collect a finite number of these equilibrium states in the snapshot matrix

$$
\mathbf { S } = \left[ \mathbf { d } _ { \mathrm { f } , 1 } - \mathbf { d } _ { \mathrm { f } , 0 } , \mathbf { \Omega } \ldots , \mathbf { \Omega } \mathbf { d } _ { \mathrm { f } , N _ { \mathrm { s n a p } } } - \mathbf { d } _ { \mathrm { f } , 0 } \right] \in \mathbb { R } ^ { N _ { \mathrm { f } } \times N _ { \mathrm { s n a p } } } ,\tag{13}
$$

where $N _ { \mathrm { f } }$ denotes the number of free degrees of freedom, $N _ { \mathrm { s n a p } }$ the number of snapshots, and $\mathbf { d } _ { \mathrm { { f } , 0 } }$ a reference state, e.g., the undeformed configuration $\mathbf { d } _ { \mathrm { f } , 0 } = \mathbf { 0 }$ . Each snapshot s is associated with the material parameters $\theta _ { s }$ and the load step at which it was computed.

A common approach to extract a reduced basis from these snapshots is the proper orthogonal decomposition [82], which relies on the thin singular value decomposition

$$
\mathbf { S } = \mathbf { U } \Sigma \mathbf { Z } ^ { T } ,\tag{14}
$$

where $\mathbf { U } \in \mathbb { R } ^ { N _ { \mathrm { f } } \times p }$ and $\mathbf { Z } \in \mathbb { R } ^ { N _ { \mathrm { s n a p } } \times p }$ have orthonormal columns, and $\Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { p } )$ contains the singular values $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq \sigma _ { p } \geq 0$ with p = min $\{ N _ { \mathrm { f } } , N _ { \mathrm { s n a p } } \}$ . The left singular vectors are ordered by their contribution to the snapshot data, i.e., the first columns of U capture the dominant deformation modes. We therefore retain only the first $N _ { \mathrm { r } } \ll N _ { \mathrm { f } }$ columns as reduced basis,

$$
\mathbf { V } = [ \mathbf { U } _ { : , 1 } , \mathbf { \alpha } _ { \cdot } \dots , \mathbf { \alpha } _ { \mathbf { V } _ { \mathrm { r } } } ] \in \mathbb { R } ^ { N _ { \mathrm { f } } \times N _ { \mathrm { r } } } .\tag{15}
$$

With this at hand, we approximate the free nodal displacements within the afine subspace

$$
\mathbf { d } _ { \mathrm { f } } \approx \mathbf { d } _ { \mathrm { f , 0 } } + \mathbf { V } \mathbf { q } ,\tag{16}
$$

where $\mathbf { q } \in \mathbb { R } ^ { N _ { \mathrm { r } } }$ denotes the vector of reduced coordinates. Instead of solving Eq. (6) in the full space, we project the residual onto the subspace in a Galerkin sense. In each Newton iteration $i ,$ this yields the reduced linear system

$$
\underbrace { \mathbf { V } ^ { T } \mathbf { K } _ { \mathrm { f f } } \mathbf { V } } _ { = : \mathbf { K } _ { \mathrm { r } } } \Delta \mathbf { q } ^ { i } = - \underbrace { \mathbf { V } ^ { T } \mathbf { r } _ { \mathrm { f } } } _ { = : \mathbf { r } _ { \mathrm { r } } } , \qquad \mathbf { q } ^ { i + 1 } = \mathbf { q } ^ { i } + \Delta \mathbf { q } ^ { i } ,\tag{17}
$$

where $\mathbf { K } _ { \mathrm { f f } }$ and $\mathbf { r } _ { \mathrm { f } }$ are evaluated at $\mathbf { d } _ { \mathrm { f } , 0 } + \mathbf { V } \mathbf { q } ^ { i }$ and θ. For hyperelastic materials, this Galerkin projection has a clear variational interpretation: it corresponds to minimizing the discrete potential $\Pi ^ { h }$ over the afine subspace in Eq. (16). Consequently, the reduced tangent $\mathbf { K } _ { \mathrm { I } }$ inherits the symmetry.

While the size of the linear system reduces from $N _ { \mathrm { f } }$ to $N _ { \mathrm { r } }$ , both $\mathbf { r } _ { \mathrm { r } }$ and $\mathbf { K } _ { \mathrm { r } }$ are still assembled over all elements. For nonlinear problems, this assembly quickly dominates the overall cost, which motivates the hyperreduction discussed next.

Hyperreduction. We employ the Energy-Conserving Sampling and Weighting (ECSW) method [83, 53], which operates directly on the projected quantities. To this end, we note that the reduced residual and tangent are exact sums of element contributions,

$$
{ \bf r } _ { \mathrm { r } } = \sum _ { e \in { \mathcal E } } [ { \bf V } ^ { e } ] ^ { T } { \bf r } ^ { e } , \qquad { \bf K } _ { \mathrm { r } } = \sum _ { e \in { \mathcal E } } [ { \bf V } ^ { e } ] ^ { T } { \bf K } ^ { e } { \bf V } ^ { e } ,\tag{18}
$$

where E denotes the set of all $n _ { \mathrm { e l } }$ elements, $\mathbf { r } ^ { e }$ and ${ \bf K } ^ { e }$ the element residual and tangent, and ${ \bf V } ^ { e }$ the submatrix of V that contains the rows associated with the free degrees of freedom of element e. ECSW replaces these sums by weighted sums over a small subset $\tilde { \mathcal { E } } \subset \mathcal { E }$

$$
\mathbf { r } _ { \mathrm { r } } \approx \tilde { \mathbf { r } } _ { \mathrm { r } } = \sum _ { e \in \tilde { \mathcal { E } } } w ^ { e } [ \mathbf { V } ^ { e } ] ^ { T } \mathbf { r } ^ { e } , \qquad \mathbf { K } _ { \mathrm { r } } \approx \tilde { \mathbf { K } } _ { \mathrm { r } } = \sum _ { e \in \tilde { \mathcal { E } } } w ^ { e } [ \mathbf { V } ^ { e } ] ^ { T } \mathbf { K } ^ { e } \mathbf { V } ^ { e } ,\tag{19}
$$

with non-negative weights $w ^ { e } \geq 0$ . Hence, the cost of assembly scales with the number of sampled elements $| \tilde { \mathcal { E } } | \ll n _ { \mathrm { e l } }$ rather than with $n _ { \mathrm { e l } }$ . Noteworthy, the hyperreduced problem remains the minimization of an approximate potential $\begin{array} { r } { \tilde { \Pi } ^ { h } = \sum _ { e \in \tilde { \mathcal { E } } } w ^ { e } \Pi ^ { h , e } } \end{array}$ , a weighted sum of element energies.

In both cases, with or without hyperreduction, the gradient computation remains in the reduced space. Since the implicit diferentiation acts on the residual that is actually solved, the adjoint problem in Eq. (12) becomes

$$
\tilde { \mathbf { K } } _ { \mathrm { r } } ^ { T } \mathbf { \Psi } \lambda _ { \mathrm { r } } = \mathbf { V } ^ { T } \left[ \frac { \partial \mathcal { L } } { \partial \mathbf { d } _ { \mathrm { f } } } \right] ^ { T } , \qquad \mathbf { \Psi } \frac { \mathrm { d } \mathcal { L } } { \mathrm { d } \theta } = \frac { \partial \mathcal { L } } { \partial \theta } - \mathbf { \Psi } \lambda _ { \mathrm { r } } ^ { T } \frac { \partial \tilde { \mathbf { r } } _ { \mathrm { r } } } { \partial \theta } ,\tag{20}
$$

with $\lambda _ { \mathrm { r } } \in \mathbb { R } ^ { N _ { \mathrm { r } } }$ , and with $\mathbf { K } _ { \mathrm { r } }$ and $\mathbf { r } _ { \mathrm { r } }$ in place of $\tilde { \mathbf { K } } _ { \mathrm { r } }$ and $\tilde { \mathbf { r } } _ { \mathrm { r } }$ without hyperreduction. Hence, the adjoint inherits the symmetry and definiteness of the reduced tangent, and $\mathbf { K } _ { \mathrm { f f } }$ is never factorized during training. Note that Eq. (20) yields the exact gradient of the reduced model, not an approximation of the full order gradient.

Computation of the ECSW weights. For each snapshot s within the snapshot matrix in Eq. (13), we evaluate every element at the projected state

$$
\tilde { \mathbf { d } } _ { \mathrm { f } , s } = \mathbf { d } _ { \mathrm { f , 0 } } + \mathbf { V } \mathbf { V } ^ { T } \left[ \mathbf { d } _ { \mathrm { f } , s } - \mathbf { d } _ { \mathrm { f , 0 } } \right]\tag{21}
$$

and the corresponding parameters $\theta _ { s }$ , i.e., at states the reduced model can actually reach. The projected element residual and tangent are stacked into

$$
\mathbf { g } _ { s } ^ { e } = \left[ \pi \mathbf { \Psi } ^ { \left[ \mathbf { V } ^ { e } \right] ^ { T } \mathbf { r } ^ { e } \left( \tilde { \mathbf { d } } _ { \mathrm { f } , s } , \theta _ { s } \right) } { \bf \Psi } \right] \in \mathbb { R } ^ { N _ { \mathrm { r } } + N _ { \mathrm { r } } ^ { 2 } } ,\tag{22}
$$

where $\operatorname { v e c } ( \bullet )$ denotes column-wise vectorization. Arranging these vectors with one column per element and one block of rows per snapshot yields

$$
\mathbf G = \left[ \begin{array} { c c c } { \mathbf g _ { 1 } ^ { 1 } } & { \cdots } & { \mathbf g _ { 1 } ^ { n _ { \mathrm { e l } } } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \mathbf g _ { N _ { \mathrm { s n a p } } } ^ { 1 } } & { \cdots } & { \mathbf g _ { N _ { \mathrm { s n a p } } } ^ { n _ { \mathrm { e l } } } } \end{array} \right] \in \mathbb { R } ^ { N _ { \mathrm { s n a p } } [ N _ { \mathrm { r } } + N _ { \mathrm { r } } ^ { 2 } ] \times n _ { \mathrm { e l } } } , \qquad \mathbf b = \mathbf G \mathbf 1 = \sum _ { e \in \mathcal E } \mathbf G _ { : , e } ,\tag{23}
$$

where 1 denotes a vector of ones. Thus, b contains the exact reduced residual and tangent of every snapshot, i.e., the full mesh with all weights equal to one. The weights follow from the non-negative least-squares (NNLS) problem

$$
\mathbf { y } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { y } } \| \mathbf { G } \mathbf { y } - \mathbf { b } \| _ { 2 } ^ { 2 } \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathbf { y } \geq \mathbf { 0 } ,\tag{24}
$$

where $\mathbf y = [ w ^ { 1 } , \dots , w ^ { n _ { \mathrm { e l } } } ] ^ { T }$ collects one weight per element. Note that the trivial solution $\mathbf y = \mathbf 1$ satisfies Eq. (24) exactly, but yields no reduction. The aim is therefore a sparse solution, in which most weights vanish. The non-negativity constraint naturally promotes such sparsity, since many weights are driven to the boundary $w ^ { e } = 0$ of the admissible set. All elements with $w ^ { e } > 0$ form the reduced set $\tilde { \mathcal { E } } .$

## 2.3. Neural Elasticity

The inverse material discovery and the reduction techniques developed so far are independent of the explicit constitutive model. We now specify the energy $\psi ( F ; \pmb \theta )$ whose parameters θ are identified in the remainder of this work. In recent years, several physics-constrained neural network material models were introduced, which all perform well in finite strain hyperelasticity [84]. Here, we follow the Physics-Augmented Neural Network approach of [4], which is polyconvex [81] by construction. In the present work, we restrict ourselves to isotropic, hyperelastic material behavior. In the following, we first introduce the neural network, then the growth term, and conclude with the logarithmic normalization.

To this end, we decompose the energy into three contributions

$$
\psi ( \pmb { F } ; \pmb { \theta } ) : = \mathcal { N } ( \pmb { x } ( \pmb { F } ) ; \pmb { \theta } ) + \psi _ { \mathrm { g r } } ( \pmb { F } ) + \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } ( \pmb { J } ; \pmb { \theta } ) ,\tag{25}
$$

where $\mathcal { N }$ denotes the physics-constrained neural network with input vector ${ \pmb x } ( { \pmb F } ) , \ \psi _ { \mathrm { g r } }$ adds a growth contribution to ensure coercivity of the overall energy, and $\psi _ { \mathrm { n r m } } ^ { \mathrm { l n } }$ depends on the determinant J of the deformation gradient and normalizes the energy to ensure a stress-free undeformed reference state. Recall that ψ is polyconvex if there exists a convex function $\mathcal { P }$ such that $\psi ( { F } ) = \mathcal { P } ( F$ , cof F, J) [81]. Each of the three terms in $\operatorname { E q . }$ (25) is constructed to satisfy this property.

Neural network. To obtain a network that is objective, isotropic, and polyconvex, we feed it with the invariants

$$
\begin{array} { r } { I _ { 1 } : = \frac { 1 } { 3 } \operatorname { t r } C , \qquad I _ { 2 } : = \frac { 1 } { 3 } \operatorname { t r } \left( \cot C \right) , \qquad J : = \operatorname* { d e t } { \cal F } , } \end{array}\tag{26}
$$

where ${ \pmb C } = { \pmb F } ^ { T } { \pmb F }$ denotes the right Cauchy-Green tensor and cof C its cofactor. Each of these invariants satisfies objectivity and material symmetry for isotropic materials, i.e., for a symmetry group equal to the full group of proper orthogonal matrices. Moreover, $I _ { 1 }$ and $I _ { 2 }$ are convex functions of $\pmb { F }$ and cof $F _ { \mathrm { { ; } } }$ , respectively, while J enters the energy directly. From these invariants, we define the input vector as

$$
\begin{array} { r } { { \pmb x } ( { \pmb F } ) : = \big [ { \boldsymbol I } _ { 1 } - 1 , ~ { \boldsymbol I } _ { 2 } - 1 , ~ { \boldsymbol J } - 1 , ~ - [ { \pmb J } - 1 ] \big ] , \qquad { \pmb x } ( { \pmb I } ) = { \bf 0 } . } \end{array}\tag{27}
$$

Since all weights of the network are constrained to be non-negative (see below), N is non-decreasing in each entry of x. The mirrored entry $- [ J - 1 ]$ therefore allows the network to also decrease with $J ,$ such that both volumetric expansion and compression can be represented.

The network follows the Input Convex Neural Network (ICNN) architecture of [85], which is convex with respect to its input. In line with [9], we slightly adapt the original architecture to be zero anchored, i.e., $\mathcal { N } ( \mathbf { 0 } ; \pmb { \theta } ) = 0$ . The first hidden layer is obtained via

$$
\begin{array} { r } { \pmb { h } _ { 1 } : = \varphi _ { 1 } \big ( \pmb { S } _ { 1 } \pmb { x } + \pmb { b } _ { 1 } \big ) - \varphi _ { 1 } \big ( \pmb { b } _ { 1 } \big ) , } \end{array}\tag{28}
$$

while all subsequent layers, including the output layer $L ,$ are obtained via

$$
\begin{array} { r } { h _ { \ell } : = \varphi _ { \ell } \bigl ( W _ { \ell } h _ { \ell - 1 } + S _ { \ell } x + b _ { \ell } \bigr ) - \varphi _ { \ell } \bigl ( b _ { \ell } \bigr ) , \qquad \ell = 2 , \ldots , L , } \end{array}\tag{29}
$$

where $h _ { \ell }$ denotes the output of layer $\ell , \ \varphi _ { \ell }$ the activation function applied component-wise, $W _ { \ell }$ and $S _ { \ell }$ the weight matrices acting on the previous layer and on the input, respectively, and $\pmb { b } _ { \ell }$ the bias vector. The network output is the scalar $\begin{array} { r } { \mathcal { N } ( \pmb { x } ; \pmb { \theta } ) : = \pmb { h } _ { L } , } \end{array}$ , and θ collects all weights and biases. The subtraction of $\varphi _ { \ell } ( b _ { \ell } )$ shifts each layer such that it vanishes at ${ \pmb x } = { \bf 0 }$ , which yields the zero anchoring. In contrast to the original ICNN architecture, we constrain all weights W and S to be non-negative, since the invariants $I _ { 1 }$ and $I _ { 2 }$ are already convex functions of F and cof $F ,$ cf. [86]. Together with convex and non-decreasing activation functions $\varphi ,$ this renders $\mathcal { N }$ convex and non-decreasing in x, and hence $\mathcal { N } ( \pmb { x } ( \pmb { F } ) ; \pmb { \theta } )$ polyconvex in ${ \pmb F } _ { \parallel }$

Growth term. The network alone does not guarantee coercivity of the energy, see Section Appendix A. To this end, we adopt the growth term of [14] and adapt it to the invariants in Eq. (26),

$$
\begin{array} { r } { \psi _ { \mathrm { g r } } ( \pmb { F } ) : = \epsilon \Big [ \big [ I _ { 1 } - 1 - \frac { 2 } { 3 } \ln J \big ] + \big [ I _ { 2 } - 1 - \frac { 4 } { 3 } \ln J \big ] + \big [ J - 1 - \ln J \big ] \Big ] , } \end{array}\tag{30}
$$

where $0 < \epsilon \ll 1$ denotes a scaling parameter. Note that each summand is greater than or equal to $\mathrm { z e r o ^ { 3 } }$ , and polyconvex, since − ln J is convex in J. Moreover, the growth term does not contribute to the stress in the reference configuration,

$$
\left. \frac { \partial \psi _ { \mathrm { g r } } } { \partial { \cal F } } \right| _ { { \cal F } = { \cal I } } = 0 ,\tag{31}
$$

which is discussed in greater detail in Section Appendix A.

Logarithmic normalization term. In general, the network $\mathcal { N }$ yields a non-zero stress at ${ \boldsymbol { F } } = { \boldsymbol { I } } .$ which has to be compensated by the normalization. To this end, we define the derivatives of the network with respect to the components of x at the undeformed configuration,

$$
a _ { 1 } : = \left. \frac { \partial N } { \partial x _ { 1 } } \right| _ { F = I } , \qquad a _ { 2 } : = \left. \frac { \partial N } { \partial x _ { 2 } } \right| _ { F = I } , \qquad a _ { J } : = \left. \frac { \partial N } { \partial x _ { 3 } } \right| _ { F = I } - \left. \frac { \partial N } { \partial x _ { 4 } } \right| _ { F = I } ,\tag{32}
$$

where $a _ { 1 } , a _ { 2 } \geq 0$ due to the monotonicity of $\mathcal { N }$ with respect to its inputs, while $a _ { J } \in \mathbb { R }$ . With these derivatives at hand, we introduce the normalization term

$$
\psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } ( J ; \theta ) : = - \left[ \textstyle { \frac { 2 } { 3 } } a _ { 1 } + \textstyle { \frac { 4 } { 3 } } a _ { 2 } \right] \ln J - a _ { J } \left[ J - 1 \right] , \qquad \left. { \frac { \partial \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } } { \partial F } } \right| _ { F = I } = - \left[ \textstyle { \frac { 2 } { 3 } } a _ { 1 } + \textstyle { \frac { 4 } { 3 } } a _ { 2 } + a _ { J } \right] I ,\tag{33}
$$

which is convex in $J ,$ and thus polyconvex, and cancels the stress contribution of $\mathcal { N }$ at the reference state. Since all three terms in $\operatorname { E q . }$ (25) vanish at $F = I ,$ , the energy is zero and the stress vanishes in the undeformed configuration. In contrast to the afine normalization introduced in [86], the logarithmic normalization, together with the zero-anchored ICNN architecture and the growth energy, ensures a non-negative energy, which is discussed in greater detail in Section Appendix A. Nevertheless, both normalizations satisfy polyconvexity and coercivity.

## 2.4. Workflow during hyperreduced inverse discovery

The reduction techniques introduced above lower the cost of a single solve, yet a fixed projection V and a fixed set of sampled elements $\tilde { \mathcal { E } }$ cannot be expected to capture the structural response for all states of the weights θ visited during training. To this end, we adapt the reduction to the evolving weights. In the following, we first introduce the bootstrap epochs, then the canary epochs, and conclude with the error measures that govern the switch between the reduced and the full order model.

Bootstrap epochs. The first $N _ { \mathrm { b o o t } }$ epochs of the training are solved in the full space for all load steps $t _ { n }$ . Their purpose is to collect representative snapshots in the snapshot matrix S. Subsequently, the projection V and the set of sampled elements $\tilde { \mathcal { E } }$ for the ECSW are computed from the snapshots collected so far. Provided the resulting reduced solution does not deviate too strongly from the full order solution, the following epochs are solved in the reduced space.

Canary epochs. Since V and $\tilde { \mathcal { E } }$ are based on past snapshots, they are in general not suitable throughout the entire training. Hence, we interleave canary epochs, in which the full order problem is solved at the current weights θ. After the bootstrap epochs, every $N _ { \mathrm { c a n a r y } }$ -th epoch is a canary epoch.<sup>4</sup> Each canary epoch consists of three steps. (i) Snapshot update. Based on the full order solution, we decide whether the current state is representable by the present projection V. If it is not, the corresponding snapshots are added to S. Between two canary epochs, S remains unchanged, since no full order solution is available for comparison. (ii) Update of the reduction. If new snapshots were added to S or if the reduced solution was found to deviate too strongly at the previous canary epoch, V and $\tilde { \mathcal { E } }$ are recomputed from the updated S. Otherwise, both are retained, as they remain constant between two canary epochs. $( i i i )$ Selection of the model. Independent of whether the reduction was updated, we compare the full order solution with the reduced solution at the same weights θ. If both deviate too strongly, the epochs up to the next canary epoch are solved with the full order model; otherwise, they are solved with the reduced order model.

The error measures underlying steps (i)-(iii) are discussed next.

Dimension of the reduced space. The projection V is defined by the first $N _ { \mathrm { r } }$ columns of U, a procedure known as proper orthogonal decomposition (POD) [82, 87, 56]. To choose $N _ { \mathrm { r } }$ , we employ the relative projection error of the snapshots,

$$
\mathrm { e r r o r } \mathrm { r } ( k ) = \sqrt { \frac { \sum _ { \substack { i = k + 1 } } ^ { p } \hat { \sigma } _ { i } ^ { 2 } } { \sum _ { \substack { i = 1 } } ^ { p } \hat { \sigma } _ { i } ^ { 2 } } } , \qquad \hat { \sigma } _ { i } : = \frac { \sigma _ { i } } { \sigma _ { 1 } } ,\tag{34}
$$

where $\sigma _ { i }$ denotes the i-th singular value of S and $\hat { \sigma } _ { i }$ its normalization by the largest one. Note that the normalization cancels in the quotient and merely renders the measure independent of the scaling of S. We choose the smallest dimension for which this error stays below a prescribed tolerance ε<sub>POD</sub>, i.e.,

$$
N _ { \mathrm { r } } = \operatorname* { m i n } \left\{ k : \mathrm { e r r o r } _ { \mathrm { r } } ( k ) \leq \varepsilon _ { \mathrm { P O D } } \right\} .\tag{35}
$$

Snapshot update. In each canary epoch, we add a new full order state $\mathbf { d } _ { \mathrm { f , n e w } }$ to S only if it cannot be suficiently well represented by the current projection V. This is quantified by the relative projection error

$$
\mathrm { e r r o r } _ { \mathrm { p r o j } } ( \mathbf { d } _ { \mathrm { f , n e w } } ) = \frac { \lVert [ \mathbf { d } _ { \mathrm { f , n e w } } - \mathbf { d } _ { \mathrm { f , 0 } } ] - \mathbf { V } \mathbf { V } ^ { T } [ \mathbf { d } _ { \mathrm { f , n e w } } - \mathbf { d } _ { \mathrm { f , 0 } } ] \rVert } { \operatorname* { m a x } \left\{ \lVert \mathbf { d } _ { \mathrm { f , n e w } } - \mathbf { d } _ { \mathrm { f , 0 } } \rVert , \beta \right\} } ,\tag{36}
$$

where $\beta > 0$ is a small number that avoids a division by zero. If erro $\mathbf { \dot { \sigma } } _ { \mathrm { p r o j } } ( \mathbf { d } _ { \mathrm { f , n e w } } ) \geq \varepsilon _ { \mathrm { p r o j } }$ , the state is added to S. In the present work, we limit the number of snapshots to $N _ { \mathrm { s n a p } } ^ { \mathrm { m a x } }$ and replace the oldest snapshot whenever the limit is reached, such that $N _ { \mathrm { r } } \leq N _ { \mathrm { s n a p } } ^ { \mathrm { m a x } }$

Number of ECSW elements. To obtain a sparse solution, we solve Eq. (24) with a greedy algorithm<sup>5</sup>, which successively adds the column of G most correlated with the current residual to the support. We terminate once the relative training error

$$
\operatorname { e r r o r } _ { \mathrm { E C S W } } = { \frac { \| \mathbf { G } \mathbf { y } - \mathbf { b } \| } { \| \mathbf { b } \| } }\tag{37}
$$

falls below a prescribed tolerance ε<sub>ECSW</sub> or the support reaches the maximum size $n _ { \mathrm { e l } }$ . The elements with $w ^ { e } > 0$ form the reduced set $\tilde { \mathcal { E } } .$

Model selection. Finally, we decide in each canary epoch whether the reduced model remains suficiently accurate. To this end, we compare the reduced and the full order loss, $\mathcal { L } ^ { \mathrm { r e d } }$ and ${ \mathcal L } ^ { \mathrm { f u l l } }$ , evaluated at identical weights $\pmb { \theta } ^ { 6 }$ . With the error measures $e ^ { \mathrm { f u l l } } \stackrel { \cdot } { = } \sqrt { \mathscr { L } ^ { \mathrm { f u l l } } }$ and $e ^ { \mathrm { r e d } } = \sqrt { \mathcal { L } ^ { \mathrm { r e d } } }$ , we define the absolute and the relative deviation

$$
\operatorname { e r r o r } _ { \mathcal { L } } = \sqrt { \left| \mathcal { L } ^ { \mathrm { r e d } } - \mathcal { L } ^ { \mathrm { f u l l } } \right| } , \qquad \operatorname { e r r o r } _ { \mathcal { L } , \mathrm { r e l } } = \frac { \left| e ^ { \mathrm { r e d } } - e ^ { \mathrm { f u l l } } \right| } { \operatorname* { m a x } \left\{ e ^ { \mathrm { f u l l } } , e ^ { \mathrm { r e d } } , \beta \right\} } .\tag{38}
$$

The reduced model is accepted if

$$
\mathrm { e r r o r } _ { \mathcal { L } } \leq \varepsilon _ { \mathcal { L } } \quad \mathrm { o r } \quad \mathrm { e r r o r } _ { \mathcal { L } , \mathrm { r e l } } \leq \varepsilon _ { \mathcal { L } , \mathrm { r e l } } .\tag{39}
$$

Both criteria are required: a purely relative tolerance tightens uncontrollably as ${ \mathcal { L } } ^ { \mathrm { f u l l } } \to 0$ during convergence, whereas a purely absolute tolerance admits large relative deviations while the loss is still large. If neither condition holds, the following epochs are solved with the full order model until a subsequent canary epoch accepts the reduction.

## 3. Results

We now examine whether the hyperreduced inverse discovery of Section 2 lowers the cost of the training without degrading the identified material. To this end, we consider three examples that probe diferent aspects of the approach. In Section 3.1, we identify the two phases of a heterogeneous plate of the Mechanical MNIST Challenge from the reaction force alone, for which a full-order reference training allows a direct comparison of accuracy and cost. In Section 3.2, we turn to experimental data of a perforated membrane and compare a force-only loss with a loss that additionally contains the measured displacement field. In Section 3.3, we consider a synthetic, three-dimensional block with two stif inclusions, for which the ground truth is known and the full-order solve dominates the cost. All settings are collected in Table 1. Throughout this section, the reported errors are evaluated by a full-order solve at the identified parameters.

The approach is implemented in our in-house code ASIMOV, which builds on JAX [88]. We run all computations in double precision on the CPU of a MacBook Pro (Apple M4, ten cores, 24 GB memory, macOS 26.6), to which all reported wall-clock times refer.

## 3.1. Case 1: Heterogeneous material system

Our first example is a heterogeneous specimen from the Mechanical MNIST Challenge [92, 93]. In every specimen of this dataset, an inclusion shaped like a handwritten digit is embedded in a compliant surrounding phase, the matrix, and is expected to respond more stifly than the latter. The plates are loaded in uniaxial tension, and the reaction force of the testing machine as well as the in-plane displacements on one face, obtained by digital image correlation (DIC), are provided for a sequence of load steps. We select sample 22, which has also served as training specimen for the full-order identification in [40]. Since both phases are present in a single specimen, the total reaction force depends on two constitutive laws at once. This raises two questions: Can the reaction force of a single test separate the two phases, and does the adaptive reduction remain reliable while the optimizer distributes the stifness between them?

Table 1: Settings of the inverse identification for all three cases. In each case, the full-order solves (bootstrap, canary and final evaluation) use the same solver settings as the reduced runs. The reduction parameters of Case 1 define the base configuration, around which the study in Fig. 7 varies one parameter at a time.
<table><tr><td>Category</td><td>Parameter</td><td>Case 1</td><td>Case 2</td><td>Case 3</td></tr><tr><td>Data</td><td>specimen</td><td>MNIST sample 22</td><td>TT1 [25]</td><td>synthetic block</td></tr><tr><td></td><td>load steps  $N _ { \mathrm { t } }$ </td><td>21</td><td>51 (DIC at 11)</td><td>10</td></tr><tr><td>loss</td><td></td><td>force</td><td>force; force + displ.</td><td>force</td></tr><tr><td></td><td>normalization of the force</td><td>target, f</td><td>target and model, f</td><td>target, f</td></tr><tr><td></td><td>f</td><td>15.94 N</td><td>236.9 N</td><td>85.61 N</td></tr><tr><td>Discretization</td><td>element</td><td>hexahedron</td><td>hexahedron, Q1/P0</td><td>hexahedron</td></tr><tr><td></td><td>elements nel</td><td>2500</td><td>1518</td><td>15625</td></tr><tr><td></td><td>degrees of freedom, free Nf</td><td>15 606, 14 994</td><td>10 206, 9 234</td><td>52 728, 48 672</td></tr><tr><td>Network</td><td>networks</td><td>one per phase</td><td></td><td>one per phase</td></tr><tr><td></td><td>hidden layers, activation</td><td colspan="3">one [8, 8], softplus; output activation softplus</td></tr><tr><td></td><td>weight constraint</td><td colspan="3"> $\bar { \bf W } , \bar { \bf S } \ge { \bf 0 } .$  enforced by projection after each update</td></tr><tr><td></td><td>growth parameter €, Eq. (30)</td><td> $1 0 ^ { \dot { - } 4 }$ </td><td></td><td></td></tr><tr><td>Optimization</td><td>optimizer, weight decay, precision</td><td>AdamW [89], 10−4, float64</td><td></td><td></td></tr><tr><td></td><td>epochs</td><td>1000</td><td>700</td><td>400</td></tr><tr><td></td><td>learning rate</td><td> $2 \cdot 1 0 ^ { - 3 }$ </td><td> $2 \cdot 1 0 ^ { - 2 }$ </td><td> $2 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>global norm clipping [90]</td><td></td><td></td><td>1</td></tr><tr><td></td><td>Newton-Raphson: tolerance, max. iterations</td><td colspan="3">1 10−8, 50, with line search</td></tr><tr><td>Solver</td><td>full order: linear solver</td><td></td><td>BiCGSTAB [91], block Jacobi preconditioner (3 × 3)</td><td></td></tr><tr><td></td><td>full order: Krylov tolerance, max. iterations</td><td>10−8, 2 000</td><td>10−8, 2000</td><td>10−9, 5000</td></tr><tr><td></td><td>reduced order: linear solver</td><td>dense direct</td><td></td><td></td></tr><tr><td></td><td>bootstrap epochs  $N _ { \mathrm { b o o t } } ,$  canary interval Ncanary</td><td>2, 50</td><td>2,50</td><td>2,50</td></tr><tr><td>Reduction</td><td>maximum number of snapshots Nmax snap</td><td>30</td><td>30</td><td>30</td></tr><tr><td></td><td>snapshot update  $\varepsilon _ { \mathrm { p r o j } } , \mathrm { E q . ~ ( 3 6 ) }$ </td><td> $5 \cdot 1 0 ^ { - 3 } $ </td><td> $5 \cdot 1 0 ^ { - 3 } $ </td><td> $5 \cdot 1 0 ^ { - 3 } $ </td></tr><tr><td></td><td>POD tolerance εPOD, Eq. (35)</td><td>10⁻4</td><td>10-6</td><td>10⁻4</td></tr><tr><td></td><td>ECSW tolerance εECsw, Eq. (37)</td><td>10-4</td><td>10-4</td><td>10-4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>model selection  $\varepsilon _ { \mathcal { L } } , \varepsilon _ { \mathcal { L } , \mathrm { r e l } } , \mathrm { E q } .$  (39) capacity of the padded set  $\tilde { \mathcal { E } }$ </td><td> $1 . 5 \cdot 1 0 ^ { - 2 } , 5 \cdot 1 0 ^ { - 2 }$  384</td><td>1024</td><td>96, then 192</td></tr></table>

Fig. 1 shows the specimen in the reference configuration. The plate is discretized by $5 0 \times 5 0 \times 1$ trilinear hexahedral elements, i.e., $n _ { \mathrm { e l } } = 2 5 0 0$ elements with one element across the thickness. Following [40], the mesh is aligned with the DIC grid, such that the nodes on the observed face coincide with the DIC points and the measured displacements can be compared without interpolation. Each element is assigned to exactly one of the two phases, and each phase is described by its own network of the form Eq. (25), i.e., $\pmb { \theta } = [ \pmb { \theta } _ { \mathrm { M } } , \pmb { \theta } _ { \mathrm { I } } ]$ . On the top and the bottom edge, we prescribe the in-plane displacements measured by DIC at the corresponding row of nodes, while the out-of-plane displacement is fixed. Since DIC observes only one face, both node layers of an edge receive the same value, which corresponds to the assumption of a uniform motion across the thickness of 2 mm. The loading path comprises $N _ { \mathrm { t } } = 2 1$ load steps up to a prescribed displacement of approximately 7.5 mm.

In the following, we first summarize the loss and the hyperparameters, before we discuss the training history together with the adaptivity of the reduction. We then assess the identified material, analyze the trial space and the ECSW rule, and conclude with a parameter study of the reduction.

Loss and hyperparameters. We identify both phases exclusively from the reaction force, i.e., we set $\alpha = 0$ in Eq. (9). The measured displacements thus never enter the training and serve as an independent check of the identified material. To obtain a loss of order one irrespective of the units, we divide the measured force by its largest absolute value ${ \bar { f } } : = \operatorname* { m a x } _ { n } | f _ { n } ^ { \mathrm { e x p } } | = 1 5 . 9 4 { \mathrm { ~ N } }$ , such that the loss reads

$$
\mathcal { L } = \frac { 1 } { { { N _ { \mathrm { t } } } } } \sum _ { n = 1 } ^ { { N _ { \mathrm { t } } } } { \left[ { \mathcal { H } ( { \bf { d } } _ { n } , \bf { \Theta } \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } \mathbf { ) } - { f _ { n } ^ { \mathrm { e x p } } } / \bar { f } } \right] ^ { 2 } } .\tag{40}
$$

Note that we scale only the target and not the model response. Since the plate is loaded exclusively by prescribed displacements, the external potential vanishes, and a positive factor in front of $\psi$ leaves the minimizer of Eq. (1)

![](images/13411d4cd9f53128788e93a06ca934fa135b1a43fc041309fd034c8954b82014.jpg)  
Figure 1: Case 1, sample 22 of the Mechanical MNIST Challenge [92, 93] in the reference configuration. The inclusion is shown in green and the matrix in grey. The plate of $4 0 \times 4 0 \times 2$ mm is discretized by $5 0 \times 5 0 \times 1$ hexahedral elements, whose nodes on the observed face coincide with the DIC points. On the top and bottom edges, the in-plane displacements $u _ { x } ( x , t )$ and $\boldsymbol { u } _ { y } ( \boldsymbol { x } , t )$ measured by DIC are prescribed for $N _ { \mathrm { t } } = 2 1$ load steps, and the out-of-plane displacement is fixed. The reaction force on the top edge is the only quantity that enters the loss.

unchanged, while it scales all reaction forces by the same factor. Consequently, the networks learn the scaled energy $\psi / { \bar { f } }$ without any efect on the displacement field, and the physical stresses and forces follow from a multiplication by <sup>¯</sup>f. Both phases share the same architecture, two hidden layers with eight neurons each, and difer only in their parameters. We initialize them with diferent random keys, since two identical networks would start at a symmetric point of the loss landscape that only the data could resolve. The non-negativity of the weights is enforced by a projection after each update. We train with AdamW [89] at a learning rate of $2 \cdot 1 0 ^ { - 3 }$ and the default weight decay of $\mathrm { \dot { 1 } 0 ^ { - 4 } }$ and clip the gradient by its global norm [90] for 1 000 epochs, each of which comprises all load steps. The full-order reference run uses identical settings and the same initial parameters, and difers only in that every epoch is solved in the full space. All remaining settings are listed in Table 1.

Training history and adaptivity of the reduction. Fig. 2(a) shows that the loss decreases by roughly four orders of magnitude within the first 150 epochs. Afterwards, it continues to decrease slowly and reaches $\mathcal { L } = 2 . 5 0 \cdot 1 0 ^ { - 4 }$ after 1 000 epochs, while the gradient norm in Fig. 2(b) drops from about $1 0 ^ { 2 }$ to $1 . 3 4 \cdot 1 0 ^ { - 2 }$ . The full-order reference follows a nearly identical path during the first 300 epochs and ends at $\mathcal { L } = 4 . 0 9 \cdot 1 0 ^ { - 4 }$ . That both runs separate afterwards is expected: the gradient in $\operatorname { E q . }$ (20) is the exact gradient of the reduced model, and small diferences to the full-order gradient accumulate over hundreds of Adam updates. In terms of the force, both final losses correspond to a root-mean-square deviation of about 2 % of <sup>¯</sup>f or less.

The cost per epoch in Fig. 2(c) reveals three distinct levels. A reduced epoch takes about 1.2 s at the beginning and 1.6 s at the end of the training. A canary epoch, which comprises a full-order solve, the snapshot update and the recomputation of the reduction, takes between 23 and 72 s, and an epoch with a rebuild of the reduced model between 78 and 114 s. For comparison, a full-order epoch of the reference run requires 33.4 s in the median, rising from about 20 s at the beginning to about 50 s towards the end of the training; the canary epochs follow the same trend, as they are dominated by the full-order solve. The median epoch, 1.23 s against 33.4 s, is thus accelerated by a factor of 27. Over the entire training, the wall-clock time decreases from 8.95 h to 0.69 h, see Fig. 2(d), i.e., by a factor of 13. The gap between both factors is explained by Fig. 3(d): although the 19 canary epochs make up less than 2 % of all epochs, they account for 41 % of the wall-clock time, while the 979 reduced epochs account for 54 %. Hence, once the reduced solve is cheap, the full-order checks rather than the reduced model limit the overall speed-up. Moreover, the reduced mode was accepted at every canary epoch, such that no epoch after the bootstrap fell back to the full-order model.

Fig. 3(a) shows how the reduction adapts to the evolving parameters. The two bootstrap epochs provide 42 equilibrium states, such that the snapshot matrix reaches its capacity of $N _ { \mathrm { s n a p } } ^ { \mathrm { m a x } } = 3 0$ right after the bootstrap. From then on, every state accepted at a canary epoch replaces the oldest snapshot, and the snapshot matrix follows the most recent part of the optimization path. At each canary epoch, those states of the full-order trajectory whose projection error Eq. (36)

![](images/2e95d8417fadcf052cb58a90a2fa85f5a1a05e16f2cbbdbd085c434a5995fb42.jpg)  
Figure 2: Case 1, training history of the hyperreduced identification. (a) The loss decreases by about four orders of magnitude within the first 150 epochs and reaches $\mathcal { L } = 2 . 5 0 \cdot 1 0 ^ { - 4 }$ , compared with $4 . 0 9 \cdot 1 0 ^ { - 4 }$ of the full-order reference. (b) The gradient norm drops from about $1 0 ^ { 2 }$ to $1 . { \dot { 3 4 } } \cdot 1 0 ^ { - 2 } \dot  $ ; rebuilds cause no visible disturbance. (c) Wall time per epoch: reduced epochs take 1.2 to 1.6 s, canary epochs (orange markers) between 23 and $^ \mathrm { 7 2 ~ s , }$ and canary epochs with a rebuild up to 114 s. (d) The cumulative wall-clock time of 0.69 h is 13 times smaller than the 8.95 h of the full-order reference (dashed). Grey dashed lines mark canary epochs and black dashed lines rebuilds of the reduced model.

exceeds $\varepsilon _ { \mathrm { p r o j } }$ are added, but at least the state with the largest error; in the present run, between 1 and 21 of the 21 states per canary epoch. The dimension of the trial space grows stepwise from $N _ { \mathrm { r } } = 9 \mathrm { t o } N _ { \mathrm { r } } = 1 3$ at the canary epochs 302, 402, 852 and 902. At exactly these canary epochs, the projection error of the updated snapshots, evaluated with the previous dimension, exceeds ε<sub>POD</sub>, see the squares in Fig. 3(b). In other words, the new snapshots carry deformation content that the previous trial space cannot represent within the tolerance, and Eq. (35) adds one mode. At all other canary epochs, both error and error remain below $1 0 ^ { - 4 }$ with the current dimension. Interestingly, nine modes sufice during the steep descent of the loss. We attribute this to the scaling argument above: as long as the optimizer mainly adjusts the overall stifness level, the displacement field barely changes, and the existing snapshots remain representative. Additional modes become necessary only later, when the ratio between the phases and the nonlinearity of the response change, which alters the pattern of deformation itself.

The reduced problem is compiled just-in-time [88], and every change in the dimension of an array triggers a new compilation. We refer to such an update as a rebuild, marked by the black dashed lines in Figs. 2 and 3. In contrast, an update that keeps all dimensions merely replaces the entries of the compiled arrays. Since the number of sampled elements changes at almost every canary epoch, from 186 elements after the bootstrap to 356 elements at the end, see Fig. 3(c), we store $\tilde { \mathcal { E } }$ in an array of fixed capacity and fill the unused entries with an element of zero weight. The capacity is chosen at the bootstrap as the smallest size of a fixed sequence 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, $\mathbf { \Omega } _ { - } 0 2 4 , \mathbf { \Omega } \cdot \mathbf { \Omega } .$ . that is not smaller than max $\{ 1 . 5 n , n + 4 \}$ , where n denotes the number of active elements; here, $1 . 5 \cdot 1 8 6 = 2 7 9$ yields 384 elements. Only if a later rule outgrows this capacity, it is raised by the same rule, and it is never decreased. These padded elements are evaluated, but by $w ^ { e } = 0$ they contribute neither to $\tilde { \mathbf { r } } _ { \mathrm { r } }$ nor to $\mathbf { K } _ { \mathrm { r } }$ in Eq. (19). Hence, as long as the active set fits into this capacity, a new ECSW rule requires no recompilation. The reduced basis, in contrast, is not padded, so that each increase of $N _ { \mathrm { r } }$ results in a rebuild; a decrease is not admitted, for the same reason. In the present run, the active set remains below the capacity throughout, and all four rebuilds after the initial construction stem from the growth of the trial space. The padding comes at the price of evaluating up to 198 elements without contribution at the beginning and 28 at the end. In return, it avoids a recompilation at 15 of the 19 canary epochs, each of which costs about 45 to 55 s here, as the comparison of the canary epochs with and without a rebuild in Fig. 2(c) shows.

![](images/e6f6481c1b15df31e967622fb46519f917d05e98bbf52f0a278cce35ef76a0f4.jpg)  
Figure 3: Case 1, adaptivity of the reduction. (a) The snapshot matrix reaches its capacity of $N _ { \mathrm { s n a p } } ^ { \mathrm { m a x } } = 3 0$ directly after the two bootstrap epochs, while the dimension of the trial space grows stepwise from $N _ { \mathrm { r } } = 9$ to $N _ { \mathrm { r } } = 1 3$ . (b) Training errors of POD and ECSW at each canary epoch; squares mark the POD error of the updated snapshots with the previous dimension, which exceeds $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ at four canary epochs and triggers the growth of $N _ { \mathrm { r } } .$ (c) The number of active ECSW elements grows from 186 to 356, whereas the padded capacity remains at 384, such that no change of the ECSW rule requires a recompilation. (d) Split of the wall-clock time: the 19 canary epochs consume 41 % of the total time, almost as much as the 979 reduced epochs.

Identified material. Fig. 4 compares the response of the identified material with the measurement. The machine curve in Fig. 4(a) is reproduced over all load steps with a root-mean-square error of 0.249 N at a maximum force of 15.9 N, i.e., a relative error of 2.5 %, and the force parity in Fig. 4(b) yields $R ^ { 2 } = 0 . 9 9 7 0$ . More informative, however, are the displacements, which the loss never contained. Their relative error amounts to 10.4 %, and the displacement parity in Fig. 4(c) yields $R ^ { 2 } = 0 . 9 0 9$ for the transverse and $R ^ { 2 } = 0 . 9 6 7$ for the loading direction, and the contour lines of the discovered and the measured displacement in Fig. 4(d)–(e) largely coincide. The relative error in Fig. 4(f) remains below 5 % in large parts of the plate and reaches up to 15 % in a band along the upper flank of the inclusion and at its lower tip close to the clamped edge. For the full-order reference, the force error is slightly larger with 0.326 N or 3.2 %, whereas the displacement agreement is better with a relative error of 4.7 %, $R ^ { 2 } = 0 . 9 7 6$ and $R ^ { 2 } = 0 . 9 9 3$ , and a pointwise maximum of about 8 %. Both parameter sets thus reproduce the force comparably well but distribute the stifness slightly diferently between the phases, which becomes visible only in the displacement field. Since both parameter sets are evaluated by the same full-order solve, this diference is not an error of the reduced model; rather, two optimization paths end in diferent minima of comparable loss. In [40], the displacements entered the loss explicitly, which constrains the choice between such minima at the expense of requiring full-field data during training; for the same specimen, they report a force error of about 3.5 % and a displacement error of about 2.8 %. The force error of the hyperreduced identification is thus of the same order, whereas its displacement error is larger, as expected for a loss that contains no displacements.

Fig. 4(g)–(h) shows the displacement in loading direction and the corresponding component $F _ { y y }$ of the deformation gradient on the deformed configuration at the last load step. The stretch difers markedly between the phases: the inclusion remains almost undeformed with $F _ { y y } \approx 1 . 0$ , while the matrix above and below the inclusion is stretched up to $F _ { y y } = 1 . 8 7$ . Since both networks start from random parameters drawn from the same distribution, such a contrast can only arise if the optimizer has assigned a substantially stifer response to the inclusion than to the matrix, which is consistent with the description of the specimens [40]. Together with the agreement of the displacement field, this indicates that the force alone sufices to separate both phases, at least qualitatively. A quantitative comparison of the individual phase laws is not possible here, since the data contain no stress–strain response of the isolated phases.

![](images/69aded39a419956828bc174df29ea8834484d21c1f8f8ba295138ae108b752e9.jpg)

(a) Machine curve  
![](images/02b0755f4b93490f41fddc779b093f9917b734a97e62f617a0e377972bba13a7.jpg)

(b) Force parity  
![](images/7d4d950e2e1de7764bbaf4dc56c6d867a5cf8210244efddb9e033577440e7afb.jpg)

(c) Displacement parity  
![](images/20966e9b09d1af3e9716a38ab0e1dcc0c45d16fd3d01c79f60c3eef6eada298d.jpg)

(d) Transverse displacement  
![](images/f25491a7daf5eb51a5e29bb1fb0675648b9e1092abb1478f167392e9cc53de9e.jpg)

(e) Loading displacement  
![](images/6ce03e4972942b2f5c7fb3e34d82a865868b9eda5dcf00286318528a9aabd1dc.jpg)

(f) Relative displacement error  
![](images/8a90f67ee9e40048c0ac96c14149aa1b676d64c3958fd373e9c203ae6c69ad9f.jpg)

![](images/9295e80a2dc8ce7b15b8fc8c79299dd0a547a9113c712e2054e9c5ec097f7108.jpg)

(g) Displacement $u _ { y }$  
![](images/73d18809af51d02a1e7010b2b41ba998e6a19652b0b13055502a5a13d8fcc6e7.jpg)  
(h) Deformation gradient $F _ { y y }$  
Figure 4: Case 1, identified material evaluated by a full-order solve, compared with the measurement. (a) The machine curve is reproduced with a root-mean-square error of 0.249 N, and (b) the force parity yields $R ^ { 2 } = 0 . 9 9 7 0 .$ (c) The displacements, which are not part of the loss, agree with the DIC measurement with $R _ { x } ^ { 2 } = 0 . 9 0 9$ and $R _ { y } ^ { 2 } = 0 . 9 6 7 . ~ ( \mathrm { d } ) { - } ( \mathrm { e } )$ Contour lines of the measured (black) and discovered (red) transverse and loading displacement nearly coincide away from the inclusion. (f) The relative displacement error remains below 5 % in most of the plate and concentrates along the upper flank of the inclusion (green outline) with up to 15 %. (g)–(h) Deformed configuration at the last load step: the inclusion moves almost rigidly and remains almost undeformed with $F _ { y y } \approx 1 . 0 $ , whereas the matrix above and below the inclusion is stretched up to $F _ { y y } = 1 . 8 7 $ . Since only the reaction force enters the loss, this contrast is learned from the structural response alone.

![](images/8d6bf071d3123edfd59c89c9a26ebf901ad7e66ca06a89f09fb8593d72fec025.jpg)  
Figure 5: Case 1, trial space at the end of the training. Top left: normalized singular values $\hat { \sigma } _ { k }$ of the snapshot matrix and projection error error (k). After a decay of four orders of magnitude within eight modes, the spectrum forms a plateau, in which the projection error crosses $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ at $N _ { \mathrm { r } } = 1 3$ . Remaining panels: magnitude (color) and direction (arrows) of the first five modes. Mode 1 describes the stretching in loading direction, mode 2 a lateral motion of the upper part, mode 3 the deformation around the upper part of the inclusion, mode 4 the region around its lower tip, and mode 5 boundary layers along the clamped edges.

Trial space and hyperreduction. Fig. 5 shows the singular values of the final snapshot matrix together with the projection error Eq. (34). The normalized singular values decay by about four orders of magnitude within the first eigh modes, followed by a plateau between $1 0 ^ { - 4 }$ and $1 0 ^ { - 5 }$ that extends up to $k \approx 2 2$ , before they drop sharply. The projection error falls below $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ within this plateau at $k = 1 3$ , which is the final dimension $N _ { \mathrm { r } } = 1 3$ in Fig. 3(a). Since the threshold intersects the flat part of the spectrum, small changes of the snapshots shift the selected dimension by one mode, which explains the stepwise growth of $N _ { \mathrm { r } }$ during training. For the same reason, the choice of $\varepsilon _ { \mathrm { P O D } }$ strongly afects the size of the trial space, as the study below confirms. The first mode represents the stretching in loading direction, whose distribution is distorted by the stifer inclusion. The second mode describes a lateral motion of the upper part of the plate, while the third mode concentrates on the upper part of the inclusion and the adjacent matrix. Notably, this is the region in which the displacement error in Fig. 4(f) is largest, $\mathrm { i . e . , }$ the region whose deformation is most sensitive to the ratio between both phases. The fourth mode captures the lower tip of the inclusion close to the clamped edge, and the fifth mode boundary layers along both clamped edges, which presumably stem from the non-uniform measured edge displacements.

Fig. 6 shows the final ECSW rule. Of the 384 entries of the padded set, 356 refer to active elements, i.e., about 14 % of the mesh, of which 21 belong to the inclusion. The remaining 28 entries are padded with zero weight and repeatedly refer to one element, such that they add no element to the figure. The sampled elements cluster along both clamped edges as well as in and around the inclusion, in particular along the interface between the phases. In the matrix away from the inclusion, in contrast, only a few elements are selected, some of them with weights of up to 220. This reflects the principle of ECSW: in regions of nearly uniform deformation, a single element with a large weight represents many similar elements, whereas regions with strong gradients and a change of material require many elements with moderate weights. Notably, the element rows along both clamped edges are selected completely, with weights between 0.994 and 1.006, i.e., they are integrated practically exactly. This is a consequence of the loss: since the trial space spans only the free degrees of freedom, whereas the reaction force lives on the prescribed ones, the ECSW training additionally contains one row per snapshot that reproduces the reaction force entering the loss. The elements adjacent to the loaded edge contribute directly to this reaction, and the boundary layers of the fifth mode involve both edges. Note that the sampled set necessarily contains elements of both phases. Otherwise, the reduced residual and thus the gradient in Eq. (20) would not depend on the parameters of the missing phase, and this phase could not be trained.

![](images/568baf2312eaedbee7b7431a8252a10a14821d39c371b96af75cb0958122484b.jpg)  
Figure 6: Case 1, final ECSW rule on the deformed mesh. The remaining mesh is shown as wireframe and the inclusion in green. The 356 active elements (about 14 % of the mesh) are colored by their weight; the 28 padded entries of the set of 384 carry zero weight and add no element. The element rows along both clamped edges are selected completely with weights close to one, and further elements concentrate in and around the inclusion, whereas the matrix away from the inclusion is represented by few elements with weights of up to about 220.

Influence of the reduction parameters. Finally, we investigate how the parameters of the reduction afect accuracy and cost. To this end, we vary one parameter at a time around the base configuration in Table 1, which results in 13 additional runs. Fig. 7 shows the force error and the speed-up per epoch for each parameter, and Fig. 8 compares all runs with each other and with the full-order reference. Here, the speed-up per epoch denotes the ratio of the median epoch times of the full-order and the reduced run, excluding the first epoch. Since the few canary epochs do not afect the median, it isolates the eficiency of the reduced solve, whereas the cost of the canary epochs, discussed above, comes on top.

All configurations reproduce the measured force, see Fig. 8(a), with root-mean-square errors between 0.22 and 0.36 N, compared with 0.33 N for the full-order reference. The tolerances of the model selection, $\varepsilon _ { \mathcal { L } }$ and $\varepsilon _ { \mathcal { L } , \mathrm { r e l } }$ , have no influence in the tested range: all runs yield the same force error, and the speed-up varies by less than 1 %, see Fig. 7(b)–(c). Hence, the model selection takes the same decisions as in the base configuration, in which the reduced model was accepted at every canary epoch, and the criterion Eq. (39) acts as a safeguard that is not triggered in this example. The canary interval afects the accuracy only for the largest value, see Fig. 7(a): for $N _ { \mathrm { c a n a r y } } \leq 1 0 0$ , the error lies between 0.244 and 0.265 N, whereas for $N _ { \mathrm { c a n a r y } } = 2 0 0$ it increases to the level of the full-order reference, and the trial space grows only to $N _ { \mathrm { r } } = 1 1$ . The speed-up per epoch remains between 26 and 28, since the median is insensitive to the canary epochs, while the overall cost grows with their number. A similar trend appears for the snapshot update in Fig. 7(d): the strictest tolerance $\varepsilon _ { \mathrm { p r o j } } = 1 0 ^ { - 3 }$ yields the smallest error of all runs with 0.223 N, and the loosest one, $\varepsilon _ { \mathrm { p r o j } } = 1 0 ^ { - 2 }$ , the only error above the full-order reference with 0.358 N. Both observations indicate that the accuracy depends primarily on how up to date the snapshots are.

The cost, in contrast, is governed by the tolerances of POD and ECSW, see Fig. 7(e)–(f) and Fig. 8(c)–(d). Fo $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 3 }$ , five modes sufice, the padded set shrinks to 192 elements, and the speed-up per epoch reaches 60 at a force error of 0.285 N. For $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 5 }$ , the trial space grows to $N _ { \mathrm { r } } = 2 3$ and the padded set to 768 elements, which reduces the speed-up to 8.6 without a noticeable gain in accuracy. This sensitivity follows directly from the plateau of the singular values in Fig. 5. Likewise, $\varepsilon _ { \mathrm { E C S W } } = 1 0 ^ { - 3 }$ reduces the padded set to 256 elements and increases the speed-up to 35.5 at an error of 0.236 N, whereas $\varepsilon _ { \mathrm { E C S W } } = 1 0 ^ { - 5 }$ doubles the padded set to 768 elements and decreases the speed-up to 12.8, again without improving the force. Overall, the diferences in accuracy between the reduced runs are of the same order as the diference between the base configuration and the full-order reference, and they show no systematic trade-of against the cost. We therefore conclude that, in this example, the remaining error stems from the non-uniqueness of the force-based identification rather than from the reduction, and that the tolerances of POD and ECSW can be relaxed considerably as long as the snapshots are updated regularly.

In summary, a single force–displacement curve sufices to identify a material that separates both phases and reproduces the unseen displacement field, while the adaptive reduction lowers the cost of the training by a factor of 13 without degrading the force accuracy. In the next case, we examine whether these observations persist for a perforated specimen.

(a)  
![](images/ac1ca8a0ae5ae0adc8dafb73c3b03e74492f877c559e8664fa18217547c28ae9.jpg)  
(d)

(b)  
![](images/29ff8ec1752216fa59ba7c94bff79156b89a93df6863a463adfc92d2880ca675.jpg)

![](images/163841d4ce47e598509de091c7196b9a580990602ba1237230aa6214a1705448.jpg)

![](images/76047f39ac4e0fd30c3780a70bd66977ad42e33289813a37236fd35766ae6c3e.jpg)

(e)  
![](images/ec7c6a0f68ba1fc5e3f5fe5b317d29de6db22cc988c6805434b472bdc4ebbf83.jpg)

(f)  
![](images/bd73f2a75f1e33e8496f0c0cd1f705d94b4d70f0db3bbe082e1d8b01d25c9ba2.jpg)  
Figure 7: Case 1, one-at-a-time variation of the reduction parameters around the base configuration of Table 1. Shown are the force error of the identified material (blue, left axis) and the speed-up per epoch against the full-order reference (red, right axis); the dotted line marks the force error of the full-order reference. (a) Canary interval $N _ { \mathrm { c a n a r y } } \mathrm { : }$ only the largest interval deteriorates the accuracy. $\mathrm { ( b ) - ( c ) }$ Tolerances of the model selection: no influence in the tested range. (d) Snapshot update $\varepsilon _ { \mathrm { p r o j } } \colon$ stricter tolerances improve the accuracy at constant cost. (e) POD tolerance ε <sub>O</sub> and (f) ECSW tolerance ε <sub>CS</sub> : the speed-up varies between 8.6 and 60, while the force error remains close to the base configuration.

## 3.2. Case 2: Homogeneous material, perforated specimen

Next, we turn to the opposite situation: a single material, but a strongly heterogeneous deformation. We use the tensile test TT1 of the open dataset by Abbasi et al. [25], in which a perforated polymer membrane with six elliptical holes is stretched in tension, see Fig. 9. Along the narrow ligaments between the holes, the material is locally stretched far beyond the nominal strain, while the regions above and below the holes are barely loaded. For each recorded frame, the dataset provides the machine force and the DIC displacements, and both have also served for the full-order identification in [40]. Unlike in Case 1, the displacement field serves here as more than a check. We compare two identifications that difer only in their loss: one from the force alone and one from the force and the displacement field. This allows us to ask what the force already implies about the field, what the field adds to the identification, and how the reduction copes with a loss that contains the displacements themselves.

The region of interest of $1 6 0 \times 5 0 \times 2 . 5$ mm is discretized with the mesh provided with the dataset, which consists of 1 518 hexahedral elements with one element across the thickness, 3 402 nodes and 10 206 degrees of freedom, of which $N _ { \mathrm { f } } = 9 2 3 4$ are free. The material of TT1 is usually treated as incompressible [25, 40]. We do not impose this assumption, since the energy of Section 2.3 is compressible, but we use an element-wise constant volume ratio $\mathrm { ( Q 1 / P 0 ) }$ [94, 95] to avoid volumetric locking should the identified material approach the incompressible limit, see Section Appendix B. Since the element potential is diferentiated with the volume ratio as a function of the nodal displacements, the variational structure of $\operatorname { E q . }$ (5) is preserved. On the top and bottom edges, we prescribe B-spline fits of the measured in-plane displacements, as in [40], together with a vanishing out-of-plane displacement. The loading path consists of $N _ { \mathrm { t } } = 5 1$ load steps, essentially every second DIC frame up to frame 100, of which 11 carry a reference displacement field. The force is evaluated as the reaction on the bottom edge, and the material is represented by a single network with the architecture of Case 1.

In the following, we first introduce the two losses, then discuss training and reduction, and assess the identified material on the training and on an unseen specimen. We then compare the initial moduli of both identified materials with the common assumption of incompressibility, and conclude with the trial space and the ECSW rule.

![](images/70d2bfbe5dc18805db6a8d1b808524f9ac29b51a35d035d16f7f394a5e8d2903.jpg)

![](images/1fd411980f51f06d03f47ffb021caecbcf8bdcad22e8d8d0a9a77458d1b77aa9.jpg)  
(d) Reduction: POD rank and ECSW elements

![](images/778d527b68dbc40e751d594613b8866f2038425462caaf56e46687de30763ead.jpg)

![](images/5ec866b4070b3168cb711e78a81e72f2aed111357fbe2bbe0f2f61228426c78f.jpg)

Figure 8: Case 1, comparison of all configurations of the parameter study. (a) Force parity: all reduced runs (squares) and the full-order reference (circles) reproduce the measured force nearly identically. (b) Force error against the speed-up per epoch: all reduced runs except $\varepsilon _ { \mathrm { p r o j } } = 1 0 ^ { - 2 }$ reach an error at or below that of the full-order reference (dotted lines) at speed-ups between 8.6 and 60. (c) Median epoch times: reduced epochs require 0.5 to 4 s, canary epochs between 25 and 85 s. (d) Final dimension of the trial space $N _ { \mathbf { r } }$ and size of the padded ECSW set |E| <sup>˜</sup> , compared with the $n _ { \mathrm { e l } } = 2 5 0 0$ elements of the mesh; only the tolerances of POD and ECSW change these sizes considerably.  
![](images/22faf51b7af06968459d2914793da202d077f0277ab5eb2f729bcffa95355250.jpg)  
Figure 9: Case 2, region of interest of the perforated specimen TT1 of Abbasi et al. [25] in the reference configuration. The membrane of $1 6 0 \times 5 0 \times 2 . 5$ mm with six elliptical holes is discretized by 1 518 hexahedral elements with one element across the thickness. On the top and bottom edges, B-spline fits of the measured in-plane displacements $u _ { x } ( x , t )$ and $\boldsymbol { u } _ { y } ( \boldsymbol { x } , t )$ are prescribed for $N _ { \mathrm { t } } = 5 1$ load steps, and the out-of-plane displacement is fixed. The reaction force on the bottom edge enters both losses, the DIC displacements at 11 load steps only the second one.

Loss and hyperparameters. The first loss contains only the force, as in Case 1. Here, however, we divide both the measured and the predicted force by $\bar { f } = \operatorname * { m a x } _ { n } | f _ { n } ^ { \mathrm { e x p } } | = 2 3 6 . 9 \ : \mathrm { N } ,$ such that the identified energy carries physical units,

$$
\mathcal { L } _ { \mathrm { F } } = \frac { 1 } { N _ { \mathrm { t } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t } } } \left[ \frac { \mathcal { H } ( \mathbf { d } _ { n } , \pmb { \theta } ) - f _ { n } ^ { \mathrm { e x p } } } { \bar { f } } \right] ^ { 2 } .\tag{41}
$$

Both scalings share the same minimizer in terms of the physical energy, but they difer in the magnitude of the loss and thus in the meaning of the absolute tolerance $\varepsilon _ { L } .$ which we calibrated for the present scaling. The second loss adds the measured displacements at the set D of the 11 observed load steps,

$$
\mathcal { L } _ { \mathrm { F + u } } = \mathcal { L } _ { \mathrm { F } } + \frac { \alpha } { 2 N _ { \mathrm { o b s } } \left| \mathcal { D } \right| } \sum _ { n \in \mathcal { D } } \sum _ { k = 1 } ^ { N _ { \mathrm { o b s } } } \left\| \boldsymbol { u } _ { n } ^ { h } ( \boldsymbol { X } _ { k } ) - \boldsymbol { u } _ { n } ^ { \mathrm { e x p } } ( \boldsymbol { X } _ { k } ) \right\| ^ { 2 } ,\tag{42}
$$

where $X _ { k }$ denote the $N _ { \mathrm { o b s } }$ DIC points on the observed face, and the factor two accounts for the two in-plane components, such that the second term is the mean squared displacement error in $\mathrm { m m ^ { 2 } }$ . In contrast to Eq. (9), the force and the displacements are thus evaluated on diferent sets of load steps. We choose α such that both contributions are equal at the initial parameters, which yields $\alpha = 0 . 4 3 4 ~ \mathrm { m m ^ { - 2 } }$ , since α converts units as much as it weights the contributions. Recall from Case 1 that the displacement field is invariant under a scaling of the energy for a purely displacement-driven problem. Hence, the displacement term alone cannot determine the stifness level, while the force term determines it. Both terms are thus complementary: the force fixes the level of the response, and the displacement constrains its spatial distribution. Consequently, both terms can be reduced at the same time only if model and measurement are consistent with each other. Both runs start from the same initial parameters, run over 700 epochs at a learning rate of $2 \cdot 1 0 ^ { - 2 }$ and use $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 6 }$ ; all further settings are listed in Table 1. No full-order training run is available for this case.

Training history and adaptivity of the reduction. For the force-based run, Fig. 10(a) shows a decrease of the loss from about $1 0 ^ { - 1 }$ to $1 0 ^ { - 4 }$ within the first 100 epochs, accompanied by pronounced oscillations, and a slow decrease to $\mathcal { L } = 6 . 0 \cdot 1 0 ^ { - 5 }$ thereafter. The gradient norm in Fig. 10(b) exhibits short peaks at several canary epochs, i.e., whenever an updated ECSW rule slightly changes the reduced landscape, but these peaks decay within a few epochs and leave no trace in the loss. For the run with both channels, Fig. 12(a) shows that the total loss is dominated by the displacement term, which levels of at about $1 . 1 \cdot 1 0 ^ { - 1 }$ , while the force term falls below $1 0 ^ { - 3 }$ . At the first canary epoch, epoch 52, the displacement term jumps from 0.128 to 0.179. At this epoch, the full-order trajectory deviates from the current trial space by a projection error of $1 . 3 \cdot 1 0 ^ { - 2 }$ , i.e., well above $\varepsilon _ { \mathrm { p r o j } } = 5 \cdot 1 0 ^ { - 3 }$ , such that all 51 states are added to the snapshot matrix and the trial space is enlarged from 25 to 26 modes. Since the displacements enter this loss through the reconstruction $\mathbf { d } _ { \mathrm { f , 0 } } + \mathbf { V } \mathbf { q } ,$ an outdated trial space directly afects the loss value, and the reduced model had apparently underestimated the displacement misfit during the first 50 epochs.

The reduced epochs take about 11 s in both runs and thus about ten times longer than in Case 1. Three efects add up. First, the loading path contains 51 instead of 21 load steps. Second, the tighter tolerance $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 6 }$ leads to $N _ { \mathrm { r } } = 2 7$ modes, see Fig. 11(a), and thus to $N _ { \mathrm { r } } + N _ { \mathrm { r } } ^ { 2 } = 7 5 6$ quantities per snapshot that ECSW has to reproduce. Third, the deformation concentrates in many small regions around the holes, and ECSW requires 613 to 801 active elements, i.e., up to 53 % of the mesh, stored in a padded set of 1 024 elements, see Fig. 11(c). The hyperreduction is therefore far less efective than for the plate of Case 1. The canary epochs take between 123 and 169 s in the force-based and between 145 and 189 s in the two-channel run. Their cost is dominated by the update of the reduction rather than by the full-order solve, since the first epoch, which is solved in full order, took only 55 s including the compilation. Without a full-order reference run, we can therefore only state that a reduced epoch is several times cheaper than a full-order epoch at this mesh size; the 685 reduced epochs account for 79 % of the total wall-clock time of 2.73 h, and the 13 canary epochs for 20 %, see Fig. 11(d).

The adaptivity difers from Case 1 in one respect. Already a single epoch provides 51 states, i.e., more than the snapshot capacity, such that the snapshot matrix is full from the first bootstrap epoch on. The trial space grows only once in the force-based run, from 26 to 27 modes at epoch 102, and twice in the two-channel run, at epochs 52 and 102, see Fig. 12. Afterwards, the projection error of the snapshots decreases by several orders of magnitude to about $1 0 ^ { - 1 1 }$ , see Fig. 11(b). As the parameters converge, the replaced snapshots stem from nearly the same material, and the snapshot matrix loses rank. The dimension of the trial space, however, is never decreased, since a smaller trial space would require another compilation, and it remains at $N _ { \mathrm { r } } = 2 7$

Identified material. The upper half of Fig. 13 evaluates the material identified from the force alone on the training specimen TT1. The machine curve is reproduced with a root-mean-square error of 1.80 N at a maximum force of 236.9 N, i.e., a relative error of 1.2 %, and the force parity yields $R ^ { 2 } = 0 . 9 9 9 2$ . For comparison, [40] report a force error of about 0.5 % for their full-order identification from the same specimen with displacements in the loss. The displacement field, which is a prediction in this run, agrees with the measurement with a relative error of 3.2 % and $R ^ { 2 } = 0 . 9 3 3$ in transverse and $R ^ { 2 } = 0 . 9 9 7$ in loading direction. The pointwise error stays below 6 % and concentrates at the right part of the specimen and around the largest hole. Adding the displacements to the loss changes this picture in a characteristic way, see Fig. C.1 in Section Appendix C. The displacement error decreases to 2.1 %, with $R ^ { 2 } = 0 . 9 9 1$ and $R ^ { 2 } = 0 . 9 9 8$ whereas the force error increases to 4.2 %, i.e., 6.47 N, with $R ^ { 2 } = 0 . 9 9 0$ . In particular, the identified material now underestimates the force at small and overestimates it at large displacements. Hence, the two measurements do not determine exactly the same material within the present model. We attribute this mainly to modeling assumptions, such as the nominal geometry of the holes, the thickness, and the prescribed edge displacements, which afect force and field diferently; a similar conclusion was drawn in [40].

![](images/698d31a77ccc2f56d7968209ac6451c75de46d50afe2efbd6331bf4243907c8b.jpg)

Figure 10: Case $^ { 2 , }$ training history of the force-based identification on TT1. (a) The loss decreases by three orders of magnitude within 100 epochs and reaches $\mathcal { L } = 6 . 0 \cdot 1 0 ^ { - 5 }$ . (b) The gradient norm shows short peaks at canary epochs, when the ECSW rule is updated, which decay within a few epochs. (c) Reduced epochs take about 11 s, canary epochs between 123 and 169 s. (d) The training requires 2.73 h in total. Grey dashed lines mark canary epochs and black dashed lines rebuilds.  
![](images/43531b90d714bd88dd97cce9123697c936f5bbeb7263b4601f5a00166c1d2b51.jpg)  
Figure 11: Case 2, adaptivity of the reduction in the force-based identification. (a) The snapshot matrix is full from the first bootstrap epoch on, since one epoch contains 51 states, and the trial space grows once from $N _ { \mathrm { r } } ~ = ~ 2 6$ to 27. (b) After epoch 300, the projection error of the snapshots drops to about $1 0 ^ { - 1 1 }$ , as the snapshots of the converging material become nearly linearly dependent. (c) ECSW requires 613 to 801 of the 1 518 elements, stored in a padded set of 1 024 entries. (d) The reduced epochs dominate the wall-clock time with 79 %.

![](images/9946e7369c291a102b0e8ea918f596bbb3a0bc2388b6e0adce64b369a3cf6bae.jpg)

![](images/b7b9843316607006869a4e1ea822264484e73d73cbd2fe752311cb6be19bed12.jpg)

![](images/33be78388bd9d2f5dd8f191e4cbe3529ca89c84f2e7f40f7fd976b6e37d14182.jpg)

![](images/70f2e070a22508b7aae60ebfc5fb1719f5bfd9a83d1a7170d604bd4ac0f73dd3.jpg)

![](images/229314171ccb87c09ef63bb087062e81dced33d6a021e5be516684c29b4af647.jpg)

![](images/6a40e04b0dcd96e3767309f448ebe8cfdf3a1bd3ce628b5f54a668673fb27499.jpg)

![](images/f36add634cdecfc7fd7dd8ce2eca1ca0803dca36ecc18d6331f76a760d866a39.jpg)  
Figure 12: Case 2, identification from force and displacements. Top: training history; the total loss (solid) is dominated by the displacement term, which levels of at about $1 . 1 \cdot 1 0 ^ { - 1 }$ , while the force term (dashed blue) falls below $1 0 ^ { - 3 }$ . The jump of the displacement term at the first canary epoch shows that the outdated trial space had underestimated the displacement misfit. Bottom: adaptivity of the reduction; the trial space grows from $N _ { \mathrm { r } } = 2 5 \ \mathrm { t o } \ 2 7$ at epochs 52 and 102, and ECSW requires up to 928 active elements. The total wall-clock time of 2.78 h is nearly identical to the force-based run.

The lower halves of Figs. C.1 and 13 apply both materials to the second specimen TT2 of the dataset, which has a diferent arrangement of holes and was not used for training. Both materials overestimate the force at large displacements by about 10 %, with root-mean-square errors of 16.4 N and 16.5 N and $R ^ { 2 } \approx 0 . 9 4$ . This overestimation is independent of whether the displacements entered the loss, and it agrees with the overestimation of about 9 % reported for TT1- trained models in [40], which were identified without any reduction. We therefore attribute it to diferences between both specimens rather than to the reduction. For the displacement field, in contrast, the additional information pays of: the material identified from both channels predicts the field of TT2 with $R ^ { 2 } = 0 . 9 9 8$ in both directions, compared with $R ^ { 2 } = 0 . 8 9 2$ and $R ^ { 2 } = 0 . 9 9 7$ for the force-based material.

At the last load step of the force-based run, the component $F _ { y y }$ of the deformation gradient ranges from 0.64 at the crowns of the holes, where the material is compressed in loading direction, to 3.94 in the vertical ligaments between the holes. A single test thus samples compression as well as large tension in loading direction.

Compressibility of the identified material. The energy of Section 2.3 does not enforce incompressibility, such that we can ask which volumetric stifness the identifications have found. To this end, we compare the initial bulk and shear moduli, which follow from the identified energy along a purely volumetric and a purely isochoric path [96],

$$
K = \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } \varepsilon ^ { 2 } } \left. \psi \big ( \exp ( \varepsilon / 3 ) I \big ) \right| _ { \varepsilon = 0 } , \qquad \mu = \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } \gamma ^ { 2 } } \left. \psi \big ( I + \gamma e _ { 1 } \otimes e _ { 2 } \big ) \right| _ { \gamma = 0 } ,\tag{43}
$$

where $\varepsilon = \ln J$ denotes the volumetric strain and $\gamma$ the shear. Both identifications yield nearly the same shear modulus, $\mu = 0 . 5 8 \mathrm { M P a }$ from the force and $\mu = 0 . 6 1$ MPa from force and displacements, but diferent bulk moduli of $K = 4 . 3 5$ MPa and $K = 1 . 1 6 ~ \mathrm { M P a }$ . This corresponds to Poisson ratios of 0.44 and 0.28, i.e., $K / \mu = 7 . 5$ and 1.9, whereas a nearly incompressible material would require $K / \mu$ of the order of $1 0 ^ { 3 }$ . Hence, neither identification supports the common assumption of incompressibility, and the displacement field mainly afects the volumetric response, presumably through the observed lateral contraction. We note, however, that the out-of-plane kinematics are only coarsely represented by one element across the thickness and the fixed out-of-plane displacement on the clamped edges

Trial space and hyperreduction. The spectrum of the final snapshot matrix in Fig. 14 drops by two orders of magnitude within the first two modes and then decays slowly over a range of almost twenty modes, before it falls sharply to $1 0 ^ { - 1 2 }$ . As in Case 1, the tolerance intersects this slowly decaying part, which explains why the strict choice $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 6 }$ results in a large trial space. The first mode represents the stretching in loading direction, and the higher modes resolve the motion of individual ligaments between the holes, i.e., the regions in which the deformation localizes. The final ECSW rule in Fig. 15 reflects the same localization. Of the 1 024 entries of the padded set, 801 refer to active elements, and the remaining 223 entries carry zero weight and add no element to the figure. The sampled element line both clamped edges and the boundaries of all holes, and only the interior of the wider ligaments is represented by few elements. Compared with Case 1, the weights remain small, with a maximum of about 17, since hardly any region deforms uniformly enough to be represented by a single heavily weighted element. In summary, the reduction remains accurate for a perforated specimen and both kinds of loss, yet its eficiency is limited by the localization of the deformation. In the last case, we turn to a problem in which the full-order solve dominates the cost.

## 3.3. Case 3: Heterogeneous material system with spherical inclusions

Our final example is a three-dimensional block with two stif spherical inclusions, for which we generate the data synthetically. This serves two purposes. First, the reference material is known, and the reference data are free of noise and of uncertainties in geometry and boundary conditions, such that any deviation stems from the identification and the reduction alone. Second, the deformation is genuinely three-dimensional, and the problem is about three times large than the plate of Case 1, such that the full-order solve becomes the dominant cost of the training.

(a) Machine curve  
![](images/bd4a970131f8b5068d0ee84fdfbd995220c5fcb9026e30a4b754ab398fa6b10f.jpg)

(b) Force parity  
![](images/9e399f9e0a7e9c279da58d131aa38386a621abf067880d85c3b81e5f07fe196c.jpg)

(c) Displacement parity  
![](images/1a7517c9c29f47236f9a5ade1f5ddfa3afed0bb9516b7e64ce08c8f2b4abfc93.jpg)

![](images/f0645a866bac8804c23cedc1db11fe92c5dce4abdd36a7d277c32314c75136cf.jpg)

![](images/731e4635d5160dfa1304b198c4314cb1d81b880e06485ea40702d17006e7c204.jpg)

![](images/ca7392ecb49d879668b5ed5254bdafad03d3f18db14424a0639a709f19a933ef.jpg)

(a) Machine curve  
![](images/9313a59e3b5c3b55b5642dd44a692595fa5d18471572567e3746c8ec27f3863d.jpg)

![](images/5e8025f2d1d7ac5429de8914fd51a2613cd06fb92bdbc7ba0a971ddc165e8d04.jpg)

(c) Displacement parity  
![](images/36521e1f30f38e4265996b5e38aad3b0d386b7aa489bea2226b8c20fa87f8cb5.jpg)

![](images/79046c3789652a3f3ced0e7fb2986c77bd137ca8e802ec9afa0bd4a80aa89567.jpg)

![](images/f4e634832151fced12cf5e1ecbbb5e2903ceaf82bbbe050c35b2bf142b73a231.jpg)

![](images/129ce7de9f26c9a086564bbde1aef10a24439359d6a22bf8695469df016e4bd4.jpg)  
Figure 13: Case 2, material identified from the force alone, evaluated by a full-order solve on the training specimen TT1 (top) and on the unseen specimen TT2 (bottom). Top: (a)–(b) The machine curve of TT1 is reproduced with a root-mean-square error of 1.80 N and $R ^ { 2 } = 0 . 9 9 9 2$ . (c) The displacements, which are not part of the loss, agree with $R _ { x } ^ { 2 } = 0 . 9 3 3$ and $R _ { y } ^ { 2 } = 0 . 9 9 7$ . (d)–(e) Measured (black) and discovered (red) contour lines nearly coincide in loading direction and deviate slightly in transverse direction. (f) The relative displacement error remains below 6 % and concentrates around the largest hole and at the right edge. Bottom: For TT2, the material overestimates the force at large displacements by about 10 % (root-mean-square error of 16.4 N), in line with the full-order results of [40], and predicts the displacement field with $R _ { x } ^ { 2 } = 0 . 8 9 2$ and $R _ { y } ^ { 2 } = 0 . 9 9 7$ . The corresponding results for the material identified from force and displacements are shown in Fig. C.1.

![](images/3667f01c6dd34b23d645fe1414a807e83e113ee42b84c154cd35b058c4bc931a.jpg)  
Figure 14: Case $^ { 2 , }$ trial space at the end of the force-based identification. Top left: normalized singular values $\hat { \sigma } _ { k }$ and projection error error<sub>r</sub>(k); the spectrum decays slowly over almost twenty modes, and the dimension $N _ { \mathrm { r } } = 2 7 .$ , set by $\bar { \varepsilon } _ { \mathrm { P O D } } = 1 0 ^ { - 6 }$ earlier in the training, is never decreased. Remaining panels: magnitude (color) and direction (arrows) of the first five modes; mode 1 describes the stretching in loading direction, while the higher modes resolve the motion of individual ligaments between the holes.

![](images/4ff3fc95b254c8840651d1e60ff2dec8520b349aa723eee542142641eddb34c9.jpg)  
Figure 15: Case 2, final ECSW rule of the force-based identification on the deformed mesh of TT1. The 801 active elements (53 % of the mesh) are colored by their weight, the remaining mesh is shown as wireframe. They line both clamped edges and the boundaries of all holes, where the deformation localizes. With weights of at most 17, the hyperreduction is considerably less sparse than for the plate of Case 1.

Fig. 16 shows the geometry. A cube with an edge length of 10 mm contains two spheres with radii $R _ { 1 } = 1 . 5$ mm and $R _ { 2 } = 2 . 0$ mm. We discretize the cube by $2 5 \times 2 5 \times 2 5$ hexahedral elements, $\mathrm { i . e . , } n _ { \mathrm { e l } } = 1 5 6 2 5$ elements with 52 728 degrees of freedom, of which $N _ { \mathrm { f } } = 4 8 6 7 2$ are free, and assign every element whose barycenter lies inside a sphere to the inclusion phase, which yields 734 inclusion elements. The bottom face is clamped, while the top face is clamped in the transverse directions and displaced by 3 mm in loading direction in $N _ { \mathrm { t } } = 1 0$ load steps, which corresponds to a nominal stretch of 1.3. The reference data stem from a full-order solve with the compressible Neo-Hookean energy

$$
\psi _ { \mathrm { N H } } ( F ) = \frac { \mu } { 2 } \big [ \mathrm { t r } C - 3 - 2 \ln J \big ] + \frac { \Lambda } { 4 } \big [ J ^ { 2 } - 1 - 2 \ln J \big ] ,\tag{44}
$$

where $\mu$ denotes the shear modulus and Λ the first Lamé parameter. We choose $\mu = 1$ MPa and $\Lambda = 4 ~ \mathrm { M P a }$ for the matrix and $\mu = 1 0$ MPa and $\Lambda = 4 0$ MPa for the inclusions, i.e., a stifness ratio of ten and a Poisson ratio of 0.4 in both phases. As in Case 1, each phase is represented by its own network, and only the total reaction force on the top face enters the loss, whose target we scale by $\bar { f } = \operatorname* { m a x } _ { n } | f _ { n } ^ { \mathrm { e x p } } | = 8 5 . 6 1 \mathrm { ~ N } .$

$$
\mathcal { L } = \frac { 1 } { { { N _ { \mathrm { t } } } } } \sum _ { n = 1 } ^ { { N _ { \mathrm { t } } } } { \left[ { \mathcal { H } ( { \bf { d } } _ { n } , \bf { \Theta } \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } \mathbf { ) } - \mathbf { f } _ { n } ^ { \mathrm { e x p } } / \bar { f } } \right] ^ { 2 } } ,\tag{45}
$$

where $f _ { n } ^ { \mathrm { e x p } }$ now denotes the reaction force of the reference solve at load step $t _ { n }$ . As in Case 1, the networks thus learn the scaled energy $\psi / { \bar { f } } .$ . The complete displacement field of the reference solve serves exclusively for validation. Note that the reference data are computed on the same mesh that is used for the identification. All further settings follow Case 1, see Table 1, except that we train for 400 epochs.

Training history and adaptivity of the reduction. Fig. 17(a) shows a nearly exponential decrease of the loss from about 60 to $1 0 ^ { - 1 }$ within the first 150 epochs, followed by a steep drop and decaying oscillations around epoch 190, and a slow decrease to $\mathcal { L } = 4 . 2 \cdot 1 0 ^ { - 5 }$ at epoch 400. In contrast to the experimental cases, a reduced epoch is now more than two orders of magnitude cheaper than a canary epoch, see Fig. $1 7 ( \mathrm { c } )$ . A reduced epoch takes between 0.17 and 0.3 s, a canary epoch about $4 0 \ \mathrm { s } ,$ and an epoch with a rebuild about 85 s. Consequently, the 391 reduced epochs amount to only 15 % of the total wall-clock time of 0.21 h, whereas the two bootstrap epochs account for 30 % and the seven canary epochs for 55 %, see Fig. 18(d). Since a canary epoch comprises a full-order solve and the update of the reduction, a full-order epoch costs at most about $^ { 4 0 \ \mathrm { s } , }$ and a full-order training over 400 epochs would take up to about four and a half hours, i.e., up to twenty times longer. Here, the speed-up is limited almost entirely by the full-order epochs that the adaptive strategy requires.

The adaptivity in Fig. 18 shows a second reason for a rebuild. Since each epoch comprises only ten load steps, the snapshot matrix fills slowly and reaches its capacity of 30 only at epoch 252. The trial space grows from four to five modes at epoch 102 and to six modes at epoch 302, in both cases because the projection error of the updated snapshots exceeds ε<sub>POD</sub> with the previous dimension, as in Case 1. At epoch 252, however, the dimension remains unchanged, and the rebuild is caused by the ECSW rule: the number of active elements grows from 92 to 101 and thus exceeds the padded capacity of 96 elements. The capacity is raised to the next size that provides the headroom introduced in Case 1, i.e., to 192 elements, which requires a recompilation. This rebuild also shows the price of the padding, since the cost of a reduced epoch increases from about 0.17 s to 0.3 s, although the number of active elements grows by only about ten percent. At the end of the training, 116 of the 15 625 elements are active, i.e., less than 1 % of the mesh.

Identified material. Fig. 19 compares the identified material with the reference. The force–displacement curve is reproduced with a root-mean-square error of 0.556 N at a maximum force of about 86 N, and the force parity yields $R ^ { 2 } = 0 . 9 9 9 5$ . The relative force error amounts to 1.0 %. The displacement field, which did not enter the loss, agrees with the reference with a relative error of 3.2 % over all nodes and load steps and with $R ^ { 2 } = 0 . 9 9 0$ in both transverse directions and $R ^ { 2 } = 0$ .9998 in loading direction. In the mid-plane $y = 5$ mm shown in Fig. 19(d)–(f), the pointwise error remains below $1 . 5 \%$ . In particular, the contour lines of the loading displacement in Fig. 19(e) follow the reference closely around both inclusions, which move almost rigidly within the softer matrix. Fig. $1 9 ( \mathrm { g } ) \mathrm { - ( h ) }$ shows the displacement in loading direction and the component $F _ { y y }$ on the deformed configuration. The stretch varies between $F _ { y y } = 1 . 0 4 4$ and 1.61, with the largest values in the vicinity of the clamped faces. Unlike that of the plate in Case 1, the force–displacement curve of the block is almost linear and samples only a moderate range of stretches. Nevertheless, the identified material reproduces the interior field of both phases, which indicates that the force sufices to separate both phases also in three dimensions.

![](images/91aca76eafd71530be4e597db1be6148d0c6f228439e5ed0a3eaaf854b888cfc.jpg)  
Figure 16: Case 3, cube with two spherical inclusions in the reference configuration. The cube of edge length 10 mm contains two spheres of radii $R _ { 1 } = 1 . 5$ mm and $R _ { 2 } = 2 . 0$ mm (green) and is discretized by 25<sup>3</sup> hexahedral elements. Top: three-dimensional view; the bottom face is clamped, and the top face is displaced by $\bar { u } _ { y } ( t )$ up to 3 mm in loading direction $_ y$ in $N _ { \mathrm { t } } = 1 0$ load steps, with the transverse displacements fixed. Bottom: front view (x–z), top view (x–y) and side view (y–z) with dimensions in mm. The inclusions are ten times stifer than the matrix.

Comparison with the reference material. Since the reference energies are known, we finally compare the identified and the reference energy of both phases at a single material point, see Fig. 20. We consider uniaxial tension, simple shear, and equibiaxial tension, given by

$$
F _ { \mathrm { u t } } = { \left[ \begin{array} { l l l } { \xi } & { 0 } & { 0 } \\ { 0 } & { \lambda } & { 0 } \\ { 0 } & { 0 } & { \xi } \end{array} \right] } , \qquad F _ { \mathrm { s s } } = { \left[ \begin{array} { l l l } { 1 } & { 0 } & { 0 } \\ { \gamma } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right] } , \qquad F _ { \mathrm { e b } } = { \left[ \begin{array} { l l l } { \lambda } & { 0 } & { 0 } \\ { 0 } & { \xi } & { 0 } \\ { 0 } & { 0 } & { \lambda } \end{array} \right] } ,\tag{46}
$$

where only $\lambda$ or $\gamma$ is prescribed and ξ follows from vanishing normal stresses. Each load case covers the range that the respective phase experiences in the reference solve. For the matrix, the identified energy coincides with the reference in uniaxial tension, whereas it is too stif in simple shear and, by a factor of about two, in equibiaxial tension. The inclusion deviates in all three load cases. This reflects what the experiment probes: the block is stretched essentially uniaxially, and the force constrains the energy only along this path, but not its extrapolation to other states, such as simple shear. Moreover, the force measures only the axial response and not the lateral contraction. Indeed, at the end of the uniaxial range, the identified matrix attains the reference energy with J = 1.03 instead of 1.08, i.e., it is les compressible than the reference. The volumetric response thus remains largely undetermined, which becomes apparent in equibiaxial tension, where the identified matrix expands to $J = 1 . 6 6$ instead of 1.18. The stif inclusions, in turn, deform little and contribute only weakly to the force, such that their energy is less well determined, in line with the observation that heterogeneous experiments do not automatically improve identifiability [40]. Note that this limitation concerns the information content of the data and not the reduction.

![](images/9580a42df1a5a1218158f6520032d38b057114cde41c70979e375e2162d089f8.jpg)  
Figure 17: Case 3, training history of the hyperreduced identification. (a) The loss decreases from about 60 to $\mathcal { L } = 4 . 2 \cdot 1 0 ^ { - 5 }$ , with a steep drop around epoch 190. (b) The gradient norm decreases from about $5 \cdot 1 0 ^ { 2 }$ to $1 . 1 \cdot 1 0 ^ { - 2 }$ . (c) Reduced epochs take $0 . 1 7 \mathrm { ~ t o ~ } 0 . 3 \mathrm { ~ s } ,$ canary epochs about 40 s and canary epochs with a rebuild about 85 s, i.e., a reduced epoch is more than two orders of magnitude cheaper than a canary epoch. (d) The training requires 0.21 h in total, most of which is spent in the full-order epochs.

Trial space and hyperreduction. In contrast to the experimental cases, the singular values in Fig. 21 decay steadily over twelve orders of magnitude without a plateau, and the tolerance $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ is reached with only six modes. We presume that the plateaus of Cases 1 and 2 stem from the measured boundary data, whose small irregularities enter every snapshot, whereas the boundary conditions of the synthetic block are smooth. The first mode represents the uniaxial stretching of the block with lateral contraction, and the higher modes describe the bulging of the lateral faces, the deformation at the edges of the clamped faces and the perturbation by the inclusions. The final ECSW rule in Fig. 22 selects 116 elements, of which 21 belong to the inclusions, stored in a padded set of 192 entries. The active elements cluster inside and around both inclusions and at the corners of the clamped faces. The matrix away from the inclusions, in contrast, is represented by a few elements with weights of up to about 1 200, each of which stands for a large region of nearly uniform deformation. With less than 1 % of the elements, this is the sparsest rule of all three cases. Together with the small trial space and the ten load steps, it explains why the reduced epochs of the largest problem are also the cheapest. In summary, the reduction pays of most where the full-order model is most expensive, while the accuracy of the identified material is limited by the information content of the data rather than by the reduction.

## 4. Discussion

Hyperreduction accelerates the discovery of neural materials without degrading the identified material. The central result of this work is that the adaptive combination of POD and ECSW lowers the cost of the adjointbased discovery by an order of magnitude, while the identified material, evaluated by a full-order solve, reproduces the structural response as accurately as a material identified with the full-order model. For the plate of Case 1, the reduction accelerates the median epoch by a factor of 27 and the entire training by a factor of 13 (Fig. 2(c)–(d)), while the force error of 2.5 % remains below that of the full-order reference (Fig. 4). Across all variations of the reduction parameters, the speed-up per epoch ranges from 8.6 to 60 without a systematic loss of accuracy (Fig. 8(b)). For the three-dimensional block of Case 3, a reduced epoch is more than two orders of magnitude cheaper than a canary epoch (Fig. 17(c)), from which we estimate that a full-order training would take up to twenty times longer. Since six modes and less than 1 % of the elements sufice (Figs. 21 and 22), the largest problem has the cheapest reduced epochs, i.e., the gain grows with the cost of the full-order model. This answers the question raised in the introduction: the reduced model remains accurate while the weights evolve, provided that it is adapted along the optimization path. In Case 1, it was accepted at every canary epoch, and the reduced and the full-order training follow nearly the same path and end at losses of comparable magnitude (Fig. 2(a)). The progressive construction of reduced models along the optimization path is established in PDE-constrained design optimization with few parameters [75, 74]; our results show that it carries over to neural materials, whose many weights and biases change at every update. Moreover, the reduction is complementary to diferentiable and GPU-accelerated full-order solvers [45, 40], which lower the cost of a single solve, but not the number of degrees of freedom that enter it. How much is gained, however, depends on two factors discussed next: the reducibility of the deformation and the cost of the full-order checks.

![](images/34ed8bac07e660f201df8cc32d6f95f526be2554120d8460f8832d2182a38b8e.jpg)  
Figure 18: Case $^ { 3 , }$ adaptivity of the reduction. (a) With ten load steps per epoch, the snapshot matrix reaches its capacity of 30 only at epoch 252; the trial space grows from $N _ { \mathrm { r } } ~ = ~ 4$ to 6. (b) At epochs 102 and 302, the POD error with the previous dimension exceeds $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ (squares). (c) At epoch 252, the active ECSW set exceeds the padded capacity of 96 elements, which is then raised to 192; this rebuild changes no dimension of the trial space. (d) Bootstrap and canary epochs account for 85 % of the wall-clock time, the 391 reduced epochs for 15 %.

Reducibility is governed by the localization of the deformation rather than by the size of the problem. The perforated membrane of Case 2 is the smallest of the three meshes, yet it requires the largest reduced model, with 27 modes and up to 53 % of the elements (Fig. 11), whereas the block of Case 3, with three times as many degrees of freedom as the plate of Case 1, requires the smallest. The decisive factor is the localization of the deformation. In Case 2, the stretch concentrates in the ligaments between the holes, where $F _ { y y }$ reaches 3.94, the higher modes resolve individual ligaments, and the singular values decay slowly (Fig. 14). Since ECSW reproduces $N _ { \mathrm { r } } + N _ { \mathrm { r } } ^ { 2 }$ projected quantities per snapshot, see $\operatorname { E q . }$ (23), a large trial space directly enlarges the sampled set, and hardly any region deforms uniformly enough to be represented by a single heavily weighted element (Fig. 15). In addition, the measured boundary data presumably cause the plateaus in the spectra of Cases 1 and 2, which are absent for the synthetic block (Figs. 5 and 21), and Case 2 employs the stricter tolerance $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 6 }$ , which afects the size of the reduced model more than any other parameter in the study of Case 1 (Fig. 7(e)). Consequently, the update of the reduction dominates the canary epochs of Case 2, and without a full-order reference, the speed-up for this specimen remains open. Case 2 also shows that a loss which observes the displacement field reacts more sensitively to an outdated trial space than a loss which observes only the reaction force (Fig. 12). These observations reflect the known limitation of linear subspaces for localized deformation [48]. Nonlinear manifolds [68, 69, 70], hyperreduction in strain space [55, 66], and component-wise reduced models [67] ofer possible routes, whose combination with the adjoint-based discovery remains to be investigated.

![](images/c1441bfc3a45ae296adf9361e6437b129b2f6b85edfb09e146e870bd8db05a20.jpg)

![](images/8f5ed781b0989c72a9ab632a9c9cf1358e6e23023da83a524c97723f5beb90a4.jpg)

![](images/0ab01e6c3d0da5136774d1201a9b9d50571cfa914ba0caf2362bdffc3a74f9cb.jpg)

![](images/a6c18c41de6938db5fea6549942a6be202d8db281aa2a0f6b6fca5612e33a248.jpg)

![](images/2f881398a0d7d98da97bf94080c238e10b91cd156de8e325531e670eff2fe937.jpg)

![](images/5e06b064c5568a2c42033b95a2d81f67bb7538148237a2ed9bbfb48cd4f6666f.jpg)

![](images/82c3f059a2611f328c380d9f297f049d612eb31580a90788fd3b8d23ed606a08.jpg)  
(g) Displacement $u _ { y }$

![](images/861abbfea32eaf9d9a0fcdb471899add7e5e930d9b8183101e6cbfc7543516a2.jpg)  
(h) Deformation gradient $F _ { y y }$  
Figure 19: Case 3, identified material compared with the synthetic reference. (a)–(b) The force is reproduced with a root-meansquare error of 0.556 N and $R ^ { 2 } = 0 . 9 9 9 5$ . (c) All three displacement components, which are not part of the loss, agree with $R ^ { 2 } \ge 0 . 9 9 0$ $\mathrm { ( d ) - ( e ) }$ Reference (black) and discovered (red) contour lines of the transverse and loading displacement in the mid-plane $y = 5$ mm nearly coincide, also around both inclusions. (f) The relative displacement error remains below 1.5 %. $\mathrm { { ( g ) - ( h ) } }$ Deformed configuration at the last load step: the displacement in loading direction increases from the clamped bottom face to the top face, which is displaced by 3 mm, and $F _ { y y }$ varies between 1.044 and 1.61 on the surface, with the largest stretches in the vicinity of the clamped faces.

![](images/f8ef953d4bde2482919f048bffc02be3e4f764a5b26fee2fc091e277241245a2.jpg)

![](images/421973fd4ed4634d96a6da9f840b6bc446bc06ef90f39ebf3eb7a903f9b51a6c.jpg)

![](images/fcf39705cd67605c7a9b68bffc4aeb8bed0a510bd336c92047dc674995637888.jpg)

![](images/ba691bd09c86a6b1d06e424e91736382667a8e1c7f5928ef7b5a4fa3b8bdb8ae.jpg)

![](images/d54a775ecf864af195ea8bc569d6387d50eb2d9bb3580edaceeebb149f25c994.jpg)

![](images/eb8a102a4ec25f438e1d896eeb90929f1f04b82b16df193e07c985c3ff4c60e9.jpg)

Figure 20: Case 3, energy density of the identified (red) and the reference Neo-Hookean material Eq. (44) (black) at a single material point. Top: matrix; bottom: inclusion; columns: uniaxial tension, simple shear and equibiaxial tension, each over the range the phase experiences in the reference solve, with the deformation gradients Eq. (46); in tension, the unprescribed diagonal components follow from vanishing normal stresses. The matrix law is reproduced in uniaxial tension (root-mean-square error of 0.004 MPa), which dominates the response of the block, whereas it is too stif in simple shear (0.048 MPa) and in equibiaxial tension (0.44 MPa), which the experiment hardly activates. The stifer and less deformed inclusion is identified less accurately in all three load cases (0.013, 0.073 and 0.069 MPa).  
![](images/f270b019e11a7a825770ec6cc210f3e44a1ef2de78f4d22b1c8c7f0356af08e5.jpg)

![](images/8f5e3b7b450c257d4494cd44040b6f5498212f9eaf294a419918cb232729f1e0.jpg)  
Figure 21: Case $^ { 3 , }$ trial space at the end of the training. Top left: normalized singular values $\hat { \sigma } _ { k }$ and projection error error $\mathbf { \nabla } _ { \mathbf { \cdot } } ( k ) ;$ in contrast to the experimental cases, the spectrum decays steadily, and $\varepsilon _ { \mathrm { P O D } } = 1 0 ^ { - 4 }$ is reached with $N _ { \mathrm { r } } = 6$ modes. Remaining panels: magnitude of the first five modes on the deformed block; mode 1 describes the uniaxial stretching, the higher modes the bulging of the lateral faces and the deformation at the edges of the clamped faces.

![](images/bd93ba5ce2c778711d58cd8c7dc0616472d755c7302bb7eb08248162c140d5a2.jpg)  
Figure 22: Case 3, final ECSW rule. The mesh is shown as wireframe and the inclusions in green. Only 116 of the 15 625 elements carry a positive weight. They cluster inside and around both inclusions and at the corners of the clamped faces, whereas the matrix is represented by a few elements with weights of up to about 1 200. The 76 padded entries carry zero weight and add no element to the figure.

Up-to-date snapshots govern the accuracy, full-order checks govern the cost. The parameter study of Case 1 separates the parameters of the reduction into two groups: the canary interval and the snapshot tolerance, which control how up to date the snapshots are, determine the accuracy, whereas the tolerances of POD and ECSW determine the cost (Fig. 7). The model selection was never triggered and acts as a safeguard. Once the reduced solve is cheap, the full-order checks dominate the cost: the canary epochs of Case 1 make up less than 2 % of all epochs, but 41 % of the wall-clock time (Fig. 3(d)), and the bootstrap and canary epochs of Case 3 even 85 % (Fig. 18(d)). Even if the reduced epochs were free, the training of Case 1 could thus be accelerated by at most a factor of about 28 for $N _ { \mathrm { c a n a r y } } = 5 0$ Recompilations add to this cost whenever a dimension of the reduced model changes, which the padded ECSW set avoids at most, but not all canary epochs (Fig. 3(c), Fig. 18(c)). Two directions follow. First, the canary interval could be chosen adaptively, since the weights change less as the training converges, cf. Section 2.4. Second, the full-order check could be complemented or replaced by an error indicator of the reduced solution, in the spirit of adaptive hyperreduction [51] or of estimators based on the constitutive relation error [97, 79]. Both would spend the expensive full-order solves where the reduced model is actually at risk.

The reaction force fixes the stifness level, but not the material uniquely. Since all identified materials are evaluated by the same full-order solve, their remaining deviations reflect the information content of the data rather than the reduction. Three observations support this view. First, in Case 1, the reduced and the full-order training reproduce the force comparably well, but distribute the stifness diferently between the phases, which only the displacement field reveals (Fig. 4). Second, in Case 2, adding the displacements to the loss trades accuracy of the force for accuracy of the field (Figs. C.1 and 13, top), improves the predicted field of the unseen specimen (Figs. C.1 and 13, bottom), and changes mainly the bulk modulus, which the force alone barely constrains. Third, in Case 3, the identified matrix coincides with the reference along the uniaxial path that the block probes, whereas its response in simple shear and equibiaxial tension, as well as the stif inclusions, remain less well determined (Fig. 20). All three follow from the scaling argument of Section 3.1: the force fixes the level of the response, the displacement field its spatial distribution, and both constrain the energy only along the activated deformation states. This agrees with the observations that the coverage of deformation states governs the generalization of the identified model [18] and that heterogeneous specimens do not automatically improve identifiability [40], see also [16]. In particular, neither identification of Case 2 supports the incompressibility commonly assumed for this material [25, 40], although the coarse out-of-plane kinematics call for caution. A natural remedy is the combination of several specimens and load cases within a single loss [18, 39]. Since this multiplies the number of nonlinear solves per epoch, it is precisely the setting in which the reduction pays of.

Extension to anisotropy and inelasticity. In the present work, we restricted ourselves to isotropic hyperelasticity. For anisotropic hyperelastic materials, the formulation of Section 2 carries over directly: the energy additionally depends on structural tensors [4, 9], the inner problem remains a minimization, ECSW still approximates a weighted sum of element energies, and the reduced tangent remains symmetric. Recent works on the discovery of anisotropic materials [33, 98, 99] and on reduced models for anisotropic problems [71] provide a starting point. Two aspects require attention. First, spatially varying preferred directions act like additional phases: as in Case 1, the sampled set must contain elements of every phase, since otherwise the gradient Eq. (20) does not depend on the corresponding parameters (Fig. 6). Second, additional invariants increase the number of parameters that the data have to determine, which aggravates the nonuniqueness discussed above. Inelasticity poses a more fundamental challenge. The load steps no longer decouple, see Section 2.1, but are connected by the evolution of internal variables, such that the adjoint problem has to be solved backward along the loading path. Moreover, ECSW requires the history only at the sampled elements; once the rule changes at a canary epoch, the history of newly sampled elements is unknown and has to be reconstructed, e.g., from the preceding full-order trajectory. Incremental variational formulations, in which each load step minimizes an incremental potential composed of the energy and a dissipation potential [8, 5], preserve the minimization structure on which the present reduction relies. Hyperreduction has been applied to forward simulations of viscoelastic, elastoplastic, and damaging solids [61, 62, 63, 64], and neural materials are available for viscoelasticity [6, 7, 10], plasticity [100, 101], anisotropic inelasticity [9], and thermomechanics [14]. Their discovery from structural data has so far relied on full-orde solvers or on local equilibrium evaluations [35, 31, 17, 41, 42, 44]. Whether the adaptive reduction remains reliable when the trial space has to represent path-dependent states, and how auxiliary networks for the internal variables [9] interact with the hyperreduction, remains open for future work.

## 5. Conclusion

This study shows that hyperreduction makes finite element model updating of neural materials computationally afordable, while preserving the advantages that distinguish it from residual-based approaches: it accommodates arbitrary geometries and boundary conditions, tolerates sparse and noisy data, and does not require full-field measurements. The approach embeds a polyconvex neural network into a finite element model, projects the equilibrium onto a POD subspace, and approximates the internal energy by a weighted sum over a small subset of elements. Adjoint sensitivities of the reduced model provide the exact gradient with respect to the weights and biases, while bootstrap and canary epochs adapt the reduction to the evolving network along the optimization path.

The numerical investigations on two experimental and one synthetic example demonstrate that the reduction lowers the cost of the training by an order of magnitude without degrading the identified material. In particular, the reduced model remains accurate while the weights evolve, and its gain grows with the size of the full-order problem. Moreover, a single reaction force sufices to separate the phases of heterogeneous specimens and to predict displacement fields that never entered the loss.

At the same time, the study highlights two limitations. The eficiency of the reduction decreases when the deformation localizes, and the full-order checks bound the overall speed-up. In addition, the accuracy of the identified material is limited by the information content of the data rather than by the reduction, since the reaction force constrains the energy only along the deformation states activated by the experiment.

Future work may focus on adaptive full-order checks, nonlinear trial manifolds for localized deformation, and the extension to anisotropic and inelastic materials.

## Acknowledgements

Hagen Holthusen gratefully acknowledges financial support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the projects 495926269 and 417002380. Kevin Linka gratefully acknowledges financial support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the project 517243167.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used Claude (Anthropic) to draft and revise text passages, to improve language and readability, and to assist in the implementation and documentation of the source code. All generated text and code were reviewed, verified, and edited by the authors, including all statements, numerical values, and references, such that the manuscript reflects the authors’ own analysis and conclusions. The authors take full responsibility for the content of the published article.

## Data availability

The data generated in this study will be made publicly available upon publication. The experimental data of Cases 1 and 2 are openly available from [92, 93] and [25], respectively.

## Appendix A. Polyconvexity and coercivity of the energy

The energy consists of the neural network ${ \mathcal { N } } ,$ a stress normalization $\psi _ { \mathrm { n r m } }$ and a growth term $\psi _ { \mathrm { g r } }$

$$
\psi ( \pmb { F } ) = \mathcal { N } ( \pmb { x } ) + \psi _ { \mathrm { n r m } } ( \pmb { J } ) + \psi _ { \mathrm { g r } } ( \pmb { F } ) , \qquad \pmb { x } = \big [ \bar { I } _ { 1 } , \bar { I } _ { 2 } , \bar { J } , - \bar { J } \big ] ,\tag{A.1}
$$

with the normalized invariants

$$
\begin{array} { r } { I _ { 1 } = \frac 1 3 \operatorname { t r } C = \frac 1 3 \| F \| ^ { 2 } , \qquad I _ { 2 } = \frac 1 3 \operatorname { t r } ( \cot C ) = \frac 1 3 \| \cot F \| ^ { 2 } , \qquad J = \operatorname* { d e t } F > 0 , } \end{array}\tag{A.2}
$$

and $\bar { I } _ { 1 } = I _ { 1 } - 1 , \bar { I } _ { 2 } = I _ { 2 } - 1 , \bar { J } = J - 1$ , such that ${ \pmb x } = { \bf 0 }$ at ${ \pmb { F } } = { \pmb { I } }$ . The network is designed as an Input Convex Neural Network (ICNN) [85], which is additionally zero-anchored $\mathcal { N } ( \mathbf { 0 } ) = 0 \ \left[ 9 \right]$ . As an ICNN, it has non-negative weights and convex and non-decreasing activations. Thus, it is therefore convex and non-decreasing in each input. Its derivatives at the reference configuration are denoted by

$$
a _ { 1 } = \left. \frac { \partial \mathcal { N } } { \partial \bar { I } _ { 1 } } \right| _ { F = I } \geq 0 , \quad a _ { 2 } = \left. \frac { \partial \mathcal { N } } { \partial \bar { I } _ { 2 } } \right| _ { F = I } \geq 0 , \quad a _ { J } = \left. \frac { \mathrm { d } } { \mathrm { d } \bar { J } } \mathcal { N } \big ( 0 , 0 , \bar { J } , - \bar { J } \big ) \right| _ { \bar { J } = 0 } \in \mathbb { R } .\tag{A.3}
$$

Since the network is non-decreasing in each input, $a _ { 1 } , a _ { 2 } \geq 0$ . By the chain rule, $a _ { J }$ is the derivative of $\mathcal { N }$ with respect to its third input minus that with respect to its fourth input. As a diference of two non-negative numbers, it may take either sign, which allows for both tensile and compressive volumetric stresses.

Normalizations and growth term. We compare two stress normalizations,

$$
\begin{array} { r } { \psi _ { \mathrm { n r m } } ^ { \mathrm { l i n } } = - \left[ \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } + a _ { J } \right] \bar { J } , \qquad \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } = - \left[ \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } \right] \ln J - a _ { J } \bar { J } , } \end{array}\tag{A.4}
$$

where the former is in line with [86]. Further, we employ the growth term with $\epsilon > 0$

$$
\begin{array} { r } { \psi _ { \mathrm { g r } } = \epsilon \Big [ \big [ I _ { 1 } - 1 - { \textstyle \frac { 2 } { 3 } } \ln J \big ] + \big [ I _ { 2 } - 1 - { \textstyle \frac { 4 } { 3 } } \ln J \big ] + \big [ J - 1 - \ln J \big ] \Big ] = \epsilon \big [ I _ { 1 } + I _ { 2 } + J - 3 - 3 \ln J \big ] . } \end{array}\tag{A.5}
$$

All three terms within this growth energy vanish at ${ \boldsymbol { F } } = { \boldsymbol { I } }$ . With $\begin{array} { r } { \partial _ { F } I _ { 1 } = \frac { 2 } { 3 } I , \partial _ { F } I _ { 2 } = \frac { 4 } { 3 } I } \end{array}$ and $\partial _ { F } J = \partial _ { F }$ ln $J = \pmb { I }$ at $\mathbf { \nabla } _ { F } = I , $ the Piola stress of each contribution at the reference configuration reads

$$
\begin{array} { r l } & { \quad \partial _ { F } {  { \mathcal N } } = \left( \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } + a _ { J } \right)  { \boldsymbol I } , \qquad \partial _ { F }  { \psi _ { \mathrm { g r } } } = { \bf 0 } , } \\ & { \quad \partial _ { F }  { \psi _ { \mathrm { n r m } } } ^ { \mathrm { l i n } } = \partial _ { F }  { \psi _ { \mathrm { n r m } } } ^ { \mathrm { l n } } = - \left( \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } + a _ { J } \right)  { \boldsymbol I } . \qquad } \end{array}\tag{A.6}
$$

Hence $\psi ( I ) = 0$ and $P ( I ) = \mathbf { 0 } $ for both normalizations. In the following, we first show polyconvexity and then coercivity.   
In both parts, network, normalization and growth term are treated one after another.

## Appendix A.1. Polyconvexity

An energy is polyconvex if it is a convex function of $F ,$ cof $\pmb { F }$ and det F [81]. As a sum of convex functions is convex, each contribution is checked separately. By Eq. (A.2), $I _ { 1 }$ is convex in $F , I _ { 2 }$ is convex in cof $\pmb { F }$ , and $J = \operatorname* { d e t } { \boldsymbol { F } }$ is itself an argument.

Neural network. The argument rests on a simple composition rule: a function that is convex and non-decreasing in each argument, applied to convex functions, is again convex [85, 102]. Every layer computes $h _ { \ell } = \varphi _ { \ell } ( W _ { \ell } h _ { \ell - 1 } +$ ${ S _ { \ell } } { x } + { b _ { \ell } } ) - \varphi _ { \ell } ( { b _ { \ell } } )$ with non-negative $W _ { \ell } , \ S _ { \ell }$ and convex, non-decreasing $\varphi _ { \ell } .$ . The argument of $\varphi _ { 1 }$ is afine in x with non-negative weights, and hence convex and non-decreasing in x, which carries over to $\boldsymbol { h } _ { 1 }$ . In every further layer, the argument is a non-negative combination of the entries of $\pmb { h } _ { \ell - 1 }$ and $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } }$ so the same reasoning applies. Thus, N is convex and non-decreasing in each entry of ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ where the anchoring term $\varphi _ { \ell } ( b _ { \ell } )$ is constant in x. Finally, $I _ { 1 } - 1$ and $I _ { 2 } - 1$ are convex in F and cof $F _ { \mathrm { { ; } } }$ , respectively, so the composition rule applies once more, while $\pm [ J - 1 ]$ are afine in $^ { J , }$ for which convexity of $\mathcal { N }$ alone sufices. Hence, $\mathcal { N } ( \pmb { x } ( \pmb { F } ) ; \pmb { \theta } )$ is convex in $( F ,$ cof $F , J )$ , i.e., polyconvex.

Normalization. As functions of J = det $F ,$ the two normalizations satisfy

$$
\frac { \mathrm { d } ^ { 2 } \psi _ { \mathrm { n r m } } ^ { \mathrm { l i n } } } { \mathrm { d } J ^ { 2 } } = 0 , \qquad \frac { \mathrm { d } ^ { 2 } \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } } { \mathrm { d } J ^ { 2 } } = \frac { \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } } { J ^ { 2 } } \geq 0 .\tag{A.7}
$$

The linear normalization is afine and thus convex for any sign of its coeficient. The logarithmic normalization is convex because $a _ { 1 } , a _ { 2 } \geq 0$ by $\mathrm { E q . \ ( A . 3 ) }$ ; here, the monotonicity of the network is essential.

Growth term. With $\epsilon > 0 , \psi _ { \mathrm { g r } }$ is a sum of the convex terms $I _ { 1 } , I _ { 2 } ,$ J and −3 ln J.

Consequently, the overall energy (A.1) is polyconvex for both normalizations.

## Appendix A.2. Coercivity

Before we investigate the coercivity by a growth condition, let us introduce two essential inequalities. Let $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } > 0$ be the singular values of F, so that $J = { \lambda } _ { 1 } { \lambda } _ { 2 } { \lambda } _ { 3 }$ . By the arithmetic-geometric mean, we obtain

$$
\begin{array} { r l } & { I _ { 1 } = \frac { 1 } { 3 } \bigl ( \lambda _ { 1 } ^ { 2 } + \lambda _ { 2 } ^ { 2 } + \lambda _ { 3 } ^ { 2 } \bigr ) \geq \bigl ( \lambda _ { 1 } ^ { 2 } \lambda _ { 2 } ^ { 2 } \lambda _ { 3 } ^ { 2 } \bigr ) ^ { 1 / 3 } = J ^ { 2 / 3 } , } \\ & { I _ { 2 } = \frac { 1 } { 3 } \bigl ( \lambda _ { 1 } ^ { 2 } \lambda _ { 2 } ^ { 2 } + \lambda _ { 2 } ^ { 2 } \lambda _ { 3 } ^ { 2 } + \lambda _ { 3 } ^ { 2 } \lambda _ { 1 } ^ { 2 } \bigr ) \geq \bigl ( \lambda _ { 1 } ^ { 4 } \lambda _ { 2 } ^ { 4 } \lambda _ { 3 } ^ { 4 } \bigr ) ^ { 1 / 3 } = J ^ { 4 / 3 } . } \end{array}\tag{A.8}
$$

Moreover, the function $y ( x ) = x - 1 -$ ln x attains its minimum zero at $x = 1$ , since its derivative $1 - 1 / x$ changes sign there. Thus

$$
y ( x ) = x - 1 - \ln x \geq 0 \qquad \forall x > 0 .\tag{A.9}
$$

Coercivity and existence of minimizers. Existence of minimizers follows from polyconvexity of $\psi$ together with a coercivity estimate of the form, cf. [81, 103],

$$
\psi ( { \pmb F } ) \ge c _ { 1 } \big ( \| { \pmb F } \| ^ { p } + \| \cot { \pmb F } \| ^ { q } + J ^ { r } \big ) + c _ { 0 } , \qquad p \ge 2 , \quad q \ge \frac { p } { p - 1 } , \quad r > 1 ,\tag{A.10}
$$

with $c _ { 1 } > 0$ , and the condition $\psi \to \infty$ as $J \to 0 ^ { + }$ . We will show the simpler bound

$$
\psi \ge c \left( I _ { 1 } + I _ { 2 } \right) + c _ { 0 }\tag{A.11}
$$

with $c > 0$ . This is suficient, since with Eq. (A.2) and $J ^ { 4 / 3 } \le I _ { 2 }$ from Eq. (A.8),

$$
\begin{array} { r } { c \left( I _ { 1 } + I _ { 2 } \right) = \frac { c } { 3 } \| \pmb { F } \| ^ { 2 } + \frac { c } { 6 } \| \cot \pmb { F } \| ^ { 2 } + \frac { c } { 2 } I _ { 2 } \geq \frac { c } { 6 } \big ( \| \pmb { F } \| ^ { 2 } + \| \cot \pmb { F } \| ^ { 2 } + J ^ { 4 / 3 } \big ) , } \end{array}\tag{A.12}
$$

which is Eq. (A.10) with $p = q = 2$ and $r = 4 / 3$

Growth term. We first show that each bracket in $\mathrm { E q . \ ( A . 5 ) }$ is non-negative and, for the first two, bounded below by half the respective invariant. For the first bracket we use $I _ { 1 } \geq J ^ { 2 / 3 }$ from Eq. (A.8) and Eq. (A.9) with $x = J ^ { 2 / 3 }$ , i.e., $x - 1 - \ln x \ge 0$

$$
\begin{array} { r } { I _ { 1 } - 1 - \frac { 2 } { 3 } \ln J = \left( J ^ { 2 / 3 } - 1 - \ln J ^ { 2 / 3 } \right) + \left( I _ { 1 } - J ^ { 2 / 3 } \right) \ge 0 . } \end{array}\tag{A.13}
$$

Using $I _ { 1 } \geq J ^ { 2 / 3 }$ for only one half of $I _ { 1 }$ and $\frac { 1 } { 2 } x - \ln x \geq 1 - \ln 2 > 0$ for all $x > 0$ (minimum at $x = 2 )$ , we further obtain

$$
\begin{array} { r } { I _ { 1 } - 1 - \frac { 2 } { 3 } \ln J \ge \frac { 1 } { 2 } I _ { 1 } + \left( \frac { 1 } { 2 } J ^ { 2 / 3 } - \ln J ^ { 2 / 3 } \right) - 1 \ge \frac { 1 } { 2 } I _ { 1 } - 1 . } \end{array}\tag{A.14}
$$

The second bracket follows in the same way, i.e. $\begin{array} { r } { I _ { 2 } - 1 - \frac { 4 } { 3 } \ln J \geq \frac { 1 } { 2 } I _ { 2 } - 1 } \end{array}$ , and the third, $J - 1 - \ln J ,$ , is non-negative by $\mathrm { E q . \ ( A . 9 ) }$ and tends to infinity for $J \to 0 ^ { + }$ . Summing the three brackets, with the first two bounded by $\operatorname { E q }$ . (A.14) and the third kept as it is, yields

$$
\begin{array} { r } { \psi _ { \mathrm { g r } } \geq \frac { \epsilon } { 2 } \left( I _ { 1 } + I _ { 2 } \right) - 2 \epsilon + \epsilon \left( J - 1 - \ln J \right) . } \end{array}\tag{A.15}
$$

The first term provides the required growth, and the last one is non-negative and tends to infinity for $J  0 ^ { + }$ , so the growth term alone satisfies the desired bound. It remains to show that network and normalization together do not destroy it, i.e., that their sum is bounded from below.

Neural Network. From the convexity of ${ \mathcal { N } } ,$ cf. [102], we obtain

$$
\begin{array} { r } { \mathcal { N } ( { \pmb x } ) \geq \mathcal { N } ( { \pmb 0 } ) + \nabla \mathcal { N } ( { \pmb 0 } ) \cdot { \pmb x } = a _ { 1 } \bar { I } _ { 1 } + a _ { 2 } \bar { I } _ { 2 } + a _ { J } \bar { J } , } \end{array}\tag{A.16}
$$

since $\mathcal { N } ( \mathbf { 0 } ) = 0$ . This bound is sharp, as equality holds for a network that consists of the output skip connection only, i.e. linear in x. A trained network can therefore be (almost) linear, so we cannot rely on its curvature to provide additional growth. The coercivity of ψ must instead come from the explicit growth term $\psi _ { \mathrm { g r } }$

Linear normalization. Adding $\psi _ { \mathrm { n r m } } ^ { \mathrm { l i n } }$ to Eq. (A.16), the terms with $a _ { J }$ cancel and the remaining terms can be grouped by the weights $a _ { 1 }$ and $a _ { 2 } .$

$$
\begin{array} { r } { \mathcal { N } + \psi _ { \mathrm { n r m } } ^ { \mathrm { l i n } } \geq a _ { 1 } \big [ \bar { I } _ { 1 } - \frac { 2 } { 3 } \bar { J } \big ] + a _ { 2 } \big [ \bar { I } _ { 2 } - \frac { 4 } { 3 } \bar { J } \big ] . } \end{array}\tag{A.17}
$$

Both brackets contain a term that is linear in J, so the question is whether $I _ { 1 }$ and $I _ { 2 }$ grow fast enough to compensate it. For the second bracket, $I _ { 2 } \geq J ^ { 4 / 3 }$ from Eq. (A.8) gives

$$
\begin{array} { r } { \bar { I } _ { 2 } - \frac { 4 } { 3 } \bar { J } \ge J ^ { 4 / 3 } - 1 - \frac { 4 } { 3 } ( J - 1 ) \ge 0 . } \end{array}\tag{A.18}
$$

For the first bracket, Eq. (A.8) only provides $I _ { 1 } \geq J ^ { 2 / 3 }$ , i.e., $J \le I _ { 1 } ^ { 3 / 2 }$ , so J may grow faster than I<sub>1</sub>. A pure dilatation $F = \lambda I$ with $I _ { 1 } = \lambda ^ { 2 }$ and $J = \lambda ^ { 3 }$ shows that this happens,

$$
{ \bar { I } } _ { 1 } - { \textstyle \frac { 2 } { 3 } } { \bar { J } } = I _ { 1 } - { \textstyle \frac { 1 } { 3 } } - { \textstyle \frac { 2 } { 3 } } J = \lambda ^ { 2 } - { \textstyle \frac { 1 } { 3 } } - { \textstyle \frac { 2 } { 3 } } \lambda ^ { 3 } \stackrel { \lambda \to \infty } { \longrightarrow } - \infty .\tag{A.19}
$$

Network and linear normalization together are therefore not bounded from below, and the growth term has to provide the missing coercivity. We add $\psi _ { \mathrm { g r } }$ , use $a _ { 1 } , a _ { 2 } \geq 0$ , drop the non-negative second bracket and the non-negative term $a _ { 1 } I _ { 1 }$ of the first one, and bound $\psi _ { \mathrm { g r } }$ by Eq. (A.15) without the non-negative term $\epsilon \left( J - 1 - \ln J \right)$ ,

$$
\begin{array} { r } { \psi \ge \frac { \epsilon } { 2 } \left( I _ { 1 } + I _ { 2 } \right) - 2 \epsilon + a _ { 1 } \left[ I _ { 1 } - \frac { 1 } { 3 } - \frac { 2 } { 3 } J \right] \ge \frac { \epsilon } { 2 } \left( I _ { 1 } + I _ { 2 } \right) - 2 \epsilon - \frac { a _ { 1 } } { 3 } - \frac { 2 } { 3 } a _ { 1 } J . } \end{array}\tag{A.20}
$$

The only negative contribution is $- { \textstyle { \frac { 2 } { 3 } } } a _ { 1 } J$ . Against it, we have the growth term $\begin{array} { r } { \frac { \epsilon } { 2 } I _ { 2 } \geq \frac { \epsilon } { 2 } J ^ { 4 / 3 } } \end{array}$ , which grows faster than J. We reserve one half of it, $\scriptstyle { \frac { \epsilon } { 4 } } I _ { 2 }$ , for this purpose and minimize the resulting expression over $J ,$

$$
\begin{array} { r } { g ( J ) = \frac { \epsilon } { 4 } J ^ { 4 / 3 } - \frac { 2 } { 3 } a _ { 1 } J , \qquad g ^ { \prime } ( J ) = 0 \Rightarrow J ^ { 1 / 3 } = \frac { 2 a _ { 1 } } { \epsilon } , \qquad g \geq - \frac { 4 a _ { 1 } ^ { 4 } } { 3 \epsilon ^ { 3 } } . } \end{array}\tag{A.21}
$$

The remaining terms in Eq. (A.20) are combined via $\begin{array} { r } { \frac { \epsilon } { 2 } I _ { 1 } + \frac { \epsilon } { 4 } I _ { 2 } \geq \frac { \epsilon } { 4 } ( I _ { 1 } + I _ { 2 } ) } \end{array}$ . Together with Eq. (A.21), this yields

$$
\psi \ge \frac { \epsilon } { 4 } \left( I _ { 1 } + I _ { 2 } \right) - C _ { \mathrm { l i n } } , \qquad C _ { \mathrm { l i n } } = \frac { 4 a _ { 1 } ^ { 4 } } { 3 \epsilon ^ { 3 } } + \frac { a _ { 1 } } { 3 } + 2 \epsilon .\tag{A.22}
$$

The energy is thus coercive, but the constant grows with $\epsilon ^ { - 3 }$ and becomes very large for a small growth term.

Logarithmic normalization. Adding $\psi _ { \mathrm { n r m } } ^ { \mathrm { l n } }$ to Eq. (A.16), the terms with $a _ { J }$ again cancel,

$$
\begin{array} { r l } & { \mathcal { N } + \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } \ \geq \ a _ { 1 } \bar { I } _ { 1 } + a _ { 2 } \bar { I } _ { 2 } + a _ { J } \bar { J } - \left( \frac { 2 } { 3 } a _ { 1 } + \frac { 4 } { 3 } a _ { 2 } \right) \ln J - a _ { J } \bar { J } } \\ & { } \\ { \quad \quad = \ a _ { 1 } \big [ I _ { 1 } - 1 - \frac { 2 } { 3 } \ln J \big ] + a _ { 2 } \big [ I _ { 2 } - 1 - \frac { 4 } { 3 } \ln J \big ] . } \end{array}\tag{A.23}
$$

These are exactly the first two brackets of the growth term, which are non-negative by Eq. (A.13). Since also $a _ { 1 } , a _ { 2 } \geq 0$ network and logarithmic normalization together are non-negative. Adding the growth term and using Eq. (A.15),

$$
\psi = \underbrace { \mathcal { N } + \psi _ { \mathrm { n r m } } ^ { \mathrm { l n } } } _ { \geq 0 } + \psi _ { \mathrm { g r } } \geq \psi _ { \mathrm { g r } } \geq \frac { \epsilon } { 2 } \left( I _ { 1 } + I _ { 2 } \right) - 2 \epsilon , \qquad \psi \geq 0 = \psi ( I ) .\tag{A.24}
$$

In contrast to Eq. (A.22), the bound in Eq. (A.24) does not depend on the trained network weights, since the constant $- 2 \epsilon$ involves only the growth parameter ϵ, whereas $C _ { \mathrm { l i n } }$ grows with $a _ { 1 } ^ { 4 }$

## Appendix B. Mean-dilatation element (Q1/P0)

For nearly incompressible materials, trilinear displacement elements enforce J ≈ 1 at every quadrature point. These are too many constraints for the available displacement degrees of freedom, and the element locks volumetrically. The mean-dilatation approach [94, 104, 95] reduces the volume change to a single value per element. In the following, we introduce the modified kinematics, derive the resulting displacement-based potential and its variation, and conclude with the consequence for the hyperreduction.

Kinematics. For each element e, we define the mean dilatation

$$
\Theta ^ { e } ( { \bf d } ^ { e } ) : = \frac { 1 } { V ^ { e } } \int _ { B ^ { e } } J { \mathrm { d } } V ,\tag{B.1}
$$

where $V ^ { e }$ denotes the reference volume of element e. With this at hand, the modified deformation gradient reads

$$
\hat { \pmb { F } } : = \Big [ \frac { \Theta ^ { e } } { J } \Big ] ^ { 1 / 3 } \pmb { F } , \qquad \operatorname* { d e t } \hat { \pmb { F } } = \Theta ^ { e } .\tag{B.2}
$$

Hence, $\hat { F }$ shares the isochoric part $J ^ { - 1 / 3 } \pmb { F }$ with $F ,$ but carries the element-wise constant volume change $\Theta ^ { e }$ instead of $J .$

Potential. We evaluate the energy at ${ \hat { F } } ,$ , such that the discrete potential becomes

$$
\Pi ^ { h } ( \mathbf { d } , \pmb \theta ) = \sum _ { e = 1 } ^ { n _ { \mathrm { e l } } } \underbrace { \int _ { \mathcal { B } ^ { e } } \psi \bigl ( \hat { F } ( \mathbf { d } ^ { e } ) ; \pmb \theta \bigr ) \mathrm { d } V } _ { = : \Pi _ { \mathrm { i n t } } ^ { h , e } ( \mathbf { d } ^ { e } , \pmb \theta ) } - \Pi _ { \mathrm { e x t } } ^ { h } ( \mathbf { d } ) .\tag{B.3}
$$

This is a pure displacement formulation: the displacements are interpolated trilinearly (Q1), and no further unknowns occur. Moreover, the construction requires no split of ψ into volumetric and isochoric parts and thus applies directly to the energy of Eq. (25).

## Appendix C. Case 2: Identification from force and displacements

This appendix complements Section 3.2 by the material identified from the loss Eq. (42), which contains both the reaction force and the displacements. The training history and the adaptivity of the reduction of this run are shown in Fig. 12. Fig. C.1 evaluates the identified material by a full-order solve on the training specimen TT1 and on the unseen specimen TT2, in the same way as Fig. 13 for the force-based material. Compared with the latter, the additional displacement data improve the agreement of the displacement field on both specimens at the expense of the force on TT1, whereas the force on TT2 is overestimated by about the same amount, see Section 3.2.

## References

[1] J. N. Fuhg, G. Anantha Padmanabha, N. Bouklas, B. Bahmani, W. Sun, N. N. Vlassis, M. Flaschel, P. Carrara, and L. De Lorenzis. A review on data-driven constitutive laws for solids. Archives of Computational Methods in Engineering, 2024. URL https://arxiv. org/abs/2405.03658.

[2] K. Linka, M. Hillgärtner, K. P. Abdolazizi, R. C. Aydin, M. Itskov, and C. J. Cyron. Constitutive artificial neural networks: A fast and general approach to predictive data-driven constitutive modeling by deep learning. Journal of Computational Physics, 429:110010, 2021. doi: 10.1016/j.jcp.2020.110010. URL https://doi.org/10.1016/j.jcp.2020.110010.

[3] Kevin Linka and Ellen Kuhl. A new family of constitutive artificial neural networks towards automated model discovery. Computer Methods in Applied Mechanics and Engineering, 403:115731, 2023. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2022.115731. URL https://www.sciencedirect.com/science/article/pii/S0045782522006867.

[4] D. K. Klein, M. Fernández, R. J. Martin, P. Nef, and O. Weeger. Polyconvex anisotropic hyperelasticity with neural networks. Journal of the Mechanics and Physics of Solids, 159:104703, 2022. doi: 10.1016/j.jmps.2021.104703. URL https://doi.org/10.1016/j.jmps. 2021.104703.

[5] M. Flaschel, P. Steinmann, L. De Lorenzis, and E. Kuhl. Convex neural networks learn generalized standard material models. Journal of the Mechanics and Physics of Solids, 200:106103, 2025. URL https://cris.fau.de/publications/338715141.

[6] Hagen Holthusen, Lukas Lamm, Tim Brepols, Stefanie Reese, and Ellen Kuhl. Theory and implementation of inelastic constitutive artificial neural networks. Computer Methods in Applied Mechanics and Engineering, 428:117063, 2024. ISSN 0045-7825. doi: https: //doi.org/10.1016/j.cma.2024.117063. URL https://www.sciencedirect.com/science/article/pii/S0045782524003190.

[7] Max Rosenkranz, Karl A. Kalina, Jörg Brummund, WaiChing Sun, and Markus Kästner. Viscoelasticty with physics-augmented neural networks: model formulation and training methods without prescribed internal variables. Computational Mechanics, 74(6):1279–1301, May 2024. ISSN 1432-0924. doi: 10.1007/s00466-024-02477-1. URL http://dx.doi.org/10.1007/s00466-024-02477-1.

[8] Hagen Holthusen, Kevin Linka, Ellen Kuhl, and Tim Brepols. A generalized dual potential for inelastic constitutive artificial neural networks: A jax implementation at finite strains. Journal of the Mechanics and Physics of Solids, 206:106337, 2026. ISSN 0022-5096. doi: https://doi.org/10.1016/j.jmps.2025.106337. URL https://www.sciencedirect.com/science/article/pii/S0022509625003084.

[9] Hagen Holthusen and Ellen Kuhl. A complement to neural networks for anisotropic inelasticity at finite strains. Computer Methods in Applied Mechanics and Engineering, 450:118612, 2026. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2025.118612. URL https://www.sciencedirect.com/science/article/pii/S0045782525008849.

[10] Karl A. Kalina, Jörg Brummund, and Markus Kästner. A physics-augmented neural network framework for finite strain incompressible viscoelasticity. Computer Methods in Applied Mechanics and Engineering, 455:118892, 2026. ISSN 0045-7825. doi: https://doi.org/10. 1016/j.cma.2026.118892. URL https://www.sciencedirect.com/science/article/pii/S0045782526001659.

(a) Machine curve  
![](images/88d1b5f77b718214333d801edfee5a81ec9f24375f18451fc96d117933899690.jpg)

![](images/3c8a9f3e808a7650be99fb3e47a61f9bbb0294b430e621656be8cf06cffdf11b.jpg)

(c) Displacement parity  
![](images/c68e4209bab2c737357611f2259ebf9c38ee9c2f40a7d013606e6853c8cb5c63.jpg)

(d) Transverse displacement  
![](images/50d7f1fa8cf566ff6ba49ba539e4c37ff18c7c680769412c354ef93a45eebd99.jpg)

![](images/22758ad3507bf6b501728afcf6b5bdd2134da2cd2061aa365bae66c229f8cd45.jpg)

![](images/19386863952a9c241129b79e12382161b0ff4d4a6be88141d0d0e20e46b478f7.jpg)  
(b) Force parity  
(c) Displacement parity

(a) Machine curve  
![](images/4a630e18c5355c21304476ac3540d44c8109c5bd33ecb8d149862b4b95995cb7.jpg)

![](images/fadcb37ff5e37578f9828a24b1ae808639ff3ff61a390729daf078f19fed6bfa.jpg)

![](images/b8f8f09c3cc08d3606a5950eef5011520d4be755648ac3b94bdecfd3671300db.jpg)

![](images/14e95ace3dd36bfb678e1b9c264fcebde495ba76329b83868407b84016369a87.jpg)

![](images/feb1a64713611de9fa21af8ba7cd450584d7462dd0afdcc8e5e4154b5c9fb184.jpg)

![](images/043ec30bf2dc58406901c7cbd2dc064c9692519722d544385153b9cf909f7b59.jpg)  
Figure C.1: Case $^ { 2 , }$ material identified from force and displacements, evaluated by a full-order solve on the training specimen TT1 (top) and on the unseen specimen TT2 (bottom). Top: (a)–(b) The force is reproduced less accurately than in Fig. 13, with a root-mean-square error of 6.47 N and $R ^ { 2 } = 0 . 9 9 0 ;$ the material is too soft at small and too stif at large displacements. (c)–(f) In turn, the displacement field agrees with $R _ { x } ^ { 2 } = 0 . 9 9 1$ and $R _ { y } ^ { 2 } = 0 . 9 9 8$ , and the relative error falls below 2 % in most of the specimen. The trade-of between both channels indicates that force and field are not fully consistent within the present model. Bottom: For TT2, the force at large displacements is overestimated by about 10 % (root-mean-square error of 16.5 N), as for the force-based material, whereas the displacement field benefits from the additional data: $R _ { x } ^ { 2 }$ increases from 0.892 to 0.998, and the relative error drops below 5 % almost everywhere.

[11] Dominik K. Klein, Rogelio Ortigosa, Jesús Martínez-Frutos, and Oliver Weeger. Nonlinear electro-elastic finite element analysis with neural network constitutive models. Computer Methods in Applied Mechanics and Engineering, 425:116910, 2024. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2024.116910. URL https://www.sciencedirect.com/science/article/pii/S004578252400166X.

[12] Jan N. Fuhg, Asghar Jadoon, Oliver Weeger, D. Thomas Seidl, and Reese E. Jones. Polyconvex neural network models of thermoelas ticity. Journal of the Mechanics and Physics of Solids, 192:105837, 2024. ISSN 0022-5096. doi: https://doi.org/10.1016/j.jmps.2024. 105837. URL https://www.sciencedirect.com/science/article/pii/S002250962400303X.

[13] R. Ortigosa, J. Martínez-Frutos, A. Pérez-Escolar, I. Castañar, N. Ellmer, and A.J. Gil. A generalized theory for physics-augmented neural networks in finite strain thermo-electro-mechanics. Computer Methods in Applied Mechanics and Engineering, 437:117741, 2025. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2025.117741. URL https://www.sciencedirect.com/science/article/ pii/S0045782525000131.

[14] H. Holthusen, P. Steinmann, and E. Kuhl. A convex route to thermoelasticity: Learning internal energy and dissipation. Computer Methods in Applied Mechanics and Engineering, 459:119082, 2026.

[15] S. Avril, M. Bonnet, A.-S. Bretelle, M. Grédiac, F. Hild, P. Ienny, F. Latourte, D. Lemosse, S. Pagano, E. Pagnacco, and F. Pierron. Overview of identification methods of mechanical parameters based on full-field measurements. Experimental Mechanics, 48(4):381–402, 2008.

[16] U. Römer, S. Hartmann, J.-A. Tröger, D. Anton, H. Wessels, M. Flaschel, and L. De Lorenzis. Reduced and all-at-once approaches for model calibration and discovery in computational solid mechanics. Applied Mechanics Reviews, 77(4):040801, 2025. doi: 10.1115/1. 4066118. URL https://doi.org/10.1115/1.4066118.

[17] B. M. Riemer, M. Kästner, and K. A. Kalina. Calibration of neural viscoelastic models via full-field data. arXiv preprint arXiv:2609.03645, 2026. URL https://arxiv.org/abs/2609.03645.

[18] K. Friedrichs, F. Dammass, K. A. Kalina, and M. Kästner. Reliable training of neural hyperelastic models via full-field data. arXiv preprint arXiv:2609.13001, 2026. URL https://arxiv.org/abs/2609.13001.

[19] D. Claire, F. Hild, and S. Roux. A finite element formulation to identify damage fields: the equilibrium gap method. International Journal for Numerical Methods in Engineering, 61(2):189–208, 2004.

[20] M. Grédiac. Principe des travaux virtuels et identification. Comptes Rendus de l’Académie des Sciences, Série II, 309(1):1–5, 1989

[21] F. Pierron and M. Grédiac. The Virtual Fields Method: Extracting Constitutive Mechanical Parameters from Full-field Deformation Measurements. Springer, New York, 2012. doi: 10.1007/978-1-4614-1824-5.

[22] M. Flaschel, S. Kumar, and L. De Lorenzis. Unsupervised discovery of interpretable hyperelastic constitutive laws. Computer Methods in Applied Mechanics and Engineering, 381:113852, 2021. doi: 10.1016/j.cma.2021.113852. URL https://doi.org/10.1016/j.cma. 2021.113852.

[23] P. Thakolkaran, A. Joshi, Y. Zheng, M. Flaschel, L. De Lorenzis, and S. Kumar. NN-EUCLID: deep-learning hyperelasticity without stress data. Journal of the Mechanics and Physics of Solids, 169:105076, 2022. doi: 10.1016/j.jmps.2022.105076. URL https: //doi.org/10.1016/j.jmps.2022.105076.

[24] B. Alheit, S. Kumar, and M. Peirlinck. CANN-EUCLID: unsupervised constitutive artificial neural network model discovery from full-field data. arXiv preprint arXiv:2606.14565, 2026. URL https://doi.org/10.48550/arXiv.2606.14565.

[25] A. Abbasi, M. Ricci, P. Carrara, M. Flaschel, S. Kumar, S. Marfia, and L. De Lorenzis. Discovery of hyperelastic constitutive laws from experimental data with EUCLID. Experimental Mechanics, 66(5):877–908, 2026. doi: 10.1007/s11340-026-01290-6. URL https://doi.org/10.1007/s11340-026-01290-6.

[26] M. Bourdyot, M. Compans, R. Langlois, B. Smaniotto, E. Baranger, and C. Jailin. Learning a hyperelastic constitutive model from 3D experimental data. Computer Methods in Applied Mechanics and Engineering, 450:118592, 2026. doi: 10.1016/j.cma.2025.118592. URL https://doi.org/10.1016/j.cma.2025.118592.

[27] K. L. Chaurasiya, S. Dutta, S. Kumar, and A. Joshi. Hetero-EUCLID: interpretable model discovery for heterogeneous hyperelastic materials using stress-unsupervised learning. Computer Methods in Applied Mechanics and Engineering, 452:118729, 2026. doi: 10.1016/j.cma.2026.118729. URL https://doi.org/10.1016/j.cma.2026.118729.

[28] V. Taç, A. Amiri-Hezaveh, G. N. Bechtel, T. Loftin, M. K. Rausch, F. Sahli Costabal, and A. B. Tepole. Fully data-driven inverse characterization of heterogeneous materials with hyper-network neural ODEs. npj Computational Materials, 12(1):165, 2026. doi: 10.1038/s41524-026-02027-8. URL https://doi.org/10.1038/s41524-026-02027-8.

[29] V. Knauf Narouie, J.-H. Urrea-Quintero, F. Cirak, and H. Wessels. Unsupervised constitutive model discovery from sparse and noisy data. Computer Methods in Applied Mechanics and Engineering, 452:118722, 2026. doi: 10.1016/j.cma.2025.118722. URL https://doi.org/10.1016/j.cma.2025.118722.

[30] S. Ghouli, M. Flaschel, S. Kumar, and L. De Lorenzis. A topology optimisation framework to design test specimens for one-shot identification or discovery of material models. Journal of the Mechanics and Physics of Solids, 203:106210, 2025. doi: 10.1016/j.jmps. 2025.106210. URL https://doi.org/10.1016/j.jmps.2025.106210.

[31] A. Benady, E. Baranger, and L. Chamoin. Unsupervised learning of history-dependent constitutive material laws with thermodynamically-consistent neural networks in the modified constitutive relation error framework. Computer Methods in Applied Mechanics and Engineering, 425:116967, 2024. doi: 10.1016/j.cma.2024.116967. URL https://doi.org/10.1016/j.cma.2024.116967.

[32] M. Flaschel, D. Martonová, C. Veil, and E. Kuhl. Material fingerprinting: A shortcut to material model discovery without solving optimization problems. Computer Methods in Applied Mechanics and Engineering, 450:118573, 2026. URL https://arxiv.org/abs/ 2508.07831.

[33] Moritz Flaschel, Hagen Holthusen, Denisa Martonová, and Ellen Kuhl. Adaptive material fingerprinting for the fast discovery of polyconvex feature combinations in isotropic and anisotropic hyperelasticity. International Journal of Engineering Science, 227:104594, 2026. ISSN 0020-7225. doi: https://doi.org/10.1016/j.ijengsci.2026.104594. URL https://www.sciencedirect.com/science/article/ pii/S0020722526001321.

[34] M. Flaschel, B. Liu, and E. Kuhl. Neural operators solve inverse problems for constitutive model discovery. arXiv preprint arXiv:2607.15049, 2026. URL https://arxiv.org/abs/2607.15049.

[35] R. Mahnken and E. Stein. A unified approach for parameter identification of inelastic material models in the frame of the finite element method. Computer Methods in Applied Mechanics and Engineering, 136(3):225–258, 1996. doi: 10.1016/0045-7825(96)00991-7.

[36] S. Wiesheier, M. A. Moreno-Mateos, and P. Steinmann. Versatile data-adaptive hyperelastic energy functions for soft materials Computer Methods in Applied Mechanics and Engineering, 430:117208, 2024. URL https://cris.fau.de/publications/325536179.

[37] S. Wiesheier, M. A. Moreno-Mateos, and P. Steinmann. Data-adaptive spline-based viscoelasticity for soft solids. Computer Methods in Applied Mechanics and Engineering, 451:118705, 2026. doi: 10.1016/j.cma.2025.118705. URL https://doi.org/10.1016/j.cma.2025.

118705.

[38] X. Wu, Y. Zhang, and S. Mao. Learning the physics-consistent material behavior from measurable data via PDE-constrained op timization. Computer Methods in Applied Mechanics and Engineering, 437:117748, 2025. doi: 10.1016/j.cma.2025.117748. URL https://doi.org/10.1016/j.cma.2025.117748.

[39] J. Tan, G. Anantha Padmanabha, S. J. Yang, and N. Bouklas. Towards rapid constitutive model discovery from multi-modal data: physics augmented finite element model updating (paFEMU). arXiv preprint arXiv:2604.07746, 2026. URL https://arxiv.org/abs/ 2604.07746.

[40] M. Knipper, C. Ji, M. Brand, and K. Linka. Automatic-diferentiation-enabled finite element model updating: learning constitutive neural networks from structural deformation data. arXiv preprint arXiv:2606.05199, 2026. URL https://arxiv.org/abs/2606.05199.

[41] G. B. Gavris and W. Sun. Discovering neural elastoplasticity from kinematic observations. Proceedings of the National Academy of Sciences, 122(38):e2508732122, 2025. doi: 10.1073/pnas.2508732122. URL https://doi.org/10.1073/pnas.2508732122.

[42] B. P. Ferreira and M. A. Bessa. Automatically diferentiable model updating (ADiMU): conventional, hybrid, and neural network material model discovery including history-dependency. Journal of the Mechanics and Physics of Solids, 206:106408, 2026. doi: 10.1016/j.jmps.2025.106408. URL https://doi.org/10.1016/j.jmps.2025.106408.

[43] G. B. Gavris and W. Sun. Discovering neural cohesive zone laws from displacement fields. Computer Methods in Applied Mechanics and Engineering, 452:118733, 2026. doi: 10.1016/j.cma.2026.118733. URL https://doi.org/10.1016/j.cma.2026.118733.

[44] L. River Spencer, William D. Meador, Brian N. Granzow, Adrian Buganza Tepole, Jin Yang, Manuel K. Rausch, D. Thomas Seidl, and Jan N. Fuhg. Full-field calibration of coupled thermomechanical material models at finite strain. arXiv preprint arXiv:2606.05465, 2026. URL https://arxiv.org/abs/2606.05465.

[45] T. Xue, S. Liao, Z. Gan, et al. JAX-FEM: A diferentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science. Computer Physics Communications, 291:108802, 2023. doi: 10.1016/j.cpc.2023.108802. URL https: //doi.org/10.1016/j.cpc.2023.108802.

[46] F. Regazzoni. Hyperelastic constitutive model discovery with diferentiable finite elements and structure-preserving neural networks. Computational Mechanics, 2026. doi: 10.1007/s00466-026-02850-2. URL https://doi.org/10.1007/s00466-026-02850-2.

[47] T. Xu, S. Atashi, and T. Xue. A diferentiable Lagrangian-coupled 3D Gaussian Splatting–SPH model for forward simulation and inverse analysis in solid mechanics. arXiv preprint arXiv:2610.04336, 2026. URL https://arxiv.org/abs/2610.04336.

[48] P. Benner, S. Gugercin, and K. Willcox. A survey of projection-based model reduction methods for parametric dynamical systems. SIAM Review, 57(4):483–531, 2015. doi: 10.1137/130932715. URL https://doi.org/10.1137/130932715.

[49] S. Chaturantabut and D. C. Sorensen. Nonlinear model reduction via discrete empirical interpolation. SIAM Journal on Scientific Computing, 32(5):2737–2764, 2010. doi: 10.1137/090766498. URL https://doi.org/10.1137/090766498.

[50] K. Carlberg, C. Farhat, J. Cortial, and D. Amsallem. The GNAT method for nonlinear model reduction: efective implementation and application to computational fluid dynamics and turbulent flows. Journal of Computational Physics, 242:623–647, 2013. doi: 10.1016/j.jcp.2013.02.028. URL https://doi.org/10.1016/j.jcp.2013.02.028.

[51] D. Ryckelynck. A priori hyperreduction method: an adaptive approach. Journal of Computational Physics, 202(1):346–366, 2005. doi: 10.1016/j.jcp.2004.07.015. URL https://doi.org/10.1016/j.jcp.2004.07.015.

[52] S. S. An, T. Kim, and D. L. James. Optimizing cubature for eficient integration of subspace deformations. ACM Transactions on Graphics, 27(5), 2008. doi: 10.1145/1409060.1409118. URL https://doi.org/10.1145/1409060.1409118.

[53] C. Farhat, T. Chapman, and P. Avery. Structure-preserving, stability, and accuracy properties of the energy-conserving sampling and weighting method for the hyperreduction of nonlinear finite element dynamic models. International Journal for Numerical Methods in Engineering, 102(5):1077–1110, 2015.

[54] J. A. Hernández, M. A. Caicedo, and A. Ferrer. Dimensional hyper-reduction of nonlinear finite element models via empirical cubature. Computer Methods in Applied Mechanics and Engineering, 313:687–722, 2017. doi: 10.1016/j.cma.2016.10.022. URL https://doi. org/10.1016/j.cma.2016.10.022.

[55] S. Wulfinghof and J. Hauck. E3C hyper-reduction for computational homogenization in nonlinear mechanics. Computational Mechanics, 76:1563–1572, 2025. doi: 10.1007/s00466-025-02662-w. URL https://doi.org/10.1007/s00466-025-02662-w.

[56] Annika Radermacher and Stefanie Reese. Pod-based model reduction with empirical interpolation applied to nonlinear elasticity. International Journal for Numerical Methods in Engineering, 107(6):477–495, 2016. doi: https://doi.org/10.1002/nme.5177. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.5177.

[57] J. A. Hernández, J. Oliver, A. E. Huespe, M. A. Caicedo, and J. C. Cante. High-performance model reduction techniques in computational multiscale homogenization. Computer Methods in Applied Mechanics and Engineering, 276:149–189, 2014. doi: 10.1016/j.cma.2014.03.011. URL https://doi.org/10.1016/j.cma.2014.03.011.

[58] O. Kunc and F. Fritzen. Finite strain homogenization using a reduced basis and eficient sampling. Mathematical and Computational Applications, 24(2):56, 2019. doi: 10.3390/mca24020056. URL https://doi.org/10.3390/mca24020056.

[59] N. Lange, G. Hütter, and B. Kiefer. A monolithic hyper ROM FE<sup>2</sup> method with clustered training at finite deformations. Computer Methods in Applied Mechanics and Engineering, 418:116522, 2024. doi: 10.1016/j.cma.2023.116522. URL https://doi.org/10.1016/ j.cma.2023.116522.

[60] M. B. Shamim and S. Wulfinghof. Variational three-field reduced order modeling for nearly incompressible materials. Computational Mechanics, 74:1073–1087, 2024. doi: 10.1007/s00466-024-02468-2. URL https://doi.org/10.1007/s00466-024-02468-2.

[61] F. Trainotti, J. Marinko, J. Maierhofer, and D. J. Rixen. ECSW hyperreduction of hyper-viscoelastic components via co-simulation with Abaqus. Finite Elements in Analysis and Design, 241:104222, 2024. doi: 10.1016/j.finel.2024.104222. URL https://doi.org/10. 1016/j.finel.2024.104222.

[62] E. Agouzal, J.-P. Argaud, M. Bergmann, G. Ferté, and T. Taddei. A projection-based reduced-order model for parametric quasi-static nonlinear mechanics using an open-source industrial code. International Journal for Numerical Methods in Engineering, 2024. doi: 10.1002/nme.7385. URL https://doi.org/10.1002/nme.7385.

[63] J. Kehls, S. Ritzert, L. Breuer, Q. Zhang, S. Reese, and T. Brepols. Multi-field decomposed hyper-reduced order modeling of damage plasticity simulations. Computer Methods in Applied Mechanics and Engineering, 451:118702, 2026. doi: 10.1016/j.cma.2025.118702. URL https://doi.org/10.1016/j.cma.2025.118702.

[64] A. Ashkpour and A. Raeisi Najafi. Hyper-reduction of nonlinear hardening plasticity models using Petrov–Galerkin projection and gappy POD. Computer Methods in Applied Mechanics and Engineering, 461:119155, 2026. doi: 10.1016/j.cma.2026.119155. URL https://doi.org/10.1016/j.cma.2026.119155.

[65] A. Larsson, M. Kim, C. Vales, S. Adriaenssens, D. M. Copeland, Y. Choi, and S. W. Cheung. Hyper-reduction methods for accelerating

nonlinear finite element simulations: open source implementation and reproducible benchmarks. arXiv preprint arXiv:2602.23551, 2026. URL https://arxiv.org/abs/2602.23551.

[66] E. Faust and L. Scheunemann. Eficient strain-space hyperreduction in large-deformation solid mechanics. arXiv preprint arXiv:2607.19330, 2026. URL https://arxiv.org/abs/2607.19330.

[67] S. Ritzert, J. Kehls, E. Faust, H. Holthusen, S. Reese, and T. Brepols. Component-wise hyperreduction for nonlinear solid mechanics problems. arXiv preprint arXiv:2608.18868, 2026. URL https://arxiv.org/abs/2608.18868.

[68] K. Lee and K. T. Carlberg. Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders. Journal of Computational Physics, 404:108973, 2020. doi: 10.1016/j.jcp.2019.108973. URL https://doi.org/10.1016/j.jcp.2019.108973.

[69] L. Scheunemann and E. Faust. A manifold learning approach to nonlinear model order reduction of quasi-static problems in solid mechanics: proof of concept and parameter study. Computational Mechanics, 77(4):983–1023, 2025. doi: 10.1007/s00466-025-02696-0. URL https://doi.org/10.1007/s00466-025-02696-0.

[70] E. Faust and L. Scheunemann. A hyperreduced manifold learning approach to nonlinear model order reduction for the homogenisation of hyperelastic RVEs. arXiv preprint arXiv:2508.21527, 2025. URL https://arxiv.org/abs/2508.21527.

[71] Jannick Kehls, Ellen Kuhl, Tim Brepols, Kevin Linka, and Hagen Holthusen. Autoencoder-based non-intrusive model order reduction for anisotropic and multi-physics problems. Computational Mechanics, Oct 2026. ISSN 1432-0924. doi: 10.1007/s00466-026-02870-y. URL https://doi.org/10.1007/s00466-026-02870-y.

[72] A. Schütz, L. Nolle, and T. Bechtold. Non-intrusive hyperreduction by a physics-augmented neural network with second-order Sobolev training. arXiv preprint arXiv:2601.10442, 2026. URL https://arxiv.org/abs/2601.10442.

[73] O. Ghattas and K. Willcox. Learning physics-based models from data: perspectives from inverse problems and model reduction. Acta Numerica, 30:445–554, 2021. doi: 10.1017/S0962492921000064. URL https://doi.org/10.1017/S0962492921000064.

[74] D. Amsallem, M. Zahr, Y. Choi, and C. Farhat. Design optimization using hyper-reduced-order models. Structural and Multidisciplinary Optimization, 51(4):919–940, 2015. doi: 10.1007/s00158-014-1183-y. URL https://doi.org/10.1007/s00158-014-1183-y.

[75] M. J. Zahr and C. Farhat. Progressive construction of a parametric reduced-order model for PDE-constrained optimization. International Journal for Numerical Methods in Engineering, 102(5):1111–1135, 2015. doi: 10.1002/nme.4770. URL https://doi.org/10.1002/nme. 4770.

[76] G. Agarwal, J.-H. Urrea-Quintero, H. Wessels, T. Wick, et al. Parameter identification and uncertainty propagation of hydrogel coupled difusion-deformation using POD-based reduced-order modeling. Computational Mechanics, 75(2):515–545, 2025. doi: 10. 1007/s00466-024-02517-w. URL https://doi.org/10.1007/s00466-024-02517-w.

[77] M. R. Pfaller, M. Cruz Varona, J. Lang, C. Bertoglio, and W. A. Wall. Using parametric model order reduction for inverse analysis of large nonlinear cardiac simulations. International Journal for Numerical Methods in Biomedical Engineering, page e3320, 2020. doi: 10.1002/cnm.3320. URL https://doi.org/10.1002/cnm.3320.

[78] V. Buljak, G. Cocchetti, A. Cornaggia, and G. Maier. Parameter identification in elastoplastic material models by small punch tests and inverse analysis with model reduction. Meccanica, 53(15):3815–3829, 2018. doi: 10.1007/s11012-018-0914-3. URL https: //doi.org/10.1007/s11012-018-0914-3.

[79] M. Bhattacharyya and L. Chamoin. Efective reduced-order approximation for fast and robust MCRE-based parametric identification of nonlinear history-dependent material laws. Computational Mechanics, 78(1):213–238, 2026. doi: 10.1007/s00466-026-02754-1. URL https://doi.org/10.1007/s00466-026-02754-1.

[80] D. Anton, J.-A. Tröger, H. Wessels, U. Römer, A. Henkes, and S. Hartmann. Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks. Advanced Modeling and Simulation in Engineering Sciences, 12:12, 2025. doi: 10.1186/s40323-025-00285-7. URL https://doi.org/10.1186/s40323-025-00285-7.

[81] J. M. Ball. Convexity conditions and existence theorems in nonlinear elasticity. Archive for Rational Mechanics and Analysis, 63(4): 337–403, 1977.

[82] L. Sirovich. Turbulence and the dynamics of coherent structures. Part I: Coherent structures. Quarterly of Applied Mathematics, 45 (3):561–571, 1987.

[83] C. Farhat, P. Avery, T. Chapman, and J. Cortial. Dimensional reduction of nonlinear finite element dynamic models with finite rotations and energy-based mesh sampling and weighting for computational eficiency. International Journal for Numerical Methods in Engineering, 98(9):625–662, 2014.

[84] Hagen Holthusen, Moritz Flaschel, Denisa Martonová, and Ellen Kuhl. Benchmarking data-driven material models on the classic treloar dataset, 2026. URL https://arxiv.org/abs/2608.14063.

[85] Brandon Amos, Lei Xu, and J Zico Kolter. Input Convex Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, page 10, Sydney, Australia, 2017.

[86] Lennart Linden, Dominik K. Klein, Karl A. Kalina, Jörg Brummund, Oliver Weeger, and Markus Kästner. Neural networks meet hyperelasticity: A guide to enforcing physics. Journal of the Mechanics and Physics of Solids, 179:105363, October 2023. ISSN 00225096. doi: 10.1016/j.jmps.2023.105363. URL https://linkinghub.elsevier.com/retrieve/pii/S0022509623001679.

[87] Gal Berkooz, Philip Holmes, and John L. Lumley. The proper orthogonal decomposition in the analysis of turbulent flows. Annual Review of Fluid Mechanics, 25(1):539–575, 1993. doi: 10.1146/annurev.fl.25.010193.002543.

[88] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs. http: //github.com/jax-ml/jax, 2018.

[89] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

[90] Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the dificulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 1310–1318, 2013.

[91] Henk A. van der Vorst. Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems. SIAM Journal on Scientific and Statistical Computing, 13(2):631–644, 1992. doi: 10.1137/0913035.

[92] Emma Lejeune. Mechanical mnist: A benchmark dataset for mechanical metamodels. Extreme Mechanics Letters, 36:100659, 2020. ISSN 2352-4316. doi: https://doi.org/10.1016/j.eml.2020.100659. URL https://www.sciencedirect.com/science/article/ pii/S2352431620300341.

[93] Manuel Rausch, Adrian Buganza Tepole, Jan Fuhg, and Francisco Sahli Costabal. Mechanical MNIST challenge data. GitHub repository, https://github.com/fsahli/mechanical-MNIST-challenge, 2026. Accessed: 2026-10-20. Challenge website: https://www. manuelrausch.com/mnist-challenge.

[94] J. C. Nagtegaal, D. M. Parks, and J. R. Rice. On numerically accurate finite element solutions in the fully plastic range. Computer Methods in Applied Mechanics and Engineering, 4(2):153–177, 1974. doi: 10.1016/0045-7825(74)90032-2.

[95] J. C. Simo, R. L. Taylor, and K. S. Pister. Variational and projection methods for the volume constraint in finite deformation elastoplasticity. Computer Methods in Applied Mechanics and Engineering, 51(1–3):177–208, 1985. doi: 10.1016/0045-7825(85)90033-7.

[96] R. W. Ogden. Non-Linear Elastic Deformations. Dover Publications, Mineola, NY, 1997.

[97] P. Ladevèze and D. Leguillon. Error estimate procedure in the finite element method and applications. SIAM Journal on Numerical Analysis, 20(3):485–509, 1983. doi: 10.1137/0720033.

[98] Dominik K. Klein, Karl A. Kalina, Rogelio Ortigosa, Jesús Martínez-Frutos, Markus Kästner, and Oliver Weeger. Polyconvex anisotropic hyperelasticity via group symmetrization. Journal of the Mechanics and Physics of Solids, 218:106815, 2027. ISSN 0022-5096. doi: https://doi.org/10.1016/j.jmps.2026.106815. URL https://www.sciencedirect.com/science/article/pii/S0022509626003169.

[99] Brain M. Riemer, Jörg Brummund, Karl A. Kalina, Abel H.G. Milor, Franz Dammaß, and Markus Kästner. Construction of minimal integrity bases for anisotropic hyperelasticity via structural tensors. Journal of the Mechanics and Physics of Solids, 216:106763, 2026. ISSN 0022-5096. doi: https://doi.org/10.1016/j.jmps.2026.106763. URL https://www.sciencedirect.com/science/article/ pii/S0022509626002632.

[100] Birte Boes, Jaan-Willem Simon, and Hagen Holthusen. Accounting for plasticity: An extension of inelastic constitutive artificial neural networks. European Journal of Mechanics - A/Solids, 117:105998, 2026. ISSN 0997-7538. doi: https://doi.org/10.1016/j.euromechsol. 2025.105998. URL https://www.sciencedirect.com/science/article/pii/S0997753825004322.

[101] Tim van der Velden, Birte Boes, Tim Brepols, Ellen Kuhl, and Hagen Holthusen. A note on constitutive artificial neural networks for finite strain plasticity. Mechanics Research Communications, 155:104707, 2026. ISSN 0093-6413. doi: https://doi.org/10.1016/j. mechrescom.2026.104707. URL https://www.sciencedirect.com/science/article/pii/S0093641326000972.

[102] Ralph Tyrell Rockafellar. Convex Analysis. Princeton University Press, Princeton, 1970. ISBN 9781400873173. doi: doi:10.1515/ 9781400873173. URL https://doi.org/10.1515/9781400873173.

[103] Jörg Schröder and Patrizio Nef. Poly-, quasi- and rank-one convexity in applied mechanics. CISM International Centre for Mechanical Sciences. Springer, Vienna, Austria, 2010 edition, June 2012.

[104] T. J. R. Hughes. Generalization of selective integration procedures to anisotropic and nonlinear media. International Journal for Numerical Methods in Engineering, 15(9):1413–1418, 1980. doi: 10.1002/nme.1620150914.