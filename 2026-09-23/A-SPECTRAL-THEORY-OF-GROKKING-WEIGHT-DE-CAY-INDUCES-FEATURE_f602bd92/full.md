# A SPECTRAL THEORY OF GROKKING: WEIGHT DE-CAY INDUCES FEATURE LEARNING

Lenz Pracher<sup>1,2∗</sup> Pascal de Jong<sup>1,∗</sup> Oskar Lieshaus<sup>1</sup> Alan Jeffares<sup>3</sup> Steffen Rulands<sup>1†</sup>

<sup>1</sup>Arnold-Sommerfeld-Center for Theoretical Physics,

Ludwig-Maximilians-Universitat M¨ unchen, Munich, Germany¨

<sup>2</sup>Department of Applied Physics, Standford University, Standford, CA, USA

<sup>3</sup>Department of Mathematics, University of Cambridge, Cambridge, UK

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author (rulands@lmu.de).

## ABSTRACT

In grokking an early fit to the training data separates from a much later improvement in generalization. During this delay, training can move from a fixed neural tangent kernel (NTK) regime to one in which task-relevant kernel eigendirections continue to evolve. We provide a quantitative theory for how this transition from lazy to rich learning can produce delayed generalization. For homogeneous networks trained with squared loss and $L _ { 2 }$ weight decay, we show that a finite residual remains after memorization, with larger residual fractions in target components associated with smaller NTK eigenvalues. These residuals feed back into the dynamics of the NTK itself, and projecting the resulting dynamics onto task-relevant spectral directions yields a reduced system in which residualdriven kernel growth competes with weight decay. This system predicts that the grokking timescale is controlled by the product of learning rate and weight decay, that feature learning slows logarithmically near a critical decay above which task-aligned NTK structure can no longer support generalization, and that stronger decay can prevent fitting altogether. We test these predictions in modular addition. In a homogeneous MLP, task-aligned Fourier structure continues to emerge in the NTK after training accuracy has saturated, and an 84×90-grid of trained networks across varying learning rate and weight decay recovers the predicted phase geometry and inverse-product scaling of the generalization time with learning rate and weight decay. A one-block Transformer shows similar macroscopic phase structure in a 42×45-grid, as well as the same transition-time scaling despite violating exact homogeneity. Together, these results provide a mechanistic derivation connecting post-fit feature learning to both the onset of generalization and its phase structure in the learning rate and weight decay plane.

## 1 INTRODUCTION

Analyzing the dynamics of generalization in deep neural networks is challenging, because optimization processes that lead to memorizing or overfitting, representation learning, and changes in validation set predictions usually occur together. However, grokking separates these timescales, as a network can reach high training accuracy, while remaining inaccurate on held-out examples for thousands of additional updates and only then generalize (Power et al., 2022). This separation lets us ask what happens after the training labels are already fitted and predicted correctly. What error remains after the training labels are already fitted, how does it reshape the learned features, and what sets the delay before those changes improve held-out predictions?

The neural tangent kernel (NTK) provides a framework for distinguishing approximately fixedfeature dynamics from feature learning. Let f denote the network outputs, $\bar { J } = \nabla _ { \theta } \mathbf { f }$ the parameter Jacobian, and $K = J J ^ { \top }$ the empirical NTK, which quantifies how network outputs adapt to changes to the parameters. For gradient flow on a differentiable data loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ , the chain rule gives

$$
\dot { \mathbf { f } } _ { \mathrm { d a t a } } = - K \nabla _ { \mathbf { f } } \mathcal { L } _ { \mathrm { d a t a } } ,\tag{1}
$$

![](images/78604f698d23ac19c56c88959aa1fd73324eb3ad8d054b817786944b9a2d5519.jpg)  
Figure 1: A fast, near-fixed-kernel stage fits the training data but leaves a finite residual under cou pled weight decay. In the reduced dynamics that follow, the residual and task-aligned NTK strength evolve together. That is, the kernel reshapes the residual, and the residual reshapes the kernel, until task-aligned structure is strong enough to move held-out predictions across their decision margins. The lower panel separates the training fit from the later generalization transition and shows the slow coordinate $\tau \simeq \eta \lambda _ { W } ( s - s _ { \mathrm { f i t } } )$

which is architecture-independent. When K remains constant, training is thus described by a fixed kernel. On the other hand, changes in K reflect changes in the tangent features of the network (Jacot et al., 2018). The neural tangent hierarchy extends this description by expressing the evolution of $K$ in terms of higher-order tangent tensors (Huang & Yau, 2020). The residual error, the deviation between the network prediction and the true output, then enters the dynamics of K itself. Hence, the neural tangent hierarchy framework links the residual that is left after saturation of the training accuracy to subsequent changes in the NTK structure that are relevant to the task at hand.

Here, we derive a post-fit mechanism for delayed generalization in homogeneous networks trained with squared loss and coupled $L _ { 2 }$ weight decay. Weight decay leaves residual error after fitting, and this drives continued task-aligned evolution of the empirical NTK. For an approximately fixed NTK during the initial fit, we prove that the fraction of each target component remaining as residual error decreases monotonically with the corresponding NTK eigenvalue. The neural tangent hierarchy couples these residuals to the evolution of the NTK, allowing task-aligned tangent structure to continue developing after memorization. Modular addition provides a natural Fourier basis for projecting these dynamics onto a reduced residual-NTK system in which residual-driven growth competes with weight decay. Its adiabatic limit predicts a slow post-fit timescale controlled by the product $\eta \lambda _ { W }$ of learning rate and weight decay, together with finite-training boundaries in the correspond ing (η, λ )-phase space. Empirically, we observe the corresponding post-fit NTK organization and optimizer-space structure in a homogeneous MLP, with the same scaling of generalization time with learning rate and weight decay also appearing in a Transformer. Figure 1 summarizes the proposed mechanism.

Related works. Delayed generalization can arise without evolving neural features. Linear estimators, Gaussian processes, and logistic models exhibit grokking under suitable conditions (Levi et al., 2024; Miller et al., 2024; Beck et al., 2025). Most directly, Xu et al. (2026) prove end-toend grokking in overparameterized ridge regression trained by gradient descent with weight decay and derive quantitative hyperparameter dependence of the grokking time, while Kim (2026) derives an exactly solvable late-time weight-decay relaxation recovering the $( \eta \lambda _ { W } ) ^ { - 1 }$ scale in linear models. These results show that $( \eta \bar { \lambda } _ { W } ) ^ { - 1 }$ -type timing alone does not distinguish feature learning from fixed-feature dynamics.

Mechanistic work on neural-network grokking instead points to gradual representation change during the apparent plateau. In modular arithmetic, this includes the emergence of Fourier-structured circuits (Nanda et al., 2023), structured features and competition between memorizing and generalizing solutions (Liu et al., 2022; 2023a; Varma et al., 2024; Merrill et al., 2023; Ding et al., 2024), and transitions from an early kernel-like regime to later feature learning (Kumar et al., 2024; Lyu et al., 2024; Mohamadi et al., 2024; Rubin et al., 2024; Tian, 2026). A related line of empirica work tracks how the tangent features of the network reorganize over this transition. In particular, leading empirical-NTK eigenfunctions become increasingly task-relevant as delayed generalization emerges (Sanguino Bautiste et al., 2024), while substantial empirical-NTK movement can precede the representational changes that more closely track generalization (Zheng et al., 2024), and empirical-NTK eigenspaces in modular-arithmetic MLPs and Transformers align increasingly with Fourier features used by the learned solution (Lin, 2025). These works show that learned representations and the empirical NTK can continue to evolve during grokking, but leave open what drives this post-memorization feature learning and how the evolution of task-aligned features is linked to generalization. As pointed out by Xu et al. (2026), a rigorous theoretical analysis that connects grokking to the transition from the lazy to the rich regime of training neural networks is missing.

In this work, we provide, to the best of our knowledge, the first quantitative theory linking this transition from lazy to rich training to delayed generalization. Appendix A develops the connections to other works in more detail. Specifically, we make the following contributions:

Contributions. We study delayed generalization in homogeneous networks trained with squared loss and coupled $L _ { 2 }$ weight decay and derive a mechanism connecting post-fit residual error to continued task-aligned feature learning:

• We prove that, after the initial fit, coupled weight decay leaves a finite residual in the eigendirections of an approximately fixed NTK, with the largest relative residuals in directions that are weakly represented by the current features (Theorem 1).

• We show that these residuals enter the subsequent evolution of the empirical NTK. Projecting the neural tangent hierarchy onto a task-aligned Fourier direction yields a reduced residual–NTK system in which residual-driven growth competes with weight decay (Theorem 2).

• We derive a decay-controlled slow timescale for post-fit feature learning, predicting $( \eta \lambda _ { W } ) ^ { - 1 }$ scaling in the grokking regime and high-decay cutoffs where feature growth or training fit fails (Section 2.3).

• We test these predictions in homogeneous MLPs on modular addition, observing continued Fourier organization of the NTK and the predicted weak-decay timing and optimizer-space structure. A non-homogeneous one-block Transformer shows similar macroscopic behavior (Section 3).

## 2 POST-FIT RESIDUALS DRIVE TANGENT FEATURE LEARNING

We start by deriving the post-fit residual under coupled weight decay, and we show how it drives subsequent NTK evolution and reduce the dynamics to a task-aligned spectral mode. We then connect this mode growth to held-out generalization and the resulting optimizer-space grokking boundaries.

## 2.1 WEAK TARGET MODES RETAIN LARGER RESIDUAL FRACTIONS

We perform our analysis in the setting of modular arithmetic, where the task is to learn the mapping $( a , \bar { b } ) \mapsto c = ( a + b )$ mod $p ,$ for a given prime $p .$ Inputs and targets are one-hot encoded such that the network has p output coordinates. Let $\mathbb { Z } _ { p }$ denote the additive group of integers modulo $p .$ Its characters are $\chi _ { k } ( z ) = \exp ( 2 \pi i k z / p ) , k \in \bar { \mathbb { Z } } _ { p }$ , such that for output class c the target can be written as

$$
y _ { c } ( a , b ) = \frac { 1 } { p } \sum _ { k = 0 } ^ { p - 1 } e ^ { - 2 \pi i k c / p } \chi _ { k } ( a ) \chi _ { k } ( b ) .\tag{2}
$$

Because the target contains only products $\chi _ { k } ( a ) \chi _ { k } ( b )$ with the same frequency k, this specifies the relevant, task-aligned Fourier components that a successful model must learn to generalize. Note that the fixed-kernel residual dynamics derived below are not specific to modular addition, but hold for arbitrary datasets and targets under the homogeneous squared-loss setting. Modularity enters only when we use the task symmetry to identify an explicit Fourier basis for the subsequent spectral reduction.

Let n be the number of training examples and stack all p-dimensional predictions and targets into $\mathbf { f } , \mathbf { y } \in \mathbb { R } ^ { n p }$ . With residual ${ \pmb { \sigma } } = \mathbf { f } - \mathbf { y } \in \mathbb { R } ^ { n p }$ , the training objective can be written as

$$
\mathscr { L } ( \pmb \theta ) = \frac { \mathscr { N } } { 2 } \left\| \pmb \sigma \right\| ^ { 2 } + \frac { \lambda _ { W } } { 2 } \left\| \pmb \theta \right\| ^ { 2 } ,\tag{3}
$$

where $\mathcal { N }$ is an arbitrary normalization factor<sup>1</sup>, and $\lambda _ { W }$ is the coefficient of the $L _ { 2 }$ penalty. Throughout, lowercase $\lambda _ { W }$ denotes optimizer weight decay, whereas capital $\Lambda$ denotes an NTK spectral strength or eigenvalue. We define $\lambda _ { W } ^ { \mathrm { n o r m } } = \bar { \lambda } _ { W } / \mathcal { N }$ as the decay coefficient in normalized gradientflow time. One gradient-descent update with learning rate η advances this time by $\mathcal { N } \eta$ to first order. For $P$ trainable parameters, $J = \dot { \nabla } _ { \pmb { \theta } } \mathbf { f } _ { \mathbf { \theta } } \in \mathbb { R } ^ { n p \times P }$ is the parameter Jacobian and $\dot { K } \bar { = } J \dot { J } ^ { \top } \in \mathbb { R } ^ { n p \times n p }$ is the empirical NTK, which measures how strongly the network output can change along any direction in output space under parameter updates. Gradient flow then leads to

$$
\dot { \pmb \theta } = - J ^ { \top } \pmb \sigma - \lambda _ { W } ^ { \mathrm { n o r m } } \pmb \theta , \qquad \dot { \pmb \sigma } = - K \pmb \sigma - \lambda _ { W } ^ { \mathrm { n o r m } } J \pmb \theta .\tag{4}
$$

Furthermore, if $\mathbf { f } _ { \theta }$ is D-homogeneous under uniform parameter rescaling, Euler’s identity gives $J { \pmb \theta } = D { \bf f }$ . Bias-free feed-forward ReLU networks satisfy this identity, whereas biases, normalization, mixed-degree residual paths, and standard attention generally break it. Substituting this identity into Eq. (4) closes the dynamics in residual space.

Theorem 1 (Persistent ridge residual under homogeneous dynamics). Let $\mathbf { f } _ { \theta }$ be D-homogeneous and let $\lambda _ { W } ^ { \mathrm { n o r m } } > 0 .$ . Gradientflow on $E q$ . (3) obeys

$$
\dot { \pmb { \sigma } } = - \big ( \boldsymbol { K } + D \lambda _ { W } ^ { \mathrm { n o r m } } I \big ) \pmb { \sigma } - D \lambda _ { W } ^ { \mathrm { n o r m } } \mathbf { y } .\tag{5}
$$

If $K ( t ) \equiv K$ , then

$$
\pmb { \sigma } ( t ) \longrightarrow \pmb { \sigma } _ { * } = - D \lambda _ { W } ^ { \mathrm { n o r m } } ( K + D \lambda _ { W } ^ { \mathrm { n o r m } } I ) ^ { - 1 } \mathbf { y } .\tag{6}
$$

Every finite-eigenvalue direction with a nonzero target projection therefore retains a nonzero resid ual.

For ReLU networks, the derivation holds on each time interval along the gradient-flow trajectory over which the activation pattern is fixed, and extends piecewise across activation-boundary crossings. We prove the theorem in Appendix B.1. Now, we diagonalize $K$ , writing $K \mathbf { u } _ { j } = \Lambda _ { j } \mathbf { u } _ { j }$ , where $\mathbf { u } _ { j }$ is an eigenvector of K with eigenvalue $\Lambda _ { j } .$ , and denote the corresponding target and residual coefficients by $y _ { j } = \mathbf { u } _ { i } ^ { \top } \mathbf { y }$ and $\sigma _ { j } = \mathbf { u } _ { i } ^ { \top } \boldsymbol { \sigma }$ . The fixed-kernel limit in Eq. (6) then gives the equilibrium residual coefficient in the jth NTK eigendirection,

$$
\sigma _ { j , \ast } = - \frac { D \lambda _ { W } ^ { \mathrm { n o r m } } } { \Lambda _ { j } + D \lambda _ { W } ^ { \mathrm { n o r m } } } y _ { j } .\tag{7}
$$

This shows that for fixed $| y _ { j } |$ , the fraction of the jth target component that remains as residual error is larger for smaller $\Lambda _ { j }$ . These weaker modes correspond to directions that the current features represent poorly. However, in a finite network the NTK can continue to evolve after training accuracy has saturated. Therefore, we now study whether the larger relative residuals in weak task-aligned modes can drive continued task-aligned NTK evolution.

## 2.2 RESIDUALS DRIVE TASK-ALIGNED TANGENT GROWTH

For a one-hidden-layer, $\textit { D } = \ 2$ , bias-free ReLU MLP, combining the neural tangent hierarchy (Huang & Yau, 2020) with the two-homogeneous weight-decay terms gives

$$
\dot { \pmb \sigma } = - ( \boldsymbol K + 2 \lambda _ { W } ^ { \mathrm { n o r m } } \boldsymbol I ) \pmb \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \mathbf { y } , \qquad \dot { \boldsymbol K } = - \boldsymbol K ^ { ( 2 ) } \pmb \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \boldsymbol K .\tag{8}
$$

Here $K ^ { ( 2 ) }$ is a third-order tangent tensor, and $K ^ { ( 2 ) } \pmb { \sigma }$ denotes contraction over its residual index, $\begin{array} { r } { ( K ^ { ( 2 ) } \pmb { \sigma } ) _ { i j } : = \sum _ { k } K _ { i j k } ^ { ( 2 ) } \pmb { \sigma } _ { k } } \end{array}$ . This contraction couples the residual and kernel dynamics, and again employing two-homogeneity the parameter decay term becomes $- 2 \lambda _ { W } ^ { \mathrm { n o r m } } K$ . Although this term shrinks the NTK, the residual-dependent part $- K ^ { ( 2 ) } \sigma$ can simultaneously increase NTK strength along task-aligned directions.

To proceed, choose a normalized real Fourier direction u with nonzero target projection and orient it so that $y = \mathbf { u } ^ { \top } \mathbf { y } \geq 0$ . We define the task-aligned NTK strength Λ and projected residual σ by

$$
\Lambda = \mathbf { u } ^ { \top } K \mathbf { u } , \qquad \boldsymbol { \sigma } = \mathbf { u } ^ { \top } \boldsymbol { \sigma } .\tag{9}
$$

Λ measures how strongly the network can change its output along the selected task direction under parameter updates.

The Fourier structure of modular addition makes u a natural task direction. This motivates a local one-mode approximation in which u remains approximately an eigendirection of the $\mathrm { N T K } , K \mathbf { u } \simeq$ Λu, with weak mixing into other directions, and its dynamics are driven mainly by the residual component along u. Appendix C.1 states these conditions explicitly. Section 3 later shows the corresponding Fourier organization empirically. Projecting the hierarchy dynamics onto u then gives

$$
\dot { \sigma } \simeq - ( \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } y , \qquad \dot { \Lambda } \simeq - a \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda .\tag{10}
$$

Here, $a ( \pmb \theta )$ is the local coupling between the projected residual and the residual-driven change in $\Lambda ,$ as defined in Appendix C.1. We consider $a ( \pmb \theta ) > 0$ , qualitatively consistent with the observed post-fit emergence of task-aligned NTK structure. It implies that correcting the output along u tends to increase the NTK strength in the same direction. This sign is a dynamical condition rather than a consequence of homogeneity or symmetry. Equation (10) then describes a feedback. For $\sigma < 0$ larger Λ accelerates residual relaxation, while the remaining residual drives Λ upward.

We further assume that $a ( \pmb \theta )$ is locally constant over the post-fit interval considered. Appendix C.1 shows that once the residual is close to its instantaneous ridge value, weight decay does not introduce an additional fast timescale for $^ { a , }$ supporting this approximation in the slow regime considered below. After the fast ridge relaxation, $\sigma < 0$ for $y > 0 , \mathrm { s o } - a \sigma > 0$ and the residual increases the NTK strength along the selected task direction. Under the locally constant-a approximation, the fixed points of Eq. (10) and their stability can be characterized exactly.

Theorem 2 (Spectral selection in the reduced tangent-mode system). Under the locally constant-a approximation,for $\lambda _ { W } ^ { \mathrm { n o r m } } > 0 , a > 0 ,$ , and $y \ge 0 , E q$ . (10) has one equilibrium with $\Lambda \geq 0 .$

$$
\Lambda _ { * } = - \lambda _ { W } ^ { \mathrm { n o r m } } + \sqrt { \lambda _ { W } ^ { \mathrm { n o r m } 2 } + a y } , \qquad \sigma _ { * } = - \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a } \Lambda _ { * } .\tag{11}
$$

The equilibrium is locally asymptotically stable. It satisfies $\Lambda _ { * } = 0 f o r y = 0$ and $\Lambda _ { * } > 0 f o r y > 0$

We give the fixed-point and stability calculations in Appendix C.2. Within the reduced system, unsupported directions decay, while positive residual-to-NTK coupling sustains NTK strength for target-aligned directions. Under the adiabatic approximation that residual relaxation is faster than mode motion, we set $\dot { \sigma } \simeq 0$ at the current Λ and obtain

$$
\sigma _ { \mathrm { a d } } ( \Lambda ) = - \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } y } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } , \qquad \dot { \Lambda } = 2 \lambda _ { W } ^ { \mathrm { n o r m } } \left( \frac { a y } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } - \Lambda \right) .\tag{12}
$$

This shows that mode evolution occurs on a normalized-time scale $1 / \lambda _ { W } ^ { \mathrm { n o r m } }$ . Since $t \simeq \mathcal { N } \eta s$ after s updates, we define the normalization-invariant slow time

$$
\tau : = \lambda _ { W } ^ { \mathrm { n o r m } } t \simeq \eta \lambda _ { W } s .\tag{13}
$$

Here, s is the optimizer update index and $\eta \lambda _ { W }$ is the per-update decay scale. $\operatorname { I f } s _ { \mathrm { f i t } }$ denotes the end of the initial fitting stage, the subsequent slow-time increment is $\Delta \tau \simeq \eta \lambda _ { W } ( s - s _ { \mathrm { f i t } } )$ . Thus the reduced dynamics predict the leading scaling of post-fit transition times with $( \eta \lambda _ { W } ) ^ { - 1 }$

## 2.3 GROKKING BOUNDARIES IN THE $( \eta , \lambda _ { W } )$ PLANE

The same adiabatic calculation shows how the task-aligned NTK strength Λ controls how much of the corresponding target component is expressed in the network output. Since $\boldsymbol { \sigma } = \mathbf { u } ^ { \top } ( \mathbf { f } - \mathbf { y } )$ , the projected output coefficient $\bar { f } =  { \mathbf { u } } ^ { \top } \mathbf { f }$ satisfies

$$
f = y + \sigma _ { \mathrm { a d } } ( \Lambda ) = g ( \Lambda ) y , \quad g ( \Lambda ) = \frac { \Lambda } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } .\tag{14}
$$

Hence, increasing Λ along a task-aligned mode increases the corresponding component of the training outputs continuously. This training coordinate is related to generalization through the evolution of held-out margins, the difference between the correct-class logit and the largest competing logit, along the same trajectory. Because modular addition is represented by several matched Fourier components as shown in Eq. (2), these margins can change smoothly as the corresponding components grow, while the predicted label changes only when a margin crosses zero. If many held-out examples cross that threshold at similar mode strengths, smooth spectral growth can therefore produce a sharp rise in validation accuracy. Appendix D.1 formalizes this connection using the empirical distribution of the mode strengths at which individual held-out examples change class.

The scalar dynamics in Eq. (10) track a single task-aligned Fourier mode and do not resolve the individual held-out margins that produce the sharp accuracy transition. We therefore now replace the collection of margin crossings by a threshold $\Lambda _ { \mathrm { g } }$ on the task-aligned NTK strength Λ. Reaching $\Lambda _ { \mathrm { g } }$ represents reaching a chosen held-out-accuracy threshold. We denote the NTK strength at the beginning of the slow stage by $\Lambda _ { 0 }$ . Further, we consider a total of $T$ training updates, and recall the stable equilibrium $\Lambda _ { * }$ <sub>∗</sub> from Eq. (11). Since $\operatorname { E q . } \left( 1 2 \right)$ gives $\dot { \Lambda } > 0$ for $\Lambda \ < \ \Lambda _ { * } ,$ a trajectory with $\Lambda _ { 0 } < \bar { \Lambda } _ { \mathrm { g } } < \Lambda$ <sub>∗</sub> crosses the threshold $\Lambda _ { \mathrm { g } }$ in finite time. Separating variables in $\operatorname { E q . } \left( \mathrm { i } 2 \right)$ and integrating then gives the boundary in the $( \eta , \lambda _ { W } )$ plane between trajectories that reach $\Lambda _ { \mathrm { g } }$ within $T$ updates and those that remain below it.

Corollary 1 (Finite-training grokking boundary). Under the one-mode, adiabatic, and gradientflow approximations,

$$
\eta _ { * } ( \lambda _ { W } ) = \frac 1 { 2 \lambda _ { W } T } \int _ { \Lambda _ { 0 } } ^ { \Lambda _ { \mathrm { g } } } \frac { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a y - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } d \Lambda .\tag{15}
$$

At small normalized decay and away from the fixed-point boundary, the leading dependence is $\eta _ { * }$ ∝ $1 / \lambda _ { W }$

A derivation and closed form solution of this integral can be found in Appendices D.2– D.3. Equation (15) predicts that increasing $\Lambda _ { 0 }$ removes a positive portion of the crossing-time integral and thereby shortens the delay before grokking occurs. We elaborate on this prediction and provide an empirical test in Appendices $\mathrm { F . 1 - } \bar { \mathrm { F } } . 2$ . This finite-training grokking boundary is obtained under the assumption that the threshold is reachable in the first place, $\Lambda _ { \mathrm { g } } < \Lambda _ { * }$ . Whether or not this is possible follows from the limiting case $\Lambda _ { \mathrm { g } } = \Lambda _ { * }$ , which gives

$$
\lambda _ { W } ^ { \mathrm { m o d e } } = \mathcal { N } \frac { a y - \Lambda _ { \mathrm { g } } ^ { 2 } } { 2 \Lambda _ { \mathrm { g } } } ,\tag{16}
$$

provided $a y > \Lambda _ { \mathrm { g } } ^ { 2 }$ . For $\lambda _ { W } < \lambda _ { W } ^ { \mathrm { m o d e } }$ , the equilibrium lies above $\Lambda _ { \mathrm { g } }$ so the mode can reach $\Lambda _ { \mathrm { g } }$ in finite time, with Corollary 1 determining whether this occurs within T updates. For $\lambda _ { W } \geq \lambda _ { W } ^ { \mathrm { m o d e } }$ the threshold cannot be crossed in finite time. As λ approaches this cutoff from below, the net growth rate of the task-aligned mode at $\Lambda = \Lambda _ { \mathrm { g } }$ approaches zero, causing $\eta _ { * } ( \lambda _ { W } )$ to diverge logarithmically as $\lambda _ { W } \to \lambda _ { W } ^ { \mathrm { m o d e } }$ (Appendix D.3). This shows that as $\lambda _ { W }$ approaches $\lambda _ { W } ^ { \mathrm { m o d e } }$ from below, reaching $\Lambda _ { \mathrm { g } }$ and thereby generalizing within a fixed update budget requires an increasingly large learning rate. In the $( \eta , \lambda _ { W } )$ plane, this produces an upward turn of the finite-time boundary near the cutoff at $\lambda _ { W } ^ { \mathrm { m o d e } }$

A further constraint arises from fitting the training data. This boundary depends on the relative gains of multiple output modes and is therefore not fixed exactly by the one-mode dynamics. To retain a scalar description, we summarize the onset of fitting failure by the gain of an effective training mode. Requiring $g ( \Lambda _ { \mathrm { f i t } } ) \geq g _ { \mathrm { m i n } }$ gives

$$
\lambda _ { W } \leq \mathcal { N } \frac { \Lambda _ { \mathrm { f i t } } ( 1 - g _ { \mathrm { m i n } } ) } { 2 g _ { \mathrm { m i n } } } ,\tag{17}
$$

This provides an approximately vertical effective decay scale beyond which the network no longer reaches the chosen training-accuracy criterion.

Finally, sufficiently large learning rates encounter the discrete-time stability edge of the approximately fixed-kernel dynamics. As derived in Appendix D.4, linear stability requires

$$
\eta < \frac { 2 } { \sqrt { \Lambda _ { \operatorname* { m a x } } ( K ) + 2 \lambda _ { W } } } ,\tag{18}
$$

where $\Lambda _ { \operatorname* { m a x } } ( K )$ is the largest eigenvalue of the frozen NTK. This is the frozen-kernel analogue of the usual edge-of-stability condition (Cohen et al., 2021). Because the NTK subsequently evolves, it provides only a local estimate of the large-η boundary of the full training dynamics. Appendices D.3–D.4 give the derivations and validity conditions for the fitting and stability bounds.

To summarize, these constraints predict the following organization of the $( \eta , \lambda _ { W } )$ plane. When the training data are fitted and $\Lambda _ { \mathrm { g } }$ is reachable but the finite-training condition in Corollary 1 is not satisfied, the network remains in the memorization regime over the available training interval. Grokking is possible when the training fit succeeds (Eq. (17)), the task-aligned NTK strength can reach $\Lambda _ { \mathrm { g } }$ (Eq. (16)), the threshold is crossed within $\dot { T }$ updates (Eq. (15)), and gradient descent remains stable (Eq. (18)). Increasing weight decay can eventually prevent either task-mode growth or training fit, while sufficiently large learning rates produce a separate instability boundary.

## 3 EXPERIMENTS

We now test these predictions and measure post-fit NTK reorganization as well as the $( \eta , \lambda _ { W } )$ )-plane boundaries, and whether similar transition-time behavior appears in Transformers. Appendix G provides experimental details, and all results can be reproduced from our GitHub page<sup>2</sup>.

## 3.1 TASK-ALIGNED NTK STRUCTURE DEVELOPS AFTER FITTING

We trained a bias-free one-hidden-layer ReLU MLP on addition modulo 97, using half of the samples for training and the remainder for evaluation. We computed the empirical NTK, projected it onto the true-label output of each sample, and averaged the resulting entries by output label. This isolates tangent feature organization with respect to the task labels rather than individual examples. To further isolate structure from changes in overall NTK scale, we normalized the resulting $9 7 \times 9 7$ kernel. Figure 2a shows that periodic structure in this kernel strengthens between updates 3,000 and 20,000, while Figure 2b shows that training accuracy has already saturated during much of this evolution. In Figure 2c, the leading eigenvectors dynamics show the emergence of standing-wave profiles in label space, consistent with the Fourier task structure in Eq. (2). Their continued organization after fitting is also qualitatively consistent with the post-fit growth of task-aligned NTK modes described by Eq. (10).

## 3.2 GROKKING BOUNDARIES IN THE MLP OPTIMIZER PLANE

Next, we trained an (84×90)-grid of MLPs for varying $( \eta , \lambda _ { W } )$ on addition modulo 23 for a fixed total of 24,000 updates. Figure 3 shows the resulting $( \eta , \lambda _ { W } )$ -plane, categorized into memorizing, grokking, forgetting, and non-fitting runs, as described in Appendix G.2. At small $\eta \lambda _ { W }$ , the taskaligned NTK strength Λ does not have enough time to reach the generalization threshold $\Lambda _ { \mathrm { g } }$ within the fixed training budget, producing the broad memorization region in Figure 3a. The red curve fits the finite-training grokking boundary from Eq. (15). Note that since the fit-parameters are inferred from the boundary, they are not direct measurements of Λ(t). Hence, this comparison tests the predicted optimizer dependence of the crossing time rather than the absolute amount of task-aligned NTK growth. At weak decay, it follows the transition from memorization to grokking and closely tracks the yellow $\eta \propto \lambda _ { W } ^ { - 1 }$ fit. As λ increases, the finite-time curve turns upward toward the blue mode-reachability cutoff from Eq. (16), beyond which the task-aligned NTK strength cannot reach the required threshold. The green line shows the training-fit cutoff from Eq. (17), while the black line corresponds to the frozen-kernel stability condition in Eq. (18), fitted to the observed large-learning-rate boundary. The full fitting procedures for all curves are given in Appendix G.2.1.

Figure 3b tests the predicted slow-time scaling. For every run that reaches 90% held-out accuracy, we plot the first update $s _ { 1 0 }$ at which that accuracy reaches 10% against $( \eta \lambda _ { W } ) ^ { - 1 }$ . Equation (13) predicts $s _ { 1 0 } \propto ( \eta \lambda _ { W } ) ^ { - 1 }$ if the transition is governed by the slow time $\tau \simeq \eta \lambda _ { W } s$ . Indeed, the observed crossing times follow this scaling across the grokking region.

![](images/ed15f3303c4cfce4a76c9da02087a89094bc0fb109886bc89996e33314e82af5.jpg)  
Figure 2: Post-fit change in the normalized label-space NTK for a bias-free MLP on addition modulo 97. (a) We project the sample NTK onto the true-label, average by output label, and normalize the resulting $9 7 \times 9 7$ label kernel. (b) Training and held-out accuracy for the same run. (c) Four leading eigenvectors of the same normalized label kernel, showing Fourier-like standing waves that sharpen after training accuracy is already high.

## 3.3 GROKKING BOUNDARIES AND TRANSITION-TIME SCALING IN A TRANSFORMER

Biases, LayerNorm, residual paths, and attention all break the global homogeneity used in Theorems 1 and 2. However, differentiating $K = J J ^ { \top }$ under gradient flow still yields an exact residualdependent contribution to $\dot { K }$ for any differentiable model (Appendix E). We therefore tested whether the homogeneous predictions also hold in a non-homogeneous case, and trained a (42×45)-grid of Transformers. In line with the MLP phase diagram, Figure 4a shows a weak-decay memorization region, an intermediate grokking band, and failure at stronger decay or large learning rate. The fitted curve of the form $\eta = A / \lambda _ { W }$ describes the grokking-memorization boundary well, in line with the homogeneous theory. Additionally, within the grokking networks, Figure 4b exhibits the same linear scaling of the first held-out 80% crossing $s _ { 8 0 }$ with $( \eta \bar { \lambda } _ { W } ) ^ { - 1 }$ . We use 80% here because early 10% accuracy crossings are often reversible in practice. This suggests that the optimizer-level phase and transition-time scaling may extend beyond exact homogeneity.

## 4 LIMITATIONS AND OUTLOOK

Our theory holds for homogeneous networks trained with squared loss, coupled weight decay, and full-batch gradient flow. Extending the reduced dynamics to more general objectives and architectures is a natural next step. The Transformer experiments already show that similar macroscopic optimizer-space structure can persist without exact homogeneity, but the corresponding microscopic dynamics remain open. Further, the scalar reduction follows a single decoupled task direction. Although the natural Fourier basis of modular addition motivates this, interactions among task-relevant components are neglected, and more general datasets need not provide an equally natural spectral basis. Future work should therefore extend the dynamics to interacting task directions and identify suitable coordinates beyond cyclic modular arithmetic. In this direction, Marchetti et al. (2026) show that for sequential group-composition tasks, networks learn irreducible representations one at a time, suggesting a route from the present one-mode reduction toward multiple representation theoretic components. Finally, modular addition and grokking provide explicit task structure and separate fitting from generalization, but parts of our mechanism are more general. The fixed-kernel residual result does not rely on modular arithmetic, and the residual-dependent contribution to NTK evolution persists for differentiable models. This raises the question of whether our framework also applies to broader representation-learning problems. The J-space framework of Gurnee et al. (2026) provides a possible connection, using an activation-to-output Jacobian to identify representations with strong downstream influence. The NTK similarly uses the parameter Jacobian to track taskaligned directions during training. Establishing whether these directions emerge together would connect the optimization dynamics studied here to the broader formation of internal representations used for generalization.

![](images/6af9fc02c15d5f65837aeedde10a77d70010221930a31244f952e29196032e3a.jpg)

![](images/9c8c9d65637903930e144513649ce54b87341793c8e2fa56257b9a1241e021ac.jpg)

Figure 3: Optimizer-plane structure for the homogeneous MLP on addition modulo 23. (a) Outcomes after 24,000 updates on an $8 4 \times 9 0$ learning-rate/decay grid. The overlays show the calibrated finite-time boundary, a separately fitted inverse-decay relation, the mode-reachability cutoff, the training-fit cutoff, and the calibrated frozen-kernel stability form. (b) First held-out 10% crossing among runs that eventually meet the 90% grokking criterion, plotted against $( \eta \lambda _ { W } ) ^ { - 1 }$ . The dashed segment indicates the predicted scaling $\bar { s } _ { 1 0 } \propto ( \bar { \eta \lambda } _ { W } ) ^ { - 1 }$ and is not fitted to the data.  
![](images/18e72cadc03155018e05143e7e864c4252ca8a8b63e3ee4d697deafcf6f5c4bb.jpg)

![](images/c614cc503edd6a98cb95bd05b3a593aeb2897d9668b0b4bcb7adab16cbf5b9ef.jpg)  
Figure 4: Phase and transition-time measurements for a one-block Transformer on the same modulo-23 split. (a) Outcomes on a $4 2 \times 4 5$ optimizer grid with an inverse-decay fit to the main memorization–grokking boundary. (b) First held-out 80% crossing among runs that ultimately grok at the 90% criterion, plotted against $( \eta \lambda _ { W } ) ^ { - 1 }$ . The dashed segment indicates the scaling $\bar { s } _ { 8 0 } \propto ( \eta \lambda _ { W } ) ^ { - 1 }$

## 5 CONCLUSIONS

We developed a spectral theory of grokking in which coupled weight decay leaves a finite post-fit residual, with target components at smaller NTK eigenvalues retaining a larger residual fraction. We used the neural tangent hierarchy to show that this residual drives task-aligned NTK growth. The resulting slow dynamics determine whether task-aligned NTK strength can grow sufficiently far and sufficiently quickly within the training budget, while training fit and discrete-time stability impose additional constraints. Empirically, a homogeneous MLP on modular addition shows continued Fourier organization of the NTK after training accuracy saturates, an MLP hyperparameter sweep recovers the predicted $( \eta , \lambda _ { W } )$ phase geometry and $( \eta \dot { \lambda } _ { W } ) ^ { - 1 }$ transition-time scaling, and a one-block Transformer reproduces the same macroscopic phase structure and transition-time scaling despite lacking exact homogeneity. Together, these results suggest that residual error left after fitting can actively shape how and when neural networks acquire generalizing representations, connecting feature learning, generalization time, and optimizer-space structure within a single mechanistic framework.

## REPRODUCIBILITY STATEMENT

We provide the proofs and derivations of the homogeneous results in Appendices B.1–D.4 and the differentiable-model identities in Appendix E.1. Appendix F.2 records the complete paired intervention, including architectures, seeds, splits, losses, optimizers, kernel normalization, and reporting convention. Appendices G.1–G.4 give the corresponding details for the spectral diagnostic, MLP and Transformer phase sweeps, boundary calibration, and reduced-mode illustration. The accompanying code<sup>3</sup> centralizes the configurations and provides training and analysis entry points.

## AI USE STATEMENT

AI was used to assist with text editing, coding, and with literature research. AI was not involved in research ideation, the development of theoretical results, research methodology, and experimental design. All AI-assisted outputs were reviewed, revised where necessary, and verified by the authors.

## REFERENCES

Aristide Baratin, Thomas George, Cesar Laurent, R. Devon Hjelm, Guillaume Lajoie, Pascal Vin-´ cent, and Simon Lacoste-Julien. Implicit regularization via neural feature alignment. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 2269–2277, 2021.

Alon Beck, Noam Itzhak Levi, and Yohai Bar-Sinai. Grokking at the edge of linear separability. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 3307–3334, 2025.

Francesco Bertolotti and Walter Cazzola. A survey on grokking. ACM Computing Surveys, 58(13): 327:1–327:25, 2026. doi: 10.1145/3814603.

Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=jh-rTtvkGeM.

Xiaoman Delores Ding, Zifan Carl Guo, Eric J. Michaud, Ziming Liu, and Max Tegmark. Survival of the fittest representation: A case study with modular addition. arXiv preprint arXiv:2405.17420, 2024.

Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. Verbalizable representations form a global workspace in language models. arXiv preprint arXiv:2607.15495, 2026.

Jianliang He, Leda Wang, Siyu Chen, and Zhuoran Yang. On the mechanism and dynamics of modular addition: Fourier features, lottery ticket, and grokking. arXiv preprint arXiv:2602.16849, 2026.

Jiaoyang Huang and Horng-Tzer Yau. Dynamics of deep neural networks and neural tangent hierarchy. In Proceedings of the 37th International Conference on Machine Learning, 2020.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, 2018.

Alan Jeffares, Alicia Curth, and Mihaela van der Schaar. Deep learning through a telescoping lens: A simple model provides empirical insights on grokking, gradient boosting and beyond. In Advances in Neural Information Processing Systems, volume 37, 2024.

Tikeng Notsawo Pascal Junior, Guillaume Dumas, and Guillaume Rabusseau. Grokking beyond the euclidean norm of model parameters. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 28552–28618, 2025.

Taeyoung Kim. Grokking on the weight-decay clock: A rate hierarchy from softly broken symmetries. arXiv preprint arXiv:2607.23967, 2026.

Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan. Grokking as the transition from lazy to rich training dynamics. In International Conference on Learning Representations, 2024.

Noam Itzhak Levi, Alon Beck, and Yohai Bar-Sinai. Grokking in linear estimators: A solvable model that groks without understanding. In International Conference on Learning Representations, 2024.

Jennifer Lin. Feature identification via the empirical ntk. arXiv preprint arXiv:2510.00468, 2025.

Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams. To wards understanding grokking: An effective theory of representation learning. In Advances in Neural Information Processing Systems, 2022.

Ziming Liu, Eric J. Michaud, and Max Tegmark. Omnigrok: Grokking beyond algorithmic data. In International Conference on Learning Representations, 2023a.

Ziming Liu, Ziqian Zhong, and Max Tegmark. Grokking as compression: A nonlinear complexity perspective. arXiv preprint arXiv:2310.05918, 2023b.

Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon Shaolei Du, Jason D. Lee, and Wei Hu. Dichotomy of early and late phase implicit biases can provably induce grokking. In International Conference on Learning Representations, 2024.

Neil Rohit Mallinar, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin. Emergence in non-neural models: Grokking modular arithmetic via average gradient outer product. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 42834–42856, 2025.

Giovanni Luca Marchetti, Daniel Kunin, Adele Myers, Francisco Acosta, and Nina Miolane. Sequential group composition: A window into the mechanics of deep learning. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=oFD9pc53pF.

William Merrill, Nikolaos Tsilivis, and Aman Shukla. A tale of two circuits: Grokking as competition of sparse and dense subnetworks. ICLR Workshop on Mathematical and Empirical Understanding ofFoundation Models, 2023.

Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. In Advances in Neural Information Processing Systems, volume 36, pp. 28699–28722, 2023.

Jack William Miller, Charles O’Neill, and Thang D. Bui. Grokking beyond neural networks: An empirical exploration with model complexity. Transactions on Machine Learning Research, 2024.

Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, and Danica J. Sutherland. Why do you grok? a theoretical analysis on grokking modular addition. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 35934–35967, 2024.

Prudhviraj Naidu, Zixian Wang, Leon Bergen, and Ramamohan Paturi. Quiet feature learning in algorithmic tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 37756–37764, 2026.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations, 2023.

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, and Tolga Birdal. Grokking at the edge of numerical stability. In International Conference on Learning Representations, 2025.

Noa Rubin, Inbar Seroussi, and Zohar Ringel. Grokking as a first order phase transition in two layer networks. In International Conference on Learning Representations, 2024.

Javier Sanguino Bautiste, Gregor Bachmann, Bobby He, Lorenzo Noci, and Thomas Hofmann. Feature learning dynamics under grokking in a sparse parity task. ICML 2024 Workshop on High-dimensional Learning Dynamics: The Emergence ofStructure and Reasoning, 2024.

Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua M. Susskind. The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon. In Has it Trained Yet? Workshop at NeurIPS, 2022.

Yuandong Tian. Provable scaling laws of feature emergence from learning dynamics of grokking. In International Conference on Learning Representations, 2026.

Vikrant Varma, Rohin Shah, Zachary Kenton, Janos Kramar, and Ramana Kumar. Explaining grokking through circuit efficiency. In International Conference on Learning Representations, 2024.

Mingyue Xu, Gal Vardi, and Itay Safran. To grok grokking: Provable grokking in ridge regression. arXiv preprint arXiv:2601.19791, 2026.

Xingyu Zheng, Kyle Daruwalla, Ari S. Benjamin, and David Klindt. Delays in generalization match delayed changes in representational geometry. In Proceedings of UniReps: the Second Edition of the Workshop on Unifying Representations in Neural Models, volume 285 of Proceedings of Machine Learning Research, pp. 324–334, 2024.

## A RELATED MECHANISMS FOR DELAYED GENERALIZATION

Grokking has been explained through circuit formation, implicit bias, tangent-feature motion, fixed spectra, and optimizer state. Surveys emphasize that several mechanisms can coexist across losses, architectures, and training regimes (Bertolotti & Cazzola, 2026). We position our account by asking which dynamical object changes during the plateau and which observations distinguish that change from a slow fixed-feature relaxation.

## A.1 MECHANISTIC PROGRESS AND REPRESENTATION COMPETITION

Modular arithmetic provides an interpretable Fourier basis for circuit analysis. Nanda et al. (Nanda et al., 2023) reverse-engineer a trigonometric algorithm in a grokked Transformer and introduce progress measures that reveal gradual circuit formation during the apparent plateau. Varma et al. (Varma et al., 2024) explain delayed generalization through competition between memorizing and generalizing circuits, while Merrill et al. (Merrill et al., 2023) study competition between sparse and dense subnetworks. Ding et al. (Ding et al., 2024) track competing circular Fourier representations with low-dimensional dynamics. He et al. (He et al., 2026) analyze how frequency competition, phase alignment, and weight decay shape Fourier-feature formation in two-layer networks on modular addition. Across a broader set of algorithmic Transformer tasks, Naidu et al. (Naidu et al., 2026) find causally relevant features developing during long plateaus before the output loss improves. Other representation-level studies connect grokking to changes in norms, learned features, and compression (Liu et al., 2022; 2023a;b). We use the same task-defined Fourier basis bu study the residual that drives tangent-feature motion within it.

## A.2 KERNEL-TO-FEATURE TRANSITIONS AND EMPIRICAL NTK DYNAMICS

Several theories describe grokking as a transition from an early linearized regime to later feature learning. Kumar et al. (Kumar et al., 2024) use a controllable laziness parameter to induce or remove grokking in polynomial regression, MNIST, and modular addition. Lyu et al. (Lyu et al., 2024) prove a separation between early kernel-like and late implicit biases in homogeneous networks with weight decay. Mohamadi et al. (Mohamadi et al., 2024) show that an early permutation-equivariant kernel regime can be sample inefficient for modular addition, whereas bounded-norm feature-learning solutions generalize from fewer examples. Rubin et al. (Rubin et al., 2024) analyze an adaptive kernel and relate grokking in two-layer networks to a first-order phase transition. Tian (Tian, 2026) derives a sequence of lazy, independent-feature, and interacting-feature stages with scaling laws for feature emergence and generalization.

Empirical NTKs distinguish departure from the lazy regime from the task content acquired by the kernel. On sparse parity, Sanguino Bautiste et al. (Sanguino Bautiste et al., 2024) observe leading NTK eigenfunctions moving from non-predictive directions toward predictive features as generalization emerges. On image classification, Zheng et al. (Zheng et al., 2024) find substantial empirical-NTK movement before delayed test improvement and closer synchronization between representational geometry and generalization. For modular arithmetic, Lin (Lin, 2025) shows that leading empirical-NTK eigenspaces align with Fourier feature families in trained MLPs and Transformers and that the alignment changes through grokking. These results establish feature motion but do not identify its source. We derive a residual-dependent equation for task-aligned kernel components.

Equation (8) inserts the post-fit residual into $\dot { K }$ , and the Fourier projection yields the reduced feedback, slow time, and finite-training boundary. This mechanism is also compatible with dynamic alignment between finite-network tangent features and task directions (Baratin et al., 2021).

## A.3 STATIC-FEATURE GROKKING AND THE WEIGHT-DECAY TRANSITION-TIME SCALING

Fixed-feature models show that delayed generalization can occur without representation change. Levi et al. (Levi et al., 2024) analyze grokking in linear estimators, Miller et al. (Miller et al., 2024) document related non-neural behavior, and Beck et al. (Beck et al., 2025) study long delays near linear separability. Xu et al. (Xu et al., 2026) prove end-to-end grokking for overparameterized ridge regression trained with gradient descent and weight decay. Their bounds separate early training fit from later population improvement and express the delay in terms of learning rate, decay, sample size, feature dimension, and initialization. Kim (Kim, 2026) derives a complementary late-time relaxation in linear models with weight decay and heavy-ball optimization; the zero-momentum limit again yields $1 / ( \eta \lambda )$ scaling.

The transition-time scaling in Eq. (13) is therefore shared by static-feature and feature-learning mechanisms. Our finite-network account adds the empirical NTK as a state variable and predicts task-aligned motion after training accuracy saturates. The normalized kernel measurement, the residual-dependent hierarchy term, and the directional intervention separate this claim from a fixed-feature delay.

Feature learning can also arise outside ordinary neural-network gradient descent. Mallinar et al. (Mallinar et al., 2025) demonstrate grokking in recursive feature machines driven by an average gradient outer product; block-circulant task features emerge after training loss has vanished. Jeffares et al. (Jeffares et al., 2024) analyze grokking through a sequence of local linear approximations, providing another time-resolved view of representation change.

## A.4 JACOBIAN REPRESENTATIONS AND EMERGENT CAPABILITIES

Jacobian geometry can isolate directions with direct behavioral leverage. Gurnee et al. (Gurnee et al., 2026) introduce the Jacobian lens and identify a small “J-space” of verbalizable representations in language models; interventions show that these directions can mediate downstream reasoning and reporting. Their Jacobian transports downstream activations to outputs, whereas our NTK uses the parameter Jacobian, but both approaches focus on directions that couple strongly to behavior. We study how such task-relevant parameter-Jacobian directions form during training.

The quantization model of neural scaling (Michaud et al., 2023) proposes that knowledge and skills may be acquired in discrete modules even when aggregate loss follows smooth power laws. Grokking offers a small setting in which abrupt behavior can be resolved in time. Equation (58) gives one route from a smooth internal coordinate to a sharp output transition, so behavioral discontinuity need not imply discontinuous internal learning. At the same time, task-aligned modes provide a concrete object for testing more modular accounts of capability acquisition.

## A.5 REGULARIZATION AND OPTIMIZATION MECHANISMS

Regularization mechanisms extend beyond Euclidean weight decay. Junior et al. (Junior et al., 2025) show that sparsity, low-rank penalties, implicit regularization, and depth can induce grokking-like transitions. Adaptive optimizers can produce late-training instabilities (Thilak et al., 2022), while cross-entropy training can involve numerical and logit-scaling effects (Prieto et al., 2025). The relevant slow variable changes with the objective and optimizer. We focus on homogeneous squaredloss training because coupled $L _ { 2 }$ decay leaves the ridge residual in Theorem 1 and directly contracts the two-homogeneous empirical NTK in Eq. (8).

## A.6 RELATION TO FIXED-FEATURE AND CIRCUIT-LEVEL ACCOUNTS

Our account links the residual left by an early approximately fixed kernel to later tangent-feature motion in a task-defined Fourier direction. The resulting feature strength determines whether heldout margins are reachable and how much optimizer time is required to reach them. Fixed-feature analyses explain how weight decay can create a long transition-time without changing the representation; circuit analyses describe the algorithm implemented after generalization. Residual-dependent neural-tangent-hierarchy dynamics connect the two by specifying how post-fit supervision can reshape the tangent geometry from which a generalizing circuit is built.

## B EXACT HOMOGENEOUS FUNCTION-SPACE DYNAMICS

For reference, the notation used throughout the derivations is summarized below.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> ${ \pmb { \sigma } } = \mathbf { f } - \mathbf { y }$ </td><td>stacked training residual</td></tr><tr><td> $J = \nabla _ { \theta } \mathbf { f }$ </td><td>training-set parameter Jacobian</td></tr><tr><td> $K = J J ^ { \top }$ </td><td>empirical neural tangent kernel</td></tr><tr><td> $D$ </td><td>degree of parameter homogeneity</td></tr><tr><td> $\mathcal { N }$ </td><td>global normalization multiplying  $\left\| \sigma \right\| ^ { 2 } / 2$  in the implemented loss</td></tr><tr><td> $\lambda _ { W }$ </td><td>optimizer coupled  $L _ { 2 }$  weight-decay coefficient</td></tr><tr><td> $\lambda _ { W } ^ { \mathrm { n o r m } } = \lambda _ { W } / \mathcal { N }$ </td><td>normalized decay used in continuous time</td></tr><tr><td> $\eta$ </td><td>optimizer gradient-descent learning rate</td></tr><tr><td>t</td><td>normalized gradient-flow time,  $t = \mathcal { N } t _ { \mathrm { o p t } }$ </td></tr><tr><td> $s$ </td><td>discrete optimizer update index</td></tr><tr><td> $\Lambda$ </td><td>task-direction NTK strength  $\mathbf { u } ^ { \top } K \mathbf { u }$ </td></tr><tr><td> $a$ </td><td>projected residual-to-NTK coupling coefficient</td></tr><tr><td> $_ y$ </td><td>target coefficient in the selected task direction</td></tr><tr><td> $_ { T } ^ { \mathrm { { ( . _ { g } } } }$ </td><td>reduced strength representing a held-out criterion</td></tr><tr><td></td><td>finite optimizer update budget</td></tr></table>

Table 1: Notation used in the main derivations.

## B.1 PROOF OF THEOREM 1

We absorb the global data-loss prefactor into the normalized time introduced in Section 2.1. The data-gradient term then has unit coefficient, and optimizer weight decay appears through $\lambda _ { W } ^ { \mathrm { n o r m } }$

$$
\dot { \pmb \theta } = - J ^ { \top } \pmb \sigma - \lambda _ { W } ^ { \mathrm { n o r m } } \pmb \theta ,\tag{19}
$$

$$
\dot { \pmb { \sigma } } = J \dot { \pmb \theta } = - J J ^ { \top } \pmb { \sigma } - \lambda _ { W } ^ { \mathrm { n o r m } } J \pmb \theta = - K \pmb \sigma - \lambda _ { W } ^ { \mathrm { n o r m } } J \pmb \theta .\tag{20}
$$

For a D-homogeneous network, uniform rescaling of all trainable weights rescales the output by degree D. Differentiating this identity at unit scale gives the Euler relation that closes the parameterspace decay term in function space.

$$
\mathbf { f } _ { c \pmb { \theta } } = c ^ { D } \mathbf { f } _ { \pmb { \theta } } ,\tag{21}
$$

$$
\left. \frac { d } { d c } { \bf f } _ { c \theta } \right| _ { c = 1 } = J \theta = D { \bf f } = D ( \pmb { \sigma } + { \bf y } ) .\tag{22}
$$

Substituting this relation removes the remaining explicit dependence on θ.

$$
\dot { \pmb { \sigma } } = - \big ( \boldsymbol { K } + D \lambda _ { W } ^ { \mathrm { n o r m } } I \big ) \pmb { \sigma } - D \lambda _ { W } ^ { \mathrm { n o r m } } \mathbf { y } ,\tag{23}
$$

which is Eq. (5).

For fixed $K$ , the residual equation is affine and linear. The matrix $M : = K + D \lambda _ { W } ^ { \mathrm { n o r m } } I$ is positive definite for $\lambda _ { W } ^ { \mathrm { n o r m } } > 0$ , even when K has null directions, so every residual component relaxes exponentially to a unique fixed point. We set

$$
M : = K + D \lambda _ { W } ^ { \mathrm { n o r m } } I \succ 0 , \qquad \pmb { \sigma } _ { \ast } : = - D \lambda _ { W } ^ { \mathrm { n o r m } } M ^ { - 1 } \mathbf { y } ,
$$

and the solution can be written as

$$
\dot { \pmb { \sigma } } = - M ( \pmb { \sigma } - \pmb { \sigma } _ { \ast } ) ,\tag{24}
$$

$$
\pmb { \sigma } ( t ) - \pmb { \sigma } _ { \ast } = e ^ { - M t } \big ( \pmb { \sigma } ( 0 ) - \pmb { \sigma } _ { \ast } \big ) .\tag{25}
$$

We diagonalize K to expose the ridge form. Each eigenmode follows an independent scalar relaxation, and the ratio of its equilibrium residual to its target coefficient is determined by $\Lambda _ { j }$ relative to $D \lambda _ { W } ^ { \mathrm { n o r m } }$ . For $K \mathbf { u } _ { j } = \Lambda _ { j } \mathbf { u } _ { j }$ ,

$$
\dot { \sigma } _ { j } = - \big ( \Lambda _ { j } + D \lambda _ { W } ^ { \mathrm { n o r m } } \big ) \sigma _ { j } - D \lambda _ { W } ^ { \mathrm { n o r m } } y _ { j } ,\tag{26}
$$

$$
\sigma _ { j , * } = - \frac { D \lambda _ { W } ^ { \mathrm { n o r m } } } { \Lambda _ { j } + D \lambda _ { W } ^ { \mathrm { n o r m } } } y _ { j } .\tag{27}
$$

For ReLU networks, these identities hold on every interval on which the activation pattern is fixed. At activation-boundary crossings the network remains continuous and the gradient-flow equation holds almost everywhere, so the same function-space relation extends piecewise across the trajectory.

This proves Theorem 1.

## C ONE-MODE NEURAL-TANGENT-HIERARCHY CLOSURE

## C.1 ASSUMPTIONS BEHIND THE MODAL CLOSURE

For the normalized Fourier direction u used in the main text, define

$$
\begin{array} { r } { q _ { \theta } : = \mathbf { u } ^ { \top } \mathbf { f } _ { \theta } , \qquad \boldsymbol { \Lambda } : = \mathbf { u } ^ { \top } \boldsymbol { K } \mathbf { u } = \left. \nabla _ { \theta } q _ { \theta } \right. ^ { 2 } , \qquad \boldsymbol { \sigma } : = \mathbf { u } ^ { \top } \boldsymbol { \sigma } . } \end{array}\tag{28}
$$

The reduction follows one task-bearing direction rather than approximating the full kernel. Over the transition interval, we assume that the kernel acts approximately diagonally on this direction and that the residual driving it is dominated by the component along u:

$$
\pmb { \sigma } \approx \sigma \mathbf { u } , \qquad K \mathbf { u } \approx \Lambda \mathbf { u } .\tag{29}
$$

These conditions suppress leading-order mixing with the remaining modes.

With

$$
K _ { i j k } ^ { \left( 2 \right) } = \left. \nabla _ { \theta } K _ { i j } , \nabla _ { \theta } f _ { k } \right. ,\tag{30}
$$

the projected hierarchy term becomes

$$
\mathbf { u } ^ { \top } ( K ^ { ( 2 ) } \pmb { \sigma } ) \mathbf { u } \approx \sigma \sum _ { i j k } u _ { i } u _ { j } u _ { k } K _ { i j k } ^ { ( 2 ) } ,\tag{31}
$$

$$
a ( \pmb \theta ) : = \sum _ { i j k } u _ { i } u _ { j } u _ { k } K _ { i j k } ^ { ( 2 ) } = \langle \nabla _ { \pmb \theta } \Lambda , \nabla _ { \pmb \theta } q _ { \pmb \theta } \rangle .\tag{32}
$$

Thus $a ( \pmb \theta )$ is the local proportionality between the residual coefficient σ and the residual-driven change in the NTK strength Λ. When $a \mathrm { ~ > ~ } 0$ and $\sigma \ < \ 0$ for a positive target coefficient, this term increases Λ. The sign of a is a dynamical property; homogeneity does not determine it. The projected kernel equation is therefore

$$
\dot { \Lambda } = - a ( \pmb { \theta } ) \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda .\tag{33}
$$

Whenever $\sigma \neq 0$ , the same equation can be inverted along a measured trajectory:

$$
a _ { \mathrm { e f f } } ( t ) = - \frac { \dot { \Lambda } ( t ) + 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda ( t ) } { \sigma ( t ) } .\tag{34}
$$

The constant-a approximation is local and does not require a to remain fixed throughout training. In the two-homogeneous MLP, both Λ and a are degree two under uniform parameter rescaling,

$$
\Lambda ( c \pmb \theta ) = c ^ { 2 } \Lambda ( \pmb \theta ) , \qquad a ( c \pmb \theta ) = c ^ { 2 } a ( \pmb \theta ) .\tag{35}
$$

Under the one-mode parameter approximation,

$$
\dot { \pmb \theta } \approx - \sigma \nabla _ { \pmb \theta } q _ { \pmb \theta } - \lambda _ { W } ^ { \mathrm { n o r m } } \pmb \theta ,\tag{36}
$$

$$
\dot { a } = - b ( \pmb \theta ) \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } a , \qquad b ( \pmb \theta ) : = \langle \nabla _ { \pmb \theta } a , \nabla _ { \pmb \theta } q \pmb \theta \rangle .\tag{37}
$$

After the residual relaxes near its instantaneous ridge value, substitution into Eq. (37) gives

$$
\dot { a } \simeq 2 \lambda _ { W } ^ { \mathrm { n o r m } } \left( \frac { b ( \pmb { \theta } ) y } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } - a \right) ,\tag{38}
$$

and

$$
\frac { d } { d t } \left( \frac { a } { \Lambda } \right) = \frac { \sigma } { \Lambda ^ { 2 } } \left( a ^ { 2 } - b \Lambda \right) .\tag{39}
$$

Equations (37)–(38) show that weight decay does not introduce an additional fast timescale for a after residual relaxation. Pure radial decay also leaves $a / \Lambda$ unchanged, so any remaining drift of this ratio is residual driven. We therefore use a locally constant coupling subject to four explicit conditions:

1. the projected residual and kernel satisfy Eq. (29);

2. $a ( t )$ remains positive over the interval of interest;

3. a(t) changes slowly relative to residual relaxation; and

4. coupling to other task modes is weak enough to be absorbed into the local coefficient a and the effective threshold $\Lambda _ { \mathrm { g } }$

The cyclic symmetry of modular addition and the difference-structured split motivate approximate Fourier decoupling. The remaining conditions are assumptions of the scalar reduction.

## C.2 PROOF OF THEOREM 2

With a fixed locally, we combine the two first-order equations into a second-order equation for Λ. This form separates damping from the force that selects the equilibrium. Equation (10) gives

$$
\ddot { \Lambda } = - a \dot { \sigma } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \dot { \Lambda }\tag{40}
$$

$$
= a ( \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) \sigma + 2 a \lambda _ { W } ^ { \mathrm { n o r m } } y - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \dot { \Lambda } ,\tag{41}
$$

$$
\sigma = - \frac { \dot { \Lambda } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } { a } ,\tag{42}
$$

hence

$$
\ddot { \Lambda } + ( \Lambda + 4 \lambda _ { W } ^ { \mathrm { n o r m } } ) \dot { \Lambda } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda ( \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) - 2 a \lambda _ { W } ^ { \mathrm { n o r m } } y = 0 .\tag{43}
$$

Equation (43) describes one-dimensional damped motion with state-dependent coefficient $\Lambda +$ $4 \bar { \lambda _ { W } } ^ { \mathrm { n o r m } }$ . We collect the remaining terms into the cubic potential

$$
V ( \Lambda ) = \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { 3 } \Lambda ^ { 3 } + 2 \lambda _ { W } ^ { \mathrm { n o r m 2 } } \Lambda ^ { 2 } - 2 a \lambda _ { W } ^ { \mathrm { n o r m } } y \Lambda ,\tag{44}
$$

$$
\mathcal { H } = \frac { 1 } { 2 } \dot { \Lambda } ^ { 2 } + V ( \Lambda ) - V ( \Lambda _ { * } ) .\tag{45}
$$

On the branch $\Lambda \geq 0$ , positive decay makes the potential strictly convex. The shifted energy H combines displacement from its unique minimum with the kinetic term, and

$$
V ^ { \prime } ( \Lambda ) = 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda ( \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) - 2 a \lambda _ { W } ^ { \mathrm { n o r m } } y ,\tag{46}
$$

$$
V ^ { \prime \prime } ( \Lambda ) = 4 \lambda _ { W } ^ { \mathrm { n o r m } } ( \Lambda + \lambda _ { W } ^ { \mathrm { n o r m } } ) > 0 ,\tag{47}
$$

$$
\dot { \mathcal { H } } = - ( \Lambda + 4 \lambda _ { W } ^ { \mathrm { n o r m } } ) \dot { \Lambda } ^ { 2 } \le 0 .\tag{48}
$$

The stationary points of the first-order system coincide with the extrema of this potential. We solve the equilibrium conditions to obtain

$$
\sigma _ { * } = - \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a } \Lambda _ { * } ,\tag{49}
$$

$$
\Lambda _ { * } ( \Lambda _ { * } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) = a y ,\tag{50}
$$

$$
\Lambda _ { \pm } = - \lambda _ { W } ^ { \mathrm { n o r m } } \pm \sqrt { \lambda _ { W } ^ { \mathrm { n o r m } 2 } + a y } .\tag{51}
$$

For $y = 0 ,$ the only nonnegative root is zero. For $y > 0 , \Lambda _ { + } > 0$ and $\Lambda _ { - } < 0 .$ , giving one positive target-bearing equilibrium. The Lyapunov calculation establishes dissipative motion toward the

![](images/b9d9d7be1b9545d65199e56fd6a794361a616a1fdb8cda04eb659ec597efddde.jpg)

![](images/ea246b905163775e22ea0d161a97a1e236ba149af5ce37d188fbc02e59019adb.jpg)  
Figure 5: Reduced potential and adiabatic trajectories. (a) With illustrative values $\lambda _ { W } ^ { \mathrm { n o r m } } = 0 . 1 5$ and $a = 1$ , the task-aligned mode $( y = 1 )$ has its potential minimum at positive $\Lambda _ { i }$ , whereas the unsupported mode $( y = 0 )$ is minimized at $\Lambda = 0$ . (b) Starting from the same initial strength, the two adiabatic trajectories move toward their respective minima. These dimensionless values illustrate the reduced dynamics and are not fitted to an empirical run.

potential minimum; we use the Jacobian to verify the local asymptotic stability in Theorem 2. At the positive equilibrium,

$$
A = \left( \begin{array} { c c } { { - ( \Lambda _ { * } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) } } & { { - \sigma _ { * } } } \\ { { - a } } & { { - 2 \lambda _ { W } ^ { \mathrm { n o r m } } } } \end{array} \right) ,\tag{52}
$$

with

$$
\mathrm { t r } A = - ( \Lambda _ { * } + 4 \lambda _ { W } ^ { \mathrm { n o r m } } ) < 0 ,\tag{53}
$$

$$
\operatorname* { d e t } A = 2 \lambda _ { W } ^ { \mathrm { n o r m } } ( \Lambda _ { * } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) - a \sigma _ { * }\tag{54}
$$

$$
= 4 \lambda _ { W } ^ { \mathrm { n o r m } } ( \Lambda _ { * } + \lambda _ { W } ^ { \mathrm { n o r m } } ) > 0 .\tag{55}
$$

The negative trace and positive determinant place both eigenvalues in the open left half-plane, so the positive equilibrium is locally asymptotically stable on the nonnegative branch. □

## D ADIABATIC REDUCTION AND FINITE-TRAINING BOUNDARY

## D.1 FROM SMOOTH MODE GROWTH TO AN ACCURACY TRANSITION

The reduced dynamics produce a smooth trajectory $\Lambda ( t )$ , whereas held-out accuracy changes only when individual predictions change class. For held-out example $\mu$ with correct class $c _ { \mu }$ , define its correct-class margin along the training trajectory by

$$
m _ { \mu } ( t ) = f _ { c _ { \mu } } ( { \bf x } _ { \mu } ; t ) - \operatorname* { m a x } _ { c \neq c _ { \mu } } f _ { c } ( { \bf x } _ { \mu } ; t ) .\tag{56}
$$

When $\Lambda ( t )$ is monotone over the transition, the trajectory can be reparameterized by Λ, so that ${ \widetilde m } _ { \mu } ( \Lambda ) : = m _ { \mu } ( t ( \Lambda ) )$ . If each relevant $\widetilde { m } _ { \mu } ( \Lambda )$ changes monotonically through its classification transition, then each example has a mode strength at which its predicted class changes. Denote this threshold by

$$
\widetilde { m } _ { \mu } ( \Lambda _ { \mu , \mathrm { g } } ) = 0 .\tag{57}
$$

In this case, held-out accuracy at a given $\Lambda$ is the empirical cumulative distribution of the examplespecific thresholds,

$$
\operatorname { A c c } ( \Lambda ) = \frac { 1 } { M } \sum _ { \mu = 1 } ^ { M } \mathbf { 1 } \{ \Lambda > \Lambda _ { \mu , \mathbf { g } } \} = \widehat { F } _ { \mathbf { g } } ( \Lambda ) .\tag{58}
$$

A narrow distribution of $\Lambda _ { \mu , \mathrm { g } }$ therefore produces a sharp accuracy rise even when $\Lambda ( t )$ changes smoothly. No discontinuity in the optimization dynamics is required. As an illustration, a logistic

threshold distribution with location $\mu _ { g }$ and scale $b _ { g }$ gives

$$
\operatorname { A c c } ( \Lambda ) \approx \frac { 1 } { 1 + \exp [ - ( \Lambda - \mu _ { g } ) / b _ { g } ] } .\tag{59}
$$

This logistic form is only an example of the readout, the one-mode dynamics do not depend on it.

## D.2 ADIABATIC SLOW EQUATION

The residual relaxes at rate $\Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } }$ , while the NTK mode moves on the slower decay-controlled scale once the residual is close to its ridge value. Under this separation of timescales, we set $\dot { \sigma } \simeq 0$ at the current Λ and define the resulting adiabatic residual $\sigma _ { \mathrm { a d } } ( \Lambda )$ . Equation (10) gives

$$
0 = - ( \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) \sigma - 2 \lambda _ { W } ^ { \mathrm { n o r m } } y ,\tag{60}
$$

$$
\sigma _ { \mathrm { a d } } ( \Lambda ) = - \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } y } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } ,\tag{61}
$$

$$
\dot { \Lambda } = - a \sigma _ { \mathrm { a d } } ( \Lambda ) - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda\tag{62}
$$

$$
= 2 \lambda _ { W } ^ { \mathrm { n o r m } } \frac { a y - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } .\tag{63}
$$

The numerator in Eq. (63) determines the sign of the mode growth. It is positive below the stable fixed point and vanishes at that fixed point, so the upward motion slows as Λ approaches equilibrium. Separating variables gives

$$
d t = \frac { 1 } { 2 \lambda _ { W } ^ { \mathrm { n o r m } } } \frac { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a y - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } d \Lambda .\tag{64}
$$

Integrating from $\Lambda _ { 0 }$ to $\Lambda _ { \mathrm { g } }$ gives the normalized gradient-flow time required to reach the held-out threshold. After $T$ optimizer updates,

$$
t \simeq \mathcal { N } \eta T ,
$$

$$
\begin{array} { r } { \mathcal { N } \lambda _ { W } ^ { \mathrm { n o r m } } = \lambda _ { W } . } \end{array}\tag{65}
$$

(66)

Equating the available normalized time to the crossing time gives Eq. (15). The conversion from updates contributes ${ \mathcal { N } } _ { : }$ , while $\mathcal { N } \lambda _ { W } ^ { \mathrm { n o r m } } = \lambda _ { W }$ , leaving optimizer decay in the prefactor and normalized decay inside the integrand.

## D.3 CLOSED FORM, CRITICAL SLOWING, AND THE FITTING CUTOFF

We factor the denominator of the crossing-time integrand at the two fixed points of the slow equation. Define

$$
q : = a y , \qquad \Lambda _ { \pm } : = - \lambda _ { W } ^ { \mathrm { n o r m } } \pm \sqrt { \lambda _ { W } ^ { \mathrm { n o r m } 2 } + q } .\tag{67}
$$

Then

$$
q - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda = - ( \Lambda - \Lambda _ { + } ) ( \Lambda - \Lambda _ { - } ) ,\tag{68}
$$

$$
C _ { - } : = \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } + \Lambda _ { + } } { \Lambda _ { + } - \Lambda _ { - } } ,\tag{69}
$$

$$
C _ { + } : = \frac { 2 \lambda _ { W } ^ { \mathrm { n o r m } } + \Lambda _ { - } } { \Lambda _ { - } - \Lambda _ { + } } .\tag{70}
$$

The integrand becomes

$$
\frac { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { q - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } = - \frac { C _ { - } } { \Lambda - \Lambda _ { + } } - \frac { C _ { + } } { \Lambda - \Lambda _ { - } } .\tag{71}
$$

Integrating between $\Lambda _ { 0 }$ and $\Lambda _ { \mathrm { g } }$ gives

$$
\eta _ { * } ( \lambda _ { W } ) = \frac 1 { 2 \lambda _ { W } T } \Biggl [ C _ { - } \log \left| \frac { \Lambda _ { 0 } - \Lambda _ { + } } { \Lambda _ { \mathrm { g } } - \Lambda _ { + } } \right| + C _ { + } \log \left| \frac { \Lambda _ { 0 } - \Lambda _ { - } } { \Lambda _ { \mathrm { g } } - \Lambda _ { - } } \right| \Biggr ] .\tag{72}
$$

As the requested threshold approaches the stable fixed point $\Lambda _ { + }$ , the pole at $\Lambda _ { + }$ reaches the upper integration limit while the $\Lambda _ { - }$ contribution remains finite. Hence, for $\Lambda _ { \mathrm { g } } \uparrow \Lambda _ { + }$ -9

$$
\eta _ { * } ( \lambda _ { W } ) \sim { \frac { C _ { - } } { 2 \lambda _ { W } T } } \log { \frac { 1 } { \Lambda _ { + } - \Lambda _ { \mathrm { g } } } } ,\tag{73}
$$

$$
\Lambda _ { + } - \Lambda _ { \mathrm { g } } = \Theta \big ( \lambda _ { W } ^ { \mathrm { m o d e } } - \lambda _ { W } \big ) \ .\tag{74}
$$

The logarithm arises because the growth rate of Λ vanishes at the reachability boundary. Away from this cutoff, the logarithm remains finite and the explicit prefactor gives the leading $\bar { 1 / ( \lambda _ { W } T ) }$ dependence.

The training-fit cutoff in $\operatorname { E q . } \left( 1 7 \right)$ follows from the same adiabatic gain. For a representative training mode, the expressed fraction of its target component is

$$
g ( \Lambda _ { \mathrm { f i t } } ) = \frac { \Lambda _ { \mathrm { f i t } } } { \Lambda _ { \mathrm { f i t } } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } .\tag{75}
$$

Requiring $g ( \Lambda _ { \mathrm { f i t } } ) \geq g _ { \mathrm { m i n } }$ gives

$$
\frac { \Lambda _ { \mathrm { f i t } } } { \Lambda _ { \mathrm { f i t } } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } \geq g _ { \mathrm { m i n } } ,\tag{76}
$$

$$
\lambda _ { W } ^ { \mathrm { n o r m } } \leq \frac { \Lambda _ { \mathrm { f i t } } ( 1 - g _ { \mathrm { m i n } } ) } { 2 g _ { \mathrm { m i n } } } ,\tag{77}
$$

$$
\lambda _ { W } \leq \mathcal { N } \frac { \Lambda _ { \mathrm { f i t } } ( 1 - g _ { \mathrm { m i n } } ) } { 2 g _ { \mathrm { m i n } } } ,\tag{78}
$$

which is Eq. (17). The approximately vertical boundary used in the phase diagram further assumes that the representative training-mode strength $\Lambda _ { \mathrm { f i t } }$ varies weakly with η over the range of interest.

## D.4 DISCRETE-TIME STABILITY OF THE FROZEN-KERNEL DYNAMICS

Our continuous-time reduction does not describe the large-step edge of the optimizer plane. We hold the kernel fixed to isolate the Euler stability condition for the fast residual dynamics and set

$$
h : = { \mathcal { N } } \eta .
$$

For fixed $K ,$

$$
\pmb { \sigma } _ { s + 1 } = \pmb { \sigma } _ { s } - h \left[ ( K + 2 \lambda _ { W } ^ { \mathrm { n o r m } } I ) \pmb { \sigma } _ { s } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } \mathbf { y } \right] ,\tag{79}
$$

$$
\pmb { \sigma } _ { \ast } = - 2 \lambda _ { W } ^ { \mathrm { n o r m } } ( K + 2 \lambda _ { W } ^ { \mathrm { n o r m } } I ) ^ { - 1 } \mathbf { y } ,\tag{80}
$$

$$
\delta _ { s } : = \pmb { \sigma } _ { s } - \pmb { \sigma } _ { \ast } ,\tag{81}
$$

$$
\delta _ { s + 1 } = \left[ I - h ( K + 2 \lambda _ { W } ^ { \mathrm { n o r m } } I ) \right] \delta _ { s } .\tag{82}
$$

After we subtract the fixed point, each kernel eigendirection evolves independently. For $K \mathbf { u } _ { j } \ =$ $\Lambda _ { j } \mathbf { u } _ { j }$

$$
\delta _ { j , s + 1 } = \left[ 1 - h ( \Lambda _ { j } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) \right] \delta _ { j , s } .\tag{83}
$$

Therefore

$$
| 1 - h ( \Lambda _ { j } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } ) | < 1 ,\tag{84}
$$

$$
\ N \eta < \frac { 2 } { \Lambda _ { \mathrm { m a x } } ( K ) + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } ,\tag{85}
$$

$$
\eta < \frac { 2 } { \sqrt { \Lambda _ { \operatorname* { m a x } } ( K ) + 2 \lambda _ { W } } } ,\tag{86}
$$

which recovers Eq. (18). The most restrictive eigenvalue is $\Lambda _ { \operatorname* { m a x } } ( K )$ . Feature learning changes this spectrum, so we use the bound only as a local stability condition for the frozen-kernel approximation, not as a global guarantee for nonlinear training.

## E JACOBIAN-MEDIATED FEATURE DYNAMICS BEYOND HOMOGENEITY

For any differentiable model trained with squared loss and coupled $L _ { 2 }$ decay, the chain rule separates the residual-dependent contribution to tangent-kernel evolution from the contribution of radial parameter decay. Homogeneity is only needed to convert the latter into fixed multiples of the output and NTK. Without homogeneity, the residual-dependent term remains exact, while the radial contractions become architecture dependent.

## E.1 EXACT DIFFERENTIABLE-MODEL IDENTITIES

Proposition 1 (Exact Jacobian-mediated dynamics for differentiable models). Let

$$
\widetilde { \mathcal { L } } ( \pmb \theta ) = \frac { 1 } { 2 } \left\| \mathbf { f } _ { \pmb \theta } - \mathbf { y } \right\| ^ { 2 } + \frac { \lambda _ { W } ^ { \mathrm { n o r m } } } { 2 } \left\| \pmb \theta \right\| ^ { 2 } ,\tag{87}
$$

with $\pmb { \sigma } = \mathbf { f } - \mathbf { y } , J = \nabla _ { \pmb { \theta } } \mathbf { f } ,$ , and $K = J J ^ { \top }$ . Then

$$
\dot { \pmb { \sigma } } = - K \pmb { \sigma } - \lambda _ { W } ^ { \mathrm { n o r m } } J \pmb { \theta } ,\tag{88}
$$

$$
\dot { K } _ { i j } = - \sum _ { k } K _ { i j k } ^ { ( 2 ) } \sigma _ { k } - \lambda _ { W } ^ { \mathrm { n o r m } } R _ { i j } ,\tag{89}
$$

where

$$
K _ { i j k } ^ { ( 2 ) } = \left. \nabla _ { \theta } K _ { i j } , \nabla _ { \theta } f _ { k } \right. , \qquad R _ { i j } = \left. \nabla _ { \theta } K _ { i j } , \theta \right. .\tag{90}
$$

Along the same flow, the loss is monotone because

$$
\frac { d } { d t } \widetilde { \mathcal { L } } ( \pmb { \theta } ( t ) ) = - \left\| J ^ { \top } \pmb { \sigma } + \lambda _ { W } ^ { \mathrm { n o r m } } \pmb { \theta } \right\| ^ { 2 } \leq 0 .\tag{91}
$$

Proof. We differentiate the model output along parameter gradient flow to obtain the residual identity. Applying the same derivative to each entry of $K$ produces one contraction with the data gradient and one with the radial weight-decay direction.

$$
\begin{array} { r } { \dot { \pmb { \theta } } = - J ^ { \top } \pmb { \sigma } - \lambda _ { W } ^ { \mathrm { n o r m } } \pmb { \theta } , } \end{array}\tag{92}
$$

$$
\dot { \pmb { \sigma } } = J \dot { \pmb { \theta } } = - K \pmb { \sigma } - \lambda _ { W } ^ { \mathrm { n o r m } } J \pmb { \theta } ,\tag{93}
$$

$$
\dot { K } _ { i j } = \left. \nabla _ { \pmb { \theta } } K _ { i j } , \dot { \pmb { \theta } } \right.\tag{94}
$$

$$
= - \sum _ { k } \left. \nabla _ { \theta } K _ { i j } , \nabla _ { \theta } f _ { k } \right. \sigma _ { k } - \lambda _ { W } ^ { \mathrm { n o r m } } \left. \nabla _ { \theta } K _ { i j } , \pmb { \theta } \right. ,\tag{95}
$$

$$
\frac { d } { d t } \widetilde { \mathcal { L } } = \Big \langle \nabla _ { \pmb { \theta } } \widetilde { \mathcal { L } } , \dot { \pmb { \theta } } \Big \rangle = - \left\| \nabla _ { \pmb { \theta } } \widetilde { \mathcal { L } } \right\| ^ { 2 } .\tag{96}
$$

Equation (89) isolates the extension needed beyond homogeneous networks. Whenever the $K ^ { ( 2 ) }$ contraction is nonzero, the residual that controls prediction error also changes the tangent kernel. The contraction may vanish at individual states; homogeneity is needed only to replace the remaining radial term by fixed coefficients.

## E.2 LOCAL PROJECTED DYNAMICS

For a task direction u, we suppose over a training interval that

$$
\begin{array} { r } { K \mathbf { u } \approx \Lambda \mathbf { u } , \quad \sigma \approx \sigma \mathbf { u } , \quad \mathbf { u } ^ { \top } J \pmb \theta \approx \beta ( \sigma + y ) , \quad \mathbf { u } ^ { \top } R \mathbf { u } \approx \rho \Lambda , \quad \mathbf { u } ^ { \top } ( K ^ { ( 2 ) } \pmb \sigma ) \mathbf { u } \approx a \sigma , } \end{array}\tag{97}
$$

with slowly varying $\beta , \rho , a .$ The coefficients $\beta$ and $\rho$ summarize radial parameter decay after projection onto the local task direction, while a retains its residual-to-kernel role. Projecting gives

$$
\dot { \sigma } \approx - ( \Lambda + \beta \lambda _ { W } ^ { \mathrm { n o r m } } ) \sigma - \beta \lambda _ { W } ^ { \mathrm { n o r m } } y , \qquad \dot { \Lambda } \approx - a \sigma - \rho \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda .\tag{98}
$$

For a two-homogeneous model, Euler identities set $\beta = \rho = 2$ and recover Eq. (10). In a nonhomogeneous model, we must measure these local contractions; they need not equal the homogeneous coefficients. If they remain order one and vary slowly through the transition, both decay terms

retain the factor $\lambda _ { W } ^ { \mathrm { n o r m } }$ , leaving $1 / \lambda _ { W } ^ { \mathrm { n o r m } }$ as the natural normalized-time scale and $( \eta \lambda _ { W } ) ^ { - 1 }$ as the corresponding update scale.

For the Transformer in Appendix G.3, we let $\mathbf { f } _ { \theta }$ denote the softmax probabilities entering the meansquared-error loss. Proposition 1 then applies despite attention, LayerNorm, residual paths, and biases.

## F DIRECTIONAL NTK-ALIGNMENT INTERVENTION

## F.1 EARLIER TASK ALIGNMENT SHORTENS THE PREDICTED DELAY

Starting closer to the held-out threshold removes a positive part of the crossing-time integral without invoking the global kinetic calibration. Equation (15) gives

$$
s _ { \mathrm { g } } = \frac { 1 } { 2 \eta \lambda _ { W } } \int _ { \Lambda _ { 0 } } ^ { \Lambda _ { \mathrm { g } } } \frac { \Lambda + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a y - \Lambda ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda } d \Lambda ,\tag{99}
$$

and differentiation with respect to the initial strength gives

$$
\frac { \partial s _ { \mathrm { g } } } { \partial \Lambda _ { 0 } } = - \frac { 1 } { 2 \eta \lambda _ { W } } \frac { \Lambda _ { 0 } + 2 \lambda _ { W } ^ { \mathrm { n o r m } } } { a y - \Lambda _ { 0 } ^ { 2 } - 2 \lambda _ { W } ^ { \mathrm { n o r m } } \Lambda _ { 0 } } < 0\tag{100}
$$

throughout the reachable branch, where the denominator is the positive spectral drive. The sign predicts that improving early alignment with a reachable task direction should advance the later held-out transition even if the auxiliary intervention is removed before generalization begins.

## F.2 A TEMPORARY NTK INTERVENTION ADVANCES HELD-OUT GENERALIZATION

We change early task-indexed tangent geometry while keeping the later objective identical, which tests the sign in Eq. (100). We use addition modulo 97 on the 4,753 unordered input pairs. We train a bias-free $1 9 4 \to \mathbf { \bar { 1 } 2 8 } \to 1 2 8 \to 1 2 8 \to 9 7$ ReLU teacher with He-normal initialization. Its 50/50 split uses NumPy seed 0 and contains 2,376 training and 2,377 held-out examples. We run 30,000 full-batch SGD updates without momentum, using learning rate 180, coupled weight decay $1 0 ^ { - 5 }$ ordinary PyTorch MSE between logits and one-hot targets averaged over samples and outputs, and model seed 0.

For examples $\mathbf { x } _ { \mu } , \mathbf { x } _ { \nu }$ with labels $c _ { \mu } , c _ { \nu } ,$ , we project the empirical NTK through the corresponding true-label logits,

$$
K _ { \mu \nu } ^ { \mathrm { T L } } = \bigl \langle \nabla _ { \pmb { \theta } } f _ { c _ { \mu } } ( { \bf x } _ { \mu } ) , \nabla _ { \pmb { \theta } } f _ { c _ { \nu } } ( { \bf x } _ { \nu } ) \bigr \rangle .\tag{101}
$$

The intervention uses a normalization order different from the diagnostic in Appendix G.1. We first cosine-normalize the sample kernel,

$$
\widehat K _ { \mu \nu } ^ { \mathrm { T L } } = \frac { K _ { \mu \nu } ^ { \mathrm { T L } } } { \sqrt { K _ { \mu \mu } ^ { \mathrm { T L } } K _ { \nu \nu } ^ { \mathrm { T L } } } } ,\tag{102}
$$

and then average between label groups,

$$
\bar { K } _ { c c ^ { \prime } } = \frac { 1 } { n _ { c } n _ { c ^ { \prime } } } \sum _ { \mu : c _ { \mu } = c } \sum _ { \nu : c _ { \nu } = c ^ { \prime } } \widehat { K } _ { \mu \nu } ^ { \mathrm { T L } } .\tag{103}
$$

We compute the fixed symmetric $9 7 \times 9 7$ target matrix from the teacher’s training split and the differentiable student matrix from each student’s training split.

We train 40 paired bias-free 194 → 256 → 97 ReLU students with He-normal initialization. Pair $q \in \{ 0 , \ldots , 3 9 \}$ uses model seed q and an independently generated $5 0 / 5 0$ split with NumPy seed $1 0 , 0 0 0 { + } q ;$ every split has 2,376 training and $^ { 2 , 3 7 7 }$ held-out examples, and both members of the pair share the same initialization and data. Every student uses 30,000 full-batch SGD updates without momentum, learning rate 35, coupled weight decay $3 \times 1 0 ^ { - 6 }$ , and ordinary PyTorch MSE between logits and one-hot targets averaged over samples and outputs. During updates 0 through 2,999, the intervention member minimizes

$$
{ \mathcal { L } } _ { \mathrm { i n t } } = { \mathcal { L } } _ { \mathrm { s t u d e n t } } + \lambda _ { \mathrm { N T K } } \left. { \bar { K } } _ { \pmb { \theta } } - { \bar { K } } _ { \mathrm { t e a c h e r } } \right. _ { F } , \qquad \lambda _ { \mathrm { N T K } } = 0 . 1 0 .\tag{104}
$$

![](images/dde4aba73b3381f6292ac87720e7ed79d50eb9a9828fe02add79c2619050aa5a.jpg)

![](images/f3764c821c5b4cd2405444e122937310510019d94088581c995eec0a065a71f3.jpg)  
Figure 6: Temporary early NTK alignment advances held-out generalization. (a) Mean held-out accuracy for the paired baseline and intervention students. The shaded vertical region marks the first 3,000 updates, when the NTK penalty is active; bands show one sample standard deviation across 40 paired seeds. (b) Mean held-out MSE for the same runs. The mean first 50% held-out crossing moves from 20,200 updates for the baseline to 13,527.5 for the intervention, a ratio of 1.493.

We remove the auxiliary term at update 3,000; the baseline never receives it. We record training and held-out accuracy and MSE every 100 updates.

The paired mean speedup is 1.498, and the median paired speedup is 1.489. The intervention students also maintain lower held-out MSE after the penalty has been removed. Because the two members of each pair share their data and initialization and use the same objective after update 3,000, the shift is consistent with the predicted negative derivative in Eq. (100). Matching the full $9 7 \times 9 7$ label kernel changes several Fourier components and may alter other parts of the trajectory, so the experiment tests the direction of the initial-alignment effect rather than the scalar one-mode closure in isolation.

## G EXPERIMENTAL CONFIGURATIONS AND ANALYSIS CONVENTIONS

The accompanying experiment package centralizes the configurations below in the model, training, and analysis code. We report optimizer weight decay in the convention passed directly to PyTorch SGD. The anonymous artifact URL remains to be inserted in the reproducibility statement.

## G.1 SPECTRAL DIAGNOSTICS ON ADDITION MODULO 97

For Figure 2, we use the $9 7 \cdot 9 8 / 2 = 4 { , } 7 5 3$ unordered input pairs allowed by commutativity. A NumPy generator with seed 0 permutes the pairs; the first 2,376 are used for training and the remaining 2,377 are held out. Each input concatenates two 97-dimensional one-hot vectors, and the target is the 97-dimensional one-hot encoding of the modular sum.

The model is a bias-free $1 9 4 \to 2 5 6 \to 9 7$ one-hidden-layer ReLU MLP with He-normal initialization and model seed 0. We train for 20,000 full-batch SGD updates with learning rate 40, coupled weight decay $2 . 5 \times 1 0 ^ { - 5 }$ , and no momentum. The loss is ordinary PyTorch MSE averaged over samples and all 97 output coordinates, so $N = 2 / ( 2 3 7 6 \cdot 9 7 ) = 1 / \dot { 1 } 1 5 , \dot { 2 } 3 6$ and $\lambda _ { W } ^ { \mathrm { n o r m } } = 2 . 8 8 0 9$ . Accuracy is recorded every 100 updates.

For training examples $\mathbf { x } _ { \mu } , \mathbf { x } _ { \nu }$ with labels $c _ { \mu } , c _ { \nu }$ , the true-label-projected sample NTK is

$$
K _ { \mu \nu } ^ { \mathrm { T L } } = \bigl \langle \nabla _ { \pmb { \theta } } f _ { c _ { \mu } } ( { \bf x } _ { \mu } ) , \nabla _ { \pmb { \theta } } f _ { c _ { \nu } } ( { \bf x } _ { \nu } ) \bigr \rangle .\tag{105}
$$

We first average these entries between groups of examples with the same output labels, producing a $9 7 \times 9 7$ matrix $\widetilde { K }$ . We then diagonal-normalize and symmetrize the label kernel,

$$
\bar { K } _ { c c ^ { \prime } } = \frac { \widetilde { K } _ { c c ^ { \prime } } } { \sqrt { \widetilde { K } _ { c c } \widetilde { K } _ { c ^ { \prime } c ^ { \prime } } } } .\tag{106}
$$

Panel (a) uses updates 0, 3,000, 8,000, and 20,000. Eigensystems are computed at updates 0, 100, 500, 1,000, 3,000, 6,000, 8,000, 10,000, and 20,000. Eigenvectors are matched backward through these checkpoints by maximum absolute overlap, with signs chosen continuously. Panel (c) displays four tracked leading eigenvectors at updates 0, 1,000, 6,000, and 20,000.

## G.2 HIGH-RESOLUTION MLP PHASE SWEEP ON ADDITION MODULO 23

We use all $2 3 ^ { 2 } = 5 2 9$ ordered input pairs. Each input concatenates two 23-dimensional one-hot vectors, and the target is a 23-dimensional one-hot encoding of the modular sum. NumPy seed 2027 selects the training differences

$$
\mathcal { D } = \{ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 1 2 , 1 3 , 1 4 , 1 6 , 1 7 , 1 9 , 2 0 , 2 2 \} ,\tag{107}
$$

and we define

$$
{ \mathcal { T } } _ { \mathcal { D } } = \{ ( a , b ) : a - b { \bmod { 2 3 } } \in { \mathcal { D } } \} .\tag{108}
$$

The split contains 368 training and 161 held-out pairs and is invariant under simultaneous shift $( a , b ) \mapsto ( a + r , b + r )$

At every grid point we initialize the same bias-free $4 6  2 5 6  2 3$ one-hidden-layer ReLU MLP with He-normal initialization and model seed 0. Training uses ordinary PyTorch MSE averaged over the $3 6 8 \times 2 3$ residual coordinates, full-batch SGD, coupled $L _ { 2 }$ weight decay, and no momentum. Thus $N = 2 / ( 3 6 8 { \cdot } 2 3 ) = 1 / 4 2 3 2$ . Each run lasts 24,000 updates and is evaluated every 100 updates. The grid contains 84 logarithmically spaced learning rates from 0.1 to 100 and 90 logarithmically spaced weight decays from $5 \times 1 0 ^ { - 6 } \mathrm { t o } 2 \times 1 0 ^ { - 3 }$ , for 7,560 runs.

We classify runs using a 90% accuracy threshold. Grokking requires final training and held-out accuracy above $9 0 \% ;$ memorization requires final training accuracy above 90% but held-out accuracy below it; forgetting means that training accuracy crossed 90% earlier but ends below it; all remaining runs are no fitting. For Figure 3b, we retain only runs that ultimately grok and record the first update at which held-out accuracy reaches 10%. This lower threshold marks the beginning of the final MLP generalization rise and does not affect the phase labels.

The representative runs in Figure 7 are fixed cells from the same dense sweep. Their $( \eta , \lambda _ { W } )$ values are $( 0 . 8 7 0 4 8 , 3 . 2 9 3 0 \times \mathrm { \bar { 1 } 0 ^ { - 5 } } )$ for memorization, (2.36316, 1.18329 $\times ~ 1 0 ^ { - 4 } )$ for grokking, $( 1 . 4 3 4 2 6 , 1 . 2 4 8 4 6 \times 1 0 ^ { - 3 } )$ for forgetting, and $( 2 8 . 6 9 6 7 , 1 . 8 9 5 6 1 \times 1 0 ^ { - 4 } )$ for no fitting. The grokking example first reaches 90% held-out accuracy at update 8,300. The forgetting example reaches peak training accuracy 0.948 and ends at 0.204.

## G.2.1 BOUNDARY EXTRACTION AND CALIBRATION

For each usable weight-decay column, we define the memorization–grokking boundary as the geometric midpoint between the last memorizing learning rate and the first grokking learning rate. We exclude the detached high-step island. The finite-time boundary uses 53 midpoints over $2 \times 1 0 ^ { - 5 } \le \lambda _ { W } \le 7 \times 1 0 ^ { - 4 }$ with equal weight in log learning rate.

To evaluate Eq. (15), define the scaled coordinates

$$
\begin{array} { r } { \widehat { \Lambda } = { \cal N } \Lambda , \qquad \widehat { q } = \Lambda ^ { 2 } a y . } \end{array}\tag{109}
$$

This removes $\mathcal { N }$ from the integrand while leaving the optimizer prefactor $1 / ( 2 \lambda _ { W } T )$ unchanged. The calibrated finite-time curve is

$$
\boxed { \eta _ { \mathrm { f i t } } ( \lambda _ { W } ) = \frac { C _ { \eta } } { 2 \lambda _ { W } T } \int _ { \hat { \Lambda } _ { 0 } } ^ { \hat { \Lambda } _ { g } } \frac { \widehat { \Lambda } + 2 \lambda _ { W } } { \widehat { q } - \widehat { \Lambda } ^ { 2 } - 2 \lambda _ { W } \widehat { \Lambda } } d \widehat { \Lambda } }\tag{110}
$$

with one global kinetic factor $C _ { \eta }$ . If omitted modes and slow variation in the projected coupling rescale the scalar velocity as $d \widehat { \Lambda } / d t \simeq \kappa F ( \widehat { \Lambda } )$ , then crossing times are multiplied by $1 / \kappa$ and $C _ { \eta } = 1 / \kappa$ . The MSE normalization $\mathcal { N }$ is known exactly and is not part of this calibration. The quantities $\widehat { \Lambda } _ { 0 } , \widehat { \Lambda } _ { g } ,$ and $\widehat { q }$ are inferred jointly from the optimizer-plane boundary rather than measured from NTK trajectories. We therefore treat them as effective coordinates of the scalar boundary model: the fit tests the predicted dependence of the crossing time on learning rate and weight decay, but does not determine the absolute magnitude of post-fit NTK growth.

![](images/1aaa70bc4596f9de7cfd3846a0d15359e3bfe69c8304d2c9a5397e86d5a96c01.jpg)  
Figure 7: Representative trajectories for the four MLP phase outcomes. We show training and held-out accuracy for fixed cells from the modulo-23 sweep. The dotted horizontal line is the 90% threshold used to classify the phase map.

Panel (a) of Table 2 collects the values used for every overlay in Figure 3a. The mode-reachability cutoff is derived from the fitted reduced coordinates rather than fitted independently. The training-fit and stability overlays use the functional forms of Eqs. (17) and (18), with their locations calibrated separately to the observed boundaries. In particular, the stability calibration is summarized by an effective scale $\widehat { \Lambda } _ { \mathrm { s t a b } }$ and is not obtained from a direct measurement of $\mathcal { N } \Lambda _ { \mathrm { m a x } } ( K )$

Over the small-decay interval used for the inverse-decay comparison, the empirical boundary has log–log slope −1.1681, while the calibrated finite-time curve has slope −0.9043. These slopes are diagnostics of the boundary shape and are not additional fit parameters.

## G.3 TRANSFORMER PHASE SWEEP ON ADDITION MODULO 23

We use the same task, difference split, split seed, training size, and held-out size as in Appendix G.2, but present each input as a sequence of two 23-dimensional one-hot tokens. A learned affine encoder maps each token to $d _ { \mathrm { m o d e l } } = 3 2$ . The model contains one pre-norm Transformer block with four attention heads, a ReLU feed-forward width of 64, residual connections, learned embeddings for the two positions, a final LayerNorm, and an affine 64 →23 decoder. We use model seed 0 at every grid point.

Training uses full-batch SGD with coupled $L _ { 2 }$ weight decay and no momentum. The loss is Py-Torch MSE between softmax probabilities and one-hot targets, averaged over samples and output coordinates, so $\mathcal { N } = 1 / 4 2 3 2$ . Each run lasts 60,000 updates and is evaluated every 250 updates. The phase file contains 42 uniformly logarithmically spaced learning rates from $5 \times 1 \mathrm { { 0 ^ { - 3 } } }$ to $\mathrm { i 0 ^ { 3 } }$ and 45 uniformly logarithmically spaced weight decays from $1 0 ^ { - 7 }$ to $1 0 ^ { - 3 }$ , for 1,890 independently trained networks. We use the same 90% phase definitions as for the MLP.

For every usable weight-decay column, we extract the geometric midpoint between the last memorizing and first grokking learning rate. We fit the inverse-decay form $\eta = A / \lambda _ { W }$ over $1 0 ^ { - 5 } \ \leq$ $\lambda _ { W } \leq 3 \times 1 0 ^ { - 4 }$ . The current phase grid yields the values in panel (b) of Table 2.

For the transition-time scaling panel, we retain runs that ultimately grok and record the first held-out 80% crossing. The 80% threshold is chosen to lie in the final generalization rise rather than near chance accuracy, $1 / 2 3 \approx 4 . 3 5 \%$ . On the current 1,890-run grid, the grokking cohort has median $\eta \lambda _ { W } s _ { 8 0 } = 0 . 9 2 0$

<table><tr><td>Curve</td><td>Relation</td><td>Calibration values</td><td>Fit range / quality</td></tr><tr><td colspan="4">(a) MLP</td></tr><tr><td>Finite-time boundary</td><td>Eq. (110)</td><td> $C _ { \eta } = 2 4 . 8 4 0 7 4 3 , \widehat { \Lambda } _ { 0 } = 0 . 9 2 8 1 8 0 6 ,$   $\widehat { \Lambda } _ { g } = 0 . 9 2 8 4 9 2 0 , \widehat { q } = 0 . 8 6 3 3 9 7 3 ;$ </td><td>53 midpoints,  $2 \times 1 0 ^ { - 5 } \ \leq \ \hat { \lambda } _ { W } \ \leq$   $7 \ \times \ 1 0 ^ { - 4 } ;$  log-MSE</td></tr><tr><td></td><td></td><td> $\kappa \overset { \cdot } { = } C _ { \eta } ^ { - 1 } = 0 . 0 4 0 2 5 6 4$   $A = 1 . 0 9 1 6 3 9 5 \times 1 0 ^ { - 4 }$ </td><td> $0 . 0 2 6 8 0 0 ;$  log-RMSE  $0 . 1 6 3 7 0 6$  34 midpoints,</td></tr><tr><td>asymptote</td><td></td><td> $\lambda _ { W } ^ { \mathrm { m o d e } } ~ = ~ 7 . 0 0 0 0 \times 1 0 ^ { - 4 }$  , derived no independent fit</td><td> $2 \times 1 0 ^ { - 5 } \ \leq \ \dot { \lambda } _ { W } \ \leq$   $2 \times 1 0 ^ { - 4 } ;$  log-RMSE  $0 . 1 1 9 9 0 2$ </td></tr><tr><td>ity Training fit</td><td>Eq. (17)</td><td>from ♀ and  $\widehat { \Lambda } _ { g }$   $\lambda _ { W } ^ { \mathrm { f i t } } = 8 . 3 8 2 1 1 5 1 \times 1 0 ^ { - 4 }$ </td><td>location minimizes</td></tr><tr><td></td><td></td><td></td><td>final-training-accuracy misclassification low- location calibrated to</td></tr><tr><td>Frozen-kernel stability</td><td>Eq. (18)</td><td>effective  $\widehat { \Lambda } _ { \mathrm { s t a b } } = 0 . 1 0 5 6 0 1 1 ;$  decay cutoff  $\eta \simeq 1 8 . 9 2 8 2$ </td><td>observed instability boundary</td></tr><tr><td colspan="4">(b) Transformer</td></tr><tr><td>Inverse-decay boundary</td><td> $\eta = A / \lambda _ { W }$ </td><td> $A = 1 . 6 8 7 5 6 2 1 \times 1 0 ^ { - 5 }$ </td><td>16 boundary points,  $1 0 ^ { - 5 } \quad \le \quad \dot { \lambda } _ { W } ^ { } \quad \le$ </td></tr></table>

Table 2: Boundary calibrations for the optimizer-plane experiments. Panel (a) lists the MLP phasediagram overlays; panel (b) gives the inverse-decay fit for the Transformer memorization–grokking boundary. The theory specifies the functional forms of the MLP curves, while their locations are determined either by the reduced finite-time fit or by separate effective calibrations to the observed boundaries.

## G.4 REDUCED-MODE POTENTIAL ILLUSTRATION

For Figure 5, we integrate Eq. (12) with dimensionless normalized decay $\lambda _ { W } ^ { \mathrm { n o r m } } = 0 . 1 5 ,$ coupling $a = 1$ , initial strength $\Lambda ( 0 ) \stackrel { - } { = } 0 . 2 5$ , fourth-order Runge–Kutta step 0.002, and final time 18. The task-aligned trajectory uses $y = 1$ , while the unsupported control uses $y = 0 .$ . These values are chosen only to display the qualitative reduced dynamics and are not fitted to an empirical run.