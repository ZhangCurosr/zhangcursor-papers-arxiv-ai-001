# A “feature ODE” describing the learning behavior of shallow MLPs on simple functions

Joseph Turnbull<sup>1,∗</sup> Berkan Ottlik<sup>2,∗</sup> Jamie Simon<sup>1,3</sup>

<sup>1</sup>UC Berkeley <sup>2</sup>University of Pennsylvania <sup>3</sup>Imbue

joeyturnbull@berkeley.edu, ottlik@seas.upenn.edu, jsi@berkeley.edu

## Abstract

We study the gradient flow training dynamics of two-layer multi-layer perceptrons (MLPs) on anisotropic Gaussian data. Even in this simple setting, solving the dynamics in full is notoriously dificult, so we try something diferent. We propose a “feature $\mathrm { O D E } ^ { \boldsymbol { \mathbf { \mathit { w } } } }$ that might be intuitively expected to describe the learning dynamics of a shallow MLP. We find that MLPs trained to learn simple target functions (low-order Hermite polynomials and staircase functions) actually roughly follow the loss trajectories from this feature ODE. This agreement is a surprise because of the size of the simplification and the fact that it does not come from a rigorous chain of approximations. We discuss this surprise and speculate on its reasons and uses.

![](images/82c5431c5b7b257ee4712f065989486768ad8d7b3e1c8195ad8b562c3e5685b3.jpg)

$$
\eta \cdot n _ { \mathrm { i t e r } }
$$

![](images/1f7b6a37613ded31cbd2c820134f5c08c8e466180788f167cb5372868f26dca4.jpg)  
Figure 1: We suggest a simple ODE of scalar variables whose solution predicts loss trajectories for shallow MLPs learning simple functions of anisotropic data. These plots treat shallow MLPs with shifted SiLU activation functions $( \sigma ( x ) = \mathrm { S i L U } ( x + 0 . 5 ) )$ learning various Hermite polynomial target functions data sampled $\pmb { x } \sim \mathcal { N } ( 0 , \mathbf { T } )$ with $\mathbf { r } = \mathrm { d i a g } ( \gamma _ { i } ) , \gamma _ { i } \propto ( i + 6 ) ^ { - 1 . 8 }$ . We write $z _ { i } \equiv { \pmb x } _ { i } / \sqrt { \gamma _ { i } }$ for normalized coordinates. Left. MLP loss trajectories closely match the ODE for example linear and quadratic target functions. Right. Efective optimization times $\eta \cdot n _ { \mathrm { i t e r } }$ required for learning to complete are well-predicted by this ODE across many such target functions. For quadratic terms of the form $\mathrm { H e } _ { 2 } ( z _ { i } )$ , the leading term is indicated in the plots $( z _ { i } ^ { 2 } )$ . See Section A for experimental details.

## 1 Introduction

How do shallow MLPs learn simple functions of anistropic Gaussian data? Many researchers have studied these training dynamics, and we have learned a lot (see Section 2). One definite conclusion has emerged: the problem is hard — surprisingly hard, for what should be a simple toy problem in deep learning theory.

What is the source of the dificulty? Taking a look at the functional form of a shallow MLP,

$$
f _ { \mathrm { M L P } } ( \pmb { x } ) = \langle \pmb { w } _ { \mathrm { o u t } } , \sigma ( \pmb { W } _ { \mathrm { i n } } \pmb { x } ) \rangle ,\tag{1}
$$

the main annoyance is the elementwise nonlinearity $\sigma ( \cdot )$ . Elementwise nonlinearities do not play well with our nicest mathematical tools: try to exactly solve the dynamics of a model parameterized as Equation (1) using e.g. regular matrix ODEs, and the elementwise nonlinearity (which is rather unnatural from a linear-algebraic point of view) gets in the way.

Rather than develop a new tool for treating the exact dynamics of a shallow MLP, we propose something diferent altogether. We design our system to be a solvable model of the shallow MLP that keeps two of the MLP’s key traits. Firstly, there should be two trainable linear transformations with a (fixed) nonlinear operation in between, and secondly, the model class needs to represent any nonlinear function. We get both of these traits in the following “deep polynomial regression” (DPR) model:

$$
f _ { \mathrm { D P R } } ( { \pmb x } ) = \langle { \pmb B } , \psi _ { c } ( { \pmb A } { \pmb x } ) \rangle ,\tag{2}
$$

where A is a square matrix, B is a readout vector, and $\psi _ { c }$ is the following polynomial featurization operation:

$$
\begin{array} { r } { \psi _ { c } ( h ) : = \mathrm { v e c } \left( \left[ \sqrt { c _ { 0 } } \mathrm { ~ \ p s ~ \ } \sqrt { c _ { 1 } } h \mathrm { ~ \ ~ } \frac { \sqrt { c _ { 2 } } } { \sqrt { 2 ! } } h h ^ { \top } \mathrm { ~ \ ~ } \frac { \sqrt { c _ { 3 } } } { \sqrt { 3 ! } } h ^ { \otimes 3 } \mathrm { ~ \ ~ \ } \cdot \cdot \cdot \right] \right) , } \end{array}\tag{3}
$$

with $\pmb { c } = \left( c _ { \ell } \right)$ a sequence of constants.

Why is this a reasonable proxy for a shallow MLP? The matrix A serves the role of $W _ { \mathrm { i n } }$ , preconditioning the input to the nonlinear operation. The operation $\psi _ { c }$ generates all monomial functions of the hidden vector $h = A x$ , giving the model full expressivity. Finally, the readout vector B maps these features to a scalar output. See Figure 2 for a visual comparison between the two models.

Note that if one simply fixes $\pmb { A } = \pmb { I } _ { 3 }$ , then Equation (2) is polynomial regression: $f ( { \pmb x } ) = \langle { \pmb b } , { \pmb \psi } _ { c } ( { \pmb x } ) \rangle$ ⟩. Equivalently, it is a kernel machine with kernel $\begin{array} { r } { K ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathsf { \tilde { L } } \pmb { \ell } \frac { c _ { \ell } } { \ell ! } \langle \pmb { x } , \pmb { x } ^ { \prime } \rangle ^ { \ell } } \end{array}$ . With A present, f is a kernel machine with trainable kernel $\begin{array} { r } { K _ { A } ( { \pmb x } , { \pmb x } ^ { \prime } ) = \sum _ { \ell } \frac { c _ { \ell } } { \ell ! } \langle { \pmb A } { \pmb x } , { \pmb A } { \pmb x } ^ { \prime } \rangle } \end{array}$

The wonderful thing about DPR is that its gradient flow dynamics on squared loss can in many cases be reduced to a simple ODE of scalar variables and solved exactly. The punchline of this paper is that the resulting ODE is often predictive of the loss trajectories and learning times for MLPs learning low-order polynomials (Figures 1 and 3).

## 2 Related works

Our work intersects with a wide range of preexisting literature, most notably: the analysis of single- and multi-index target functions, saddle-to-saddle dynamics, staircase function learning, mean-field treatments of two layer networks, and kernel learning. We also briefly discuss potential relations to the “neural feature matrix” line of study.

Single- and multi- index models Single- and multi- index models (SIMs, MIMs respectively) have been proposed as the simplest non-trivial targets for shallow MLPs to learn. Therefore, a great deal of literature covers how MLPs are trained on them [Ben Arous et al., 2021, Bietti et al., 2022, Mousavi-Hosseini et al., 2023, Damian et al., 2023, 2022]. Similar to our experiments on staircase functions, Berthier et al. [2024] studied the gradient flow trajectory of wide networks in the mean-field regime and found the same multi-timescale plateau-and-drop structure we showcase. Similar to our ODE, Arnaboldi et al. [2023, 2024] derived a dimensionless ODE detailing the batch size an MLP requires to escape a saddle point when trained on a SIM. We occasionally induce a separation of timescales argument to reduce our feature ODE, which is preceeded by Dandi et al. [2024a], Damian et al. [2023], Dandi et al. [2024b], Mousavi-Hosseini et al. [2025] making batch size arguments for when this is valid on MIMs.

![](images/c18070d9a11fa5703488c1ed95275179263e43714d513a31b01da468c5cc3335.jpg)  
Figure 2: Comparison between a shallow MLP and deep polynomial regression (DPR). Left. A shallow MLP has nonlinearities applied elementwise, and each nonlinearity contains many polynomial orders. Diferent elements tend to diferent components of the preactivation. Right. In DPR, the nonlinear operation is a preactivation expansion which separately generates monomial interactions of the input vector.

Saddle-to-saddle and step-like learning dynamics The empirical observation that gradient flow on smallinitialization MLPs proceeds via long plateaus punctuated by rapid drops has by now been documented from many angles: saddle-to-saddle dynamics in deep linear networks [Jacot et al., 2021]; plateaus and timescale separation in two-layer SIMs [Berthier et al., 2024]; the implicit bias perspective of Pesme et al. [2021]; alternating gradient flows [Kunin et al., 2025]; and the structured plateaus exhibited by the teacher-student ODEs of Saad and Solla [1995b]. Our feature ODE reproduces this plateau structure (Figure 3) without explicitly solving for saddles.

Staircase functions The staircase functions studied in our experiments $( f ^ { \star } = z _ { 1 } , z _ { 1 } + z _ { 1 } z _ { 2 } , z _ { 1 } + z _ { 1 } z _ { 2 } + z _ { 1 } z _ { 2 } z _ { 3 } )$ come from the line initiated by Abbe et al. [2021, 2022, 2023]. Our work difers by analyzing and characterizing the efects of the anisotropy on the learning dynamics. Closely related, the global convergence results of Boursier et al. [2022b] for ReLU networks on orthogonal inputs also exhibit a staircase decomposition of the loss.

Mean-field analyses of two-layer networks Our DPR model and feature ODE can be compared against the more rigorous yet less intuitive mean-field literature of [Mei et al., 2018, Chizat and Bach, 2018, Sirignano and Spiliopoulos, 2020, Rotskof and Vanden-Eijnden, 2018]. At a high level, mean-field studies derive a partial diferential equation for the distribution of parameters in the N → ∞ limit and gives global-convergence results under suitable assumptions. These analyses take a slighly diferent approach to removing the efect of neuron specialization, taking an infinite-dimensional measure over neurons. Close historical analogs to our ODE can be found from Saad and Solla [1995b,a], Goldt et al. [2019], deriving order-parameter ODEs for teacher-student target functions (similar to our explicit Hermite target decomposition). Refinetti et al. [2021] used analogous ODE reductions for high-dimensional classification problems. Perhaps most similar to our work is the DMFT literature of Bordelon and Pehlevan [2022, 2023] that captures the full evolution of inner-product kernels during feature learning.

Efective ODEs and high-dimensional limit theorems for SGD A separate analytical thread, instead of taking a width limit, takes a high-dimensional input limit and derives rigorous ODE/SDE descriptions of summary statistics under one-pass SGD. Collins-Woodfin et al. [2024] established that online gradient flow on MIMs with general data covariance, admits an ODE that exactly predicts the risk and other observables. Arnaboldi et al. [2023] gave a parallel result for two-layer networks, providing a dimensionless ODE that unifies mean-field and high-dimensional descriptions.

Kernels, Hermites, and random features Several of the technical ingredients we use—the Hermite decomposition of activations to read of the level coeficients c<sub>ℓ</sub> (Section C), the dot-product kernel approximation in the “near-origin”

or “on-sphere” regime, and the polynomial truncation of the random-feature kernel are similar to results found in [Karkada et al., 2025, Wortsman and Loureiro, 2025]. The semi-frozen network discussed in Section B.1 can be viewed as a random feature approximation to the DPR model; we owe Rahimi and Recht [2007] for their random feature work. Mei et al. [2022] used Hermite expansions to characterize precisely when neural networks beat their associated kernels, providing the polynomial-by-polynomial picture of approximation that our DPR truncation order p inherits.

Neural feature matrix and feature-learning mechanisms We believe DPR is a sound tool to accurately describe the evolution of the neural feature matrix (NFM) $W _ { \mathrm { i n } } ^ { \top } W _ { \mathrm { i n } }$ , introduced in [Radhakrishnan et al., 2024].

Preconditioned kernel learning We are not the first to suggest applying a learned preconditioner to the inputs of a kernel machine. Wilson et al. [2015] first proposed this technique, using it to improve the sample complexity of kernel regression.

## 3 The feature ODE

We draw data as x $\sim \mathcal { N } ( \mathbf { 0 } , \mathbf { T } )$ , with anisotropic covariance $\mathbf { r } = \mathrm { d i a g } ( \gamma _ { 1 } , \cdot \cdot \cdot , \gamma _ { d } )$ . Given a target function $f ^ { * }$ , we construct the square loss $\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 } \mathbb { E } _ { \pmb { x } \sim \mathcal { N } ( 0 , \pmb { \Gamma } ) } \Big [ \big ( f ^ { * } \big ( \pmb { x } \big ) - f _ { \mathrm { D P R } } \big ( \pmb { x } \big ) \big ) ^ { 2 } \Big ] } \end{array}$ . We study gradient flow optimization of the DPR parameters:

$$
\dot { \pmb { A } } = - \nabla _ { \pmb { A } } \mathcal { L } , \qquad \dot { \pmb { B } } = - \nabla _ { B } \mathcal { L } , \qquad \pmb { A } ( 0 ) = 1 / \sqrt { d } \cdot \pmb { I } , \qquad \pmb { B } ( 0 ) = \pmb { 0 } .\tag{4}
$$

Observe that the model function $f _ { \mathrm { D P R } }$ can be written out explicitly in polynomial form:

$$
f _ { \mathrm { D P R } } ( \pmb { x } ) = \sum _ { \ell \geq 0 } \sum _ { \beta \in [ d ] ^ { \ell } } \frac { \sqrt { c _ { \ell } } } { \sqrt { \ell ! } } B _ { \beta } \prod _ { m = 1 } ^ { \ell } ( \pmb { A } \pmb { x } ) _ { \beta _ { m } } .\tag{5}
$$

The great simplifying blessing is that, when $f ^ { * }$ depends on only a handful of the coordinates of $^ { \mathrm { ~ ~ } } { \mathbf { x } } ,$ then most of these terms can be neglected. In particular, $\operatorname { i f } f ^ { * }$ depends on only a subset of coordinates $\{ x _ { i } \}$ such that $i \in \mathcal { T } \subset [ d ]$ , then any readout parameters $b _ { \beta }$ with $\beta \notin \mathcal { L } ^ { d }$ will remain identically zero under gradient flow. Furthermore, first-layer parameters $A _ { i j }$ with $j \not \in \mathcal { I }$ will feel no gradient and remain equal to $\kappa \delta _ { i j }$

We now invoke two assumptions on the data. First, we assume that $\gamma _ { i } \ll 1 . ^ { 3 }$ With this assumption, we find that terms of higher orders are learned more slowly, and only the lowest-order terms of $f _ { \mathrm { D P R } }$ required to represent the target function $f ^ { * }$ grow substantially and are relevant to the dynamics. Second, we expand the target function as a sum of Hermite polynomials: $\begin{array} { r } { f ^ { * } ( \pmb { x } ) \overset {  } { = } \sum _ { \pmb { \alpha } \in \mathbb { N } _ { \Omega } ^ { d } } v _ { \pmb { \alpha } } h _ { \pmb { \alpha } } ( \mathbf { I } ^ { - 1 / 2 } \pmb { x } ) } \end{array}$ , and make a sparseness assumption on the coeficients $v _ { \alpha }$ which will be satisfied in every case we discuss (the per-order parity criterion, Section B.3). See Section B for full discussion of these assumptions and the consequent derivations.

After some algebra, we are left with the following ODE:

$$
\mathcal { L } = \frac { 1 } { 2 } \sum _ { \alpha \in \mathcal { N } _ { 0 } ^ { d } } \Bigg ( { v _ { \alpha } } - \tilde { c } _ { \alpha } \cdot b _ { \alpha } \prod _ { i } a _ { i } ^ { \alpha _ { i } } \Bigg ) ^ { 2 } ,\tag{6}
$$

$$
\dot { a } _ { i } = - \partial _ { a _ { i } } \mathcal { L } , \qquad \dot { b } _ { \alpha } = - \partial _ { b _ { \alpha } } \mathcal { L } ,\tag{7}
$$

where $\begin{array} { r } { \tilde { c } _ { \pmb { \alpha } } : = \sqrt { c _ { | \pmb { \alpha } | } } \cdot \prod _ { i = 1 } ^ { d } \gamma _ { i } ^ { \alpha _ { i } / 2 } } \end{array}$ . We refer to the above as the feature ODE. The point of the feature ODE is that (a) when the target function is very sparse (i.e. only a few $v _ { \alpha }$ are nonzero), then we need only track a handful of scalar variables, and (b) the resulting loss dynamics resemble those of a shallow MLP trained on the same target function.

When the target function consists of a single Hermite term $f ^ { * } ( x ) = h _ { \alpha } ( x )$ , the loss in the feature ODE simplifies to just $\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 } ( 1 - \tilde { c } _ { \pmb { \alpha } } \cdot b \prod _ { i } a _ { i } ^ { \alpha _ { i } } ) ^ { 2 } } \end{array}$ , and the resulting dynamics of a few variables can be studied nicely. The loss and its dynamics cannot be directly solved, but a number of approximations can be made, and are done so in Section D, such

![](images/e38b3cbdacc1f1aa61deb6489b555eb4a4796aa3157ec9102eb1b90dfde14f53.jpg)  
Figure 3: The feature ODE predicts loss trajectories for shallow MLPs learning low-degree “staircase functions.” Predicted loss trajectories are staircases with a number of steps equal to the order of the target function. True loss curves for shallow MLPs roughly follow these staircases, with steps at roughly the predicted times. See Section A for experimental details.

that an explicit solution for the efective optimization time $\tau \equiv \eta \cdot n _ { \mathrm { i t e r } }$ can be found. In short, we define the efective optimization time by having b reach its final value, of $b _ { f } = \bar { c } _ { | \alpha | } ^ { - } \overline { { { \sum _ { i } \alpha _ { i } + 1 } } }$

$$
\int _ { 0 } ^ { \tau } \mathrm { d } t = \tau = \int _ { 0 } ^ { b _ { f } } \frac { \mathrm { d } b } { \tilde { c } _ { | \alpha | } \prod _ { i = 1 } ^ { d } \left( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) ^ { \frac { \alpha _ { i } } { 2 } } } .\tag{8}
$$

We experimentally validate that MLPs learn target functions in an amount of gradient steps defined by Equation (8), finding great agreement with results summarized in the right side of Figure 1.

Example calculations. To concretize our findings, we provide an in-depth discussion on how the feature ODE can be used to analyze an MLP learning a number of targets presented in the paper. Throughout our discussion, we will use normalized coordinates $z _ { i } = x _ { i } / \sqrt { \gamma } _ { i }$ such that $\left| f ^ { \star } \right| = 1$

Linear target. On a linear target, the feature ODE and dynamics reduce to the form

$$
{ \mathcal { L } } = { \frac { 1 } { 2 } } \left( 1 - { \tilde { c } } _ { 1 } a _ { i } b \right) ^ { 2 } \qquad { \dot { a } } _ { i } = \left( 1 - { \tilde { c } } _ { 1 } a _ { i } b \right) \cdot { \tilde { c } } _ { 1 } b \qquad { \dot { b } } = \left( 1 - { \tilde { c } } _ { 1 } a _ { i } b \right) \cdot { \tilde { c } } _ { 1 } a _ { i } .
$$

where we would like to pause to appreciate the simplicity of our ODE: describing the learning dynamics can reliably be captured by only two dynamical variables! For the remainder of the examples, we leave out the dynamics in a and b as they follow from simple derivatives of the loss.

For the loss trajectory, refer to the top left of Figure 1, with the efective optimization time being

$$
\tau = \int _ { 0 } ^ { \tilde { c } ^ { \frac { 1 } { 2 } } } \frac { \mathrm { d } b } { \tilde { c } _ { 1 } \sqrt { b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) } } \approx \frac { 1 } { \tilde { c } _ { 1 } } \ln \left( \frac { 2 } { \sqrt { \tilde { c } _ { 1 } } a _ { i } ( 0 ) } \right)
$$

Quadratic targets. There are two diferent types of quadratic targets to consider: the first consists of a mixedindex target, z<sub>i</sub>z<sub>j</sub>, and the second is a pure-quadratic target $h _ { 2 } ( z _ { i } )$ . We review the pure-quadratic case’s efective optimization time, with the mixed-index being developed similarly. We note that the feature ODE can be obtained by

![](images/cabe96395e514cfe8e6e954a4834c7fe3259743a1ef730640e0642c67aa9c058.jpg)  
Figure 4: The feature ODE can predict MLP loss dynamics despite neuron-level specialization in MLPs. We train a two-layer MLP on a two-term staircase $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 }$ and overlay the feature ODE prediction. Top. We plot the online loss of the MLP and of the feature ODE. Bottom. Each row of the heatmap tracks one neuron’s preactivation variance over training, and neurons are sorted by their variance at the end of training. The horizontal striations show neuron-level specialization: individual neurons carry distinctly diferent preactivation variances. Nevertheless, the feature ODE captures the MLP’s staircase loss.

similar arguments used for the linear case,<sup>4</sup> We find the dynamics and efective optimization time as,

$$
\tau \approx \int _ { 0 } ^ { \infty } \frac { \mathrm { d } b } { \tilde { c } _ { 2 } \left( 2 b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) } = \frac { a _ { 0 } ^ { - 1 } } { \tilde { c } _ { 2 } } \frac { \pi } { 2 ^ { 3 / 2 } } ,
$$

The mixed-index case has one small but noticeable change: the efective optimization time has the $2 ^ { 3 / 2 }$ in the denominator replaced with a 2. Surprisingly, the repeated-index quadratic is learned with exactly $\sqrt { 2 }$ times fewer steps than it would take to learn a mixed-index quadratic, all other properties remaining constant (validated in Figure 1)!

Staircase with two terms Thus far, we have only discussed target functions with a single term. Notably, this is vastly diferent from what most real targets consist of. To get closer to real targets, we analyze feature ODE on a “staircase function” (defined in Abbe et al. [2023]) with two terms: $f ^ { \star } = z _ { i } + z _ { i } z _ { j }$ . The dynamics of the feature ODE are not discussed in detail, however they follow similar arguments to those of the linear and quadratic cases and are instead shown in Figure 3. We note that there is currently no clean expression for the optimization time of the feature ODE on staircase functions, although presume it is possible.

As more terms are added, the feature ODE will undoubtedly become more complex, however we believe the feature ODE will still be a useful tool that can be used to analyze the learning dynamics of the shallow MLP.

## 4 The feature ODE cannot describe neuron-level specialization

Although our models are fully expressive and capture the training dynamics of simple target functions well, we cannot fully capture all of the training dynamics of MLPs. We removed the elementwise nonlinearity and replaced it with a global nonlinear operation. With each neuron being indistinguishable, we have thus eliminated all neuron-level specialization.

MLPs trained on general tasks, however, are known to develop sparse, specialized weights (i.e. some rows $w _ { \mathrm { i n } , k }$ of $W _ { \mathrm { i n } }$ satisfy $\lVert \pmb { w } _ { \mathrm { i n } , k } \rVert _ { 1 } \ll \sqrt { d } \lVert \pmb { w } _ { \mathrm { i n } , k } \rVert _ { 2 } )$ . Select neurons can attend to few input coordinates or directions [Maennel et al., 2018, Barak et al., 2022, Boursier and Flammarion, 2024], notably when developing localized receptive fields on image data [Ingrosso and Goldt, 2022, Lufkin et al., 2024] or solving algorithmic tasks like modular addition [Gromov, 2023, Nanda et al., 2023, Kunin et al., 2025, He et al., 2026]. Our model is unable to capture these neuron-level dynamics.

For simple target functions such as the staircase in Figure 4, the efects of neuron-level specialization are minor and the feature ODE can predict the MLP’s loss well even though MLP neurons clearly specialize. For other target functions, specialization becomes significant, and the feature ODE predicts the loss less accurately. See Section E for details.

## 5 Outlook & Conclusion

We are excited about future work utilizing this framework to study feature learning. We believe our framework allows for a principled study of the evolution of the neural feature matrix $W _ { \mathrm { i n } } ^ { \top } W _ { \mathrm { i n } }$ (introduced in Radhakrishnan et al. [2024]), with some, albeit weaker, evidence for a principled study of the output layer weight vector ${ \pmb w } _ { \mathrm { o u t } }$ being possible. We are optimistic we can predict dynamics of MLPs learning more complicated functions in varied hyperparameter regimes by further studying DPR and studying more general simplified ODEs.

## Acknowledgements

All authors were supported by the FLAB initiative at Imbue and are grateful for Imbue’s support of fundamental science. All authors are grateful to Dhruva Karkada for extensive discussions about the project.

JT thanks the 2025 learning mechanics retreat group for the realization of the promising potential for the feature ODE, along with the space given to meet with co-authors in-person. BO acknowledges support by the National Science Foundation Graduate Research Fellowship Program. BO is also grateful to Surbhi Goel for extensive discussion about the project.

## References

Emmanuel Abbe, Enric Boix-Adserà, Matthew Brennan, Guy Bresler, and Dheeraj Nagaraj. The staircase property: How hierarchical structure can guide deep learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2108.10573.

Emmanuel Abbe, Enric Boix-Adserà, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly suficient condition for SGD learning of sparse functions on two-layer neural networks. In Conference on Learning Theory (COLT), 2022. URL https://arxiv.org/abs/2202.08658.

Emmanuel Abbe, Enric Boix-Adsera, and Theodor Misiakiewicz. Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics, 2023. URL https://arxiv.org/abs/2302.11055.

Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks. In Conference on Learning Theory (COLT), 2023.

Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. Online learning and information exponents: On the importance of batch size, and time/complexity tradeofs. International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2406.02157.

Boaz Barak, Benjamin L. Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: SGD learns parities near the computational limit. In Advances in Neural Information Processing Systems, 2022.

Gérard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal ofMachine Learning Research, 22(106):1–51, 2021. URL https: //arxiv.org/abs/2003.10409.

Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations ofComputational Mathematics, 2024. URL https://arxiv.org/abs/2303.00055.

Alberto Bietti, Joan Bruna, Clayton Sanford, and Min Jae Song. Learning single-index models with shallow neural networks. Advances in neural information processing systems, 35:9768–9783, 2022.

Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org abs/2205.09653

Blake Bordelon and Cengiz Pehlevan. Dynamics of finite width kernel and prediction fluctuations in mean field neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv. org/abs/2304.03408.

Etienne Boursier and Nicolas Flammarion. Early alignment in two-layer networks training is a two-edged sword. arXiv preprint arXiv:2401.10791, 2024.

Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs. In Advances in Neural Information Processing Systems, 2022a.

Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs. In Advances in Neural Information Processing Systems (NeurIPS), 2022b.

Lénaïc Chizat and Francis Bach. On the global convergence of gradient descent for over-parameterized models using optimal transport. In Advances in Neural Information Processing Systems (NeurIPS), 2018. URL https: //arxiv.org/abs/1805.09545.

Elizabeth Collins-Woodfin, Courtney Paquette, Elliot Paquette, and Loucas Pillaud-Vivien. Hitting the highdimensional notes: An ODE for SGD learning dynamics on GLMs and multi-index models. Information and Inference: A Journal ofthe IMA, 13(4), 2024. URL https://arxiv.org/abs/2308.08977.

Alex Damian, Jason D. Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory (COLT), 2022. URL https://arxiv.org/abs/2206.15144.

Alex Damian, Eshaan Nichani, Loucas Pillaud-Vivien, and Jason D. Lee. Smoothing the landscape boosts the signal for SGD: Optimal sample complexity for learning single index models. Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.10633.

Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How two-layer neural networks learn, one (giant) step at a time. Journal of Machine Learning Research, 25(349), 2024a. URL https://arxiv. org/abs/2305.18270.

Yatin Dandi, Luca Pesce, Hugo Cui, Florent Krzakala, Yue M. Lu, and Bruno Loureiro. Repetita iuvant: Data repetition allows SGD to learn high-dimensional multi-index functions. arXiv preprint, 2024b. URL https: //arxiv.org/abs/2405.15459.

Sebastian Goldt, Madhu Advani, Andrew M. Saxe, Florent Krzakala, and Lenka Zdeborová. Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1906.08632.

Andrey Gromov. Grokking modular arithmetic. arXiv preprint arXiv:2301.02679, 2023.

Jianliang He, Leda Wang, Siyu Chen, and Zhuoran Yang. On the mechanism and dynamics of modular addition: Fourier features, lottery ticket, and grokking. arXiv preprint arXiv:2602.16849, 2026.

Alessandro Ingrosso and Sebastian Goldt. Data-driven emergence of convolutional structure in neural networks. Proceedings ofthe National Academy ofSciences, 119(40), 2022.

Arthur Jacot, François Ged, Berfin Şimşek, Clément Hongler, and Franck Gabriel. Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. arXiv preprint, 2021. URL https: //arxiv.org/abs/2106.15933.

Dhruva Karkada, Joseph Turnbull, Yuxi Liu, and James B. Simon. Predicting kernel regression learning curves from only raw data statistics, 2025. URL https://arxiv.org/abs/2510.14878.

Daniel Kunin, Giovanni Luca Marchetti, Feng Chen, Dhruva Karkada, James B. Simon, Michael R. DeWeese, Surya Ganguli, and Nina Miolane. Alternating gradient flows: A theory of feature learning in two-layer neural networks, 2025. URL https://arxiv.org/abs/2506.06489.

Leon Lufkin, Andrew M. Saxe, and Erin Grant. Nonlinear dynamics of localization in neural receptive fields. In Advances in Neural Information Processing Systems, 2024

Hartmut Maennel, Olivier Bousquet, and Sylvain Gelly. Gradient descent quantizes ReLU network features. arXiv preprint arXiv:1803.08367, 2018.

Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings ofthe National Academy ofSciences, 115(33):E7665–E7671, 2018. URL https://arxiv. org/abs/1804.06561.

Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis, 59, 2022.

Alireza Mousavi-Hosseini, Denny Wu, Taiji Suzuki, and Murat A. Erdoğdu. Gradient-based feature learning under structured data. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Alireza Mousavi-Hosseini, Denny Wu, and Murat A. Erdoğdu. Learning multi-index models with neural networks via mean-field Langevin dynamics. In International Conference on Learning Representations (ICLR), 2025.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023. URL https://arxiv.org/abs/2301.05217.

Scott Pesme and Nicolas Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. In Advances in Neural Information Processing Systems, 2023.

Scott Pesme, Loucas Pillaud-Vivien, and Nicolas Flammarion. Implicit bias of SGD for diagonal linear networks: a provable benefit of stochasticity. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, and Mikhail Belkin. Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science, 383(6690):1461–1467, 2024.

Ali Rahimi and Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems (NeurIPS), volume 3, page 5, 2007.

Maria Refinetti, Sebastian Goldt, Florent Krzakala, and Lenka Zdeborová. Classifying high-dimensional Gaussian mixtures: Where kernel methods fail and neural networks succeed. In International Conference on Machine Learning (ICML), 2021. URL https://arxiv.org/abs/2102.11742.

Grant M. Rotskof and Eric Vanden-Eijnden. Trainability and accuracy of neural networks: An interacting particle system approach. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

David Saad and Sara A. Solla. Exact solution for on-line learning in multilayer neural networks. Physical Review Letters, 74(21):4337–4340, 1995a

David Saad and Sara A. Solla. On-line learning in soft committee machines. Physical Review E, 52(4):4225–4243, 1995b.

Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks: A central limit theorem. Stochastic Processes and their Applications, 130(3):1820–1852, 2020.

Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning, 2015. URL https://arxiv.org/abs/1511.02222.

Arie Wortsman and Bruno Loureiro. Kernel ridge regression under power-law data: spectrum and generalization. arXiv preprint arXiv:2510.04780, 2025.

Greg Yang and Edward J Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In International Conference on Machine Learning, pages 11727–11737. PMLR, 2021.

## A Further experimental details

For the MLP, we use $\mu \mathrm { P }$ parameterization Yang and Hu [2021]:

$$
W _ { \mathrm { i n } } [ n , i ] \sim \mathcal { N } ( 0 , 1 / d ) , \quad W _ { \mathrm { o u t } } [ n ] = \mathbf { 0 } , \quad \eta _ { W _ { \mathrm { i n } } } = \eta N , \quad \eta _ { W _ { \mathrm { o u t } } } = \eta / N ,
$$

where $W _ { \mathrm { i n } } = \mathbb { R } ^ { N \times d }$ , and we set $W _ { \mathrm { o u t } }$ to be the all zero’s vector for consistency with the feature ODE (as $N  \infty ,$ $W _ { \mathrm { { o u t } } }$ tends to 0). For the SFN and DPR, we initialize $\begin{array} { r } { { \cal A } = { \frac { 1 } { \sqrt { d } } } { \cal I } _ { d } } \end{array}$ and zero out the readout tensors $B ^ { ( \ell ) }$ . All three models train with online SGD on streamed Gaussian data.

Figure 1 experimental details. Experiments are performed with data $\mathbf { \boldsymbol { x } } \sim \mathcal { N } ( 0 , \Gamma )$ , with $\pmb { x } \in \mathbb { R } ^ { 1 0 0 }$ and $\Gamma =$ diag(γ ) where $\gamma _ { i } \sim ( i + 6 ) ^ { - 1 . 8 }$ normalized such that $\dot { \sum } _ { i } \gamma _ { i } = 1$ . The learning rate is set to $\eta = 1 0 ^ { - 2 }$ which we find to be near the gradient flow regime. Our data is streamed online with a batch size $b s z = 4 0 0 0 - \mathrm { v }$ e note that any reasonably large choice for the batch size reliably reproduces our results. Our MLP is setup in the $\mu \mathrm { P }$ regime Yang and Hu [2021], with a width $N = 8 1 9 2$ . Our activation function of choice is a shifted SiLU: $\sigma ( x ) = \mathrm { S i L U } ( x + 0 . 5 )$ which ofers the nice properties of (1) being relatively close to the standard practice ReLU, and (2) having a full power series, with the Taylor expansion around $x = 0$ having no coeficients of 0. We make explicit note that the vast majority of biasless nonlinearities are typically not full in their power series. An exponential moving average smoother with constant 0.9 $( \mathcal { L } ( t ) = 0 . 9 \mathcal { L } ( t - 1 ) + 0 . 1 \Delta \mathcal { L } ( t )$ is used to smooth over random Gaussian fluctuations in the data. Loss curves. Training is done with a set 100, 000 gradient steps, at which point the run terminates. Efective optimization time predictions. We train on a number of targets finding great agreement on all linear terms with slight disagreement on quadratic terms with high covariance $\gamma _ { i } .$ Cubic and higher order target functions are not included as the predicted optimization times were prohibitive. We believe anisotropic efects that make real MLP optimization times difer than the ODE predicted times are more prevalent for these higher order targets.

Figure 3 experimental details. Both panels train an MLP of width N = 8192 online with batch size 10,000 and a single init seed, and overlay the feature ODE prediction. Panel (a) targets the 2-staircase $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 }$ with $d = 2$ , a power-law exponent $\alpha = 1 0$ yielding $\gamma \approx ( 2 . 0 0 \times 1 0 ^ { - 2 } , 1 . 9 \bar { 5 } \times 1 0 ^ { - 5 } )$ , learning rate $\eta = 0 . 5 ,$ and cosine activation $\sigma ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 )$ . Panel (b) targets the 3-staircase $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 } + z _ { 1 } z _ { 2 } z _ { 3 }$ with $d = 3 ,$ a power-law exponent $\alpha = 6$ yielding $\gamma \approx ( 2 . 9 5 \times 1 0 ^ { - 2 } , 4 . 6 1 \times 1 0 ^ { - 4 } , 4 . 0 5 \times 1 0 ^ { - 5 } )$ , learning rate $\eta = 0 . 0 3$ , and tanh-half activation. We were only able to see the staircase behavior with a large ratio of data eigenvalues. With larger data eigenvalues and smaller learning rate the feature ODE becomes less predictive.

Figure 5 and Figure 6 experimental details. Both figures share the same configuration: $d = 3 ,$ a power-law exponent α = 1.5 yielding γ $\mathbf { \tilde { \Sigma } } \approx ( 3 . 1 1 \times 1 0 ^ { - 3 } , 1 . 1 0 \times 1 0 ^ { - 3 } , 5 . 9 8 \times 1 0 ^ { - 4 } )$ , the cosine activation $\sigma ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 )$ with no biases, learning rate $\eta = 0 . 2 ,$ , batch size $1 0 { , } 0 0 0 ,$ and a single init seed. The SFN uses width $N = 8 1 9 2 .$ and DPR uses truncation order $p = 3$ to accommodate the highest-order target. We sweep across four targets in z coordinates: $z _ { 1 } , \mathrm { H e } _ { 2 } ( z _ { 1 } ) , z _ { 1 } + z _ { 1 } z _ { 2 } ,$ , and $z _ { 1 } + z _ { 1 } z _ { 2 } + z _ { 1 } z _ { 2 } z _ { 3 }$

Figure 7, Figure 8, and Figure 9 experimental details. Each figure compares the diagonal-A dynamics of an MLP run against an SFN run on the same target. All six runs use width $N = 8 1 9 2$ , batch size 10,000, the cosine activation $\overset { \_ } { \sigma } ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 )$ with no biases, and a single init seed. The per-run hyperparameters are:

• Figure 7, target $f ^ { * } ( z ) = z _ { 1 } \colon \mathrm { M L P }$ uses $d = 2$ with uniform $\gamma = ( 2 . 5 0 \times 1 0 ^ { - 3 } , 2 . 5 0 \times 1 0 ^ { - 3 } )$ and $\eta = 0 . 1 ;$ SFN uses $d = 3$ with α = 1.5 yielding $\gamma \approx ( 3 . 1 1 \times 1 0 ^ { - 3 } , 1 . 1 0 \times 1 0 ^ { - 3 } , 5 . 9 8 \times 1 0 ^ { - 4 } )$ and $\eta = 0 . 2 .$

• Figure 8, target $f ^ { * } ( z ) = \mathrm { H e } _ { 2 } ( z _ { 1 } ) + \mathrm { H e } _ { 2 } ( z _ { 2 } )$ : MLP and SFN both use $d = 2$ with uniform $\gamma = ( 2 . 5 0 \times$ $1 0 ^ { - 3 } , 2 . 5 0 \times 1 0 ^ { - 3 } )$ and $\eta = 0 . 1$

• Figure 9, target $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 } \colon \mathrm { M L P }$ uses $d = 2$ with $\alpha = 1 0$ yielding $\gamma \approx ( 2 . 0 0 \times 1 0 ^ { - 2 } , 1 . 9 5 \times 1 0 ^ { - 5 } )$ and $\eta = 0 . 5 ; \mathrm { S F N }$ uses $d = 3$ with $\alpha = 1 . 5$ yielding $\gamma \approx ( 3 . 1 1 \times 1 0 ^ { - 3 } , 1 . 1 0 \times 1 0 ^ { - 3 } , 5 . 9 8 \times 1 0 ^ { - 4 } )$ and $\eta = 0 . 2$

## B Detailed derivation of the feature ODE from MLPs

We derive the feature ODE from an MLP with MSE loss in three steps. Rather than conjecturing DPR directly from the MLP, we introduce an intermediate model—the semi-frozen network (SFN)—whose infinite-width limit recovers DPR, and we derive the SFN from the MLP by a conjecture on the input layer’s weight matrix.

## B.1 Further details on MLPs to Semi-Frozen Networks

A one-hidden-layer MLP computes

$$
f _ { \mathrm { M L P } } ( \pmb { x } ) = W _ { \mathrm { o u t } } \sigma ( \pmb { W } _ { 1 } \pmb { x } ) , \qquad \pmb { W } _ { 1 } \in \mathbb { R } ^ { N \times d } ,
$$

where σ acts pointwise. To pass from the MLP to a semi-frozen network, we assume that the hidden neurons form a homogeneous population: no single neuron encodes special information, and adding or removing any one has a negligible efect on the output. We require this homogeneity to hold throughout training, not just at initialization.

The input layer plays two roles. It rotates the data, upweighting coordinates that matter for the output; and it routes the rotated data into the pointwise nonlinearity, where diferent neurons see diferent mixtures. The SFN disentangles these roles by freezing the routing matrix and training only the rotation.

Skinny SVD. Let $\pmb { W } _ { 1 } = \pmb { U } \pmb { S } \pmb { V } ^ { \top }$ be the thin SVD, with $U \in \mathbb { R } ^ { N \times d }$ having orthonormal columns, $S \in \mathbb { R } ^ { d \times d }$ diagonal, and $V \in \mathbb { R } ^ { d \times d }$ orthogonal. Defining $\pmb { A } : = \pmb { S V } ^ { \top }$ and $W _ { \mathrm { f r o z e n } } : = U$ gives

$$
f _ { \mathrm { S F N } } ( x ) = W _ { \mathrm { o u t } } \tilde { \sigma } ( W _ { \mathrm { f r o z e n } } A x ) ,
$$

where $\begin{array} { r } { \tilde { \sigma } ( x ) = \sum _ { \ell } q _ { \ell } c _ { \ell } x ^ { \ell } , c _ { \ell } \ ' s } \end{array}$ given by the Taylor series expansion of the ML ${ \mathbf { } } P _ { } ^ { } s \sigma$ (see Section C). We introduce a factor of q as a numerical parameter which we found to follow the pattern of $0 : 1 , 1 : 1 , 2 : 2$ for our experiments. We suspect $q _ { \ell } = \ell !$ , although we provide no such proof. σ˜ still acts pointwise, but now on the random features $W _ { \mathrm { f r o z e n } } A x$ . We initialize $\begin{array} { r } { { \cal A } = \frac { 1 } { \sqrt { d } } { \bar { \cal I } } _ { d } } \end{array}$ to remain consistent with $\mu \mathrm { P }$ scaling.

The SFN freezes U at its initialization and trains only A, reducing the first-layer parameter count from Nd to $d ^ { 2 }$ In practice, rather than taking U from the SVD of a specific $W _ { 1 }$ , we initialize each entry of $W _ { \mathrm { f r o z e n } }$ as iid Gaussian with variance $\sigma _ { \mathrm { f r o z e n } } ^ { 2 } / d$ (the standard $\mu \mathrm { P } / $ random-feature scaling) and hold it fixed throughout training. The scale $\sigma _ { \mathrm { f r o z e n } }$ is a free hyperparameter that controls the preactivation magnitude; we absorb it into the data eigenvalues $\gamma _ { i }$ throughout (see the footnote in Section B.2), so it does not appear explicitly in the kernel or the feature ODE. This shift assigns the nonlinearity’s expressive role to a fixed random projection. The price is any reorganization of the frozen directions—basis rotations, ICA-style updates—which we discuss in Section E.

Width and concentration. As $N \to \infty$ , the SFN’s random feature kernel concentrates around its expectation over the frozen weights, which is the SFN kernel we use in Section B.2. At finite N, the SFN’s Gram matrix fluctuates by $O ( 1 / \sqrt { N } )$ across seeds, so the SFN matches the kernel only up to width corrections.

Neural feature matrix. The MLP carries a “neural feature matrix” $\mathrm { N F M } : = W _ { \mathrm { i n } } ^ { \top } W _ { \mathrm { i n } }$ . For an SFN, A plays the analogous role:

$$
\begin{array} { r } { A \ \sim \ \sqrt { \frac { 1 } { N } { W _ { \mathrm { i n } } ^ { \top } } { W _ { \mathrm { i n } } } } . } \end{array}
$$

## B.2 Further details on Semi-Frozen Networks to DPR

Here, we replace the pointwise activation in the SFN with explicit polynomial features and obtain deep polynomial regression (DPR). At a high level, the SFN’s feature kernel admits a polynomial decomposition, and DPR is the explicit-feature model that realizes the same decomposition.

The SFN feature kernel. Because $W _ { \mathrm { f r o z e n } }$ is frozen, the random feature kernel holds throughout training, not only at initialization (unlike the NNGP). Taking $N  \infty ,$ the random feature average converges to its expectation over the frozen weights<sup>5</sup> [Rahimi and Recht, 2007]:

$$
\begin{array} { r } { K _ { \mathrm { S F N } } ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathbb { E } _ { \pmb { w } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } / d ) } \left[ \tilde { \sigma } ( \pmb { w } ^ { \top } \pmb { A } \pmb { x } ) \sigma ( \pmb { w } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } ) \right] . } \end{array}
$$

The $1 / d$ per-component variance is the standard $\mu \mathrm { P } / $ random-feature scaling that keeps the preactivation $O ( 1 )$ as d grows. The preactivations $\pmb { w } ^ { \top }$ Ax and $\mathbf { \boldsymbol { w } } ^ { \top } \mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { x } } ^ { \prime }$ are jointly Gaussian, so the kernel reduces to a two-dimensional Gaussian expectation:

$$
K _ { \mathrm { S F N } } ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathbb { E } _ { z _ { 1 } , z _ { 2 } \sim \mathcal { N } ( 0 , \pmb { \Sigma } ^ { \prime } ) } \left[ \tilde { \sigma } \left( z _ { 1 } \right) \tilde { \sigma } \left( z _ { 2 } \right) \right] , \quad \pmb { \Sigma } ^ { \prime } = \frac { 1 } { d } \left( \begin{array} { c c } { \left\| \pmb { A } \pmb { x } \right\| ^ { 2 } } & { \pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } } \\ { \pmb { x } ^ { \prime } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } } & { \left\| \pmb { A } \pmb { x } ^ { \prime } \right\| ^ { 2 } } \end{array} \right) .
$$

The kernel therefore depends on three scalars: the two norms $\left\| A x \right\| ^ { 2 } , \left\| A x ^ { \prime } \right\| ^ { 2 }$ and the dot product $\pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } ^ { \prime }$

Dot-product approximation. The kernel depends on all three scalars in general, but reduces to a function of the dot product alone in two regimes. First, when $\| \pmb { A x } \| ^ { 2 }$ concentrates across data points (the “on-sphere” regime), the diagonal entries factor out as a data-independent prefactor. Second, when $\left\| A x \right\| ^ { 2 } / d$ stays small for all data points (the “near-origin” regime), the diagonal entries sit near zero and the kernel ignores their variation entirely. The near-origin regime corresponds to the wide-kernel-theory limit; the on-sphere regime is weaker, allowing larger norms but requiring them to be approximately equal. Small data eigenvalues place us in the near-origin regime, since $\begin{array} { r } { \mathbb { E } _ { \pmb { x } } \left[ \left\| \pmb { A } \pmb { x } \right\| ^ { 2 } / d \right] = ( 1 / d ) \sum _ { j } a _ { j } ^ { 2 } \gamma _ { j } } \end{array}$ . In both cases, the kernel reduces to

$$
\begin{array} { r } { K _ { \mathrm { S F N } } ( \pmb { x } , \pmb { x } ^ { \prime } ) \approx k _ { \mathrm { S F N } } \big ( \frac { 1 } { d } \pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } \big ) . } \end{array}
$$

Polynomial decomposition. Taylor-expanding k<sub>SFN</sub> gives

$$
 { K } _ { \mathrm { S F N } } (  { \boldsymbol { \mathbf { x } } } ,  { \boldsymbol { \mathbf { x } } } ^ { \prime } ) \approx \sum _ { \ell = 0 } ^ { \infty } \frac { c _ { \ell } } { \ell ! } \left( \frac { 1 } { d }  { \boldsymbol { \mathbf { x } } } ^ { \top }  { \boldsymbol { A } } ^ { \top }  { \boldsymbol { A } }  { \boldsymbol { \mathbf { x } } } ^ { \prime } \right) ^ { \ell } \equiv K _ { \mathrm { D P R } } (  { \boldsymbol { \mathbf { x } } } ,  { \boldsymbol { \mathbf { x } } } ^ { \prime } ) ,
$$

where the $( 1 / d ) ^ { \ell }$ factors are absorbed into $c _ { \ell } .$ . Applying Mercer’s theorem with eigenfunctions $\phi \sim \sqrt { c _ { \ell } / \ell ! } ( A x ) ^ { \otimes \ell }$ yields the DPR model:

$$
\begin{array} { r } { f _ { \mathrm { D P R } } ( \pmb { x } ) = \displaystyle \sum _ { \ell = 0 } ^ { p } \sqrt { \frac { c _ { \ell } } { \ell ! } } B ^ { ( \ell ) } ( \pmb { A } \pmb { x } ) ^ { \otimes \ell } , } \end{array}
$$

with trainable symmetric order-ℓ tensors $B ^ { ( \ell ) }$ and an order-p truncation.

Truncation error. DPR drops all terms $\ell > p$ in the expansion above. The order-ℓ kernel contribution is $\begin{array} { r } { \frac { c _ { \ell } } { \ell ! } \big ( \frac { 1 } { d } \pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } \big ) ^ { \ell } } \end{array}$ , which decays with ℓ when $\left\| A \bar { \boldsymbol { x } } \right\| ^ { 2 } / d$ stays small. Taking the expectation over data $\pmb { x } \sim \mathcal { N } ( 0 , \mathbf { T } )$ gives $\begin{array} { r } { \mathbb { E } _ { \pmb { x } } \Big [ \big \| \pmb { A } \pmb { x } \big \| ^ { 2 } / d \Big ] = ( 1 / d ) \sum _ { j } a _ { j } ^ { 2 } \gamma _ { j } } \end{array}$ , so smaller data eigenvalues and larger p both tighten the approximation.

## B.3 Further details on DPR to feature ODE

We simplify DPR’s gradient flow ODE to a system of scalar variables in A and the target-aligned entries of B, recovering the feature ODE defined in the main text. The reduced state consists of $A \mathit { \ ' } _ { s }$ diagonal entries ${ \textbf { \em a } } =$ $( a _ { 1 } , \ldots , a _ { d } )$ together with one scalar $b _ { \alpha }$ per target Hermite mode—the common value of $B ^ { ( | \alpha | ) }$ on permutations of α.

The reduction rests on the same two assumptions the main text invokes (Section 3): the data eigenvalues are small, which separates the timescales of diferent polynomial orders; and the target’s nonzero Hermite coeficients $v _ { \alpha }$ obey a per-order parity criterion, which keeps A diagonal under gradient flow.

The reduced equations. Under both assumptions, only the target-aligned B entries drive the dynamics, and we replace each permutation-symmetric block $B ^ { ( | \alpha | ) }$ on multi-index α by a single scalar $b _ { \alpha }$ . Expanding f around diagonal A and computing the Hermite coeficient at each target mode α yields, after some algebra,

$$
\mathbb { E } _ { \boldsymbol { x } } \Big [ f _ { \mathrm { D P R } } ( \boldsymbol { x } ) h _ { \alpha } \big ( \Gamma ^ { - 1 / 2 } \boldsymbol { x } \big ) \Big ] = \tilde { c } _ { \alpha } b _ { \alpha } \prod _ { j } a _ { j } ^ { \alpha _ { j } } , \qquad \tilde { c } _ { \alpha } : = \sqrt { c _ { | \alpha | } } \prod _ { j } \gamma _ { j } ^ { \alpha _ { j } / 2 } ,
$$

matching the main-text definition (Section 3). The residual at mode α is $\begin{array} { r } { e _ { \pmb { \alpha } } : = v _ { \pmb { \alpha } } - \tilde { c } _ { \pmb { \alpha } } b _ { \pmb { \alpha } } \prod _ { j } a _ { j } ^ { \alpha _ { j } } } \end{array}$ , and the population gradient flow produces the feature ODE:

$$
\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 } \displaystyle \sum _ { \alpha } e _ { \alpha } ^ { 2 } , } \end{array}
$$

$$
\dot { b } _ { \alpha } = e _ { \alpha } \tilde { c } _ { \alpha } \prod _ { j } a _ { j } ^ { \alpha _ { j } } , \qquad \dot { a } _ { j } = \sum _ { \alpha : \alpha _ { j } \geq 1 } e _ { \alpha } \tilde { c } _ { \alpha } \alpha _ { j } \frac { \prod _ { k } a _ { k } ^ { \alpha _ { k } } } { a _ { j } } b _ { \alpha } .
$$

The sum runs over target multi-indices, since $v _ { \alpha } = 0$ kills the residual elsewhere.

Parity criterion for diagonal A. The of-diagonal $A _ { i j }$ gradient takes the form $\mathbb { E } _ { \pmb { x } } \bigg [ \big ( f - f ^ { * } \big ) \pmb { x } ^ { \alpha ^ { \prime } } \bigg ]$ , with $\mathbf { { \boldsymbol { \alpha } } ^ { \prime } }$ shifted by one in coordinates $i , j .$ Gaussian parity makes this expectation vanish unless every entry of $\bar { \mathbf { \alpha } } \bar { \mathbf { \alpha } } ^ { \prime }$ is even, which in turn forces two target multi-indices at the same order to difer in parity at exactly $i , j$ . So A stays diagonal whenever no two same-order target multi-indices difer in parity at exactly two coordinates—the per-order parity criterion. Staircases $( z _ { 1 } , z _ { 1 } z _ { 2 } , z _ { 1 } z _ { 2 } z _ { 3 } , . . . )$ and single Hermite modes $\mathrm { ( H e } _ { \ell } ( z _ { 1 } ) )$ satisfy this criterion, and we restrict to such targets throughout the paper. Targets that fail include $z _ { 1 } + z _ { 2 }$ and $\mathrm { H e _ { 3 } } ( z _ { 1 } ) + \mathrm { H e _ { 3 } } ( z _ { 2 } )$ , where two same-order multi-indices difer in exactly two coordinates and seed an of-diagonal gradient.

Timescale separation. The order-ℓ Hermite mode’s gradient carries a factor of order $\gamma ^ { \alpha }$ , so higher-order modes relax much more slowly than lower-order ones whenever the data eigenvalues are small. Lower-order Hermite modes therefore settle long before higher-order modes have moved, and of-target B entries don’t backreact on the target dynamics at leading order in the eigenvalues. We can therefore analyze each target multi-index independently—the property the reduced ODE above relies on.

## B.4 Validation of the approximation chain

We validate the two links of the approximation chain on synthetic Gaussian data: the SFN-to-feature-ODE reduction (Figure 5) and the DPR-to-feature-ODE reduction (Figure 6). In both cases, the feature ODE is solved with the same hyperparameters as the trained network. We compare online MSE across four targets of increasing complexity: a linear monomial $f ^ { * } = z _ { 1 }$ , a single Hermite mode $f ^ { * } = \mathrm { H e } _ { 2 } ( z _ { 1 } )$ , a two-term staircase $f ^ { * } = z _ { 1 } + z _ { 1 } z _ { 2 }$ , and a three-term staircase $f ^ { * } = z _ { 1 } + z _ { 1 } z _ { 2 } + z _ { 1 } z _ { 2 } z _ { 3 }$ . See Section A for the experimental setup.

![](images/70e22f21ed2615a5a73507ad7c8dc00ff873f52c99053565a5274caa1d9bde94.jpg)  
Figure 5: SFN tracks the feature ODE across all four targets. Online MSE for an SFN (colored) and the feature ODE (dashed black) over efective optimization time. The ODE recovers both the plateau lengths and the loss drops for each target, including the staircase structure of the two- and three-term staircase targets.

![](images/46a51edd879b2ca9d70ef705348e869a8b58493d33fa13a3a72228f431028d7e.jpg)  
Figure 6: DPR tracks the feature ODE across all four targets. Same as Figure 5 but with DPR replacing the SFN. The agreement is tighter than for the SFN, consistent with DPR’s explicit polynomial features removing the random-feature concentration error.

## C Level coeficients for general activations

## C.1 General formula

For a general activation $\sigma ,$ with data on a sphere $\| A { \boldsymbol { \mathbf { \mathit { x } } } } \| = r ,$ the level coeficients $c _ { \ell }$ of the dot-product kernel can be computed from the Hermite decomposition of the rescaled activation:

$$
c _ { \ell } = \frac { \ell ! } { r ^ { 2 \ell } } \left( \mathbb { E } _ { z \sim \mathcal { N } ( 0 , 1 ) } \Big [ h _ { \ell } \big ( z \big ) \sigma \Big ( \frac { r } { \sqrt { d } } \cdot z \Big ) \Big ] \right) ^ { 2 } ,\tag{9}
$$

where $\begin{array} { r } { h _ { \ell } ( z ) = \frac { 1 } { \sqrt { \ell ! } } \mathrm { H e } _ { \ell } ( z ) } \end{array}$ is the ℓ-th normalized probabilist’s Hermite polynomial. In words, we extract the ℓ-th Hermite coeficient of the function $z \mapsto \sigma ( r z / { \sqrt { d } } )$ , where the prefactor $r / \sqrt { d }$ is the standard deviation of the preactivations.

The activation’s symmetry constrains which orders are present: odd activations (e.g. tanh) have $c _ { 2 \ell } = 0$ ; activations that are linear plus even (e.g. ReLU) have $c _ { \ell } = 0 ~ \mathrm { f o r } ~ \ell = 3 , 5 , 7 , . . . .$ . For the reason, we often use either a shifted activation function $( \mathrm { e . g . \ S i L U } ( x + 0 . 5 )$ , tan $_ 1 ( x + 0 . 5 ) )$ or a cosine activation function $( \sigma ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 ) )$ , all of which are full in their Taylor series.

Approximation. While we are in the near-origin regime (discussed in Section B.2), we find that the level coeficients are found from taking it’s Taylor series expansion:

$$
c _ { \ell } = \left( \sigma ^ { ( \ell ) } ( 0 ) \right) ^ { 2 } ,\tag{10}
$$

the exponent $( \ell )$ in parentheses denoting the ℓ-th derivative. Equation (10) gives the level coeficients used in our validation experiments

## C.2 Cosine activation

Here, we derive the dot-product kernel for the special case of a cosine activation $\sigma ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 )$ and show that it gives $c _ { \ell } = 1$ for all ℓ.

The SFN feature kernel is

$$
\begin{array} { r } { K ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathbb { E } _ { \pmb { w } \sim \mathcal { N } ( 0 , { I } _ { d } / d ) } \left[ \sigma ( \pmb { w } ^ { \top } \pmb { A } \pmb { x } ) \sigma ( \pmb { w } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } ) \right] . } \end{array}
$$

The preactivations $z _ { 1 } = { w } ^ { \top }$ Ax and $z _ { 2 } = { \pmb w } ^ { \top } A { \pmb x } ^ { \prime }$ are jointly Gaussian with covariance

$$
\Sigma ^ { \prime } = \frac { 1 } { d } \left( \vphantom { \frac { 1 } { x ^ { \prime } } ^ { \top } A ^ { \top } A x } \frac { \left. A x \right. ^ { 2 } } { A ^ { \top } A ^ { \top } } \right. \left. \overline { { x ^ { \top } A ^ { \top } A x ^ { \prime } } } \right) .
$$

Write $\begin{array} { r } { s _ { 1 1 } = \frac { 1 } { d } \left\| A \pmb { x } \right\| ^ { 2 } , s _ { 2 2 } = \frac { 1 } { d } \left\| A \pmb { x } ^ { \prime } \right\| ^ { 2 } } \end{array}$ , and $\begin{array} { r } { s _ { 1 2 } = \frac { 1 } { d } \pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { A } \pmb { x } ^ { \prime } } \end{array}$

Expanding the activation:

$$
K = 2 \mathbb { E } [ \cos ( z _ { 1 } + \pi / 4 ) \cos ( z _ { 2 } + \pi / 4 ) ]\tag{11}
$$

$$
= \mathbb { E } [ \cos ( z _ { 1 } - z _ { 2 } ) ] + \mathbb { E } [ \cos ( z _ { 1 } + z _ { 2 } + \pi / 2 ) ] .\tag{12}
$$

For the first term, $z _ { 1 } - z _ { 2 } \sim \mathcal { N } ( 0 , s _ { 1 1 } - 2 s _ { 1 2 } + s _ { 2 2 } )$ . Using the Gaussian identity $\mathbb { E } _ { u \sim \mathcal { N } ( 0 , \tau ^ { 2 } ) } [ \cos ( u ) ] = e ^ { - \tau ^ { 2 } / 2 }$

$$
\begin{array} { r } { \mathbb { E } [ \cos ( z _ { 1 } - z _ { 2 } ) ] = \exp \left( - \frac { 1 } { 2 } ( s _ { 1 1 } - 2 s _ { 1 2 } + s _ { 2 2 } ) \right) . } \end{array}
$$

For the second term, $z _ { 1 } + z _ { 2 } \sim \mathcal { N } ( 0 , s _ { 1 1 } + 2 s _ { 1 2 } + s _ { 2 2 } )$ , and cos $( z _ { 1 } + z _ { 2 } + \pi / 2 ) = - \sin ( z _ { 1 } + z _ { 2 } )$ . Since $\mathbb { E } _ { u \sim \mathcal { N } ( 0 , \tau ^ { 2 } ) } [ \sin ( u ) ] = 0$ by symmetry, this term vanishes. Therefore:

$$
\begin{array} { r } { K ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathrm { e x p } \big ( - \frac 1 2 \big ( s _ { 1 1 } - 2 s _ { 1 2 } + s _ { 2 2 } \big ) \big ) = \mathrm { e x p } \Big ( - \frac 1 { 2 d } \left\| \pmb { A } \pmb { x } - \pmb { A } \pmb { x } ^ { \prime } \right\| ^ { 2 } \Big ) . } \end{array}
$$

This is an RBF kernel in the transformed coordinates ${ \pmb A } { \pmb x } .$ . Factoring:

$$
\begin{array} { r } { K ( { \pmb x } , { \pmb x } ^ { \prime } ) = \underbrace { \exp \left( - \frac { 1 } { 2 } \big ( s _ { 1 1 } + s _ { 2 2 } \big ) \right) } _ { \mathrm { l e n g t h - d e p e n d e n t p r e f a c t o r } } \cdot \underbrace { \exp \left( s _ { 1 2 } \right) } _ { \mathrm { d o t - p r o d u c t ~ t e r m } } . } \end{array}
$$

Under the dot-product kernel approximation (length terms approximately constant, see Section B.2), the kernel is proportional to

$$
\exp ( s _ { 1 2 } ) = \exp \bigl ( { \textstyle { \frac { 1 } { d } } } { \pmb x } ^ { \top } { \pmb A } ^ { \top } { \pmb A } { \pmb x } ^ { \prime } \bigr ) = \sum _ { \ell = 0 } ^ { \infty } { \textstyle { \frac { 1 } { \ell ! } } } \left( { \textstyle { \frac { 1 } { d } } } { \pmb x } ^ { \top } { \pmb A } ^ { \top } { \pmb A } { \pmb x } ^ { \prime } \right) ^ { \ell } .
$$

Comparing with the general form $\begin{array} { r } { k ( t ) = \sum _ { \ell } c _ { \ell } t ^ { \ell } / \ell ! } \end{array}$ , we read of $c _ { \ell } = 1$ for all $\ell .$

## D Feature ODE Rise Time Derivations

This appendix discusses our methodology for arriving at the efective optimization time for MLPs to learn monomial target functions. Beginning with the feature ODE, we have

$$
\mathcal { L } = \frac { 1 } { 2 } \left( 1 - \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } a _ { i } ^ { \alpha _ { i } } b \right) ^ { 2 }
$$

$$
\dot { a } _ { i } = - \frac { \partial \mathcal { L } } { \partial a _ { i } } = \left( 1 - \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } a _ { i } ^ { \alpha _ { i } } b \right) \frac { \alpha _ { i } \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } a _ { i } ^ { \alpha _ { i } - 1 } b } { a _ { i } }
$$

$$
\dot { b } = - \frac { \partial \mathcal { L } } { \partial b } = \left( 1 - \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } a _ { i } ^ { \alpha _ { i } } b \right) \frac { \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } a _ { i } ^ { \alpha _ { i } } b } { b } .
$$

Noting $\begin{array} { r } { \frac { 1 } { \alpha _ { i } } a _ { i } \dot { a } _ { i } = b \dot { b } } \end{array}$ , we find $a _ { i } ( t ) = \sqrt { \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) }$ Plugging this into the feature ODE, we get

$$
\dot { b } = \left( 1 - \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } \big ( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \big ) ^ { \frac { \alpha _ { i } } { 2 } } b \right) \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } \big ( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \big ) ^ { \frac { \alpha _ { i } } { 2 } } .
$$

This equation is not generally solvable, but we can make a number of simplifications to estimate an efective optimization time. In a gradient flow regime from small initialization, the network output tends to follow two distinct regimes: near a saddle-point and growth [Jacot et al., 2021, Boursier et al., 2022a, Pesme and Flammarion, 2023]. We assume that the number of gradient steps to escape from a saddle-point will dominate over the number it takes to reach the next saddle-point, forcing $\begin{array} { r } { \left( 1 - { \tilde { c } _ { \ell } } \prod _ { i = 1 } ^ { d } \left( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) ^ { \frac { \alpha _ { i } } { 2 } } b \right) \approx 1 } \end{array}$ , which simplifies the ODE to

$$
\int _ { 0 } ^ { b _ { f } } \frac { \mathrm { d } b } { \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } \left( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) ^ { \frac { \alpha _ { i } } { 2 } } } = \int _ { 0 } ^ { \tau } \mathrm { d } t ,
$$

where

$$
b _ { f } \equiv \tilde { c } \sum _ { i } ^ { 1 } { \pmb { \alpha } } _ { i } + 1\tag{13}
$$

represents the final value of b and τ represents the learning-rate scaled number of gradient steps. We provide exact results of this integral for the cases of $\pmb { \alpha } _ { i } = 0 \forall i$ and $\pmb { \alpha } _ { i } = 1 , \ \pmb { \alpha } _ { j } = 0 \forall j \neq i$ (a constant and linear target). For higher order integrals, exact results exist, although to high precision we can replace the upper bound $b _ { e q }$ with infinity, remaining up to a $\mathcal { O } \left( c ^ { - } \overline { { \sum _ { i } ^ { 2 } \alpha _ { i } + 1 } } \right)$ constant.

$$
\tau _ { c o n s t } = \int _ { 0 } ^ { \frac { 1 } { \tilde { c } } } \frac { { \mathrm { d } } b } { \tilde { c } } = \frac { 1 } { \tilde { c } ^ { 2 } }\tag{14}
$$

$$
\tau _ { i n } = \int _ { 0 } ^ { \frac { 1 } { \sqrt { \bar { c } } } } \frac { \mathrm { d } b } { \bar { c } \sqrt { b ^ { 2 } + a _ { i } ^ { 2 } ( 0 ) } } = \frac { 1 } { \tilde { c } } \mathrm { l n } \left( \frac { \frac { 1 } { \sqrt { \bar { c } } } + \sqrt { \frac { 1 } { \bar { c } } + a _ { i } ^ { 2 } ( 0 ) } } { a _ { i } ( 0 ) } \right) \approx \frac { 1 } { \tilde { c } } \mathrm { l n } \left( \frac { 2 } { \sqrt { \bar { c } } a _ { i } ( 0 ) } \right)\tag{15}
$$

$$
\begin{array} { l } { \displaystyle \tau _ { q u a d + } = \int _ { 0 } ^ { \tilde { c } } \overline { { \left( \sum _ { i } ^ { \perp } \alpha _ { i } + 1 \right) } } \frac { \mathrm { d } b } { \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } \left( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) ^ { \frac { \alpha _ { i } } 2 } } } \\ { \displaystyle = \int _ { 0 } ^ { \infty } \frac { \mathrm { d } b } { \tilde { c } _ { \ell } \prod _ { i = 1 } ^ { d } \left( \alpha _ { i } b ^ { 2 } ( t ) + a _ { i } ^ { 2 } ( 0 ) \right) ^ { \frac { \alpha _ { i } } 2 } } - \mathcal { O } \left( { \tilde { c } } ^ { - \sum _ { i } ^ { \frac { 2 } { \alpha _ { i } + 1 } } } \right) } \end{array}\tag{16}
$$

(17)

Below, we provide a table of the rise times found with the ∞ upper bound. For shorthand, any $\alpha _ { i }$ that is unspecified takes a value of $^ { \dag } 0 ,$ and $\alpha _ { i , j , k , \cdots } = a$ denotes that $\alpha _ { i } = \alpha _ { j } = \alpha _ { k } = \cdot \cdot \cdot = a$ . Some integrals (ie $\alpha _ { i } = 3 , \alpha _ { j } = 1 )$ are incomplete elliptics—in such cases, we provide a numerical estimate. We drop the subscript of c˜ as it should be clear from the context.

• Quadratics $\left\{ \begin{array} { l l } { { \alpha _ { i } = 2 } } & { { \tau = \frac { a _ { 0 } ^ { - 1 } } { \tilde { c } } \frac { \pi } { 2 ^ { 3 / 2 } } } } \\ { { \alpha _ { i , j } = 1 } } & { { \tau = \frac { a _ { 0 } ^ { - 1 } } { \tilde { c } } \frac { \pi } { 2 } } } \end{array} \right.$

• Cubics

$$
\left\{ \begin{array} { l l } { \alpha _ { i } = 3 } & { \tau = \frac { a _ { 0 } ^ { - 2 } } { \tilde { c } } \frac { 1 } { \sqrt { 3 } } } \\ { \alpha _ { i } = 2 , \alpha _ { j } = 1 } & { \tau = \frac { a _ { 0 } ^ { - 2 } } { \tilde { c } } \frac { \pi } { 4 } } \\ { \alpha _ { i , j , k } = 1 } & { \tau = \frac { a _ { 0 } ^ { - 2 } } { \tilde { c } } ( 1 ) } \end{array} \right.
$$

$$
\begin{array} { r l } { \left\{ \alpha _ { i } = 4 \right. } & { { } \ \tau = \frac { a _ { 0 } ^ { - 3 } } { \tilde { c } _ { s } } \frac { \pi } { 8 } } \\ { \alpha _ { i } = 3 , \alpha _ { j } = 1 } & { { } \ \tau = \frac { a _ { 0 } ^ { - 3 } } { \tilde { c } _ { s } } ( 0 . 4 6 2 3 ) } \end{array}
$$

$$
\left. \begin{array} { r l } { \mathbf { \cdot } \ \mathbf { Q u a r t i c s } \ \middle \{ \alpha _ { i , j } = 2 } & { \tau = \frac { a _ { 0 } ^ { - 3 } } { \tilde { c } } \frac { \pi \sqrt { 2 } } { 8 } } \\ { \alpha _ { i } = 2 , \alpha _ { j , k } = 1 } & { \tau = \frac { a _ { 0 } ^ { - 3 } } { \tilde { c } } \frac { \pi } { 2 } ( \sqrt { 2 } - 1 ) } \\ { \alpha _ { i , j , k , l } = 1 } & { \tau = \frac { a _ { 0 } ^ { - 3 } } { \tilde { c } } \frac { \pi } { 4 } } \end{array} \right.
$$

$$
\begin{array} { r l } { \mathbb { E } \operatorname* { m i n } _ { t  0 } ( { { { \frac { 1 } { 2 } } } _ { 1 / 2 } } - \frac { 5 } { 4 } ) } & { { } \Gamma = \frac { 3 \pi \hat { \nu } _ { 1 } ^ { 2 } } { \nu _ { 1 } ^ { 2 } } \frac { \mathcal { E } } { \nu _ { 2 } ^ { 3 } } } \\ { \approx \frac { 1 } { 8 \pi \hat { \nu } _ { 1 } ^ { 4 } - 3 \sigma _ { 1 } ^ { 2 } - 1 } } & { { } \Gamma = \frac { 3 \pi \hat { \nu } _ { 1 } ^ { 4 } ( \sqrt { \frac { 3 } { 2 } } - \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } + \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } ) } { \nu _ { 1 } ^ { 2 } } } \\ { \approx \frac { 3 \pi \hat { \nu } _ { 1 } ^ { 4 } - 3 \sigma _ { 1 } - 2 } { \nu _ { 2 } ^ { 2 } - 1 } } & { { } \Gamma = \frac { 3 \pi \hat { \nu } _ { 1 } ^ { 5 } ( \sqrt { \frac { 3 } { 2 } } - \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } + \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } ) } { \nu _ { 1 } ^ { 2 } } } \\ { \times \frac { 1 } { 8 \pi \hat { \nu } _ { 1 } ^ { 2 } - 2 \sigma _ { 1 } \nu _ { 1 } - 1 } } & { { } \Gamma = \frac { \sqrt { 3 } \pi \hat { \nu } _ { 1 } ^ { 4 } ( \sqrt { \frac { 3 } { 2 } } - \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } + \frac { 1 } { 2 } \sqrt { \frac { 3 } { 2 } } ) } { \nu _ { 1 } ^ { 2 } } } \\ { \times \frac { 1 } { 8 \pi \hat { \nu } _ { 1 } ^ { 2 } - 2 \sigma _ { 1 } \nu _ { 1 } - 1 } } &  { } \Gamma = \frac  3 \pi \hat   \end{array}
$$

## E Neuron specialization and target functions

We diagnose neuron specialization with a three-panel comparison: training loss, per-direction $A _ { i i } \ \mathrm { o r } \ \sqrt { \mathbf { N F M } _ { i i } }$ trajectories, and a heatmap of per-neuron preactivation variance (all $N = 8 0 0 0$ neurons ranked by variance, plotted against efective optimization time). The heatmap is the workhorse. Neurons all contribute to the dynamics when there are consistent color changes across neurons at a given time step and there are smooth color gradients. Neuron subpopulations drive the dynamics when groups of neurons change color independently of the rest. All experiments in this appendix use a biasless network with the cosine activation $\sigma ( z ) = \sqrt { 2 } \cos ( z + \pi / 4 )$

DPR has no neurons — its $B ^ { ( \ell ) }$ tensors act on all features at once — so specialization cannot arise. The frozen weights in the SFN make neuron specialization dificult and we see more little specialized behavior in Figures 7 to 9. The feature ODE in turn predicts the loss and per-direction $A _ { i i }$ trajectories well.

MLP neurons can specialize. On tasks like modular arithmetic, for instance, small subpopulations align with Fourier frequencies of the input [Kunin et al., 2025, Nanda et al., 2023]. On the simpler targets we study, whether the MLP specializes depends on the target itself. On a linear target $f ^ { * } ( z ) = z _ { 1 }$ (Figure 7), neither the MLP nor the SFN specializes much: a linear target produces a per-neuron gradient with no direction-dependent amplification, so all neurons evolve symmetrically. Their heatmaps are rather uniform, and the feature ODE matches loss and $\sqrt { \mathrm { N F M } _ { i i } }$ throughout training. On a pure-Hermite target $f ^ { * } ( z ) = \mathrm { H e } _ { 2 } ( z _ { 1 } ) + \mathrm { H e } _ { 2 } ( z _ { 2 } )$ (Figure 8), the MLP specializes sharply: roughly 500 neurons at the top of the ranking drive their preactivation variance orders of magnitude above the bulk. We think this specialized subpopulation drives the MLP loss down earlier than the feature ODE predicts and pins $\sqrt { \mathrm { N F M } _ { i i } }$ below the ODE-predicted $A _ { i i }$ . On a two-term staircase $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 } { \mathrm { ~ ( F i g u r e ~ } } 9 )$ , the MLP specializes only mildly — faint bands rather than a dominant subpopulation. The feature ODE captures the staircase structure of the loss and features, with only a small late-time drift in $\sqrt { \mathrm { N F M _ { 2 , 2 } } }$

We leave a more detailed study hyperparameter and target function’s impact on neuron specialization to future work.

![](images/12578a0a0057bfeda6ad5aac6622d29726ea8f50e195bf866cca98ef37c853fc.jpg)

![](images/d6f9dfc96287994b1e9c41e42678f4ffe266705e68312c75a6e2bd5d67b33e21.jpg)

![](images/d86c0e3a69c478d7e08f11ef65c3fe28514f2268d43a9c74957381ee2db7852c.jpg)

![](images/52ba9b9fd06a7867e1c78ea4bd00340a6d7f49d55a43aa058f71412a3a4f15cb.jpg)  
Figure 7: Linear target $f ^ { * } ( z ) = z _ { 1 }$ . Top: MLP. Bottom: SFN. Neurons are sorted by preactivation variance at the end of training in the heatmaps. Loss, $\sqrt { { \mathrm { N F M } _ { i i } } } \left( { \mathrm { M L P } } \right)$ or $A _ { i i }$ (SFN), and per-neuron preactivation variance. Both models track the feature ODE, and every neuron gains variance on the same timescale.

![](images/9e585dec9e38f8460a8076f7d0be7ca56064c66f14020e188ca8ad9673a1a6be.jpg)

![](images/368a33cf1491cb86c0e3be088f573ee81998187dc960c10ad4235006bfdddd05.jpg)

![](images/cf7de2ace791c64c491d46eb4750e3fa9fcf3af8f408deeb05d4783023b4f86b.jpg)

![](images/6b24302a3b25317351192582556c617d6e94cc144fc182c71731b64de12a15cb.jpg)  
Figure 8: Pure-Hermite target $f ^ { * } ( z ) = \mathrm { H e } _ { 2 } ( z _ { 1 } ) + \mathrm { H e } _ { 2 } ( z _ { 2 } )$ . Top: MLP. Bottom: SFN. Neurons are sorted by preactivation variance at the end of training in the heatmaps. A specialized subpopulation of roughly 500 MLP neurons (top, right column) gains preactivation variance orders of magnitude above the bulk, drives the loss down earlier than the feature ODE predicts, and pins $\sqrt { \mathrm { N F M } _ { i i } }$ below the ODE-predicted $A _ { i i }$ . The SFN carries no such subpopulation and tracks the ODE.

![](images/90a10742a63d016304fd41107c51cd931bd4fb9fe4dfe8d83e5835379faabd44.jpg)

![](images/752dc5e05c10f84ae4a846a8e1d7cf57227d93a3773883dcaf724c115e00a11a.jpg)

![](images/de108c7efdec2f9d3caa354bea980e57bb9dbca06319ae67eb1dc50787e8fafd.jpg)

![](images/5f0f3683fc27a4a8ae26c72eaa21016c5c50ae4254bec67ca1a89ae798ac5fdb.jpg)  
Figure 9: Two-term staircase $f ^ { * } ( z ) = z _ { 1 } + z _ { 1 } z _ { 2 }$ . Top: MLP. Bottom: SFN. Neurons are sorted by preactivation variance at the end of training in the heatmaps. A faint specialization band in the MLP heatmap accompanies a small late-time drift of $\sqrt { \mathrm { N F M _ { 2 , 2 } } }$ above the ODE prediction. The staircase loss is otherwise well captured. The SFN matches the ODE throughout.

## F DPR population gradient

This appendix derives the DPR population gradient in closed form, using Isserlis’ theorem (Wick contractions) to evaluate the Gaussian expectations. The main text (Section 3) states the resulting gradient flow for some specific target functions. Here, we provide the gradient of DPR for any polynomial target function.

Let $\pmb { a } _ { m } \in \mathbb { R } ^ { d }$ denote the m-th row of A written as a column vector, so that $( A x ) _ { m } = \pmb { a } _ { m } ^ { \top } \pmb { x } .$ , and let $\boldsymbol { e } _ { \nu } \in \mathbb { R } ^ { d }$ denote the ν-th standard basis vector. Working in the normalized coordinates $z = \mathbf { { \Gamma } } ^ { - 1 / 2 } x$ (so that $\boldsymbol { z } \sim \mathcal { N } ( 0 , \boldsymbol { I } _ { d } )$ when $\pmb { x } \sim \mathcal { N } ( 0 , \mathbf { T } ) )$ , we expand the target polynomially as $\begin{array} { r } { f ^ { \star } ( z ) = \sum _ { j = 0 } ^ { p } F ^ { ( j ) } z ^ { \otimes j } } \end{array}$ with $\pmb { F } ^ { ( j ) }$ a symmetric order-j tensor. The main text (Section 3) writes the target in Hermite basis, $\begin{array} { r } { f ^ { \star } ( z ) = \sum _ { \alpha } v _ { \alpha } h _ { \alpha } ( z ) } \end{array}$ , with coeficients $v _ { \alpha }$ indexed by multi-index. The monomial form used here is equivalent: standard monomial-to-Hermite expansion identifies $\pmb { F } ^ { ( j ) }$ with a linear combination of $v _ { \alpha } \mathrm { a t } \left| \pmb { \alpha } \right| \leq j$ . We use the monomial representation because Isserlis’ theorem evaluates Gaussian moments of monomial products directly. All other symbols $( A , B ^ { ( \ell ) } , a _ { m } , \Gamma , \gamma _ { i } , c _ { \ell } )$ follow the main-text conventions.

## F.1 Derivatives of the DPR loss

We take the DPR model

$$
f _ { \mathrm { D P R } } ( \pmb { x } ) = \sum _ { \ell = 0 } ^ { p } \sqrt { \frac { c _ { \ell } } { \ell ! } } B ^ { ( \ell ) } ( \pmb { A } \pmb { x } ) ^ { \otimes \ell } ,\tag{18}
$$

and expand its squared loss

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { \pmb { x } \sim \mathcal { N } ( 0 , \mathbf { T } ) } \Big [ \big ( f _ { \mathrm { D P R } } ( \pmb { x } ) - f ^ { \star } ( z ) \big ) ^ { 2 } \Big ] = \mathbb { E } \left[ f _ { \mathrm { D P R } } ^ { 2 } \right] - 2 \mathbb { E } \left[ f _ { \mathrm { D P R } } f ^ { \star } \right] + \mathbb { E } \left[ \left( f ^ { \star } \right) ^ { 2 } \right] .
$$

The last term is parameter-free and drops out of the gradient. The first two split into a model-model piece (which is quadratic in $B )$ and a model-target cross piece (which is linear in $B )$ ; we compute the parameter derivatives of $f _ { \mathrm { D P R } }$ before taking expectations.

B gradient. At entry $( k _ { 1 } , \ldots , k _ { \ell } )$ of $B ^ { ( \ell ) }$ , the derivative of $f _ { \mathrm { D P R } } ( \pmb { x } )$ is the matching monomial in Ax:

$$
\frac { \partial f _ { \mathrm { D P R } } ( \pmb x ) } { \partial \pmb { B } _ { k _ { 1 } , \ldots , k _ { \ell } } ^ { ( \ell ) } } = \sqrt { \frac { c _ { \ell } } { \ell ! } } \left( \pmb { A } \pmb { x } \right) _ { k _ { 1 } } \cdot \cdot \cdot \left( \pmb { A } \pmb { x } \right) _ { k _ { \ell } } .\tag{19}
$$

For the constant term $( \ell = 0 ) , B ^ { ( 0 ) }$ is a scalar and $\partial _ { B ^ { ( 0 ) } } f _ { \mathrm { D P R } } = \sqrt { c _ { 0 } }$

A gradient. An increment in $\scriptstyle A _ { \mu \nu }$ shifts $( \pmb { A } \pmb { x } ) _ { \mu }$ by $x _ { \nu } ,$ , so each order-ℓ term contributes ℓ identical copies, one per tensor slot:

$$
{ \frac { \partial f _ { \mathrm { D P R } } ( { \pmb x } ) } { \partial { \pmb A } _ { \mu \nu } } } = x _ { \nu } \sum _ { \ell = 1 } ^ { p } \ell \sqrt { \frac { c _ { \ell } } { \ell ! } } { \pmb B } _ { \mu , b _ { 2 } , \dots , b _ { \ell } } ^ { ( \ell ) } \left( { \pmb A } { \pmb x } \right) _ { b _ { 2 } } \cdot \cdot \cdot \cdot \left( { \pmb A } { \pmb x } \right) _ { b _ { \ell } } ,\tag{20}
$$

where repeated b-indices are summed. The constant term $( \ell = 0 )$ does not depend on A and contributes nothing.

## F.2 Expectations via Isserlis’ theorem

Each piece of the loss gradient is the expectation of a polynomial in the jointly Gaussian vectors x $\sim \mathcal { N } ( 0 , \mathbf { \Gamma } \mathbf { T } )$ and $z = \bar { \mathbf { T } } ^ { - 1 / 2 } x$ . Isserlis’ theorem states that for jointly Gaussian zero-mean random variables $X _ { 1 } , \ldots , X _ { 2 n }$

$$
\mathbb { E } \bigl [ X _ { 1 } X _ { 2 } \cdot \cdot \cdot X _ { 2 n } \bigr ] = \sum _ { \mathrm { p a i r i n g s } } \prod _ { \pi ( m , n ) \in \pi } \mathbb { E } \bigl [ X _ { m } X _ { n } \bigr ] ,\tag{21}
$$

where the sum runs over the $( 2 n - 1 ) ! !$ partitions of $\left\{ 1 , \ldots , 2 n \right\}$ into unordered pairs; odd-order moments vanish. Three types of slots appear in our expressions — transformed slots $( A x ) _ { k }$ , raw slots $x _ { \nu }$ (from the A derivative), and target slots $z _ { f }$ . Assigning each slot a vector $\pmb { u } \in \mathbb { R } ^ { d }$

$$
{ \bf { u } } _ { k } = { \bf { a } } _ { k } ( \mathrm { t r a n s f o r m e d } ) ,
$$

$$
{ \pmb u } _ { \nu } = e _ { \nu } ~ \mathrm { ( r a w ) } , \qquad { \pmb u } _ { f } = \Gamma ^ { - 1 / 2 } e _ { f } ~ ( z \mathrm { - t a r g e t } ) ,\tag{22}
$$

every pairwise covariance takes the unified form $\mathrm { C o v } [ \cdot , \cdot ] =  { \boldsymbol { u } } _ { m } ^ { \top } \boldsymbol { \Gamma }  { \boldsymbol { u } } _ { n }$ . This evaluates to $\pmb { a } _ { m } ^ { \top } \pmb { \Gamma } \pmb { a } _ { n }$ between two transformed slots, $\sqrt { \gamma _ { f } } ( \pmb { a } _ { m } ) _ { f }$ between a transformed slot and a z-slot, $\delta _ { f f ^ { \prime } }$ between two z-slots, and analogously for the raw slot.

B expectations. For the model-model piece, the order-ℓ contribution at index $( k _ { 1 } , \ldots , k _ { \ell } )$ is

$$
\begin{array} { r l r } {  { 2 \sqrt { \frac { c _ { \ell } } { \ell ! } } \mathbb { E } [ ( A { \pmb x } ) _ { k _ { 1 } } \cdot \cdot \cdot ( { \pmb A } { \pmb x } ) _ { k _ { \ell } } f _ { \mathrm { D P R } } ( { \pmb x } ) ] } } \\ & { } & { = 2 \sqrt { \frac { c _ { \ell } } { \ell ! } } \sum _ { i = 0 } ^ { p } \sqrt { \frac { c _ { i } } { i ! } } \pmb { B } _ { b _ { 1 } , \dots , b _ { i } } ^ { ( i ) } \sum _ { k \mathrm { - } b \mathrm { p a i r i n g s } } \prod _ { ( m , n ) } \pmb { a } _ { m } ^ { \top } \mathbf { \Gamma } \mathbf { r } \pmb { a } _ { n } , } \end{array}\tag{23}
$$

where $\cdots k { - } b$ pairings” enumerates all unordered pair partitions of the $\ell + i$ indices $\{ k _ { 1 } , \dotsc , k _ { \ell } , b _ { 1 } , \dotsc , b _ { i } \}$ . The pairing sum vanishes whenever $\ell + i$ is odd. The model-target cross piece is analogous,

$$
- 2 \sqrt { \frac { c _ { \ell } } { \ell ! } } \sum _ { j = 0 } ^ { p } \pmb { F } _ { f _ { 1 } , \dots , f _ { j } } ^ { ( j ) } \sum _ { k \not = f \mathrm { p a i r i n g s } } \prod _ { ( m , n ) } \pmb { u } _ { m } ^ { \top } \pmb { \Gamma } \pmb { u } _ { n } ,\tag{24}
$$

with ${ \pmb u } _ { m } = { \pmb a } _ { m }$ on k-slots and ${ \pmb u } _ { m } = { \bf \cal T } ^ { - 1 / 2 } e _ { m }$ on f-slots (z-coordinates of the target).

A expectations. The same Wick reduction applies, except one slot is now “frozen” to the raw coordinate $x _ { \nu } = e _ { \nu } ^ { \top }$ x from the A derivative:

$$
\begin{array} { r l } & { \mathbb { E } \big [ \nabla _ { A } \mathcal { L } \big ] _ { \mu , \nu } = 2 \displaystyle \sum _ { i = 1 } ^ { p } i \sqrt { \frac { c _ { i } } { i ! } } \ \pmb { B } _ { \mu , b _ { 2 } , \dots , b _ { i } } ^ { ( i ) } [ \displaystyle \sum _ { j = 1 } ^ { p } \sqrt { \frac { c _ { j } } { j ! } } \ B _ { c _ { 1 } , \dots , c _ { j } } ^ { ( j ) } \displaystyle \sum _ { \nu = b - c \mathrm { p a r i n g s } } \prod _ { m , n }   u _ { m } ^ { \top } \Gamma \pmb { u } _ { n }  } \\ & { \qquad - \displaystyle \sum _ { j = 1 } ^ { p } { \cal F } _ { f _ { 1 } , \dots , f _ { j } } ^ { ( j ) } \displaystyle \sum _ { \nu \mathrm { - } b \mathrm { - } f \mathrm { \ p a i r i n g s } } \prod _ { m , n } ] _ { m } \mathbf { u } _ { m } ^ { \top } \mathbf { T } \pmb { u } _ { n } ] , } \end{array}\tag{25}
$$

where u takes value $e _ { \nu }$ on the ν-slot, $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \bf \alpha } \mathbf { \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm \mathrm { \bf \alpha } \mathrm { \bf \alpha } \mathrm \mathrm { \bf \alpha } \mathrm \mathrm { \bf \alpha } \mathrm \mathrm { \bf \alpha } \mathrm \mathrm { \alpha \mathrm \alpha } \mathrm \mathrm \mathrm { \alpha \mathrm } \mathrm \mathrm \mathrm { \alpha \alpha } \mathrm \mathrm \mathrm \mathrm { \alpha \mathrm \alpha } \mathrm \mathrm \mathrm \mathrm { \alpha \mathrm \mathrm \alpha } \mathrm \mathrm \mathrm { \alpha \mathrm \mathrm \mathrm \alpha \mathrm \mathrm } \mathrm \mathrm \mathrm  \alpha \mathrm \mathrm \mathrm \alpha \mathrm \mathrm \mathrm \mathrm { \alpha \alpha \alpha } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm  \alpha \alpha \alpha \mathrm \mathrm \mathrm \mathrm \alpha \mathrm \mathrm \mathrm \mathrm \alpha \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \alpha \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \alpha \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm$ on b- and c-slots, and $\mathbf { \Gamma } \mathbf { \Gamma } ^ { - 1 / 2 } e _ { f }$ on f-slots.

## F.3 Full population gradient

The pieces combine into two equivalent expressions for the full population gradient.

Pairing form. Collecting the model-model and model-target reductions,

$$
\begin{array} { r } { \nabla _ { B _ { k _ { 1 } , \ldots , k _ { \ell } } ^ { ( \ell ) } } \mathcal { L } = 2 \sqrt { \frac { c _ { \ell } } { \ell ! } } \displaystyle \sum _ { i = 0 } ^ { p } \sqrt { \frac { c _ { i } } { i ! } } B _ { b _ { 1 } , \ldots , b _ { i } } ^ { ( i ) } \sum _ { k \mathrm { ~ b a i r i n g s } } \prod _ { ( m , n ) } a _ { m } ^ { \top } \Gamma a _ { n } } \\ { - 2 \sqrt { \frac { c _ { \ell } } { \ell ! } } \displaystyle \sum _ { j = 0 } ^ { p } F _ { f _ { 1 } , \ldots , f _ { j } } ^ { ( j ) } \sum _ { k - f \mathrm { ~ p a i r i n g s } } \prod _ { ( m , n ) } u _ { m } ^ { \top } \Gamma u _ { n } , } \end{array}\tag{26}
$$

and the A gradient is given by the expression above.

Moment-tensor form. Define $S = A \Gamma A ^ { \top }$ and two Gaussian moment tensors:

$$
M _ { k _ { 1 } , \ldots , k _ { \ell } } ^ { ( \ell ) } = \mathbb { E } [ ( A x ) _ { k _ { 1 } } \cdot \cdot \cdot ( A x ) _ { k _ { \ell } } ] = \sum _ { \mathrm { p a i r i n g s } } \prod _ { ( m , n ) } S _ { m , n } ,\tag{27}
$$

$$
M _ { s _ { 1 } , \ldots , s _ { r } ; q _ { 1 } , \ldots , q _ { t } } ^ { ( r ; t ) } = \mathbb { E } [ z _ { s _ { 1 } } \cdot \cdot \cdot z _ { s _ { r } } \left( A x \right) _ { q _ { 1 } } \cdot \cdot \cdot \left( A x \right) _ { q _ { t } } ] ,\tag{28}
$$

where $M ^ { ( r ; t ) }$ contracts r z-target slots against t transformed slots through Gaussian pairings. Using $x _ { \nu } = \sqrt { \gamma _ { \nu } } z _ { \nu }$ the raw-slot factor in the A derivative pulls a $\sqrt { \gamma _ { \nu } }$ outside the moment, leaving pure z-and-transformed moments. The gradient then takes the compact contraction form

$$
\begin{array} { r } { \nabla _ { B _ { k _ { 1 } , \ldots , k _ { \ell } } ^ { ( \ell ) } } \mathcal { L } = 2 \sqrt { \frac { \underline { { c } } _ { \ell } } { \ell ! } } \bigg [ \displaystyle \sum _ { i = 0 } ^ { p } \sqrt { \frac { c _ { i } } { i ! } } B _ { b _ { 1 } , \ldots , b _ { i } } ^ { ( i ) } M _ { k _ { 1 } , \ldots , k _ { \ell } , b _ { 1 } , \ldots , b _ { i } } ^ { ( \ell + i ) } } \\ { - \displaystyle \sum _ { j = 0 } ^ { p } F _ { f _ { 1 } , \ldots , f _ { j } } ^ { ( j ) } M _ { f _ { 1 } , \ldots , f _ { j } ; k _ { 1 } , \ldots , k _ { \ell } } ^ { ( j ; \ell ) } \bigg ] , } \end{array}\tag{29}
$$

$$
\begin{array} { c } { { \nabla _ { A _ { \mu , \nu } } \mathcal { L } = 2 \sqrt { \gamma _ { \nu } } \displaystyle \sum _ { i = 1 } ^ { p } i \sqrt { \frac { c _ { i } } { i ! } } B _ { \mu , b _ { 2 } , \dots , b _ { i } } ^ { ( i ) } \left[ \displaystyle \sum _ { j = 1 } ^ { p } \sqrt { \frac { c _ { j } } { j ! } } B _ { c _ { 1 } , \dots , c _ { j } } ^ { ( j ) } M _ { \nu ; b _ { 2 } , \dots , b _ { i } , c _ { 1 } , \dots , c _ { j } } ^ { ( 1 ; i + j - 1 ) } \right. } } \\ { { \displaystyle \left. - \sum _ { j = 1 } ^ { p } F _ { f _ { 1 } , \dots , f _ { j } } ^ { ( j ) } M _ { \nu , f _ { 1 } , \dots , f _ { j } ; b _ { 2 } , \dots , b _ { i } } ^ { ( j + 1 ; i - 1 ) } \right] . } } \end{array}\tag{30}
$$

The pairing form is more explicit and easier to truncate to a single order; the moment-tensor form makes the contraction structure transparent. Both feed directly into the diagonal-A and timescale-separation reductions used in Section B.3