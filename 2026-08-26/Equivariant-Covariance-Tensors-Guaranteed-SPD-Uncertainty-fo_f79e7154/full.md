# Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning

Ruihan Liu <sup>1</sup> Yu Ji <sup>1</sup> Jianbo Yu <sup>2</sup> Shifu Yan <sup>3</sup> Qingchao Jiang <sup>4</sup>

## Abstract

Tensor-valued prediction is fundamental to geometric deep learning, yet uncertainty quantification (UQ) for such outputs remains an open challenge. While E(3)-equivariant neural networks excel at point estimates, they lack rigorous confidence measures. We focus on symmetric rank-2 tensor prediction, where the target has six Kelvin– Mandel coordinates and full uncertainty is represented by a 6×6 covariance matrix. We introduce a framework for E(3)-equivariant UQ, modeling the full predictive distribution where both mean and covariance preserve rotational symmetry. Our approach decomposes the covariance into irreducible representations $\mathrm { S y m } ^ { 2 } ( \rho _ { c } ) \cong 2 \times ( l =$ $0 ) \oplus 2 \times ( l = 2 ) \oplus 1 \times ( l = 4 )$ . By mapping from the flat Lie algebra sym(6) to the curved SPD manifold via matrix exponentiation, we strictly ensure positive-definite covariances while maintaining exact equivariance. Furthermore, we formulate a Log-Euclidean Equivariant Scoring Objective (LE-ESO)—a robust surrogate loss based on the Multivariate Laplace distribution—providing robustness to heavy-tailed errors and stable optimization. Validation on ModelNet40 inertia tensors and Materials Project dielectric tensors demonstrates that our method achieves competitive performance and provides physically consistent, symmetry-preserving uncertainty estimates with useful risk and OOD sensitivity.

## 1. Introduction

Tensor-valued predictions are fundamental to scientific computing and geometric deep learning, with applications spanning material properties (elasticity tensors, dielectric response), biomedical imaging (diffusion MRI), and computational fluid dynamics. While E(3)-equivariant neural networks (ENNs) have achieved remarkable success in predicting tensorial properties like dielectric constants (Batatia et al., 2022; Heilman et al., 2024), they are inherently deterministic—outputting single point estimates without any confidence measure. This is a critical limitation: overconfident tensor predictions in scientific decisions can lead to costly experimental failures.

In this paper, the primary output is a symmetric rank-2 tensor $\bar { C } \in \mathbb { R } _ { \mathrm { { s y m } } } ^ { 3 \times 3 }$ , such as a dielectric tensor. Although C is a $3 \times 3$ matrix, symmetry leaves six independent degrees of freedom. We represent it by the Kelvin-Mandel vector $\mathbf { c \in }$ $\mathbb { R } ^ { 6 }$ , for which rotations act by an orthogonal six-dimensional representation $\rho _ { c } ( R )$ . Therefore, uncertainty over C is not a $3 \times 3$ covariance, but a $6 \times 6$ covariance over the Kelvin-Mandel coordinates.

Extending ENNs with uncertainty quantification poses a fundamental challenge: uncertainty estimates must themselves transform equivariantly. Mathematically, the covariance $\Sigma \in \mathbb { R } ^ { 6 \times 6 }$ must satisfy:

$$
\Sigma ( R \cdot X ) = \rho _ { c } ( R ) \Sigma ( X ) \rho _ { c } ( R ) ^ { \top } , \quad \forall R \in O ( 3 ) ,
$$

where $\rho _ { c } ( R )$ is the rotation matrix in Kelvin-Mandel notation. This creates a parameterization challenge: Choleskytype covariance heads guarantee SPD but are not equivariant in Kelvin–Mandel covariance coordinates, while direct equivariant regression preserves the transformation law but does not guarantee SPD.

In this work, we introduce a principled framework for E(3)-equivariant full-covariance uncertainty quantification in symmetric tensor-valued prediction. Our contributions address this equivariant SPD parameterization challenge through: (1) an equivariant matrix-exponential head parameterizing the SPD manifold via irreducible decomposition ${ \mathrm { S y m } } ^ { 2 } ( \bar { \rho _ { c } } ) \cong 2 \times ( \ell = 0 ) \oplus 2 \times ( \ell = 2 ) \oplus 1 \times ( \ell = 4 )$ (2) a stable joint optimization strategy via a Log-Euclidean scoring objective that integrates uncertainty calibration with geometric feature extraction; and (3) rigorous validation on ModelNet40 inertia tensors with competitive performance on Materials Project dielectric prediction (MAE 1.55).

## 2. Related Work

Equivariant Tensor Prediction. E(3)-equivariant neural networks (ENNs) achieve state-of-the-art tensor prediction across materials science and 3D geometry (Batatia et al., 2022; Heilman et al., 2024; Hua et al., 2026; Pakornchote et al., 2023; Fung et al., 2021; Reiser et al., 2022; Du et al., 2024; Equer et al., 2023). However, all existing ENNs are inherently deterministic—they cannot quantify confidence in their predictions. Our framework preserves equivariance guarantees while adding rigorous uncertainty quantification, addressing this critical gap. While our approach utilizes the spherical harmonic basis provided by e3nn (Geiger & Smidt, 2022), recent advances in representation theory, such as the High-Rank Irreducible Cartesian Tensor (ICT) decomposition framework (Shao et al., 2025), offer alternative analytical paths for constructing equivariant bases directly in Cartesian space. Such methods could potentially simplify the implementation for higher-rank tensor properties.

SPD Constraints in Neural Networks. Ensuring symmetric positive-definite (SPD) covariances is essential for valid uncertainty. Prior work uses Cholesky decomposition, eigendecomposition, or Riemannian optimization (Jekel et al., 2022; Pouliquen et al., 2025; Zhao et al., 2023). These methods guarantee SPD in a fixed coordinate parameterization, but they do not by themselves provide an equivariant map $X \mapsto \Sigma ( X )$ under the covariance representation $\rho _ { c }$ . Orthogonal conjugation preserves SPD; thus the difficulty is not a conflict between the SPD property and rotation itself. Rather, the challenge is to design a neural parameterization that is simultaneously equivariant and constrained to the SPD cone. Cholesky-type heads enforce SPD but are not equivariant in Kelvin-Mandel covariance coordinates, whereas direct equivariant regression can preserve the transformation law but does not guarantee SPD. Our matrix-exponential head resolves this parameterization problem by predicting an equivariant symmetric operator $A ( X )$ and setting $\Sigma ( X ) = \exp ( A ( X ) )$ , which satisfies $\begin{array} { r } { \exp ( \rho _ { c } ( R ) A \rho _ { c } ( R ) ^ { \top } ) = \rho _ { c } ( R ) \exp ( A ) \rho _ { c } ( R ) ^ { \top } } \end{array}$ and therefore is jointly SPD and equivariant in all rotated frames.

Probabilistic Methods and Equivariant GPs. Probabilistic extensions for ENNs are severely underdeveloped. While ensemble methods can provide heuristic uncertainty estimates (Rudner et al., 2022) and even exhibit emergent equivariance (Gerken & Kessel, 2024), our framework explicitly learns the full 21-parameter aleatoric covariance tensor, which is essential for modeling the inherent anisotropic noise in physical properties. Bayesian neural networks face computational challenges at scale (Rensmeyer et al., 2024; Olivier et al., 2021; Doan et al., 2025; Sheinkman & Wade, 2025). Existing equivariant Bayesian approaches focus on scalar or vector quantities (Zhou et al., 2024), missing tensorvalued uncertainty. E(3)-equivariant Gaussian processes provide theoretically sound uncertainties but are typically restricted to isotropic or diagonal covariance approximations (Steinert et al., 2025; Bevanda et al., 2025)—scaling them to the full 21-parameter covariance structure of rank-2 tensors remains computationally prohibitive. Our neural network approach offers computational scalability while learning complete equivariant tensor correlations. Recent work on uncertainty calibration and stable tensor operations provides theoretical grounding for our design (Berman et al., 2026; Gruber & Buettner, 2022; Fakour et al., 2024; Newman et al., 2024).

## 3. Methods

## 3.1. Problem Formulation

We address the fundamental challenge of predicting tensorvalued quantities while quantifying predictive uncertainty. We present the construction for symmetric rank-2 tensors $C \in \mathbb { R } _ { \mathrm { s y m } } ^ { 3 \times 3 }$ , which already require a full $6 \times 6$ covariance representation. The SPD construction and scoring objective are representation-agnostic once an equivariant symmetric operator $A ( X )$ is available. However, the parameterization of $A ( X )$ is representation-specific and must be constructed separately for each tensor order and symmetry group. We predict C from a 3D structure $X = \bar { \{ ( \mathbf { x } _ { i } , f _ { i } ) \} } _ { i = 1 } ^ { \bar { N } }$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ denotes spatial coordinates and $f _ { i }$ represents associated features (such as atomic species in materials). This formulation encompasses important applications such as predicting dielectric tensors in crystal structures. Although we demonstrate this framework on materials science, the approach is applicable to any 3D point cloud with rank-2 tensorial attributes, ranging from biological molecules to geometric shapes.

Rather than producing deterministic point estimates, we model the full predictive distribution

$$
p ( C \mid X ) = \operatorname { L a p l a c e } ( C \mid \mu ( X ) , \Sigma ( X ) ) ,\tag{1}
$$

where $\mu ( X ) \in \mathbb { R } _ { \mathrm { { s y m } } } ^ { 3 \times 3 }$ is the predicted mean tensor and $\Sigma ( X )$ captures the predictive uncertainty through a covariance structure.

To maintain geometric consistency across all domains, both the mean prediction $\mu$ and covariance Σ must satisfy equivariance constraints with respect to rotations and reflections. Since we predict global tensor properties (e.g., dielectric tensor of a unit cell), the predictions are invariant to translations but equivariant to orthogonal transformations:

![](images/9a17cc0349d396081f9bda9f5d5592fb93ee0bb8120886c94800de79dd572f22.jpg)  
Figure 1. E(3)-equivariant tensor uncertainty framework. (1) Feature Decomposition: covariance head models symmetry via irreducible representations $( \ell = 0 , 2 , 4 )$ . (2) Manifold Mapping: network predicts unconstrained A ∈ sym(6). (3) Validity Constraint: $\Sigma = \exp ( A )$ projects to SPD manifold ${ \mathcal { P } } _ { 6 }$

$$
\begin{array} { r l } & { \mu ( R \cdot X ) = R \mu ( X ) R ^ { \top } , } \\ & { \Sigma ( R \cdot X ) = \rho _ { c } ( R ) \Sigma ( X ) \rho _ { c } ( R ) ^ { \top } . } \end{array}\tag{2}
$$

Our construction ensures full O(3) equivariance. For symmetric rank-2 tensors, the transformation under reflection (det $R = - 1 )$ is handled by the even-parity representation of the Kelvin-Mandel basis $\rho _ { c } ( R ) = R \otimes _ { s } R .$ , where (det $R ) ^ { 2 } = 1$ guarantees consistent behavior for both chiral and achiral structures. We numerically verify $O ( 3 )$ equivariance in Section 4.5 (Table 4), with detailed analysis in Appendix E.3. Predictions are translation-invariant since global tensor properties depend only on relative atomic positions. This universal equivariance constraint forms the foundation of our domain-agnostic framework. We emphasize that while Eq. 1 presents the distribution in standard predictive form, the training optimizes the robustified LE-ESO objective (Section 3.5) which generalizes the Multivariate Laplace negative log-likelihood with enhanced stability against outliers.

## 3.2. Voigt Representation and Covariance Structure

To facilitate neural network implementation while preserving the tensor’s geometric structure, we employ the Kelvin-Mandel notation to flatten the symmetric tensor C into a 6-dimensional vector. Unlike standard Voigt representation, Kelvin-Mandel notation maintains the isometry property between tensor and vector spaces:

$$
\mathbf { c } _ { \mathrm { K M } } = [ C _ { 1 1 } , C _ { 2 2 } , C _ { 3 3 } , \sqrt { 2 } C _ { 2 3 } , \sqrt { 2 } C _ { 1 3 } , \sqrt { 2 } C _ { 1 2 } ] ^ { \top } ,\tag{3}
$$

which preserves the Frobenius norm: $\| \mathbf { c } _ { \mathrm { K M } } \| _ { 2 } = \| C \| _ { F }$ Under rotation $R , { \bf c } _ { \mathrm { K M } } ^ { \prime } = \rho _ { c } ( R ) { \bf c } _ { \mathrm { K M } }$ where $\rho _ { c } ( R )$ is an

orthogonal $6 \times 6$ matrix. The covariance transforms as $\Sigma ^ { \prime } = \rho _ { c } ( R ) \Sigma \rho _ { c } ( R ) ^ { \top }$ , maintaining coordinate invariance of the physical uncertainty.

## 3.3. Irreducible Representation Decomposition

The mathematical structure of equivariant uncertainty quantification becomes clear through representation theory. In group theory, irreducible representations are fundamental building blocks that cannot be further decomposed into smaller invariant subspaces. The 6D representation $\rho _ { c }$ for symmetric $3 \times 3$ tensors decomposes into irreducible representations of SO(3) as

$$
\rho _ { c } \cong l = 0 \oplus l = 2 ,\tag{4}
$$

corresponding respectively to the isotropic (trace) and deviatoric (traceless) components of the tensor. Here <sup>∼</sup>= denotes an isomorphism of representations, not equality of matrices: after a fixed change of basis, the six Kelvin-Mandel coordinates split into a one-dimensional isotropic trace component and a five-dimensional traceless deviatoric component.

Since Σ transforms as $\rho _ { c } \otimes \rho _ { c }$ , its representation decomposes as:

$$
\begin{array} { r l } & { \rho _ { c } \otimes \rho _ { c } = ( l = 0 \oplus l = 2 ) \otimes ( l = 0 \oplus l = 2 ) } \\ & { \qquad = ( l = 0 ) \oplus 2 ( l = 2 ) \oplus ( l = 4 ) \oplus ( l = 1 , 3 ) _ { \mathrm { a n t i s y m } } } \end{array}\tag{5}
$$

The $\ell = 1$ and $\ell = 3$ components lie in the antisymmetric part of $\rho _ { c } \otimes \rho _ { c } ,$ , corresponding to operators that change sign under exchange of the two covariance indices. Since covariance matrices satisfy $\Sigma = \Sigma ^ { \top }$ , only the symmetric square $\mathrm { S y m } ^ { 2 } ( \rho _ { c } )$ remains, yielding:

$$
\mathrm { S y m } ^ { 2 } ( \rho _ { c } ) \cong 2 \times ( l = 0 ) \oplus 2 \times ( l = 2 ) \oplus 1 \times ( l = 4 ) ,\tag{6}
$$

This provides 21 independent degrees of freedom for symmetric $6 \times 6 \ : \mathrm { S P D }$ covariance matrices.

## 3.4. Equivariant Neural Architecture

Figure 1 summarizes the data flow. A shared E(3)- equivariant encoder first maps the input structure X to latent irreducible features. The mean head selects the $\ell = 0 \oplus \ell = 2$ components and outputs the Kelvin-Mandel mean vector $\mu _ { \mathrm { K M } } ( X ) \in \mathbb { R } ^ { 6 }$ , which is mapped back to a symmetric $3 \times 3$ tensor. The covariance head outputs coefficients in $2 \times ( \ell = 0 ) \oplus 2 \times ( \ell = 2 ) \oplus 1 \times ( \ell = 4 )$ which are assembled into an equivariant symmetric operator $A ( X ) \in { \mathfrak { s u m } } ( 6 )$ . Finally, the predictive covariance is $\Sigma ( X ) = \exp ( A ( X ) )$ ), and the loss is computed from the Kelvin-Mandel residual $\Delta \mathbf { c } = \mathbf { c } _ { \mathrm { K M } } - \pmb { \mu } _ { \mathrm { K M } } ( X )$

The covariance head $f _ { \Sigma } : X \mapsto A ( X )$ must satisfy the transformation property

$$
A ( R \cdot X ) = \rho _ { c } ( R ) A ( X ) \rho _ { c } ( R ) ^ { \top } .\tag{7}
$$

Following the irreducible representation decomposition in Eq. 6, we construct $A ( X )$ through structured Clebsch-Gordan combinations. Let $\overset { \vartriangle } { \boldsymbol { \phi } ^ { ( L ) } } ( \boldsymbol { X } ) \in \mathbb { R } ^ { F _ { L } \times ( 2 L + 1 ) }$ denote spherical tensor features of order L produced by the equivariant backbone. We index the irreps appearing in $\mathrm { S y m ^ { 2 } } ( \rho _ { c } )$ by

$$
{ \mathcal { T } } = \{ ( 0 , 1 ) , ( 0 , 2 ) , ( 2 , 1 ) , ( 2 , 2 ) , ( 4 , 1 ) \} ,
$$

where $( L , r )$ refers to the r-th copy of the L-irrep, and write

$$
A ( X ) = \sum _ { ( L , r ) \in \mathbb { Z } } \sum _ { m = - L } ^ { L } a _ { L , m } ^ { ( r ) } ( X ) B _ { L , m } ^ { ( r ) } ,\tag{8}
$$

with the equivariant coefficients

$$
a _ { L , m } ^ { ( r ) } ( X ) = \sum _ { f = 1 } ^ { F _ { L } } w _ { L , r , f } \phi _ { f , m } ^ { ( L ) } ( X ) .
$$

Here $B _ { L , m } ^ { ( r ) } \in \mathbb { R } _ { \mathrm { s y m } } ^ { 6 \times 6 }$ arefixed, input-independent basis matrices for the r-th copy of the L-irrep in $\mathrm { S y m } ^ { 2 } ( \rho _ { c } )$ , computed once from the Clebsch-Gordan decomposition. All input dependence resides in the equivariant coefficients $a _ { L , m } ^ { ( r ) } ( X )$ Under rotation, the coefficient vector $( a _ { L , m } ^ { ( r ) } ) _ { m = - L } ^ { L }$ and the basis $( B _ { L , m } ^ { ( r ) } ) _ { m = - L } ^ { L }$ transform by the same $\mathrm { W i g n e r } { - } D ^ { ( L ) }$ representation, so their contraction yields $A ( R \cdot X ) \ =$ $\rho _ { c } ( R ) A ( X ) \rho _ { c } ( R ) ^ { \intercal }$ . This explicit tensor basis construction guarantees that any choice of weights $w _ { L , r , f }$ preserves the equivariance property, providing hard-constrained geometric consistency rather than soft-regularized approximation. We note that while we rely on spherical tensor products, the orthogonal ICT decomposition matrices (Shao et al., 2025) provide an equivalent and highly efficient basis for higher-order Cartesian tensors, which may offer computational advantages for future extensions to rank-4 tensors like elasticity.

Our architecture implements an E(3)-equivariant neural network using the e3nn library, following the established paradigm of equivariant message passing. The implementation proceeds through two key stages. First, an equivariant message passing backbone processes the atomic structure to produce latent features transforming under mixed irreps up to $\ell _ { m a x } \ = \ 4$ Second, an equivariant linear layer—implemented via a fourth-order Cartesian tensor with symmetry $i j k l = j i k l = i j l k = k l i j$ —assembles these latent features into the symmetric block structure of A, ensuring consistency with the $\mathrm { S y m } ^ { 2 } ( \rho _ { c } )$ representation while automatically filtering to the $l = 0 , 2 .$ 4 components required for rank-4 covariance output. The covariance head uses a residual connection $A = A _ { \mathrm { b a s e } } \cdot I + \Delta A$ , where $A _ { \mathrm { b a s e } }$ provides an isotropic baseline and $\Delta A$ captures the anisotropic uncertainty structure learned from the data.

We employ an end-to-end joint optimization strategy, where both the mean and covariance heads are trained simultaneously. The inherent stability of our matrix-exponential mapping and the LE-ESO loss eliminates the need for gradient detachment, allowing the backbone to learn geometric features that are mutually informative for both point prediction and uncertainty quantification. This design guarantees that the raw network output A naturally possesses the correct equivariance properties, setting the stage for positive-definite covariance construction.

## 3.5. Positive-Definite Covariance Construction and Training Objective

From Curved Manifold to Flat Tangent Space. To strictly enforce the SPD constraint while maintaining equivariance, we leverage the Log-Euclidean framework (Arsigny et al., 2006) and parameterize Σ(X) via the matrix exponential mapping:

$$
\Sigma ( X ) = \exp ( A ( X ) ) ,\tag{9}
$$

We thus optimize within the tangent space sym $( 6 ) - \mathtt { a }$ flat Euclidean vector space at the identity—and the matrix exponential lifts $A ( X )$ to the SPD manifold ${ \mathcal { P } } _ { 6 }$ while preserving $E ( 3 )$ -equivariance, since $\rho _ { c } ( R )$ acts by orthogonal conjugation (Figure 2).

Log-Euclidean Equivariant Scoring Objective (LE-ESO). Given the Kelvin-Mandel residual $\Delta \mathbf { c } = \mathbf { c } _ { \mathrm { K M } } -$ $\mu _ { \mathrm { K M } } ( X )$ ), we define the Log-Euclidean Mahalanobis distance

$$
D _ { M } ( A , \Delta \mathbf { c } ) = \sqrt { \Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c } } .\tag{10}
$$

![](images/2a84ca1535c96b0f58aff10d56c26fca493ec3c4619fbdeaab924cea9fac2656.jpg)  
Figure 2. Geometric interpretation of equivariant covariance construction. The network operates in the Lie algebra $\mathfrak { g } \cong \mathbb { R } _ { \mathrm { s y m } } ^ { 6 \times 6 }$ (left), producing A(X). The exponential map Σ = exp(A) (center) projects onto the SPD manifold ${ \bar { \mathcal { P } } } _ { 6 }$ (right), guaranteeing valid uncertainties while preserving E(3)-equivariance.

To control the influence of extreme outliers, we use a robustified distance $\tilde { D } _ { M }$ with transition threshold $\tau { : }$

$$
\tilde { D } _ { M } = \left\{ \begin{array} { l l } { D _ { M } , } & { D _ { M } < \tau , } \\ { \tau + \log ( 1 + D _ { M } - \tau ) , } & { D _ { M } \geq \tau . } \end{array} \right.\tag{11}
$$

The final LE-ESO objective combines uncertainty volume regularization with the (robustified) data-fit term:

$$
\mathcal { L } _ { \mathrm { L E - E S O } } = \alpha \mathrm { T r } ( A ) + \tilde { D } _ { M } ,\tag{12}
$$

where $\alpha > 0$ controls the trade-off between uncertainty volume and data fit.

When $\alpha = 1$ and ${ \tilde { D } } _ { M } = D _ { M }$ , this is the multivariate Laplace negative log-likelihood in Log-Euclidean form, a strictly proper scoring rule on sym(6) (Gneiting & Raftery, 2007). With log-tail robustification or $\alpha \neq 1$ , the objective is a robust surrogate scoring objective, trading strict propriety for outlier stability. We set $\alpha = 1$ in our primary experiments; Appendix D.4 reports a validation sweep over $\alpha \in \{ 0 . 0 3 , 0 . 1 , 0 . 3 , 1 . 0 \}$ showing stable training, with α = 1 also giving the lowest validation MAE. Detailed derivation and gradient analysis are in Appendix A.4.

Training Stability via Joint Optimization. In our experiments, the Lie algebra parameterization combined with eigenvalue clamping and the log-tail robustification in Eq. 11 keeps gradients and the matrix exponential numerically wellbehaved during training, enabling stable end-to-end joint optimization of both the mean and covariance heads without gradient detachment, so the backbone can receive informative gradients from the uncertainty quantification objective. This joint training paradigm allows the learned geometric features to be optimized for both prediction accuracy and uncertainty calibration, without sacrificing numerical stability or equivariance guarantees.

## 4. Experiments

We evaluate the proposed framework in two main settings: (i) controlled geometric validation on ModelNet40 inertia tensors (Wu et al., 2015), and (ii) real-data dielectric tensor prediction on the Materials Project (Barroso-Luque et al., 2024; Jain et al., 2013). These main experiments are complemented by additional studies in Appendix D, including ModelNet40 shape-covariance prediction (Appendix D.1), rank-4 elasticity tensor prediction (Appendix D.2), runtime profiling (Appendix D.3), and sensitivity to the LE-ESO weight α (Appendix D.4). Dataset statistics are detailed in Appendix C.1.

We compare our equivariant full-covariance model against two primary baselines: (i) a deterministic model trained with MSE, and (ii) a diagonal UQ model assuming independent components. For ablation analysis, we evaluate nonequivariant and non-SPD variants (Section 4.5). All estimators utilize an E(3)-equivariant backbone with $\ell _ { m a x } = 4$ to support rank-4 covariance output; detailed hyperparameters and training protocols are provided in Appendix B.

Our experiments verify three central hypotheses: (1) geometric validation—exact equivariance preservation on complex 3D shapes; (2) accuracy—maintained point-prediction performance while modeling full covariance structure; and (3) calibration—covariance matrices that reflect true error distributions while respecting underlying geometry.

## 4.1. Controlled Geometric Validation: Inertia Tensor Prediction

The ModelNet40 experiments are intended as controlled geometric validation rather than as replacements for closedform tensor estimators. For inertia tensors, analytic formulas exist; our goal is to isolate whether the learned covariance remains equivariant, SPD, and geometrically meaningful under controlled point-cloud perturbations. The inertia tensor

![](images/291e0d0072e8df78d4772e93df78b4b0f1ec2ccca8efea3ccbebd92def984de1.jpg)  
Figure 3. 3D uncertainty visualization on ModelNet40. (Top) Point clouds with uncertainty ellipsoids (blue) and principal axes: predicted (red) vs ground truth (green). (Bottom) Anisotropic covariance matrices. Samples show varying errors (0.293–0.472) and Mahalanobis distances (0.37–0.50), demonstrating geometric-adaptive uncertainty that rotates with the object shape.

I transforms equivariantly under rotation $( \boldsymbol { \mathcal { Z } ^ { \prime } } = \boldsymbol { R } \boldsymbol { \mathcal { Z } } \boldsymbol { R } ^ { \intercal } )$ making it a clean target for verifying geometric consistency. We use the official split of 12,311 CAD models and introduce aleatoric uncertainty via Gaussian jitter applied to point positions (detailed preprocessing in Appendix C.1). For this validation task, the mean and covariance heads are trained jointly without gradient detachment, as the synthetic noise is well-behaved. We provide a second controlled rank-2 tensor validation on ModelNet40 shape covariance in Appendix D.1, showing that the construction is not tied to the inertia target.

We quantify equivariance error using the relative Frobenius norm $\begin{array} { r l r } { E _ { \mathrm { e q u i v } } } & { { } = } & { \| \Sigma ( R \cdot X ) - } \end{array}$ $\rho _ { c } ( R ) \Sigma ( X ) \rho _ { c } ( R ) ^ { \top } \| _ { F } / \| \Sigma ( \dot { X ) } \| _ { F }$ Prediction accuracy and uncertainty scores are reported in Table 1; equivariance and SPD-validity results are analyzed separately in Table 4, where the equivariance errors remain at the level of $1 0 ^ { - 7 }$ , confirming near-machine-precision symmetry preservation. Our full-covariance model reduces MAE by 15% relative to the diagonal baseline while maintaining perfect SPD properties (>99.9% validity, median condition number 6.8). Detailed SPD analysis is provided in Appendix E.5.

Table 1. ModelNet40 inertia tensor prediction. Our equivariant full-covariance framework achieves strong performance with exact geometric consistency and physical constraints.
<table><tr><td>METHOD</td><td>MAE</td><td>RMSE</td><td>LE-ESO</td><td>SPD</td></tr><tr><td>OURS (FULL COV)</td><td>0.078</td><td>0.128</td><td>10.66</td><td>√</td></tr><tr><td>DIAGONAL COV</td><td>0.092</td><td>0.156</td><td>12.84</td><td>√</td></tr><tr><td>DETERMINISTIC (MSE)</td><td>0.083</td><td>0.135</td><td>N/A</td><td>x</td></tr></table>

Figure 3 provides visual validation that our uncertainty estimates are physically meaningful: uncertainty ellipsoids align with principal shape axes (demonstrating E(3)- equivariance), expand in regions with sparse point density

(capturing sampling ambiguity), and preserve tensorial correlations across components.

## 4.2. Application: Dielectric Tensor Prediction

Dataset and Preprocessing. We utilize the Materials Project dielectric tensor dataset (Barroso-Luque et al., 2024; Jain et al., 2013) with static dielectric tensors computed via DFPT. To ensure data quality and consistency, we apply systematic filtering criteria (detailed in Appendix C.1), including structure size constraints $( 3 ~ \leq ~ \mathrm { a t o m s } ~ \leq ~ 3 0 )$ SPD positive-definiteness verification, and value range constraints. We apply Matrix Log-Normalization to reduce the dynamic range of dielectric tensors before converting them to Kelvin–Mandel coordinates, ensuring that the six independent components are scaled consistently across diverse crystal structures. We process atomic structures into graphs with 5.0A cutoff. Architecture and training details <sup>˚</sup> are provided in Appendix B.

## 4.3. Prediction Accuracy

Our goal is not state-of-the-art point prediction alone, but competitive accuracy with full-covariance, symmetrypreserving uncertainty. Table 2 shows our method achieves competitive MAE among UQ models (1.55, vs. 1.96 for the MACE deep ensemble and 2.25 for diagonal UQ), and remains close to deterministic point predictors such as DT-Net (Mao et al., 2024) (1.91) and GoeCTP (Hua et al., 2026) (1.41), which do not provide calibrated equivariant covariance estimates. The primary contribution is the principled, backbone-agnostic UQ mechanism, not a new point estimator. Modeling the full covariance manifold does not hinder mean estimation; the gains are most visible on off-diagonal components, which capture anisotropic directional dependencies.

Crucially, the parity plot in Figure 4a demonstrates that the high $R ^ { 2 }$ in Kelvin-Mandel log-space suggests that the E(3)- equivariant backbone effectively captures the underlying physics of dielectric properties while the UQ branch provides necessary aleatoric regularization. The predictive covariance is SPD by construction due to the matrix exponential. Separately, we also check whether the predicted mean dielectric tensors satisfy the expected positive-definiteness of the physical tensor; all test-set mean predictions satisfy this constraint in our run.

![](images/84f14a5dfbead075a9dbf6544c812d57408312bda21c0644db0eb32f777ddb4d.jpg)

![](images/63eec68247abd25e0f4b1a0d7a6efa67d07a50db45a0e2bc1b6ea978b67a3f58.jpg)  
Figure 4. Performance on Materials Project. (a) Parity plot of diagonal components $( R ^ { 2 } = 0 . 6 5 9 )$ . (b) Multivariate Laplace reliability diagram (MACE= 0.0489).

Table 2. Performance on Materials Project dielectric dataset. GoeCTP (Hua et al., 2026) employs a scalable equivariant architecture. DTNet (Mao et al., 2024) uses universal potential embeddings. MACE-Ens. is a 5-model deep ensemble. Ours (Calib.) applies temperature scaling (T ≈ 0.05). ES generalizes CRPS to multivariate settings.
<table><tr><td>METHOD</td><td>TYPE</td><td>MAE (↓)</td><td>LE-ESO (↓)</td><td>ES  $( \downarrow )$ </td><td>SPD?</td></tr><tr><td>SECONV (HEILMAN ET AL., 2024)</td><td>POINT</td><td>4.702</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>DTNET (MAO ET AL., 2024)</td><td>POINT</td><td>1.91</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>GOECTP (HUA ET AL., 2026)</td><td>POINT</td><td>1.41</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>DETERMINISTIC (MACE)</td><td>POINT</td><td>2.10</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MACE ENSEMBLE (N = 5)</td><td>UQ</td><td>1.96</td><td>-2.55</td><td>0.78</td><td>≈</td></tr><tr><td>DIAGONAL UQ</td><td>UQ</td><td>2.25</td><td>-1.85</td><td>0.87</td><td>√</td></tr><tr><td>OURS (FULL COV.)</td><td>UQ</td><td>1.55</td><td>-2.43</td><td>0.87</td><td>√</td></tr><tr><td>OURS (CALIBRATED)</td><td>UQ</td><td>1.55</td><td>-2.61</td><td>0.66</td><td>√</td></tr></table>

Extensibility to Higher-Order Tensors. To test extensibility beyond rank-2, Appendix D.2 reports a real-data rank-4 elasticity experiment, where the mean target is a rank-4 elasticity tensor with 21 independent components under standard minor/major symmetries. This is supporting evidence rather than a comprehensive rank-4 benchmark; the model achieves competitive MAE, improves empirical coverage from ∼35% for a naive UQ baseline to ∼52%, and preserves numerical equivariance and SPD validity.

## 4.4. Uncertainty Calibration

A primary contribution of our work is the calibration of tensor-valued uncertainty. Our model is trained using the Multivariate Laplace negative log-likelihood via the LE-ESO objective (Eq. 12), which naturally accounts for the heavy-tailed error distributions common in materials data. Figure 4b shows the reliability diagram evaluated against this same Multivariate Laplace distribution, confirming that our training objective and calibration assessment are properly matched.

The model achieves a MACE of 0.0489, indicating good agreement between predicted and empirical confidence levels under the multivariate Laplace evaluation protocol. Compared with the Gaussian-NLL objective in Table 3, LE-ESO gives lower calibration error and better accuracy on this dataset. While the curve remains slightly below the diagonal in the high-confidence regime, this indicates that the model is conservative (under-confident), which is preferable for high-stakes materials screening as it avoids over-optimistic

![](images/3363afb49dde6f1e65d92f84119fd199c62b891faef3b2d9db83433f954a3b1b.jpg)

![](images/2e3118962874cca99e2e73d16af405a04f21efd173454a4d2bb94be1c05fb73d.jpg)  
Figure 5. Uncertainty diagnostics and utility analysis. (a) Sharpness distribution of 95% confidence volumes, demonstrating the model’s ability to assign heteroscedastic uncertainties. (b) Risk-coverage analysis comparing ranking by directional uncertainty $( \lambda _ { \mathrm { m a x } } ( \Sigma ) )$ and total uncertainty (Tr(Σ)). Directional uncertainty provides a modest but consistent advantage in identifying high-error samples.

predictions.

Utility for Risk-Informed Decision Making. To evaluate the utility of our equivariant covariance tensors for riskinformed decision making, we perform a risk-coverage analysis comparing two ranking metrics: the total uncertainty (Trace(Σ)) and the directional uncertainty $\left( \lambda _ { \operatorname* { m a x } } ( \Sigma ) \right)$ . As illustrated in Figure 5b, the maximum eigenvalue $\lambda _ { \operatorname* { m a x } } -$ representing the variance along the most uncertain principal axis—serves as a more targeted indicator of directional prediction risk, capturing failure modes that are partially obscured by scalar total uncertainty.

At 90% coverage, ranking by $\lambda _ { \mathrm { m a x } } ( \Sigma )$ improves retainedset MAE by 3.1% relative to the full test set, compared with a 0.8% improvement when ranking by Trace(Σ). When compared directly against Trace-based ranking under the same retained-set protocol, the advantage of $\lambda _ { \mathrm { m a x } }$ is smaller but consistent, and persists at lower coverage levels where Trace-based ranking can fall below the full-dataset baseline. Appendix D.5 provides the full retained-set comparison, including the diagonal-UQ baseline. These results indicate that for anisotropic physical properties like dielectric tensors, capturing the directional components of uncertainty is informative for identifying potential failure modes beyond what isotropic or diagonal approximations expose.

Ablation of Training Objectives. This ablation isolates the effect of the scoring objective while keeping the equivariant backbone and matrix-exponential covariance head fixed. We compare LE-ESO against the standard Gaussian NLL and Multivariate Energy Score (ES) training. As summarized in Table 3, the Laplace-based LE-ESO achieves the lowest MAE (1.55) and calibration error (MACE 0.049, ES 0.66). The performance gap relative to Gaussian NLL (MAE 1.78) is consistent with the heavy-tailed nature of materials residuals, where the linear Mahalanobis penalty of the Laplace formulation is less sensitive to extreme deviations than the quadratic Gaussian penalty. Energy Score training is competitive on calibration but exhibits higher MAE (1.64). Overall, the results suggest that the Laplace-based LE-ESO objective provides a better accuracy-calibration trade-off than Gaussian NLL or Energy Score training in this dataset.

Table 3. Ablation of training objectives on Materials Project. All variants use the same E(3)-equivariant backbone and matrixexponential head. LE-ESO (Ours) demonstrates superior robustness to heavy-tailed noise compared to Gaussian NLL.
<table><tr><td>TRAINING OBJECTIVE</td><td>MAE (↓)</td><td>MACE (↓)</td><td>ES (↓)</td></tr><tr><td>GAUSSIAN NLL</td><td>1.78</td><td>0.092</td><td>1.05</td></tr><tr><td>ENERGY SCORE (ES)</td><td>1.64</td><td>0.054</td><td>0.81</td></tr><tr><td>LE-ESO (OURS)</td><td>1.55</td><td>0.049</td><td>0.66</td></tr></table>

Chemical Out-of-Distribution Analysis. Beyond internal calibration, a key utility of symmetry-preserving UQ is identifying Out-of-Distribution (OOD) samples during materials screening. We perform a Chemical Substitution Analysis by replacing common atoms in the test set with unseen elements (e.g., Actinides U, Pu; Rare Earths Gd, Sm). As shown in Figure 6, the predicted directional uncertainty $\lambda _ { \mathrm { m a x } }$ rises monotonically from 2.71 to 5.23 (+93.2%) as the substitution ratio reaches 100%, suggesting that the covariance head captures patterns correlated with chemical distribution shift; we do not claim to disentangle aleatoric and epistemic uncertainty in this analysis.

## 4.5. Equivariance and Validity Verification

Table 4 isolates equivariance and SPD validity across four baselines: A (non-equivariant GNN + Cholesky), B (E3NN + coordinate-wise heads), B<sup>′</sup> (our equivariant mean head + Cholesky covariance), and C (direct equivariant regression of $A ( X )$ without the matrix exponential). Implementation details are in Appendix C.2.

Table 4. Equivariance and SPD-validity analysis. Baseline B<sup>′</sup> isolates the Cholesky covariance failure mode (equivariant $\mu ,$ non-equivariant Σ). Only the matrix-exponential head achieves both covariance equivariance and strict SPD validity.
<table><tr><td>METHOD</td><td> $\mathcal { E } _ { \mu }$ </td><td> $\mathcal { E } _ { \Sigma }$ </td><td>MEAN EQUIV.? COV. EQUIV.? SPD?</td><td></td><td></td></tr><tr><td>BASELINE  $\mathrm { A \ ( N O N \mathrm { - } E Q U I V A R I A N T \ G N N + C H O L E S K Y ) }$ </td><td> $1 . 3 6 \times 1 0 ^ { 0 }$ </td><td> $1 . 0 7 \times 1 0 ^ { 0 }$ </td><td>x</td><td>x</td><td>√</td></tr><tr><td>BASELINE  $\mathrm { \Delta B \ ( E 3 N N + C O O R D I N A T E - W I S E ~ H E A D S ) }$ </td><td> $1 . 2 8 \times 1 0 ^ { 0 }$ </td><td> $1 . 0 3 \times 1 0 ^ { 0 }$ </td><td>x</td><td>x</td><td>√</td></tr><tr><td>BASELINE  $\mathbf { B } ^ { \prime } \left( \mathrm { E Q U I V A R I A N T M E A N + C H O L E S K Y C O V . } \right)$ </td><td> $1 . 4 3 \times 1 0 ^ { - 6 }$ </td><td> $4 . 3 0 \times 1 0 ^ { - 1 }$ </td><td>√</td><td>x</td><td>√</td></tr><tr><td>BASELINE  $\mathrm { \Delta C \ ( E 3 N N + D I R E C T ~ E Q U I V A R I A N T ~ C O V . ) }$ </td><td> $7 . 5 9 \times 1 0 ^ { - 7 }$ </td><td> $4 . 7 6 \times 1 0 ^ { - 7 }$ </td><td>√</td><td>√</td><td>x</td></tr><tr><td>OURS (E3NN + MATRIX EXP.)</td><td> $\mathbf { 2 . 3 9 \times 1 0 ^ { - 7 } }$ </td><td> $\mathbf { 2 . 7 5 \times 1 0 ^ { - 7 } }$ </td><td>√</td><td>√</td><td>√</td></tr></table>

![](images/bfef4f7261d00d7d7574c919f7514b47054aacb2a600d272ffd47ff182043713.jpg)  
Figure 6. Chemical OOD sensitivity analysis. The predicted uncertainty increases as the crystal lattice is populated with unseen chemical species, suggesting that the model provides a useful distribution-shift risk signal.

Baseline B<sup>′</sup> reaches near-machine-precision $\mathcal { E } _ { \mu } \approx 1 . 4 \times 1 0 ^ { - 6 }$ but $\mathcal { E } _ { \Sigma } \approx 0 . 4 3 .$ , confirming that the failure is specific to Cholesky: its lower-triangular structure is not preserved under orthogonal conjugation by $\rho _ { c } ( R )$ . Only the matrixexponential head achieves $O ( 1 0 ^ { - 7 } )$ equivariance error with strict SPD validity.

Computational Overhead. Appendix D.3 reports runtime profiling. The full-covariance model is more expensive than the deterministic baseline because of the covariance branch and its backpropagation, but it provides the full anisotropic covariance in a single forward pass without ensembling.

## 5. Discussion

Our primary validated setting is E(3)-equivariant UQ for symmetric rank-2 tensors. The additional shape-covariance experiment in Appendix D.1 shows that the construction is not specific to the inertia target, while the rank-4 elasticity experiment in Appendix D.2 provides supporting evidence for higher-order tensor targets. However, the higher-order empirical study is not yet exhaustive, and broader validation across additional equivariant backbones, tensor orders, and symmetry groups remains future work. The construction separates a group-agnostic SPD/UQ core—namely, the matrix-exponential mapping and the Log-Euclidean scoring objective—from a representation-specific equivariant parameterization of the symmetric operator $A ( X )$ . Extending the framework to other groups therefore depends on the availability of suitable representation-theoretic bases and implementation tools.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62573132) and the Fundamental Research Funds for the Central Universities (Grant No. 2025SMECP012).

## Impact Statement

This paper introduces a framework for equivariant uncertainty quantification in tensor-valued geometric learning. Its primary potential impact lies in scientific machine learning, particularly in materials discovery tasks where tensor-valued properties such as dielectric or elastic responses are important. By providing symmetry-preserving uncertainty estimates, the method may support more reliable and risk-aware computational screening, helping researchers prioritize candidates for further validation. At the same time, predictions from such models should not be treated as substitutes for experimental or domain-expert verification. Responsible use of the framework requires human-in-the-loop assessment, careful calibration checks, and validation on the target scientific domain.

## References

Arsigny, V., Fillard, P., Pennec, X., and Ayache, N. Logeuclidean metrics for fast and simple calculus on diffusion tensors. Magnetic Resonance in Medicine, 56(2):411– 421, 2006.

Barroso-Luque, L., Shuaibi, M., Fu, X., Wood, B. M., Dzamba, M., Gao, M., Rizvi, A., Zitnick, C. L., and Ulissi, Z. W. Open materials 2024 (omat24) inorganic materials dataset and models. arXiv preprint arXiv:2410.12771, 2024.

Batatia, I., Kovacs, D. P., Simm, G., Ortner, C., and Csanyi,´ G. Mace: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in neural information processing systems, 35: 11423–11436, 2022.

Berman, E., Ginesin, J., Pacini, M., and Walters, R. On Uncertainty Calibration for Equivariant Functions. Transactions on Machine Learning Research, 2026. URL https://mlanthology.org/tmlr/2026/ berman2026tmlr-uncertainty/.

Bevanda, P., Beier, M., Capone, A., Sosnowski, S. G., Hirche, S., and Lederer, A. Koopman-equivariant gaussian processes. In Li, Y., Mandt, S., Agrawal, S., and Khan, E. (eds.), Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pp. 3151–3159. PMLR, 2025. URL https://proceedings.mlr.press/ v258/bevanda25a.html.

Doan, B. G., Shamsi, A., Guo, X.-Y., Mohammadi, A., Alinejad-Rokny, H., Sejdinovic, D., Teney, D., Ranasinghe, D. C., and Abbasnejad, E. Bayesian low-rank learning (Bella): A practical approach to Bayesian neural networks. Proceedings ofthe AAAI Conference on Artificial Intelligence, 39(15):16298–16307, 2025. doi: 10. 1609/aaai.v39i15.33790. URL https://ojs.aaai. org/index.php/AAAI/article/view/33790.

Du, H., Wang, J., Hui, J., Zhang, L., and Wang, H. Densegnn: universal and scalable deeper graph neural networks for high-performance property prediction in crystals and molecules. npj Computational Materials, 10 (1):292, 2024.

Equer, L., Rusch, T. K., and Mishra, S. Multi-scale message passing neural pde solvers. arXiv preprint arXiv:2302.03580, 2023.

Fakour, F., Mosleh, A., and Ramezani, R. A structured review of literature on uncertainty in machine learning & deep learning. arXiv preprint arXiv:2406.00332, 2024.

Fung, V., Zhang, J., Juarez, E., and Sumpter, B. G. Benchmarking graph neural networks for materials chemistry. npj Computational Materials, 7(1):84, 2021.

Geiger, M. and Smidt, T. e3nn: Euclidean neural networks. arXiv preprint arXiv:2207.09453, 2022.

Gerken, J. E. and Kessel, P. Emergent equivariance in deep ensembles. arXiv preprint arXiv:2403.03103, 2024.

Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477):359–378, 2007.

Gruber, S. and Buettner, F. Better uncertainty calibration via proper scores for classification and beyond. Advances in Neural Information Processing Systems, 35:8618–8632, 2022.

Heilman, A., Schlesinger, C., and Yan, Q. Equivariant graph neural networks for prediction of tensor material properties of crystals. arXiv preprint arXiv:2406.03563, 2024.

Hua, H., Yang, J., Lin, W., and Zhou, P. Revisiting the Canonicalization for Fast and Accurate Crystal Tensor Property Prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 40(1):417–425, 2026. doi: 10. 1609/aaai.v40i1.37004. URL https://ojs.aaai. org/index.php/AAAI/article/view/37004.

Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G., et al. Commentary: The materials project: A materials genome approach to accelerating materials innovation. APL materials, 1(1), 2013.

Jekel, C. F., Swartz, K. E., White, D. A., Tortorelli, D. A., and Watts, S. E. Neural network layers for prediction of positive definite elastic stiffness tensors. arXiv preprint arXiv:2203.13938, 2022.

Kuleshov, V., Fenner, N., and Ermon, S. Accurate uncertainties for deep learning using calibrated regression. In International Conference on Machine Learning, pp. 2796– 2804. PMLR, 2018.

Mao, Z., Li, W., and Tan, J. Dielectric tensor prediction for inorganic materials using latent information from preferred potential. npj Computational Materials, 10(1):265, 2024.

Newman, E., Horesh, L., Avron, H., and Kilmer, M. E. Stable tensor neural networks for efficient deep learning. Frontiers in Big Data, 7:1363978, 2024.

Olivier, A., Shields, M. D., and Graham-Brady, L. Bayesian neural networks for uncertainty quantification in datadriven materials modeling. Computer methods in applied mechanics and engineering, 386:114079, 2021.

Pakornchote, T., Ektarawong, A., and Chotibut, T. Straintensornet: Predicting crystal structure elastic properties using se(3)-equivariant graph neural networks. Physical Review Research, 5(4):043198, 2023.

Pouliquen, C., Massias, M., and Vayer, T. Schur’s positivedefinite network: Deep learning in the spd cone with structure. In International Conference on Learning Representations, volume 2025, pp. 71401–71416, 2025.

Reiser, P., Neubert, M., Eberhard, A., Torresi, L., Zhou, C., Shao, C., Metni, H., van Hoesel, C., Schopmans, H., Sommer, T., et al. Graph neural networks for materials science and chemistry. Communications Materials, 3(1): 93, 2022.

Rensmeyer, T., Craig, B., Kramer, D., and Niggemann, O. High accuracy uncertainty-aware interatomic force modeling with equivariant bayesian neural networks. Digital Discovery, 3(11):2356–2366, 2024.

Rudner, T. G., Chen, Z., Teh, Y. W., and Gal, Y. Tractable function-space variational inference in bayesian neural networks. Advances in Neural Information Processing Systems, 35:22686–22698, 2022.

Shao, S., Li, Y., Lin, Z., and Cui, Q. High-rank irreducible cartesian tensor decomposition and bases of equivariant spaces. Journal ofMachine Learning Research, 26(175): 1–53, 2025. URL http://jmlr.org/papers/ v26/25-0134.html.

Sheinkman, A. and Wade, S. The architecture and evaluation of bayesian neural networks. arXiv e-prints, pp. arXiv– 2503, 2025.

Steinert, T., Ginsbourger, D., Lykke-Møller, A., Christiansen, O., and Moss, H. Integration-free kernels for equivariant gaussian process modelling. In Forty-second International Conference on Machine Learning, 2025.

Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., and Xiao, J. 3d shapenets: A deep representation for volumetric shapes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 1912– 1920, 2015.

Zhao, W., Lopez, F., Riestenberg, J. M., Strube, M., Taha, D., and Trettel, S. Modeling graphs beyond hyperbolic: Graph neural networks in symmetric positive definite matrices. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 122–139. Springer, 2023.

Zhou, X.-H., Liu, Z.-R., and Xiao, H. Bi-eqno: generalized approximate bayesian inference with an equivariant neural operator framework. arXiv preprint arXiv:2410.16420, 2024.

## A. Theoretical Proofs and Derivations

In this section, we provide formal statements and proofs establishing the mathematical validity of our equivariant uncertainty formulation. We establish the theoretical foundations for (i) the representation-theoretic decomposition of covariance tensors, (ii) the matrix exponential construction ensuring both positive-definiteness and equivariance, and (iii) the numerical stability of our loss formulation.

## A.1. Rotation Matrices in Kelvin-Mandel Space

For a rotation matrix $R \in S O ( 3 )$ , the corresponding $6 \times 6$ transformation matrix $\rho _ { c } ( R )$ in Kelvin-Mandel space can be derived from the Kronecker product structure. The vectorization operation vec(C) maps a symmetric tensor $C$ to a 9-dimensional vector, and under rotation:

$$
\operatorname { v e c } ( C ^ { \prime } ) = ( R \otimes R ) \operatorname { v e c } ( C ) ,\tag{13}
$$

where $\otimes$ denotes the Kronecker product.

The matrix $\rho _ { c } ( R )$ is obtained by projecting $R \otimes R$ onto the 6-dimensional symmetric subspace and applying the Kelvin-Mandel scaling matrix P:

$$
\rho _ { c } ( R ) = \mathbf { P } \cdot \boldsymbol { S } \cdot ( R \otimes R ) \cdot \boldsymbol { S } ^ { T } \cdot \mathbf { P } ^ { - 1 } ,\tag{14}
$$

where $s$ is the $6 \times 9$ selection matrix.

For practical computation, $\rho _ { c } ( R )$ has the explicit block structure:

$$
\rho _ { c } ( R ) = [ \begin{array} { c c c c c c } { R _ { 1 1 } ^ { 2 } } & { R _ { 1 2 } ^ { 2 } } & { R _ { 1 3 } ^ { 2 } } & { \sqrt { 2 } R _ { 1 2 } R _ { 1 3 } } & { \sqrt { 2 } R _ { 1 1 } R _ { 1 3 } } & { \sqrt { 2 } R _ { 1 1 } R _ { 1 2 } } \\ { R _ { 2 1 } ^ { 2 } } & { R _ { 2 2 } ^ { 2 } } & { R _ { 2 3 } ^ { 2 } } & { \sqrt { 2 } R _ { 2 2 } R _ { 2 3 } } & { \sqrt { 2 } R _ { 2 1 } R _ { 2 3 } } & { \sqrt { 2 } R _ { 2 1 } R _ { 2 2 } } \\ { R _ { 3 1 } ^ { 2 } } & { R _ { 3 2 } ^ { 3 } } & { R _ { 3 3 } ^ { 2 } } & { \sqrt { 2 } R _ { 3 2 } R _ { 3 3 } } & { \sqrt { 2 } R _ { 3 1 } R _ { 3 3 } } & { \sqrt { 2 } R _ { 3 1 } R _ { 3 2 } } \\ { \sqrt { 2 } R _ { 2 1 } R _ { 3 1 } } & { \sqrt { 2 } R _ { 2 2 } R _ { 3 2 } } & { \sqrt { 2 } R _ { 2 3 } R _ { 3 3 } } & { R _ { 2 2 } R _ { 3 3 } + R _ { 2 3 } R _ { 3 2 } } & { R _ { 2 1 } R _ { 3 3 } + R _ { 2 3 } R _ { 3 1 } } & { R _ { 2 1 } R _ { 3 2 } + R _ { 2 2 } R _ { 3 1 } } \\ { \sqrt { 2 } R _ { 1 1 } R _ { 3 1 } } & { \sqrt { 2 } R _ { 1 2 } R _ { 3 2 } } & { \sqrt { 2 } R _ { 1 3 } R _ { 3 3 } } & { R _ { 1 2 } R _ { 3 3 } + R _ { 1 3 } R _ { 3 2 } } & { R _ { 1 1 } R _ { 3 3 } + R _ { 1 3 } R _ { 3 1 } } & { R _ { 1 1 } R _ { 3 2 } + R _ { 1 2 } R _ { 3 1 } } \\  \sqrt { 2 } R _ { 1 1 } R \end{array}\tag{15}
$$

This explicit form ensures that $\rho _ { c } ( R )$ maintains orthogonality in Kelvin-Mandel space: $\rho _ { c } ( R ) ^ { T } \rho _ { c } ( R ) = I _ { 6 }$

Note on Voigt vs. Kelvin-Mandel Notation. While standard Voigt notation maps $C _ { i j }$ to $[ C _ { 1 1 } , C _ { 2 2 } , C _ { 3 3 } , C _ { 2 3 } , C _ { 1 3 } , C _ { 1 2 } ] ^ { T }$ it does not preserve the Frobenius norm. Kelvin-Mandel notation applies $\sqrt { 2 }$ scaling to shear components, ensuring $\| \mathbf { c } _ { \mathrm { K M } } \| _ { 2 } = \| C \| _ { F }$ . This isometric property is crucial for maintaining geometric consistency in uncertainty quantification.

## A.2. Irreducible Representation Decomposition

Proposition A.1 (Irreducible decomposition of the covariance representation). Let $\rho _ { c }$ denote the 6-dimensional real representation of $\mathrm { \ddot { S } O ( 3 ) }$ corresponding to symmetric rank-2 tensors, i.e.

$$
\rho _ { c } \cong l = 0 \oplus l = 2 .
$$

Then the symmetric tensor product representation of $\dot { \rho } _ { c }$ decomposes as

$$
{ \mathrm { S y m } } ^ { 2 } ( \rho _ { c } ) \cong 2 \times ( l = 0 ) \oplus 2 \times ( l = 2 ) \oplus 1 \times ( l = 4 ) ,
$$

which possesses 21 independent degrees offreedom—equal to that ofa symmetric $6 \times 6$ covariance matrix.

Proof. Since $\rho _ { c } \cong l = 0 \oplus l = 2$ , the symmetric square decomposes as

$$
\mathrm { S y m } ^ { 2 } ( \rho _ { c } ) \cong \mathrm { S y m } ^ { 2 } ( l = 0 ) \oplus ( l = 0 \otimes l = 2 ) \oplus \mathrm { S y m } ^ { 2 } ( l = 2 ) .
$$

We have $\mathrm { S y m } ^ { 2 } ( l = 0 ) = l = 0 , l = 0 \otimes l = 2 = l = 2 ,$ , and $\mathrm { S y m } ^ { 2 } ( l = 2 ) = l = 0 \oplus l = 2 \oplus l = 4$ . Combining these yields

$$
\mathrm { S y m } ^ { 2 } ( \rho _ { c } ) \cong 2 \times ( l = 0 ) \oplus 2 \times ( l = 2 ) \oplus 1 \times ( l = 4 ) ,
$$

which has dimension $2 \cdot 1 + 2 \cdot 5 + 1 \cdot 9 = 2 1$

## A.3. Matrix Exponential Properties

Proposition A.2 (Positive-definiteness and equivariance of the exponential map). Let $A ( X ) \in \mathbb { R } _ { \mathrm { s y m } } ^ { 6 \times 6 }$ satisfy the equivariance condition

$$
A ( R { \cdot } X ) = \rho _ { c } ( R ) A ( X ) \rho _ { c } ( R ) ^ { \top } \quad \forall R \in O ( 3 ) .
$$

Then the matrix exponential

$$
\Sigma ( X ) = \exp ( A ( X ) )
$$

is (i) symmetric positive-definite for all X, and (ii) equivariant under the same group action:

$$
\Sigma ( R \cdot X ) = \rho _ { c } ( R ) \Sigma ( X ) \rho _ { c } ( R ) ^ { \top } .
$$

Proof. For any real symmetric A, there exists an orthogonal Q and real diagonal Λ such that $A = Q \Lambda Q ^ { \top }$ . Then

$$
\exp ( A ) = Q \exp ( \Lambda ) Q ^ { \top } ,
$$

where $\exp ( \Lambda )$ has strictly positive diagonal entries $\exp ( \lambda _ { i } ) > 0$ . Thus $\exp ( A )$ is symmetric positive-definite.

For equivariance, note that $\rho _ { c } ( R )$ is orthogonal. The matrix exponential satisfies $\exp ( S A S ^ { - 1 } ) = S \exp ( A ) S ^ { - 1 }$ for any invertible S. Taking $S = \rho _ { c } ( R )$ gives

$$
\begin{array} { r } { \exp ( \rho _ { c } ( R ) A \rho _ { c } ( R ) ^ { \top } ) = \rho _ { c } ( R ) \exp ( A ) \rho _ { c } ( R ) ^ { \top } , } \end{array}
$$

which proves equivariance.

Proposition A.3 (Equivariance of spectral functions). Let $f : \mathbb { R } \to \mathbb { R }$ be a scalarfunction. For a symmetric matrix A with eigenvalue decomposition $A = Q \Lambda Q ^ { \top }$ , define the spectral function $F ( A ) = Q d i a g ( f ( \lambda _ { 1 } ) , \ldots , f ( \lambda _ { n } ) ) Q ^ { \top }$ . Since $\rho _ { c } ( R )$ is orthogonalfor all $R \in O ( 3 )$ , it follows that

$$
\begin{array} { r } { F ( \rho _ { c } ( R ) A \rho _ { c } ( R ) ^ { \top } ) = \rho _ { c } ( R ) F ( A ) \rho _ { c } ( R ) ^ { \top } . } \end{array}
$$

Thus, eigenvalue clamping and anisotropic jitter (as defined in Eq. 18) preserve $O ( 3 )$ equivariance.

Proof. For any orthogonal matrix $U ,$ , the spectral function commutes with orthogonal similarity transformations: $F ( \bar { U } A U ^ { \top } ) = \bar { U } F ( A ) \bar { U } ^ { \top }$ . This follows from the fact that $U A U ^ { \top } = Q \Lambda Q ^ { \top }$ where $Q = U Q _ { 0 }$ for the original eigenvectors $Q _ { 0 }$ of A. Applying the definition of $F \colon$

$$
F ( U A U ^ { \top } ) = ( U Q _ { 0 } ) \mathrm { d i a g } ( f ( \lambda _ { i } ) ) ( U Q _ { 0 } ) ^ { \top } = U ( Q _ { 0 } \mathrm { d i a g } ( f ( \lambda _ { i } ) ) Q _ { 0 } ^ { \top } ) U ^ { \top } = U F ( A ) U ^ { \top } .
$$

Taking $U = \rho _ { c } ( R )$ completes the proof.

## A.4. Numerically Stable Loss Function

We present two formulations: the standard Gaussian NLL (for comparison) and the Multivariate Laplace NLL used in our implementation.

Proposition A.4 (Gaussian NLL in Log-Euclidean form). Let A be a symmetric matrix, $\Sigma = \exp ( A )$ , and $\Delta \mathbf { c } = \mathbf { c } _ { t r u e } - \mu .$ The standard Gaussian negative log-likelihood is

$$
\mathcal { L } _ { G a u s s } = \frac { 1 } { 2 } \log \operatorname* { d e t } \Sigma + \frac { 1 } { 2 } \Delta \mathbf { c } ^ { \top } \Sigma ^ { - 1 } \Delta \mathbf { c } .
$$

Then the following loss is algebraically equivalent and numerically stable:

$$
{ \bigg | } { \mathcal { L } } _ { G a u s s } = { \frac { 1 } { 2 } } \operatorname { T r } ( A ) + { \frac { 1 } { 2 } } \Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c } . { \bigg | }
$$

Proof. $\operatorname { U s i n g } \Sigma \ = \ \exp ( A )$ and the identity det $( \exp ( A ) ) ~ = ~ \exp ( \mathrm { T r } ( A ) )$ , we obtain log det $\Sigma ~ = ~ \operatorname { T r } ( A )$ Since $( \exp ( A ) ) ^ { - 1 } = \exp ( - A )$ , the Mahalanobis term becomes $\Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c }$ □

Proposition A.5 (Multivariate Laplace NLL in Log-Euclidean form). Let A be a symmetric matrix, $\Sigma = \exp ( A )$ , and $\Delta \mathbf { c } = \mathbf { c } _ { t r u e } - \mu .$ . The Multivariate Laplace negative log-likelihood (with unit scale) is

$$
\mathcal { L } _ { L a p l a c e } = \log \operatorname* { d e t } \Sigma + \sqrt { \Delta \mathbf { c } ^ { \top } \Sigma ^ { - 1 } \Delta \mathbf { c } } .
$$

The numerically stable form in the Lie algebra sym(6) is:

$$
\boxed { \mathcal { L } _ { L a p l a c e } = \mathrm { T r } ( A ) + D _ { M } , }
$$

where $D _ { M } = \sqrt { \Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c } }$ is the Mahalanobis distance in the Log-Euclidean metric.

Proof. The log-determinant term follows identically: log det $\Sigma = \operatorname { \mathrm { T r } } ( A )$ . For the Mahalanobis distance term, note that $\Sigma ^ { - 1 } = \exp ( - A )$ , so:

$$
D _ { M } = { \sqrt { \Delta \mathbf { c } ^ { \top } \Sigma ^ { - 1 } \Delta \mathbf { c } } } = { \sqrt { \Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c } } } .
$$

The key difference from the Gaussian case is the square root: the Laplace NLL is linear in $D _ { M }$ rather than quadratic in $D _ { M } ^ { 2 } .$ This provides robustness to outliers, as large residuals contribute linearly rather than quadratically to the loss. □

Laplacian-Huber Compound Robust Loss. To handle extreme outliers in materials data, we introduce a Laplacian-Huber scheme that combines two complementary robustness mechanisms. For the Mahalanobis distance $D _ { M } =$ $\sqrt { \Delta \mathbf { c } ^ { \top } \Sigma ^ { - 1 } \Delta \mathbf { c } }$ , our robust loss is:

$$
\tilde { D } _ { M } = \left\{ \begin{array} { l l } { { D _ { M } , } } & { { D _ { M } < \tau \quad \mathrm { ( l i n e a r r e g i o n ) } } } \\ { { \tau + \log ( 1 + D _ { M } - \tau ) , } } & { { D _ { M } \geq \tau \quad \mathrm { ( l o g \mathrm { - } t a i l r e g i o n ) } } } \end{array} \right.\tag{16}
$$

This design has a clear statistical interpretation: (1) the linear region $( D _ { M } < \tau )$ preserves the core Laplace distribution assumption, providing natural robustness through a linear rather than quadratic penalty on residuals; (2) the log-tail region $( D _ { M } \ge \tau )$ smoothly compresses the contribution of extreme residuals so that the residual penalty grows logarithmically rather than linearly, mitigating large updates from rare outliers in practice. We set $\tau = 5 . 0$ based on validation analysis—approximately 99% of well-predicted samples have $D _ { M } < 5$ , while extreme outliers beyond this threshold are smoothly bounded without affecting the majority of the data distribution

Gradient Behavior of the Laplace NLL. Using eigenvalue decomposition $A = Q \Lambda Q ^ { \top }$ , define the whitened residual $\mathbf { z } = Q ^ { \top } \Delta \mathbf { c }$ . The Mahalanobis distance becomes:

$$
D _ { M } = \sqrt { \sum _ { i = 1 } ^ { 6 } z _ { i } ^ { 2 } \exp ( - \lambda _ { i } ) } .
$$

The gradient with respect to eigenvalues Λ in the linear region $( D _ { M } < \tau )$ is:

$$
\frac { \partial \mathcal { L } _ { \mathrm { L a p l a c e } } } { \partial \lambda _ { k } } = 1 - \frac { z _ { k } ^ { 2 } \exp ( - \lambda _ { k } ) } { 2 D _ { M } } ,\tag{17}
$$

Compared to the Gaussian gradient $\begin{array} { r } { \frac { 1 } { 2 } \big ( 1 - z _ { k } ^ { 2 } \exp ( - \lambda _ { k } ) \big ) } \end{array}$ , the Laplace gradient carries an additional $1 / ( 2 D _ { M } )$ factor that reduces the growth rate of the data-fit term, but it does not by itself yield a uniform bound: when a single direction k dominates $D _ { M }$ (so $D _ { M } \approx | z _ { k } | \exp ( - \lambda _ { k } / 2 ) ;$ ), the contribution behaves like $| z _ { k } | \exp ( - \lambda _ { k } / 2 ) / 2$ and is unbounded as $\lambda _ { k } \to - \infty$ . In the logarithmic tail region $( D _ { M } \ge \tau )$ , the gradient becomes:

$$
\frac { \partial \tilde { D } _ { M } } { \partial \lambda _ { k } } = - \frac { z _ { k } ^ { 2 } \exp ( - \lambda _ { k } ) } { 2 ( 1 + D _ { M } - \tau ) D _ { M } } ,
$$

which approaches a finite constant $(  - 1 / 2$ in a single dominant direction) rather than diverging linearly with $z _ { k } ^ { 2 } \exp ( - \lambda _ { k } )$ as in the unmodified Laplace case. We therefore do not claim that the Lie algebra parameterization alone guarantees bounded gradients near the SPD boundary; rather, in our implementation, gradient and matrix-exponential overflow are controlled jointly by (i) eigenvalue clamping $\lambda _ { k } \in [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ before the exponential map, and (ii) the log-tail robustification in Eq. 16.

Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning
<table><tr><td>Property</td><td>Gaussian NLL</td><td>Laplace NLL (LE-ESO)</td></tr><tr><td>Mahalanobis term</td><td> $\phantom { } _ { 2 } ^ { 1 } D _ { M } ^ { 2 }$ </td><td> $D _ { M }$ </td></tr><tr><td>Penalty shape</td><td>Quadratic</td><td>Linear</td></tr><tr><td>Log-det coefficient</td><td> $\frac { 1 } { 2 }$ </td><td>1 (tunable as α)</td></tr><tr><td>Outlier sensitivity</td><td>High (quadratic growth)</td><td>Low (linear growth)</td></tr><tr><td>Gradient for large  $D _ { M }$ </td><td> $\propto D _ { M }$ </td><td>∝1</td></tr><tr><td>Statistical assumption</td><td>Light-tailed errors</td><td>Heavy-tailed errors</td></tr></table>

Table 5. Comparison of Gaussian and Laplace negative log-likelihood formulations.

Comparison: Gaussian vs. Laplace NLL. The key distinctions are summarized below:

## B. Model Architecture and Implementation Details

## B.1. Network Architecture and Data Flow

Our architecture implements an E(3)-equivariant neural network using standard message passing layers. The input atomic numbers are first projected into 119-dimensional Magpie feature embeddings, which then pass through $L = 2$ interaction layers with a hidden dimension of 64. The backbone employs SiLU activations for scalar features and gated-tanh for higher-order tensors to preserve equivariance throughout computation. To support rank-4 covariance output, we set the maximum rotation order $\ell _ { m a x } = 4 .$ , enabling the full $\mathrm { \bar { S y m } } ^ { 2 } ( \rho _ { c } )$ representation required by our theoretical decomposition.

The network branches into two distinct heads that operate in parallel. The mean head predicts Voigt components through $\ell = 0 \oplus \ell = 2$ irreducible representations, directly outputting the tensor mean prediction. The covariance head outputs the symmetric tensor basis defined as $2 \times ( \ell = 0 ) \oplus 2 \times ( \ell = 2 ) \oplus 1 \times ( \ell = 4 )$ , which is then linearly projected to the Lie algebra element $A ( X ) \in \mathbb { R } _ { s y m } ^ { 6 \times 6 }$ . This dual-head architecture ensures that both mean and uncertainty predictions respect the underlying geometric symmetries.

Joint Training Stability. The Lie algebra parametrization combined with the LE-ESO loss provides inherent numerical stability that, in our experiments, enables end-to-end joint optimization of the mean and covariance heads without gradient detachment. To validate this robustness, we conducted an ablation study comparing joint training with gradient-detached training (where UQ gradients are blocked from flowing back to the backbone). Both approaches achieved comparable performance (MAE difference $< 0 . 0 2 )$ , with joint training showing marginally better uncertainty calibration. This empirica finding is consistent with the geometric advantage of the Log-Euclidean framework: by operating in the flat tangent space sym(6) rather than on the curved SPD manifold directly, gradients remain well-conditioned even when the covariance head receives informative error signals from the scoring objective. We did not observe variance collapse or shortcut learning in this setting.

## B.2. Implementation Details of the Equivariant Covariance Head

To strictly enforce the symmetry properties of the covariance tensor, we employ the CartesianTensor formalism from the e3nn library (Geiger & Smidt, 2022). The covariance of a symmetric rank-2 tensor is mathematically a rank-4 tensor $\mathcal { C } _ { i j k l }$ with specific permutation symmetries. First, the covariance exhibits symmetry of the first tensor argument such that $\mathcal { C } _ { i j k l } = \mathcal { C } _ { j i k l }$ . Second, it maintains symmetry of the second tensor argument with $\mathcal { C } _ { i j k l } = \mathcal { C } _ { i j l k }$ . Third, the covariance itself is symmetric, satisfying $\mathcal { C } _ { i j k l } = \mathcal { C } _ { k l i j }$

In our implementation, we define the output space using the formula $\ " \mathrm { \mathrm { ~ i ~ j ~ k ~ l = j ~ i ~ k ~ l = i ~ j ~ l ~ k = k ~ l ~ i ~ j ~ } } ^ { \prime }$ , which restricts the learnable basis to the subspace of R<sup>3×3×3×3</sup> satisfying these symmetries. The e3nn library automatically computes the change-of-basis matrix from the irreducible representations (irreps) of $S O ( 3 )$ to this symmetric Cartesian basis. The projection to the $6 \times 6$ Kelvin-Mandel matrix $A ( X )$ proceeds in two systematic steps.

First, in the irreps to Cartesian mapping, the features are mapped to the rank-4 Cartesian tensor $\mathcal { C } _ { i j k l }$ using the precomputed equivariant basis:

$$
\mathcal { C } _ { i j k l } = \sum _ { L , m } w _ { L , m } Y _ { i j k l } ^ { L } ,
$$

where $Y _ { i j k l } ^ { L }$ are the Clebsch-Gordan coefficients projecting the spherical harmonics onto the Cartesian tensor components. Second, in the Cartesian to Kelvin-Mandel transformation, the $3 \times 3 \times 3 \times 3$ tensor is flattened into a $6 \times 6$ matrix $A _ { K M }$ using the Kelvin-Mandel isometry. This mapping preserves the Frobenius norm $( \mathrm { i . e . , } \| \mathcal C \| _ { F } = \| A _ { K M } \| _ { F } )$ by scaling the off-diagonal shear components by ${ \sqrt { 2 } } .$ . For indices mapping $i j \to \alpha$ and $k l  \beta$ (where $\alpha , \beta \in \{ 1 . 6 \} )$ , the entry $A _ { \alpha \beta }$ is given by:

$$
A _ { \alpha \beta } = \eta _ { \alpha } \eta _ { \beta } \mathcal { C } _ { i j k l } ,
$$

where $\eta = \{ 1 , 1 , 1 , \sqrt { 2 } , \sqrt { 2 } , \sqrt { 2 } \}$ corresponds to the indices $\{ x x , y y , z z , y z , x z , x y \}$ . This construction guarantees that the predicted matrix $A ( X )$ strictly lies in the symmetric subspace sym(6) and transforms exactly according to $\rho _ { c } \otimes \rho _ { c }$

## B.3. Training Protocol and Stability Measures

All models were optimized using AdamW with hyperparameters $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ and a weight decay of $1 0 ^ { - 4 }$ . We employed a OneCycleLR scheduler with a peak learning rate of $1 0 ^ { - 3 } .$ , warming up for 20% of the total 50 epochs before gradually decaying. To ensure training stability during the critical early phases of covariance learning, we implemented several key strategies.

We introduced a critical loss annealing strategy where the auxiliary MSE warmup weight λ gradually decays from 0.9 to full LE-ESO optimization by epoch 5 (note that $\lambda _ { \mathrm { M S E } }$ is distinct from the LE-ESO weight α in Eq. 12). This gradual transition is essential for preventing early training instability - it allows the network to first learn reasonable mean predictions before tackling the more complex uncertainty quantification task. Furthermore, we applied eigenvalue clamping within $[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ] = [ - 4 , 3 ]$ to prevent numerical overflow in the matrix exponential computation. This constraint ensures that the resulting covariance eigenvalues remain in $[ e ^ { - 4 } , e ^ { 3 } ] \approx [ 0 . 0 1 8 , 2 0 . { \overset { \cdot } { 1 } } ]$ , preventing both variance collapse and explosion during early training.

Anisotropic Jitter as Numerical Gradient Stabilizer. A subtle numerical issue arises in automatic differentiation of eigenvalue decompositions: when the covariance matrix has degenerate eigenvalues $( \lambda _ { i } = \lambda _ { j } )$ , the Jacobian contains singular terms $( \lambda _ { i } - \lambda _ { j } ) ^ { - 1 }$ . This is particularly problematic for high-symmetry crystals $( \mathrm { e . g . }$ , cubic systems) where physical symmetry can cause eigenvalue degeneracy. To ensure differentiability, we introduce a numerical gradient stabilizer—a tiny anisotropic perturbation applied only during the spectral decomposition step:

$$
\tilde { \lambda } _ { i } = \lambda _ { i } + \epsilon \cdot i , \quad i = 0 , \ldots , 5 ,\tag{18}
$$

with $\epsilon \approx 1 0 ^ { - 6 }$ for float64 precision. This design is crucial: an isotropic shift $( \epsilon \cdot I )$ would preserve degeneracy and fail to resolve the singularity, whereas the anisotropic pattern guarantees $\lambda _ { i } - \lambda _ { j } \neq 0$ for all $i \neq j$ . Importantly, this jitter is not an architectural choice—it is a numerical safeguard with magnitude $O ( 1 0 ^ { - 6 } )$ that is negligible compared to typical eigenvalue scales $( \sim 1 )$ . Empirically, our equivariance verification (Table 4) shows errors on the order of $\bar { 1 } 0 ^ { - 7 }$ , confirming that this minimal perturbation does not compromise the geometric fidelity of the learned representations. The jitter operates entirely within the numerical solver and is invisible to the upstream equivariant architecture.

Discussion on Jitter and Calibration Impact. The introduced jitter $( \epsilon \approx 1 0 ^ { - 6 } )$ is several orders of magnitude smaller than the predicted eigenvalues $( \sim 1 . 0 ) $ . We observe that this perturbation is essential for maintaining stable gradients during joint training but has a negligible impact on both calibration (MACE $\mathrm { c h a n g e } < 1 0 ^ { - 4 } )$ ) and equivariance (errors remain at the level of $1 0 ^ { - 7 }$ as reported in Table 4).

Hyperparameter Details for Numerical Stability. We provide the specific hyperparameter values used in our implementation. For eigenvalue clamping, we use $[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ] = [ - 4 , 3 ]$ , which constrains the covariance eigenvalues to $[ e ^ { - 4 } , e ^ { 3 } ] \approx [ 0 . 0 1 \bar { 8 } , 2 0 . 1 ]$ . This range was chosen to prevent both variance collapse (eigenvalues $\ll 1 )$ and explosion (eigenvalues $\gg 1 )$ during early training. For the Huber robustification, the threshold is set to $\tau = 5 . 0$ , meaning that Mahalanobis distances above 5.0 transition to logarithmic scaling. This threshold was selected based on validation set analysis to be significantly above typical well-predicted samples $( D _ { M } \approx 2 \mathrm { - } 3 )$ while effectively capping the influence of extreme outliers.

Log-Euclidean Framework and Information Geometry. The space of SPD matrices ${ \mathcal { P } } _ { 6 }$ is not a vector space but a Riemannian manifold with non-Euclidean geometry. Direct optimization on this manifold introduces path-dependent gradients and numerical instabilities near the boundary. By working in the tangent space sym(6)—the Lie algebra o symmetric matrices—we obtain a flat Euclidean vector space where standard optimization is geometrically well-defined. The matrix exponential serves as the Riemannian exponential map, lifting points from the tangent space to the curved manifold while preserving the geometric structure.

Geometric Interpretation of α. The parameter α controls the tightness of the equivariant confidence hull, a process analogous to entropy regularization in information-theoretic learning. The term log det Σ represents the infinitesimal volume element of the uncertainty manifold in the Riemannian geometry of SPD matrices. By adjusting $\alpha ,$ we effectively control the trade-off between: (i) Information-theoretic volume: α log det $\Sigma = \alpha \operatorname { T r } ( A )$ penalizes excessive uncertainty spread; and (ii) Geometric fit: $D _ { M }$ measures the normalized prediction error in the metric induced by Σ. Theoretically, for the standard Multivariate Laplace distribution, $\alpha = 1$ (no $1 / 2$ coefficient as in the Gaussian case). However, we treat α as a tunable hyperparameter to balance model confidence with coverage: larger α encourages tighter confidence regions (lower uncertainty volume), while smaller α allows more conservative uncertainty estimates. This flexibility is valuable for materials science applications where the true noise level may vary across different datasets and measurement modalities.

Temperature Scaling for Calibration. To ensure the predicted covariance tensors Σ reflect the empirical error distribution, we apply post-hoc temperature scaling (Kuleshov et al., 2018). The optimal temperature $T \approx 0 . 0 5$ was determined on the validation set via a robust median-matching strategy. The small value of $T$ reflects the heavy-tailed nature of the initia residuals, requiring the model to significantly contract its uncertainty hulls after training with the robustified LE-ESO. This adjustment yields a calibrated covariance $\Sigma ^ { \prime } = T \cdot \Sigma$ , which is equivalent to an additive shift $A ^ { \prime } = A + \ln ( T ) I$ in the Lie algebra. This scaling effectively aligns the predictive distribution with the requirements of scoring rules $( \mathrm { e . g . }$ , Energy Score) without affecting the mean prediction or the exact $\operatorname { E } ( 3 )$ -equivariance.

Spectral Bounding for Manifold Consistency. To maintain numerical consistency with the Riemannian structure of ${ \mathcal { P } } _ { 6 } ,$ we constrain the Lie algebra eigenvalues to a bounded interval before computing $\exp ( \Lambda )$ . This spectral bounding ensures that the resulting covariance eigenvalues remain in a geometrically valid range, preventing both variance collapse (near-zero eigenvalues) and explosion (excessively large eigenvalues) during early training when predictions may be far from the data manifold. The bounds $[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ are chosen to map to a physically meaningful covariance spectrum $[ e ^ { \lambda _ { \mathrm { m i n } } } , e ^ { \lambda _ { \mathrm { m a x } } } ]$ under the matrix exponential.

Laplacian-Huber Compound Robust Loss. To handle extreme outliers in material property data, we introduce a Laplacian-Huber scheme with two regimes: when the Mahalanobis distance $D _ { M }$ is below threshold $\tau = 5 . 0$ , we apply a linear penalty (the Laplace distribution core); when $D _ { M }$ exceeds τ, we switch to a logarithmic penalty $( \tau + \log ( 1 +$ $D _ { M } - \tau ) )$ that compresses the residual contribution for very large $D _ { M }$ . This design preserves the statistical interpretation of the Multivariate Laplace distribution for normal samples while limiting the influence of extreme outliers, so that the residual term grows logarithmically instead of linearly in the tail.

Gradient Analysis: Practical Control on the Lie Algebra. We do not claim that the Lie algebra parameterization by itself yields a uniform gradient bound near the SPD-cone boundary; rather, in practice, gradient and matrix-exponential overflow are controlled jointly by eigenvalue clamping and the log-tail robustification. Let $\mathbf { z } = Q ^ { \top } ( \mathbf { c } _ { \mathrm { t r u e } } - \mu )$ be the rotated residuals in the eigenbasis. The loss gradient with respect to eigenvalues $\boldsymbol { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { 6 } )$ is:

$$
\frac { \partial \mathcal { L } _ { \mathrm { L E - E S O } } } { \partial \Lambda } = \alpha I - \frac { \partial \tilde { D } _ { M } } { \partial \Lambda } .\tag{19}
$$

As derived in Proposition $\mathrm { A . 5 } ,$ the Laplace residual term carries an extra $1 / ( 2 D _ { M } )$ factor relative to the Gaussian case, which reduces the growth rate of $\partial D _ { M } / \partial \lambda _ { k }$ but does not by itself produce a uniform bound: when a single direction dominates $D _ { M }$ , the contribution to $\partial D _ { M } / \partial \lambda _ { k }$ still grows as $| z _ { k } | \exp ( - \lambda _ { k } / 2 ) / 2$ as $\lambda _ { k } \to - \infty$ . We therefore enforce the constraint $\lambda _ { k } \in [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ before the matrix exponential, which prevents $\exp ( - \lambda _ { k } )$ from diverging and keeps $\partial D _ { M } / \partial \lambda _ { k }$ finite over the optimization trajectory. The log-tail region $( D _ { M } \ge \tau )$ further compresses the residual gradient: $\partial \tilde { D } _ { M } / \partial \lambda _ { k }$ approaches a finite constant $(  - 1 / 2$ in a dominant direction) instead of growing with $z _ { k } ^ { 2 } \exp ( - \lambda _ { k } )$ , mitigating the influence of rare extreme outliers. Empirically, this combination keeps training stable enough to support end-to-end joint optimization without gradient detachment. The loss maintains $O ( 3 )$ )-invariance since both the trace and matrix exponential preserve equivariance under orthogonal transformations.

The numerically stable loss in Eq. 12 follows directly from algebraic identities proven in Proposition A.5. Importantly, both the trace term $\operatorname { T r } ( A )$ and the Mahalanobis distance $D _ { M } = \sqrt { \Delta \mathbf { c } ^ { \top } \exp ( - A ) \Delta \mathbf { c } }$ are invariant under any orthogonal

transformation $\rho _ { c } ( R )$

$$
\mathrm { T r } ( \rho _ { c } ( R ) A \rho _ { c } ( R ) ^ { \top } ) = \mathrm { T r } ( A ) , \qquad D _ { M } ( \rho _ { c } ( R ) \Delta \mathbf { c } , \rho _ { c } ( R ) A \rho _ { c } ( R ) ^ { \top } ) = D _ { M } ( \Delta \mathbf { c } , A ) .
$$

Consequently, the loss function provides an exact symmetry-preserving training objective, in contrast to approximate equivariant regularizations or data augmentation-based approaches.

Training was conducted on a single NVIDIA RTX 4060 Ti with batch sizes of 32 for ModelNet40 and 16 for the Materials Project, requiring approximately 10 hours for complete convergence. Additional implementation details include processing atomic structures into graphs with a 5.0A cutoff distance and using Magpie feature embeddings of dimension 119 for atomic<sup>˚</sup> number representations.

## C. Experimental Setup and Analysis

## C.1. Dataset Configuration and Preprocessing

We evaluate our framework on two distinct datasets that provide complementary validation of our equivariant uncertainty quantification approach. ModelNet40 serves for geometric validation with physically defined tensor properties, while the Materials Project provides a real-world materials science application with experimentally relevant predictions.

ModelNet40. The dataset comprises 12,311 CAD models across 40 categories. We adhere to the official split, utilizing 9,843 models for training and 2,468 for testing. To simulate measurement uncertainty and validate our probabilistic framework, we sample $N = 2 0 4 8$ points uniformly from mesh surfaces and apply Gaussian jitter with $\sigma _ { n o i s e } = 0 . 0 1$ . This noise injection creates the aleatoric uncertainty necessary for testing our framework’s ability to capture geometric ambiguity arising from point cloud sampling.

Materials Project Dielectric Dataset. We source precomputed dielectric tensor predictions from the Materials Project database (Barroso-Luque et al., 2024; Jain et al., 2013). To ensure data quality and consistency, we apply systematic filtering criteria: (1) structure size—we exclude crystals with fewer than 3 atoms or more than 30 atoms to balance computational efficiency and representation learning; (2) positive-definiteness—we verify that all dielectric tensors have eigenvalues strictly greater than $1 0 ^ { - 4 } ,$ excluding numerically singular matrices; (3) value range—we remove samples with dielectric constants outside [−10, 50] or with diagonal entries below 1.0. After filtering, the dataset comprises 5,002 crystalline structures, partitioned into 4,236 for training, 485 for validation, and 281 for testing. We apply Matrix Log-Normalization with parameters $\mu _ { l o g } = 1 . 2 4$ and $\sigma _ { l o g } = 0 . 8 6$ to handle the wide dynamic range while preserving the relationships between different crystal structures.

## C.2. Equivariance Ablation Study Design

To systematically isolate the contributions of equivariance and SPD constraints, we designed four baseline variants tha progressively incorporate different architectural components. Baseline A employs a standard non-equivariant GNN with a Cholesky covariance head to test the necessity of equivariant message passing. Baseline B upgrades the backbone to an equivariant neural network (ENN), but keeps coordinate-wise scalar MLP outputs for both the mean and Cholesky covariance heads, evaluating whether equivariant features alone are sufficient to guarantee equivariant tensor outputs. Baseline B<sup>′</sup> further replaces the scalar mean head with the same equivariant mean construction used in our model while retaining the Cholesky covariance head. This baseline isolates the covariance-parameterization failure mode: the mean branch is equivariant by construction, whereas the Cholesky covariance is SPD but not equivariant under the Kelvin–Mandel covariance representation. Baseline C uses an ENN backbone with direct equivariant regression of the symmetric operator A(X) but omits the matrix exponential, testing the importance of the SPD projection. Finally, our full method combines the ENN backbone with the matrix-exponential covariance head to simultaneously guarantee covariance equivariance and SPD validity.

## D. Additional Experimental Results

This appendix collects supplementary experiments that complement the two main experiments in the body of the paper. They are intended as supporting evidence for the scope and robustness of the proposed equivariant SPD/UQ construction rather than as a comprehensive benchmarking study.

## D.1. ModelNet40 Shape-Covariance Validation

The shape-covariance experiment is a controlled geometric validation benchmark on the ModelNet40 dataset (Wu et al., 2015). Like inertia tensor prediction, the target admits a closed-form estimator from the point cloud. We therefore do not present this task as a real-world setting where neural prediction is necessary. Instead, it tests whether the proposed equivariant SPD/UQ construction remains valid on a second symmetric rank-2 tensor target beyond inertia, demonstrating that the framework is not specific to the inertia formulation.

Table 6. ModelNet40 shape-covariance validation. The goal is controlled geometric validation rather than replacing the closed-form estimator. “Mean Tensor PSD Rate” refers to the fraction of predicted mean shape-covariance tensors that are positive semi-definite, distinct from the SPD validity of the predictive covariance Σ(X).
<table><tr><td>METHOD</td><td>MAE↓</td><td>RMSE↓</td><td>MEAN TENSOR PSD RATE</td></tr><tr><td>DETERMINISTIC (MSE)</td><td>0.0333</td><td>0.0616</td><td>99.2%</td></tr><tr><td>FULL UQ (OURS)</td><td>0.0346</td><td>0.0633</td><td>99.5%</td></tr></table>

The point-prediction MAE/RMSE of the UQ model is comparable to the deterministic baseline, while the predictive covariance $\Sigma ( X )$ is exactly E(3)-equivariant and SPD by construction. As in the inertia setting, we observe near-machine precision equivariance (errors on the order of $1 0 ^ { - 7 } )$ and strict SPD validity for the predictive covariance.

## D.2. Rank-4 Elasticity Tensor Prediction

We evaluate the framework on a real-data elasticity tensor prediction task from the Materials Project (Jain et al., 2013). Unlike the rank-2 dielectric setting, the mean target here is directly a rank-4 elasticity tensor. Under the standard minor and major symmetries, the elasticity tensor has 21 independent components. This experiment is intended as supporting evidence that the proposed structured equivariant SPD/UQ construction can be extended beyond the six-dimensional symmetric rank-2 setting; it is not intended as a comprehensive study of all higher-order tensor parameterizations.

On this benchmark, the model achieves a test MAE of approximately 5.0 GPa, which is essentially on par with a deterministic baseline and noticeably better than a naive UQ baseline. The structured UQ model also improves uncertainty quality over the naive baseline: empirical coverage rises from approximately 35% to 52%, and the uncertainty–error correlation rises from approximately −0.15 to 0.31. At the same time, both numerical equivariance/prediction consistency and predictive covariance SPD validity remain at 100%, matching the structural behavior observed in the rank-2 dielectric setting. We emphasize that this single experiment is supporting evidence that the proposed structured equivariant SPD/UQ construction remains feasible on a higher-order tensor target, rather than an exhaustive higher-order benchmark.

## D.3. Computational Overhead

We profile per-batch wall-clock time on the Materials Project dielectric task using a single NVIDIA RTX 4060 Ti, with batch size 16 and identical input pipelines.

Table 7. Runtime profiling on Materials Project (RTX 4060 Ti, batch size 16). The full-covariance model is more expensive primarily because of the covariance branch and its backpropagation, rather than the matrix exponential alone.
<table><tr><td>MODEL</td><td>TIME / BATCH</td><td>RELATIVE COST</td></tr><tr><td>DETERMINISTIC</td><td>130 MS</td><td>1.00×</td></tr><tr><td>DIAGONAL UQ</td><td>132 MS</td><td>1.015×</td></tr><tr><td>FULL COVARIANCE UQ</td><td>570 MS</td><td>4.4×</td></tr></table>

The diagonal-UQ overhead is negligible (1.5%), confirming that anisotropy modeling, not uncertainty quantification per se, dominates cost. For inference, a single forward pass yields the full anisotropic covariance, in contrast to ensemble methods that require N forward passes.

## D.4. Sensitivity to the LE-ESO Weight

The weight α controls the trade-off between the log-volume term $\mathrm { T r } ( A ) = \log \mathrm { d e t } \Sigma$ and the geometric data-fit term in LE-ESO. We use $\alpha = 1$ in the main experiments because it corresponds to the canonical coefficient in the multivariate

Table 8. Sensitivity to the LE-ESO weight α on the Materials Project dielectric task. MAE is measured in log-Kelvin–Mandel space and is comparable across rows. The LE-ESO value is evaluated with the same α used for training, so it reflects the optimized objective for each setting rather than a fixed cross-α NLL.
<table><tr><td>α</td><td>BEST VAL MAE↓</td><td>BEST VAL LE-ESO ↓</td></tr><tr><td>0.03</td><td>0.3634</td><td>0.7911</td></tr><tr><td>0.10</td><td>0.4176</td><td>0.7165</td></tr><tr><td>0.30</td><td>0.4566</td><td>0.3469</td></tr><tr><td>1.00</td><td>0.3519</td><td>-2.9393</td></tr></table>

Laplace objective motivating LE-ESO.

To evaluate sensitivity, we run a short validation sweep over $\alpha \in \{ 0 . 0 3 , 0 . 1 0 , 0 . 3 0 , 1 . 0 0 \}$ on the Materials Project dielectric task. Table 8 reports the best validation MAE in log-Kelvin–Mandel space. Across the tested values, the validation MAE remains in a moderate range (0.352–0.457), indicating that performance is not tied to a narrow value of α. The canonical choice $\alpha = 1$ also gives the lowest validation MAE in this sweep.

We also report the best validation LE-ESO value for completeness. Importantly, this value is evaluated using the same α as the corresponding training run, and therefore should be interpreted as the optimized objective for that setting rather than as a fixed cross-α negative log-likelihood. Since changing α changes the scoring objective itself, these LE-ESO values are not directly comparable as absolute NLL values across different α.

## D.5. Additional Risk-Coverage Analysis

To complement the risk-coverage discussion in the main paper, we report the full retained-set comparison between $\lambda _ { \operatorname* { m a x } } ( \Sigma )$ ranking, Trace(Σ) ranking, and a diagonal-UQ baseline that ignores off-diagonal correlations. At 90% coverage, ranking by $\lambda _ { \mathrm { m a x } }$ improves retained-set MAE by 3.1% relative to the full test set. The improvement of $\lambda _ { \mathrm { m a x } }$ over Trace under the same retained-set protocol is approximately 1.5%—smaller than the headline 3.1% number but consistent across coverage levels. At 80% coverage, $\lambda _ { \mathrm { m a x } }$ continues to retain a positive improvement, while Trace-based ranking can fall slightly below the full-dataset baseline. The diagonal-UQ baseline, which lacks off-diagonal covariance information, ranks the test set less informatively than either $\lambda _ { \mathrm { m a x } }$ or Trace from the full-covariance model. These results support the interpretation that directional uncertainty captures failure modes that scalar total uncertainty partially obscures, while clarifying that the practical advantage over Trace is moderate rather than dramatic.

## E. Additional Results and Analysis

## E.1. Training Dynamics and Loss Analysis

Figure 7 illustrates the complete training dynamics of our equivariant uncertainty framework. To ensure a stable optimization landscape, we employ a two-stage curriculum: the model is initially warmed up with a combined MSE-LE-ESO objective for 5 epochs to establish a reliable mean prediction baseline before transitioning to heavy-tailed LE-ESO optimization.

Panel (a) reveals that the loss stabilizes rapidly upon transition, with no numerical spikes despite the non-linear nature of the matrix exponential map. Panel (b) demonstrates that the addition of the uncertainty branch does not compromise the underlying point-prediction accuracy; instead, the MAE for both diagonal $( \varepsilon _ { i i } )$ and off-diagonal $( \varepsilon _ { i j } )$ components plateaus at a state-of-the-art level, benefiting from the robust regularization provided by the UQ branch.

Most importantly, panel (c) highlights the sophisticated trade-off mechanism inherent in our loss formulation. As the validation epoch progresses, the network balances the data fit term (Mahalanobis distance) against the uncertainty regularization term (log det Σ). The joint optimization allows both branches to benefit from shared geometric representations, preventing “shortcut learning” where the model might collapse its uncertainty to minimize the scoring rule. The eventual convergence of the Mahalanobis distance toward a steady value confirms that the model has effectively learned to characterize the aleatoric noise in the dielectric property space.

![](images/278f8161f6685172e453e901f940c0777b167968f3aaa35c33a83ed1ed128212.jpg)

![](images/ee034fb7096dd15469b217022aefb4c42170d2880b64bfb41c8cc745e7c30d04.jpg)

![](images/262d7fe024c07a5697af7f4d867f278520be2e3085513da88993cb3d1d80822b.jpg)  
Figure 7. Training dynamics. Two-stage optimization: warmup (5 epochs) then LE-ESO. (a) LE-ESO convergence. (b) MAE stability for diagonal/off-diagonal components. (c) Balance between data fit $( \dot { \mathbb { E } } [ D _ { M } ] )$ and regularization (log det Σ).

## E.2. Empirical Verification of Theoretical Guarantees

To validate the theoretical guarantees established in Appendix A, we performed rigorous numerical checks throughout training that confirm both the mathematical correctness and practical stability of our implementation.

The numerical stability of our approach stems from the eigenvalue decomposition $A = Q \Lambda Q ^ { \top }$ used to compute the loss without explicitly forming $\Sigma = \exp ( A )$ . Since A is symmetric, all eigenvalues $\lambda _ { i }$ are real and $\exp ( \lambda _ { i } )$ remains positive. In practice, eigenvalue clamping keeps these exponentials bounded, preventing numerical overflow and improving gradient conditioning. Together with the log-tail robustification, this enables joint end-to-end training without explicit regularization on the covariance spectrum, addressing a critical limitation of direct covariance optimization approaches.

For equivariance verification, we continuously monitored the relative Frobenius-norm difference between rotated predictions and transformed predictions:

$$
E _ { \mathrm { e q u i v } } = \frac { \| \Sigma ( R \cdot X ) - \rho _ { c } ( R ) \Sigma ( X ) \rho _ { c } ( R ) ^ { \top } \| _ { F } } { \| \Sigma ( X ) \| _ { F } } .
$$

Across all random rotations tested during training, this error consistently remained at the level of $1 0 ^ { - 7 }$ , confirming that our implementation achieves near-machine-precision equivariance rather than approximate symmetry preservation.

For SPD validation, we monitored the spectrum of predicted covariance matrices throughout training. The minimum eigenvalue of $\Sigma ( X )$ across all batches remained strictly positive $( > 1 0 ^ { - 5 } )$ , with no numerical violations of the SPD constraint observed (see Figure 8a).

Beyond the geometric validation on ModelNet40, we further analyzed the conditioning of the predicted covariances for the dielectric tensor task (Materials Project). As shown in Figure 8b, the distribution of condition numbers $\kappa ( \Sigma )$ remains numerically well-conditioned for the final model, with a mean of 3.80 and a maximum of 16.4.

This result is particularly significant because, unlike the synthetic jitter in ModelNet40, the uncertainty in dielectric tensors arises from complex physical and DFT approximation errors. The low condition numbers indicate that our matrix exponential mapping naturally induces numerically stable, non-degenerate uncertainty estimates without requiring auxiliary regularization terms (e.g., hinge loss penalties on eigenvalues). This confirms that the optimization landscape remains well-behaved even for high-dimensional material representations.

## E.3. Reflection Symmetry and Chirality Handling

Our framework explicitly accounts for improper rotations (reflections) by ensuring that the representation $\rho _ { c }$ correctly tracks the parity of the tensorial outputs. For the symmetric rank-2 tensors considered here—such as dielectric or inertia tensors—the physical quantities are even tensors under parity, meaning they are invariant to inversion. The Kelvin-Mandel representation $\rho _ { c } ( R )$ used throughout this paper is defined by projecting $R \otimes R$ onto the symmetric subspace, as given in $\operatorname { E q } .$ . 14 of Appendix A.1; we use that construction directly here rather than introducing a separate definition. Since this transformation is built from two factors of $R ,$ the determinant contribution (det $R ) ^ { 2 } = 1$ ensures that the framework handles chiral structures and their mirror images with consistent physical semantics. We numerically verified full $O ( 3 )$ equivariance by testing improper rotations, achieving errors at the level of $1 0 ^ { - 7 }$ consistent with the SO(3) results reported in Table 4. Consequently, our uncertainty quantification remains valid regardless of the handedness of the coordinate system, a critical requirement for modeling both chiral and achiral materials.

## E.4. Spectral Analysis and Sharpness Distribution

To further investigate the UQ quality, we provide detailed spectral analysis in Figure 8. The eigenvalue distribution (Panel a) confirms that all predicted covariances maintain strict positive-definiteness with a minimum eigenvalue $\lambda _ { \operatorname* { m i n } } \approx 0 . 4 4 9$ , safely avoiding variance collapse. The condition number distribution in Figure 8b shows that the predicted covariance matrices remain numerically well-conditioned, consistent with the verification in Appendix E.2. Complementary to the risk-coverage analysis in Section 4.4, the sharpness distribution in Figure 5a reveals that the model effectively differentiates between “simple” and “complex” atomic environments by assigning confidence volumes spanning several orders of magnitude.

![](images/f5cc8d8503a880fd21780ffa67f3c35bfc23877d84a6ffc6ae89ad04179bdfce.jpg)  
(a) Spectrum Validity

![](images/7328ec3ca4e6dda3fa6b064a87fcdfe20edb10ac969a98c469f86e7576afb98e.jpg)  
(b) Conditioning  
Figure 8. Numerical stability analysis. (a) Positive eigenvalues ensure SPD validity. (b) Moderate condition numbers indicate numerical stability.

## E.5. ModelNet40 SPD Analysis

The 3D uncertainty visualization in Figure 3 demonstrates that our framework produces physically meaningful uncertainty estimates where uncertainty ellipsoids align with principal shape axes (demonstrating E(3)-equivariance), expand in regions with sparse point density (capturing sampling ambiguity), and preserve tensorial correlations across components.

We verify physical consistency through systematic validation of SPD properties (Figure 9). Our predictions maintain strict SPD requirements (>99.9% validity) with well-conditioned covariance structures (median condition number 6.8), contrasting sharply with unconstrained baselines that frequently violate physical constraints.

## E.6. Limitations and Future Work

The SPD construction and scoring objective are representation-agnostic once an equivariant symmetric operator $A ( X )$ is available. However, extending the full parameterization to higher-order tensor predictions requires group- and representationspecific basis construction. Our main implementation and empirical validation focus on symmetric rank-2 tensors, with the rank-4 elasticity experiment in Appendix D.2 serving as preliminary supporting evidence. Extending to fourth-order tensors (e.g., elasticity tensors) and beyond introduces two key computational challenges. First, the tensor basis construction scales as $O ( d ^ { \ell } )$ where ℓ is the tensor rank, making the basis enumeration for rank-4 and higher tensors substantially more expensive. Second, the covariance matrix dimension grows combinatorially—for a rank-k symmetric tensor in 3D, the Kelvin-Mandel representation has dimension $( k + 1 ) ( k + 2 ) / 2$ , leading to covariance matrices of size $O ( k ^ { 4 } )$ . This scaling necessitates careful memory management and may require approximations such as low-rank covariance factorization or hierarchical uncertainty modeling. Future work should explore more efficient equivariant basis constructions for higher-order tensors. In particular, integrating path-matrix based ICT decompositions (Shao et al., 2025) could significantly reduce the overhead of basis enumeration for rank-4 and higher tensors, enabling the extension of our uncertainty framework to complex properties like the full elasticity tensor.

![](images/fee0abff3886c375e46365293005439445f9d122b4a5f56c8cd499136ea7ccbc.jpg)

![](images/fe9d7537a05ec60de3959edc490269e954c3ef6b5ec167357f6e89277276f1c8.jpg)

![](images/eb675f6b3c47a9087fb1676e557c6413dfbbc5f2bdc241097e41757af0bbf6b5.jpg)  
Figure 9. SPD validity on inertia-tensor task. (a) Minimum eigenvalue distribution (all positive). (b) $\log _ { 1 0 }$ condition numbers (well-conditioned). (c) Total uncertainty Tr(Σ).

Modularity and Backbone Extensibility. A key strength of our framework is its modular design: the matrix-exponential UQ head is completely backbone-agnostic and can be integrated with any E(3)-equivariant architecture. While this study utilizes a standard message-passing backbone to validate the UQ mechanism, future work will explore pairing our UQ head with higher-accuracy architectures such as GoeCTP (Hua et al., 2026) to combine state-of-the-art point prediction with calibrated, symmetry-preserving uncertainty estimates. This plug-and-play capability allows practitioners to add rigorous uncertainty quantification to existing equivariant models without architectural reengineering.