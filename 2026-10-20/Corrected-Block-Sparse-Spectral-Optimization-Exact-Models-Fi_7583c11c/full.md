# Corrected Block Sparse Spectral Optimization: Exact Models, Finite Identification, and Local Dynamics<sup>1</sup>

Omid Shams Solari<sup>1</sup>

<sup>1</sup>sAIm Labs

## ABSTRACT

The block generalized-power method pairs threshold-based support selection with inexpensive Stiefel updates, but the standard block formulation conflates two nonequivalent objectives and leaves inactive components, threshold ties, and post-selection refitting ill-specified. We rebuild the model from the primal. Eliminating each loading column from the normalized loading-ball problem yields a sum of soft-thresholded Euclidean norms (A-SUM); the squared-threshold objective in common use arises instead from a quadratic lift (A-SQ). We prove both identities, exhibit a two-component instance whose maximizers disagree, and specify explicit conventions for zero-response columns, threshold ties, rank-deficient gradient matrices, and the mask-before-refi inconsistency. For A-SQ we separate the historical full-polar iteration from a safeguarded Riemannian ascent and prove, for the latter, whole-sequence convergence through a suficient-ascent lemma, a relative-error lemma, and the Kurdyka– Lojasiewicz property, which applies here because the soft-threshold objective and the Stiefe constraint are both semi-algebraic. Once the iterates converge, a strict threshold margin forces the support and sign pattern to stabilize in finitely many steps. The post-identification map is ordinary power iteration on the active covariance in the rank-one hard-threshold case; the block case with component-specific supports requires the Fr´echet derivative of a coupled polar map rather than a single pooled operator, which we derive explicitly. A radius-based safe-screening rule follows from the same local analysis. Planted-spike experiments match the predicted identification time and local rates, including a two-branch validation that cleanly separates ordinary power contraction from the shifted contraction the soft-threshold gradient introduces. Modern-AI panels push the same estimator onto a GPT-2 small residual stream, a ViT CLS stream, and the Pitprops benchmark; the thresholded block-sparse components match dense PCA on reconstruction, improve on it in monosemanticity and class-conditional specialisation, and avoid the reconstruction cost that a tied-decoder sparse autoencoder of matching width pays for its ℓ<sub>1</sub> sparsity. Four real-world matrices add weight: the Fama–French 48 industry daily returns (sparse loadings that align with GICS sectors and sit on the dense-PCA Pareto frontier); three single-cell RNA-seq datasets with native cell-type labels, namely Zheng17 bead-purified FACS-sorted PBMCs, the fresh 68k donor-A PBMC matrix with FACS-correlated per-barcode labels, and Tabula Muris FACS (Marrow, Spleen, Thymus) with native mouse cell ontology class, on all three of which A-SQ beats dense PCA on cluster-recovery AMI at matched sparsity (margins +0.023 to +0.044); and a multi-turn GPT-2 tool-use conversation stream in which the components resolve into phase-selective activations that the matched PCA components do not expose. Finally, routing a certified safe-screen mask through a dedicated screened operato agrees with the unscreened solve to machine precision while attaining the full (n−k)/n oracle-work ceiling

Keywords FACET, sparse PCA, Stiefel manifold, support identification, K L convergence, local spectral dynamics Emails omid.solari@saimai.com,solari@berkeley.edu Updated October 2026 Version v0.5

## CONTENTS

## 1 Introduction

2 Related Work 6   
2.1 Sparse PCA and multi-component estimation 6   
2.2 Convex maximization and generalized Frank–Wolfe 6   
2.3 Sparse optimization on matrix manifolds 6   
2.4 Sparse neural representations   
3 Corrected Problem Formulation   
3.1 A-SUM: normalized loading-ball model   
3.2 A-SQ: squared-threshold model   
3.3 The objectives are not equivalent 8   
3.4 Numerical conventions   
4 Algorithms and Refitting 8   
4.1 Historical full-polar iteration 8   
4.2 FACET safeguarded iteration 9   
4.3 Support-conditioned refitting 10   
4.4 Complexity 10   
5 Theory 10   
5.1 Monotonicity of the full-polar step 11   
5.2 Whole-sequence convergence of the safeguarded method 12   
5.3 Finite active-set identification 12   
5.4 Post-identification dynamics 13   
5.5 Locally safe screening 13   
5.6 Second-order scope 13   
6 Proofs and Extended Analysis 13   
6.1 Exact value functions and recovery maps . 13   
6.2 Diferentiability and active-region smoothness 14   
6.3 Safeguarded ascent and whole-sequence convergence 15   
6.4 Identification with explicit margins 16   
6.5 Common-support block rate . 17   
6.6 Heterogeneous supports and the polar Jacobian 17   
6.7 Second-order statements with explicit justification 17   
Algorithmic Variants and Fully Proved Special Cases 17   
7.1 A-SUM alternating method . 18   
7.2 A complete rank-one convergence theorem 18   
7.3 Fixed-support second-order classification 18   
7.4 Complexity accounting 18   
8 Experiments 19   
8.1 Controlled sparse spectral models 19   
8.2 Controlled neural bridge 19   
8.3 Transformer activations 19   
8.4 Ablations . 19   
9 Results 20   
9.1 Real-world domain panels 20   
9.2 Discussion 21   
10 Limitations 21   
11 Conclusion 22   
A Additional Derivations and Proof Obligations 38   
A.1 Gradient of the squared-threshold objective 38   
A.2 Riemannian gradient and tangent projection 38   
A.3 Polar derivative obligation 38   
A.4 Counterexample suite 38   
A.5 Proof checklist . 38   
A.6 Reproducibility checklist . 38   
B Detailed Proofs, Counterexamples, and Numerical Contracts 39   
B.1 Counterexample: normalize then mask 39   
B.2 Counterexample: threshold ties 39   
B.3 Polar diferential . 39   
B.4 Proof contract for the K L theorem 39   
C Whole-Sequence Proof under Explicit Algorithmic Assumptions 39   
D Derivation of the Polar Diferential 40   
E Support-Specific Hessian Calculations 40   
F Extended Counterexample Catalogue 41

## CONTENTS

## 1 Introduction

Sparse spectral methods look for low-dimensional directions that explain the data while depending on few variables. The generalized power method of Journ´ee et al. (2010) is influential because it recasts high-dimensional sparse-loading problems as thresholded convex objectives on a sphere or a Stiefel manifold, with a particularly simple linear-maximization oracle: normalization on the sphere, the polar factor on the Stiefel. Each iteration is inexpensive, and support discovery is cleanly separated from support-conditioned refitting. The same thresholded block-sparse structure already drives the paired cross-covariance setting in our prior BLOCCS block-sparse CCA estimator (Solari et al., 2019b) and in the concave-minimization sparse-CCA formulation (Solari et al., 2019a); this paper is the sparse-PCA / generalized-power counterpart, developed inside the broader multi-view sparse-spectral framework of Solari (2019).

The multi-component version carries three mathematical ambiguities that the implementation then inherits. First, eliminating each loading column from the printed linear-minus-ℓ primal yields a sum of thresholded norms, while the paper’s displayed block objective sums their squares. The two agree in the singlecomponent case only because a nonnegative scalar shares its maximizers with its square; after summing several components they are diferent estimators. Second, the printed unit-sphere loading domain excludes zero loading columns, yet the surrounding reductions and prose silently allow inactive components. Third, the published post-processing normalizes a loading column before applying its support mask, which breaks unit norm the moment the mask has any zeros. Taken together, these three issues alter the estimator, the feasible set, the fixed points, and the validity of every convergence argument built on top.

We treat the audit as the starting point rather than the conclusion. Two coherent block estimators replace the ambiguous one. A-SUM is the exact score-space reduction of the normalized loading-ball primal, with objective a sum of thresholded norms. A-SQ sums squared thresholded norms and arises from a distinct quadratic-lift primal. For A-SQ we retain the historical full-polar iteration on the record (and credit it only with what a convex minorization can prove), and we introduce a safeguarded Riemannian ascent: an ascent direction, a retraction or polar candidate, an Armijo test, and an explicit fallback. The safeguarded method is what the whole-sequence analysis is built on.

The dynamics have two phases. Before identification the iterates cross threshold boundaries, so the support and sign pattern vary. Under a strict margin, convergence restricts these jumps to finitely many. After identification the objective is smooth on a fixed active region and the iteration reduces to a spectral map. For a rank-one hard threshold that map is ordinary power iteration on the active covariance; for a block with a single shared active operator it is orthogonal iteration. With component-specific supports neither picture applies, and the correct object is the Fr´echet derivative of a coupled polar map on the Stiefel tangent space. We make this distinction explicit so that the paper does not implicitly claim that every block instance reduces to a single-covariance eigenproblem.

The modern-AI setting is a stress test for the mathematics, not a replacement for it. Transformer activation matrices are wide and structured, and current practice analyses them with sparse autoencoders or native-neuron selection. We ask three concrete questions: do measurable threshold margins predict support stabilization; do active eigengaps or Jacobian radii predict the observed local slopes; and do the resulting sparse components carry reproducible causal efects. Reconstruction alone does not count as evidence of interpretability. The Section 9 panels report top-token coherence against GPT-2 input embeddings (an SAEBench-style quality diagnostic (Karvonen et al., 2025)), multi-probe ablation KL against a matched-norm random-direction null (an MIB-style causal diagnostic (Mueller et al., 2025; O’Neill et al., 2025)), and a head-to-head comparison against a gated-SAE-style tied-decoder dictionary of matching width (Rajamanoharan et al., 2024).

## Contributions.

1. We derive the exact loading-ball reduction and the exact quadratic-lift representation, and exhibit a two-component counterexample proving that their block aggregations define diferent estimators.

2. We give complete numerical conventions for inactive components, threshold ties, rank-deficient gradient matrices, nonunique polar factors, component permutations, and support-conditioned refitting.

![](images/84bd7f8a708ae782c0f17e43a65bfea94df7121babb9ab821594c8e1545f33e2.jpg)  
Figure 1. Printed-vs-corrected model map. The left column lists four defects of the seed-paper block formulation; the right column states the FACET correction that each defect motivates. The downstream theorems A1–A10 depend on all four corrections being in force simultaneously.

![](images/d101c8f1bfaad62bb2fbd77c791a16c7a6ac965751bcae885cd6203694367888.jpg)  
Figure 2. The corrected model family. A-SUM and A-SQ have diferent exact primal interpretations and are not silently identified.

3. We separate the historical full-polar algorithm from a safeguarded Riemannian ascent, establish monotonicity of the full-polar step, and prove last-iterate convergence of the safeguarded method via a suficient-ascent lemma, a relative-error lemma, and the Kurdyka– Lojasiewicz property, which applies because the soft-threshold objective and the algebraic Stiefel constraint are semi-algebraic.

4. We prove conditional finite identification under strict margins, derive the rank-one fixed-support power rate (validated on a controlled fixture to within 5% of the classical $\lambda _ { 2 } / \lambda _ { 1 }$ prediction, and to within 1% of a corrected prediction that includes the soft-threshold shift $\gamma \sigma _ { 1 } \sqrt { k } )$ , formulate the correct block tangent-Jacobian condition, and obtain a locally safe screening rule from certified distance radii; the screening rule is realized at the solver level by a dedicated screened operator and attains the full $( n - k ) / n$ oracle-work ceiling with machine-precision objective agreement against the unscreened solve.

5. We validate the theory on planted spikes and extend the pipeline to a GPT-2 small residual stream, a vision-transformer CLS stream, and the Jefers Pitprops classical sparse-PCA benchmark (Jefers, 1967); FACET components match dense PCA on reconstruction $R ^ { 2 }$ while exceeding it on Hoyer sparsity, top-token coherence, and class-conditional specialisation, and avoid the ∼8-point $R ^ { 2 }$ cost that a tied-decoder sparse autoencoder incurs for its $\ell _ { 1 } \cdot$ -driven sparsity at matched dictionary width.

Organization. Section 2 positions the work. Section 3 fixes the models. Section 4 states the algorithms and the refit. Section 5 lays out the result hierarchy and Section 6 carries the whole-sequence proofs. Sections 8 and 9 report the controlled and modern-AI panels.

## 2 Related Work

## 2.1 Sparse PCA and multi-component estimation

Regression formulations, semidefinite relaxations, penalized matrix decompositions, Fantope methods, and truncated power algorithms occupy distinct trade-ofs between statistical guarantees, exact sparsity, computation, and component coupling (Zou et al., 2006; d’Aspremont et al., 2007, 2008; Witten et al., 2009; Vu et al., 2013; Yuan & Zhang, 2013). Recent work tightens polynomial-time sample complexity in spiked models and strengthens exact or certifiable optimization (Cai et al., 2025; Bertsimas et al., 2022). The contribution of the present paper is not a new sparsity penalty. It is a correction of the block value function, a precise separation of two estimators that the literature has conflated, and an analysis of the discrete-to-smooth dynamics that thresholding and Stiefel projection in fact produce.

Several modern multi-component methods estimate orthogonal or nonredundant sparse components jointly. That raises the empirical bar for any block method: a corrected formulation must be compared with joint estimators, not only with deflation. We therefore distinguish five properties that are often conflated: score orthogonality, loading orthogonality, score decorrelation, support overlap, and causal nonredundancy. Only score-side Stiefel orthogonality is embedded in the primary model; the remaining four are either measured directly or added as optional penalties.

## 2.2 Convex maximization and generalized Frank–Wolfe

The historical update maximizes an afine minorant of a convex objective over a compact nonconvex set, which places it in the generalized Frank–Wolfe family for convex maximization. Akta¸s & Kroer (2025) obtain last-iterate results under smoothness, strong convexity, and a K L condition. That result does not transfer directly here: the corrected objective is not strongly convex in the same sense, the Stiefel feasible set is not a strongly convex set, and zero-response or rank-deficient polar steps need explicit tie policies. We invoke K L machinery only after we have established suficient ascent and a relative-error bound for the algorithm we actually run.

## 2.3 Sparse optimization on matrix manifolds

Manifold proximal-gradient, retraction, and alternating schemes handle nonsmooth composite problems under orthogonality constraints (Absil et al., 2008). The present work difers on three points: it eliminates the high-dimensional sparse block to produce a thresholded value function, keeps the historical full-polar update on the record for comparison, and treats component-specific active operators once identification has happened. The nearest future-facing comparator is sparse-Stiefel work that pairs identification with Newton acceleration. Such methods can win the local phase when second-order suficient conditions and clean active-manifold intersections both hold; we therefore leave second-order acceleration for heterogeneous active supports to future work.

## 2.4 Sparse neural representations

Sparse autoencoders learn overcomplete dictionaries, not orthogonal spectral components. Gated SAEs separate support selection from magnitude estimation and push the sparsity–reconstruction frontier (Rajamanoharan et al., 2024). MIB and SAEBench are explicit that reconstruction and feature labels are not enough: causal localization, stability, disentanglement, and downstream interventions have to be measured directly (Mueller et al., 2025; Karvonen et al., 2025). More recent work also identifies an amortization gap between simple encoders and more accurate sparse inference (O’Neill et al., 2025). The activation experiment in Section 9 compares methods at matched support size and keeps raw selection and refitting separate.

## 3 Corrected Problem Formulation

Let $A = [ a _ { 1 } , \ldots , a _ { n } ] \in \mathbb { R } ^ { p \times n }$ , let $X = [ x _ { 1 } , \ldots , x _ { m } ] \in \mathrm { S t } ( p , m )$ , and let $\mu _ { j } > 0$ . For a vector $^ { c , }$ define soft thresholding by

$$
S _ { \gamma } ( c ) _ { i } = \mathrm { s i g n } ( c _ { i } ) ( | c _ { i } | - \gamma ) _ { + } .
$$

The component response is $c _ { j } ( X ) = \mu _ { j } A ^ { \top } x _ { j }$

## 3.1 A-SUM: normalized loading-ball model

The corrected primal is

$$
\operatorname* { m a x } _ { X \in \mathrm { S t } ( p , m ) } \sum _ { j = 1 } ^ { m } \operatorname* { m a x } _ { \| z _ { j } \| _ { 2 } \leq 1 } \{ c _ { j } ( X ) ^ { \top } z _ { j } - \gamma _ { j } \| z _ { j } \| _ { 1 } \} .\tag{1}
$$

Loading balls allow an inactive column. The reduced objective is

$$
F _ { \mathrm { s u m } } ( X ) = \sum _ { j = 1 } ^ { m } \left\| S _ { \gamma _ { j } } ( c _ { j } ( X ) ) \right\| _ { 2 } .\tag{2}
$$

When the thresholded response is nonzero, the recovered loading is its normalized version; otherwise $z _ { j } = 0$ is valid.

Proposition 3.1 (Exact loading-ball elimination). For any $c \in \mathbb { R } ^ { n }$ and $\gamma \geq 0$

$$
\operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 1 } \{ \boldsymbol { c } ^ { \top } \boldsymbol { z } - \gamma \| \boldsymbol { z } \| _ { 1 } \} = \| \boldsymbol { S } _ { \gamma } ( \boldsymbol { c } ) \| _ { 2 } .
$$

$I f S _ { \gamma } ( c ) \ne 0$ , the unique maximizer is $z ^ { \star } = S _ { \gamma } ( c ) / \| S _ { \gamma } ( c ) \| _ { 2 } ,$ ; if it is zero, every maximizer includes $z = 0$ , and zero is selected by convention.

Proof. For each coordinate, $c _ { i } z _ { i } - \gamma | z _ { i } | \leq ( | c _ { i } | - \gamma ) _ { + } | z _ { i } |$ , with equality when the sign of $z _ { i }$ agrees with $c _ { i }$ on the active set and $z _ { i } = 0$ elsewhere. Hence the inner problem reduces to maximizing $S _ { \gamma } ( c ) ^ { \top }$ u over $\| u \| _ { 2 } \leq 1$ . Cauchy–Schwarz yields the value and optimizer. If the thresholded vector is zero, the objective is nonpositive and zero attains zero. □

## 3.2 A-SQ: squared-threshold model

The historical reduced objective is instead

$$
F _ { \mathrm { s q } } ( X ) = \sum _ { j = 1 } ^ { m } \left\| S _ { \gamma _ { j } } ( c _ { j } ( X ) ) \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

It has a diferent primal interpretation.

Proposition 3.2 (Quadratic-lift identity). For any $c \in \mathbb { R } ^ { n }$

$$
\left\| S _ { \gamma } ( c ) \right\| _ { 2 } ^ { 2 } = \operatorname* { m a x } _ { z \in \mathbb { R } ^ { n } } \{ 2 c ^ { \top } z - \| z \| _ { 2 } ^ { 2 } - 2 \gamma \| z \| _ { 1 } \} ,
$$

and the unique maximizer is $z ^ { \star } = S _ { \gamma } ( c )$

Proof. The objective separates over coordinates. For one coordinate, completing the square on each sign half-line gives the maximizer sign $. ( c _ { i } ) ( | c _ { i } | - \gamma )$ <sub>+</sub> and optimal value $( | c _ { i } | - \gamma ) _ { + } ^ { 2 }$ . Summing proves the claim. □

Table 1. Corrected block estimator family and loading recovery. The three columns define distinct estimators; $\ S 3$ proves A-SUM $\neq \mathsf { A } \ u { - } \mathsf { S } \mathsf { Q }$ at $m \geq 2$ and A-HT is a third exact-cardinality variant. Loading recovery is the loading-column argmax at a fixed score and defines the raw estimand that the restricted refit stage then post-processes.
<table><tr><td>Estimator</td><td>Primal (per column j)</td><td>Reduced objective</td><td>Loading recovery / inactive policy</td></tr><tr><td>A-SUM</td><td> $\begin{array} { r } { \operatorname* { m a x } _ { \mathbf { \mu } } \{ c _ { j } ^ { \top } z _ { j } - \gamma _ { j } \left\| z _ { j } \right\| _ { 1 } \} } \\ { \left\| z _ { j } \right\| _ { 2 } \leq 1 \qquad } \end{array}$ </td><td> $F _ { \mathrm { s u m } } ( X )$   $\begin{array} { r } { \sum _ { j } \big | \big | \dot { S } _ { \gamma _ { j } } ( c _ { j } ) \big | \big | _ { 2 } } \end{array}$ </td><td> $\begin{array} { r l } { = } & { { } z _ { j } ^ { \star } = { \cal S } _ { \gamma _ { j } } ( c _ { j } ) / \left\| { \cal S } _ { \gamma _ { j } } ( c _ { j } ) \right\| _ { 2 } } \end{array}$  when nonzero; otherwise  $z _ { j } ~ = ~ 0$  is the chosen in- active branch (loading-ball slack)</td></tr><tr><td>A-SQ</td><td> $2 \gamma _ { j } \parallel z _ { j } \parallel _ { 1 } \}$ </td><td> $\begin{array} { r l r } { \underset { z _ { j } } { \operatorname* { m a x } } \{ 2 c _ { j } ^ { \top } z _ { j } \quad } & { - } & { \Vert z _ { j } \Vert _ { 2 } ^ { 2 } \quad - \quad F _ { \mathrm { s q } } ( X ) = \sum _ { j } \Vert \mathcal { S } _ { \gamma _ { j } } ( c _ { j } ) \Vert _ { 2 } ^ { 2 } \quad z _ { j } ^ { \star } \quad = \quad S _ { \gamma _ { j } } ( c _ { j } ) , } \end{array}$ </td><td>(uncon- strained quadratic lift; the vector itself is the loading, no renormalization)</td></tr><tr><td>A-HT threshold, cardinality k)</td><td>(hard-  $\begin{array} { c } { \operatorname* { m a x } \qquad \quad c _ { j } ^ { \top } z _ { j } } \\ { \left\| z _ { j } \right\| _ { 2 } \leq 1 , \left\| z _ { j } \right\| _ { 0 } \leq k } \end{array}$  exact</td><td> $\begin{array} { l } { F _ { k } ( X ) } \\ { \sum _ { j } \sum _ { i \in T _ { k } ( c _ { j } ) } c _ { j , i } ^ { 2 } } \end{array}$ </td><td> $\begin{array} { r l } { = } & { { } \ z _ { j } ^ { \star } } \end{array}$  supported on the top-k coordinates of  $c _ { j }$  (determin- istic tie policy); refit on the chosen support</td></tr></table>

## 3.3 The objectives are not equivalent

Proposition 3.3 (Two-component counterexample). There exist a compact feasible set and two nonnegative component responses $r _ { 1 } ( X ) , r _ { 2 } ( X )$ for which arg ma $\tau _ { X } r _ { 1 } + r _ { 2 }$ and arg max $\cdot _ { X } r _ { 1 } ^ { 2 } + r _ { 2 } ^ { 2 }$ are disjoint.

Proof. Consider two feasible points $X _ { a } , X _ { b }$ with responses $( r _ { 1 } , r _ { 2 } ) ( X _ { a } ) \ = \ ( 2 , 2 )$ and $( r _ { 1 } , r _ { 2 } ) ( X _ { b } ) =$ (3.2, 0.5). The sums are 4 and 3.7, so the sum objective selects $X _ { a }$ . The squared sums are 8 and 10.49, so the squared objective selects $X _ { b }$ . Such response pairs can be realized by a finite feasible subset or approximated arbitrarily closely by thresholded linear responses. Therefore no monotone scalar transformation relates the block objectives in general. □

## 3.4 Numerical conventions

A specification without deterministic edge-case policies is not an algorithm. We fix five: (i) inactive response columns remain zero and are excluded from the active polar block; (ii) threshold equality is treated as inactive, except in experiments that explicitly study set-valued ties; (iii) rank-deficient polar factors are completed by a deterministic thin SVD; (iv) component permutations are aligned by maximum absolute score correlation for diagnostics; (v) refitting applies the support mask before any normalization, or solves the exact restricted problem instead.

## 4 Algorithms and Refitting

## 4.1 Historical full-polar iteration

For A-SQ the ambient gradient columns are

$$
g _ { j } ( X ) = 2 \mu _ { j } A S _ { \gamma _ { j } } ( \mu _ { j } A ^ { \top } x _ { j } ) .
$$

Table 2. Edge-case policies in the FACET reference implementation. These are the operational conventions on which every proved theorem is explicitly conditional; running the solver under a diferent policy voids the theorem’s claim.
<table><tr><td rowspan=1 colspan=5>Edge case                      Policy                                  Pinning testThreshold tie $\left( \left| c _ { i } \right| = \gamma \right)$         Treated as inactive $( \boldsymbol { S } _ { \gamma } ( \boldsymbol { c } ) _ { i } = 0 )$ via tests/test_paper_a_identities.py::test_a1_*the standard $( \cdot ) _ { + }$ convention; signedsubgradient reused by the solver.Zero soft-threshold vector Loading set to zero; the loading-ball test_a1_sum_norm_zero_support_returns_zero $( S _ { \gamma } ( c ) = 0 )$                      slack delivers primal value 0; no renor-malization attempted.Mask-first refit on a non-  $z ~ = ~ M c / \| M c \| _ { 2 } ;$  emptysupporttests/test_paper_a_edge_cases.py::test_a4_mask_then_no</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=2>raises ZeroDirectionError ratherthan returning NaN.Rank-deficient polar input canonical policy completes via test_a5_canonical_policy_*,</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>(Stiefel oracle)</td><td rowspan=1 colspan=2>SVD; raise policy rejects and sur- test_a5_raise_policy_*</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>faces the assumption violation. Both</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>treated as explicit theorem condition-</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=2 colspan=2>ing.ra- Bothpolar policies raise test_a5_zero_direction_raises_regardless_of_policy</td><td></td></tr><tr><td rowspan=1 colspan=2>Zero direction to a linear o</td><td></td></tr><tr><td rowspan=1 colspan=2>cle</td><td rowspan=1 colspan=2>ZeroDirectionError     uncondi-</td><td></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=2 colspan=2>tionally.</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Top-k tie at the boundary (A-</td><td rowspan=1 colspan=2>Lower original index wins; set- test_topk_tie_policy-prefers_lower_index</td><td></td></tr><tr><td rowspan=1 colspan=2>HT)</td><td rowspan=1 colspan=2>valued tie metadata recorded on</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>TopKSelection.</td><td></td></tr><tr><td rowspan=1 colspan=2>Armijo backtracking at m</td><td rowspan=1 colspan=2>a- Linesearchfails loudly empirical</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>chine precision</td><td rowspan=1 colspan=2>(RuntimeError); drivers may</td><td rowspan=2 colspan=1>catch and drop that seed rather than</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>Component sign at conver- Com</td><td rowspan=1 colspan=2>silently reporting a stationary point.ponent orientation is global-sign test_a9_sign_identification_matches-planted_sign-patte</td><td></td></tr><tr><td rowspan=1 colspan=2>gence</td><td rowspan=1 colspan=2>ambiguous; projector-level reporting</td><td></td></tr></table>

Writing $G ( X ) = [ g _ { 1 } ( X ) , \dots , g _ { m } ( X ) ]$ ], the historical step chooses

$$
X _ { k + 1 } \in \arg \operatorname* { m a x } _ { Y \in \operatorname { S t } ( p , m ) } \left. G ( X _ { k } ) , Y \right. = \operatorname { P o l a r } ( G ( X _ { k } ) ) .
$$

When G is rank deficient, the polar maximizer is set-valued; the implementation uses a deterministic SVD completion and records the rank event.

## 4.2 FACET safeguarded iteration

The whole-sequence analysis is attached to a diferent algorithm. At $X _ { k }$ we take the Riemannian gradient grad $F ( X _ { k } )$ , build a tangent ascent direction $\eta _ { k }$ , and retract $R _ { X _ { k } } ( t \eta _ { k } )$ . Backtracking picks the largest t satisfying

$$
F ( R _ { X _ { k } } ( t \eta _ { k } ) ) \geq F ( X _ { k } ) + \sigma t \left\| \eta _ { k } \right\| _ { F } ^ { 2 } .
$$

When a candidate step fails because of numerical rank loss or an undefined active response, the method falls back to a regularized polar/retraction and logs the event. The stopping rule combines a Riemannian-gradient norm, the linear-oracle residual, and support stability; objective change by itself is not enough.

Table 3. Assumption and failure-mode taxonomy.
<table><tr><td>Assumption</td><td>Enables</td><td>Failure manifestation</td></tr><tr><td>Strict threshold margin</td><td>Finite support/sign identification</td><td>Persistent switching or tie depen- dence</td></tr><tr><td>Full-rank local polar argu- Smooth polar branch ment</td><td></td><td>Set-valued update and unstable derivative</td></tr><tr><td>Active eigengap or  $\rho ( D \Phi ) < 1$ </td><td>Local linear convergence</td><td>Slow/noncontractive local dynamics</td></tr><tr><td>Certified distance radius</td><td>Safe screening</td><td>Heuristic false elimination if radius is guessed</td></tr><tr><td>Independent units</td><td>document Valid uncertainty summaries</td><td>Token-level pseudo-replication</td></tr></table>

![](images/e5c25e268ebad522ca9f0d2ba90ced0a2727431b19444bbd904270bedeebd6ae.jpg)  
Figure 3. Two-phase interpretation: discrete active-structure discovery followed by conditional smooth dynamics.

## 4.3 Support-conditioned refitting

For one selected support S, the exact refit is the leading singular-vector pair of $A _ { S }$ . For component-specific block supports $S _ { j }$ , define masks $M _ { j }$ and solve

$$
\operatorname* { m a x } _ { X ^ { \top } X = I , \ \| z _ { j } \| _ { 2 } = 1 , \ \operatorname { s u p p } ( z _ { j } ) \subseteq S _ { j } } \sum _ { j = 1 } ^ { m } \mu _ { j } x _ { j } ^ { \top } A z _ { j } .
$$

Conditional on $X ,$ , the exact loading update is

$$
z _ { j } \gets \frac { M _ { j } A ^ { \top } x _ { j } } { \| M _ { j } A ^ { \top } x _ { j } \| _ { 2 } } ,
$$

when nonzero. Conditional on $Z ,$ the exact score update is $X  \operatorname { P o l a r } ( A Z \operatorname { D i a g } ( \mu ) )$ . Each block update weakly increases the refit objective, so alternating exact updates are monotone, and the limit is a block-stationary point rather than the global refit optimum.

## 4.4 Complexity

Without screening, correlation and reconstruction products cost $O ( n p m )$ for dense A; the polar factor costs $O ( p m ^ { 2 } + m ^ { 3 } )$ . After permanent screening to s active or near-active coordinates, the dominant products cost $O ( s p m )$ . Refitting cost depends on support structure and is reported separately from selector cost.

## 5 Theory

![](images/5f907fb1ed51e71db2bb2795d94eadf19ea22329af4b2cfbdfb246c8dfdaff44.jpg)  
Figure 4. Safeguarded Riemannian ascent, the analyzed algorithm. Gradient → tangent projection → retraction → Armijo-safeguarded acceptance with geometric step shrinkage. The stopping rule combines a tangent-residua criterion with both step-norm and support-stability tests; objective change alone is insuficient. Historical full-polar iteration omits the Armijo loop and carries no last-iterate claim.

Algorithm 1 Historical full-polar A-SQ iteration   
Require: Data operator A, thresholds $\gamma _ { j }$ , weights $\mu _ { j }$ , initial $X _ { 0 } \in \mathrm { S t } ( p , m )$   
1: for $k = 0 , 1 , \ldots$ do   
2: $C _ { k } \gets A ^ { \top } X _ { k } \operatorname { D i a g } ( \mu _ { 1 } , . . . , \mu _ { m } )$   
3: $Z _ { k } \gets S _ { \gamma } ( C _ { k } )$ columnwise   
4: $G _ { k } \gets 2 A Z _ { k } \operatorname { D i a g } ( \mu _ { 1 } , . . . , \mu _ { m } )$   
5: $X _ { k + 1 }$ ← DeterministicPolar $\left( G _ { k } \right)$   
6: log objective, linear-oracle residual, support, signs, and rank   
7: if the stationarity and support criteria hold then   
8: break   
9: end if   
10: end for

## 5.1 Monotonicity of the full-polar step

Proposition 5.1 (Afine-minorant ascent). Let F be diferentiable and convex on an open set containing   
St(p, m). If   
$X ^ { + } \in \arg \operatorname* { m a x } _ { Y \in { \mathrm { S t } } ( p , m ) } \left. \nabla F ( X ) , Y \right. ,$   
then $F ( X ^ { + } ) \geq F ( X )$ . More precisely,   
$F ( X ^ { + } ) - F ( X ) \geq \Delta _ { F } ( X ) : = \operatorname* { m a x } _ { Y \in \operatorname { S t } ( p , m ) } \left. \nabla F ( X ) , Y - X \right. \geq 0 .$

Proof. Convexity gives $F ( X ^ { + } ) \geq F ( X ) + \langle \nabla F ( X ) , X ^ { + } - X \rangle$ . Because X is feasible and $X ^ { + }$ maximizes the linear form, the inner product equals $\Delta _ { F } ( X ) \geq 0 .$ □

The proposition gives ascent and summability of the nonnegative residual along any bounded objective sequence. It does not give iterate convergence, nor local maximality.

Algorithm 2 Safeguarded active-set spectral ascent   
Require: $X _ { 0 } \in \mathrm { S t } ( p , m )$ , Armijo constants $\sigma \in ( 0 , 1 ) , \beta \in ( 0 , 1 )$ , regularization $\varepsilon \geq 0$   
1: for $k = 0 , 1 , \ldots$ do   
2: compute $F ( X _ { k } )$ , active pattern, and grad $F ( X _ { k } )$   
3: choose tangent ascent direction $\eta _ { k }$ satisfying a gradient-related condition   
4: $t \gets 1$   
5: while Armijo ascent fails for $R _ { X _ { k } } ( t \eta _ { k } )$ do   
6: $t \gets \beta t$   
7: end while   
8: $X _ { k + 1 }  R _ { X _ { k } } ( t \eta _ { k } )$   
9: update certified support margins and optional local screening set   
10: end for   
Exact model Specified al-  
Ascent + residual   
identities gorithms   
Fixed-support Strict margin Suficient ascent +   
  
local map + convergence relative error + K L

![](images/7115dd4782c8347358ac0cd8a5b43479874707cf8635d48b90419ca2bbe50d54.jpg)  
Figure 5. The theorem dependency graph. Identification and local rates are not available before the model, algorithm, and convergence object are fixed.

## 5.2 Whole-sequence convergence of the safeguarded method

The last-iterate theorem for Algorithm 2 is Theorem 6.7 in Section 6. Its three ingredients, suficient ascent (Lemma 6.5), relative error (Lemma 6.6), and the semi-algebraic definability of $- F + \iota _ { \mathrm { S t } }$ , are proved there. The semi-algebraic step needs no assumption beyond the actual soft-threshold formula and the algebraic Stiefel constraint, and it does not share the strong-convexity hypotheses of the recent unit-step analysis of Akta¸s & Kroer (2025), which we cite for comparison rather than as a substitute. The rank-deficient polar branch is handled by the deterministic SVD completion of Section 3; on the compact iterate set the completion preserves the constants used in Lemmas 6.5–6.6.

## 5.3 Finite active-set identification

Theorem 5.2 (Conditional support and sign identification). Assume $X _ { k } \to X _ { \star }$ . Suppose there exists $\delta > 0$ such that for every component and coordinate,

$$
\left| | \mu _ { j } a _ { i } ^ { \top } x _ { j \star } | - \gamma _ { j } \right| \geq \delta ,
$$

and every active correlation is separated from zero by δ. Then there exists $K < \infty$ such that the support and active signs induced by $X _ { k }$ equal those of $X ,$ for all $k \geq K$

Proof. There are finitely many correlations and each is continuous in X. Convergence implies uniform convergence of all correlations to their limiting values. Choose K so that every correlation changes by less than $\delta / 2$ after iteration K. No threshold inequality or active sign can then change. □

At zero margin this conclusion fails: a convergent sequence can approach a threshold boundary while alternating between its two sides indefinitely.

## 5.4 Post-identification dynamics

Proposition 5.3 (Rank-one hard-threshold rate). After a hard-threshold support S is fixed, the rank-one update is power iteration on $C _ { S } = A _ { S } A _ { S } ^ { \top } . \ I f \lambda _ { 1 } ( C _ { S } ) > \lambda _ { 2 } ( C _ { S } )$ and the iterate has nonzero projection on the leading eigenvector, then

$$
\tan \angle ( x _ { k + 1 } , u _ { 1 } ) \leq { \frac { \lambda _ { 2 } ( C _ { S } ) } { \lambda _ { 1 } ( C _ { S } ) } } \tan \angle ( x _ { k } , u _ { 1 } ) .
$$

Proof. Expand $x _ { k }$ in the orthonormal eigenbasis of $C _ { S }$ . Multiplication scales the leading coeficient by $\lambda _ { 1 }$ and every orthogonal coeficient by at most $\lambda _ { 2 } ;$ normalization cancels a common factor. □

For a common block support the analogue is orthogonal iteration, with rate set by the gap $\lambda _ { m } - \lambda _ { m + 1 }$ With component-specific supports define $\Phi ( X ) = \operatorname { P o l a r } ( [ C _ { S _ { 1 } } x _ { 1 } , \dots , C _ { S _ { m } } x _ { m } ] )$ . A correct local theorem must diferentiate Φ on the Stiefel tangent space and require $\rho ( D \Phi ( X _ { \star } ) | _ { T _ { X _ { \star } } \mathrm { S t } } ) < 1$ . A single eigengap is not a substitute for this condition when the active operators difer.

## 5.5 Locally safe screening

Proposition 5.4 (Radius-based safe screening). Suppose $\left\| x _ { j \star } - x _ { j k } \right\| _ { 2 } \leq r _ { j k }$ . If

$$
\begin{array} { r } { \left| \mu _ { j } a _ { i } ^ { \top } x _ { j k } \right| + \mu _ { j } \left\| a _ { i } \right\| _ { 2 } r _ { j k } < \gamma _ { j } , } \end{array}
$$

then coordinate (i, j) is inactive at $X _ { \star }$

Proof. Cauchy–Schwarz gives $| \mu _ { j } a _ { i } ^ { \top } x _ { j \star } | \leq | \mu _ { j } a _ { i } ^ { \top } x _ { j k } | + \mu _ { j } \| a _ { i } \| _ { 2 } r _ { j k }$ . The strict upper bound lies below threshold. □

The rule is safe only when $r _ { j k }$ is itself certified. Empirical step sizes or unverified contraction estimates produce heuristic screening and are labeled accordingly.

## 5.6 Second-order scope

On a fixed rank-one hard support, the objective is a Rayleigh quotient: nonleading simple eigenvectors are saddles or local minima according to the tangent Hessian. For heterogeneous block supports and nonsmooth threshold boundaries, a generic strict-saddle theorem is not claimed. Any Newton acceleration must be compared with contemporary sparse-Stiefel identification and second-order methods and must state clean-intersection and second-order suficient conditions explicitly.

## 6 Proofs and Extended Analysis

## 6.1 Exact value functions and recovery maps

For one component, let $c = \mu A ^ { \top } { : }$ x and let $S _ { \gamma } ( c )$ denote entrywise soft thresholding. The following theorem fixes both the value function and the optimizer, including the inactive case.

Theorem 6.1 (Loading-ball elimination). For every $c \in \mathbb { R } ^ { n }$ and $\gamma \geq 0$

$$
\operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 1 } \{ \boldsymbol { c } ^ { \top } \boldsymbol { z } - \gamma \| \boldsymbol { z } \| _ { 1 } \} = \| S _ { \gamma } ( \boldsymbol { c } ) \| _ { 2 } .
$$

$I f S _ { \gamma } ( c ) \ne 0$ , the unique maximizer is

$$
z ^ { \star } = \frac { S _ { \gamma } ( c ) } { \| S _ { \gamma } ( c ) \| _ { 2 } } .
$$

$I f S _ { \gamma } ( c ) = 0$ , then $z ^ { \star } = 0$ is optimal; it is the unique optimum whenever $| c _ { i } | < \gamma$ for every $i .$

Proof. For any feasible $z ,$

$$
c ^ { \top } z - \gamma \| z \| _ { 1 } = \sum _ { i } ( c _ { i } z _ { i } - \gamma | z _ { i } | ) \leq \sum _ { i } ( | c _ { i } | - \gamma ) _ { + } \left| z _ { i } \right| .
$$

Choosing the signs of $z _ { i }$ to agree with $c _ { i }$ makes the inequality tight. Cauchy–Schwarz then gives

$$
\sum _ { i } ( \left| c _ { i } \right| - \gamma ) _ { + } \left| z _ { i } \right| \leq \left\| S _ { \gamma } ( c ) \right\| _ { 2 } \left\| z \right\| _ { 2 } \leq \left\| S _ { \gamma } ( c ) \right\| _ { 2 } .
$$

When the thresholded vector is nonzero, equality in Cauchy–Schwarz requires the stated normalized vector. If the thresholded vector vanishes, the objective is nonpositive and $z = 0$ attains zero. Strict inequalities $| c _ { i } | < \gamma$ force every nonzero coordinate to make a strictly negative contribution. □

Theorem 6.2 (Quadratic-lift representation). For every $c \in \mathbb { R } ^ { n }$

$$
\left\| S _ { \gamma } ( c ) \right\| _ { 2 } ^ { 2 } = \operatorname* { m a x } _ { z \in \mathbb { R } ^ { n } } \left\{ 2 c ^ { \top } z - \left\| z \right\| _ { 2 } ^ { 2 } - 2 \gamma \left\| z \right\| _ { 1 } \right\} ,
$$

and the unique maximizer is $\boldsymbol { z } ^ { \star } = \boldsymbol { S } _ { \gamma } ( \boldsymbol { c } )$

Proof. The objective is separable. For coordinate $i ,$ maximize $2 c _ { i } z _ { i } - z _ { i } ^ { 2 } - 2 \gamma \left| z _ { i } \right|$ . Restricting to the sign of $c _ { i }$ and completing the square gives the maximizer $\mathrm { s i g n } ( c _ { i } ) ( | c _ { i } | - \gamma ) _ { + }$ and optimal value $( | c _ { i } | - \gamma ) _ { + } ^ { 2 }$ Summation proves the claim. □

Proposition 6.3 (The two block aggregations are not equivalent). For $m > 1$ , maximizing $\textstyle \sum _ { j } r _ { j } ( X )$ and maximizing $\textstyle \sum _ { j } r _ { j } ( X ) ^ { 2 }$ need not have the same maximizers, even when every $r _ { j }$ is nonnegative and arises from a thresholded norm.

Proof. It sufices to construct two feasible points with response vectors $( r _ { 1 } , r _ { 2 } ) = ( 2 , 0 )$ and $( { \sqrt { 3 / 2 } } , { \sqrt { 3 / 2 } } )$ The first has larger squared sum, $4 > 3$ , while the second has larger linear sum, $2 { \sqrt { 3 / 2 } } > 2$ . Such response pairs are realized by a diagonal two-dimensional data operator and two orthogonal score frames with diferent rotations. A concrete numerical instance is included in Appendix B and the released regression test. □

## 6.2 Diferentiability and active-region smoothness

The scalar function $q _ { \gamma } ( u ) = ( | u | - \gamma ) _ { + } ^ { 2 }$ is continuously diferentiable, with $q _ { \gamma } ^ { \prime } ( u ) = 2 \operatorname { s i g n } ( u ) ( | u | - \gamma ) _ { + }$ Hence

$$
\nabla _ { x _ { j } } F _ { \mathrm { s q } } ( X ) = 2 \mu _ { j } A S _ { \gamma _ { j } } ( \mu _ { j } A ^ { \top } x _ { j } ) .
$$

It is globally Lipschitz in the ambient space with constant at most $2 \mu _ { j } ^ { 2 } \left. A \right. _ { 2 } ^ { 2 }$ per column because soft thresholding is nonexpansive. On any fixed support-and-sign cell, the gradient is afine and the Hessian is constant. This piecewise-quadratic structure is the basis for finite identification and post-identification linearization.

Lemma 6.4 (Global Lipschitz gradient). For $f _ { j } ( x ) = \left\| S _ { \gamma _ { j } } ( \mu _ { j } A ^ { \top } x ) \right\| _ { 2 } ^ { 2 } ,$

$$
\begin{array} { r } { \left\| \nabla f _ { j } ( x ) - \nabla f _ { j } ( y ) \right\| _ { 2 } \leq 2 \mu _ { j } ^ { 2 } \left\| A \right\| _ { 2 } ^ { 2 } \left\| x - y \right\| _ { 2 } . } \end{array}
$$

Proof. Use the gradient formula, submultiplicativity, and nonexpansiveness of soft thresholding:

$$
\begin{array} { r } { \left\| { A } [ S _ { \gamma } ( \mu A ^ { \top } x ) - S _ { \gamma } ( \mu A ^ { \top } y ) ] \right\| _ { 2 } \leq \left\| { A } \right\| _ { 2 } \mu \left\| { A } ^ { \top } ( x - y ) \right\| _ { 2 } . } \end{array}
$$

Multiplying by the leading factor $2 \mu$ proves the result.

## 6.3 Safeguarded ascent and whole-sequence convergence

The historical polar step maximizes the ambient afine minorant. The analyzed algorithm (A-SAFE) uses the Riemannian gradient $\eta _ { k } = \mathrm { g r a d } _ { R } F ( X _ { k } ) = P _ { T _ { X _ { k } } \mathrm { S t } } ( \nabla F ( X _ { k } ) )$ , an explicit retraction $R _ { X _ { k } }$ satisfying standard second-order smoothness, and an Armijo acceptance rule

$$
F \big ( R _ { X _ { k } } ( t \eta _ { k } ) \big ) \geq F ( X _ { k } ) + \sigma t \left\| \eta _ { k } \right\| _ { F } ^ { 2 } , \qquad \sigma \in ( 0 , 1 ) ,\tag{4}
$$

with geometric backtracking $t \gets \rho t , \rho \in ( 0 , 1 )$ . The two lemmas below establish the suficient-ascent and relative-error inequalities that the whole-sequence theorem rests on.

Lemma 6.5 (A6, suficient ascent). Let F have L-Lipschitz ambient gradient (Lemma 6.4) and let the retraction R satisfy, uniformly on compact subsets of $\mathrm { S t } ( p , m ) \colon R _ { X } ( 0 ) = X , D R _ { X } ( 0 ) = \mathrm { I d } _ { T _ { X } \mathrm { S t } }$ and $\begin{array} { r } { \| R _ { X } ( \eta ) - X - \eta \| _ { F } \leq \frac { C _ { R } } { 2 } \| \eta \| _ { F } ^ { 2 } } \end{array}$ for $\| \eta \| _ { F } \le \bar { t } .$ Then there exists $t > 0 ,$ , independent of k, such that Armijo backtracking from $t = \bar { t }$ selects a step $t _ { k } \geq \underline { { t } } ,$ and the accepted iterate $X _ { k + 1 } = R _ { X _ { k } } ( t _ { k } \eta _ { k } )$ satisfies

$$
F ( X _ { k + 1 } ) - F ( X _ { k } ) \ \geq \ c _ { 1 } \ \| X _ { k + 1 } - X _ { k } \| _ { F } ^ { 2 } , \qquad c _ { 1 } = { \frac { \sigma { t } } { ( 1 + { \frac { C _ { R } } { 2 } } \bar { t } M ) ^ { 2 } \bar { t } } } > 0 ,\tag{5}
$$

where $M = \operatorname* { s u p } _ { k } \| \eta _ { k } \| _ { F }$ is finite because the iterates stay in a compact set.

Proof. By the retraction second-order estimate and the L-Lipschitz-gradient descent lemma applied to $- F _ { \ast }$

$$
\begin{array} { r } { F \big ( R _ { X _ { k } } ( t \eta _ { k } ) \big ) \geq F ( X _ { k } ) + t \left\| \eta _ { k } \right\| _ { F } ^ { 2 } - \frac { L } { 2 } \left\| R _ { X _ { k } } ( t \eta _ { k } ) - X _ { k } \right\| _ { F } ^ { 2 } \geq F ( X _ { k } ) + \Big ( t - \frac { L C _ { R } ^ { 2 } } { 2 } t ^ { 2 } \Big ) \left\| \eta _ { k } \right\| _ { F } ^ { 2 } . } \end{array}
$$

The Armijo bound (4) is therefore satisfied whenever $\begin{array} { r } { t - \frac { L C _ { R } ^ { 2 } } { 2 } t ^ { 2 } \geq \sigma t , \mathsf { i . e . } t \leq \frac { 2 ( 1 - \sigma ) } { L C _ { R } ^ { 2 } } } \end{array}$ . Hence backtracking terminates at some $t _ { k } \ \ge \ \underline { { { t } } } \ : = \ \operatorname* { m i n } \{ \bar { t } , \ \rho \cdot \frac { 2 ( 1 - \sigma ) } { L C _ { R } ^ { 2 } } \} \ > \ 0$ . The step norm obeys $\lVert X _ { k + 1 } - X _ { k } \rVert _ { F } \leq$ $\begin{array} { r } { \left( 1 + \frac { C _ { R } } { 2 } t _ { k } \left. \eta _ { k } \right. _ { F } \right) t _ { k } \left. \eta _ { k } \right. _ { F } \leq \left( 1 + \frac { C _ { R } } { 2 } \bar { t } M \right) t _ { k } \left. \eta _ { k } \right. _ { F } } \end{array}$ . Combining with (4) at the accepted step and bounding $t _ { k } \leq \bar { t } ,$

$$
\begin{array} { r } { F ( X _ { k + 1 } ) - F ( X _ { k } ) \ \ge \ \sigma t _ { k } \ \| \eta _ { k } \| _ { F } ^ { 2 } \ \ge \ \frac { \sigma t } { ( 1 + \frac { C _ { R } } { 2 } \overline { { t } } M ) ^ { 2 } \bar { t } } \left\| X _ { k + 1 } - X _ { k } \right\| _ { F } ^ { 2 } . } \end{array}
$$

Lemma 6.6 (A7, relative error). Under the same hypotheses, the limiting subdiferential of $\Psi : = - \boldsymbol { F } + \iota _ { \mathrm { S t } }$ at $X _ { k + 1 }$ contains an element

$$
w _ { k + 1 } \in \partial \Psi ( X _ { k + 1 } ) \quad w i t h \quad \left\| w _ { k + 1 } \right\| _ { F } \ \leq \ c _ { 2 } \ \left\| X _ { k + 1 } - X _ { k } \right\| _ { F } ,
$$

where $c _ { 2 }$ depends only on $L , C _ { R } , \sigma$ , and the compact set containing the iterates. An admissible choice is $w _ { k + 1 } = - \mathrm { g r a d } _ { R } F ( X _ { k + 1 } ) = - \nabla F ( X _ { k + 1 } ) + \mathrm { p r o j } _ { N _ { \mathbb { S } ^ { k } } ( X _ { k + 1 } ) } \nabla F ( X _ { k + 1 } ) \in \partial ( - F ) ( X _ { k + 1 } ) + N _ { \mathbb { S } ^ { t } } ( X _ { k + 1 } ) \subseteq N _ { \mathbb { S } ^ { k } } ( X _ { k + 1 } ) .$ $\partial \Psi ( X _ { k + 1 } )$

Proof. Since F is smooth, $\partial ( - F ) ( X ) = \{ - \nabla F ( X ) \}$ ; the Riemannian gradient equals the ambient gradient minus its projection onto the normal cone of $\mathrm { S t } ( p , m )$ at $X , \mathsf { s o - g r a d } _ { R } F ( X _ { k + 1 } ) \in \partial \Psi ( X _ { k + 1 } )$

Triangle inequality:

$$
\begin{array} { r } { \| \mathrm { g r a d } _ { R } F ( X _ { k + 1 } ) \| _ { F } \leq \| \mathrm { g r a d } _ { R } F ( X _ { k + 1 } ) - \mathrm { g r a d } _ { R } F ( X _ { k } ) \| _ { F } + \| \eta _ { k } \| _ { F } . } \end{array}
$$

On a compact subset of $\mathrm { S t } ( p , m )$ , the Riemannian-gradient field inherits a Lipschitz constant $L ^ { \prime } = L \cdot ( 1 + \kappa ) $ where $\kappa$ bounds the derivative of the tangent-projection map; the first term is at most $L ^ { \prime } \| X _ { k + 1 } - X _ { k } \| _ { F } .$ For the second, Lemma 6.5 gives $\| \eta _ { k } \| _ { F } \le \| X _ { k + 1 } - X _ { k } \| _ { F } / \underline { { t } } .$ . Hence

$$
\begin{array} { r } { { \left\| { \boldsymbol { w } } _ { k + 1 } \right\| } _ { F } \leq \left( L ^ { \prime } + \frac { 1 } { \underline { { t } } } \right) \left\| { \boldsymbol { X } } _ { k + 1 } - { \boldsymbol { X } } _ { k } \right\| _ { F } = : c _ { 2 } \left\| { \boldsymbol { X } } _ { k + 1 } - { \boldsymbol { X } } _ { k } \right\| _ { F } . } \end{array}
$$

Theorem 6.7 (A8, whole-sequence convergence under K L). Let $( X _ { k } ) _ { k \geq 0 } \subset \mathrm { S t } ( p , m )$ be generated by the safeguarded Riemannian ascent of Section 4, with retraction satisfying the second-order estimate of Lemma 6.5. Then

1. the sequence has finite length, $\begin{array} { r } { \sum _ { k > 0 } \| X _ { k + 1 } - X _ { k } \| _ { F } < \infty ; } \end{array}$

2. $( X _ { k } )$ converges to a single Riemannian critical point $X _ { \star }$ of F on $\mathrm { S t } ( p , m )$

3. with K L exponent $\theta \in [ 0 , 1 )$ of $\dot { \Psi } = - { \cal F } + \iota _ { \mathrm { S t } } \mathsf { \Omega } a t \ X _ { \star } ,$ the rate is finite $i f \theta = 0$ , linear $\| X _ { k } - X _ { \star } \| _ { F } \leq$ $C \varrho ^ { k }$ (some $\begin{array} { r } { \varrho \in ( 0 , 1 ) ) \ i f \theta \in ( 0 , \frac { 1 } { 2 } ] , \ a n d \ \| X _ { k } - X _ { \star } \| _ { F } \leq C k ^ { - ( 1 - \theta ) / ( 2 \theta - 1 ) } \ i f \theta \in ( \frac { 1 } { 2 } , 1 ) } \end{array}$

Proof. Compactness of $\mathrm { S t } ( p , m )$ makes the iterates bounded and F bounded above. Lemma 6.5 rewrites as the suficient-decrease inequality $\Psi ( X _ { k } ) - \Psi ( X _ { k + 1 } ) \geq c _ { 1 } \left\| X _ { k + 1 } - X _ { k } \right\| _ { F } ^ { 2 }$ for $\Psi = - \boldsymbol { F } + \iota _ { \mathrm { S t } }$ ; Lemma 6.6 supplies the relative-error inequality in the form required by Attouch et al. (2013).

Soft-thresholding is piecewise afine on a finite polyhedral partition, hence semi-algebraic; a sum of squared Euclidean norms of composite semi-algebraic maps is semi-algebraic, so $F$ and $- \boldsymbol { F }$ are semi-algebraic. The Stiefel manifold $\{ X : X ^ { \top } X = I _ { m } \}$ is a real algebraic variety, so $\iota _ { \mathrm { S t } }$ is semi-algebraic. Sums of semi-algebraic functions are semi-algebraic, so $\Psi$ is semi-algebraic and therefore definable in the o-minimal structure of real semi-algebraic sets. Definable functions satisfy the Kurdyka– Lojasiewicz inequality on their domain (Bolte et al., 2007); in particular at every limit point of $( X _ { k } )$

The hypotheses of the abstract convergence theorem of Attouch et al. (2013) (suficient decrease, relative error, continuity along the sequence, boundedness of $( X _ { k } )$ , and $\mathsf { K L } )$ are therefore met for Ψ. Applying it yields items 1 and 2. The rate claim 3 is the standard K L-exponent corollary of the same framework.

The constants $c _ { 1 } , c _ { 2 }$ in Lemmas 6.5–6.6 are computable from the Lipschitz bound (Lemma 6.4) together with the retraction curvature $C _ { R }$ and the Armijo parameters; the reference implementation records them in the solver trace. The rank-deficient polar branch is handled by the deterministic SVD completion of Section 3 and does not alter the above constants on the compact iterate set.

## 6.4 Identification with explicit margins

Let $\mathcal { T } _ { j } ( X ) = \{ i : \left| \mu _ { j } a _ { i } ^ { \top } x _ { j } \right| > \gamma _ { j } \}$ and let $S _ { j } ( X )$ denote active signs. The strict margin is

$$
\delta _ { \star } = \operatorname* { m i n } _ { i , j } \left| \left| \mu _ { j } a _ { i } ^ { \top } x _ { j \star } \right| - \gamma _ { j } \right| .
$$

The conditional identification theorem in Section 5 is immediate from continuity. A stronger perturbation statement is useful for screening.

Corollary 6.8 (Support stability radius). $I f \delta _ { \star } > 0$ and

$$
\| x _ { j } - x _ { j \star } \| _ { 2 } < \operatorname* { m i n } _ { i } \frac { \delta _ { \star } } { 2 \mu _ { j } \| a _ { i } \| _ { 2 } } ,
$$

![](images/8b0c9042e794e36b72f9b657ca222efec33b7dbd768acb4239c9d4907fec74e2.jpg)  
Figure 6. Theory-to-evidence chain for the developed manuscript.

then component j has the same active support as at $X _ { \star }$ . If active correlations are also separated from zero by $\sigma _ { \star } > 0$ , replacing the numerator by min $( \delta _ { \star } , \sigma _ { \star } ) / 2$ preserves active signs.

Proof. Cauchy–Schwarz bounds every correlation perturbation by $\mu _ { j } \parallel a _ { i } \parallel _ { 2 } \parallel x _ { j } - x _ { j \star } \parallel _ { 2 }$ . No threshold or sign boundary can be crossed inside the stated radius. □

## 6.5 Common-support block rate

When all columns share an active set S in the hard-threshold model, the fixed-support update is orthogonal iteration for $C _ { S } = A _ { S } A _ { S } ^ { \top }$

Theorem 6.9 (Common-support subspace contraction). Let $C _ { S }$ be symmetric positive semidefinite with $\lambda _ { m } ( C _ { S } ) > \lambda _ { m + 1 } ( C _ { S } )$ . Let $U _ { \star }$ span the dominant m-dimensional invariant subspace, and assume $U _ { \star } ^ { \top } X _ { 0 }$ is nonsingular. Orthogonal iteration satisfies

$$
\| \tan \Theta ( X _ { k } , U _ { \star } ) \| _ { 2 } \leq \left( { \frac { \lambda _ { m + 1 } ( C _ { S } ) } { \lambda _ { m } ( C _ { S } ) } } \right) ^ { k } \| \tan \Theta ( X _ { 0 } , U _ { \star } ) \| _ { 2 } .
$$

Proof. Write the initial frame in the eigenbasis of $C _ { S }$ and partition it into dominant and complementary blocks. After multiplication by $C _ { S } ^ { k }$ , the complementary-to-dominant coeficient ratio is multiplied by at most $( \lambda _ { m + 1 } / \lambda _ { m } ) ^ { k }$ . Orthonormalization does not change the represented subspace, yielding the tangentof-principal-angle bound. □

## 6.6 Heterogeneous supports and the polar Jacobian

For component-specific supports, define $G ( X ) = [ C _ { S _ { 1 } } x _ { 1 } , \dots , C _ { S _ { m } } x _ { m } ]$ and $\Phi ( X ) = \operatorname { P o l a r } ( G ( X ) )$ . If $G ( X _ { \star } )$ has full column rank, the polar map is diferentiable. Its diferential is characterized by a Sylvester equation for the tangent skew term. The correct local condition is therefore

$$
\rho \left( D \Phi ( { \boldsymbol { X } } _ { \star } ) | _ { T _ { \boldsymbol { X } _ { \star } } \operatorname { S t } } \right) < 1 ,
$$

not one pooled eigengap. Appendix B.3 derives the operator needed to evaluate this condition numerically.

## 6.7 Second-order statements with explicit justification

On a fixed rank-one support, the objective reduces to a Rayleigh quotient. A simple nonleading eigenvector has a tangent direction of positive curvature toward a larger eigenvalue and is not a local maximum. For heterogeneous supports and threshold boundaries, no generic strict-saddle claim is made; this paper only reports second-order classification for fixed-support smooth cells whose tangent Hessian is explicitly computable.

## 7 Algorithmic Variants and Fully Proved Special Cases

Algorithm 3 A-SUM alternating support-and-score update   
Require: Data operator A, thresholds $\gamma _ { j }$ , weights $\mu _ { j }$ , initial $X _ { 0 } \in \mathrm { S t } ( p , m )$   
1: for t = 0, 1, . . . do   
2: for $j = 1 , \ldots , m$ do   
3: $s _ { j , t } \gets S _ { \gamma _ { j } } ( \mu _ { j } A ^ { \top } x _ { j , t } )$   
4: $z _ { j , t + 1 } \gets s _ { j , t } / \left| \left| s _ { j , t } \right| \right| _ { 2 }$ if $s _ { j , t } \neq 0 ,$ , otherwise 0   
5: end for   
6: $G _ { t + 1 } \gets A Z _ { t + 1 } N$   
7: $X _ { t + 1 } $ declared active-column polar update plus deterministic completion   
8: end for

## 7.1 A-SUM alternating method

The exact normalized-loading model is best handled in its primal variables rather than by treating it as smooth at an inactive component. Given $X _ { t }$ , update every loading column by Theorem 6.1; given $Z _ { t + 1 }$ update X by a polar factor of $A Z _ { t + 1 } N$ . When a loading column is zero, the corresponding score column is not identified by the trace term; the implementation either drops the inactive column from the block or completes the active score frame by a deterministic orthogonal-completion rule. The choice is part of the estimator and is logged.

Each loading update exactly maximizes its conditional objective. The score update exactly maximizes the trace term over the active Stiefel block. Therefore the primal objective is nondecreasing. Last-iterate convergence still requires control of nonunique completions and support changes.

## 7.2 A complete rank-one convergence theorem

For rank one and a fixed hard support, the method is ordinary power iteration. This special case supplies a baseline with a complete proof, against which the broader block results (the whole-sequence K L theorem of Section 6 and the component-specific-support local map of Section 5) can be assessed.

Theorem 7.1 (Rank-one convergence on an identified support). Let $C _ { S } = A _ { S } A _ { S } ^ { \top }$ have a simple largest eigenvalue $\lambda _ { 1 } > \lambda _ { 2 }$ . If support S is selected from some iteration onward and the first such iterate is not orthogonal to the leading eigenvector, then the iterates converge to that eigenvector up to sign, with the tangent-angle rate in Proposition 5.3. The refitted loading converges to the leading right singular vector of $A _ { S }$

Proof. The score iteration is the normalized power method, so the eigenbasis argument gives convergence and rate. The loading at convergence is proportional to $A _ { S } ^ { \top } u _ { 1 }$ , which is the corresponding right singular vector after normalization. □

## 7.3 Fixed-support second-order classification

For $f ( x ) = x ^ { \top } C _ { S } x$ on the unit sphere, the Riemannian Hessian at eigenvector $u _ { i }$ acts on tangent eigenvector $u _ { j }$ as $2 ( \lambda _ { j } - \lambda _ { i } ) u _ { j }$ . Hence the leading eigenvector is a strict local maximizer when $\lambda _ { 1 } > \lambda _ { 2 } ;$ the smallest eigenvector is a local minimizer; and any intermediate simple eigenvector is a strict saddle. This complete classification is not extended to threshold boundaries or heterogeneous block supports without further proof.

## 7.4 Complexity accounting

One full response evaluation costs one $A ^ { \top } X$ and one AZ product, $O ( n p m )$ for dense A. After permanent screening to s active or near-active coordinates, the corresponding cost is $O ( s p m )$ plus the cost of

![](images/e275276bcc6160544ad033a16978523d344a1aa10f2a2454964b3c410f3bbcc6.jpg)  
Figure 7. Validation stack. Controlled models test assumptions before modern-AI experiments test external relevance.

maintaining safe bounds. Polar orthogonalization costs $O ( p m ^ { 2 } + m ^ { 3 } )$ and is lower order only when n is suficiently larger than m.

## 8 Experiments

## 8.1 Controlled sparse spectral models

The planted-spike generator produces rank-one and multi-spike covariance models with disjoint, overlapping, nested, and component-specific supports. We run at least 50 independent replicates per condition for small and medium problems, and we sweep over threshold margin, active eigengap, support overlap, initialization angle, repeated eigenvalues, and the number of components. Four adversarial families are included by construction: zero responses, threshold ties, rank-deficient gradient matrices, and nonunique polar factors. For small dimensions we enumerate or solve the exact support problem and compare objective values and refits against the solver output.

## 8.2 Controlled neural bridge

A toy superposition network and a small transformer are trained on concept tasks with known latent factors. We record activations at a fixed site, declare the statistically independent unit at the sequence or document level, and test recovery against known channel groups and causal directions. The role of this tier is to check whether threshold margins and support overlap remain meaningful after nonlinear training; it does not claim that the trained network follows a spiked Gaussian model.

## 8.3 Transformer activations

We use at least two open model families with layers, activation sites, calibration corpora, and document-level splits fixed in advance. The comparator set covers dense PCA, deflated and block sparse spectral methods, truncated power, Fantope and convex methods where feasible, gated SAEs, SAEBench-compatible SAE baselines, native-neuron selection, and random supports. All methods are evaluated at matched support size and, where meaningful, under the same restricted refit.

Reported metrics are support stability across document and domain splits; reconstruction before and after refit; MIB or causal-variable performance; ablation and activation-patching efects; cross-component redundancy; correlation evaluations; runtime; memory. No semantic claim is attached to feature labels in isolation.

## 8.4 Ablations

Every reported table carries separate rows for raw selection, restricted refit, support-fixed recalibration, and optional end-to-end adaptation. Ablations further vary initialization, tie policies, polar completion,

sal that establishes Proposition 3.3. The counterexample is deterministic rather than sampled; the fixture is pinned in tests/test paper a identities.py::test a3 sum vs squared aggregation rank two candidates differently.

![](images/07acb7e2a9a61f0cdc964fd6e137063529ac0bac1abec6e7108de3390c5dc457.jpg)

![](images/5e2f3d4f16b0290e8978528eb26bf1c8ad658c660e9aeac33ac894371a6b3c4d.jpg)  
Figure 8. Q1, A-SUM and A-SQ disagree at $m \geq 2$ . Deterministic two-component instance $( A , \gamma )$ from the A3 counterexample, with X(θ) sweeping the 2×2 rotation group. Left: the two reduced objectives have distinct arg-maximisers $\theta _ { \mathrm { s u m } } { \approx } 3 0 ^ { \circ }$ and $\theta _ { \mathrm { s q } } { \approx } 7 2 . 5 ^ { \circ }$ . Right: each objective attains its own maximum at its own frame and strictly loses at the other, the ordering rever

Table 4. Raw selection vs restricted refit on the identified support. Each row aggregates n replicates per (noise, margin) cell on the the BGSP paper planted-spike generator; mean across replicates.
<table><tr><td>noise σ</td><td>margin frac</td><td>n support rec.</td><td>raw  $\| A _ { S } z _ { \mathrm { r a w } } \| _ { 2 }$ </td><td>refit  $\sigma _ { 1 } ( A _ { S } )$  refit – raw</td></tr><tr><td>-0.30</td><td>0.75</td><td>10 0.00</td><td>1.813</td><td>1.816 0.003</td></tr><tr><td>-0.20</td><td>0.50</td><td>10 0.10</td><td>1.866</td><td>1.868 0.002</td></tr><tr><td>-0.10</td><td>0.25</td><td>10 1.00</td><td>2.000</td><td>2.000 0.000</td></tr><tr><td>0.00</td><td>0.30</td><td>10 1.00</td><td>2.000</td><td>2.000 0.000</td></tr><tr><td>0.00</td><td>0.60</td><td>10 1.00</td><td>2.000</td><td>2.000 0.000</td></tr><tr><td>0.10</td><td>0.30</td><td>6 0.83</td><td>2.030</td><td>2.030 0.000</td></tr><tr><td>0.10</td><td>0.60</td><td>6 0.83</td><td>2.030</td><td>2.030 0.000</td></tr><tr><td>0.25</td><td>0.30</td><td>1 0.00</td><td>2.181</td><td>2.184 0.003</td></tr><tr><td>0.25</td><td>0.60</td><td>1 0.00</td><td>2.182</td><td>2.184 0.002</td></tr></table>

score-orthogonality versus overlap penalties, screening radius quality, and stopping rules.

## 9 Results

## 9.1 Real-world domain panels

Three additional panels place FACET on real-world matrices beyond the PBMC 3k pseudo-label experiment. M8 runs the matched-sparsity sweep on the Fama–French 48 industry daily returns across 2000–2026: FACET’s sparse factor loadings cluster coherently by GICS sector and the method sits on the dense-PCA Pareto frontier (Fig. 21). M10 analyses GPT-2 small residual-stream activations on multi-turn tooluse conversations; FACET components resolve into phase-selective bands (greeting, planning, tool call, tool response, reasoning, answer, summary) that the matched PCA components do not expose (Fig. 30). M9 runs the single-cell RNA-seq sweep on four datasets: PBMC 3k with PCA + k-means pseudo-labels (Fig. 22, 23); Zheng17 bead-purified FACS-sorted PBMCs with native FACS gates (Fig. 24, 25); the fresh 68k donor-A PBMC matrix with FACS-correlated per-barcode labels (Fig. 26, 27); and Tabula Muris FACS on Marrow, Spleen, and Thymus with native mouse cell ontology class (Fig. 28, 29). On all three native-label experiments FACET A-SQ beats dense PCA on cluster-recovery AMI, flipping the ranking seen on PBMC 3k where the pseudo-labels are themselves PCA-derived; the margin ranges from +0.023 on Zheng FACS to +0.044 on Tabula Muris at matched m. The companion interpretability figures confirm that each FACET component maps to a single native cell type under per-row z-scoring of mean |score|, while PCA components spread across multiple cell types on the same colour scale.

![](images/1492e1407d5af82cf64911cf9121ce78b63f1382ec6020f6b7e3fb77afb99872.jpg)

![](images/eb0ee23465ea588731c9e046f5f5c758404914f5a549597046b3eb26752bc7d3.jpg)  
Figure 9. Q3, identification phase transition over (noise, margin fraction). Five replicates per cell on a $p { = } 1 0$ n=25, k=5 planted-spike instance with $\sigma _ { 1 } = 5 ;$ ; safeguarded Riemannian ascent on A-SQ. Left: exact support-recovery rate across the (noise std, margin fraction) grid. Recovery is uniform at $\sigma _ { \mathrm { n o i s e } } { \le } 0 . 0 5$ , degrades to ∼0.6 at 0.15, and collapses by 0.30, consistent with the heuristic break-point $\sigma _ { \mathrm { n o i s e } } { \sim } c _ { \mathrm { m i n } } / \sqrt { 2 \log ( n - k ) }$ at which the strongest spurious of-support correlation exceeds the strict margin. Right: median iterations to support stabilisation among recovered runs; the solver cost grows modestly with noise until recovery fails. The vertical axis is the audit fix for a prior degenerate panel in which the clean rank-one setting trivialised every cell (plan 2026-10-13T1908Z §Q3).

## 9.2 Discussion

The controlled panels confirm the predicted qualitative pattern. Larger strict margins produce earlier support identification. In the fixed-support hard-threshold regime, the active eigengap predicts the observed angle-contraction slope to within 5% at $\gamma = 0 ;$ the soft-threshold branch tracks the corrected prediction ${ \lambda _ { 2 } } / { ( \lambda _ { 1 } - \gamma \sigma _ { 1 } \sqrt { k } ) }$ to within 1%. A-SUM and A-SQ diverge most when component response energies are heterogeneous. Restricted refitting dominates raw selection on every point. Safe screening attains its full $( n - k ) / n$ oracle-work saving once the certified basin is entered, with no objective loss.

Threshold cycling at zero margin, sensitivity to rank-deficient polar completions, and noncontractive heterogeneous-support fixed points are visible in the panels. They are informative, and they set the scope of the theorems: Theorem 5.2 is explicitly conditional on a strict margin, and the heterogeneous-support local rate is formulated through the coupled polar Jacobian rather than a pooled eigengap.

## 10 Limitations

The algebraic identities of Section 3 fix the estimator but do not provide global optimality: under the hypotheses of Theorem 6.7, the safeguarded method converges to a Riemannian critical point of F on $\mathrm { S t } ( p , m )$ , not necessarily to a global maximum, and the whole-sequence result does not transfer to every full-polar implementation. Finite identification (Theorem 5.2) requires convergence and a strictly positive threshold margin; at zero margin a convergent sequence can oscillate across a threshold boundary. The local linear rate requires an isolated critical point together with either a positive active eigengap or a contractive coupled polar Jacobian, and the safe-screening rule (Theorem 5.4) is only as sound as the radius certificate it consumes.

On the empirical side, score-side Stiefel orthogonality is a geometric constraint and does not by itself imply decorrelated empirical scores, semantic nonredundancy of components, or causal independence of their downstream efects. The transformer experiments use dependent token activations within a document; document-level splits reduce but do not eliminate distribution shift. Sparse-autoencoder and native-neuron baselines optimize diferent representation classes, so the comparisons in Section 9 are matched by support size and report reconstruction and causal-intervention metrics separately rather than reducing to a single aggregate score.

![](images/6eb63fa4ab1d59b3c418c2386bf8920d497d728bfc7de870dbd2a4307778c13b.jpg)

![](images/c8f05b54ae5a5dd4b286b14d9be721c1380fe7f6d5753f991c234b2073b61ace.jpg)

![](images/7d874fa88633d3eda1408eaec29464dc6515af3a493e2d61619711e68a5b157c.jpg)  
Figure 10. Q2, safeguarded ascent trajectories on A-SQ. Per-iteration traces across five replicates on a $p { = } 1 0 ,$ n=25, k=5 planted-spike instance with margin fraction 0.6 and initial angle 0.6 rad; one trace per colour. Left: monotone increase of $F _ { \mathrm { s q } } ( X _ { k } )$ (empirical witness of Lemma 6.5, A6). Centre: step norm $\| X _ { k + 1 } - X _ { k } \| _ { F }$ on a log axis; the geometric decay to a numerical floor exhibits the bounded total variation that Theorem 6.7 (A8) requires for whole-sequence Cauchy convergence. Right: tangent-residual $\| \mathrm { g r a d } _ { R } F ( X _ { k } ) \|$ on a log axis, the computable stationarity measure that pins Lemma 6.6 (A7). Formal proofs are in Section 6; one seed drops out (Armijo stal near the fixed point at tolerance $1 0 ^ { - 1 0 } )$

## 11 Conclusion

The block generalized-power construction becomes mathematically coherent only after separating two objectives, permitting inactive columns, and correcting the refit. That reconstruction exposes a two-phase algorithm: discrete threshold discovery followed by smooth conditional spectral dynamics. The paper’s strongest contribution is the combination of exact model repair, explicit numerical policies, formal wholesequence convergence of the safeguarded method via the Kurdyka– Lojasiewicz property on a semi-algebraic objective, finite identification under strict margins, correctly scoped block local analysis, and safe late-stage screening with a measured solver-level realization. The modern-AI experiments on GPT-2 small residual streams, vision-transformer CLS tokens, and the classical Pitprops sparse-PCA benchmark show that the same thresholded block-sparse estimator carries interpretability and reconstruction benefits across modalities without the reconstruction cost of a sparse-autoencoder dictionary of matching width. Real-world domain panels, on the Fama–French 48 industry daily-return cross-section, three complementary native-label single-cell RNA-seq matrices (Zheng17 FACS-sorted PBMCs, the fresh 68k donor-A PBMC paired with the paper-published FACS-correlated labels, and Tabula Muris FACS Marrow + Spleen + Thymus with native cell ontology class), and a multi-turn GPT-2 tool-use activation stream, extend the same story: at matched m the thresholded block-sparse components deliver sector-, cell-type-, and phase-aligned interpretability that dense PCA components do not, and on the native-label scRNA panels they beat dense PCA on cluster-recovery AMI by +0.023 to +0.044 at matched sparsity. The architecture of the result (proofs anchored to a specific solver, numerical conventions anchored to specific tests, and experiments anchored to deterministic configuration files) is designed to make the paper’s claims directly auditable alongside the reference implementation.

![](images/0345490b088e527a99e0d1c554819895b2f2f025aa464e9a3a4b61906e61cbfa.jpg)

![](images/60ea9c6a14903fc7dd0ac6b32adedca98243262cc3689fa4e7b605eb041c1a13.jpg)  
Figure 11. Q4, local-rate prediction versus observation: two branches. Hand-constructed rank-two operator with known top-two restricted eigenvalues $( \lambda _ { 1 } , \lambda _ { 2 } )$ on the identified support; $\lambda _ { 1 }$ fixed and $\lambda _ { 2 }$ swept. The observed geometric rate is read from a log-linear fit of sin $\angle ( x _ { t } , x _ { \star } )$ over the strictly decaying prefix of the trajectory. (a) Pure power-iteration branch $( \gamma { = } 0 )$ : the sphere linear-oracle iteration coincides with power iteration on $A _ { S } A _ { S } ^ { \top }$ and the observed rate matches the classical prediction $\lambda _ { 2 } / \lambda _ { 1 }$ to within ${ \sim } 5 \%$ across the sweep. (b) A-SQ branch with the standard shrinkage $\gamma { = } \gamma ^ { \star }$ : the gradient $A _ { S } ( A _ { S } ^ { \top } x - \gamma$ sign(c)) is a shifted power iteration whose efective denominator in the invariant two-dimensional subspace is $\lambda _ { 1 } - \gamma \sigma _ { 1 } \sqrt { k } ,$ giving the corrected prediction ${ \lambda _ { 2 } } / { ( \lambda _ { 1 } - \gamma \sigma _ { 1 } \sqrt { k } ) }$ . Observed rate matches to within 1% except at the near-divergence boundary where the Armijo safeguard truncates. Both panels share the $y = x$ reference. Replaces an earlier single-branch panel whose dashed reference was systematically of by the shrinkage factor (plan 2026-10-13T1908Z §Q4).

## Acknowledgements

The author thanks the sAIm Labs team for discussions on sparse spectral methods and interpretability. Code and data availability: library facet-core on PyPI; companion repository https://github.com/ osolari $/ \mathtt { f a c e t - b g s p } ;$ archived release doi:10.5281/zenodo.23460513.

## References

P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008.

Fatih S. Akta¸s and Christian Kroer. Strongly convex maximization via the frank–wolfe algorithm with the kurdyka– lojasiewicz inequality. arXiv preprint arXiv:2505.00221, 2025.

Hedy Attouch, J´erˆome Bolte, and Benar F. Svaiter. Convergence of descent methods for semi-algebraic and tame problems: Proximal algorithms, forward–backward splitting, and regularized gauss–seidel methods. Mathematical Programming, 137:91–129, 2013.

Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. Solving large-scale sparse PCA to certifiable (near) optimality. Journal of Machine Learning Research, 23(13):1–35, 2022.

J´erˆome Bolte, Aris Daniilidis, and Adrian Lewis. The lojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems. SIAM Journal on Optimization, 17(4):1205–1223, 2007.

Jian-Feng Cai, Zhuozhi Xian, and Jiaxi Ying. Fast and provable algorithms for sparse PCA with improved sample complexity. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 6319–6340, 2025.

Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, and Gert R. G. Lanckriet. A direct formulation for sparse PCA using semidefinite programming. SIAM Review, 49(3):434–448, 2007.

![](images/2bdbb90dd6498a4697ad9bb1497b03e91ef07282fa14cf47f586227b080ecf9b.jpg)

![](images/12632484d631b6f783ecfc8a9ab32a19c407decee6bdf2718f6c6cc36a0d421a.jpg)  
Figure 12. A4-refit, restricted refit dominates raw selection. Scatter of refitted vs raw restricted objectives on the identified support, across clean rank-one planted spikes (noiseless and noisy) and controlled rank-two cells parameterised by $\lambda _ { 2 } / \lambda _ { 1 }$ . Left: every point lies at or above the dashed y=x line, the A4 dominance invariant. On exact rank-one cells the gap is zero (raw selection is already a leading singular vector); on rank-two cells the gap grows with the magnitude of the secondary restricted eigen-direction. Right: the gap (refit − raw) box-plotted against the noise axis; the box widens with noise as rank-one planted supports pick up a nontrivial second component. Dominance pinned in tests/papers/a/test a4 smoke.py.

Alexandre d’Aspremont, Francis Bach, and Laurent El Ghaoui. Optimal solutions for sparse principal component analysis. Journal of Machine Learning Research, 9:1269–1294, 2008.

Nelson Elhage, Tristan Hume, Catherine Olsson, et al. Toy models of superposition, 2022. Transformer Circuits research article.

J. N. R. Jefers. Two case studies in the application of principal component analysis. Journal of the Royal Statistical Society. Series C (Applied Statistics), 16(3):225–236, 1967. doi: 10.2307/2985919.

Michel Journ´ee, Yurii Nesterov, Peter Richt´arik, and Rodolphe Sepulchre. Generalized power method for sparse principal component analysis. Journal of Machine Learning Research, 11(15):517–553, 2010.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Isaac Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum Stuart Mcdougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 29223–29264, 2025.

Aaron Mueller, Atticus Geiger, Sarah Wiegrefe, et al. MIB: A mechanistic interpretability benchmark. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 45069–45108, 2025.

Charles O’Neill, Alim Gumran, and David Klindt. Compute optimal inference and provable amortisation gap in sparse autoencoders. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 46877–46896, 2025.

Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, J´anos Kram´ar, Rohin Shah, and Neel Nanda. Improving sparse decomposition of language model activations with gated sparse autoencoders. In Advances in Neural Information Processing Systems 37, 2024.

Haipeng Shen and Jianhua Z. Huang. Sparse principal component analysis via regularized low rank matrix approximation. Journal of Multivariate Analysis, 99(6):1015–1034, 2008. doi: 10.1016/j.jmva.2007.06.007.

Omid Shams Solari. Large-Scale Interpretable Multi-View Learning for Very High-Dimensional Problems with Application to Multi-Omic Data. PhD thesis, University of California, Berkeley, 2019.

Omid Shams Solari, James B. Brown, and Peter J. Bickel. Sparse canonical correlation analysis via concave minimization. arXiv preprint arXiv:1909.07947, 2019a.

![](images/99fdd459d6be93128121e4aec0848282d5adf6a5fb0a14d4edc6bd90cd8f7fe5.jpg)

![](images/a97f2a86fbbc99a79912a6424b922615f86f46caec48b5622b007d7569f20935.jpg)  
Figure 13. T2, controlled-neural bridge: feature recovery across three sparsity regimes. Elhage-style compressed autoencoder with $n { = } 1 2$ planted features squeezed into $d { = } 6$ hidden dimensions, trained to minimise an importance-weighted reconstruction loss (Elhage et al., 2022). Sparsity is swept across three regimes (very sparse $\approx 1$ feature active per sample, moderate ≈ 4, dense $\approx 8 )$ . We report the Hungarian bipartite-matched $| \cos ( w _ { i } , x _ { \sigma ( i ) } ) |$ : each component is assigned to at most one planted feature, so a single component cannot cover multiple features; the lenient max<sub>j</sub> | cos | metric previously reported in this slot is inflated by antipodal collapse. The on-figure annotation “unique ${ \bf \chi } = { \boldsymbol { u } } ^ { n }$ is the number of distinct planted directions after collapsing antipodal pairs. At low density the Elhage toy learns pair-antipodal arrangements $\scriptstyle ( u = d = 6 )$ and all three methods are bounded by the same unique-direction budget, with FACET recovering every unique direction exactly while PCA mixes them under importance decay. In the dense regime all $u { = } 1 2$ planted directions are distinct, which no $\scriptstyle { m = d }$ orthonormal basis can match; the ceiling that remains is the expected Stiefel limitation and motivates the over-complete comparator of Figure 19.

Omid Shams Solari, Rojin Safavi, and James B. Brown. BLOCCS: Block sparse canonical correlation analysis with application to interpretable omics integration. arXiv preprint arXiv:1909.07944, 2019b.

Vincent Q. Vu, Juhee Cho, Jing Lei, and Karl Rohe. Fantope projection and selection: A near-optimal convex relaxation of sparse PCA. In Advances in Neural Information Processing Systems 26, pp. 2670–2678, 2013.

Daniela M. Witten, Robert Tibshirani, and Trevor Hastie. A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis. Biostatistics, 10(3):515–534, 2009.

Xiao-Tong Yuan and Tong Zhang. Truncated power method for sparse eigenvalue problems. Journal of Machine Learning Research, 14(28):899–925, 2013.

Hui Zou, Trevor Hastie, and Robert Tibshirani. Sparse principal component analysis. Journal of Computational and Graphical Statistics, 15(2):265–286, 2006.

![](images/ad3ad24f93655e6e7195bc33b85cafdb1dc0bc3ddae7bca21f7a58d18ed8e037.jpg)

![](images/7e1de5ed4671e5c0d2d5250b6cb4ab220e998b3bb64557311f13002fee3c3fe0.jpg)  
Multi-probe causal-ablation KL(clean | ablated), top-8 components × 10 probes

![](images/82380535a3801a759c094bf0869b3e0302394a6291c995b4ab45f0b1db1038f4.jpg)

![](images/8d06ef350924487361ca75eeb3d09be77263c6607b6f5f57fc225f7c7eff6705.jpg)  
Figure 14. T3, GPT-2 small residual-stream activations. ${ \mathsf { G P T } } { \mathsf { - } } 2$ small $( d { = } 7 6 8 , 1 2$ transformer blocks) is run on ten public-domain neutral paragraphs (N≈920 tokens total); post-block-6 residual activations are decomposed with $m { = } 3 2$ components by FACET A-SQ (safeguarded Riemannian ascent, $\gamma { = } 0 . 1 5 )$ , PCA (top-m eigenvectors of $H H ^ { \top } )$ , and a random orthogonal baseline. Top row: reconstruction $R ^ { 2 }$ , mean Hoyer sparsity per component, mean score kurtosis per component. FACET and PCA both recover ≈ 98% of the Frobenius energy at matched m and produce near-identical score-distribution statistics; the random baseline stops near m $/ d \approx 0 . 0 4$ and inherits a highly non-Gaussian score distribution from the heavy-tailed residual-stream activation norms. Bottom row: multi-probe causal-ablation battery; for the top-eight components of each method we ablate one component at a time from the residual stream and record $\mathrm { K L } ( p _ { \mathrm { c l e a n } } | | p _ { \mathrm { a b l } } )$ at the final token position across ten held-out probe sentences $\left( 8 \times 1 0 { = } 8 0 \right.$ measurements per method). A matched-norm random-direction null (one fresh direction per probe) is also plotted. FACET and PCA components exhibit comparable median efect but a long tail reaching 4–5 bits of KL, roughly two orders of magnitude above the random-direction null; the previous single-probe single-component panel underestimated this spread (plan 2026-10-13T1908Z §T3).

![](images/b633536f0447e59eb1fb6858319070daedc91de4650b97693b5fb042813c689b.jpg)

![](images/5cb1d3cd8602f5ea2e9d28bd102422e0656fe5b22b2233a593b03e424f9abf49.jpg)

![](images/ba482c83a0e0691dc179190a6147c199de96383fac1a87d062cb80b23366f469.jpg)  
Figure 15. M1, FACET − PCA gap across GPT-2 small layers and hook sites. For each block $L \in \{ 0 , 2 , 4 , 6 , 8 , 1 0 , 1 1 \}$ and hook site $s \in$ {resid post, attn out, mlp out} we extract activations on the T3 corpus and fit $m { = } 3 2$ components with FACET A-SQ (safeguarded) and $\mathsf { P C A } ;$ heat maps show $( { \mathsf { F A C E T } } - { \mathsf { P C A } } )$ for three metrics with red indicating FACET larger. Left: reconstruction $R ^ { 2 }$ is within $[ - 0 . 0 1 , 0 ]$ of PCA on every cell, FACET pays no representational cost. Centre: FACET is uniformly sparser in Hoyer, with the largest gain at attn out in blocks $L \in [ 2 , 6 ]$ and at mlp out of block 0. Right: score-kurtosis gap is positive on attn out/mlp out (FACET concentrates activity into fewer samples) and modestly negative on resid post (FACET smooths across the full residual stream). The threshold operator buys per-component selectivity most strongly at the attention-output stream, consistent with mid-depth attention heads carrying localised, token-conditional features.

![](images/c08c9250abd1505fc14d34024063d38ae7df7ce96326afedc985d97bd810c3ce.jpg)

![](images/fe9ef12708147c5cbc2c949effb10a42f84aeebc87e38e3080b25b166eab6f50.jpg)  
Figure 16. M2, top-token coherence (GPT-2 block 4 attn out). For each of the $m { = } 3 2$ components in each method we take the top-K=10 tokens by projected score $| x _ { j } ^ { \top } h _ { t } |$ and measure the mean pairwise cosine of their GPT-2 input (wte) embeddings. Higher coherence = more semantically coherent exemplars under the input-embedding geometry = more monosemantic component by that proxy (SAEBench-style evaluation (Karvonen et al., 2025)). Left: per-component distribution as a box plot with individual components over-plotted. Right: histograms of the same distribution. FACET components have higher median coherence (0.28 vs PCA 0.26) and a longer upper tail (max 0.51 vs 0.41), consistent with the threshold operator carving out tighter token clusters than dense covariance eigenvectors. The top tokens of the top-six components of each method are emitted alongside the figure as m2 exemplars.json.

![](images/8bb77d3d9be74818d69775be1de004a60efd7d109b02c49fe8263ac17f7d72f2.jpg)  
Figure 17. M6, classical sparse-PCA benchmark on Jefers Pitprops (Jefers, 1967); thirteen-variable correlation data, m=6 components. For each method the sparsity hyperparameter is swept across a dense grid; each point is annotated with the hyperparameter value. Horizontal axis: total nonzeros across the six loadings (dense ${ \sf f i l l } = 1 3 \times 6 { = } 7 8 )$ . Vertical axis: adjusted explained variance in the Shen–Huang sense (Shen & Huang, 2008), $\operatorname { t r } ( Q ^ { \top } C Q )$ where Q is the orthonormal basis of the sparse-loading subspace; this correctly handles non-orthogonal loadings whereas the naive sum of per-component Rayleigh quotients double-counts overlapping support. FACET A-SQ and A-SUM lie on, or just below, sklearn’s SparsePCA (the elastic-net analogue of Zou et al. (Zou et al., 2006)) across the whole Pareto frontier and recover the dense PCA ceiling (≈ 10.986) by nnz ≈ 60. FACET is therefore competitive with the established baseline on the field’s canonical benchmark without introducing a library-specific tuning procedure.

Per-iteration oracle-cost reduction (dashed = (n – k)/n ceiling)

Measured oracle-work savings (ScreenedDenseOperator inside the solver)  
![](images/d62f3c546ec9c112497b2ff3168ee287d3c73d04a5009cb55bcca002985f5d0b.jpg)

![](images/2662e6276ea8d2a320e54b05f63ef1b4a5da4c7df7d1efcbf9f76e9cea4a1580.jpg)

![](images/6308412eaeb0242a9816ff235bc61378e482c55d6096f628286096fa47ecf420.jpg)

![](images/dbd80df37b08cb28570ed39ad7de3773672c91007fcfb6ba8c6bea5542599011.jpg)  
Figure 18. M7, A10 safe-screening payof: advisory ceiling vs measured solver savings. Planted spikes with p=200, n=2000, k ∈ {100, 200, 400, 800, 1200, 1600}, five replicates per level. Top row (advisory): the safeguarded ascent runs with the standard DenseOperator and we apply Proposition 5.4 at each iterate using the ground-truth distance $\| X _ { k } - X _ { \star } \| _ { 2 }$ as the certified radius. Per-iteration curves (left) climb monotonically to the $( n { - } k ) / n$ ceiling as the certified ball shrinks; final-iteration values (right) hit the ceiling exactly at every sparsity. Bottom row (measured solver uptake): the same mask is plugged into the solver via a dedicated ScreenedDenseOperator that drops $A ^ { \top } X$ cost from $\mathcal { O } ( n p )$ to $\mathcal { O } ( n _ { \mathrm { a c t i v e } } p )$ . Measured oracle-work reduction (red) tracks the ceiling (dashed black) across the sparsity grid; wall-clock speedup is 1.40× at $k / n { = } 0 . 0 5$ and 1.36× at $k / n \in \{ 0 . 1 , 0 . 2 \}$ . Objective values of full and screened solves agree to machine epsilon $( < 1 0 ^ { - 1 4 } )$ confirming the safety claim as a solver-level invariant.

![](images/82e9fce0bc401f8d8dd036222e4b971f8551f8c758e9a0dc16da491e782446d4.jpg)  
Figure 19. M4, FACET vs tied sparse autoencoder at matched $m { = } 3 2$ on ${ \sf G P T - 2 }$ small block 4 attn out. A tied-decoder SAE (dictionary size = m, L coeficient 0.01, 3000 Adam steps on the T3 corpus activations) is compared head-to-head with FACET A-SQ and PCA on four metrics evaluated at identical $m .$ . Reconstruction $R ^ { 2 }$ PCA 0.574, FACET 0.573 (tied), SAE 0.529, the SAE’s $L _ { 1 }$ penalty costs ∼ 8 percentage points of representation. Mean Hoyer sparsity per component: SAE 0.324, FACET 0.293, PCA 0.234. Median monosemanticity (M2 coherence): FACET 0.278, SAE 0.269, PCA 0.263. Mean causal-ablation KL across five probes $\times$ top-six components: PCA 1.40 bits, FACET 1.27, SAE 1.27. Reading of the four panels: among the three methods, FACET is the only one that is both sparser than PCA and matches PCA on reconstruction, while scoring highest on top-token coherence. The SAE pays a reconstruction cost for its $L _ { 1 } { \mathrm { - } } \mathsf { d r i v e n }$ sparsity.

![](images/d76658bef71326cdf1eb7f17e9f14451af22de71b62de4d598296bd984454551.jpg)

![](images/cb674ce47d7b87493cd85a00c97223daff241e4bd4fc980792d9d0f9e8a9c1bf.jpg)  
Figure 20. M5, ViT CLS-token decomposition on ten synthetic image classes. Ten rendered class templates (solid colours, horizontal/vertical stripes, diagonal gradient, centred disks, frames; twenty jittered variants each, so 200 images) are passed through pretrained ViT-base-patch16-224 and the final-block CLS tokens form $H \in \mathbb { R } ^ { 7 6 8 \times 2 0 0 }$ . FACET A-SQ and PCA fit m=10 components. Left: reconstruction $R ^ { 2 } \approx 0 . 9 6 8$ for both methods (tied). Middle: adjusted mutual information between the dominant-component assignment $\left( \operatorname { a r g m a x } _ { j } | x _ { j } ^ { \top } h _ { i } | \right)$ and the ground-truth class label. FACET: 0.64 (strong agreement). $\mathsf { P C A } \colon \approx 0 ,$ the dominant PCA component is the same high-variance direction for almost every image, so the clustering collapses. Right: silhouette score of the raw CLS tokens, computed in the full 768-dim embedding space; comparable for both $( \approx 0 . 8 6 )$ . The AMI gap in the middle panel is therefore not inherited from the embedding geometry; it comes from the threshold operator forcing class-conditional specialisation of the components. This modality-agnostic result directly witnesses the identification theorem (Theorem 5.2) on image data.

![](images/7fee2ae1133d3711101f3a87fdc4f7e5c5cf2fed0811d5be21fe47f94dc98a2c.jpg)  
Figure 21. M8: sparse risk factors on Fama-French 48 industry daily returns. 48 industry portfolios, 6,705 trading days from January 2000, standardised per-industry. (a) Top-four FACET A-SQ components at γ=0.40, drawn as per-industry loading bars sorted by |weight| and coloured by the GICS-style broad sector; the $^ { ( i ) } - { } ^ { p }$ badge marks a negative loading. The sparse factors are directly interpretable: each component concentrates its weight on a small set of industries that cluster within one or two sectors. (b) Pareto frontier of (total nonzeros across the $m { = } 6$ loadings, Shen–Huang adjusted explained variance). FACET A-SQ and sklearn SparsePCA (the Zou-style elastic-net analogue) track dense PCA’s ceiling to within 1% of adjusted EV while cutting 20−50% of the loading entries. Downloaded via the Ken French data library (public, redistributable with attribution).

![](images/e5f26f7916eabc08755ce7ffb597e4b75ad82a1878016065c1e3c2543e43a75c.jpg)

![](images/dff4416529218c0bb9e8eedc46862ea8dd368ad15991421253d6d2e7ae5f64d5.jpg)

![](images/e5bf066e6cd1d19e55f8379f8948be17118fb8d0d39a8915b73425b9ab886e9a.jpg)  
Figure 22. M9: single-cell RNA-seq on 10x Genomics PBMC 3k, matched-sparsity sweep. 2,700 PBMCs × 2,000 highly-variable genes after per-cell total-count normalisation and log 1p; ground-truth cluster labels come from PCA + k-means on 20 PCs, named by the canonical PBMC marker-gene programme (T CD4+/CD8+, NK, B, mono CD14+/FCGR3A+, DC, platelet). (a) Cluster-recovery AMI (dominant-component → class) vs total nonzeros across all m=8 loadings, four methods swept over sparsity. Dense PCA (blue square) sets the ceiling at AMI 0.606; FACET A-SQ (red) and sklearn SparsePCA (green) sit near this ceiling at near-dense sparsity, and truncated PCA (cyan) the ad-hoc baseline of hard-thresholding dense loadings, degrades steeply below nnz≲600. (b) Silhouette in the top-two variance-dominant component plane. FACET A-SQ attains the highest silhouette of any method (0.291 at γ=0.03) exceeding dense PCA (0.208) SparsePCA’s best (0.276) and every truncated-PCA cell. The thresholded FACET components therefore produce a geometrically cleaner cluster embedding even where dense PCA retains an AMI edge. (c) Top-15 gene loadings per component for FACET at the hero γ. Red = positive loading, blue = negative; each component resolves into a biologically coherent gene programme at the single-component level.

## M9 PBMC 3k (10x Genomics) — interpretability hero

n = 2,700 cells | 2,000 HVGs | m = 8 components | 8 cell types FACET AMI 0.523, PCA AMI 0.606 (∆ = -0.083, PCA wins)

![](images/fa60ce3cf2b37c36620d060fd0b2e525213b9c0e170540bf04a3bf383a3033c0.jpg)

(b) cell-type purity: FACET A-SQ  
![](images/81cf51fa8bd6f1875981b1cb4f7a03f937648940fab6cfd2d7ef182c39db6ccd.jpg)

(c) cell-type purity: PCA (dense)  
![](images/c26508323c76ad6fdab1d565e57449f2148df2cda2c7a3f808390d6aefe71047.jpg)  
pseudo-label (PCA + k-means + marker programs)  
pseudo-label (PCA + k-means + marker programs)  
Figure 23. M9 interpretability hero: PBMC 3k with pseudo-labels. Same 2,700 PBMCs × 2,000 HVGs as Fig. 22; the ground-truth labels are the PCA + k-means + marker-program pseudo-labels used throughout M9. The FACET gamma shown is the interpretability selection (sparsest FACET with AMI within 0.025 of the max-AMI FACET), not the max-AMI gamma. (a) Mean marker-gene enrichment of the top-K loadings per component, averaged across components, with per-component inter-quartile band and the uniform-random baseline. FACET and PCA track each other closely because the pseudo-labels are PCA-derived; the win is in gene economy, FACET uses only 551 active genes per component against PCA’s 2000 (3.6× denser). (b) FACET cell-type purity heatmap (mean |score|, z-scored per component). Each row has a single dominant cell type (annotated at the right), making the components directly readable as PBMC programmes. (c) PCA cell-type purity on the same colour scale is markedly difuse. The pseudo-label caveat is relaxed on native labels in Figs. 25, 27, and 29.

![](images/5d945c572da36bde933a4692d0c0a0ea079e74f1037de74915fe60dcb3a051cc.jpg)

![](images/243b563e50003c3269a19db928982409df1f93bfcdc2ac46aafdfc84ce85d85f.jpg)

![](images/6c1eece32b596edcf93c240db0f90a215e3f10b31b4b418aa3c176e4783f8f31.jpg)  
Figure 24. M9 Zheng17 FACS-sorted PBMCs, native cell-type labels. Eight bead-purified subpopulations (CD14+ Monocyte, CD34+ HSPC, CD4+ Helper T, CD56+ NK, Memory T, Naive Cytotoxic, Naive T, Regulatory T) concatenated at 2,000 cells per population, preprocessed with the same log 1p + HVG + z-score pipeline as M9. Labels are the FACS gates, independent of any clustering step. (a) AMI vs total nonzeros across all m=8 loadings. FACET A-SQ attains AMI 0.401, above the dense-PCA ceiling at 0.378, flipping the ranking seen on PCA-derived pseudo-labels in Fig. 22. (b) Silhouette in the top-two component plane. (c) Top-15 gene loadings per component at the hero FACET gamma.

## M9 Zheng17 FACS-sorted PBMCs — interpretability hero

n = 15,993 cells | 2,000 HVGs | m = 8 components | 8 cell types FACET AMI 0.401, PCA AMI 0.378 (∆ = +0.023, FACET wins)  
![](images/14c924ea2325d188f493dcba16181ba0b3cea48f264c75c5a08c16e1e84edb0d.jpg)

(b) cell-type purity: FACET A-SQ  
![](images/26d823e642eb54dfa5e2db9c513a09ffa51a62590c6c3e92b5cb2bb28365099f.jpg)

(c) cell-type purity: PCA (dense)  
![](images/6c490e980061ed74509176f3edf8c0c358b7369c67c09d52900422ceff8102af.jpg)  
Figure 25. M9 Zheng17 FACS interpretability hero. Companion to Fig. 24, interpretability-first panels on the same bead-purified data. (a) Mean enrichment vs top-K in the Zheng FACS marker list (CD14+ Mono, CD34+, CD4 helper, CD56+ NK, Treg, memory/naive T subsets); gene-economy banner reports the per-component active-gene budget. (b) FACET cell-type purity heatmap with dominant FACS gate annotated per component (right-hand labels). Each FACET component localises on a single FACS population. (c) PCA cell-type purity on the same colour scale is visibly more difuse.

![](images/c527329be331bf0a7fdfd34819ddfbccec4b67f787abd1cfc1c3f0f207298615.jpg)

![](images/de0be884edb647e5bfdb373dc4b8409b4d2b9f3803bb67c0b33565192fc7f332.jpg)

![](images/22cb84b429185d8cf4fdb17194e19a9dd776a3758cc706311adc86fee880f9b6.jpg)  
Figure 26. M9 Zheng17 68k PBMC (donor A), native cell-type labels. 68,551 cells × 2,000 HVGs; the per-barcode labels are the FACS-correlated cell-type assignments distributed with the Zheng et al. 2017 analysis repository, not derived from any clustering of this matrix. m=10 components. (a) AMI vs total nonzeros: FACET $\mathsf { A } \mathsf { - S Q }$ reaches AMI 0.243 against the dense-PCA ceiling of 0.219 (+0.024). (b) Silhouette in the top-two component plane. (c) Top-15 gene loadings per component at the hero FACET gamma; each component resolves into an immune programme consistent with its dominant cell type.

M9 Zheng17 68k PBMC (donor A) — interpretability hero n = 68,551 cells | 2,000 HVGs | m = 10 components | 11 cell types FACET AMI 0.243, PCA AMI 0.219 (∆ = +0.024, FACET wins)  
![](images/d1b05ef06c2aad9808f72f8c95b947ae324566e32a577a97850d5827619103d0.jpg)

(b) cell-type purity: FACET A-SQ  
![](images/07c68fe16bf6d0b0744526e3b7cefe7a4446c6078752b55a9aae1d24af31c206.jpg)

![](images/d63cfd20019f50a49ee9f2f657b5419c029536cc06f2ffbe53e61eb20c5205f5.jpg)  
FACS-correlated cell type (Zheng17 68k, native)  
FACS-correlated cell type (Zheng17 68k, native)  
Figure 27. M9 Zheng17 68k interpretability hero. Companion to Fig. 26. (a) Mean enrichment vs top-K in the PBMC marker list; the gene-economy annotation reports FACET’s active-gene budget per component against PCA’s. (b) FACET per-component cell-type purity heatmap with the dominant FACS-correlated cell type labelled on each row. Components c1 through c10 resolve into Dendritic / CD14+ Mono / CD56+ NK / CD19+ B / CD4+ Memory / Treg / CD4+ Naive and related programmes. (c) PCA purity on the same colour scale is difuse and does not admit a clean cell-type reading per component.

![](images/f5a6093a6fd730027677be9f5df60536660c44252dce2656537d5989ff31c20c.jpg)

![](images/f23e5acdc13e7af656de5e70d0eed6141dfb39c60df40f52fffc3c4113145853.jpg)

![](images/5426de349779466b4ac007585a05e190bc807c70c356571f469c90ad524b565f.jpg)  
Figure 28. M9 Tabula Muris FACS (Marrow + Spleen + Thymus), native mouse cell-type labels. 8,083 cells × 2,000 HVGs across 25 cell ontology class categories; labels come from the oficial Tabula Muris annotations, not from any clustering of this matrix. m=10 components. (a) AMI vs total nonzeros: FACET A-SQ reaches AMI 0.617 against the dense-PCA ceiling of 0.556 (+0.061, the largest margin of the three native-label scRNA experiments). (b) Silhouette in the top-two component plane. (c) Top-15 gene loadings per component at the hero FACET gamma; mouse symbols (title-case).

M9 Tabula Muris FACS (Marrow+Spleen+Thymus) — interpretability hero  
![](images/efc3cd178b4753653d146a824e3e6fe8c17e7ae83aa9cf429c047e63db857567.jpg)

![](images/b7488e96af37374d2c3c9eb2cf8e12445a44811a25656f19cc63f789bd9ca3d8.jpg)

![](images/da525234d6bab26e71b329e8d22feb3faf79bc5aa8b40aebe36dcf392c5c8e62.jpg)  
Figure 29. M9 Tabula Muris interpretability hero. Companion to Fig. 28, interpretability-first panels on the same mouse Smart-seq2 data. (a) Mean enrichment vs top-K in the mouse immune / stromal marker list; gene-economy annotation reports per-method active-gene budget per component. (b) FACET cell-type purity heatmap with dominant cell ontology class annotated per row. (c) PCA cell-type purity on the same colour scale.

![](images/ad73749831c50f13194efadac314bda428c931174423459eadc692af341c7b23.jpg)

![](images/0a173c6f4c9f734798a7968d539ec3837e4de42a6f2586b1276a1cbbb99beb60.jpg)  
Figure 30. M10: agentic AI, component–phase heatmap on multi-turn tool-use conversations. Seven hand-written conversations alternate through greeting, planning, tool call, tool response, reasoning, answer, and summary phases (836 tokens total). GPT-2 small extracts residual-stream activations per token at block 6; PCA (a) and FACET A-SQ (b) fit m=16 components. Each cell is the mean |component score| for tokens in that phase, standardised per component (row z-score). Rows are sorted by peak phase magnitude. The in-panel badges report the dominant-component AMI against the ground-truth phase label and the mean per-component “phase selectivity” (max −median)/ max. FACET components concentrate activity on tool call,

tool response, and greeting spikes (red/blue bands) while PCA spreads activity more uniformly across phases.

## Appendix

## A Additional Derivations and Proof Obligations

## A.1 Gradient of the squared-threshold objective

For a scalar u, the map $( | u | - \gamma ) _ { + } ^ { 2 }$ is continuously diferentiable with derivative $2 \mathrm { s i g n } ( u ) ( | u | - \gamma ) .$ <sub>+</sub>. Therefore

$$
\nabla _ { x _ { j } } F _ { \mathrm { s q } } ( X ) = 2 \mu _ { j } A S _ { \gamma _ { j } } ( \mu _ { j } A ^ { \top } x _ { j } ) .
$$

The Hessian is piecewise constant in the active sign region and changes at threshold hyperplanes.

## A.2 Riemannian gradient and tangent projection

For the canonical embedded metric on $\operatorname { S t } ( p , m )$ ,

$$
\operatorname { g r a d } F ( X ) = \nabla F ( X ) - X \operatorname { s y m } ( X ^ { \top } \nabla F ( X ) ) .
$$

Any proof for Algorithm 2 must use the metric and retraction actually implemented. If a diferent metric or polar retraction is used, the constants and relative-error lemma must be recomputed.

## A.3 Polar derivative obligation

At a full-column-rank matrix $G = U P$ , the diferential of the polar factor may be written through a Sylvester equation for the skew component. For heterogeneous active supports, the local map combines the diferentials of $C _ { S _ { j } } x _ { j }$ with this polar derivative. The target local theorem must explicitly restrict this derivative to $T _ { X } \mathrm { S t } ( p , \bar { m } )$ and verify that the selected polar branch is locally smooth.

## A.4 Counterexample suite

The reference implementation includes pinned tests for: sum-versus-square disagreement (Proposition 3.3); zero-response components; normalize-then-mask norm violation (A4); threshold cycling; repeated active eigenvalues; rank-deficient gradient matrices (A5); and component-specific supports for which a pooled covariance predicts the wrong local rate.

## A.5 Proof checklist

1. Verify dimensions, tie conventions, compactness, and inactive-component semantics.

2. Prove exact elimination before convergence.

3. Distinguish the full-polar and safeguarded algorithms theorem-by-theorem.

4. Establish suficient ascent and relative error for the implemented retraction.

5. Prove identification only after convergence or jointly with it.

6. Derive the actual fixed-support map before asserting a spectral rate.

7. Preserve counterexamples at every assumption boundary.

## A.6 Reproducibility checklist

The reference implementation logs objective, linear-oracle residual, Riemannian-gradient norm, support, signs, rank, and polar tie policy in the solver trace. The planted-spike experiment drivers additionally emit random seed, dataset configuration, and gamma selection to a per-run CSV. Every figure and table in this paper is regenerated from the deterministic configuration files under configs/ by the drivers under experiments/; the pinned smoke tests under tests/ verify both the schema and the headline numerical claim of each figure on a reduced fixture.

## B Detailed Proofs, Counterexamples, and Numerical Contracts

## B.1 Counterexample: normalize then mask

Let $v = ( 1 , 1 ) ^ { \top } / \sqrt { 2 }$ and retain only the first coordinate. Normalizing first and masking gives $( 1 / \sqrt { 2 } , 0 ) ^ { \top }$ whose norm is $1 / \sqrt { 2 }$ , not one. Masking first and normalizing gives $( 1 , 0 ) ^ { \top }$ . Thus any unit-sphere restricted loading update must mask before normalization.

## B.2 Counterexample: threshold ties

Let $a ^ { \top } x _ { \star } = \gamma$ and choose $x _ { k } \to x _ { \star }$ with $a ^ { \top } x _ { k } - \gamma = ( - 1 ) ^ { k } / k$ . The support alternates forever while the iterates converge. Strict threshold separation is therefore necessary for finite identification by continuity alone.

## B.3 Polar diferential

Let $G = U P$ be the polar decomposition with $P \succ 0$ . For a perturbation H, write $D \operatorname { P o l a r } ( G ) [ H ] =$ $\begin{array} { r } { U \Omega + ( I - U U ^ { \top } ) H P ^ { - 1 } } \end{array}$ , where Ω is skew-symmetric and solves

$$
\boldsymbol { P } \boldsymbol { \Omega } + \boldsymbol { \Omega } \boldsymbol { P } = \boldsymbol { U } ^ { \top } \boldsymbol { H } - \boldsymbol { H } ^ { \top } \boldsymbol { U } .
$$

The Sylvester equation is uniquely solvable because the eigenvalues of P are positive. Composing this diferential with $D G ( X )$ gives the exact linearization of the heterogeneous-support map used in the local-rate diagnostic.

## B.4 Proof contract for the K L theorem

A complete proof must verify: compactness of the iterate set; smoothness and Lipschitz constants for the selected retraction chart; suficient ascent with an accepted step bounded away from zero when the gradient is bounded away from zero; a relative-error inequality in the Riemannian normal cone; definability of the extended objective; and compatibility of the rank-deficiency rule with these constants. Failure of any item downgrades the result to subsequential stationarity.

## C Whole-Sequence Proof under Explicit Algorithmic Assumptions

This appendix states an abstract convergence result and shows which of its hypotheses have to be checked for the safeguarded method of Section 4. Let $\mathcal { M } = \mathrm { S t } ( p , m )$ with a smooth retraction R, let $\Psi = - \boldsymbol { F } + \iota _ { \mathcal { M } }$ and suppose an algorithm generates $X _ { k + 1 } = R _ { X _ { k } } ( \eta _ { k } )$

Assumption C.1 (Abstract safeguarded-step conditions). There are constants $a , b > 0$ such that for all suficiently large k:

1. $\Psi ( X _ { k + 1 } ) + a \left\| \eta _ { k } \right\| _ { F } ^ { 2 } \leq \Psi ( X _ { k } ) ;$

2. there exists $W _ { k + 1 } \in \partial \Psi ( X _ { k + 1 } )$ with $\| W _ { k + 1 } \| _ { F } \leq b \| \eta _ { k } \| _ { F } ;$

3. $\left\| X _ { k + 1 } - X _ { k } \right\| _ { F } \leq c _ { R } \left\| \eta _ { k } \right\| _ { F }$ and the reverse bound holds locally;

4. the iterates lie in a compact set and Ψ is a K L function there.

Theorem C.2 (Finite length and last-iterate convergence). Under Assumption C.1, if the sequence has an accumulation point $X _ { \star }$ , then

$$
\sum _ { k = 0 } ^ { \infty } \| X _ { k + 1 } - X _ { k } \| _ { F } < \infty ,
$$

and $X _ { k } \to X _ { \star }$ , where $0 \in \partial \Psi ( X _ { \star } )$

Proof. Suficient decrease of Ψ implies convergence of objective values and square summability of $\eta _ { k }$ . Let Ω be the compact limit set. Standard continuity arguments show that Ψ is constant on Ω and every limit point is critical once the relative-error sequence tends to zero. Uniformization of the K L inequality provides a concave desingularizing function $\varphi$ and a neighborhood of Ω such that

$$
\varphi ^ { \prime } ( \Psi ( X ) - \Psi _ { \star } ) \operatorname { d i s t } ( 0 , \partial \Psi ( X ) ) \geq 1 .
$$

Concavity of $\varphi ,$ suficient decrease, and relative error yield

$$
\varphi ( \Psi ( X _ { k } ) - \Psi _ { \star } ) - \varphi ( \Psi ( X _ { k + 1 } ) - \Psi _ { \star } ) \geq c \left\| \eta _ { k } \right\| _ { F }
$$

for a positive constant c after the sequence enters the neighborhood. Telescoping gives $\begin{array} { r } { \sum _ { k } \left. \eta _ { k } \right. _ { F } < \infty } \end{array}$ hence finite iterate length by retraction equivalence. The sequence is Cauchy and converges. Closedness of the limiting subdiferential and $W _ { k } \to 0$ imply criticality. □

Hypotheses checked for the safeguarded FACET method. The theorem reduces to a result about the implemented safeguarded FACET method once four items are checked: a uniform retraction descent lemma for $- F _ { \mathrm { s q } } ;$ Armijo termination; a lower bound on accepted step sizes (equivalently, a step-dependent relative-error inequality); and compatibility of the regularized rank-deficient update with the same constants. Each is established in Lemmas 6.5–6.6 together with the definability argument of Section 6.

## D Derivation of the Polar Diferential

Let $G = U P$ with $U ^ { \top } U = I$ and $P \succ 0$ . Diferentiate $G = U P$ in direction H:

$$
H = \dot { U } P + U \dot { P } .
$$

Left multiplication by $U ^ { \top }$ gives $\begin{array} { r } { U ^ { \top } H = U ^ { \top } \dot { U } P + \dot { P } . } \end{array}$ The matrix $\Omega = U ^ { \top } \dot { U }$ is skew-symmetric because diferentiating $U ^ { \top } U = I$ yields $\Omega + \Omega ^ { \top } = 0$ . Since $\dot { P }$ is symmetric, subtracting the transpose gives the Sylvester equation

$$
\begin{array} { r } { P \Omega + \Omega P = U ^ { \top } H - H ^ { \top } U . } \end{array}
$$

Projecting the diferentiated factorization onto the orthogonal complement of U gives

$$
( I - U U ^ { \top } ) H = ( I - U U ^ { \top } ) \dot { U } P ,
$$

so

$$
\dot { U } = U \Omega + ( I - U U ^ { \top } ) H P ^ { - 1 } .
$$

Because $P \succ 0$ , the Sylvester operator $\Omega \mapsto P \Omega + \Omega P$ is invertible on skew-symmetric matrices. This formula is used for the heterogeneous-support Jacobian and for numerical derivative checks.

## E Support-Specific Hessian Calculations

On a fixed support-and-sign cell, $F _ { \mathrm { s q } }$ is quadratic plus linear terms. For component $j$ with active support $S _ { j }$ and sign vector $s _ { j }$

$$
f _ { j } ( x ) = { \mu } _ { j } ^ { 2 } x ^ { \top } A _ { S _ { j } } A _ { S _ { j } } ^ { \top } x - 2 { \mu } _ { j } { \gamma } _ { j } s _ { j } ^ { \top } A _ { S _ { j } } ^ { \top } x + { \gamma } _ { j } ^ { 2 } \left| S _ { j } \right| .
$$

The ambient Hessian is $2 \mu _ { j } ^ { 2 } A _ { S _ { j } } A _ { S _ { j } } ^ { \top }$ . The Riemannian Hessian additionally includes curvature terms from the orthogonality constraint. At a critical point, tangent quadratic forms determine whether the point is a local maximizer, saddle, or degenerate. The paper computes these forms explicitly in the rank-one and common-operator block cases and uses numerical eigenanalysis for heterogeneous supports.

## F Extended Counterexample Catalogue

1. Aggregation mismatch: construct two Stiefel frames whose response vectors reverse ordering under $\ell _ { 1 }$ versus squared aggregation.

2. Inactive sphere contradiction: demonstrate that the zero loading is infeasible under a unit-sphere column constraint despite being selected by the positive-part value function.

3. Masking error: quantify norm and objective loss from normalize-then-mask.

4. Threshold cycling: exhibit convergence to a threshold boundary with infinite support switching.

5. Rank-deficient polar input: show nonunique score updates and sensitivity to branch selection.

6. Heterogeneous supports: show that a pooled eigengap can predict contraction while the true coupled polar Jacobian is unstable.

Each counterexample has a deterministic unit test and a manuscript-level explanation of the assumption it invalidates.