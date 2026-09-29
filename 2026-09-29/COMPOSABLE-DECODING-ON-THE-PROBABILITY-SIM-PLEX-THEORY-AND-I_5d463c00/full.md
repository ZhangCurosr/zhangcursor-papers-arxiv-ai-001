# COMPOSABLE DECODING ON THE PROBABILITY SIM-PLEX: THEORY AND IMPLEMENTATION

Xiaotong Ji<sup>∗</sup> Huawei Noah’s Ark Lab

Ahmed Khaled Khamis<sup>∗</sup> ahmedkkhamis@outlook.com

Rasul Tutunov Huawei Noah’s Ark Lab

Matthieu Zimmer Huawei Noah’s Ark Lab

Haitham Bou-Ammar UCL Centre for AI

## ABSTRACT

Decoding for large language models is typically treated as a collection of isolated sampling strategies, with limited theoretical understanding of the behaviours they induce and how their underlying objectives relate. We formulate decoding as an optimisation problem over next-token distributions on the probability simplex, balancing expected model score against regularisation under support constraints. This view recovers familiar decoding methods through choices of regularisers and support constraints; more importantly, it enables new decoders to be constructed by composing distributional preferences within a single optimisation problem without external rewards, learned critics, or model parameter updates. We introduce COMPOSIMPLEX <sup>§</sup>, a library with configurable support rules, regularisation primitives, and simplex solvers for constructing and evaluating compositional decoders. We evaluate standard samplers, individual regularisers, and compositions across multiple models and reasoning tasks. Our results show that compositions can realise trade-offs between single-sample quality, multi-sample quality, and diversity that are not attained by individual decoding objectives.

## 1 INTRODUCTION

Every large language model pipeline ends with a decoding step, yet decoding remains the least principled component in the stack. Practitioners choose from a shelf of isolated tricks: greedy decoding, temperature sampling (Nadeem et al., 2020), Top-K (Fan et al., 2018), Top-P (nucleus) sampling (Holtzman et al., 2020), and recent variants (Meister et al., 2023; Hewitt et al., 2022; Nguyen et al., 2025), each tuned by intuition and trial-and-error. Prior work has identified shared properties of sampling transformations and studied their quality–diversity trade-offs (Nadeem et al., 2020; Wiher et al., 2022). A practical challenge is to turn these insights into explicit objectives that can be configured, combined, and evaluated within a common interface.

We adopt an optimisation perspective: decoding distributions can be constructed by solving explicit optimisation problems on the probability simplex. The key insight is that a decoder need not choose a token directly; at each step, it can first choose a distribution over tokens, and only then sample or take the mode. This reframes decoding as a regularised optimisation problem: maximise expected model score subject to a regulariser that encodes structural preferences, e.g., diversity, sparsity, stability, etc. From this single template, familiar decoding algorithms emerge as special cases: greedy decoding is the limit with no regularisation, softmax sampling is the unique optimum under negative Shannon entropy, Top-K and Top-P arise from negative entropy on restricted supports, and Sparsemax-style sparsity follows from an $\ell _ { 2 }$ penalty (Martins & Astudillo, 2016). Decoders differ not by “how they sample” but by “what objective they implicitly optimise”. This formulation connects regularised prediction and optimisation-based decoding (Blondel et al., 2020; Noarov et al., 2025; Mudgal et al., 2024). Our focus is on jointly optimising complementary distributional objectives at each decoding step without external rewards, learned critics or model parameter updates.

![](images/ae4c0f4ab86ccf66e4e34bf515bb2689c8b646b251e61279e8f24d6ca730b090.jpg)  
Figure 1: Overview of composable decoding and the COMPOSIMPLEX library. Given model scores $s _ { t } ,$ a decoder is configured by (a) a support constraint $C _ { t } ,$ (b) regularisers $\Omega _ { i }$ and (c) a simplex solver. The weighted regularisers are optimised jointly to construct (d) the next-token distribution $q _ { t } ^ { \star }$

This optimisation view does more than unify: it provides a principled way to construct practical decoders that jointly balance multiple distributional preferences. When a distributional preference is represented by a regulariser, multiple preferences can be composed: a weighted sum of regularisers yields a new decoder that combines their behaviours within a single optimisation problem. A practitioner who wants a decoder that simultaneously covers high-quality alternatives, stays anchored to the model distribution via KL divergence, and maintains entropy for diversity can declare $\Omega ( q ) = \alpha _ { 1 } \Omega _ { \mathrm { K L } } ( q ) + \alpha _ { 2 } \Omega _ { \mathrm { c o v } } ( q ) + \alpha _ { 3 } \Omega _ { \mathrm { e n t } } ( \bar { q } )$ and solve on the simplex. This compositional perspective opens up a vast design space that the community has only begun to explore. Existing generation libraries such as Transformers (Wolf et al., 2020) and vLLM (Kwon et al., 2023) expose sampling parameters and extensible logits processors, while DISCO provides a toolkit for distributional control (Kruszewski et al., 2023). We implement this view in COMPOSIMPLEX, a library with configurable support rules, distributional regularisers, and simplex solvers to examine how different distributional preferences affect the performance obtained from a language model. Figure 1 illustrates how these components define and solve a composed decoding objective.

## Our contributions are as follows:

1. Decoding as optimisation on the simplex. We formalise decoding as a regularised optimisation problem over the probability simplex and derive the KKT optimality conditions that recover existing decoders as special cases.

2. Objective composition. We express decoder composition through a weighted sum of regularisers that combines multiple distributional preferences within a single optimisation problem. The regularisers contribute additively to the optimality conditions, and mirror ascent on the simplex provides a general solver for composed objectives that lack closed-form solutions. Based on this formulation, we introduce Best-of-K decoding, which combines KL regularisation with a local token-coverage utility for a K-sample budget.

3. CompoSimplex: a library for composable decoding. We implement the framework as a library with configurable components, including support constraints, regularisation primitives, and simplex solvers. These components serve as flexible building blocks for constructing compositional decoders through a shared interface for Transformers and vLLM.

4. Systematic decoding benchmark. We provide a decoding benchmark for evaluating model performance across different support rules and sampling budgets, jointly measuring accuracy, multi-sample success, and diversity. Across four models and benchmarks, we compare standard samplers, individual regularisers, and compositions, showing that composition can retain the distributional preferences of individual primitives.

## 2 DECODING ON THE PROBABILITY SIMPLEX

We formulate decoding as the problem of choosing a distribution over the vocabulary at each generation step. Given a prefix $x _ { < t } = ( x _ { 1 } , \dots , x _ { t - 1 } )$ , the language model assigns a score $s _ { t } ( v ) \in \mathbb { R }$ to each token v in the vocabulary $V$ at step t. We view decoding as selecting a next-token distribution $q _ { t } \in \Delta ( V )$ , where $\Delta ( V )$ is the collection of all probability distributions defined over the vocabulary $V$ . The next token is then obtained by sampling $x _ { t } \sim q _ { t }$ or by selecting a mode of the distribution $\boldsymbol { x } _ { t } \in$ arg m $\mathbf { \boldsymbol { x } } _ { v \in V } q _ { t } ( v )$ . Thus deterministic and stochastic decoding differ in how the final token is selected from $q _ { t } ,$ , while both require the decoder to construct a distribution on the simplex.

## 2.1 DECODING AS OPTIMISATION OVER DISTRIBUTIONS

We define the decoding distribution as the solution of a regularised optimisation problem:

$$
q _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { q \in \Delta ( V ) } \left[ \langle q , s _ { t } \rangle - \lambda \Omega ( q ) \right] , \qquad { \mathrm { s . t . ~ } } q \in C _ { t } ,\tag{1}
$$

where $\begin{array} { r } { \langle q , s _ { t } \rangle = \sum _ { v \in V } q ( v ) s _ { t } ( v ) } \end{array}$ is the expected model score under $q , \Omega ( q )$ is the regulariser that encodes preferences over the decoding distribution, and $\lambda \geq 0$ controls its strength. The set $C _ { t }$ specifies a decoding-time feasibility constraint; for example, a support constraint restricts sampling to a selected set of candidate tokens $S _ { t } \subseteq V$ by requiring $q ( v ) = 0$ for all $v \not \in S _ { t }$ . This formulation separates the model score from the decoding rule: the model provides $s _ { t } ,$ , while the decoder is specified by $\Omega , \lambda$ and $C _ { t }$

The score term places probability mass on high-scoring tokens, while the regulariser $\Omega ( q )$ shapes how this mass is allocated across the feasible simplex. For example, negative entropy encourages probability mass to spread across the support, whereas a divergence penalty discourages differences from a reference distribution. In this view, a decoding rule is specified by the pair $( { \bar { \Omega } } , C _ { t } )$ with the regularisation strength λ, and the output of the rule is always the distribution $q _ { t } ^ { \star }$

## 2.2 OPTIMALITY CONDITIONS ON THE SIMPLEX

We now derive the optimality condition for Eq. 1. For clarity, we first omit the support constraint $C _ { t }$ and rewrite the maximisation as the equivalent minimisation problem

$$
q _ { t } ^ { \star } = \arg \operatorname* { m i n } _ { q \in \Delta ( V ) } \left[ \lambda \Omega ( q ) - \langle q , s _ { t } \rangle \right] .\tag{2}
$$

Eq. 2 can be solved as a constrained optimisation problem over the simplex. The simplex constraint consists of the normalisation condition $\textstyle \sum _ { v \in V } q ( { \bar { v } } ) = 1$ and the non-negativity conditions $q ( v ) \geq 0$ for all $v \in V$ . We first derive the stationarity condition for coordinates in the interior of the simplex, where $q ( v ) > 0$ . On these active coordinates, the non-negativity constraints are inactive, so we can impose only the normalisation condition with a Lagrange multiplier

$$
\mathcal { L } ( q , \eta ) = \lambda \Omega ( q ) - \langle q , s _ { t } \rangle + \eta \left( \sum _ { v \in V } q ( v ) - 1 \right) ,
$$

where $\eta$ is the multiplier for the simplex normalisation. Assuming $\Omega ( \cdot )$ is differentiable with respect to primal variables $q ( v )$ for any coordinate with strictly positive mass $( q _ { t } ^ { \star } ( v ) > 0 )$ , stationarity gives

$$
\frac { \partial \mathcal { L } } { \partial q ( v ) } ( q _ { t } ^ { \star } ) = 0 \implies s _ { t } ( v ) - \lambda \frac { \partial \Omega ( q _ { t } ^ { \star } ) } { \partial q ( v ) } = \eta .\tag{3}
$$

For coordinates with an optimal primal solution at the boundary $q _ { t } ^ { \star } ( v ) = 0$ , moving slightly into the feasible region must not decrease the objective, and the corresponding KKT condition gives:

$$
\frac { \partial \mathcal { L } } { \partial q ( v ) } ( q _ { t } ^ { \star } ) \geq 0 \implies s _ { t } ( v ) - \lambda \frac { \partial \Omega ( q _ { t } ^ { \star } ) } { \partial q ( v ) } \leq \eta .\tag{4}
$$

The quantity $\begin{array} { r } { s _ { t } ( v ) - \lambda \frac { \partial \Omega ( q _ { t } ^ { \star } ) } { \partial q ( v ) } } \end{array}$ can be viewed as the regularised score of token v at the optimum. All tokens assigned strictly positive probability have the same regularised score $\eta ,$ while tokens at the boundary cannot exceed this value when the derivative at zero is finite. When $C _ { t }$ is a support constraint, the same condition applies on the feasible face of the simplex, with tokens excluded by $C _ { t }$ fixed to zero. This optimality view recovers familiar decoding rules through specific choices of $\Omega , \lambda ,$ , and $C _ { t }$ . Appendix C.1 provides detailed derivations for greedy, Top-K, Top-P, softmax and sparsemax decoding as special cases under our formulation. We next apply the same formulation to composed decoding objectives.

## 3 DECODING BY OBJECTIVE COMPOSITION

We use the optimisation view above to construct new compositional decoders. Many existing decoding methods are designed to control a single property, such as staying close to the base distribution, smoothing the distribution, or encouraging broader coverage across samples. Our goal is to combine such behaviours without introducing a separate decoding rule for each combination. The formulation in Eq. 1 makes this possible: we can express composition by combining different regularisers $\Omega ,$ while keeping the same score term and feasible set. In this section, we describe how composition enters the objective and its optimality condition, and how the resulting problem can be solved when no closed-form solution is available, then introduce Best-of-K decoding as a special use case.

## 3.1 COMPOSITION THROUGH THE REGULARISER

A regulariser Ω specifies one way of shaping the decoding distribution by encoding a bias over the simplex, for example, keeping close to a reference model distribution or encouraging probability mass to cover more tokens. To obtain a decoder with multiple such characteristics, we define a composed regulariser

$$
\Omega _ { \alpha } ( { q } ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } \Omega _ { i } ( { q } ) ,\tag{5}
$$

where $\Omega _ { i }$ is the i-th regulariser and the weights $\alpha _ { i } \geq 0$ satisfy $\textstyle \sum _ { i = 1 } ^ { m } \alpha _ { i } = 1$ . A component may penalise an undesirable property directly, or it may be written as the negative of a quantity to be encouraged. In both cases, the composed expression is treated as a single regulariser in the original decoding objective, and we define the composed problem by substituting Eq. 5 into Eq. 1

$$
\boldsymbol { q } _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { \boldsymbol { q } \in \Delta ( V ) } \left[ \langle \boldsymbol { q } , \boldsymbol { s } _ { t } \rangle - \lambda \sum _ { i = 1 } ^ { m } \alpha _ { i } \Omega _ { i } ( \boldsymbol { q } ) \right] , \qquad \mathrm { s . t . } \ \boldsymbol { q } \in C _ { t } .\tag{6}
$$

The optimisation variable remains the distribution $q ,$ and the decoder still returns a distribution $q _ { t } ^ { \star }$ on the feasible simplex. For the composed regulariser, for every active token v with $q _ { t } ^ { \star } ( v ) > 0$ , the condition in Eq. 3 becomes

$$
s _ { t } ( v ) - \lambda \sum _ { i = 1 } ^ { m } \alpha _ { i } \frac { \partial \Omega _ { i } ( q _ { t } ^ { \star } ) } { \partial q ( v ) } = \eta .\tag{7}
$$

Feasible tokens on the boundary satisfy the corresponding inequality in Eq. 4 when the derivatives at zero are finite. Each component regulariser contributes an additive term to the regularised score through its derivative, and the optimum balances the combined regularisation effect against the model score. This yields a simple mechanism for objective composition: multiple decoding preferences interact through additive gradient contributions within a shared optimality condition. Consequently, new decoding behaviours can be introduced by modifying or combining regularisers, without altering the underlying decoding formulation.

## 3.2 SOLVING THE COMPOSED OBJECTIVE

In special cases, the optimisation in Eq. 2 can be solved analytically from the optimality condition. For example, if the derivative of Ω in Eq. 3 can be inverted coordinate-wise, the normalisation constraint can determine the multiplier η and yield a closed-form distribution. Appendix C.1 works through standard decoders induced by simple regularisers and support constraints. For a composed regulariser, however, Eq. 7 contains a sum of derivative terms, making it difficult to isolate each coordinate $q _ { t } ^ { \star } ( v )$ in closed form. We therefore solve the objective directly on the simplex.

One seemingly natural choice to tackle this problem is projected gradient ascent:

$$
q _ { j + 1 } = \arg \operatorname* { m a x } _ { q \in \Delta ( V ) } \left[ \langle \nabla f ( q _ { j } ) , q - q _ { j } \rangle - \frac { 1 } { 2 \rho } \| q - q _ { j } \| _ { 2 } ^ { 2 } \right] ,\tag{8}
$$

where $\rho > 0$ is the step size and $f ( q ) = \langle q , s _ { t } \rangle - \lambda \Omega _ { \alpha } ( q )$ denotes the objective function in Eq. 6. This form shows that projected gradient ascent uses Euclidean distance to keep the next iterate close to $q _ { j }$ . However, the optimisation variable is a probability distribution. Euclidean distance does not reflect the geometry of the simplex, and the update requires an explicit projection step to return to a valid distribution. Mirror ascent addresses the geometry mismatch of projected gradient ascent by replacing the Euclidean distance in Eq. 8 with a divergence defined on the simplex, leading to updates that remain valid distributions without an explicit Euclidean projection. For a strictly convex function $\psi ,$ define

$$
D _ { \psi } ( q , q _ { j } ) = \psi ( q ) - \psi ( q _ { j } ) - \left. \nabla \psi ( q _ { j } ) , q - q _ { j } \right. .\tag{9}
$$

The mirror ascent update becomes:

$$
q _ { j + 1 } = \arg \operatorname* { m a x } _ { q \in \Delta ( V ) } \left[ \langle \nabla f ( q _ { j } ) , q - q _ { j } \rangle - \frac { 1 } { \rho } D _ { \psi } ( q , q _ { j } ) \right] .\tag{10}
$$

Using the negative entropy potential $\begin{array} { r } { \psi ( q ) = \sum _ { v \in V } q ( v ) } \end{array}$ log q(v) gives $D _ { \psi } ( q , q _ { j } ) = K L ( q | | q _ { j } )$ Under this choice, Eq. 10 reduces to the multiplicative update:

$$
q _ { j + 1 } = \frac { q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } } { \lVert q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } \rVert _ { 1 } } ,\tag{11}
$$

which preserves non-negativity and normalisation by construction. The derivation is provided in Appendix B. Here, ⊙ denotes the component-wise product of two vectors in $\mathbb { R } ^ { | V | }$ . Please note that the composed regulariser contributes to this equation via the gradient term $\nabla f ( q _ { j } ) =$ $\begin{array} { r } { s _ { t } - \lambda \sum _ { i = 1 } ^ { m } \alpha _ { i } \nabla \dot { \Omega } _ { i } ( q _ { j } ) } \end{array}$ . When feasibility conditions $C _ { t }$ impose a support constraint, the update is applied and normalised on the feasible face of the simplex. After a fixed number of steps, the final iterate is used as the decoding distribution.

## 3.3 USE CASE: BEST-OF-K DECODING

We introduce Best-of-K (BoK) decoding as an example of constructing a new decoder through objective composition in Algorithm 1. There are existing generation pipelines that draw multiple completions and then apply self-consistency or reranking (Wang et al., 2023). In these settings, the usefulness of the candidate set depends on whether it contains good alternatives. BoK is designed to encourage coverage across multiple samples while keeping the decoding distribution close to the model distribution. For a selected token set $S _ { t }$ defining the support constraint $C _ { t } .$ let $p _ { t }$ be a positive reference model distribution on $S _ { t }$ . We compose the KL regulariser with the negative of a weighted coverage utility:

$$
\Omega _ { \mathrm { K L } } ( q ) = \mathrm { K L } ( q \| p _ { t } ) , \quad \Omega _ { U _ { K } } ( q ) = - \sum _ { v \in S _ { t } } w _ { t } ( v ) \left[ 1 - ( 1 - q ( v ) ) ^ { K } \right] ,\tag{12}
$$

Algorithm 1 BoK Decoder via Mirror Ascent (one decoding step)   
Require: candidate tokens $S _ { t } .$ , scores $s _ { t } ,$ reference $p _ { t } ,$ weights $w _ { t }$   
Require: hyperparameters $K , \lambda , \alpha _ { \mathrm { K L } } , \alpha _ { U }$ , step size $\rho ,$ iterations J   
1: Initialise $q _ { 0 } \gets p _ { t }$   
2: for $j = 0 , \bar { 1 } , \dots , J - 1$ do   
3: for each token v $\in S _ { t }$ do   
4: $g _ { j } ( v ) \gets s _ { t } ( v ) - \lambda \alpha _ { \mathrm { K L } } \bigg ( \log \frac { q _ { j } ( v ) } { p _ { t } ( v ) } + 1 \bigg ) + \lambda \alpha _ { U } w _ { t } ( v ) K ( 1 - q _ { j } ( v ) ) ^ { K - 1 }$   
5: end for   
6: $M _ { j } \gets \operatorname* { m a x } _ { v \in S _ { t } } \rho g _ { j } ( v )$ ▷ Log-Sum-Exp stabilisation   
7: $\widetilde { q } _ { j + 1 } ^ { \textit { \textbf { \ i } } } ( v ) \gets q _ { j } ( v ) \exp ( \rho g _ { j } ( v ) - M _ { j } ) , \quad v \in S _ { t }$   
8: $q _ { j + 1 }  \widetilde { q } _ { j + 1 } / \| \widetilde { q } _ { j + 1 } \| _ { 1 }$   
9: end for   
10: return $q _ { J }$

where $w _ { t } ( v ) \geq 0$ . The utility adapts weighted expected coverage from classical occupancy models (Boneh & Hofri, 1997), where the bracketed term is the probability of observing token v at least once in K independent draws at a given prefix. Different choices of $w _ { t }$ give the KL-Coverage and KL-Diversity variants, with the weighting schemes defined in Section 4.

Let $U _ { K , t } ( q ) = - \Omega _ { U _ { K } } ( q )$ denote the weighted coverage utility. The resulting composed objective is

$$
\boldsymbol { q } _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { \boldsymbol { q } \in \Delta ( S _ { t } ) } \left[ \langle \boldsymbol { q } , \boldsymbol { s } _ { t } \rangle - \lambda \alpha _ { \mathrm { K L } } \mathrm { K L } ( \boldsymbol { q } \| \boldsymbol { p } _ { t } ) + \lambda \alpha _ { \boldsymbol { U } } \boldsymbol { U } _ { K , t } ( \boldsymbol { q } ) \right]\tag{13}
$$

The shared mirror-ascent solver uses the gradient

$$
g _ { j } ( v ) = s _ { t } ( v ) - \lambda \alpha _ { \mathrm { K L } } \left( \log \frac { q _ { j } ( v ) } { p _ { t } ( v ) } + 1 \right) + \lambda \alpha _ { U } w _ { t } ( v ) K ( 1 - q _ { j } ( v ) ) ^ { K - 1 } , \qquad v \in S _ { t } .\tag{14}
$$

For $K > 1$ , the coverage term gives diminishing returns to tokens that are already likely to appear among the samples, while the KL term penalises departures from $p _ { t }$ . This illustrates how combining a utility function with distributional preferences can shape the next-token distribution beyond what temperature scaling alone can achieve; see Appendix C.2 for a detailed discussion.

## 4 COMPOSIMPLEX: A LIBRARY FOR COMPOSABLE DECODING

COMPOSIMPLEX is an open-source library that implements the formulation in Section 2 and objective composition in Section 3 through the configurable support rules, regularisation primitives, and simplex solvers shown in Figure 1. A new decoder is specified by a configuration that selects its support, regularisation primitives, and optimiser settings. The regularisation coefficient λ controls the overall regularisation strength, and the weights $\alpha _ { i } \geq 0 .$ , with $\textstyle \sum _ { i } \alpha _ { i } = 1$ , control the relative contribution of each primitive. At each generation step, these components define an optimisation problem whose solution gives the next-token distribution. The library integrates this computation with generation backends, allowing decoding methods to be constructed through configuration.

Support. A support rule selects candidate tokens $S _ { t } \subseteq V$ , defining the constraint $C _ { t }$ through $q ( v ) = 0$ outside $\bar { S } _ { t }$ . We support the full vocabulary, Top-k with a fixed candidate count (Fan et al., 2018), Top-p based on cumulative probability mass (Holtzman et al., 2020), Min-p with a threshold relative to the highest token probability (Nguyen et al., 2025), η-sampling with an entropy-adaptive threshold (Hewitt et al., 2022), and typical sampling based on proximity of token information content to the distribution’s entropy (Meister et al., 2023).

Regularisation primitives. An objective primitive with a computable gradient with respect to q can be added and combined with others through configuration. We provide KL and JS divergences to control deviation from a reference distribution (Kullback & Leibler, 1951; Lin, 1991), and negative entropy to encourage broader sampling (Shannon, 1948; Jaynes, 1957). The KL regulariser is defined in $\operatorname { E q . }$ . 12, while $\Omega _ { \mathrm { J S } } ( q ) = \mathrm { \bar { J S } } ( \bar { q } \| p _ { t } )$ and $\Omega _ { \mathrm { E n t } } ( q ) \stackrel { \cdot } { = } - H ( q )$ . The reference $p _ { t }$ is the softmax of the model logits on $S _ { t }$ with a configurable temperature. For multi-sample generation, Coverage and Diversity instantiate $\Omega _ { U _ { K } }$ in Eq. 12 through different choices of $w _ { t }$ . Coverage assigns equal positive weights to the top-r tokens under $p _ { t }$ and zero elsewhere. Diversity uses $w _ { t } ( \bar { v } ) \propto \bar { d } _ { t } ( v ) \bar { \exp ( - d _ { t } ( v ) / \tau ) }$ , where $d _ { t } ( \boldsymbol { v } )$ is the gap from the largest logit and $\tau > 0 .$ , favouring alternatives with moderate logit gaps.

Optimiser. The optimiser combines the weighted gradients of the selected primitives to compute the decoding distribution. COMPOSIMPLEX provides closed-form solutions for supported cases, including single KL and entropy objectives, and otherwise uses mirror ascent as described in Section 3.2. Appendix C.2 gives the KL and entropy solutions and discusses their relation to temperature scaling. For the primitives above, these updates reuse the current model logits and require no additional model forward passes. We use a small number of mirror-ascent steps to limit the added computation and report the resulting inference overhead in Section 5.

Backend integration. COMPOSIMPLEX integrates with Hugging Face Transformers (Wolf et al., 2020) and vLLM (Kwon et al., 2023) through custom logits processors. The processor returns the computed log probabilities, with tokens outside the support masked, and the backend performs multinomial sampling or argmax according to the configured selection rule. This allows the same decoder configuration to be used with either backend.

## 5 EVALUATION: DECODING BY OBJECTIVE COMPOSITION

We use COMPOSIMPLEX as a shared benchmarking framework to examine how different decoding objectives affect multiple dimensions of model performance. We compare standard sampling methods, individual regularisation primitives, and composed objectives in terms of accuracy, multisample success, and diversity, using a common evaluation setup. Our evaluation addresses two questions: (i) Can composition combine the preferences encoded by individual regularisers? (ii) How do composed objectives affect distributional behaviour compared with individual primitives?

## 5.1 PERFORMANCE EVALUATION

Models and benchmarks. We evaluate four models and benchmarks with different scales and across base models and instruct versions. We use LFM2.5-1.2B-Base (Amini et al., 2025) on IFEval (Zhou et al., 2023), which evaluates compliance with verifiable instructions; Qwen3-4B-Base (Yang et al., 2025) on GPQA Diamond (Rein et al., 2024), which contains 198 science questions; and Qwen2.5-7B (Qwen et al., 2025) on MATH500 (Lightman et al., 2024) for mathematical problem solving. For code generation, we evaluate the instruct model Gemma-4-26B-A4B-IT (Gemma Team, 2026) on new problems introduced in LiveCodeBench v6 (Jain et al., 2025).

Decoder configurations. The standard sampling support rules we evaluate include Top-k (Fan et al., 2018), Top-p (Holtzman et al., 2020), Min-p (Nguyen et al., 2025), typical sampling (Meister et al., 2023), and η-sampling (Hewitt et al., 2022). Each support rule selects the candidate tokens at a generation step. The single primitives include KL divergence (Kullback & Leibler, 1951), JS divergence (Lin, 1991), entropy (Jaynes, 1957), and Coverage and Diversity primitives (Boneh & Hofri, 1997). Composed objectives combine two or more primitives through weights $\alpha _ { i } .$ In particular, KL-Coverage and KL-Diversity instantiate the two weighted variants introduced in Section 3.3. We evaluate them alongside other compositions, compare each composition with its constituent primitives, and examine how these objectives behave across different support constraints.

Evaluation metrics. We report pass@k $( k \in \{ 1 , 4 , 1 6 \} )$ , the fraction of prompts with at least one correct completion among the first k samples. On MATH500 and GPQA Diamond, selfconsistency accuracy (SC@16) uses majority voting over 16 extracted answers. Semantic diversity averages pairwise cosine distances between embeddings of sampled reasoning completions within each prompt, then across prompts. For IFEval and LiveCodeBench, all-pass@16 is the fraction of prompts whose 16 completions all pass strict prompt-level instruction checks or all test cases, respectively. We also report LiveCodeBench’s Pass@1 on hard problems.

Base KL Diversity Composition  
![](images/31f64ae345530b5cfab77c31f60005d85b46e9f1a8fbfdc3319789724108cece.jpg)

(a) MATH500, Top-k  
![](images/fa33abbc475dbbbe50bc5417bbe56e321467af8371081b4bfa0295e448c2cd4b.jpg)

![](images/8f69c10e26b9513420c058f03490dfa796792c4d64c9f6ebdef6ef195e511dd4.jpg)

![](images/21b00b6259268ad8aa407482f3b7d4c0fc3d5371fe1d95039cd2467d52f3fe79.jpg)

![](images/28f56f9c36b733136a46cbad9cfcccb1d859a5984fedeaf92a06b2b4cc96adfd.jpg)  
(e) IFEval, Top-p

![](images/8a42f377ff94714e4b6f9d187f57d76842cfba711c344c1a76f2cae9786f0055.jpg)  
(f) IFEval, η-sampling

![](images/375fde4c003545aabf40675d4585ed7688d6a237ccaffceaff65ece4604af710.jpg)  
(g) LiveCodeBench, Top-p

![](images/35a124c53a67410359865ee38a29b947221dd3924d2fb2be954eda3194c71526.jpg)  
(h) LiveCodeBench, Min-p  
Figure 2: Performance profiles across MATH500 (Qwen2.5-7B), GPQA Diamond (Qwen3-4B-Base), IFEval (LFM2.5-1.2B-Base), and LiveCodeBench v6 (Gemma-4-26B-A4B-IT). Each panel compares a standard sampler with selected individual and composed objectives under the indicated support rule. The axes show the accuracy and diversity metrics labelled in each panel.

Main results. Figure 2 summarises selected performance profiles across four model–benchmark pairs and different support rules. On MATH500 with Qwen2.5-7B and Top-k support, KL+Diversity matches KL’s pass@1 of 64.4%, 8.4 percentage points above Diversity, while its pass@16 reaches 90.0%, close to Diversity’s 91.0% and above KL’s 88.4%. Across GPQA Diamond, IFEval, and LiveCodeBench, the plots also show that compositions cover a larger area than either constituent primitive in most settings, while some metrics may fall below individual regularisers.

These results show that composition can balance the preferences of individual primitives. We also observe larger maximum gains over the corresponding base sampler in pass@1 (+10.6 percentage points) than in pass@16 (+5.1 percentage points), suggesting that regularisation can effectively concentrate probability mass on correct completions, making them easier to obtain with fewer samples. This pattern is consistent with prior findings on decoding and post-training (Wiher et al., 2022; Yue et al., 2025), and motivates evaluating model performance across decoding objectives and sampling budgets within a unified framework. Full results and seed variation are reported in Appendix D.2.

Computational efficiency. The generation cost for single primitives with a closed-form solution, including KL and entropy, is similar to that of the corresponding standard samplers. For single and compositional objectives without closed-form solutions, we solve the objective approximately using mirror ascent by combining the weighted gradients within each update. These updates reuse the model logits and require no additional forward passes at a given generation step. The empirical generation cost ranges from approximately 1.13× to 2.88× the corresponding base decoding method for our main evaluation. We report these costs and also examine sensitivity to the number of iterations and step size for the optimisation in Appendix D.3.

## 5.2 DISTRIBUTIONAL BEHAVIOUR ANALYSIS

We compare compositions with their constituent primitives under different regularisation strengths and composition weights. We examine how these settings change the distributional metrics and how the resulting preferences affect task performance.

Regularisation strength. We vary $\lambda \in \{ 0 . 5 , 1 , 2 \}$ with fixed Top-k support and equal composition weights on MATH500. Increasing λ gives the selected regularisation preferences more influence relative to the model score. Across all four compositions, the corresponding utility increases while

![](images/a85adef5994e00c12ac1bdc9a1f3b8956bb6eb26cf39b1046b97165fc4b134c5.jpg)  
Figure 3: Distributional trade-offs for individual objectives and equally weighted compositions on MATH500 with Qwen2.5-7B and Top-k support. Marker size denotes $\lambda \in \ \{ 0 . 5 , 1 , 2 \}$ ; dashed curves are the Pareto guides.

KL or JS divergence decreases, as shown in Figure 3. At λ = 2, three of the four composed points are non-dominated among the evaluated configurations in their respective divergence–utility planes.

Strong utility regularisation can nevertheless reduce task accuracy. As λ increases from 0.5 to 2, Diversity’s utility rises but its pass@1 falls from 62.4% to 39.8%, while KL + Diversity retains 57.0% pass@1 at λ = 2. At this largest tested strength, all four compositions achieve higher pass@1 than their corresponding pure utility primitives. It is always difficult to decide the regularisation strength in the regularised objective, and the sweep also shows that compositions can retain robust task performance under strong regularisation compared with single primitives.

Composition weights. We also vary the composition weights of the two BoK variants at fixed λ and examine both distributional metrics and task performance. Across the tested weights α ∈ {0, 0.25, 0.5, 1}, increasing the utility weight α monotonically raises the corresponding utility and reduces the measured KL divergence. The effects on task performance vary by metric, with no consistent improvement as α increases. Appendix D.4 provides the settings and full results.

## 6 RELATED WORK

Sampling Methods. Sampling methods control which tokens remain eligible and how probability mass is distributed among them. Top-k retains a fixed number of candidates (Fan et al., 2018), while nucleus sampling adapts the support to retain a prescribed probability mass (Holtzman et al., 2020). Temperature scaling adjusts concentration within the resulting distribution. Studies of these transformations identify shared properties and show that their quality–diversity trade-offs depend on the task and configuration (Nadeem et al., 2020; Wiher et al., 2022). For multi-sample inference, Du et al. (2025) use an entropy-based criterion to select temperatures for answer aggregation without task-specific validation data. These results motivate combining several distributional preferences to retain model fidelity and encourage exploration. We express these preferences as explicit objectives and study their joint effects through objective composition.

Optimisation-based Decoding. Optimisation-based generation often targets complete sequences: DAEMON controls expected text metrics (Ji et al., 2024), while power sampling sharpens the sequence distribution without external rewards (Karan & Du, 2026; Ji et al., 2026). Controlled Decoding instead applies tokenwise control using prefix value functions learned from reward supervision (Mudgal et al., 2024). Direct optimisation of the next-token distribution offers a complementary route: Bregman decoding uses a divergence and an $\ell _ { 0 }$ penalty to recover a sparse distribution, with an adaptively selected support (Noarov et al., 2025). We likewise optimise a next-token distribution, but focus on jointly balancing directly computable preferences. Their weighted combination yields a regularised simplex problem at each step, using current model scores without external rewards, learned critics, future rollouts, or model parameter updates. Appendix A further compares the objectives and information used by these methods.

## 7 CONCLUSION

We presented a framework for decoding through regularised optimisation on the probability simplex, recovering familiar decoders as special cases and composing distributional preferences within a single objective. Our library COMPOSIMPLEX implements support rules, regularisers, and solvers as flexible building blocks, with Best-of-K decoding combining KL regularisation and local token coverage. Experiments across four models and benchmarks show that composition can retain complementary strengths of individual primitives in single-sample accuracy, multi-sample success, and diversity. Compositions can also achieve non-dominated points in distribution space while maintaining more robust task performance than single primitives under strong regularisation. This general framework and library provide a practical basis for designing and evaluating decoding strategies through explicit, composable objectives.

## REFERENCES

Alexander Amini, Anna Banaszak, Harold Benoit, Arthur Bo¨ok, Tarek Dakhran, Song Duong, Al-¨ fred Eng, Fernando Fernandes, Marc Hark¨ onen, Anne Harrington, Ramin Hasani, Saniya Karwa,¨ Yuri Khrustalev, Maxime Labonne, Mathias Lechner, Valentine Lechner, Simon Lee, Zetian Li, Noel Loo, Jacob Marks, Edoardo Mosca, Samuel J. Paech, Paul Pak, Rom N. Parnichkun, Alex Quach, Ryan Rogers, Daniela Rus, Nayan Saxena, Bettina Schlager, Tim Seyde, Jimmy T. H. Smith, Aditya Tadimeti, and Neehal Tumma. LFM2 technical report. arXiv:2511.23404, 2025. URL https://arxiv.org/abs/2511.23404.

Mathieu Blondel, Andre F. T. Martins, and Vlad Niculae. Learning with Fenchel–Young losses.´ Journal of Machine Learning Research, 21(35):1–69, 2020. URL https://jmlr.org/ papers/v21/19-021.html.

Arnon Boneh and Micha Hofri. The coupon-collector problem revisited—a survey of engineering problems and computational methods. Stochastic Models, 13(1):39–66, 1997. doi: 10.1080/ 15326349708807412.

Souradip Chakraborty, Soumya Suvra Ghosal, Ming Yin, Dinesh Manocha, Mengdi Wang, Amrit Singh Bedi, and Furong Huang. Transfer Q-star: Principled decoding for LLM alignment. In Advances in Neural Information Processing Systems, volume 37, pp. 101725–101761, 2024.

Sijin Chen, Omar Hagrass, and Jason Klusowski. Decoding game: On minimax optimality of heuristic text generation strategies. In The Thirteenth International Conference on Learning Representations, 2025.

Yuanhao Ding, Meimingwei Li, Esteban Garces Arias, Matthias Aßenmacher, Christian Heumann, and Chongsheng Zhang. Min-k sampling: Decoupling truncation from temperature scaling via relative logit dynamics. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 14932–14948. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.681. URL https://aclanthology. org/2026.acl-long.681/.

Weihua Du, Yiming Yang, and Sean Welleck. Optimizing temperature for language models with multi-sample inference. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 14648–14668. PMLR, 2025. URL https://proceedings.mlr.press/v267/du25f.html.

Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 889–898, 2018. doi: 10.18653/v1/P18-1082. URL https://aclanthology. org/P18-1082/.

Gemma Team. Gemma 4 technical report. arXiv:2607.02770, 2026. URL https://arxiv. org/abs/2607.02770.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.

John Hewitt, Christopher D Manning, and Percy Liang. Truncation sampling as language model desmoothing. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 3414–3427, 2022. doi: 10.18653/v1/2022.findings-emnlp.249. URL https:// aclanthology.org/2022.findings-emnlp.249/.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In The Eighth International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rygGQyrFvH.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html.

Edwin T Jaynes. Information theory and statistical mechanics. Physical review, 106(4):620–630, 1957. doi: 10.1103/PhysRev.106.620.

Haozhe Ji, Pei Ke, Hongning Wang, and Minlie Huang. Language model decoding as direct metrics optimization. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 7416573f05b50beac6d0aef3abc805c0-Abstract-Conference.html.

Xiaotong Ji, Rasul Tutunov, Matthieu Zimmer, and Haitham Bou Ammar. Scalable power sampling: Unlocking efficient, training-free reasoning for LLMs via distribution sharpening. arXiv:2601.21590, 2026.

Aayush Karan and Yilun Du. Reasoning with sampling: Your base model is smarter than you think. In The Fourteenth International Conference on Learning Representations, 2026.

Muhammad Khalifa, Hady Elsahar, and Marc Dymetman. A distributional approach to controlled text generation. In The Ninth International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=jWkw45-9AbL.

Wouter Kool, Herke Van Hoof, and Max Welling. Stochastic beams and where to find them: The Gumbel-top-k trick for sampling sequences without replacement. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 3499–3508. PMLR, 2019. URL https://proceedings.mlr.press/v97/ kool19a.html.

German Kruszewski, Jos Rozen, and Marc Dymetman. disco: a toolkit for distributional con-´ trol of generative models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 144–160. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-demo.14. URL https: //aclanthology.org/2023.acl-demo.14/.

Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86, 1951. doi: 10.1214/aoms/1177729694.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Prin ciples, pp. 611–626. ACM, 2023. doi: 10.1145/3600006.3613165.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html.

Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information theory, 37(1):145–151, 1991. doi: 10.1109/18.61115.

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, and Yejin Choi. DExperts: Decoding-time controlled text generation with experts and antiexperts. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6691–6706. Association for Computational Linguistics, 2021. doi: 10.18653/ v1/2021.acl-long.522. URL https://aclanthology.org/2021.acl-long.522/.

Andre Martins and Ramon Astudillo. From softmax to sparsemax: A sparse model of attention and multi-label classification. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pp. 1614–1623, 2016. URL https://proceedings.mlr.press/v48/martins16.html.

Clara Meister, Ryan Cotterell, and Tim Vieira. If beam search is the answer, what was the question? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 2173–2185. Association for Computational Linguistics, 2020.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally typical sampling. Transactions of the Association for Computational Linguistics, 11:102–121, 2023. doi: 10.1162/ tacl a 00536. URL https://aclanthology.org/2023.tacl-1.7/.

Sidharth Mudgal, Jong Lee, Harish Ganapathy, Yaguang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, Jilin Chen, Alex Beutel, and Ahmad Beirami. Controlled decoding from language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 36486–36503, 2024. URL https://proceedings.mlr.press/v235/mudgal24a. html.

Moin Nadeem, Tianxing He, Kyunghyun Cho, and James Glass. A systematic characterization of sampling algorithms for open-ended language generation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pp. 334–346. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.aacl-main.36. URL https://aclanthology.org/2020.aacl-main.36/.

Minh Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, and Ravid Shwartz-Ziv. Turning up the heat: Min-p sampling for creative and coherent LLM outputs. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ afa5f124e36bed5cc2125067005d43f5-Abstract-Conference.html.

Georgy Noarov, Soham Mallick, Tao Wang, Sunay Joshi, Yan Sun, Yangxinyu Xie, Mengxin Yu, and Edgar Dobriban. Foundations of Top-k decoding for language models. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.nips.cc/paper\_files/paper/2025/hash/ a96d4fda3017f1773b261a52a3efc8dd-Abstract-Conference.html.

Ben Peters, Vlad Niculae, and Andre F. T. Martins. Sparse sequence-to-sequence models. In´ Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1504– 1519. Association for Computational Linguistics, 2019.

Lianhui Qin, Sean Welleck, Daniel Khashabi, and Yejin Choi. COLD decoding: Energy-based constrained text generation with langevin dynamics. In Advances in Neural Information Processing Systems, volume 35, 2022. URL

https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 3e25d1aff47964c8409fd5c8dc0438d7-Abstract-Conference.html.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv:2412.15115, 2025. URL https://arxiv.org/abs/2412.15115.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x.

Chenxia Tang, Jianchun Liu, Hongli Xu, and Liusheng Huang. Top-nσ: Eliminating noise in logit space for robust token sampling of LLM. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10758–10774. Association for Computational Linguistics, 2025a. doi: 10.18653/v1/2025.acl-long.528. URL https://aclanthology.org/2025.acl-long.528/.

Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, and Remi Munos. Optimizing language models for inference time objectives using reinforcement learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 59066–59085. PMLR, 2025b. URL https://proceedings.mlr.press/ v267/tang25o.html.

Luke Vilnis, Yury Zemlyanskiy, Patrick Murray, Alexandre Tachard Passos, and Sumit Sanghai. Arithmetic sampling: Parallel diverse decoding for large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 35120–35136. PMLR, 2023. URL https://proceedings.mlr. press/v202/vilnis23a.html.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.

Gian Wiher, Clara Meister, and Ryan Cotterell. On decoding strategies for neural text generators. Transactions of the Association for Computational Linguistics, 10:997–1012, 2022. doi: 10.1162/ tacl a 00502. URL https://aclanthology.org/2022.tacl-1.58/.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-demos.6. URL https://aclanthology. org/2020.emnlp-demos.6/.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Kevin Yang and Dan Klein. FUDGE: Controlled text generation with future discriminators. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3511–3535. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.naacl-main.276. URL https: //aclanthology.org/2021.naacl-main.276/.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv:2311.07911, 2023. URL https://arxiv.org/abs/2311.07911.

## A ADDITIONAL RELATED WORK

Support Selection and Distribution Shaping. Desmoothing interprets truncation as removing probability mass introduced by model smoothing (Hewitt et al., 2022). Top-nσ thresholds logits using their maximum and standard deviation, while Min-k identifies truncation boundaries from relative changes in sorted logits (Tang et al., 2025a; Ding et al., 2026). These methods inform the choice of admissible support in our framework. Sparsemax and α-entmax obtain sparse distribu tions through regularisation, illustrating how the objective itself can determine zero-probability co ordinates (Martins & Astudillo, 2016; Peters et al., 2019). Other theoretical accounts explain beam search through information-density objectives (Meister et al., 2020) and truncation-normalisation through approximations to a minimax strategy (Chen et al., 2025). Together, these connections motivate separating support selection from distribution shaping, while expressing the latter through configurable objectives.

Regularised Prediction and Local Decoding. Blondel et al. (2020) define prediction as maximising a score minus an output regulariser and derive corresponding losses for supervised learning. Noarov et al. (2025) apply local optimisation directly to decoding: they minimise a Bregman divergence from the model’s next-token distribution together with an $\ell _ { 0 }$ sparsity penalty. Under their assumptions, the optimal support consists of the highest-probability tokens, its size can be selected adaptively, and the divergence determines how retained probabilities are reweighted. Our focus is on a different use of local optimisation: for a chosen support, we combine divergence, entropy, and token-coverage terms to control several distributional preferences jointly. This combination does not require external reward models, learned value functions or additional model training.

Sequential and Reward-Guided Decoding. Distributional control specifies desired output properties through constraints on expected features (Khalifa et al., 2021). DAEMON uses multiple text metrics to define a sequence-level energy-based target and approximates sampling through sampling-importance-resampling (Ji et al., 2024). COLD enforces differentiable constraints by applying Langevin dynamics to a continuous relaxation of a token sequence (Qin et al., 2022). Power sampling also acts on complete-sequence distributions, but sharpens model likelihoods without external rewards or additional training, using MCMC (Karan & Du, 2026) or autoregressive corrections estimated from future rollouts (Ji et al., 2026). Some methods apply sequence-level preferences through tokenwise control: Controlled Decoding uses reward-trained prefix value functions in a KL-regularised objective and supports combinations of reward scorers (Mudgal et al., 2024), while Transfer $\mathbf { Q } ^ { * }$ estimates values for a target reward using a baseline model (Chakraborty et al., 2024). FUDGE uses learned predictors of future attributes and supports their composition (Yang & Klein, 2021); DExperts combines expert and anti-expert language-model logits (Liu et al., 2021). Our implemented objectives directly shape the current next-token distribution using model scores and configured references and weights, without evaluating complete trajectories or training critics.

Multi-sample Generation and Selection. Self-consistency improves answer reliability by aggregating independently sampled reasoning paths (Wang et al., 2023). Stochastic beam search reduces repeated sequences through sampling without replacement (Kool et al., 2019), while arithmetic sampling coordinates draws to obtain diverse candidates (Vilnis et al., 2023). Tang et al. (2025b) train models to improve inference-time objectives such as pass@k and majority voting. Our question is how changing the conditional sampling distribution of a frozen model affects candidate utility at a fixed sampling budget. The proposed BoK decoding method rewards the probability of covering weighted token alternatives in K independent draws at the same prefix. This local surrogate shapes candidate generation; its effects on accuracy and semantic diversity are tested empirically.

Decoding Infrastructure. Transformers and vLLM support custom decoding behaviour through generation settings and extensible logits processors (Wolf et al., 2020; Kwon et al., 2023). DISCO makes distributional control methods accessible through reusable software components (Kruszewski et al., 2023). COMPOSIMPLEX exposes the optimisation problem itself: users select a support, declare weighted regularisers, and choose a simplex solver. The solver combines the regulariser gradients to optimise the declared objective jointly. This interface connects the theoretical formulation to practical experimentation, allowing individual objectives and their compositions to be configured and compared within the same implementation.

## B MIRROR ASCENT CLOSED-FORM EXPRESSION

Let us consider the mirror ascent update:

$$
q _ { j + 1 } = \arg \operatorname* { m a x } _ { q \in \Delta ( V ) } \left[ \langle \nabla f ( q _ { j } ) , q - q _ { j } \rangle - \frac { 1 } { \rho } D _ { \psi } ( q , q _ { j } ) \right] .
$$

Next, we will show that using the negative entropy potential $\begin{array} { r } { \psi ( q ) = \sum _ { v \in V } q ( v ) } \end{array}$ log $q ( v )$ gives $D _ { \psi } ( q , q _ { j } ) = K L ( q | | q _ { j } )$ and the update $q _ { j + 1 }$ allows the following closed-form expression:

$$
q _ { j + 1 } = \frac { q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } } { \left\| q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } \right\| _ { 1 } } .
$$

The corresponding optimisation problem has the following form:

$$
\operatorname* { m i n } _ { \substack { q ( v ) \geq 0 , v \in V } } \frac { 1 } { \rho } \sum _ { v \in V } q ( v ) \log \frac { q ( v ) } { q _ { j } ( v ) } - \nabla ^ { \top } f ( q _ { j } ) ( q - q _ { j } )
$$

$$
\operatorname { s . t . } \ \sum _ { v \in V } q ( v ) = 1 .
$$

The Lagrangian has the following form:

$$
\mathcal { L } ( q , \eta ) = \frac { 1 } { \rho } \sum _ { v \in V } q ( v ) \log \frac { q ( v ) } { q _ { j } ( v ) } - \nabla ^ { \top } f ( q _ { j } ) ( q - q _ { j } ) + \eta \left( \sum _ { v \in V } q ( v ) - 1 \right) .
$$

The first-order stationary conditions give:

$$
\frac { 1 } { \rho } \left[ \log \frac { q ( v ) } { q _ { j } ( v ) } + 1 \right] - [ \nabla f ( q _ { j } ) ] _ { v } + \eta = 0 \Longrightarrow q ( v ) = q _ { j } ( v ) \exp \left( \rho [ \nabla f ( q _ { j } ) ] _ { v } - \rho \eta - 1 \right) .
$$

Using the normalisation condition $\textstyle \sum _ { v \in V } [ q ( v ) ] = 1$ gives:

$$
\sum _ { v \in V } q _ { j } ( v ) \exp \left( \rho [ \nabla f ( q _ { j } ) ] _ { v } \right) \exp ( - \rho \eta - 1 ) = 1 \implies \exp \left( \rho \eta + 1 \right) = \sum _ { v \in V } q _ { j } ( v ) \exp \left( \rho [ \nabla f ( q _ { j } ) ] _ { v } \right) .
$$

This gives the final expression for the optimal primal variable:

$$
q ( v ) = \frac { q _ { j } ( v ) \exp { ( \rho [ \nabla f ( q _ { j } ) ] _ { v } ) } } { \sum _ { v \in V } q _ { j } ( v ) \exp { ( \rho [ \nabla f ( q _ { j } ) ] _ { v } ) } } , \forall v \in V .
$$

Using non-negativity of all terms and the component-wise product ⊙ between two vectors $q _ { j }$ and $\nabla f ( q _ { j } )$ gives:

$$
q _ { j + 1 } = \frac { q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } } { \left\| q _ { j } \odot \exp { ( \rho \nabla f ( q _ { j } ) ) } \right\| _ { 1 } } .
$$

## C ANALYSIS OF STANDARD AND COMPOSED DECODERS

## C.1 STANDARD DECODERS AS SPECIAL CASES

The following examples recover standard decoders from Eq. 1 through choices of $\Omega , \lambda ,$ , and $C _ { t }$

Greedy decoding. Set $\Omega ( q ) = 0 , \lambda = 0 .$ , and $C _ { t } = \Delta ( V )$ . The objective reduces to maximising $\langle q , s _ { t } \rangle$ . The optimality conditions become

$$
\begin{array} { r } { q _ { t } ^ { \star } ( v ) > 0 \implies s _ { t } ( v ) = \eta , } \\ { q _ { t } ^ { \star } ( v ) = 0 \implies s _ { t } ( v ) \leq \eta . } \end{array}\tag{15}
$$

Since at least one probability is positive, $\eta = \operatorname* { m a x } _ { v \in V } s _ { t } ( v )$ . Thus any optimum places all its mass on the highest-scoring tokens. If the maximiser $v _ { t } ^ { \star }$ is unique, the solution is $q _ { t } ^ { \star } ( v _ { t } ^ { \star } ) = 1$ and zero elsewhere. With tied scores, choosing a point mass on a maximiser according to the tie-breaking rule recovers deterministic greedy decoding.

Softmax sampling. For the negative Shannon entropy regulariser $\begin{array} { r } { \Omega ( q ) = \sum _ { v \in V } q ( v ) } \end{array}$ log $q ( v )$ $\lambda > 0$ , and $C _ { t } = \Delta ( V )$ , we have $\partial \Omega ( q ) / \partial q ( v ) = 1 + \log q ( v )$ , and Eq. 3 becomes $s _ { t } ( v ) - \lambda ( 1 +$ log $q _ { t } ^ { \star } ( v ) ) = \eta$ . Solving for $q _ { t } ^ { \star } ( v )$ and imposing normalisation gives

$$
q _ { t } ^ { \star } ( v ) = \frac { \exp ( s _ { t } ( v ) / \lambda ) } { \sum _ { u \in V } \exp ( s _ { t } ( u ) / \lambda ) } .\tag{16}
$$

This recovers softmax sampling with temperature $\lambda .$

Top-K sampling. Let $S _ { t }$ contain the K highest-scoring tokens. Choose negative Shannon entropy $\begin{array} { r } { \Omega ( \bar { q } ) = \sum _ { v \in V } \bar { q } ( v ) } \end{array}$ log q(v), $\lambda > 0$ , and

$$
C _ { t } = \{ q \in \Delta ( V ) : q ( v ) = 0 \mathrm { f o r } v \notin S _ { t } \} .\tag{17}
$$

The objective is therefore restricted to $\Delta ( S _ { t } )$ ):

$$
\operatorname* { m a x } _ { q \in \Delta ( S _ { t } ) } \left[ \sum _ { v \in S _ { t } } q ( v ) s _ { t } ( v ) - \lambda \sum _ { v \in S _ { t } } q ( v ) \log q ( v ) \right] .\tag{18}
$$

The entropy-regularised optimum is positive on $S _ { t } .$ . Substituting its derivative into Eq. 3 gives

$$
s _ { t } ( v ) - \lambda ( 1 + \log q _ { t } ^ { \star } ( v ) ) = \eta \quad \Longrightarrow \quad q _ { t } ^ { \star } ( v ) \propto \exp ( s _ { t } ( v ) / \lambda ) .\tag{19}
$$

Normalising over $S _ { t }$ yields

$$
q _ { t } ^ { \star } ( v ) = \left\{ \begin{array} { l l } { \frac { \exp ( s _ { t } ( v ) / \lambda ) } { \sum _ { u \in S _ { t } } \exp ( s _ { t } ( u ) / \lambda ) } , } & { v \in S _ { t } , } \\ { 0 , } & { v \not \in S _ { t } . } \end{array} \right.\tag{20}
$$

This is Top-K sampling with temperature λ; zeros outside $S _ { t }$ are enforced by the support constraint.

Top-P (nucleus) sampling. Top-P retains the same regulariser and changes the support selection rule. Let $p _ { t } ^ { ( \lambda ) }$ be the full-vocabulary softmax distribution in Eq. 16, and order tokens by decreasing probability. For a threshold $p \in ( 0 , 1 ]$ ], define

$$
m _ { t } = \operatorname* { m i n } \left\{ m : \sum _ { i = 1 } ^ { m } p _ { t } ^ { ( \lambda ) } ( v _ { ( i ) } ) \geq p \right\} , \qquad S _ { t } = \{ v _ { ( 1 ) } , \ldots , v _ { ( m _ { t } ) } \} .\tag{21}
$$

Using this $S _ { t }$ in Eq. 17 gives the same restricted entropy objective as Top-K. Its solution is therefore Eq. 20, which renormalises $p _ { t } ^ { ( \lambda ) }$ over the nucleus. This construction applies temperature scaling before nucleus selection. The support is determined from the base distribution and held fixed when optimising $q .$

Sparsemax. Choose $\begin{array} { r } { \Omega ( q ) = \frac { 1 } { 2 } \lVert q \rVert _ { 2 } ^ { 2 } , \lambda > 0 } \end{array}$ , and $C _ { t } = \Delta ( V )$ . The objective becomes

$$
\operatorname* { m a x } _ { q \in \Delta ( V ) } \left[ \langle q , s _ { t } \rangle - \frac { \lambda } { 2 } \| q \| _ { 2 } ^ { 2 } \right] .\tag{22}
$$

Since $\partial \Omega ( q ) / \partial q ( v ) = q ( v )$ , the two optimality conditions give

$$
\begin{array} { l l l } { q _ { t } ^ { \star } ( v ) > 0 \implies q _ { t } ^ { \star } ( v ) = \frac { s _ { t } ( v ) - \eta } { \lambda } , } \\ { q _ { t } ^ { \star } ( v ) = 0 \implies s _ { t } ( v ) \leq \eta . } \end{array}\tag{23}
$$

Combining them with normalisation yields

$$
q _ { t } ^ { \star } ( v ) = \frac { [ s _ { t } ( v ) - \eta ] _ { + } } { \lambda } , \qquad \sum _ { v \in V } [ s _ { t } ( v ) - \eta ] _ { + } = \lambda ,\tag{24}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ and the second equation uniquely determines η. Equivalently, completing the square gives $q _ { t } ^ { \star } =$ sparsemax $ { \left( s _ { t } / \lambda \right) }$ , with standard sparsemax recovered at $\lambda = 1$ . Here zero probabilities arise from the quadratic regulariser and the boundary condition, without a prescribed support set.

Table 1: Top-k sampling and BoK (KL + Diversity) at three different temperatures on MATH500 with Qwen2.5-7B. Both use Top-k support with $k = 2 0 0 ;$ ; τ denotes temperature.
<table><tr><td>Method</td><td>T</td><td>Pass@1↑</td><td>Pass@4↑</td><td>Pass@16 ↑</td></tr><tr><td rowspan="3">Top-k (k = 200)</td><td>0.25</td><td>62.8</td><td>82.0</td><td>92.0</td></tr><tr><td>0.5</td><td>59.8</td><td>81.2</td><td>90.0</td></tr><tr><td>0.7</td><td>52.0</td><td>77.2</td><td>87.6</td></tr><tr><td rowspan="3">BoK (KL + Diversity)</td><td>0.25</td><td>65.2</td><td>84.0</td><td>92.0</td></tr><tr><td>0.5</td><td>64.4</td><td>81.4</td><td>90.0</td></tr><tr><td>0.7</td><td>60.4</td><td>77.8</td><td>88.6</td></tr></table>

## C.2 SINGLE AND COMPOSED REGULARISERS: RELATION TO TEMPERATURE SCALING

We consider a single decoding step on a fixed support $S _ { t }$ , with $\lambda > 0$ . The scores $s _ { t }$ are temperaturescaled model logits, and $p _ { t }$ is the softmax of the model logits. We write softmax $\cdot { \cal S } _ { t }$ for normalisation over $S _ { t } ,$ , with zero probability outside the support.

KL and entropy. KL regularisation is negative entropy regularisation with an additional linear term determined by the reference distribution.

$$
\mathrm { K L } ( q | | p _ { t } ) = - H ( q ) - \left. q , \log p _ { t } \right.\tag{25}
$$

For the single-regulariser objective in Eq. 1, the corresponding solutions are

$$
\begin{array} { r } { q _ { t } ^ { \star } = \left\{ \begin{array} { l l } { \operatorname { s o f t m a x } _ { S _ { t } } ( s _ { t } / \lambda ) , } & { \Omega ( q ) = - H ( q ) , } \\ { \operatorname { s o f t m a x } _ { S _ { t } } ( \log p _ { t } + s _ { t } / \lambda ) , } & { \Omega ( q ) = \mathrm { K L } ( q \| p _ { t } ) . } \end{array} \right. } \end{array}\tag{26}
$$

Because log $p _ { t }$ is a positive rescaling of $s _ { t }$ up to an additive constant, both solutions amount to temperature scaling of the same model logits on $S _ { t }$ . At the same $\lambda ,$ the extra log $p _ { t }$ term makes the KL solution more concentrated on high-scoring tokens. The distinction is clearest when regularisation dominates the score term: as $\lambda \to \infty$ , the entropy solution approaches the uniform distribution on $S _ { t } .$ , while the KL solution approaches $p _ { t }$

Composition goes beyond temperature scaling. We note that including KL or entropy in a composed objective does not restrict the decoder to temperature scaling. For BoK in Eq. 13, with α<sub>KL</sub> $> 0$ and $\alpha _ { U } > 0$ , the stationarity condition gives

$$
\begin{array} { c } { q _ { t } ^ { \star } = \displaystyle { \mathrm { s o f t m a x } } _ { S _ { t } } \left( \log p _ { t } + \frac { s _ { t } } { \lambda \alpha _ { \mathrm { K L } } } + \frac { \alpha _ { U } } { \alpha _ { \mathrm { K L } } } \nabla U _ { K , t } ( q _ { t } ^ { \star } ) \right) , } \\ { \left[ \nabla U _ { K , t } ( q _ { t } ^ { \star } ) \right] _ { v } = w _ { t } ( v ) K ( 1 - q _ { t } ^ { \star } ( v ) ) ^ { K - 1 } , \qquad v \in S _ { t } . } \end{array}\tag{27}
$$

The first two terms inside the softmax have the same temperature-scaling form as the KL decoder above. The utility term adds a separate bonus to each token: the bonus increases with $w _ { t } ( v )$ and, for $K > 1$ , decreases as $q _ { t } ^ { \star } ( v )$ increases. This gives a smaller reward for increasing a token’s probability when it is already likely to appear among the K samples. Temperature scaling multiplies all score differences by the same factor. The utility bonuses need not change these differences in the same proportion, so BoK is not restricted to temperature scaling. The equation describes the exact optimum, which the finite-step solver approximates. Table 1 compares the Top-k baseline and BoK (KL + Diversity) across three configured temperatures. On this grid, BoK matches or improves on the baseline with different preferences compared with the Top-k decoder at each temperature.

## D ADDITIONAL EXPERIMENTAL RESULTS

This section supplements the experimental results in Section 5. We first describe the model checkpoints, benchmarks, decoding settings, and evaluation protocols. We then present detailed results for selected configurations and their variation across random seeds, examine solver convergence and computational cost, and analyse the effects of regularisation strength and composition weights.

## D.1 EXPERIMENTAL SETUP

Our experiments use the shared decoding and evaluation interface of COMPOSIMPLEX. The implementation, experiment configurations, and evaluation scripts are provided in our code repository<sup>1</sup>. The configurations specify the model checkpoint, sampling parameters, random seed, and evaluation settings.

Models and benchmarks. We evaluate four model–benchmark pairs across instruction following, scientific reasoning, mathematical reasoning, and code generation, using base and instruction-tuned models at different scales. For instruction following, we use LFM2.5-1.2B-Base (Amini et al., 2025)<sup>2</sup> on the 541 evaluation prompts of IFEval (Zhou et al., 2023)<sup>3</sup>. For scientific reasoning, we use Qwen3-4B-Base (Yang et al., 2025)<sup>4</sup> on all 198 questions in GPQA Diamond (Rein et al., 2024)<sup>5</sup>. For mathematical reasoning, we use Qwen2.5-7B (Qwen et al., 2025)<sup>6</sup> on the 500-problem MATH500 test split of MATH (Hendrycks et al., 2021)<sup>7</sup>. For code generation, we use Gemma-4-26B-A4B-IT (Gemma Team, 2026)<sup>8</sup> on the 175 new problems introduced in LiveCodeBenchv6 (Jain et al., 2025)<sup>9</sup>.

Generation and decoding settings. We use the Hugging Face Transformers backend for the evaluation while also providing the vLLM backend in the open-source library. Unless otherwise stated, we sample 16 completions per prompt at temperature $\dot { T } = 0 . 5 ,$ , regularised objectives use λ = 1, and compositions assign equal weights to their constituent primitives. Top-k support uses k = 200 and Top-p support uses $p = 0 . 9$ unless otherwise indicated. Min-p uses a relative threshold of 0.05, typical sampling uses a cumulative mass of 0.95, and η-sampling uses eta cutof $\mathrm { \Delta f = 5 \times 1 0 ^ { - 4 } }$ Completions terminate at a model-specific end-of-sequence token or after a maximum of 3072 completion tokens. For a fixed task, decoder comparisons use the same prompt and completion-index seed schedule. The exact benchmark prompts and model-specific chat-template settings are provided in our code repository.

We use the closed-form solution for single KL and entropy objectives. Other evaluated regularised objectives use 10 mirror-ascent updates with step size 0.1. The solver and coefficient studies vary these settings explicitly.

Task-specific grading. For MATH500, we extract the final answer with the last boxed expression. The grader normalises mathematical expressions and checks agreement with the reference through exact comparison and symbolic equivalence checks using SymPy. For GPQA, we extract an option letter from {A, B, C, D} and compare it with the reference option. For MATH500, GPQA, and IFEval, we encode extracted reasoning traces using sentence-transformers/all-MiniLM-L6-v2, and calculate the mean pairwise cosine distance within each prompt, averaged across prompts. For IFEval, we use the official instructionfollowing evaluator<sup>10</sup> and report strict prompt-level correctness: a completion passes only if it satisfies every instruction associated with the prompt. For LiveCodeBench, we extract Python code from the generated response and use the official execution-based grader<sup>11</sup>. A completion passes only when all test cases returned by the evaluator pass; compilation errors, runtime errors, and timeouts (6s) count as failures.

Distributional metrics. At each generation step, we measure $\mathrm { K L } ( q | | p ) , \mathrm { J S } ( q , p )$ , entropy, expected coverage, and diversity-gap utility on the same selected support. For comparable coverage scores across decoders, the reported metric uses the top $k = \mathrm { m i n } ( 8 , | S | )$ reference tokens $I _ { k }$ and the normalisation $\begin{array} { r } { \mathrm { C o v } _ { k } ( q ) = \frac { \sum _ { v \in I _ { k } } \left[ 1 - ( 1 - q ( v ) ) ^ { 1 6 } \right] } { k \left[ 1 - ( 1 - 1 / k ) ^ { 1 6 } \right] } } \end{array}$ . This reporting normalisation differs from the $\ell _ { 2 }$ -normalised weights in the Coverage objective. Diversity-gap utility uses $K = 1 6$ and normalised weights proportional to $\Delta _ { v } \exp ( - \Delta _ { v } )$ , where $\Delta _ { v }$ is the gap between the largest supported logit and token $v { \mathrm { s } }$ logit. Each distributional metric is first averaged over generation steps within a completion and then across completions. These statistics describe the distributions encountered along each decoder’s generated trajectories on average.

<sup>1</sup>https://github.com/KickItLikeShika/composimplex   
<sup>2</sup>https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base   
<sup>3</sup>https://huggingface.co/datasets/google/IFEval   
<sup>4</sup>https://huggingface.co/Qwen/Qwen3-4B-Base   
<sup>5</sup>https://huggingface.co/datasets/Idavidrein/gpqa   
<sup>6</sup>https://huggingface.co/Qwen/Qwen2.5-7B   
<sup>7</sup>https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark   
<sup>8</sup>https://huggingface.co/google/gemma-4-26B-A4B-it   
<sup>9</sup>https://huggingface.co/datasets/livecodebench/code\_generation\_lite   
<sup>10</sup>https://github.com/google-research/google-research/tree/master/   
instruction\_following\_eval   
<sup>11</sup>https://github.com/LiveCodeBench/LiveCodeBench

## D.2 DETAILED PERFORMANCE EVALUATION

We provide detailed numerical results for the four model–benchmark pairs evaluated in Section 5: MATH500 with Qwen2.5-7B, GPQA Diamond with Qwen3-4B-Base, IFEval with LFM2.5-1.2B-Base, and LiveCodeBench v6 with Gemma-4-26B-A4B-IT. Each table groups the standard sampler, single primitives, and evaluated compositions within the corresponding support setting. We compare each composition with its constituent primitives to examine which aspects of their performance profiles are retained or changed.

Qwen2.5-7B. Table 2 reports single-sample and multi-sample success, self-consistency accuracy, and semantic diversity under Top-k and Min-p support. Under Top-k, KL + Diversity retains KL’s pass@1 of 64.4% while increasing pass@16 from 88.4% to 90.0%, below Diversity’s 91.0%. Its SemDiv also lies between the two constituents. Under Min-p, JS + Coverage matches Coverage’s pass@4 and SC@16 and exceeds both constituent primitives on pass@1 and pass@16, while its SemDiv lies between them.
<table><tr><td>Method</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16↑</td><td>SC@16↑</td><td>SemDiv ↑</td></tr><tr><td colspan="6">Qwen2.5-7B</td></tr><tr><td>Top-k</td><td>59.8</td><td>81.2</td><td>90.0</td><td>78.0</td><td>0.150</td></tr><tr><td colspan="6">Single objective primitives</td></tr><tr><td>KL</td><td>64.4</td><td>80.6</td><td>88.4</td><td>76.8</td><td>0.132</td></tr><tr><td>JS</td><td>63.6</td><td>82.2</td><td>89.0</td><td>76.2</td><td>0.136</td></tr><tr><td>Entropy</td><td>59.8</td><td>81.2</td><td>90.0</td><td>78.0</td><td>0.150</td></tr><tr><td>Coverage</td><td>63.4</td><td>82.8</td><td>89.4</td><td>78.4</td><td>0.157</td></tr><tr><td>Diversity</td><td>56.0</td><td>79.4</td><td>91.0</td><td>76.2</td><td>0.159</td></tr><tr><td colspan="6">Compositions</td></tr><tr><td>KL + Diversity</td><td>64.4</td><td>81.4</td><td>90.0</td><td>77.2</td><td>0.149</td></tr><tr><td>JS + Coverage</td><td>64.8</td><td>82.2</td><td>89.4</td><td>76.2</td><td>0.144</td></tr><tr><td>JS + Coverage + Diversity</td><td>65.2</td><td>81.2</td><td>89.4</td><td>76.8</td><td>0.145</td></tr><tr><td>Min-p</td><td>64.4</td><td>84.2</td><td>89.6</td><td>77.6</td><td>0.1396</td></tr><tr><td colspan="6">Single objective primitives</td></tr><tr><td>KL</td><td>65.2</td><td>81.6</td><td>87.8</td><td>76.2</td><td>0.1275</td></tr><tr><td>JS</td><td>66.6</td><td>83.0</td><td>89.2</td><td>77.0</td><td>0.1298</td></tr><tr><td>Entropy</td><td>64.4</td><td>84.2</td><td>89.6</td><td>77.6</td><td>0.1396</td></tr><tr><td>Coverage</td><td>65.4</td><td>84.6</td><td>89.4</td><td>78.0</td><td>0.1395</td></tr><tr><td>Diversity</td><td>65.2</td><td>84.4</td><td>90.4</td><td>76.8</td><td>0.1402</td></tr><tr><td colspan="6">Compositions</td></tr><tr><td>JS + Coverage</td><td>67.2</td><td>84.6</td><td>90.2</td><td>78.0</td><td>0.1347</td></tr><tr><td>KL + Coverage</td><td>64.6</td><td>81.6</td><td>89.2</td><td>77.0</td><td>0.1380</td></tr><tr><td>KL + Diversity + Entropy</td><td>65.4</td><td>82.8</td><td>90.0</td><td>77.8</td><td>0.1390</td></tr></table>

Table 2: MATH500 performance of Qwen2.5-7B under Top-k and Min-p support. Rows compare the standard sampler, individual regularisers, and compositions. Accuracy values are percentages.

Qwen3-4B-Base. Table 3 presents the same metrics under Top-k and typical support. Under typical support, JS + Diversity retains JS’s pass@1 of 26.3% while increasing pass@16 from 80.8% to 82.8%, closer to Diversity’s 83.3%; its SC@16 lies between the constituent values. Under Top-k, the same composition exceeds both constituents on pass@1 and pass@4, but records lower pass@16 than either constituent. The resulting profile therefore depends on the support rule as well as the composed objectives.
<table><tr><td>Method</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>SC@16↑</td><td>SemDiv ↑</td></tr><tr><td colspan="6">Qwen3-4B-Base</td></tr><tr><td>Top-k</td><td>15.7</td><td>51.0</td><td>80.3</td><td>12.6</td><td>0.710</td></tr><tr><td colspan="6">Single objective primitives</td></tr><tr><td>KL</td><td>21.2</td><td>50.0</td><td>78.3</td><td>24.7</td><td>0.564</td></tr><tr><td>JS</td><td>23.2</td><td>52.5</td><td>80.3</td><td>22.2</td><td>0.584</td></tr><tr><td>Entropy</td><td>15.7</td><td>51.0</td><td>80.3</td><td>12.6</td><td>0.710</td></tr><tr><td>Coverage</td><td>13.1</td><td>54.0</td><td>84.3</td><td>22.2</td><td>0.694</td></tr><tr><td>Diversity</td><td>24.7</td><td>55.6</td><td>85.4</td><td>28.8</td><td>0.555</td></tr><tr><td colspan="6">Compositions</td></tr><tr><td>KL + Coverage</td><td>15.2</td><td>54.5</td><td>83.3</td><td>20.7</td><td>0.670</td></tr><tr><td>KL + Diversity</td><td>21.2</td><td>50.5</td><td>83.8</td><td>21.2</td><td>0.619</td></tr><tr><td>JS + Diversity</td><td>26.3</td><td>57.6</td><td>79.8</td><td>23.2</td><td>0.581</td></tr><tr><td>Typical</td><td>20.7</td><td>57.6</td><td>80.3</td><td>14.1</td><td>0.708</td></tr><tr><td colspan="6">Single objective primitives</td></tr><tr><td>KL</td><td>23.7</td><td>57.1</td><td>79.8</td><td>24.7</td><td>0.561</td></tr><tr><td>JS</td><td>26.3</td><td>54.0</td><td>80.8</td><td>23.2</td><td>0.578</td></tr><tr><td>Entropy</td><td>20.7</td><td>57.6</td><td>80.3</td><td>14.1</td><td>0.708</td></tr><tr><td>Coverage</td><td>14.1</td><td>56.1</td><td>82.3</td><td>21.2</td><td>0.686</td></tr><tr><td>Diversity</td><td>21.7</td><td>52.5</td><td>83.3</td><td>25.8</td><td>0.579</td></tr><tr><td colspan="6">Compositions</td></tr><tr><td>KL + Diversity</td><td>24.2</td><td>56.1</td><td>79.8</td><td>23.2</td><td>0.620</td></tr><tr><td>JS + Diversity</td><td>26.3</td><td>56.6</td><td>82.8</td><td>24.7</td><td>0.577</td></tr><tr><td>KL + Coverage + Diversity</td><td>25.3</td><td>60.1</td><td>83.3</td><td>22.2</td><td>0.635</td></tr></table>

Table 3: GPQA Diamond performance of Qwen3-4B-Base under Top-k and typical support. Rows compare the standard sampler, individual regularisers, and compositions. Accuracy values are percentages.

LFM2.5-1.2B-Base. Table 4 reports strict prompt-level success and semantic diversity under Topp and η-sampling. With η-sampling, KL + Diversity reaches pass@16 of 81.9%, compared with 80.0% for KL and 81.5% for Diversity, and records higher pass@4 and SemDiv than either constituent. Its pass@1 and all-pass@16 are lower than those of both constituents. Here, higher multisample success and semantic diversity coexist with lower reliability across repeated responses.

Gemma-4-26B-A4B-IT. Table 5 reports execution-based success on the 175 problems in the Live-CodeBench v6 increment, together with all-pass@16 and pass@1 on its 80 Hard problems. Under Top-p, JS + Coverage matches Coverage’s pass@1 of 56.6%, while its Hard pass@1 lies between those of JS and Coverage. Its pass@4 reaches 66.9%, above JS’s 65.7% and Coverage’s 64.6%, but its pass@16 falls to 67.4%, compared with 69.1% for both constituents.

Standard deviation across random seeds. Table 6 lists standard deviations with Qwen2.5-7B on MATH500 under Top-k support using seeds 0, 42 and 1234. For pass@1, pass@4, pass@16, and SC@16, most listed standard deviations are below one percentage point, with a range of 0.12–1.33 percentage points.

<table><tr><td>Method</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16↑</td><td>all-pass@16↑</td><td>SemDiv ↑</td></tr><tr><td>LFM2.5-1.2B-Base</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Top-p</td><td>52.1</td><td>70.2</td><td>80.0</td><td>21.8</td><td>0.284</td></tr><tr><td>Single objective primitives</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL</td><td>49.9</td><td>70.1</td><td>79.7</td><td>21.6</td><td>0.272</td></tr><tr><td>JS</td><td>51.4</td><td>67.8</td><td>79.5</td><td>20.9</td><td>0.274</td></tr><tr><td>Coverage</td><td>50.5</td><td>67.5</td><td>79.5</td><td>19.4</td><td>0.286</td></tr><tr><td>Diversity</td><td>49.9</td><td>69.5</td><td>78.9</td><td>21.1</td><td>0.285</td></tr><tr><td>Compositions</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JS + Entropy</td><td>53.1</td><td>70.4</td><td>80.0</td><td>22.0</td><td>0.283</td></tr><tr><td>KL + Coverage</td><td>53.0</td><td>69.7</td><td>77.8</td><td>20.5</td><td>0.283</td></tr><tr><td>JS + Entropy + Diversity</td><td>52.7</td><td>70.8</td><td>79.5</td><td>21.1</td><td>0.282</td></tr><tr><td>η-sampling</td><td>50.5</td><td>68.9</td><td>80.6</td><td>17.2</td><td>0.294</td></tr><tr><td>Single objective primitives</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL</td><td>51.8</td><td>68.9</td><td>80.0</td><td>20.1</td><td>0.276</td></tr><tr><td>JS</td><td>51.9</td><td>68.9</td><td>80.0</td><td>20.9</td><td>0.279</td></tr><tr><td>Coverage</td><td>52.7</td><td>70.8</td><td>81.5</td><td>14.8</td><td>0.305</td></tr><tr><td>Diversity</td><td>51.6</td><td>70.8</td><td>81.5</td><td>16.6</td><td>0.286</td></tr><tr><td>Compositions</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL + Diversity</td><td>50.6</td><td>71.5</td><td>81.9</td><td>14.8</td><td>0.296</td></tr><tr><td>Coverage + Entropy</td><td>51.4</td><td>70.8</td><td>81.0</td><td>14.6</td><td>0.302</td></tr><tr><td>KL + Coverage</td><td>51.9</td><td>72.3</td><td>83.4</td><td>16.1</td><td>0.298</td></tr></table>

Table 4: IFEval performance of LFM2.5-1.2B-Base under Top-p and η-sampling support. Success uses strict prompt-level grading, and success rates are percentages.
<table><tr><td>Method</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>all-pass@16 ↑</td><td>Hard pass@1 ↑</td></tr><tr><td>Gemma-4-26B-A4B-IT</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Top-p</td><td>56.6</td><td>63.4</td><td>69.7</td><td>41.7</td><td>25.0</td></tr><tr><td>Single objective primitives</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL</td><td>55.4</td><td>63.4</td><td>69.7</td><td>40.0</td><td>26.3</td></tr><tr><td>JS</td><td>57.1</td><td>65.7</td><td>69.1</td><td>38.9</td><td>27.5</td></tr><tr><td>Coverage</td><td>56.6</td><td>64.6</td><td>69.1</td><td>38.9</td><td>25.0</td></tr><tr><td>Diversity</td><td>55.4</td><td>64.0</td><td>70.3</td><td>41.1</td><td>27.5</td></tr><tr><td>Compositions</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL + Diversity</td><td>56.6</td><td>65.7</td><td>68.6</td><td>41.7</td><td>26.3</td></tr><tr><td>JS + Coverage</td><td>56.6</td><td>66.9</td><td>67.4</td><td>39.4</td><td>26.3</td></tr><tr><td>Min-p</td><td>55.4</td><td>62.9</td><td>68.6</td><td>38.3</td><td>23.8</td></tr><tr><td>Single objective primitives</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL</td><td>59.4</td><td>65.1</td><td>68.6</td><td>40.6</td><td>33.8</td></tr><tr><td>Coverage</td><td>57.1</td><td>63.4</td><td>69.1</td><td>38.3</td><td>26.3</td></tr><tr><td>JS</td><td>54.8</td><td>63.4</td><td>68.6</td><td>38.9</td><td>27.5</td></tr><tr><td>Diversity</td><td>58.3</td><td>65.1</td><td>69.1</td><td>38.9</td><td>25.0</td></tr><tr><td>Compositions</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KL + Coverage</td><td>56.0</td><td>65.7</td><td>69.1</td><td>38.3</td><td>26.3</td></tr><tr><td>JS + Diversity</td><td>54.8</td><td>65.1</td><td>69.7</td><td>37.7</td><td>27.5</td></tr></table>

Table 5: LiveCodeBench v6 performance of Gemma-4-26B-A4B-IT under Top-p and Min-p support. Hard pass@1 is measured on the 80 Hard problems. All reported success rates are percentages.

<table><tr><td></td></tr><tr><td>Method</td><td>pass@1  $\mathrm { p a s s } @ 4$ </td><td>pass@16</td><td>SC@16</td><td>SemDiv  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>Top-k</td><td>0.70</td><td>0.12</td><td>0.81</td><td>0.92</td></tr><tr><td>KL</td><td>0.31</td><td>1.13</td><td>0.42</td><td>2.63 0.53 3.99</td></tr><tr><td>JS</td><td>0.12</td><td>0.83</td><td>0.42 0.31</td><td>3.64</td></tr><tr><td>Coverage</td><td>0.83</td><td>0.71</td><td>0.71</td><td>0.83 2.56</td></tr><tr><td>Diversity</td><td>1.33</td><td>0.12</td><td>1.27</td><td>0.71 3.64</td></tr><tr><td> ${ \mathrm { K L } } + { \mathrm { D i v e r s i t y } }$ </td><td>0.72</td><td>0.71</td><td>1.11</td><td>0.31 3.42</td></tr><tr><td> $\mathbf { J } \mathbf { S } + \mathbf { C } \mathbf { o v e r a g e } + \mathbf { D i v e r s i t y }$ </td><td>0.90</td><td>0.53</td><td>0.42</td><td>0.42 3.19</td></tr></table>

Table 6: Sample standard deviations across seeds 0, 42, and 1234 on MATH500 with Qwen2.5-7B and Top-k support. Deviations for pass@k and SC@16 are in percentage points; SemDiv deviations are shown in units of $1 0 ^ { - 3 }$

## D.3 COMPUTATIONAL EFFICIENCY

Solver convergence. We examine the effect of the learning rate (step size) and iteration budget using the same 128 cached prefixes for every configuration. We fix λ = 1 and vary the step size over {0.05, 0.1, 0.5} and the number of updates over {5, 10, 25, 50}. Figure 4 reports the mean $L _ { 1 }$ distance between the iteratively computed token distribution and a reference optimum. We use analytic solutions for KL and Entropy and independently compute numerical reference solutions for the remaining objectives by solving the KKT conditions in float64, using bisection on the simplex normalisation multiplier and nested coordinate bisection where required.

The distance generally decreases with more updates. At step size 0.1, the mean $L _ { 1 }$ distance is below 0.009 for every displayed objective after 50 updates. A step size of 0.5 often reaches a smaller distance with fewer updates, but is not uniformly better. Our task-level runs use 10 updates with step size 0.1 for iterative objectives, for which the mean distances range from 0.033 to 0.062. These runs therefore use finite-step approximations. KL and entropy use closed-form solutions in the tasklevel experiments.

Computational cost. Table 7 reports generation cost for selected configurations, using Top-k support on MATH500 and GPQA and Top-p support on IFEval and LiveCodeBench. All timing runs use seed 0. We compute the amortised milliseconds per output token as 1000 divided by the recorded output token throughput. The closed-form KL decoder has a recorded cost close to baseline decoding. Iterative optimisation for both single and compositional objectives incurs additional cost: on MATH500, JS, Coverage and Diversity require 4.329–5.751 ms/token, compared with 2.258 ms/token for the baseline; on LiveCodeBench, they require 20.954–21.255 ms/token, compared with 18.560 ms/token. The relative overhead varies across the recorded model and batching configurations.

![](images/8fbcfbcb7a9c567a85e935675ced2faa21a9c9b46f65b1ae7f3b2dfb38755a18.jpg)  
Figure 4: Mirror-ascent convergence for individual and composed objectives over 128 fixed prefixes. Each panel reports mean $L _ { 1 }$ distance to a reference optimum against the number of updates; curves correspond to step sizes 0.05, 0.1, and 0.5. The vertical axis uses a square-root scale with tick labels in the original $L _ { 1 }$ units.

<table><tr><td>Method</td><td>MATH500 Top-k</td><td>GPQA Top-k</td><td>IFEval Top-p</td><td>LiveCodeBench Top-p</td></tr><tr><td>Baseline</td><td>2.258</td><td>3.093</td><td>1.833</td><td>18.560</td></tr><tr><td>KL</td><td>2.238</td><td>3.068</td><td>1.838</td><td>18.308</td></tr><tr><td>JS</td><td>4.329</td><td>5.266</td><td>3.924</td><td>21.255</td></tr><tr><td>Coverage</td><td>5.004</td><td>5.910</td><td>4.399</td><td>20.954</td></tr><tr><td>Diversity</td><td>5.751</td><td>6.273</td><td>4.624</td><td>21.245</td></tr><tr><td>KL + Coverage</td><td>5.308</td><td>6.121</td><td>4.656</td><td>21.378</td></tr><tr><td>JS + Entropy + Diversity</td><td>5.832</td><td>6.341</td><td>5.285</td><td>21.652</td></tr></table>

Table 7: Generation cost in milliseconds per output token for selected objectives. Columns use the model–benchmark pair and support rule indicated in the table. The baseline uses the base support rule without an added regulariser.

## D.4 REGULARISATION STRENGTH AND COMPOSITION WEIGHTS

We study two ways of changing the decoding objective: varying the global regularisation strength λ at fixed composition weights, and varying the relative weights at fixed λ. All experiments in this subsection use Qwen2.5-7B on MATH500, Top-k support, and seed 0. Other settings follow Appendix D.1.

Regularisation-strength sweep. Table 8 groups the results into four families: KL + Coverage, KL + Diversity, JS + Coverage, and JS + Diversity. Each block compares the two constituent primitives with their equally weighted composition at λ ∈ {0.5, 1, 2}. Alongside pass@1, pass@4, pass@16, SC@16, and SemDiv, the final two columns report the divergence and utility associated with that family.

Increasing λ increases the corresponding utility and SemDiv within each of the four evaluated compositions, while reducing its KL or JS divergence. Stronger utility regularisation does not necessarily improve accuracy: at λ = 2, single Coverage and Diversity reach their highest respective utilities, but their pass@1 falls to 41.0% and 39.8%. The four compositions retain pass@1 between 57.0% and 62.6% at the same global strength, with lower utility values than the corresponding pure utility objectives.

![](images/fdd917c3bd716e0c26f45ee2b9e618223a86cb44a51fccca98639a82472f1200.jpg)  
Figure 5: Effect of composition weight α on MATH500 with Qwen2.5-7B and Top-k support at $\lambda = 1$ . The top row shows KL + Coverage and the bottom row KL + Diversity. Columns show the corresponding utility, KL divergence, SemDiv, and pass@1 and pass@16.

Relative composition weights. At λ = 1, we vary the utility weight $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 1 \}$ in KL + Coverage and KL + Diversity. Figure 5 reports the corresponding utility, KL divergence, Sem-Div, pass@1, and pass@16. Across the evaluated weights, increasing α monotonically increases the corresponding utility and decreases KL divergence in both families. These improvements show that the composed objectives shape the next-token distributions in the desired directions. These distributional improvements do not translate into consistent gains in pass@1 or pass@16 as α increases. Nearby composition weights nevertheless yield broadly similar task performance, suggesting limited sensitivity to the precise choice of α within a small range.

<table><tr><td colspan="9">KL + Coverage</td></tr><tr><td>Method</td><td>λ</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>SC@16↑</td><td>SemDiv ↑</td><td>KL↓</td><td>Coverage ↑</td></tr><tr><td rowspan="4">KL</td><td>0.5</td><td>65.8</td><td>82.0</td><td>88.0</td><td>73.4</td><td>0.111</td><td>0.0501</td><td>0.1498</td></tr><tr><td>1</td><td>64.4</td><td>80.6</td><td>88.4</td><td>76.8</td><td>0.132</td><td>0.0380</td><td>0.1564</td></tr><tr><td>2</td><td>59.8</td><td>81.2</td><td>90.0</td><td>78.0</td><td>0.150</td><td>0.0240</td><td>0.1663</td></tr><tr><td>0.5</td><td>64.4</td><td>82.4</td><td>89.4</td><td>76.8</td><td>0.142</td><td>0.0310</td><td>0.1622</td></tr><tr><td rowspan="4">Coverage</td><td>1</td><td>63.4</td><td>82.8</td><td>89.4</td><td>78.4</td><td>0.157</td><td>0.0225</td><td>0.1729</td></tr><tr><td>2</td><td>41.0</td><td>75.2</td><td>87.8</td><td>74.2</td><td>0.225</td><td>0.0182</td><td>0.2415</td></tr><tr><td>0.5</td><td>62.8</td><td>82.2</td><td>88.8</td><td>76.0</td><td>0.140</td><td>0.0323</td><td>0.1604</td></tr><tr><td>1</td><td>61.8</td><td>82.4</td><td>89.0</td><td>76.6</td><td>0.148</td><td>0.0263</td><td>0.1659</td></tr><tr><td>KL + Coverage</td><td>2</td><td>60.2</td><td>81.4</td><td>89.2</td><td>77.2</td><td>0.168</td><td>0.0158</td><td>0.1817</td></tr><tr><td>KL + Diversity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Method</td></tr><tr><td rowspan="3">KL</td><td>λ</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>SC@16↑</td><td>SemDiv ↑</td><td>KL↓</td><td>DivGap ↑</td></tr><tr><td>0.5</td><td>65.8</td><td>82.0</td><td>88.0</td><td>73.4</td><td>0.111</td><td>0.0501</td><td>0.0226</td></tr><tr><td>1 2</td><td>64.4 59.8</td><td>80.6</td><td>88.4</td><td>76.8</td><td>0.132</td><td>0.0380</td><td>0.0444</td></tr><tr><td rowspan="4">Diversity</td><td></td><td></td><td>81.2</td><td>90.0</td><td>78.0</td><td>0.150</td><td>0.0240</td><td>0.0747</td></tr><tr><td>0.5</td><td>62.4</td><td>81.4</td><td>89.2</td><td>76.8</td><td>0.144</td><td>0.0287</td><td>0.0808</td></tr><tr><td>1</td><td>56.0</td><td>79.4</td><td>91.0</td><td>76.2</td><td>0.159</td><td>0.0237</td><td>0.1377</td></tr><tr><td>2</td><td>39.8</td><td>71.2</td><td>86.8</td><td>71.2</td><td>0.185</td><td>0.0431</td><td>0.2662</td></tr><tr><td rowspan="3">KL + Diversity</td><td>0.5</td><td>64.2</td><td>81.4</td><td>89.2</td><td>77.4</td><td>0.141</td><td>0.0313</td><td>0.0640</td></tr><tr><td>1</td><td>64.4</td><td>81.4</td><td>90.0</td><td>77.2</td><td>0.149</td><td>0.0247</td><td>0.0909</td></tr><tr><td>2</td><td>57.0</td><td>79.4</td><td>89.6</td><td>77.0</td><td>0.166</td><td>0.0171</td><td>0.1445</td></tr><tr><td>JS + Coverage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Method</td><td>λ</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>SC@16↑</td><td>SemDiv ↑</td><td>JS↓</td><td>Coverage ↑</td></tr><tr><td rowspan="3">JS</td><td>0.5</td><td>63.6</td><td>81.2</td><td>88.6</td><td>76.8</td><td>0.134</td><td>0.0113</td><td>0.1572</td></tr><tr><td>1</td><td>63.6</td><td>82.2</td><td>89.0</td><td>76.2</td><td>0.136</td><td>0.0109</td><td>0.1581</td></tr><tr><td>2</td><td>60.6</td><td>82.0</td><td>89.2</td><td>76.6</td><td>0.140</td><td>0.0100</td><td>0.1598</td></tr><tr><td rowspan="4">Coverage</td><td>0.5</td><td>64.4</td><td>82.4</td><td>89.4</td><td>76.8</td><td>0.142</td><td>0.0094</td><td>0.1622</td></tr><tr><td>1</td><td>63.4</td><td>82.8</td><td>89.4</td><td>78.4</td><td>0.157</td><td>0.0065</td><td>0.1729</td></tr><tr><td>2</td><td>41.0</td><td>75.2</td><td>87.8</td><td>74.2</td><td>0.225</td><td>0.0053</td><td>0.2415</td></tr><tr><td>0.5</td><td>63.8</td><td></td><td>90.2</td><td>75.8</td><td></td><td></td><td></td></tr><tr><td rowspan="3">JS + Coverage</td><td>1</td><td></td><td>82.0</td><td></td><td></td><td>0.138</td><td>0.0105</td><td>0.1594 0.1634</td></tr><tr><td>2</td><td>64.8</td><td>82.2</td><td>89.4</td><td>76.2</td><td>0.144</td><td>0.0089</td><td>0.1755</td></tr><tr><td></td><td>62.6</td><td>81.0</td><td>89.8</td><td>75.8</td><td>0.162</td><td>0.0059</td><td></td></tr><tr><td>JS + Diversity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Method JS</td><td>λ</td><td>pass@1↑</td><td>pass@4↑</td><td>pass@16 ↑</td><td>SC@16↑</td><td>SemDiv ↑</td><td>JS↓</td><td>DivGap ↑</td></tr><tr><td>0.5</td><td>63.6</td><td>81.2</td><td>88.6</td><td>76.8</td><td>0.134</td><td>0.0113</td><td>0.0471</td></tr><tr><td>1</td><td>63.6</td><td>82.2</td><td>89.0</td><td>76.2</td><td>0.136</td><td>0.0109</td><td>0.0500</td></tr><tr><td>2</td><td>60.6</td><td>82.0</td><td>89.2</td><td>76.6</td><td>0.140</td><td>0.0100</td><td>0.0555</td></tr><tr><td rowspan="4">Diversity</td><td>0.5</td><td>62.4</td><td>81.4</td><td>89.2</td><td>76.8</td><td>0.144</td><td>0.0085</td><td>0.0808</td></tr><tr><td>1</td><td>56.0</td><td>79.4</td><td>91.0</td><td>76.2</td><td>0.159</td><td>0.0069</td><td>0.1377</td></tr><tr><td>2</td><td>39.8</td><td>71.2</td><td>86.8</td><td>71.2</td><td>0.185</td><td>0.0088</td><td>0.2662</td></tr><tr><td>0.5</td><td>62.8</td><td>82.4</td><td>89.8</td><td>76.4</td><td>0.139</td><td>0.0100</td><td>0.0605</td></tr><tr><td rowspan="3">JS + Diversity</td><td>1</td><td>63.0</td><td>82.4</td><td>89.0</td><td>77.0</td><td>0.145</td><td>0.0081</td><td>0.0838</td></tr><tr><td>2</td><td>57.0</td><td>79.8</td><td>89.4</td><td>76.4</td><td>0.160</td><td>0.0062</td><td>0.1398</td></table>

Table 8: Effect of regularisation strength $\lambda \ \in \ \{ 0 . 5 , 1 , 2 \}$ on KL–Coverage, KL–Diversity, JS– Coverage, and JS–Diversity on MATH500 with Qwen2.5-7B and Top-k support. Each family reports task performance, semantic diversity, the indicated divergence, and its coverage or diversity utility.