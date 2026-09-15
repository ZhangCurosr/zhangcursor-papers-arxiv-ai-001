# CROSS-BLOCK CONDITIONING IN DEEP BOLTZMANN MACHINES FOR STATISTICAL DATA FUSION

A PREPRINT

Junichiro Niimi\* <sup>1</sup> <sup>1</sup>Meijo University

## ABSTRACT

Statistical data fusion combines two panels that share a block of covariates but observe disjoint outcome blocks, and in its traditional form no row observes both outcomes at once. That rules out the discriminative criterion one would rather train a Deep Boltzmann Machine with, since multi prediction training needs ground truth for whatever it holds out. We propose observed-block multiprediction, which restricts the multi-prediction objective to targets drawn from what each row actually observes. It is well defined for any missingness pattern and reduces to the original criterion when rows are complete. Having a discriminative criterion that survives the setting lets us ask whether the joint model is needed at all, by separating what it contributes into a representation part and an inference part. On two consumer panels, on grids over sample size and covariate width spanning 35 cells and 875 runs, the fine-tuned DBM is the best of fifteen methods in every cell; but almost none of that advantage comes from generative pre-training, which is confined to the smallest sample size on one dataset and absent on the other. It comes from conditioning on one outcome block when predicting the other. This term amounts to +0.19 and +0.07 percentage points, is positive in all 35 cells, and, unlike every other contribution we measure, neither decays as the panels grow, nor requires a second hidden layer, nor requires more inference. Permuting one outcome block to destroy its association with the other removes the gain entirely, which is what the account predicts. The margins are small. But a small effect that does not decay is a different object from one that does, because it rests on evidence that no model mapping covariates to outcomes can accept.

## 1 Introduction

## 1.1 Background

The recent resurgence of energy-based models (EBMs) is unmistakable. Hinton’s 2025 Nobel Lecture [1] recentered Boltzmann machines [2] as a foundational paradigm in machine learning. The Energy-Based Transformer [3] recasts next-token prediction as iterative energy descent. LeCun and collaborators continue to advocate the Joint-Embedding Predictive Architecture (JEPA) family [4, 5, 6] as an EBM-flavored alternative to autoregressive generative modeling. Survey and tutorial work [7, 8, 9] together with new theoretical instruments [10] make the revival concrete. Against this backdrop the Deep Boltzmann Machine (DBM) [11], the classical deep EBM with two or more stacked hidden layers, remains underexplored as an applied tool. Its native compatibility with partial observation is a structural advantage that contemporary EBMs lack, and the setting that exercises that advantage most directly is statistical data fusion in marketing [12]: two panels share a common covariate block X but observe disjoint outcome blocks, $Y _ { A }$ in one and Y<sub>B</sub> in the other, and the task is to fill in what each panel did not measure.

What makes data fusion hard is not the amount of data but its shape. The two outcome blocks are never observed together (Figure 1). This is a property of the design, not of the sample size: collecting more rows adds more source-A rows and more source-B rows, and never a single row in which $Y _ { A }$ and $Y _ { B }$ appear side by side. Any model of the joint distribution p(y<sub>A</sub>, y | x) therefore has to obtain the association between the two blocks from something other than paired examples, and any model that simply maps X to outcomes has given up on that association by construction.

<table><tr><td>Groups</td><td>Users</td><td>Source A</td><td>Source B</td><td>Covariates</td></tr><tr><td>Group 1</td><td>user 1 user 2 user 3</td><td>Survey responses</td><td>Did not answer (Unobserved)</td><td rowspan="2">Common variables among two sources (e.g., Demographics)</td></tr><tr><td>Group 2</td><td>… user N</td><td>Did not answer (Unobserved)</td><td>Behavioral logs</td></tr></table>

Figure 1: Typical data structure (missing at random, MAR) for statistical data fusion.

For a DBM the natural response is to marginalize: missing visible dimensions are summed out of the training objective, which stays well-defined under any observation pattern [13]. This generative criterion is available in the fusion regime, but it is not, on its own, a strong predictor. The discriminative alternative — the Multi-Prediction DBM [14], which trains the model to be a good inference machine by holding out a random subset of visible dimensions and predicting it — is the criterion one would rather use, and it is exactly the one that data fusion forbids: its targets must be observed, so when no row observes both blocks, the objective has nothing to score.

This paper closes that gap. We restrict the multi-prediction objective to targets drawn from what each row actually observes. The resulting criterion, which we call observed-block multi-prediction (OBMP), is well-defined for any observation pattern, reduces to MP-DBM when every row happens to be complete, and draws gradient from every row in the fusion regime.

Having a discriminative criterion that survives the fusion setting lets us ask the question the two criteria were obscuring, which is the question this paper is really about: is a joint model necessary here at all? A discriminative model fitted on the common block is simpler, cheaper, and — as we will show — close behind. If what the DBM adds is a better representation, then the difference is one of degree, and a sufficiently flexible discriminative model with enough data should erase it. If instead part of what it adds is unavailable to any model of the form x 7→ y, then the gap has a floor that data cannot lower. Telling those two apart requires separating the contributions rather than reporting their sum, and that is what the experiments below do.

## 1.2 Contributions

A decomposition that separates representation from inference. We separate the advantage of the fine-tuned DBM into two parts that can be measured independently: what generative pre-training contributes over a randomly initialized network of the same capacity trained on the same objective and budget, and what conditioning on one outcome block contributes when predicting the other, measured on a single checkpoint under two conditioning sets. The second term is roughly an order of magnitude larger than the first, and the first is confined to the smallest sample size we study. The case for the joint model rests on its inference structure, not on generative pre-training as an initializer.

An observed-only multi-prediction criterion. We formalize OBMP: a conditioning mask and a target mask constrained to be disjoint and contained in the observed set, scored by a row-normalized cross-entropy on a differentiable mean-field unroll. The criterion is a strict generalization of multi-prediction training to arbitrary missingness, and it is defined in the traditional fusion regime where no training row is complete.

The mechanism does not weaken as data accumulate. On a two-dimensional grid over sample size $n _ { \mathrm { t r a i n } }$ and common-block width k (20 cells over five seeds, 100 runs), the cross-block term is flat: between +0.17 and +0.20 pp at every sample size from 500 to 10,000, with no significant trend on either axis. Every other contribution we measure decays in $n _ { \mathrm { t r a i n } } .$ including the generative pre-training term, which is worth +0.19 pp at $n _ { \mathrm { t r a i n } } { = } 5 0 0$ and is indis tinguishable from zero above $n _ { \mathrm { t r a i n } } { = } 1 0 0 0$ . What the joint model contributes at inference time is therefore not a small-sample effect that more data would erase.

A like-for-like baseline comparison. We compare against ten imputation baselines spanning marginal predictors, supervised regression, nearest neighbors, chained equations, classical data fusion, and deep generative imputation, with every method fitted on the same training rows as the model. OBMP is the best of the fifteen methods in all 20 cells of the grid. We read this as evidence that the effect holds without exception rather than that it is large: the margins are fractions of a percentage point, on a task where the distance between a constant predictor and the strongest baseline is itself only two points (§6.2).

A practical reading. The decomposition says which part of the advantage a practitioner should expect to keep. The part that comes from representation shrinks as panels grow, and is the part a tuned discriminative model will eventually match; the part that comes from conditioning on the other outcome block does not shrink, because it is not a matter of what the model learned.

## 2 Related Study

## 2.1 Energy-based models and their resurgence

An energy-based model defines a probability distribution $p _ { \theta } ( x ) \propto \exp ( - E _ { \theta } ( x ) )$ over a configuration space through a learned scalar energy $E _ { \theta }$ . The framework is by construction more flexible than autoregressive likelihood models — any non-negative function of x is a valid unnormalized density — at the price of an intractable partition function whose handling is the central technical concern of the field [8, 9, 7].

Three concurrent developments have brought EBMs back to the centre of the deep-learning research agenda. First, Hinton’s 2025 Nobel Lecture [1] explicitly recentered the Boltzmann machine [2] as a foundational paradigm, foregrounding the maximum-likelihood gradient with its positive and negative phases as a unifying principle. Second, the Energy-Based Transformer [3] reframes next-token prediction as iterative energy descent over candidate continuations, exchanging the rigid autoregressive factorization for a more flexible test-time inference procedure. Third, LeCun and collaborators have developed the JEPA family [6, 4, 5] as an EBM-flavored alternative to autoregressive generative modeling for image and video. New theoretical instruments [10] continue to fill in the picture.

Against this backdrop the DBM [11] — the classical deep EBM, predating the contemporary revival by more than a decade — has not received commensurate attention as an applied tool. Its compatibility with partial observation is native rather than retrofitted, and that is the property this paper exploits.

## 2.2 Boltzmann machines and the DBM

A Boltzmann machine is a stochastic recurrent network of binary units with symmetric interactions and an energy $E ( s ) = - s ^ { \top } W s / 2 - b ^ { \top } s .$ , where the configuration s comprises both visible and hidden units [2]. The restricted Boltzmann machine (RBM) [15, 16] removes within-layer connections, factoring the energy into bipartite blocks and admitting a closed-form free energy. The DBM [11] stacks two or more RBM-style layers, restoring deep-network expressivity while retaining tractable mean-field inference: a factorized variational posterior over the hidden layers can be optimized by coordinate ascent, each update reducing to a sigmoid of the weighted activations from neighboring layers. Training combines greedy layer-wise RBM pre-training with joint persistent contrastive divergence on the full model, with the partition function estimated by annealed importance sampling [17] when a likelihood bound is needed.

Relation to factor analysis. A useful conceptual anchor is that the DBM can be read as a non-linear, multilayer extension of factor analysis. A single RBM with linear hidden units and Gaussian observations reduces to classical factor analysis with mean-field inference; replacing the linear hidden activation with a sigmoid — and the observations with Bernoulli units, as here — yields a non-linear single-layer factor model, and stacking these layers yields the multilayer non-linear case [12, 11]. The mean-field updates (§3.5) are the direct analogue of expectation maximization in factor analysis. The factor-analytic baseline (§5.2) is therefore a comparison against the DBM’s own linear singlelayer ancestor.

## 2.3 Statistical data fusion

Statistical data fusion has been approached with a variety of methodologies, including propensity-score matching [18], clustering-based methods [19], and factor-analytic models [12]. In the broader setting of missing-value imputation, common choices include multivariate imputation by chained equations (MICE) [20], random-forest imputation [21], and deep generative models [22, 23]. We compare against representatives of all of these families (§5.2).

Despite their natural fit, DBMs have seen little use in data fusion. An earlier preliminary proposal [13] suggested casting the problem as a DBM with per-sample observation masks — missing visible dimensions are marginalized during training, and mean-field inference fills them in at test time — but did not provide a formal derivation or numerical evaluation. We adopt that generative criterion as the first of two training stages (§3.2) and show that on its own it is the weakest arm we test; the contribution of this paper is what is built on top of it.

The closest analogue outside marketing is multimodal learning. The multimodal DBM [24] joins modality-specific pathways, such as images and text tags, through a shared hidden layer, and fills in an absent modality by inference conditioned on the one that is present; this is the same operation as our cross-block conditioning, with modalities in place of outcome blocks. What differs is where the association is learned from. A multimodal model sees its modalities together in training, whereas at $p _ { \mathrm { c o m p l e t e } } = 0$ no row observes $Y _ { A }$ and $Y _ { B }$ together, and their association is identified only through the common block X (§3.6). Data fusion is, in this sense, the multimodal problem with the paired examples removed.

## 2.4 Multi-prediction training and the fusion regime

The Multi-Prediction DBM (MP-DBM) [14] was proposed specifically for missing-value prediction with DBMs. It replaces greedy layer-wise pre-training with a multi-prediction objective: at each minibatch a stochastic mask partitions the visible vector into “observed” and “predicted” subsets, and the model is trained with a cross-entropy loss on the predicted set, with the gradient flowing through the unrolled mean-field inference. The design principle — train the parameters for the inference procedure that will be used at test time — is the one we retain.

The objective requires ground-truth values for the predicted subset, and therefore presupposes fully observed training samples. This is precisely what the fusion setting withholds: source-A rows lack ${ \dot { Y } } _ { B }$ entirely and source-B rows lack $Y _ { A }$ entirely, so a random subset of visible dimensions will in general contain entries no row can score. MP-DBM is thus undefined at $p _ { \mathrm { c o m p l e t e } } = 0$ , the traditional data-fusion regime of Kamakura and Wedel [12]. §3.4 shows that the restriction needed to repair this is mild — draw the targets from the observed set instead of from all visible dimensions — and that the repaired criterion contains MP-DBM as the special case in which every row is complete.

## 2.5 Missing-data taxonomy and our regime

Rubin’s taxonomy [25] distinguishes three regimes by the relationship between the missingness mechanism and the underlying values: missing completely at random (MCAR, independent of all values), missing at random (MAR, independent of the missing values given the observed ones), and missing not at random (MNAR, where missingness can depend on the unobserved values themselves).

The data-fusion setting is MAR conditional on the source indicator $z \in \{ A , B \}$ , which is part of the design rather than of nature: by construction source-A rows never observe $Y _ { B }$ and source-B rows never observe $Y _ { A }$ , irrespective of what those values would have been. The likelihood-based MAR-correct response is to marginalize the missing dimensions out of the training objective, which is what (3) does, and the identifiability argument for OBMP (§3.6) rests on the same conditional independence. MNAR regimes are out of scope here.

## 3 Methodology

We formalize a fine-tuning criterion for the Deep Boltzmann Machine that draws a discriminative training signal from partially observed rows. The criterion, which we call observed-block multi-prediction (OBMP), restricts the multi-prediction objective of the MP-DBM [14] to entries that are genuinely observed under a per-sample mask. It is therefore well-defined in the traditional data-fusion regime in which no training row is fully observed, where the original criterion is not.

## 3.1 DBM with per-sample observation masks

Consider a DBM with one visible layer $v \in \{ 0 , 1 \} ^ { D _ { \tau } }$ and L hidden layers ${ { h } ^ { ( 1 ) } } , \dots , { { h } ^ { ( L ) } }$ , with $h ^ { ( l ) } \in \{ 0 , 1 \} ^ { D _ { h _ { l } } }$ Under the standard Bernoulli–Bernoulli parametrization [11], the joint energy is

$$
E ( \boldsymbol { v } , h ^ { ( 1 : L ) } ; \boldsymbol { \theta } ) = - a ^ { \top } \boldsymbol { v } - \boldsymbol { v } ^ { \top } W ^ { ( 0 ) } h ^ { ( 1 ) } - \sum _ { l = 1 } ^ { L - 1 } h ^ { ( l ) \top } W ^ { ( l ) } h ^ { ( l + 1 ) } - \sum _ { l = 1 } ^ { L } b ^ { ( l ) \top } h ^ { ( l ) } ,\tag{1}
$$

with parameters $\theta = \{ a , b ^ { ( 1 : L ) } , W ^ { ( 0 : L - 1 ) } \} \mathrm { ~ a n d ~ } p _ { \theta } ( v , h ^ { ( 1 : L ) } ) = Z ( \theta ) ^ { - 1 } \exp ( - E ( v , h ^ { ( 1 : L ) } ; \theta ) ) .$

Each training sample i carries an observation mask $m _ { i } \in \{ 0 , 1 \} ^ { D _ { v } }$ , with $m _ { i j } = 1$ iff visible dimension $j$ is observed in sample i. In data fusion the visible dimensions partition into three blocks $\{ \check { X } , Y _ { A } , Y _ { B } \}$ : a common covariate block X observed in every row, and two outcome blocks observed in disjoint panels. The uppercase names denote sets of visible dimensions; we write the matching lowercase symbols for the subvectors of v on those dimensions, $\boldsymbol { v } = ( x , y _ { A } , y _ { B } )$ with $x \in \{ 0 , 1 \} ^ { k } , y _ { A } \in \{ 0 , 1 \} ^ { d _ { a } }$ and $y _ { B } \in \{ 0 , 1 \} ^ { d _ { b } }$ , so that $D _ { v } = k + d _ { a } + d _ { b }$ . Writing $\mathbf { 1 } _ { B }$ for the indicator vector of

a block B, each row falls into one of three groups,

$$
m _ { i } = \left\{ \begin{array} { l l } { { { \bf 1 } } } & { { \mathrm { c o m p l e t e } , } } \\ { { { \bf 1 } _ { X } + { \bf 1 } _ { Y _ { A } } } } & { { \mathrm { s o u r c e } \ A , } } \\ { { { \bf 1 } _ { X } + { \bf 1 } _ { Y _ { B } } } } & { { \mathrm { s o u r c e } \ B , } } \end{array} \right.\tag{2}
$$

and we write $p _ { \mathrm { c o m p l e t e } }$ for the fraction of complete rows and $n _ { \mathrm { c o m p l e t e } }$ for their count. The regime of interest throughout this paper is $p _ { \mathrm { c o m p l e t e } } = 0 :$ the two panels never overlap, so $\bar { Y _ { A } }$ and $Y _ { B }$ are never observed together in training.

## 3.2 Stage one: generative pre-training

The mask-aware marginal log-likelihood

$$
\mathcal { L } ^ { o } ( \theta ) = \sum _ { i } \log p _ { \theta } ( v _ { i } ^ { \mathrm { o b s } } ) = \sum _ { i } \log \sum _ { v _ { i } ^ { \mathrm { m i s s } } , h _ { i } } \exp \bigl ( - E ( v _ { i } , h _ { i } ; \theta ) \bigr ) - n _ { \mathrm { t r a i n } } \log Z ( \theta )\tag{3}
$$

remains well-defined for any observation pattern, since missing visible dimensions are folded into the latent variables and summed out. We approximate the positive phase by a factorized mean-field posterior and the negative phase by persistent contrastive divergence over the full DBM state, giving the gradient

$$
\nabla _ { \theta } \mathcal { L } ^ { o } ( \theta ) \approx - \sum _ { i } \mathbb { E } _ { q _ { i } } \bigl [ \nabla _ { \theta } E ( v _ { i } , h _ { i } ; \theta ) \bigr ] + n _ { \mathrm { t r a i n } } \mathbb { E } _ { \tilde { p } _ { \theta } } \bigl [ \nabla _ { \theta } E ( v , h ; \theta ) \bigr ] .\tag{4}
$$

We refer to a DBM trained by (3) as an ML-DBM and use it as the initialization for stage two. It is a purely generative criterion: no term in (3) distinguishes covariates from outcomes.

## 3.3 Why multi-prediction training is undefined here

The MP-DBM [14] replaces (3) with a discriminative criterion. At each minibatch it draws a random subset $S$ of visible dimensions, runs mean-field inference conditioned on $v _ { \bar { S } }$ , and penalises the cross-entropy between the inferred marginals on S and the true values v . The construction presupposes that $v _ { S }$ is available, i.e. that the row is fully observed; on a row from source A, any S intersecting $Y _ { B }$ has no ground truth to compare against. Consequently the criterion draws gradient signal only from the $n _ { \mathrm { c o m p l e t e } }$ complete rows, and is undefined outright when $n _ { \mathrm { c o m p l e t e } } = 0$

## 3.4 Observed-block multi-prediction

OBMP removes this restriction by choosing the prediction targets inside the observed set. For each row we introduce two binary masks, a conditioning mask $c _ { i } \in \{ \stackrel { \cdot } { 0 } , 1 \} ^ { D _ { \iota } }$ <sup>v</sup> marking the entries supplied to inference, and a target mask $t _ { i } \in \{ 0 , 1 \} ^ { D _ { \iota } }$ marking the entries whose known values define the loss, subject to

$$
c _ { i } \odot t _ { i } = { \bf 0 } , \qquad t _ { i } \preceq m _ { i } ,\tag{5}
$$

where ⊙ is the elementwise product and $\preceq$ holds elementwise. The first constraint keeps a target from being fed to the model as evidence; the second is the substantive one, and states that every target must be a genuinely observed value. Entries that are neither conditioned on nor targeted are treated as missing and marginalized, exactly as in (3).

Let $\mu ^ { ( v ) } ( c _ { i } ) \in [ 0 , 1 ] ^ { D _ { v } }$ denote the visible marginals returned by the mean-field procedure (§3.5) when the entries selected by $c _ { i }$ are clamped to their observed values, suppressing its dependence on $v _ { i }$ and θ in the notation. With $\ell ( p , y ) = { \dot { - y } } \log p - ( { \bar { 1 } } - y ) \log ( 1 - p )$ the elementwise binary cross-entropy, the OBMP criterion is

$$
\mathcal { L } ^ { \mathrm { O B M P } } ( \theta ) = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { I } } \frac { \sum _ { j } t _ { i j } \ell \big ( \mu _ { i j } ^ { ( v ) } ( c _ { i } ) , v _ { i j } \big ) } { \sum _ { j } t _ { i j } } ,\tag{6}
$$

where $\begin{array} { r } { \mathcal { T } = \{ i : \sum _ { i } t _ { i j } > 0 \} } \end{array}$ . The inner normalization by $\textstyle \sum _ { j } t _ { i j }$ weights every row equally regardless of how many targets it contributes, so that rows from a panel with a wider outcome block do not dominate the gradient; I excludes rows with no target at all. In practice $\mu ^ { ( v ) }$ is clamped to $[ \varepsilon , 1 - \varepsilon ]$ before the logarithm.

Two properties follow directly from (5). First, OBMP is defined for any mask pattern, including $m _ { i } \neq { \bf 1 }$ for every i: the targets are drawn from what each row actually observes, so every row contributes gradient. Second, when every row is complete $( m _ { i } \equiv \mathbf { 1 } )$ and $c _ { i } = \mathbf { 1 } - t _ { i }$ with $t _ { i }$ drawn at random, (6) reduces to the MP-DBM criterion. OBMP is thus a strict generalization of multi-prediction training to arbitrary observation patterns, and inherits its interpretation as training the model to be a good inference machine rather than a good density model.

In the general case the two masks are obtained by splitting the observed set at random: draw a per-row keep probability $\pi _ { i } \sim \mathcal { U } ( \pi _ { \operatorname* { m i n } } , \pi _ { \operatorname* { m a x } } )$ , keep each observed entry independently with probability $\pi _ { i }$ to form $c _ { i }$ , and target the rest,

$$
c _ { i } = m _ { i } \odot \rho _ { i } , \qquad t _ { i } = m _ { i } - c _ { i } , \qquad \rho _ { i j } \sim \mathrm { B e r n } ( \pi _ { i } ) .\tag{7}
$$

This recovers the stochastic masking of MP-DBM restricted to the observed support. The data-fusion instantiation below replaces it with a deterministic split that matches the deployment pattern.

## 3.5 Differentiable mean-field unroll

Evaluating (6) requires $\mu ^ { ( v ) }$ to be a differentiable function of θ. We therefore run mean-field inference as a fixed-length unrolled computation and backpropagate through all of it, rather than treating the fixed point as a constant. Write $\mu ^ { ( v ) , \ast }$ τ and $\mu ^ { ( l ) , \tau }$ for the visible and hidden marginals after τ passes, and let σ denote the elementwise logistic function. The initialization clamps the conditioned entries and sets the remainder from the visible bias, then propagates upward:

$$
\begin{array} { r } { \mu ^ { ( v ) , 0 } = c \odot v + ( 1 - c ) \odot \sigma ( a ) , } \end{array}\tag{8}
$$

$$
\mu ^ { ( l ) , 0 } = \sigma \big ( \mu ^ { ( l - 1 ) , 0 } W ^ { ( l - 1 ) } + b ^ { ( l ) } \big ) , \quad l = 1 , \ldots , L ,\tag{9}
$$

with the convention $\mu ^ { ( 0 ) , \tau } : = \mu ^ { ( v ) , \tau }$ . Each subsequent pass $\tau = 1 , \dots , T$ updates the visible marginals from the first hidden layer, re-clamping the conditioned entries, and then sweeps the hidden layers bottom-up:

$$
\begin{array} { r } { \mu ^ { ( v ) , \tau } = c \odot v + ( \mathbf { 1 } - c ) \odot \sigma \big ( \mu ^ { ( 1 ) , \tau - 1 } W ^ { ( 0 ) \top } + a \big ) , } \end{array}\tag{10}
$$

$$
\begin{array} { r } { \boldsymbol { \mu } ^ { ( l ) , \tau } = \sigma \big ( \boldsymbol { \mu } ^ { ( l - 1 ) , \tau } \boldsymbol { W } ^ { ( l - 1 ) } + \boldsymbol { b } ^ { ( l ) } + \boldsymbol { \mu } ^ { ( l + 1 ) , \tau - 1 } \boldsymbol { W } ^ { ( l ) \top } \big ) , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \boldsymbol { \mu } ^ { ( L ) , \tau } = \sigma \big ( \boldsymbol { \mu } ^ { ( L - 1 ) , \tau } \boldsymbol { W } ^ { ( L - 1 ) } + \boldsymbol { b } ^ { ( L ) } \big ) , } \end{array}\tag{12}
$$

where $l = 1 , \ldots , L - 1$ . Equation (11) carries the top-down term at pass $\tau - 1$ because the sweep proceeds bottom-up within a pass, so the layer above has not yet been refreshed. Conditioned entries enter (11) through $\mu ^ { ( v ) , \tau }$ but are never themselves updated, so the clamping in (10) is exact at every pass. Optionally each update may be damped, $\mu  ( 1 - \lambda ) \mu ^ { \mathrm { n e w } } \overset { * } { + } \lambda \mu ^ { \mathrm { o l d } }$ ; we use $\lambda = 0$ throughout.

We take $\mu ^ { ( v ) } : = \mu ^ { ( v ) , T }$ in (6). Because (8)–(12) are compositions of affine maps and logistic nonlinearities, the whole unroll is differentiable, and the gradient of (6) flows through all $T$ passes. The criterion therefore trains the parameters for the inference procedure that will be used at test time, which is the same design principle that motivates MP-DBM; the difference is only in which entries are allowed to serve as targets.

## 3.6 Instantiation for data fusion

In the fusion setting the split is deterministic. We condition on the common block and target every observed outcome:

$$
\begin{array} { r } { c _ { i } = \mathbf { 1 } _ { X } , \qquad t _ { i } = m _ { i } - \mathbf { 1 } _ { X } = m _ { i } \odot \big ( \mathbf { 1 } _ { Y _ { A } } + \mathbf { 1 } _ { Y _ { B } } \big ) , } \end{array}\tag{13}
$$

which by (2) gives $t _ { i } = { \bf 1 } _ { Y _ { A } }$ for a source-A row, $t _ { i } = \mathbf { 1 } _ { Y _ { B } }$ for a source-B row, and $t _ { i } = \mathbf { 1 } _ { Y _ { A } } + \mathbf { 1 } _ { Y _ { B } }$ for a complete row. The constraints (5) hold by construction, and I contains every row, so all $n _ { \mathrm { t r a i n } }$ rows contribute gradient at $p _ { \mathrm { c o m p l e t e } } = 0$

This choice makes the criterion identifiable under the missingness regime of data fusion. Missingness is determined by the source indicator, which is independent of the outcome values given $X ;$ ; conditioning on X alone and predicting the observed outcome block therefore targets $p ( \boldsymbol { y } _ { A } \mid x )$ on source-A rows and $p ( \boldsymbol { y } _ { B } \mid \boldsymbol { x } )$ on source-B rows, both of which are identified from the observed data. No term in (6) requires the joint $p ( y _ { A } , y _ { B } \mid x )$ , which is not identified without complete rows.

The inference (§3.7) is a different matter, and we state the asymmetry plainly because it bounds what the experiments can show. Predicting $y _ { A }$ from x and $y _ { B }$ requires $p ( \boldsymbol { y } _ { A } \mid \boldsymbol { x } , \boldsymbol { y } _ { B } )$ , a functional of exactly the joint the objective avoided, and nothing in fusion data identifies it. What the model supplies in its place is an inductive bias: the hidden layers carry all dependence between the outcome blocks, so conditioning on $y _ { B }$ acts through them. This is the same structure as the conditional independence assumption underlying factor-analytic and matching approaches to statistical matching [12, 26], where $Y _ { A }$ and $Y _ { B }$ are taken to be independent given the common variables; here they are taken to be independent given the common variables and the hidden state. That is a weaker assumption, since the hidden state is learned rather than fixed to $X ,$ , but it is an assumption and it is not testable on fusion data. Our test rows happen to observe both blocks, so we can check after the fact whether it paid off; a practitioner with genuine fusion data cannot. §6.3 reports a dataset where it does not.

## 3.7 Inference-time conditioning

The conditioning mask is a property of the inference call, not of the trained parameters, so a single OBMP checkpoint can be evaluated under different conditioning sets. We distinguish two.

Cross-block conditioning. To predict outcome block $Y _ { A }$ we clamp everything else that the row observes, including the other outcome block:

$$
c = { \bf 1 } - { \bf 1 } _ { Y _ { A } } = { \bf 1 } _ { X } + { \bf 1 } _ { Y _ { B } } ,\tag{14}
$$

and symmetrically $c = \mathbf { 1 } _ { X } + \mathbf { 1 } _ { Y _ { A } }$ to predict $Y _ { B }$ . This matches the operational task in panel fusion: a source-A row is one whose $Y _ { A }$ is recorded and whose $Y _ { B }$ must be filled in, so $Y _ { A }$ is available as evidence precisely when $Y _ { B }$ is the quantity of interest.

X-only conditioning. Alternatively both outcome blocks are hidden and predicted in a single pass,

$$
c = \mathbf { 1 } _ { X } ,\tag{15}
$$

which is the conditioning set used during training (13), and the only one available to a discriminative model fitted on X alone.

The gap between (14) and (15), measured on a single checkpoint, isolates the value of conditioning on one outcome block when predicting the other. It is a quantity that no model trained to map X to outcomes can produce, and, because $Y _ { A }$ and $Y _ { B }$ are never observed together in training, it is supplied entirely by the generative structure of the joint model rather than by a fitted association between the two blocks.

## 3.8 Two-stage procedure

Training proceeds in two stages on the same architecture and the same rows. Stage one fits the ML-DBM by (4), initialising the visible bias from the observed marginals $a _ { j } = \mathrm { l o g i t } ( \bar { v } _ { j } )$ computed over the entries with $m _ { i j } = 1$ since with $p _ { \mathrm { c o m p l e t e } } = 0$ there is no complete subset on which to pre-train layerwise. Stage two initialises from the stage-one parameters and minimises (6) with the fusion split (13), selecting the checkpoint by validation performance. Stage two changes the criterion but not the model: the parameters θ remain those of the DBM in (1), and the same mean-field procedure is used for training and for both inference modes of §3.7.

## 4 Experimental design

## 4.1 Datasets and fusion protocol

We report two datasets, chosen so that the block geometry differs as much as the fusion setting allows. Blocks are assigned by subject matter, not at random, so that each panel is one a real study might have collected.

Instacart is an online-grocery panel binarized to purchase indicators at the aisle level, with aisles assigned to blocks by department. X holds fresh-food departments (produce, dairy and eggs, beverages, bakery, meat and seafood), $Y _ { A }$ the ambient-food departments (pantry, frozen, snacks, breakfast, canned goods, dry goods, deli, $d _ { a } = 5 3 )$ and $Y _ { B }$ the nonfood departments (personal care, household, babies, pets, alcohol, international, bulk, other, $d _ { b } = 4 6 )$ , so $D _ { v } = k + 9 9$ Restricting X to width k keeps its k most-purchased aisles and drops the rest from the visible layer entirely; they are not moved into an outcome block. We draw from a pool of 80,000 users and hold out $n _ { \mathrm { v a l } } = n _ { \mathrm { t e s t } } = 5 { , } 0 0 0$

Bank Marketing is a telemarketing record, not a behavioral panel, and we use it because its block geometry is close to the reverse of Instacart’s. X holds demographics (age quartiles, occupation, marital status, education), Y the customer’s financial standing (default flag, balance quartiles, housing and personal loans, $d _ { a } = 7 )$ and $Y _ { B }$ the campaign contact history (contact channel, day and season, call-duration quartiles, contact and previous-contact counts, previous outcome, and whether the customer subscribed, $d _ { b } = 2 9 )$ , so $D _ { v } = k + 3 6$ . All variables are one-hot or quartile encoded. We draw from 35,000 rows with $n _ { \mathrm { v a l } } = n _ { \mathrm { t e s t } } = 5 { , } 0 0 0$ . The fusion task here is the realistic one of joining a demographic-plus-financial panel to a demographic-plus-campaign panel.

The two datasets differ in which outcome block is the wide one, and by a factor of nearly three in how much of the outcome space the first block occupies — 53 of 99 dimensions against 7 of 36 — which turns out to matter for reading the headline numbers (§5.5).

Every training row is assigned to source A or source B with probability $p _ { a } = 0 . 5$ and masked according to (2). We set $p _ { \mathrm { c o m p l e t e } } = 0$ throughout: no training row observes both outcome blocks. This is the regime in which the MP-DBM criterion is undefined, and it is the regime that motivates OBMP. Validation and test rows retain both blocks, since scoring requires ground truth for the block being predicted; the conditioning sets (§3.7) control what the model is allowed to see.

## 4.2 The $( n _ { \mathrm { t r a i n } } , k )$ grids

We vary two quantities jointly: the number of training rows $n _ { \mathrm { t r a i n } } \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 5 0 0 0 , 1 0 , 0 0 0 \}$ , drawn from the pool described above and disjoint from the $n _ { \mathrm { v a l } }$ validation and $n _ { \mathrm { t e s t } }$ test rows, and the width of the common block, $k \in \{ 5 , 1 0 , 2 0 , 3 5 \}$ on Instacart and $k \in \{ 5 , 1 0 , 2 0 \}$ on Bank Marketing, whose X block has 20 dimensions in total. Each cell is repeated over five seeds, for 100 and 75 runs respectively. The two axes are the natural stress directions for a data-fusion method: $n _ { \mathrm { t r a i n } }$ controls how much evidence is available, and k controls how much of the outcome variation the shared covariates can explain. Training subsets are nested within a seed, drawn by a fixed permutation rule so that a larger $n _ { \mathrm { t r a i n } }$ contains the rows of every smaller $n _ { \mathrm { t r a i n } }$

## 4.3 Methods compared

Five arms share the DBM architecture — L = 2 hidden layers with $D _ { h _ { 1 } } = 6 4$ and $D _ { h _ { 2 } } = 3 2$ — and are run on every cell of both grids:

• ML-DBM: generative training by (4) alone, no discriminative fine-tuning.

• OBMP (cross-block): the ML-DBM fine-tuned by (6), evaluated with the conditioning of (14).

• OBMP (X-only): the same checkpoint evaluated with (15). The difference between these two arms is measured on identical parameters and so is free of training confounds.

• MLP: a randomly initialized sigmoid network $k  6 4  3 2  ( d _ { a } + d _ { b } )$ trained on the same observed-block cross-entropy, with weight decay chosen on validation. It matches the DBM’s capacity and objective but has no generative pre-training and no cross-block inference path.

• X-logistic: per-dimension logistic regression on $X .$ , with the regularization strength chosen on validation.

The MLP and X-logistic arms are chosen to bracket the two mechanisms that could explain a DBM advantage without invoking the joint model: a nonlinear shared representation, and a well-tuned conditional mean. We additionally compare against ten imputation baselines spanning marginal predictors, nearest neighbours, chained equations, classical data fusion, and deep generative imputation (Table 2).

Sample-size parity. Every method, including all ten baselines, is fitted on the same $n _ { \mathrm { t r a i n } }$ training rows, verified per cell by recomputing the tuned X-logistic arm inside the baseline runner and matching it against the value recorded by the main run (deviation exactly zero on all 100 Instacart runs and all 75 Bank Marketing runs).

## 4.4 Training budget and checkpoint selection

OBMP fine-tuning and the MLP arm are trained for up to 200 epochs on every cell, evaluated on validation every five epochs with checkpoint selection and a patience of 60. Holding the two arms to one budget is what makes the decomposition (§5.3) interpretable: a difference between them is a difference of method, not of how long each was allowed to train. The budget is generous enough that OBMP selects its final epoch strictly below the cap in 98 of the 100 runs. The mean-field unroll (§3.5) runs a fixed $T = 1 0$ passes during training, since the loss has to differentiate through all of them. At evaluation, where no gradient is needed, T is a cap: the sweep stops early once no coordinate moves by more than $1 0 ^ { - 4 }$ , and we set the cap to 15. §5.7 reports what happens when it is varied.

Both arms select their checkpoint on X-only validation accuracy, the conditioning that the training objective itself uses (13). This matters for the contrast (§3.7). Selecting the OBMP checkpoint on cross-block validation instead — the conditioning under which it will be reported — and then subtracting its X-only accuracy would let the selection contribute to the gap: with roughly forty candidate epochs per cell, choosing the one that scores best under cross-block evaluation picks up the upper tail of the validation noise in exactly the direction the contrast is measured. We quantify that effect (§5.3) by tracking both checkpoints from the same run.

## 4.5 Metrics

All metrics are computed over the positions being predicted, never over observed ones. We report a dimensionweighted combination of the two outcome blocks,

$$
{ \mathrm { c o m b i n e d } } = { \frac { d _ { a } \cdot s _ { Y _ { A } } + d _ { b } \cdot s _ { Y _ { B } } } { d _ { a } + d _ { b } } } ,\tag{16}
$$

where $s _ { B }$ is the metric on block B: either accuracy at a threshold of 0.5 or binary cross-entropy. Accuracy is the headline number throughout. We quote cross-entropy only where calibration is the point (§6.2); the three baselines that emit hard $0 / 1$ values rather than probabilities are not comparable on it at all.

Every test in this paper is paired and clustered on seed. Runs sharing a seed share a test split and nested training subsets, so they are not independent replications; averaging within seed and testing the five means gives four degrees of freedom rather than the spuriously large ones a run-level test would assume. The seed-to-seed standard deviation of the contrasts we report is 0.011–0.044 pp, so the conclusions survive the stricter accounting, but the p-values are correspondingly less extreme.

## 5 Results

## 5.1 OBMP is best in every cell

Table 1 reports both grids. OBMP with cross-block conditioning is the best of the five arms in every cell of both $( 2 0 / 2 0$ on Instacart and $1 5 / 1 5$ on Bank Marketing) and in 98 of 100 and 70 of 75 runs. Against the ten imputation baselines as well (§5.2) it is best in every cell of both grids. The margin is widest at the low-resource corner and narrows as $n _ { \mathrm { t r a i n } }$ grows, but it does not change sign anywhere.

Two features are worth noting before any decomposition. First, ML-DBM alone is the weakest arm on both datasets, and on Bank Marketing it is degenerate: its accuracy is identical across all 15 cells within a seed and equals the column-mean baseline exactly. On Instacart it moves a little with $n _ { \mathrm { t r a i n } }$ and k but still trails tuned X-logistic by 1.4 to 2.5 pp, and its $Y _ { B }$ accuracy never leaves the base rate. Whatever OBMP contributes on either dataset, it is not inherited from the generative stage. Second, the ranking of the two OBMP arms is stable: cross-block conditioning beats X-only conditioning in every cell of both grids, on parameters that are identical.

## 5.2 Against imputation baselines

Table 2 compares OBMP against ten imputation methods fitted on the same rows. It wins every pairwise comparison on both datasets, all with $p < 0 . 0 0 1$ clustered on seed, and loses at most four of 100 Instacart runs and two of 75 Bank Marketing runs to any single baseline. The strongest baseline is dataset- and regime-dependent: MIWAE at small $n _ { \mathrm { t r a i n } }$ on Instacart with tuned X-logistic taking over from $n _ { \mathrm { t r a i n } } { = } 5 0 0 0$ , MICE at narrow k on Bank Marketing with X-logistic taking over as k widens. That a deep generative imputer is the toughest competitor in one corner and a regularized linear model in another is itself a useful observation for practitioners choosing an imputation method.

## 5.3 Decomposing the advantage

Table 3 splits the total into two additive parts. Generative pre-training (OBMP X-only − MLP) isolates what the generative stage adds over a randomly initialized network of the same capacity, trained on the same objective, for the same number of epochs, and selected on the same criterion. Cross-block conditioning (OBMP cross-block − OBMP X-only) isolates what the joint model adds at inference time.

The two parts behave differently, and they behave differently in the same way on both datasets. Generative pre-training contributes $+ 0 . 0 4 6 \mathrm { p p }$ on Instacart, resting almost entirely on the smallest sample size (+0.193 pp at $n _ { \mathrm { t r a i n } } { = } 5 0 0$ +0.042 pp at $n _ { \mathrm { t r a i n } } { = } 1 0 0 0$ , and within 0.004 pp of zero at every larger $n _ { \mathrm { t r a i n } } )$ and on Bank Marketing it is absent altogether (−0.005 pp over 75 runs). As a way of initializing parameters, the generative stage buys nothing that enough epochs of discriminative training on a plain network would not also buy.

Cross-block conditioning is present in every cell of both grids: +0.188 pp on Instacart, +0.065 pp on Bank Marketing, and every one of the 35 cells across the two grids is positive. It is what the total advantage over the discriminative arms is made of, and §5.5 shows that the apparent gap between the two datasets is mostly an artifact of how the two outcome blocks are weighted rather than a weaker mechanism.

The cross-block term deserves emphasis because of how it is measured. It compares one checkpoint against itself under two conditioning sets, so no training difference can contribute to it. And because $Y _ { A }$ and $Y _ { B }$ are never observed together in training, the association it exploits was never fitted from paired data; it is propagated through the shared hidden layers of the joint model. No model that maps X to outcomes can produce this term, whatever its capacity.

The contrast is sensitive to how the checkpoint is chosen. Tracking both checkpoints from each run (§4.4) makes the size of that sensitivity explicit. On Instacart, under cross-block evaluation the cross-block-selected checkpoint scores +0.026 pp higher than the X-only-selected one $( p = 0 . 0 0 2 )$ , and reporting the contrast from it would inflate

Table 1: Test combined accuracy [%] on the $( n _ { \mathrm { t r a i n } } , k )$ grids, mean over five seeds (standard deviation in parentheses). Best per cell in bold. best BL is the strongest of the ten imputation baselines on that cell, all fitted on the same training rows: MIWAE at small $n _ { \mathrm { t r a i n } }$ and X-logistic elsewhere on Instacart, MICE and X-logistic on Bank Marketing. The three ablations each remove one component of OBMP and share its training budget (§4.4).
<table><tr><td rowspan="2"> $n _ { \mathrm { t r a i n } }$ </td><td rowspan="2">k</td><td rowspan="2">proposed</td><td colspan="2">baselines</td><td colspan="3">ablations of OBMP</td></tr><tr><td>OBMP X-log.</td><td>best BL</td><td>ML-DBM</td><td>X-only</td><td>MLP</td></tr><tr><td>Instacart</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>500</td><td>5</td><td>83.24 (0.08)</td><td>82.93 (0.05)</td><td>82.94</td><td>81.50 (0.07)</td><td>83.06 (0.11)</td><td>82.68 (0.56)</td></tr><tr><td rowspan="5"></td><td>10</td><td>83.50 (0.12)</td><td>82.94 (0.07)</td><td>83.25</td><td>81.50 (0.03)</td><td>83.30 (0.08)</td><td>83.17 (0.10)</td></tr><tr><td>20</td><td>83.80 (0.07)</td><td>83.35 (0.06)</td><td>83.76</td><td>81.52 (0.14)</td><td>83.66 (0.08)</td><td>83.50 (0.09)</td></tr><tr><td>35</td><td>83.86 (0.12)</td><td>83.38 (0.09)</td><td>83.85</td><td>81.51 (0.08)</td><td>83.70 (0.08)</td><td>83.60 (0.07)</td></tr><tr><td>5</td><td>83.32 (0.08)</td><td>83.05 (0.07)</td><td>83.03</td><td>81.54 (0.04)</td><td>83.11 (0.09)</td><td>83.06 (0.07)</td></tr><tr><td>10</td><td>83.64 (0.08)</td><td>83.21 (0.09)</td><td>83.21</td><td>81.56 (0.06)</td><td>83.37 (0.07)</td><td>83.33 (0.07)</td></tr><tr><td rowspan="4">2000</td><td>20</td><td>84.03 (0.04)</td><td>83.63 (0.04)</td><td>83.69</td><td>81.55 (0.09)</td><td>83.86 (0.04)</td><td>83.80 (0.05)</td></tr><tr><td>35</td><td>84.07 (0.06)</td><td>83.71 (0.07)</td><td>83.88</td><td>81.59 (0.06)</td><td>83.94 (0.06)</td><td>83.91 (0.07)</td></tr><tr><td>5</td><td>83.27 (0.10)</td><td>83.11 (0.09)</td><td>83.10</td><td>81.57 (0.07)</td><td>83.14 (0.08)</td><td>83.13 (0.07)</td></tr><tr><td>10</td><td>83.71 (0.06)</td><td>83.39 (0.07)</td><td>83.39</td><td>81.57 (0.09)</td><td>83.42 (0.07)</td><td>83.44 (0.08)</td></tr><tr><td rowspan="5">5000</td><td>20</td><td>84.14 (0.08)</td><td>83.90 (0.07)</td><td>83.89</td><td>81.58 (0.04)</td><td>83.97 (0.07)</td><td>83.95 (0.06)</td></tr><tr><td>35</td><td>84.20 (0.08)</td><td>83.95 (0.03)</td><td>83.90</td><td>81.61 (0.08)</td><td>84.07 (0.05)</td><td>84.08 (0.02)</td></tr><tr><td>5</td><td>83.29 (0.12)</td><td>83.15 (0.07)</td><td>83.15</td><td>81.62 (0.10)</td><td>83.17 (0.08)</td><td>83.17 (0.08)</td></tr><tr><td>10</td><td>83.74 (0.04)</td><td>83.48 (0.07)</td><td>83.48</td><td>81.59 (0.05)</td><td>83.47 (0.06)</td><td>83.48 (0.06)</td></tr><tr><td>20</td><td>84.26 (0.04)</td><td>84.07 (0.05)</td><td>84.07</td><td>81.64 (0.07)</td><td>84.06 (0.05)</td><td>84.05 (0.05)</td></tr><tr><td rowspan="5">10000</td><td>35</td><td>84.40 (0.04)</td><td>84.16 (0.03)</td><td>84.14</td><td>81.70 (0.05)</td><td>84.22 (0.02)</td><td>84.23 (0.04)</td></tr><tr><td>5</td><td>83.27 (0.10)</td><td>83.16 (0.07)</td><td>83.16</td><td>81.68 (0.09)</td><td>83.18 (0.08)</td><td>83.17 (0.07)</td></tr><tr><td>10</td><td>83.79 (0.04)</td><td>83.50 (0.07)</td><td>83.50</td><td>81.76 (0.09)</td><td>83.49 (0.06)</td><td>83.51 (0.05)</td></tr><tr><td>20</td><td>84.32 (0.07)</td><td>84.12 (0.07)</td><td>84.12</td><td>82.41 (0.12)</td><td>84.12 (0.08)</td><td>84.11 (0.07)</td></tr><tr><td>35</td><td>84.47 (0.03)</td><td>84.26 (0.05)</td><td>84.25</td><td>82.82 (0.14)</td><td>84.26 (0.04)</td><td>84.27 (0.04)</td></tr><tr><td colspan="8">Bank Marketing</td></tr><tr><td rowspan="4">500</td><td>5</td><td>77.20 (0.05)</td><td>77.18 (0.03)</td><td>77.17</td><td>77.17 (0.03)</td><td></td><td></td></tr><tr><td>10</td><td>77.36 (0.05)</td><td>77.29 (0.05)</td><td>77.27</td><td>77.17 (0.03)</td><td>77.18 (0.05) 77.32 (0.03)</td><td>77.19 (0.05) 77.36 (0.04)</td></tr><tr><td>20</td><td>77.37 (0.05)</td><td>77.28 (0.02)</td><td>77.25</td><td>77.17 (0.03)</td><td>77.31 (0.04)</td><td>77.31 (0.04)</td></tr><tr><td>5</td><td>77.23 (0.03)</td><td>77.18 (0.03)</td><td>77.18</td><td>77.17 (0.03)</td><td>77.20 (0.04)</td><td>77.20 (0.05)</td></tr><tr><td rowspan="4">2000</td><td>10</td><td>77.43 (0.07)</td><td>77.35 (0.06)</td><td>77.30</td><td>77.17 (0.03)</td><td>77.35 (0.05)</td><td>77.36 (0.06)</td></tr><tr><td>20</td><td>77.46 (0.03)</td><td>77.33 (0.02)</td><td>77.31</td><td>77.17 (0.03)</td><td>77.36 (0.04)</td><td>77.35 (0.05)</td></tr><tr><td>5</td><td>77.24 (0.03)</td><td>77.20 (0.04)</td><td>77.20</td><td>77.17 (0.03)</td><td>77.21 (0.04)</td><td>77.21 (0.03)</td></tr><tr><td>10</td><td>77.45 (0.04)</td><td>77.37 (0.05)</td><td>77.36</td><td>77.17 (0.03)</td><td>77.39 (0.04)</td><td>77.39 (0.04)</td></tr><tr><td rowspan="4">5000</td><td>20</td><td>77.50 (0.03)</td><td>77.38 (0.05)</td><td>77.36</td><td>77.17 (0.03)</td><td>77.41 (0.05)</td><td>77.41 (0.04)</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>77.26 (0.02) 77.45 (0.06)</td><td>77.22 (0.03) 77.38 (0.05)</td><td>77.22 77.39</td><td>77.17 (0.03) 77.17 (0.03)</td><td>77.22 (0.03) 77.40 (0.03)</td><td>77.22 (0.03) 77.40 (0.03)</td></tr><tr><td>20</td><td>77.55 (0.03)</td><td>77.43 (0.04)</td><td>77.43</td><td>77.17 (0.03)</td><td>77.44 (0.04)</td><td>77.44 (0.04)</td></tr><tr><td rowspan="4">10000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>77.27 (0.04)</td><td>77.22 (0.03)</td><td>77.22</td><td>77.17 (0.03)</td><td>77.22 (0.03)</td><td>77.22 (0.03)</td></tr><tr><td>10</td><td>77.50 (0.03)</td><td>77.39 (0.04)</td><td>77.38</td><td>77.17 (0.03)</td><td>77.39 (0.04)</td><td>77.39 (0.04)</td></tr><tr><td>20</td><td>77.56 (0.03)</td><td>77.45 (0.04)</td><td>77.45</td><td>77.17 (0.03)</td><td>77.46 (0.04)</td><td>77.45 (0.02)</td></tr></table>

the cross-block term from +0.188 to +0.230 pp. The inflation is not uniform: it reaches +0.166 pp at $n _ { \mathrm { t r a i n } } { = } 1 0 , 0 0 0 ,$ k=5 and is near zero at k=35, because a narrow common block leaves X-only validation least able to discriminate between candidate epochs. A decomposition that selects on the criterion it then reports would therefore attribute part of the selection procedure’s behavior to the model. All numbers we report use the X-only-selected checkpoint.

Table 2: OBMP against ten imputation baselines, all fitted on the same training rows. $\Delta$ is the paired difference from OBMP [pp] and wins counts runs where OBMP is ahead. <sup>∗</sup> marks $p < 0 . 0 5$ against zero, clustered on seed. <sup>†</sup> returns hard 0/1 values, so its cross-entropy is not comparable (§6.3).
<table><tr><td rowspan="2">method</td><td colspan="2">Instacart (100)</td><td colspan="2">Bank Marketing (75)</td></tr><tr><td>∆</td><td>wins</td><td> $\Delta$ </td><td>wins</td></tr><tr><td>X-logistic (default C)</td><td>+0.387*</td><td>99/100</td><td>+0.169*</td><td>75/75</td></tr><tr><td>factor analysis</td><td>+0.538*</td><td>100/100</td><td>+0.178*</td><td>74/75</td></tr><tr><td>MIWAE</td><td>+0.811*</td><td>96/100</td><td>+1.349*</td><td>75/75</td></tr><tr><td>MICE (ridge solver)</td><td>+0.922*</td><td>100/100</td><td>+0.108*</td><td>73/75</td></tr><tr><td>k-NN</td><td>+1.524*</td><td>100/100</td><td>+2.255*</td><td>75/75</td></tr><tr><td>column mean</td><td>+2.255*</td><td>100/100</td><td>+0.219*</td><td>74/75</td></tr><tr><td>mode zero</td><td>+2.493*</td><td>100/100</td><td>+2.320*</td><td>75/75</td></tr><tr><td>hot deck†</td><td>+6.799*</td><td>100/100</td><td>+10.556*</td><td>75/75</td></tr><tr><td>propensity matching†</td><td>+8.281*</td><td>100/100</td><td>+10.537*</td><td>75/75</td></tr><tr><td>MissForest†</td><td>+19.229*</td><td>100/100</td><td>+8.579*</td><td>75/75</td></tr></table>

Table 3: The two contributions, as paired differences in test combined accuracy [pp], mean over five seeds. Positive favors OBMP. Cross-block conditioning is positive in every cell of both grids.
<table><tr><td> $n _ { \mathrm { t r a i n } }$ </td><td>k</td><td>pre- training</td><td>cross- block</td><td>VS. MLP</td><td>VS. X-log.</td><td>pre- training</td><td>cross- block</td><td>VS. MLP</td><td>VS. X-log.</td></tr><tr><td>Instacart</td><td></td><td></td><td></td><td></td><td></td><td>Bank Marketing</td><td></td><td></td><td></td></tr><tr><td>500</td><td>5</td><td>+0.380</td><td>+0.186</td><td>+0.566</td><td>+0.311</td><td>-0.011</td><td>+0.015</td><td>+0.005</td><td>+0.022</td></tr><tr><td></td><td>10</td><td>+0.133</td><td>+0.203</td><td>+0.336</td><td>+0.566</td><td>-0.041</td><td>+0.049</td><td>+0.009</td><td>+0.077</td></tr><tr><td></td><td>20</td><td>+0.159</td><td>+0.141</td><td>+0.299</td><td>+0.444</td><td>+0.004</td><td>+0.057</td><td>+0.061</td><td>+0.090</td></tr><tr><td></td><td>35</td><td>+0.100</td><td>+0.161</td><td>+0.261</td><td>+0.475</td><td></td><td></td><td></td><td></td></tr><tr><td>1000</td><td>5</td><td>+0.049</td><td>+0.208</td><td>+0.257</td><td>+0.272</td><td>-0.005</td><td>+0.034</td><td>+0.029</td><td>+0.054</td></tr><tr><td></td><td>10</td><td>+0.034</td><td>+0.273</td><td>+0.307</td><td>+0.430</td><td>-0.012</td><td>+0.074</td><td>+0.062</td><td>+0.072</td></tr><tr><td></td><td>20</td><td>+0.052</td><td>+0.169</td><td>+0.222</td><td>+0.392</td><td>+0.010</td><td>+0.096</td><td>+0.106</td><td>+0.125</td></tr><tr><td></td><td>35</td><td>+0.033</td><td>+0.134</td><td>+0.167</td><td>+0.366</td><td></td><td></td><td></td><td></td></tr><tr><td>2000</td><td>5</td><td>+0.014</td><td>+0.129</td><td>+0.143</td><td>+0.163</td><td>-0.003</td><td>+0.029</td><td>+0.026</td><td>+0.037</td></tr><tr><td></td><td>10</td><td>-0.021</td><td>+0.289</td><td>+0.268</td><td>+0.313</td><td>-0.001</td><td>+0.064</td><td>+0.063</td><td>+0.079</td></tr><tr><td></td><td>20</td><td>+0.025</td><td>+0.167</td><td>+0.192</td><td>+0.242</td><td>-0.003</td><td>+0.093</td><td>+0.090</td><td>+0.122</td></tr><tr><td></td><td>35</td><td>-0.010</td><td>+0.130</td><td>+0.120</td><td>+0.247</td><td></td><td></td><td></td><td></td></tr><tr><td>5000</td><td>5</td><td>+0.001</td><td>+0.115</td><td>+0.116</td><td>+0.135</td><td>+0.001</td><td>+0.042</td><td>+0.043</td><td>+0.041</td></tr><tr><td></td><td>10</td><td>-0.004</td><td>+0.266</td><td>+0.262</td><td>+0.264</td><td>-0.006</td><td>+0.051</td><td>+0.045</td><td>+0.063</td></tr><tr><td></td><td>20</td><td>+0.005</td><td>+0.207</td><td>+0.212</td><td>+0.198</td><td>-0.005</td><td>+0.115</td><td>+0.110</td><td>+0.126</td></tr><tr><td></td><td>35</td><td>-0.010</td><td>+0.179</td><td>+0.169</td><td>+0.231</td><td></td><td></td><td></td><td></td></tr><tr><td>10000</td><td>5</td><td>+0.011</td><td>+0.089</td><td>+0.099</td><td>+0.111</td><td>+0.004</td><td>+0.047</td><td>+0.051</td><td>+0.049</td></tr><tr><td></td><td>10</td><td>-0.018</td><td>+0.296</td><td>+0.278</td><td>+0.290</td><td>-0.001</td><td>+0.103</td><td>+0.102</td><td>+0.107</td></tr><tr><td></td><td>20</td><td>+0.003</td><td>+0.205</td><td>+0.207</td><td>+0.204</td><td>+0.001</td><td>+0.106</td><td>+0.108</td><td>+0.109</td></tr><tr><td></td><td>35</td><td>-0.012</td><td>+0.207</td><td>+0.194</td><td>+0.210</td><td></td><td></td><td></td><td></td></tr><tr><td>all cells</td><td></td><td>+0.046</td><td>+0.188</td><td>+0.234</td><td>+0.293</td><td>-0.005</td><td>+0.065</td><td>+0.061</td><td>+0.078</td></tr></table>

## 5.4 Is it really the association?

The account above says the cross-block term uses the dependence between the two outcome blocks. That is a causal claim about the data, and it can be tested directly by removing the dependence and watching the term go away. Permuting the rows of $Y _ { B }$ does exactly that: every column marginal is preserved to the bit, $d _ { b }$ and the base rates are unchanged, and only the correspondence with $Y _ { A }$ is destroyed. Permuting a fraction α of the rows scales the damage. Table 4 reports the result on Instacart at $n _ { \mathrm { t r a i n } } = 2 0 0 0$ , and the mean absolute between-block correlation confirms the manipulation does what it should, falling from 0.060 to the 0.012 finite-sample noise floor.

The cross-block term tracks it. From +0.178 pp on untouched data it falls $\mathrm { { t o + 0 . 0 6 3 } }$ , crosses zero near $\alpha = 0 . 5$ , and settles at about −0.03 pp once the association is gone (Spearman $\rho = - 0 . 9 3 \mathrm { o v e r }$ the 25 seed means, $p < 1 0 ^ { - 1 0 }$ ; the change from $\alpha = 0 \tan \alpha = 1 { \mathrm { i s } } - 0 . 2 0 8 { \mathrm { p p } }$ , larger than the term itself). The overshoot is what one would expect rather than a puzzle: conditioning on a block that carries no information about the target is not merely useless but costs a little, since the inference now has to accommodate values that explain nothing. All four values of k show the same profile, positive at $\alpha = 0$ and significantly negative at $\alpha = 1$ , so this is not one column of the grid driving the result.

Two features of the table make the reading unambiguous. First, generative pre-training — the contribution that does not involve conditioning on $Y _ { B } - \mathrm { i s }$ flat across the whole sweep, within ±0.01 pp; the manipulation moves the term that should move and leaves alone the one that should not. Second, the collateral damage is small: permuting $Y _ { B }$ also destroys $x \mapsto y _ { B }$ , but X carries so little about $Y _ { B }$ on this data that the X-only logistic arm loses only 0.048 pp on that block. Almost all of the 0.24 pp that OBMP gives up is the cross-block channel itself.

Table 4: Breaking the association between the outcome blocks. Rows of $Y _ { B }$ are permuted for a fraction α of the data, which leaves $d _ { b } ,$ the base rates and every column marginal untouched and destroys only the dependence on $Y _ { A } ; \overline { { | r | } }$ is the resulting mean absolute correlation between the blocks on the test split. Instacart, $n _ { \mathrm { t r a i n } } = 2 0 0 0$ , all four k, five seeds [pp]. <sup>∗</sup> marks $p < 0 . 0 5$ against zero, clustered on seed.
<table><tr><td colspan="4">cross-block conditioning</td><td></td><td>control</td></tr><tr><td>α</td><td> $\overline { { | r | } }$ </td><td>combined</td><td> $Y _ { A }$ </td><td> $Y _ { B }$ </td><td>pre-training</td></tr><tr><td>0.00</td><td>0.060</td><td> $+ 0 . 1 7 8 ^ { * }$ </td><td>+0.288</td><td>+0.052</td><td>+0.002</td></tr><tr><td>0.25</td><td>0.048</td><td>+0.063*</td><td>+0.103</td><td>+0.017</td><td>+0.009*</td></tr><tr><td>0.50</td><td>0.034</td><td>+0.002</td><td>+0.003</td><td>+0.002</td><td>+0.005</td></tr><tr><td>0.75</td><td>0.019</td><td>-0.030*</td><td>-0.056</td><td>-0.000</td><td>+0.009</td></tr><tr><td>1.00</td><td>0.012</td><td>-0.029*</td><td>-0.055</td><td>-0.000</td><td>+0.005</td></tr></table>

## 5.5 The same mechanism, differently weighted

Read off the combined metric, cross-block conditioning looks three times larger on Instacart than on Bank Marketing. Table 5 shows that most of that gap is the dimension weighting in (16) rather than the mechanism. Restricting both grids to the k they share, the gain on $Y _ { A } \ \mathrm { i s } + 0 . 3 2 6 \ p \mathrm p$ on Instacart and +0.239 pp on Bank Marketing — a factor of 1.4 — while $\gamma _ { A } \mathrm { \dot { s } }$ share of the outcome dimensions differs by a factor of 2.8 $( \bar { 5 } \bar { 3 } / 9 9$ against $7 / 3 6 )$ . The combined figure is the weighted average of two block figures, so a dataset whose interesting block is narrow reports a smaller number for the same behavior.

The gain also lands on the same block in both cases. On Instacart it is $+ 0 . 3 2 6 \mathrm { p p }$ on $Y _ { A }$ against $+ 0 . 0 4 6$ on $Y _ { B } ;$ on Bank Marketing, +0.239 against +0.023. $Y _ { B }$ is the near-saturated block on both datasets, and conditioning on the other block buys little there. That the effect concentrates on the harder block, on two datasets whose block sizes are reversed, is what one would expect if the mechanism is the one we claim rather than an artifact of either dataset’s particular geometry.

Table 5: Cross-block conditioning by outcome block [pp], restricted to the $k \in \{ 5 , 1 0 , 2 0 \}$ the two grids share. The combined figure is the dimension-weighted average (16) of the two block figures. Most of the gap between the combined columns is that weighting rather than a weaker mechanism: $Y _ { A }$ carries 7 of 36 dimensions on Bank Marketing against 53 of 99 on Instacart, a factor of 2.8, while the $Y _ { A }$ gains themselves differ by a factor of 1.4.
<table><tr><td>dataset</td><td> $d _ { a }$ </td><td> $d _ { b }$ </td><td> $Y _ { A }$ </td><td> $Y _ { B }$ </td><td>combined</td></tr><tr><td>Instacart</td><td>53</td><td>46</td><td>+0.326</td><td>+0.046</td><td>+0.196</td></tr><tr><td>Bank Marketing</td><td>7</td><td>29</td><td>+0.239</td><td>+0.023</td><td>+0.065</td></tr></table>

## 5.6 Trends in $n _ { \mathrm { t r a i n } }$ and k

Table 6 regresses each contribution on $\log _ { 2 } ( n _ { \mathrm { t r a i n } } / 5 0 0 )$ and $k - 5$ with seed fixed effects. The sample-size axis is where the two datasets agree and where the claim lies. Cross-block conditioning has a non-negative $n _ { \mathrm { t r a i n } }$ slope on both $- + 0 . 0 0 4$ on Instacart $( p = 0 . 3 2 , \mathrm { i . e }$ . flat) and +0.008 on Bank Marketing $( p = 0 . 0 2 , \mathrm { i . e . }$ . growing). Nothing else we measure behaves that way: generative pre-training falls at −0.039 pp per doubling on Instacart $( p = 0 . 0 3 )$

and the total advantage over X-logistic falls at −0.059. For a term whose value is the claim, not decaying across a twentyfold range of sample sizes is the result.

The width axis does not agree. Instacart shows no significant k slope $( - 0 . 0 0 1 , p = 0 . 2 0 )$ and a non-monotone profile peaking at k=10; Bank Marketing shows a significantly positive one $( + 0 . 0 0 4 , p = 0 . 0 0 2 )$ rising monotonically across all three widths, and the slope survives dropping the k=5 column. We therefore make no claim about how the mechanism scales with the width of the common block: on the evidence here that relationship is dataset-specific, and a study designed to answer it would need to vary k while holding the composition of X fixed, which the popularityranked construction here does not do (§6.3).

Table 6: Trend of each contribution in sample size and common-block width: OLS of the paired difference [pp] on $\log _ { 2 } ( n _ { \mathrm { t r a i n } } / 5 0 0 )$ and $k - 5$ with seed fixed effects. Coefficients come from the pooled fit; <sup>∗</sup> marks $p < 0 . 0 5$ from the same slope estimated within each seed and tested across the five. The sample-size slope is never negative for cross-block conditioning; the width slope does not agree between datasets.
<table><tr><td>dataset</td><td>contribution</td><td> $\beta _ { \log _ { 2 } n _ { \mathrm { t r a i n } } }$ </td><td> $\beta _ { k }$ </td><td> $R ^ { 2 }$ </td></tr><tr><td>Instacart</td><td>generative pre-training</td><td>-0.0390*</td><td>-0.0016</td><td>0.247</td></tr><tr><td rowspan="3">Bank Marketing</td><td>cross-block conditioning</td><td>+0.0044</td><td>-0.0010</td><td>0.070</td></tr><tr><td>generative pre-training</td><td>+0.0029</td><td>+0.0004</td><td>0.106</td></tr><tr><td>cross-block conditioning</td><td>+0.0082*</td><td>+0.0038*</td><td>0.530</td></tr></table>

## 5.7 Does the mechanism need depth?

The model we have been calling a DBM has two hidden layers, and the mean-field sweep (§3.5) carries a top-down term because of it. Table 7 repeats both grids with that depth removed: L = 1 with $D _ { h _ { 1 } } = 9 6$ , holding all of the DBM’s hidden units in one layer, and L = 1 with $D _ { h _ { 1 } } = 6 4$ , matching its first layer only. With one weight matrix the sweep loses its top-down term and the model is an RBM; nothing else changes.

Table 7: Removing depth. The ablations replace the two hidden layers with one, keeping either the same total number of hidden units (96) or the first layer’s (64). Accuracy and contrasts are means over the whole grid [%, pp]. <sup>∗</sup> marks a contrast differing from that dataset’s DBM row at $p < 0 . 0 5 ,$ , paired over cells.
<table><tr><td>dataset</td><td>hidden layers</td><td>OBMP</td><td>cross-block</td><td>pre-training</td></tr><tr><td>Instacart</td><td>64 → 32 (DBM)</td><td>83.815</td><td>+0.188</td><td>+0.046</td></tr><tr><td rowspan="5">Bank Marketing</td><td>96 (RBM)</td><td>83.843</td><td>+0.243*</td><td>+0.018*</td></tr><tr><td>64 (RBM)</td><td>83.797</td><td> $+ 0 . 2 2 9 ^ { \ast }$ </td><td>-0.014*</td></tr><tr><td>64 → 32 (DBM)</td><td>77.388</td><td>+0.065</td><td>-0.005</td></tr><tr><td>96 (RBM)</td><td>77.391</td><td>+0.068</td><td>-0.004</td></tr><tr><td>64 (RBM)</td><td>77.396</td><td>+0.069</td><td>-0.000</td></tr></table>

Depth is not what produces the effect. The cross-block term is present in all six configurations and on Instacart is in fact larger without depth (+0.243 and +0.229 pp against +0.188, both differing from the DBM at $p < 0 . 0 5 ) ;$ ; on Bank Marketing the three architectures are within 0.01 pp of one another. Every cell of every architecture on both datasets has a positive cross-block term. Whatever the joint model is doing at inference time, it does not require a second hidden layer to do it.

On overall accuracy the architectures are indistinguishable for practical purposes: the three are within 0.05 pp on Instacart and 0.01 pp on Bank Marketing, and where a difference reaches significance it favors the shallower model, not the deeper one. We keep the two-layer model as the main configuration because it is the one the decomposition was developed on, but the honest summary is that the second hidden layer is not carrying the result.

Nor more inference. How long inference is allowed to run is a second thing one might expect the mechanism to depend on, since it is the one quantity here that can be spent freely at test time. Varying the cap over $T \in \{ 1 , \ldots , 5 0 \}$ on the primary checkpoints of both grids, it does not. The cross-block term is already significantly positive at $T = \dot { 1 }$ — one pass is visible-to-hidden-to-visible, which is all an observed $Y _ { B }$ needs to reach $Y _ { A } -$ and settles onto a plateau of +0.18 to +0.23 pp on Instacart and +0.04 to +0.08 pp on Bank Marketing from $T = 1 0$ onward. It converges no more slowly than the X-only term does, which is the opposite of what one would expect if the joint model’s advantage had to be iterated into existence.

The two datasets differ in how far the iteration actually runs, and that difference rather than the mechanism explains the shapes. On Bank Marketing, with a visible layer under half the size, the sweep genuinely converges: at a cap of 25, 71 of 75 runs already return exactly what they return at 50, and the cross-block term is flat in $T$ throughout. On Instacart the tolerance is still not met at a cap of 50 for most cells, so raising the cap keeps changing the predictions. What the extra passes buy there is absolute accuracy for both conditioning sets, and slightly more of it for X-only, so the gap narrows rather than widens as inference proceeds. $\mathrm { A t } T \le 2$ both arms score under a constant predictor, and X-only still does at $T = 3$ once $n _ { \mathrm { t r a i n } } \geq 5 0 0 0$ , so the larger gaps in that region are between two failed predictors and we do not read them as the mechanism

## 6 Discussion

## 6.1 What the decomposition implies

Taken together, the results give a specific account of what a joint model contributes in data fusion. The generative criterion by itself is not competitive: ML-DBM is the weakest arm on both grids and on Bank Marketing collapses onto the column-mean predictor exactly. Discriminative fine-tuning through OBMP is what makes the model competitive. And what OBMP adds over a plain network of the same capacity is almost entirely the ability to condition on one outcome block when predicting the other, not the pre-trained initialization: the former is present in every cell of both datasets and does not decay in $n _ { \mathrm { t r a i n } } .$ , the latter is confined to $n _ { \mathrm { t r a i n } } { = } 5 0 0$ on one dataset and absent on the other.

The practical consequence follows from which term survives. Every contribution that comes from representation — the generative initialization, and the total margin over a tuned discriminative model — shrinks as the panels grow, which is the ordinary expectation for anything a flexible discriminative model can eventually learn on its own. The cross-block term does not. Two things are being claimed there and they have different standing. That the term cannot be reproduced by a discriminative model is structural: such a model has no argument to put $y _ { B }$ into, whatever its capacity or sample size. That the term is positive is empirical, and rests on the conditional independence assumption $( \ S 3 . 6 )$ holding well enough on the data at hand — on one of the three datasets we tried, it does not. That is available to a joint model and unavailable to a discriminative one, and no amount of additional data changes which side of that line a method sits on. §6.2 sets these magnitudes against what the task allows, and §5.7 shows the term does not depend on the model being deep.

## 6.2 The size of the effect

The margins reported here are small in absolute terms, and it is worth being explicit about what they are small relative to. Both tasks admit little movement: on Instacart, predicting zero everywhere already scores 81.3% combined accuracy and the strongest baseline reaches 83.4%, so the entire distance any method can travel is 2.1 points; on Bank Marketing the corresponding figures are 75.1% and 77.3%, a distance of 2.2 points. OBMP’s +0.39 pp over the strongest Instacart baseline is 18% of the distance that baseline itself covered from the constant predictor. On cross-entropy, where the ceiling does not bind in the same way, the reduction against X-logistic is 2.7%.

The two outcome blocks also differ sharply in how far they can be predicted at all. On Instacart’s $Y _ { A }$ the constant predictor scores 72.7% and tuned X-logistic 76.7%; on $Y _ { B }$ they score 91.3% and 91.2%, that ${ \mathrm { i s } } ,$ the baseline does not beat a constant on $Y _ { B }$ . Averaging the blocks by dimension, as (16) does, therefore dilutes every contrast in this paper, and by different amounts on the two datasets (§5.5). We report the diluted figure throughout because it is the honest summary of the task as posed, but a reader setting these numbers against gains reported on tasks with more headroom should keep the difference in view.

None of this makes the effect large, and the argument of this paper does not rest on its size. It rests on which part of it survives. A margin that shrinks with $n _ { \mathrm { t r a i n } }$ is one a discriminative model will close given enough data, and most of what we measure behaves that way. The cross-block term does not, on either dataset, and the reason is structural: it uses evidence that a model mapping X to outcomes has no way to accept. A small effect that does not decay is a different kind of finding from a small effect that does, and it is the first kind we claim here.

## 6.3 Limitations

Three baselines do not emit probabilities. Hot deck, propensity matching and MissForest return hard values; MissForest in particular scores below a constant zero predictor because a regression forest’s continuous output is thresholded. Their accuracy gaps overstate the methodological distance and their cross-entropies are not comparable.

The k axis is confounded. Training subsets are drawn with a generator seeded jointly on the seed and k, so cells that differ in k use different rows, and the k trends in Table 6 are estimated across that extra variation. More importantly, increasing k adds progressively less informative covariates — aisles further down the popularity ranking, features further down the column order — so the k slope conflates “more covariates” with “sparser covariates”, and the two datasets rank their covariates by different criteria. This is the most likely reason the k slopes disagree (§5.6), and it is why we make no claim on that axis.

Panels split rather than collected. Both grids are built by partitioning one collected table into blocks, which is standard practice for evaluating fusion but is not the situation the method is for: in a real fusion the two panels come from different instruments, with different measurement error, coverage and definitions, and nothing guarantees the shared covariates mean the same thing in both. Our outcomes are also all binary. The mechanism should be tested on genuinely separate panels and on continuous outcomes before the magnitudes here are taken as representative.

## 7 Conclusion

Data fusion withholds the one thing a discriminative criterion needs: a row where the quantity to be predicted is recorded next to the evidence for it. Restricting multi-prediction training to targets the data actually contains removes that obstacle, and the resulting criterion is defined for any missingness pattern while reducing to the original one when rows are complete. That is a modest technical step, but it is what makes the substantive question askable, because it puts a discriminatively fine-tuned joint model and a discriminative model of the same capacity on the same footing

Asked that way, the answer is narrower than the totals suggest. The fine-tuned DBM is the best of fifteen methods in every cell of both grids, but the generative stage it starts from contributes almost nothing — it is confined to the smallest sample size on one dataset and absent on the other, and removing the second hidden layer costs nothing either. What the joint model supplies is the ability to condition on one outcome block while predicting the other, worth +0.19 and +0.07 pp, positive in every cell, and alone among the contributions in not fading as the panels grow, as the architecture is flattened, or as inference is allowed to run longer. It is small, and it is the part a discriminative model cannot reach at any capacity or sample size, because the evidence it uses is not of the form such a model accepts.

Two results mark the boundaries of that claim. How the mechanism scales with the width of the common block does not agree between our two datasets, and we make no claim on that axis. And on a third dataset whose partner block is only seven dimensions wide, the mechanism disappears and turns slightly negative — the gain is bounded by the association between the two outcome blocks, and there is a regime where that bound binds. Locating it, on panels less alike than the ones studied here, is what we would do next.

## Acknowledgment

This study is supported by JSPS KAKENHI (Grant No. JP24K16472).

## A Implementation details

Every arm is trained per cell with the seed fixed to the cell’s seed, so a cell is reproducible from its $( n _ { \mathrm { t r a i n } } , k , \mathrm { s e e d } )$ alone. Code and the per-cell result files are released with the paper.

Stage one (ML-DBM). SGD, learning rate 0.005, batch size $6 4 , \operatorname* { m a x } ( 1 5 , \lceil 2 5 0 0 0 / n _ { \mathrm { t r a i n } } \rceil )$ epochs. The positive phase uses 15 mean-field passes; the negative phase is persistent contrastive divergence with a chain of 64 states carried across minibatches and 5 Gibbs steps per update. With $p _ { \mathrm { c o m p l e t e } } = 0$ there is no complete subset to pre-train layerwise on, so we skip greedy pre-training and initialize the visible bias to $\log \mathrm { i t } ( \bar { v } _ { j } )$ over the entries with $m _ { i j } = 1$ clamped to $[ 1 0 ^ { - 3 } , 1 - 1 0 ^ { - 3 } ]$

Stage two (OBMP). AdamW, learning rate $1 0 ^ { - 3 }$ , no weight decay, batch size 64, gradient-norm clipping at 5.0, up to 200 epochs with validation every 5 and patience 60. The differentiable unroll runs a fixed $T = 1 0$ passes with no damping.

MLP. The same optimizer, learning rate, batch size, epoch budget, evaluation cadence and patience as stage two, with weight decay selected on validation from $\{ 0 , 1 0 ^ { - 4 } , \dot { 1 } 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ . Architecture $k  6 4  \bar { 3 } 2  ( d _ { a } + \bar { d _ { b } } )$ with sigmoid activations, matching the DBM’s hidden widths. The loss is the observed-entry cross-entropy normalized within each row before averaging over rows, so that it matches (6) rather than weighting rows by how many targets they contribute.

Tuned X-logistic. One liblinear logistic regression per outcome dimension, max $\mathtt { i t e r } = 5 0 0$ , inverse regularization strength selected on validation from $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 \}$ . A dimension with fewer than ten observed training rows, or with only one class among them, falls back to the observed marginal.

Baselines. All ten are fitted on the same training rows as the model arms and run at library defaults except where noted: k-NN with $k = 1 5 ;$ factor analysis with 8 components; MICE as IterativeImputer with 3 cycles and a ridge solver (the library default, BayesianRidge, raised a LAPACK convergence failure on every cell we tried); MissForest as IterativeImputer with a random-forest regressor, 10 trees, depth 8, 3 cycles; MIWAE with 128 hidden units, 16 latent dimensions, 10 importance samples in training and 50 at evaluation, 200 epochs, batch 256, learning rate $1 0 ^ { - 3 }$ . Hot deck and propensity-score matching have no free parameters in our implementation. None of the ten has its hyperparameters tuned per cell, which favors OBMP (§6.3).

Data preparation. Instacart is reduced to a user × aisle purchase-indicator matrix over 134 aisles; users are sampled without replacement from the full pool by a seeded permutation and sliced into train, validation and test. Bank Marketing is one-hot encoded, with continuous fields (age, balance, day, duration, campaign counts, days since previous contact, previous contacts) discretized to quartiles. Block membership is fixed by subject matter as described (§4.1); within a cell, the training subset of size $n _ { \mathrm { t r a i n } }$ is drawn by a permutation seeded on (seed, k), which makes subsets nested in $n _ { \mathrm { t r a i n } }$ within a seed but not aligned across k (§6.3).

Evaluation. Test-time mean-field runs to a cap of $T = 1 5$ passes with early exit once no coordinate moves by more than $1 0 ^ { - 4 }$ . Predictions are thresholded at 0.5 for accuracy and used as probabilities for cross-entropy.

## References

[1] Geoffrey Hinton. Nobel lecture: Boltzmann machines. Reviews of Modern Physics, 97(3):030502, 2025.

[2] David H Ackley, Geoffrey E Hinton, and Terrence J Sejnowski. A learning algorithm for boltzmann machines. Cognitive science, 9(1):147–169, 1985.

[3] Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, and Tariq Iqbal. Energy-based transformers are scalable learners and thinkers. In The Fourteenth International Conference on Learning Representations, 2026.

[4] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[5] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

[6] Yann LeCun. A path towards autonomous machine intelligence. OpenReview https://openreview.net/ forum?id=BZ5a1r-kVsf, 2022.

[7] Davide Carbone. Hitchhiker’s guide on the relation of energy-based models with other generative models, sampling and statistical physics: a comprehensive review. Transactions on Machine Learning Research, 2025.

[8] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, Fujie Huang, et al. A tutorial on energy-based learning. Predicting structured data, 2006.

[9] Yang Song and Diederik P Kingma. How to train your energy-based models. arXiv preprint arXiv:2101.03288, 2021.

[10] Louis Bethune, David Vigouroux, Yilun Du, Rufin VanRullen, Thomas Serre, and Victor Boutin. Follow the en-´ ergy, find the path: Riemannian metrics from energy-based models. Advances in Neural Information Processing Systems, 38:97824–97870, 2026.

[11] Ruslan Salakhutdinov and Geoffrey Hinton. Deep boltzmann machines. In Artificial intelligence and statistics, pages 448–455. PMLR, 2009.

[12] Wagner A Kamakura and Michel Wedel. Statistical data fusion for cross-tabulation. Journal of Marketing Research, 34(4):485–498, 1997.

[13] Junichiro Niimi and Takahiro Hoshino. A method for data fusion using deep boltzmann machine: An application of dbm in data fusion to predict customers behavior at competitors. Proceedings of the Annual Conference of JSAI, page 1I12, 2017.

[14] Ian Goodfellow, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Multi-prediction deep boltzmann machines. Advances in Neural Information Processing Systems, 26, 2013.

[15] Paul Smolensky. Information processing in dynamical systems: Foundations of harmony theory. Technical report, Department of Computer Science, University of Colorado at Boulder, 1986.

[16] Geoffrey E Hinton. A practical guide to training restricted boltzmann machines. In Neural Networks: Tricks of the Trade: Second Edition, pages 599–619. Springer, 2012.

[17] Ruslan Salakhutdinov and Iain Murray. On the quantitative analysis of deep belief networks. In Proceedings of the 25th international conference on Machine learning, pages 872–879, 2008.

[18] Paul R Rosenbaum and Donald B Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika, pages 41–55, 1983.

[19] Bailey K Fosdick, Maria DeYoreo, Jerome P Reiter, et al. Categorical data fusion using auxiliary information. The Annals ofApplied Statistics, 10(4):1907–1929, 2017.

[20] Stef Van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal ofstatistical software, 45:1–67, 2011.

[21] Daniel J Stekhoven and Peter Buhlmann. Missforest—non-parametric missing value imputation for mixed-type ¨ data. Bioinformatics, 28(1):112–118, 2012.

[22] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings ofthe 2nd International Conference on Learning Representations (ICLR 2014), 2014.

[23] Pierre-Alexandre Mattei and Jes Frellsen. Miwae: Deep generative modelling and imputation of incomplete data sets. In International conference on machine learning, pages 4413–4423. PMLR, 2019.

[24] Nitish Srivastava and Russ R Salakhutdinov. Multimodal learning with deep boltzmann machines. Advances in neural information processing systems, 25, 2012.

[25] Donald B Rubin. Inference and missing data. Biometrika, 63(3):581–592, 1976.

[26] Susanne Rassler. Data fusion: Identification problems, validity, and multiple imputation.¨ Austrian Journal of Statistics, 33(1–2):153–171, 2004.