# General Quantification of Covariate and Concept Shifts

Hongbo Chen <sup>1</sup> Li Charlie Xia <sup>1</sup>

## Abstract

Generalization under distribution shift remains a core challenge in modern machine learning, yet existing learning bound theory is limited to narrow, idealized settings and is non-estimable from samples. In this paper, we bridge the gap between theory and practical applications. We first show that existing definition of concept shift breaks when the source and target supports mismatch. Leveraging entropic optimal transport, we propose a key notion: γ<sup>∗</sup>-concept shifts, and derive a general error bound unifying covariate and γ<sup>∗</sup>- concept shifts, which applies to broad loss functions, label spaces, and stochastic labeling. We further develop estimators for these shifts with concentration guarantees, and the DataShifts algorithm, which can quantify distribution shifts and estimate the error bound in most applications - a rigorous and general tool for analyzing learning error under distribution shift.

## 1. Introduction

With the growth of data and computing power, supervised learning has achieved remarkable success. Nonetheless, traditional supervised learning assumes that the training data (source domain) and the test or deployment data (target domain) share the same distribution. However, in many real-world applications, the test data distribution can differ substantially from the training distribution, and this discrepancy can significantly impact model performance in the target domain. To analyze such challenges, researchers theorized that the distributions between the source and target domains are shifted and developed methods to assess how learners trained in the source domain could perform on the target domain. Depending on whether the target domain data is accessible, the problems are further categorized into domain adaptation and domain generalization.

Theoretical results on distribution shift usually bound a model’s target domain error by its source domain error plus a measure of the distribution shift. The shift is further dissected into X (covariate) and Y|X (concept) shifts (Moreno-Torres et al., 2012; Liu et al., 2021; Zhang et al., 2023). Early studies proposed using H-divergence to measure X shift and derived an error bound for binary classification (Ben-David et al., 2006; 2010). Later works improved X shift using the maximum mean discrepancy (Long et al., 2015) and the Wasserstein distance (Shen et al., 2018; Courty et al., 2017). Further studies proposed more complex metrics for the X shift to obtain bounds for multiclass classification (Zhang et al., 2020; 2019). These results focused only on X shift and additionally relied on a joint-error term between the source and target domains. Lately, Zhao et al. (2019) improved this loose joint-error term and proposed a bound that explicitly considers both X and Y|X shifts for binary classification, and Zhang et al. (2023) extended the theory to multiclass classification.

In this paper, we focus on two remaining key problems in the existing theoretical frameworks, which significantly hinder researchers from analyzing X and Y|X shifts in applications:

• Generalizability. Existing theories rely on restrictive assumptions: they require deterministic labeling, omitting label noise and latent confounders commonly seen in practice; moreover, they only apply to classification with absolute error, excluding broader tasks such as regression and loss families.

• Estimability. Although existing theories provide a preliminary definition of Y|X shift, this definition and the accompanying error bounds—are not estimable. As a result, one cannot rigorously quantify Y|X shift on real data, nor assess its impact on model performance.

We aim to provide a general theoretical framework unifying X and Y|X shifts that widely applies to stochastic labeling, most supervised tasks, and broader loss functions. Moreover, these two shifts can be accurately estimated from samples, thereby offering a rigorous, plug-and-play tool for quantifying and analyzing distribution shifts in real applications.

Specifically, our key observation is that if X shift occurs, the supports of the source and target covariate distributions may not overlap. Such a support mismatch renders the existing theories’ Y|X shift ill-defined, fundamentally causing the error bounds non-estimable and loose. Our key innovation is to employ the entropic optimal transport to give a general definition for X and $\mathrm { Y } | \mathrm { X }$ shift. The $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift we propose, which depends on the entropic optimal transport coupling of X shift, stays well-defined even when supports mismatch, and applies to the stochastic labeling and general label space. Based on that, we derived a new error bound that considers both the X and the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shifts. Our new bound relies only on the Lipschitz continuity of the hypothesis h and loss $\ell ,$ and is agnostic to specific hypothesis space, label space, or loss function, which naturally generalizes to binary and multiclass classification, regression and other tasks.

Since our $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift is well-defined regardless of support mismatch, X and Y|X shifts, and our new error bound becomes estimable from samples. Nonetheless, for X shift, the traditional plug-in estimator for entropic optimal transport tends to overestimate due to the curse of dimensionality. We thus further developed a debiased estimator that remains accurate in high dimensions. For our $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift, we also proposed an estimator. We proved both estimators concentration inequalities to their true values. Leveraging these two estimators, we presented the DataShifts algorithm, enabling quantification of X and Y|X shifts on general labeled data. Finally, we apply our theoretical framework and DataShifts to three distinct tasks: Novozymes enzyme prediction, ColoredMNIST, and PACS, clearly validating the general effectiveness of our theoretical results.

Contributions. In summary, our major contributions are:

• We show that support mismatch makes the existing Y|X shift ill-defined—hence loose and non-estimable. We introduce the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift as a well-defined concept shift, which possesses many desirable properties.

• We derive a general error bound unifying covariate and γ<sup>∗</sup>-concept shifts. Our bound covers broader learning scenarios, far beyond traditional binary classification.

• For the X and $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shifts in our theory, we propose two estimators and prove their concentration inequalities, ensuring that these shifts can be rigorously estimated from finite samples.

• We integrate our theoretical results into the DataShifts algorithm, which can estimate X and $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shifts from real data, supplying a rigorous and general tool for quantifying and analyzing distribution shifts.

The paper is organized as follows: Section 2 covers the preliminaries, Section 3 presents the population-level theoretical results, Section 4 focuses on the statistical results, and Section 5 presents experiments validating our theory.

## 2. Preliminary

## 2.1. Problem Setup

Our problem is to bound the error of models under distribution shift. Let X and Y be the covariate space and the label space, respectively. Let $\mathcal { D } _ { X Y } ^ { S }$ and $\mathcal { D } _ { X Y } ^ { T }$ be the joint distributions of covariates and labels on $\mathcal { X } \times \mathcal { V }$ for the source and target domains, respectively. $\mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$ are their covariate marginals on $\mathcal { X } .$ For any $x \in \mathcal { X }$ , we let $\mathcal { D } _ { Y \mid X = x } ^ { S }$ and $\mathcal { D } _ { Y \mid X = x } ^ { T }$ be the conditional label distributions at x in the source and target domain. Let $\mathcal { V } ^ { \prime }$ be the output space of the learner, $\ell : \mathcal { V } \times \mathcal { V } ^ { \prime } $ R the loss, and ${ \mathcal { H } } \subseteq \{ g : \mathcal { X } \to \mathcal { Y } ^ { \prime } \}$ the hypothesis space. For a hypothesis $h \in \mathcal H$ , the learning errors for source and target domains are:

$$
\epsilon _ { S } ( h ) = \mathbb { E } _ { ( x _ { S } , y _ { S } ) \sim \mathcal { D } _ { X Y } ^ { S } } \left[ \ell ( y _ { S } , h ( x _ { S } ) ) \right]
$$

$$
\epsilon _ { T } ( h ) = \mathbb { E } _ { ( x _ { T } , y _ { T } ) \sim \mathcal { D } _ { X Y } ^ { T } } \left[ \ell ( y _ { T } , h ( x _ { T } ) ) \right]
$$

We hope to bound $\epsilon _ { T } ( h )$ by $\epsilon _ { S } ( h )$ and a measure of distribution shift, consistent with existing theoretical results (Ben-David et al., 2006; Zhao et al., 2019). Notably, the space X can be the raw input space or a representation space output by an upstream learner (Ben-David et al., 2006). Our theory treats them in the same way, so it applies to both raw data and learned representations.

Stochastic and deterministic labeling. Above, we assume that the label $y$ follows the conditional distribution at point x: $y \sim \mathcal { D } _ { Y | X = x }$ , namely the stochastic labeling setting (Zhao et al., 2019). It enables our theory to accommodate latent confounders and label noise, which are common in practice. In contrast, existing theories oversimplify by using deterministic labeling: a labeling function $f : \mathcal { X }  \mathcal { Y }$ with $y = f ( x )$ , a special case of stochastic labeling in which the conditional distribution collapses to a Dirac mass at $f ( x ) , \mathrm { i . e . , } { \mathcal { D } } _ { Y | X = x } = \delta _ { f ( x ) }$

## 2.2. Existing Theory and Ill-Defined Y|X Shift

We first demonstrate that support mismatch leads to an illdefined Y|X shift, a key flaw in existing theory.

Definition 2.1 (Support). Let µ be a probability measure on the topological space $( \mathcal { X } , \tau )$ . Its support is defined as:

$$
\operatorname { s u p p } ( \mu ) \ = \ \{ x \in \mathcal { X } \mid \forall U \in \tau , \ x \in U , \ \mu ( U ) > 0 \}
$$

supp $( \mu )$ is the closure of every region where $\mu$ has positive measure, or equivalently, the complement of the union of all µ-null open sets:

$$
\operatorname { s u p p } ( \mu ) \ = \ \mathcal { X } \setminus \left\{ \bigcup \{ U \in \tau \mid \mu ( U ) = 0 \} \right\}
$$

Lemma 2.2 (Ill-Defined Expectation of Conditional Probability). For probability measure $P _ { X } , a \ P _ { X }$ -measurable function $P ( A \mid X = x ) : { \mathcal { X } } \to [ 0 , 1 ]$ is the conditional probability of event A given $X = x .$ . For probability measure $P _ { X } ^ { \prime }$ satisfying supp $( P _ { X } ^ { \prime } ) \setminus \operatorname { s u p p } ( P _ { X } ) \neq \emptyset$ , the expectation $\mathbb { \bar { E } } _ { P _ { X } ^ { \prime } } [ P ( A \mid X = x ) ]$ is arbitrary.

Remark This problem arises because the conditional probability $P ( A \mid X = x )$ is unique almost everywhere with respect to $P _ { X }$ (unique $P _ { X ^ { - } \mathbf { a } . \mathbf { e } . ) }$ , rather than to $P _ { X } ^ { \prime }$ . When supp $( P _ { X } ^ { \prime } ) \ \backslash$ supp $( P _ { X } ) \neq \varnothing , P _ { X } ^ { \prime }$ assigns positive mass to some $P _ { X } \mathrm { - n u l l }$ sets. In this case, $\mathbb { E } _ { P _ { \mathrm { x } } ^ { \prime } } [ P ( A \mid X = x ) ]$ depends on the values of $P ( A \mid X = { \overset {  } { x } } )$ on sets where it is not uniquely determined. Hence, support mismatch leads to the ill-defined expectation of conditional probability. This issue directly impacts the definition and computation of $\mathrm { Y } | \mathrm { X }$ shift in existing theories, since the $\mathrm { Y } | \mathrm { X }$ shift is typically formulated as an expectation over conditional labeling distributions or its collapsed labeling functions.

With deterministic labeling, Zhao et al. (2019) used $\mathcal { H } -$ divergence (Ben-David et al., 2006) to derive an error bound for soft-label binary classification under X and $\mathrm { Y } | \mathrm { X }$ shifts:

Theorem 2.3 (Existing Learning Bound on Distribution Shift). Let $\mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$ be the covariate distributions and $f _ { S } , f _ { T } : \mathcal { X }  [ 0 , 1 ]$ be the labelingfunctionsfor the source and target domain. Using absolute error loss | · |, for any hypothesis space $\mathcal { H } \subseteq [ 0 , 1 ] ^ { \chi }$ define:

$$
\tilde { \mathcal { H } } = \left\{ \mathrm { s g n } ( | h ( x ) - h ^ { \prime } ( x ) | - t ) ~ \middle | ~ h , h ^ { \prime } \in \mathcal { H } , ~ t \in [ 0 , 1 ] \right\} ,
$$

then for any $h \in \mathcal { H } ,$

$$
\begin{array} { r l r } & { } & { \epsilon _ { T } ( h ) \ \leq \ \epsilon _ { S } ( h ) + d _ { \tilde { \mathcal { H } } } \big ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } \big ) + \operatorname* { m i n } \Big \{ \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { S } } \big [ | f _ { S } ( x ) } \\ & { } & { \quad \quad - f _ { T } ( x ) | \big ] , \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { T } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ] \Big \} \quad ( 1 ) } \end{array}
$$

Here, $d _ { \tilde { \mathcal { H } } }$ measures X shift, and the two expectations $\mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \boldsymbol { x } } ^ { S } } [ | \dot { \boldsymbol { f } } _ { S } ( \boldsymbol { x } ) - \boldsymbol { f } _ { T } ( \boldsymbol { x } ) | ] , \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \boldsymbol { x } } ^ { T } } [ | \boldsymbol { f } _ { S } ( \boldsymbol { x } ) - \boldsymbol { f } _ { T } ( \boldsymbol { x } ) | ]$ are both $\mathrm { Y } | \mathrm { X }$ shift but measured with the source or target covariate distribution, with the smaller one taken in the bound.

Remark\* (Limitations of Theorem 2.3) First, it is highly specialized, applying only to deterministic labeling, binary classification, and absolute loss. Moreover, when X shift causes support mismatch : $\operatorname { s u p p } ( { \mathcal { D } } _ { X } ^ { T } ) \backslash \operatorname { s u p p } ( { \mathcal { D } } _ { X } ^ { S } ) \neq \emptyset$ , the conditional distribution $\mathcal { D } _ { Y \mid X = x } ^ { S } \left( \mathrm { i . e . , } f _ { S } ( x ) \right)$ is only unique $\mathcal { D } _ { X } ^ { S } \mathrm { - a . e . }$ . and arbitrary on the mismatched region, so the $\mathrm { Y } | \mathrm { X }$ shift $\mathbb { E } _ { { x } \sim { D _ { \mathrm { ~ v ~ } } ^ { T } } } [ | f _ { S } ( x ) - f _ { T } ( x ) | ]$ is ill-defined. Similarly, if supp $( \mathcal { D } _ { X } ^ { S } ) \ \stackrel { \cdot \cdot } { \setminus } \ \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { T } ) \neq \emptyset$ , the other Y|X shift term is also ill-defined. This additionally causes two problems:

• Loose bound. The ill-defined $Y | X$ shifts can take arbitrary values, making the bound in Eq. (1) loose.

• Non-estimable. When the real concept $\mathcal { D } _ { Y \mid X = x } ^ { S }$ or $f _ { S } ( x )$ is unknown, we cannot sample it outside supp $( \mathcal { D } _ { X } ^ { S } )$ ; hence, under support mismatch the expectations $\bar { \mathbb { E } } _ { x \sim \mathcal { D } _ { x } ^ { T } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ]$ is non-estimable, as well as $\mathbb { E } _ { { x } \sim { D _ { x } ^ { s } } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ]$

## 2.3. Entropic Optimal Transport

We introduce entropic optimal transport, which we will employ to give the general definitions of X and $\mathrm { Y } | \mathrm { X }$ shifts properly. It measures the distance between two probability distributions, augmenting conventional optimal transport with a relative-entropy regularizer. Given two probability distributions P and $\mathbb { Q }$ on the metric space $( \Omega , \rho )$ and a parameter $\beta \geq 0$ , the order-1 entropic optimal transport is:

$$
W _ { \beta } ( \mathbb { P } , \mathbb { Q } ) = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mathbb { P } , \mathbb { Q } ) } \left\{ \int \rho d \gamma + \beta H ( \gamma \mid \mathbb { P } \otimes \mathbb { Q } ) \right\}\tag{2}
$$

where $\begin{array} { r } { H ( \gamma \mid \mathbb { P } \otimes \mathbb { Q } ) = \int \log \left( \frac { d \gamma ( x _ { 1 } , x _ { 2 } ) } { d \mathbb { P } ( x _ { 1 } ) d \mathbb { Q } ( x _ { 2 } ) } \right) d \gamma ( x _ { 1 } , x _ { 2 } ) } \end{array}$ Here, $\rho ( x _ { 1 } , x _ { 2 } )$ is the cost of transporting mass between points, Γ(P, Q) is the set of joint distributions with marginals P and $\mathbb { Q } ,$ and $\gamma ( x _ { 1 } , x _ { 2 } )$ is a transport coupling. Hence, $W _ { \beta } ( \mathbb { P } , \mathbb { Q } )$ is the minimum total transport cost between P and Q under an entropy regularizer. When $\beta = 0 .$ this reduces to the Wasserstein-1 distance, denoted by $W _ { 1 } ( \mathbb { P } , \mathbb { Q } )$ . Compared with other distribution distances, (entropic) optimal transport has a well-established geometric meaning (Gangbo & McCann, 1996), and we will consistently use it to measure distribution shifts.

## 3. Theoretical Results

In this section, we focus on population-level theoretical results. We assume that covariate X and label Y are distributed on the metric spaces $( \mathcal { X } , \rho _ { \mathcal { X } } )$ and $( \mathcal { V } , \rho _ { \mathcal { V } } )$ , respectively, and give the rigorous and general definitions of their distribution shifts and essential theorems below.

## 3.1. General X and Y|X Shifts

Definition 3.1 (X Shift). Using entropic optimal transport, the X (covariate) shift is defined as:

$$
\begin{array} { r l r } { S _ { C o v } = W _ { \beta } \big ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } \big ) } & { = } & { \underset { \gamma \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } ) } { \operatorname* { i n f } } \Bigg \{ \int \rho _ { \mathcal { X } } \big ( x _ { S } , x _ { T } \big ) } \\ & { } & { d \gamma \big ( x _ { S } , x _ { T } \big ) + \beta H \big ( \gamma \mid \mathcal { D } _ { X } ^ { S } \otimes \mathcal { D } _ { X } ^ { T } \big ) \Bigg \} \quad \quad ( \mathcal { D } _ { X } ^ { \perp } \big ( x _ { S } , x _ { T } \big ) } \end{array}\tag{3}
$$

where the optimal coupling is denoted as $\gamma ^ { * }$ , a joint distribution of $\mathcal { D } _ { X } ^ { \bar { S } } , \mathcal { D } _ { X } ^ { T }$ that gives the minimum transport cost.

With realistic stochastic labeling, the label follows a conditional distribution given the covariate; we give a general and rigorous definition of Y|X shift below:

Definition 3.2 $( \gamma ^ { * } \mathrm { - } \Upsilon | \mathrm { X }$ Shift). Let $\gamma ^ { * }$ be the optimal transport coupling as in Definition 3.1, $S _ { p a i r } ( x _ { S } , x _ { T } ) =$

$W _ { 1 } \big ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \big )$ , then the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift is defined as the expectation of $S _ { p a i r } ( x _ { S } , x _ { T } )$ under $\gamma ^ { * }$ :

$$
\begin{array} { l } { { S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \left[ S _ { p a i r } ( x _ { S } , x _ { T } ) \right] } } \\ { { \mathrm { ~ \ ~ \ } = \displaystyle \int W _ { 1 } \left( { \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } } \right) d \gamma ^ { * } ( x _ { S } , x _ { T } ) } } \end{array}\tag{4}
$$

This definition is based on the optimal transport coupling $\gamma ^ { * }$ for X shift. $S _ { p a i r } ( x _ { S } , x _ { T } )$ denotes the paired conditional distribution shift between the source domain at $x _ { S }$ and the target domain at $x _ { T }$ . Intuitively, since optimal transport coupling $\gamma ^ { * }$ places most of its mass on nearby pairs $( x _ { S } , x _ { T } )$ $S _ { C p t } ^ { \gamma ^ { * } }$ is evaluated mainly on such neighboring points.

With deterministic labeling, this definition reduces to:

Corollary 3.3 (Consistency under Deterministic Labeling). Assume deterministic labeling ${ \mathcal D } _ { Y \mid X = x } ^ { S } \ = \ \delta _ { f s ( x ) } ,$ $\mathcal { D } _ { Y | X = x } ^ { T } = \delta _ { f _ { T } ( x ) }$ , then:

$$
{ S _ { C p t } ^ { \gamma ^ { * } } } = \mathbb { E } _ { \gamma ^ { * } } \left[ S _ { p a i r } ( x _ { S } , x _ { T } ) \right] = \mathbb { E } _ { \gamma ^ { * } } \left[ \rho _ { \mathcal { V } } ( f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } ) ) \right]
$$

The following three lemmas guarantee the good properties of $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ Shift.

Lemma 3.4 (Support of $\gamma ^ { * } )$ . For any $\gamma \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ $\mathrm { s u p p } ( \gamma ) \subseteq \mathrm { s u p p } ( \bar { \mathcal { D } } _ { X } ^ { S } ) \times \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { T } )$

Lemma 3.5 (Uniqueness of $\gamma ^ { * } )$ $H \beta > 0 ,$ , then the entropic optimal transport coupling $\gamma ^ { * }$ in Definition 3.1 is unique.

Lemma 3.6 (Collapse of $\gamma ^ { * } ) . \ f \ \beta \ = \ 0$ and $\begin{array} { r l } { \mathcal { D } _ { X } ^ { S } } & { { } = } \end{array}$ $\mathcal { D } _ { X } ^ { T } , ~ \gamma ^ { * }$ collapses to diagonal (identity) coupling $\gamma ^ { * } =$ $( \mathrm { I d } , \mathrm { I d } ) _ { \# } \mathcal { D } _ { X } ^ { S }$ , equivalently, for every measurable $A \subseteq$ $\begin{array} { r } { \mathcal { X } \times \mathcal { X } , \gamma ^ { * } \overset {  } { ( } A ) = \int \mathbf { 1 } _ { \{ ( x , x ) \in A \} } d \mathcal { D } _ { X } ^ { S } ( x ) } \end{array}$ , where $\mathbf { 1 } _ { \{ \cdot \} }$ is the indicatorfunction.

Combining Lemmas 3.4 and 3.5, the rigor of $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ Shift is ensured:

Theorem 3.7 (Well-Definedness of $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ Shift). For $\beta >$ 0, the $\gamma ^ { * } { \mathrm { - } } Y | X$ shift $S _ { C p t } ^ { \gamma ^ { * } }$ in Definition 3.2 is unique even when supp $( \mathcal { D } _ { X } ^ { S } ) \neq \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { T } )$ .

Remark Such $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift not only avoids the illdefinition encountered by existing theory, but also makes $\mathrm { Y } | \mathrm { X }$ shift tight and estimable. On the other hand, combining Corollary 3.3 and Lemma $3 . 6 ,$ the relationship between $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift and existing Y|X shift is shown as follows:

Proposition 3.8 (Relationship to existing Y|X shift). Assume deterministic labeling $\mathscr { D } _ { Y | X = x } ^ { S } = \delta _ { f s ( x ) } , \mathscr { D } _ { Y | X = x } ^ { T } =$ $\delta _ { f _ { T } ( x ) }$ and $\mathcal { V } \subset \mathbb { R }$ with $\rho _ { \mathcal { Y } } = | \cdot | , i f \beta = 0$ and $\mathcal { D } _ { X } ^ { S ^ { \prime } } = \mathcal { D } _ { X } ^ { T }$ then:

$$
\begin{array} { r } { { S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { S } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ] } } \\ { { = \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { T } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ] } } \end{array}
$$

Remark The above $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift, defined via the entropic optimal transport coupling $\gamma ^ { * }$ , applies to general settings including stochastic labeling. When $\mathcal { D } _ { X } ^ { S } = \bar { \mathcal { D } } _ { X } ^ { T }$ , it recovers the existing $\mathrm { Y } | \mathrm { X }$ shift; and when the supports of $\mathcal { D } _ { X } ^ { S }$ and $\mathcal { D } _ { X } ^ { T }$ are mismatched, it still remains rigorous.

## 3.2. General Learning Bound

We now give a new cross-domain learning error bound based on X and $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift. To be general, the output space of the learner is a metric space $( \mathcal { V } ^ { \prime } , \rho _ { \mathcal { V } } ^ { \prime } )$ , possibly different from the true label space $( \mathcal { V } , \rho _ { \mathcal { V } } )$ . For the loss function $\ell : \mathcal { V } \times \mathcal { V } ^ { \prime } \to \mathbb { R }$ , we require the following basic assumption from them:

Assumption 3.9 (Separately Lipschitz Continuity). For metric spaces $( \mathcal { V } , \rho _ { \mathcal { V } } )$ and $( \mathcal { V } ^ { \prime } , \rho _ { \mathcal { V } } ^ { \prime } )$ , a function $\ell : \mathcal { V } \times$ $\mathcal { V } ^ { \prime } $ R satisfies separately $( L _ { \ell } , L _ { \ell } ^ { \breve { \prime } } )$ -Lipschitz if there exist $L _ { \ell } , L _ { \ell } ^ { \prime } \ge 0$ such that for any $y _ { 1 } , y _ { 2 } \in \mathcal { D }$ and $y _ { 1 } ^ { \prime } , y _ { 2 } ^ { \prime } \in \mathcal { V } ^ { \prime }$ there is:

$$
\left| \ell ( y _ { 1 } , y _ { 1 } ^ { \prime } ) - \ell ( y _ { 2 } , y _ { 2 } ^ { \prime } ) \right| \leq L _ { \ell } \rho _ { \mathcal { V } } ( y _ { 1 } , y _ { 2 } ) + L _ { \ell } ^ { \prime } \rho _ { \mathcal { V } } ^ { \prime } ( y _ { 1 } ^ { \prime } , y _ { 2 } ^ { \prime } )\tag{5}
$$

Remark This is a mild assumption: most loss functions are differentiable, which already implies continuity. And their stable optimization usually needs bounded gradients in a region, further ensuring Lipschitz continuity (Boyd & Vandenberghe, 2004). Note that this assumption also covers asymmetric losses. For instance, for binary classification with label space [0, 1], output space is typically $[ a , 1 - a ]$ $( a \in ( 0 , 0 . 5 ) )$ by Sigmoid function and $\rho y = \rho _ { y } ^ { \prime } = | \cdot | .$ the cross-entropy loss: $\ell _ { C E } ( y , \hat { y } ) = - y \log { \hat { y } } - \mathbf { \mu } ( 1 -$ $y ) \log ( 1 - \hat { y } )$ satisfies separately $( L _ { \ell } , L _ { \ell } ^ { \prime } ) - 1$ Lipschitz with $\begin{array} { r } { L _ { \ell } = \log \left( \frac { 1 - a } { a } \right) } \end{array}$ and $\begin{array} { r } { L _ { \ell } ^ { \prime } = \frac { 1 } { a } } \end{array}$

Corollary 3.10 (Composition Preserves Separate Lipschitzness). Let $h : \mathcal { X } \ :  \ : \mathcal { Y } ^ { \prime }$ be $L _ { h } – L i p s c h i t z$ and let $\ell : \mathcal { Y } \times \mathcal { Y } ^ { \prime } \to \mathbb { R }$ be separately $( L _ { \ell } , L _ { \ell } ^ { \prime } ) { \ – } L i p s c h i t z .$ . Then compositefunction $\ell \big ( y , h ( x ) \big ) : y \times \mathcal { X }  .$ R is separately $( L \ell , L _ { h } L _ { \ell } ^ { \prime } ) { - } L i p s c h i t z$

The following two lemmas characterize the transport coupling of joint distributions $\mathcal { D } _ { X Y } ^ { S }$ and $\mathcal { D } _ { X Y } ^ { T }$

Lemma 3.11 (Separately Weak Duality). Let $\mathcal { G } \left( y , x \right) : \mathcal { V } \times$ $\mathcal { X }  \mathbb { R }$ is separately $( L _ { \mathcal { Y } } , L _ { \mathcal { X } } )$ -Lipschitz,for any coupling of joint distributions $\hat { \gamma } _ { X Y } \in \Gamma ( \mathcal { D } _ { X Y } ^ { S } , \mathcal { D } _ { X Y } ^ { T } )$ , there exists:

$$
| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { S } } [ \mathcal { G } ] ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] | \leq L _ { \mathcal { X } } E _ { \gamma _ { X Y } } [ \rho _ { \mathcal { X } } ] + L _ { \mathcal { Y } } E _ { \gamma _ { X Y } } [ \rho _ { \mathcal { Y } } ]\tag{6}
$$

Lemma 3.12 (Gluing Construction for Joint Coupling). Let $\gamma ^ { * }$ be the optimal transport coupling of X shift in Definition $3 . I , \ \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * }$ be the optimal transport coupling of $S _ { p a i r } ( x _ { S } , \stackrel { . . } { x _ { T } } )$ in Definition 3.2, construct the joint distribution of couplings: $\gamma _ { _ { X Y } }$ (dx<sub>S</sub>, dx<sub>T</sub>, $d y _ { S } , d y _ { T } ) ~ =$ $\gamma ^ { * } ( d x _ { S } , d x _ { T } ) \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } ( d y _ { S } , d y _ { T } )$ , then $\gamma _ { x Y }$ is the coupling ofjoint distributions: $\gamma _ { _ { X Y } } \in \Gamma ( \mathcal { D } _ { X Y } ^ { S } , \mathcal { D } _ { X Y } ^ { T } )$

Combining Corollary 3.10, Lemmas 3.11 and 3.12, we obtain the general cross-domain error bound as follows:

Theorem 3.13 (General Learning Bound). Given the covariate space $( \mathcal { X } , \rho _ { \mathcal { X } } )$ , the label space $( \mathscr { y } , \rho _ { \mathscr { y } } )$ , and the output space $( \mathcal { V } ^ { \prime } , \rho _ { \mathcal { V } } ^ { \prime } )$ , the source and target distributions are $( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { Y \mid X = x } ^ { S } )$ and $( \mathcal { D } _ { X } ^ { T } , \mathcal { D } _ { Y \mid X = x } ^ { T } )$ , respectively. If the loss $\ell : \dot { \mathcal { y } } \times \mathcal { y } ^ { \prime } $ R satisfies separately $( L _ { \ell } , L _ { \ell } ^ { \prime } )$ Lipschitz, then for any hypothesis h : $\mathcal { X }  \mathcal { V } ^ { \prime }$ that satisfies $L _ { h } – L i p s c h i t z ,$ the following bound holds:

$$
\epsilon _ { T } ( h ) \ \leq \ \epsilon _ { S } ( h ) \ + \ L _ { h } L _ { \ell } ^ { \prime } S _ { C o v } \ + \ L _ { \ell } S _ { C p t } ^ { \gamma ^ { * } }\tag{7}
$$

where $S _ { C o v }$ is the X shift in Definition 3.1 and $S _ { C p t } ^ { \gamma ^ { * } }$ is the $\gamma ^ { \ast } \ – Y | X s h i f t$ in Definition 3.2.

Remark This elegant bound unifies the covariate shift $S _ { C o v }$ and the concept shift $( \gamma ^ { * } \mathrm { - } \mathrm { Y } | \mathrm { X }$ shift) $S _ { C p t } ^ { \gamma ^ { * } } \mathrm { } ^ { , } \mathrm { } s$ effect on target error via the Lipschitz factors $L _ { h } L _ { \ell } ^ { \prime }$ and $L _ { \ell } .$ . Notably, it depends only on the Lipschitz continuity of the hypothesis $h$ and the loss ℓ, without any other specific restriction on the label space $\mathcal { V }$ or loss ℓ. It naturally covers binary classification or regression tasks when $\mathcal { V }$ is one-dimensional, and multiclass classification or multi-label tasks when $\mathcal { V }$ is multi-dimensional. Besides, since the bound holds under stochastic labeling, it applies to a wide range of supervised learning scenarios in practice. On the other hand, by using the well-defined $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift, our bound can be tighter than the existing bound, and more crucially, both the covariate and concept shifts in our theory can be robustly estimated from the finite samples. Additionally, since the entropic optimal transport $S _ { C o v }$ increases with the hyperparameter $\beta ,$ we recommend choosing $\beta$ as a small non-zero value to balance the tightness and the rigor of the theory.

## 4. Statistical Results

In practice, true domain distributions are all unknown. We wish to estimate the shifts from domain samples and analyze how these shifts will influence model performance. In this section, we focus on sample-level theoretical results. Suppose we have i.i.d. samples $\{ ( \boldsymbol { X } _ { i } ^ { ( S ) } , \boldsymbol { Y } _ { i } ^ { ( S ) } ) \}$ and $\{ ( X _ { j } ^ { ( T ) } , Y _ { j } ^ { ( T ) } ) \}$ drawn from $\mathcal { D } _ { X Y } ^ { S }$ and $\mathcal { D } _ { X Y } ^ { T }$ with sizes $N _ { S }$ and $\hat { N } _ { T }$ , respectively. This section involves three parts: the estimation of X shift, the estimation of Y|X shift, and the DataShifts algorithm to estimate the overall bound.

## 4.1. Estimation of X Shift

By Definition 3.1, the X shift is the entropic optimal transport between $\mathcal { D } _ { X } ^ { S }$ and $\mathcal { D } _ { X } ^ { T }$ . The traditional estimation method uses the entropic optimal transport of empirical distributions—known as the plug-in estimator.

Traditional Plug-in Estimator Given i.i.d. samples $\{ X _ { i } ^ { ( S ) } \} ~ \sim ~ \mathcal { D } _ { X } ^ { S } , ~ \{ X _ { j } ^ { ( T ) } \} ~ \sim ~ \mathcal { D } _ { X } ^ { T }$ with sample sizes $N _ { S } , N _ { T }$ respectively, set the empirical measures: ${ \widehat { \mathcal { D } _ { X } ^ { S } } } =$ $\begin{array} { r } { \frac { 1 } { N _ { S } } \sum _ { i = 1 } ^ { N _ { S } } \delta _ { X _ { i } ^ { ( S ) } } , \widehat { \mathcal { D } _ { X } ^ { T } } = \frac { 1 } { N _ { T } } \sum _ { j = 1 } ^ { N _ { T } } \delta _ { X _ { j } ^ { ( T ) } } } \end{array}$ , their entropic optimal transport is:

$$
\begin{array} { l } { { \displaystyle { \cal W } _ { \beta } \big ( \widehat { \mathcal D _ { X } ^ { S } } , \widehat { \mathcal D _ { X } ^ { T } } \big ) = \operatorname* { m i n } _ { \hat { \gamma } } \langle C , \hat { \gamma } \rangle + \beta \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } \hat { \gamma } _ { i j } \log \hat { \gamma } _ { i j } } } \\ { { \displaystyle ~ + \beta \log \big ( N _ { S } N _ { T } \big ) \quad \mathrm { s . t . } \quad \hat { \gamma } { \bf 1 } = \frac 1 { N _ { S } } { \bf 1 } , \hat { \gamma } ^ { \top } { \bf 1 } = \frac 1 { N _ { T } } { \bf 1 } ( \ S } } \end{array}\tag{}
$$

where $C ~ \in ~ \mathbb { R } _ { + } ^ { N _ { S } \times N _ { T } }$ is the cost matrix with $\begin{array} { r l } { c _ { i j } } & { { } = } \end{array}$ $\rho _ { \mathcal { X } } \big ( X _ { i } ^ { ( S ) } , X _ { j } ^ { ( T ) } \big )$ , and $\hat { \gamma } \in \mathbb { R } _ { + } ^ { N _ { S } \times N _ { T } }$ represents any discretized transport coupling satisfying the given linear constraints. Such an optimization problem can be solved efficiently at a large scale by the Sinkhorn algorithm (Cuturi, 2013; Genevay et al., 2016).

Curse of Dimensionality. However, in modern applications, the covariate space X is often high-dimensional. Even when two distributions are very close, their samples’ distance can be large. Such curse of dimensionality makes the plug-in estimator greatly overestimated (Verleysen & Franc¸ois, 2005; Panaretos & Zemel, 2019) (Fig.1(a)). When $\beta$ is small, entropic optimal transport behaves similarly to Wasserstein distance (Carlier et al., 2017; 2023), and the upward bias decays only in $O ( N ^ { - 1 / d } )$ order (Fournier & Guillin, 2015), which implies that increasing N has a very limited debiasing effect when d is high (Fig.1(b)).

To address the overestimation problem of the traditional plug-in estimator, we propose the following debiased estimator.

Definition 4.1 (Debiased Estimator). Given i.i.d. samples $\{ X _ { i } ^ { ( S ) } \} \sim \mathcal { D } _ { X } ^ { S } , \{ X _ { j } ^ { ( T ) } \} \sim \mathcal { D } _ { X } ^ { T }$ with sample sizes $N _ { S } , N _ { T }$ respectively, split the samples in half to obtain four independent empirical measures: $\begin{array} { r } { \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime } \ = \ \frac { 2 } { N _ { S } } \sum _ { i = 1 } ^ { N _ { S } / 2 } \delta _ { X _ { i } ^ { ( S ) } } } \end{array}$ $\begin{array} { r } { \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime \prime } = \frac { 2 } { N _ { S } } \sum _ { i = N _ { S } / 2 + 1 } ^ { N _ { S } } \delta _ { X _ { i } ^ { ( S ) } } , \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime } = \frac { 2 } { N _ { T } } \sum _ { j = 1 } ^ { N _ { T } / 2 } \delta _ { X _ { j } ^ { ( T ) } } \delta _ { } } \end{array}$ $\begin{array} { r } { \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime \prime } = \frac { 2 } { N _ { T } } \sum _ { j = N _ { T } / 2 + 1 } ^ { N _ { T } } \delta _ { X _ { j } ^ { ( T ) } } } \end{array}$ . The debiased estimator is:

$$
\begin{array} { r l r } & { } & { W _ { \beta } ^ { d e b } \big ( \widehat { \mathcal { D } _ { X } ^ { S } } , \widehat { \mathcal { D } _ { X } ^ { T } } \big ) = \bigg | \frac { 1 } { 2 } W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { S } } , \widehat { \mathcal { D } _ { X } ^ { T } } \big ) ^ { 2 } + \frac { 1 } { 2 } W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime \prime } , \widehat { \mathcal { D } _ { X } ^ { T } } \big ) ^ { 2 } } \\ & { } & { - \frac { 1 } { 2 } W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { S } } , \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime \prime } \big ) ^ { 2 } - \frac { 1 } { 2 } W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { T } } , \widehat { \mathcal { D } _ { X } ^ { T } } \big ) ^ { 2 } \bigg | ^ { 1 / 2 } \quad \quad ( 9 ) } \end{array}
$$

Remark This debiased estimator uses four plug-in estimators. The first two terms, $W _ { \beta } ( \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime } , \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime } )$ and $W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { S } } ^ { \prime \prime } , \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime \prime } \big )$ , estimate the distance between $\mathcal { D } _ { X } ^ { S }$ and $\mathcal { D } _ { X } ^ { T }$ , including the sample bias. And the last two terms, $\overrightarrow { W _ { \beta } } ( \overrightarrow { \mathcal { D } _ { X } ^ { S } } ^ { \prime } , \overbrace { \mathcal { D } _ { X } ^ { S } } ^ { \prime \prime } )$ and $W _ { \beta } \big ( \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime } , \widehat { \mathcal { D } _ { X } ^ { T } } ^ { \prime \prime } \big )$ estimate the distance arising from sample bias only. Subtracting the two parts reduces the sample bias, thus giving a better estimate of $W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ . Compared with the traditional plug-in estimator, our debiased estimator $W _ { \beta } ^ { d e b }$ reduces the overestimation and remains accurate regardless of the true distribution distance (see Fig.1(a) and 1(c)).

![](images/e8343a9ac865e771a75ec65326bd8619e865a72c99d3ee219982faf95402b4a7.jpg)

![](images/749a8aa92cd1a4c56646ba203ba25955ad6ccdfbecc51cbecd5df7596415af07.jpg)  
(a) Empirical distance vs. dimension (d)

![](images/b0509d62f871f4decafeceb81f5e118992c608dabec3db1fb37779e7b9e16869.jpg)  
(b) Empirical distance vs. sample size (N)  
(c) Empirical distance vs. true $W _ { 1 }$  
Figure 1. (a)–(c) show the empirical distance from entropic optimal transport $( \beta = 0 . 0 0 1 )$ versus dimension $d ,$ sample size $N$ , and true Wasserstein-1 distance. In (a)–(b), ηˆ and $\hat { \eta } ^ { \prime }$ are independent empirical measures from high-dimensional standard normals, thus the true distance is zero. The traditional estimator $\dot { W } _ { \beta } ( \boldsymbol { \hat { \eta } } , \boldsymbol { \hat { \eta } ^ { \prime } } )$ greatly overestimates as d increases, as shown in (a), and even a much larger N only brings a small improvement, as shown in (b). In (c), with $d = 7 0 , \hat { \eta } _ { t }$ is the empirical measure of shifted standard normal $\bar { \mathcal { N } } ( t , \bar { I } )$ , whose true Wasserstein-1 distance to the standard normal is ∥t∥. Our debiased estimator remains accurate in every case.

Moreover, when the covariate space is Euclidean, we derived a concentration inequality guaranteeing that the debiased estimator converges to the true distance:

Theorem 4.2 (Concentration Inequality of Debiased Estimator). Let $\mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$ be two distributions on $( \mathbb { R } ^ { d } , \lVert \cdot \rVert )$ withfinite squared-exponential moments. For i.i.d. samples $\{ X _ { i } ^ { ( S ) } \} \sim \mathcal { D } _ { X } ^ { S } , \{ X _ { j } ^ { ( T ) } \} \sim \mathcal { D } _ { X } ^ { T }$ with sample sizes $N _ { S } , N _ { T }$ respectively, when $\begin{array} { r } { \check { \beta } = 0 , } \end{array}$ , and for any $\varepsilon > 0 ,$ , there exists N such that $i f N _ { S } , N _ { T } > N ,$ , then:

$$
\mathbb { P } \Big ( \big | W _ { \beta } ^ { d e b } ( \widehat { \mathcal { D } _ { X } ^ { S } } , \widehat { \mathcal { D } _ { X } ^ { T } } ) - W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } ) \big | > \varepsilon \Big ) \leq
$$

$$
\begin{array} { r } { 2 \exp \left( - \frac { \lambda _ { S } N _ { S } V _ { \varepsilon } \varepsilon ^ { 2 } } { 3 2 } \right) + 2 \exp \left( - \frac { \lambda _ { T } N _ { T } V _ { \varepsilon } \varepsilon ^ { 2 } } { 3 2 } \right) } \end{array}\tag{10}
$$

where $\lambda _ { S } , \lambda _ { T } ~ > ~ 0$ depend only on squared-exponential moments of $\mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$ , respectively, and $V _ { \varepsilon } \in [ 2 - \sqrt { 3 } , 2 )$ depends only on $W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } ) / \varepsilon$

Remark This theorem implies that the deviation probability of the debiased estimator decays exponentially with sample sizes, so with high probability, our debiased estimator can well approximate the true entropic optimal transport distance $W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ . Notably, it also shows the enlightening fact that our estimator’s concentration depends not only on each distribution’s scale characterized by $\lambda _ { S } , \lambda _ { T }$ but also on the true distance between two distributions.

## 4.2. Estimation of Y|X Shift

As above, we defined the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift $S _ { C p t } ^ { \gamma ^ { * } }$ in Definition 3.2; we now estimate it from samples.

Definition 4.3 (Estimator for $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ Shift). Given i.i.d. samples $\{ ( \boldsymbol { X } _ { i } ^ { ( S ) } , \boldsymbol { Y } _ { i } ^ { ( S ) } ) \} ~ \sim ~ \mathcal { D } _ { X Y } ^ { S } , ~ \{ ( \boldsymbol { X } _ { j } ^ { ( T ) } , \boldsymbol { Y } _ { j } ^ { ( T ) } ) \} ~ \sim$ $\mathcal { D } _ { X Y } ^ { T }$ with sample sizes $N _ { S } , N _ { T }$ respectively, the estimator of $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift is defined as:

$$
\hat { S } _ { C p t } = \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } \rho _ { \mathcal { V } } \big ( Y _ { i } ^ { ( S ) } , Y _ { j } ^ { ( T ) } \big ) \hat { \gamma } _ { i j } ^ { * } ,\tag{11}
$$

where $\hat { \gamma } ^ { \ast } \in \mathbb { R } _ { + } ^ { N _ { S } \times N _ { T } }$ represents the discrete optimal transport coupling for $\widehat { W _ { \beta } ( T _ { X } ^ { S } } , \widehat { T _ { X } ^ { T } } )$ .

Remark In Section 4.1, we use the debiased estimator $W _ { \beta } ^ { d e b } ( \widehat { \mathcal { D } _ { X } ^ { S } } , \widehat { \mathcal { D } _ { X } ^ { T } } )$ for X shift, whereas Definition 4.3 still estimates $\gamma ^ { * } { \mathrm { - } } \dot { \Upsilon } | \mathrm { X }$ shift via the optimal transport coupling from the plug-in estimator $\widehat { W _ { \beta } ( D _ { X } ^ { S } } , \widehat { D _ { X } ^ { T } } )$ . This is because the curse of dimensionality mainly affects the transport cost $( \mathrm { i . e . }$ ., inter-sample distances), rather than the transport coupling. Leveraging the stability of entropic optimal transport (Eckstein & Nutz, 2022), the following lemma establishes the convergence of the coupling for plug-in estimator:

Lemma 4.4 (Stability of Entropic Optimal Transport Coupling). Let $\mathcal { D } _ { X } ^ { S } , \bar { \mathcal { D } } _ { X } ^ { T }$ be two distributions on $( \mathcal { X } , \rho _ { \mathcal { X } } )$ with finite squared-exponential moments. $\gamma ^ { * }$ and $\hat { \gamma } ^ { * }$ are the optimal transport couplings of $W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ and $\widehat { W _ { \beta } ( D _ { X } ^ { S } } , \widehat { D _ { X } ^ { T } } )$ respectively, then:

$$
W _ { 1 } \left( \gamma ^ { * } , \hat { \gamma } ^ { * } \right) \leq \Lambda + 2 \sqrt { \frac { 2 } { \beta \lambda _ { \gamma ^ { * } } } } \sqrt { \varLambda }
$$

$$
\varLambda = W _ { 1 } \left( \mathcal { D } _ { X } ^ { S } , \widehat { \mathcal { D } _ { X } ^ { S } } \right) + W _ { 1 } \left( \mathcal { D } _ { X } ^ { T } , \widehat { \mathcal { D } _ { X } ^ { T } } \right)\tag{12}
$$

where $W _ { 1 } \left( \gamma ^ { * } , \hat { \gamma } ^ { * } \right)$ is computed on $\mathcal { X } \times \mathcal { X }$ with metric $\rho \big ( ( x _ { 1 } , x _ { 2 } ) , ( x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime } ) \big ) : = \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 1 } ^ { \prime } ) + \rho _ { \mathcal { X } } ( x _ { 2 } , x _ { 2 } ^ { \prime } ) , \lambda _ { \gamma ^ { * } } >$ 0 depend only on squared-exponential moments of $\mathrm { \dot { \mathcal { D } } } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$

Remark This lemma shows that the Wasserstein-1 distance between the plug-in estimator coupling and the population optimal transport coupling, is bounded by the sum of the marginal Wasserstein-1 distance between each population and its empirical distribution, with the bound scaled by the parameter $\beta .$ Such a uniform bound is guaranteed only when $\beta > 0$ , which further highlights the necessity of using entropic optimal transport.

By Lemma 4.4, when covariate space X is Euclidean and label space $\mathcal { V }$ is a bounded set in an Euclidean space, we derive a concentration inequality guaranteeing estimator in Definition 4.3 approximates true value:

Theorem 4.5 (Concentration Inequality for Definition 4.3). Let $\mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T }$ be two distributions on $( \mathbb { R } ^ { d } , \parallel \cdot \parallel )$ with $f _ { - }$ nite squared-exponential moments. Let the label space $\mathcal { V } \subset \mathbb { R } ^ { d ^ { \prime } }$ be bounded by $\begin{array} { r } { M = \operatorname* { s u p } _ { y , y ^ { \prime } \in \mathcal { Y } } \| y - y ^ { \prime } \| } \end{array}$ , on which conditional distributions $\mathscr { D } _ { Y | X = x _ { S } } ^ { S } , \mathscr { D } _ { Y | X = x _ { T } } ^ { T }$ satisfy $L _ { Y \mid X }$ -Lipschitz respectively: d<sub>TV</sub> $\left( { \mathcal { D } } _ { Y \mid X = x } , { \mathcal { D } } _ { Y \mid X = x ^ { \prime } } \right) \ \leq$ $L _ { Y \mid X } \parallel x - x ^ { \prime } \parallel$ . For i.i.d. samples $\{ ( \boldsymbol { X } _ { i } ^ { ( S ) } , \boldsymbol { Y } _ { i } ^ { ( S ) } ) \} \sim \mathcal { D } _ { X Y } ^ { S }$ $\{ ( \boldsymbol { X } _ { j } ^ { ( T ) } , \boldsymbol { Y } _ { j } ^ { ( T ) } ) \} \sim \mathcal { D } _ { X Y } ^ { T }$ with sample sizes $N _ { S } , N _ { T }$ , when $\beta > 0 ,$ and for any $\varepsilon > 0$ , there exists N such that if $N _ { S } , N _ { T } > N _ { ; }$ , then:

$$
\begin{array} { r l r } & { } & { \mathbb { P } \big ( \vert \hat { S } _ { C p t } - S _ { C p t } ^ { \gamma ^ { * } } - \varDelta \vert > \varepsilon \big ) \leq 2 \exp \biggl ( - \frac { N _ { S } N _ { T } \varPhi \varepsilon ^ { 2 } } { ( N _ { S } + N _ { T } ) M ^ { 2 } } \biggr ) } \\ & { } & { + \exp \biggl ( - \frac { \lambda _ { S } ^ { 1 / 2 } N _ { S } \varPhi \varepsilon ^ { 2 } } { 4 \lambda _ { T } ^ { 1 / 2 } M ^ { 2 } } \biggr ) + \exp \biggl ( - \frac { \lambda _ { T } ^ { 1 / 2 } N _ { T } \varPhi \varepsilon ^ { 2 } } { 4 \lambda _ { S } ^ { 1 / 2 } M ^ { 2 } } \biggr ) ( 1 3 ) } \end{array}
$$

where d<sub>TV</sub> is total-variation distance, $\lambda _ { S } , \lambda _ { T } > 0$ depend only on the squared-exponential moments of $\mathbf { \bar { \mathcal { D } } } _ { X } ^ { S } , \mathbf { \bar { \mathcal { D } } } _ { X } ^ { T } , \mathbf { \bar { \phi } } >$ 0 depends on $L _ { Y | X } , \lambda _ { S } , \lambda _ { T } , \beta ,$ , the bias $\varDelta$ is a constant.

Remark This theorem implies that the deviation probability of the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ Shift estimator $\hat { S } _ { C p t }$ decays exponentially with sample sizes, so with high probability, the estimator can well approximate the $S _ { C p t } ^ { \gamma ^ { * } } + \varDelta$ . The bias term $\varDelta$ arises because Definition 4.3 uses $\rho _ { \mathcal { V } } ( Y _ { i } ^ { ( S ) } , Y _ { j } ^ { ( T ) } )$ as a single-point estimate of $S _ { p a i r } \big ( X _ { i } ^ { ( S ) } , X _ { j } ^ { ( T ) } \big )$ in Definition 3.2. The following two propositions show that the bias ∆ is controlled. Proposition 4.6 (∆ in Deterministic Labeling). Assume deterministic labeling $\mathcal { D } _ { Y | X = x } ^ { S } = \delta _ { f _ { S } ( x ) } , \mathcal { D } _ { Y | X = x } ^ { T } = \delta _ { f _ { T } ( x ) }$ then the $\varDelta$ in Theorem 4.5 satisfies:

$$
\varDelta = 0\tag{14}
$$

Remark This proposition shows that under deterministic labeling, the estimator $\hat { S } _ { C p t }$ accurately estimates the $\mathrm { Y } | \mathrm { X }$ shift $S _ { C p t } ^ { \gamma ^ { * } }$ itself. For stochastic labeling, we show that $\varDelta$ can be bounded by the irreducible error (James et al., 2013) (also known as the Bayes risk (Berger, 2013)) in traditional statistical learning.

Proposition 4.7 (∆ in Stochastic Labeling). When $( \mathcal { V } , \bar { \rho } _ { \mathcal { V } } ) = ( \mathbb { R } ^ { d ^ { \prime } } , \| \cdot \| )$ , the irreducible error ofjoint distribution $\mathcal { D } _ { X Y }$ under squared loss $\| \cdot \| ^ { 2 }$ is defined as: $\begin{array} { r } { \operatorname { I } ( \mathcal { D } _ { X Y } ) = \operatorname* { i n f } _ { g : \mathcal { X } \to \mathcal { Y } } \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { X Y } } \left[ | | y - g ( x ) | | ^ { 2 } \right] } \end{array}$ , then the $\varDelta$ in Theorem 4.5 satisfies:

$$
0 \leq \varDelta \leq \sqrt { \mathrm { I } ( \mathcal { D } _ { X Y } ^ { S } ) } + \sqrt { \mathrm { I } ( \mathcal { D } _ { X Y } ^ { T } ) }\tag{15}
$$

Remark The irreducible error is the fundamental error inherent to stochastic labeling that no model can overcome. When the problem is learnable, covariate and label are often well correlated, and the irreducible error is small relative to the overall label variability. In this case, Proposition 4.7 guarantees that the estimator $\hat { S } _ { C p t }$ does not substantially overestimate the Y|X shift $S _ { C p t } ^ { \gamma ^ { * } }$

## 4.3. DataShifts Algorithm

On the Lipschitz Constant of Learners. The Lipschitz constant of the learner have been well studied, such as logistic regression (Roux et al., 2012), multi-layer perceptron (MLP) (Fazlyab et al., 2019), convolutional neural networks (CNN) (Virmaux & Scaman, 2018; Zou et al., 2019) and attention mechanism (Kim et al., 2021; Castin et al., 2023). Although the Lipschitz constants of modern large-scale neural networks such as ResNet-50 or Transformers remain difficult to analyze, researchers are more concerned with whether these models learn distribution-robust representations (Hendrycks et al., 2020), rather than distribution shift in the raw input space. In this setting, the covariate space X is the representation space, and the downstream model (the hypothesis h in our theory) is often a simple learner—such as a linear classifier—whose Lipschitz constant is still easy to handle. We summarize the Lipschitz constants of various learners in Appendix D.

Algorithm 1 DataShifts   
Input: hyperparameter $\beta$ (default 0.01),   
samples $\{ ( \boldsymbol { X } _ { i } ^ { ( S ) } , \boldsymbol { Y } _ { i } ^ { ( S ) } ) \} , \{ ( \boldsymbol { X } _ { j } ^ { ( T ) } , \boldsymbol { Y } _ { j } ^ { ( T ) } ) \}$   
Lipschitz constants $L _ { \ell } , L _ { \ell } ^ { \prime } , \bar { L _ { h } }$ (optional),   
source domain empirical error $\hat { \epsilon } _ { S }$ (optional)   
Do:   
Estimate X shift by 4.1 as $\hat { S } _ { C o v }$   
Estimate $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift by 4.3 as $\hat { S } _ { C p t }$   
if $L _ { \ell } , L _ { \ell } ^ { \prime } , L _ { h }$ , and $\hat { \epsilon } _ { S }$ are provided then   
Estimate bound: $B = \hat { \epsilon } _ { S } + L _ { h } L _ { \ell } ^ { \prime } \hat { S } _ { C o v } + L _ { \ell } \hat { S } _ { C p t }$   
end if   
Return: $\hat { S } _ { C o v } , \hat { S } _ { C p t }$ and B (optional)

![](images/2a18c912d6b7260d1feee81611fb689baf586754edb637b99dc9de7bf09ca838.jpg)  
(a) Novozymes

![](images/7abf33994afd4ceae63db3f6ecacb7306ecd79173d9dc6a320fa80ba31c41678.jpg)  
(b) ColoredMNIST

![](images/270bb0099eca53c233b330c3d6b807b1a723b9b82aeaa6a5ecbe311f913bcd8b.jpg)  
(c) PACS  
Figure 2. (a)–(c) show that the estimated error bounds track the test error well across three distinct tasks, corroborating the general effectiveness of our learning bound and estimators.

By leveraging the theoretical results above, we give the plugand-play Algorithm 1 (DataShifts) for quantifying X and Y|X shifts from finite samples and estimating error bounds.

## 5. Experiments

In this section, we apply our estimable general theory to three distinct practical tasks: tabular regression, image binary classification, and image multi-class classification – to validate the general effectiveness of our bound and estimators. We also conduct experiments on synthetic tasks where existing theory apply, demonstrating our bound is tighter.

## 5.1. Novozymes Enzyme Stability Prediction

The Novozymes Enzyme Prediction Competition (Pultz et al., 2022) is a large-scale Kaggle contest. It is a tabular regression task, with 9,000 point-mutation samples spanning 180 enzyme families, aiming to predict transition temperatures for unseen families. Each enzyme family is treated as a separate domain; due to distribution shifts between them, thousands of participants found it difficult to develop any effective solution.

We select the 60 enzyme families with the smallest pretraining error as the source domain, and treat each of the remaining 120 enzyme families as a target domain. In this task, we focus on the distribution shift between the raw data of different enzyme families, taking the 20-dimensional input feature space as covariate space X. We train a 3-layer MLP on the source domain; an analysis of its Lipschitz constant is provided in the Appendix D.4. We use the absolute loss (separately (1, 1)-Lipschitz) to evaluate the source domain error and the test error on each target domain, and apply our DataShifts algorithm $\beta = 0 . 2 )$ to estimate an error bound for each target domain.

We plot the test error and the estimated error bound on each target domain in Fig. 2(a). In this figure, the overall trend of the test error and the error bound lies just above the diagonal, indicating that our bound is tight and effectively captures the test error under distribution shift. Meanwhile, it directly shows the contributions of X and Y|X shifts on the error bound. The large Y|X shift across enzyme families is what drives the generalization failure in this contest.

## 5.2. ColoredMNIST and PACS

ColoredMNIST and PACS are two standard tasks in the DomainBed benchmark(Gulrajani & Lopez-Paz, 2020). ColoredMNIST is a binary classification task with 70,000 handwritten digit images exhibiting color shift, while PACS is a multi-class object recognition task with 9,991 images exhibiting style shift. For both tasks, we treat the model’s representation space as the covariate space X . Following the DomainBed setup, we use the simple CNN with a 128- dimensional representation for ColoredMNIST, and ResNet-50 with a 2048-dimensional representation for PACS. In this setting, the hypothesis h in our theory corresponds to the model’s final layer (a linear classifier), whose Lipschitz constant is analyzed in the Appendix D.3.

We evaluate three methods: ERM, CORAL, and MMD. Adhering to DomainBed, we select the best hyperparameters via training-domain validation over 20 random hyperparameter trials for each method, domain, and trial. For ColoredM-NIST, we train each best-hyperparameter run for 5,000 steps and save checkpoints every 100 steps. For PACS, since the model converges earlier, we train for 1,000 steps and save checkpoints every 20 steps. At each checkpoint, we regard the mixture distribution over training domains as the source domain and the test domain as the target domain. We still use the absolute loss to measure source and target errors, and run our DataShifts algorithm $( \beta = 0 . 2 )$ using each domain’s representation-label pairs to estimate the test domain error bound at every checkpoint.

![](images/e33cf0118265ad17c399e402cea70f6b0a97607df0009695306f7ba64bab5106.jpg)  
(a) Error vs. X shift

![](images/6fc7147e55fe1e9993d60a4200bd00abb15e42a856bc5ff875aaee7c5a021e4c.jpg)  
(b) Error vs. Y|X shift (θ)

![](images/6356494286e843c78f48d1e4a47ab0f4546ba827f9ff190e261acb38ab56dbdc.jpg)  
(c) Error vs. Y|X shift (b)  
Figure 3. (a)–(c) respectively show the relationship between the estimated learning bounds and the X or Y|X shift in synthetic binary classification tasks. As these shifts increase, our bound becomes significantly tighter than the existing bound.

We plot the test error and the estimated error bound for both tasks in Fig. 2(b) and 2(c). In Fig. 2(b), the bound tracks the test error well across checkpoints. The MMD (purple) and CORAL (yellow) points lie closer to the lower-left region, indicating that these X shift-reducing methods indeed yield smaller bounds, and consequently lower error on ColoredM-NIST. In Fig. 2(c), although the bound becomes looser in magnitude when applied to PACS, a more complex image classification task, it still exhibits a consistent trend with the test error in each run.

## 5.3. Synthetic Binary Classification

We compare our theory with the existing bound in Zhao et al. (2019), which is shown to be tighter than previous learning bounds. As discussed in Remark\*, the existing learning bound only applies to soft-label binary classification with deterministic labeling and absolute loss. Since its estimation requires sampling the true concepts $f _ { S }$ and $f _ { T }$ outside the supports of the covariate distributions $\mathcal { D } _ { X } ^ { S } , ~ \mathcal { D } _ { X } ^ { T }$ (oracle), it can only be estimated on synthetic tasks where the true concepts are known.

We construct a synthetic binary classification task using logistic regression. Let the covariate space be 10-dimensional, and the source inputs are sampled from standard normal distribution: $\mathcal { D } _ { X } ^ { S } ~ = ~ \mathcal { N } ( \mathbf { 0 } , I )$ , with labels generated by $f _ { S } ( x ) = \sigma ( w _ { S } ^ { \top } x + b _ { S } )$ , where $w _ { S } = \mathbf { 1 } / \sqrt { 1 0 } , b _ { S } = 0$ and σ is the sigmoid function. The target inputs are generated by shifting $\mathcal { D } _ { X } ^ { S }$ along a random direction, thereby controlling the X shift. And the target labels are generated by another logistic regression: $f _ { T } ( x ) = \sigma ( w _ { T } ^ { \top } x + b _ { T } )$ , where w is obtained by rotating w<sub>S</sub> by angle θ toward a random direction. θ and $b _ { T }$ further control the Y|X shift. By varying the X shift, θ, and $b _ { T }$ , we obtain a series of target domains.

We also use logistic regression as the learner and train it on the source domain. Its Lipschitz constant is analyzed in Appendix D.2. For each target domain, we estimate the test error, the existing and our bounds under the absolute loss.

We plot the learning bounds with respect to the X shift and the Y|X shifts (θ and $b _ { T } )$ in Fig. 3. In Fig. 3(a), since both the source concept and the learner are logistic regression models, the learner can fit the source concept well, and the X shift in the target domain has only a minor effect on the test error. As the X shift increases, our bound becomes tighter than the existing bound. In Figs. 3(b) and 3(c), as the $\mathrm { Y } | \mathrm { X }$ shift increases, the existing bound becomes looser than ours. This verifies our point in Remark\*: the ill-defined Y|X shift in the existing bound is loose, whereas our $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift and the accompanying learning bound can be tighter.

In addition to the above results, sensitivity experiments of the proposed estimators with respect to the parameter $\beta$ are provided in Appendix E.1. Experiments on the bias of the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift estimator under stochastic labeling are presented in Appendix E.2.

## 6. Conclusion

In this paper, we focus on a general and estimable theoretical framework for learning under distribution shift. We first introduce a key notion, $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift, via entropic optimal transport, which addresses the ill-definedness in existing theory. Then we derive a general learning bound unifying X shift and $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift. We further develop concentrationguaranteed estimators for both shifts, and integrate our theory into the plug-and-play DataShifts algorithm, enabling researchers to quantify and analyze distribution shift in broad settings. Experiments on practical and synthetic tasks validate the general effectiveness and tightness of our theory. We believe our theoretical framework takes an important step toward learning under distribution shift and will spur further algorithmic advances.

## Acknowledgements

We also thank Dr. Jie Ren for suggestions and feedback on this work. This study was funded by the National Natural Science Foundation of China (12571529) and Guangdong Basic and Applied Basic Research Foundation (2024A1515- 010699) to LCX.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Ben-David, S., Blitzer, J., Crammer, K., and Pereira, F. Analysis of representations for domain adaptation. Advances in neural information processing systems, 19, 2006.

Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., and Vaughan, J. W. A theory of learning from different domains. Machine learning, 79:151–175, 2010.

Berger, J. O. Statistical decision theory and Bayesian analysis. Springer Science & Business Media, 2013.

Bolley, F., Guillin, A., and Villani, C. Quantitative concentration inequalities for empirical measures on non-compact spaces. Probability Theory and Related Fields, 137(3–4):541–593, 2007. doi: 10.1007/ s00440-006-0004-7. URL https://doi.org/10. 1007/s00440-006-0004-7.

Bonet, C., Vauthier, C., and Korba, A. Flowing datasets with wasserstein over wasserstein gradient flows. arXiv preprint arXiv:2506.07534, 2025.

Boyd, S. P. and Vandenberghe, L. Convex optimization. Cambridge university press, 2004.

Carlier, G., Duval, V., Peyre, G., and Schmitzer, B. Con-´ vergence of entropic schemes for optimal transport and gradient flows. SIAM Journal on Mathematical Analysis, 49(2):1385–1418, 2017.

Carlier, G., Pegon, P., and Tamanini, L. Convergence rate of general entropic optimal transport costs. Calculus of Variations and Partial Differential Equations, 62(4):116, 2023.

Castin, V., Ablin, P., and Peyre, G. How smooth is attention?´ arXiv preprint arXiv:2312.14820, 2023.

Cole, S. R. and Frangakis, C. E. The consistency statement in causal inference: a definition or an assumption? Epidemiology, 20(1):3–5, 2009. doi: 10.1097/ EDE.0b013e31818ef366. URL https://doi.org/ 10.1097/EDE.0b013e31818ef366.

Courty, N., Flamary, R., Habrard, A., and Rakotomamonjy, A. Joint distribution optimal transportation for domain adaptation. Advances in neural information processing systems, 30, 2017.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

Duncan, T. E. On the absolute continuity of measures. The Annals ofMathematical Statistics, 41(1):30–38, 1970.

Eckstein, S. and Nutz, M. Quantitative stability of regularized optimal transport and convergence of sinkhorn’s algorithm. SIAM Journal on Mathematical Analysis, 54 (6):5922–5948, 2022.

El Hamri, M., Bennani, Y., and Falih, I. Theoretical guarantees for domain adaptation with hierarchical optimal transport. Machine Learning, 114(5):119, 2025.

Fazlyab, M., Robey, A., Hassani, H., Morari, M., and Pappas, G. Efficient and accurate estimation of lipschitz constants for deep neural networks. Advances in neural information processing systems, 32, 2019.

Fazlyab, M., Morari, M., and Pappas, G. J. Safety verification and robustness analysis of neural networks via quadratic constraints and semidefinite programming. IEEE Transactions on Automatic Control, 67(1):1–15, 2020.

Fournier, N. and Guillin, A. On the rate of convergence in wasserstein distance of the empirical measure. Probability theory and relatedfields, 162(3):707–738, 2015.

Gangbo, W. and McCann, R. J. The geometry of optimal transportation. 1996.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., and Lempitsky, V. Domainadversarial training of neural networks. Journal of machine learning research, 17(59):1–35, 2016.

Gelman, A. and Meng, X.-L. Simulating normalizing constants: From importance sampling to bridge sampling to path sampling. Statistical science, pp. 163–185, 1998.

Genevay, A., Cuturi, M., Peyre, G., and Bach, F. Stochastic´ optimization for large-scale optimal transport. Advances in neural information processing systems, 29, 2016.

Glorot, X., Bordes, A., and Bengio, Y. Domain adaptation for large-scale sentiment classification: A deep learning approach. In Proceedings ofthe 28th international conference on machine learning (ICML-11), pp. 513–520, 2011.

Gulrajani, I. and Lopez-Paz, D. In search of lost domain generalization. arXiv preprint arXiv:2007.01434, 2020.

Hajek, A. What conditional probability could not be. ´ Synthese, 137(3):273–323, 2003.

Hendrycks, D., Liu, X., Wallace, E., Dziedzic, A., Krishnan, R., and Song, D. Pretrained transformers improve out-ofdistribution robustness. arXiv preprint arXiv:2004.06100, 2020.

James, G., Witten, D., Hastie, T., Tibshirani, R., et al. An introduction to statistical learning, volume 112. Springer, 2013.

Kallenberg, O. and Kallenberg, O. Foundations of modern probability, volume 2. Springer, 1997.

Kim, H., Papamakarios, G., and Mnih, A. The lipschitz constant of self-attention. In International Conference on Machine Learning, pp. 5562–5571. PMLR, 2021.

Krueger, D., Caballero, E., Jacobsen, J.-H., Zhang, A., Binas, J., Zhang, D., Le Priol, R., and Courville, A. Out-ofdistribution generalization via risk extrapolation (rex). In International conference on machine learning, pp. 5815– 5826. PMLR, 2021.

Liu, J., Shen, Z., He, Y., Zhang, X., Xu, R., Yu, H., and Cui, P. Towards out-of-distribution generalization: A survey. arXiv preprint arXiv:2108.13624, 2021.

Long, M., Cao, Y., Wang, J., and Jordan, M. Learning transferable features with deep adaptation networks. In International conference on machine learning, pp. 97– 105. PMLR, 2015.

Moreno-Torres, J. G., Raeder, T., Alaiz-Rodr´ıguez, R., Chawla, N. V., and Herrera, F. A unifying view on dataset shift in classification. Pattern recognition, 45(1):521–530, 2012.

Panaretos, V. M. and Zemel, Y. Statistical aspects of wasserstein distances. Annual review of statistics and its application, 6(1):405–431, 2019.

Pei, Z., Cao, Z., Long, M., and Wang, J. Multi-adversarial domain adaptation. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Pultz, D., Friis, E., Salomon, J., Hallin, P. F., Jørgensen, S. B., Reade, W., and Demkin, M. Novozymes enzyme stability prediction. https://kaggle.com/competitions/ novozymes-enzyme-stability-prediction, 2022. Kaggle.

Rame, A., Dancette, C., and Cord, M. Fishr: Invariant gradient variances for out-of-distribution generalization. In International Conference on Machine Learning, pp. 18347–18377. PMLR, 2022.

Redko, I., Habrard, A., and Sebban, M. Theoretical analysis of domain adaptation with optimal transport. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 737–753. Springer, 2017.

Roux, N., Schmidt, M., and Bach, F. A stochastic gradient method with an exponential convergence rate for finite training sets. Advances in neural information processing systems, 25, 2012.

Shen, J., Qu, Y., Zhang, W., and Yu, Y. Wasserstein distance guided representation learning for domain adaptation. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Sugiyama, M., Krauledat, M., and Muller, K.-R. Covariate¨ shift adaptation by importance weighted cross validation. Journal ofMachine Learning Research, 8(5), 2007.

Sun, B. and Saenko, K. Deep coral: Correlation alignment for deep domain adaptation. In Computer vision–ECCV 2016 workshops: Amsterdam, the Netherlands, October 8-10 and 15-16, 2016, proceedings, part III 14, pp. 443– 450. Springer, 2016.

Verleysen, M. and Franc¸ois, D. The curse of dimensionality in data mining and time series prediction. In International work-conference on artificial neural networks, pp. 758– 770. Springer, 2005.

Virmaux, A. and Scaman, K. Lipschitz regularity of deep neural networks: analysis and efficient estimation. Advances in Neural Information Processing Systems, 31, 2018.

Zhang, X., He, Y., Xu, R., Yu, H., Shen, Z., and Cui, P. Nico++: Towards better benchmarking for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16036– 16047, 2023.

Zhang, Y., Liu, T., Long, M., and Jordan, M. Bridging theory and algorithm for domain adaptation. In International conference on machine learning, pp. 7404–7413. PMLR, 2019.

Zhang, Y., Deng, B., Tang, H., Zhang, L., and Jia, K. Unsupervised multi-class domain adaptation: Theory, algorithms, and practice. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2775–2792, 2020.

Zhao, H., Des Combes, R. T., Zhang, K., and Gordon, G. On learning invariant representations for domain adaptation. In International conference on machine learning, pp. 7523–7532. PMLR, 2019.

Zou, D., Balan, R., and Singh, M. On lipschitz bounds of general convolutional neural networks. IEEE Transactions on Information Theory, 66(3):1738–1759, 2019.

## A. Related Work

Generalization theory under distribution shift is an important and long-standing problem. Researchers usually seek to measure distribution shift using some distribution divergence and derive corresponding learning bounds. Early theoretical works mainly focused on the X (covariate) shift. Ben-David et al. (2006; 2010) used the H-divergence to measure X shift between the source and target domains for binary classification, thereby deriving learning bounds for domain adaptation. Subsequent studies characterized the X shift via maximum mean discrepancy (MMD), leading to new learning bounds and improved domain adaptation algorithms (Long et al., 2015). Redko et al. (2017); Courty et al. (2017); Shen et al. (2018) adopted optimal transport (OT) to characterize X shift and obtained similar bounds. Further studies proposed more complex metrics for the X shift to extend the theory to multiclass classification (Zhang et al., 2019; 2020; El Hamri et al., 2025).

Most of these theories focus only on the X shift and follow a similar proof strategy, resulting in a joint error term between the source and target domains, $\lambda = \operatorname* { m i n } _ { h } \epsilon _ { S } ( h ) + \epsilon _ { T } ( h )$ , which is widely recognized as loose and non-estimable. Our baseline, Zhao et al. (2019), showed that the Y|X (concept) shift is implicitly contained in this joint error term λ, and derived a tighter learning bound by explicitly introducing the Y|X shift instead. Zhang et al. (2023) extended this theory to multiclass classification. We follow this path of explicitly characterizing the Y|X shift, and point out that the existing definition of the Y|X shift becomes ill-defined when the supports of the covariate distributions are mismatched, which still makes the Y|X shift loose and non-estimable. Through entropic optimal transport, we introduce a well-defined $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift and the corresponding learning bound, which can be rigorously estimated from samples. Our theory can be tighter than existing bounds and further generalizes to a broader tasks, losses, and stochastic labeling.

Table 1. Comparison of learning bounds for distribution shift.
<table><tr><td rowspan="2">Bound</td><td rowspan="2">Divergence</td><td colspan="3">Terms</td><td colspan="3">Properties</td></tr><tr><td>X shift</td><td>Y|X shift</td><td>Remaining Term</td><td>Tight</td><td>Estimable General</td><td></td></tr><tr><td>Ben-David et al. (2006)</td><td>H-divergence</td><td>√</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Ben-David et al. (2010)</td><td>H-divergence</td><td>√</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Long et al. (2015)</td><td>MMD</td><td>V</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Redko et al. (2017)</td><td>OT</td><td>√</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Courty et al. (2017)</td><td>OT</td><td></td><td>estimated XY shift</td><td>joint-error λ, kMΦ</td><td></td><td></td><td>√</td></tr><tr><td>Shen et al. (2018)</td><td>OT</td><td>√</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Zhao et al. (2019)</td><td>H-divergence</td><td>√</td><td>√(ill-defined)</td><td></td><td>√</td><td></td><td></td></tr><tr><td>Zhang et al. (2023)</td><td>H-divergence</td><td>√</td><td>√(ill-defined)</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>El Hamri et al. (2025)</td><td>hierarchical OT</td><td>√</td><td></td><td>joint-error λ</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>entropic OT</td><td>√</td><td>√</td><td></td><td>√√</td><td>√</td><td>√√</td></tr></table>

✓✓indicates a stronger degree than ✓.

Distribution shift theory often serves as a theoretical framework for domain adaptation and domain generalization, where many algorithms have been developed (Sugiyama et al., 2007; Glorot et al., 2011; Sun & Saenko, 2016; Ganin et al., 2016; Pei et al., 2018; Arjovsky et al., 2019; Krueger et al., 2021; Rame et al., 2022). However, the DomainBed benchmark (Gulrajani & Lopez-Paz, 2020) shows that many algorithms do not outperform standard empirical risk minimization. This calls for tighter and more general theoretical frameworks to explain the generalization failures of these algorithms. In addition, the proposed $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift takes a nested OT form, whose other mathematical properties have also been studied recently, such as the gradient flow of the Wasserstein-over-Wasserstein distance (Bonet et al., 2025).

## B. Technical Tools

In this section, we introduce several existing mathematical tools to prove the propositions in Appendix C.

## B.1. Conditional Probability

Definition B.1 (Conditional Probability). Let $( \Omega , { \mathcal { F } } , P )$ be a probability space and $X : \Omega \to \mathcal { X }$ a random variable with distribution $P _ { X }$ . For event $A \in { \mathcal { F } } , \mathtt { a } P _ { X }$ -measurable function $P ( A \mid X = x ) : { \mathcal { X } } \to [ 0 , 1 ]$ is the conditional probability of

A given X if:

$$
\forall B \in { \mathcal { B } } _ { \mathcal { X } } , P ( A \cap \{ X \in B \} ) = \int _ { B } P ( A \mid X = x ) d P _ { X } ( x )
$$

where $B _ { \mathcal { X } }$ denotes the Borel σ-algebra on $\mathcal { X } .$

The conditional probability is unique almost everywhere with respect to $P _ { X }$ (unique $P _ { X ^ { - } \mathbf { a } . \mathbf { e } . ) }$ (Kallenberg & Kallenberg, 1997). That is, for any $P _ { X } \mathrm { - n u l l }$ set $Z$ with $\begin{array} { r } { P _ { X } ( Z ) = 0 , \int _ { Z } P ( A \mid X = x ) d P _ { X } ( x ) = 0 } \end{array}$ , implying that $P ( A \mid X = x )$ can be assigned arbitrarily on $Z$ without affecting its overall properties. Intuitively, it means discussing conditional probabilities on a null set is meaningless (Hajek´ , 2003).

## B.2. McDiarmid’s Inequality

Lemma B.2 (McDiarmid’s Inequality). Let $Z _ { 1 } , \ldots , Z _ { n }$ be independent random variables taking values in sets $\mathcal { Z } _ { 1 } , \ldots , \mathcal { Z } _ { n } ,$ and let $F : { \mathcal { Z } } _ { 1 } \times \cdots \times { \mathcal { Z } } _ { n } \to \mathbb { R } . A$ ssume F satisfies the bounded-differences property: there exist constants $c _ { 1 } , \ldots , c _ { n } \geq 0$ such thatfor every $i \in [ n ]$ and any two inputs $\left( z _ { 1 } , \ldots , z _ { n } \right)$ and $( z _ { 1 } , \ldots , z _ { i } ^ { \prime } , \ldots , z _ { n } )$ differing only at the i-th coordinate

$$
\left| F ( z _ { 1 } , \dots , z _ { i } , \dots , z _ { n } ) - F ( z _ { 1 } , \dots , z _ { i } ^ { \prime } , \dots , z _ { n } ) \right| \ \leq \ c _ { i } .
$$

Then for any $\varepsilon > 0 ,$

$$
\mathbb { P } \Big ( F ( Z _ { 1 } , \ldots , Z _ { n } ) - \mathbb { E } \big [ F ( Z _ { 1 } , \ldots , Z _ { n } ) \big ] \geq \varepsilon \Big ) \ \leq \ \exp \left( - \frac { 2 \varepsilon ^ { 2 } } { \sum _ { i = 1 } ^ { n } c _ { i } ^ { 2 } } \right) ,
$$

and

$$
\mathbb { P } \Big ( \big | F ( Z _ { 1 } , \dots , Z _ { n } ) - \mathbb { E } [ F ( Z _ { 1 } , \dots , Z _ { n } ) ] \big | \geq \varepsilon \Big ) \ \leq \ 2 \exp \Bigg ( { - } \frac { 2 \varepsilon ^ { 2 } } { \sum _ { i = 1 } ^ { n } c _ { i } ^ { 2 } } \Bigg ) .
$$

## B.3. Concentration for Empirical $W _ { 1 }$

Lemma B.3 (Square-Exponential Moment Implies $T _ { 1 }$ (Bolley et al., 2007)). Let $( \mathcal { X } , \rho )$ be a Polish metric space and $\mu$ a Borel probability measure on $\mathcal { X } .$ Assume $\mu$ admits a square-exponential moment: there exist $a > 0$ and $x _ { 0 } \in \mathcal { X }$ such that

$$
\int _ { \mathcal { X } } \exp \bigl ( a \rho ( x , x _ { 0 } ) ^ { 2 } \bigr ) d \mu ( x ) < \infty .
$$

Then $\mu$ satisfies a $T _ { 1 } ( \lambda )$ transport-entropy inequality for some $\lambda > 0 ;$ for all probability measures ν on $x ,$

$$
W _ { 1 } ( \mu , \nu ) \leq \sqrt { \frac { 2 } { \lambda } H ( \nu \mid \mu ) } .
$$

Lemma B.4 (Bolley–Guillin Concentration for Empirical $W _ { 1 }$ (Bolley et al., 2007)). Let $\mu$ be a probability measure on $( \mathbb { R } ^ { d } , \lVert \cdot \rVert )$ that satisfies the transport inequality $T _ { 1 } ( \lambda )$ for some $\lambda > 0 ,$ , namelyfor all probability measures $\nu ,$

$$
W _ { 1 } ( \mu , \nu ) \leq \sqrt { \frac { 2 } { \lambda } H ( \nu \mid \mu ) } ,
$$

where $H ( \nu \mid \mu )$ denotes the relative entropy. Let $\begin{array} { r } { \hat { \mu } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } } } \end{array}$ be the empirical measure ofi.i.d. samples $X _ { 1 } , \dots , X _ { N } \sim$ $\mu .$ Thenfor any $d ^ { \prime } > d$ and any $\lambda _ { \mu } \in ( 0 , \lambda )$ , there exists a constant $N _ { 0 }$ depending only on $\lambda _ { \mu } , d ^ { \prime }$ , and the squared-exponentia moments of µ such that for any $\varepsilon > 0$ and any

$$
N \geq N _ { 0 } \operatorname* { m a x } \bigl ( \varepsilon ^ { - ( d ^ { \prime } + 2 ) } , 1 \bigr ) ,
$$

we have

$$
\mathbb { P } \big ( W _ { 1 } ( \mu , \hat { \mu } ) > \varepsilon \big ) \ \le \ \exp \biggl ( - \frac { \lambda _ { \mu } } { 2 } N \varepsilon ^ { 2 } \biggr ) .
$$

## C. Full proofs

## C.1. Proof of Lemma 2.2

Proof. Since supp $( P _ { X } ^ { \prime } ) \setminus \operatorname { s u p p } ( P _ { X } ) \neq \emptyset .$ pick $x _ { 0 } \in \mathrm { s u p p } ( P _ { X } ^ { \prime } ) \setminus \mathrm { s u p p } ( P _ { X } )$ . By Definition $2 . 1 , x _ { 0 } \notin \mathrm { s u p p } ( P _ { X } )$ implies that there exists an open neighborhood $U \subseteq { \mathcal { X } }$ with $x _ { 0 } \in U$ such that

$$
P _ { X } ( U ) = 0 .
$$

On the other hand, $x _ { 0 } \in \mathrm { s u p p } ( P _ { X } ^ { \prime } )$ implies $P _ { X } ^ { \prime } ( V ) > 0$ for every open neighborhood $V ~ \ni ~ x _ { 0 }$ , hence in particular $P _ { X } ^ { \prime } ( U ) > 0$ . Let $Z : = U$ . Then $Z \in B _ { \mathcal { X } }$ and

$$
P _ { X } ( Z ) = 0 , \qquad P _ { X } ^ { \prime } ( Z ) > 0 .
$$

Let $P ( A \mid X = x ) : { \mathcal { X } } \to [ 0 , 1 ]$ be a conditional probability in Definition B.1, i.e., for all $B \in B _ { \mathcal { X } }$

$$
P { \big ( } A \cap \{ X \in B \} { \big ) } = \int _ { B } P ( A \mid X = x ) d P _ { X } ( x ) .
$$

For any constant $c \in [ 0 , 1 ]$ , define a modified function

$$
{ \widetilde { P } } ( A \mid X = x ) : = { \left\{ { P } ( A \mid X = x ) , \begin{array} { l l } { x \not \in Z , } \\ { c , } & { x \not \in Z . } \end{array} \right.}  
$$

For any $B \in B _ { \mathcal { X } }$

$$
\begin{array} { l } { \displaystyle \int _ { B } \widetilde P ( A \mid X = x ) d P _ { X } ( x ) = \int _ { B \setminus Z } P ( A \mid X = x ) d P _ { X } ( x ) + \int _ { B \cap Z } c d P _ { X } ( x ) } \\ { \displaystyle \qquad = \int _ { B \setminus Z } P ( A \mid X = x ) d P _ { X } ( x ) + c P _ { X } ( B \cap Z ) } \\ { \displaystyle \qquad = \int _ { B \setminus Z } P ( A \mid X = x ) d P _ { X } ( x ) \qquad ( \mathrm { s i n c e } \ P _ { X } ( Z ) = 0 ) } \\ { \displaystyle \qquad = \int _ { B } P ( A \mid X = x ) d P _ { X } ( x ) = P \big ( A \cap \{ X \in B \} \big ) , } \end{array}
$$

so ${ \tilde { P } } ( A \mid X = x )$ is also a valid conditional probability of A given X by Definition B.1, as another version of $P ( A \mid X = x )$ Now take expectation under $P _ { X } ^ { \prime }$ :

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { P _ { X } ^ { \prime } } [ \widetilde { P } ( A \mid X = x ) ] = \int _ { \mathcal { X } } \widetilde { P } ( A \mid X = x ) d P _ { X } ^ { \prime } ( x ) } \\ { \displaystyle = \int _ { { \mathcal { X } \backslash Z } } P ( A \mid X = x ) d P _ { X } ^ { \prime } ( x ) + \int _ { Z } c d P _ { X } ^ { \prime } ( x ) } \\ { \displaystyle = \int _ { { \mathcal { X } \backslash Z } } P ( A \mid X = x ) d P _ { X } ^ { \prime } ( x ) + c P _ { X } ^ { \prime } ( Z ) . } \end{array}
$$

Since $P _ { X } ^ { \prime } ( Z ) > 0$ and $c \in [ 0 , 1 ]$ is arbitrary, the value of E ${ } _ { P _ { \mathbf { v } } ^ { \prime } } [ P ( A \mid X = x ) ]$ can be changed by choosing different c while preserving the defining property of conditional probability under $P _ { X }$ . Hence $\mathbb { E } _ { P _ { X } ^ { \prime } } [ P ( A \mid X = x ) ]$ is arbitrary. □

Remark The above proof shows that support mismatch leads to arbitrariness. Broadly speaking, support mismatch commonly breaks the rigor of theoretical analyses. Some studies explicitly exclude it from analysis by extra assumptions, such as done in the absolute continuity in measure theory (Duncan, 1970), the positivity assumption in causal inference (Cole & Frangakis, 2009), or support overlap in importance sampling (Gelman & Meng, 1998).

## C.2. Proof of Corollary 3.3

Proof. By Definition 3.2,

$$
S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { ( \boldsymbol { x } _ { S } , \boldsymbol { x } _ { T } ) \sim \gamma ^ { * } } \left[ W _ { 1 } \big ( \mathcal { D } _ { Y | X = \boldsymbol { x } _ { S } } ^ { S } , \mathcal { D } _ { Y | X = \boldsymbol { x } _ { T } } ^ { T } \big ) \right] .
$$

Under deterministic labeling, $\mathcal { D } _ { Y | X = x _ { S } } ^ { S } = \delta _ { f _ { S } ( x _ { S } ) }$ and $\mathcal { D } _ { Y | X = x _ { T } } ^ { T } = \delta _ { f _ { T } ( x _ { T } ) }$ , hence

$$
S _ { p a i r } ( x _ { S } , x _ { T } ) = W _ { 1 } \bigl ( \delta _ { f _ { S } ( x _ { S } ) } , \delta _ { f _ { T } ( x _ { T } ) } \bigr ) .
$$

For any $a , b \in \mathcal { D }$ , any coupling $\pi \in \Gamma ( \delta _ { a } , \delta _ { b } )$ must satisfy $\pi ( \{ a \} \times \mathcal { Y } ) = 1$ and $\pi ( \mathcal { V } \times \{ b \} ) = 1$ , hence $\pi = \delta _ { ( a , b ) }$ is the unique coupling. Therefore,

$$
W _ { 1 } ( \delta _ { a } , \delta _ { b } ) = \operatorname* { i n f } _ { \pi \in \Gamma ( \delta _ { a } , \delta _ { b } ) } \int \rho _ { \mathcal { V } } ( y _ { 1 } , y _ { 2 } ) d \pi = \int \rho _ { \mathcal { V } } ( y _ { 1 } , y _ { 2 } ) d \delta _ { ( a , b ) } ( y _ { 1 } , y _ { 2 } ) = \rho _ { \mathcal { V } } ( a , b ) .
$$

Applying this with $a = f _ { S } ( x _ { S } )$ and $b = f _ { T } ( x _ { T } )$ yields $S _ { p a i r } ( x _ { S } , x _ { T } ) = \rho _ { \mathcal V } ( f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } ) )$ , and thus

$$
S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \left[ \rho _ { \mathcal { V } } ( f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } ) ) \right] ,
$$

as claimed.

## C.3. Proof of Lemma 3.4

Proof. Any $\gamma \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ is a coupling of $\mathcal { D } _ { X } ^ { S }$ and $\mathcal { D } _ { X } ^ { T }$ , hence its first marginal is $\mathcal { D } _ { X } ^ { S }$ and its second marginal is $\mathcal { D } _ { X } ^ { T }$ Take any $( x _ { S } , x _ { T } ) \not \in \mathrm { s u p p } ( { \mathcal { D } } _ { X } ^ { S } ) \times \mathrm { s u p p } ( { \mathcal { D } } _ { X } ^ { T } )$ . Then either $x _ { S } \notin \operatorname* { s u p p } ( \mathcal { D } _ { X } ^ { S } )$ or $x _ { T } \notin \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { T } )$

If $x _ { S } \notin \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { S } )$ , by Definition 2.1 there exists an open neighborhood $U \subseteq { \mathcal { X } }$ of $x _ { S }$ such that $\mathcal { D } _ { X } ^ { S } ( U ) = 0$ . Since the first marginal of γ is $\mathcal { D } _ { X } ^ { S }$

$$
\gamma ( U \times \mathcal { X } ) = { \mathcal { D } } _ { X } ^ { S } ( U ) = 0 .
$$

In particular, for any open neighborhood V of $x _ { T }$ we have $\gamma ( U \times V ) \leq \gamma ( U \times \mathcal { X } ) = 0 , \operatorname { s o } \gamma ( U \times V ) = 0$ . Thus, taking the product open set $U \times V$ containing $( x _ { S } , x _ { T } )$ , we conclude that $( x _ { S } , x _ { T } ) \not \in \mathrm { s u p p } ( \gamma )$ by Definition 2.1 applied on the product topology of $\mathcal { X } \times \mathcal { X }$

The case $x _ { T } \notin \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { T } )$ is symmetric: there exists an open $V \ni x _ { T }$ with $\mathcal { D } _ { X } ^ { T } ( V ) = 0$ , and using the second marginal of γ gives $\gamma ( \mathcal { X } \times V ) = 0 .$ , hence $\gamma ( U \times V ) = 0$ for any open $U \ni x _ { S }$ , implying $( x _ { S } , x _ { T } ) \not \in \mathrm { s u p p } ( \gamma )$

Therefore every point outside $\mathrm { s u p p } ( \mathcal { D } _ { X } ^ { S } ) \times \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { T } )$ is outside supp(γ), i.e.,

$$
\mathrm { s u p p } ( \gamma ) \subseteq \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { S } ) \times \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { T } ) .
$$

## C.4. Proof of Lemma 3.5

Proof. Let $\nu : = \mathcal { D } _ { X } ^ { S } \otimes \mathcal { D } _ { X } ^ { T }$ and consider the entropic optimal transport objective

$$
J ( \gamma ) : = \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma ( x _ { S } , x _ { T } ) + \beta H ( \gamma \mid \nu ) , \qquad \gamma \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } ) .
$$

We claim that J is strictly convex over the convex set $\Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ . Indeed, the transport cost term $\gamma \mapsto \textstyle \int \rho _ { \mathcal { X } } d \gamma$ is linear in $\gamma .$ . For the entropy term, write $\begin{array} { r } { r = \frac { d \gamma } { d \nu } } \end{array}$ ; then

$$
H ( \gamma \mid \nu ) = \int r \log r d \nu .
$$

For any $\gamma _ { 1 } , \gamma _ { 2 } \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ with $\gamma _ { 1 } \neq \gamma _ { 2 }$ , by Lemma 3.4,

$$
\mathrm { s u p p } ( \gamma _ { 1 } ) \subseteq \mathrm { s u p p } ( { \mathcal D } _ { X } ^ { S } ) \times \mathrm { s u p p } ( { \mathcal D } _ { X } ^ { T } ) = \mathrm { s u p p } ( \nu ) .
$$

$$
\mathrm { s u p p } ( \gamma _ { 2 } ) \subseteq \mathrm { s u p p } ( { \mathcal D } _ { X } ^ { S } ) \times \mathrm { s u p p } ( { \mathcal D } _ { X } ^ { T } ) = \mathrm { s u p p } ( \nu ) .
$$

Because $\varphi ( t ) = t \log t$ is strictly convex on $[ 0 , \infty )$ , for and any $\lambda \in ( 0 , 1 )$

$$
H \big ( \lambda \gamma _ { 1 } + ( 1 - \lambda ) \gamma _ { 2 } \mid \nu \big ) < \lambda H ( \gamma _ { 1 } \mid \nu ) + ( 1 - \lambda ) H ( \gamma _ { 2 } \mid \nu ) ,
$$

hence (multiplying by $\beta > 0 )$ the whole objective satisfies

$$
J \big ( \lambda \gamma _ { 1 } + ( 1 - \lambda ) \gamma _ { 2 } \big ) < \lambda J ( \gamma _ { 1 } ) + ( 1 - \lambda ) J ( \gamma _ { 2 } ) .
$$

Now suppose, toward a contradiction, that there are two distinct optimal couplings $\gamma _ { 1 } \neq \gamma _ { 2 }$ minimizing J over $\Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ . By convexity of $\Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ , their mixture $\gamma _ { \lambda } : = \lambda \gamma _ { 1 } + ( 1 - \lambda ) \gamma _ { 2 }$ is feasible. Strict convexity then yields

$$
J ( \gamma _ { \lambda } ) < \lambda J ( \gamma _ { 1 } ) + ( 1 - \lambda ) J ( \gamma _ { 2 } ) = \operatorname* { i n f } _ { \gamma \in \Gamma } J ( \gamma ) ,
$$

a contradiction. Therefore the optimizer $\gamma ^ { * }$ is unique.

## C.5. Proof of Lemma 3.6

Proof. When $\beta = 0$ , Definition 3.1 reduces to the Wasserstein-1 problem

$$
\operatorname* { i n f } _ { \gamma \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } ) } \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma ( x _ { S } , x _ { T } ) ,
$$

where $\rho _ { \mathcal { X } }$ is a metric, hence $\rho _ { \mathcal { X } } \geq 0$ and $\rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) = 0 { \mathrm { i f } } x _ { S } = x _ { T }$

Consider the diagonal (identity) coupling $\bar { \gamma } : = ( \mathrm { I d } , \mathrm { I d } ) _ { \# } \mathcal { D } _ { X } ^ { S }$ , i.e.,

$$
\bar { \gamma } ( A ) = \int \mathbf { 1 } _ { \{ ( x , x ) \in A \} } d { \cal D } _ { X } ^ { S } ( x ) .
$$

It is immediate that $\bar { \gamma } \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { S } )$ , and its transport cost is

$$
\int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \bar { \gamma } ( x _ { S } , x _ { T } ) = \int \rho _ { \mathcal { X } } ( x , x ) d \mathcal { D } _ { X } ^ { S } ( x ) = 0 .
$$

Therefore the optimal value is at most 0, hence equals 0.

Let $\gamma ^ { \ast } \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { S } )$ be any optimal coupling. Then

$$
0 = \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) .
$$

Since the integrand is nonnegative, this implies $\rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) = 0$ holds $\gamma ^ { * }$ -almost surely, i.e., $( x _ { S } , x _ { T } ) \in \zeta : = \{ ( x , x ) : x \in$ ${ \mathcal { X } } \} \gamma ^ { * } { \mathrm { - a . s } }$ . Hence $\operatorname { s u p p } ( \gamma ^ { * } ) \subseteq \zeta .$

Finally, because $\gamma ^ { * }$ is supported on $\zeta$ and has first marginal $\mathcal { D } _ { X } ^ { S }$ , for every measurable $A \subseteq { \mathcal { X } } \times { \mathcal { X } }$ we have

$$
\gamma ^ { * } ( A ) = \gamma ^ { * } ( A \cap \zeta ) = \int \mathbf { 1 } _ { \left\{ ( x , x ) \in A \right\} } d \mathscr { D } _ { X } ^ { S } ( x ) = ( \mathrm { I d } , \mathrm { I d } ) _ { \# } \mathscr { D } _ { X } ^ { S } ( A ) ,
$$

which proves $\gamma ^ { * } = ( \mathrm { I d } , \mathrm { I d } ) _ { \# } \mathcal { D } _ { X } ^ { S }$ and the stated equivalent form.

## C.6. Proof of Theorem 3.7

Proof. For $\beta > 0$ , Lemma 3.5 ensures that the entropic optimal transport coupling $\gamma ^ { * }$ in Definition 3.1 is unique. Hence any potential ambiguity of $S _ { C p t } ^ { \gamma ^ { * } }$ can only come from the choice of versions of the conditional distributions.

Given any two versions of the source conditionals $\{ \mathcal { D } _ { Y \mid X = x } ^ { S } \} _ { x \in \mathcal { X } }$ and $\{ \widetilde { D } _ { Y \mid X = x } ^ { S } \} _ { x \in \mathcal { X } }$ , and any two versions of the targe conditionals $\{ \mathcal { D } _ { Y \mid X = x } ^ { T } \} _ { x \in \mathcal { X } }$ and $\{ \widetilde { D } _ { Y | X = x } ^ { T } \} _ { x \in \mathcal { X } }$ , such that $\mathcal { D } _ { Y | X = x } ^ { S } = \widetilde { \mathcal { D } } _ { Y | X = x } ^ { S }$ holds ${ \mathcal { D } } _ { X } ^ { S } \mathrm { - a . e . }$ ., and $\mathcal { D } _ { Y | X = x } ^ { T } = \widetilde { \mathcal { D } } _ { Y | X = x } ^ { T }$

holds $\mathcal { D } _ { X } ^ { T } \mathrm { - a . e }$ . Therefore, there exist measurable sets $E _ { S } , E _ { T } \subseteq \mathcal { X }$ with $\mathcal { D } _ { X } ^ { S } ( E _ { S } ) = 0$ and $\mathcal { D } _ { X } ^ { T } ( E _ { T } ) = 0$ such that the equalities hold for all $x \notin E _ { S }$ and all x $\notin E _ { T }$ , respectively.

Define the $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift under the two choices as

$$
S _ { C p t } ^ { \gamma ^ { * } } : = \int W _ { 1 } \big ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \big ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) ,
$$

$$
\widetilde { S } _ { C p t } ^ { \gamma ^ { * } } : = \int W _ { 1 } \left( \widetilde { \mathcal { D } } _ { Y | X = x _ { S } } ^ { S } , \widetilde { \mathcal { D } } _ { Y | X = x _ { T } } ^ { T } \right) d \gamma ^ { * } ( x _ { S } , x _ { T } ) .
$$

Let $E : = ( E _ { S } \times \mathcal { X } ) \cup ( \mathcal { X } \times E _ { T } )$ . For any $( x _ { S } , x _ { T } ) \in E ^ { c }$ we have $x s \notin E _ { S }$ and $x _ { T } \notin E _ { T }$ , hence

$$
\mathcal { D } _ { Y | X = x _ { S } } ^ { S } = \widetilde { \mathcal { D } } _ { Y | X = x _ { S } } ^ { S } , \qquad \mathcal { D } _ { Y | X = x _ { T } } ^ { T } = \widetilde { \mathcal { D } } _ { Y | X = x _ { T } } ^ { T } ,
$$

which implies

$$
\begin{array} { r } { W _ { 1 } \big ( \mathcal D _ { Y | X = x _ { S } } ^ { S } , \mathcal D _ { Y | X = x _ { T } } ^ { T } \big ) = W _ { 1 } \big ( \widetilde D _ { Y | X = x _ { S } } ^ { S } , \widetilde D _ { Y | X = x _ { T } } ^ { T } \big ) \quad \mathrm { f o r ~ a l l ~ } ( x _ { S } , x _ { T } ) \in E ^ { c } . } \end{array}
$$

Consequently, the difference between the two versions satisfies

$$
S _ { C p t } ^ { \gamma ^ { * } } - \widetilde { S } _ { C p t } ^ { \gamma ^ { * } } = \int _ { E } \left[ W _ { 1 } \left( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \right) - W _ { 1 } \left( \widetilde { \mathcal { D } } _ { Y | X = x _ { S } } ^ { S } , \widetilde { \mathcal { D } } _ { Y | X = x _ { T } } ^ { T } \right) \right] d \gamma ^ { * } ( x _ { S } , x _ { T } ) ,
$$

By Lemma 3.4, supp $\begin{array} { r } { \mathfrak { d } ( \gamma ^ { * } ) \subseteq \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { S } ) \times \operatorname { s u p p } ( \mathcal { D } _ { X } ^ { T } ) } \end{array}$ , so $\gamma ^ { * }$ never places mass outside the support product region. Furthermore, since $\gamma ^ { \ast } \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ , hence

$$
\gamma ^ { * } ( E _ { S } \times \mathcal { X } ) = \mathcal { D } _ { X } ^ { S } ( E _ { S } ) = 0 , \qquad \gamma ^ { * } ( \mathcal { X } \times E _ { T } ) = \mathcal { D } _ { X } ^ { T } ( E _ { T } ) = 0 ,
$$

and therefore $\gamma ^ { * } ( E ) = 0$ , and thus

$$
S _ { C p t } ^ { \gamma ^ { * } } - \widetilde { S } _ { C p t } ^ { \gamma ^ { * } } = 0 .
$$

Therefore $S _ { C p t } ^ { \gamma ^ { * } }$ does not depend on the choice of versions of the conditional distributions. This proves that $S _ { C p t } ^ { \gamma ^ { * } }$ is unique, regardless of whether $\mathrm { s u p p } ( \mathcal { D } _ { X } ^ { S } ) \neq \mathrm { s u p p } ( \mathcal { D } _ { X } ^ { T } )$ ). □

## C.7. Proof of Proposition 3.8

Proof. Since $\beta = 0$ and $\mathcal { D } _ { X } ^ { S } = \mathcal { D } _ { X } ^ { T }$ , Lemma 3.6 gives that the optimal coupling collapses to the diagonal:

$$
\gamma ^ { * } = ( \operatorname { I d } , \operatorname { I d } ) _ { \# } D _ { X } ^ { S } .
$$

Under deterministic labeling and $\rho _ { \mathscr { y } } = | \cdot |$ , Corollary 3.3 yields

$$
S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \big [ | f _ { S } ( x _ { S } ) - f _ { T } ( x _ { T } ) | \big ] .
$$

Substituting $\gamma ^ { * } = ( \mathrm { I d } , \mathrm { I d } ) _ { \# } D _ { X } ^ { S }$ implies $( x _ { S } , x _ { T } ) = ( x , x )$ with $x \sim D _ { X } ^ { S }$ , hence

$$
S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { \boldsymbol { x } \sim D _ { \boldsymbol { X } } ^ { S } } \left[ | \boldsymbol { f } _ { S } ( \boldsymbol { x } ) - \boldsymbol { f } _ { T } ( \boldsymbol { x } ) | \right] .
$$

Finally, since $\mathcal { D } _ { X } ^ { S } = \mathcal { D } _ { X } ^ { T }$ , we obtain

$$
S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { S } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ] = \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { T } } \big [ | f _ { S } ( x ) - f _ { T } ( x ) | \big ] ,
$$

as claimed.

## C.8. Proof of Corollary 3.10

Proof. Take any $y _ { 1 } , y _ { 2 } \in \mathcal { D }$ and $x _ { 1 } , x _ { 2 } \in \mathcal { X }$ . By the separate $( L _ { \ell } , L _ { \ell } ^ { \prime } )$ -Lipschitzness of $\ell ,$

$$
{ \big | } \ell ( y _ { 1 } , h ( x _ { 1 } ) ) - \ell ( y _ { 2 } , h ( x _ { 2 } ) ) { \big | } \leq L _ { \ell } \rho y ( y _ { 1 } , y _ { 2 } ) + L _ { \ell } ^ { \prime } \rho y \prime { \big ( } h ( x _ { 1 } ) , h ( x _ { 2 } ) { \big ) } .
$$

By the L -Lipschitzness of $h ,$ we further have $\rho _ { \mathcal { V } ^ { \prime } } ( h ( x _ { 1 } ) , h ( x _ { 2 } ) ) \leq L _ { h } \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 2 } )$ . Substituting this bound yields

$$
\left| \ell ( y _ { 1 } , h ( x _ { 1 } ) ) - \ell ( y _ { 2 } , h ( x _ { 2 } ) ) \right| \le L _ { \ell } \rho _ { \mathcal { V } } ( y _ { 1 } , y _ { 2 } ) + ( L _ { h } L _ { \ell } ^ { \prime } ) \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 2 } ) ,
$$

which is exactly the separate $( L _ { \ell } , L _ { h } L _ { \ell } ^ { \prime } )$ -Lipschitz condition for the composite function $\ell ( y , h ( x ) )$

## C.9. Proof of Lemma 3.11

Proof. Write the two expectations explicitly:

$$
\mathbb { E } _ { \mathcal { D } _ { X Y } ^ { s } } [ \mathcal { G } ] = \int _ { \mathcal { X } \times \mathcal { Y } } \mathcal { G } ( y s , x _ { S } ) d \mathcal { D } _ { X Y } ^ { s } ( x _ { S } , y s ) , \qquad \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] = \int _ { \mathcal { X } \times \mathcal { Y } } \mathcal { G } ( y _ { T } , x _ { T } ) d \mathcal { D } _ { X Y } ^ { T } ( x _ { T } , y _ { T } ) .
$$

Let $\gamma _ { X Y } \in \Gamma ( \mathcal { D } _ { X Y } ^ { S } , \mathcal { D } _ { X Y } ^ { T } )$ be any coupling on $( \mathcal { X } \times \mathcal { Y } ) \times ( \mathcal { X } \times \mathcal { Y } )$ , i.e., its first marginal is $\mathcal { D } _ { X Y } ^ { S }$ and second marginal is $\mathcal { D } _ { X Y } ^ { T }$ . Hence, for any integrable functions $\varphi , \psi$

$$
\int \varphi ( x _ { S } , y _ { S } ) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) = \int \varphi ( x _ { S } , y _ { S } ) d D _ { X Y } ^ { S } ( x _ { S } , y _ { S } ) ,
$$

$$
\int \psi ( x _ { T } , y _ { T } ) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) = \int \psi ( x _ { T } , y _ { T } ) d D _ { X Y } ^ { T } ( x _ { T } , y _ { T } ) .
$$

Applying these identities to $\varphi ( x _ { S } , y _ { S } ) = { \mathcal { G } } ( y _ { S } , x _ { S } ) { \mathrm { ~ a n d ~ } } \psi ( x _ { T } , y _ { T } ) = { \mathcal { G } } ( y _ { T } , x _ { T } )$ gives

$$
\begin{array} { l } { \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { s } } [ \mathcal { G } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] = \displaystyle \int \mathcal { G } ( y _ { s } , x _ { s } ) d \mathcal { D } _ { X Y } ^ { s } ( x _ { s } , y _ { s } ) - \int \mathcal { G } ( y _ { T } , x _ { T } ) d \mathcal { D } _ { X Y } ^ { T } ( x _ { T } , y _ { T } ) } \\ { = \displaystyle \int \mathcal { G } ( y _ { s } , x _ { s } ) d \gamma _ { X Y } ( x _ { s } , y _ { s } , x _ { T } , y _ { T } ) - \int \mathcal { G } ( y _ { T } , x _ { T } ) d \gamma _ { X Y } \big ( x _ { S } , y _ { S } , x _ { T } , y _ { T } \big ) } \\ { = \displaystyle \int \Big ( \mathcal { G } ( y _ { S } , x _ { S } ) - \mathcal { G } ( y _ { T } , x _ { T } ) \Big ) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) . } \end{array}
$$

Taking absolute values and using $\begin{array} { r } { | \int f d \mu | \leq \int | f | d \mu } \end{array}$ yields

$$
\left| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { s } } [ \mathcal { G } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] \right| \leq \int \left| \mathcal { G } ( y _ { S } , x _ { S } ) - \mathcal { G } ( y _ { T } , x _ { T } ) \right| d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) .
$$

By the separate $( L _ { \mathcal { Y } } , L _ { \mathcal { X } } ) .$ -Lipschitz property of ${ \mathcal { G } } ,$ we have

$$
{ \left| { \mathcal { G } } ( y _ { S } , x _ { S } ) - { \mathcal { G } } ( y _ { T } , x _ { T } ) \right| } \leq L _ { { \mathcal { Y } } } \rho _ { { \mathcal { Y } } } ( y _ { S } , y _ { T } ) + L _ { { \mathcal { X } } } \rho _ { { \mathcal { X } } } ( x _ { S } , x _ { T } ) .
$$

Substituting and splitting the integral,

$$
\begin{array} { r l } & { \left| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { \delta } } [ \mathcal { G } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] \right| \leq \displaystyle \int \left( L _ { \mathcal { Y } } \rho _ { \mathcal { Y } } ( y _ { S } , y _ { T } ) + L _ { \mathcal { X } } \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) \right) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) } \\ & { \qquad = L _ { \mathcal { Y } } \displaystyle \int \rho _ { \mathcal { Y } } ( y _ { S } , y _ { T } ) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) + L _ { \mathcal { X } } \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma _ { X Y } ( x _ { S } , y _ { S } , x _ { T } , y _ { T } ) } \\ & { \qquad = L _ { \mathcal { Y } } \mathbb { E } _ { \mathcal { Y } _ { X Y } } \bigl [ \rho _ { \mathcal { Y } } ( y _ { S } , y _ { T } ) \bigr ] + L _ { \mathcal { X } } \mathbb { E } _ { \mathcal { T } _ { X X } } \bigl [ \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) \bigr ] , } \end{array}
$$

as claimed.

## C.10. Proof of Lemma 3.12

Proof. We show that the first marginal of $\gamma _ { X Y }$ equals $\mathcal { D } _ { X Y } ^ { S }$ and the second marginal equals $\mathcal { D } _ { X Y } ^ { T }$

First marginal. Take any measurable $A \subseteq { \mathcal { X } }$ and $B \subseteq { \mathcal { V } } .$ . By the definition of $\gamma _ { X Y }$

$$
\gamma _ { X Y } \big ( ( A \times B ) \times ( \mathcal { X } \times \mathcal { Y } ) \big ) = \int _ { ( x _ { S } , x _ { T } ) \in \mathcal { X } \times \mathcal { X } } \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } \big ( ( B \times \mathcal { Y } ) \big ) \mathbf { 1 } _ { \{ x _ { S } \in A \} } d \gamma ^ { * } ( x _ { S } , x _ { T } ) .
$$

For each fixed $( x _ { S } , x _ { T } )$ , the coupling $\gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } \in \Gamma ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } )$ has first marginal $\mathcal { D } _ { Y \mid X = x _ { S } } ^ { S }$ , hence

$$
\gamma _ { Y \mid ( x _ { S } , x _ { T } ) } ^ { * } ( B \times \mathcal { Y } ) = \mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } ( B ) .
$$

Substituting gives

$$
\gamma _ { X Y } { \big ( } ( A \times B ) \times ( { \mathcal { X } } \times { \mathcal { Y } } ) { \big ) } = \int \mathbf { 1 } _ { \{ x _ { S } \in A \} } { \mathcal { D } } _ { Y | X = x _ { S } } ^ { S } ( B ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) .
$$

Finally, since the first marginal of $\gamma ^ { * }$ is $\mathcal { D } _ { X } ^ { S }$ , integrating out $x _ { T }$ yields

$$
\gamma _ { X Y } \big ( ( A \times B ) \times ( \mathcal { X } \times \mathcal { Y } ) \big ) = \int _ { x _ { S } \in A } \mathcal { D } _ { Y | X = x _ { S } } ^ { S } ( B ) d \mathcal { D } _ { X } ^ { S } ( x _ { S } ) = \mathcal { D } _ { X Y } ^ { S } ( A \times B ) ,
$$

Second marginal. Similarly, for measurable $A \subseteq { \mathcal { X } }$ and $B \subseteq \mathcal { V }$

$$
\begin{array} { r l } { \gamma _ { X Y } \big ( ( \mathcal { X } \times \mathcal { Y } ) \times ( A \times B ) \big ) = \displaystyle \int \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } ( \mathcal { V } \times B ) \mathbf { 1 } _ { \{ x _ { T } \in A \} } d \gamma ^ { * } ( x _ { S } , x _ { T } ) } & { } \\ { \quad } & { = \displaystyle \int \mathbf { 1 } _ { \{ x _ { T } \in A \} } \mathcal { D } _ { Y | X = x _ { T } } ^ { T } ( B ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) } \\ { \quad } & { = \displaystyle \int _ { x _ { T } \in A } \mathcal { D } _ { Y | X = x _ { T } } ^ { T } ( B ) d \mathcal { D } _ { X } ^ { T } ( x _ { T } ) = \mathcal { D } _ { X Y } ^ { T } ( A \times B ) . } \end{array}
$$

Therefore $\gamma _ { X Y } \in \Gamma ( \mathcal { D } _ { X Y } ^ { S } , \mathcal { D } _ { X Y } ^ { T } )$

## C.11. Proof of Theorem 3.13

Proof. Define the composite function

$$
\mathcal G ( \boldsymbol y , \boldsymbol x ) : = \ell \big ( \boldsymbol y , h ( \boldsymbol x ) \big ) , \qquad ( \boldsymbol y , \boldsymbol x ) \in \boldsymbol { \mathcal { V } } \times \boldsymbol { \mathcal X } .
$$

By Corollary 3.10, G is separately $( L _ { \ell } , \ L _ { h } L _ { \ell } ^ { \prime } ) \mathrm { - I }$ Lipschitz, i.e.,

$$
\left| \mathcal { G } ( y _ { 1 } , x _ { 1 } ) - \mathcal { G } ( y _ { 2 } , x _ { 2 } ) \right| \le L _ { \ell } \rho _ { \mathcal { V } } ( y _ { 1 } , y _ { 2 } ) + \left( L _ { h } L _ { \ell } ^ { \prime } \right) \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 2 } ) .
$$

Hence

$$
\begin{array} { r l } & { \epsilon _ { T } ( h ) = \epsilon _ { S } ( h ) + \left( \epsilon _ { T } ( h ) - \epsilon _ { S } ( h ) \right) } \\ & { \qquad \leq \epsilon _ { S } ( h ) + \left| \epsilon _ { T } ( h ) - \epsilon _ { S } ( h ) \right| } \\ & { \qquad = \epsilon _ { S } ( h ) + \left| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { S } } [ \mathcal { G } ] \right| . } \end{array}
$$

Step 1: construct a specific joint coupling. Let $\gamma ^ { \ast } \in \Gamma ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ be the optimal coupling in Definition 3.1. For each $( x _ { S } , x _ { T } )$ , let $\gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } \in \Gamma ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } )$ be an optimal coupling of $S _ { p a i r } ( x _ { S } , x _ { T } )$ in Definition 3.2, define the joint distribution of couplings

$$
\gamma _ { X Y } ( d x _ { S } , d x _ { T } , d y _ { S } , d y _ { T } ) : = \gamma ^ { * } ( d x _ { S } , d x _ { T } ) \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } ( d y _ { S } , d y _ { T } ) .
$$

By Lemma 3.12, $\gamma _ { X Y } \in \Gamma ( \mathcal { D } _ { X Y } ^ { S } , \mathcal { D } _ { X Y } ^ { T } )$

Step 2: apply Lemma 3.11. Apply Lemma 3.11 to $\mathcal { G }$ with the coupling γ<sub>XY</sub>, we obtain:

$$
\begin{array} { r } { \left| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { \mathcal { S } } } [ \mathcal { G } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \mathcal { G } ] \right| \leq ( L _ { h } L _ { \ell } ^ { \prime } ) \mathbb { E } _ { \gamma _ { X Y } } \big [ \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) \big ] + L _ { \ell } \mathbb { E } _ { \gamma _ { X Y } } \big [ \rho _ { \mathcal { V } } ( y _ { S } , y _ { T } ) \big ] . } \end{array}
$$

It remains to bound the two expectations on the right-hand side.

Step 3: bound the X-term by $S _ { C o v }$ . By the construction of $\gamma _ { X Y }$

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { \gamma _ { X Y } } \big [ \rho \chi ( x _ { S } , x _ { T } ) \big ] = \int \rho \chi ( x _ { S } , x _ { T } ) \gamma ^ { * } ( d x _ { S } , d x _ { T } ) \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } ( d y _ { S } , d y _ { T } ) } \\ & { } & { \quad \quad \quad = \int \rho \chi ( x _ { S } , x _ { T } ) \gamma ^ { * } ( d x _ { S } , d x _ { T } ) = \mathbb { E } _ { \gamma ^ { * } } \big [ \rho \chi ( x _ { S } , x _ { T } ) \big ] . } \end{array}
$$

Since $S _ { C o v } = W _ { \beta } ( \mathcal { D } _ { X } ^ { S } , \mathcal { D } _ { X } ^ { T } )$ and $\gamma ^ { * }$ is optimal,

$$
S _ { C o v } = \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) + \beta H \big ( \gamma ^ { * } \mid \mathcal { D } _ { X } ^ { S } \otimes \mathcal { D } _ { X } ^ { T } \big ) \geq \int \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) d \gamma ^ { * } ( x _ { S } , x _ { T } ) ,
$$

because $\beta \geq 0$ and relative entropy is nonnegative. Therefore,

$$
\mathbb { E } _ { \gamma _ { X Y } } \big [ \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) \big ] = \mathbb { E } _ { \gamma ^ { * } } \big [ \rho _ { \mathcal { X } } ( x _ { S } , x _ { T } ) \big ] \leq S _ { C o v } .
$$

Step 4: identify the Y-term with $S _ { C p t } ^ { \gamma ^ { * } }$ . Again by the construction of $\gamma _ { X Y }$

$$
\begin{array} { r l } { \mathbb { E } _ { \gamma _ { X Y } } \big [ \rho y ( y _ { S } , y _ { T } ) \big ] = \displaystyle \int \rho y \big ( y _ { S } , y _ { T } \big ) \gamma ^ { * } \big ( d x _ { S } , d x _ { T } \big ) \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } \big ( d y _ { S } , d y _ { T } \big ) } & { } \\ { = \displaystyle \int _ { \mathcal { X } \times \mathcal { X } } \left[ \int _ { \mathcal { X } \times \mathcal { Y } } \rho y ( y _ { S } , y _ { T } ) d \gamma _ { Y | ( x _ { S } , x _ { T } ) } ^ { * } \big ( y _ { S } , y _ { T } \big ) \right] d \gamma ^ { * } \big ( x _ { S } , x _ { T } \big ) } & { } \\ { = \displaystyle \int _ { \mathcal { X } \times \mathcal { X } } W _ { 1 } \big ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \big ) d \gamma ^ { * } \big ( x _ { S } , x _ { T } \big ) } & { } \\ { = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \left[ S _ { p a i r } \big ( x _ { S } , x _ { T } \big ) \right] = S _ { C p t } ^ { \gamma ^ { * } } , } \end{array}
$$

Step 5: conclude the bound. Combining Steps 2–4 yields

$$
\left| \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { s } } [ \boldsymbol { \mathcal { G } } ] - \mathbb { E } _ { \mathcal { D } _ { X Y } ^ { T } } [ \boldsymbol { \mathcal { G } } ] \right| \leq \left( L _ { h } L _ { \ell } ^ { \prime } \right) S _ { C o v } + L _ { \ell } S _ { C p t } ^ { \gamma ^ { * } } .
$$

Substituting this into the earlier inequality for $\epsilon _ { T } ( h )$ gives

$$
\epsilon _ { T } ( h ) \leq \epsilon _ { S } ( h ) + \left( L _ { h } L _ { \ell } ^ { \prime } \right) S _ { C o v } + L _ { \ell } S _ { C p t } ^ { \gamma ^ { * } } ,
$$

which proves the theorem.

## C.12. Proof of Theorem 4.2

Proof. We work with $\beta = 0$ , hence $W _ { \beta } = W _ { 1 }$ . For brevity, denote

$$
\mu : = { \mathcal { D } } _ { X } ^ { S } , \qquad \nu : = { \mathcal { D } } _ { X } ^ { T } , \qquad c : = W _ { 1 } ( \mu , \nu ) .
$$

Let $\hat { \mu } ^ { \prime } , \hat { \mu } ^ { \prime \prime }$ (resp. $\hat { \nu } ^ { \prime } , \hat { \nu } ^ { \prime \prime } )$ be the two half-sample empirical measures constructed in Definition 4.1. They are independent within each domain because they are built from disjoint halves of i.i.d. samples.

A useful inequality. We will use the elementary fact: for any $a , b \geq 0 ,$

$$
| { \sqrt { a } } - { \sqrt { b } } | \leq { \sqrt { | a - b | } } .
$$

Indeed, if $a \geq b$ then $\begin{array} { r } { \sqrt { a } - \sqrt { b } = \frac { a - b } { \sqrt { a } + \sqrt { b } } \leq \frac { a - b } { \sqrt { a - b } } = \sqrt { a - b } , } \end{array}$ and the case $b \geq a$ is symmetric.

Step 1: rewrite the debiased estimator and introduce an auxiliary statistic. Define

$$
\begin{array} { r } { S : = \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) ^ { 2 } + \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } - \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \mu } } ^ { \prime \prime } ) ^ { 2 } - \frac { 1 } { 2 } W _ { 1 } ( { \hat { \nu } } ^ { \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } . } \end{array}
$$

Then Definition 4.1 (with $\beta = 0 )$ gives

$$
{ \widehat { W _ { 1 } ^ { d e b } ( { \widehat { D _ { X } ^ { S } } } , { \widehat { D _ { X } ^ { T } } } ) } } = { \sqrt { | S | } } .
$$

Introduce the auxiliary random variable

$$
T : = { \big | } S - c ^ { 2 } { \big | } .
$$

Step 2: reduce $\{ | W _ { 1 } ^ { d e b } - c | > \varepsilon \}$ to an event on T. First note the deterministic inequality

$$
\begin{array} { r } { | | S | - c ^ { 2 } | \leq | S - c ^ { 2 } | = T , } \end{array}
$$

which holds because $c ^ { 2 } \geq 0$ . Hence, using the inequality in the first paragraph,

$$
| W _ { 1 } ^ { d e b } - c | = \big | \sqrt { | S | } - \sqrt { c ^ { 2 } } \big | \leq \sqrt { \big | | S | - c ^ { 2 } \big | } \leq \sqrt { T } .
$$

Therefore, for any $\varepsilon > 0 ,$

$$
\begin{array} { r } { \mathbb { P } \big ( | W _ { 1 } ^ { d e b } - c | > \varepsilon \big ) \le \mathbb { P } \big ( \sqrt { T } > \varepsilon \big ) = \mathbb { P } \big ( T > \varepsilon ^ { 2 } \big ) . } \end{array}
$$

This bound is always valid and is particularly useful when c is small.

When $c \geq \varepsilon .$ , we can obtain a complementary reduction by squaring the deviation event. Indeed, $| W _ { 1 } ^ { d e b } - c | > \varepsilon$ implies either $W _ { 1 } ^ { d e b } > c + \varepsilon$ or $W _ { 1 } ^ { d e b } < c - \varepsilon$ . In both cases,

$$
\left| ( W _ { 1 } ^ { d e b } ) ^ { 2 } - c ^ { 2 } \right| > ( c + \varepsilon ) ^ { 2 } - c ^ { 2 } = 2 c \varepsilon + \varepsilon ^ { 2 } \quad \mathrm { o r } \quad c ^ { 2 } - ( c - \varepsilon ) ^ { 2 } = 2 c \varepsilon - \varepsilon ^ { 2 } ,
$$

hence in particular

$$
\left| ( W _ { 1 } ^ { d e b } ) ^ { 2 } - c ^ { 2 } \right| > 2 c \varepsilon - \varepsilon ^ { 2 } .
$$

Since $( W _ { 1 } ^ { d e b } ) ^ { 2 } = | S |$ , we have

$$
\{ | W _ { 1 } ^ { d e b } - c | > \varepsilon \} \subseteq \{ \left| | S | - c ^ { 2 } \right| > 2 c \varepsilon - \varepsilon ^ { 2 } \} \subseteq \{ T > 2 c \varepsilon - \varepsilon ^ { 2 } \} .
$$

Combining both regimes, define the threshold

$$
t ( c , \varepsilon ) : = { \left\{ \begin{array} { l l } { \varepsilon ^ { 2 } , } & { c < \varepsilon , } \\ { 2 c \varepsilon - \varepsilon ^ { 2 } , } & { c \geq \varepsilon , } \end{array} \right. }
$$

so that for all $c \geq 0 ,$

$$
\begin{array} { r } { \mathbb { P } \big ( | W _ { 1 } ^ { d e b } - c | > \varepsilon \big ) \le \mathbb { P } \big ( T > t ( c , \varepsilon ) \big ) . } \end{array}
$$

Step 3: bound T by marginal empirical Wasserstein errors. Start from the definition of T and use the triangle inequality for | · |:

$$
\begin{array} { r l } & { T = \Big | \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) ^ { 2 } + \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } - \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \mu } } ^ { \prime \prime } ) ^ { 2 } - \frac { 1 } { 2 } W _ { 1 } ( { \hat { \nu } } ^ { \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } - c ^ { 2 } \Big | } \\ & { \quad \leq \frac { 1 } { 2 } \Big | W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) ^ { 2 } - c ^ { 2 } \Big | + \frac { 1 } { 2 } \Big | W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } - c ^ { 2 } \Big | + \frac { 1 } { 2 } W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \mu } } ^ { \prime \prime } ) ^ { 2 } + \frac { 1 } { 2 } W _ { 1 } ( { \hat { \nu } } ^ { \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } . } \end{array}
$$

Step 3.1: control the cross-domain square terms. Let $u : = W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } )$ . Then

$$
| u ^ { 2 } - c ^ { 2 } | = | u - c | | u + c | \leq | u - c | \left( | u - c | + 2 c \right) = 2 c | u - c | + | u - c | ^ { 2 } .
$$

Multiplying by $\textstyle { \frac { 1 } { 2 } }$ gives

$$
\begin{array} { r } { \frac 1 2 | u ^ { 2 } - c ^ { 2 } | \leq c | u - c | + \frac 1 2 | u - c | ^ { 2 } . } \end{array}
$$

Applying this with $u = W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } )$ and again with $u = W _ { 1 } ( \hat { \mu } ^ { \prime \prime } , \hat { \nu } ^ { \prime \prime } )$ yields

$$
\begin{array} { r } { \frac { 1 } { 2 } \Big | W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) ^ { 2 } - c ^ { 2 } \Big | \leq c \big | W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) - c \big | + { \frac { 1 } { 2 } } \big | W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) - c \big | ^ { 2 } , } \end{array}
$$

$$
\begin{array} { r } { \frac { 1 } { 2 } \Big | W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) ^ { 2 } - c ^ { 2 } \Big | \leq c \big | W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) - c \big | + \frac { 1 } { 2 } \big | W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) - c \big | ^ { 2 } . } \end{array}
$$

Step 3.2: relate $| W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) - c | t o W _ { 1 } ( \mu , { \hat { \mu } } ^ { \prime } )$ and $W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } )$ . Using the triangle inequality for $W _ { 1 }$

$$
W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } ) \leq W _ { 1 } ( \hat { \mu } ^ { \prime } , \mu ) + W _ { 1 } ( \mu , \nu ) + W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } ) = W _ { 1 } ( \hat { \mu } ^ { \prime } , \mu ) + c + W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } ) ,
$$

hence $W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } ) - c \leq W _ { 1 } ( \hat { \mu } ^ { \prime } , \mu ) + W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } )$ . Similarly,

$$
c = W _ { 1 } ( \mu , \nu ) \leq W _ { 1 } ( \mu , \hat { \mu } ^ { \prime } ) + W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } ) + W _ { 1 } ( \hat { \nu } ^ { \prime } , \nu ) ,
$$

so $c - W _ { 1 } ( { \hat { \mu } } ^ { \prime } , { \hat { \nu } } ^ { \prime } ) \leq W _ { 1 } ( \mu , { \hat { \mu } } ^ { \prime } ) + W _ { 1 } ( \nu , { \hat { \nu } } ^ { \prime } )$ . Combining the two inequalities,

$$
\left| W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \nu } ^ { \prime } ) - c \right| \leq W _ { 1 } ( \mu , \hat { \mu } ^ { \prime } ) + W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } ) .
$$

The same argument gives

$$
\left| W _ { 1 } ( { \hat { \mu } } ^ { \prime \prime } , { \hat { \nu } } ^ { \prime \prime } ) - c \right| \leq W _ { 1 } ( \mu , { \hat { \mu } } ^ { \prime \prime } ) + W _ { 1 } ( \nu , { \hat { \nu } } ^ { \prime \prime } ) .
$$

Step 3.3: relate $W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \mu } ^ { \prime \prime } )$ and $W _ { 1 } ( \hat { \nu } ^ { \prime } , \hat { \nu } ^ { \prime \prime } )$ to the same marginal errors. Again by the triangle inequality,

$$
\begin{array} { r } { W _ { 1 } ( \hat { \mu } ^ { \prime } , \hat { \mu } ^ { \prime \prime } ) \leq W _ { 1 } ( \hat { \mu } ^ { \prime } , \mu ) + W _ { 1 } ( \mu , \hat { \mu } ^ { \prime \prime } ) , \qquad W _ { 1 } ( \hat { \nu } ^ { \prime } , \hat { \nu } ^ { \prime \prime } ) \leq W _ { 1 } ( \hat { \nu } ^ { \prime } , \nu ) + W _ { 1 } ( \nu , \hat { \nu } ^ { \prime \prime } ) . } \end{array}
$$

Step 3.4: assemble the bound. Introduce the shorthand nonnegative random variables

$$
\begin{array} { r } { u ^ { \prime } : = W _ { 1 } ( \mu , \hat { \mu } ^ { \prime } ) , \quad u ^ { \prime \prime } : = W _ { 1 } ( \mu , \hat { \mu } ^ { \prime \prime } ) , \quad v ^ { \prime } : = W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } ) , \quad v ^ { \prime \prime } : = W _ { 1 } ( \nu , \hat { \nu } ^ { \prime \prime } ) . } \end{array}
$$

Then Steps 3.1–3.3 imply

$$
\begin{array} { r } { T \leq c ( u ^ { \prime } + v ^ { \prime } ) + \frac { 1 } { 2 } ( u ^ { \prime } + v ^ { \prime } ) ^ { 2 } + c ( u ^ { \prime \prime } + v ^ { \prime \prime } ) + \frac { 1 } { 2 } ( u ^ { \prime \prime } + v ^ { \prime \prime } ) ^ { 2 } + \frac { 1 } { 2 } ( u ^ { \prime } + u ^ { \prime \prime } ) ^ { 2 } + \frac { 1 } { 2 } ( v ^ { \prime } + v ^ { \prime \prime } ) ^ { 2 } . } \end{array}
$$

Now use $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ repeatedly to simplify:

$$
\begin{array} { r } { \frac { 1 } { 2 } ( u ^ { \prime } + v ^ { \prime } ) ^ { 2 } \leq u ^ { \prime 2 } + v ^ { \prime 2 } , \qquad \frac { 1 } { 2 } ( u ^ { \prime \prime } + v ^ { \prime \prime } ) ^ { 2 } \leq u ^ { \prime \prime 2 } + v ^ { \prime \prime 2 } , \qquad \frac { 1 } { 2 } ( u ^ { \prime } + u ^ { \prime \prime } ) ^ { 2 } \leq u ^ { \prime 2 } + u ^ { \prime \prime 2 } , \qquad \frac { 1 } { 2 } ( v ^ { \prime } + v ^ { \prime \prime } ) ^ { 2 } \leq v ^ { \prime 2 } + v ^ { \prime \prime 2 } . } \end{array}
$$

Substituting yields the clean bound

$$
T \le c ( u ^ { \prime } + u ^ { \prime \prime } + v ^ { \prime } + v ^ { \prime \prime } ) + 2 \big ( u ^ { \prime 2 } + u ^ { \prime \prime 2 } + v ^ { \prime 2 } + v ^ { \prime \prime 2 } \big ) .
$$

In particular, on the event $\left\{ u ^ { \prime } \leq \varepsilon _ { 0 } , u ^ { \prime \prime } \leq \varepsilon _ { 0 } , v ^ { \prime } \leq \varepsilon _ { 0 } , v ^ { \prime \prime } \leq \varepsilon _ { 0 } \right\}$ , we have

$$
\begin{array} { r } { T \leq 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } . } \end{array}
$$

Hence,

$$
\{ T > 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } \} \subseteq \{ u ^ { \prime } > \varepsilon _ { 0 } \} \cup \{ u ^ { \prime \prime } > \varepsilon _ { 0 } \} \cup \{ v ^ { \prime } > \varepsilon _ { 0 } \} \cup \{ v ^ { \prime \prime } > \varepsilon _ { 0 } \} ,
$$

and by the union bound,

$$
\begin{array} { r } { \mathbb { P } \big ( T > 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } \big ) \le \mathbb { P } ( u ^ { \prime } > \varepsilon _ { 0 } ) + \mathbb { P } ( u ^ { \prime \prime } > \varepsilon _ { 0 } ) + \mathbb { P } ( v ^ { \prime } > \varepsilon _ { 0 } ) + \mathbb { P } ( v ^ { \prime \prime } > \varepsilon _ { 0 } ) + \mathbb { P } ( v ^ { \prime \prime } > \varepsilon _ { 0 } ) . } \end{array}
$$

Step 4: apply traditional concentration inequality. By assumption, $\mu$ and ν have finite squared-exponential moments on $\mathbb { R } ^ { d }$ . Lemma B.3 and B.4 implies that there exist constants $\lambda _ { S } , \lambda _ { T } > 0 .$ , depending only on the squared-exponential moments of $\mu$ and ν respectively, such that for any $\varepsilon _ { 0 } > 0$ there exists $N$ with the following property: whenever $N _ { S } / 2 \geq N$ and $N _ { T } / 2 \geq N$

$$
\begin{array} { r } { \mathbb { P } \big ( W _ { 1 } ( \mu , \hat { \mu } ^ { \prime } ) > \varepsilon _ { 0 } \big ) \leq \mathrm { e x p } \Big ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Big ) , \qquad \mathbb { P } \big ( W _ { 1 } ( \mu , \hat { \mu } ^ { \prime \prime } ) > \varepsilon _ { 0 } \big ) \leq \mathrm { e x p } \Big ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Big ) , } \end{array}
$$

and similarly

$$
\begin{array} { r } { \mathbb { P } \big ( W _ { 1 } ( \nu , \hat { \nu } ^ { \prime } ) > \varepsilon _ { 0 } \big ) \leq \exp \Bigl ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Bigr ) , \qquad \mathbb { P } \big ( W _ { 1 } ( \nu , \hat { \nu } ^ { \prime \prime } ) > \varepsilon _ { 0 } \big ) \leq \exp \Bigl ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Bigr ) . } \end{array}
$$

Therefore, for $N _ { S } , N _ { T }$ large enough,

$$
\begin{array} { r } { \mathbb { P } \big ( T > 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } \big ) \leq 2 \exp \\\Bigl ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Bigr ) + 2 \exp \Bigl ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Bigr ) . } \end{array}
$$

Step $\mathbf { 5 } \colon$ choose $\varepsilon _ { 0 }$ to match the deviation level ε. Recall from Step 2 that

$$
\begin{array} { r } { \mathbb { P } \big ( | W _ { 1 } ^ { d e b } - c | > \varepsilon \big ) \le \mathbb { P } \big ( T > t ( c , \varepsilon ) \big ) , \qquad t ( c , \varepsilon ) = \left\{ \begin{array} { l l } { \varepsilon ^ { 2 } , } & { c < \varepsilon , } \\ { 2 c \varepsilon - \varepsilon ^ { 2 } , } & { c \ge \varepsilon . } \end{array} \right. } \end{array}
$$

We now select $\varepsilon _ { \mathrm { 0 } }$ so that

$$
4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } = t ( c , \varepsilon ) .
$$

Let $a : = c / \varepsilon \geq 0 .$ Solving this quadratic in the two regimes yields the closed-form expression

$$
\varepsilon _ { 0 } ^ { 2 } = { \frac { V ( a ) \varepsilon ^ { 2 } } { 8 } } ,
$$

where

$$
V ( a ) = \left\{ \begin{array} { l l } { a ^ { 2 } + 1 - a \sqrt { a ^ { 2 } + 2 } , } & { a < 1 , } \\ { a ^ { 2 } + 2 a - 1 - a \sqrt { a ^ { 2 } + 4 a - 2 } , } & { a \geq 1 . } \end{array} \right.
$$

Define $V _ { \varepsilon } : = V \big ( c / \varepsilon \big ) = V \big ( W _ { 1 } ( \mu , \nu ) / \varepsilon \big )$ . As shown in the “Range of $V ( a ) ^ { \mathfrak { r } }$ derivation, $V _ { \varepsilon } \in [ 2 - \sqrt { 3 } , 2 )$ and depends only on $c / \varepsilon$

With this choice, $\{ T > t ( c , \varepsilon ) \} = \{ T > 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } \}$ , hence

$$
\begin{array} { r l } & { \mathbb { P } \big ( | W _ { 1 } ^ { d e b } - c | > \varepsilon \big ) \leq \mathbb { P } \big ( T > 4 c \varepsilon _ { 0 } + 8 \varepsilon _ { 0 } ^ { 2 } \big ) } \\ & { \qquad \leq 2 \exp \\Big ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Big ) + 2 \exp \Big ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 4 } \Big ) } \\ & { \qquad = 2 \exp \Big ( - \frac { \lambda _ { S } N _ { S } V _ { \varepsilon } \varepsilon ^ { 2 } } { 3 2 } \Big ) + 2 \exp \Big ( - \frac { \lambda _ { T } N _ { T } V _ { \varepsilon } \varepsilon ^ { 2 } } { 3 2 } \Big ) . } \end{array}
$$

Finally, note $c = W _ { 1 } ( \mu , \nu ) = W _ { \beta } ( { \mathcal D } _ { X } ^ { S } , { \mathcal D } _ { X } ^ { T } )$ ) when $\beta = 0 ;$ , and $W _ { 1 } ^ { d e b } = W _ { \beta } ^ { d e b }$ when $\beta = 0 ;$ , so the above inequality is exactly (10). This completes the proof.

Range of $V ( a )$ . Recall

$$
V ( a ) = \left\{ \begin{array} { l l } { V _ { 1 } ( a ) : = a ^ { 2 } + 1 - a \sqrt { a ^ { 2 } + 2 } , } & { 0 \leq a < 1 , } \\ { V _ { 2 } ( a ) : = a ^ { 2 } + 2 a - 1 - a \sqrt { a ^ { 2 } + 4 a - 2 } , } & { a \geq 1 . } \end{array} \right.
$$

First, the two branches agree at $a = 1$

$$
V _ { 1 } ( 1 ) = 2 - \sqrt { 3 } , \qquad V _ { 2 } ( 1 ) = 2 - \sqrt { 3 } .
$$

(i) Monotonicity on [0, 1]. Differentiate $V _ { 1 }$ on (0, 1):

$$
V _ { 1 } ^ { \prime } ( a ) = 2 a - \sqrt { a ^ { 2 } + 2 } - \frac { a ^ { 2 } } { \sqrt { a ^ { 2 } + 2 } } = 2 a - \frac { 2 ( a ^ { 2 } + 1 ) } { \sqrt { a ^ { 2 } + 2 } } .
$$

For $a \geq 0$ , we have $a { \sqrt { a ^ { 2 } + 2 } } \leq a ^ { 2 } + 1$ since

$$
( a ^ { 2 } + 1 ) ^ { 2 } - a ^ { 2 } ( a ^ { 2 } + 2 ) = 1 > 0 .
$$

Thus $\textstyle { \frac { a ^ { 2 } + 1 } { \sqrt { a ^ { 2 } + 2 } } } \geq a .$ , which implies $V _ { 1 } ^ { \prime } ( a ) \leq 0 \mathrm { o n } \left( 0 , 1 \right)$ . Hence $V _ { 1 }$ is decreasing on [0, 1], so

$$
V _ { 1 } ( a ) \in \left[ V _ { 1 } ( 1 ) , V _ { 1 } ( 0 ) \right] = [ 2 - \sqrt { 3 } , 1 ] .
$$

(ii) Monotonicity on $\lbrack 1 , \infty )$ . Let $s ( a ) : = { \sqrt { a ^ { 2 } + 4 a - 2 } } .$ . Then for $a > 1$

$$
V _ { 2 } ^ { \prime } ( a ) = 2 a + 2 - s ( a ) - \frac { a ( a + 2 ) } { s ( a ) } .
$$

Since $s ( a ) > 0$ , the inequality $V _ { 2 } ^ { \prime } ( a ) \geq 0$ is equivalent to

$$
( a + 1 ) s ( a ) \geq a ^ { 2 } + 3 a - 1 .
$$

Squaring both sides (both sides are nonnegative for $a \geq 1 )$ gives

$$
( a + 1 ) ^ { 2 } ( a ^ { 2 } + 4 a - 2 ) \geq ( a ^ { 2 } + 3 a - 1 ) ^ { 2 } ,
$$

and the difference factors as

$$
( a + 1 ) ^ { 2 } ( a ^ { 2 } + 4 a - 2 ) - ( a ^ { 2 } + 3 a - 1 ) ^ { 2 } = 3 ( 2 a - 1 ) \geq 0 \qquad ( a \geq 1 ) .
$$

Therefore $V _ { 2 } ^ { \prime } ( a ) \geq 0$ for all $a \geq 1 , \mathrm { i . e . , } V _ { 2 }$ is increasing on $\lbrack 1 , \infty )$ . In particular,

$$
V _ { 2 } ( a ) \geq V _ { 2 } ( 1 ) = 2 - \sqrt { 3 } \qquad ( a \geq 1 ) .
$$

(iii) Upper bound $V ( a ) < 2$ and li $\mathrm { n } _ { a \to \infty } V ( a ) = 2 .$ . For $a \ge 1$ , write $s ( a ) = \sqrt { ( a + 2 ) ^ { 2 } - 6 }$ . Then

$$
( a + 2 - s ( a ) ) ( a + 2 + s ( a ) ) = ( a + 2 ) ^ { 2 } - s ( a ) ^ { 2 } = 6 , \quad \mathrm { ~ s o ~ } \quad a + 2 - s ( a ) = \frac { 6 } { a + 2 + s ( a ) } .
$$

Using $V _ { 2 } ( a ) = a ( a + 2 ) - 1 - a s ( a )$ , we get

$$
2 - V _ { 2 } ( a ) = 3 - a \left( a + 2 - s ( a ) \right) = 3 - { \frac { 6 a } { a + 2 + s ( a ) } } .
$$

Since $s ( a ) > a - 2$ for $a \ge 1$ (indeed $s ( a ) = \sqrt { a ^ { 2 } + 4 a - 2 } > a )$ , we have $a + 2 + s ( a ) > 2 a$ , hence $\frac { 6 a } { a + 2 + s ( a ) } < 3 .$ , which implies $2 - V _ { 2 } ( a ) > 0$ and thus $V _ { 2 } ( a ) < 2$ for every finite a. Moreover, as $a  \infty ,$ , one has $a + 2 + s ( a ) \sim 2 a$ , hence $\frac { 6 a } { a + 2 + s ( a ) }  3$ and therefore $V _ { 2 } ( a )  2$

Conclusion. Combining (i)–(iii), the global minimum is attained at $a = 1$ with

$$
\operatorname* { m i n } _ { a \geq 0 } V ( a ) = 2 - { \sqrt { 3 } } ,
$$

and the supremum equals 2 but is not attained:

$$
V ( a ) \in [ 2 - { \sqrt { 3 } } , 2 ) , \qquad a \geq 0 .
$$

## C.13. Proof of Lemma 4.4

Proof. We follow (Eckstein & Nutz, 2022). Recall that the entropically regularized OT problem is

$$
S _ { \mathrm { e n t } } ^ { \varepsilon } ( \mu , \nu , c ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int c d \pi + \varepsilon \mathrm { K L } ( \pi \| \mu \otimes \nu ) ,
$$

and for fixed $\varepsilon > 0$ one may assume $\varepsilon = 1$ by dividing the objective by ε and using the cost $c / \varepsilon$

Step 1 (Reduction to $\varepsilon = 1$ and identification of $L = 1 / \beta )$ . Let $c ( x _ { 1 } , x _ { 2 } ) : = \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 2 } )$ . The $\beta .$ -entropic OT objective

$$
\int c d \pi + \beta \mathrm { K L } ( \pi \| \mu \otimes \nu )
$$

has the same optimizer as

$$
\int { \frac { 1 } { \beta } } c d \pi + \mathrm { K L } ( \pi \parallel \mu \otimes \nu ) ,
$$

since the two objectives differ only by a multiplicative factor $\beta .$ . Hence the optimal coupling for $W _ { \beta } ( \mu , \nu )$ equals the optimizer of $S _ { \mathrm { e n t } } ^ { 1 } ( \mu , \nu , c _ { \beta } )$ with $c _ { \beta } : = c / \beta$

Consider the product space $\mathcal { X } \times \mathcal { X }$ with metric

$$
\rho \big ( ( x _ { 1 } , x _ { 2 } ) , ( x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime } ) \big ) : = \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 1 } ^ { \prime } ) + \rho _ { \mathcal { X } } ( x _ { 2 } , x _ { 2 } ^ { \prime } ) .
$$

By the triangle inequality,

$$
\left| c ( x _ { 1 } , x _ { 2 } ) - c ( x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime } ) \right| = \left| \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 2 } ) - \rho _ { \mathcal { X } } ( x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime } ) \right| \leq \rho _ { \mathcal { X } } ( x _ { 1 } , x _ { 1 } ^ { \prime } ) + \rho _ { \mathcal { X } } ( x _ { 2 } , x _ { 2 } ^ { \prime } ) = \rho ( ( x _ { 1 } , x _ { 2 } ) , ( x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime } ) ) ,
$$

so c is 1-Lipschitz w.r.t. $\rho ,$ and therefore $c _ { \beta } = c / \beta$ satisfies the Lipschitz condition (AL) of Eckstein & Nutz (2022) with constant

$$
L = \operatorname { L i p } ( c _ { \beta } ) = { \frac { 1 } { \beta } } .
$$

Step 2 (Apply Theorem 3.11). Let $\mu : = \mathcal { D } _ { X } ^ { S } , \nu : = \mathcal { D } _ { X } ^ { T }$ , and similarly $\hat { \mu } : = \widehat { \mathcal { D } _ { X } ^ { S } } , \hat { \nu } : = \widehat { \mathcal { D } _ { X } ^ { T } }$ . Let $\gamma ^ { * }$ and $\hat { \gamma } ^ { * }$ be the optimizers of $S _ { \mathrm { e n t } } ^ { 1 } ( \mu , \nu , c _ { \beta } )$ and $S _ { \mathrm { e n t } } ^ { 1 } ( \hat { \mu } , \hat { \nu } , c _ { \beta } )$ , respectively.

Assume $\mu$ and $\nu$ have finite squared-exponential moments. Then by Lemma 3.10(ii) in Eckstein & Nutz (2022), the pair of marginals satisfies the inequality $\left( \mathrm { I } _ { 1 } \right)$ with some finite constant $C _ { 1 } > 0$ depending only on these squared-exponentia moments.

Now apply Theorem 3.11 of Eckstein & Nutz (2022) with $N = 2$ and $p = q = 1$ . Since $N ^ { 1 / q - 1 / p } = 1$ , we obtain

$$
W _ { 1 } ( \gamma ^ { * } , \hat { \gamma } ^ { * } ) \leq \Delta + C _ { 1 } ( 2 L \Delta ) ^ { 1 / 2 } , \qquad \Delta : = W _ { 1 } \bigl ( ( \mu , \nu ) ; ( \hat { \mu } , \hat { \nu } ) \bigr ) ,
$$

Step 3 (Upper bound $\Delta$ by Λ). Let $\pi _ { S }$ be an optimal coupling between $\mu$ and $\hat { \mu } ,$ , and $\pi _ { T }$ an optimal coupling between ν and $\hat { \nu } .$ . Then $\pi _ { S } \otimes \pi _ { T }$ is a coupling between $( \mu , \nu )$ and $( \hat { \mu } , \hat { \nu } )$ on $\mathcal { X } \times \mathcal { X }$ , and its expected ρ-cost equals

$$
\int \rho _ { \mathcal K } ( x , x ^ { \prime } ) d \pi _ { S } ( x , x ^ { \prime } ) + \int \rho _ { \mathcal K } ( y , y ^ { \prime } ) d \pi _ { T } ( y , y ^ { \prime } ) = W _ { 1 } ( \mu , \hat { \mu } ) + W _ { 1 } ( \nu , \hat { \nu } ) = : A .
$$

Taking the infimum over all couplings yields $\Delta \le A$ . Therefore,

$$
W _ { 1 } ( \gamma ^ { * } , \hat { \gamma } ^ { * } ) \leq A + C _ { 1 } \sqrt { 2 L A } = A + C _ { 1 } \sqrt { \frac { 2 } { \beta } \Lambda } .
$$

Step 4 (Definition of $\lambda _ { \gamma } .$ ∗ and the factor $2 \sqrt { 2 } )$ . Define

$$
\lambda _ { \gamma ^ { * } } : = \frac { 4 } { C _ { 1 } ^ { 2 } } \quad \Longleftrightarrow \quad C _ { 1 } = \frac { 2 } { \sqrt { \lambda _ { \gamma ^ { * } } } } .
$$

Then

$$
C _ { 1 } \sqrt { \frac { 2 } { \beta } \ { \cal A } } = \frac { 2 } { \sqrt { \lambda _ { \gamma ^ { * } } } } \sqrt { \frac { 2 } { \beta } } \sqrt { \cal A } = 2 \sqrt { \frac { 2 } { \beta \lambda _ { \gamma ^ { * } } } } \sqrt { \cal A } ,
$$

which yields the claimed bound. Finally, $\lambda _ { \gamma ^ { * } } > 0$ depends only on the squared-exponential moments of $\mu$ and $\nu$ because $C _ { 1 }$ does via Lemma 3.10(ii) (Eckstein & Nutz, 2022). □

## C.14. Proof of Theorem 4.5

Proof. Let

$$
\mu : = \mathcal { D } _ { X } ^ { S } , \qquad \nu : = \mathcal { D } _ { X } ^ { T } , \qquad \hat { \mu } : = \widehat { \mathcal { D } _ { X } ^ { S } } = \frac { 1 } { N _ { S } } \sum _ { i = 1 } ^ { N _ { S } } \delta _ { X _ { i } ^ { ( S ) } } , \qquad \hat { \nu } : = \widehat { \mathcal { D } _ { X } ^ { T } } = \frac { 1 } { N _ { T } } \sum _ { j = 1 } ^ { N _ { T } } \delta _ { X _ { j } ^ { ( T ) } } .
$$

Let $\gamma ^ { * }$ be the population entropic OT coupling between $\mu$ and ν (with $\beta > 0 )$ , and let $\hat { \gamma } ^ { * } = ( \hat { \gamma } _ { i j } ^ { * } ) \in \mathbb { R } _ { + } ^ { N _ { S } \times N _ { T } }$ be the discrete optimal coupling solving $W _ { \beta } ( \boldsymbol { \hat { \mu } } , \boldsymbol { \hat { \nu } } )$ . It satisfies the marginal constraints

$$
\sum _ { j = 1 } ^ { N _ { T } } \hat { \gamma } _ { i j } ^ { * } = \frac { 1 } { N _ { S } } \quad ( \forall i ) , \qquad \sum _ { i = 1 } ^ { N _ { S } } \hat { \gamma } _ { i j } ^ { * } = \frac { 1 } { N _ { T } } \quad ( \forall j ) .
$$

Associate to $\hat { \gamma } ^ { * }$ a probability measure on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d } ;$

$$
\bar { \gamma } ^ { * } : = \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } \hat { \gamma } _ { i j } ^ { * } \delta _ { ( X _ { i } ^ { ( S ) } , X _ { j } ^ { ( T ) } ) } \in \Gamma ( \hat { \mu } , \hat { \nu } ) .
$$

Throughout, Wasserstein–1 distances on $\mathbb { R } ^ { d }$ use the Euclidean norm $\| \cdot \|$ , and on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ we use the metric

$$
\rho \big ( ( x _ { S } , x _ { T } ) , ( x _ { S } ^ { \prime } , x _ { T } ^ { \prime } ) \big ) : = \| x _ { S } - x _ { S } ^ { \prime } \| + \| x _ { T } - x _ { T } ^ { \prime } \| .
$$

Step 1: define the population quantity we will concentrate around and the bias $\varDelta .$ Define the function $f : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } $ R<sub>+</sub> by

$$
f ( x _ { S } , x _ { T } ) : = \mathbb { E } _ { y _ { S } \sim \mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } , y _ { T } \sim \mathcal { D } _ { Y \mid X = x _ { T } } ^ { T } } \big [ \| y _ { S } - y _ { T } \| \big ] .
$$

Since $\mathcal { y } \subset \mathbb { R } ^ { d ^ { \prime } }$ is bounded with $M = \operatorname* { s u p } _ { y , y ^ { \prime } \in \mathcal { V } } \| y - y ^ { \prime } \|$ , we have $0 \leq f ( x _ { S } , x _ { T } ) \leq M$ for all $( x _ { S } , x _ { T } )$

Recall that

$$
S _ { p a i r } ( x _ { S } , x _ { T } ) = W _ { 1 } \bigl ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \ \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \bigr ) , \qquad S _ { C p t } ^ { \gamma ^ { * } } = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \bigl [ S _ { p a i r } ( x _ { S } , x _ { T } ) \bigr ] .
$$

Because $f ( x _ { S } , x _ { T } )$ is the transport cost under the independent coupling $\mathcal { D } _ { Y | x _ { S } } ^ { S } \otimes \mathcal { D } _ { Y | x _ { T } } ^ { T }$ , while $S _ { p a i r } ( x _ { S } , x _ { T } )$ is the infimum over all couplings, we have

$$
f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \geq 0 .
$$

Define the (constant) bias term

$$
\begin{array} { r } { \varDelta : = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \Big [ f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \Big ] . } \end{array}
$$

Then

$$
\begin{array} { r } { { S _ { C p t } ^ { \gamma ^ { * } } + \varDelta = \mathbb E _ { \gamma ^ { * } } \left[ f ( x _ { S } , x _ { T } ) \right] } . } \end{array}
$$

Hence it suffices to prove concentration of $\hat { S } _ { C p t }$ around $\mathbb { E } _ { \gamma ^ { * } } [ f ]$

Step 2: decompose the error into a Y -noise term and a coupling-stability term. By Definition 4.3,

$$
\hat { S } _ { C p t } = \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } \Vert Y _ { i } ^ { ( S ) } - Y _ { j } ^ { ( T ) } \Vert \hat { \gamma } _ { i j } ^ { * } .
$$

Add and subtract $\mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ]$

$$
| \hat { S } _ { C p t } - \mathbb { E } _ { \gamma ^ { * } } [ f ] | \leq \underbrace { \left| \hat { S } _ { C p t } - \mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ] \right| } _ { = : A } + \underbrace { \left| \mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ] - \mathbb { E } _ { \gamma ^ { * } } [ f ] \right| } _ { = : B } .
$$

We will bound A by McDiarmid’s inequality and B by Lipschitzness of $f$ and stability of entropic OT.

Step 3: concentration of A via McDiarmid inequality B.2. Condition on all covariates $\{ X _ { i } ^ { ( S ) } \} _ { i = 1 } ^ { N _ { S } }$ and $\{ X _ { j } ^ { ( T ) } \} _ { j = 1 } ^ { N _ { T } }$ Under this conditioning, $\hat { \gamma } ^ { * }$ is fixed (it depends only on the covariates), and the labels $\{ Y _ { i } ^ { ( S ) } \}$ are independent with $Y _ { i } ^ { ( S ) } \sim \mathcal { D } _ { Y | X = X _ { i } ^ { ( S ) } } ^ { S }$ , and similarly $\{ Y _ { j } ^ { ( T ) } \}$ are independent with $Y _ { j } ^ { ( T ) } \sim \mathcal { D } _ { Y \mid X = X _ { j } ^ { ( T ) } } ^ { T }$

Define the function of all labels

$$
F \big ( \{ Y _ { i } ^ { ( S ) } \} _ { i = 1 } ^ { N _ { S } } , \{ Y _ { j } ^ { ( T ) } \} _ { j = 1 } ^ { N _ { T } } \big ) : = \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } \| Y _ { i } ^ { ( S ) } - Y _ { j } ^ { ( T ) } \| \hat { \gamma } _ { i j } ^ { * } .
$$

Then $A = \left| F - \mathbb { E } [ F \mid \{ X \} ] \right|$

Bounded differencesfor changing one source label. Fix an index i and replace $Y _ { i } ^ { ( S ) }$ by another value $Y _ { i } ^ { ( S ) ^ { \prime } }$ in $\mathcal { V } ,$ , keeping all other labels the same. Then, by the triangle inequality,

$$
\begin{array} { r l } & { \Big | F ( \ldots , Y _ { i } ^ { ( S ) } , \ldots ) - F ( \ldots , Y _ { i } ^ { ( S ) ^ { \prime } } , \ldots ) \Big | } \\ & { = \Big | \displaystyle \sum _ { j = 1 } ^ { N _ { T } } \Big ( \| Y _ { i } ^ { ( S ) } - Y _ { j } ^ { ( T ) } \| - \| Y _ { i } ^ { ( S ) ^ { \prime } } - Y _ { j } ^ { ( T ) } \| \Big ) \hat { \gamma } _ { i j } ^ { * } \Big | } \\ & { \le \displaystyle \sum _ { j = 1 } ^ { N _ { T } } \Big | \| Y _ { i } ^ { ( S ) } - Y _ { j } ^ { ( T ) } \| - \| Y _ { i } ^ { ( S ) ^ { \prime } } - Y _ { j } ^ { ( T ) } \| \Big | \hat { \gamma } _ { i j } ^ { * } } \\ & { \le \displaystyle \sum _ { j = 1 } ^ { N _ { T } } \| Y _ { i } ^ { ( S ) } - Y _ { i } ^ { ( S ) ^ { \prime } } \| \hat { \gamma } _ { i j } ^ { * } \le { M } \displaystyle \sum _ { j = 1 } ^ { N _ { T } } \hat { \gamma } _ { i j } ^ { * } = \frac { M } { N _ { S } } . } \end{array}
$$

Bounded differences for changing one target label. Similarly, replacing one $Y _ { j } ^ { ( T ) }$ by $Y _ { j } ^ { ( T ) ^ { \prime } }$ changes $F$ by at most

$$
{ \frac { M } { N _ { T } } } .
$$

Apply McDiarmid. By McDiarmid nequality B.2, for any $\varepsilon _ { 1 } > 0$

$$
\begin{array} { r l } & { \mathbb { P } \bigg ( | F - \mathbb { E } [ F \mid \{ X \} ] | > \varepsilon _ { 1 } \biggm | \{ X \} \bigg ) \leq 2 \exp \bigg ( - \frac { 2 \varepsilon _ { 1 } ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { S } } ( M / N _ { S } ) ^ { 2 } + \sum _ { j = 1 } ^ { N _ { T } } ( M / N _ { T } ) ^ { 2 } } \bigg ) } \\ & { \qquad = 2 \exp \bigg ( - \frac { 2 \varepsilon _ { 1 } ^ { 2 } } { M ^ { 2 } \big ( \frac { 1 } { N _ { S } } + \frac { 1 } { N _ { T } } \big ) } \bigg ) } \\ & { \qquad = 2 \exp \bigg ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 1 } ^ { 2 } } { \big ( N _ { S } + N _ { T } \big ) M ^ { 2 } } \bigg ) . } \end{array}
$$

Taking expectation over $\{ X \}$ gives the unconditional bound

$$
\mathbb { P } ( A > \varepsilon _ { 1 } ) \le 2 \exp \biggl ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 1 } ^ { 2 } } { ( N _ { S } + N _ { T } ) M ^ { 2 } } \biggr ) .
$$

Compute the conditional mean $\mathbb { E } [ F \mid \{ X \} ]$ and identify $\mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ]$ . Since $Y _ { i } ^ { ( S ) }$ and $Y _ { j } ^ { ( T ) }$ are independent given $\{ X \}$ , we have

$$
\mathbb { E } \big [ \| Y _ { i } ^ { ( S ) } - Y _ { j } ^ { ( T ) } \| ~ | ~ \{ X \} \big ] = f \big ( X _ { i } ^ { ( S ) } , X _ { j } ^ { ( T ) } \big ) ,
$$

hence

$$
\mathbb { E } [ F \mid \{ X \} ] = \sum _ { i = 1 } ^ { N _ { S } } \sum _ { j = 1 } ^ { N _ { T } } f \left( X _ { i } ^ { ( S ) } , X _ { j } ^ { ( T ) } \right) \hat { \gamma } _ { i j } ^ { * } = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \bar { \gamma } ^ { * } } \left[ f ( x _ { S } , x _ { T } ) \right] = \mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ] .
$$

Therefore $A = \left| F - \mathbb { E } [ F \mid \{ X \} ] \right|$ is exactly the deviation term we bounded.

Step 4: bound B using Lipschitzness of $f$ and stability of entropic OT.

Step 4.1: show $f$ is $( M L _ { Y \mid X } ) .$ -Lipschitz on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d } .$ . We prove Lipschitzness in the first coordinate; the second is symmetric. Fix $x _ { T }$ and let $x _ { S } , x _ { S } ^ { \prime } \in \mathbb { R } ^ { d }$ . Write $\pi _ { x _ { S } } ^ { S } : = { \cal D } _ { Y | X = x _ { S } } ^ { S }$ and $\pi _ { x _ { T } } ^ { T } : = { \cal D } _ { Y | X = x _ { T } } ^ { T }$ . Then

$$
\begin{array} { r l } & { f ( x _ { S } , x _ { T } ) - f ( x _ { S } ^ { \prime } , x _ { T } ) } \\ & { \ = \displaystyle \int _ { \mathcal { V } } \displaystyle \int _ { \mathcal { V } } \| y _ { S } - y _ { T } \| \left( \mathrm { d } \pi _ { x _ { S } } ^ { S } ( y _ { S } ) - \mathrm { d } \pi _ { x _ { S } ^ { \prime } } ^ { S } ( y _ { S } ) \right) \mathrm { d } \pi _ { x _ { T } } ^ { T } ( y _ { T } ) . } \end{array}
$$

Taking absolute values and using Fubini,

$$
\left. f ( x _ { S } , x _ { T } ) - f ( x _ { S } ^ { \prime } , x _ { T } ) \right. \le \int _ { \mathcal { Y } } \left. \int _ { \mathcal { Y } } \left. y _ { S } - y _ { T } \right. \left( \mathrm { d } \pi _ { x _ { S } } ^ { S } ( y _ { S } ) - \mathrm { d } \pi _ { x _ { S } ^ { \prime } } ^ { S } ( y _ { S } ) \right) \right. \mathrm { d } \pi _ { x _ { T } } ^ { T } ( y _ { T } ) .
$$

For each fixed $y _ { T }$ , define

$$
g _ { y _ { T } } ( y _ { S } ) : = \frac { 2 } { M } \| y _ { S } - y _ { T } \| - 1 .
$$

Since $\lVert y _ { S } - y _ { T } \rVert \in [ 0 , M ]$ for $y _ { S } , y _ { T } \in \mathcal { V }$ , we have $g _ { y _ { T } } \in [ - 1 , 1 ]$ and thus $\| g _ { y _ { T } } \| _ { \infty } \leq 1$ . Also note that

$$
\| y _ { S } - y _ { T } \| = \frac { M } { 2 } \big ( g _ { y _ { T } } ( y _ { S } ) + 1 \big ) ,
$$

and $\begin{array} { r } { \int { 1 \left( \mathrm { d } \pi _ { x _ { S } } ^ { S } - \mathrm { d } \pi _ { x _ { S } ^ { \prime } } ^ { S } \right) } = 0 } \end{array}$ , hence

$$
\begin{array} { l l } { \displaystyle \left. \int _ { \mathcal { V } } \left. y s - y _ { T } \right. \left( \mathrm { d } \pi _ { x s } ^ { S } - \mathrm { d } \pi _ { x _ { S } ^ { \prime } } ^ { S } \right) \right. = \displaystyle \frac { M } { 2 } \Big \lvert \int _ { \mathcal { V } } g _ { y _ { T } } ( y _ { S } ) \left( \mathrm { d } \pi _ { x _ { S } } ^ { S } - \mathrm { d } \pi _ { x _ { S } ^ { \prime } } ^ { S } \right) \Big \rvert } \\ { \displaystyle \qquad \leq M d _ { \mathrm { T V } } ( \pi _ { x s } ^ { S } , \pi _ { x _ { S } ^ { \prime } } ^ { S } ) , } \end{array}
$$

where we used the dual representation

$$
d _ { \mathrm { T V } } ( P , Q ) = \frac { 1 } { 2 } \operatorname* { s u p } _ { \| g \| _ { \infty } \leq 1 } \Bigl | \int g \mathrm { d } ( P - Q ) \Bigr | .
$$

Plugging this into the previous bound and using that $\pi _ { x _ { T } } ^ { T }$ has total mass 1 gives

$$
\begin{array} { r } { \big | f ( x _ { S } , x _ { T } ) - f ( x _ { S } ^ { \prime } , x _ { T } ) \big | \leq M d _ { \mathrm { T V } } \big ( \mathcal { D } _ { Y | X = x _ { S } } ^ { S } , \mathcal { D } _ { Y | X = x _ { S } ^ { \prime } } ^ { S } \big ) \leq M L _ { Y | X } \| x _ { S } - x _ { S } ^ { \prime } \| . } \end{array}
$$

Similarly,

$$
\left| f ( x _ { S } , x _ { T } ) - f ( x _ { S } , x _ { T } ^ { \prime } ) \right| \leq M L _ { Y | X } \| x _ { T } - x _ { T } ^ { \prime } \| .
$$

Combining both, for all $( x _ { S } , x _ { T } ) , ( x _ { S } ^ { \prime } , x _ { T } ^ { \prime } )$ ，

$$
\begin{array} { r } { \left| f ( x _ { S } , x _ { T } ) - f ( x _ { S } ^ { \prime } , x _ { T } ^ { \prime } ) \right| \leq M L _ { Y | X } \big ( \| x _ { S } - x _ { S } ^ { \prime } \| + \| x _ { T } - x _ { T } ^ { \prime } \| \big ) = M L _ { Y | X } \rho \big ( ( x _ { S } , x _ { T } ) , ( x _ { S } ^ { \prime } , x _ { T } ^ { \prime } ) \big ) . } \end{array}
$$

Thus $f \mathrm { i s } ( M L _ { Y | X } ) \ –$ Lipschitz on $( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } , \rho )$

Step 4.2: convert B to a Wasserstein distance on couplings. By the Kantorovich–Rubinstein duality for $W _ { 1 }$ on $( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } , \rho )$ , for any K-Lipschitz function $\varphi ,$

$$
\left| \mathbb { E } _ { \pi } [ \varphi ] - \mathbb { E } _ { \pi ^ { \prime } } [ \varphi ] \right| \le K W _ { 1 } ( \pi , \pi ^ { \prime } ) .
$$

Apply this with $\varphi = f$ and $K = M L _ { Y \mid X } , \pi = \bar { \gamma } ^ { * } , \pi ^ { \prime } = \gamma ^ { * }$

$$
B = \left| \mathbb { E } _ { \bar { \gamma } ^ { * } } [ f ] - \mathbb { E } _ { \gamma ^ { * } } [ f ] \right| \leq M L _ { Y | X } W _ { 1 } ( \gamma ^ { * } , \bar { \gamma } ^ { * } ) .
$$

Step 4.3: apply entropic OT stability and then concentration of marginals. By Lemma $4 . 4 ,$

$$
W _ { 1 } ( \gamma ^ { * } , \bar { \gamma } ^ { * } ) \leq \varLambda + 2 \sqrt { \frac { 2 } { \beta \lambda _ { \gamma ^ { * } } } } \sqrt { \varLambda } , \qquad \varLambda : = W _ { 1 } ( \mu , \hat { \mu } ) + W _ { 1 } ( \nu , \hat { \nu } ) .
$$

Therefore,

$$
B \leq M L _ { Y | X } \Big ( A + 2 \sqrt { \frac { 2 } { \beta \lambda _ { \gamma ^ { * } } } } \sqrt { \varLambda } \Big ) .
$$

Fix $\varepsilon _ { 2 } > 0$ . If $\varLambda \leq 2 \varepsilon _ { 2 }$ , then

$$
\sqrt { A } \leq \sqrt { 2 \varepsilon _ { 2 } } , \qquad A + 2 \sqrt { \frac { 2 } { \beta \lambda _ { \gamma ^ { * } } } } \sqrt { A } \leq 2 \varepsilon _ { 2 } + 4 \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } ,
$$

and hence

$$
B \leq 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } .
$$

Consequently,

$$
\bigg \{ B > 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } \bigg \} \subseteq \{ A > 2 \varepsilon _ { 2 } \} .
$$

By the union bound,

$$
\begin{array} { r } { \mathbb { P } ( A > 2 \varepsilon _ { 2 } ) \leq \mathbb { P } \big ( W _ { 1 } ( \mu , \hat { \mu } ) > \varepsilon _ { 2 } \big ) + \mathbb { P } \big ( W _ { 1 } ( \nu , \hat { \nu } ) > \varepsilon _ { 2 } \big ) . } \end{array}
$$

Since $\mu , \nu$ have finite squared-exponential moments on $\mathbb { R } ^ { d } .$ , by Lemma B.3 and B.4, there exist constants $\lambda _ { S } , \lambda _ { T } > 0$ depending only on these moments such that for any $\varepsilon _ { 2 } > 0$ there exists N with, whenever $N _ { S } , N _ { T } > N$

$$
\mathbb { P } \big ( W _ { 1 } ( \mu , \hat { \mu } ) > \varepsilon _ { 2 } \big ) \le \exp \Big ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 2 } ^ { 2 } } { 2 } \Big ) , \qquad \mathbb { P } \big ( W _ { 1 } ( \nu , \hat { \nu } ) > \varepsilon _ { 2 } \big ) \le \exp \Big ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 2 } ^ { 2 } } { 2 } \Big ) .
$$

Thus we conclude

$$
\mathbb { P } \bigg ( B > 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } \bigg ) \leq \exp \bigg ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 2 } ^ { 2 } } { 2 } \bigg ) + \exp \bigg ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 2 } ^ { 2 } } { 2 } \bigg ) .
$$

Step 5: combine the two parts. For any $\varepsilon _ { 1 } , \varepsilon _ { 2 } > 0$ , by the union bound,

$$
\begin{array} { r l } & { \mathbb { P } \bigg ( \big | \hat { S } _ { C p t } - \mathbb { E } _ { \gamma ^ { * } } [ f ] \big | > \varepsilon _ { 1 } + 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } \bigg ) \leq \mathbb { P } \big ( A > \varepsilon _ { 1 } \big ) } \\ & { \qquad + \mathbb { P } \bigg ( B > 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma ^ { * } } } } \bigg ) } \\ & { \leq 2 \exp \bigg ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 1 } ^ { 2 } } { \big ( N _ { S } + N _ { T } \big ) M ^ { 2 } } \bigg ) } \\ & { \qquad + \exp \Big ( { - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 2 } ^ { 2 } } { 2 } } \Big ) + \exp \Big ( { - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 2 } ^ { 2 } } { 2 } } \Big ) . } \end{array}
$$

Recalling $\mathbb { E } _ { \gamma ^ { * } } [ f ] = S _ { C p t } ^ { \gamma ^ { * } } + \varDelta$ , the left-hand side is exactly $\mathbb { P } ( | \hat { S } _ { C p t } - S _ { C p t } ^ { \gamma ^ { * } } - \varDelta | > \cdots )$

Step 6: parameter tying and the explicit Φ. Introduce a single auxiliary parameter $\varepsilon _ { 0 } > 0$ and set

$$
\varepsilon _ { 1 } : = M \varepsilon _ { 0 } , \qquad \varepsilon _ { 2 } : = \lambda _ { S } ^ { - 1 / 4 } \lambda _ { T } ^ { - 1 / 4 } \varepsilon _ { 0 } .
$$

Then the three exponential terms become

$$
\begin{array} { c } { { 2 \displaystyle \exp \Bigl ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 1 } ^ { 2 } } { ( N _ { S } + N _ { T } ) M ^ { 2 } } \Bigr ) = 2 \exp \Bigl ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { N _ { S } + N _ { T } } \Bigr ) , } } \\ { { \mathrm { e x p } \Bigl ( - \frac { \lambda _ { S } N _ { S } \varepsilon _ { 2 } ^ { 2 } } { 2 } \Bigr ) = \exp \Bigl ( - \frac { \lambda _ { S } ^ { 1 / 2 } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 2 \lambda _ { T } ^ { 1 / 2 } } \Bigr ) , } } \\ { { \mathrm { e x p } \Bigl ( - \frac { \lambda _ { T } N _ { T } \varepsilon _ { 2 } ^ { 2 } } { 2 } \Bigr ) = \exp \Bigl ( - \frac { \lambda _ { T } ^ { 1 / 2 } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 2 \lambda _ { S } ^ { 1 / 2 } } \Bigr ) . } } \end{array}
$$

The deviation level on the left-hand side becomes

$$
\begin{array} { r } { \varepsilon _ { 1 } + 2 M L _ { Y | X } \varepsilon _ { 2 } + 4 M L _ { Y | X } \sqrt { \frac { \varepsilon _ { 2 } } { \beta \lambda _ { \gamma * } } } = M \varepsilon _ { 0 } + 2 M L _ { Y | X } \lambda _ { S } ^ { - 1 / 4 } \lambda _ { T } ^ { - 1 / 4 } \varepsilon _ { 0 } + 4 M L _ { Y | X } \beta ^ { - 1 / 2 } \lambda _ { \gamma * } ^ { - 1 / 2 } \lambda _ { S } ^ { - 1 / 8 } \lambda _ { T } ^ { - 1 / 8 } \varepsilon _ { 0 } ^ { 1 / 2 } \varepsilon _ { 1 } } \end{array}
$$

Define the constants

$$
\begin{array} { r } { c _ { 2 } : = 1 + 2 L _ { Y | X } \lambda _ { S } ^ { - 1 / 4 } \lambda _ { T } ^ { - 1 / 4 } , \qquad c _ { 1 } : = 1 6 L _ { Y | X } ^ { 2 } \beta ^ { - 1 } \lambda _ { \gamma ^ { * } } ^ { - 1 } \lambda _ { S } ^ { - 1 / 4 } \lambda _ { T } ^ { - 1 / 4 } . } \end{array}
$$

Then the deviation level can be written compactly as

$$
M \Bigl ( c _ { 2 } \varepsilon _ { 0 } + \sqrt { c _ { 1 } \varepsilon _ { 0 } } \Bigr ) .
$$

Now choose $\varepsilon _ { \mathrm { 0 } }$ so that

$$
M \Bigl ( c _ { 2 } \varepsilon _ { 0 } + \sqrt { c _ { 1 } \varepsilon _ { 0 } } \Bigr ) = \varepsilon .
$$

Let u $: = \sqrt { \varepsilon _ { 0 } }$ . The equation becomes $c _ { 2 } u ^ { 2 } + \sqrt { c _ { 1 } } u = \varepsilon / M$ , whose positive solution is

$$
u = \frac { \sqrt { c _ { 1 } + 4 c _ { 2 } ( \varepsilon / M ) } - \sqrt { c _ { 1 } } } { 2 c _ { 2 } } , \qquad \varepsilon _ { 0 } = u ^ { 2 } .
$$

It is convenient to express $\varepsilon _ { 0 } ^ { 2 }$ in the form $\textstyle \varepsilon _ { 0 } ^ { 2 } = { \frac { \varPhi \varepsilon ^ { 2 } } { 2 M ^ { 2 } } }$ . Let

$$
a : = \frac { c _ { 1 } M } { c _ { 2 } \varepsilon } \quad ( > 0 ) , \qquad A ( a ) : = a ^ { 2 } + 4 a + 2 - ( a + 2 ) \sqrt { a ^ { 2 } + 4 a } , \qquad \phi : = \frac { A ( a ) } { c _ { 2 } ^ { 2 } } .
$$

A direct algebraic simplification (expanding $( a + 2 - \sqrt { a ^ { 2 } + 4 a } ) ^ { 2 } )$ shows

$$
\varepsilon _ { 0 } ^ { 2 } = \frac { \varPhi \varepsilon ^ { 2 } } { 2 M ^ { 2 } } .
$$

Substituting this $\varepsilon _ { 0 } ^ { 2 }$ into the exponential terms yields

$$
2 \exp \Bigl ( - \frac { 2 N _ { S } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { N _ { S } + N _ { T } } \Bigr ) = 2 \exp \Bigl ( - \frac { N _ { S } N _ { T } \phi \varepsilon ^ { 2 } } { \left( N _ { S } + N _ { T } \right) M ^ { 2 } } \Bigr ) ,
$$

$$
\exp \Bigl ( - \frac { \lambda _ { S } ^ { 1 / 2 } N _ { S } \varepsilon _ { 0 } ^ { 2 } } { 2 \lambda _ { T } ^ { 1 / 2 } } \Bigr ) = \exp \Bigl ( - \frac { \lambda _ { S } ^ { 1 / 2 } N _ { S } \varPhi \varepsilon ^ { 2 } } { 4 \lambda _ { T } ^ { 1 / 2 } M ^ { 2 } } \Bigr ) ,
$$

$$
\exp \Bigl ( - \frac { \lambda _ { T } ^ { 1 / 2 } N _ { T } \varepsilon _ { 0 } ^ { 2 } } { 2 \lambda _ { S } ^ { 1 / 2 } } \Bigr ) = \exp \Bigl ( - \frac { \lambda _ { T } ^ { 1 / 2 } N _ { T } \varPhi \varepsilon ^ { 2 } } { 4 \lambda _ { S } ^ { 1 / 2 } M ^ { 2 } } \Bigr ) .
$$

This is exactly the claimed bound, completing the proof.

## C.15. Proof of Proposition 4.6

Proof. Recall the notation used in the proof of Theorem 4.5: for $( x _ { S } , x _ { T } ) \in \mathcal { X } \times \mathcal { X }$ define

$$
f ( x _ { S } , x _ { T } ) : = \mathbb { E } _ { y s \sim { D _ { Y } ^ { s } } | x = x _ { S } } , y r \sim { D _ { Y } ^ { x } } _ { | X = x _ { T } } \left[ \rho y \left( y _ { S } , y _ { T } \right) \right] , \qquad S _ { p a i r } \left( x _ { S } , x _ { T } \right) : = W _ { 1 } \left( { \mathcal { D } _ { Y | X = x _ { S } } ^ { s } } , { \mathcal { D } _ { Y | X = x _ { T } } ^ { T } } \right) ,
$$

and the bias is

$$
\varDelta = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \bigl [ f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \bigr ] .
$$

Under deterministic labeling, $\mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } = \delta _ { f _ { S } ( x _ { S } ) }$ and $\mathcal { D } _ { Y | X = x _ { T } } ^ { T } = \delta _ { f _ { T } ( x _ { T } ) }$ . Hence the random variables $y _ { S } , y _ { T }$ are almost surely equal to $f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } )$ , and thus

$$
f ( x _ { S } , x _ { T } ) = \mathbb { E } { \left[ \rho _ { \mathcal { V } } ( y _ { S } , y _ { T } ) \right] } = \rho _ { \mathcal { V } } { \left( f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } ) \right) } .
$$

On the other hand, the Wasserstein-1 distance between two Dirac measures is exactly the ground metric:

$$
S _ { p a i r } ( x _ { S } , x _ { T } ) = W _ { 1 } \big ( \delta _ { f _ { S } ( x _ { S } ) } , \delta _ { f _ { T } ( x _ { T } ) } \big ) = \rho _ { \mathcal { V } } \big ( f _ { S } ( x _ { S } ) , f _ { T } ( x _ { T } ) \big ) .
$$

Therefore $f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) = 0$ for all $( x _ { S } , x _ { T } )$ , and so

$$
\begin{array} { r } { \varDelta = \mathbb { E } _ { \gamma ^ { * } } \bigl [ f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \bigr ] = 0 . } \end{array}
$$

This concludes the proof.

## C.16. Proof of Proposition 4.7

Proof. We follow the notation used in the proof of Theorem 4.5. For $( x _ { S } , x _ { T } ) \in \mathcal { X } \times \mathcal { X }$ , define

$$
f ( x _ { S } , x _ { T } ) : = \mathbb { E } _ { y _ { S } \sim \mathcal { D } _ { Y | X = x _ { S } } ^ { s } , \ y _ { T } \sim \mathcal { D } _ { Y | X = x _ { T } } ^ { T } } \big [ \| y _ { S } - y _ { T } \| \big ] , \qquad S _ { p a i r } ( x _ { S } , x _ { T } ) : = W _ { 1 } \big ( \mathcal { D } _ { Y | X = x _ { S } } ^ { s } , \mathcal { D } _ { Y | X = x _ { T } } ^ { T } \big ) ,
$$

and recall that

$$
\Delta = \mathbb { E } _ { ( x _ { S } , x _ { T } ) \sim \gamma ^ { * } } \left[ f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \right] .
$$

Lower bound $\begin{array} { r l r } { \varDelta } & { { } \geq } & { 0 . } \end{array}$ . For each fixed $( x _ { S } , x _ { T } )$ , the product measure $\mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } ~ \otimes ~ \mathcal { D } _ { Y \mid X = x _ { T } } ^ { T }$ belongs to $\Gamma \big ( \mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } , \mathcal { D } _ { Y \mid X = x _ { T } } ^ { T } \big )$ . Since $S _ { p a i r } ( x _ { S } , x _ { T } )$ is the infimum of $\mathbb { E } [ \| y _ { S } - y _ { T } \| ]$ over all such couplings,

$$
S _ { p a i r } ( x _ { S } , x _ { T } ) \leq \mathbb { E } _ { \mathcal { D } _ { Y \mid X = x _ { S } } ^ { S } \otimes \mathcal { D } _ { Y \mid X = x _ { T } } ^ { T } } [ \| y _ { S } - y _ { T } \| ] = f ( x _ { S } , x _ { T } ) .
$$

Thus $f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \geq 0$ pointwise, hence $\varDelta \geq 0$

An upper bound for each pair $( x _ { S } , x _ { T } )$ . Fix $( x _ { S } , x _ { T } )$ and write $P : = { \mathcal { D } } _ { Y | X = x _ { S } } ^ { S }$ and $Q : = { \mathcal { D } } _ { Y \mid X = x _ { T } } ^ { T }$ . Let their means be

$$
m _ { S } ( x _ { S } ) : = \mathbb { E } _ { y \sim P } [ y ] , \qquad m _ { T } ( x _ { T } ) : = \mathbb { E } _ { y \sim Q } [ y ] ,
$$

For independent draws $y _ { S } \sim P$ and $y _ { T } \sim Q$ , the triangle inequality yields

$$
\begin{array} { r } { \| y _ { S } - y _ { T } \| \leq \| y _ { S } - m _ { S } ( x _ { S } ) \| + \| m _ { S } ( x _ { S } ) - m _ { T } ( x _ { T } ) \| + \| m _ { T } ( x _ { T } ) - y _ { T } \| . } \end{array}
$$

Taking expectation gives

$$
f ( x _ { S } , x _ { T } ) \leq \mathbb { E } _ { y \sim P } \big [ \| y - m _ { S } ( x _ { S } ) \| \big ] + \| m _ { S } ( x _ { S } ) - m _ { T } ( x _ { T } ) \| + \mathbb { E } _ { y \sim Q } \big [ \| y - m _ { T } ( x _ { T } ) \| \big ] .
$$

Next we lower bound $S _ { p a i r } ( x _ { S } , x _ { T } ) = W _ { 1 } ( P , Q )$ by the distance between the means. Let $\pi \in \Gamma ( P , Q )$ be any coupling, and denote $m _ { P } : = \mathbb { E } _ { y \sim P } [ y ] , m _ { Q } : = \mathbb { E } _ { y \sim Q } [ y ]$ . Since $\| \cdot \|$ is convex, Jensen’s inequality yields

$$
\begin{array} { r } { { \mathbb E } ( y , y ^ { \prime } ) { \sim } { \pi } \left[ \| y - y ^ { \prime } \| \right] ~ \geq ~ \left\| { \mathbb E } ( y , y ^ { \prime } ) { \sim } { \pi } \left[ y - y ^ { \prime } \right] \right\| . } \end{array}
$$

By the marginal constraints of π, we have $\mathbb { E } _ { ( y , y ^ { \prime } ) \sim \pi } [ y ] = m _ { P }$ and $\mathbb { E } _ { ( y , y ^ { \prime } ) \sim \pi } [ y ^ { \prime } ] = m _ { Q }$ , hence

$$
\left\| \mathbb { E } _ { ( y , y ^ { \prime } ) \sim \pi } [ y - y ^ { \prime } ] \right\| = \| m _ { P } - m _ { Q } \| .
$$

Therefore, for every $\pi \in \Gamma ( P , Q )$

$$
\begin{array} { r } { { \mathbb { E } } _ { ( y , y ^ { \prime } ) \sim \pi } \left[ \left\| y - y ^ { \prime } \right\| \right] \ \geq \ \left\| m _ { P } - m _ { Q } \right\| . } \end{array}
$$

Taking the infimum over $\pi \in \Gamma ( P , Q )$ gives

$$
W _ { 1 } ( P , Q ) = \operatorname* { i n f } _ { \pi \in \Gamma ( P , Q ) } \mathbb { E } _ { ( y , y ^ { \prime } ) \sim \pi } \big [ \| y - y ^ { \prime } \| \big ] \ \ge \ \| m _ { P } - m _ { Q } \| .
$$

Combining the two displays and canceling the middle term yields

$$
f ( x _ { S } , x _ { T } ) - S _ { p a i r } ( x _ { S } , x _ { T } ) \leq \mathbb { E } _ { y \sim P } \big [ \| y - m _ { S } ( x _ { S } ) \| \big ] + \mathbb { E } _ { y \sim Q } \big [ \| y - m _ { T } ( x _ { T } ) \| \big ] .
$$

Integrate over $\gamma ^ { * }$ and relate to irreducible errors. Taking expectation ove $( x _ { S } , x _ { T } ) \sim \gamma ^ { * }$ and using that the marginals of $\gamma ^ { * }$ are $\mathcal { D } _ { X } ^ { S }$ and $\mathcal { D } _ { X } ^ { T }$

$$
\begin{array} { r l } & { \varDelta \leq \mathbb { E } _ { x \sim \mathcal { D } _ { x } ^ { s } } \Big [ \mathbb { E } \big [ \| Y ^ { ( S ) } - m _ { S } ( x ) \| \mid X ^ { ( S ) } = x \big ] \Big ] + \mathbb { E } _ { x \sim \mathcal { D } _ { x } ^ { T } } \Big [ \mathbb { E } \big [ \| Y ^ { ( T ) } - m _ { T } ( x ) \| \mid X ^ { ( T ) } = x \big ] \Big ] . } \end{array}
$$

For any random variable $Z \ge 0 , \mathbb { E } [ Z ] \le \sqrt { \mathbb { E } [ Z ^ { 2 } ] }$ , hence

$$
\begin{array} { r } { \mathbb { E } \big [ \| Y ^ { ( S ) } - m _ { S } ( x ) \| | X ^ { ( S ) } = x \big ] \leq \sqrt { \mathbb { E } \big [ \| Y ^ { ( S ) } - m _ { S } ( x ) \| ^ { 2 } | X ^ { ( S ) } = x \big ] } , } \end{array}
$$

and similarly for the target term. Thus

$$
\begin{array} { r } { \varDelta \leq \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { \delta } } \left[ \sqrt { \mathbb { E } \left[ \left\| Y ^ { ( S ) } - m _ { S } ( x ) \right\| ^ { 2 } \mid X ^ { ( S ) } = x \right] } \right] + \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { T } } \left[ \sqrt { \mathbb { E } \left[ \left\| Y ^ { ( T ) } - m _ { T } ( x ) \right\| ^ { 2 } \mid X ^ { ( T ) } = x \right] } \right] . } \end{array}
$$

Since $\sqrt { \cdot }$ is concave on $\mathbb { R } _ { + }$ , Jensen’s inequality gives

$$
\mathbb { E } _ { x \sim { D } _ { X } ^ { s } } \left[ \sqrt { \mathbb { E } \left[ \Vert Y ^ { ( S ) } - m _ { S } ( x ) \Vert ^ { 2 } \mid X ^ { ( S ) } = x \right] } \right] \leq \sqrt { \mathbb { E } _ { ( x , y ) \sim { D } _ { X _ { Y } } ^ { s } } \left[ \Vert y - m _ { S } ( x ) \Vert ^ { 2 } \right] } ,
$$

$$
\begin{array} { r l } & { \mathbb { E } _ { x \sim \mathcal { D } _ { X } ^ { T } } \left[ \sqrt { \mathbb { E } \left[ \| Y ^ { ( T ) } - m _ { T } ( x ) \| ^ { 2 } \mid X ^ { ( T ) } = x \right] } \right] \leq \sqrt { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { X Y } ^ { T } } \left[ \| y - m _ { T } ( x ) \| ^ { 2 } \right] } . } \end{array}
$$

Finally, under squared loss on $\mathbb { R } ^ { d ^ { \prime } }$ , the minimizer of $\mathbb { E } [ \| Y - g ( x ) \| ^ { 2 } \mid X = x ] { \mathrm { ~ i s ~ } } g ( x ) = \mathbb { E } [ Y \mid X = x ]$ . Therefore the functions $m _ { S } ( x ) { \bar { = } } \mathbb { E } [ Y ^ { ( S ) } | X ^ { ( S ) } = x ]$ and $m _ { T } ( x ) = \mathbb { E } [ Y ^ { ( T ) } | X ^ { ( T ) } = x ]$ achieve the infima of $\operatorname { I } ( \mathcal { D } _ { X Y } ^ { S } )$ and $\mathrm { I } ( \mathcal { D } _ { X Y } ^ { T } )$ , i.e.,

$$
\begin{array} { r l r } { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { X Y } ^ { s } } \left[ \| y - m _ { S } ( x ) \| ^ { 2 } \right] = \mathbf { I } ( \mathcal { D } _ { X Y } ^ { s } ) , } & { \quad \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { X Y } ^ { T } } \left[ \| y - m _ { T } ( x ) \| ^ { 2 } \right] = \mathbf { I } ( \mathcal { D } _ { X Y } ^ { T } ) . } \end{array}
$$

Putting the above bounds together yields

$$
\begin{array} { r } { \Delta \leq \sqrt { \mathrm { I } ( \mathcal { D } _ { X Y } ^ { S } ) } + \sqrt { \mathrm { I } ( \mathcal { D } _ { X Y } ^ { T } ) } , } \end{array}
$$

which completes the proof.

## D. Analysis of Lipschitz Constants: Results and Proofs

## D.1. Lipschitz Constant of the Sigmoid Function

Proposition D.1 (Lipschitz constant of the sigmoid). Let $\sigma : \mathbb { R } $ R be the sigmoid function

$$
\sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } .
$$

Then σ is globally $\scriptstyle { \frac { 1 } { 4 } } - L i p s c h i t z o n$ R.

Proof. We first compute the derivative:

$$
\sigma ^ { \prime } ( x ) = \frac { d } { d x } \Bigl ( 1 + e ^ { - x } \Bigr ) ^ { - 1 } = \frac { e ^ { - x } } { \bigl ( 1 + e ^ { - x } \bigr ) ^ { 2 } } .
$$

Using $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } } \end{array}$ and $\begin{array} { r } { 1 - \sigma ( x ) = \frac { e ^ { - x } } { 1 + e ^ { - x } } } \end{array}$ , we can rewrite

$$
\sigma ^ { \prime } ( x ) = \sigma ( x ) { \big ( } 1 - \sigma ( x ) { \big ) } .
$$

Let $u = \sigma ( x ) \in ( 0 , 1 )$ . Then $\sigma ^ { \prime } ( x ) = u ( 1 - u )$ , and for all $u \in [ 0 , 1 ]$

$$
u ( 1 - u ) = u - u ^ { 2 } \ \leq \ \operatorname* { m a x } _ { t \in [ 0 , 1 ] } ( t - t ^ { 2 } ) = \frac 1 4 ,
$$

where the maximum is attained at $\begin{array} { r } { t = \frac { 1 } { 2 } } \end{array}$ . Hence $\begin{array} { r } { 0 < \sigma ^ { \prime } ( x ) \le \frac { 1 } { 4 } } \end{array}$ for all $x .$

Now fix any $x , y \in \mathbb { R }$ . Since σ is differentiable on R (hence continuous), by the mean value theorem, there exists c between x and y such that

$$
\sigma ( x ) - \sigma ( y ) = \sigma ^ { \prime } ( c ) ( x - y ) .
$$

Taking absolute values and using $\begin{array} { r } { \sigma ^ { \prime } { \left( c \right) } \leq \frac { 1 } { 4 } } \end{array}$ yields

$$
| \sigma ( x ) - \sigma ( y ) | = | \sigma ^ { \prime } ( c ) | | x - y | \leq { \frac { 1 } { 4 } } | x - y | ,
$$

so σ is <sup>1</sup><sub>4</sub> -Lipschitz.

## D.2. Lipschitz Constant of the Logistic Regression

Proposition D.2 (Lipschitz constant of logistic regression under $\ell _ { p } / \ell _ { q } )$ . Fix $w \in \mathbb { R } ^ { d }$ and $b \in \mathbb { R } .$ . Define

$$
h _ { w , b } ( x ) \ : = \ : \sigma ( w ^ { \top } x + b ) , \sigma ( t ) = \frac { 1 } { 1 + e ^ { - t } } .
$$

Let $p \in [ 1 , \infty ]$ and let $q \in [ 1 , \infty ]$ be its Holder conjugate, i.e.,¨ $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ . Then $h _ { w , b }$ is globally $\frac { \| \boldsymbol { w } \| _ { q } } { 4 }$ -Lipschitz with respect $t o \parallel \cdot \parallel _ { p } ,$ namely for all $x , x ^ { \prime } \in \mathbb { R } ^ { d }$

$$
\big | h _ { w , b } ( x ) - h _ { w , b } ( x ^ { \prime } ) \big | \ \leq \ \frac { \| w \| _ { q } } { 4 } \| x - x ^ { \prime } \| _ { p } .
$$

In particular, under $\| \cdot \| _ { 2 } ,$ the Lipschitz constant equals $\| w \| _ { 2 } / 4 .$

Proof. For any $x , x ^ { \prime } \in \mathbb { R } ^ { d }$ , apply the mean value theorem to σ to obtain some ξ between $\boldsymbol { w } ^ { \top } \boldsymbol { x } + \boldsymbol { b }$ and $w ^ { \top } x ^ { \prime } + b$ such that

$$
h _ { w , b } ( \boldsymbol { x } ) - h _ { w , b } ( \boldsymbol { x } ^ { \prime } ) = \sigma ^ { \prime } ( \boldsymbol { \xi } ) w ^ { \top } ( \boldsymbol { x } - \boldsymbol { x } ^ { \prime } ) .
$$

Thus

$$
\big | h _ { w , b } ( x ) - h _ { w , b } ( x ^ { \prime } ) \big | \leq | \sigma ^ { \prime } ( \xi ) | \big | w ^ { \top } ( x - x ^ { \prime } ) \big | .
$$

By Proposition D.1, we have $\begin{array} { r } { | \sigma ^ { \prime } ( \xi ) | \le \frac { 1 } { 4 } } \end{array}$ . By Holder’s inequality with conjugate exponents ¨ $( p , q )$

$$
\begin{array} { r } { | \boldsymbol { w } ^ { \intercal } ( \boldsymbol { x } - \boldsymbol { x } ^ { \prime } ) | \leq \| \boldsymbol { w } \| _ { q } \| \boldsymbol { x } - \boldsymbol { x } ^ { \prime } \| _ { p } . } \end{array}
$$

Combining the two inequalities yields

$$
\left| h _ { w , b } ( x ) - h _ { w , b } ( x ^ { \prime } ) \right| \leq \frac { 1 } { 4 } \| w \| _ { q } \| x - x ^ { \prime } \| _ { p } ,
$$

which proves the claim.

## D.3. Lipschitz Constant of the Linear Multi-class Classifier

Proposition D.3.1 (Lipschitz constant of the linear classifier under $\| \cdot \| _ { 2 } ) .$ . Let $W \in \mathbb { R } ^ { N \times d }$ and $b \in \mathbb { R } ^ { N }$ . Define

$$
f ( x ) = \mathrm { s o f t m a x } ( W x + b ) \in \Delta ^ { N - 1 } , \qquad x \in \mathbb { R } ^ { d } .
$$

Here $\Delta ^ { N - 1 } : = \{ p \in \mathbb { R } ^ { N } : p _ { i } \geq 0 , \forall i , \sum _ { i = 1 } ^ { N } p _ { i } = 1 \}$ denotes the probability simplex. Let $\boldsymbol { w _ { i } ^ { \intercal } }$ denote the i-th row ofW and define

$$
D ( W ) : = \operatorname* { m a x } _ { i , j \in [ N ] } \| w _ { i } - w _ { j } \| _ { 2 } .
$$

Then $f$ is Lipschitz from $( \mathbb { R } ^ { d } , \| \cdot \| _ { 2 } ) t o ( \mathbb { R } ^ { N } , \| \cdot \| _ { 2 } )$ and its optimal Lipschitz constant equals

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla f ( x ) \| _ { 2 } = { \frac { 1 } { 2 { \sqrt { 2 } } } } D ( W ) ,
$$

where $\| \cdot \| _ { 2 }$ denotes the matrix operator norm induced by the vector $\| \cdot \| _ { 2 }$ norm.

Proof. For $z \in \mathbb { R } ^ { N }$ , write $p = \mathrm { s o f t m a x } ( z ) \in \Delta ^ { N - 1 }$ . The Jacobian of softmax is

$$
J ( p ) = \nabla _ { z } \operatorname { s o f t m a x } ( z ) = \operatorname { D i a g } ( p ) - p p ^ { \top } ,
$$

where $\mathrm { D i a g } ( p )$ is the diagonal matrix with diagonal entries $p _ { 1 } , \dotsc , p _ { N } . \mathrm { ~ B y ~ }$ the chain rule, for $p = \mathrm { s o f t m a x } ( W x + b )$

$$
\nabla f ( x ) = J ( p ) W .
$$

Therefore,

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla f ( x ) \| _ { 2 } = \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W \| _ { 2 } .\tag{16}
$$

Fix $p \in \Delta ^ { N - 1 }$ and a unit vector $u \in \mathbb { R } ^ { d }$ with $\| u \| _ { 2 } = 1$ . Let

$$
v : = W u \in \mathbb { R } ^ { N } , \qquad \mu : = p ^ { \top } v .
$$

and $\mu$ represents the mean of v under the probability distribution $p .$ We have

$$
J ( p ) v = { \big ( } { \mathrm { D i a g } } ( p ) - p p ^ { \top } { \big ) } v = { \mathrm { D i a g } } ( p ) { \big ( } v - \mu \mathbf { 1 } { \big ) } ,
$$

hence

$$
\| J ( p ) W u \| _ { 2 } ^ { 2 } = \| J ( p ) v \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 } ( v _ { i } - \mu ) ^ { 2 } .\tag{17}
$$

Now fix $v \in \mathbb { R } ^ { N }$ and consider the quantity

$$
\Phi ( v ) : = \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 } ( v _ { i } - p ^ { \top } v ) ^ { 2 } .
$$

Indeed, for any $p \in \Delta ^ { N - 1 }$

$$
\sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 } ( v _ { i } - \mu ) ^ { 2 } \leq \Big ( \operatorname* { m a x } _ { i } p _ { i } \Big ) \sum _ { i = 1 } ^ { N } p _ { i } ( v _ { i } - \mu ) ^ { 2 } = \Big ( \operatorname* { m a x } _ { i } p _ { i } \Big ) \mathrm { V a r } _ { p } ( v ) ,
$$

where $\begin{array} { r } { \operatorname { V a r } _ { p } ( v ) : = \sum _ { i } p _ { i } ( v _ { i } - \mu ) ^ { 2 } } \end{array}$ , which represents the variance of v under the probability distribution p. If p is supported on two values $m : = \operatorname* { m i n } _ { i } v _ { i }$ and $M : = \operatorname* { m a x } _ { i } v _ { i }$ with masses α and $1 - \alpha$ , let $r : = M - m$ , then

$$
\sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 } ( v _ { i } - \mu ) ^ { 2 } = 2 \alpha ^ { 2 } ( 1 - \alpha ) ^ { 2 } r ^ { 2 } \le \frac { r ^ { 2 } } { 8 } ,
$$

with equality at $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ . Since any mass placed at intermediate values of v cannot increase the range-based extremum, we obtain $\Phi ( v ) = r ^ { 2 } / 8$ , and the optimal p is supported on indices attaining M and $m$

Combining (17), for any unit u,

$$
\operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W u \| _ { 2 } = \frac { 1 } { 2 \sqrt { 2 } } \Big ( \operatorname* { m a x } _ { i } ( W u ) _ { i } - \operatorname* { m i n } _ { j } ( W u ) _ { j } \Big ) .\tag{18}
$$

Finally, maximize the range over u. Writing $( W u ) _ { i } = w _ { i } ^ { \top }$ u, we have

$$
\operatorname* { m a x } _ { i } ( W u ) _ { i } - \operatorname* { m i n } _ { j } ( W u ) _ { j } = \operatorname* { m a x } _ { i , j } ( w _ { i } - w _ { j } ) ^ { \top } u .
$$

Thus,

$$
\operatorname* { s u p } _ { \| \boldsymbol { u } \| _ { 2 } = 1 } \left( \operatorname* { m a x } _ { i } ( W \boldsymbol { u } ) _ { i } - \operatorname* { m i n } _ { j } ( W \boldsymbol { u } ) _ { j } \right) = \operatorname* { m a x } _ { i , j } _ { \| \boldsymbol { u } \| _ { 2 } = 1 } \operatorname* { s u p } _ { ( w _ { i } - w _ { j } ) } \top _ { \boldsymbol { u } } = \operatorname* { m a x } _ { i , j } \| w _ { i } - w _ { j } \| _ { 2 } = D ( W ) .
$$

Plugging this into (18) and then into (16) yields

$$
\begin{array} { r l } & { \quad \displaystyle \operatorname* { s u p } _ { x \in \mathbb R ^ { d } } \| \nabla f ( x ) \| _ { 2 } } \\ & { = \displaystyle \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W \| _ { 2 } } \\ & { = \displaystyle \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \displaystyle \operatorname* { s u p } _ { \| u \| _ { 2 } = 1 } \| J ( p ) W u \| _ { 2 } } \\ & { = \displaystyle \operatorname* { s u p } _ { \| u \| _ { 2 } = 1 } \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W u \| _ { 2 } } \\ & { = \displaystyle \frac { 1 } { 2 \sqrt { 2 } } p ( W ) } \end{array}
$$

This completes the proof.

Proposition D.3.2 (Lipschitz constant of the linear classifier under $\| \cdot \| _ { 2 } \to \| \cdot \| _ { 1 } )$ . Let $W \in \mathbb { R } ^ { N \times d }$ and $b \in \mathbb { R } ^ { N }$ . Define

$$
f ( x ) = \mathrm { s o f t m a x } ( W x + b ) \in \Delta ^ { N - 1 } , \qquad x \in \mathbb { R } ^ { d } .
$$

Here $\Delta ^ { N - 1 } : = \{ p \in \mathbb { R } ^ { N } : p _ { i } \geq 0 , \forall i , \sum _ { i = 1 } ^ { N } p _ { i } = 1 \}$ denotes the probability simplex. Let $\boldsymbol { w _ { i } ^ { \intercal } }$ denote the i-th row of W and define

$$
D ( W ) : = \operatorname* { m a x } _ { i , j \in [ N ] } \| w _ { i } - w _ { j } \| _ { 2 } .
$$

Then $f$ is Lipschitz from $( \mathbb { R } ^ { d } , \| \cdot \| _ { 2 } ) t o ( \mathbb { R } ^ { N } , \| \cdot \| _ { 1 } )$ , and its optimal Lipschitz constant equals

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla f ( x ) \| _ { 2  1 } = \frac { 1 } { 2 } D ( W ) ,
$$

where $\| \cdot \| _ { 2 \to 1 }$ denotes the operator norm induced by $\| \cdot \| _ { 2 }$ on the input and $\| \cdot \| _ { 1 }$ on the output.

Proof. For $z \in \mathbb { R } ^ { N }$ , write $p = \mathrm { s o f t m a x } ( z ) \in \Delta ^ { N - 1 }$ and recall

$$
J ( p ) = \nabla _ { z } \operatorname { s o f t m a x } ( z ) = \operatorname { D i a g } ( p ) - p p ^ { \top } .
$$

By the chain rule, for $p = \mathrm { s o f t m a x } ( W x + b )$ we have

$$
\nabla f ( x ) = J ( p ) W .
$$

Hence

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla f ( x ) \| _ { 2 \to 1 } = \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W \| _ { 2 \to 1 } .\tag{19}
$$

Fix $p \in \Delta ^ { N - 1 }$ and $v \in \mathbb { R } ^ { N }$ . Let $\mu : = p ^ { \top } i$ v. Using

$$
J ( p ) v = ( { \mathrm { D i a g } } ( p ) - p p ^ { \top } ) v = { \mathrm { D i a g } } ( p ) { \big ( } v - \mu \mathbf { 1 } { \big ) } ,
$$

we obtain

$$
\| J ( p ) v \| _ { 1 } = \sum _ { i = 1 } ^ { N } p _ { i } | v _ { i } - \mu | .\tag{20}
$$

The right-hand side is the mean absolute deviation of v under the probability distribution $p .$

Fix $v \in \mathbb { R } ^ { N }$ and define $m : = \mathrm { m i n } _ { i } v _ { i } , M : = \mathrm { m a x } _ { i } v _ { i }$ , and $r : = M - m$ . We claim that

$$
\operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \sum _ { i = 1 } ^ { N } p _ { i } | v _ { i } - p ^ { \top } v | = \frac { r } { 2 } .\tag{21}
$$

Upper bound. Let $\mu = p ^ { \intercal } v$ . By Cauchy–Schwarz,

$$
\sum _ { i = 1 } ^ { N } p _ { i } | v _ { i } - \mu | \leq \Big ( \sum _ { i = 1 } ^ { N } p _ { i } \Big ) ^ { 1 / 2 } \Big ( \sum _ { i = 1 } ^ { N } p _ { i } ( v _ { i } - \mu ) ^ { 2 } \Big ) ^ { 1 / 2 } = \sqrt { \mathrm { V a r } _ { p } ( v ) } .
$$

By Popoviciu’s inequality for bounded random variables, $\mathrm { V a r } _ { p } ( v ) \leq r ^ { 2 } / 4$ , hence

$$
\sum _ { i = 1 } ^ { N } p _ { i } | v _ { i } - \mu | \leq \frac { r } { 2 } .
$$

Lower bound (attainability). Let $i _ { \mathrm { m a x } } \in$ arg max<sub>i</sub> $v _ { i }$ and $i _ { \mathrm { m i n } } \in$ arg min<sub>i</sub> $v _ { i }$ , and take

$$
p = \frac { 1 } { 2 } ( e _ { i _ { \operatorname* { m a x } } } + e _ { i _ { \operatorname* { m i n } } } ) .
$$

Here $e _ { i } \in \mathbb { R } ^ { N }$ denotes the i-th standard basis vector, $\mathrm { i . e . , } ( e _ { i } ) _ { j } = { \bf 1 } \{ j = i \}$ . Then $\mu = ( M + m ) / 2 ,$ , and

$$
\sum _ { i = 1 } ^ { N } p _ { i } | v _ { i } - \mu | = \frac 1 2 \Big | M - \frac { M + m } { 2 } \Big | + \frac 1 2 \Big | m - \frac { M + m } { 2 } \Big | = \frac { r } { 2 } .
$$

This proves (21).

Combining (20) and (21), for any $u \in \mathbb { R } ^ { d }$ with $\| u \| _ { 2 } = 1$

$$
\operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W u \| _ { 1 } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } _ { i } ( W u ) _ { i } - \operatorname* { m i n } _ { j } ( W u ) _ { j } \Big ) .\tag{22}
$$

Writing $( W u ) _ { i } = w _ { i } ^ { \top } ,$ u, we have

$$
\operatorname* { m a x } _ { i } ( W u ) _ { i } - \operatorname* { m i n } _ { j } ( W u ) _ { j } = \operatorname* { m a x } _ { i , j } ( w _ { i } - w _ { j } ) ^ { \top } u .
$$

Therefore,

$$
\operatorname* { s u p } _ { \| { \boldsymbol u } \| _ { 2 } = 1 } \left( \operatorname* { m a x } _ { i } ( W { \boldsymbol u } ) _ { i } - \operatorname* { m i n } _ { j } ( W { \boldsymbol u } ) _ { j } \right) = \operatorname* { m a x } _ { i , j } _ { \| { \boldsymbol u } \| _ { 2 } = 1 } \operatorname* { s u p } _ { ( w _ { i } - w _ { j } ) } \top _ { \boldsymbol u } = \operatorname* { m a x } _ { i , j } \| { \boldsymbol w } _ { i } - { \boldsymbol w } _ { j } \| _ { 2 } = D ( W ) .
$$

From (19) and (22),

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { x \in \mathbb R ^ { d } } \| \nabla f ( x ) \| _ { 2  1 } } \\ { \displaystyle = \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } \| u \| _ { 2 } = 1 } \| J ( p ) W u \| _ { 1 } } \\ { \displaystyle = \operatorname* { s u p } _ { y \in \Delta ^ { N - 1 } \| u \| _ { 2 } = 1 } \| J ( p ) W u \| _ { 1 } } \\ { \displaystyle = \frac 1 2 \operatorname* { s u p } _ { p \in \Delta ^ { N - 1 } } \| J ( p ) W u \| _ { 1 } } \\ { \displaystyle = \frac 1 2 \operatorname* { s u p } _ { \| u \| _ { 2 } = 1 } \Big ( \operatorname* { m a x } _ { i } ( W u ) _ { i } - \operatorname* { m i n } _ { j } ( W u ) _ { j } \Big ) } \\ { \displaystyle = \frac 1 2 D ( W ) . } \end{array}
$$

This completes the proof.

□

## D.4. Lipschitz Constant of MLPs

Based on the previous theory for the Lipschitz constant of MLPs by Fazlyab et al. (2020), we obtain the following direct result:

Proposition D.4 (SDP-certified Lipschitz constant for MLPs). Consider the ℓ-hidden-layer MLP

$$
\begin{array} { r l } { x _ { 0 } = x \in \mathbb { R } ^ { n _ { 0 } } , \quad } & { x _ { k } = \phi ( W _ { k - 1 } x _ { k - 1 } + b _ { k - 1 } ) \in \mathbb { R } ^ { n _ { k } } \left( k = 1 , \ldots , \ell \right) , } \\ & { \quad \quad f ( x ) = W _ { \ell } x _ { \ell } + b _ { \ell } \in \mathbb { R } ^ { m } , } \end{array}
$$

where $\boldsymbol { \phi } ( z ) = [ \varphi ( z _ { 1 } ) , \dots , \varphi ( z _ { n _ { k } } ) ] ^ { \intercal }$ acts componentwise and the scalar activation $\varphi$ is slope-restricted on $\left[ 0 , \beta \right] ( i . e .$ $\begin{array} { r } { 0 \leq \frac { \varphi ( u ) - \varphi ( v ) } { u - v } \leq \beta f o r a l l u \neq v ) . } \end{array}$

Let $N : = n _ { 0 } + n _ { 1 } + \cdot \cdot \cdot + n _ { \ell }$ and $n : = n _ { 1 } + \cdots + n _ { \ell } .$ . Define the matrices $A \in \mathbb { R } ^ { n \times N }$ and $B \in \mathbb { R } ^ { n \times N } b \}$

$$
A = \left[ { \begin{array} { c c c c c } { W _ { 0 } } & { 0 } & { \cdots } & { 0 } & { 0 } \\ { 0 } & { W _ { 1 } } & { \cdots } & { 0 } & { 0 } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { 0 } & { 0 } & { \cdots } & { W _ { \ell - 1 } } & { 0 } \end{array} } \right] , \qquad B = \left[ { \begin{array} { c c c c c } { 0 } & { I _ { n _ { 1 } } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 0 } & { I _ { n _ { 2 } } } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { 0 } & { 0 } & { \cdots } & { I _ { n _ { \ell } } } \end{array} } \right] .
$$

Let the decision variable $T _ { n } \in \mathbb { S } ^ { n }$ be block-diagonal across layers:

$$
T _ { n } = { \mathrm { b l k d i a g } } ( T _ { n _ { 1 } } , \ldots , T _ { n _ { \ell } } ) , \qquad T _ { n _ { k } } = \mathrm { d i a g } ( t _ { k , 1 } , \ldots , t _ { k , n _ { k } } ) , \quad t _ { k , i } \geq 0 .
$$

Consider the SDP (with variable $H \geq 0 )$

$$
\begin{array} { r l } { \underset { \{ t _ { k , i } \} , H } { \operatorname* { m i n } } } & { H } \\ { \mathrm { s . t . } } & { M \succeq 0 , } \\ & { M = - \bigg ( M _ { L } ( H ) + M _ { A } ( T _ { n } ) \bigg ) , } \end{array}
$$

$$
M _ { L } ( H ) = \left[ \begin{array} { c c c c } { - H I _ { n _ { 0 } } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 0 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { 0 } & { \cdots } & { W _ { \ell } ^ { \top } W _ { \ell } } \end{array} \right] \in \mathbb { S } ^ { N } ,
$$

$$
M _ { A } ( T _ { n } ) = \left[ \boldsymbol { \cal B } \right] ^ { \top } \left[ \begin{array} { c c } { 0 } & { \beta T _ { n } } \\ { \beta T _ { n } } & { - 2 T _ { n } } \end{array} \right] \left[ \boldsymbol { \cal B } \right] \in \mathbb { S } ^ { N } .
$$

Let $H ^ { \star }$ be the optimal value and define $L _ { \mathrm { S D P } } : = \sqrt { H ^ { \star } }$ . Then L<sub>SDP</sub> is a global ∥ · ∥<sub>2</sub>-Lipschitz constant of f, i.e.,

$$
\| f ( x ) - f ( y ) \| _ { 2 } \leq L _ { \mathrm { S D P } } \| x - y \| _ { 2 } , \qquad \forall x , y \in \mathbb { R } ^ { n _ { 0 } } .
$$

## E. Additional Experiments

## E.1. Experiments on Estimator Sensitivity

In this section, we conduct a sensitivity analysis of the proposed estimators $\hat { S } _ { C o v }$ and $\hat { S } _ { C p t }$ with respect to the parameter $\beta .$ We take the ”-90%” domain of ColoredMNIST as the target domain and regard the mixture of the other two domains as the source domain. We train the model on the source domain using ERM with the best hyperparameters in DomainBed. At the last checkpoint, we store all representation-label pairs from both the source and target domains. For each $\beta ,$ we randomly sample 10,000 pairs from each domain, run the DataShifts algorithm to obtain the estimated results, repeat this procedure 50 times, and report the average. The results are shown below:

Table 2. Sensitivity analysis with respect to $\beta .$
<table><tr><td> $\beta$ </td><td>0.001</td><td>0.002</td><td>0.005</td><td>0.01</td><td>0.02</td><td>0.05</td><td>0.1</td><td>0.2</td></tr><tr><td> $\hat { S } _ { C o v }$ </td><td>0.5081</td><td>0.5080</td><td>0.5078</td><td>0.5074</td><td>0.5066</td><td>0.5041</td><td>0.4995</td><td>0.4916</td></tr><tr><td> $\hat { S } _ { C p t }$ </td><td>1.4456</td><td>1.4458</td><td>1.4467</td><td>1.4478</td><td>1.4501</td><td>1.4567</td><td>1.4674</td><td>1.4895</td></tr><tr><td>Time (s)</td><td>15.12</td><td>14.09</td><td>12.62</td><td>11.60</td><td>10.54</td><td>9.09</td><td>8.03</td><td>6.83</td></tr></table>

Across magnitude changes in $\beta ,$ the coefficients of variation of $\hat { S } _ { C o v }$ and $\hat { S } _ { C p t }$ are 1.076% and 0.990%, so both estimators are stable for small $\beta .$ Since entropic OT gets faster as $\beta$ grows, we use $\beta = 0 . 2$ in the main experiments to balance speed and accuracy.

## E.2. Experiments on the Bias under Stochastic Labeling

We test the bias of our $\gamma ^ { * } { \mathrm { - } } \Upsilon | \mathrm { X }$ shift estimator $\hat { S } _ { C p t }$ on the following stochastic-labeling synthetic data. In both source and target domains, covariates are drawn from a 10-dim standard normal distributions. At each $x ,$ scalar labels are sampled from normal distribution $\mathcal { N } ( \| x \| _ { 2 } , \sigma ^ { 2 } )$ in the source and $\mathcal { N } ( \| x \| _ { 2 } + 1 . 0 , \sigma ^ { 2 } )$ in the target. Thus, the true $\mathrm { Y } | \mathrm { X }$ shift $S _ { C p t } ^ { \gamma ^ { * } } = 1 . 0$ and standard deviation σ controls the noise. By Proposition 4.7, the irreducible errors of both domains are $\sigma ^ { 2 }$ , and the bias $\Delta \geq 0$ of our estimator should be bounded by 2σ. $\hat { \Delta } = \hat { S } _ { C p t } - S _ { C p t } ^ { \gamma ^ { * } }$ is an estimate of the bias $\Delta .$ . For each $\sigma ,$ we randomly sample 10,000 points from each domain, and run DataShifts algorithm with $\beta = 0 . 2 .$ , repeat this procedure 50 times, and report the average $\hat { \Delta }$ . The results are shown below:

Table 3. Bias of $\hat { S } _ { C p t }$ under different noise levels σ.
<table><tr><td>σ</td><td>0.01</td><td>0.1</td><td>0.3</td><td>0.5</td><td>1.0</td><td>2.0</td><td>5.0</td><td>10.0</td></tr><tr><td>Irreducible error bound (2σ)</td><td>0.02</td><td>0.2</td><td>0.6</td><td>1.0</td><td>2.0</td><td>4.0</td><td>10.0</td><td>20.0</td></tr><tr><td>Bias ()</td><td>0.0036</td><td>0.0389</td><td>0.0702</td><td>0.1440</td><td>0.4865</td><td>1.4524</td><td>4.7314</td><td>10.3467</td></tr></table>

When the noise $\sigma$ is small, the bias $\hat { \Delta }$ is also very small, which means our estimator $\hat { S } _ { C p t }$ stays close to the true $\mathrm { Y } | \mathrm { X }$ shift $S _ { C p t } ^ { \gamma ^ { * } } = 1 . 0$ . Even when the noise in both domains reaches half of the true Y|X shift $( \sigma = 0 . 5 )$ , the bias $\hat { \Delta }$ remains small, and the $\hat { S } _ { C p t }$ still does not notably overestimate. As σ grows far beyond 1.0, Y|X shift becomes less identifiable. The bias increases and overestimation appears, but it stays below the bound set by irreducible error, as proved in Proposition 4.7.