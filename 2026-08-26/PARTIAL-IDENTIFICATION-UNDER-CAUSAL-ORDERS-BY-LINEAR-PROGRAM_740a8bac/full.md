# PARTIAL IDENTIFICATION UNDER CAUSAL ORDERS BY LINEAR PROGRAMMING

A PREPRINT

Eric Rossetto and Alessandro Antonucci

Istituto Dalle Molle di Studi sull’Intelligenza Artificiale (IDSIA)   
Scuola Universitaria Professionale della Svizzera Italiana (SUPSI) Lugano, Switzerland {eric.rossetto, alessandro.antonucci}@supsi.ch

## ABSTRACT

Non-parametric (partial) identification of counterfactual queries typically relies on a fully specified causal graph. Motivated by settings with incomplete domain knowledge, we challenge this requirement by leveraging structural assumptions that are inherently implied by the query itself. We show that any counterfactual inquiry induces a, mostly partial, topological ordering over relevant variables, which, in turn, enables an explicit query parametrisation reducing the identification task to a linear program. This allows bounding arbitrary counterfactual and nested counterfactual queries. Our work can be viewed as a generalisation of the classical bounding framework of Tian and Pearl (2000), originally developed for probabilities of causation. We also prove the tightness of our bounds by constructing structural causal models that attain the bounds whilst being compatible with both the observed data and the query-implied order. To assess both the generality and practical utility of the proposed bounding procedure, we revisit several case studies from the literature, demonstrating how the derived bounds can be used to yield informative insights even in the absence of an input causal graph.

Keywords Structural Causal Models · Partial Identifiability · Linear Programming

## 1 Introduction

Causal reasoning deepens our understanding of the effects of policies, actions, and treatments in domains that require accountable decision-making, such as algorithmic fairness, personalised medicine, economics, and the broader social sciences. Quantifying such causal inquiries demands a rigorous mathematical foundation. The work of Pearl (2009) formalised the systems under study using structural causal models (SCMs), which elegantly encode prior causal beliefs via directed graphs (Pearl, 1995). Ideally, these structural assumptions allow causal queries to be uniquely computed—formally identified—from the available set of observations. In many practical settings, however, the available assumptions are insufficient for point identification, necessitating partial identification: computing a range within which the target quantity is guaranteed to lie.

Early work by Manski (1990) derived closed-form bounds for simple treatment effect estimation without relying on structural assumptions. Similarly, Tian and Pearl (2000) formulated a linear program to compute sharp bounds on the probabilities of causation (Pearl, 1999) from experimental and observational data under either general or mild assumptions. Whilst weak structural assumptions typically yield substantially wide bounds, assuming a fully specified causal graph might produce tighter intervals. Graph-based reductions to linear programs were first developed for the instrumental variable setting (Balke and Pearl, 1994a,b) and later extended to generalised instrumental constraints (Sachs et al., 2023). For general graphical assumptions, partial identification is reduced to a polynomial program via canonicalisation, though the exponential growth of the resulting exogenous state space ultimately necessitates approximate methods (Zhang et al., 2022; Duarte et al., 2024).

We revisit partial identification in the spirit of Tian and Pearl (2000), whose bounding method did not require a specification of a causal graph; instead, their proposal relied on a direct parametrisation of the joint counterfactual distribution. To extend this framework to arbitrary queries, we leverage the minimal structural assumptions inherently implied by the query itself, reducing the identification task to a linear program without imposing additional structural commitments. Our specific contributions are as follows: (i) we reduce partial identification under a total order of variables to a linear program, capable of handling arbitrary counterfactual, including nested, queries; (ii) we prove that the resulting bounds are tight by constructing witnessing structural causal models that attain them; (iii) we prove that the order-dependent optimisation task is invariant under both the marginalisation of unqueried variables and the specific choice of total order (compatible with the query); and (iv) we show that the linear program admits a lifted reformulation by exploiting symmetries induced by the observations and the query’s logical evaluations. The presented results define an automated framework that enables the bounding of arbitrary queries without the need for explicit graphical assumptions. It is important to emphasise that graph-based methods generally yield narrower bounds by exploiting conditional independencies; in contrast, we pose ourselves in a regime of incomplete domain knowledge, abstaining from such graphical commitments.

After introducing the necessary background in Sect. 2, Sect. 3 reduces partial identification to a linear programme, which Sect. 4 extends to accommodate conditional queries, nested counterfactuals, and structural constraints. Sect. 5 demonstrates practical case studies, followed by conclusions in Sect. 6; proofs, additional experiments and discussions are in Apps. A, B and C, respectively.

## 2 Background

Notation. Let us first review some necessary background on causality. We denote variables by capital letters, whilst small and calligraphic letters are used instead for the states and the sample spaces. Thus, $x \in \mathcal { X }$ is a state of X. We consider discrete variables only. Bold is used for sets of variables. The indicator function that is one when its argument is true and zero otherwise is denoted as · , whilst | · | is the cardinality of the set in the argument.

Structural Causal Models (SCMs). Following Pearl (2009), an SCM $\mathcal { M } : = \langle V , U , \mathcal { F } , P ( U ) \rangle$ is such that V is its set of endogenous variables, U the set of exogenous variables distributed according to $P ( U )$ , and $\mathcal { F }$ a set $\{ f _ { V } \} _ { V \in V }$ of structural equations (SEs). The SE of each $V \in V$ determines $v = f _ { V } ( \mathrm { p a } _ { V } , \pmb { u } _ { V } )$ , where $\operatorname { p a } _ { V } \in \operatorname { P a } _ { V } \subseteq { \dot { V } } { \dot { \cup } } { \dot { \{ V \} } }$ and $U _ { V } \subseteq U$ are, respectively, the endogenous and the exogenous inputs of the SE. We may view an SE as a collection of deterministic mechanisms, or mappings, from input configurations to output states. $\operatorname { I f } { \dot { P } } ( U )$ is unavailable, $M : = \langle V , U , { \mathcal { F } } \rangle$ is termed partially specified SCM (PSCM).

Both SCM M and PSCM M induce a directed graph G whose nodes represent variables and an edge $( X , Y )$ exists iff $X \in { \mathrm { P a } } _ { Y }$ . We restrict our analysis to recursive SCMs, i.e., such that $\mathcal { G }$ is acyclic. We adopt a kinship terminology (e.g., parents) to present graphical relations between variables. The notation $\mathbb { M } _ { M }$ is finally used for the set of all the SCMs that share the $\mathrm { S E s } ^ { \prime }$ signature, i.e., parent and child variables, with M.

Interventions. In an SCM M, $P ( U )$ and $\mathcal { F }$ induce a joint observational distribution $P _ { \mathcal { M } } ( V )$ . An SCM natural regime can be modified by acting on its SEs. For an arbitrary set $X \subseteq V$ , an intervention yields a new model $\mathcal { M } _ { x }$ $( \mathrm { i . e . }$ , a sub-model) by replacing the SEs of X with constant assignments $X  x ,$ whilst keeping $P ( U )$ and all other SEs unchanged. The resulting, interventional, distribution is denoted as $P _ { \mathcal { M } _ { x } } ( V \setminus X )$ . For any $\bar { Y } \in V$ , the potential response $Y _ { \pmb { x } } ( \pmb { u } )$ denotes the value of Y in $\mathcal { M } _ { x }$ given $U = { \pmb u }$ . A counterfactual variable (or potential outcome) $Y _ { x }$ is the variable induced by $Y _ { \pmb { x } } ( \pmb { u } )$ when $\mathbf { } u \sim P ( U )$

Counterfactuals. Following the logical formalisation of a PSCM by Halpern (2000), we define an atomic proposition as the assignment $Y _ { x } = y$ (for short, $y _ { \pmb { x } } )$ , where $Y \in V , X \subseteq V$ and $y \in \mathcal { V }$ . A counterfactual event $\gamma$ is a conjunction of k atomic propositions, $\gamma \equiv y _ { \pmb { x } ^ { ( 1 ) } } ^ { ( 1 ) } \wedge \cdots \wedge y _ { \pmb { x } ^ { ( k ) } } ^ { ( k ) }$ . A counterfactual query $P _ { \mathcal { M } } ( \gamma )$ asks for the probability of γ with respect to an SCM M. This requires the simultaneous computation of potential responses of multiple sub-models $\{ \bar { \mathcal { M } } _ { \pmb { x } ^ { ( i ) } } \} _ { i = 1 } ^ { k }$ , coupled by the shared P(U), i.e.,

$$
P _ { \mathbf { \mathcal { M } } } ( \gamma ) = \sum _ { { \boldsymbol { \mathbf { u } } } \in \mathcal { U } } \left[ \bigwedge _ { i = 1 } ^ { k } Y _ { { \boldsymbol { \mathbf { x } } } ^ { ( i ) } } ^ { ( i ) } ( { \boldsymbol { \mathbf { u } } } ) = y ^ { ( i ) } \right] \cdot P ( { \boldsymbol { \mathbf { u } } } ) .\tag{1}
$$

The evaluation accumulates over the exogenous instantiations u satisfying the formula $\gamma .$ We denote such an entailment as $( { \mathcal { M } } , { \boldsymbol { \mathscr { u } } } ) \models \gamma$ . When it is clear from the context which SCM we refer to, we omit specifying it and rewrite the right-hand side of $\begin{array} { r } { \mathrm { E q . ~ } ( 1 ) \mathrm { a s } \sum _ { u } \mathbb { I } u \Vdash \gamma \mathbb { I } P ( \pmb { u } ) } \end{array}$ . If all the subscripts agree on x, then Eq. (1) can be retrieved from the interventional distribution $P _ { \mathcal { M } _ { x } } .$ . In general, this is not the case and we eventually characterise a counterfactual distribution. Finally, we denote as $\overset { \mathbf { \omega } } { V } _ { \overset { \mathbf { \displaystyle } } { \ b { \gamma } } } \subseteq \mathbf { \bar { V } }$ the endogenous, no matter whether queried or intervened, variables in $\gamma$

Nested Counterfactuals. We can further generalise the intervention for the counterfactual variable $Y _ { x }$ to allow representing settings where the variable X is set to behave as another variable, say $X _ { z } . \mathrm { ~ A ~ }$ random variable Y in such system is represented with a potential outcome of the form $Y _ { X _ { z } } = y _ { : }$ , which is called a nested counterfactual. This expresses the statement $^ { 6 6 } Y$ becomes $y$ when we set X to whatever value it would have taken if we had set $Z \mathrm { t o } \ z ^ { \prime \prime }$ Queries involving nested counterfactuals can be rewritten into a marginalisation over conjunctions of standard, atomic counterfactual events (Correa et al., 2021), i.e.,

$$
( Y _ { X _ { z } } = y ) \equiv \bigvee _ { x \in \mathcal { X } } \left( Y _ { x } = y \wedge X _ { z } = x \right) .\tag{2}
$$

(Partial) Identifiability. Eq. (1) allows to compute counterfactual queries in an SCM M. Yet, the exogenous distribution $P ( U )$ is rarely available, and one must cope with the corresponding PSCM M and a dataset D of endogenous observations, used to estimate an empirical distribution $\hat { P } ( V )$ . In general, there are multiple SCMs in $\mathbb { M } _ { M }$ that could have generated $\hat { P } ( V )$ and yield different values to a query over γ. In such a partially identifiable setting, we can only compute bounds to the query, i.e., the minimum:

$$
\operatorname* { m i n } _ { \mathcal { M } \in \mathbb { M } _ { M } } P _ { \mathcal { M } } ( \gamma ) \quad \mathrm { s . t . } \quad P _ { \mathcal { M } } ( V ) = \hat { P } ( V ) ,\tag{3}
$$

and the maximum, which is analogously defined. In the remainder of the paper, just for the sake of brevity, we focus on the minimisation problem, noticing that the maximisation counterpart can be handled analogously.

Those optimisation tasks have been shown to be NP-hard (Zaffalon et al., 2024). Nevertheless, approximate techniques were proposed to estimate bounds efficiently (Zhang et al., 2022; Zaffalon et al., 2024; Duarte et al., 2024; Bjøru et al., 2025). Standard (point) identifiability emerges as a special case when the two bounds coincide.

Queries and Partial Orders. Although potential outcome semantics allow us to consider a variable $Y _ { x }$ for any $\bar { Y } , X \in V$ and $x \in \mathcal { X }$ , an intervention on $X ^ { \mathit { \Pi } }$ can alter the realisation of $Y$ only if there exists a directed path from X to Y in the causal graph induced by the assumed SCM M over V. This is formalised as follows.

Proposition 1 (Tian, 2002). $I f Y , X \in V$ are distinct endogenous variables of an SCM M such that X /∈ $\operatorname { A n c } ( Y )$ then, $\forall u , Y _ { x } ( { \pmb u } ) = Y ( { \pmb u } )$ , and thus $P _ { \mathcal { M } } ( Y _ { x } ) = P _ { \mathcal { M } } ( Y )$ .

Thus, a counterfactual variable $Y _ { x }$ can be non-trivial only if X precedes Y in the underlying causal graph. In this paper, we restrict attention to non-trivial queries: consequently, when the graph is not available, a query over γ can be also used to induce a collection of precedence constraints, corresponding to a partial order $\prec _ { \gamma }$ on $\bar { V _ { \gamma \cdot } } \bar { \mathrm { A } ^ { \enspace  } } \mathrm { c y c l i c } ^ { , }$ event such as $y _ { x } \wedge x _ { y }$ would be ill-posed as it induces incompatible precedence constraints. Such cases would naturally call for non-recursive models (see, for instance, Cozman et al., 2025 and Bongers et al., 2021), which fall outside the scope of this paper.

## 3 Bounding Framework

The main goal of this paper is to relax the optimisation task in Eq. (3) to cope with situations where the underlying PSCM M is unavailable. We start by assuming that the endogenous variables are sorted by a total order σ, corresponding to $\pmb { V } : = ( V ^ { ( 1 ) } , \ldots , V ^ { ( k ) } )$ . In this setting, we can replace in $\operatorname { E q . } \left( 3 \right)$ the set $\mathbb { M } _ { M }$ of SCMs compatible with the PSCM M with the set $\mathbb { M } _ { \sigma }$ of all SCMs whose causal graph is consistent with $\sigma , \mathrm { i . e . }$

$$
\operatorname* { m i n } _ { \mathcal { M } \in \mathbb { N } _ { \sigma } } P _ { \mathcal { M } } ( \gamma ) \quad \mathrm { s . t . } \quad P _ { \mathcal { M } } ( V ) = \hat { P } ( V ) .\tag{4}
$$

For the moment, we consider $\gamma$ such that $V _ { \gamma } = V$ , i.e., all endogenous variables appear in the query. A more general account of the case $V _ { \gamma } \subset V$ is addressed later.

Response Representation. Unlike Eq. (3), the optimisation task in Eq. (4) refers to a collection of SCMs possibly based on different PSCMs. Yet, given only σ and $V$ , we consider a complete canonical PSCM over V denoted as $M _ { o }$ and such that: (i) $\mathrm { P a } _ { V ^ { ( i ) } }$ coincides with all the predecessors of $V ^ { ( i ) }$ according to σ and they are denoted as $V _ { \prec i } ; ( \mathrm { i i } )$ there is a single exogenous variable $U _ { \textrm { \scriptsize : } }$ , which is a parent of all the endogenous variables; (iii) the SEs of each $V \in V$ and the cardinality of $U$ are such that $M _ { \sigma }$ is canonical in the sense of Zhang et al. (2022), this basically meaning that the exogenous states of $U$ are indexing all possible structural relations between the endogenous variables and their endogenous parents (see App. C for a more detailed account).

The optimisation task in Eq. (4) can be equivalently achieved by considering the complete canonical PSCM $M _ { \sigma }$ only, as stated by the following result.

Theorem 1. Eq. (3) with $M = M _ { \sigma }$ gives the same minimum of $E q . \ ( 4 ) , i . e .$ ,

$$
\operatorname* { m i n } _ { \mathcal { M } \in \mathbb { M } _ { \sigma } } P _ { \mathcal { M } } ( \gamma ) = \operatorname* { m i n } _ { \mathcal { M } \in \mathbb { M } _ { M _ { \sigma } } } P _ { \mathcal { M } } ( \gamma ) ,\tag{5}
$$

where $P _ { \mathcal { M } } ( V ) = \hat { P } ( V )$ is required on both sides.

The above result reduces the optimisation in Eq. (4)—which considers a collection of structurally heterogenous SCMs— to span only those SCMs sharing the same causal graph as $M _ { \sigma }$ . This reduction provides a parametrisation of the task, where the optimisation variables are associated with the exogenous probabilities as in Eq. (1). To avoid explicitly modelling these latent variables, we build upon the approach proposed by Balke and Pearl (1994a,b). Here, we consider all the potential outcomes that $M _ { \sigma }$ is allowed to generate as to enumerate all possible endogenous mappings from parents to child. This is achieved by considering a response signature $S _ { \sigma }$ for $M _ { \sigma }$ , corresponding to the following collection of potential outcomes:

$$
S _ { \sigma } : = \left\{ V _ { \pmb { v } _ { < i } } ^ { ( i ) } \right\} _ { \pmb { v } _ { < i } \in \pmb { \mathscr { V } } _ { \prec i } } ^ { i = 1 , \dots , k } .\tag{6}
$$

The following result shows that the probability of any counterfactual query can be expressed not only as a linear combination of joint probabilities over the exogenous variables, as in Eq. (1), but also as a linear combination of joint probabilities over $S _ { \sigma }$

Theorem 2. A query $P _ { \mathcal { M } } ( \gamma )$ in an SCM $\mathcal { M } \in \mathbb { M } _ { \sigma }$ can be written as a linear combination ofthe probabilities ofthe response signature, i.e.,

$$
P _ { \mathcal { M } } ( \gamma ) = \sum _ { \pmb { s } _ { \sigma } \in \pmb { \mathscr { S } } _ { \sigma } } \mathbb { \mathbb { \mathbb { s } } } _ { \sigma } \Vdash \gamma \mathbb { \mathbb { I } } \cdot P ( \pmb { s } _ { \sigma } ) .\tag{7}
$$

By viewing observational queries as a special case of counterfactual queries in which interventions are performed on the empty set, we also obtain the following additional result.

Corollary 1. Any observational probability $P _ { \mathcal { M } } ( \pmb { v } )$ in an $S C M \mathcal M \in \mathbb M _ { c }$ can be written as a linear combination ofthe probabilities of the response signature, i.e.,

$$
P _ { \mathcal { M } } ( \pmb { v } ) = \sum _ { \pmb { s } _ { \sigma } \in \pmb { S } _ { \sigma } } \mathbb { \mathbb { \mathbb { s } } } _ { \sigma } \Vdash \pmb { v } \mathbb { I } \cdot P ( \pmb { s } _ { \sigma } ) .\tag{8}
$$

Linear Programming (LP) Formulation. Thm. 2 and Cor. 1 allow us to address the optimisation task in Eq. (4) via the signature joint states. The resulting linearity of the constraints and the objective function allows us to rewrite the bounding task as a LP task:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { \displaystyle \sum _ { s _ { \sigma } \in S _ { \sigma } } \left[ | s _ { \sigma } | = \gamma \right] \cdot q _ { s _ { \sigma } } } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { s _ { \sigma } \in S _ { \sigma } } \left[ | s _ { \sigma } | = v \right] \cdot q _ { s _ { \sigma } } = \hat { P } ( v ) \ , \forall v \in \mathcal { V } \ ; \quad \displaystyle \sum _ { s _ { \sigma } \in S _ { \sigma } } q _ { s _ { \sigma } } = 1 \ ; \quad q _ { s _ { \sigma } } \geq 0 \ , \forall s _ { \sigma } \in \mathcal { S } _ { \sigma } \ . } \end{array}\tag{9}
$$

with semantics $q _ { { \pmb s } _ { \sigma } } = P ( { \pmb s } _ { \sigma } )$ for the optimisation variables.

The following result establishes that the above LP task solves the relaxed optimisation over SCMs consistent with σ by yielding tight identification bounds.

Theorem 3. The LP task in Eq. (9) gives the solution of Eq. (4). Furthermore, there exists an SCM $\mathcal { M } \in \mathbb { M } _ { o }$ such that $P _ { \mathcal { M } } ( \gamma )$ coincides with the solution ofthe LP.

Signature State Partitioning. The latent space reduction to finite response-function classes (Balke and Pearl, 1994a; Duarte et al., 2024) enables us to neatly solve Eq. (4) through Eq. (9). However, a further reduction can be achieved by observing that the LP in Eq. (9) depends exclusively on the Boolean evaluations of the signatures against the observations v and the counterfactual event γ. Signatures states that evaluate identically against these observations and events can be grouped into equivalence classes $\smash { \pmb { t } } _ { \sigma } \in \mathcal { T } _ { \sigma } : = \pmb { S } _ { \sigma } / \sim _ { \gamma }$ , where:

$$
s _ { \sigma } \sim _ { \gamma } s _ { \sigma } ^ { \prime } \iff ( ( s _ { \sigma } \vdash \gamma ) \equiv ( s _ { \sigma } ^ { \prime } \vdash \gamma ) ) \land ( \exists v \in \mathcal { V } \mathrm { ~ s . t . ~ } ( s _ { \sigma } \vdash v ) \land ( s _ { \sigma } ^ { \prime } \vdash v ) ) ~ .\tag{10}
$$

Each class $\mathbf { \Delta } _ { t _ { \sigma } } \in \mathcal { T } _ { \sigma }$ inherits the binary evaluations from the signatures it includes, hence $[ [ t _ { \sigma } \mid = v ] ]$ and $[ [ t _ { \sigma } \vdash \gamma ] ]$ . By defining aggregated masses as variables, $\begin{array} { r } { q _ { \pmb { t } _ { \sigma } } : = \sum _ { \pmb { s } _ { \sigma } \in \pmb { t } _ { \sigma } } q _ { \pmb { s } _ { \sigma } } } \end{array}$ , and applying Thm. 2 and Cor. 1, the LP in Eq. (9) reduces to a lifted LP task (Kersting et al., 2017):

$$
\begin{array} { r l } { \operatorname* { m i n } } & { \displaystyle \sum _ { t _ { \sigma } \in \mathcal { T } _ { \sigma } } \left\| t _ { \sigma } \left| - \gamma \right\| \cdot q _ { t _ { \sigma } } \right. } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { t _ { \sigma } \in \mathcal { T } _ { \sigma } } \left[ t _ { \sigma } \left| - v \right. \right] \cdot q _ { t _ { \sigma } } = \hat { P } ( v ) \ , \forall v \in \mathcal { V } \ ; \quad \displaystyle \sum _ { t _ { \sigma } \in \mathcal { T } _ { \sigma } } q _ { t _ { \sigma } } = 1 \ ; \quad q _ { t _ { \sigma } } \geq 0 \ , \forall t _ { \sigma } \in \mathcal { T } _ { \sigma } \ . } \end{array}\tag{11}
$$

The equivalence mapping improves the tractability of the LP in Eq. (9) whilst maintaining both soundness and tightness, as it follows from the following result.

Theorem 4. The lifted LP task in Eq. (11) gives the solution ofEq. (4). Furthermore, there exists an SCM $\mathcal { M } \in \mathbb { M } _ { o }$ such that $P _ { \mathcal { M } } ( \gamma )$ coincides with the solution ofthe LP.

Marginalising Non-Queried Variables. So far, we assumed that all the endogenous variables to appear in the query, i.e., $V _ { \gamma } = V$ . We now extend our results to the more general case in which $V _ { \gamma } \subseteq V$ . To this end, let $\mathbb { M } _ { \sigma } ^ { V _ { \gamma } }$ denote the set of SCMs defined only over $V _ { \gamma }$ and compatible with $\sigma .$ Observe that $\mathbb { M } _ { \sigma } ^ { V } = \mathbb { M } _ { o }$ and, in general, $\mathbb { M } _ { \sigma } ^ { V _ { \gamma } } \subseteq \mathbb { M } _ { \sigma } .$ where, for notational ease, we assumed that the endogenous variables might not appear explicitly in an SCM, thus being uniformly distributed. The following result holds.

Theorem 5. The solution of Eq. (4) coincides with:

$$
\operatorname* { m i n } _ { \mathcal { M } \in \mathbb { M } _ { \sigma } ^ { V _ { \gamma } } } P _ { \mathcal { M } } ( \gamma ) s . t . P _ { \mathcal { M } } ( V _ { \gamma } ) = \hat { P } ( V _ { \gamma } ) .\tag{12}
$$

where $\hat { P } ( V _ { \gamma } )$ is obtained from $\hat { P } ( V )$ by marginalisation.

As a consequence of the above result, when $V _ { \gamma } \subset V$ , we marginalise the empirical distribution before applying the LP formulation to solve Eq. (12) and hence Eq. (4).

Coping with Partial Orders. The optimisation task in Eq. (4) assumes a total order σ over the endogenous variables. However, as discussed in Sect. 2, a counterfactual query over γ might induce only a partial order $\prec _ { \gamma }$ . Nevertheless, a general formulation is readily obtained by considering the bound over all linear extensions $\Sigma ( \prec _ { \gamma } )$

$$
\operatorname* { m i n } _ { \sigma \in \Sigma ( \prec _ { \gamma } ) } \operatorname* { m i n } _ { \mathcal { M } \in \mathbb { M } _ { \sigma } ^ { V _ { \gamma } } } P _ { \mathcal { M } } ( \gamma ) \quad \mathrm { s . t . } \quad P _ { \mathcal { M } } ( V _ { \gamma } ) = \hat { P } ( V _ { \gamma } ) .\tag{13}
$$

When $\gamma$ contains no interventions, $\prec _ { \gamma }$ imposes no ordering constraints, so that all possible total orders must be considered, and we have $| \Sigma ( \prec _ { \gamma } ) | = | V _ { \gamma } | !$ . Interventions restrict this search space by ruling out inconsistent orderings; nonetheless, the number of linear extensions can still scale exponentially with respect to $| V _ { \gamma } | .$ In spite of this, the following result establishes that evaluating an arbitrary linear extension is sufficient, thus eliminating the need for a brute-force optimisation over all the orders in $\Sigma ( \prec _ { \gamma } )$

Theorem 6. Consider a counterfactual query over γ inducing a partial order $\prec _ { \gamma } .$ . Any total order $\sigma \in \Sigma ( \prec _ { \gamma } )$ gives the same minimum in Eq. (12) and, hence, the solution ofEq. (13).

As a consequence of Thm. 6, we are not required to assume or enforce any specific total order. The choice of a particular σ serves only as a possible parametrisation of the task.

Complexity. Thm. 3 enables us to solve the optimisation problem in Eq. (4) using a linear solver that implements the LP defined in Eq. (9). It is straightforward to verify that, if the number of variables in the query is $k : = \bar { | V _ { \gamma } | }$ (see Thm. 5) and $n : = \mathrm { m a x } _ { i = 1 } ^ { k } \vert \mathcal { V } ^ { ( i ) } \vert$ denotes the maximum endogenous cardinality, the number of potential outcomes comprising the signature $S _ { \sigma }$ in Eq. (6) is bounded by $\begin{array} { r } { \sum _ { i = 0 } ^ { k - 1 } n ^ { i } = \frac { n ^ { k } - 1 } { n - 1 } } \end{array}$ . The number of joint signature states $| \pmb { S } _ { \sigma } |$ which dictates the number of optimisation variables in the na¨ıve LP of Eq. (9), is therefore double exponential, namely ${ \mathcal { O } } ( n ^ { n ^ { k - 1 } } )$ . In contrast, by Thm. 4, the size of the quotient space $| \tau _ { \sigma } |$ , which governs the size of the lifted LP in Eq. (11), is bounded by $\begin{array} { r } { \mathcal { O } \big ( \prod _ { i = 1 } ^ { k } | \mathcal { V } ^ { ( i ) } | \big ) = \mathcal { O } ( n ^ { k } ) } \end{array}$ . Thus, to partially identify any query given the empirical marginal $\hat { P } ( V _ { \gamma } )$ , we only need to solve Eq. (11), whose optimisation space scales linearly with the size of the sample space of $V _ { \gamma }$

## 4 Extensions

In the current section, we present some important extensions of the LP mapping introduced in the previous section.

Conditional Queries. To bound a conditional query $P ( \gamma \mid \delta ) : = P ( \gamma \wedge \delta ) / P ( \delta )$ , it suffices to note that both terms decompose linearly over $\scriptstyle { \pmb { S } } _ { \sigma }$ via Thm. 2 and Cor. 1. This reformulates the bounding task as a linear-fractional programme, which is reduced to a standard LP via the Charnes-Cooper transformation (Charnes and Cooper, 1962). We incorporate evidence by refining the equivalence relation in Eq. (10) to evaluate both $\gamma \wedge \delta$ and δ:

$$
{ s _ { \sigma } } \sim _ { \gamma \mid \delta } s _ { \sigma } ^ { \prime } \iff s _ { \sigma } \sim _ { \gamma \wedge \delta } s _ { \sigma } ^ { \prime } \wedge \left( \left( s _ { \sigma } \fbox { = } \delta \right) \equiv \left( s _ { \sigma } ^ { \prime } \fbox { = } \delta \right) \right) .\tag{14}
$$

![](images/ef5cb77762fa78572fa0fe40218dff671c1b4dc7554e01ba462216cf8939c4ca.jpg)  
Figure 1: Causal diagrams considered in the examples. (a) Confounded treatment-outcome pair. (b) Instrumental Variable (IV) setting, where $Z$ serves as an instrument for X. (c) Mediation model with a direct effect and confounding between the mediator W and the outcome. Grey nodes denote unobserved exogenous variables.

Letting $b _ { \psi } : = [ [ s _ { \sigma } \vert = \psi ] ]$ , the logical constrain $\gamma \wedge \delta \Rightarrow \delta$ enforces $b _ { \gamma \wedge \delta } \leq b _ { \delta } \in \{ 0 , 1 \}$ . This restricts evaluations to three valid Boolean states: $( \bar { b } _ { \gamma \wedge \delta } , \bar { b } _ { \delta } ) \in \{ ( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 1 ) \}$ }. The optimisation space retains the same asymptotic size discussed in Sect. 3.

Nested Queries. The LP mapping in Eq. (9), and by extension the lifted LP in Eq. (11), accommodates nested counterfactuals. Although nested variables do not appear directly in the response signature $S _ { \sigma }$ , their evaluation remains a linear optimisation task. For a single nested counterfactual, it is sufficient to consider the probabilistic version of Eq. (2), i.e.,

$$
P _ { \mathcal M } ( Y _ { X _ { z } } = y ) = \sum _ { x \in { \mathcal X } } P _ { \mathcal M } ( Y _ { x } = y , X _ { z } = x ) .\tag{15}
$$

In the general case, it is sufficient to perform such un-nesting on all the nested terms to obtain a linear function of standard counterfactual queries and thus a linear objective function.

Experimental Data. Beyond observational data, it is straightforward to incorporate experimental distributions obtained directly from randomised controlled trials. By Thm. 2, experimental probabilities translate directly into linear constraints on the signature probabilities in Eq. (9). To preserve the validity of the reduction in Eq. (11), the equivalence relation is refined by further requiring that equivalent signatures agree on the Boolean evaluation of the relevant interventional events. For instance, given an experimental probability $P ( y _ { x } )$ , the relation in Eq. (10) is augmented with the logical condition $\left( \pmb { \mathscr { s } } _ { \sigma } \mp \mathcal { Y } _ { x } \right) \equiv \left( \pmb { \mathscr { s } } _ { \sigma } ^ { \prime } \mp \mathcal { Y } _ { x } \right)$ . Tracking these additional constraints partitions the signature space more finely, thereby scaling the quotient space by a factor determined by the number of interventional events considered. Furthermore, when multiple studies or heterogeneous data sources are available, one can seamlessly incorporate all evidence by appending the constraints induced by each trial in the same LP. Such a procedure natively performs automated data fusion, analogously to the approach developed by Zaffalon et al. (2023) for PSCMs.

Monotonicity and Weak Exogeneity. For ordinal variables, monotonicity (Angrist et al., 1996; Manski, 1997) posits a directional effect $( \mathbf { e . g . } , Y _ { x _ { 1 } } \geq Y _ { x _ { 0 } } { \mathrm { i f ~ } } x _ { 1 } > x _ { 0 } )$ . Weak exogeneity (Tian and Pearl, 2000) requires that a variable is unaffected by unobserved confounders. Both assumptions translate expert knowledge into logical restrictions on the response signature $\scriptstyle { \pmb { S } } _ { \sigma } :$ : monotonicity excludes states violating the inequality by fixing relevant optimisation variables to zero; exogeneity requires that causal effects are identified from data, i.e., $\mathbf { \bar { \psi } } _ { P ( y _ { x } ) } = \mathbf { \bar { \psi } } _ { P ( y \mid x ) }$ , functionally mirroring the embedding of experimental data.

## 5 Examples

Let us validate the application of our LP-based bounding scheme across a range of scenarios. We compare our bounds against analytical formulae and results from the literature.

Probabilities of Causation. As a first illustrative application of our method, we derive bounds for the well-known probability ofnecessity and sufficiency (PNS, Pearl, 1999). This query refers to the counterfactual event $\gamma : = ( Y _ { X = 0 } =$ $0 ) \wedge ( Y _ { X = 1 } = 1 )$ , which is intended to capture the causal relationship between two Boolean endogenous variables, X and Y . The only total order σ compatible with $\prec _ { \gamma }$ is $\sigma : = ( X , Y )$ . The associated response signature is therefore ${ \cal S } _ { \sigma } = \{ X , Y _ { X = 0 } , \dot { Y } _ { X = 1 } \}$ . Accordingly, PNS bounds are obtained by solving LPs with eight variables, corresponding to the joint states of $S _ { \sigma } ;$ ; and four linear constraints induced by the observational distribution ${ \hat { P } } ( X , Y )$ . The resulting solutions coincide with the classical closed-form bounds derived by Tian and Pearl (2000). The equivalence relation in Eq. (10) prunes the parameter space by identifying empty equivalence classes. Specifically, the candidate classes defined by the Boolean tuples $\langle ( X \overset { \cdot } { = } x , \overset { \cdot } { Y } = y ) , \overset { \cdot } { \mathbb { I } } \gamma \overset { \cdot } { \mathbb { I } } \rangle$ for $\mathsf { \bar { ( } } \dot { x } , y \bar { ) } \in \{ ( 1 , 0 ) , ( 0 , 1 ) \}$ contain no valid signatures. This is because an observational state of $( X = 1 , Y = \overset { \cdot } { 0 } )$ forces $Y _ { X = 1 } = 0$ via consistency (Galles and Pearl, 1998), directly contradicting the requirement in $\gamma$ that $Y _ { X = 1 } = 1$ . We similarly proceed for $( X = \overset { \cdot } { 0 } , Y = 1 )$ .

The tightness (referred to as sharpness in the original work) of these bounds follows from Thm. 3: one can construct two complete canonical SCMs under which the query attains the lower and upper bounds (see also the constructive argument in the proof).

Finally, by considering the extension to conditional queries discussed in Sect. 4, analogous results can be readily obtained for the other probabilities ofcausation (PN and PS, Pearl, 1999). Moreover, unlike the analytical results of Tian and Pearl (2000), which are restricted to binary variables, our approach naturally generalises to the non-binary case. By simply adjusting the cardinality of the variables in the response signature $S _ { \sigma } ,$ , we can derive tight bounds for probabilities of causation involving multi-valued treatments and outcomes (Sun et al., 2025) without requiring new algebraic derivations. From this perspective, our work is a genuine generalisation of the analysis in Tian and Pearl (2000), with a clear semantics, and allowing to bound arbitrary counterfactual queries, whether nested or not.

Average Causal Effect (ACE). The ACE measures the average change due to treatment:

$$
\mathrm { A C E } _ { x , x ^ { \prime } } ( y ) : = P ( Y _ { x } = y ) - P ( Y _ { x ^ { \prime } } = y ) .\tag{16}
$$

For an outcome $Y \in \{ y , y ^ { \prime } \}$ and treatment $| \mathcal { X } | = 3 ,$ , the unique order $\sigma = ( X , Y )$ induces an LP with 24 signature variables and six empirical constraints in Eq.(9). The resulting bounds, $[ P ( x , y ) + P ( x ^ { \prime } , y ^ { \prime } ) - 1 , 1 - P ( x , y ^ { \prime } ) - \bar { P ( } x ^ { \prime } , y ) ]$ (Balke and Pearl, 1997), recover those of Sachs et al. (2023, Sect. 6.1) whose causal assumptions are in Fig. 1a. Crucially, our method achieves this without explicitly parametrising a fully specified causal graph nor the SEs. A different account of the ACE bounding can be found in App. B.

<table><tr><td rowspan="2">Assumption Set</td><td colspan="2"> $\mathrm { C D E } _ { x _ { 0 } , x _ { 1 } } ^ { w _ { 0 } } \left( y _ { 1 } \right)$ </td><td colspan="2"> $\mathrm { C D E } _ { x _ { 0 } , x _ { 1 } } ^ { w _ { 1 } } \big ( y _ { 1 } \big )$ </td><td rowspan="2"> $| \pmb { S } _ { \sigma } | \quad | \pmb { T } _ { \sigma } |$ </td><td rowspan="2"></td></tr><tr><td>Lower</td><td>Upper</td><td>Lower</td><td>Upper</td></tr><tr><td>No additional assumptions</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cai et al. (2008) (Fig. 1c)</td><td>-0.200</td><td>0.385</td><td>-0.781</td><td>0.634</td><td></td><td></td></tr><tr><td>This paper</td><td>-0.599</td><td>0.694</td><td>-0.891</td><td>0.817</td><td>128</td><td>20</td></tr><tr><td>Monotonicity</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cai et al. (2008)</td><td>0.000</td><td>0.036</td><td>0.000</td><td>0.582</td><td></td><td></td></tr><tr><td>This paper</td><td>0.000</td><td>0.519</td><td>0.000</td><td>0.791</td><td>36</td><td>13</td></tr><tr><td>Weak exogeneity</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>This paper</td><td>-0.200</td><td>0.385</td><td>-0.782</td><td>0.633</td><td>128</td><td>64</td></tr><tr><td>Weak exogeneity + monotonicity</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>This paper</td><td>0.000</td><td>0.035</td><td>0.000</td><td>0.580</td><td>36</td><td>24</td></tr></table>

Table 1: Comparison of CDE bounds on the (binary) Lipid data (Cai et al., 2008, Tab. 1) against analytical bounds under varying assumptions. In this experiment, Monotonicity forces positive effects of X on ${ \check { W } } , X$ on $Y ,$ , and $W$ on $Y ;$ whilst, weak exogeneity imposes no-interaction between X and $W$ on $Y$

Controlled Direct Effect (CDE). Consider a setting in which the effect of a treatment X on an outcome Y is mediated by a variable W. In this context, the ACE in Eq. (16) can be refined into a CDE to measure the change due to treatment at different mediator strata:

$$
\mathrm { C D E } _ { x _ { 0 } , x _ { 1 } } ^ { w } ( y ) : = P ( Y _ { x _ { 1 } , w } = y ) - P ( Y _ { x _ { 0 } , w } = y ) .\tag{17}
$$

The query induces a partial order, namely $\{ X , W \} \prec Y$ . By Thm. 6, we can choose an arbitrary order among its set of linear extensions, together with an empirical distribution ${ \hat { P } } ( X , W , Y )$ , to solve Eq. (11). We validate the method against the problem proposed by Cai et al. (2008) of bounding Eq. (17) under the assumptions encoded in Fig. 1c. In such a setting, we first consider all binary variables and a randomised treatment (i.e., no confounders between X and Y ), examining data from the Coronary Primary Prevention Trial (Cai et al., 2008).

A summary of the computed bounds is reported in Tab. 1. One can observe that when no assumptions are made, or when solely monotonicity is assumed, the minimal structure inherently implied by the query yields valid, but wide, intervals. Notably, the tight bounds reported by Cai et al. (2008) rely on a fully specified causal graph. To mirror their structural assumptions within our framework, one must incorporate a weak exogeneity constraint—also

referred to as the no-interaction assumption—into the signature space $\scriptstyle { \pmb { S } } _ { \sigma }$ . Specifically, enforcing the independence $\{ Y , W \}$ ⊥⊥ X recovers Cai et al. (2008) analytical bounds. As a special instance, one can also require $Y \perp \perp \{ X , W \}$ to collapse the bound to a point estimate, since $P ( y _ { x , w } ) = P ( y \mid \bar { x } , w )$ . As expected, introducing these independence constraints expands the number of optimisation parameters $| \tau _ { \sigma } |$ , whilst introducing monotonicity constraints shrinks the space (cf. Tab. 1). A similar discussion holds for categorical variables, where the LP reduction can be systematically constructed and compared against the equations presented in Cai et al. (2008).
<table><tr><td></td><td>Lower</td><td>Upper</td><td> $| \pmb { S } _ { \sigma } |$ </td><td> $| \mathcal { T } _ { \sigma } |$ </td></tr><tr><td>Assumptions for  $\mathrm { N D E } _ { x _ { 0 } , x _ { 1 } } \left( y _ { 1 } \right)$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Query-induced Order</td><td>-0.691</td><td>0.688</td><td>294,912</td><td>60</td></tr><tr><td> $\bar { Y } \perp \perp X$ </td><td>-0.503</td><td>0.497</td><td>294,912</td><td>96</td></tr><tr><td> $\{ Y , W \} \perp \perp X$ </td><td>-0.503</td><td>0.497</td><td>294,912</td><td>528</td></tr><tr><td> $\grave { Y } \perp \perp \left\{ X , W \right\}$ </td><td>-0.628</td><td>0.679</td><td>294,912</td><td>89,172</td></tr><tr><td> $Y \not \perp \downarrow \ \mathrm { \bar { \{ X , } }   \bar { W \} } \ \& \ W \not \perp X$ </td><td>-0.489</td><td>0.637</td><td>294,912</td><td>294,912</td></tr><tr><td>Query</td><td></td><td></td><td></td><td></td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 0 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 0 } )$ </td><td>0.000</td><td>0.610</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 1 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 1 } )$ </td><td>0.000</td><td>0.493</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 2 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 2 } )$ </td><td>0.000</td><td>0.365</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 3 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 3 } )$ </td><td>0.000</td><td>0.415</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 4 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 4 } )$ </td><td>0.000</td><td>0.357</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 1 } , W = w _ { 5 } } = y _ { 1 } , W _ { x _ { 0 } } = w _ { 5 } )$ </td><td>0.000</td><td>0.391</td><td>294,912</td><td>37</td></tr><tr><td> $P ( Y _ { x _ { 0 } } = y _ { 1 } )$ </td><td>0.312</td><td>0.691</td><td>8</td><td>6</td></tr></table>

Table 2: UC Berkeley dataset: NDE bounds under various weak exogeneity assumptions (top) and individual unnested query terms under query-induced assumptions (bottom).

Natural Direct Effect (NDE). The CDE in Eq. (17) refers to a fixed state of the mediator. By contrast, the NDE evaluates the effect of a treatment when the mediator is set to the value it would have attained in the absence of treatment. This leads to the nested counterfactual query:

$$
\mathrm { N D E } _ { x _ { 0 } , x _ { 1 } } ( y _ { 1 } ) : = P ( Y _ { x _ { 1 } , W _ { x _ { 0 } } } = y _ { 1 } ) - P ( Y _ { x _ { 0 } } = y _ { 1 } ) .\tag{18}
$$

As a matter of fact, the nested syntactical structure of the query induces a unique complete order $X \prec W \prec Y$ . We proceed to apply the un-nesting procedure in Eq. (15), which, in turn, allows one to rewrite nested counterfactuals like in Eq. (18) as sums of standard (non-nested) counterfactual queries. This transformation preserves the linear structure of the objective function in our LP formulation and one can consider the PSCM whose endogenous structure is like in Fig. 1c but with one exogenous U parent of all variables. As discussed previously, by Thm. 4, the optimisation space’s size is bounded by $O ( | \boldsymbol { \mathcal { X } } | \cdot | \boldsymbol { \mathcal { W } } | \cdot | \dot { \boldsymbol { \mathcal { V } } } | )$ (cf. Tab. 2), making our approach computationally feasible for modern LP solvers. To the best of our knowledge, no general bounding techniques are currently available for queries of this kind without imposing additional structural assumptions about the underlying causal graph.

Berkeley Admissions Dataset. For a numerical illustration, we consider the popular UC Berkeley admissions dataset (Bickel et al., 1975). In this dataset, the binary variables X and $Y$ represent gender (with $x _ { 1 }$ indicating female) and admission outcome (with $y _ { 1 }$ denoting acceptance). The mediator W is a categorical variable with six states corresponding to the department to which the applicant applied. We omit monotonicity assumptions, as deterministic effects are tricky to justify in social contexts.

Tab. 2 summarises the total NDE bounds under varying weak exogeneity assumptions, alongside the decomposed counterfactual terms from Eq. (15). In the context of university admissions, enforcing Y ⊥⊥ X assumes no confounding jointly affecting gender and admission. The stricter condition $\{ Y , W \} \stackrel { \cdot \mathrm { l } } { \bot } X$ treats gender as if it were completely randomised, assuming department choice does not depend on gender. Conversely, $\breve { Y } \perp \perp \{ X , W \}$ dictates that the admission committee’s decision is entirely independent of unobserved background factors influencing either the applicant’s gender or department preference. It is worth noting that when all possible exogeneity assumptions are employed, the equivalence reduction from Sect. 2 fails to compress the optimisation space, as enforcing mutual independence across all response mechanisms requires tracking each signature state individually. Whilst these assumptions narrow the bounds, at the cost of expanding the optimisation space $| \tau _ { \sigma } |$ , they risk misrepresenting complex dynamics.

Unlike bespoke analytical methods that require graphical constraints to bound nested counterfactuals, our LP formulation automatically evaluates the NDE using only the minimal assumptions inherent to the query itself. For reference, under the standard fairness model (Plecko and Bareinboim, 2024), the NDE evaluates to a point estimate of 0.043. Concludingˇ a lack of discriminative policies based on this point estimate might be completely inaccurate, as the stark contrast with our wider, assumption-free intervals demonstrates how such conclusions rely heavily on rigid structural assumptions Furthermore, it is interesting to notice that directly bounding the full NDE yields substantially tighter intervals than combining separately bounded decomposed components via interval arithmetic.

## 6 Conclusions

We presented a linear programming framework for partially identifying arbitrary, including nested, counterfactual queries, relying exclusively on structural assumptions inherent to the query itself. In doing so, we successfully extended the classical bounding results of Tian and Pearl (2000) for probabilities of causation. We proved that partial identification under these minimal assumptions yields tight intervals, and established their invariance with respect to the marginalisation of the variables not appearing in the query. Rather than requiring an input causal graph, an SCM attaining the bounds emerges constructively as a by-product. Furthermore, exploiting query symmetries drastically reduces the optimisation space, rendering the LP tractable. We acknowledge an inherent trade-off in the proposed method: reducing the reliance on prior domain knowledge, and explicit conditional independencies, produces intervals that are naturally wider than those derived from a fully specified causal graph. Nevertheless, as demonstrated in our case studies, this framework serves as a general and computationally efficient method to bound arbitrary inquiries under minimal assumptions. Future work could investigate extending this framework to cyclic causal structures, examining the impact of different valid causal orders, and investigate finer quotient-space reductions to further enhance scalability.

## Acknowledgements

The research of Eric Rossetto was funded by Swiss Post AG as part of a Doctoral Studies Grant on AI and Robustness. The authors thank the anonymous reviewers for their constructive comments, and in particular the first reviewer for insightful remarks on the underlying assumptions of our approach and its possible extension to cyclic models.

## References

Joshua D. Angrist, Guido W. Imbens, and Donald B. Rubin. Identification of causal effects using instrumental variables. Journal ofthe American Statistical Association, 91(434):444–455, 1996.

Alexander Balke and Judea Pearl. Counterfactual probabilities: Computational methods, bounds and applications. In Proceedings ofthe Tenth International Conference on Uncertainty in Artificial Intelligence, pages 46–54, 1994a.

Alexander Balke and Judea Pearl. Probabilistic evaluation of counterfactual queries. In Probabilistic and Causal Inference, volume Proceedings of the Twelfth National Conference on Artificial Intelligence of AAAI ’94, pages 230–237. American Association for Artificial Intelligence, 1994b.

Alexander Balke and Judea Pearl. Bounds on treatment effects from studies with imperfect compliance. Journal ofthe American Statistical Association, 92(439):1171–1176, 1997.

Peter Bickel, Eugene Hammel, and William O’Connell. Sex bias in graduate admissions: data from Berkeley. Science, 187(4175):398–404, 1975.

Anna Rodum Bjøru, Rafael Cabanas, Helge Langseth, and Antonio Salmer ˜ on. Divide and conquer for causal computa-´ tion. International Journal ofApproximate Reasoning, 186:109520, 2025.

Stephan Bongers, Patrick Forre, Jonas Peters, and Joris M Mooij. Foundations of structural causal models with cycles´ and latent variables. The Annals ofStatistics, 49(5):2885–2915, 2021.

Zhihong Cai, Manabu Kuroki, Judea Pearl, and Jin Tian. Bounds on direct effects in the presence of confounded intermediate variables. Biometrics, 64(3):695–701, 2008.

Abraham Charnes and William W. Cooper. Programming with linear fractional functionals. Naval Research Logistics Quarterly, 9(3-4):181–186, 1962.

Juan D. Correa, Sanghack Lee, and Elias Bareinboim. Nested counterfactual identification from arbitrary surrogate experiments. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems. Curran Associates Inc., 2021.

Fabio G. Cozman, Radu Marinescu, Junkyu Lee, Alexander Gray, and Denis D. Maua. Dealing with cycles in graph-´ based probabilistic models: the case of logical credal networks. In Proceedings of the Fourteenth International Symposium on Imprecise Probabilities: Theories and Applications, volume 290 of Proceedings ofMachine Learning Research, pages 93–102. PMLR, 15–18 Jul 2025.

Guilherme Duarte, Noam Finkelstein, Dean Knox, Jonathan Mummolo, and Ilya Shpitser. An automated approach to causal inference in discrete settings. Journal ofthe American Statistical Association, 119(547):1778–1793, 2024.

Robin J. Evans. Graphs for margins of Bayesian networks. Scandinavian Journal ofStatistics, 43(3):625–648, 2016.

David Galles and Judea Pearl. An Axiomatic Characterization of Causal Counterfactuals. Foundations of Science, 3(1): 151–182, 1998.

Joseph Y. Halpern. Axiomatizing Causal Reasoning. Journal ofArtificial Intelligence Research, 12:317–337, 2000.

Yuta Kawakami, Manabu Kuroki, and Jin Tian. Probabilities of causation for continuous and vector variables. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pages 1901–1921. PMLR, 15–19 Jul 2024.

Kristian Kersting, Martin Mladenov, and Pavel Tokmakov. Relational linear programming. Artificial Intelligence, 244: 188–216, 2017.

Charles F. Manski. Nonparametric bounds on treatment effects. The American Economic Review, 80(2):319–323, 1990.

Charles F. Manski. Monotone Treatment Response. Econometrica, 65(6):1311, 1997.

Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–688, 1995.

Judea Pearl. Probabilities of causation: Three counterfactual interpretations and their identification. Synthese, 121(1): 93–149, 1999.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009.

Drago Plecko and Elias Bareinboim. Causal Fairness Analysis: A Causal Toolkit for Fair Machine Learning. ˇ Foundations and Trends® in Machine Learning, 17(3):304–589, 2024.

Michael C. Sachs, Gustav Jonzon, Arvid Sjolander, and Erin E. Gabriel. A General Method for Deriving Tight Symbolic¨ Bounds on Causal Effects. Journal ofComputational and Graphical Statistics, 32(2):567–576, 2023.

Ilya Shpitser and Judea Pearl. What Counterfactuals Can Be Tested. 2007.

Hanmei Sun, Chengfeng Shi, and Qiang Zhao. Bounding the probability of causation under ordinal outcomes. Communications in Statistics-Theory and Methods, 54(24):8121–8132, 2025.

Jin Tian. Studies in causal reasoning and learning. PhD thesis, University of California, Los Angeles, 2002.

Jin Tian and Judea Pearl. Probabilities of causation: Bounds and identification. Annals of Mathematics and Artificial Intelligence, 28(1-4):287–313, 2000.

Marco Zaffalon, Alessandro Antonucci, Rafael Cabanas, and David Huber. Approximating counterfactual bounds while˜ fusing observational, biased and randomised data sources. International Journal ofApproximate Reasoning, 162: 109023, 2023.

Marco Zaffalon, Alessandro Antonucci, Rafael Cabanas, David Huber, and Dario Azzimonti. Efficient computation of˜ counterfactual bounds. International Journal ofApproximate Reasoning, 171:109111, 2024.

Junzhe Zhang, Jin Tian, and Elias Bareinboim. Partial counterfactual identification from observational and experimental data. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 26548–26558, 2022.

## A Proofs

Proof of Thm. 1. Let us separately prove the non-strict inequalities between the two sides of Eq. (5). By definition, $M _ { \sigma }$ is based on a causal graph compatible with $\sigma .$ . This implies $\mathbb { M } _ { M _ { \sigma } } \supseteq \mathbb { M } _ { \sigma }$ and hence the $\leq$ inequality.

In order to prove the opposite inequality, let $\mathcal { M } ^ { \ast }$ be an SCM that solves Eq. (4). By definition of $\mathbb { M } _ { \sigma }$ , the causal graph of $\mathcal { M } ^ { * }$ is consistent with the total order $\sigma .$ . Therefore, for each $V ^ { ( i ) } \in V$ , the endogenous parents of $V ^ { ( i ) }$ in $\mathcal { M } ^ { \ast }$ form a subset of its predecessors $V _ { \prec i }$ under $\sigma$ . We can enlarge each SE of $V ^ { ( i ) }$ by including as inputs all variables in $V _ { \prec i }$ that are not already endogenous parents of $V ^ { ( i ) }$ , and by similarly incorporating all exogenous variables not already among its exogenous parents. In this way, the collection U of exogenous variables can be treated as a single aggregated variable U. Consequently, $\mathcal { M } ^ { * }$ may be regarded as an SCM that is consistent with the same causal graph as the PSCM $M _ { \sigma }$ . Since $M _ { \sigma }$ is canonical, there exists an equivalent SCM in ${ \mathbb M } _ { M _ { \sigma } }$ . This implies the $\geq$ inequality. □

Lemma 1. For any atomic counterfactual proposition $V _ { \mathbf { x } } ~ = ~ v$ and $\smash { \pmb { s } _ { \sigma } \in \pmb { S } _ { \sigma } }$ , the truth value of the entailment $\mathbf { \mathscr { s } } _ { \sigma }  = ( V _ { \mathbf { \mathscr { x } } } = v )$ is determined. Furthermore, $i f \gamma$ is a conjunction of such propositions, the truth value of $s _ { \sigma } \models \gamma$ is also determined.

Proof. Consider an arbitrary intervention x. We proceed by induction on the topological order $\sigma$ to show that the potential response of every variable $V ^ { ( i ) }$ is uniquely determined by $\scriptstyle { \pmb { s } } _ { \sigma }$ and x.

$\operatorname { I f } i = 1$ , the variable $V ^ { ( 1 ) }$ has no endogenous parents. $\mathrm { I f } V ^ { ( 1 ) } \in { \pmb X }$ , its value is fixed to the assignment specified in x. If $V ^ { ( 1 ) } \not \in { \cal X }$ , its value is uniquely determined by the realization encoded in $\scriptstyle { \pmb { s } } _ { \sigma }$

Assume now that for some $i > 1$ , the interventional values of all preceding variables $V _ { \prec i }$ are fixed to $( { \pmb v } _ { \prec i } ) _ { \pmb x }$ . We determine the value of $V ^ { ( i ) }$ as follows:

1. If $V ^ { ( i ) } \in X$ , its value is determined by the assignment in x.

2. If $V ^ { ( i ) } \not \in { \cal X }$ , its value is uniquely determined by $\scriptstyle { \pmb { s } } _ { \sigma }$ given the values of its parents, which are fixed to $( { \pmb v } _ { \prec i } ) _ { \mathfrak { s } }$ x by the induction hypothesis.

Since $| V | < \infty$ , every counterfactual variable $V _ { \pmb { x } }$ evaluates to a unique constant under $\scriptstyle { \pmb { s } } _ { \sigma }$ . Consequently, the atomic entailment is well-defined. By extension, the entailment for a conjunction $\begin{array} { r } { \gamma = \bigwedge _ { j = 1 } ^ { k } ( Y _ { \pmb { x } ^ { ( j ) } } ^ { ( j ) } = y ^ { ( j ) } ) } \end{array}$ is determined by:

$$
( s _ { \sigma } \Vdash \gamma ) \iff \bigwedge _ { j = 1 } ^ { k } \left( s _ { \sigma } \Vdash Y _ { x ^ { ( j ) } } ^ { ( j ) } = y ^ { ( j ) } \right) .\tag{19}
$$

Lemma 2. For any arbitrary SCM $\mathcal { M } \in \mathbb { M } _ { M _ { o } }$ , there exists a bijection $h : \mathcal { U } \to \pmb { \mathscr { S } } _ { \sigma }$ between its exogenous domain U and the response signature domain $\scriptstyle { \pmb { s } } _ { \sigma }$

Proof. Let $h : \mathcal { U } \to \pmb { \mathscr { S } } _ { \sigma }$ map each exogenous state u to its induced signature $s _ { \sigma }$ . Because SEs are deterministic, each u induces exactly one signature, making h a well-defined function. We establish that h is a bijection.

By definition, $s _ { \sigma }$ comprises all valid configurations of potential outcomes which $M _ { \sigma }$ generates. Because $\mathcal { M } \in \mathbb { M } _ { M _ { \sigma } }$ , it follows a canonical specification (Zhang et al., 2022, Def. 2.3); its exogenous domain U explicitly indexes the Cartesian product of every possible deterministic function mapping from endogenous parents to children. Thus, for any arbitrary signature $\pmb { s } _ { \sigma } \in \pmb { S } _ { \sigma }$ , there is at least one state $u \in \mathcal { U }$ that generates it. We conclude that h is surjective.

By the same canonical specification of $\mathcal { M } ,$ , the cardinality of U is exactly equal to the cardinality of the Cartesian product of all possible functional mappings—which coincides with $| \pmb { S } _ { \sigma } |$ by construction in Eq. (6). Because h is a surjective function mapping between two finite sets of identical cardinality, it must necessarily be injective. Therefore, every state u corresponds uniquely to a signature $\scriptstyle { \pmb { s } } _ { \sigma }$ □

Proof of Thm. 2. By Thm. 1, without loss of generality, we can restrict our analysis to an arbitrary SCM $\mathcal { M } \in \mathbb { M } _ { M _ { \sigma } }$

By Lem. 2, there exists a bijection h mapping each exogenous state $u \in \mathcal { U }$ to a unique response signature $\boldsymbol { s } _ { \sigma } \in \boldsymbol { s } _ { \sigma }$ Consequently, the logical equivalence $( u \ | = \gamma ) \iff \ ( s _ { \sigma } \ | = \gamma )$ holds true for $\begin{array} { r } { \pmb { s } _ { \sigma } = h ( \ b { u } ) } \end{array}$ . The left-hand side is well-defined by the soundness axiomatisation of SCMs (Halpern, 2000), and the right-hand side is determined by Lem. 1.

Because h is bijective, the probability measure over U directly translates to $\scriptstyle { \pmb { S } } _ { \sigma } .$ , yielding $P ( U = u ) = P ( S _ { \sigma } = \pmb { s } _ { \sigma } )$ Substituting u with $s _ { \sigma }$ in Eq. (1), we obtain:

$$
P _ { \mathcal { M } } ( \gamma ) = \sum _ { \pmb { s } _ { \sigma } \in \pmb { \mathscr { S } } _ { \sigma } } \mathbb { \mathbb { \mathbb { E } } } _ { \sigma } \Vdash \gamma \mathbb { \mathbb { I } } \cdot P ( \pmb { S } _ { \sigma } = \pmb { s } _ { \sigma } ) .
$$

Proof of Thm. 3. Let $L ^ { * }$ be the minimum of the optimisation problem in $\operatorname { E q . } \ ( 4 )$ , and let L be the minimum of the LP task defined in Eq. (9). To prove the theorem, we first show that $L \leq L ^ { * }$ , and then $L \geq L ^ { * }$

Assume the optimal value $L ^ { * }$ is achieved by some SCM $\mathcal { M } ^ { \ast } \in \mathbb { M } _ { o }$ for query $P _ { \mathcal { M } ^ { \ast } } ( \gamma )$ . By Thm. 1, we can consider $\mathcal { M } ^ { * }$ to be an SCM parametrising the canonical PSCM $M _ { \sigma }$ . By construction of $\scriptstyle { \pmb { S } } _ { \sigma }$ , the exogenous distribution of $\mathcal { M } ^ { * }$ maps to a valid joint probability distribution $\mathbf { q } ^ { * }$ over the states $\pmb { s } _ { \sigma } \in \pmb { S } _ { \sigma }$ . By Thm. 2, the objective query can be expressed as a linear combination of the probabilities associated with such states. Furthermore, since $\mathcal { M } ^ { * }$ is a feasible solution to Eq. (4), it must satisfy the observational constraint $P _ { \mathrm { \mathcal { M } ^ { \ast } } } ( V ) = \hat { P } ( V )$ . Correspondingly, $\mathbf { q } ^ { * }$ satisfies the $\mathrm { L P } ^ { * } \mathrm { s }$ constraint imposed by observational data as by Cor. 1. Consequently, $\mathbf { q } ^ { * }$ constitutes a feasible solution within the constraints of the LP formulation in Eq. (9). Because the LP identifies the global minimum L over all feasible distributions, it must be that $L \leq L ^ { * }$

Conversely, let $\mathbf { q } ^ { * }$ be the optimal solution that minimises the LP task in Eq. (9), such that the objective evaluates to $L$ Consider a specific SCM $\mathcal { M } \in \mathbb { M } _ { \sigma }$ ; our argument will equivalently consider $\mathcal { M } \in \mathbb { M } _ { M _ { \alpha } }$ (after Thm. 1). We define its exogenous distribution by assigning $\bar { P _ { \mathcal M } ( u ) } = q _ { h ( u ) } ^ { * }$ , where h is the bijection established in Lem. 2. Additionally, because the LP enforces the observational constraints, M matches the empirical distribution, then it is also a feasible candidate for the optimisation task in Eq. (4). Since $L ^ { * }$ represents the minimum over all feasible SCMs in $\mathbb { M } _ { \sigma }$ , and we have found a specific feasible SCM M that achieves L, it must hold that $L ^ { * } \leq L$

Since $L \leq L ^ { * }$ and $L ^ { * } \leq L$ , we conclude that $L = L ^ { * }$ . Furthermore, the explicit construction in the second half of the proof demonstrates that there exists an SCM $\mathcal { M } \in \mathbb { M } _ { M _ { \sigma } }$ that precisely attains the LP bounds. □

Proof of Thm. 4. Let $Q _ { L P }$ and $Q _ { L L P }$ denote the feasible regions of the LP in Eq. (9) and of the lifted LP in Eq. (11), respectively.

For any feasible solution $\pmb q ^ { * } \in Q _ { L P }$ , by definition of the equivalence relation in Eq. (10), $\begin{array} { r } { q _ { { t _ { \sigma } } } = \sum _ { s _ { \sigma } \in t _ { \sigma } } q _ { s _ { \sigma } } ^ { * } } \end{array}$ trivially satisfies the constraints of Eq. (11) whilst preserving the objective value, as the Boolean evaluations are constant within a class $\mathbf { \Delta } _ { t _ { \sigma } } \in \mathcal { T } _ { \sigma }$

Conversely, consider any feasible solution $\pmb q _ { t _ { \sigma } } ^ { * } \in Q _ { L L P }$ . Let $k _ { t _ { \sigma } } : = | t _ { \sigma } |$ be the number of signatures $s _ { \sigma }$ contained within a given equivalence class $t _ { \sigma }$ . Note that by construction every class $t _ { \sigma }$ is non-empty, hence $k _ { t _ { \sigma } } > 0$ . We construct a valid distribution in the original space by assigning a uniform mass $q _ { s _ { \sigma } } = q _ { t _ { \sigma } } ^ { \ast } / k _ { t _ { c } }$ to every signature $\mathbf { \boldsymbol { s } } _ { \sigma } \in t _ { \sigma }$ , for all $\mathbf { \Delta } \mathbf { \mathbf { t } } _ { \sigma } \in \mathcal { T } _ { \sigma }$ . Because the logical constraints evaluate identically across $\scriptstyle t _ { \sigma } ,$ , summing over this uniformly distributed mass recovers the constraints and the objective value of the lifted LP, satisfying Eq. (9).

Since these mappings strictly preserve both feasibility and the objective value in both directions, the extrema evaluated over $Q _ { L L P }$ must exactly match those over $Q _ { L P }$ . Following the constructive argument detailed in the proof of Thm. 3, there exists a witness SCM that attains these bounds, ensuring the tightness is inherited by the lifted $\mathrm { L P }$ in Eq. (11).

Proof of Thm. 5. Let $\mathcal { M } ^ { \ast } \in \mathbb { M } _ { \sigma }$ and $\tilde { \mathcal { M } } ^ { \ast } \in \mathbb { M } _ { \sigma } ^ { V _ { \gamma } }$ denote the SCMs giving the optima in, respectively, Eq. (4) and Eq. (12). Let also $L : = P _ { \mathcal { M } ^ { \ast } } ( \gamma )$ and $L ^ { \prime } : = P _ { \tilde { \mathcal { M } } ^ { \ast } } ( \gamma )$ . Following, for instance, Bongers et al. (2021), we can marginalise out from $\mathcal { M } ^ { * }$ the variables in $\dot { V } \setminus V _ { \gamma }$ and obtain a well-defined SCM over $V _ { \gamma }$ . Moreover, by construction, we have $P _ { \mathcal { M } } ^ { * } ( V ) = \hat { P } ( V )$ , and this implies, in the marginal SCM, $P _ { \mathcal M } ^ { * } ( V _ { \gamma } ) = \hat { P } ( V _ { \gamma } )$ . Overall, we have an SCM in $\mathbb { M } _ { \sigma } ^ { V _ { \gamma } }$ giving L, and this proves $L \geq \bar { L } ^ { \prime }$

To prove the inverse inequality, let $P ( V ^ { \prime } \mid V _ { \gamma } ) : = \hat { P } ( V ) / \hat { P } ( V _ { \gamma } )$ , where $V ^ { \prime } : = V \setminus V _ { \gamma }$ . We augment the SCM $\tilde { \mathcal { M } } ^ { * }$ with the variables in $V ^ { \prime }$ . By setting all the variables in $V _ { \gamma }$ as endogenous parents of those in $V ^ { \prime }$ , we can easily obtain $P ( V ) = \hat { P } ( V )$ . Thus we have an SCM in $\mathbb { M } _ { \sigma }$ such that $P ( V ) = \hat { P } ( V )$ and giving $L ^ { \prime } .$ . This proves $L \leq L ^ { \prime }$ , and hence the thesis. □

Proof of Thm. 6. Let $\sigma \in \Sigma ( \prec _ { \gamma } )$ be an arbitrary total order compatible with $\prec _ { \gamma } .$ , and consider the lifted LP in Eq. (11) constructed over $\sigma .$ Each equivalence class $\mathbf { \Delta } _ { t _ { \sigma } } \in \mathcal { T } _ { \sigma }$ is uniquely characterised by an observational assignment $v \in \nu$ and a Boolean query evaluation $b = \ [ t _ { \sigma } \models \gamma \mathbb { I } \in \{ 0 , 1 \}$

An equivalence class indexed by $\langle { \pmb v } , b \rangle$ is non-empty (i.e., contains at least one signature $\textstyle s \in { \mathcal { S } } _ { \sigma } )$ if and only if v and b do not contradict the consistency axiom (Galles and Pearl, 1998). Specifically, for any atomic proposition $\mathrm { \dot { \cal Y } } _ { x } = y ^ { \ast }$ in $\gamma ,$ if v assigns $X = x$ , consistency forces $Y _ { x } = v _ { Y }$ . Hence, if $v _ { Y } \ne y ^ { * }$ , consistency prevents γ from being satisfied, ruling out $b = 1 ;$ ; conversely, if v forces every atomic term in γ to hold, consistency rules out $b = 0$

Crucially, whether a pair $\langle { \pmb v } , b \rangle$ satisfies the consistency axiom depends solely on the assignment v and the syntax of $\gamma _ { : }$ , independently of the topological order σ. Moreover, because the canonical specification contains all possible functional responses, every pair $\langle { \pmb v } , b \rangle$ that is compatible with consistency is realised by at least one signature $\textstyle s \in { \mathcal { S } } _ { \sigma } :$ the observational response is fixed to v, while the counterfactual responses under unobserved interventions can be set to satisfy γ (yielding $b \overset { \_ } { = } 1 )$ or violate at least one atom of $\gamma$ (yielding $b = 0 )$

We can conclude that the quotient space $\boldsymbol { \mathcal { T } } _ { \sigma }$ of non-empty equivalence classes is invariant to $\sigma .$ Indexing the lifted optimisation variables as $\boldsymbol { q } _ { \langle \boldsymbol { v } , \boldsymbol { b } \rangle }$ , the LP objective function is $\textstyle \sum _ { v } q _ { \langle v , 1 \rangle }$ and the empirical constraints are $q _ { \langle v , 0 \rangle } + q _ { \langle v , 1 \rangle } =$ $\hat { P } ( v )$ for each $v \in \nu .$ Because the optimisation variables, constraints, and objective function are entirely decoupled from σ, the resulting LP is identical for all linear extensions $\sigma \in \Sigma ( \prec _ { \gamma } )$ . Solving the $\mathrm { L P }$ for any arbitrary compatible σ therefore yields the same bounds as Eq. (12) and solves the global optimisation task in Eq. (13). □

## B Additional Experiments

<table><tr><td>Method</td><td>Lower</td><td>Upper</td></tr><tr><td>Vitamin A</td><td></td><td></td></tr><tr><td>Balke and Pearl (1997) This paper</td><td>-0.195 -0.587</td><td>0.005 0.412</td></tr><tr><td>Coronary</td><td></td><td></td></tr><tr><td>Balke and Pearl (1997)</td><td></td><td></td></tr><tr><td></td><td>0.262</td><td>0.868</td></tr><tr><td>This paper</td><td>-0.685</td><td>0.971</td></tr></table>

Table 3: Comparison of ACE bounds. Lower and upper bounds for the ACE grouped by Study.

ACE with Imperfect Compliance. We consider a model with three Boolean variables. In addition to the outcome ${ \mathit { Y } } ,$ we distinguish between treatment exposure X and treatment assignment $Z .$ To evaluate the causal effect of the treatment when assigned, we focus on the query ${ \mathrm { A C E } } _ { X = 0 , X = 1 } ( Y = 1 )$ , defined as in Eq. (16). Observational information is available only in the form of a conditional empirical distribution ${ \hat { P } } ( X , Y | Z )$ . In our framework, consistency with the empirical distribution is expressed as:

$$
\frac { \sum _ { \pmb { s } _ { \sigma } \in { \pmb { S } } _ { \sigma } } \left[ \pmb { s } _ { \sigma } \left[ = ( x , y , z ) \right] \right] \cdot q _ { \pmb { s } _ { \sigma } } } { \sum _ { \pmb { s } _ { \sigma } \in { \pmb { S } } _ { \sigma } } \left[ \pmb { s } _ { \sigma } \left[ = z \right] \right] \cdot q _ { \pmb { s } _ { \sigma } } } = \hat { P } ( x , y | z ) ,\tag{20}
$$

which yields linear constraints over the optimisation variables $q _ { s _ { \sigma } }$ , thus preserving the LP structure of our method. Although the query involves only X and Y, the conditional nature of the empirical distribution prevents a marginalisation of $Z .$ Consequently, the response signature must be based on all three Boolean variables. By Thm. 6 we choose an arbitrary admissible order σ such that $X \prec Y$

We compare our bounds with those obtained from the LP formulation introduced by Balke and Pearl (1997) and later generalised by Sachs et al. (2023). That approach requires an explicit causal graph. In particular, the model in Fig. 1b assumes that $Z$ affects Y only through X, and that $\bar { Z }$ is independent of an unobserved confounder U affecting both X and Y. Tab. 3 reports the resulting comparison for two observational studies: the vitamin A supplementation study in northern Sumatra and the coronary primary prevention trial. Our bounds remain informative, but are substantially wider than those of Balke and Pearl (1997). This difference reflects the fact that our framework relies only on the order restrictions induced by the query, whereas the graphical model in Fig. 1b imposes additional structure, most notably the exclusion of a direct effect from Z to Y. To model these effects with such a granularity one would either impose additional (non-linear) constraints or eventually require a causal graph, but this lies outside manuscript’s scope. By Thm. 3, the more extreme values admitted by our procedure are attainable under SCMs compatible with our weaker assumptions, including models in which Z may directly affect Y.

As pointed out by Balke and Pearl (1994a), bounds can be further tightened if additional assumptions are made about subject behaviours: imposing a monotonic constraint, e.g. a positive response on treatment, $\bar { Y _ { x _ { 1 } , z } } \geq Y _ { x _ { 0 } , z }$ for every $z \in { \mathcal { Z } }$ . As a consequence, negative lower bounds on the effects are raised to zero. Nevertheless, even when combined with additional assumptions such as weak exogeneity, these restrictions are still insufficient to recover the analytical bounds implied by the graphical model, as discussed in Sect. 5.

![](images/34addec5ca4da7cd9e277304f21b85d57fa723dab0d0f17001fc2b65199c1094.jpg)  
Figure 2: End-to-end computation time for GPNS(r) bounds, on a semi-logarithmic scale, as a function of the variable cardinalities $( n = | \mathcal { X } | = \mathrm { \bar { | } } \mathcal { Y } | )$ .

PNS Generalisation for Non-Binary Variables To assess the practical scalability of the lifted LP in Eq. (11), we measured the end-to-end time required to construct the signature equivalence classes, aggregate the lifted variables, and solve the resulting LP. The benchmark query is a multi-hypothetical generalisation of PNS for ordinal treatment and outcome variables, in the spirit of Kawakami et al. (2024). Let X and Y be ordinal treatment and effect variables, respectively, defined over finite domains $\mathcal { X } = \{ x ^ { ( 0 ) } , \dots , x ^ { ( n - 1 ) } \}$ } and $\mathcal { Y } = \{ \boldsymbol { y } ^ { ( 0 ) } , \dots , \boldsymbol { y } ^ { ( n - 1 ) } \}$ where n is an experimental parameter. The GPNS of order r (where $2 \leq r \leq n )$ evaluates the joint probability of strict compliance across r distinct interventions, defined as:

$$
{ \mathrm { G P N S } } ( r ) : = P \left( \bigwedge _ { i = 0 } ^ { r - 1 } Y _ { x ^ { ( i ) } } = y ^ { ( i ) } \right) .\tag{21}
$$

For binary X and Y , GPNS(2) reduces to the standard PNS event. For each $n \in \{ 4 , \ldots , 1 5 \}$ , we randomly parametrise a Bayesian network with the endogenous structure in Fig. 1a, estimated ${ \hat { P } } ( X , Y )$ from simulated observational data, and computed bounds for $\mathrm { G P N S } ( \bar { r } )$ with $r \in \{ 2 , 3 , 4 \}$ . Figure 2 reports the resulting end-to-end computation times.

![](images/981c5244454a59d32603b885ddb8ab39680585b991b59092e75311451239d2ca.jpg)  
Figure 3: Examples of canonical PSCM specifications compatible with the endogenous order $\sigma : = ( X , W , Y )$ (a) Markovian specification. (b) Specification with one exogenous variable shared by W and $Y .$ . (c) Specification with one exogenous variable shared by all endogenous variables. Grey nodes denote exogenous variables.

For this benchmark, $V _ { \gamma } = \{ X , Y \}$ , so the lifted LP has at most $| { \mathcal { X } } | | { \mathcal { V } } | = n ^ { 2 }$ variables, as derived in Sect. 3. Results are consistent with the polynomial scaling of the lifted formulation, although we must consider the overhead introduced by the symmetry detection, LP construction, and numerical optimisation. As expected, solving the LP remains tractable for standard solvers as the domain expands.

## C Canonical Specifications

We expand the discussion of canonicalisation of SCMs and PSCMs briefly introduced in Sect. 3. We recall here that the canonical PSCM used in the proof of Thm. 1 builds upon the theoretical contributions of Zhang et al. (2022) and Balke and Pearl (1994a). Generally speaking, an SCM takes its structural equations (SEs) as given. When these are unknown and the endogenous variables take on finitely many values, the literature shows that one may still specify a PSCM from the causal graph alone by adopting a canonical representation.

For an endogenous variable $V \in V$ with parents $\operatorname { P a } _ { V } .$ , a state of its exogenous parent $U _ { V }$ acts as a selector for a deterministic mapping (or responsefunction) from the domain of $\operatorname { P a } _ { V }$ to $V$ . There are exactly $| \mathcal { V } | ^ { | \mathrm { P a } _ { V } | }$ such possible functions, where $\mathrm { | } \mathrm { \hat { P } a } _ { V } \mathrm { | } : = \mathrm { et { } { ' } \sum _ { Z \in \mathrm { P a } _ { V } } \tilde { | Z | } }$ . In the absence of unobserved confounding, as in Markovian models, each $U _ { V }$ is independent of all other exogenous variables and can be replaced without loss of generality by a discrete canonical variable of finite cardinality $| \check { \mathcal { U } } _ { V } | = | \mathcal { V } | ^ { | \mathrm { P a } _ { V } | }$ When latent confounding is present, as in semi-Markovian models, unobserved variables may jointly affect multiple endogenous variables. The graph is then partitioned into confounded components (or c-components) (Tian, 2002), defined as the maximal sets of endogenous variables connected by paths of latent confounders (see confounded paths in Shpitser and Pearl, 2007). In a canonical specification, all latent confounding within each c-component ${ \bar { C } } \subseteq V$ is replaced without loss of generality by a single discrete exogenous variable $U _ { C }$ that jointly selects the response functions for every variable in $C ,$ whilst the exogenous variables associated with distinct c-components remain mutually independent. Because the state space of $U _ { C }$ must encompass all joint assignments of response functions, its canonical space $\mathcal { U } _ { C }$ indexes all possible combinations of individual mappings, yielding $\begin{array} { r } { | \mathcal { U } _ { C } | = \bar { \prod } _ { V \in C } | \mathcal { V } | ^ { | \mathrm { P a } _ { V } | } } \end{array}$ . Canonicalisation guarantees that any discrete SCM (or PSCM) admits a discrete equivalent inducing an identical set of observational and counterfactual distributions (Evans, 2016; Duarte et al., 2024).

To ground the ideas, we proceed by means of an example. Consider $V : = \{ X , W , Y \}$ with binary domains $\mathcal { X } = \mathcal { Y } =$ $\{ 0 , \bar { 1 } \}$ and ternary domain $\mathcal { W } = \mathrm { \bar { \{ 0 , 1 , 2 \} } }$ . We assume the causal order $\dot { X } \prec W \prec Y$ . The three panels in Fig. 3 share the same endogenous structure compatible with this order. A canonical specification of these causal diagrams is obtained by enumerating the deterministic response functions for each endogenous variable. First, consider the unconfounded setting in Fig. 3a, where every variable forms a singleton c-component. Since $\mathrm { P a } _ { X } = \varnothing .$ , there are two constant mappings, giving $\vert \bar { \mathcal { U } } _ { X } \vert = 2$ . For W, whose parent is X, the mechanism $w  f _ { W } ( x , u _ { W } )$ assigns a state in W to each $x \in { \mathcal { X } } ;$ hence, $| \mathcal { U } _ { W } | = | \mathcal { W } | ^ { | \mathcal { X } | } = 3 ^ { 2 } = 9$ . Finally, the mapping $y  f _ { Y } ( x , w , u _ { Y } )$ assigns a value in $\mathcal { V }$ to every configuration $( x , w ) \in \mathcal { X } \times \mathcal { W }$ , yielding $| \mathcal { U } _ { Y } | = | \mathcal { V } | ^ { | \mathcal { X } | \cdot | \mathcal { W } | } = \bar { 2 } ^ { \bar { 2 } \cdot 3 } = 6 4$ . Next, consider Fig. 3b, where unobserved confounding between $W$ and Y induces the partition into c-components {X} and $\{ W , Y \}$ . The canonical confounder $U _ { W Y }$ jointly indexes the mappings of both $\bar { W }$ and Y , yielding $| \bar { \mathcal { U } } _ { W Y } | = | \bar { \mathcal { U } } _ { W } | \cdot | \bar { \mathcal { U } } _ { Y } | = 9 \cdot 6 4 = 5 7 6$ , whilst $\vert \mathcal { U } _ { X } \vert = 2$ Lastly, in the fully confounded model of Fig. 3c, all endogenous variables belong to a single c-component $\dot { C } = \mathbf { \dot { V } } . \mathrm { ~ A ~ }$ single global exogenous parent U governs all mechanisms, with cardinality $| \mathcal { U } | = \mathbf { \bar { \lvert } } \mathcal { U } _ { X } | \cdot | \mathcal { U } _ { W } ^ { - } | \cdot | \mathcal { U } _ { Y } | \stackrel { \cdot } { = } 2 \cdot 9 \cdot 6 4 = 1 , 1 5 2$ The canonical PSCM constructed in the proof of Thm. 1 matches this full-confounding specification.