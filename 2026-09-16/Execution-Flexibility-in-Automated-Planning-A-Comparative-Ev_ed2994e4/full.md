# Execution Flexibility in Automated Planning: A Comparative Evaluation of Deordering and Reordering Strategies

Md. Monjurul Islam<sup>1</sup>, Sabah Binte Noor<sup>1</sup>, Fazlul Hasan Siddiqui<sup>1</sup>, Gahangir Hossain

<sup>1</sup>Department of Computer Science and Engineering, Dhaka University of Engineering & Technology, Gazipur, Gazipur-1707, Bangladesh. <sup>2</sup>Department of Data Science, , University of North Texas, Denton, TX, USA.

\*Corresponding author(s). E-mail(s): Gahangir.Hossain@unt.edu; Contributing authors: monjurul.islam.cs@gmail.com; sabah@duet.ac.bd; siddiqui@duet.ac.bd;

## Abstract

This study covers foundational concepts for enhancing plan-execution flexibility, including partial-order planning, the producer-consumer-threat formalism, and a range of deordering and reordering strategies. Creating a partial-order plan from a sequential one by removing unnecessary ordering constraints is a practical way to improve execution flexibility, and several methods have been proposed for this task. This study analyzes their capabilities across ordering, action handling, parameter handling, plan structure, concurrency, and complexity, and evaluates them against each other on a shared benchmark. The central finding is that block deordering-based approaches, which restructure causal dependencies through block-level grouping and subplan substitution, substantially outperform MaxSAT-based approaches despite the latter’s theoretical guarantees of minimum reordering. The reason is structural: minimum reordering optimizes within the causal structure already present in the plan, whereas block deordering-based methods change that structure, exposing orderings that would otherwise appear necessary. A further distinction is practical: block deordering-based methods are anytime algorithms that always return a valid result, while MaxSAT-based methods fail entirely on a substantial portion of plans and ofer no partial solution when they do. Block substitution further extends the parallel execution by formalizing non-concurrency constraints, though its impact is limited to

domains with resource-based interactions. On eficiency, block deordering-based approaches achieve the highest flex gain per unit of computation time, while MaxSAT-based encodings incur large computational overhead.

Keywords: Automated Planning, Flexibility, Concurrency, Partial Order Planning

## 1 Introduction

The capability of autonomous agents to adapt to unforeseen circumstances represents a fundamental requirement for efective operation in dynamic environments. Within the domain of automated planning, multiple methodologies have been developed to enhance agent flexibility, ranging from providing agents with alternative plan selections (Graham et al. 2003) to extending the applicability of existing plans through generalization techniques (Anderson and Farley 1988). Plan generalization approaches generate partial-order plans (POPs) by reprieving specific action orderings until such orderings become inevitable. This strategy of minimizing action ordering decisions has been explored through various concepts, including plan deordering (Nguyen and Kambhampati 2001; Veloso et al. 2002; Siddiqui and Haslum 2012) and reordering (Muise et al. 2016) methodologies. While deordering the plan removes inessential action orderings, reordering the plan permits arbitrary changes to the action sequence.

One way to produce a POP is through partial-order causal link (POCL) planning (Weld 1994), which constructs a plan incrementally while maintaining causal support for every action precondition. A practical alternative is to take a sequential plan from a fast heuristic planner and remove the ordering constraints that are not necessary.

Several methods have been developed for this post-processing task. Explanationbased order generalization (EOG) (Kambhampati and Kedar 1994; Veloso et al. 2002) annotates each ordering with its causal justification and removes those without one. Block deordering (BD) (Siddiqui and Haslum 2012) groups coherent operators into blocks and eliminates additional orderings by treating each block as a unit. Minimum reordering (MR) (Muise et al. 2016) encodes the deordering problem as partial weighted MaxSAT to find provably minimum reorderings and remove redundant actions; minimum reinstantiated reordering (MRR) (Waters et al. 2020) extends this with parameter rebinding within same-name operators. More recently, FIBS (Noor and Siddiqui 2024) replaces subplans with alternatives using block-substitution, and CIBS (Noor and Siddiqui 2025) extends FIBS to model and minimize non-concurrency constraints for parallel execution.

We analyze capabilities of diferent algorithms across multiple dimensions (ordering, action handling, parameter handling, plan structure, concurrency, and complexity), and evaluate on 3,345 plans from 46 IPC domains. The analysis produces a result that is not obvious from the descriptions above. MR and MRR are designed to find minimum reorderings. Yet BD and FIBS produce substantially higher flex values on the same benchmarks, with large and statistically significant efect sizes against all other methods. The reason is that minimum reordering optimizes within the causal structure the original plan already has. Block deordering works diferently: it reorganizes which operators share causal dependencies by grouping them into blocks, exposing ordering constraints that would otherwise appear necessary. Blocksubstitution goes further, replacing subplans with alternatives that carry diferent causal entailments entirely. The two families of methods are not in competition so much as they are solving diferent problems; restructuring causal structure turns out to be the more productive one for increasing flexibility.

## 2 Preliminaries

This section outlines the foundational concepts required to comprehend the methods. It begins with defining planning tasks and plans within the finite-domain representation, followed by notions of partial-order planning, causal dependencies, and threats. Finally, it outlines key concepts of plan deordering and reordering.

## 2.1 Planning Task

Classical planning problems are commonly represented using finite domain representation (FDR) (Helmert 2011), which describes planning tasks through state variables and their associated value domains. The following definitions establish the mathematical foundation for subsequent analysis.

Definition 1. In FDR, a planning task is denoted as $\Pi = \langle { \mathcal { X } } , { \mathcal { O } } , s _ { i } , s _ { g } \rangle$ , where:

• X represents a finite non-empty set of state variables. A domain $\mathcal { D } _ { x }$ with a finite number of elements is associated with each $x \in \mathcal { X }$ . The pair $\langle x , d \rangle$ with $x \in \mathcal { X }$ and $d \in \mathcal { D } _ { x }$ is a fact. The notion of state corresponds to a function s defined over X, such that $s ( x ) \in \mathcal { D } _ { x }$ for all $x \in \mathcal { X }$ . The set of variables involved in a function is denoted as vars(s). A function sˆ similar to a state s, but defined over a proper subset of X, represents a partial state where vars $( \hat { s } ) \subset \mathcal { X }$

• O represents the finite non-empty set of operators. Every operator $o \in \mathcal { O }$ is associated with two partial states, precondition $\left( p r e _ { o } \right)$ and efect $( e f f _ { o } ) _ { ; }$ , along with an associated cost that is nonnegative $( c o s t _ { o } \in \mathbb { R } _ { 0 } ^ { + } )$ . An operator o can be applied in a state s if $p r e _ { o } \subseteq s$ . Applying the operator o within state s transforms it into state $s ^ { \prime } = a p p l y ( s , o )$ as specified in (1).

$$
s ^ { \prime } ( x ) = a p p l y ( s , o ) = \left\{ \begin{array} { l l } { e f f _ { o } ( x ) } & { i f x \in v a r s ( e f f _ { o } ) } \\ { s ( x ) } & { o t h e r w i s e } \end{array} \right.\tag{1}
$$

$s _ { i }$ corresponds to the initial state,

• $s _ { g }$ refers to a partial state that specifies the goal conditions.

Definition 2. Let $\pi = \left. o _ { 1 } , o _ { 2 } , \ldots , o _ { k } , \ldots , o _ { n } \right.$ be a plan, a sequence of operators, for a planning task $\Pi = \langle { \mathcal { X } } , { \mathcal { O } } , s _ { i } , s _ { g } \rangle$ . The plan π is valid $i f f p r e _ { o _ { 1 } } \subseteq s _ { i } , p r e _ { o _ { k + 1 } } \subseteq s _ { k }$ for every $k \in \{ 1 , 2 , \ldots , n - 1 \}$ where $s _ { k } = a p p l y ( s _ { k - 1 } , o _ { k } )$ , and $s _ { g } \subseteq s _ { n }$

Definition 3. The prod<sub>o</sub>, cons<sub>o</sub>, and $d e l _ { o }$ represent the facts that an operator o produces, consumes, and deletes, respectively. The fact−

$\left. x , d \right. \in p r o d _ { o } \ i f f \left. x , d \right. \in e f f _ { o } .$

$\langle x , d \rangle \in c o n s _ { o } \ i f f \ \langle x , d \rangle \in p r e _ { o } $

$\langle x , d \rangle \in d e l _ { o } \ i f f -$

i. either x /∈ vars(cons ) or cons (x) = d, and

ii. ef<sub>o</sub>(x) = d<sup>′</sup> such that $d ^ { \prime } \in ( D _ { x } \setminus \{ d \} )$

## 2.2 Partial-Order Planning

Operators can be executed in any possible sequence using the partial-order plan (POP) framework while imposing partial orderings over them. It is assumed that each operator is individually identifiable, even if an operator may appear multiple times within a POP.

Definition 4. A POP is represented as $\pi _ { p o p } = \langle \mathcal { O } , \prec \rangle$ , regarding a planning task $\Pi = \langle { \mathcal { X } } , { \mathcal { O } } , s _ { i } , s _ { g } \rangle$ , where:

• O denotes the set of operators, and

• ≺ represents ordering constraints over O. The notation $\mathbf { o _ { a } } \prec \mathbf { o _ { b } }$ represents a ordering constraint between two operators $o _ { a } , o _ { b } \in \mathcal { O }$ , which specifies that the operator $o _ { a }$ needs to be executed at any time before the execution of operator $o _ { b }$ . ≺ exhibits transitive property, which means that if $o _ { a } ~ \prec ~ o _ { b }$ and $o _ { b } ~ \prec ~ o _ { c }$ , then $o _ { a } \prec o _ { c }$ . Basic orderings represent constraints that are not transitively inferred by other constraints in the set.

Plan flexibility quantification uses the metric flex (Siddiqui and Haslum 2012), which is the proportion of the pairs of operators that do not have fundamental or transitive ordering to all operator pairs. Higher flex values indicate greater execution flexibility, though the relationship between flexibility and practical adaptability depends on domain-specific factors.

Definition 5. Let $\pi _ { p o p } = \langle \mathcal { O } , \prec \rangle$ be a POP. The flex value of the plan $\pi _ { p o p }$ is $\mathit { f l e x } ( \pi _ { p o p } )$ and is calculated as specified in (2).

$$
\mathrm { \Delta } f \mathrm { \Delta } e x ( \pi _ { p o p } ) = 1 - \frac { | \prec | } { \Sigma _ { i = 1 } ^ { | \mathcal { O } | - 1 } i }\tag{2}
$$

where $\Sigma _ { i = 1 } ^ { | \mathcal { O } | - 1 } i$ is the maximum number of pairings that might be constructed with a collection of |O| items.

The framework of producer-consumer-threat (PCT) (Bäckström 1998) establishes ordering structures by identifying operator relationships regarding fact production, consumption, and deletion. This formalism introduces causal links that map operator preconditions to efect providers.

Definition 6. A causal link $o _ { p } \xrightarrow { \langle x , d \rangle } o _ { c }$ between $o _ { p }$ and $o _ { c }$ specifies that $o _ { p } \prec o _ { c }$ and the operator $o _ { p }$ produces $\langle x , d \rangle$ for the operator $o _ { c }$ , where $\langle x , d \rangle \in ( p r o d _ { o _ { p } } \cap$ con ${ \bf \Pi } _ { { \bf \Pi } _ { { \bf \Pi } _ { { \bf \Pi } _ { { \bf \Pi } _ { { \bf \Pi } } } } } } ^ { 3 } \partial _ { c _ { c _ { } } } \left( \frac { { \bf \Pi } } { \Pi _ { { \bf \Pi } _ { { \bf \Pi } } c _ { } } } \right)$ A threat is defined as a conflict between a causal link $o _ { p } \xrightarrow { \langle x , d \rangle } o _ { c }$ and the efect of an operator $o _ { d }$ , where $o _ { d }$ deletes $\langle x , d \rangle$ and can be ordered in between $o _ { p }$ and $O _ { C } .$

A promotion, which adds the ordering $o _ { d } \ \prec \ o _ { p } ,$ or a demotion, which adds the ordering $o _ { c } \prec o _ { d } .$ , can resolve a threat between a causal link $o _ { p } \xrightarrow { \langle x , d \rangle } o _ { c }$ and an operator $o _ { d } . \mathrm { ~ A ~ P O P }$ is valid if causal relationships support each operator precondition without any threat (Weld 1994). Three labels, $P C , C T$ , and $T P _ { : }$ are introduced by Siddiqui and Haslum (2012) to outline a $\mathrm { P O P } \mathrm { { s } }$ ordering constraints.

Definition 7. Let $\pi _ { p o p } = \langle \mathcal { O } , \prec \rangle$ be a POP and $R e ( o _ { a } \prec o _ { b } )$ represents the ordering reasons for an ordering constraint $o _ { a } ~ \prec ~ o _ { b }$ . An ordering restriction $o _ { a } ~ \prec ~ o _ { b }$ might form for three reasons:

$P C ( \langle x , d \rangle ) \in R e ( o _ { a } \prec o _ { b } )$ is the producer-consumer of a fact $\langle x , d \rangle$ , where $o _ { a }$ produces $\langle x , d \rangle$ and $o _ { b }$ consumes $\langle x , d \rangle$ . The fact $\langle x , d \rangle$ may be produced and consumed by multiple operators. One operator $o _ { a }$ is assigned via a causal link to obtain $\langle x , d \rangle \ f o r \ o _ { b }$

$C T ( \langle x , d \rangle ) \in R e ( o _ { a } \prec o _ { b } )$ is the consumer-threat ofa fact $\langle x , d \rangle$ , where operator $o _ { a }$ consumes $\langle x , d \rangle$ and $o _ { b }$ deletes $\langle x , d \rangle$

$T P ( \langle x , d \rangle ) \in R e ( o _ { a } \prec o _ { b } )$ is the threat-producer of a fact $\langle x , d \rangle$ , where operator $o _ { a }$ deletes the fact $\langle x , d \rangle$ and there exist one or more causal link $o _ { b } \xrightarrow { \langle x , d \rangle } o _ { c }$ for some $o _ { c } \in \mathcal { O }$

Causal links of a POP are denoted by the label PC, and the demotion and promotion ordering constraints of the plan are denoted by $C T$ and $T P _ { : }$ respectively. These labels assist in identifying and monitoring the rationale behind the orderings in a POP.

## 2.3 Plan Deordering and Reordering

Plan deordering and reordering are two key concepts for attaining plan execution flexibility. Formal definitions of these notions are given in Bäckström (1998). A POP is assumed to be transitively closed in the Definition 8.

Definition 8. Let $\pi _ { p } = \langle \mathcal { O } , \prec _ { p } \rangle$ and $\pi _ { q } = \langle \mathcal { O } , \prec _ { q } \rangle$ be two diferent POPs with respect to Π (a planning task). Therefore:

• $\pi _ { q }$ is a deordering of $\pi _ { p }$ with respect to Π $i f f$ both $\pi _ { p }$ and $\pi _ { q }$ are valid $P O P$ and $\prec _ { q } \subseteq \prec _ { p }$

$\pi _ { q }$ is a proper deordering $o f \pi _ { p }$ with respect to Π $i f \pi _ { q }$ is a deordering of $\pi _ { p }$ $a n d \prec _ { q } \subset \prec _ { p }$

• $\pi _ { q }$ is a reordering of $\pi _ { p }$ with respect to Π if both $\pi _ { p }$ and $\pi _ { q }$ are valid POP.

• $\pi _ { q }$ is a proper reordering of $\pi _ { p }$ with respect to Π $i f \pi _ { q }$ is a reordering of $\pi _ { p }$ $w i t h \prec _ { q } \ne \prec _ { p }$

• $\pi _ { q }$ is a minimum deordering of $\pi _ { p }$ with respect to Π $i f f -$

1. $\pi _ { q }$ is a deordering $o f \pi _ { p } ,$ and

2. there exists no partial-order plan $\pi _ { r } = \langle \mathcal { O } , \prec _ { r } \rangle$ such that $\pi _ { r }$ is a deordering $o f \pi _ { p } \ w i t h \mid \prec _ { r } \mid < \mid \prec _ { q } \mid ,$

$\pi _ { q }$ is a minimum reordering of $\pi _ { p }$ with respect to Π $i f f -$

1. $\pi _ { q }$ is a reordering $o f \pi _ { p } ,$ and

2. there exists no partial-order plan $\pi _ { r } = \langle \mathcal { O } , \prec _ { r } \rangle$ such that $\pi _ { r }$ is a reordering of $\pi _ { p }$ and $| \prec _ { r } | < | \prec _ { q } |$

## 3 Partial-Order Causal Link-based Approach

The partial-order causal link (POCL) (Weld 1994) planning framework serves as the foundation for the traditional method of creating a partial-order plan (POP). An initial POP is iteratively refined through modifications involving operators, causal links, and ordering constraints in this paradigm. Such a preliminary plan typically begins with an initial operator and a goal operator. Refinement steps may introduce a new operator, insert a causal link connecting two operators, or establish an ordering constraint between them. A $\mathrm { P O P }$ is considered completed once each operator’s precondition is protected by a threat-free causal link. The two most popular POCL-based partial-order planners are VHPOP (Simmons and Younes 2011) and UCPOP (Penberthy and Weld 1992). Furthermore, the POCL strategy has been adapted to find temporal plans through state-based forward search techniques (Coles et al. 2021), and it also forms the foundation for numerous hierarchical planning approaches (Bercher et al. 2017, 2016; Bit-Monnot et al. 2020, 2016).

In addition to POCL, other techniques have been developed to generate $\mathrm { P O P s } .$ One such method is Petri net unfolding (Hickmott et al. 2007), which uses iterative unfolding of a specifically constructed Petri net to simulate the execution of a forward planning system. Another prominent approach is Graphplan (Blum and Furst 1997), which leverages a compact structure called a planning graph to derive optimal POPs. Over time, this planning graph has been used as a preprocessing mechanism for diverse planning systems, including STAN (Fox and Long 2001), IPP (Koehler 1999), and Blackbox (Kautz and Selman 1998).

## 4 Partial Weighted MaxSAT-based Approach

Transforming sequential plans into partial-order plans through deordering or reordering is a prominent approach for generating POPs. Early strategies for deordering involved generalizing and storing sequential plans in triangle tables (Regnier and Fade 1991; Fikes and Nilsson 1971), primarily to support plan modification and reuse. Triangle tables were later utilized as a preprocessing technique to take a sequential plan with conditional efects and extract partial ordering (Winner and Veloso 2002). More recent work has applied partial weighted MaxSAT encodings to enhance execution flexibility by reducing the orderings in a plan (Muise et al. 2016). Building on this, the action reinstantiation extends the MaxSAT formulation with additional constraints, allowing operator parameter reassignment(Waters et al. 2018, 2020).

The partial weighted MaxSAT problem extends the traditional satisfiability (SAT) problem by introducing two categories of clauses: hard and soft. Hard clauses function like those in the standard SAT problem and must always be satisfied. Soft clauses are not mandatory; instead, each is assigned a weight indicating its relative importance. Finding an assignment that satisfies all hard clauses and maximizes the cumulative weight of completely satisfied soft clauses is the goal.

## 4.1 MR Encodings

The task of achieving a plan’s minimum reordering (MR) as a partial weighted MaxSAT problem instance is formulated by Muise et al. (2016). The solution of the instance corresponds to a POP, which they denote as the target POP. For a given POP or sequential plan $\pi = \langle \mathcal { O } , \prec \rangle$ , the encoding employs three categories of propositional variables−

$x _ { o } \colon$ Indicates the presence of operator o in the target POP for each $o \in \mathcal { O }$

$\kappa ( o _ { a } , o _ { b } )$ : Indicates that the the target POP includes the ordering constraint $o _ { a } \prec o _ { b }$ for each operator pair $o _ { a } , o _ { b } \in \mathcal { O }$

$\Upsilon ( o _ { p } , \langle x , d \rangle , o _ { c } )$ : Indicates that a causal link $o _ { p } \xrightarrow { \langle x , d \rangle } o _ { c }$ is present in the target POP, for a pair of operator $o _ { p } , o _ { c } \in \mathcal { O }$ , specifies that the operator $o _ { p }$ produces $\langle x , d \rangle$ for the operator $o _ { c } ,$ where $\langle x , d \rangle \in ( p r o d _ { o _ { p } } \cap c o n s _ { o _ { c } } )$ .

The encoding procedure starts with defining hard clauses as Boolean formulae, which are then converted into conjunctive normal form (CNF), followed by the introduction of soft clauses and their associated weights. A soft clause with weight k is

$$
\scriptstyle { \binom { k } { \cdot \cdot \cdot } }
$$

represented by the syntax , whereas no weight indication represents a hard clause. To guarantee correctness, the encoding defines formulae (3) and (4) that enforce acyclicity in the target POP, where (4) ensures that the ordering constraints include the transitive closure. Formula (5) includes the initial and goal operators. Operators are treated as universally quantified, and in the case of formula (6), it is assumed that $o _ { I } \neq o _ { i } \neq o _ { G }$ . Formula (7) guarantees that every causal link remains threat-free by enforcing the required promotion or demotion orderings, while formula (8) ensures that each precondition is backed by an appropriate causal link.

$$
( \neg \kappa ( o , o ) )\tag{3}
$$

$$
\kappa ( o _ { p } , o _ { q } ) \wedge \kappa ( o _ { q } , o _ { r } ) \to \kappa ( o _ { p } , o _ { r } )\tag{4}
$$

$$
( x _ { o _ { I } } ) \wedge ( x _ { o _ { G } } )\tag{5}
$$

$$
x _ { o _ { i } }  \kappa ( o _ { I } , o _ { i } ) \wedge \kappa ( o _ { i } , o _ { G } )\tag{6}
$$

$$
\Upsilon ( o _ { p } , \langle x , d \rangle , o _ { c } ) \to \bigwedge _ { o _ { d } : \langle x , d \rangle \in d e l _ { o _ { d } } } x _ { o _ { d } } \to \kappa ( o _ { d } , o _ { p } ) \vee \kappa ( o _ { c } , o _ { d } )\tag{7}
$$

$$
x _ { o _ { c } }  \bigwedge _ { \langle x , d \rangle \in c o n s _ { o _ { c } } } \bigvee _ { o _ { p } \colon \langle x , d \rangle \in p r o d _ { o _ { p } } } \kappa ( o _ { p } , o _ { c } ) \wedge \Upsilon ( o _ { p } , \langle x , d \rangle , o _ { c } )\tag{8}
$$

For each operator and ordering variable in the encoding, a soft unit clause is introduced in formulae (9) and (10) containing the negation of that variable. If any of these unit clauses is violated, it indicates that the operator or the ordering constraint associated with the variable of the violated clause is incorporated into the solution. As a result, solutions with higher weights in the encoding correspond to target POPs that contain fewer ordering constraints.

$$
\begin{array}{c} \begin{array} { r } { \begin{array} { c c c } { { c o s t } _ { o } + | { \mathcal { O } } | ^ { 2 } + 1 } \\ { \left( \neg x _ { o } \right) } \end{array} , } & { \forall o \in \left( \mathcal { O } \setminus \left\{ o _ { I } , o _ { G } \right\} \right) } \end{array}  \end{array}\tag{9}
$$

$$
\begin{array} { c } { \mathrm { ~ \sigma ~ } _ { { \left( \neg \kappa \left( o _ { i } , o _ { j } \right) \right) } } ^ { 1 } \mathrm { ~ , ~ } \forall o _ { i } , o _ { j } \in \mathcal O } \end{array}\tag{10}
$$

## 4.2 MRR Encodings

The partial weighted MaxSAT encoding for achieving a plan’s minimum reordering is extended by Waters et al. (2020) through allowing modifications to both operator parameters and ordering constraints to achieve greater flexibility, defined as minimum reinstantiated reordering (MRR). Variables are denoted by symbols like $x , y ,$ , and $z ,$ constants by $^ { c , }$ and terms by $t , u ,$ and v. The definition of a term t is an ordered series of items, where $t [ i ]$ is the i-th element in the sequence. A mapping of terms with variables is designated as substitution $\Psi ;$ for instance, $\Psi = \{ x _ { 1 } / t _ { 1 } , x _ { 2 } / t _ { 2 } , \dots , x _ { n } / t _ { n } \}$ maps $x _ { i }$ to the corresponding term $t _ { i }$ for all $i \in \{ 1 , 2 , \ldots , n \}$ . The sets of variables and constants that appear in a given structure η are denoted by vars(η) and $c o n s t s ( \eta )$ In this approach, a POP is represented as $\pi = \langle \mathcal { O } , \Psi , \prec \rangle$ , where the set of operators is represented by $\mathrm { { } } ^ { \mathcal { O } , \mathrm { { \prec } } }$ is a strict partial order over O that is transitively closed, and Ψ is complete with respect to O.

Definition 9. Let $\pi _ { p } = \langle \mathcal { O } , \Psi , \prec \rangle$ and $\pi _ { q } = \langle \mathcal { O } , \Psi ^ { \prime } , \prec ^ { \prime } \rangle$ be two diferent $P O P s ,$ then:

$\pi _ { q }$ is a reinstantiated reordering of $\pi _ { p } \ i f \pi _ { p }$ and $\pi _ { q }$ both are valid.

$\pi _ { q }$ is a minimum reinstantiated reordering of $\pi _ { p } ~ i f \pi _ { q }$ is a reinstantiated reordering of $\pi _ { p }$ and there does not exist a $P O P \pi _ { r } \stackrel { . } { = } \langle \mathcal { O } , \hat { \Psi } ^ { \prime \prime } , \prec ^ { \prime \prime } \rangle$ such that $\pi _ { r }$ is a reinstantiated reordering of $\pi _ { p }$ with $| \prec ^ { \prime \prime } | < | \prec ^ { \prime } |$

An operator is defined as a tuple $\textit { o } = \langle v a r s _ { o } , p r e _ { o } , e f f _ { o } \rangle$ , where $v a r s _ { o }$ denotes the set of variables, and both $e f f _ { o }$ and $p r e _ { o }$ are the finite sets of facts containing var ${ \cal s } _ { o } \mathrm { \large { s } }$ variables. If both $p r e _ { o }$ and $e f f _ { o }$ are made up entirely of ground facts, then o is considered as a ground operator. $\langle o _ { p } , q ( \overrightarrow { t } ) , o _ { c } , q ( \overrightarrow { u } ) \rangle$ represents a causal link, where $q ( \vec { u } )$ and $q ( \overrightarrow { t } )$ are literals, with $q ( \overrightarrow { t } ) \in p r o d _ { o }$ and $q ( \vec { u } ) \in c o n s _ { o }$ . The MRR encoding adds two additional categories of propositional variables in addition to the x and κ utilized in the MR encoding:

$\varepsilon ( u , v )$ : Encodes that in the final POP the condition $\Psi ( t ) ~ = ~ \Psi ( u )$ must be satisfied.

$\tau ( o _ { p } , \overrightarrow { t } , o _ { c } , \overrightarrow { u } ) \mathrm { : }$ : Encodes that there exists some q such that the $\langle o _ { p } , q ( \overrightarrow { t } ) , o _ { c } , q ( \overrightarrow { u } ) \rangle$ causal link remains threat-free within the final POP.

To compute the MRR of a partial-order plan $\pi _ { p } = \langle \mathcal { O } , \Psi , \prec \rangle$ , the following encoding together with formulae (3) through (6) is used. Formulae (11) and (12) specify that the equality relation over variables and constants is both symmetric and transitive, whereas the mapping of each variable to a single object is guaranteed by Formula (13). Formulae (14) and (15) formalize the conditions required for the validity of the target POP.

$$
\varepsilon ( u , v )  \varepsilon ( v , u )\tag{11}
$$

$$
\varepsilon ( u , v ) \land \varepsilon ( v , t ) \to \varepsilon ( u , t )\tag{12}
$$

$$
\bigwedge \quad ( \quad \bigvee \quad \varepsilon ( x , c ) \quad \wedge \quad \bigwedge \quad \neg \varepsilon ( x , c ^ { \prime } ) \vee \neg \varepsilon ( x , c ^ { \prime \prime } ) )\tag{13}
$$

$$
\begin{array} { c c } { { x \in v a r s _ { \mathcal { O } } ~ c \in c o n s t s _ { \mathcal { O } } ~ } } & { { ~ c ^ { \prime } , c ^ { \prime \prime } \in c o n s t s _ { \mathcal { O } } : } } \\ { { } } & { { } } \\ { { ~ c ^ { \prime } \ne c ^ { \prime \prime } } } \end{array}
$$

$$
\bigwedge \qquad \bigvee \qquad \tau ( o _ { p } , \vec { t } , o _ { c } , \vec { w } ) \wedge \kappa ( o _ { p } , o _ { c } )\tag{14}
$$

$$
q ( \overrightarrow { \mathcal { U } } ) { \in } c o n s _ { o _ { c } } q ( \overrightarrow { t } ) { \in } p r o d _ { o _ { p } }
$$

$$
\tau ( o _ { p } , \vec { t } , o _ { c } , \vec { u } ) \to \bigwedge _ { \substack { 1 \leq i \leq | \vec { t } | } } \varepsilon ( \vec { t } \vert , \vec { u } \vert \vert ) \quad \wedge \underset { \underset { \vec { t } = \vec { v } , o _ { d } \neq o _ { c } } { \uparrow \sqrt { v } \in d e _ { d } } } { \bigwedge } ( \kappa ( o _ { d } , o _ { p } ) \vee \kappa ( o _ { c } , o _ { d } ) )\tag{15}
$$

If a POP has fewer than k ordering constraints, determining if it admits a minimal reinstantiated reordering is NP-complete. It is impossible to approximate the problem of finding such a reordering within a constant factor. A key limitation of MRR is that substitutions are restricted to operators sharing the same name, but it does not permit replacing operator sets with diferent operator names or sizes.

## 5 Explanation-based Order Generalization

Another well-known approach, Explanation-based Order Generalization (EOG), employs validation structures as proofs of correctness and modifies plans to resolve inconsistencies (Veloso et al. 2002; Kambhampati 1994; Kambhampati and Hendler 1992). EOG has been further refined to support conditional efects (Noor and Siddiqui 2022). The EOG deorders plans by creating a validation structure, creating a causal link for each operator’s preconditions, and using promotions or demotions to mitigate threats to those causal links (Kambhampati and Kedar 1994; Veloso et al. 2002). Given a total-order plan π of a planning task $\Pi = \langle { \mathcal { X } } , { \mathcal { O } } , s _ { i } , s _ { g } \rangle$ , EOG emulates the initial and goal conditions of Π by augmenting π with a couple of additional operators, o<sub>I</sub> and $o _ { G }$ . Specifically, $p r e _ { o _ { I } } = \emptyset , e f f _ { o _ { I } } = s _ { i } , p r e _ { o _ { G } } = s _ { g }$ , and $e f f _ { o _ { G } } = \emptyset$ with $o _ { I } \prec o _ { G }$ and $o _ { I } \prec o \prec o _ { G }$ for all $o \in ( \mathcal { O } \setminus \{ o _ { I } , o _ { G } \} )$ . The validation structure is subsequently constructed, and threats are resolved by introducing promotion and demotion orderings. To minimize unnecessary transitive orderings, it systematically binds causal links to the earliest possible producers.

## 5.1 Plan Deordering with Conditional Efects

Conditional efects (Haslum et al. 2019) enable operators to present complex scenarios more efectively by allowing a single operator to produce diferent outcomes depending on the specific conditions. Noor and Siddiqui (2022) introduce a method for annotating operator orderings in plans with conditional efects, allowing such plans to be transformed into a Partial-Order Plan (POP).

Definition 10. Let $\Pi = \langle { \mathcal X } , { \mathcal O } _ { c } , s _ { i } , s _ { g } \rangle$ be a planning task, where $\mathcal { O } _ { c }$ is a set of a finite number of operators with conditional efects. An efect of an operator $o \in \mathcal { O } _ { c }$ consists of triples ⟨cond, x, d⟩, forming a partial state where cond denotes the efect condition, which may be empty. When o is applicable in the state $s ,$ applying it yields the state $s ^ { \prime } = a p p l y ( o , s )$ , where $x \in \mathcal { X }$ takes the value $d \in D _ { x }$ if and only if cond $\subseteq s$

Definition 11. Let $\pi _ { p o p } = \langle \mathcal { O } , \prec \rangle$ be a POP and two operators $o _ { i } , o _ { j } ~ \in ~ \mathcal { O }$ with conditional efects.

$o _ { i }$ is considered as a candidate producer of $\langle x , d \rangle$ for operator $o _ { j } i f f .$

$$
\begin{array} { r } { 1 . \ o _ { j } \ \nprec \ o _ { i } , } \end{array}
$$

2. ⟨cond, $x , d \rangle \in e f f _ { o _ { i } }$ and there exists other candidate producer $o _ { p } \in \mathcal { O }$ for every $\langle x ^ { \prime } , d ^ { \prime } \rangle \in c o n d$ where $o _ { p } \neq o _ { j }$ , and

3. there exists no operator $o _ { k } \in \mathcal { O }$ with $\langle c o n d _ { 1 } , x , d _ { 1 } \rangle \in ~ e f f _ { o _ { k } }$ such that $o _ { j } \not \prec$ $o _ { k } \not \sim o _ { i } , d _ { 1 } \in ( { \cal D } _ { x } \backslash \{ d \} )$ , and there exists other candidate producer $o _ { p } \in \mathcal { O }$ for each $\langle x ^ { \prime \prime } , d ^ { \prime \prime } \rangle \in c o n d _ { 1 }$

$o _ { i }$ is considered as an earliest candidate producer of $\langle x , d \rangle$ for operator $o _ { j } \ i f f$ $o _ { i }$ is a candidate producer of $\langle x , d \rangle$ for $o _ { j }$ and there does not exists any candidate producer $o _ { k }$ of $\langle x , d \rangle$ for o<sub>j</sub> with $o _ { k } \prec o _ { i }$

Definition 12. Let $\pi _ { p o p } = \langle \mathcal { O } , \prec \rangle$ be a POP and $o \in \mathcal { O }$ be an operator with conditional efects. The operator o produces, consumes, and deletes facts, which are represented by prod , cons , and $\bf d e l _ { o _ { \perp } }$ , respectively. The fact−:

$\langle x , d \rangle \in c o n s _ { o } \ i f \ \langle x , d \rangle \in p r e _ { o }$ or $\langle x , d \rangle \in$ cond such that $\langle c o n d , x ^ { \prime } , d ^ { \prime } \rangle \in e f f _ { o }$ and there exists an operator $o _ { k } \in \mathcal { O }$ with $o \xrightarrow { \langle x ^ { \prime } , d ^ { \prime } \rangle } o _ { k }$

$\langle x , d \rangle \in$ prod $i f \ \langle c o n d , x , d \rangle \in \ e f f _ { o }$ and there exists other candidate producer $o _ { p } \in \mathcal { O }$ for each $\langle x ^ { \prime } , d ^ { \prime } \rangle \in$ cond such that $o _ { p } \prec o _ { \ell }$

$\langle x , d \rangle \in d e l _ { o } ~ i f ~ x ~ \notin$ vars(cons<sub>o</sub>) or con $s _ { o } ( x ) = d ,$ ⟨cond $, x , d ^ { \prime } \rangle \in e f f _ { o }$ , where $d ^ { \prime } \in$ $( \mathcal { D } _ { x } \setminus \{ d \} )$ , and there exists a candidate producer $o _ { p } \in \mathcal { O }$ for each $\langle x ^ { \prime } , d ^ { \prime \prime } \rangle \in c o n d$

$R e ( o _ { a } \prec o _ { b } )$ denotes the ordering reasons between two operators $o _ { a }$ and $o _ { b }$ with conditional efects, which can fall into the categories PC, CT, TP, and $\mathrm { O C . } O C ( \langle x , d \rangle )$

denotes an obstructor-consumer relation for a fact $\langle x , d \rangle$ , which occurs when $o _ { a }$ prevents $o _ { b }$ from receiving the fact $\langle x , d \rangle \in$ cond, such that ⟨cond, $x ^ { \prime } , d ^ { \prime } \rangle \in e f f _ { o _ { h } }$ . If there is an operator $o _ { k }$ that yields $\langle x , d \rangle$ and $o _ { b } \prec o _ { k } \prec o _ { a }$ , then $O C ( \langle x , d \rangle ) \in R { \bar { e } } ( o _ { a } \prec o _ { b } )$ is considered threatened. This identified threat may be mitigated in two ways: through Promotion, by including $p O C ( \langle x , d \rangle )$ in $R e ( o _ { k } ~ \prec ~ o _ { a } )$ , or through Demotion, by including $O C p ( \langle x , d \rangle )$ in $R e ( o _ { b } \prec o _ { k } )$

The Deordering Plan with Conditional Efects (DConE) algorithm constructs causal links to build a validation framework for a POP. Similar to the EOG, DConE identifies causal links using the earliest candidate producers for each fact $\langle x , d \rangle$ required by an operator. While determining a candidate producer of $\langle x , d \rangle$ for an operator $o _ { i } .$ , additional orderings may be needed due to $O C$ (obstructor-consumer) reasons. This occurs when an operator $o _ { d }$ prevents another operator $o _ { p }$ from providing $\langle x , d \rangle$ to $o _ { i }$ through a conditional efect $\langle c o n d , x , d ^ { \prime } \rangle \in e f f _ { o _ { d } } ,$ where $\bar { d ^ { \prime } } \bar { \in } \left( \mathcal { D } _ { x } \backslash \{ d \} \right)$ . To address this, the algorithm introduces an ordering reason $O C ( \langle x , d \rangle )$ in $R e ( o _ { q } \prec o _ { d } )$ for some operator $o _ { q }$ that obstructs the efect $\langle c o n d , x , d ^ { \prime } \rangle$ of $o _ { d }$ from being triggered. The ordering ${ \cal O } _ { q } \prec { \cal O } _ { d }$ prevents cond from holding, ensuring that $o _ { d }$ no longer deletes $\langle x , d \rangle$ . This process repeats iteratively until a valid candidate producer is found. After establishing a causal link between two operators, DConE adds conditional causal links for each fact in the producer’s efect condition and resolves all threats to $P C$ and OC reasons through promotion or demotion of the threatening operators.

## 6 Block Deordering-based Approach

Block deordering (Siddiqui and Haslum 2012) identifies coherent sets of operators, known as blocks, to remove additional orderings from the plan. Block deordering groups coherent operators into blocks to reduce the $\mathrm { P O P } \ ' _ { \mathrm { s } }$ ordering constraints, resulting in a block decomposed partial-order (BDPO) plan (Siddiqui and Haslum 2015). Each block contains a set of operators; operators from two disjoint blocks cannot interleave, allowing the blocks to be executed in any sequence. Additionally, blocks can be nested, which allows a block to have one or more inner blocks. Overlapping between blocks is not permitted. A block fully enclosed within another is called an inner block, while a block not contained within any other is referred to as an outer block. Block deordering has also been leveraged to improve overall plan quality (Siddiqui and Haslum 2015), generate macro-actions (Chrpa and Siddiqui 2015), and improve flexibility via block-substitution (Noor and Siddiqui 2024).

Definition 13. A BDPO plan is denoted as $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ , where $\mathcal { O }$ be a collection of operator $s , \prec$ is a collection of orderings within ${ \mathcal { O } } ,$ and B is a set of blocks. For a block $b \in B _ { : }$ ${ i f \ o _ { p } , o _ { q } \ \in \ b }$ and $o _ { p } \prec o _ { q }$ , then there does not exist an operator $o _ { r } \notin b$ such that $o _ { p } \ \prec \ o _ { r } \ \prec \ o _ { q }$ . For any two blocks $b _ { p } , b _ { q } \in B$ , exactly one of the following holds: $b _ { p } \subset b _ { q } , b _ { q } \subset b _ { p } ,$ , or $b _ { p } \cap b _ { q } = \emptyset$

A block b can be characterized by its preconditions and efects, similar to an operator. If some operator $o \in b$ consumes $\langle x , d \rangle$ and no operator $\boldsymbol { o } ^ { \prime } \in \boldsymbol { b }$ provides $\langle x , d \rangle$ to operator $^ { O , }$ then a fact $\langle x , d \rangle$ is one of the preconditions of b. Conversely, if an operator $o \in b$ generates $\langle x , d \rangle$ and no subsequent operator within b modifies ${ \mathrm { i t } } ,$ then $\langle x , d \rangle$ is included in the efects of b. Unlike an individual operator, a block can have multiple efects on the same or diferent variables. For example, if $o _ { i } , o _ { j } \in b$ with $o _ { i } \ \not \to \ o _ { j } , \ o _ { j } \ \not \to \ o _ { i } , \ \langle x , d \rangle \in \ e f f _ { o _ { i } }$ , and $\langle x , d ^ { \prime } \rangle \in \ e f f _ { o _ { i } }$ , then a block b has both $\langle x , d \rangle$ and $\langle x , d ^ { \prime } \rangle$ as its efects (where d $\neq d ^ { \prime } )$ . The labels $P C$ (producer-consumer), $C T$ (consumer-threat), and $T \dot { P }$ (threat-producer) can also be used to annotate ordering relations between blocks.

Definition 14. Let a BDPO plan $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ with $b \in B$ be a block, then:

$I f o _ { i } \in b$ be an operator with $\langle x , d \rangle \in p r e _ { o _ { i } }$ and there exists no operator $o _ { j } \in b$ with $i \neq j$ that delivers $\langle x , d \rangle$ to $o _ { i }$ via a causal link $o _ { j } \xrightarrow { \langle x , d \rangle } o _ { i }$ , then the fact $\langle x , d \rangle \in p r e _ { b }$

$I f o _ { i } \in b$ be an operator with $\langle x , d \rangle \in \ e f f _ { o , }$ and no subsequent operator $o _ { j } ~ \in ~ b$ (with $o _ { i } ~ \prec ~ o _ { j } )$ produces a diferent value $\langle x , d ^ { \prime } \rangle$ for the same variable x with $d ^ { \prime } \in ( \mathcal { D } _ { x } \mid \{ d \} )$ , then the fact $\langle x , d \rangle \in e f f _ { b }$

Definition 15. For a block $b ,$ the sets of facts that it consumes, produces, and deletes are denoted as cons<sub>b</sub>, prod<sub>b</sub>, and $\mathbf { \Delta } d e l _ { b } .$ , respectively. Where:

• A fact $\langle x , d \rangle \in$ cons<sub>b</sub> if and only if $\langle x , d \rangle \in p r e _ { b }$

• A fact $\langle x , d \rangle \in$ prod<sub>b</sub> if and only $\} \mathinner { \langle { x , d } \rangle } \notin$ cons<sub>b</sub>, $\langle x , d \rangle \in \mathfrak { e f f } _ { b }$ and no other efect $\langle x , d ^ { \prime } \rangle \in e f f _ { b }$ exists with $d ^ { \prime } \in ( D _ { x } \setminus \{ d \} )$

• A fact $\langle x , d \rangle \in d e l _ { b }$ if and only if either x /∈ vars(cons<sub>b</sub>) or $c o n s _ { b } ( x ) = d ,$ and there exists an efect $\langle x , d ^ { \prime } \rangle \in e f f _ { b }$ with $d ^ { \prime } \in ( \mathcal { D } _ { x } \setminus \{ d \} )$

Definition 16. Let a BDPO plan $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ with a pair of blocks $b , b ^ { \prime } \in B$ and $\langle x , d \rangle \in c o n s _ { b ^ { \prime } }$ , then:

$I f \left. x , d \right. \in e f f _ { b } , b \prec b ^ { \prime }$ , and no other block $b ^ { \prime \prime } \in B$ exists such that $\langle x , d \rangle \in d e l _ { b ^ { \prime \prime } }$ and $b ^ { \prime } \not \prec b ^ { \prime \prime } \not \prec b ,$ , then block b is a candidate producer of the fact $\langle x , d \rangle$ for $b ^ { \prime }$

• $I f$ there is no other candidate producer b<sup>′′</sup> of ⟨x, d⟩ for b<sup>′</sup> with $b ^ { \prime \prime } \prec b ,$ then block b is considered as an earliest candidate producer of $\langle x , d \rangle \ f o r \ b ^ { \prime }$

Block deordering builds a valid BDPO plan by taking a total-order plan as input. The total-order plan is first converted into a $\mathrm { P O P } ~ \pi = \langle { \mathcal O } , \prec \rangle$ using EOG. Next, a block $b = \{ o \}$ is created and added to B for every operator $o \in \mathcal { O }$ to construct an initial BDPO plan $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ . Ordering constraint $b \prec b ^ { \prime }$ is introduced for every ordering $\textit { o } \prec \textit { o } ^ { \prime }$ present in $\prec ,$ where $o \in b , o ^ { \prime } \in b ^ { \prime }$ , and $b , b ^ { \prime } \in B$ . The blocks with a single operator are denoted as primitive block and the blocks with multiple operators are denoted as compound block. The term block is used in a general sense to cover both. Block deordering then applies a set of rules to remove additional orderings in $\pi _ { b d p o p }$

Rule 1. A valid BDPO plan $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , \mathcal { B } \rangle$ with an ordering $b _ { p } \prec b _ { q }$ and let b be a block.

i. $H f b _ { p } \in b , b _ { q } \notin b _ { \mathrm { ~ } }$ , and $b _ { p } \ \not \prec \ b ^ { \prime }$ for all $b ^ { \prime } \in ( b \setminus b _ { p } )$ , therefore, $P C ( \langle x , d \rangle )$ can be eliminated from $R e ( b _ { p } \prec b _ { q } ) ~ i f ~ \langle x , d \rangle \in p r e _ { b }$ and there exists $b _ { p } \notin b$ which can create causal links $b _ { p } \xrightarrow { \langle x , d \rangle } b _ { q }$ and $b _ { p } \xrightarrow { \langle x , d \rangle } b$

ii. $H f b _ { p } \in b , b _ { q } \notin b _ {  } ,$ , and $b \cap b _ { q } = \emptyset$ , therefore, $C T ( \langle x , d \rangle )$ can be eliminated from $R e ( b _ { p } \prec b _ { q } ) ^ { - } i f \left. x , d \right. \notin c o n s _ { b }$

iii. $H f b _ { p } \notin b , b _ { q } \in b ,$ and $b _ { p } \cap b = \emptyset$ , therefore, $C T ( \langle x , d \rangle )$ can be eliminated from $R e ( b _ { p } \prec b _ { q } ) ~ i f \left. x , d \right. \notin d e l _ { b }$

iv. $I f b _ { q } \in b$ and $b _ { p } \notin b ,$ therefore, $T P ( \langle x , d \rangle )$ can be eliminated from $R e ( b _ { p } \prec b _ { q } )$ if b contains all blocks b<sup>′</sup> with $b _ { q } \xrightarrow { \langle x , d \rangle } b ^ { \prime }$

Block deordering begins by examining each ordering in the initial BDPO plan from top to bottom, attempting to remove them greedily. If Rule 1 can eliminate all of the ordering reasons for an ordering $b _ { p } \prec b _ { q }$ , then that ordering is eliminated. The ordering $b _ { p } \ \prec \ b _ { q }$ is kept, and the algorithm proceeds to the subsequent ordering if some reasons cannot be eliminated. Whenever an ordering is successfully removed, the modified BDPO plan is returned by the algorithm. The deordering process is then restarted by the algorithm from the top of this most recent plan. Until no more orderings can be eliminated from the most recent $\mathrm { B D P O }$ plan, this iterative process keeps going.

## 6.1 Flexibility Improvement via Block-Substitution

Block-substitution (Noor and Siddiqui 2024) permits changing a block inside a valid BDPO plan without compromising the plan’s validity. The block that is being replaced is referred to as the original block, and the block that is replacing it is referred to as the substituting block. Block substitution permits the replacement block to originate either from within the plan or from an external source. When the replacement block is drawn from within the same plan, the process is referred to as an internal block substitution. Establishing causal links for the replacing block’s preconditions is required for the substitution process, as well as restoring all causal links that the original block had previously supported. Furthermore, any threats introduced by the substitution must be identified and resolved to maintain the validity of the plan.

Definition 17. Let $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ be a valid BDPO plan, $b \in B _ { : }$ , and a subplan $\tilde { b } = \langle \tilde { \mathcal { O } } , \tilde { \mathcal { Q } } \rangle$ with $\tilde { \mathcal { O } } \subset \mathcal { O }$ . A new BDPO plan $\pi _ { b d p o p } ^ { \prime } = \langle \mathcal { O } ^ { \prime } , \prec ^ { \prime } , \mathcal { B } ^ { \prime } \rangle$ is produced $i f b$ is substituted with $\tilde { b }$ where $b \notin B ^ { \prime }$ and $\tilde { b } \in B ^ { \prime } . \ I f \pi _ { b d p o p } ^ { \prime }$ is a valid BDPO plan, then the block-substitution is considered valid.

Definition 18. Let $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , B \rangle$ be a valid BDPO plan and some causal link $b _ { p } \ { \xrightarrow { \langle x , d \rangle } } \ b _ { c }$ is threatened by $b _ { t } \in B$ , where $\langle x , d \rangle \in d e l _ { b _ { i } }$ and $b _ { p } , b _ { c } \in B$ . A threat is considered resolved if a threat-resolution strategy from the following can be used without creating a cycle in π<sub>bdpop</sub>:

i. Promotion: introduce the ordering $b _ { t } \prec b _ { p }$ int $\cdot o \prec$

$i i .$ Demotion: introduce the ordering $b _ { c } \prec b _ { t }$ into ≺.

iii. Internal substitution: replace $b _ { t }$ with $b _ { c } ,$ or $b _ { c }$ with $b _ { t }$ .

The procedure replaces a block b with a substituting block ${ \tilde { b } } .$ The process starts by making sure that every precondition of the $\tilde { b }$ is supported by a causal link, but $\tilde { b } \mathrm { { ^ * _ { s } } }$ preconditions are already satisfied if it is internal. For an external block, <sup>˜</sup>b is first included in the plan, and causal links are established using the earliest candidate producers for each precondition; if any precondition cannot be supported, the substitution fails. Then, using <sup>˜</sup>b as the new producer, every causal link that block b initially supported is restored. If <sup>˜</sup>b does not produce a required fact for any dependent block, the substitution is unsuccessful. Once these causal links are in place, b is removed from the plan. Finally, any threats introduced by the substitution are identified and resolved using threat-resolution strategies, including promotion, demotion, or internal block-substitution. In some situations, replacing a conflicting block with <sup>˜</sup>b resolves threats while maintaining plan validity. The block-substitution process has an overall worst-case complexity of $O ( n ^ { 2 } p ^ { 2 } )$ , where $p$ is the highest possible number of facts in any operator’s efect or precondition and n is the number of operators in the plan.

Flexibility Improvement via Block-Substitution (FIBS) (Noor and Siddiqui 2024) algorithm generates a valid BDPO plan given a valid total-order plan $\pi$ as input. First, using EOG, a partial-order plan $\pi _ { p o p } = ( \mathcal { O } , \prec )$ is created from π. A BDPO plan $\pi _ { b d p o p } = \langle \mathcal { O } , \prec , \mathcal { B } \rangle$ is then created from this partial-order plan by constructing a block $b = \{ o \}$ for each $o \in \mathcal { O }$ . In the next phase, ordering constraints are removed from π<sub>bdpop</sub> by substituting blocks, where initially only primitive blocks are considered, since no compound blocks have yet been formed. Subsequently, block deordering introduces compound blocks to eliminate additional orderings, further increasing plan flexibility. Finally, the procedure of substituting blocks is employed once more to replace both compound and primitive blocks, minimizing orderings in the plan. The procedure of substituting blocks examines each basic ordering $b _ { i } \prec b _ { j }$ in the BDPO plan and attempts to remove it by substituting $b _ { j } \ { \mathrm { o r } } ,$ if necessary, $b _ { i } .$ while ensuring causal links and plan validity are maintained. After a successful removal, the procedure is restarted from the current BDPO plan’s beginning, iterating until no further orderings can be eliminated. This two-stage application, before and after block deordering, allows the algorithm to separately evaluate the impact of primitive and compound block substitutions on overall plan flexibility.

FIBS significantly enhances plan flexibility while reducing computational and plan costs through eficient block-substitution within BDPO plans, outperforming existing approaches like EOG and MaxSAT reorderings. However, FIBS restricts candidate subplans for substitution to blocks within a BDPO plan, leaving other potentially useful subplans unexplored, and it has not yet been evaluated across diverse planning paradigms or with planners beyond LAMA.

## 6.2 Concurrency Improvement via Block- Substitution

Partial-order plans enable parallel execution by indicating which operators cannot run concurrently. A parallel plan is a $\mathrm { P O P }$ that permits concurrent operator execution. Noor and Siddiqui (2025) transform sequential plans into parallel ones and define nonconcurrency constraints in FDR. They improved the flexibility of a plan’s execution by expanding block deordering and block substitution approaches, thereby exploiting potential parallelism to minimize overall execution time.

Definition 19. A parallel plan is denoted as $\pi _ { \mathcal { P } } ~ = ~ \langle \mathcal { O } , \prec , \# \rangle$ , which basically extends a $P O P \ \langle O , \prec \rangle$ with an irreflexive and symmetric relation $\#$ , which represents non-concurrency constraints over O. Two operators, $o _ { a }$ and $o _ { b }$ , have a nonconcurrency constraint, expressed as $O _ { a } \# O _ { b }$ , if cons $\Pr _ { a } ( x ) \neq$ con $s _ { o _ { b } } ( x )$ ， $p r o d _ { o _ { a } } ( x ) \neq$ $p r o d _ { o _ { b } } ( x )$ , or $c o n s _ { o _ { a } } ( x ) \ne p r o d _ { o _ { b } } ( x )$ for any variable $x \in \mathcal { X }$ , which means that they cannot be executed simultaneously.

Definition 20. Let $\pi _ { \mathcal { P } } = \langle O , \prec , \# \rangle$ be a parallel plan. The ratio of the operator pairs that can be executed simultaneously to the maximum possible operator pairs is the concurrent flexibility value of the parallel plan $\pi p$ , as specified in (16), expressed as $c f l e x ( \pi p )$

$$
\mathop { c f l e x } ( \pi _ { \mathcal { P } } ) = 1 - \frac { \displaystyle \sum _ { 1 \leq b < a \leq | \mathcal { O } | } \left\{ 1 \quad i f o _ { a } \prec o _ { b } , o _ { b } \prec o _ { a } , o r o _ { a } \# o _ { b } \right. } { }  { \sum _ { n = 1 } ^ { | \mathcal { O } | - 1 } n } \mathop { o r o _ { a } } \# \theta \mathop { r o v o l o c } _ { \left. \iota \right. }\tag{16}
$$

Two unordered operators of a partial-order plan can be executed in any sequence without invalidating the plan, but they can’t always run at the same time because they might use the same resources. To safely run operators in parallel, we must make sure they don’t interfere with each other. Knoblock (1994) grouped parallel plans based on how operators depend on each other, describing the simplest kind as those where operators are independent with respect to the goal, meaning the final result is the same whether they run one after another or together. Noor and Siddiqui (2025) focus on using such independent operators to speed up plan execution, not because parallelism is required, but because it can make plans more eficient. Methods like Knoblock (1994) and Boutilier and Brafman (2001) handle this by explicitly defining which resources each operator used, but none established how to identify parallelism without such information. Noor and Siddiqui (2025) extended the idea of the BDPO plan by adding the non-concurrency relation to allow blocks to be executed in parallel.

Definition 21. A parallel block decomposed partial-order $( P B D P O )$ plan is denoted as $\pi _ { p b d p } = \langle \mathcal { O } , \prec , B , \# \rangle$ , which basically extends a BDPO plan $\langle { \mathcal { O } } , \prec , { \mathcal { B } } \rangle$ with an symmetric, irreflexive relation $\#$ over $B _ { i }$ depicting non-concurrency constraints between blocks. Blocks $b _ { p }$ and $b _ { q }$ are considered non-concurrent, expressed as $b _ { p } \# b _ { q } ,$ if two primitive operators $o _ { p } \in b _ { p }$ and $o _ { q } \in b _ { q }$ are non-concurrent $( o _ { p } \# o _ { q } )$ . Variables causing this non-concurrency are denoted by the set $v a r s ( b _ { p } \# b _ { q } )$

Definition 22. Let $\pi _ { p b d p } = \langle \mathcal { O } , \prec , B , \# \rangle$ be a valid parallel BDPO plan with $b _ { p } \# b _ { q } .$ where blocks $b _ { p } , b _ { q } \in B$ are two disjoint blocks. The $b _ { p } \# b _ { q }$ is considered necessary non-concurrency constraint within $\pi _ { p b d p } ~ i f -$

i. $b _ { p } \nless b _ { q } \nless b _ { p }$ , and

ii. $i f b _ { p }$ is in the block $b _ { x } \in B$ , then $b _ { q }$ is also in the block $b _ { x }$

The Concurrent Flexibility Improvement via Block-Substitution (CIBS) algorithm generates a valid PBDPO plan from a valid total-order plan π for a planning task Π. The CIBS algorithm extends blocks before substitution. It selects two blocks $b _ { p }$ and $b _ { q }$ with $b _ { p } \# b _ { q }$ in a PBDPO plan $\pi _ { p b d p }$ with respect to a planning task Π. Then, the procedure includes the necessary blocks to the block $b _ { p }$ to achieve a new block $b _ { p } ^ { \prime }$ so that another block $\hat { b } _ { p }$ can be used in place of $b _ { p } ^ { \prime }$ where vars $( \hat { b } _ { p } \# b _ { q } ) = \emptyset$

The CIBS algorithm enhances the concurrent flexibility through block-substitution in three steps. In the first step, EOG is applied to convert π into a parallel plan $\pi _ { e o g } = \langle \mathcal { O } , \prec , \# \rangle$ first, and then non-concurrency constraints between every pair of operators are identified. In the next step, cohesive operators are encapsulated into blocks using block deordering, which removes ordering constraints from $\pi _ { e o g }$ and yields a PBDPO plan $\pi _ { p b d p } = \langle \mathcal { O } , \prec , B , \# \rangle$ . In the last stage, blocks are replaced to improve the concurrent flexibility of $\pi _ { p b d p }$

The CIBS efectively enhances concurrent flexibility across a wide range of planning domains, outperforming both EOG and block deordering in most cases. Its consistent improvements in cflex , particularly in domains rich with resource-based interactions, show the benefit of incorporating substitution alongside block deordering. The observed correlations suggest that CIBS scales reasonably well with plan complexity, though execution time grows notably with larger plan sizes and more variables. However, the performance drop in extensive plans and the limited impact of block deordering on cflex highlight scalability and eficiency challenges. Additionally, the algorithm’s reliance on available resources for substitution restricts its applicabil ity in domains lacking such elements, indicating room for improvement in generalizing CIBS to more diverse planning tasks.

## 7 Analysis

The optimization capabilities of the six algorithms are summarized in Table 1 across six categories: ordering, action handling, parameter handling, plan structure, concurrency, and complexity and optimality.

All six algorithms remove unnecessary orderings from a plan, but they difer in scope. EOG operates at the operator level and produces a standard POP without block structure. Block Deordering (BD) groups coherent operators into blocks and eliminates more orderings than EOG by treating each block as a unit. MR and MRR allow arbitrary reordering and find minimum reorderings via MaxSAT encodings, though neither produces block-structured plans. Flexibility Improvement via Block-Substitution (FIBS) and Concurrency Improvement via Block-Substitution (CIBS)

Table 1: Optimization capability comparison among algorithms
<table><tr><td>Criteria</td><td>EOG</td><td>BD</td><td></td><td>MR MRR</td><td>FIBS</td><td>CIBS</td></tr><tr><td>Ordering</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Remove Orderings</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Arbitrary Reordering</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Block-level Deordering</td><td>No</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Minimizes Orderings</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Action Handling</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Remove Redundant Actions</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Introduce New Actions</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Replace Action Sets (Subplans)</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Replace Actions of Different Name</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Parameter Handling</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rebind Parameters</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Same-name Operator Rebinding Only</td><td></td><td></td><td>一</td><td>Yes</td><td></td><td></td></tr><tr><td>Plan structure</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Produces Partial-order Plan</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Produces BDPO Plan</td><td>No</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Produces Parallel Plan</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Produces Parallel BDPO Plan</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>External Planner for Subplans</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Concurrency</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Non-concurrency Constraints (#)</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Minimizes #</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Improves Parallel Execution</td><td>No</td><td>Partial</td><td>No</td><td>No</td><td>Partial</td><td>Yes</td></tr><tr><td>Uses FDR for # Conditions</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Block Extension via DTG Analysis</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Complexity and Optimality</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Polynomial-time Algorithm</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>Partial</td><td> Partial</td></tr><tr><td>MaxSAT-based Encoding</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Anytime / Iterative Improvement</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Composable on top of MR/MRR</td><td>No</td><td>No</td><td></td><td></td><td>Yes</td><td>No</td></tr></table>

use block-level deordering but, unlike MR and MRR, do not seek a globally minimum reordering; instead, they improve the plan iteratively through substitution.

On action handling, EOG and BD leave the operator set unchanged. MR removes redundant actions as part of its encoding. MRR does the same and additionally reorders operator parameters, though only within operators of the same name. FIBS and CIBS replace entire subplans with new ones drawn from the planning task, including operators with diferent names and diferent numbers of actions. Block-substitution is therefore the only mechanism in the table that can introduce actions absent from the original plan.

Parameter rebinding is exclusive to MRR, and even there it is constrained to same-name operators. All other algorithms treat operator parameters as fixed.

All six algorithms produce a valid POP. BD, FIBS, and CIBS additionally produce BDPO plans by organizing operators into blocks. Only CIBS produces a parallel plan by incorporating a non-concurrency constraint relation (#), which is a parallel block decomposed (PBD) plan. FIBS and CIBS both use an external planner to generate candidate subplans for substitution; the remaining algorithms are self-contained.

The concurrency category distinguishes CIBS from everything else in the table. EOG, BD, MR, MRR, and FIBS do not model non-concurrency constraints (#). CIBS formalizes necessary and suficient conditions for non-concurrency using FDR variables, uses a Domain Transition Graph (DTG) to determine when a block must be extended before substitution, and directly minimizes # constraints to maximize parallel execution flexibility.

EOG and BD run in polynomial time. MR and MRR are NP-complete due to their MaxSAT formulations. FIBS and CIBS are partially polynomial: the blocksubstitution procedure runs in $O ( n ^ { 2 } p ^ { 2 } )$ time, but both use an external planner, which is not guaranteed to be polynomial. Both are anytime algorithms that produce a valid result at any point during execution. Finally, FIBS can be applied on top of MR or MRR-generated POPs for further flexibility gains; CIBS takes a sequential plan directly as input and is not designed to compose with MaxSAT reorderings.

## 8 Evaluation

We evaluate seven plan execution flexibility-improving methods on 3,345 plans from 46 IPC domains. These methods are EOG, BD, MR, MRR, FIBS, FIBS applied on MR output (FIBS+MR), and FIBS applied on MRR output (FIBS+MRR). EOG and BD achieve full coverage by construction, while MR, MRR, FIBS+MR, and FIBS+MRR solve 81.4%, 47.8%, 48.2%, and 21.0% of plans, respectively. When an algorithm fails to find a solution, we assume its flex value with the EOG flex for that plan, which is the lowest flex any plan achieves before any reordering or substitution.

We measure flexibility using the flex metric defined in (2). For a multi-dimensional comparison, we compute five normalized scores per algorithm: Flex (mean flex value), Coverage (proportion of plans solved), Speed (inverse mean execution time), Consistency (inverse coeficient of variation of flex), and Unique Best (proportion of plans on which the algorithm achieves the strictly highest flex). All five are min-max normalized, so a value of 1.0 denotes the best algorithm on that dimension.

## 8.1 Flex-Time Trade-of

Figure 1 plots mean flex against mean time per plan for all seven methods, with the Pareto frontier connecting those not dominated on both dimensions at once. EOG is the fastest at 1.06 seconds with a mean flex of 0.247. MR and FIBS+MR reach mean flex values of 0.247 and 0.249 at 29.3 and 30.4 seconds, respectively, both on the frontier. BD and FIBS improve mean flex to 0.339 and 0.341 at 44.0 and 47.4 seconds, also on the frontier. MRR and FIBS+MRR take 58.8 and 59.9 seconds but reach only 0.249 mean flex each, placing them of the frontier. The Pareto front therefore has two natural operating points: fast with modest gains (EOG, MR) and moderate cost with substantially better flex (BD, FIBS). MRR and FIBS+MRR roughly double MR’s computation time without improving its position on the frontier.

![](images/6228175099c5728955c3c276b68831824ad1918b3d23b3fc743444fb66674e4b.jpg)  
Fig. 1: Mean flex score vs. mean planning time with Pareto Frontier.

## 8.2 Flex Eficiency

Figure 2 shows flex eficiency per algorithm and domain, measured as flex gain over EOG per additional second spent (∆flex/∆time). BD has the highest median eficiency across domains: block deordering removes a large number of ordering constraints at relatively low per-plan overhead. FIBS reaches comparable eficiency because its substitution phases build directly on BD. MR and FIBS+MR cluster at intermediate values. MRR and FIBS+MRR fall to the lowest eficiency, as MRR’s encoding cost is large relative to the flex improvement it achieves.

Bubble size in Figure 2 encodes absolute mean flex gain per domain; the largest bubbles belong to BD and FIBS, confirming that these two algorithms also produce the greatest absolute improvements.

## 8.3 Statistical Efect Size

Figure 3 presents a pairwise efect-size heatmap using the rank-biserial correlation r from a two-sided Mann–Whitney U test on the flex distributions. A positive r in row A and column B means algorithm A produces higher flex on more plans than algorithm B; significance levels are marked as $^ { * } ( p < 0 . 0 5 )$ \*\* $\left( p < 0 . 0 1 \right)$ ), and \*\*\* $( p < 0 . 0 0 1 )$ .

BD and FIBS show strong positive r values against all other algorithms, with statistically significant diferences in the majority of pairwise comparisons. The efect size between BD and FIBS itself is small and non-significant, so the two algorithms perform equivalently at the population level despite FIBS having a marginally higher mean. EOG, MR, MRR, FIBS+MR, and FIBS+MRR cluster together, with nearzero or weakly negative r values against the block-based methods. Within this cluster, pairwise efect sizes are consistently small and non-significant, indicating that the MaxSAT-based methods do not meaningfully separate from one another in terms of flex on the full plan distribution.

![](images/b055c2fc6d5e416ab3c4503b514819036d491d77f5845dcdbb3ceb1f6491244c.jpg)  
Fig. 2: Flex eficiency

## 8.4 Multi-Criteria Comparison

Figure 4 places all seven methods on a radar chart with one axis per normalized dimension; the outer rim corresponds to the best score on each axis. BD and FIBS occupy the largest area, with normalized Flex scores of 0.985 and 1.0, perfect Coverage, and perfect Consistency. FIBS has a Unique Best score of 0.119, meaning it produces the strictly highest flex on roughly 12% of individual plans. BD’s normalized Unique Best score is 1.0, as it achieves the best flex on the largest number of plans overall. EOG scores perfectly on Speed and Coverage, and its normalized Unique Best of 0.625 is notably high: since EOG is used as the fallback for plans where MaxSAT methods time out, it retains the best score wherever all alternatives fail to terminate.

MR achieves moderate Speed (0.520) and Coverage (0.765) but a near-zero normalized Flex score relative to BD and FIBS. MRR scores near zero on Speed, Coverage, and Unique Best. FIBS+MR and FIBS+MRR also score near zero on Unique Best, since when the MaxSAT component does find a solution, FIBS adds only a small additional flex increment, and the combined pipeline rarely produces the globally best result for any individual plan.

r>0: row has higher flex | \*, \*\*, \*\*\* = p<0.05/0.01/0.001  
![](images/0cc788f528634cd86f90efdaa38ba9eb14a6dfc29c986c45468b68765987ba30.jpg)  
Fig. 3: Efect Size - Rank-biserial r

## 9 Conclusion

The methods reviewed in this study approach plan flexibility from three distinct angles: causal-link reasoning (POCL, EOG), operator coherence and block structure (BD, FIBS, CIBS), and constraint optimization (MR, MRR). Each angle captures something real about what makes a plan flexible, but the empirical results show that these angles are not equally productive. Restructuring causal dependencies at the block level, without any guarantee of minimality, yields substantially more flexibility than minimizing ordering constraints within a fixed causal structure. This is the central finding of the study, and is somewhat counterintuitive given how much theoretical work has gone into minimum reordering.

The theoretical promise of finding minimum reorderings with the MaxSAT-based methods is legitimate. Yet minimum reordering within a fixed action set is a ceiling set by the original plan’s causal structure. Block deordering raises that ceiling by reorganizing how causal dependencies are grouped, which exposes ordering constraints that would otherwise appear necessary. Block-substitution raises it further by replacing subplans with alternatives that carry diferent causal entailments entirely. BD’s greedy, polynomial-time heuristic therefore outperforms MR’s NP-complete search on a broad collection of IPC benchmarks.

![](images/088605e3d558f6e1404c77e70e5dccfca1199fd56fd2d231027396a8f0f998b3.jpg)  
Fig. 4: Radar Chart

Minimum reordering has been studied extensively as an optimization target, but the quality of the causal structure itself has received less attention. The success of block-substitution suggests that reordering and action-set modification interact in non-trivial ways. MRR takes a step in this direction by allowing parameter rebinding, but only within same-name operators, and the evaluation shows this restriction is too tight to close the gap with BD. A method that jointly optimizes ordering and action selection without the coverage failures of MaxSAT encodings is still missing.

CIBS introduces a separate question that none of the other methods address. Two unordered operators may still be concurrent if they do not share a state variable. CIBS introduces cflex to account for this, instead of relying on flex, and the gap between the two metrics is non-trivial in resource-rich domains. However, whether a domain-independent characterization of practical concurrency is achievable, and how to extend concurrent flexibility improvement to domains without explicit resource contention, remain open.

## Statements and Declarations

## Competing Interests

The authors have no relevant financial or non-financial interests to disclose.

## References

Anderson JS, Farley AM (1988) Plan abstraction based on operator generalization. In: Proceedings of the Seventh AAAI National Conference on Artificial Intelligence. AAAI Press, AAAI’88, p 100–104

Bercher P, Höller D, Behnke G, et al (2016) More than a name? on implications of preconditions and efects of compound HTN planning tasks. In: Proceedings of the Twenty-Second European Conference on Artificial Intelligence. IOS Press, NLD, ECAI’16, p 225–233, https://doi.org/10.3233/978-1-61499-672-9-225

Bercher P, Behnke G, Höller D, et al (2017) An admissible HTN planning heuristic. In: Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17, pp 480–488, https://doi.org/10.24963/ijcai.2017/68

Bit-Monnot A, Smith DE, Do M (2016) Delete-free Reachability Analysis for Temporal and Hierarchical Planning (full version). In: ICAPS Workshop on Heuristics and Search for Domain-independent Planning (HSDIP), London, United Kingdom, URL https://ut3-toulouseinp.hal.science/hal-01319768

Bit-Monnot A, Ghallab M, Ingrand F, et al (2020) FAPE: a constraint-based planner for generative and hierarchical temporal planning. arXiv:2010.13121

Blum AL, Furst ML (1997) Fast planning through planning graph analysis. Artificial Intelligence 90(1):281–300. https://doi.org/10.1016/S0004-3702(96)00047-1

Boutilier C, Brafman RI (2001) Partial-order planning with concurrent interacting actions. Journal of Artificial Intelligence Research 14:105–136. https://doi.org/10. 1613/jair.740

Bäckström C (1998) Computational aspects of reordering plans. Journal of Artificial Intelligence Research 9:99–137

Chrpa L, Siddiqui FH (2015) Exploiting block deordering for improving planners eficiency. In: Proceedings of the 24th International Conference on Artificial Intelligence. AAAI Press, IJCAI’15, p 1537–1543

Coles A, Coles A, Fox M, et al (2021) Forward-chaining partial-order planning. Proceedings of the International Conference on Automated Planning and Scheduling 20(1):42–49. https://doi.org/10.1609/icaps.v20i1.13403, URL https://ojs.aaai.org index.php/ICAPS/article/view/13403

Fikes RE, Nilsson NJ (1971) STRIPS: A new approach to the application of theorem proving to problem solving. Artificial Intelligence 2(3):189–208. https://doi.org/10. 1016/0004-3702(71)90010-5

Fox M, Long D (2001) Hybrid STAN: Identifying and managing combinatorial optimisation sub-problems in planning. In: Proceedings of the 17th International Joint Conference on Artificial Intelligence - Volume 1. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, IJCAI’01, p 445–450

Graham JR, Decker KS, Mersic M (2003) DECAF - a flexible multi agent system architecture. Autonomous Agents and Multi-Agent Systems 7(1–2):7–27. https:// doi.org/10.1023/A:1024120703127

Haslum P, Lipovetzky N, Magazzeni D, et al (2019) An Introduction to the Plan ning Domain Definition Language. Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan & Claypool

Helmert M (2011) The fast downward planning system. Journal of Artificial Intelligence Research 26. https://doi.org/10.1613/jair.1705

Hickmott S, Rintanen J, Thiébaux S, et al (2007) Planning via petri net unfolding. In: Proceedings of the 20th International Joint Conference on Artificial Intelligence. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, IJCAI’07, p 1904–1911

Kambhampati S (1994) Multi-contributor causal structures for planning: a formalization and evaluation. Artificial Intelligence 69(1-2):235–278. https://doi.org/10. 1016/0004-3702(94)90083-3

Kambhampati S, Hendler JA (1992) A validation-structure-based theory of plan modification and reuse. Artificial Intelligence 55(2):193–258. https://doi.org/10.1016/ 0004-3702(92)90056-4

Kambhampati S, Kedar S (1994) A unified framework for explanation-based general ization of partially ordered and partially instantiated plans. Artificial Intelligence 67(1):29–70. https://doi.org/10.1016/0004-3702(94)90011-6

Kautz H, Selman B (1998) BLACKBOX: A new approach to the application of theorem proving to problem solving. In: AIPS98 Workshop on Planning as Combinatorial Search

Knoblock CA (1994) Generating parallel execution plans with a partial-order planner. In: International Conference on Artificial Intelligence Planning Systems, pp 98–103

Koehler J (1999) Handling of conditional efects and negative goals in IPP

Muise C, Beck J, McIlraith S (2016) Optimal partial-order plan relaxation via MaxSAT. Journal of Artificial Intelligence Research 57:113–149. https://doi.org/ 10.1613/jair.5128

Nguyen X, Kambhampati S (2001) Reviving partial order planning. pp 459–466

Noor SB, Siddiqui FH (2022) Plan deordering with conditional efects. In: Arai K (ed) Intelligent Systems and Applications. Springer International Publishing, pp 852–870

Noor SB, Siddiqui FH (2024) Improving plan execution flexibility using blocksubstitution. arXiv preprint arXiv:240603091

Noor SB, Siddiqui FH (2025) Improving execution concurrency in partial-order plans via block-substitution. Autonomous Agents and Multi-Agent Systems 39(1):1–43

Penberthy JS, Weld DS (1992) UCPOP: A sound, complete, partial order planner for adl. In: Proceedings of the Third International Conference on Principles of Knowledge Representation and Reasoning. Morgan Kaufmann Publishers Inc., KR’92, p 103–114

Regnier P, Fade B (1991) Complete determination of parallel actions and temporal optimization in linear plans of action. In: European Workshop on Planning, volume 522 of Lecture. Springer-Verlag, pp 100–111

Siddiqui FH, Haslum P (2012) Block-structured plan deordering. In: Thielscher M, Zhang D (eds) AI 2012: Advances in Artificial Intelligence. Springer Berlin Heidelberg, Berlin, Heidelberg, pp 803–814

Siddiqui FH, Haslum P (2015) Continuing plan quality optimisation. Journal of Artificial Intelligence Research 54:369–435

Simmons R, Younes H (2011) VHPOP: Versatile heuristic partial order planner. Journal of Artificial Intelligence Research 20. https://doi.org/10.1613/jair.1136

Veloso M, Perez M, Carbonell J (2002) Nonlinear planning with parallel resource allocation

Waters M, Nebel B, Padgham L, et al (2018) Plan relaxation via action debinding and deordering. Proceedings of the International Conference on Automated Planning and Scheduling 28(1):278–287. https://doi.org/10.1609/icaps.v28i1.13901, URL https://ojs.aaai.org/index.php/ICAPS/article/view/13901

Waters M, Padgham L, Sardina S (2020) Optimising partial-order plans via action reinstantiation. In: Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI’20

Weld DS (1994) An introduction to least commitment planning. AI Magazine 15(4):27. https://doi.org/10.1609/aimag.v15i4.1109

Winner E, Veloso MM (2002) Analyzing plans with conditional efects. In: AIPS, pp 23–33