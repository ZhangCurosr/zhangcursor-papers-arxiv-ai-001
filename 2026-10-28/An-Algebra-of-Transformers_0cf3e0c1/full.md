# An Algebra of Transformers

— Yong Wang, Zhiyan Wang, Hexiao Kang —

## Contents

1 Introduction 1   
2 Preliminaries 3   
2.1 Multiset 3   
2.2 Equational Logic . 4   
2.3 Pomset Labelled Transition System 5   
3 Basic Algebra for Transformers 8   
3.1 Axiomatic System 8   
3.2 Operational Semantics 11   
3.3 Soundness and Completeness . 11   
4 Algebra of Parallelism for Transformers 13   
4.1 Operational Semantics . 13   
4.2 Axiomatic System 13   
4.3 Soundness and Completeness . 16   
5 Recursion 18   
5.1 Operational Semantics . 19   
5.2 Axiomatic System 19   
5.3 Soundness and Completeness . 20   
6 Abstraction 21   
6.1 Silent Step . 21   
6.1.1 Operational Semantics 21   
6.1.2 Axiomatic System . 23   
6.1.3 Soundness and Completeness 23   
6.2 Abstraction 24   
6.2.1 Operational Semantics 25   
6.2.2 Axiomatic System . 25   
6.2.3 Soundness and Completeness 26   
Application 29   
7.1 Verification of a Transformer for Most-Recent Induction 29   
7.1.1 A Transformer Construction of the Most-Recent Induction 29   
7.1.2 Algebraic Representation . 30   
7.1.3 Verification of the Most-Recent Induction Transformer . 32   
7.2 Verification of a Transformer for Dyck-1 33   
7.2.1 A Transformer Construction for Dyck-1 33   
7.2.2 Algebraic Representation . 35   
7.2.3 Verification of the Dyck-1 Transformer 37   
7.3 Verification of a Transformer for Dyck-1-2 38   
7.3.1 A Transformer Construction of Dyck-1-2 38   
7.3.2 Algebraic Representation . 40   
7.3.3 Verification of the Dyck-1-2 Transformer 44

## 1 Introduction

Because of the parallelizability of transformers [1], transformers are becoming the mainstream techniques in large language models (LLMs). On the theoretic analyses on transforms, they mainly focus on the computability analysis, such the Turing-completeness of unbounded precision transformers [2], the advantage of parallelizability of unbounded precision transformers [3] and the limitations of bounded precision transformers [4].

This paper give transformers an algebra for reasoning, which relies heavily on the functional decomposition results of transformers [5], we follow its notations and definitions of transformers.

We write [n] for the set $\{ 1 , \cdots , n \} , \ X ^ { * }$ for the set of sequences over set $X , X ^ { + }$ for the set of non-empty sequences over $X$ , string $\Sigma ^ { * }$ as $w = w _ { 1 } { \cdots } w _ { n }$ for a finite alphabet Σ, 0 for zero vector or matrix, I for identity matrix, 1 for a vector or matrix whose entries are all $\mathrm { 1 3 . }$ For any function $f : \mathbb { R } \to \mathbb { R }$ and any vector $\mathbf { x } = [ x _ { i } ] _ { i = 1 } ^ { d } \in \mathbb { R } ^ { d } ,$ f is extended coordinate-wise:

$$
f ( \mathbf { x } ) = f \left( { \left[ \begin{array} { l } { x _ { 1 } } \\ { x _ { 2 } } \\ { \vdots } \\ { x _ { d } } \end{array} \right] } \right) = { \left[ \begin{array} { l } { f ( x _ { 1 } ) } \\ { f ( x _ { 2 } ) } \\ { \vdots } \\ { f ( x _ { d } ) } \end{array} \right] }
$$

We write $f : X ^ { + } \xrightarrow { l p } Y ^ { + }$ for a function from $X ^ { + }$ to $Y ^ { + }$ , which is length-preserving. And we write $f . { \mathsf { a } }$ for a parameter a of $f .$

Definition 1.1 (Transformers). A transformer is a function tf $: \Sigma ^ { + } \stackrel { l p } { \longrightarrow } \mathcal { y } ^ { + } , { \mathrm { t f } } ( x _ { 1 } { \cdots } x _ { n } ) = y _ { 1 } { \cdots } y _ { n }$ where $\begin{array} { r } { \mathbf { z } _ { i } ^ { ( 0 ) } = \mathsf { t f } . \mathsf { w e } ( x _ { i } ) + \mathsf { t f } . \mathsf { p e } _ { n } ( i ) , ( \mathbf { z } _ { 1 } ^ { ( 1 ) } , \cdots , \mathbf { z } _ { n } ^ { ( 1 ) } ) = \mathsf { t f } . \mathsf { t l } ^ { ( 1 ) } ( \mathbf { z } _ { 1 } ^ { ( 0 ) } , \cdots , \mathbf { z } _ { n } ^ { ( 0 ) } ) , \cdots , ( \mathbf { z } _ { 1 } ^ { ( L ) } , \cdots , \mathbf { z } _ { n } ^ { ( L ) } ) = } \end{array}$ $\mathfrak { t f . t } | ^ { ( L ) } ( \mathbf { z } _ { 1 } ^ { ( L - 1 ) } , \cdots , \mathbf { z } _ { n } ^ { ( L - 1 ) } ) , \ \mathbf { y } _ { i } = \mathfrak { t f . o u t } ( \mathbf { z } _ { i } ^ { ( L ) } )$ . Where tf.we and tf. $\boldsymbol { \cdot } \mathsf { p e } _ { n }$ are word embedding function and position embedding function, tf. $. \mathsf { t l } ^ { ( \ell ) }$ are transformer layers, tf.out is an unembedding function with some output space Y. For the details of these functions, please refer to $[ 5 ] .$

The decomposition of transformers coming from [5] is shown in table 1. The idea of decomposition of a transformer is dividing a transformer into a set of atomic constructions, which are operations of vectors or matrices. For the details of these constructions, please refer to [5].

Table 1: Decomposition of Transformers
<table><tr><td>Names</td><td>Functions</td><td>Types</td></tr><tr><td>α1</td><td>word embedding</td><td> $\overline { { \Sigma \to \mathbb { R } ^ { d } } }$   $\mathbb { N } \to \mathbb { R } ^ { d }$ </td></tr><tr><td> $\alpha _ { 2 }$ </td><td>position embedding</td><td></td></tr><tr><td>α3</td><td>identity</td><td> $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ </td></tr><tr><td> $\alpha _ { 4 }$ </td><td>min/max</td><td> $\mathbb { R } ^ { 2 } \to \mathbb { R }$ </td></tr><tr><td> $\alpha _ { 5 }$ </td><td>add/subtract</td><td> $\mathbb { R } ^ { 2 } \to \mathbb { R }$ </td></tr><tr><td> $\alpha _ { 6 }$ </td><td>multiply by c</td><td> $\mathbb { R } \to \mathbb { R }$ </td></tr><tr><td> $\alpha _ { 7 }$ </td><td>multiply xy</td><td> $\mathbb { R } ^ { 2 } \to \mathbb { R }$ </td></tr><tr><td> $\alpha _ { 8 }$ </td><td>comparators</td><td> $\mathbb { R }  \{ 0 , 1 \}$ </td></tr><tr><td> $\alpha _ { 9 }$ </td><td>boolean functions</td><td> $\{ 0 , 1 \} ^ { m }  \{ 0 , 1 \}$ </td></tr></table>

to be continued. . .

continuing: Decomposition of Transformers
<table><tr><td>Names</td><td>Functions</td><td> $\mathrm { T y p e s }$ </td></tr><tr><td> $\alpha _ { 1 0 }$ </td><td>continuous piecewise linear functions</td><td> $\overline { { \mathbb { R } ^ { d } \to \mathbb { R } } }$ </td></tr><tr><td> $\alpha _ { 1 1 }$ </td><td>cancel residual</td><td> $\mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ </td></tr><tr><td> $\alpha _ { \mathrm { 1 2 } }$ </td><td>identity</td><td> $( \mathbb { R } ^ { d } ) ^ { + } \to ( \mathbb { R } ^ { d } ) ^ { + }$ </td></tr><tr><td> $\alpha _ { 1 3 }$ </td><td>uniform average</td><td> $( \mathbb { R } ^ { d } ) ^ { + }  ( \mathbb { R } ^ { d } ) ^ { + }$ </td></tr><tr><td> $\alpha _ { 1 4 }$ </td><td>first</td><td> $( \mathbb { R } ^ { d } ) ^ { + } \to ( \mathbb { R } ^ { d } ) ^ { + }$ </td></tr><tr><td> $\alpha _ { 1 5 }$ </td><td>index lookup</td><td> $( \mathbb { R } ^ { d } ) ^ { + } \to \mathbb { R } ^ { + }$ </td></tr><tr><td> $\alpha _ { 1 6 }$ </td><td>predecessor</td><td> $( \mathbb { R } ^ { d } ) ^ { + } \to ( \mathbb { R } ^ { d } ) ^ { + }$ </td></tr><tr><td> $\alpha _ { 1 7 }$ </td><td>tie-breaking</td><td></td></tr><tr><td> $\alpha _ { 1 8 }$ </td><td>multi-head</td><td></td></tr><tr><td> $\alpha _ { 1 9 }$ </td><td>unembedding</td><td></td></tr><tr><td> $\alpha _ { 2 0 }$ </td><td>layer normalization</td><td> $\mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ </td></tr><tr><td> $\phi _ { 1 }$ </td><td>conditional</td><td> $\{ 0 , 1 \} \times \mathbb { R } ^ { 2 } \to \mathbb { R }$ </td></tr></table>

end of Decomposition of Transformers

Note that, the atomic constructions include two kinds: one for atomic actions, the other for guards in programming language. The set of atomic actions is denoted ${ \sf A c t } = \left\{ \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 } , \alpha _ { 5 } \right\} \cup$ J $\{ \alpha _ { 6 } , \alpha _ { 7 } , \alpha _ { 8 } , \alpha _ { 9 } , \alpha _ { 1 0 } , \alpha _ { 1 1 } , \alpha _ { 1 2 } , \alpha _ { 1 3 } , \alpha _ { 1 4 } , \alpha _ { 1 5 } , \alpha _ { 1 6 } , \alpha _ { 1 7 } , \alpha _ { 1 8 } , \alpha _ { 1 9 } , \alpha _ { 2 0 } \}$ and the set of guards is denoted $\mathsf { G r d } = \{ \phi _ { 1 } \}$ . In the boolean functions, the truth value true is interpreted as number 1 and false as $0 ;$ in the general vectors or matrices operations, true is interpreted as I and false as 0.

## 2 Preliminaries

For self-satisfactory, in this chapter, we introduce the preliminaries on multiset in section 2.1, equational logic in section 2.2 and pomset labelled transition systems in section 2.3.

## 2.1 Multiset

Definition 2.1 (Labelled poset). A labelled poset is a tuple $\mathbf { u } = \langle S , \leq , \lambda \rangle$ , where S is the carrier set, ≤ is a partial order on S and λ is a labelling function $\lambda : S \to \Sigma$

For a labelled poset $u , S _ { \mathbf { u } } , \le _ { \mathbf { u } }$ and $\lambda _ { \mathbf { u } }$ denote the carrier, the partial order and the labelling of u respectively. The set of labelled posets is denoted LP and the empty labelled poset is 1.

Definition 2.2 (Poset morphism). For posets $\langle S , \leq \rangle$ and $\langle S ^ { \prime } , \leq ^ { \prime } \rangle$ and function $\varphi : S  S ^ { \prime }$ , f is called a poset morphism if for $a _ { 0 } , a _ { 1 } \in S$ with $a _ { 0 } \leq a _ { 1 }$ , then $\varphi ( a _ { 0 } ) \leq ^ { \prime } \varphi ( a _ { 1 } )$ holds.

Definition 2.3 (Labelled poset isomorphism). Let $\mathbf { u } = \langle S _ { 1 } , \leq _ { 1 } , \lambda _ { 1 } \rangle$ and $\mathbf { v } = \langle S _ { 2 } , \leq _ { 2 } , \lambda _ { 2 } \rangle$ be labelled posets. A labelled poset morphism φ from $\mathbf { u } = \langle S _ { 1 } , \leq _ { 1 } , \lambda _ { 1 } \rangle$ to $\mathbf { v } = \langle S _ { 2 } , \leq _ { 2 } , \lambda _ { 2 } \rangle$ is a poset morphism from $\langle S _ { 1 } , \leq _ { 1 } \rangle$ and $\langle S _ { 2 } , \leq _ { 2 } \rangle$ with $\lambda _ { 2 } \circ \varphi = \lambda _ { 1 }$ . Moreover, φ is a labelled poset isomorphism if it is a bijection with $\varphi ^ { - 1 }$ is a poset isomorphism from $\langle S _ { 2 } , \leq _ { 2 } , \lambda _ { 2 } \rangle$ to $\langle S _ { 1 } , \leq _ { 1 } , \lambda _ { 1 } \rangle$ . We say that $\mathbf { u } = \langle S _ { 1 } , \leq _ { 1 } , \lambda _ { 1 } \rangle$ is isomorphic to $\mathbf { v } = \langle S _ { 2 } , \leq _ { 2 } , \lambda _ { 2 } \rangle$ denoted $\langle S _ { 1 } , \Sigma _ { 1 } , \lambda _ { 1 } \rangle \sim \langle S _ { 2 } , \Sigma _ { 2 } , \lambda _ { 2 } \rangle$ , if there exists a poset isomorphism φ between $\langle S _ { 1 } , \leq _ { 1 } , \lambda _ { 1 } \rangle$ and $\langle S _ { 2 } , \leq _ { 2 } , \lambda _ { 2 } \rangle$

Definition 2.4 (Multiset). A multiset is a kind of set of objects which may be repetitive denoted $\left\{ - \right\}$ , such that $\{ | a , b , b | \}$ is significantly distinguishable from $\{ \boldsymbol { a } , \boldsymbol { b } \}$

Definition 2.5 (Pomset). A partially ordered multiset, pomset, is a ∼-equivalence class of posets. The set of pomsets is denoted Pom; the empty pomset is denoted I and the ∼-equivalence class of I is also denoted by I.

Definition 2.6 (Pomset composition in parallel). Let U, V ∈ Pom with $U = [ \mathbf { u } ]$ and $V = [ \mathbf { v } ]$ We write U ∥ V for the parallel composition of U and V , which is the pomset represented $b y$ $\mathbf { u } \parallel \mathbf { v } _ { : }$ , where

$$
A _ { \mathbf { u } \parallel \mathbf { v } } = A _ { \mathbf { u } } \cup A _ { \mathbf { v } } \qquad \leq _ { \mathbf { u } \parallel \mathbf { v } } = \leq _ { \mathbf { u } } \cup \leq _ { \mathbf { v } }
$$

Definition 2.7 (Pomset composition in sequence). Let $U , V \in$ Pom with $U = [ \mathbf { u } ]$ and $V = [ \mathbf { v } ]$ We write $U \cdot V$ for the sequential composition of U and V, which is the pomset represented by $\mathbf { u } \cdot \mathbf { v } _ { \mathrm { ~ : ~ } }$ , where

$$
A _ { \mathbf { u } \cdot \mathbf { v } } = A _ { \mathbf { u } } \cup A _ { \mathbf { v } } \qquad \quad \leq _ { \mathbf { u } \cdot \mathbf { v } } = \leq _ { \mathbf { u } } \cup \leq _ { \mathbf { v } } \cup \big ( A _ { \mathbf { u } } \times A _ { \mathbf { v } } \big )
$$

Definition 2.8 (Pomset types). Let U ∈ Pom, U is sequential (resp. parallel) if there exist non-empty pomsets $U _ { 1 }$ and $U _ { 2 }$ such that $U = U _ { 1 } \cdot U _ { 2 }$ (resp. $U = U _ { 1 } \parallel U _ { 2 } )$

Definition 2.9 (Factorization). Let U ∈ Pom. (1) When $U = U _ { 1 } \cdot \cdots \cdot U _ { i } \cdot \cdots \cdot U _ { n }$ with each $U _ { i }$ non-sequential and non-empty, the sequence $U _ { 1 } , \cdots , U _ { i } , \cdots , U _ { n }$ is called a sequential factorization of U. (2) When $U = U _ { 1 } \parallel \cdots \parallel U _ { i } \parallel \cdots \parallel U _ { n }$ with each $U _ { i }$ non-parallel and non-empty, the multiset $\{ U _ { 1 } , \cdots , U _ { i } , \cdots , U _ { n } \}$ is called a parallel factorization of U.

Lemma 2.10 (Factorization). Sequential and parallel factorizations exist uniquely.

## 2.2 Equational Logic

Definition 2.11 (Signature). A signature Σ consists of a finite set of function symbols (or operators) $f , g , \cdots$ , where each function symbol f has an arity ar(f), being its number of arguments. A function symbol $\alpha , \beta , \gamma , \cdots$ of arity zero is called a constant, a function symbol of arity one is called unary, and a function symbol of arity two is called binary.

Definition 2.12 (Term). Let Σ be a signature. The set T(Σ) of (open) terms s, t, u, ⋯ over Σ is defined as the least set satisfying: (1)each variable is in T(Σ); (2) if f ∈ Σ and ${ t } _ { 1 } , \cdots , { t } _ { a r ( f ) } \in$ T(Σ), then $f ( t _ { 1 } , \cdots , t _ { a r ( f ) } \in \mathbb { T } ( \Sigma ) )$ . A term is closed if it does not contain free variables. The set of closed terms is denoted by T (Σ).

Definition 2.13 (Substitution). Let Σ be a signature. A substitution is a mapping σ from variables to the set T(Σ) of open terms. A substitution extends to a mapping from open terms to open terms: the term σ(t) is obtained by replacing occurrences of variables x in t by σ(x). A substitution σ is closed $i f \sigma ( x ) \in \mathcal { T } ( \Sigma )$ for all variables x.

Definition 2.14 (Axiomatization). An axiomatization over a signature Σ is a finite set of equations, called axioms, of the form s = t with $s , t \in \mathbb { T } ( \Sigma )$

Definition 2.15 (Equality relation). An axiomatization over a signature Σ induces a binary equality relation = on T(Σ) as follows. (1)(Substitution) If s = t is an axiom and σ a substitution, then $\sigma ( s ) = \sigma ( t )$ . (2)(Equivalence) The relation = is closed under reflexivity, symmetry, and transitivity. (3)(Context) The relation = is closed under contexts: if t = u and f is a function symbol with ar(f) > 0, then $f \bigl ( s _ { 1 } , \cdots , s _ { i - 1 } , t , s _ { i + 1 } , \cdots , s _ { a r ( f ) } \bigr ) = f \bigl ( s _ { 1 } , \cdots , s _ { i - 1 } , u , s _ { i + 1 } , \cdots , s _ { a r ( f ) } \bigr )$

Definition 2.16 (Model). Assume an axiomatization E over a signature Σ, which induces an equality relation =. A model for E consists of a set M together with a mapping ϕ ∶ T(Σ) → M. (1) (M, ϕ) is sound for E if s = t implies $\phi ( s ) \equiv \phi ( t )$ for s, t ∈ T (Σ); (2) (M, ϕ) is complete for E if ϕ(s) ≡ ϕ(t) implies s = t for s, t ∈ T (Σ).

Definition 2.17 (Term rewriting system). Assume a signature Σ. A rewrite rule is an expression s → t with s, t ∈ T(Σ), where: (1) the left-hand side s is not a single variable; (2) all variables that occur at the right-hand side t also occur in the left-hand side s. A term rewriting system (TRS) is a finite set of rewrite rules.

Definition 2.18 (Rewrite relation). A TRS over a signature Σ induces an one-step rewrite relation → on T(Σ) as follows. (1) (Substitution) If s → t is a rewrite rule and σ a substitution, then $\sigma ( s ) \to \sigma ( t )$ . (2) (Context) The relation → is closed under contexts: if t → u and f is a function symbol with ar(f) > 0, then $f \bigl ( s _ { 1 } , \cdots , s _ { i - 1 } , t , s _ { i + 1 } , \cdots , s _ { a r ( f ) } \bigr ) \to f \bigl ( s _ { 1 } , \cdots , s _ { i - 1 } , u , s _ { i + 1 } , \cdots , s _ { a r ( f ) } \bigr )$ . The rewrite relation →<sup>∗</sup> is the reflexive transitive closure of the one-step rewrite relation →: (1) if s → t, then s →<sup>∗</sup> t; (2) t →<sup>∗</sup> t; (3) if s →<sup>∗</sup> t and $t  ^ { * } u ,$ then $s \to ^ { * } u$

Definition 2.19 (Normal form). A term is called a normal form for a TRS if it cannot be reduced by any of the rewrite rules.

Definition 2.20 (Termination). A TRS is terminating if it does not induce infinite reductions $t _ { 0 }  t _ { 1 }  t _ { 2 }  \cdots$

Definition 2.21 (Weak confluence). A TRS is weakly confluent if for each pair of one-step reductions $s \to t _ { 1 }$ and $s \to t _ { 2 }$ , there is a term u such that $t _ { 1 } \to ^ { * }$ u and $t _ { 2 } \to ^ { * } u$

Theorem 2.22 (Newman’s lemma). If a TRS is terminating and weakly confluent, then it reduces each term to a unique normal form.

Definition 2.23 (Convergence). A pair of terms s and t is said to be convergent if there exists a term u such that $s \to ^ { * }$ u and $t  ^ { * } u$

Axiomatizations can give rise to TRSs that are not weakly confluent, which can be remedied by Knuth-Bendix completion [6]. It determines overlaps in left hand sides of rewrite rules, and introduces extra rewrite rules to join the resulting right hand sides, witch are called critical pairs.

Theorem 2.24 (Weak confluence). A TRS is weakly confluent if and only if all its critical pairs are convergent.

Definition 2.25 (Elimination property). Let a process algebra with a defined set of basic terms as a subset of the set of closed terms over the process algebra. Then the process algebra has the elimination to basic terms property if for every closed term s of the algebra, there exists a basic term t of the algebra such that the algebra⊢ $s = t$

Definition 2.26 (Strongly normalizing). A term $s _ { 0 }$ is called strongly normalizing if does not an infinite series of reductions beginning in $s _ { 0 }$

Definition 2.27. We write $s > _ { l p o } t$ if $s  ^ { + } t$ where →<sup>+</sup> is the transitive closure of the reduction relation defined by the transition rules of an algebra.

Theorem 2.28 (Strong normalization). Let a term rewriting system (TRS) with finitely many rewriting rules and let > be a well-founded ordering on the signature of the corresponding algebra. $I f s > _ { l p o } t$ for each rewriting rule s → t in the TRS, then the term rewriting system is strongly normalizing.

## 2.3 Pomset Labelled Transition System

Definition 2.29 (Pomset labelled transition system). A pomset labelled transition system (PLTS) is a quadruple $\langle \mathsf { P r o c } , \mathsf { A c t } , \{ \overset { U } { \longrightarrow } | U \in \mathsf { P o m } \} , \mathsf { P r e d } \rangle$ , where:

1. Proc is a set of processes, ranged over by $p , q$

2. Act is a set of actions, ranged over by $a , b .$

3. Pom is the set of pomsets over Act, ranged over by $U , V$

4. <sup>U</sup>Ð→⊆ Proc × Pom × Proc is called a pomset transition for every $U \in \mathsf { P o m }$ . We write $p \stackrel { U } { \longrightarrow } q$ instead of $( p , q ) \in { \xrightarrow { U } }$ , and write $p \not \to i f p \stackrel { U } { \to } q$ with no state q. Intuitively, $p \stackrel { U } {  } q$ means that state p can evolve into state q by the execution of pomset U. We see that traditional single action transition $p { \stackrel { a } { \to } } q$ with $a \in \mathsf { A c t }$ is a special case of pomset transition in which the pomset is primitive.

5. For every P ∈ Pred, we write pP (resp. p¬P) if state p satisfies (resp. does not satisfy) predicate P. Intuitively, pP means that predicate P holds in state p.

The binary pomset transitions $p \stackrel { U } { \to } q$ and unary predicates pP in a PLTS are called transitions. Note that, by replacing $U \in$ Pom by $a \in \mathsf { A c t }$ , we can get the definition of traditional labelled transition system (LTS). And we use $\mathsf { A c t } ( p ) \subseteq \mathsf { A c t }$ to denote the set of actions occurring in $p .$

Definition 2.30 (Finiteness conditions on a PLTS). A PLTS is:

• Finitely branching: if for every state p there are only finitely many outgoing pomset transitions $p \stackrel { U } { \longrightarrow } q$

• Regular: if it is finitely branching and each state can reach only finitely many other states.

• Finite: if it is finitely branching and there is no infinite sequence of pomset transitions $p _ { 0 } \xrightarrow { U _ { 0 } } p _ { 1 } \xrightarrow { U _ { 1 } } . . .$

A pomset transition system specification is a collection of inductive proof rules to derive the pomset transitions over states in Proc.

Definition 2.31 (Pomset transition system specification). Let Σ be a signature, and $t , t ^ { \prime } \in \mathbb { T } ( \Sigma )$ A pomset transition rule ρ is of the $\begin{array} { r } { f o r m \ \frac { H } { \alpha } } \end{array}$ , where H is the set of premises with positive premises $t \stackrel { U } { \longrightarrow } t ^ { \prime }$ and tP, and negative premises $t \stackrel { U } { \not \to }$ and $t \neg P ;$ α is the conclusion with the form $t \stackrel { U } { \longrightarrow } t ^ { \prime }$ and $^ { t P , }$ and $U \in \mathsf { P o m }$ and P is a predicate. For the conclusion with the form of $t \stackrel { U } { \longrightarrow } t ^ { \prime } ,$ the left-hand side of the conclusion t is called the source of ρ and the right-hand side of the conclusion $t ^ { \prime }$ is called the target of ρ. A transition rule is closed if it does not contain variables.

A pomset transition system specification (PTSS) is a set of pomset transition rules. A PTSS is positive if its transition rules do not contain negative premises.

Definition 2.32 (Pomset transitions). Let $p , q \in$ Proc and U ∈ Pom, $p \stackrel { U } { \longrightarrow } q$ is called a pomset transition from p to q.

Definition 2.33 (Pomset bisimulation). Let $\langle \mathsf { P r o c } , \mathsf { A c t } , \{ \overset { U } { \longrightarrow } | U \in \mathsf { P o m } \}$ , Pred⟩ be a PLTS, S a set of states, and $p , q \in { \mathsf { P r o c } }$ and $s , s ^ { \prime } \in S$ . A pomset bisimulation is a relation $R \subseteq \langle \mathsf { P r o c } , S \rangle \times$ $\langle \mathsf { P r o c } , S \rangle$ , such that:

1. $I f \left. p , s \right. R \left. q , s \right.$ and $\langle p , s \rangle \stackrel { U } { \longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ , then $\langle q , s \rangle \stackrel { U } { \longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ , with U ∈ Pom, and $\langle p ^ { \prime } , s ^ { \prime } \rangle R \langle q ^ { \prime } , s ^ { \prime } \rangle$

2. $I f \left. p , s \right. R \left. q , s \right.$ and $\langle q , s \rangle \stackrel { U } { \longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ , then $\langle p , s \rangle \stackrel { U } { \longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ , with U ∈ Pom, and $\langle p ^ { \prime } , s ^ { \prime } \rangle R \langle q ^ { \prime } , s ^ { \prime } \rangle$

3. If $\langle p , s \rangle R \langle q , s \rangle$ and $\langle p , s \rangle P$ , then $\langle q , s \rangle P$

4. If $\langle p , s \rangle R \langle q , s \rangle$ and $\langle q , s \rangle P$ , then $\langle p , s \rangle P$

We say that p, q are pomset bisimilar, written p↔q, if there exists a pomset bisimulation R, such that $\langle p , s \rangle R \langle q , s \rangle$

Definition 2.34 (Congruence). Let Σ be a signature. An equivalence relation R on $\tau ( \Sigma )$ is a congruence if for each $f \in \Sigma , \ i f \ s _ { i } R t _ { i } \ f o r \ i \in \{ 1 , \cdots , a r ( f ) \}$ , then $f ( s _ { 1 } , \cdots , s _ { a r ( f ) } ) R f ( t _ { 1 } , \cdots , t _ { a r ( f ) } )$

Definition 2.35 (Operational conservative extension). $\Sigma _ { 0 }$ and $\Sigma _ { 1 }$ are two signatures, the union of $\Sigma _ { 0 }$ and $\Sigma _ { 1 }$ , denoted $\Sigma _ { 0 } \oplus \Sigma _ { 1 }$ , agrees on the arity of the function symbols in the intersection $o f \Sigma _ { 0 }$ and $\Sigma _ { 1 } . \ T _ { 0 }$ and $T _ { 1 }$ are $P T S S s$ over $\Sigma _ { 0 }$ and $\Sigma _ { 1 }$ , the sum of $T _ { 0 }$ and $T _ { 1 }$ , denoted $T _ { 0 }$ ⊕ $T _ { 1 }$ , is the PTSS over signature $\Sigma _ { 0 } \oplus \Sigma _ { 1 }$ containing the pomset transition rules in $T _ { 0 }$ and $T _ { 1 }$ . A PTSS $T _ { 0 } \oplus T _ { 1 }$ is an operational conservative extension of PTSS $T _ { 0 }$ , if for each closed transition rule $\textstyle { \frac { N } { \alpha } }$ such that:

• N contains only negative literals.

• The left-hand side of α is in $T ( \Sigma _ { 0 } )$

$\begin{array} { r } { T _ { 0 } \oplus T _ { 1 } \vdash \frac { N } { \alpha } } \end{array}$

we have that $T _ { 0 } \vdash { \frac { N } { \alpha } }$

Definition 2.36 (Source-dependency). The source-dependent variables in a pomset transition rule $\rho$ are defined inductively as follows:

• All variables in the source of ρ are source-dependent.

$I f t \stackrel { U } { \longrightarrow } t ^ { \prime }$ is a premise of ρ and all variables in t are source-dependent, then all variables in $t ^ { \prime }$ are source-dependent.

A pomset transition rule is source-dependent if all its variables are.

Definition 2.37 (Freshness). We say that a term in $\mathbb { T } ( \Sigma _ { 0 } \oplus \Sigma _ { 1 } )$ is fresh if it contains a function symbol (may be a constant) from $\Sigma _ { 1 } \setminus \Sigma _ { 0 }$ . Similarly, a predicate symbol in $T _ { 1 }$ is fresh if it does not occur in $T _ { 0 }$

Theorem 2.38 (Operational conservative extension). Let $T _ { 0 }$ and $T _ { 1 }$ be PTSSs over signature $\Sigma _ { 0 }$ and $\Sigma _ { 1 }$ , respectively. $T _ { 0 } \oplus T _ { 1 }$ is an operational conservative extension of $T _ { 0 }$ $i f { \mathrm { : } }$

1. Each $\rho \in T _ { 0 }$ is source-dependent.

2. For each $\rho \in T _ { 1 }$

• Either the source of ρ is fresh.

• Or $\rho$ has a premise of the form $t \stackrel { U } { \longrightarrow } t ^ { \prime } o r t { \cal P } ,$ , where:

$t \in \mathbb { T } ( \Sigma _ { 0 } )$

– All variables in t occur in the source of $\rho .$

$t ^ { \prime } , P$ and every $a \in U$ are fresh.

## 3 Basic Algebra for Transformers

In this section, we will discuss the algebraic laws for the set of atomic actions Act the set of guards Grd, the deadlock constant 0 and the identity I, the set of pomsets over Act ∪ $\{ \mathbf { 0 } \} \cup \{ \mathbf { I } \}$ denoted Pom, and the serial composition ○ and the alternative composition +, and let $\alpha \in \mathsf { A c t \cup \{ \mathbf { 0 } \} \cup \{ \mathbf { I } \} }$ $U \in \mathsf { P o m }$ . The resulted algebra is called Basic Algebra for True Concurrency, abbreviated BAT.

## 3.1 Axiomatic System

In the following, let $\alpha , \beta , \alpha ^ { \prime } , \beta ^ { \prime } \in \mathsf { A c t } , \phi , \psi \in \mathsf { G r d } .$ and let variables $x , y , z$ range over the set of terms BAT, $p , q , s$ range over the set of closed terms. The set of axioms of BAT consists of the laws given in Table 2. The predicate $t e s t ( \phi , s )$ represents that $\phi$ holds in the state $s ,$ and test(I, s) holds and test $( \mathbf { 0 } , s )$ does not hold. $e f f e c t ( U , s ) \in S$ denotes $s ^ { \prime }$ in $s \stackrel { U } { \longrightarrow } s ^ { \prime }$ for $U \in \mathsf { P o m } .$ The predicate weakest precondition $w p ( U , \phi )$ denotes that $\forall s , s ^ { \prime } \in S , t e s t ( \phi , e f f e c t ( U , s ) )$ holds. We extend Grd to the set of basic guards G with element $\phi , \psi , \cdots$ , which is generated by the following formation rules:

$$
\phi : = 0 \mid \mathbf { I } \mid \lnot \phi \mid \psi \in { \mathsf { G r d } } \mid \phi + \psi \mid \phi \circ \psi
$$

The deadlock 0 is caused by incomposibility of serial composition, which may be caused by the mismatch of dimensions of vectors or matrices, we define the composibility function of two actions $a , b \in \mathsf { A c t }$ as follows:

$$
\gamma ( a , b ) = { \left\{ \begin{array} { l l } { a \circ b , } & { a , b { \mathrm { ~ a r e ~ c o m p o s i b l e } } ; } \\ { \mathbf { 0 } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Each serial composition implicitly executes this function.

Table 2: Axioms of BAT
<table><tr><td> $\mathrm { N o . }$ </td><td>Axioms</td></tr><tr><td>A1</td><td> $x + y = y + x$ </td></tr><tr><td>A2</td><td> $( x + y ) + z = x + ( y + z )$ </td></tr><tr><td>A3</td><td> $x + x = x$ </td></tr><tr><td>A4</td><td> $( x + y ) \circ z = x \circ z + y \circ z$ </td></tr><tr><td>A5</td><td> $( x \circ y ) \circ z = x \circ ( y \circ z )$ </td></tr><tr><td>A6</td><td> $x + \mathbf { 0 } = x$ </td></tr><tr><td>A7</td><td> $\mathbf { 0 } \circ x = \mathbf { 0 }$ </td></tr><tr><td>A8</td><td> $\mathbf { I } \circ x = x$ </td></tr><tr><td>A9</td><td> $x \circ \mathbf { I } = x$ </td></tr><tr><td>G1</td><td> $\phi \circ \lnot \phi = \mathbf { 0 }$ </td></tr><tr><td>G2</td><td></td></tr><tr><td></td><td> $\phi + \lnot \phi = \mathbf { I }$ </td></tr><tr><td>G3</td><td> $\phi \circ \mathbf { 0 } = \mathbf { 0 }$ </td></tr><tr><td>G4</td><td> $\phi \circ ( x + y ) = \phi \circ x + \phi \circ y$ </td></tr></table>

to be continued. . .

continuing: Axioms of BAT
<table><tr><td>No.</td><td>Axioms</td></tr><tr><td>G5 G6 G7 G8 G9 G10 G11</td><td> $\phi \circ ( x \circ y ) = ( \phi \circ x ) \circ y$   $\left( \phi + \psi \right) \circ x = \phi \circ x + \psi \circ x$   $( \phi \circ \psi ) \circ x = \phi \circ ( \psi \circ x )$   $\phi = \mathbf { I } { \mathrm { ~ i f ~ } } \forall s \in S . t e s t ( \phi , s )$   $\phi _ { 0 } \circ \cdots \circ \phi _ { n } = \mathbf { 0 } { \mathrm { ~ i f ~ } } \forall s \in S , \exists i \leq n . t e s t ( \lnot \phi _ { i } , s )$   $w p ( U , \phi ) U \phi = w p ( U , \phi ) U$   $\neg w p ( U , \phi ) U \neg \phi = \neg w p ( U , \phi ) U$  end of Axioms of BAT</td></tr></table>

Definition 3.1 (Basic terms of BAT). The set of basic terms of BAT, B(BAT), is inductively defined as follows:

1. $\mathsf { A c t \cup \{ \mathbf { 0 } \} \cup \{ \mathbf { I } \} } \subset B ( B A T )$

2. Grd ⊂ B(BAT).

3. $I f U \in \mathsf { P o m } , t \in \mathcal { B } ( B A T ) \ t h e n \ U \circ t \in \mathcal { B } ( B A T )$

4. $I f \phi \in { \mathsf { G r d } } , t \in B ( B A T )$ then $\phi \circ t \in { \cal { B } } ( B A T )$

5. $I f t , s \in B ( B A T )$ then $t + s \in { \mathcal { B } } ( B A T )$

Theorem 3.2 (Elimination theorem of BAT). Let p be a closed BAT term. Then there is a basic BAT term q such that $B A T \vdash p = q$

Proof. (1) Firstly, suppose that the following ordering on the signature of BAT is defined: $\circ > +$ and the symbol ○ is given the lexicographical status for the first argument, then for each rewrite rule $p  q$ in Table 3 relation $p > _ { l p o } q$ can easily be proved. We obtain that the term rewrite system shown in Table 3 is strongly normalizing, for it has finitely many rewriting rules, and > is a well-founded ordering on the signature of BAT, and if $s > _ { l p o } t ,$ for each rewriting rule $s  t$ is in Table 3 (see Theorem 2.28).

Table 3: Rewriting rules of BAT
<table><tr><td>No.</td><td>Rewriting rules</td></tr><tr><td>RA3 RA4 RA5 RA6 RA7 RA8 RA9 RG1</td><td> $x + x  x$   $( x + y ) \circ z \to x \circ z + y \circ z$   $( x \circ y ) \circ z \to x \circ ( y \circ z )$   $x + \mathbf { 0 }  x$   $\mathbf { 0 } \circ x  \mathbf { 0 }$   $\mathbf { I } \circ x  x$   $x \circ \mathbf { I }  x$ </td></tr></table>

to be continued. . .

continuing: Rewriting rules of BAT
<table><tr><td>No.</td><td>Rewriting rules</td></tr><tr><td>RG2 RG3 RG4 RG5 RG6 RG7</td><td> $\phi + \lnot \phi \to \mathbf { I }$   $\phi \circ \mathbf { 0 }  \mathbf { 0 }$   $\phi \circ ( x + y )  \phi \circ x + \phi \circ y$   $\phi \circ ( x \circ y )  ( \phi \circ x ) \circ y$   $( \phi + \psi ) \circ x  \phi \circ x + \psi \circ x$ </td></tr><tr><td colspan="2">RG8  $\phi \to \mathbf { I } { \mathrm { ~ i f ~ } } \forall s \in S . t e s t ( \phi , s )$  RG9  $\phi _ { 0 } \circ \cdots \circ \phi _ { n }  \mathbf { 0 } { \mathrm { ~ i f ~ } } \forall s \in S , \exists i \leq n . t e s t ( \lnot \phi _ { i } , s )$  RG10  $w p ( U , \phi ) U \phi \to w p ( U , \phi ) U$  RG11</td></tr><tr><td colspan="2"> ${ \neg { w p } } ( U , \phi ) U { \neg { \phi } } \to { \neg { w p } } ( U , \phi ) U$  end of Axioms of BAT</td></tr></table>

(2) Then we prove that the normal forms of closed BAT terms are basic BAT terms.

Suppose that $p$ is a normal form of some closed BAT term and suppose that $p$ is not a basic term. Let $p ^ { \prime }$ denote the smallest sub-term of $p$ which is not a basic term. It implies that each sub-term of $p ^ { \prime }$ is a basic term. Then we prove that $p$ is not a term in normal form. It is suficient to induct on the structure of $p ^ { \prime } { : }$

• Case $p ^ { \prime } \equiv U , U \in \mathsf { P o m . } p ^ { \prime }$ is a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term, so this case should not occur.

• Case $p ^ { \prime } \equiv \phi , \phi \in { \sf G r d . } \ p ^ { \prime }$ is a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term, so this case should not occur.

• Case $p ^ { \prime } \equiv p _ { 1 } \circ p _ { 2 }$ . By induction on the structure of the basic term $p _ { 1 } .$ :

– Subcase $p _ { 1 } \in$ Pom. $p ^ { \prime }$ would be a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term;

– Subcase $p _ { 1 } \in { \mathsf { G r d } } . \ p ^ { \prime }$ would be a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term;

– Subcase $p _ { 1 } \equiv U \circ p _ { 1 } ^ { \prime }$ . RA5 or RA9 rewriting rule can be applied. So $p$ is not a normal form;

– Subcase $p _ { 1 } \equiv \phi \circ p _ { 1 } ^ { \prime }$ . RG1, RG3, RG4, RG5, RG7, or RG8 − 9 rewriting rules can be applied. So $p$ is not a normal form;

– Subcase $p _ { 1 } \equiv p _ { 1 } ^ { \prime } + p _ { 1 } ^ { \prime \prime }$ . RA4, RA6, RG2, or RG6 rewriting rules can be applied. So $p$ is not a normal form.

• Case $p ^ { \prime } \equiv p _ { 1 } + p _ { 2 }$ . By induction on the structure of the basic terms both $p _ { 1 }$ and $p _ { 2 }$ , all subcases will lead to that $p ^ { \prime }$ would be a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term.

$$
\begin{array} { r l } & { \frac { \partial \big ( \mathrm { J } , \boldsymbol { s } \big ) } { \partial x } \big ( \mathrm { J } , \boldsymbol { s } \big ) = \big ( \mathrm { J } , \boldsymbol { s } \big ) } \\ & { \big ( \mathrm { J } , \boldsymbol { s } \big ) \frac { \partial \mathrm { J } } { \partial y } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \big ( \mathrm { J } , \boldsymbol { s } \big ) - \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) \big ( \mathrm { H } ^ { \mathcal { I } } \big ( \mathrm { e } ^ { \mathcal { I } } , \boldsymbol { s } \big ) } \\ & { \big ( \mathrm { J } , \boldsymbol { s } \big ) \big ) \frac { \partial \mathrm { J } } { \partial x } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \frac { \partial ( \mathrm { J } , \boldsymbol { s } , \boldsymbol { s } ) } { \partial x } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) , \qquad \frac { \partial ( \mathrm { J } , \boldsymbol { s } ) } { \partial x } \big ( \mathrm { J } ^ { \mathcal { I } } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \big ( x + y , s \big ) \frac { \partial \mathrm { J } } { \partial y } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \frac { ( y , s ) } { \partial x } \frac { \partial \mathrm { J } } { \partial y } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \big ( x + y , s \big ) \frac { \partial ^ { 2 } } { \partial x } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \frac { \partial ( x , s ) } { \partial y } \frac { \partial \mathrm { J } } { \partial y } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \\ & { \big ( x , s \big ) \frac { \partial ^ { 2 } } { \partial x } \big ( \mathrm { J } , \boldsymbol { s } ^ { \prime } \big ) } \end{array} \frac  \big ( \mathrm { J } , \boldsymbol { s } \big ) ^
$$

Table 4: Transition rules of BAT

## 3.2 Operational Semantics

We give the operational transition rules for I, atomic action $\alpha \in \mathsf { A c t } .$ , pomset U ∈ Pom, guard $\phi \in$ Grd and operators ○ and + as Table 4 shows. And the predicate $\xrightarrow { U } \surd$ represents successful termination after execution of U.

Theorem 3.3 (Congruence of BAT with respect to pomset bisimulation equivalence). Pomset bisimulation equivalence is a congruence with respect to BAT.

Proof. Since the PTSS of BAT in Table 4 is positive after reduction and in panth format, pomset bisimulation equivalence is a congruence with respect to BAT. □

## 3.3 Soundness and Completeness

Theorem 3.4 (Soundness of BAT modulo pomset bisimulation equivalence). The axiomatization of BAT is sound modulo pomset bisimulation equivalence, i.e., let x and y be BAT terms. If $B A T \vdash x = y$ , then x↔y.

Proof. Since pomset bisimulation ↔ is both an equivalent and a congruent relation, we only need to check if each axiom in Table 2 is sound modulo pomset bisimulation equivalence ↔. The proof is trivial and we omit it. □

Theorem 3.5 (Completeness of BAT modulo pomset bisimulation equivalence). The axiomatization of BAT is complete modulo pomset bisimulation equivalence, i.e., let p and q be closed BAT terms, if p↔q then $p = q$

Proof. Firstly, by the elimination theorem of BAT, we know that for each closed BAT term $p ,$ there exists a closed basic BAT term $p ^ { \prime }$ , such that $B A T \vdash p = p ^ { \prime }$ , so, we only need to consider closed basic BAT terms.

The basic terms (see Definition 3.1) modulo associativity and commutativity $\mathrm { ( A C ) }$ of conflict + (defined by axioms A1 and A2 in Table 2), and this equivalence is denoted $\mathrm { b y } = _ { A C }$ . Then, each equivalence class s modulo AC of + has the following normal form

$$
s _ { 1 } + \cdots + s _ { k }
$$

with each $s _ { i }$ either an atomic event or of the form $t _ { 1 } \circ t _ { 2 }$ , and each $s _ { i }$ is called the summand of s.

Now, we prove that for normal forms n and $n ^ { \prime } .$ , if $n { \stackrel {  } { \longrightarrow } } n ^ { \prime }$ then $n = _ { A C } n ^ { \prime }$ . It is suficient to induct on the sizes of n and $n ^ { \prime }$

• Consider a summand U of n. Then $\langle n , s \rangle \stackrel { U } { \longrightarrow } \langle \surd , s ^ { \prime } \rangle$ , so $n \sim _ { p } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle \stackrel { U } { \longrightarrow } \langle \surd , s \rangle$ 2 meaning that $n ^ { \prime }$ also contains the summand U.

• Consider a summand $\phi$ of $n .$ . Then $\langle n , s \rangle \to \langle \sqrt { , } s \rangle$ , if $t e s t ( \phi , s )$ holds, so $n \sim _ { p } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle  \langle \sqrt { , } s \rangle$ , if $t e s t ( \phi , s )$ holds, meaning that $n ^ { \prime }$ also contains the summand $\phi .$

• Consider a summand $t _ { 1 } \circ t _ { 2 }$ of n. Then $\langle n , s \rangle \stackrel { t _ { 1 } } { \longrightarrow } \langle t _ { 2 } , s ^ { \prime } \rangle$ , so $n \sim _ { p } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle \stackrel { t _ { 1 } } { \longrightarrow } \langle t _ { 2 } ^ { \prime } , s ^ { \prime } \rangle$ with $t _ { 2 } \sim _ { p } t _ { 2 } ^ { \prime } ,$ , meaning that $n ^ { \prime }$ contains a summand $t _ { 1 } \circ t _ { 2 } ^ { \prime }$ . Since $t _ { 2 }$ and $t _ { 2 } ^ { \prime }$ are normal forms and have sizes smaller than n and $n ^ { \prime } { \mathrm { . } }$ by the induction hypotheses $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ implies $t _ { 2 } = _ { A C } t _ { 2 } ^ { \prime }$

So, we get $n = _ { A C } n ^ { \prime }$

Finally, let s and t be basic terms, and $s {  } t$ , there are normal forms n and $n ^ { \prime } ,$ , such that $s = n$ and $t = n ^ { \prime }$ . The soundness theorem of BAT modulo pomset bisimulation equivalence (see Theorem 3.4) yields $s {  } n$ and $t { \stackrel {  } { \mathop {  } } } n ^ { \prime }$ , so $n { \stackrel {  } { \mathop { = } } } s { \stackrel {  } { \mathop { = } } } t { \stackrel {  } { \mathop { = } } } n ^ { \prime }$ . Since if $n { \stackrel {  } { \longrightarrow } } n ^ { \prime }$ then $n = _ { A C } n ^ { \prime } , s = n = _ { A C } n ^ { \prime } = t$ , as desired. □

$$
{ \frac { x \stackrel { \alpha } {  } \sqrt { \quad y \stackrel { \alpha ^ { \prime } } { \longrightarrow } \vee } } { x \oplus y \stackrel { \{ \| \alpha , \alpha ^ { \prime } \| } } { \longrightarrow } }  \quad { \frac { x \stackrel { \alpha } {  } x ^ { \prime } \quad y \stackrel { \alpha ^ { \prime } } { \longrightarrow } \vee } { x \oplus y \stackrel { \{ \| \alpha , \alpha ^ { \prime } \| } } { \longrightarrow } x ^ { \prime } } 
$$

$$
{ \frac { x \stackrel { \alpha } {  } \sqrt { \quad y \stackrel { \alpha ^ { \prime } } { \longrightarrow } y ^ { \prime } } } { x \oplus y { \frac { \{ \| \alpha , \alpha ^ { \prime } \| } } { x \Longleftrightarrow y ^ { \prime } } } }  \quad { \frac { x \stackrel { \alpha } {  } x ^ { \prime } \quad y \stackrel { \alpha ^ { \prime } } { \longrightarrow } y ^ { \prime } } { x \oplus y { \frac { \{ \| \alpha , \alpha ^ { \prime } \| } } { x \Longleftrightarrow x ^ { \prime } \oplus y ^ { \prime } } } } 
$$

Table 5: Transition rules of APT

## 4 Algebra of Parallelism for Transformers

In this section, we will discuss parallel composition. The resulted algebra is called Algebra of Parallelism for Transformers, abbreviated APT.

## 4.1 Operational Semantics

Firstly, we give the transition rules for parallelism as table 5 shows.

Theorem 4.1 (Congruence theorem of APT). Pomset bisimulation equivalences is a congruence with respect to APT.

Proof. Since the PTSS of APT in Table 5 is positive after reduction and in panth format, pomset bisimulation equivalence ↔ is a congruence with respect to APT. □

## 4.2 Axiomatic System

We design the axioms of APT in Table 6.

Table 6: Axioms of APT
<table><tr><td>No.</td><td>Axioms</td></tr><tr><td>P1</td><td> $\alpha \oplus ( \alpha ^ { \prime } \circ y ) = ( \alpha \oplus \alpha ^ { \prime } ) \circ y$ </td></tr><tr><td>P2 P3</td><td> $\left( \alpha \circ x \right) \oplus \alpha ^ { \prime } = \left( \alpha \oplus \alpha ^ { \prime } \right) \circ x$ </td></tr><tr><td>P4</td><td> $\left( \alpha \circ x \right) \oplus \left( \alpha ^ { \prime } \circ y \right) = \left( \alpha \oplus \alpha ^ { \prime } \right) \circ \left( x \oplus y \right)$ </td></tr><tr><td>P5</td><td> $( x + y ) \oplus z = ( x \oplus z ) + ( y \oplus z )$ </td></tr><tr><td></td><td> $x \oplus ( y + z ) = ( x \oplus y ) + ( x \oplus z )$ </td></tr><tr><td>P6</td><td>0 ⊕ x = 0</td></tr><tr><td>P7</td><td>x ⊕ 0 = 0</td></tr><tr><td>P8</td><td>I ⊕ x = x</td></tr><tr><td>P9</td><td>x ⊕ I = x</td></tr><tr><td>G12</td><td> $\phi \circ ( x \oplus y ) = \phi \circ x \oplus \phi \circ y$ </td></tr><tr><td>G13</td><td>φ⊕0 = 0</td></tr></table>

to be continued. . .

continuing: Axioms of APT
<table><tr><td>No.</td><td>Axioms</td></tr><tr><td>G14</td><td> $\mathbf { 0 } \oplus \phi = \mathbf { 0 }$ </td></tr><tr><td>G15</td><td> $\phi \oplus \mathbf { I } = \phi$ </td></tr><tr><td>G16</td><td> $\mathbf { I } \oplus \phi = \phi$ </td></tr><tr><td>G17</td><td> $\phi \oplus \lnot \phi = \mathbf { 0 }$ </td></tr><tr><td>G18</td><td> $\phi _ { 0 } \oplus \cdots \oplus \phi _ { n } = \mathbf { 0 } { \mathrm { ~ i f ~ } } \forall s _ { 0 } , \cdots , s _ { n } \in S , \exists i \leq n . t e s t { \bigl ( } \lnot \phi _ { i } , s _ { 0 } \cup \cdots \cup s _ { n } { \bigr ) }$ </td></tr></table>

end of Axioms of APT

Definition 4.2 (Basic terms of APT). The set of basic terms of APT, B(AP T), is inductively defined as follows:

1. $\mathsf { A c t \cup \{ \mathbf { 0 } \} \cup \{ \mathbf { I } \} } \subset B ( A P T )$

2. $\mathsf { G r d } \subset B ( A P T )$

3. $I f U \in \mathsf { P o m } , t \in { \mathcal { B } } ( A P T )$ then $U \circ t \in { \cal { B } } ( A P T )$

4. $I f \phi \in { \mathsf { G r d } } , t \in B ( A P T )$ then $\phi \circ t \in { \cal B } ( A P T )$ ).

5. $I f t , s \in B ( A P T )$ then $t + s \in { \mathcal { B } } ( A P T )$

6. $I f t , s \in B ( A P T )$ then $t \oplus s \in \mathcal { B } ( A P T )$

Theorem 4.3 (Elimination theorem of parallelism). Let p be a closed APT term. Then there is a basic APT term q such that $A P T \vdash p = q$

Proof. (1) Firstly, suppose that the following ordering on the signature of APT is defined: $\oplus > \circ > +$ and the symbol ○ is given the lexicographical status for the first argument, then for each rewrite rule $p \to q$ in Table 7 relation $p > _ { l p o } q$ can easily be proved. We obtain that the term rewrite system shown in Table 7 is strongly normalizing, for it has finitely many rewriting rules, and > is a well-founded ordering on the signature of APT, and if $s > _ { l p o } t ,$ , for each rewriting rule s → t is in Table 7 (see Theorem 2.28).

Table 7: Rewriting rules of APT
<table><tr><td>No.</td><td>Rewriting rules</td></tr><tr><td>RP1 RP2</td><td> $\alpha \oplus ( \alpha ^ { \prime } \circ y )  ( \alpha \oplus \alpha ^ { \prime } ) \circ y$   $( \alpha \circ x ) \oplus \alpha ^ { \prime } \to ( \alpha \oplus \alpha ^ { \prime } ) \circ x$ </td></tr><tr><td>RP3</td><td> $( \alpha \circ x ) \oplus ( \alpha ^ { \prime } \circ y )  ( \alpha \oplus \alpha ^ { \prime } ) \circ ( x \oplus y )$ </td></tr><tr><td>RP4</td><td> $\left( x + y \right) \oplus z \to \left( x \oplus z \right) + \left( y \oplus z \right)$ </td></tr><tr><td>RP5</td><td> $x \oplus ( y + z ) \to ( x \oplus y ) + ( x \oplus z )$ </td></tr><tr><td>RP6</td><td> $\mathbf { 0 } \oplus x \to \mathbf { 0 }$ </td></tr><tr><td>RP7</td><td> $x \oplus \mathbf { 0 } \to \mathbf { 0 }$ </td></tr><tr><td>RP8</td><td></td></tr><tr><td></td><td> $\mathbf { I } \oplus x \to x$ </td></tr></table>

to be continued. . .

continuing: Rewriting rules of APT
<table><tr><td> $\mathrm { N o } .$ </td><td>Rewriting rules</td></tr><tr><td>RP9</td><td> $x \oplus \mathbf { I } \to x$ </td></tr><tr><td>RG12</td><td> $\phi \circ ( x \oplus y ) \to \phi \circ x \oplus \phi \circ y$ </td></tr><tr><td>RG13</td><td> $\phi \oplus \mathbf { 0 } \to \mathbf { 0 }$ </td></tr><tr><td>RG14</td><td> $\mathbf { 0 } \oplus \phi \to \mathbf { 0 }$ </td></tr><tr><td>RG15</td><td> $\phi \oplus \mathbf { I } \to \phi$ </td></tr><tr><td>RG16</td><td> $\mathbf { I } \oplus \phi \to \phi$ </td></tr><tr><td>RG17</td><td> $\phi \oplus \lnot \phi \to \mathbf { 0 }$ </td></tr><tr><td>RG18</td><td> $\phi _ { 0 } \oplus \cdots \oplus \phi _ { n } \to \mathbf { 0 } { \mathrm { ~ i f ~ } } \forall s _ { 0 } , \cdots , s _ { n } \in S , \exists i \leq n . t e s t { \bigl ( } \lnot \phi _ { i } , s _ { 0 } \cup \cdots \cup s _ { n } { \bigr ) }$ </td></tr></table>

end of Rewriting rules of $\mathrm { A P T }$

(2) Then we prove that the normal forms of closed APT terms are basic APT terms.

Suppose that $p$ is a normal form of some closed APT term and suppose that $p$ is not a basic APT term. Let $p ^ { \prime }$ denote the smallest sub-term of $p$ which is not a basic APT term. It implies that each sub-term of $p ^ { \prime }$ is a basic APT term. Then we prove that $p$ is not a term in normal form. It is suficient to induct on the structure of $p ^ { \prime } { : }$

• Case $p ^ { \prime } \equiv U , U \in$ Pom. $p ^ { \prime }$ is a basic APT term, which contradicts the assumption that $p ^ { \prime }$ is not a basic APT term, so this case should not occur.

• Case $p ^ { \prime } \equiv \phi , \phi \in$ Grd. $p ^ { \prime }$ is a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term, so this case should not occur.

• Case $p ^ { \prime } \equiv p _ { 1 } \circ p _ { 2 }$ . By induction on the structure of the basic APT term $p _ { 1 }$ :

– Subcase $p _ { 1 } \in \mathsf { P o m . } \ p ^ { \prime }$ would be a basic APT term, which contradicts the assumption that $p ^ { \prime }$ is not a basic APT term;

– Subcase $p _ { 1 } \in { \mathsf { G r d } } . \ p ^ { \prime }$ would be a basic term, which contradicts the assumption that $p ^ { \prime }$ is not a basic term;

– Subcase $p _ { 1 } \equiv U \circ p _ { 1 } ^ { \prime }$ . RA5 or RA9 rewriting rules in Table 3 can be applied. So $p$ is not a normal form;

– Subcase $p _ { 1 } \equiv \phi \cdot p _ { 1 } ^ { \prime }$ . RG1, RG3, RG4, RG5, RG7, or RG8 − 9 rewriting rules can be applied. So p is not a normal form;

– Subcase $p _ { 1 } \equiv p _ { 1 } ^ { \prime } + p _ { 1 } ^ { \prime \prime }$ . RA4, RA6, RG2, or RG6 rewriting rules in Table 3 can be applied. So p is not a normal form;

– Subcase $p _ { 1 } \equiv p _ { 1 } ^ { \prime } \oplus p _ { 1 } ^ { \prime \prime }$ . RP1-RP9 rewrite rules in Table 7 can be applied. So $p$ is not a normal form.

• Case $p ^ { \prime } \equiv p _ { 1 } + p _ { 2 }$ . By induction on the structure of the basic $A P T$ terms both $p _ { 1 }$ and $p _ { 2 }$ , all subcases will lead to that $p ^ { \prime }$ would be a basic $A P T$ term, which contradicts the assumption that $p ^ { \prime }$ is not a basic APT term.

• Case $p ^ { \prime } \equiv p _ { 1 } \oplus p _ { 2 }$ . By induction on the structure of the basic $A P T$ terms both $p _ { 1 }$ and p , all subcases will lead to that $p ^ { \prime }$ would be a basic $A P T$ term, which contradicts the assumption that $p ^ { \prime }$ is not a basic APT term.

## 4.3 Soundness and Completeness

Theorem 4.4 (Generalization of APT with respect to BAT). $A P T$ is a generalization of BAT.   
Proof. It follows from the following three facts.

1. The transition rules of BAT Table 4 are all source-dependent;

2. The sources of the transition rules for APT in Table 5 contain an occurrence of ⊕;

3. The transition rules of APT are all source-dependent.

So, APT is a generalization of BAT, i.e., BAT is an embedding of APT, as desired. □

Theorem 4.5 (Soundness of APT modulo pomset bisimulation equivalence). The axiomatization of APT is sound modulo pomset bisimulation equivalence, i.e., let x and y be APT terms. If $A P T \vdash x = y$ , then $x {  } y$

Proof. Since pomset bisimulations ↔ is both an equivalent and a congruent relation, we only need to check if each axiom in Table 6 is sound modulo pomset bisimulation equivalence ↔. The proof is trivial and we omit it. □

Theorem 4.6 (Completeness of APT modulo pomset bisimulation equivalence). The axiomatization of APT is complete modulo pomset bisimulation equivalence, i.e., let p and q be closed APT terms, if $p {  } q$ then $p = q$

Proof. Firstly, by the elimination theorem of APT (see Theorem 4.3), we know that for each closed APT term $p ,$ there exists a closed basic APT term $p ^ { \prime } .$ such that $A P T \vdash p = p ^ { \prime } .$ , so, we only need to consider closed basic APT terms.

The basic terms (see Definition 4.2) modulo associativity and commutativity (AC) of conflict + (defined by axioms A1 and A2 in Table 2), and these equivalences is denoted by $= A C$ . Then, each equivalence class s modulo AC of + has the following normal form

$$
s _ { 1 } + \cdots + s _ { k }
$$

with each $s _ { i }$ either an atomic event or of the form

$$
t _ { 1 } \circ \cdots \circ t _ { m }
$$

with each $t _ { j }$ either an atomic event or of the form

with each $u _ { \ell }$ an atomic event, and each $s _ { i }$ is called the summand of $s .$

Now, we prove that for normal forms n and $n ^ { \prime } .$ , if $n { \stackrel { } { \mathop {  } } n ^ { \prime } }$ then $n = _ { A C } n ^ { \prime }$ . It is suficient to induct on the sizes of n and $n ^ { \prime }$

• Consider a summand $U$ of n. Then $\langle n , s \rangle \stackrel { U } { \longrightarrow } \langle \surd , s ^ { \prime } \rangle$ , so $n { \stackrel { } { \mathop {  } } n ^ { \prime } }$ implies $\langle n ^ { \prime } , s \rangle \stackrel { U } { \longrightarrow } \langle \surd , s \rangle$ meaning that $n ^ { \prime }$ also contains the summand $U .$

• Consider a summand ϕ of $n .$ . Then $\langle n , s \rangle \to \langle \sqrt { , } s \rangle$ , if $t e s t ( \phi , s )$ holds, so $n { \stackrel { } { \mathop {  } } } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle  \langle \sqrt { , } s \rangle$ , if $t e s t ( \phi , s )$ holds, meaning that $n ^ { \prime }$ also contains the summand $\phi .$

• Consider a summand $t _ { 1 }$ ⋅ t<sub>2</sub> of $n _ { \mathrm { : } }$

– if $t _ { 1 } \equiv \alpha ^ { \prime }$ , then $\langle n , s \rangle \stackrel { \alpha ^ { \prime } } { \longrightarrow } \langle t _ { 2 } , s ^ { \prime } \rangle$ , so n↔n<sup>′</sup> implies $\langle n ^ { \prime } , s \rangle \stackrel { \alpha ^ { \prime } } { \longrightarrow } \langle t _ { 2 } ^ { \prime } , s ^ { \prime } \rangle$ with $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ , meaning that $n ^ { \prime }$ contains a summand $\alpha ^ { \prime } \cdot t _ { 2 } ^ { \prime }$ . Since $t _ { 2 }$ and $t _ { 2 } ^ { \prime }$ are normal forms and have sizes smaller than n and $n ^ { \prime } { \mathrm { . } }$ by the induction hypotheses if $t _ { 2 } { \stackrel {  } { \geq } } t _ { 2 } ^ { \prime }$ then $t _ { 2 } = _ { A C } t _ { 2 } ^ { \prime } ;$

– if $t _ { 1 } \equiv \phi ^ { \prime }$ , then $\langle n , s \rangle \to \langle t _ { 2 } , s \rangle$ , if $t e s t ( \phi ^ { \prime } , s )$ holds, so $n { \stackrel {  } { \Longrightarrow } } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle \to \langle t _ { 2 } ^ { \prime } , s \rangle$ with $t _ { 2 } { \stackrel {  } { \geq } } t _ { 2 } ^ { \prime }$ , if $t e s t ( \phi ^ { \prime } , s )$ holds, meaning that $n ^ { \prime }$ contains a summand $\phi ^ { \prime } \cdot t _ { 2 } ^ { \prime }$ . Since $t _ { 2 }$ and $t _ { 2 } ^ { \prime }$ are normal forms and have sizes smaller than n and $n ^ { \prime }$ , by the induction hypotheses if $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ then $t _ { 2 } = _ { A C } t _ { 2 } ^ { \prime }$ <sub>2</sub>;

– if $t _ { 1 } \equiv \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { e l l }$ , then $\left. n , s \right. \xrightarrow { \left\{ \alpha _ { 1 } , \cdots , \alpha _ { e l l } \right\} } \left. t _ { 2 } , s ^ { \prime } \right.$ , so $n { \stackrel {  } { \longrightarrow } } n ^ { \prime }$ implies $\begin{array} { r } { \langle n ^ { \prime } , s \rangle \xrightarrow { \{ \alpha _ { 1 } , \cdots , \alpha _ { e l l } \} } } \end{array}$ $\langle t _ { 2 } ^ { \prime } , s ^ { \prime } \rangle$ with $t _ { 2 } {  } t _ { 2 } ^ { \prime } .$ , meaning that $n ^ { \prime }$ contains a summand $\left( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { e l l } \right) \cdot t _ { 2 } ^ { \prime }$ . Since $t _ { 2 }$ and $t _ { 2 } ^ { \prime }$ are normal forms and have sizes smaller than n and $n ^ { \prime } { \mathrm { . } }$ , by the induction hypotheses if $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ then $t _ { 2 } = _ { A C } t _ { 2 } ^ { \prime } ;$

– if $t _ { 1 } \equiv \phi _ { 1 } \oplus \cdots \oplus \phi _ { e l l }$ , then $\langle n , s \rangle \to \langle t _ { 2 } , s \rangle$ , if test $( \phi _ { 1 } , s ) , \cdots , t e s t ( \phi _ { e l l } , s )$ hold, so $n { \stackrel {  } { \longrightarrow } } n ^ { \prime }$ implies $\langle n ^ { \prime } , s \rangle \to \langle t _ { 2 } ^ { \prime } , s \rangle$ with $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ , if te ${ \sf 3 } t ( \phi _ { 1 } , s ) , \cdots , t e s t ( \phi _ { e l l } , s )$ hold, meaning that $n ^ { \prime }$ contains a summand $\left( \phi _ { 1 } \oplus \cdots \oplus \phi _ { e l l } \right) \cdot t _ { 2 } ^ { \prime }$ . Since $t _ { 2 }$ and $t _ { 2 } ^ { \prime }$ are normal forms and have sizes smaller than n and $n ^ { \prime }$ , by the induction hypotheses if $t _ { 2 } {  } t _ { 2 } ^ { \prime }$ then $t _ { 2 } = _ { A C } t _ { 2 } ^ { \prime }$

${ \mathrm { S o } } ,$ we get $n = _ { A C } n ^ { \prime } .$

Finally, let s and t be basic APT terms, and $s {  } t ,$ , there are normal forms n and $n ^ { \prime }$ , such that $s = n$ and $t = n ^ { \prime }$ . The soundness theorem of parallelism modulo pomset bisimulation equivalence (see Theorem 4.5) yields $s {  } n$ and $t { \stackrel {  } { \longrightarrow } } n ^ { \prime } .$ , so $n { \stackrel {  } { \_ } } s { \stackrel {  } { \_ } } t { \stackrel {  } { \_ } } n ^ { \prime }$ . Since if $n { \stackrel {  } { \Longrightarrow } } n ^ { \prime }$ then $n = _ { A C } n ^ { \prime } , s =$ $n = _ { A C } n ^ { \prime } = t .$ , as desired. □

## 5 Recursion

In this section, we introduce recursion to capture infinite computations based on APT.   
In the following, E, F, G are recursion specifications, X, Y, Z are recursive variables.

Definition 5.1 (Recursive specification). A recursive specification is a finite set of recursive equations

$$
X _ { 1 } = t _ { 1 } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

$$
X _ { n } = t _ { n } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

where the left-hand sides of $X _ { i }$ are called recursion variables, and the right-hand sides $t _ { i } ( X _ { 1 } , \cdots , X _ { n } )$ are process terms in APT with possible occurrences of the recursion variables $X _ { 1 } , \cdots , X _ { n }$

Definition 5.2 (Solution). Processes $p _ { 1 } , \cdots , p _ { n }$ are solutions for a recursive specification $\left\{ X _ { i } = \right.$ $t _ { i } ( X _ { 1 } ,  &  . . . , X _ { n } ) | i \in \{ 1 , \cdots , n \} \}$ (with respect to pomset bisimulation equivalence ↔) i $f p _ { i } {  } t _ { i } ( p _ { 1 } , \cdots , p _ { n } )$ $f o r \ i \in \{ 1 , \cdots , n \}$

Definition 5.3 (Guarded recursive specification). A recursive specification

$$
X _ { 1 } = t _ { 1 } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

$$
X _ { n } = t _ { n } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

is guarded if the right-hand sides of its recursive equations can be adapted to the form by $a p \cdot$ plications of the axioms in APT and replacing recursion variables by the right-hand sides of their recursive equations, and there does not exist an infinite sequence of I-transitions $\left. X | E \right. \to$ $\langle X ^ { \prime } | E \rangle \to \langle X ^ { \prime \prime } | E \rangle \to \cdots$

$$
U _ { 1 } \circ s _ { 1 } \bigl ( X _ { 1 } , \cdots , X _ { n } \bigr ) + \cdots + U _ { k } \circ s _ { k } \bigl ( X _ { 1 } , \cdots , X _ { n } \bigr ) + V _ { 1 } + \cdots + V _ { \ell }
$$

where $U _ { 1 } , \cdots , U _ { k } , V _ { 1 } , \cdots , V _ { \ell } \ \in$ Pom, and the sum above is allowed to be empty, in which case it represents the deadlock 0.

Definition 5.4 (Linear recursive specification). A recursive specification is linear $i f$ its recursive equations are of the form

$$
U _ { 1 } \circ X _ { 1 } + \cdots + U _ { k } \circ X _ { k } + V _ { 1 } + \cdots + V _ { \ell }
$$

where $U _ { 1 } , \cdots , U _ { k } , V _ { 1 } , \cdots , V _ { \ell } \ \in$ Pom, and the sum above is allowed to be empty, in which case it represents the deadlock 0.

$$
\begin{array} { r l } & { \frac { t _ { i } \left( \left. X _ { 1 } | E \right. , \cdots , \left. X _ { n } | E \right. \right) \stackrel { U } { \longrightarrow } \check { \mathcal { V } } } { \left. X _ { i } | E \right. \stackrel { U } { \longrightarrow } \check { \mathcal { V } } } } \\ & { \frac { t _ { i } \left( \left. X _ { 1 } | E \right. , \cdots , \left. X _ { n } | E \right. \right) \stackrel { U } { \longrightarrow } y } { \left. X _ { i } | E \right. \stackrel { U } { \longrightarrow } y } } \end{array}
$$

Table 8: Transition rules of guarded recursion

For a guarded recursive specifications E with the form

$$
X _ { 1 } = t _ { 1 } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

$$
X _ { n } = t _ { n } ( X _ { 1 } , { \cdots } , X _ { n } )
$$

the behavior of the solution $\langle X _ { i } | E \rangle$ for the recursion variable $X _ { i }$ in $E ,$ where $i \in \{ 1 , \cdots , n \}$ ， is exactly the behavior of their right-hand sides $t _ { i } ( X _ { 1 } , \cdots , X _ { n } )$ , which is captured by the two transition rules in Table 8.

## 5.1 Operational Semantics

Theorem 5.5 (Conservativity of APT with guarded recursion). APT with guarded recursion is a conservative extension of APT.

Proof. Since the transition rules of APT are source-dependent, and the transition rules for guarded recursion in Table 8 contain only a fresh constant in their source, so the transition rules of APT with guarded recursion are conservative extensions of those of APT. □

Theorem 5.6 (Congruence theorem of APT with guarded recursion). Pomset bisimulation equivalence ↔ is a congruences with respect to APT with guarded recursion.

Proof. Since the PTSS of APT with guarded recursion in Table 8 is positive after reduction and in panth format, pomset bisimulation equivalence is a congruence with respect to APT with guarded recursion. □

## 5.2 Axiomatic System

The RDP (Recursive Definition Principle) and the RSP (Recursive Specification Principle) are shown in Table 9.

Table 9: Recursive definition and specification principle
<table><tr><td rowspan=1 colspan=1>No.</td><td rowspan=1 colspan=1>Axioms</td></tr><tr><td rowspan=1 colspan=1>RDPRSP</td><td rowspan=1 colspan=1> $\langle X _ { i } | E \rangle = t _ { i } { \big ( } \langle X _ { 1 } | E \rangle , \cdots , \langle X _ { n } | E \rangle { \big ) } \quad ( i \in \{ 1 , \cdots , n \} { \big ) }$ if $y _ { i } = t _ { i } ( y _ { 1 } , \cdots , y _ { n } ) { \mathrm { ~ f o r ~ } } i \in \{ 1 , \cdots , n \}$ , then $y _ { i } = \langle X _ { i } | E \rangle \quad ( i \in \{ 1 , \cdots , n \} )$ </td></tr><tr><td rowspan=1 colspan=2>end of Recursive definition and specification principle</td></tr></table>

RDP follows immediately from the two transition rules for guarded recursion, which express that $\langle X _ { i } | E \rangle$ and $t _ { i } ( \langle X _ { 1 } | E \rangle , \cdots , \langle X _ { n } | E \rangle )$ have the same initial transitions for $i \in \{ 1 , \cdots , n \}$ . RSP follows from the fact that guarded recursive specifications have only one solution.

## 5.3 Soundness and Completeness

Theorem 5.7 (Soundness of APT with guarded recursion). Let x and $y$ be APT with guarded recursion terms. If APT with guarded recursion $\vdash x = y$ , then $x {  } y$

Proof. Since pomset bisimulation equivalence is both an equivalent and a congruent relation with respect to APT with guarded recursion, we only need to check if each axiom in Table 9 is sound modulo pomset bisimulation equivalence. The proof is trivial and we omit it. □

Theorem 5.8 (Completeness of APT with linear recursion). Let p and q be closed APT with linear recursion terms, then, if p↔q then $p = q$

Proof. Firstly, we know that each process term in APT with linear recursion is equal to a process term $\langle X _ { 1 } | E \rangle$ with E a linear recursive specification.

Let $E _ { 1 }$ consist of recursive equations $X = t _ { X }$ for $X \in { \mathcal { X } }$ and $E _ { 2 }$ consists of recursion equations $Y = t _ { Y }$ for $Y \in \mathcal { V }$ . Let the linear recursive specification E consist of recursion equations $Z _ { X Y } =$ $t _ { X Y }$ , and $\langle X | E _ { 1 } \rangle \sim _ { s } \langle Y | E _ { 2 } \rangle$ , and $t _ { X Y }$ consists of the following summands:

1. $t _ { X Y }$ contains a summand $( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } ) Z _ { X ^ { \prime } Y ^ { \prime } } \operatorname { i f f } t _ { X }$ contains the summand $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) X ^ { \prime }$ and $t _ { Y }$ contains the summand $( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } ) Y ^ { \prime }$ such that $\langle X ^ { \prime } | E _ { 1 } \rangle \sim _ { s } \langle Y ^ { \prime } | E _ { 2 } \rangle$ ;

2. $t _ { X Y }$ contains a summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ if $t _ { X }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ and $t _ { Y }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$

Let σ map recursion variable X in $E _ { 1 }$ to $\langle X | E _ { 1 } \rangle$ , and let $\psi$ map recursion variable $Z _ { X Y }$ in E to $\langle X \vert E _ { 1 } \rangle$ . So, $\sigma ( ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } ) X ^ { \prime } ) \equiv \bigl ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \bigr ) \bigl \langle X ^ { \prime } \big | E _ { 1 } \bigr \rangle \equiv \psi \bigl ( \bigl ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \bigr ) Z _ { X ^ { \prime } Y ^ { \prime } } \bigr )$ , so by $R D P$ we get $\langle X | E _ { 1 } \rangle = \sigma ( t _ { X } ) = \psi ( t _ { X Y } )$ . Then by RSP, $\langle X | E _ { 1 } \rangle = \langle Z _ { X Y } | E \rangle$ , particularly, $\left. X _ { 1 } | E _ { 1 } \right. =$ $\langle Z _ { X _ { 1 } Y _ { 1 } } | E \rangle$ . Similarly, we can obtain $\langle Y _ { 1 } | E _ { 2 } \rangle ~ = ~ \langle Z _ { X _ { 1 } Y _ { 1 } } | E \rangle$ . Finally, $\langle X _ { 1 } | E _ { 1 } \rangle \ = \ \langle Z _ { X _ { 1 } Y _ { 1 } } | E \rangle \ =$ $\left. Y _ { 1 } | E _ { 2 } \right.$ , as desired. □

$$
\overline { { \langle \tau _ { \phi } , s \rangle \to \langle \sqrt { , } s \rangle } } ^ { \mathrm { ~ i f ~ } t e s t ( \tau _ { \phi } , s ) }
$$

$$
\langle \tau , s \rangle \stackrel { \tau } {  } \langle \sqrt { , \tau ( s ) \rangle }
$$

Table 10: Transition rule of the silent step

## 6 Abstraction

To abstract away from the internal implementations of a transformer, and verify that the transformer exhibits the desired external behaviors, the silent step τ and abstraction operator $\tau _ { I }$ are introduced, where $I \subseteq { \mathsf { A c t } } \cup$ Grd denotes the internal events or guards.

## 6.1 Silent Step

The silent step τ represents the internal actions, and $\tau _ { \phi }$ for internal guards, when we consider the external behaviors of a process, τ steps can be removed, i.e., τ steps must keep silent. The transition rule of τ is shown in Table 10. In the following, let the atomic action a range over Act ∪ $\{ \mathbf { 0 } \} \cup \{ \mathbf { I } \} \cup \{ \tau \}$ , and $\phi$ range over $\mathsf { G r d } \cup \{ \tau \}$ . We use $\tau ( s )$ to denote $e f f e c t ( \tau , s )$ , for the fact that τ only change the state of internal data environment, that is, for the external data environments, $s = \tau ( s )$

## 6.1.1 Operational Semantics

Definition 6.1 (Weak pomset transitions). Let $p , q \in$ Proc and $U \in$ Pom, $p \stackrel { U } { \Longrightarrow } q$ is called a weak pomset transition from p to $q ,$ where we define $\stackrel { a } { \Longrightarrow } \triangleq \stackrel { \tau ^ { * } } { \longrightarrow } \stackrel { a } { \longrightarrow } \stackrel { \tau ^ { * } } { \longrightarrow } \textnormal { }$ . And $\stackrel { U } { \Longrightarrow } \triangleq \stackrel { \tau ^ { * } } { \longrightarrow } \stackrel { U } { \longrightarrow } \stackrel { \tau ^ { * } } { \longrightarrow }$ , for every $a \in U$

Definition 6.2 (Branching pomset bisimulation). Assume a special termination predicate ↓, and let $\surd$ represent a state with $\surd \downarrow$ . Let S be a set of states, p, q ∈ Proc, U ∈ Pom and $s , s ^ { \prime } \in S$ A branching pomset bisimulation is a relation $R \subseteq \langle \mathsf { P r o c } , S \rangle \times \langle \mathsf { P r o c } , S \rangle$ , such that:

1. if $( \langle p , s \rangle , \langle q , s \rangle ) \in R$ , and $\langle p , s \rangle \stackrel { U } { \longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ then

• either $U \equiv \tau ^ { * }$ , and $( \langle p ^ { \prime } , s ^ { \prime } \rangle , \langle q , s \rangle ) \in R$ with $s ^ { \prime } \in \tau ( s )$ ;

• or there is a sequence of (zero or more) τ-transitions $\left. q , s \right. \stackrel { \tau ^ { * } } { \longrightarrow } \langle q ^ { 0 } , s ^ { 0 } \rangle$ , such that $( \langle p , s \rangle , \langle q ^ { 0 } , s ^ { 0 } \rangle ) \in R$ and $\langle q ^ { 0 } , s ^ { 0 } \rangle \overset { U } { \Longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ with $( \langle p ^ { \prime } , s ^ { \prime } \rangle , \langle q ^ { \prime } , s ^ { \prime } \rangle ) \in R ;$

2. $i f \left( \langle p , s \rangle , \langle q , s \rangle \right) \in R$ , and $\langle q , s \rangle \stackrel { U } { \longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ then

• either $U \equiv \tau ^ { * }$ , and $( \langle p , s \rangle , \langle q ^ { \prime } , s ^ { \prime } \rangle ) \in R ,$

• or there is a sequence of (zero or more) τ-transitions $\langle p , s \rangle \stackrel { \tau ^ { * } } { \longrightarrow } \langle p ^ { 0 } , s ^ { 0 } \rangle$ , such that $( \langle p ^ { 0 } , s ^ { 0 } \rangle , \langle q , s \rangle ) \in R$ and $\langle p ^ { 0 } , s ^ { 0 } \rangle \overset { U } { \Longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ with $( \langle p ^ { \prime } , s ^ { \prime } \rangle , \langle q ^ { \prime } , s ^ { \prime } \rangle ) \in R ;$

3. if $( \langle p , s \rangle , \langle q , s \rangle ) \in R$ and $\langle p , s \rangle \downarrow$ , then there is a sequence of (zero or more) τ -transitions $\left. q , s \right. \stackrel { \tau ^ { * } } { \longrightarrow } \left. q ^ { 0 } , s ^ { 0 } \right.$ such that $( \langle p , s \rangle , \langle q ^ { 0 } , s ^ { 0 } \rangle ) \in R$ and $\langle q ^ { 0 } , s ^ { 0 } \rangle \downarrow$ ;

4. if $( \langle p , s \rangle , \langle q , s \rangle ) \in R$ and $\langle q , s \rangle$ ↓, then there is a sequence of (zero or more) τ-transitions $\langle p , s \rangle \stackrel { \tau ^ { * } } { \longrightarrow } \langle p ^ { 0 } , s ^ { 0 } \rangle$ such that $( \langle p ^ { 0 } , s ^ { 0 } \rangle , \langle q , s \rangle ) \in R$ and $\langle p ^ { 0 } , s ^ { 0 } \rangle \downarrow$

We say that $p , q$ are branching pomset bisimilar, written $p {  } { _ { b } } q .$ , if there exists a branching pomset bisimulation R, such that $( \langle p , s \rangle , \langle q , s \rangle ) \in R$

Definition 6.3 (Rooted branching pomset, step bisimulation). Assume a special termination predicate ↓, and let represent a state with $\surd \downarrow$ . Let S be a set of states, $p , q \in { \mathsf { P r o c } }$ , U ∈ Pom and $s , s ^ { \prime } \in S$ . A rooted branching pomset bisimulation is a relation $R \subseteq \langle \mathsf { P r o c } , S \rangle \times \langle \mathsf { P r o c } , S \rangle$ , such that:

1. $i f \left( \langle p , s \rangle , \langle q , s \rangle \right) \epsilon R$ , and $\langle p , s \rangle \stackrel { U } { \longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ then $\langle q , s \rangle \stackrel { U } { \longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ with $\langle p ^ { \prime } , s ^ { \prime } \rangle { \stackrel { \left. \right.} { \mathop {  } } } { } _ { b } \langle q ^ { \prime } , s ^ { \prime } \rangle$ ;

2. $i f \left( \langle p , s \rangle , \langle q , s \rangle \right) \epsilon R$ , and $\langle q , s \rangle \stackrel { U } { \longrightarrow } \langle q ^ { \prime } , s ^ { \prime } \rangle$ then $\langle p , s \rangle \stackrel { U } { \longrightarrow } \langle p ^ { \prime } , s ^ { \prime } \rangle$ with $\langle p ^ { \prime } , s ^ { \prime } \rangle { \stackrel { \left. \right.} { \mathop {  } } } { } _ { b } \langle q ^ { \prime } , s ^ { \prime } \rangle$ ;

3. $i f \left( \langle p , s \rangle , \langle q , s \rangle \right) \in R$ and $\langle p , s \rangle \downarrow _ { ; }$ then ⟨q, s⟩ ↓;

4. $i f \left( \langle p , s \rangle , \langle q , s \rangle \right) \in R$ and $\langle q , s \rangle \downarrow$ , then $\langle p , s \rangle \downarrow$

We say that $p , q$ are rooted branching pomset bisimilar, written $p {  } _ { r b } q$ , if there exists a rooted branching pomset bisimulation R, such that $( \langle p , s \rangle , \langle q , s \rangle ) \in R$

The silent step τ as an atomic event, is introduced into E. Considering the recursive specification $X = \tau X , \tau s , \tau \tau s . $ , and $\tau { \cdot } \cdot \cdot s$ are all its solutions, that is, the solutions make the existence of $\tau -$ loops which cause unfairness. To prevent τ-loops, we extend the definition of linear recursive specification (Definition 5.4) to the guarded one.

Definition 6.4 (Guarded linear recursive specification). A recursive specification is linear if its recursive equations are of the form

$$
U _ { 1 } \circ X _ { 1 } + \cdots + U _ { k } \circ X _ { k } + V _ { 1 } + \cdots + V _ { \ell }
$$

where $U _ { 1 } , \cdots , U _ { k } , V _ { 1 } , \cdots , V _ { \ell } \in \mathsf { P o m }$ , and the sum above is allowed to be empty, in which case it represents the deadlock 0.

A linear recursive specification E is guarded if there does not exist an infinite sequence of Itransitions $\langle X | E \rangle  \langle X ^ { \prime } | E \rangle  \langle X ^ { \prime \prime } | E \rangle$ → ⋯ and τ-transitions $\langle X | E \rangle \stackrel { \tau } { \to } \langle X ^ { \prime } | E \rangle \stackrel { \tau } { \to } \langle X ^ { \prime \prime } | E \rangle \stackrel { \tau } { \to } \cdots .$

Theorem 6.5 (Conservativity of APT with silent step and guarded linear recursion). APT with silent step and guarded linear recursion is a conservative extension of $A P T$ with linear recursion.

Proof. Since the transition rules of APT with linear recursion are source-dependent, and the transition rules for silent step in Table 10 contain only a fresh constant τ in their source, so the transition rules of APT with silent step and guarded linear recursion is a conservative extension of those of APT with linear recursion. □

Theorem 6.6 (Congruence theorem of APT with silent step and guarded linear recursion). Rooted branching pomset bisimulation equivalence $\stackrel {  } { \mathop { : } } r b$ is a congruence with respect to APT with silent step and guarded linear recursion.

Proof. Since the PTSS of APT with silent step and guarded linear recursion in Table 10 is positive after reduction and in RBB cool format, rooted branching pomset bisimulation equivalence $\stackrel {  } { \mathop { : - r } } b$ is a congruence with respect to APT with with silent step and guarded linear recursion.

## 6.1.2 Axiomatic System

We design the axioms for the silent step τ in Table 11.

Table 11: Axioms of silent step
<table><tr><td> $\mathrm { N o . }$ </td><td>Axioms</td></tr><tr><td>B1 B2</td><td> $U \circ \tau = U$   $U \circ ( \tau \circ ( x + y ) + x ) = U \circ ( x + y )$ </td></tr><tr><td>B3</td><td> $x \oplus \tau = x$ </td></tr><tr><td>B4</td><td> $\tau \oplus x = x$ </td></tr><tr><td>G26</td><td></td></tr><tr><td>G19</td><td> $\tau _ { \phi } \cdot x = x$ </td></tr><tr><td>G20</td><td> $x \cdot \tau _ { \phi } = x$ </td></tr><tr><td></td><td> $x \parallel \tau _ { \phi } = x$ </td></tr><tr><td>G21</td><td> $\tau _ { \phi } \parallel x = x$ </td></tr></table>

end of Axioms of silent step

## 6.1.3 Soundness and Completeness

Theorem 6.7 (Soundness of APT with silent step and guarded linear recursion). Let x and y be APT with silent step and guarded linear recursion terms. If APT with silent step and guarded linear recursion $\vdash x = y$ , then $x {  } _ { r b } y$

Proof. Since rooted branching pomset bisimulation $\stackrel {  } { \mathop { : } } r b$ is both an equivalent and a congruent relation with respect to APT with silent step and guarded linear recursion, we only need to check if each axiom in Table 11 is sound modulo rooted branching pomset bisimulation equivalence. The proof is trivial and we omit it. □

Theorem 6.8 (Completeness of APT with silent step and guarded linear recursion). Let p and q be closed $A P T$ with silent step and guarded linear recursion terms, then, $i f p {  } { } _ { r b } q$ then $p = q$

Proof. Firstly, we know that each process term in APT with silent step and guarded linear recursion is equal to a process term $\langle X _ { 1 } | E \rangle$ with E a guarded linear recursive specification.

The recursive equation $W = \tau + \cdots + \tau$ with $W \not \equiv X _ { 1 }$ in $E _ { 1 }$ and $E _ { 2 }$ , can be removed, and the corresponding summands aW are replaced by α, to get $E _ { 1 } ^ { \prime }$ and $E _ { 2 } ^ { \prime }$ , by use of the axioms $R D P$ A3 and B1, and $\langle X | E _ { 1 } \rangle = \langle X | E _ { 1 } ^ { \prime } \rangle , \langle Y | E _ { 2 } \rangle = \langle Y | E _ { 2 } ^ { \prime } \rangle$

Let $E _ { 1 }$ consists of recursive equations $X = t _ { X }$ for $X \in \mathcal { X }$ and $E _ { 2 }$ consists of recursion equations $Y = t _ { Y }$ for $Y \in \mathcal { V }$ , and are not the form $\tau + \cdots + \tau$ . Let the guarded linear recursive specification E consists of recursion equations $Z _ { X Y } = t _ { X Y }$ , and $\langle X | E _ { 1 } \rangle \approx _ { r b s } \langle Y | E _ { 2 } \rangle$ , and $t _ { X Y }$ consists of the following summands:

1. $t _ { X Y }$ contains a summand ${ \left( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \right) } Z _ { X ^ { \prime } Y ^ { \prime } }$ if $t _ { X }$ contains the summand $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) X ^ { \prime }$ and $t _ { Y }$ contains the summand $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) Y ^ { \prime }$ such that $\left. X ^ { \prime } | E _ { 1 } \right. \approx _ { r b s } \left. Y ^ { \prime } | E _ { 2 } \right.$ ;

2. $t _ { X Y }$ contains a summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ if $t _ { X }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ and $t _ { Y }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n } ;$

3. $t _ { X Y }$ contains a summand $\tau Z _ { X ^ { \prime } Y }$ if $X Y \not \equiv X _ { 1 } Y _ { 1 } , t _ { X }$ contains the summand $\tau X ^ { \prime }$ , and $\left. X ^ { \prime } | E _ { 1 } \right. \approx _ { r b s } \left. Y | E _ { 2 } \right.$ ;

4. $t _ { X Y }$ contains a summand $\tau Z _ { X Y ^ { \prime } }$ if $X Y \ \not \equiv \ X _ { 1 } Y _ { 1 }$ , t<sub>Y</sub> contains the summand $\tau Y ^ { \prime }$ , and $\langle X | E _ { 1 } \rangle \approx _ { r b s } \left. Y ^ { \prime } | E _ { 2 } \right.$

Since $E _ { 1 }$ and $E _ { 2 }$ are guarded, E is guarded. Constructing the process term $u X Y$ consist of the following summands:

1. $u _ { X Y }$ contains a summand ${ \bigl ( } \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } { \bigr ) } \langle X ^ { \prime } | E _ { 1 } \rangle$ if $t _ { X }$ contains the summand $( \alpha _ { 1 } \oplus \cdots \oplus$ $\alpha _ { m } ) X ^ { \prime }$ and $t _ { Y }$ contains the summand $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) Y ^ { \prime }$ such that $\left. X ^ { \prime } | E _ { 1 } \right. \approx _ { r b s } \left. Y ^ { \prime } | E _ { 2 } \right.$ ;

2. $u _ { X Y }$ contains a summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ if $t _ { X }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n }$ and $t _ { Y }$ contains the summand $\beta _ { 1 } \oplus \cdots \oplus \beta _ { n } ;$

3. $u _ { X Y }$ contains a summand $\tau \langle X ^ { \prime } | E _ { 1 } \rangle$ if $X Y \not \equiv X _ { 1 } Y _ { 1 } , t _ { X }$ contains the summand $\tau X ^ { \prime }$ , and $\left. X ^ { \prime } | E _ { 1 } \right. \approx _ { r b s } \left. Y | E _ { 2 } \right.$

Let the process term $s _ { X Y }$ be defined as follows:

1. $s _ { X Y } \triangleq \tau \langle X \vert E _ { 1 } \rangle + u _ { X Y }$ if $X Y \not \equiv X _ { 1 } Y _ { 1 } , t _ { Y }$ contains the summand $\tau Y ^ { \prime }$ , and $\left. X | E _ { 1 } \right. \approx _ { r b s }$ $\langle Y ^ { \prime } | E _ { 2 } \rangle$

2. $s _ { X Y } \triangleq \langle X | E _ { 1 } \rangle$ , otherwise.

So, $\begin{array} { r } { \langle X | E _ { 1 } \rangle = \langle X | E _ { 1 } \rangle + u _ { X Y } , \mathrm { ~ a n d ~ } \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) \big ( \tau \langle X | E _ { 1 } \rangle + u _ { X Y } \big ) = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) \big ( ( \tau \langle X | E _ { 1 } \rangle + u _ { X Y } ) \big ) \quad , } \end{array}$ $\begin{array} { r } { u _ { X Y } \Big ) + u _ { X Y } \Big ) = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) \big ( \langle X | E _ { 1 } \rangle + u _ { X Y } \big ) = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \big ) \big \langle X | E _ { 1 } \big \rangle + u _ { X Y } \big ) . } \end{array}$ , hence, $\bigl ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } \bigr ) s _ { X Y } =$ $( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { m } ) \langle X | E _ { 1 } \rangle$

Let σ map recursion variable X in $E _ { 1 }$ to $\langle X | E _ { 1 } \rangle$ , and let $\psi$ map recursion variable $Z _ { X Y }$ in E to $s _ { X Y }$ . It is suficient to prove $s _ { X Y } = \psi ( t _ { X Y } )$ for recursion variables $Z _ { X Y }$ in $E$ . Either $X Y \equiv X _ { 1 } Y _ { 1 }$ or $X Y \not \equiv X _ { 1 } Y _ { 1 }$ , we all can get $s _ { X Y } = \psi ( t _ { X Y } )$ . So, $s _ { X Y } = \left. Z _ { X Y } | E \right.$ for recursive variables $Z _ { X Y }$ in E is a solution for E. Then by RSP, particularly, $\langle X _ { 1 } | E _ { 1 } \rangle = \left. Z _ { X _ { 1 } Y _ { 1 } } | E \right.$ . Similarly, we can obtain $\langle Y _ { 1 } | E _ { 2 } \rangle = \langle Z _ { X _ { 1 } Y _ { 1 } } | E \rangle$ . Finally, $\langle X _ { 1 } | E _ { 1 } \rangle = \langle Z _ { X _ { 1 } Y _ { 1 } } | E \rangle = \langle Y _ { 1 } | E _ { 2 } \rangle$ , as desired. □

## 6.2 Abstraction

The unary abstraction operator $\tau _ { I } ~ ( I \subseteq \mathsf { A c t } )$ renames all atomic events in I into τ. APT with silent step and abstraction operator is called $A P T _ { \tau }$

$$
{ \frac { \langle x , s \rangle \stackrel { \alpha } { \to } \langle \sqrt { , s ^ { \prime } } \rangle } { \langle \tau _ { I } ( x ) , s \rangle \stackrel { \alpha } { \to } \langle \sqrt { , s ^ { \prime } } \rangle } } \quad \alpha \not \in I \qquad { \frac { \langle x , s \rangle \stackrel { \alpha } { \to } \langle x ^ { \prime } , s ^ { \prime } \rangle } { \langle \tau _ { I } ( x ) , s \rangle \stackrel { \alpha } { \to } \langle \tau _ { I } ( x ^ { \prime } ) , s ^ { \prime } \rangle } } \quad \alpha \not \in I
$$

$$
{ \frac { \langle x , s \rangle \stackrel { \alpha } { \to } \langle { \sqrt { , } } s ^ { \prime } \rangle } { \langle \tau _ { I } ( x ) , s \rangle \stackrel { \tau } { \to } \langle { \sqrt { , } } \tau ( s ) \rangle } } \quad \alpha \in I \qquad { \frac { \langle x , s \rangle \stackrel { \alpha } { \to } \langle x ^ { \prime } , s ^ { \prime } \rangle } { \langle \tau _ { I } ( x ) , s \rangle \stackrel { \tau } { \to } \langle \tau _ { I } ( x ^ { \prime } ) , \tau ( s ) \rangle } } \quad \alpha \in I
$$

Table 12: Transition rule of the abstraction operator

## 6.2.1 Operational Semantics

The transition rules of operator $\tau _ { I }$ are shown in Table 12.

Theorem 6.9 (Conservativity of $A P T _ { \tau }$ with guarded linear recursion). $A P T _ { \tau }$ with guarded linear recursion is a conservative extension of $A P T$ with silent step and guarded linear recursion.

Proof. Since the transition rules of APT with silent step and guarded linear recursion are sourcedependent, and the transition rules for abstraction operator in Table 12 contain only a fresh operator $\tau _ { I }$ in their source, so the transition rules of $A P T _ { \tau }$ with guarded linear recursion is a conservative extension of those of APT with silent step and guarded linear recursion. □

Theorem 6.10 (Congruence theorem of $A P T _ { \tau }$ with guarded linear recursion). Rooted branching pomset bisimulation equivalence $\stackrel {  } { \mathop { : } } r b$ is a congruence with respect to $A P T _ { \tau }$ with guarded linear recursion.

Proof. Since the PTSS of $A P T _ { \tau }$ with guarded linear recursion in Table 12 is positive after reduction and in RBB cool format, rooted branching pomset bisimulation equivalence $\stackrel {  } { \mathop { : } } r b$ is a congruence with respect to $A P T _ { \tau }$ with guarded linear recursion. □

## 6.2.2 Axiomatic System

We design the axioms for the abstraction operator $\tau _ { I }$ in Table 13.

Table 13: Axioms of abstraction
<table><tr><td> $\mathrm { N o . }$ </td><td>Axioms</td></tr><tr><td>TI1</td><td> $\begin{array} { r l } { \alpha \not \in I } & { { } \tau _ { I } ( \alpha ) = \alpha } \end{array}$ </td></tr><tr><td>TI2</td><td> $\begin{array} { r l } { \alpha \in I } & { { } \tau _ { I } ( \alpha ) = \tau } \end{array}$ </td></tr><tr><td>TI3</td><td> ${ \boldsymbol \tau } _ { I } ( \mathbf { 0 } ) = \mathbf { 0 }$ </td></tr><tr><td>TI4</td><td> $\tau _ { I } ( \mathbf { I } ) = \mathbf { I }$ </td></tr><tr><td>TI5</td><td> $\tau _ { I } ( x + y ) = \tau _ { I } ( x ) + \tau _ { I } ( y )$ </td></tr><tr><td>TI6</td><td> $\tau _ { I } ( x \circ y ) = \tau _ { I } ( x ) \circ \tau _ { I } ( y )$ </td></tr><tr><td>TI7</td><td> $\tau _ { I } ( x \oplus y ) = \tau _ { I } ( x ) \oplus \tau _ { I } ( y )$ </td></tr><tr><td>G22</td><td>φ∉I τ1(φ) = φ</td></tr><tr><td colspan="2"></td></tr></table>

continuing: Axioms of abstraction
<table><tr><td> $\mathrm { N o . }$ </td><td>Axioms</td></tr><tr><td>G23</td><td> $\phi \in I$   $\tau _ { I } ( \phi ) = \tau _ { \phi }$ </td></tr></table>

end of Axioms of abstraction

## 6.2.3 Soundness and Completeness

Theorem 6.11 (Soundness of APT with guarded linear recursion). Let x and y be $A P T _ { \tau }$ with guarded linear recursion terms. If $A P T _ { \tau }$ with guarded linear recursion $\vdash x = y$ , then $x {  } _ { r b } y$

Proof. Since rooted branching pomset bisimulation $\stackrel {  } { \mathop { - r } } b$ is both an equivalent and a congruent relation with respect to $A P T _ { \tau }$ with guarded linear recursion, we only need to check if each axiom in Table 13 is sound modulo rooted branching pomset bisimulation equivalence. The proof is trivial and we omit it. □

Though τ -loops are prohibited in guarded linear recursive specifications (see Definition 6.4) in a specifiable way, they can be constructed using the abstraction operator, for example, there exist τ-loops in the process term $\tau _ { \{ \alpha \} } ( \langle X | X = a X \rangle )$ . To avoid τ -loops caused by $\tau _ { I }$ and ensure fairness, the concept of cluster and CF AR (Cluster Fair Abstraction Rule) [7] are still valid, we introduce them below.

Definition 6.12 (Cluster). Let E be a guarded linear recursive specification, and $I \subseteq { \mathsf { A c t } }$ . Two recursion variable X and Y in E are in the same cluster for I if there exist sequences of transitions $\begin{array} { r } { \langle X | E \rangle \xrightarrow { \{ \beta _ { 1 1 } , \cdots , \beta _ { 1 i } \} } \cdots \xrightarrow { \{ \beta _ { m 1 } , \cdots , \beta _ { m i } \} } \langle Y | E \rangle } \end{array}$ and $\langle Y | E \rangle \xrightarrow  \{ \} \gamma _ { 1 1 } , \cdots , \gamma _ { 1 j } \} \cdots \xrightarrow  \{ \} \gamma _ { n 1 } , \cdots , \gamma _ { n j } \} \langle X | E \rangle$ where $\beta _ { 1 1 } , \cdots , \beta _ { m i } , \gamma _ { 1 1 } , \cdots , \gamma _ { n j } \in I \cup \{ \tau \}$

α<sub>1</sub> ⊕⋯⊕α<sub>k</sub> or $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { k } \big ) X$ is an exit for the cluster $C \ i f f \colon ( 1 ) \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { k } \ o r \ ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { k } ) X$ is a summand at the right-hand side of the recursive equation for a recursion variable in $C ,$ and (2) in the case of $\big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { k } \big ) X$ , either $\alpha _ { \ell } \not \in I \cup \{ \tau \} ( \ell \in \{ 1 , 2 , \cdots , k \} )$ or $X \notin C$

Table 14: Cluster fair abstraction rule
<table><tr><td> $\mathrm { N o } .$ </td><td>Axioms</td></tr><tr><td>CFAR</td><td>If X is in a cluster for I with exits</td></tr><tr><td></td><td> $\{ ( \alpha _ { 1 1 } \oplus \cdots \oplus \alpha _ { 1 i } ) Y _ { 1 } , \cdots , ( \alpha _ { m 1 } \oplus \cdots \oplus \alpha _ { m i } ) Y _ { m } , \beta _ { 1 1 } \oplus \cdots \oplus \beta _ { 1 j } , \cdots , \beta _ { n 1 } \oplus \cdots \oplus \beta _ { n j } \} .$ </td></tr><tr><td></td><td> $\mathrm { t h e n } \tau \circ \tau _ { I } ( \langle X | E \rangle ) =$   ${ \textrm { 7 } } \circ \tau _ { I } { \big ( } ( \alpha _ { 1 1 } \oplus \cdots \oplus \alpha _ { 1 i } ) \langle Y _ { 1 } | E \rangle + \cdots + ( \alpha _ { m 1 } \oplus \cdots \oplus \alpha _ { m i } ) \langle Y _ { m } | E \rangle + \beta _ { 1 1 } \oplus \cdots \oplus \beta _ { 1 j } + \cdots + \beta _ { n 1 } \oplus \cdots \oplus \beta _ { n j } { \big ) }$ </td></tr></table>

end of Cluster fair abstraction rule

Theorem 6.13 (Soundness of CF AR). CF AR is sound modulo rooted branching pomset bisimulation equivalence $\stackrel {  } { \mathop { : - r } { b } }$

Proof. Let X be in a cluster for I with exits $\{ ( \alpha _ { 1 1 } \oplus \cdots \oplus \alpha _ { 1 i } ) Y _ { 1 } , \cdots , ( \alpha _ { m 1 } \oplus \cdots \oplus \alpha _ { m i } ) Y _ { m } , \beta _ { 1 1 }$ ⊕ $\cdots \oplus \beta _ { 1 j } , \cdots , \beta _ { n 1 } \oplus \cdots \oplus \beta _ { n j } \}$ . Then $\langle X \vert E \rangle$ can execute a string of atomic events from $I \cup \{ \tau \}$ inside the cluster of $X$ , followed by an exit $( \alpha _ { i ^ { \prime } 1 } \oplus \cdots \oplus \alpha _ { i ^ { \prime } i } ) Y _ { i ^ { \prime } }$ for $i ^ { \prime } \in \{ 1 , \cdots , m \}$ or $\beta _ { j ^ { \prime } 1 } \oplus \cdots \oplus \beta _ { j ^ { \prime } j }$ for $j ^ { \prime } \in \{ 1 , \cdots , n \}$ . Hence, $\tau _ { I } ( \langle X | E \rangle )$ can execute a string of $\tau ^ { * }$ inside the cluster of $X$ , followed by an exit $\tau _ { I } ( ( \alpha _ { i ^ { \prime } 1 } \oplus \cdots \oplus \alpha _ { i ^ { \prime } i } ) \langle Y _ { i ^ { \prime } } | E \rangle )$ for $i ^ { \prime } \in \{ 1 , \cdots , m \}$ or $\tau _ { I } ( \beta _ { j ^ { \prime } 1 } \oplus \cdots \oplus \beta _ { j ^ { \prime } j } )$ for $j ^ { \prime } \in \{ 1 , \cdots , n \}$ And these $\tau ^ { * }$ are non-initial in $\tau \tau _ { I } ( \langle X | E \rangle )$ , so they are truly silent by the axiom B1, we obtain $\tau \tau ( \langle X | E \rangle ) \approx _ { \tau b s } \tau \circ \tau _ { T } ( ( \alpha _ { 1 1 } \oplus \cdots \oplus \alpha _ { 1 i } ) \langle Y _ { 1 } | E \rangle + \cdots + ( \alpha _ { m 1 } \oplus \cdots \oplus \alpha _ { m i } ) \langle Y _ { m } | E \rangle + \beta _ { 1 1 } \oplus \cdots \oplus \beta _ { 1 j } + \cdots + ( \alpha _ { m 1 } \oplus \cdots \oplus \alpha _ { m i } ) \langle Y _ { m } | E \rangle )$ $\beta _ { n 1 } \oplus \cdots \oplus \beta _ { n j } )$ , as desired. □

Theorem 6.14 (Completeness of APT<sub>τ</sub> with guarded linear recursion and CFAR). Let p and q be closed $A P T _ { \tau }$ with guarded linear recursion and CF AR terms, then, $i f p {  } { } _ { r b } q$ then $p = q$

Proof. Firstly, in the proof the Theorem 6.8, we know that each process term p in APT with silent step and guarded linear recursion is equal to a process term $\langle X _ { 1 } | E \rangle$ with E a guarded linear recursive specification. And we prove if $\left. X _ { 1 } | E _ { 1 } \right. \approx _ { r b s } \left. Y _ { 1 } | E _ { 2 } \right.$ , then $\left. X _ { 1 } | E _ { 1 } \right. = \left. Y _ { 1 } | E _ { 2 } \right.$

The only new case is $p \equiv \tau _ { I } ( q )$ . Let $q = \left. X | E \right.$ with E a guarded linear recursive specification, so $p = \tau _ { I } ( \langle X | E \rangle )$ . Then the collection of recursive variables in E can be divided into its clusters $C _ { 1 } , \cdots , C _ { N }$ for I. Let

$$
( \alpha _ { 1 i } \oplus \cdots \oplus \alpha _ { k _ { \mathrm { i } i } 1 } ) Y _ { i 1 } + \cdots + ( \alpha _ { 1 i m _ { i } } \oplus \cdots \oplus \alpha _ { k _ { i m _ { i } } i m _ { i } } ) Y _ { i m _ { i } } + \beta _ { 1 i 1 } \oplus \cdots \oplus \beta _ { \ell _ { i 1 } i 1 } + \cdots + \beta _ { 1 i m _ { i } } \oplus \cdots \oplus \beta _ { \ell _ { i m _ { i } } i m _ { i } }
$$

be the conflict composition of exits for the cluster $C _ { i } ,$ with $i \in \{ 1 , \cdots , N \}$

For $Z \in C _ { i }$ with $i \in \{ 1 , \cdots , N \}$ , we define

$$
\begin{array} { r l } & { s _ { Z } \triangleq \big ( \alpha _ { \hat { 1 } i _ { 1 } } \oplus \dots \oplus \alpha _ { k _ { i _ { 1 } i } ^ { - 1 } 1 } \big ) \tau _ { I } \big ( \langle Y _ { i _ { 1 } } | E \rangle \big ) + \cdots + \big ( \alpha _ { \hat { 1 } i m _ { i } } \oplus \dots \oplus \alpha _ { k _ { i m _ { i } } i m _ { i } } \big ) \tau _ { I } \big ( \langle Y _ { i m _ { i } } | E \rangle \big ) + \beta _ { \hat { 1 } i 1 } ^ { \circ } \oplus \dots \oplus \beta _ { \ell _ { i 1 } i _ { 1 } } + } \\ & { \cdots + \beta _ { \hat { 1 } i m _ { i } } \oplus \cdots \oplus \beta _ { \ell _ { i m _ { i } } i m _ { i } } } \end{array}
$$

For $Z \in C _ { i }$ and $\alpha _ { 1 } , \cdots , \alpha _ { j } \in \mathsf { A c t \cup \{ \tau \} }$ with $j \in \mathbb N$ , we have

$$
\begin{array} { r l } & { \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } \big ) \tau _ { I } \big ( \langle Z | E \rangle \big ) } \\ & { = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } \big ) \tau _ { I } \big ( \big ( \alpha _ { 1 i 1 } \oplus \cdots \oplus \alpha _ { k _ { i 1 } i 1 } \big ) \langle Y _ { i 1 } | E \rangle + \cdots + \big ( \alpha _ { 1 i m _ { i } } \oplus \cdots \oplus \alpha _ { k _ { i m _ { i } i } m _ { i } } \big ) \langle Y _ { i m _ { i } } | E \rangle + \beta _ { 1 i 1 } \oplus \cdots \oplus } \\ & { \beta _ { \ell _ { i 1 } i 1 } + \cdots + \beta _ { 1 i m _ { i } } \oplus \cdots \oplus \beta _ { \ell _ { i m _ { i } } i m _ { i } } \big ) } \end{array}
$$

$$
\mathbf { \alpha } = ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } ) s _ { Z }
$$

Let the linear recursive specification F contain the same recursive variables as E, for $Z \in C _ { i } , F$ contains the following recursive equation

$$
Z = \bigl ( \alpha _ { \hat { 1 } i 1 } \oplus \cdots \oplus \alpha _ { \hat { k _ { i 1 } } i } \bigr ) Y _ { i 1 } + \cdots + \bigl ( \alpha _ { \hat { 1 } i m _ { i } } \oplus \cdots \oplus \alpha _ { { k _ { i m _ { i } } } i m _ { i } } \bigr ) Y _ { i m _ { i } } + \beta _ { \hat { 1 } i 1 } ^ { \circ } \oplus \cdots \oplus \beta _ { \ell _ { i 1 } i 1 } ^ { \circ } + \cdots + \beta _ { \hat { 1 } i m _ { i } } ^ { \circ } \oplus \cdots \oplus \beta _ { \ell _ { i m _ { i } } i m _ { i } } ,
$$

It is easy to see that there is no sequence of one or more τ -transitions from $\langle Z \vert F \rangle$ to itself, so $F$ is guarded.

For

$$
\begin{array} { r l } & { s _ { Z } = \big ( \alpha _ { \hat { 1 } i 1 } \oplus \cdots \oplus \alpha _ { k _ { i 1 } i 1 } \big ) Y _ { i 1 } + \cdots + \big ( \alpha _ { \hat { 1 } i m _ { i } } \oplus \cdots \oplus \alpha _ { k _ { i m _ { i } } i m _ { i } } \big ) Y _ { i m _ { i } } + \beta _ { \hat { 1 } i 1 } ^ { \setminus } \oplus \cdots \oplus \beta _ { \hat { \ell } _ { i 1 i } ^ { \setminus } 1 } + \cdots + \beta _ { \hat { 1 } i m _ { i } } \oplus \cdots \oplus \beta _ { \hat { \ell } _ { i m _ { i } } i m _ { i } } , } \\ & { \mathrm { i s ~ a ~ s o l u t i o n ~ f o r } \ F . \ \mathrm { S o } , \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } \big ) \tau _ { I } \big ( \langle Z | E \rangle \big ) = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } \big ) s _ { Z } = \big ( \alpha _ { 1 } \oplus \cdots \oplus \alpha _ { j } \big ) \langle Z | F \rangle . } \\ & { \mathrm { S o } , } \end{array}
$$

$$
\begin{array} { r l } & { \langle Z | F \rangle = \bigl ( \alpha _ { \hat { 1 } i 1 } \oplus \cdots \oplus \alpha _ { \hat { k } _ { i 1 } i 1 } \bigr ) \langle Y _ { i 1 } | F \rangle + \cdots + \bigl ( \alpha _ { 1 \hat { 1 } m _ { i } } \oplus \cdots \oplus \alpha _ { { k } _ { i m _ { i } } i m _ { i } } \bigr ) \langle Y _ { i m _ { i } } | F \rangle + \beta _ { 1 i 1 } ^ { \circ } \oplus \cdots \oplus \beta _ { \ell _ { i 1 } i 1 } ^ { \circ } + \cdots + } \\ & { \beta _ { 1 \hat { m } m _ { i } } \oplus \cdots \oplus \beta _ { \ell _ { i m _ { i } } \hat { m } m _ { i } } } \end{array}
$$

Hence, $\tau _ { I } ( \langle X | E \rangle = \langle Z | F \rangle )$ , as desired.

## 7 Application

This chapter applies APT to three Transformer constructions from The Transformer Cookbook [5]: Most-Recent Induction, Dyck-1 recognition, and Dyck-1-2 recognition. Each application first describes the construction and establishes its functional specification, then represents its components using the existing atomic constructions, and finally derives its external input-output behavior from recursive state-transition equations by abstraction. Functional correctness is established from the concrete computation before internal actions are hidden.

## 7.1 Verification of a Transformer for Most-Recent Induction

The subsection apply APT to the Most-Recent Induction construction presented in the Transformer Cookbook[5]. At each position, the construction predicts the symbol following the most recent earlier occurrence of the current symbol. If no earlier occurrence exists, it predicts the current symbol.

We first describe the construction and its functional specification. We then represent initialization, predecessor retrieval, induction retrieval, and unembedding using existing APT atomic constructions. Finally, we connect these components through recursive process equations and derive the expected external input-output behavior by abstraction.

## 7.1.1 A Transformer Construction of the Most-Recent Induction

Let Σ be a finite nonempty alphabet, and let

$$
\begin{array} { r } { x = x _ { 1 } { \cdots } x _ { n } \in \Sigma ^ { + } , \qquad n = | x | , } \end{array}
$$

be a nonempty input sequence, where $x _ { i }$ denotes the symbol at position $i ,$ where $1 \leq i \leq n$ . Each symbol $\sigma \in \Sigma$ is represented by its one-hot encoding $e _ { \sigma }$

$$
\mathrm { W E } ( x ) = ( e _ { x _ { 1 } } , \ldots , e _ { x _ { n } } ) .
$$

For the example ,

$$
\mathrm { W E } \big ( \mathtt { A C A B D A C D C A } \big ) = \big ( e _ { A } , e _ { C } , e _ { A } , e _ { B } , e _ { D } , e _ { A } , e _ { C } , e _ { D } , e _ { C } , e _ { A } \big ) .
$$

For every symbol, use the predecessor construction to retrieve the embedding of the predecessor symbol (returning 0 when none exists).

$$
\operatorname { P r e d } ( x _ { i } ) = { \left\{ \begin{array} { l l } { 0 , } & { i = 1 , } \\ { e _ { x _ { i - 1 } } , } & { 2 \leq i \leq n . } \end{array} \right. }
$$

Thus,

$$
\operatorname* { P r e d } \bigl ( \mathtt { A C A B D A C D C A } \bigr ) = \bigl ( 0 , e _ { A } , e _ { C } , e _ { A } , e _ { B } , e _ { D } , e _ { A } , e _ { C } , e _ { D } , e _ { C } \bigr ) .
$$

At position $i ,$ attention compares $e _ { x _ { i } }$ with the predecessor embeddings at candidate positions $j \leq i .$ . A position $j$ matches when

$$
\operatorname { P r e d } ( x _ { j } ) = e _ { x _ { i } } .
$$

For such a position, $x _ { j - 1 } = x _ { i } ,$ and the value e represents the symbol following that earlier $e _ { x _ { j } }$ occurrence. Rightmost selection chooses the largest matching position index and returns its value:

$$
\operatorname { R e t r i e v e } ( x ) _ { i } = { \left\{ \begin{array} { l l } { e _ { x _ { j } } , } & { { \mathrm { i f ~ t h e r e ~ e x i s t s ~ } } j { \mathrm { ~ w i t h ~ } } 2 \leq j \leq i { \mathrm { ~ a n d ~ } } x _ { j - 1 } = x _ { i } , } \\ { e _ { x _ { i } } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Here, i is the current query position, and j denotes the largest index satisfying $2 \leq j \leq i$ and $x _ { j - 1 } = x _ { i }$ , whenever such an index exists. The largest candidate index corresponds to the most recent earlier occurrence.

For the running example,

$$
\mathrm { R e t r i e v e } ( \mathtt { A C A B D A C D C A } ) = ( e _ { A } , e _ { C } , e _ { C } , e _ { B } , e _ { D } , e _ { B } , e _ { A } , e _ { A } , e _ { D } , e _ { C } ) .
$$

Unembedding decodes the retrieved embeddings:

$$
\mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x _ { i } ) = \left\{ \begin{array} { l l } { x _ { j } , } & { \mathrm { i f ~ a ~ m a t c h i n g ~ c a n d i d a t e ~ e x i s t s , } } \\ { x _ { i } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

with j defined above.

Thus,

$$
\mathrm { I n d u c t _ { r i g h t m o s t } ( A C A B D A C D C A ) = A C C B D B A A D C . }
$$

## 7.1.2 Algebraic Representation

Input interface. The action

$$
R ( x ) , \quad x \in \Sigma ^ { + }
$$

receives the input sequence $x = x _ { 1 } \cdot x _ { n }$ and stores it. It performs no Transformer computation.   
Sequential composition transfers control to initialization.

The input and output actions describe interaction with the environment; they are not additional Transformer computational primitives.

Embedding. The initialization component is represented by

$$
\left( \alpha _ { 1 } \oplus \alpha _ { 2 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 }$ is the word-embedding action. It supplies the one-hot current-symbol embeddings, computes $\mathrm { W E } ( x _ { i } ) = e _ { x } $ at each position.

$\alpha _ { 2 }$ is the position-embedding action. It supplies $i / n$ for predecessor retrieval in the first self-attention component and rightmost selection in the second self-attention component.

$\alpha _ { 5 }$ is the addition action, which merges the outputs of $\alpha _ { 1 }$ and $\alpha _ { 2 }$ coordinatewise.

The two embedding actions execute in parallel, followed by their coordinatewise addition.

First Transformer layer. The first transformer component uses the predecessor construction to retrieve the embedding of the immediately preceding symbol. It is represented by

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 6 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 2 }$ is the sequence identity action. It preserves the current-symbol embeddings and positional information.

$\alpha _ { 1 6 }$ is the predecessor action. It retrieves $\operatorname { P r e d } ( x _ { i } )$ and routes the result to coordinates separate from the preserved embeddings.

$\alpha _ { 5 }$ combines the preserved information and predecessor embeddings.

The parallel composition preserves the current information while computing the predecessor embedding. The subsequent addition retains both results for the second Transformer layer.

Second Transformer layer. The second Transformer layer retrieves the symbol following the most recent earlier occurrence of the current symbol. Its self-attention component and residual connection are represented by

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 2 }$ is the sequence identity action. It preserves the information produced by the first Transformer layer.

$\alpha _ { 1 8 }$ is the multi-head action, configured with one efective retrieval head. It uses $\operatorname { W E } ( x _ { i } )$ as the query content, $\operatorname { P r e d } ( x _ { j } )$ as the key content, and $\operatorname { W E } ( x _ { j } )$ as the value at each candidate position $j \leq i .$ Content matching and rightmost selection produce $\mathrm { R e t r i e v e } ( x _ { i } )$ If no match exists, the configuration returns $\operatorname { W E } ( x _ { i } )$

$\alpha _ { 5 }$ adds the attention output to the preserved input, retaining the retrieved embedding for unembedding.

The second-layer FFN is configured as zero because the attention component already produces the required prediction embedding. This inactive FFN is omitted from the process representation.

Unembedding. The unembedding action

$$
\alpha _ { 1 9 }
$$

decodes each retrieved one-hot embedding Retrieve $( x _ { i } )$ into its corresponding symbol. Applied at every position, it produces the prediction sequence

$$
{ \mathrm { I n d u c t } } _ { \mathrm { r i g h t m o s t } } ( x ) .
$$

Output interface. The output action reads and emits the computed prediction sequence. Its occurrence is therefore labelled

$$
W \big ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \big )
$$

## 7.1.3 Verification of the Most-Recent Induction Transformer

We now verify that the Most-Recent Induction Transformer exhibits the expected external behavior: it receives an input sequence, computes its Most-Recent Induction predictions, emits the result, and returns to the input-waiting process.

State-transition equations. Using the component representations in Section 7.1.2, define

$$
\begin{array} { r l } & { T = R ( x ) \circ T _ { 2 } , } \\ & { T _ { 2 } = \left( \left( \alpha _ { 1 } \oplus \alpha _ { 2 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 3 } , } \\ & { T _ { 3 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 6 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 4 } , } \\ & { T _ { 4 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 5 } , } \\ & { T _ { 5 } = \alpha _ { 1 9 } \circ T _ { 6 } , } \\ & { T _ { 6 } = W \left( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \right) \circ T . } \end{array}
$$

Abstraction and external behavior. Use the abstraction set I defined earlier in the manuscript.

$$
I \subseteq { \mathsf { A c t } } \cup { \mathsf { G r d } }
$$

It contains all configured internal actions occurring in the process equations, while the input and output actions remain visible.

Theorem 7.1 (Correctness of the Most-Recent Induction Transformer). For every received input sequence x, the transformer construction of Most-Recent Induction computes and emits its Most-Recent Induction prediction sequence and returns to input-waiting state. Thus, $\tau _ { I } ( T )$ satisfies the external behavior equation:

$$
\tau _ { I } ( T ) = R ( x ) \circ W \bigl ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \bigr ) \circ \tau _ { I } ( T ) .
$$

Proof. Section 7.1.1 establishes that the predecessor and retrieval computations return the required prediction embedding at each position, including the no-match case. Unembedding therefore produces the stated output sequence.

Unfolding the state-transition equations and applying TI1, TI2, TI6, and TI7 gives

$$
\begin{array} { r l } & { \tau _ { I } ( T ) = R ( x ) \circ \tau _ { I } ( T _ { 2 } ) , } \\ & { \tau _ { I } ( T _ { 2 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 3 } ) , } \\ & { \tau _ { I } ( T _ { 3 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 4 } ) , } \\ & { \tau _ { I } ( T _ { 4 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 5 } ) , } \\ & { \tau _ { I } ( T _ { 5 } ) = \tau \circ \tau _ { I } ( T _ { 6 } ) , } \\ & { \tau _ { I } ( T _ { 6 } ) = W \big ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \big ) \circ \tau _ { I } ( T ) . } \end{array}
$$

By B4, $\tau \oplus \tau = \tau$ , together with A5, this gives

$$
\begin{array} { r l } & { \tau _ { I } ( T ) = R ( x ) \circ \tau _ { I } ( T _ { 2 } ) , } \\ & { \tau _ { I } ( T _ { 2 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 3 } ) , } \\ & { \tau _ { I } ( T _ { 3 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 4 } ) , } \\ & { \tau _ { I } ( T _ { 4 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 5 } ) , } \\ & { \tau _ { I } ( T _ { 5 } ) = \tau \circ \tau _ { I } ( T _ { 6 } ) , } \\ & { \tau _ { I } ( T _ { 6 } ) = W \big ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \big ) \circ \tau _ { I } ( T ) . } \end{array}
$$

For each input branch, substitution gives seven sequential silent actions between input and output.

$$
\begin{array} { r } { \tau _ { I } ( T ) = R ( x ) \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau } \\ { \circ W \bigl ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \bigr ) \circ \tau _ { I } ( T ) . } \end{array}
$$

For any continuation Y , A5 and B1 give

$$
R ( x ) \circ ( \tau \circ Y ) \stackrel { A 5 } { = } ( R ( x ) \circ \tau ) \circ Y \stackrel { B 1 } { = } R ( x ) \circ Y .
$$

Applying this identity repeatedly within each input branch removes the seven sequential silent actions. Hence,

$$
\tau _ { I } ( T ) = R ( x ) \circ W \bigl ( \mathrm { I n d u c t } _ { \mathrm { r i g h t m o s t } } ( x ) \bigr ) \circ \tau _ { I } ( T ) .
$$

The resulting equation exhibits the required external behavior: the transformer receives an input sequence, computes its Most-Recent Induction predictions, emits the result, and returns to the input-waiting process. □

## 7.2 Verification of a Transformer for Dyck-1

In this subsection, we apply APT to the verification of the uniform two-layer Transformer for the Dyck-1 language presented in the Transformer Cookbook[5]. The construction considered here is uniform: its parameters are independent of the input length, and the transformer computes the quantities required for recognition rather than an internal binary acceptance bit.

The verification has three complementary parts. First, we describe the numerical construction and establish its recognition criterion. Second, we represent its embedding, attention, and feedforward components using existing atomic constructions. Finally, we connect these components through recursive equations and derive the external input-output behavior by abstraction.

## 7.2.1 A Transformer Construction for Dyck-1

The Dyck-1 language consists of well-nested strings over the alphabet

$$
\Sigma = \{ ( , ) \} .
$$

A string $w \in \Sigma ^ { * }$ belongs to Dyck-1 exactly when it satisfies the following two conditions:

• Property 1 (Prefix Condition): In every prefix of the string, the count of open brackets must be greater than or equal to the count of close brackets.

• Property 2 (Balance Condition): The total count of open and close brackets in the entire string must be equal.

The construction below is stated for nonempty inputs because it inspects the representation at the final position. The empty string may be accepted separately by convention. Let $w = w _ { 1 } \cdot w _ { n }$ ∈ $\Sigma ^ { + }$ be a nonempty input string. Encode its symbols by

$$
o _ { i } = \{ \begin{array} { l l } { 1 , } & { w _ { i } = ( , } \\ { - 1 , } & { w _ { i } = ) . } \end{array}  
$$

For each $i \in [ n ]$ , let $O _ { i }$ and $C _ { i }$ be the numbers of opening and closing parentheses, respectively, in the prefix $w _ { 1 } { \cdots } w _ { i }$ . The corresponding prefix balance is

$$
B _ { i } = O _ { i } - C _ { i } = \sum _ { j = 1 } ^ { i } o _ { j } .
$$

The embedding initializes

$$
( o _ { i } , 0 , 0 , 0 ) ^ { \mathsf { T } } .
$$

The first attention component computes the normalized prefix balance:

$$
\frac { 1 } { i } \sum _ { j = 1 } ^ { i } o _ { j } = \frac { B _ { i } } { i } .
$$

The first FFN computes the prefix-error signal

$$
E _ { i } = \mathrm { R e L U } \left( - \frac { B _ { i } } { i } \right) .
$$

The second attention component computes its prefix average:

$$
t _ { i } = { \frac { 1 } { i } } \sum _ { j = 1 } ^ { i } E _ { j } .
$$

The final residual representation at position i is

$$
z _ { i } = \left[ \begin{array} { c } { z _ { i , 1 } } \\ { z _ { i , 2 } } \\ { z _ { i , 3 } } \\ { z _ { i , 4 } } \end{array} \right] = \left[ \begin{array} { c } { \boldsymbol { \mathbf { \rho } } _ { i } } \\ { \boldsymbol { B } _ { i } / i } \\ { \boldsymbol { E } _ { i } } \\ { \boldsymbol { t } _ { i } } \end{array} \right] .
$$

The representation therefore evolves as

$$
\left[ \begin{array} { c } { \rho _ { i } } \\ { 0 } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right] \xrightarrow { \mathrm { C o m p u t e ~ B a l a n c e } } \left[ \begin{array} { c } { \rho _ { i } } \\ { B _ { i } / i } \\ { 0 } \\ { 0 } \end{array} \right]  \xrightarrow { \mathrm { N e g a t i v e ~ } B _ { i } \mathrm { ~ E r r o r ? } } \left[ \begin{array} { c } { \rho _ { i } } \\ { B _ { i } / i } \\ { E _ { i } } \\ { 0 } \end{array} \right]  \xrightarrow { \mathrm { S u m ~ o f ~ E r r o r s } } \left[ \begin{array} { c } { \rho _ { i } } \\ { B _ { i } / i } \\ { E _ { i } } \\ { t _ { i } } \end{array} \right] { }
$$

Write the final representation as

$$
z _ { i } = ( o _ { i } , B _ { i } / i , E _ { i } , t _ { i } ) ^ { \mathsf { T } } .
$$

At the final position,

$$
z _ { n , 2 } = \frac { B _ { n } } { n } , \qquad z _ { n , 4 } = t _ { n } .
$$

Since $n > 0$

$$
z _ { n , 2 } = \frac { B _ { n } } { n } = 0 \Longleftrightarrow B _ { n } = 0 ,
$$

so $z _ { n , 2 } = 0$ is equivalent to the balance condition. Moreover,

$$
E _ { i } = 0 \Longleftrightarrow B _ { i } \geq 0 .
$$

and all error signals $E _ { i }$ are nonnegative. Therefore,

$$
t _ { n } = 0 \iff E _ { i } = 0 { \mathrm { ~ f o r ~ e v e r y ~ } } i \in [ n ] \iff B _ { i } \geq 0 { \mathrm { ~ f o r ~ e v e r y ~ } } i \in [ n ] .
$$

Thus, $z _ { n , 4 } = 0$ is equivalent to the prefix condition. It follows that

$$
w \in { \mathrm { D y c k - 1 } } \iff z _ { n , 2 } = 0 \land z _ { n , 4 } = 0 .
$$

For later use, define the expected numerical output for w by

$$
o _ { D } : = \left( \frac { B _ { n } ( w ) } { n } , t _ { n } ( w ) \right) .
$$

Then

$$
w \in \mathrm { D y c k - 1 } \quad \Longleftrightarrow \quad o _ { D } = ( 0 , 0 ) .
$$

The equality test is an external observation. It is not an internal comparison layer or Boolean classifier.

## 7.2.2 Algebraic Representation

The construction uses causal uniform averaging, residual connections, and a four-dimensional residual workspace. No positional embedding or layer normalization is required.

Repeated occurrences of the same action symbol may therefore denote diferently configured instances of the same construction. For example, the two occurrences of $\alpha _ { 1 3 }$ below use diferent source and target workspace coordinates. All such parameter choices are meta-level configurations only and introduce neither new atomic actions nor new operators into APT.

Input interface.The input action

$$
R ( w )
$$

receives the input string w from the external environment and stores it. This interface action does not itself perform a Transformer computation. Once it has terminated, sequential composition transfers control to embedding.

Embedding. The first internal Transformer action is the existing word-embedding action

$$
\alpha _ { 1 }
$$

which reads the symbols $w _ { i }$ from the input component, yields the first component $o _ { i }$ at each position, and initializes the residual representation at position i as

$$
\begin{array} { r } { \left[ { O _ { i } } \right] } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} .
$$

No separate position-embedding action $\alpha _ { 2 }$ is required in this construction.

First self-attention component. The first self-attention component represented by

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 3 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 2 }$ is the sequence-identity construction used to preserve input information;

$\alpha _ { 1 3 }$ is the uniform-average action, which reads the first coordinate and computes its uniform average over positions $1 , \cdot , i .$ . It writes the causal prefix average $B _ { i } / i$ to the second workspace coordinate and produce 0 in the other output coordinates;

$\alpha _ { 5 }$ merges the preserved input representation with the newly attention output.

Consequently, this component performs

$$
\begin{array}{c} \begin{array} { r } { \left[ { o _ { i } } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right] \longmapsto \left[ { \begin{array} { c } { { o _ { i } } } \\ { { B _ { i } } / { i } } \\ { { 0 } } \\ { { 0 } } \end{array} } \right] . } \end{array}
$$

First feed-forward component. The first feed-forward component and its residual connection are

$$
\left( \alpha _ { 3 } \oplus \left( \alpha _ { 6 } \circ \alpha _ { 1 0 } \right) \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 3 }$ is the vector-level identity construction that preserves the residual representation while the FFN branch computes the new error signal.

$\alpha _ { 6 }$ reads the normalized prefix balance $B _ { i } / i$ and multiplies it by $^ { \cdot } - 1$ ;

$\alpha _ { 1 0 }$ is the continuous-piecewise-linear action, applies ReLU to produce the prefix error signal $E _ { i } = \mathrm { R e L U } ( - B _ { i } / i )$ and routes the result to the third coordinate;

$\alpha _ { 5 }$ adds the computational output to the retained identity-branch output.

The computational branch produces

$$
\begin{array} { r } { \left[ \begin{array} { l } { 0 } \\ { 0 } \\ { E _ { i } } \\ { 0 } \end{array} \right] } \end{array}
$$

Thus, this is the correction required to obtain

$$
\begin{array} { r } { \left[ \begin{array} { c } { o _ { i } } \\ { B _ { i } / i } \\ { 0 } \\ { 0 } \end{array} \right] \longmapsto \left[ \begin{array} { c } { o _ { i } } \\ { B _ { i } / i } \\ { E _ { i } } \\ { 0 } \end{array} \right] . } \end{array}
$$

Second self-attention component The second self-attention component is represented by

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 3 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 2 }$ preserves the residual sequence;

$\alpha _ { 1 3 }$ now reads the third coordinate (the error signals $E _ { i } )$ and computes its causal prefix average $t _ { i } .$ . It writes this value into the fourth coordinate;

$\alpha _ { 5 }$ adds the routed output to the preserved representation.

The resulting transformation is

$$
\begin{array} { r } { \left[ \begin{array} { c } { o _ { i } } \\ { B _ { i } / i } \\ { E _ { i } } \\ { 0 } \end{array} \right] \longmapsto \left[ \begin{array} { c } { o _ { i } } \\ { B _ { i } / i } \\ { E _ { i } } \\ { t _ { i } } \end{array} \right] . } \end{array}
$$

Second feed-forward component. No additional feed-forward computation is required after the second self-attention stage. In the Dyck-1 construction, the second attention head already writes the accumulated error $t _ { i }$ into the final workspace coordinate, after which recognition is specified by inspecting the final-position values. We therefore omit the second-layer FFN.

Output interface The output action reads the second and fourth coordinates at the final position:

$$
( z _ { n , 2 } , z _ { n , 4 } ) = \left( \frac { B _ { n } ( w ) } { n } , t _ { n } ( w ) \right) = o _ { \cal D } .
$$

Its occurrence is therefore labelled

$$
\mathsf { W } ( o _ { D } ) ,
$$

which emits a two-dimensional numerical result without performing the external acceptance test.

## 7.2.3 Verification of the Dyck-1 Transformer

We now verify that the $\mathrm { D y c k - 1 }$ Transformer exhibits the expected external behavior: it receives a nonempty input string, computes and emits the corresponding numerical result, and returns to its input-waiting process.

State-transitions equations. For each w $\in \Sigma ^ { + }$ , define its processes are as follows:

$$
\begin{array} { r l } & { T = R ( w ) \circ T _ { 2 } , } \\ & { T _ { 2 } = \alpha _ { 1 } \circ T _ { 3 } , } \\ & { T _ { 3 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 3 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 4 } , } \\ & { T _ { 4 } = \left( \left( \alpha _ { 3 } \oplus \left( \alpha _ { 6 } \circ \alpha _ { 1 0 } \right) \right) \circ \alpha _ { 5 } \right) \circ T _ { 5 } , } \\ & { T _ { 5 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 3 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 6 } , } \\ & { T _ { 6 } = W ( o _ { D } ) \circ T . } \end{array}
$$

Abstraction of internal actions. All configured internal actions belong to I. The interface actions $R ( w )$ and $W ( o _ { D } )$ remain visible.

Theorem 7.2 (Correctness of the Dyck-1 Transformer.). For every received input string $w _ { ; }$ the transformer construction of Dyck-1 computes and emits its numerical result $o _ { D }$ and returns to input-waiting state. Thus, $\tau _ { I } ( T )$ satisfies the external behavior equation:

$$
\tau _ { I } ( T ) = R ( w ) \circ W ( o _ { D } ) \circ \tau _ { I } ( T ) .
$$

Proof. First, for the process calculation, axiom P1 gives

$$
\alpha _ { 3 } \oplus \left( \alpha _ { 6 } \circ \alpha _ { 1 0 } \right) = \left( \alpha _ { 3 } \oplus \alpha _ { 6 } \right) \circ \alpha _ { 1 0 } .
$$

Together with A5, this rewrites the feed-forward residual process as

$$
T _ { 4 } = \left( \alpha _ { 3 } \oplus \alpha _ { 6 } \right) \circ \alpha _ { 1 0 } \circ \alpha _ { 5 } \circ T _ { 5 } .
$$

The abstraction axioms TI1, TI2, TI6, and TI7 apply to the residual-process equations, and using B4 to simplify

$$
\tau \oplus \tau = \tau ,
$$

we obtain

$$
\begin{array} { r l } & { \tau _ { I } ( T ) = R ( w ) \circ \tau _ { I } ( T _ { 2 } ) , } \\ & { \tau _ { I } ( T _ { 2 } ) = \tau \circ \tau _ { I } ( T _ { 3 } ) , } \\ & { \tau _ { I } ( T _ { 3 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 4 } ) , } \\ & { \tau _ { I } ( T _ { 4 } ) = \tau \circ \tau \circ \tau \circ \tau _ { I } ( T _ { 5 } ) , } \\ & { \tau _ { I } ( T _ { 5 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 6 } ) , } \\ & { \tau _ { I } ( T _ { 6 } ) = W ( o _ { D } ) \circ \tau _ { I } ( T ) . } \end{array}
$$

Substitution yields eight sequential silent actions between the visible input and output actions. For any continuation Y , A5 and B1 give

$$
R ( x ) \circ ( \tau \circ Y ) \stackrel { A 5 } { = } ( R ( x ) \circ \tau ) \circ Y \stackrel { B 1 } { = } R ( x ) \circ Y .
$$

Applying this equality repeatedly within each input branch yields

$$
\tau _ { I } ( T ) = R ( w ) \circ W ( o _ { D } ) \circ \tau _ { I } ( T )
$$

The resulting equation exhibits the required external behavior: the transformer receives a string, emits its corresponding numerical result, and returns to the input-waiting process. □

## 7.3 Verification of a Transformer for Dyck-1-2

In this subsection, we apply APT to the depth-two bracket-matching construction for the Dyck-1-2 presented in the Transformer Cookbook[5]. Unlike the running-balance construction for Dyck-1, this Transformer maintains an active bit at every position and performs two rounds of nearest-neighbor matching.

We first summarize the construction and its recognition criterion. We then represent its embedding, positional encoding, attention, and feed-forward components using existing APT atomic constructions. Finally, we connect these components through recursive process equations and derive the expected external input-output behavior by abstraction.

## 7.3.1 A Transformer Construction of Dyck-1-2

The Dyck-1-2 language consists of well-nested strings with one type of bracket and maximum nesting depth two. Let $\Sigma = \{ ( , ) \}$ , and let

$$
w = w _ { 1 } . . . w _ { n } \in \Sigma ^ { + } , \qquad n = | w | ,
$$

be a nonempty input string, where $w _ { i }$ is the bracket at position i.

For $1 \leq i \leq n ,$ define

$$
o _ { i } = \left\{ \begin{array} { l l } { + 1 , } & { w _ { i } = \left( , \right. } \\ { - 1 , } & { w _ { i } = \left. \right) . } \end{array} \right.
$$

We initialize our embedding vector $x _ { i }$ at each position i as follows:

$$
\begin{array} { r } { \boldsymbol { x } _ { i } = \left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] , } \end{array}
$$

The second coordinate is an ”active bit”. Its initial value is 1, indicating that the bracket has not yet been matched. A successful match changes this coordinate to 0. The third and fourth coordinates provide scratch space for intermediate computations, and the fifth stores positional information.

Identifying depth-one matches. The first step retrieves the bracket types immediately to the left and right of each position using the predecessor and successor constructions with the corresponding strict masks. The intermediate representation contains the current bracket type and its two neighboring bracket types. An FFN then marks adjacent opening–closing pairs as matched and clears the scratch coordinates:

$$
\left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] \xrightarrow { A ^ { \mathrm { l e f t } } , A ^ { \mathrm { r i g h t } } } \left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { o _ { i - 1 } } \\ { o _ { i + 1 } } \\ { i / n } \end{array} \right] \xrightarrow { \mathrm { F F N : ~ M a r k ~ M a t c h e d } } \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] ,
$$

where

$$
a _ { i } = \left\{ \begin{array} { l l } { 0 , } & { o _ { i } = + 1 \mathrm { ~ a n d ~ } o _ { i + 1 } = - 1 , } \\ { 0 , } & { o _ { i } = - 1 \mathrm { ~ a n d ~ } o _ { i - 1 } = + 1 , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Thus, both brackets in each adjacent opening–closing pair are marked as matched.

Identifying depth-two matches. The second step searches for the nearest unmatched bracket on each side, ignoring positions whose active bits are zero. This selection through the UHAT expression

$$
\operatorname { U H A T } \left( \left[ { \frac { 1 } { n } } - \left( a _ { 1 } - 1 \right) , { \frac { 2 } { n } } - \left( a _ { 2 } - 1 \right) , \ldots , { \frac { n } { n } } - \left( a _ { n } - 1 \right) \right] \right) _ { i } , \qquad i = 1 , \ldots , n .
$$

The source describes obtaining the required quantities through a linear transformation of the intermediate representation, using the attention query $W _ { ( Q ) }$ and key matrices $W _ { K }$ with a bias

term. We then proceed analogously to first step, the workspace changes as follows:

$$
\left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] \xrightarrow { \mathrm { S e l e c t ~ N e a r e s t ~ A c t i v e } } \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { l _ { i } } \\ { r _ { i } } \\ { i / n } \end{array} \right] \xrightarrow [ ] { \mathrm { C h e c k ~ M a t c h } } \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } ^ { \prime } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] .
$$

Here, $l _ { i }$ and $r _ { i }$ denote the types of the nearest unmatched brackets to the left and right, with 0 denoting a missing neighbor. The nearest unmatched bracket types $l _ { i }$ and $r _ { i }$ are defined as above:

$$
l _ { i } = \left\{ \begin{array} { l l } { o _ { \operatorname* { m a x } \{ j \in \{ 1 , \dots , i - 1 \} : a _ { j } = 1 \} } , } & { \mathrm { i f ~ } \{ j \in \{ 1 , \dots , i - 1 \} : a _ { j } = 1 \} \not = \emptyset , } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

$$
r _ { i } = \left\{ \begin{array} { l l } { o _ { \mathrm { m i n } \{ j \in \{ i + 1 , \dots , n \} : a _ { j } = 1 \} } , } & { \mathrm { i f ~ } \{ j \in \{ i + 1 , \dots , n \} \not = \emptyset , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The maximum selects the largest active position index smaller than $i ,$ while the minimum selects the smallest active position index greater than $i .$

The FFN updates the active bit according to

$$
a _ { i } ^ { \prime } = \left\{ \begin{array} { l l } { 0 , } & { a _ { i } = 0 , } \\ { 0 , } & { a _ { i } = 1 , \ o _ { i } = + 1 , \ r _ { i } = - 1 , } \\ { 0 , } & { a _ { i } = 1 , \ o _ { i } = - 1 , \ l _ { i } = + 1 , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

After the second step, the string is accepted if and only if all active bits are $0 .$

Recognition criterion. The final numerical output is the active-bit sequence

$$
( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) .
$$

The recognition condition is

$$
w \in \mathrm { D y c k - 1 - 2 } \iff a _ { i } ^ { \prime } = 0 \quad \mathrm { f o r ~ e v e r y ~ } i = 1 , \dots , n .
$$

The all-zero test is performed by the external environment.

## 7.3.2 Algebraic Representation

Input interface. The input action is represented by $R ( w )$ which receives the input string w defined in Section 7.3.1 and stores it. Sequential composition then transfers control to the embedding component.

Embedding. The initialization component is represented by

$$
\left( \alpha _ { 1 } \oplus \alpha _ { 2 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 }$ is the word embedding action, which supplies the bracket type $o _ { i }$ and initializes the active bit $a _ { i }$

• α<sub>2</sub> is the position embedding action used to supply $i / n$ , which enables attention to identify neighboring positions and select the nearest unmatched brackets in the second matching step.

$\alpha _ { 5 }$ is the addition action, which adds the outputs of $\alpha _ { 1 }$ and $\alpha _ { 2 }$ coordinatewise to obtain the initial workspace vector $x _ { i } = ( o _ { i } , 1 , 0 , 0 , i / n ) ^ { \mathsf { T } }$

First attention component. The first attention component retrieves the bracket types immediately preceding and succeeding each position and stores them in the two scratch coordinates. Its algebraic representation is

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 1 2 }$ is the sequence identity action. It preserves the current residual sequence, including the bracket types, active bits, and positional encodings.

$\alpha _ { 1 8 }$ is a configured multi-head attention action containing two directional heads. The left head retrieves the bracket type $o _ { i - 1 }$ at position i − 1, while the right head retrieves $o _ { i + 1 }$ at position $i + 1$ , using the corresponding strict masks. Their outputs are routed to the third and fourth coordinates, a missing neighbor contributes 0.

• α<sub>5</sub> adds the preserved residual sequence and the attention output coordinatewise.

Thus, the component performs

$$
\begin{array} { r } { \left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] \longmapsto \left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { 1 } \\ { o _ { i - 1 } } \\ { o _ { i + 1 } } \\ { i / n } \end{array} \right] . } \end{array}
$$

First feed-forward component. The first feed-forward component identifies adjacent opening–closing pairs, changes their active bits to 0, and resets the third and fourth coordinates to 0. The active-bit update is

$$
U _ { 1 } = \left( \alpha _ { 3 } \oplus \alpha _ { 1 0 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 3 }$ is the vector identity action. It preserves the input vector at each position.

$\alpha _ { 1 0 }$ is a configured piecewise-linear action. It reads the current bracket type and the two retrieved neighbor types, determines $a _ { i }$ according to the first-step matching rule, and routes it only to the second coordinate.

$\alpha _ { 5 }$ adds the correction to the preserved input vector. It changes the active bit from 1 to ${ { a } _ { i } } ,$ resets the third and fourth coordinates to 0, and preserves the bracket type and positional encoding.

The temporary coordinates are cleared by

$$
C _ { 3 } = ( \alpha _ { 3 } \oplus \alpha _ { 6 } ) \circ \alpha _ { 5 } ,
$$

$$
C _ { 4 } = ( \alpha _ { 3 } \oplus \alpha _ { 6 } ) \circ \alpha _ { 5 } .
$$

In $C _ { 3 } { \mathrm { : } }$

$\alpha _ { 3 }$ preserves the representation by $U _ { 1 }$

$\alpha _ { 6 }$ reads the third coordinate, multiplies its value by −1, and routes the result to the third output coordinate, with all other output coordinates set to zero;

$\alpha _ { 5 }$ adds the separately retained branch outputs coordinatewise. The third coordinate becomes $o _ { i - 1 } + \left( - o _ { i - 1 } \right) = 0$ , while the remaining coordinates are preserved.

In $C _ { 4 }$ :

$\alpha _ { 3 }$ preserves the representation produced by $C _ { 3 }$ ;

$\alpha _ { 6 }$ reads the fourth coordinate, multiplies its value by -1, and routes the result to the fourth output coordinate, with all other output coordinates set to zero.

$\alpha _ { 5 }$ adds the branch outputs coordinatewise. The fourth coordinate becomes $o _ { i + 1 } + \left( - o _ { i + 1 } \right) =$ 0, while the remaining coordinates are preserved.

Thus, the complete feed-forward component

$$
\begin{array} { r l } & { F _ { 1 } = U _ { 1 } \circ C _ { 3 } \circ C _ { 4 } } \\ & { \qquad = \big ( \big ( \alpha _ { 3 } \oplus \alpha _ { 1 0 } \big ) \circ \alpha _ { 5 } \big ) \circ \big ( \big ( \alpha 3 \oplus \alpha _ { 6 } \big ) \circ \alpha _ { 5 } \big ) \circ \big ( \big ( \alpha _ { 3 } \oplus \alpha _ { 6 } \big ) \circ \alpha _ { 5 } \big ) . } \end{array}
$$

updates the active bit and then resets the third and fourth coordinates to zero. It performs

$$
\begin{array} { r } { \left[ \begin{array} { l } { o _ { i } } \\ { 1 } \\ { o _ { i - 1 } } \\ { o _ { i + 1 } } \\ { i / n } \end{array} \right] \longmapsto \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] . } \end{array}
$$

Second attention component. The second attention component retrieves the nearest unmatched bracket on each side whenever such a bracket exists. Its algebraic representation is

$$
\left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } .
$$

Here:

• α<sub>12</sub> preserves the residual sequence produced by the first matching step, including its updated active bits.

$\alpha _ { 1 8 }$ is a configured multi-head attention action containing two directional heads. The left and right heads retrieve $l _ { i }$ and $r _ { i }$ , respectively, and route them to the third and fourth coordinates.

$\alpha _ { 5 }$ adds the attention output to the preserved residual sequence.

Whenever the corresponding active neighbors exist, this gives

$$
\begin{array} { r } { \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] \longmapsto \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { l _ { i } } \\ { r _ { i } } \\ { i / n } \end{array} \right] . } \end{array}
$$

Second feed-forward component. The second feed-forward component matches adjacent opening–closing pairs in the remaining active sequence, preserves previously matched positions, and resets the third and fourth coordinates to 0. The Second active bit update is

$$
U _ { 2 } = \left( \alpha _ { 3 } \oplus \alpha _ { 1 0 } \right) \circ \alpha _ { 5 } .
$$

Here:

$\alpha _ { 3 }$ preserves the input vector at each position.

$\alpha _ { 1 0 }$ reads $o _ { i } , a _ { i } , l _ { i } , r _ { i }$ and implements

$$
a _ { i } ^ { \prime } = \left\{ \begin{array} { l l } { 0 , } & { a _ { i } = 0 , } \\ { 0 , } & { a _ { i } = 1 , \ o _ { i } = + 1 , \ r _ { i } = - 1 , } \\ { 0 , } & { a _ { i } = 1 , \ o _ { i } = - 1 , \ l _ { i } = + 1 , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

This correction updates the active bit.

• $\alpha _ { 5 }$ adds the correction to the preserved input vector, producing $a _ { i } ^ { \prime }$ in the second coordinate.

Using the same clearing components $C _ { 3 }$ and $C _ { 4 }$ as in the first feed-forward component, the second feed-forward component is represented by

$$
\begin{array} { r l } & { F _ { 2 } = U _ { 2 } \circ C _ { 3 } \circ C _ { 4 } } \\ & { \quad = \big ( \big ( \alpha _ { 3 } \oplus \alpha _ { 1 0 } \big ) \circ \alpha _ { 5 } \big ) \circ \big ( \big ( \alpha _ { 3 } \oplus \alpha _ { 6 } \big ) \circ \alpha _ { 5 } \big ) \circ \big ( \big ( \alpha _ { 3 } \oplus \alpha _ { 6 } \big ) \circ \alpha _ { 5 } \big ) } \end{array}
$$

The resulting transformation is

$$
\begin{array} { r } { \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } } \\ { l _ { i } } \\ { r _ { i } } \\ { i / n } \end{array} \right] \longmapsto \left[ \begin{array} { l } { o _ { i } } \\ { a _ { i } ^ { \prime } } \\ { 0 } \\ { 0 } \\ { i / n } \end{array} \right] . } \end{array}
$$

Output interface. The output action reads and emits the final active-bit sequence:

$$
W \big ( ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) \big ) .
$$

The emitted values are determined by the computed residual state. The external environment accepts exactly when every emitted entry is 0. After output, control returns to the input-waiting process $T$

## 7.3.3 Verification of the Dyck-1-2 Transformer

We now verify that the Dyck-1-2 Transformer exhibits the expected external behavior: it receives a nonempty input string, computes and emits the corresponding numerical result, and returns to its input-waiting process.

State-transition equations. For each $w \in \Sigma ^ { + }$ , define

$$
\begin{array} { r l } & { T = R ( w ) \circ T _ { 2 } , } \\ & { T _ { 2 } = \left( \left( \alpha _ { 1 } \oplus \alpha _ { 2 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 3 } , } \\ & { T _ { 3 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 4 } , } \\ & { T _ { 4 } = F _ { 1 } \circ T _ { 5 } , } \\ & { T _ { 5 } = \left( \left( \alpha _ { 1 2 } \oplus \alpha _ { 1 8 } \right) \circ \alpha _ { 5 } \right) \circ T _ { 6 } , } \\ & { T _ { 6 } = F _ { 2 } \circ T _ { 7 } , } \\ & { T _ { 7 } = W \left( \left( a _ { 1 } ^ { \prime } , . . . , a _ { n } ^ { \prime } \right) \right) \circ T . } \end{array}
$$

Here, $F _ { 1 }$ and $F _ { 2 }$ are the composite feed-forward terms defined above.

Abstraction and external behavior. Use the abstraction set I defined earlier in the manuscript, containing all configured internal actions occurring above. The interface actions $R ( w )$ and $W ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } )$ remain visible.

Theorem 7.3 (External behavior of the Dyck-1 Transformer.). For every received input string $w$ , the transformer construction of Dyck-1-2 computes and emits final active-bit sequence and returns to input-waiting state. Thus, $\tau _ { I } ( T )$ satisfies the external behavior equation

$$
\tau _ { I } ( T ) = R ( w ) \circ W ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) \circ \tau _ { I } ( T ) .
$$

Proof. Applying the abstraction axioms to the state-transition equations and expanding $F _ { 1 } =$ $U _ { 1 } \circ C _ { 3 } \circ C _ { 4 }$ and $F _ { 2 } = U _ { 2 } \circ C _ { 3 } \circ C _ { 4 }$ gives

$$
\begin{array} { r l } & { \tau _ { I } ( T ) = R ( w ) \circ \tau _ { I } ( T _ { 2 } ) , } \\ & { \tau _ { I } ( T _ { 2 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 3 } ) , } \\ & { \tau _ { I } ( T _ { 3 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 4 } ) , } \\ & { \tau _ { I } ( T _ { 4 } ) = \tau _ { I } ( U _ { 1 } ) \circ \tau _ { I } ( C _ { 3 } ) \circ \tau _ { I } ( C _ { 4 } ) \circ \tau _ { I } ( T _ { 5 } ) , } \\ & { \tau _ { I } ( T _ { 5 } ) = ( \tau \oplus \tau ) \circ \tau \circ \tau _ { I } ( T _ { 6 } ) , } \\ & { \tau _ { I } ( T _ { 6 } ) = \tau _ { I } ( U _ { 2 } ) \circ \tau _ { I } ( C _ { 3 } ) \circ \tau _ { I } ( C _ { 4 } ) \circ \tau _ { I } ( T _ { 7 } ) , } \\ & { \tau _ { I } ( T _ { 7 } ) = W \left( ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) \right) \circ \tau _ { I } ( T ) . } \end{array}
$$

Since $U _ { 1 } , U _ { 2 } , C _ { 3 }$ , and $C _ { 4 }$ each consist of two parallel internal actions followed by addition, TI2, TI6, TI7, and B4 give

$$
\tau _ { I } ( U _ { 1 } ) = \tau _ { I } ( U _ { 2 } ) = \tau _ { I } ( C _ { 3 } ) = \tau _ { I } ( C _ { 4 } ) = ( \tau \oplus \tau ) \circ \tau = \tau \circ \tau .
$$

Substituting these equalities, simplifying the remaining parallel silent actions by B4, and applying A5 gives

$$
\begin{array} { r l } & { \tau _ { I } ( T ) = R ( w ) \circ \tau _ { I } ( T _ { 2 } ) , } \\ & { \tau _ { I } ( T _ { 2 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 3 } ) , } \\ & { \tau _ { I } ( T _ { 3 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 4 } ) , } \\ & { \tau _ { I } ( T _ { 4 } ) = \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau _ { I } ( T _ { 5 } ) , } \\ & { \tau _ { I } ( T _ { 5 } ) = \tau \circ \tau \circ \tau _ { I } ( T _ { 6 } ) , } \\ & { \tau _ { I } ( T _ { 6 } ) = \tau \circ \tau \circ \tau \circ \tau \circ \tau \circ \tau _ { I } ( T _ { 7 } ) , } \\ & { \tau _ { I } ( T _ { 7 } ) = W \left( ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) \right) \circ \tau _ { I } ( T ) . } \end{array}
$$

For any continuation Y, A5 and B1 establish

$$
R ( w ) \circ ( \tau \circ Y ) \stackrel { A 5 } { = } ( R ( w ) \circ \tau ) \circ Y \stackrel { B 1 } { = } R ( w ) \circ Y .
$$

Substituting the abstracted continuation equations into each input branch and repeatedly applying this identity removes the finite sequence of internal silent actions. Consequently,

$$
\tau _ { I } ( T ) = R ( w ) \circ W \bigl ( ( a _ { 1 } ^ { \prime } , \ldots , a _ { n } ^ { \prime } ) \bigr ) \circ \tau _ { I } ( T ) .
$$

The resulting equation exhibits the required external behavior: the transformer receives an input string, emits its computed final active-bit sequence, and returns to its input-waiting process.

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need (2023). arXiv:1706.03762. URL https://arxiv.org/abs/1706.03762

[2] J. P´erez, P. Barcel´o, J. Marinkovic, Attention is turing complete, Journal of Machine Learning Research 22 (2021) 1–35.

[3] D. H. C. Sanford, M. Telgarsky, Transformers, parallel computation, and logarithmic depth, Under reviewPreprint, available at arXiv:2402.09268 (2024).

[4] W. Merrill, A. Sabharwal, The parallelism tradeof: Limitations of log-precision transformers, Transactions of the Association for Computational Linguistics 11.

[5] A. Yang, C. Watson, A. Xue, S. Bhattamishra, J. Llarena, W. Merrill, E. D. S. Ferreira, A. S., D. Chiang, The transformer cookbook (2025). arXiv:2510.00368. URL https://arxiv.org/abs/2510.00368

[6] D. Knuth, P. Bendix, Simple word problems in universal algebras, Computational Problems in Abstract Algebra (1970) 263–297.

[7] F. Vaandrager, Verification of two communication protocols by means of process algebra, Tech. Rep. Report CS-R8608, CWI, Amsterdam (1986).