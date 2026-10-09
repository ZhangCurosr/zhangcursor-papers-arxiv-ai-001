# Complexity of Grounded Semantics and Preferred Semantics in Finitary Argumentation Frameworks

Xu Jinfan and Luo Jieting

School of Philosophy, Zhejiang University

Abstract. Abstract argumentation frameworks (AFs) introduced by Dung provide a formal foundation for non-monotonic reasoning in artificial intelligence. While decision problems for general infinite AFs typically reside at high levels of the analytical hierarchy (Σ<sup>1</sup> or Π<sup>1</sup>), restricting the framework to be computably finitary reduces some of the complexity to the arithmetical hierarchy. In this paper, we present a complexity mapping of grounded and preferred semantics in computably finitary AFs across standard decision problems: credulous acceptance (Cred), skeptical acceptance (Skep), extension existence (Exists), uniqueness (Uni), and non-empty existence (NE). For grounded semantics, credulous and skeptical acceptance are already known to be Σ<sup>0</sup><sub>1</sub>-complete. We show that non-empty existence is also Σ<sup>0</sup><sub>1</sub>-complete, whereas existence and uniqueness are trivial. These classifications are understood within the domain of valid computably finitary representations. For preferred semantics, using a computably finitely branching computation tree, $\operatorname { C r e d } _ { p r e f }$ is shown to be in $\varPi _ { \mathrm { 1 } } ^ { \mathrm { 0 } } \mathrm { - c }$ and $\mathrm { N E } _ { p r e f }$ is $\Sigma _ { \mathrm { 2 } ^ { \mathrm { - C } } } ^ { 0 }$ . However, it is insuficient to reduce universal quantification and global uniqueness, leaving ${ \mathrm { S k e p } } _ { p r e f }$ in $\boldsymbol { \varPi } _ { 1 } ^ { 1 }$ and ${ \mathrm { U n i } } _ { p r e f }$ in ${ \Sigma } _ { \mathrm { 2 } } ^ { 1 } { \mathrm { - c } }$ . Our results show the precise boundary where finitarity succeeds to bring reasoning down to the arithmetical hierarchy and where second-order quantification forces problems back into the analytical hierarchy.

Keywords: Abstract Argumentation · Finitary AFs · Computability Theory

## 1 Introduction

Abstract Argumentation Frameworks (AFs), introduced by Dung [6], have become a fundamental part of formal reasoning in artificial intelligence. While the computational properties of finite AFs are well-understood, recent years have witnessed a surge of interest in infinite argumentation frameworks (IAF). Infinite AFs enable the formal modeling of domains involving infinite events, continuous temporal reasoning, or infinite debate games; however, they introduce substantial computational challenges. As shown by Andrews and San Mauro [2], standard reasoning tasks in general infinite AFs often reside at the upper levels of the analytical hierarchy (e.g., being Π<sup>1</sup><sub>1</sub> - or Σ<sup>1</sup><sub>1</sub> -complete), rendering them algorithmically intractable in the general case.

Nevertheless, many realistic infinite scenarios satisfy a natural structural constraint known as the finitary restriction: although the total pool of arguments may be infinite, each individual argument is attacked by at most finitely many predecessors. This structural property provides a powerful graphical tool. In finitary $\mathrm { A F s , }$ the defense space of an argument can be modeled as a finitely branching tree. By exploiting classical logical tools, such as the ω-continuity of Dung’s characteristic operator and K¨onig’s Lemma, the computational complexity of several core semantics drops from the analytical hierarchy down to the arithmetical hierarchy.

Andrews and San Mauro have already established upper and lower bounds for credulous and skeptical acceptance in computably finitary AFs [1]. In particular, grounded acceptance is $\Sigma _ { 1 } ^ { 0 }$ -complete within the domain of valid representations. Building on these results, this paper studies grounded and preferred semantics across five decision problems. For grounded semantics, we recall the acceptance classification and prove that non-empty existence $( \mathrm { N E } _ { g r d } ^ { \mathrm { c f i n } } )$ is $\Sigma _ { 1 } ^ { 0 } \mathrm { \cdot }$ complete; existence and uniqueness are trivial. For preferred semantics, we use the correspondence between admissible sets and preferred extensions to obtain $\varPi _ { 1 } ^ { 0 }$ -completeness of credulous acceptance and $\Sigma _ { 2 } ^ { 0 }$ -completeness of non-empty existence. We also study the analytical complexity of skeptical acceptance and uniqueness under preferred semantics.

## 2 Computability Theoretic Background

In this section, we present the necessary foundations of recursion theory and descriptive complexity used to classify decision problems on infinite structures. We assume all elements of our domain can be efectively encoded as natural numbers ω.

## 2.1 Numbers, Strings, and Trees

To formulate our problems as subsets of $\omega ,$ it is convenient to encode pairs of numbers into single numbers. For this purpose, we fix a computable bijection $p : \omega \times \omega  \omega$ and adopt the standard notation $\langle x , y \rangle$ for $p ( x , y )$ . Furthermore, given a finite set $A = \left\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \right\} \subseteq \omega ,$ , its canonical index is defined as

$$
y = 2 ^ { x _ { 1 } } + 2 ^ { x _ { 2 } } + \cdot \cdot \cdot + 2 ^ { x _ { n } } .
$$

We let $D _ { y }$ denote the finite set whose canonical index is $y .$ . By leveraging canonical indices, we can quantify over all finite subsets of an argumentation framework.

We denote the set of all finite strings of natural numbers by $\omega ^ { < \omega }$ , and the set of all infinite strings by $\omega ^ { \omega }$ . A string σ is called a prefix of $\tau _ { : }$ written as $\sigma \preceq \tau ,$ if τ extends $\sigma$ . If neither $\sigma \preceq \tau$ nor $\tau \preceq \sigma$ holds, σ and τ are said to be incomparable. The restriction of an infinite string $\pi \in \omega ^ { \omega }$ to its first n bits is denoted by $\pi \ \lceil _ { n }$

A tree is a set $T \subseteq \omega ^ { < \omega }$ that is closed under prefixes. An infinite string $\pi \in \omega ^ { \omega }$ is called a path through $T$ if every finite prefix of $\pi$ belongs to $T \ ( \mathrm { i . e . }$ $\sigma \in T$ for all $\sigma \prec \pi , \mathrm { o r } \pi \vdash \Gamma _ { n } \in T$ for all numbers n). We denote the set of all paths through T by [T]. A tree T is said to have infinite height if it contains strings of arbitrarily large finite lengths.

## 2.2 The Arithmetical Hierarchy: Quantifying over Numbers

The arithmetical hierarchy classifies subsets of $\omega ^ { k }$ based on the number of alternating first-order quantifiers $( \exists , \forall )$ required to define them over a computable predicate.

Let $R \subseteq \omega ^ { k + n }$ be a recursive (computable) relation.

A set $S \subseteq \omega ^ { k }$ is in the class $\textstyle \sum _ { n } ^ { 0 }$ if there exists a recursive relation R such that:

$$
S = \{ \pmb { x } \ | \ \exists y _ { 1 } \forall y _ { 2 } \exists y _ { 3 } \dotsc Q y _ { n } R ( \pmb { x } , y _ { 1 } , \dotsc , y _ { n } ) \}
$$

where the quantifier string contains n alternating quantifiers, and Q is ∃ if n is odd and ∀ if n is even.

$- \textrm { A }$ set $S \subseteq \omega ^ { k }$ is in the class $\boldsymbol { \varPi } _ { n } ^ { 0 }$ if its complement is in $\textstyle \sum _ { n } ^ { 0 }$ . Formally:

$$
S = \{ \pmb { x } \ | \ \forall y _ { 1 } \exists y _ { 2 } \forall y _ { 3 } \ldots Q y _ { n } R ( \pmb { x } , y _ { 1 } , \ldots , y _ { n } ) \}
$$

where Q is ∀ if n is odd and ∃ if n is even.

The class $\varDelta _ { n } ^ { 0 }$ is defined as the intersection $\Sigma _ { n } ^ { 0 } \cap { \cal { I } } I _ { n } ^ { 0 }$ , representing properties that are structurally equivalent to both formats at level n.

## 2.3 The Analytical Hierarchy: Quantifying over Sets

When the definition of a problem requires quantifying over all possible infinite sets or functions, the arithmetical hierarchy is no longer suficient to classify its complexity. In these instances, we move into the analytical hierarchy. The fundamental distinction between these hierarchies lies in the domain of quantification. While the arithmetical hierarchy is restricted to quantification over individuals (first-order quantification, $\mathrm { e . g . } , \exists n \in \omega )$ , the analytical hierarchy allows for quantification over subsets of individuals (second-order quantification, $\mathrm { e . g . , } \exists S \subseteq \omega )$

Definition 1 (Analytical Hierarchy). A set $S \subseteq \omega ^ { k }$ belongs to the class $\textstyle \sum _ { n } ^ { 1 }$ or $\boldsymbol { \Pi } _ { n } ^ { 1 } \boldsymbol { i f } \boldsymbol { i t }$ can be defined by a second-order formula of the form:

$$
S = \{ { \pmb x } \ | \ ( Q _ { 1 } X _ { 1 } ) ( Q _ { 2 } X _ { 2 } ) \ldots ( Q _ { n } X _ { n } ) { \pmb \phi } ( { \pmb x } , X _ { 1 } , \ldots , X _ { n } ) \}
$$

where:

$- \ X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ are second-order variables ranging over subsets of ω (or functions from ω to ω).

$- \ Q _ { 1 } , Q _ { 2 } , . . . , Q _ { n }$ are alternating second-order quantifiers $\left( \exists ^ { 1 } , \forall ^ { 1 } \right)$

$- \ \varPhi$ is an arithmetical formula containing only first-order quantifiers.

A set is $\begin{array} { r } { \sum _ { n } ^ { 1 } \ i f Q _ { 1 } = \exists ^ { 1 } } \end{array}$ , and it is $I I _ { n } ^ { 1 } \ i f Q _ { 1 } = \forall ^ { 1 }$

## 3 Argumentation Theoretic Background

## 3.1 Abstract Argumentation

An argumentation framework (AF) is a pair ${ \mathcal { F } } = \langle A , R \rangle$ , where A is a set of arguments and $R \subseteq A \times A$ is a binary relation representing attacks. For $a , b \in A$ we say a attacks b if $( a , b ) \in R .$ . A set $S \subseteq A$ is conflict-free if there are no $a , b \in S$ such that $( a , b ) \in R$ . A set $S \subseteq A$ defends an argument $a \in A$ if for every $b \in A$ that attacks $^ { a , }$ there exists some $c \in S$ that attacks b.

Definition 2 (Characteristic Function). The characteristic function $T _ { \mathcal { F } }$ $2 ^ { A }  2 ^ { A }$ is defined as ${ \cal T } _ { \mathcal { F } } ( S ) = \{ a \in A \mid S$ defends a}.

## 3.2 Argumentation Semantics

In the extension-based approach to argumentation, a semantics σ associates with every AF ${ \mathcal { F } } = \langle A , R \rangle$ a set of extensions $\sigma ( { \mathcal { F } } ) \subseteq 2 ^ { A }$ . We follow the standard definitions introduced by Dung.

Definition 3 (Admissible-based Semantics). Let $\mathcal { F } = \langle A , R \rangle$ be an $A F$ and $S \subseteq A$ be a conflict-free set.

S is admissible $( S \in a d m ( { \mathcal { F } } ) )$ if S defends all its elements, i.e., $S \subseteq T _ { \mathcal { F } } ( S )$

$- \textit { S }$ is a complete extension $( S \in c o m ( { \mathcal { F } } ) )$ if it is a fixed point of the characteristic function, $i . e . , S = { \cal { T } } \mathcal { F } ( S )$

– S is the grounded extension $( S = g r d ( { \mathcal { F } } ) )$ if it is the least fixed point of $T _ { \mathcal { F } }$ – S is a preferred extension $( S \in p r e f ( \mathcal { F } ) ) \ i f$ it is a maximal $( w . r . t . \subseteq )$ admissible set.

While the existence of preferred and grounded extensions is guaranteed in any AF (including infinite ones) by Zorn’s Lemma and the Knaster-Tarski theorem respectively, the complexity of identifying these sets varies significantly depending on the graph’s topology.

## 3.3 Computably Finitary AFs and Computational Problems

We focus on computably finitary AFs, which are infinite structures where the local interaction is efectively bounded.

Definition 4 (Finitary AF). An $A F { \mathcal { F } } = \langle A , R \rangle$ is finitary if for every $a \in A$ the set of its attackers $A t t ( a ) = \{ b \in A \mid ( b , a ) \in R \}$ is finite.

Definition 5 (Computably Finitary AF). An argumentation framework ${ \mathcal { F } } =$ $\langle A , R \rangle$ , where A is the set of arguments and $R \subseteq A \times A$ is the attack relation, is said to be computably finitary if it satisfies the following two conditions $I I { \big / } .$

1. Recursive Set of Arguments: The set A is a recursive set. Formally, a set $A \subseteq \omega$ is recursive (or decidable) if its characteristic function $\chi _ { A } : \omega $ $\{ 0 , 1 \}$ is a total computable function, defined as:

$$
\chi _ { A } ( n ) = { \left\{ \begin{array} { l l } { 1 } & { i f n \in A } \\ { 0 } & { i f n \notin A } \end{array} \right. }
$$

In the context of infinite argumentation, A is typically identified with the set of natural numbers ω.

2. Computable Attackers: There exists a total computable function $f : A  \omega$ such that for every argument $a \in A , f ( a )$ returns the canonical index of the finite set of its attackers, $A t t ( a ) = \{ b \in A \mid ( b , a ) \in R \}$

Following the methodology of Andrews and San Mauro [2], we analyze five fundamental decision problems for each semantics σ:

1. Credulous Acceptance $\left( { \mathrm { C r e d } } _ { \sigma } \right)$ : Given $\mathcal { F }$ and $x \in A$ , is x contained in at least one $S \in \sigma ( \mathcal { F } ) ?$

2. Skeptical Acceptance $\left( \mathrm { S k e p } _ { \sigma } \right)$ : Given $\mathcal { F }$ and $x \in A$ , is x contained in every $S \in \sigma ( { \mathcal { F } } ) ?$

3. Existence $\left( \mathrm { E x i s t s } _ { \sigma } \right)$ : Does there exist a set $S \in \sigma ( \mathcal { F } ) ?$ (Note: for grounded/preferred, this is trivial).

4. Non-empty Existence $\left( \mathrm { N E } _ { \sigma } \right)$ : Does there exist a non-empty set $S \in \sigma ( \mathcal { F } ) ?$

5. Uniqueness $( \mathrm { U n i } _ { \sigma } )$ : Is there exactly one set $S \in \sigma ( \mathcal { F } ) ?$

For the computably finitary representation, fix a standard enumeration $( \varphi _ { e } ) _ { e \in \omega }$ of the partial computable functions and let

$$
\mathrm { T o t } = \{ e \in \omega \mid \varphi _ { e } \mathrm { i s \ t o t a l } \} .
$$

For each $e \in \mathrm { T o t }$ , let $\mathcal { F } _ { e }$ have argument set $A = \{ a _ { x } \mid x \in \omega \}$ , with the finite attacker set of $a _ { x }$ encoded by $D _ { \varphi _ { e } ( x ) }$ . Here a set of numerical codes is identified with the corresponding set of arguments, so that

$$
A t t _ { \mathcal { F } _ { e } } ( a _ { x } ) = \{ a _ { y } ~ | ~ y \in D _ { \varphi _ { e } ( x ) } \} .
$$

Thus, a valid index provides an efective finite list of all attackers of each argument. Complexity classifications below are understood within Tot (or Tot $\times \ \omega$ for acceptance): the upper-bound formulas need only agree with the decision problems on valid indices, and hardness reductions must always produce valid indices. They do not include the complexity of testing whether an arbitrary index belongs to Tot.

We code the decision problems as follows:

1. Cred<sup>cfin</sup><sub>σ</sub> := {⟨e, n⟩ | e ∈ Tot, (∃S ∈ σ(F<sub>e</sub>))(a<sub>n</sub> ∈ S)};

$$
{ \mathrm { S k e p } } _ { \sigma } ^ { \mathrm { c f i n } } : = \{ \langle e , n \rangle \mid e \in { \mathrm { T o t } } , ( \forall S \in \sigma ( { \mathcal { F } } _ { e } ) ) ( a _ { n } \in S ) \}
$$

3. Exists<sup>cfin</sup><sub>σ</sub> := {e ∈ Tot | σ(F<sub>e</sub>) ̸= ∅};

4. $\mathrm { N E } _ { \sigma } ^ { \mathrm { c f i n } } : = \{ e \in \mathrm { T o t } \ | \ ( \exists S \in \sigma ( { \mathcal F } _ { e } ) ) ( S \neq \emptyset ) \}$

$$
\begin{array} { r } { 5 . \mathrm { ~ U n i } _ { \sigma } ^ { \mathrm { c f i n } } : = \{ e \in \mathrm { T o t } | | \sigma ( \mathcal { F } _ { e } ) | = 1 \} . } \end{array}
$$

For grounded semantics, $\sigma ( \mathcal { F } _ { e } )$ in these definitions is the singleton $\{ g r d ( \mathcal { F } _ { e } ) \}$ We omit the superscript cfin when the representation is clear from context. We also write $\operatorname { C r e d } _ { \sigma } ( \mathcal { F } _ { e } )$ and $\mathrm { S k e p } _ { \sigma } ( \mathcal { F } _ { e } )$ for the corresponding sets of accepted arguments.

## 4 Grounded Semantics

We begin with grounded semantics. The grounded extension $G$ is the least fixed point of ${ \cal T } _ { \mathcal { F } } ( S ) = \{ a \in A \mid S$ defends $a \}$

Lemma 1 (ω-continuity by Dung [6]). $I f { \mathcal { F } }$ is finitary, $T _ { \mathcal { F } }$ is ω-continuous, $i . e . , T ( \bigcup S _ { i } ) = \bigcup T ( S _ { i } )$ . Consequently, $\begin{array} { r } { G = \bigcup _ { n < \omega } { \cal { T } } ^ { n } ( \emptyset ) } \end{array}$

Proposition 1. Let ${ \mathcal { F } } = \langle A , R \rangle$ be a computably finitary argumentation framework. For any argument $x \in A$ and any finite step $n \in \omega$ , the predicate $x \in T _ { \mathcal { F } } ^ { n } ( \varnothing )$ is decidable $( \varDelta _ { 1 } ^ { 0 } )$ ).

Proof. We proceed by induction on n.

– Base Case $( n = 0 )$ : By definition, $T _ { \mathcal { F } } ^ { 0 } ( \emptyset ) = \emptyset$ . The predicate $x \in \varnothing$ always evaluates to false, which is trivially decidable.

– Inductive Step: Assume that for a given $n \in \omega ,$ , the predicate $y \in T _ { \mathcal { F } } ^ { n } ( \varnothing )$ is decidable for all $y \in A$ . The next iteration of the characteristic operator is defined as:

$$
\Gamma _ { \mathcal { F } } ^ { n + 1 } ( \emptyset ) = \{ x \in A \mid \forall y \in A ( ( y , x ) \in R \to \exists z \in T _ { \mathcal { F } } ^ { n } ( \emptyset ) ( ( z , y ) \in R ) ) \}
$$

Since $\mathcal { F }$ is computably finitary, the set $A t t ( x )$ is finite and its canonical index can be efectively computed. Thus, the universal quantification over $y$ reduces to a finite check over the explicit members of $A t t ( x )$ . For each individual attacker $y \in A t t ( x )$ , we must check if $y$ is counter-attacked by some element currently in $T _ { \mathcal { F } } ^ { n } ( \emptyset )$ . This is equivalent to verifying whether $A t t ( y ) \cap T _ { \mathcal { F } } ^ { n } ( \emptyset ) \neq \emptyset$ . Again, because $\mathcal { F }$ is computably finitary, $A t t ( y )$ is a finite, computable set. $\mathrm { B y }$ the inductive hypothesis, membership in $T _ { \mathcal { F } } ^ { n } ( \emptyset )$ is decidable. Therefore, testing whether the finite set $A t t ( y )$ contains at least one element belonging to $T _ { \mathcal { F } } ^ { n } ( \varnothing )$ takes finite time and is decidable. Thus, the entire predicate $x \overset { \cdot } { \in } \bar { { \cal { T } } } _ { \mathcal { F } } ^ { n + 1 } ( \overset { \cdot } { \varnothing } )$ is decidable $( \varDelta _ { 1 } ^ { 0 } )$

By mathematical induction, the predicate $x \in T _ { \mathcal { F } } ^ { n } ( \varnothing )$ is decidable for any $n \in \omega .$

## 4.1 Credulous and Skeptical Acceptance

Andrews and San Mauro have already established upper and lower bounds for credulous and skeptical acceptance in computably finitary AFs. Their construction for complete semantics directly uses the grounded extension to encode

computably enumerable sets, yielding $\Sigma _ { 1 } ^ { 0 } { \mathrm { \cdot } }$ -completeness of grounded acceptance within $\mathrm { T o t } \times \omega \ [ 1 ]$

Since the grounded extension is unique, credulous and skeptical acceptance coincide. The upper bound can also be seen from ω-continuity and Proposition 1:

$$
a _ { x } \in g r d ( \mathcal { F } _ { e } ) \quad \iff \quad \exists n \in \omega \left( a _ { x } \in T _ { \mathcal { F } _ { e } } ^ { n } ( \emptyset ) \right) .
$$

On valid indices, the finite-stage predicate is decidable uniformly in $e , x , n .$ Equivalently, a finite computation witnessing membership at some stage can be efectively verified. Hence $\operatorname { C r e d } _ { g r d } ^ { \mathrm { c f i n } }$ and $\mathrm { S k e p } _ { g r d } ^ { \mathrm { c f i n } }$ are $\Sigma _ { 1 } ^ { 0 }$ -complete within Tot $\times \omega$

## 4.2 Non-empty Existence

Proposition 2 (Non-empty existence under grounded semantics). $U n -$   
der the computably finitary representation, $N E _ { g r d } ^ { \mathrm { c f i n } }$ is $\Sigma _ { 1 } ^ { 0 }$ -complete within Tot.

Proof. First observe that, for every argumentation framework ${ \mathcal { F } } ,$

$$
g r d ( { \mathcal F } ) \neq \emptyset \quad \iff \quad T _ { { \mathcal F } } ( \emptyset ) \neq \emptyset \quad \iff \quad \exists a \in A \left( A t t _ { { \mathcal F } } ( a ) = \emptyset \right) .
$$

Indeed, $T _ { \mathcal { F } } ( \emptyset )$ consists precisely of the unattacked arguments and is contained in the grounded extension. If there are no unattacked arguments, then $T _ { \mathcal { F } } ( \emptyset ) = \emptyset$ so $\varnothing$ is already a fixed point and $g r d ( \mathcal { F } ) = \emptyset$

For $e \in$ Tot, the attacker set of $a _ { x }$ is encoded by $D _ { \varphi _ { e } ( x ) }$ , and therefore

$$
e \in \mathrm { N E } _ { g r d } ^ { \mathrm { c f i n } } \quad \Longleftrightarrow \quad \exists x \in \omega \left( D _ { \varphi _ { e } \left( x \right) } = \emptyset \right) .
$$

Given a valid index e and an argument code $x ,$ we can compute this finite set and decide whether it is empty. More explicitly, the condition can be expressed as the existence of x and a finite computation in which $\varphi _ { e } ( x )$ halts with output 0, the canonical index of the empty set. This gives a $\Sigma _ { 1 } ^ { 0 }$ definition on the valid representation domain.

For $\Sigma _ { 1 } ^ { 0 }$ -hardness, we $\mathrm { g i }$ ve a many-one reduction from the halting set

$$
K = \{ e \mid \varphi _ { e } ( e ) \mathrm { { \ e v e n t u a l l y } \ h a l t s } \} .
$$

Given $e ,$ construct a framework $\mathcal { F } _ { f ( e ) }$ with argument set $\{ a _ { n } \mid n \in \omega \}$ . For each $s \in \omega .$ , set

$$
A t t ( a _ { 2 s } ) = \left\{ \begin{array} { l l } { { \emptyset , } } & { { \mathrm { i f ~ } \varphi _ { e } ( e ) \mathrm { ~ h a l t s ~ w i t h i n ~ t h e ~ f i r s t ~ } s \mathrm { ~ s t e p s } , } } \\ { { \{ a _ { 2 s + 1 } \} , } } & { { \mathrm { o t h e r w i s e } , } } \end{array} \right.
$$

and set $A t t ( a _ { 2 s + 1 } ) = \left\{ a _ { 2 s + 1 } \right\}$ . Given e and s, simulating $\varphi _ { e } ( e )$ for the finite number of steps s sufices to compute these attacker sets. Their canonical indices are respectively 0 or $2 ^ { 2 s + 1 }$ for $a _ { 2 s }$ , and $2 ^ { 2 s + 1 }$ for $a _ { 2 s + 1 }$ . The construction is uniform, so the s-m-n theorem yields a total computable index map $f ,$ , with $f ( e ) \in$ Tot for every e.

If $e \in K$ and $\varphi _ { e } ( e )$ halts within t steps, then $a _ { 2 s }$ is unattacked for every $s \geq t ,$ and hence $g r d ( \mathcal { F } _ { f ( e ) } ) \neq \emptyset . \mathrm { ~ I f ~ } e \notin K .$ every $a _ { 2 s }$ is attacked by $a _ { 2 s + 1 }$ , and every $a _ { 2 s + 1 }$ attacks itself. Thus there are no unattacked arguments and $g r d ( \mathcal { F } _ { f ( e ) } ) = \varnothing$ Consequently,

$$
e \in { \cal K } \quad \Longleftrightarrow \quad f ( e ) \in \mathrm { N E } _ { g r d } ^ { \mathrm { c f i n } } .
$$

Therefore $\mathrm { N E } _ { g r d } ^ { \mathrm { c f i n } }$ is $\Sigma _ { 1 } ^ { 0 }$ -complete within Tot.

## 4.3 Existence and Uniqueness

Since every argumentation framework has exactly one grounded extension, both Exist $^ { \mathrm { c f i n } } _ { g r d }$ and $\bar { \mathrm { U n i } } _ { g r d } ^ { \mathrm { c f i n } }$ are trivial on the valid representation domain: every $e \in$ Tot is a positive instance.

## 5 Preferred Semantics

We now turn to preferred semantics. A preferred extension is a maximal (w.r.t. $\subseteq$ admissible set. Because every chain of admissible sets has an upper bound, by Zorn’s Lemma, a preferred extension always exists, and any argument contained in an admissible set is also contained in at least one preferred extension.

Lemma 2. Let ${ \mathcal { F } } = \langle A , R \rangle$ be an argumentation framework. For any admissible set $S _ { 0 } ~ \in ~ a d m ( \mathcal { F } )$ , there exists a preferred extension $E \in \mathit { p r e f } ( \mathcal { F } )$ such that $S _ { 0 } \subseteq E$

Proof. Let $\mathcal { M } = \{ S \in a d m ( \mathcal { F } ) \mid S _ { 0 } \subseteq S \}$ be the collection of all admissible sets containing $S _ { 0 }$ . The poset $\langle { \mathcal { M } } , { \subseteq } \rangle$ is partially ordered by set inclusion. We show that every totally ordered chain in M has an upper bound in $\mathcal { M }$

Let $\mathcal { C } \subseteq \mathcal { M }$ be an arbitrary totally ordered chain, and consider its union $U = \textstyle \bigcup _ { X \in { \mathcal { C } } } X$ . Clearly, U is an upper bound of every $X \in { \mathcal { C } }$ under set inclusion. To establish that $U \in \mathcal { M }$ , we must prove that $U$ is an admissible set $( { \mathrm { i . e . } }$ conflict-free and self-defending):

Conflict-freeness of U: Suppose for contradiction that U is not conflictfree. Then, there exist $a , b \in U$ such that $( a , b ) \in R$ . By definition of the union, there exist $X _ { a } , X _ { b } \in \mathcal { C }$ such that $a \in X _ { a }$ and $b \in X _ { b }$ . Since $\mathcal { C }$ is a totally ordered chain, we must have either $X _ { a } \subseteq X _ { b }$ or $X _ { b } \subseteq X _ { a }$ . Assume $X _ { a } ~ \subseteq ~ X _ { b }$ , which implies $a , b \in X _ { b }$ . However, since $X _ { b } \in \mathcal { C } \subseteq \mathcal { M } , X _ { b }$ is admissible and must be conflict-free, contradicting $( a , b ) \in R$ . Thus, $U$ is conflict-free.

– Self-defense of $U { : }$ Let $a \in U$ be an arbitrary argument, and suppose there exists $y \in A$ such that $( y , a ) \in R$ . Since $a \in U$ , there exists some $X _ { a } \in \mathcal { C }$ such that $a \in X _ { a }$ . Since $X _ { a }$ is admissible, it defends $^ { a , }$ meaning there exists $z \in X _ { a }$ such that $( z , y ) \in R$ . Since $X _ { a } \subseteq U$ , we have $z \in U$ . Thus, U defends all its elements.

Since U is both conflict-free and self-defending, $U \in a d m ( { \mathcal { F } } )$ . Furthermore, since $S _ { 0 } \subseteq X$ for all $X \in { \mathcal { C } } .$ , we have $S _ { 0 } \subseteq U$ , which guarantees $U \in \mathcal { M } . \ \mathrm { B y }$ Zorn’s Lemma, $\langle { \mathcal { M } } , { \subseteq } \rangle$ contains at least one maximal element E. By definition, E is an admissible set containing $S _ { 0 }$ . The maximality of E in M ensures that $E \in p r e f ( { \mathcal { F } } )$

For proving the upper bound complexity of decision problems ${ \mathrm { S k e p } } _ { p r e f }$ and $\mathrm { U n i } _ { p r e f }$ , we will utilize the technique from [1,1] to build a bridge between paths in a computably finitary tree and admissible sets, as well as K¨onig’s Lemma.

Definition 6. Any given string $\sigma \in \boldsymbol { \omega } ^ { < \omega }$ defines two subsets of arguments in ${ \mathcal { F } } \colon$

$$
\begin{array} { r l } & { - \mathrm { \normalfont ~ I n } _ { \sigma } = D \cup \{ a _ { j } : \sigma ( 2 j ) = 1 \} \cup \{ a _ { k } : ( \exists j ) ( \sigma ( 2 j + 1 ) = k + 1 ) \} \cup \{ a _ { i } : } \\ & { \quad ( \exists j ) ( \sigma ( 2 j + 1 ) > 0 \land g _ { j } ^ { + } = a _ { i } ) \} ; } \end{array}
$$

$$
{ \begin{array} { r l } & { - \operatorname { O u t } _ { \sigma } = E \cup \{ a _ { j } : \sigma ( 2 j ) = 0 \} \cup \{ a _ { i } : ( \exists j ) ( \sigma ( 2 j + 1 ) > i + 1 \wedge a _ { i } \longmapsto g _ { j } ^ { - } ) \} \cup \{ a _ { i } : } \\ & { \ ( \exists j ) ( \sigma ( 2 j + 1 ) = 0 \wedge g _ { j } ^ { + } = a _ { i } ) \} . } \end{array} }
$$

We define $T _ { \mathcal { F } }$ as the set of σ such that

– In<sub>σ</sub> does not contain elements $a _ { i }  a _ { j }$ with $i , j < | \sigma | ;$

– In<sub>σ</sub> ∩ Out<sub>σ</sub> ∩ {a<sub>i</sub> : i < |σ|} = ∅;

$I f 2 j < | \sigma |$ , then $\sigma ( 2 j ) \in \{ 0 , 1 \}$ ;

$I f 2 j + 1 < | \sigma |$ and $\sigma ( 2 j + 1 ) = k + 1$ , then $a _ { k }  g _ { j } ^ { - }$

Note that if F is computable, then $T _ { \mathcal { F } }$ is computable, and $i f ~ { \mathcal { F } }$ is computably finitary, then $T _ { \mathcal { F } }$ is computably finitely branching.

Theorem 1 (). Let $\mathcal { F } = \langle A , R \rangle$ be a computably finitary argumentation framework and $x \in A$ . There exists a non-empty admissible set $S \in a d m ( \mathcal { F } )$ with $x \in S \mathrm { \ } i f$ and only if the computation tree $T _ { \mathcal { F } }$ contains an infinite path $\pi \in [ T _ { \mathcal { F } } ]$

Proof. See [1] for a complete proof.

Lemma 3 (K¨onig’s Lemma). Let T be an infinite tree. If every node in T has a finite number of children $( i . e . , T$ is finitely branching), then T contains at least one infinite path.

## 5.1 Credulous Acceptance

Lemma 4. Let $\mathcal { F } = \langle A , R \rangle$ be an argumentation framework. An argument $x \in A$ is credulously accepted under the preferred semantics $i f f$ it is credulously accepted under the admissible semantics. That is,

$$
C r e d _ { p r e f } ( \mathcal { F } ) = C r e d _ { a d m } ( \mathcal { F } )
$$

Proof. We establish the identity by bidirectional inclusion:

$- \ ( \subseteq )$ Let $x \in \operatorname { C r e d } _ { p r e f } ( { \mathcal { F } } )$ . There exists a preferred extension $E \in p r e f ( { \mathcal { F } } )$ containing x. Since every preferred extension is admissible by definition $( E \in$ $a d m ( \mathcal { F } ) )$ , E serves as a witnessing admissible set. Thus, $x \in \operatorname { C r e d } _ { a d m } ( { \mathcal { F } } )$

$- \ ( \supseteq )$ Let $x \in \operatorname { C r e d } _ { a d m } ( { \mathcal { F } } )$ . There exists an admissible set $S _ { 0 } \in a d m ( \mathcal { F } )$ containing x. By Lemma 2, there exists a preferred extension $E \in p r e f ( { \mathcal { F } } )$ such that $S _ { 0 } \subseteq E$ . Since $x \in S _ { 0 }$ , we have $x \in E .$ , which implies $x \in { \mathrm { C r e d } } _ { p r e f } ( { \mathcal { F } } )$

Theorem 2. In a computably finitary $A F { \mathcal { F } } , C r e d _ { p r e f } ( { \mathcal { F } } )$ is $I I _ { 1 } ^ { 0 } { - } c o m p l e t e$

Proof. In Dung’s abstract argumentation frameworks, By Lemma 4, for any argument $x \in A$ , x belongs to some preferred extension $E \in p r e f ( \mathcal { F } )$ if x belongs to some admissible set $S \in a d m ( \mathcal { F } )$ . Thus, we have Cred $_ { \mathit { p r e f } } ( { \mathcal { F } } ) = { \mathrm { C r e d } } _ { a d m } ( { \mathcal { F } } )$

By the established results [2], while ${ \mathrm { C r e d } } _ { a d m }$ is $\Sigma _ { 1 } ^ { 1 }$ -complete for general infinite frameworks, introducing the computably finitary restriction reduces its complexity to $\varPi _ { 1 } ^ { 0 }$ . Due to the extensional identity $\mathrm { C r e d } _ { p r e f } ( { \mathcal F } ) = \mathrm { C r e d } _ { a d m } ( { \mathcal F } )$ ， the $\varPi _ { 1 } ^ { 0 }$ -completeness of ${ \mathrm { C r e d } } _ { p r e f }$ follows immediately.

## 5.2 Non-empty Existence

Lemma 5. Let $\mathcal { F } = \langle A , R \rangle$ be an argumentation framework. There exists a nonempty preferred extension $E \in p r e f ( { \mathcal { F } } )$ if there exists a non-empty admissible set $S \in a d m ( \mathcal { F } )$

Proof. We prove the equivalence in both directions:

( =⇒ ) Assume there exists a non-empty preferred extension $E \in p r e f ( { \mathcal { F } } )$ Since every preferred extension is an admissible set, $E \in a d m ( { \mathcal { F } } )$ . Because $E \neq$ ∅, it immediately serves as a witnessing non-empty admissible set.

$( \Leftarrow )$ Assume there exists a non-empty admissible set $S _ { 0 } \in a d m ( \mathcal { F } )$ . By Lemma 2, there exists a preferred extension $E \in p r e f ( { \mathcal { F } } )$ such that $S _ { 0 } \subseteq E$ Since $S _ { 0 } \ne \emptyset$ , it follows immediately that $E \neq \emptyset$

Theorem 3. In a computably finitary $A F { \mathcal { F } } ,$ the non-empty existence problem under preferred semantics $( N E _ { p r e f } )$ is $\Sigma _ { 2 } ^ { 0 }$ -complete.

Proof. By Lemma 5, an argumentation framework F contains a non-empty preferred extension if and only if it contains a non-empty admissible set. This equivalence immediately establishes an equivalence between the decision problems of non-empty existence under preferred and admissible semantics. That is, $\mathrm { N E } _ { p r e f } = \mathrm { N E } _ { a d m }$

Since the non-empty existence problem under admissible semantics $\left( \mathrm { N E } _ { a d m } \right)$ has already been established as $\scriptstyle \sum _ { 2 } ^ { 0 } - \mathrm { c o m p l e t e }$ in computably finitary $\mathrm { A F s . }$ it follows directly that $\mathrm { N E } _ { p r e f }$ is also $\Sigma _ { \mathrm { 2 } } ^ { 0 } \mathrm { - c }$

## 5.3 Skeptical Acceptance

So far all problems remain arithmetical; skeptical acceptance is the turning point. Under the preferred semantics, an argument is skeptically accepted if and only if it belongs to every preferred extension (defined as maximal admissible sets under set inclusion).

Lemma 6. A path $P \in [ T _ { \mathcal { F } } ]$ corresponds to a preferred extension if for every $n \in \omega$ , and for every child node τ of $P \ \lceil \ n$ in $T _ { \mathcal { F } } .$ :

$I f \tau \neq P \mid ( n + 1 )$ , then the subtree rooted at $\tau$ contains no infinite path.

Proof. Let $P , T \in [ T _ { \mathcal { F } } ]$ be two infinite paths corresponding to admissible sets $S$ and $S ^ { \prime } ,$ respectively. The set inclusion $S \subset S ^ { \prime }$ corresponds to $P ( 2 n ) = 1 \implies$ $T ( 2 n ) = 1$ , which also corresponds to $P$ and $T$ share a prefix. Thus, the maximality of an admissible set $S \ ( \mathrm { i . e . , } \ S$ is a preferred extension) can be characterized by $T _ { \mathcal { F } }$ as a path where there is no divergent infinite branch.

Theorem 4. In a computably finitary argumentation framework $\mathcal { F }$ , the skeptical acceptance problem under preferred semantics $( S k e p _ { p r e f } )$ is in $\varPi _ { 1 } ^ { 1 }$

Proof. An argument $x \in A$ fails to be skeptically accepted with respect to preferred semantics $( x \not \in \operatorname { S k e p } _ { p r e f } ( { \mathcal { F } } ) )$ if and only if there exists a preferred extension $E \in p r e f ( { \mathcal { F } } )$ such that $x \notin E$ . The admissible sets correspond to the set of infinite paths $[ T _ { \mathcal { F } } ]$ through a computation tree $T ^ { \mathcal { F } }$ . Using Lemma $6 ,$ the non-skeptical acceptance problem x $\not \in \mathrm { S k e p } _ { p r e f } ( \mathcal { F } )$ is equivalent to the assertion:

There exists an infinite path $P \in [ T _ { \mathcal { F } } ]$ such that its corresponding admissible set excludes argument $x ,$ and for any node $\sigma = P \uparrow n .$ , every child $\tau \neq P \ \lceil \ ( n + 1 )$ roots a subtree $\left( T _ { \mathcal { F } } \right)$ <sub>τ</sub> that eventually dies out $( { \mathrm { i . e . } }$ contains no infinite paths).

The condition that a sub-tree contains no infinite paths is a $\varPi _ { 1 } ^ { 0 }$ property by K¨onig’s Lemma. However, asserting the existence of an infinite path $P$ with a maximality condition that evaluates infinitely many subtrees introduces an existential quantification over infinite paths $\left( \exists P \in \left[ T _ { \mathcal { F } } \right] \right)$ , placing the non-skeptical acceptance problem in the $\Sigma _ { 1 } ^ { 1 }$ level of the analytical hierarchy. Taking the complement, the skeptical acceptance problem $x \in \mathrm { S k e p } _ { p r e f } ( \mathcal { F } )$ is expressible as a $\varPi _ { 1 } ^ { 1 }$ formula.

## 5.4 The Uniqueness Problem

Lemma 7. Let ${ \mathcal { F } } = \langle A , R \rangle$ be an argumentation framework. $\mathcal { F }$ has a unique preferred extension $i f f$ there exists $E \in$ adm(F), for all $S \in a d m ( { \mathcal { F } } ) , S \subseteq E$

Proof. Proof is straightforward by the definition of preferred extensions.

Theorem 5. In a computably finitary argumentation framework $\mathcal { F } _ { : }$ , the uniqueness problem for preferred extensions $( U n i _ { p r e f } )$ is in $\Sigma _ { 2 } ^ { 1 }$

Proof. Let $T _ { \mathcal { F } }$ be the computable, finitely branching tree whose infinite paths $[ T _ { \mathcal { F } } ]$ correspond to all admissible sets in $\mathcal { F }$

Encoding argument membership(i.e., let $S$ be an admissible set and $P$ be its corresponding path, argument $a _ { n } \in S \iff P ( 2 n ) = 1 )$ , a framework $\mathcal { F }$ belongs

to $\mathrm { U n i } _ { p r e f }$ if and only if there exists a witness path $P \in [ T _ { \mathcal { F } } ]$ such that every path $T \in [ T _ { \mathcal { F } } ]$ is contained in $P .$ Formally:

$$
{ \mathcal { F } } \in { \mathrm { U n i } } _ { p r e f } \ \Longleftrightarrow \ \exists P \in [ T _ { { \mathcal { F } } } ] \forall T \in [ T _ { { \mathcal { F } } } ] \forall n \in \omega \left( T ( 2 n ) = 1 \implies P ( 2 n ) = 1 \right)
$$

Analyzing the quantifier complexity of this formalization:

1. The inner relation ∀n $\in \omega \left( T ( 2 n ) = 1 \implies P ( 2 n ) = 1 \right)$ evaluates a universal quantifier over the natural numbers on a decidable relation, forming a $\varPi _ { 1 } ^ { 0 }$ predicate.

2. Universal quantification over all infinite paths $T \in \ [ T _ { \mathcal F } ]$ adds a universal second-order quantifier.

3. Existential quantification over the witness path $P \in [ T _ { \mathcal { F } } ]$ adds an existential second-order quantifier.

Consequently, the quantifier prefix takes the form $\exists P \forall T \forall n ,$ , which places the uniqueness problem $\mathrm { U n i } _ { p r e f }$ precisely in the $\Sigma _ { 2 } ^ { 1 }$ tier of the analytical hierarchy.

Theorem 6 (Σ<sup>1</sup>-Hardness of Uniqueness). In computably finitary argumentation frameworks, the uniqueness problem for preferred extensions $U n i _ { p r e f }$ is Σ<sup>1</sup><sub>2</sub>-hard.

Proof. We reduce a canonical $\Sigma _ { 2 } ^ { 1 }$ -complete set E to $\mathrm { U n i } _ { p r e f }$ . Let $e \in { \mathcal { E } }$ be represented as:

$$
e \in { \mathcal { E } } \ \Longleftrightarrow \ \exists P \in 2 ^ { \omega } \forall T \in 2 ^ { \omega } \exists n \in \omega \ R ( e , P \mid n , T \mid n )
$$

where R is a computable relation over finite sequences. We construct an infinite argumentation framework $\mathcal { F } _ { e } = \langle A , A t t \rangle$ corresponding to $e .$

The set of arguments A is defined as:

$$
A = \{ p _ { \sigma } , e _ { \sigma } ^ { P } \mid \sigma \in 2 ^ { < \omega } \} \cup \{ t _ { \tau } , e _ { \tau } ^ { T } \mid \tau \in 2 ^ { < \omega } \} \cup \{ U \}
$$

where $p _ { \sigma }$ and $t _ { \tau }$ represent finite path choices in the candidate tree $P$ and challenger tree $T$ respectively, $e _ { \sigma } ^ { P }$ and $e _ { \tau } ^ { T }$ act as enforcer arguments, and $U$ is the global dominator argument.

The attack relation Att consists of the following pairs for all $\sigma , \tau \in 2 ^ { < \omega }$ :

1. $p _ { \sigma \cdot 0 }  p _ { \sigma \cdot 1 }$ and $t _ { \tau \cdot 0 }  t _ { \tau \cdot 1 }$

2. $e _ { \sigma } ^ { P } \to p _ { \sigma }$ with children counter-attacking $p _ { \sigma \cdot 0 }  e _ { \sigma } ^ { P }$ and $p _ { \sigma \cdot 1 } \to e _ { \sigma } ^ { P }$ . Similarly, $e _ { \tau } ^ { T } \to t _ { \tau }$ with $t _ { \tau \cdot 0 } \to e _ { \tau } ^ { T }$ and $t _ { \tau \cdot 1 } \to e _ { \tau } ^ { T }$

3. For $| \sigma | = | \tau | = n , { \mathrm { i f ~ } } R ( e , \sigma , \tau )$ holds (is True), then $p _ { \sigma }  t _ { \tau }$

4. For all $\tau \in { 2 ^ { < \omega } , t _ { \tau } \to U }$ . Conversely, U attacks all non-root tree arguments: $U  p _ { \sigma } , e _ { \sigma } ^ { P } , t _ { \tau } , e _ { \tau } ^ { T }$ for all $\sigma \neq \emptyset , \tau \neq \emptyset$

We show that $e \in \mathcal { E } \iff \mathcal { F } _ { e } \in \operatorname { U n i } _ { p r e f }$

(⇒) Suppose $e \in { \mathcal { E } }$ . Then there exists a witness path $P ^ { * } \in 2 ^ { \omega }$ such that for every challenger path $T \in 2 ^ { \omega }$ , there is an integer $n _ { T }$ where $R ( e , P ^ { * } \mid n _ { T } , T \mid n _ { T } )$ holds. Let $S _ { P ^ { * } } = \{ p _ { P ^ { * } \uparrow k } \ | \ k \in \omega \}$ . For any path $T ,$ , the attack $p _ { P ^ { * } \mathsf { f } n _ { T } } \to t _ { T \mathsf { f } n _ { T } }$ defeats argument $t _ { T \lceil n _ { T } \rceil }$ . Consequently, $t _ { T \lceil n _ { T } \rceil }$ fails to defend against $e _ { T \mid n _ { T } - 1 } ^ { T } ,$ which in turn defeats $t _ { T \lceil n _ { T } - 1 }$ . By induction, this collapse propagates upward to the root, destroying all admissible sets along $T .$ . Since every challenger node $t _ { \tau }$ is either directly defeated by $S _ { P } ,$ or falls to a cascading collapse, no $t _ { \tau }$ survives to attack $U .$ . Thus, $S _ { P ^ { * } }$ defends $U ,$ making $U$ admissible. The set $E ^ { * } = S _ { P ^ { * } } \cup \{ U \}$ forms a preferred extension. Since $U \in E ^ { * }$ defeats all alternative candidate paths in the $P -$ tree and $T \cdot$ -tree, $E ^ { * }$ is the unique preferred extension of $\mathcal { F } _ { e }$

![](images/55c727906055a2576782881b9966fb42d5116569014b138ac5d0c3d258991450.jpg)

(⇐) Suppose $e \not \in \mathcal { E }$ . Then for every path $P \in 2 ^ { \omega }$ , there exists a counterpath $T _ { P } \in 2 ^ { \omega }$ such that $R ( e , P \ \mid \ n , T _ { P } \ \mid \ n )$ is False for all $n \in \omega .$ . For any candidate path $P _ { : }$ the argument set $S _ { P } = \{ p _ { P \mid k } ~ | ~ k \in \omega \}$ fails to launch any attack against $T _ { P }$ . Thus, the nodes $\left\{ t _ { T _ { P } \cap \{ n \} } ~ n ~ \in ~ \omega \right\}$ remain undefeated and self-defending against their enforcers. Since these nodes continuously attack $U$ $( t _ { T _ { P } \cap \mathop { n } } \to U )$ ， $U$ can never be defended and cannot belong to any admissible set. Without $U$ to eliminate competing choices, each distinct path $P \in 2 ^ { \omega }$ paired with its counter-path $T _ { P }$ yields a distinct preferred extension:

$$
E _ { P } = \{ p _ { P \mid k } \mid k \in \omega \} \cup \{ t _ { T _ { P } \mid n } \mid n \in \omega \}
$$

Because there are uncountably many choices for $P , \mathcal { F } _ { e }$ possesses uncountably many preferred extensions. Hence, $\mathcal { F } _ { e } \notin \mathrm { U n i } _ { p r e f }$

This completes the reduction and proves that $\mathrm { U n i } _ { p r e f }$ is $\Sigma _ { 2 } ^ { 1 }$ -hard.

Theorem 7. $U n i _ { p r e f }$ is $\Sigma _ { 2 } ^ { 1 }$ -complete.

Proof. Hardness follows from Theorem 6 (reduction from the canonical $\Sigma _ { 2 } ^ { 1 } \mathrm { - }$ complete problem). For membership, $\mathcal { F } \in \mathrm { \mathrm { ~ U n i } } _ { p r e f }$ if and only if there exists a preferred extension $E _ { 1 }$ such that all preferred extensions $E _ { 2 }$ equal $E _ { 1 }$ . Since being a preferred extension is a $\varPi _ { 1 } ^ { 1 }$ property, as in Theorem $5 ,$ the predicate $\exists ^ { 1 } E _ { 1 } ( { \mathrm { P r e f } } ( E _ { 1 } ) \land \forall ^ { 1 } E _ { 2 } ( { \mathrm { P r e f } } ( E _ { 2 } )  { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } E _ { 1 } = E _ { 2 } ) )$ contains two alternating secondorder quantifiers $\big ( \exists ^ { 1 } \forall ^ { 1 } \big )$ , placing $\mathrm { U n i } _ { p r e f }$ in $\Sigma _ { 2 } ^ { 1 }$

## 6 Related Work

The formal study of computability and expressiveness in infinite argumentation was pioneered by Dung [6], who established fundamental properties such as the ω-continuity of the characteristic function in finitary frameworks. Baumann and Spanring [5]established the theoretical baseline for unrestricted AFs (UAFs), examining whether fundamental semantic properties hold when argument sets are infinite. A key structural challenge in UAFs is the emergence of infinitely ascending Strongly Connected Component chains, which break the classical SCC-recursive schema [4]. To address these, Andrews and San Mauro [3] proposed repairs using compactness techniques and ordinal-indexing. The computational landscape transitions significantly when moving from finite graphs to infinite structures: Firstly, in Unrestricted ${ \mathrm { I A F s } } ,$ complexity shifts to the analytical hierarchy. Decidability for primary semantics (admissible, stable, complete) inhabits $\Sigma _ { 1 } ^ { 1 }$ -completeness or $\varPi _ { 1 } ^ { 1 }$ -completeness [2]. Then, under finitary restrictions, the characteristic operator becomes ω-continuous, causing a complexity collapse into the arithmetical hierarchy [1]. Andrews and San Mauro further showed that stable extensions can code arbitrary $\varPi _ { 1 } ^ { 0 }$ classes, proving that IAFs are computationally universal for tree-search problems [2].

## 7 Conclusions

In this paper, we establish a computational classification for computably finitary AFs:

– Grounded Semantics: Credulous and skeptical acceptance coincide and are $\scriptstyle \sum _ { 1 } ^ { 0 } - \mathrm { c o m p l e t e }$ within Tot $\times \omega ,$ as established by Andrews and San Mauro [1]. Proposition 2 shows that non-empty existence is $\Sigma _ { 1 } ^ { 0 }$ -complete within Tot. Existence and uniqueness are trivial on valid indices.

Preferred Semantics: Some problems like Cred $_ { p r e f } \ ( { \cal I } _ { 1 } ^ { 0 } { \mathrm { - c } } )$ and $\mathrm { N E } _ { p r e f } ~ ( \Sigma _ { 2 } ^ { 0 } { \mathrm { - c } } )$ remain arithmetical. In contrast, others escape the arithmetical hierarchy: universal quantification over infinite extensions pushes Skep $) _ { p r e f }$ to $\varPi _ { 1 } ^ { 1 }$ , while ruling out competing infinite branches renders $\mathrm { U n i } _ { p r e f } \ \sum _ { 2 } ^ { 1 } - \dot { \bf c } .$

Several directions remain open. While we showed $\mathrm { S k e p } _ { p r e f } \in \varPi _ { 1 } ^ { 1 }$ , its exact completeness level within the analytical hierarchy is still undetermined. Finally, it remains to investigate whether similar collapses occur for other semantics.

## References

1. Andrews, U., Mauro, L.S.: Complexity in finitary argumentation (extended version). arXiv preprint arXiv:2508.16986 (2025)

2. Andrews, U., San Mauro, L.: On computational problems for infinite argumentation frameworks: The complexity of finding acceptable extensions. Ceur Workshop Proceedings (2024)

3. Andrews, U., San Mauro, L.: On the complexity of the grounded semantics for infinite argumentation frameworks. In: Bjorndahl, A. (ed.) Proceedings Twentieth Conference on Theoretical Aspects of Rationality and Knowledge, D¨usseldorf, Germany, July 14-16, 2025. Electronic Proceedings in Theoretical Computer Science, vol. 437, pp. 112–127. Open Publishing Association (2025). https://doi.org/10. 4204/EPTCS.437.13

4. Baumann, R., Spanring, C.: Infinite argumentation frameworks: On the existence and uniqueness of extensions. In: Advances in Knowledge Representation, Logic Programming, and Abstract Argumentation: Essays Dedicated to Gerhard Brewka on the Occasion of his 60th Birthday, pp. 281–295. Springer (2015)

5. Baumann, R., Spanring, C.: A study of unrestricted abstract argumentation frameworks. In: International Joint Conference on Artificial Intelligence (2017), https: //api.semanticscholar.org/CorpusID:34510332

6. Dung, P.M.: On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial intelligence 77(2), 321–357 (1995)