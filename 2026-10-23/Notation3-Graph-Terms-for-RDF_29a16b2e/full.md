# Notation3: Graph Terms for RDF

Dörthe Arndt # Ñ

TU Dresden, Germany

ScaDS.AI Dresden/Leipzig, Germany

Pierre-Antoine Champin # Ñ

W3C

Inria, University Côte d’Azur, CNRS, I3S UMR 7271, France

## Abstract

Notation3 Logic was introduced as an extension of RDF allowing the user to express rules on top of triples and to make statements about graphs. The original proposal of the logic only came with an informal specification of its semantics which has been refined later on. While this refinement clarified the meaning of rules and implicitly quantified variables, it left the meaning of graphs used as terms

– so-called graph terms – open. In this paper we aim to close this gap. We discuss in detail, how graph terms are used in Notation3, which formal requirements they need to meet, and provide examples to illustrate the dificulties when defining their semantics. We then introduce an abstract

syntax for the logic and define its semantics in two steps: similarly to RDF’s simple interpretations, we define interpretations and basic entailment for Notation3 which covers plain RDF triples but also triples containing graph terms. N3-entailment is then an extension of basic entailment which assigns meaning to the diferent special predicates of the logic like for example the implication symbol. N3-entailment is defined in a way that it can be extended whenever a new predicate is added to the logic. As Notation3 Logic is currently under development, we expect many predicates to be added. Our research provides and important step towards the full standardisation of the logic.

2012 ACM Subject Classification Information systems → Resource Description Framework (RDF); Theory of computation → Logic and verification; Theory of computation → Semantics and reasoning

Keywords and phrases Notation3 Logic, Graph terms, Semantics, Rules, RDF

Digital Object Identifier 10.4230/TGDK.1.1.42

Supplementary Material The authors declare that this article involves no relevant supplementary resources.

Acknowledgements The authors would like to thank all the participants of the W3C Notation 3 Community Group<sup>a</sup> for fruitful discussions and their feedback on early stages of this work.

Received Date of submission Accepted Date of acceptance Published Date of publishing Editor TGDK section area editor

<sup>a</sup> https://www.w3.org/community/n3-dev/

## 1 Introduction

Ever since the invention of the Semantic Web, the community saw the need to allow users to make statements about statements. Diferent approaches like singleton properties [31], the inclusion of named graphs to RDF [30, Section 4], or the integration of RDF-star (formally RDF\*) [21, 22] into the upcoming RDF 1.2, are only some indicators for it. And while these proposals and standards to some extend address the problem, they all have some draw-backs: singleton properties and RDF-star only support statements about triples but not about RDF graphs and for named graphs, there is so far no agreement about their semantics [39].

In this paper we focus on another framework which supports statements about statements: Notation3 Logic (N3) [13]. N3 was proposed by Tim Berners-Lee and others to provide a simple syntax for RDF – this resulted in RDF-Turtle [32] – but also as a logical extension for RDF. N3

## 42:2 Notation3: Graph Terms for RDF

supports so-called graph terms which are conceptually very similar to RDF-star’s triple terms with the only diference that N3 does not only allow triples as terms but entire graphs. Of course, this comes with challenges: graph terms could stand for themselves or for bigger graphs, scoping of blank nodes is far from trivial, and graph terms are used to express logical constructs. When originally proposed, the semantics of N3 has only been defined informally and diferent implementations like Cwm [10], EYE [37] or Data-Fu [27] were developed to reason on it. The theoretical works which followed later on to define a model theoretic semantics for N3 did focus on rules [3, 4] and on implicit quantification [5], but did not fully cover the semantics of graph terms as it was originally intended.

In this paper we aim to fill this gap: We analyse the requirements a graph term semantics has to comply with according to the informal specifications and the existing implementations. We then propose an abstract syntax for N3 and define its semantics in two steps: we first specify a base semantics which covers the meaning of graph terms but does not assign any meaning to the built-in predicates of the logic. In that sense our base semantics is very similar to RDF’s simple entailment. In a second step we define N3-semantics which covers the logical predicates in N3’s log-namespace<sup>1</sup>. Here we mainly focus on logical implication (log:implies) and on graph inclusion (log:includes). Especially the former is crucial for N3 as this predicate enables rule-based reasoning. We furthermore show that our definitions are in-line with the requirements.

The remainder of this paper is structured as follows: In Section 2 we analyse how graph terms are used in N3, and we derive the requirements for their semantics in Section 3. In Section 4 we define an abstract syntax for N3 and explain how it maps to the concrete serialization syntax. In Section 5 we then provide a formal semantics for the abstract syntax, and we show in Section 6 that the semantics of graph terms meets our requirements. In Section 7 we discuss the related work and explain how our formalisation relates to other approaches to enable statements about statements in RDF. With Section 8 we conclude our paper.

## 2 Graph terms in N3

Before formally introducing N3 and our proposal for a graph term semantics, we give a few examples to make the reader familiar with the logic. N3 extends the Turtle syntax for RDF [33], which means that each RDF turtle triple is also an N3 triple. We can, for example, state that<sup>2</sup>

:alice :likes :pasta.

(1)

This triple informally means that “Alice likes pasta.” Just as in RDF, we have literals and IRIs which refer to resources, and blank nodes which are used as existentially quantified variables. N3 furthermore supports graph terms. We can, for example, state that

:bob :says {:alice :likes :pizza, :pasta}.

(2)

Which informally means that “Bob says that Alice likes pizza and pasta.” Note that, while RDF-star only allows a single triple to be used as a term, here the object is a set of triples which we interpret as a conjunction. This makes it possible to have co-referring blank nodes in graph terms. We could modify the previous statement and get:

$$
\mathrm  : b o b ~ : s a y s ~ \ell \ell _ { - } : x ~ : l i k e s ~ : p i z z a . ~ \ell _ { - } : x ~ : l i k e s ~ : p a s t a \} .\tag{3}
$$

Despite being a little bit vague, we could give an informal interpretation before. In this case it is more complex because there is implicit existential quantification involved. In our proposal, the above means “Bob says that there exists someone who likes pizza and pasta”. The blank node is existentially quantified within the graph term and not outside of it<sup>3</sup>. It has to be said, that this interpretation is still subject to discussion, which is why a semantics of N3 needs to be flexible to be able to adjust possible future changes in the scoping of implicit quantification.

Graph terms are not only used to make statements about statements but also to express logical operations. A rule can be written as follows:

$$
\{ \mathfrak { T x } \ : \mathrm { { 1 i k e s } \ : p a s t a \ } \} \ = > \ \{ \mathfrak { T x } \ : \mathrm { { 1 i k e s } \ : s p a g h e t t i \ } \} .\tag{4}
$$

In this rule, the ?x stands for a (implicitly) universally quantified variable and the sign => is syntactic sugar for the N3 predicate log:implies. The above rule informally means that “If someone likes pasta, then they also like spaghetti.”. We see that the rule above is simply a triple with graph terms as subject and object and a special predicate log:implies. This is a design choice of N3: All data and axioms in N3 are expressed through triples, and all semantic decisions (e.g., scoping of variables) we take for graph terms in statements like Formula 2 also apply to the graph terms occurring in rules like Formula 4 above.

We illustrate this by providing the following example:

$$
\{ : { \mathrm { b o b } } : { \mathrm { s a y s } } ? { \mathrm { x } } \} \ \Rightarrow \ ? { \mathrm { x } } \ .\tag{5}
$$

This can be freely interpreted as “Everything Bob says is true.” In order to use this formula, apply it to Formula 2 and derive the triples

$$
: \mathsf { a l i c e \ t : l i k e s \ t s : p a s t a , \ t p i z z a . }\tag{6}
$$

we need to consider the dual nature of Formula 5. Understanding the IRI log:implies as a predicate, we have an atomic formula (in first-order-like syntax it would be ∀x.implies(says(bob, x), x)) and we can unify the universal variable ?x with the graph term {:alice :likes :pasta, :pizza.}. As a consequence, we get the formula

$$
\begin{array} { r l }  \{ : \mathrm { b o b ~ : s a y s ~ \{ : a l i c e ~ : l i k e s ~ p a s t a , ~ : p i z z a . \} ~ \} } & { } \\ &  = > \{ : \mathrm { a l i c e ~ \ x l i k e s ~ : p a s t a , ~ : p i z z a . \} . } \end{array}\tag{7}
$$

This, in turn, can now be seen as an implication and applied on Formula 2 it yields Formula 6.

## 3 Requirements on a formal semantics

The examples in the previous section reveal that there are several dificulties and uncertainties when dealing with graph terms in N3. Some requirements on N3’s semantics only become clear after further analysing the logic. Below, we list and explain the main requirements to be taken into account when defining the semantics of N3, with a special focus on graph terms.

Semantic compatibility with RDF N3 syntactically extends RDF Turtle and each RDF Turtle triple is also an N3 triple. N3 semantics should thus be compatible with RDF semantics. As N3 does not assign special semantics to the diferent predicates of the RDF and RDFS namespace (e.g., rdf:type, rdfs:domain), we limit this requirement to compatibility withe RDF

## 42:4 Notation3: Graph Terms for RDF

simple entailment. N3 should additionally support D-Entailment, but we leave that detail out of the considerations of this paper as our main focus is on graph terms.

Referential Opacity One detail which often leads to discussion when a logic supports statements about statements is the question whether or not a constant occurring in a formula term (as a generalised graph term) refers to the same resource as it does in a stated formula (e.g. [22, Section 6.4.4]). This is often illustrated by the superman example which in N3 syntax would be:

:superman owl:sameAs :clarkKent.

(8)

$$
\mathrm { : l o i s L a n e : k n o w s ~ \lbrace : s u p e r m a n ~ : c a n ~ : f 1 y \rbrace } .\tag{9}
$$

Above, we borrowed the predicate owl:sameAs from OWL [38] to express equality. The above means in a free translation: “Superman is the same as Clark Kent and Lois Lane knows that Superman can $\mathit { \Pi } _ { f l y . \mathit { \Pi } ^ { \prime \prime } }$ Under the assumption that we support equality, we ask the question whether we can derive that

$$
\mathrm { : l o i s L a n e : k n o w s ~ \{ : c l a r k K e n t ~ : c a n ~ : f 1 y \} }\tag{10}
$$

or in words “Lois Lane knows that Clark Kent can $\mathit { \Pi } ^ { f l y . ^ { \prime \prime } }$ In case of a positive answer, we say that graph terms are transparent, otherwise they are opaque.

Opaque graph terms are considered a better design choice for N3. Firstly, as the Superman example above demonstrates, there are use-cases where the derivations entailed by transparency are not desirable. Secondly, if such derivations are desired, it is still possible to enable them by adding a rule of the form:

$$
\{ \{ \texttt { T x } \ \mathrm { o u l } : \mathrm { s a m e A s } \ \mathrm { ~ ? y } , \quad \texttt { ? s } \ \mathrm { ~ ? p } \ \{ \mathrm { ~ ? x } \ \mathrm { ~ ? q } \ \mathrm { ~ ? r } \} \} = > \{ \mathrm { ~ ? s } \ \mathrm { ~ ? p } \ \{ \mathrm { ~ ? y } \ \mathrm { ~ ? q } \ \mathrm { ~ ? r } \} \} \} .\tag{11}
$$

or a variant thereof. Referential opacity is a weaker, less constraining, option than transparency. Closedness of graph terms Another detail to be discussed in the context of graph terms is whether we consider them closed or open. By this we mean whether the term {:clarkKent :can :fly} from Formula 10 refers to a graph (a set of triples) containing exactly one triple (closed), namely :clarkKent :can :fly. or to a potentially bigger graph containing at least that triple – but may contain others (open). Both options have been considered as proposed semantics for datasets in RDF [39]. Consider the rule expressed in Formula 7. If the graph term on the left-hand side could denote a graph containing arbitrary extra triples, there would be no way to prove it to be satisfied by the currently known triples, and therefore no way to derive the consequence of that (or any) rule. Therefore, we consider graph terms to be closed in our semantics.

Clarity and flexibility of quantification scopes N3 has two kinds of variables: existentially quantified (or blank nodes, inherited from RDF) and universally quantified (in order to express rules, as illustrated earlier). When combined with graph terms, this immediately raises the question of the scope of these quantifications. Historically, blank nodes (such as \_:y) have been scoped to the innermost graph containing them (“locally scoped”), while universal variables (such as ?x) have been scoped to the top-level graph (“globally scoped”). While this fits most use-cases, those scoping rules have always been a topic of discussions and a point of divergence among diferent implementations of N3 [5]. For example, they do not tell us precisely how to interpret the following simple graph:

$$
\ ? \mathbf { x } \ : \mathsf { l o v e s \ \_ : y . }\tag{12}
$$

We know that both variables are scoped to the same (top-level) graph, but it is not clear how they interact. Without additional information, the reader could either interpret the above as (a) “Everybody loves somebody” or (b) “There is someone who is loved by everyone”.

Even if we specify the scoping rule to disambiguate the example above (and we will, making (a) the correct interpretation), we also need a way to explicitly express the quantification scope of a given variable when the implicit scoping rule are not adequate. Besides the example of interpretation (b) above, consider the example below where we want to describe, in an N3 graph, the content of another N3 document:

$$
< \mathrm { e x a m p l e . n 3 > \mathrm { ~ \ l o g : s e m a n t i c s ~ \{ ~  ~ \emptyset \mathrm { f o r A l ~  ~ : x . ~  ~ : x ~  ~ : l o v e s ~  ~ : y ~  ~ \} ~ } }  }\tag{13}
$$

Here, we could not have used the implicitly quantified universal ?x, because its scope would have been the whole graph, while what we want to convey is that it is scoped to the graph term representing the content of example.n3. Explicit quantification was also introduced early in N3 (with the keywords @forAll and @forSome), but like the implicit scoping rules, their interpretations and implementations have varied largely.

## 4 Syntax

Before defining the semantics of N3, we introduce its syntax. The concrete syntax of N3, which is used to illustrate examples throughout this paper, is described in the draft report<sup>4</sup> of the W3C Notation 3 Community Group. Space limitations prevent from presenting it in details, but the examples provided in Section 2 present its main features, such as graph terms (Formula 2) and rules (Formula 4).

To address the last requirement from above, the flexibility to express diferent quantification scopes, but also to have a way to easily deal with quantification, we define an abstract syntax. Below, we introduce this abstract syntax, explain its relation to concrete syntax and provide the syntactic definitions needed to define the semantics.

## 4.1 Abstract Syntax

We first define atomic triples, the building blocks of our abstract syntax.

▶ Definition 1 (Atomic Term). The set of atomic terms consists of

the set C of IRIs,

the set L of Literals,

the set V of Variables.

As we can see, the abstract syntax has only one kind of variable. It does not distinguish between existential (blank nodes) and universal variables, because contrarily to the concrete syntax, all variables in the abstract syntax will be explicitly quantified.

▶ Definition 2 (Term). A term can be

an atomic term,

a graph term G where G is a finite graph (see below).

Note that we make a clear diference between a graph and a graph term. This diference will become important in Section 5.

▶ Definition 3 (Triple). A triple is a 3-tuple of terms, respectively called the subject, predicate and object of the triple.

## 42:6 Notation3: Graph Terms for RDF

Note that, in contrast to RDF, N3 allows for any kind of term in all positions of a triple.

While in RDF a graph is simply defined as a set of triples, N3 needs a way to specify which variables are universally or existentially quantified, and in which scope.

▶ Definition 4 (Graph). An N3 graph (or simply graph) is a tuple (U, E, F) where

$U \subseteq V$ is a (possibly empty) set of universally quantified variables,

$E \subseteq V$ is a (possibly empty) set of existentially quantified variables,

U and E are disjoint,

F is a (possibly empty) set of triples.

A graph $G = ( U , E , F )$ is finite if the sets U, E and F are finite.

Note that this model is a strict superset of RDF: any RDF graph can be seen as an N3 graph, where F is the set of triples consituting the graph, $U = \emptyset$ , and E is the set of all blank nodes used in F. Of course, an RDF graph will use no graph term in F.

▶ Remark (Notation). Consider a graph $G ~ = ~ ( U , E , F )$ . To improve readability, in our examples, we will use the quantifier ∀ to mark the set of universally quantified variables and the quantifier ∃ to mark the set of existentially quantified variables, and ommit those sets when they are empty. For example, ({x}, {y}, {(x, :likes, y)}) will be noted $\forall \{ x \} \exists \{ y \} \{ ( x , \mathrm { { : l i k e s } } , y ) \} , ( \{ x \} , \{ \} , \{ ( x , \mathrm { { : l i k e s } } , \mathrm { { : p i z z a } } ) \} )$ will be noted ∀{x} {(x, :likes, :pizza)}, and ({}, {}, {(:alice, :likes, :pizza)}) will be noted {(:alice, :likes, :pizza)}.

## 4.2 From concrete to abstract syntax

From a syntactic point of view, N3 can be seen as a superset of Turtle [32], with which we assume the reader to be familiar. The main N3-specific features of the concrete syntax were introduced in Sections 2. N3 supports implicitly scoped universal variables, starting with a question mark (e.g. ?x); graph terms, enclosed in curly brackets ({}); and the symbol => which is an alias for the predicate log:implies. IRIs in the concrete syntax translate to IRIs (C) in the abstract syntax (except when quantified with the keywords @forAll and @forSome, see below). Literals in the concrete syntax translate to literals (L) in the abstract syntax. Blank nodes and ?-prefixed universals translate to variables (V ). Diferent concrete symbols are translated to distinct variables in the abstract syntax. In particular ?x and \_:x are not the same and if we modify Formula 12 as follows

?x :loves \_:x.

(12<sup>′</sup>)

it corresponds to the abstract N3 graph

$$
\forall \{ x _ { 1 } \} \exists \{ x _ { 2 } \} \{ ( x _ { 1 } , : \mathrm { l o v e s } , x _ { 2 } ) \}\tag{14}
$$

For the translation of implicit quantification, we adhere to the following schema: the variable corresponding to any ?-prefixed universals is automatically added to the U set of the outermost graph. The variable corresponding to a blank node is automatically added to the E set of the innermost graph. For example, in the RDF triple

\_:x :likes :pasta.

(15)

we have only one level, and the blank node is quantified there, we get

$$
\exists \{ x \} \{ ( x , : { \mathrm { l i k e s } } , : { \mathrm { p a s t a } } ) \}\tag{16}
$$

while in Formula 3 from above, the blank nodes occurring in the graph term are quantified there, and we get

$$
\{ ( { \mathrm { : b o b , : s a y s , } } \exists \{ x \} \{ ( x , { \mathrm { : l i k e s , : p i z z a } } ) , ( x , { \mathrm { : l i k e s , : p a s t a } } ) \} ) \}\tag{17}
$$

In the following example:

$$
\{ \mathfrak { T x } \ \textbf { \textsf { a } } : \mathrm { H u m a n } \} \ \Rightarrow \ \{ \mathfrak { T x } \ : \mathrm { h a s M o t h e r } \ \mathbf { \Omega } _ { - } : \mathfrak { y } \} .\tag{18}
$$

variable ?x is universally quantified on the outermost graph, the blank node \_:y is local to the consequent of the rule. We get

$$
\forall \{ x \} \left\{ ( \{ ( x , { \mathrm { r d f } } \mathrm { : t y p e } , \mathrm { : H u m a n } ) \} , { \mathrm { l o g } } { \mathrm { : i m p l i e s } } , \exists \{ y \} \{ ( x , { \mathrm { : h a s M o t h e r } } , y ) \} ) \right\}\tag{19}
$$

which is an existential rule [4].

Explicit quantification in the concrete syntax is expressed with the @forAll or @forSome keyword, followed by an IRI and by a period (.). Such quantified IRIs will then translate to variables (V) in the abstract syntax rather than IRIs (C). Formula 12 from above could also be written as

$$
\mathfrak { Q } \mathbf { f } \mathrm { o r } \mathtt { A } \mathtt { l } \mathtt { l } \mathtt { l } \mathtt { \Omega } : \mathtt { x } . \mathtt { \Omega } \mathtt { Q } \mathtt { f } \mathrm { o r } \mathtt { S } \mathrm { o m e } \mathtt { \Omega } : \mathtt { y } . \mathtt { \Omega } : \mathtt { x } : \mathtt { l } \mathtt { o v e s } \ : \mathtt { y } .\tag{12<sup>′′</sup>}
$$

The formula again corresponds to the abstract formula 14. The quantifier keywords can be used in the top-level graph or in any graph term, but must always occur before the first triple of the graph to which they belong, and all @forAlls must be placed before any @forSome of the same graph<sup>5</sup>. Any further use of the quantified IRI in the same graph, or nested graph terms, will be represented in the abstract syntax by the corresponding variable. For example, the formula

$$
\begin{array} { r }  \mathfrak { Q } \mathtt { f o r S o m e \ : y . ~ : y . ~ : s a y s \ : \mathtt { s a y s \ : \mathtt { d e f o r A l l } { \mathtt { \alpha : x . ~ : y } ~ : l i k e s \ : x } \mathtt { y . } } } \end{array}\tag{20}
$$

is represented as

$$
\exists \{ y \} \{ ( y , : \mathrm { s a y s } , \forall \{ x \} \{ ( y , : \mathrm { l i k e s } , x \} ) \}\tag{21}
$$

## 4.3 Variables

For the definitions which follow later in this document, we need to make some statements about the variables occurring in N3 graphs. Here we distinguish between variables occurring directly in a graph an those which are nested. To be able to diferentiate them, we introduce the following concept of constituents.

▶ Definition 5 (Direct Constituent). A term is a direct constituent of a graph G = (U, E, F) if it is the subject, predicate or object of a triple t ∈ F. We denote the set of direct constituents of a graph G as DC(G).

We now define the more general notion of constituents, which are informally speaking all terms occurring in a graph.

▶ Definition 6 (Constituent). A term is a constituent of a graph G<sub>1</sub> if it is either

a direct constituent of G<sub>1</sub>, or

## 42:8 Notation3: Graph Terms for RDF

1 a constituent of graph $G _ { 2 }$ , where the graph term $G _ { 2 }$ is a direct constituent of $G _ { 1 }$

We denote the set of constituents of a graph G as $C ( G )$

In other words: direct constituents are all terms which directly form the graph's triples. Constituents also include all terms found in nested graph terms independently of the depth of the nesting.

We illustrate these definitions on an earlier example: Formula 3, whose abstract syntax is given in Formula 17. We encounter three direct constituents: :bob, :says, ∃{x} {(x, :likes, :pizza), (x, :likes, :pasta)}; and seven constituents: all the above plus x, :likes, :pizza, :pasta.

With that definition, we now introduce the notion of free variable.

▶ Definition 7 (Free Variable). The set of free variables in a term, triple, set of triples or graph x

is noted $F V ( x )$ and defined as follows:

$F V ( x ) = \varnothing \ i f \ x \in C \cup L ,$

$F V ( x ) = \{ x \} ~ i f ~ x \in V ,$

$F V ( x ) = F V ( s ) \cup F V ( p ) \cup F V ( o ) \ i f \ x = ( s , p , o )$ is a triple,

F V (x) = ∪<sub>t∈x</sub>F V (t) if x is a set of triples,

F V (x) = F V (F) \ (U ∪ E) if x = (U, E, F) is a graph,

$F V ( x ) = F V ( G ) ~ i f ~ x = \underline { { G } }$ is a graph term.

For a graph $G = ( U , E , F )$ and a variable $v \in C ( G ) \cap V .$ : We say that v is free in G $i f f v \in F V ( G )$ Otherwise, v is said to be scoped in G.

Examples:

In $\forall \{ x \} \exists \{ y \} \{ ( x , : \mathrm { l o v e s } , y ) \}$ (Formula 12) all variables are scoped.

In ∃{y} {(x, :knows, y)} the variable x is free while y is scoped.

In $\{ ( y , \mathrm { { : s a y s } , \exists \{ x \} \{ ( x , \mathrm { { r d f : t y p e } , \mathrm { { : M o r t a l } ) \} } } } ) \}$ , the variable x is scoped while the variable y is free.

With this definition, we now introduce ground graphs:

▶ Definition 8 (Ground term, triple and graph).

A term is ground if it is either

an IRI,

a Literal,

a graph term G with $F V ( G ) = \varnothing .$

We denote the set of ground terms as $T _ { G }$

A triple $( s , p , o )$ is ground $i f s , p$ and o are ground.

A graph $G = ( U , E , F )$ is ground if $E = U = \emptyset$ and all triples in $F$ are ground.

Note that this definition treats graphs and graph terms diferently: a graph term $( U , E , F )$ may be ground while the corresponding graph $( U , E , F )$ is not. More precisely: a graph term is ground if all its variable constituents are scoped, while to be ground, a graph must additionally have no variable in its direct constituents. A ground graph may still contain variables, as long as these are scoped within a graph term. This is because these variables are not accessible by outside quantification. If we modify Formula 17 from above by adding quantification we get

$$
\forall \{ x \} \{ ( { \mathrm { : b o b , : s a y s , } } \exists \{ x \} \{ ( x , { \mathrm { : l i k e s , : p i z z a } } ) , ( x , { \mathrm { : l i k e s , : p a s t a } } ) \} ) \} .\tag{22}
$$

As the variable x in this graph only occurs, scoped, in the graph term $\exists \{ x \} \{ ( x ,$ , :likes, :pizza), $( x , \mathrm { { : l i k e s } , \mathrm { { : p a s t a } ) } } \} )$ , it can not be "reached" by the universal quantifier we just added, and therefore Formulae 17 and 22 will be interpreted equally.

The example illustrates that a graph $G = ( U , E , F )$ can contain quantified variables $v \in U \cup E$ which do not occur freely in any triple of F and thus do not contribute to the interpretation of the graph. In order to simplify our considerations for the following sections, we introduce the notion of normalised graphs which exclude such quantification.

▶ Definition 9 (Normalisation). For each graph $G = ( U , E , F )$ , we call $G ^ { N } = ( U \cap ( F V ( F ) )$ , E ∩ $( F V ( F ) ) , F )$ the normalisation of G.

The normalisation of Formula 22 is Formula 17. We provide more examples below:   
∀{x, z} ∃{y} {(x, :knows, y)}<sup>N</sup> = ∀{x} ∃{y} {(x, :knows, y)}   
∀{x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>} ∃{y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>} {(:socrates, :knows, :plato)})<sup>N</sup> = {(:socrates, :knows, :plato)}   
∀{x} ∃{y} {(y, :says, ∃{x} {(x, rdf:type, :Mortal)})}<sup>N</sup>   
= ∃{y} {(y, :says, ∃{x} {(x, rdf:type, :Mortal)}}

## 4.4 Isomorphisms

Having the syntactic definition in place, we can now take a closer look at the relation between two graphs. Just like in RDF, we can define isomorphisms between N3 graphs and between graph terms. The latter will become particularly important in Section 5 where we define the semantics. When interpreting graph terms, the naming of a variable should not matter. Just as the Formulae 12, 12<sup>′</sup> and $1 2 ^ { \prime \prime }$ had the same meaning and could be translated to the abstract formula 14, the latter formula should have the meaning as

$$
\forall \{ y \} \exists \{ z \} \{ ( y , : \mathrm { l o v e s } , z ) \}\tag{14<sup>′</sup>}
$$

where we only changed the variable names. And that behavior should also hold for graph terms: we would like our semantics to assign the same meaning to the abstract graph term

```lisp
∃{x} {(x, :likes, :pizza), (x, :likes, :pasta)}
```

(23)

from Formula 17 as for

```lisp
∃{y} {(y, :likes, :pizza), (y, :likes, :pasta)}
```

(23<sup>′</sup>)

where we only replaced the variable x by y.

In order to be able to define isomorphisms between graph terms, we first introduce mechanisms to combine mappings. Here and later in this paper, we write dom(M) to denote the domain X of a mapping $M : X  Y$

▶ Definition 10 (Mapping Combination). Let M and N be two mappings from terms to terms, we define their combination, noted $M \bullet N$ as follows:

$$
M \bullet N ( v ) = \left\{ { \begin{array} { l } { N ( v ) { \mathrm { ~ } } i f v \in d o m ( N ) } \\ { M ( v ) { \mathrm { ~ } } i f v \notin d o m ( N ) { \mathrm { ~ } } a n d v \in d o m ( M ) } \end{array} } \right.
$$

Informally speaking, this definition means that the second mapping “overrides” the first one whenever a term is in both of their domains. This is very similar to the behavior of quantifiers in first-order logic where the scope of a quantifier is always the set of quantified variables occurring freely in the quantified formula. This similarity is no coincidence. In our semantics, combinations will be mainly used to handle nested variable scopes.

For a mapping $M : X  S$ , we denote by $M | _ { Y }$ the restriction of M to the set $Y \subset X$

We can now define what it means for N3 graphs to be isomorphic to each other.

## 42:10 Notation3: Graph Terms for RDF

▶ Definition 11 (Isomorphic under M). Let M be a bijection between sets of variables in $V :$

$$
G _ { 1 }
$$

$$
G _ { 2 }
$$

$$
G _ { 1 } ^ { N } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )
$$

$$
G _ { 2 } ^ { N } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )
$$

$M ( U _ { 1 } ) : = \{ M ( u ) | u \in U _ { 1 } \} = U _ { 2 }$ and $M ( E _ { 1 } ) : = \{ M ( e ) | e \in E _ { 1 } \} = E _ { 2 }$

$F _ { 1 }$ is isomorphic to $F _ { 2 }$ under M

a set of triples $F _ { 1 }$ is isomorphic to a set of triples $F _ { 2 }$ under M $i f f$

M is a bijection from $F V ( F _ { 1 } )$ to $F V ( F _ { 2 } )$

there is a bijection T from the triples in $F _ { 1 }$ to those in $F _ { 2 }$ such that, for each triple $t _ { 1 } = ( s _ { 1 } , p _ { 1 } , o _ { 1 } ) \in F _ { 1 }$ and for the image $T ( ( s _ { 1 } , p _ { 1 } , o _ { 1 } ) ) = ( s _ { 2 } , p _ { 2 } , o _ { 2 } ) \in F _ { 2 }$ it holds that $s _ { 1 } , p _ { 1 }$ and $o _ { 1 }$ are isomorphic under M to $s _ { 2 }$ , p<sub>2</sub> and $O _ { 2 }$ , respectively

a variable v is isomorphic under M to $M ( v )$ 2

an IRI or Literal is isomorphic under M to itself,

a graph term $\underline { { G _ { 1 } } }$ is isomorphic under M to $G _ { 2 }$ if for $G _ { 1 } ^ { N } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } ^ { N } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ there is a bijection $P$ between $U _ { 1 } \cup E _ { 1 }$ and $U _ { 2 } \cup E _ { 2 }$ such that $M | _ { F V ( G _ { 1 } ) } \bullet P$ is a bijection and $G _ { 1 }$ is isomorphic to $G _ { 2 }$ under $M | _ { F V ( G _ { 1 } ) } \bullet P$

As we will further discuss below (Lemma 13), if a graph $G _ { 1 }$ is isomorphic to a graph $G _ { 2 }$ under $M .$ , then $G _ { 2 }$ is isomorphic to $G _ { 1 }$ under $M ^ { - 1 }$ . This close connection motivates us to omit mentioning the mapping when talking about isomorphisms. We get the following more general definition:

▶ Definition 12 (Isomorphic). A graph $G _ { 1 }$ is said to be isomorphic to a graph $G _ { 2 }$ , written as $G _ { 1 } \simeq G _ { 2 }$ , if there exists some M such that $G _ { 1 }$ is isomorphic to $G _ { 2 }$ under $M .$

We note that, for the subset of N3 graphs that are also RDF graphs, our notion of isomorphism is equivalent to that defined in the RDF specification [30]. We also note the following lemma.

▶ Lemma 13 (Equivalence property of Isomorphisms). The isomorphic relation from above is reflexive, symmetric, and transitive.

To prove this lemma, we first remark that, given a graph G such that $G ^ { N } = ( U , E , F )$ , it holds that $F V ( F ) = U \cup E \cup F V ( G )$ per Definitions 7 and 9. For convenience, we denote this set by var(G).

Given two mappings A and B with disjoint domains, we remark that $A \bullet B = A \cup B$ (treating mappings as set of pairs). If additionally A, B and A • B are bijections, this implies that $( A \bullet B ) ^ { - 1 } = A ^ { - 1 } \bullet B ^ { - 1 }$

Isomorphism of N3 graphs (≃) is reflexive. This is trivially true by choosing the identity for the bijections M, T and $P .$ ◀

Isomorphism of N3 graphs (≃) is symmetric. Let us first prove by induction that if $G _ { 1 }$ is isomorphic to $G _ { 2 }$ under a bijection M, then $G _ { 2 }$ is isomorphic to $G _ { 1 }$ under $M ^ { - 1 }$ . Let $G _ { 1 } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ . There must be a bijection $T$ mapping triples from $F _ { 1 }$ to triples from $F _ { 2 }$ that satisfies the definition of isomorphism under $M$

If we assume that neither $G _ { 1 }$ nor $G _ { 2 }$ contain graph terms, then trivially $G _ { 2 }$ is isomorphic to $G _ { 1 }$ under $M ^ { - 1 }$ by choosing $T ^ { - 1 }$ to map from triples from $F _ { 2 }$ to $F _ { 1 }$

If $G _ { 1 }$ contains graph terms, then T must map every graph term $\underline { { H _ { 1 } } }$ occurring in $G _ { 1 }$ to a graph term $H _ { 2 }$ occurring in $G _ { 2 }$ , and for each such pair there must exist a bijection P such that $H _ { 1 }$ is isomorphic to $H _ { 2 }$ under $M | _ { F V ( H _ { 1 } ) } \bullet P$

Per our induction hypothesis, H<sub>2</sub> is also isomorphic to $H _ { 1 }$ under $( M | _ { F V ( G 1 ) } \bullet P ) ^ { - 1 }$ . We remark that the domains of $M | _ { F V ( H _ { 1 } ) }$ and P are disjoint, so $( M | _ { F V ( H _ { 1 } ) } \bullet P ) ^ { - 1 } = ( M | _ { F V ( H _ { 1 } ) } ) ^ { - 1 } \bullet P ^ { - 1 } =$ $M ^ { - 1 } | _ { F V ( H _ { 2 } ) } \bullet P ^ { - 1 }$ . We can then conclude that the graph term $H _ { 2 }$ is isomorphic to $H _ { 1 }$ under $M ^ { - 1 }$ , choosing $P ^ { - 1 }$ , and therefore that $G _ { 2 }$ is isomorphic to $G _ { 1 }$ under $M ^ { - 1 }$

From the above, it follows that if $G _ { 1 } \simeq G _ { 2 }$ , then $G _ { 2 } \simeq G _ { 1 }$

Isomorphism of N3 graphs $( \simeq )$ is transitive. Let us first prove by induction that if $G _ { 1 }$ is isomorphic to $G _ { 2 }$ under $M _ { 1 }$ , and $G _ { 2 }$ is isomorphic to $G _ { 3 }$ under $M _ { 2 }$ , then $G _ { 1 }$ is isomorphic to $G _ { 3 }$ under $M _ { 2 } \circ M _ { 1 }$ . Let $G _ { i } ^ { N } = ( U _ { i } , E _ { i } , F _ { i } )$ for $i \in \{ 1 , 2 , 3 \}$ . There must be a bijection $T _ { 1 }$ from $F _ { 1 }$ to $F _ { 2 }$ and a bijection $T _ { 2 }$ from $F _ { 2 }$ to $F _ { 3 }$ that satisfy the definition of isomorphism under $M _ { 1 }$ and $M _ { 2 }$ respectively.

If we assume that neither $G _ { 1 } , G _ { 2 }$ nor $G _ { 3 }$ contain graph terms, then trivially $G _ { 1 }$ is isomorphic to $G _ { 3 }$ under $M _ { 2 } \circ M _ { 1 }$ by chosing $T _ { 2 } \circ T _ { 1 }$ to map from $F _ { 1 }$ to $F _ { 3 }$

If $G _ { 1 }$ contains graph terms, then $T _ { 1 }$ must map every graph term $H _ { 1 }$ occurring in $G _ { 1 }$ to a graph term $H _ { 2 }$ occurring in $G _ { 2 } .$ , and for each such pair there must exist a bijection $P _ { 1 }$ such that $H _ { 1 }$ is isomorphic to $H _ { 2 }$ under $M _ { 1 } | _ { F V ( H _ { 1 } ) } \bullet P _ { 1 }$ . Similarly $T _ { 2 }$ must map every graph term $H _ { 2 }$ occurring in $G _ { 2 }$ to a graph term $\underline { { H 3 } }$ occurring in $G _ { 3 } ,$ , and for each such pair there must exist a bijection $P _ { 2 }$ such that $H _ { 2 }$ is isomorphic to $H _ { 3 }$ under $M _ { 2 } | _ { F V ( H _ { 2 } ) } \bullet P _ { 2 }$ . Per our induction hypothesis, $H _ { 1 }$ is isomorphic to $H _ { 3 }$ under $( M _ { 2 } | _ { F V ( H _ { 2 } ) } \bullet P _ { 2 } ) \circ ( M _ { 1 } | _ { F V ( H _ { 1 } ) } \bullet P _ { 1 } )$ ). Since the domain of $M \vert _ { F V ( H _ { i } ) }$ and $P _ { i }$ are disjoint for all $i ,$ this is equal to $( M _ { 2 } | _ { F V ( H _ { 2 } ) } \cup P _ { 2 } ) \circ ( M _ { 1 } | _ { F V ( H _ { 1 } ) } \cup P _ { 1 } )$ Since the image of $M _ { 1 } | _ { F V ( H _ { 1 } ) }$ is $F V ( H _ { 2 } )$ , and the image of $P _ { 1 }$ is the domain of $P _ { 2 }$ , this is equal to $( M _ { 2 } | _ { F V ( H _ { 2 } ) } \circ M _ { 1 } | _ { F V ( H _ { 1 } ) } \cup ( P _ { 2 } \circ P _ { 1 } ) = ( M _ { 2 } \circ M _ { 1 } ) | _ { F V ( H _ { 1 } ) } \bullet ( P _ { 2 } \circ P _ { 1 } )$ and therefore each graph term $H _ { 1 }$ is isomorphic to $H _ { 3 }$ under $M _ { 2 } \circ M _ { 1 }$ , choosing $P _ { 2 } \circ P _ { 1 }$ . Hence, $G _ { 1 }$ is isomorphic to $G _ { 3 }$ under $M _ { 2 } \circ M _ { 1 }$

From the above, it follows that if $G _ { 1 } \simeq G _ { 2 }$ and $G _ { 2 } \simeq G 3$ , then $G _ { 1 } \simeq G _ { 3 }$

We give an example to further illustrate the definition. Consider the following graphs in concrete syntax

$$
G _ { 1 } = \{ \\\\l _ { - } \colon \mathrm { x ~ : k n o w s ~ \ t y . ~ : p ~ : s a y s ~ \cup ~ \{ \ l _ - : x ~ \in ~ : G e n i u s \} ~ . } \}
$$

$$
G _ { 2 } = \{ \\\ l _ { - } \colon \mathrm { y } ~ : \mathrm { k n o w s ~ \ t p . ~ \ddots ~ \mathrm { p } ~ : s \mathtt { a y s } ~ \{ \ l _ - : x ~ \mathtt { a } ~ : G e n i u s \} . } \}
$$

written in abstract syntax these are

$$
G _ { 1 } = \exists \{ x \} \{ ( x , \mathrm { { : k n o w s } , : p } ) , ( \mathrm { { : p } , \mathrm { { : s a y s } , \exists \{ x \} \{ ( x , \mathrm { { r d f : t y p e } , : G e n i u s } ) \} } } ) \}
$$

$$
G _ { 2 } = \exists \{ y \} \{ ( y , \mathrm { { : k n o w s } , \mathrm { { : p } } ) , ( \mathrm { { : p } , \mathrm { { : s a y s } , \exists \{ x \} \{ x \} \{ ( x , \mathrm { { r d f : t y p e } , \mathrm { { : G e n i u s } } ) \} } } ) } } \}
$$

$$
G _ { 3 } = \exists \{ y \} \{ ( y , \mathrm { : k n o w s , : p } ) , ( \mathrm { : p } , \mathrm { : s a y s } , \exists \{ x , z \} \{ ( x , \mathrm { r d f : t y p e } , z ) \} ) \}
$$

Then $G _ { 1 }$ is isomorphic to $G _ { 2 }$ but not to $G _ { 3 }$

## 4.5 Syntactic relations between graphs

Beyond isomorphism, N3 graphs can have other useful relationships with each other. In this section we define the notion of subgraph, instance/generalisation, and merge.

▶ Definition 14 (Subgraph). A graph $G _ { 1 } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ is a subgraph of a graph $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ written as $G _ { 1 } \subseteq G _ { 2 } , i f F V ( G _ { 1 } ) \subseteq F V ( G _ { 2 } ) , U _ { 1 } \subseteq U _ { 2 } , E _ { 1 } \subseteq E _ { 2 } , a n d F _ { 1 } \subseteq F _ { 2 }$

Note that this definition is slightly more complicated than the subgraph relation in RDF which is basically the same as a subset [24], since in N3, we need to also take variable bindings into account. Given the graphs

$$
G = \forall \{ x \} \exists \{ y \} \{ ( x , : { \mathrm { k n o w s } } , y ) , ( x , { \mathrm { r d f } } \colon { \mathrm { t y p e } } , : { \mathrm { L o g i c i a n } } ) \}
$$

{(:alice, :likes, :pizza)} is a U-instance of $\forall \{ x \} \left\{ ( x , : { \mathrm { l i k e s } } , : { \mathrm { p i z z a } } ) \right\}$ , with $M = \{ x \mapsto \mathrm { { a l i c e } } \}$

G<sub>4</sub> =∀{x} {(x, rdf:type, :Logician)}   
G<sub>5</sub> =∃{x} {(x, rdf:type, :Logician)}   
G<sub>6</sub> =∀{x} {(x, :knows, y), (x, rdf:type, :Logician)}

## 42:12 Notation3: Graph Terms for RDF

and

$G _ { 4 }$ is a subgraph of G, but neither are $G _ { 5 }$ (quantifiers difer) nor $G _ { 6 }$ (y is free in $G _ { 6 } )$ Note that for the definition of subgraphs, concrete variable names matter. The graph

G<sub>7</sub> = ∀{z} {(z, rdf:type, :Logician)}

is not a subgraph of G, it is, however, isomorphic to one of its subgraphs, namely $G _ { 4 }$

In order to define the notions of instance and generalisation, we first need to define a way to properly substitute variables in an N3 graph.

▶ Definition 15 (Total application). Let M be a mapping from a set of variable $W \subset V$ into a set S. A total application $M ^ { t }$ of M on graphs and terms is defined as follows:

if x is a variable M<sup>t</sup>(x) = M(x) if x ∈ dom(M), x otherwise   
if x is an IRI or a Literal M<sup>t</sup>(x) = x   
if x = H is a graph term M<sup>t</sup>(x) = M<sup>t</sup>(H)   
if x = (s, p, o) M<sup>t</sup>(x) = (M<sup>t</sup>(s), M<sup>t</sup>(p), M<sup>t</sup>(o)),   
if x = F is a set of triples M<sup>t</sup>(x) = {M<sup>t</sup>(t)|t ∈ F}   
if x = (U, E, F) is a graph M<sup>t</sup>(x) = (U, E, M|<sup>t</sup><sub>FV(x)</sub>(F))

The total application of M replaces all free variables occurring directly or nested in a graph with their image through M. As an example we consider a modification of the rule in Formula 18, where the universal ?x is replaced with a free variable:

{({(x, rdf:type, :Human)}, log:implies, ∃{y} {(x, :hasMother, y)})}

(24)

Let $M : \{ x , y \} \to S$ be a mapping such that M(x) = tom and $M ( y ) = m a r y$ , then we get:

$$
\begin{array} { r l } & { M ^ { t } ( \{ ( \{ ( \underline { { \land \mathrm { ( f i } } . \mathrm { r d f } . \mathrm { t y p e } , \mathrm { : H u m a n } } ) \} , \log \mathrm { : i m p l i e s } , \underline { { \exists \{ y \} \{ ( x , \cdot \mathrm { h a s M o t h e r } , y ) \} \} } ) \} ) \} ) } \\ & { \quad \quad \quad = \{ ( \{ ( \{ ( \{ o m , \mathrm { r d f } \mathrm { t y p e } , \mathrm { : H u m a n } } ) \} , \log \cdot \mathrm { i m p l i e s } , \exists \{ y \} \{ ( t o m , \cdot \mathrm { h a s M o t h e r } , y ) \} ) \} ) \} } \end{array}\tag{25}
$$

In this example, we replace x with its value because it occurs freely in the graph while y does not get replaced since it is quantified in the nested graph.

▶ Definition 16 (Instance and generalisation of a graph). Let $G = ( U , E , F )$ be a graph.

We call a graph $G ^ { \prime }$ an E-instance of G if there exists a mapping $M : X \subseteq E \to T _ { G } \cup E$ such   
that $G ^ { \prime }$ is the normalisation of $( U , E , M ^ { t } ( F ) )$ . Conversely, we call G an E-generalisation of $G ^ { \prime }$ . We call a graph G<sup>′′</sup> a U-instance of G if there exists a mapping $M : X \subseteq U \to T _ { G } \cup U$ such   
that $G ^ { \prime \prime }$ is the normalisation of $( U , E , M ^ { t } ( F ) )$ . Conversely, we call G a U-generalization of $G ^ { \prime \prime }$

{(:alice, :likes, :pizza)} is an E-instance of $\exists \{ x \} \{ ( x , : { \mathrm { l i k e s } } , : { \mathrm { p i z z a } } ) \}$ , with $M = \{ x \mapsto \mathrm { { a l i c e } } \}$

$$
= \exists \{ x \} \{ ( x , : { \mathrm { l i k e s } } , : { \mathrm { p i z z a } } ) , ( x , : { \mathrm { l i k e s } } , : { \mathrm { p a s t a } } ) \}
$$

$$
\exists \{ x , y \} \{ ( x , : \mathrm { l i k e s } , : \mathrm { p i z z a } ) , ( y ,
$$

$$
M = \{ y \mapsto x \}
$$

$$
M = \{ y \mapsto x \}
$$

$= \exists \{ x \} \{ ( x , : { \mathrm { l i k e s } } , : { \mathrm { a l i c e } } ) \}$ is an E-generalization of {(:bob, :likes, :alice)}, which is itself a Uinstance of $\forall \{ y \} \left\{ ( y , : { \mathrm { l i k e s } } , : { \mathrm { a l i c e } } ) \right\}$ ,

We will see below (Section 5.3) that these definitions allow us to formulate a counterpart to RDF’s interpolation lemma [24, Section 5.2]. In particular, note that our notion of E-instance extends the notion of instance of an RDF graph. In contrast to RDF, N3 also covers universal quantification, hence the need to diferentiate two kinds of instances/generalisations.

▶ Definition 17 (Merge of graphs). Let $G _ { 1 } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ be two graphs such that $U _ { 1 } \cap U _ { 2 } = E _ { 1 } \cap E _ { 2 } = F V ( G _ { 1 } ) \cap F V ( G _ { 2 } ) = \emptyset . \ B y \ G _ { 1 } \sqcup G _ { 2 } : = ( U _ { 1 } \cup U _ { 2 } , E 1 \cup E 2 , F _ { 1 } \cup F _ { 2 } )$ we denote the disjoint union $o f G _ { 1 }$ and $G _ { 2 }$

Let $\{ G _ { 1 } , \ldots , G _ { n } \}$ be a set of graphs. We call a graph G a merge of $G _ { 1 } , \ldots , G _ { n }$ if, for all $i \in [ 1 , n ]$ there exist a graph $G _ { i } ^ { \prime }$ isomorphic to $G _ { i } ,$ such that $G = G _ { 1 } ^ { \prime } \sqcup \cdot \cdot \cdot \sqcup G _ { n } ^ { \prime }$

This definition extends that of the merge of RDF graphs as defined in Section 4.1 of RDF Semantics [24].

▶ Definition 18 (Closure of a graph). A graph G is said to be closed if it does not contain any free variable, i.e. $F V ( G ) = \varnothing$ . Given a graph $G = ( U , E , F )$ , the closed graph $( U \cup F V ( G ) , E , F )$ is called the closure of G.

As we will see in the following section, any graph is semantically equivalent to its closure.

## 5 Semantics

Having introduced N3’s abstract syntax in the previous section, we now define N3’s semantics. We do that in two steps: we first define N3’s base semantics which only covers the meaning of terms and variables but does not take any special predicates into account. After that, we discuss N3 semantics which assigns meaning to N3’s built-in predicates (Section 5.4). Here, we limit our considerations to the most important predicates (including log:implies). We plan to extend the definitions in future work.

## 5.1 Base Semantics

To define N3’s base semantics, we start with an interpretation function:

▶ Definition 19 (Interpretation). An interpretation I is defined by:

a set $\Delta _ { I }$ (domain) whose elements are referred to as resources,

a mapping $D _ { I }$ (denotation) from $C \cup L$ to $\Delta _ { I }$ , such that its restriction to C is a total mapping,

a mapping $Q _ { I }$ from ground graph terms to ground graph terms, such that

for any graph term $\underline { { { G } } } \colon Q _ { I } ( \underline { { { G } } } ) = \underline { { { H } } } \implies H \simeq G$

$f o r$ two graph terms $G _ { 1 }$ and $G _ { 2 } .$

$$
G _ { 1 } \simeq G _ { 2 } \implies Q _ { I } ( G _ { 1 } ) = Q _ { I } ( G _ { 2 } )
$$

a subset $E X T _ { I }$ (extension) of $\Delta _ { I } ^ { 3 }$ .

This definition is generally “compatible” with RDF simple interpretations, with the following superficial changes:

IR becomes $\Delta _ { I } ,$

IP is assumed, without loss of generality, to be a subset of $\Delta _ { I }$

1 IS and IL have been merged in $D _ { I }$ ,

IEXT can be derived from $E X T _ { I } ;$ formally: $I E X T ( p ) = \{ ( s , o ) | ( s , p , o ) \in E X T _ { I } \}$ ; conversely:

$$
\begin{array} { r } { E X T _ { I } = \bigcup _ { p \in I P } \{ ( s , p , o ) | ( s , o ) \in I E X T ( p ) \} } \end{array}
$$

## 42:14 Notation3: Graph Terms for RDF

The interpretation maps each ground graph term to an isomorphic copy of itself. In that sense, ground graph terms are similar to literals in RDF (they have a “fixed” denotation). As a consequence, two non-isomorphic graphs cannot have the same meaning.

However, N3-entailment(see Section 5.4) provides mechanisms to “unquote” graphs and to compare them based on the meaning of the formula they quote.

We now inductively define how to interpret ground graphs with an interpretation I:

▶ Definition 20 (Application of an interpretation). For ground terms, triples, and graphs, we define the application of an interpretation I as follows:

if x = F is a set of ground triples I(x) = false if I(t) = false for some t ∈ F,

if x = (∅, ∅, F) is a ground graph I(x) = I(F)

The definition implies that the empty graph is true under any interpretation I.

In order to extend the definition towards non-ground graphs, we rely on a few additional definitions. To better understand the need for these, consider the following example:

?x :says {?x :is :wise}.

(26)

which freely translates to “everyone says about themselves that they are wise”. In abstract syntax, the formula would be expressed as:

```lisp
∀{x} {(x, :says, {(x, :is, :wise))}}
```

(27)

This formula is interesting because the universally quantified variable appears once as a direct constituent in the triple and once nested. The semantics for ground graphs maps direct constituents to (not further specified) domain elements, while ground nested graphs are mapped to an isomorphic version of themselves.

If we replace the variable x with, for example, :bob, we get

{(:bob, :says, {(:bob, :is, :wise))}}

(28)

With an interpretation I:

Note that the mapping $D _ { I }$ maps the URI :bob directly to a resource of $\Delta _ { I } .$ , while $Q _ { I }$ operates on the graph term which contains the URI :bob. If we go back to Formula 27 where we have a variable x instead of :bob, the first occurrence of x should stand for a resource, while the second occurrence (in the graph term) depends on the syntactical representation of that resource. Note that this diverges from how RDF deals with variables, i.e. blank nodes [24, Section 5.1].

While RDF semantics maps blank nodes to resources, our semantics for N3 maps variables to ground terms, which are then, depending on the context, mapped to resource via an interpretation I, or kept as is. While this shift is not neutral (the domain may contain unnamed elements, which are therefore unreachable by quantified variables), we will prove later that it has no significant impact on basic entailment: N3 basic entailment on RDF graphs is equivalent to RDF simple entailment, despite this diferent treatment of existentials (Lemma 26 below). The diference is likely to matter, however, with more complex inference regimes introducing negation [20].

▶ Definition 21 (Interpretation with assignment). Let I be an interpretation and $A : V \to T _ { G }$ a mapping (assignment) from V to $T _ { G }$ . We define the application $I [ A ] ( x )$ on terms, triples, and graphs as follows:

if x is a variable I[A](x) = I(A(x))   
if x is an IRI or a Literal I[A](x) = D<sub>I</sub>(x)   
if x is a graph term I[A](x) = Q<sub>I</sub>(A<sup>t</sup>(x))   
if x = (s, p, o) I[A](x) = true if (I[A](s), I[A](p), I[A](o)) ∈ EXT<sub>I</sub>,   
false otherwise   
if x = F is a set of triples I[A](x) = f alse if I[A](t) = f alse for some $t \in F ,$   
true otherwise   
if x = (U, E, F) is a graph I[A](x) = true if for each mapping $B : U \to T _ { G }$   
there exists a mapping C : E → T   
such that I[A • B • C](F) = true,   
false otherwise

When x is a graph, A will only afect the free variables in x (quantified variables will be handled by B and C). Then if x is a closed graph (i.e. without free variables), $I [ A ] ( x )$ is not influenced by A. In such cases, we can omit the assignment A and simply write $I ( x )$

▶ Definition 22 (Basic models). Let I be an interpretation and G a graph. We call I a basic model of G, noted $I \left. _ { b } G , i f I [ A ] ( G ) \right.$ = true for all mappings $A : V \to T _ { G }$

The above definition handles all variables occurring freely in a graph as universally quantified. Note that the interpretation of a variable x depends on its position in a graph. In the expression $( x , \cdot \mathrm { p } , \{ ( x , \cdot \mathrm { q } , \cdot \mathrm { r } ) \} )$ , the first x is interpreted as $I ( A ( x ) )$ (an element of the domain) while the second is interpreted as A(x) (a ground term).

▶ Definition 23 (Basic entailment). Let G and H be graphs. We say that G b-entails H, written as G |=<sub>b</sub> H if for each basic model I of G, i.e. such that $I \Vdash _ { b } G ,$ , it holds that $I \models _ { b } H$

We give some examples for b-enatilemnt. Formula 27 from above b-entails Formula 28 and

∃{x} {(x, :says, {(x, is, :wise)})}

(29)

but not

$$
\forall \{ x \} \{ ( \mathrm { x } , \mathrm { : s a y s } , \underline { { \{ ( \mathrm { : b o b , i s , : w i s e } ) \} } } ) \}\tag{30}
$$

Formula 30 b-entails Formulae 28 and 29, and Formula 28 b-entails Formula 29. Note that basic entailment does not support rules. Rule 19 from above

$$
\forall \{ x \} \left\{ ( \{ ( x , { \mathrm { r d f } } \mathrm { : t y p e } , \mathrm { : H u m a n } ) \} , { \mathrm { l o g } } { \mathrm { : i m p l i e s } } , \exists \{ y \} \{ ( x , { \mathrm { : h a s M o t h e r } } , y ) \} ) \right\}\tag{19}
$$

b-entails

$$
\{ ( \{ \mathrm { ( \{ \cdot t o m , r d f : t y p e , : H u m a n } )  \} , \mathrm { l o g : i m p l i e s } , \exists \{ y \} \{ \mathrm { ( : t o m , : h a s M o t h e r , } y ) \} ) \}\tag{31}
$$

but even in combination with

{(:tom, rdf:type, :Human)}

(32)

## 42:16 Notation3: Graph Terms for RDF

we cannot b-entail that

$$
\exists \{ y \} \{ ( : \mathrm { t o m } , : \mathrm { h a s M o t h e r } , y ) \} .\tag{33}
$$

This kind of entailment is only supported by N3-entailment which will be discussed below in Section 5.4.

As basic entailment on its own does not support concepts like rules or negation, it can not be used to construct contradictions and all N3 graphs have a basic model.

▶ Lemma 24 (Basic satisfiability). For each N3 graph there exists a basic model.

Proof. Consider the interpretation with $\Delta _ { I } = T _ { G } , D _ { I } = \mathrm { i d }$ the identity mapping, $E X T _ { I } = \Delta _ { I } ^ { 3 }$ and $Q _ { I }$ mapping all ground graph terms to one of their isomorphic representations (such as required in Definition 19). This interpretation is a model for all graphs. ◀

For the closure of a graph we furthermore observe the following:

▶ Lemma 25 (Closure). Each N3 graph has the same models as its closure.

Proof. Let $G _ { 1 } = ( U , E , F )$ be an N3 graph and $G _ { 2 } = ( U \cup F V ( G _ { 1 } ) , E , F )$ its closure. Let $I \ | = _ { b } \ G _ { 1 }$ . We show that for all $B _ { 2 } : U \cup F V ( G _ { 1 } )  T _ { G }$ there exists $C _ { 2 } : E \to T _ { G }$ such that $I [ B _ { 2 } \bullet C _ { 2 } ] ( F ) = { \mathrm { t r u e } }$ . As $I \Vdash _ { b } G _ { 1 }$ , we know that for $A _ { 1 } : V \to T _ { G }$ and $B _ { 1 } : U \to T _ { G }$ , there exists $C _ { 1 } : E \to T _ { G }$ . If we chose $A _ { 1 }$ and $B _ { 1 }$ such that $B _ { 2 } = A _ { 1 } | _ { F V ( G _ { 1 } ) } \bullet B _ { 1 }$ , we set $C _ { 2 } : = C _ { 1 }$ , then we have $I [ B _ { 2 } \bullet C _ { 2 } ] ( F ) =$ true and therefore $I \left| = _ { b } G _ { 2 } \right.$

Conversely, let $I \left| = _ { b } G _ { 2 } \right.$ . We show that for all $A _ { 1 } : V \to T _ { G }$ and $B _ { 1 } : U \to T _ { G }$ there exists $C _ { 1 }$ $E  T _ { G }$ such that $I [ A _ { 1 } \bullet B _ { 1 } \bullet C _ { 1 } ] ( F ) = { \mathrm { t r u e } }$ . As $I \Vdash _ { b } G _ { 2 }$ , we know that for $B _ { 2 } : = A _ { 1 } | _ { F V ( G _ { 1 } ) } \bullet B _ { 1 }$ there exists $C _ { 2 } : E \to T _ { G }$ . We set $C _ { 1 } : = C _ { 2 }$ , then we have $I [ A _ { 1 } \bullet B _ { 1 } \bullet C _ { 1 } ] ( F )$ = true and therefore $I \Vdash _ { b } G _ { 1 }$ ◀

From the lemma above, we conclude that we can, without loss of generality, focus on closed graphs for the rest of the proofs. Any subsequent result is straightforwardly extensible to non-closed graph by first taking their closure.

For pure RDF graphs (i.e. graphs without graph terms and universal quantification) bentailmant coincides with RDF simple entailment [24, Section 5.2]. For an RDF graph G let E be the set of blank nodes occurring in it. We denote its N3 representation with explicit quantifiers as detailed in Section 4.2 as $G ^ { N 3 } : = ( \emptyset , E , G )$

▶ Lemma 26 (Simple Entailment). For two RDF graphs $G _ { 1 }$ and $G _ { 2 }$ with blank nodes $E _ { 1 }$ and $E _ { 2 }$ respectively, the following holds:

$$
G _ { 1 } ~ s i m p l y ~ e n t a i l s ~ G _ { 2 } ~ i f f ~ G _ { 1 } ^ { N 3 } = ( \emptyset , E _ { 1 } , G _ { 1 } ) ~ b - e n t a i l s ~ G _ { 2 } ^ { N 3 } = ( \emptyset , E _ { 2 } , G _ { 2 } )
$$

Proof. In the following, we note RDF simple entailment as $\succcurlyeq { \succcurlyeq { s } } ,$ to clearly diferentiate it from Notation3 basic entailment, still noted $\models _ { b } .$ . We also use RDF simple interpretations and Notation3 interpretations interchangeably, since we showed after Definition 19 that they can easily be transformed into each other.

⇒ Assume $G _ { 1 } \models _ { s } G _ { 2 }$ . Let $I \left| = _ { b } G _ { 1 } ^ { N 3 } \right.$ , i.e. there exists $B _ { 1 } : E _ { 1 } \to T _ { G }$ such that $I [ B _ { 1 } ] ( G _ { 1 } ^ { N 3 } ) =$ true. We build a new model $I ^ { \prime }$ such that $\Delta _ { I } ^ { \prime } = I m g ( D _ { I } )$ , and $E X T _ { I ^ { \prime } }$ ′ is the restriction of $E X T _ { I }$ to this new domain. Clearly $I ^ { \prime } [ B _ { 1 } ] ( G _ { 1 } ^ { N 3 } ) = \mathrm { t r u e }$ , so ${ \cal I } ^ { \prime } \models _ { b } G _ { 1 } ^ { N 3 }$ . We also have ${ \cal I } ^ { \prime } \models _ { s } G _ { 1 }$ 2 choosing $A _ { 1 } ( e ) = I ^ { \prime } ( B _ { 1 } ( e ) )$ . Per our assumption, $I ^ { \prime } \models _ { s } G _ { 2 } { : }$ there exists $A _ { 2 } : E _ { 2 } \to \Delta _ { I ^ { \prime } }$ such that $[ I ^ { \prime } + A _ { 2 } ] ( G _ { 2 } ) =$ true. By construction, all elements r of $\Delta _ { I ^ { \prime } }$ are named, so there exists a function $\nu : \Delta _ { I ^ { \prime } }  T _ { G }$ such that $I ^ { \prime } ( \nu ( r ) ) = r$ . So we can build $B _ { 2 } : E \to T _ { G }$ defined by $B _ { 2 } ( e ) = \nu ( A _ { 2 } ( e ) )$ 2 and therefore $I ^ { \prime } [ B _ { 2 } ] ( G _ { 2 } ^ { N 3 } )$ = true. It follows, by construction of $I ^ { \prime } ,$ that $I [ B 2 ] ( G _ { 2 } ^ { N 3 } )$ = true, and therefore $I \left| = _ { b } G _ { 2 } ^ { N 3 } . \right.$

$\Leftarrow \mathrm { A }$ ssume $G _ { 1 } ^ { N 3 } \left| = _ { b } G _ { 2 } ^ { N 3 } \right.$ . Let $I \Vdash _ { s } G _ { 1 }$ , i.e. there exists $A _ { 1 } : C \to \Delta _ { I }$ such that $[ I + A _ { 1 } ] ( G _ { 1 } ) =$ true. Without loss of generality, we assume that IR does not contain any graph term. We set $N _ { 1 } : = \{ e \in E _ { 1 } | \exists t \in C \cup L , D _ { I } ( t ) = A _ { 1 } ( e ) \}$ (the set of existentials that $A _ { 1 }$ maps to a named individual), and $O _ { 1 } : = E _ { 1 } \setminus N _ { 1 }$ . We can define an injection<sup>6</sup> δ which assigns to every resource in $A _ { 1 } ( O _ { 1 } )$ a graph term of the form $( c , c , c )$ with $c \in C$ . We define the interpretation $I ^ { \prime }$ as follows: $\begin{array} { r l } { = } & { { } \Delta _ { I ^ { \prime } } = \Delta I \setminus A _ { 1 } ( O _ { 1 } ) \cup \delta ( A _ { 1 } ( O _ { 1 } ) ) } \end{array}$

$$
\begin{array} { r l } { \mathrm { ~  ~ \tau ~ } } & { { } = \ Dot { D } _ { I ^ { \prime } } = D _ { I } } \end{array}
$$

Q<sub>I</sub>′ = identity function on $\delta ( A _ { 1 } ( O _ { 1 } ) )$

$$
\begin{array} { r l } & { E X T _ { I ^ { \prime } } = \left\{ \left( \delta ^ { * } ( s ) , p , \delta ^ { * } ( o ) \right) \mid ( s , p , o ) \in E X T _ { I } \right\} \mathrm { ~ w h e r e ~ } \delta ^ { * } ( x ) = \left\{ \begin{array} { l l } { \delta ( r ) } & { \mathrm { i f ~ } r \in A _ { 1 } ( O _ { 1 } ) } \\ { r } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

In other words, I<sup>′</sup> is obtained by replacing in I all resources $r \in A _ { 1 } ( O _ { 1 } )$ by $\delta ( r )$ . Clearly, $[ I ^ { \prime } + \delta ^ { * } \circ A _ { 1 } ] ( G _ { 1 } ) = { \mathrm { t r u e } } .$ , so ${ \cal I } ^ { \prime } \models _ { s } G _ { 1 }$ . We also have ${ \cal I } ^ { \prime } \models _ { b } G _ { 1 } ^ { N 3 }$ by choosing

$$
B _ { 1 } ( e ) = { \left\{ \begin{array} { l l } { t \in C \cup L { \mathrm { ~ s u c h ~ t h a t ~ } } D _ { I } ( t ) = A _ { 1 } ( e ) } & { { \mathrm { i f ~ } } e \in N _ { 1 } } \\ { \delta ( A _ { 1 } ( e ) ) } & { { \mathrm { i f ~ } } e \in O _ { 1 } } \end{array} \right. }
$$

Per our assumption, ${ \cal I } ^ { \prime } \left| = _ { b } G _ { 2 } ^ { N 3 } \right. :$ : there exists $B _ { 2 } : E _ { 2 } \to T _ { G }$ such that $I ^ { \prime } [ B _ { 2 } ] ( G _ { 2 } ^ { N 3 } ) = \mathrm { t r u e }$ . We can build $A _ { 2 } : E _ { 2 }  I R$ as follows:

$$
A _ { 2 } ( e ) = \left\{ \begin{array} { l l } { D _ { I ^ { \prime } } ( B _ { 2 } ( e ) ) } & { \mathrm { i f ~ } B _ { 2 } ( e ) \in C \cup L } \\ { B _ { 2 } ( e ) } & { \mathrm { i f ~ } B _ { 2 } ( e ) \mathrm { i s ~ a ~ g r a p h ~ t e r m } } \end{array} \right.
$$

$[ I + A _ { 2 } ] ( G _ { 2 } )$ = true by construction, and therefore $I \left| = _ { s } G _ { 2 } . \right.$

## 5.2 Semantic relations between graphs

Having introduced basic models, we now discuss the semantic implications of the syntactic relations between graphs we introduced in Sections 4.4 and 4.5. We start with isomorphic graphs and show:

▶ Lemma 27 (Isomorphic graphs). If two closed graphs are isomorphic to each other, they share the same models.

Proof. Given two closed graphs $G _ { 1 } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ where $G _ { 1 }$ is isomorphic to $G _ { 2 }$ under M. Let $I \Vdash _ { b } G _ { 1 }$

If $\dot { G } _ { 1 }$ and $G _ { 2 }$ are ground $( U _ { 1 } = U _ { 2 } = E _ { 1 } = E _ { 2 } = \emptyset )$ , we simply need to show that $I ( F _ { 2 } ) = I ( F _ { 1 } )$ This holds because $F _ { 1 }$ is isomorphic to $F _ { 2 }$ under M, and $Q _ { I }$ maps any two isomorphic ground graph terms to the same image.

If $G _ { 1 }$ and $G _ { 2 }$ are not ground, we show that for each mapping $B _ { 2 } : U _ { 2 } $ T there exists a mapping $C _ { 2 } : E _ { 2 } \to T _ { G }$ such that $I [ B _ { 2 } \bullet C _ { 2 } ] ( F _ { 2 } ) = t r u e$ . As $I \ | = _ { b } \ G _ { 1 }$ , we know that for $B _ { 1 } : = B _ { 2 } \circ M : U _ { 1 }  T _ { G }$ there exists a mapping $C _ { 1 } : E _ { 1 } \to T _ { G }$ such that $I [ B _ { 1 } \bullet C _ { 1 } ] ( F _ { 1 } ) = t r u e$ We set $C _ { 2 } : = C _ { 1 } \circ M ^ { - 1 }$ , then $I [ B 2 \bullet C _ { 2 } ] ( F _ { 2 } ) :$ = true and therefore $I \Vdash _ { b } G _ { 2 }$ ◀

We can furthermore show, that N3 graphs b-entail their subgraphs.

▶ Lemma 28 (Subgraphs). A closed N3 graph b-entails any of its subgraphs.

Proof. Let $G _ { 1 } = ( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ be a closed N3 graph, and $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ a subgraph of $G _ { 1 }$ . We notice that $G _ { 2 }$ is necessarily closed (as $F V ( G _ { 2 } ) \subseteq F V ( G _ { 1 } ) = \varnothing )$ and therefore $F V ( F _ { 2 } ) \subseteq U _ { 2 } \cup E _ { 2 }$ We also notice that, if $B : U _ { 1 } \to \ T _ { G }$ and $C : E _ { 1 } \to T _ { G }$ $I [ B \bullet C ] ( F 2 ) = I [ B | _ { U _ { 2 } } \bullet C | _ { E _ { 2 } } ] ( F 2 )$ Let I be an interpretation such that $I \left| = _ { b } G _ { 1 } \right.$ . We show that for all $B _ { 2 } : U _ { 2 } \to T _ { G }$ , there exists $C _ { 2 } : E _ { 2 } \to T _ { G }$ such that $I [ B _ { 2 } \bullet E _ { 2 } ] ( F _ { 2 } ) =$ true. As $I \Vdash _ { b } G _ { 1 }$ , we know that for $B _ { 1 } : U 1 \to T _ { G }$ such that $B _ { 1 } | _ { U _ { 2 } } = B _ { 2 }$ , there exists $C _ { 1 } : E _ { 1 } \to T _ { G }$ such that $I [ B _ { 1 } \bullet C _ { 1 } ] ( F _ { 1 } ) = \mathrm { t r u e }$ . It also holds that $I [ B _ { 1 } \bullet C _ { 1 } ] ( F _ { 2 } ) = \mathrm { t r u e } .$ as $F _ { 2 } \subseteq F _ { 1 }$ , and so $I [ B _ { 1 } | _ { U _ { 2 } } \bullet C _ { 1 } | _ { E _ { 2 } } ] ( F _ { 2 } ) = \mathrm { t r u e }$ . We set $C _ { 2 } : = C _ { 1 } | _ { E _ { 2 } }$ , then $I [ B _ { 2 } \bullet C _ { 2 } ] ( F _ { 2 } ) = { \mathrm { t r u e } }$ and therefore $I \Vdash _ { b } G _ { 2 }$ ◀

Similarly to RDF simple entailment, we see that E-instances entail their generalisations.

▶ Lemma 29 (E-Instances). A closed N3 graph is b-entailed by any of its E-instances.

Proof. Let $G _ { 1 } = ( U , E _ { 1 } , F _ { 1 } )$ be a closed N3 graph, and $G _ { 2 } = ( U , E _ { 2 } , F _ { 2 } )$ an E-instance of $G _ { 1 }$ with the instantiation mapping $M : E _ { 1 } \to T _ { G } \cup E _ { 2 }$ . Let $I \Vdash _ { b } G _ { 2 }$ . We show that for all $B : U \to T _ { G }$ there exists $C _ { 1 } : E _ { 1 } \to T _ { G }$ such that $I [ B \bullet C _ { 1 } ] ( F _ { 1 } ) = { \mathrm { t r u e } }$ . As $I \Vdash _ { b } G _ { 2 }$ , we know that there exists $C _ { 2 } : E _ { 2 } \to T _ { G }$ such that $I [ B \bullet C _ { 2 } ] ( F _ { 2 } ) = { \mathrm { t r u e } }$ . We set

$$
C _ { 1 } ( v ) : = \left\{ { \begin{array} { l l } { C _ { 2 } ( M ( v ) ) { \mathrm { ~ i f ~ } } M ( v ) \in E _ { 2 } } \\ { M ( v ) { \mathrm { ~ i f ~ } } M ( v ) \in T _ { G } } \end{array} } \right.
$$

We have

For U-instances, the opposite is true: U-generalisations entail their instances.

▶ Lemma 30 (U-Instances). A closed N3 graph entailes any of its U-instances.

Proof. Let $G _ { 1 } = ( U _ { 1 } , E , F _ { 1 } )$ be a closed N3 graph, and $G _ { 2 } = ( U _ { 2 } , E , F _ { 2 } )$ a U-instance of $G _ { 1 }$ with the instantiation mapping $M : U _ { 1 } \to T _ { G } \cup U _ { 2 }$ . Let $I \Vdash _ { b } G _ { 1 }$ . We show that for all $B _ { 2 } : U _ { 2 } \to T _ { G }$ there exists $C : E \to T _ { G }$ such that $I [ B _ { 2 } \bullet C ] ( F _ { 2 } ) =$ true. As $I \ \models _ { b } \ G _ { 1 }$ , we know that for all $B _ { 1 } : U _ { 1 } \to T _ { G }$ there exists $C : E  T _ { G }$ such that $I [ B _ { 1 } \bullet C _ { ] } ( F _ { 1 } )$ = true. We chose $B _ { 1 }$ such that

$$
B _ { 1 } ( v ) = { \left\{ \begin{array} { l l } { B _ { 2 } ( M ( v ) ) { \mathrm { ~ i f ~ } } M ( v ) \in U _ { 2 } } \\ { M ( v ) { \mathrm { ~ i f ~ } } M ( v ) \in T _ { G } } \end{array} \right. }
$$

We have

$$
{ \begin{array} { r l r l } & { I [ B _ { 1 } \bullet C ] ( F _ { 1 } ) = { \mathrm { t r u e } } } \\ { = } & { I [ B _ { 2 } \bullet C ] ( M ^ { t } ( F _ { 1 } ) ) } & & { { \mathrm { b y ~ c o n s t r u c t i o n ~ o f ~ } } B _ { 1 } } \\ { = } & { I [ B _ { 2 } \bullet C _ { 1 } ] ( F _ { 2 } ) } & & { { \mathrm { a s ~ } } F _ { 2 } = M ^ { t } ( F _ { 1 } ) { \mathrm { ~ p e r ~ D e f i n i t i o n ~ 1 6 } } } \end{array} }
$$

The merge of two logical consequences is furthermore a logical consequence.

▶ Lemma 31 (Merge). Let $G , G _ { 1 } , G _ { 2 }$ be closed N3 graphs, let $G _ { m }$ be a merge of $G _ { 1 }$ and $G _ { 2 }$ , and let $G \models _ { b } G _ { 1 }$ and $G \models _ { b } G _ { 2 }$ then $G \left| = _ { b } G _ { m } \right.$

Proof. Let $G , G _ { 1 } , G _ { 2 } , G _ { m }$ be as stated above, and let $I \Vdash _ { b } G$ . Then $I \Vdash _ { b } G _ { \exists }$ <sub>1</sub> and $I \Vdash _ { b } G _ { 2 }$ . As $G _ { m }$ is a merge of $G _ { 1 }$ and $G _ { 2 }$ , there exist graphs $G _ { 1 } ^ { \prime } = ( U _ { 1 } ^ { \prime } , E _ { 1 } ^ { \prime } , F _ { 1 } ^ { \prime } ) \simeq G _ { 1 }$ and $G _ { 2 } ^ { \prime } = ( U _ { 2 } ^ { \prime } , E _ { 2 } ^ { \prime } , F _ { 2 } ^ { \prime } ) \simeq G _ { 2 }$ such that $G _ { m } = G _ { 1 } ^ { \prime } \sqcup G _ { 2 } ^ { \prime }$ (Definition 17). By Lemma 27 it holds that $I \Vdash _ { b } G _ { 1 } ^ { \prime }$ and $I \Vdash _ { b } G _ { 2 } ^ { \prime }$ and as $U _ { 1 } ^ { \prime } \cap U _ { 2 } ^ { \prime } = E _ { 1 } ^ { \prime } \cap E _ { 2 } ^ { \prime } = \emptyset$ , this implies that $I \left| = _ { b } G _ { 1 } ^ { \prime } \sqcup G _ { 2 } ^ { \prime }\right.$ and thus $I \Vdash _ { b } G _ { m }$ ◀

## 5.3 Interpolation Lemma

In the previous section we showed how several syntactic relations between graphs ensure entailment but we did not yet characterise the other direction: the syntactical relation between graphs which entail each other. The goal of this section is to find an N3 counterpart for RDF’s interpolation lemma.

To do so, we first remind that, according to Lemma 26 that for pure RDF graphs b-entailmant coincides with RDF simple entailment. In N3, we furthermore need to take universal quantification into account. The graph

$$
{ G } _ { 1 } = \forall \{ x \} \{ ( x , : p , : o ) \}\tag{34}
$$

b-entails

$$
G _ { 2 } = \{ ( \boldsymbol { \it { s } } , \boldsymbol { \it { s p } } , \boldsymbol { \it { z o } } ) \}\tag{35}
$$

We replaced the universal variable x by the ground term :s and retrieved a U-instance (Lemma 30). This form of syntactical derivation gets more complicated if there are existentially quantified variables depending on the universals like for example in the following case:

$$
{ G _ { 3 } = \forall \{ x \} \exists \{ y \} \{ ( x , : h a s M o t h e r , y ) \} }\tag{36}
$$

Here, we state that everyone (and everything) has a mother. We can safely b-entail that

$$
G _ { 4 } = \exists \{ y \} \{ ( : b o b , : h a s M o t h e r , y ) \}\tag{37}
$$

but the entailment

$$
{ \cal G } _ { 4 } = \exists \{ y \} \{ ( : b o b , : h a s M o t h e r , y ) , ( : l i z , : h a s M o t h e r , y ) \}\tag{38}
$$

is incorrect and a correct version of this kind of entailment would be

$$
G _ { 4 } = \exists \{ y , z \} \{ ( : b o b , : h a s M o t h e r , y ) , ( : l i z , : h a s M o t h e r , z ) \}\tag{39}
$$

to make sure that :bob and :liz possibly have diferent mothers. As we quantify over all ground terms and thus over an infinite set – even if the set of constants is limited, we can construct infinitely many graph terms – we cannot easily identify a finite graph from which all possible consequences can be derived, as it is the case for simple entailment in RDF where we have RDF’s interpolation lemma. Instead, we construct the universal extension of a graph which might be infinite. The idea is motivated by the concept of universal models for existential rules [19].

▶ Definition 32 (Universal Extension). Let $G = ( U , E , F )$ be a finite and normalised closed N3 graph, $| U | = n$ . Let N ⊂ V be a set of variables such that $N \cap C ( G ) = \emptyset$ . We enumerate the variables in $U = \{ u _ { 1 } , u _ { 2 } , . . . u _ { n } \}$ and define for each combination of ground terms $( t _ { 1 } , . . . , t _ { n } ) = \bar { t } \in T _ { G } ^ { n }$ two mappings $A _ { \bar { t } } : U \to T _ { G }$ and $S _ { \bar { t } } : E \to N$ , such that:

(i) $A _ { \overline { { t } } } ( u _ { i } ) = t _ { i }$ for all $1 \leq i \leq n$

$$
( i i ) \ S _ { \overline { { { t } } } } ( y _ { 1 } ) \neq S _ { \overline { { { t ^ { \prime } } } } } ( y _ { 2 } ) \ i f \ \overline { { { t } } } \neq \overline { { { t ^ { \prime } } } } \ o r \ y _ { 1 } \neq y _ { 2 }
$$

If N is minimal in the sense that for each $x \in N$ there exists a value $\bar { t } \in T _ { G } ^ { n }$ and a value $y \in E$ such that $S _ { \bar { t } } ( y ) = x ,$ we call the graph $U _ { N } ( G ) = ( \emptyset , N , F ^ { \prime } )$ with $\begin{array} { r } { F ^ { \prime } : = \bigcup _ { \bar { t } \in { T } _ { \mathrm { { \diamond } } } ^ { n } } ( A _ { \bar { t } } \bullet S _ { \bar { t } } ) ^ { t } ( F ) } \end{array}$ a universal extension of G via N. In cases where the exact choice of N is not important, we omit N and call $U ( G )$ a universal extension of G.

Note that, modulo an appropriate mapping $B : N \to T _ { G }$ , the universal extension contains all triples which must be true in any model of G. This is captured by the following lemma:

▶ Lemma 33 (Universal Extension equivalence). Let $G = ( U , E , F )$ be a finite normalized closed N3 graph, and $U _ { N } ( G ) = ( \emptyset , N , F ^ { \prime } )$ a universal extension of G. Let I be an interpretation, then the following holds:

$$
I \left| = _ { b } G \right. \quad i f f \quad I \left| = _ { b } U _ { N } ( G ) \right.
$$

Proof. $\Rightarrow$ Let I be a model of G. We know that for all mappings $A _ { \bar { t } } : U \to T _ { G }$ , there exists a mapping $B _ { \bar { t } } : E \to T _ { G }$ such that $I [ A _ { \bar { t } } \bullet B _ { \bar { t } } ] ( F )$ is true. We also know (from Definition 32) that there exists a mapping $S _ { \overline { { t } } }$ such that $( A _ { \bar { t } } \bullet S _ { \bar { t } } ) ^ { t } ( F ) \subseteq F ^ { \prime }$ . We also know that, for every $x \in N$ there exists $\bar { t } \in T _ { G } ^ { n }$ and $y \in E$ such that $S _ { \bar { t } } ( y ) = x$ . We set $B : = B _ { \overline { { t } } } \circ S _ { \overline { { t } } } ^ { - 1 }$ . It follows that $I [ B ] ( F ^ { \prime } )$ is true, thus $I \left| = _ { b } U _ { N } ( G ) \right.$

⇐ Let I be a model of $U _ { N } ( G )$ . We know that there exists $B : N \to T _ { G }$ such that $I [ B ] ( F ^ { \prime } )$ is true. We also know (from Definition 32) that for all mapping $A _ { \bar { t } } : U \to T _ { G }$ there exists a mapping $S _ { \overline { { t } } }$ such that $( A _ { \bar { t } } \bullet S _ { \bar { t } } ) ^ { t } ( F ) \subseteq F ^ { \prime }$ . We set $B _ { \bar { t } } : = B \circ S _ { \bar { t } }$ . Then $I [ A _ { \bar { t } } \bullet B _ { \bar { t } } ] ( F )$ is true, thus $ I \left| = _ { b } G . \right. \ \left. \right.$

With universal extensions, we can provide the N3 version of the interpolation lemma for basic entailment:

▶ Lemma 34 (Generalised interpolation lemma). Given two finite normalised closed graphs $G _ { 1 } =$ $( U _ { 1 } , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } = ( U _ { 2 } , E _ { 2 } , F _ { 2 } )$ and universal extensions $U _ { N _ { 1 } } ( G _ { 1 } ) = ( \emptyset , N _ { 1 } , F _ { 1 } ^ { \prime } )$ of $G _ { 1 }$ and $U _ { N _ { 2 } } ( G _ { 2 } ) = ( \emptyset , N _ { 2 } , F _ { 2 } ^ { \prime } )$ of $G _ { 2 }$ then the following holds:

$G _ { 1 } \left| = _ { b } G _ { 2 } \right. \quad i f f$ there exists a mapping $M : N _ { 2 }  N _ { 1 }$ such that, for every triple $\left( { { s _ { 2 } } , { p _ { 2 } } , { o _ { 2 } } } \right)$ of an E-instance of $U _ { N _ { 2 } } ( G _ { 2 } )$ there is some triple $\left( s _ { 1 } , p _ { 1 } , o _ { 1 } \right) o f U _ { N _ { 1 } } ( G _ { 1 } )$ such that $s _ { 2 } , \ p _ { 2 }$ , and $O _ { 2 }$ , are isomorphic under M to $s _ { 1 } , p _ { 1 }$ , and $o _ { 1 }$ respectively.

Before proving the lemma, we demonstrate how it relates to the RDF interpolation lemma for simple entailment [24, Section 5.3] with the following corollary:

▶ Corollary 35 (RDF interpolation lemma). Given two finite normalised closed graphs $G _ { 1 } =$ $( \varnothing , E _ { 1 } , F _ { 1 } )$ and $G _ { 2 } = ( \emptyset , E _ { 2 } , F _ { 2 } )$ which do not contain graph terms. $G _ { 1 }$ b-entails a graph $G _ { 2 }$ if and only if a subgraph of $G _ { 1 }$ is an E-instance of $G _ { 2 }$

To see how the corollary is a special case of the lemma we first we take a closer look at the right hand side of the lemma. Note that universal extensions $U _ { N _ { 1 } } ( G _ { 1 } )$ and $U _ { N _ { 2 } } ( G _ { 2 } )$ are variants of $G _ { 1 }$ and $G _ { 2 }$ in which only the variables in $E _ { 1 }$ and $E _ { 2 }$ are renamed. As we exclude graph terms as components, the isomorphism M between the elements $s _ { 2 } , p _ { 2 } , o _ { 2 }$ and $s _ { 1 } , p _ { 1 } , o _ { 1 }$ is only an isomorphism in the RDF sense.

We come back to Lemma 34 and provide its proof:

Proof. ⇒ Let $G _ { 1 } \models _ { b } G _ { 2 }$ . We construct a model $I _ { U }$ for $U _ { N _ { 1 } } ( G _ { 1 } )$ . We set $\Delta _ { I _ { U } } = T _ { G } , D _ { I _ { U } } = i d$ where id stands for the identity function. We choose a representation of the isomorphic graph terms and let $Q _ { I _ { U } }$ map graph terms to their isomorphic image. For each $n \in N _ { 1 }$ we choose a unique IRI $B _ { 1 } ( n ) \not \in C ( G _ { 1 } ) \cup C ( G _ { 2 } )$ . We set $\mathit { E X T } _ { I _ { U } } : = \{ ( I _ { U } [ B _ { 1 } ] ( s _ { 1 } ) , I _ { U } [ B _ { 1 } ] ( p _ { 1 } ) , I _ { U } [ B _ { 1 } ] ( o _ { 1 } ) ) | ( s _ { 1 } , p _ { 1 } , o _ { 1 } ) \in F _ { 1 } ^ { \prime } \}$ We clearly see that ${ \cal I } _ { U } \quad \Longleftarrow _ { b } \ U _ { N _ { 1 } } ( G _ { 1 } )$ . By Lemma 33, it follows that $I _ { U } \subseteq U \subseteq G _ { 1 }$ , which by our assumption means that $I _ { U } \ \models _ { b } \ G _ { 2 }$ and thus, again by Lemma 33, ${ \cal I } _ { U } ~ \left| = _ { b } ~ U _ { N _ { 2 } } ( G _ { 2 } ) \right.$ That is, there is a mapping $B _ { 2 } \ : \ N _ { 2 } \ \to \ T _ { G }$ such that $I _ { U } [ B _ { 2 } ] ~ \ v { U } \ v { = } _ { b } ~ F _ { 2 } ^ { \prime }$ , or written diferently $( I _ { U } [ B _ { 2 } ] ( s _ { 2 } ) , I _ { U } [ B _ { 2 } ] ( p _ { 2 } ) , I _ { U } [ B _ { 2 } ] ( o _ { 2 } ) ) \in E X T _ { I _ { U } }$ for each triple $( s _ { 2 } , p _ { 2 } , o _ { 2 } ) \in F _ { 2 } ^ { \prime }$ . We can therefore always find a triple $( s _ { 1 } , p _ { 1 } , o _ { 1 } ) \in F _ { 1 } ^ { \prime }$ such that $( I _ { U } [ B _ { 1 } ] ( s _ { 1 } ) , I _ { U } [ B _ { 1 } ] ( p _ { 1 } ) , I _ { U } [ B _ { 1 } ] ( o _ { 1 } ) ) =$ $( I _ { U } [ B _ { 2 } ] ( s _ { 2 } ) , I _ { U } [ B _ { 2 } ] ( p _ { 2 } ) , I _ { U } [ B _ { 2 } ] ( o _ { 2 } ) )$ . We apply the mapping

$$
B _ { 2 } ^ { \prime } ( x ) : = { \left\{ \begin{array} { l l } { B _ { 2 } ( x ) } & { { \mathrm { ~ i f ~ } } B _ { 2 } ( x ) \in C ( G _ { 1 } ) } \\ { x } & { { \mathrm { ~ e l s e . } } } \end{array} \right. }
$$

to create an E-instance of $U _ { N _ { 2 } } ( G _ { 2 } )$ . The claim follows for the mapping

$$
M ( x ) : = { \left\{ \begin{array} { l l } { B _ { 1 } ^ { - 1 } \circ B _ { 2 } ( x ) } & { { \mathrm { ~ i f ~ t h e ~ v a l u e ~ e x i s t s , } } } \\ { n _ { 1 } } & { { \mathrm { ~ e l s e . } } } \end{array} \right. }
$$

where $n _ { 1 } \in N _ { 1 }$ is an arbitrary value.

⇐ Let $M : N _ { 2 }  N _ { 1 }$ be a mapping such that, for every triple $\left( { { s _ { 2 } } , { p _ { 2 } } , { o _ { 2 } } } \right)$ of an E-instance $G _ { 2 } ^ { \prime }$ of $U _ { N _ { 2 } } ( G _ { 2 } )$ there exists some triple $\left( { { s _ { 1 } } , { p _ { 1 } } , { o _ { 1 } } } \right)$ of $U _ { N _ { 1 } } ( G _ { 1 } )$ such that $s _ { 2 } , p _ { 2 } .$ , and $O _ { 2 }$ are isomorphic under M to $s _ { 1 } , p _ { 1 }$ , and $o _ { 1 } .$ , respectively. Let $I \left| = _ { b } U _ { N _ { 1 } } ( G _ { 1 } ) \right.$ . As G<sup>′</sup> is an E-instance of $U _ { N _ { 2 } } ( G _ { 2 } )$ there exists a mapping $W : X \subseteq N _ { 2 } \to T _ { G } \cup N _ { 2 }$ such that $G _ { 2 } ^ { \prime } = ( \emptyset , N _ { 2 } , W ^ { t } ( F _ { 2 } ^ { \prime } ) ) ^ { N }$ . We thus have that $( \emptyset , N _ { 1 } , ( M ^ { t } ( W ^ { t } ( F _ { 2 } ^ { \prime } ) ) )$ is isomorphic to a subgraph of $U _ { N _ { 1 } } ( G _ { 1 } )$ . It follows by Lemma 28 and Lemma 27, that $I \left| = _ { b } \right. ( \emptyset , N _ { 1 } , ( M ^ { t } ( W ^ { t } ( F _ { 2 } ^ { \prime } ) ) ) )$ ), that is, there exists a mapping $B : N _ { 1 } \to T _ { G }$ 2 such that $I [ B ] ( ( M ^ { t } ( W ^ { t } ( F _ { 2 } ^ { \prime } ) ) ) ) = t r u e$ . We extend W and M to total mappings $W ^ { \prime }$ and $M ^ { \prime }$ on $N _ { 2 } \cup T _ { G }$ which behave like the identity function in cases where $W ( x )$ , respectively $M ( x )$ are not defined. Then $I [ B \circ M ^ { \prime } \circ W ^ { \prime } ] ( F _ { 2 } ^ { \prime } ) = t r u \epsilon$ holds and thus $I \left| = _ { b } \ U _ { N _ { 2 } } ( G _ { 2 } ) \right.$ . The claim follows by Lemma 33. ◀

Note that Corollary 35 difers from $\mathrm { R D F } \mathrm { s }$ interpolation lemma as stated in the specification in one detail: RDF allows the graphs to be infinite while we only talk about finite graphs. The reason is that is that our proof relies on finding "fresh" ground terms the existential variables can map to (variables need to correspond to ground terms). This is not possible for a graph G using all possible ground terms, i.e. $C ( G ) = T _ { G }$ . For such a graph these terms always carry additional meaning. We aim to address this issue in future work.

## 5.4 N3-entailment

In the previous sections we introduced and discussed N3’s basic-entailment which – as we explained – is very similar to RDF’s simple entailment. On top of this, we define N3-entailment which covers the N3-predicates collected under the log-namespace<sup>7</sup>. We introduce the predicates log:implies and log:includes.

▶ Definition 36 (N3 models and entailment). Let I be an interpretation and G and H be graphs. We call I an N3 model of G, noted ${ \cal I } | = _ { n 3 } G , i f f { \cal I } | = _ { b } G $ and I satisfies the following conditions:

I satisfies the the conditions for log:implies (see Definition 37)

I satisfies the the conditions for log:includes (see Definition 40)

We say that G N3-entails H, written as $G \models _ { n 3 } H$ if for each interpretation I it holds that if $I | = _ { n 3 } G$ then $I \Vdash _ { n 3 } H$

Note, that the log-namespace covers a wider range of predicates supporting, for example operations on literals (log:dtlit, log:langlit) or types of terms (log:rawType). We omit those here as the primary focus of this paper are graph terms. To take into account the missing cases, the above definition can be extended.

Below, we provide the definitions for the two predicates. We start with the predicate log:implies, which is used for implications.

▶ Definition 37 (Semantic conditions for log:implies). The semantic conditions on an interpretation I for log:implies are:

for all s, o ∈ ∆<sub>I</sub>, (s, D<sub>I</sub>(log:implies), o) ∈ EXT<sub>I</sub> holds only if either

## 42:22 Notation3: Graph Terms for RDF

$s = G _ { s }$ and $o = G _ { o }$ are both graph terms and ${ \cal I } \vdash _ { n 3 } G _ { o }$ holds $i f I \models _ { n 3 } G _ { s }$

$\mathbf { \xi } = \mathbf { \zeta } \mathbf { \mathcal { S } } \mathbf { \zeta } O r \mathbf { \zeta } g$ is not a graph term

To illustrate how this definition makes it possible to apply rules on data, we come back to our example from the previous section:

$$
\left\{ \left( : \mathrm { t o m } , \mathrm { r d f : t y p e } , \mathrm { : H u m a n } \right) \right\} \qquad \mathrm { }\tag{32}
$$

$$
\begin{array} { r l } & { \mathrm { A n y ~ m o d e l ~ } I \mathrm { ~ o f ~ t h e s e ~ t w o ~ t r i p l e s ~ m u s t ~ s a t i s f y : } } \\ & { = \begin{array} { r l } & { ( D _ { I } ( : \mathrm { t o m } ) , D _ { I } ( \mathrm { r d f . t y p e } ) , D _ { I } ( : \mathrm { H u m a n } ) ) \in E X T _ { I } } \\ & { = \hphantom { ( } ( D _ { I } ( \{ \left( : \mathrm { t o m } , \mathrm { r d f . t y p e } , \cdot \mathrm { H u m a n } \right) \} ) , D _ { I } ( \mathrm { l o g . i m p l i e s } ) , D _ { I } ( \exists \{ y \} \{ \left( : \mathrm { t o m } , \cdot \mathrm { h a s M o t h e r } , \cdot \mathrm { y } \right) \} ) } \end{array} } \end{array}\tag{31}
$$

By the semantic condition on log:implies (first bullet), and the fact that I is a model for the graph on the left hand side (“tom type human”), we derive that I is also a model for the graph on the right-hand side. The graph above thus N3-entails

$$
\exists \{ y \} \{ ( : \mathrm { t o m } , : \mathrm { h a s M o t h e r } , y ) \} .\tag{33}
$$

If we were to replace :tom in the second line with a universally quantified variable x (as in Formula 19), the implication still holds. First, consider the implication as a “regular” assertion containing a universal. This assertion must hold for every possible value of ?x, including :tom, so the second bullet above would still hold in every basic model of the graph. This would in turn lead to the N3-entailment that $\exists \{ y \}$ {(:tom, :hasMother, y)}., as above.

The example also shows that N3-entailment supports a form of existential rules. Thus, in general, it is not decidable whether one graph is N3-entailed by another [9]. Other rule frameworks for RDF either exclude blank nodes on rule heads (e.g. SWRL [25]) or limit how often such rules can be applied (e.g. SPARQL RL [18]). As blank nodes are widely used and important in diferent contexts, N3 allows them in rule heads. Some reasoners like for example EYE [37] provide mechanisms to tame such rules by diferent mechanisms.

The above mechanism even allows us to cover the "unquoting" example from Section 2 which had a variable as rule head:

$$
\mathrm { : b o b ~ : s a y s ~ \{ : a l i c e ~ : l i k e s ~ : p i z z a , ~ : p a s t a \} . }\tag{2}
$$

$$
\{ : { \mathrm { b o b } } : { \mathrm { s a y s } } ? { \mathrm { x } } \} \ \Rightarrow \ ? { \mathrm { x } } \ .\tag{5}
$$

Represented in abstract syntax, these formulae can be written as:

$$
\{ ( { \mathrm { : b o b , : s a y s , } } \{ ( { \mathrm { : a l i c e , : l i k e s , : p i z z a } } ) , ( { \mathrm { : a l i c e , : l i k e s , : p a s t a } } ) \} ) \}\tag{40}
$$

$$
\forall \{ x \} \{ ( \{ ( { \mathrm { : b o b , : s a y s , } } x ) \} , { \mathrm { l o g : i m p l i e s } } , x ) \}\tag{41}
$$

By simple entailment, we obtain Formula 7 which we represent in abstract semantics as:

$$
\{ ( \{ ( \mathrm { : b o b , : s a y s , \{ ( : a l i c e , : l i k e s , : p i z z a ) , ( : a l i c e , : l i k e s , : p a s t a ) \} } ) \} , \mathrm { l o g : i m p l i e s , } \}
$$

$$
\{ ( { \mathrm { : a l i c e , : l i k e s , : p i z z a } } ) , ( { \mathrm { : a l i c e , : l i k e s , : p a s t a } } ) \} \}\tag{42}
$$

With the condition for log:implies, we derive from Formulae 40 and 41:

$$
\{ ( \mathrm { { : a l i c e } , \mathrm { { : l i k e s } , \mathrm { { : p i z z a } } } } ) , ( \mathrm  { : a l i c e } , \mathrm { { : l i k e s } , \mathrm { { : p a s t a } } } ) \}\tag{43}
$$

which is the abstract representation of Formula 6.

Note that the semantic condition above is a one-way implication (“only if”). This means that given two graphs G and H such that $G \models _ { n 3 } H$ , it does not necessarily hold that G log:implies H. Note also that the semantic condition allows for log:implies to hold between a non-graph term and any other term (last sub-bullet in Definition 37), in which case, it does not have any special meaning.

Similarly to the conditions shown above, we define the semantics for the predicate log:includes. The predicate was initially introduced to mimic basic entailment (more precisely its predecessor N3 derivation [13, Section 5.7]). It has two ground graph terms as arguments and is true if the second graph is b-entailed by the first one. A simple example of such a true triple (in N3 syntax) is:

$$
\{ : { \texttt { s } } : { \texttt { p } } : { \texttt { o . } } : { \texttt { a } } : { \texttt { b } } : { \texttt { c } } \} \ \mathrm { ~ l o g : i n c l u d e s ~ } \{ : { \texttt { a } } : { \texttt { b } } : { \texttt { c } } \}\tag{44}
$$

As the second argument of the predicate is a subgraph of the first we clearly see it is b-entailed by it (see Lemma 28). This example is easy to implement and supported by most reasoners we are aware of. However, b-entailment including universal quantification is rather complex as we illustrated earlier on Formulae 36-39. To the best of our knowledge, there is furthermore no reasoner supporting it and most practical use cases only require a simpler form of entailment which does not $_ \mathrm { g o }$ beyond the expressivity of RDF’s interpolation lemma (Corollary 35)<sup>8</sup>. We therefore exclude universal quantification here and define:

▶ Definition 38 (Existential consequence). If a subgraph of a finite closed N3 graph $G _ { 1 }$ is isomorphic to an E-instance of an N3 graph $G _ { 2 }$ , we call $G _ { 2 }$ a existential consequence of $G _ { 1 }$

From our considerations in Section 5.2, we immediately see:

▶ Corollary 39 (Existential Consequence). An N3 graph b-entails any of its existential consequences.

Proof. This follows directly from Lemmas 27, 28 and 29.

Note that even graphs with universally quantified variables can have existential consequences. The concept only limits the kind of derivation steps taken into account and explicitly excludes consequences produced by instantiating universals. The graph

$$
\boldsymbol { G } = \forall \{ x \} \exists \{ y \} \{ ( x , : p , y ) \}\tag{45}
$$

is an existential consequence of graph $G _ { 1 }$ in Formula 34.

We use the existential consequence to define the semantics of the predicate log:includes.

▶ Definition 40 (Semantic conditions for log:includes). The semantic conditions on an interpretation I for log:includes are:

$\begin{array} { r l } { = } & { { } ( s , D _ { I } ( l o g \cdot i n c l u d e s ) , o ) \in E X T _ { I } } \end{array}$ if and only if $s = \underline { { G _ { s } } }$ and $o = \underline { { G _ { o } } }$ are both graph terms and $G _ { o }$ is an existential consequence of $G _ { s }$

Note that the semantic condition above is a two-way implication (“if and only if”). In other words, the extension of log:includes contains exactly all the pairs of graph terms present in $\Delta _ { I }$ such that the second one is an existential consequence of the first one.

## 6 Discussion of the requirements

Having defined the concept of basic semantics and N3 semantics, we now come back to the requirements identified in Section 3 discussing them one by one.

Semantic compatibility with RDF We defined basic entailment (Definition 23) and characterised it through a generalised interpolation lemma (Lemma 34). We furthermore showed that for RDF graphs b-entailment and simple entailment coincide (Lemma 26) and that the N3 interpolation lemma for b-entailment (Lemma 34) is an extension of the RDF interpolation lemma for simple entailment on finite graphs (Corollary 35). The main diference between the two is that the RDF interpolation lemma also holds for infinite graphs while the N3 interpolation lemma is limited to finite graphs. As entailment between infinite RDF graphs is rarely needed in practice, we consider the diference between both forms of entailment minimal<sup>9</sup>. The semantics we defined is otherwise in line with RDF-entailment and the requirement is met.

Referential Opacity The second requirement we identified was referential opacity. We meet this requirement by semantically representing graph terms as syntactic objects in base semantics which can then be further refined in N3 semantics – or possible other extensions – like we exemplified on implications and inclusion. We come back to our superman example (Formulae 8 and 9), which we represent in abstract syntax

$$
\{ ( \mathrm { { : s u p e r m a n } , o w l { : s a m e A s } , : c l a r k K e n t } ) \}\tag{46}
$$

$$
\left\{ \mathrm { ( : l o i s L a n e , : k n o w s , \{ ( : s u p e r m a n , : c a n , : f l y ) \} } \right\}\tag{47}
$$

Even with an interpretation I where I(:superman) = I(:clarkKent), we have:

$$
\begin{array} { r } { I ( \underbrace { \{ \{ \{ \langle { \mathrm { s u p e r m a n } } , \mathrm { : c a n } , \{ \mathrm { f l y } } \} \} }  _ { \ne \ I ( \{ \{ \mathrm { : c l a r k K e n t , : c a n } , \mathrm { : f l y } } \} )  ) = \underbrace { \{ \{ ( \mathrm { : s u p e r m a n } , \mathrm { : c a n } , \mathrm { : f l y } ) \} }  _ { \downarrow } )  \\ { \ne I ( \{ \{ \{ \mathrm { : c l a r k K e n t , : c a n , : f l y } \} \} } ) = \underbrace { \{ ( \mathrm { : c l a r k K e n t , : c a n , : f l y } ) \} } _ { \ge \ I ( \{ \mathrm { : c l a r k K e n t , : c a n , : f l y } \} ) }  \end{array}
$$

With Rule 11 which we write in abstract syntax as

$$
\forall \{ x , y , s , p , q , r \} \{ ( \{ ( x , \mathrm { o w t . s a m e A s } , y ) , ( s , p , \{ x , q , r \} ) \} , \mathrm { l o g . i m p l i e s } , \{ ( s , p , \{ ( y , q , r ) \} ) \} ) \}\tag{48}
$$

This rule b-entails

$$
\{ ( \{ ( \mathrm { s u p e r m a n } , \mathrm { o w l } \mathrm { s a m e A s } , \mathrm { z e l a r k K e n t } ) , ( \mathrm { : l o i s L a n e } , \mathrm { : k n o w s } , \{ ( \mathrm { : s u p e r m a n } , \mathrm { : c a n } , \mathrm { : d y } ) \} )  \} ,
$$

$$
\mathrm { l o g  i m p l i e s , \{ ( : l o i s L a n e , : k n o w s , \{ ( : c l a r k K e n t , : c a n , : d y ) \} ) \} } \}\tag{49}
$$

and with N3-entailment we get

$$
\{ ( { \mathrm { : l o i s L a n e , : k n o w s , } } \{ ( { \mathrm { : c l a r k K e n t , : c a n , : f l y } } ) \} ) \}\tag{50}
$$

Our semantics supports referential opacity, while allowing to locally enable the derivations that would follow from referential transparency, as illustrated above.

Closedness of graph terms Another requirement we identified was that graph terms in N3 need to be closed as they are used to represent premises and conclusions of rules. We again meet this criterion by interpreting graph terms by an isomorphic syntactical representation. In the case of implications, we further interpret the graph terms as graphs which in case they are true in a model can be used to derive new facts. If needed, IRIs could be used to represent open graphs. We could for example state

{(:loisLanesKnowledge, :contains, {(:superman, :can, :fly)})}

and provide semantics for the predicate :contains which ensure that the subject of the predicate represents a graph consisting of at least the triples occurring in its object graphs. This idea has been discussed in previous work to represent RDF datasets using N3 [7].

Clarity and flexibility of quantification scopes We introduced explicit quantification to avoid possible ambiguities arising from implicit scoping. For each graph we declare the set of universally and existentially quantified variables. This makes our semantics independent of the concrete syntax whose scoping of implicit quantification is still subject to discussion. Our semantics with explicit scoping makes it possible to better understand the impact of future scoping decisions and possible restrictions to the language. In our approach, we chose to follow N3’s original specification in which the scope of universal specification is always outside the scope of existential quantification when these are declared on the same graph [13, p.20]. At least if we employ N3 entailment and not just basic entailment, this design decision does not restrict the expressivity of our logic. For example, if we want to modify our initial example on quantification ("Everybody loves somebody", Formula 12) to express that there exists someone who is loved by everyone, we can state

$$
\exists \{ y \} \{ ( \underline { { \{ \} } } , \mathrm { l o g : i m p l i e s } , \forall \{ x \} \{ ( x , : l o v e s , y ) \} ) \}\tag{51}
$$

According to our semantics, the empty graph is always true and in an N3 model the formula has thus the intended meaning.

## 7 Related work

The work presented in this paper has been influenced by several other publications.

First, it relies on the original contributions introducing Notation3 Logic: W3C team submission [12], a web page by Tim Berners-Lee explaining design issues of the framework [11], and a research paper detailing the logic [13]. All N3 features discussed in this paper were informally introduced in these early specifications of the logic; however, none of them had formal semantics. This explicitly includes graph terms and implicit quantification of two diferent types. N3 also ofers the means to state explicit quantification (see also Formulae $1 2 ^ { \prime \prime }$ and 20). In the past, this feature often lead to problems because N3 treats the quantifiers themselves as parts of the graphs and thus as elements of sets. This makes it dificult to determine the proper scope of a quantifier and their orders. The specification allows us to write Formula $1 2 ^ { \prime \prime }$ from above as

$$
\begin{array} { r } { \mathfrak { Q } \mathbf { f } \mathrm { o r } \mathbf { S } \mathrm { o m e } \mathbf { \Lambda } : \mathbf { y } . \quad \mathfrak { Q } \mathbf { f } \mathrm { o r } \mathbf { A } \mathbf { l } \mathbf { l } \textbf { \Lambda } : \mathbf { x } . \quad : \mathbf { x } \textbf { \Lambda } : \mathbf { l o v e s \ } : \mathbf { y } . } \end{array}\tag{12<sup>′′′</sup>}
$$

without changing its meaning ("everybody loves somebody"). As we consider this detail confusing, we impose a fixed order for the two kinds of quantifiers. Not only explicit quantification but also its implicit counterparts has lead to contradicting interpretations. In previous work, we investigated the extent and impact of these disagreements and proposed a mapping to a logic with explicit quantification [5, 6]. This mapping could cope with formulae which are similar to First Order Logic like, for example, the implication detailed in Formula 4. But it did not cover cases like the unquoting exemplified in Formula 5 or the built-in log:includes. The development of N3 was continued by a W3C community group which provided specifications for the syntax of the language [35], its semantics [8], and its built-in functions [36]. The semantics developed in this context forms the base of the work we present in this paper. In addition to refining the definitions of that specification, we furthermore investigate N3’s formal relationship to RDF simple entailment and add a semantic definition for the predicate log:includes. In earlier work this predicate relies on a simplified form of entailment called n3-derivation [13, p.13]. Even though n3-derivation was not fully detailed when introduced, we see a strong similarity to basic entailment as it was introduced in this paper.

The idea of referencing to graphs, formulae, or statements has not only been discussed in the context of N3 but can also be found in diferent frameworks for the web and beyond. We first discuss web formalisms to then broaden our discussion towards logic in general. In the moment of writing, the W3C RDF & SPARQL working group is developing RDF 1.2 [28] and SPARQL 1.2 [23] which add triple terms to RDF and SPARQL. Conceptually, these are very similar to graph terms with the main diference that users can not use whole graphs as terms of triples but only single triples. RDF 1.2 originates from RDF-star (also RDF<sup>∗</sup>) [22] which was originally proposed by Hartig [21]. In RDF 1.2 the use of triple terms is restricted and they are only allowed in the object position of a triple. Blank nodes in triple terms are globally quantified and triple terms are referentially transparent. In both of these aspects, triple terms difer from N3 graph terms. A singleton N3 graph term is not the same as a triple term in RDF 1.2. RDF does furthermore not support rules and it does not support any mechanism to unquote triple terms as illustrated for N3 with Example 5.

Before the rise and standardisation of triple terms, RDF reification [24, Appendix D1] was used to make statements about triples. There was no special syntactic construct to refer to triples but these were described in RDF’s triple syntax. A reified triple was represented by an IRI or blank node in combination with predicates making connections to the diferent components of a triple (rdf:subject, rdf:predicate, rdf:object). There was no semantics for RDF reification and no guarantees or checks that a reified triple was well-formed (we could for example have two subject declarations for the same term). However, by the way RDF reification is done, we can assume that it is referentially transparent and that all existential quantification is global.

As an alternative to RDF reification, Nguyen et al. proposed singleton properties [31]. The idea behind this concept is to use, in each triple to be reified, a fresh predicate that is related to the original property. That predicate can then be used to refer to the single triple where it is used. This approach to refer to triples does not only difer syntactically from the former two, it also difers in another sense: while triple terms and also reification, just like N3 graph terms, can be used without stating the triple, this is not possible with singleton properties. We can, for example, not express Formula 8 without asserting that :superman :can :fly.

To refer to graphs RDF 1.1 furthermore introduced the concept of RDF datasets [30, Section 4]. Here, a combination of a graph name and a graph can be stated in an RDF document and the name can then further be used to refer to that construct. Although RDF datasets are part of the RDF 1.1 recommendation, their semantics has not been fixed and there exist diferent proposals [39] addressing the diferent aspects identified in Section 3 in various ways most of which can be represented by N3 [7]. The semantics introduced by Carroll et al. [16] is the closest to the graph term semantics introduced in this paper. A named graph in this semantics is a pair composed of a name and a graph, and that name can be used in triples and always refers to the pair itself. As N3 graph terms do not have names, we observe a diference here. The graph is closed, blank nodes are local, and the graph itself is interpreted as a syntactical object thereby supporting referential opacity.

Outside of the web context, Common Logic [26] defines constructs similar to RDF datasets and their semantics according to Carroll et al. Here, we can again combine formulae with names to then use these names in other formulae. In contrast to RDF datasets, Common Logic supports universal quantification but it does not support to quantify into a formula. The expression

$$
\forall \{ x \} \{ ( x , \mathrm { s a y s } , \{ ( x , \mathrm { i s } , \mathrm { w i s e } ) \} ) \}\tag{52}
$$

is supported by our semantics of N3 but cannot be expressed in Common Logic. The dificulty here is that the quantified variable x occurs inside and outside of the graph term. We solve that by first grounding the variable, then further interpret it according to the position where it is used (once as a direct term, once as part of the graph term). Common Logic does furthermore not support unquoting. In contrast to that, Qiana, a framework developed by Coumes et al. [17], supports both quantifying into a formula/graph term and unquoting. The mechanism to do so difers from our approach, as its formalisation treats quoted formulae as logical functions which directly take terms of the domain of discourse as arguments. This can be used to resolve quantification over variables used simultaneously inside and outside a quoted formula – we use grounding instead. Qiana furthermore diverges from N3 in that its main focus is to display formulae together with some context under which they may or may not be true. In that regard, it is close to the diferent variants of modal logic [14]. We adopt Qiana’s notation of underlining formulae/graph terms to indicate their mention rather than use.

The concept of universal extension of N3 graphs (Definition 32) and the related theory have been influenced by the work around existential rules [9] (also called tuple-generating dependencies [1] or $D a t a l o g ^ { \pm }$ [15]). Here, the chase procedure can be used to construct universal models [19]: existential rules are applied repeatedly to a growing set of facts, where each rule application adds new facts unless its head already maps homomorphically into the existing ones. In the case of basic entailment, we do not have rules but ∀∃-formulae which could be seen as existential rules with empty bodies. To mimic RDF’s interpolation lemma, we therefore have to apply them on all terms producing infinite models. Existential rules are furthermore closely related to N3 rules employing the built-in predicate log:implies [4, 3] and query answering for N3 entailment faces the same problems as for existential rules and is in general undecidable.

## 8 Conclusion and future work

We have proposed an abstract syntax and a formal semantics for N3. Compared to previous eforts in that direction, our proposal aims to cover the diferent ways in which graph terms have been used historically in N3: statements about statements, and rules. This formal semantics is defined in two levels: basic semantics (comparable to RDF’s simple entailment), and N3 semantics, which includes N3’s special predicates, notably the rule-defining predicate log:implies. We have shown that this formal semantics satisfies the requirement we identified as meeting those diferent use-cases, and that RDF’s interpolation lemma for simple entailment can be extended to N3’s basic semantics (on finite graphs).

While we proved our generalized interpolation lemma for finite graphs only, we conjecture that a proof may exist for infinite graphs as well. As a matter of fact, our proof does not prevent the graphs to have an infinite number of triples; it only requires them to not use all the variables (so that we can build a universal extension) nor all the ground terms (so that we can build a mapping $B _ { 1 } )$ . But we believe that even those constrains could be lifted, and plan to address them in future work.

We also acknowledge that our generalized interpolation lemma is far less practical than its counterpart in RDF: it relies in the universal extension of N3 graphs, which are infinite as soon as the graph contains a universal variable (unlike universal models [19], the concept of which universal extensions are inspired). In the future, we wish to provide an entailment procedure which only takes the terms occurring in the two involved graphs into account similar to the RDFS entailment rules provided in the RDF specification [34, 24].

## 42:28 Notation3: Graph Terms for RDF

Finally, while we formally define in N3-entailment the semantics of the special predicates log:implies and log:includes, the original N3 specification (and its implementations) provide much more special predicates. We plan to include them, following the example of the two we described in this paper. Note that some of these additional properties will include new challenges, such as negation (e.g. log:notIncludes<sup>10</sup>) and aggregation (e.g. log:forAllIn<sup>11</sup>). As N3 allows variables to occur in the predicate position of triples, these concepts cannot rely on classical stratification [2]. We plan to base our definitions on chain stratification [29], a concept recently proposed to overcome that problem.

## References

1 Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley, 1995. URL: http://webdam.inria.fr/Alice/.

2 Krzysztof R. Apt, Howard A. Blair, and Adrian Walker. Towards a theory of declarative knowledge. In Jack Minker, editor, Foundations of Deductive Databases and Logic Programming, pages 89–148. Morgan Kaufmann, 1988. doi:10.1016/B978-0-934613-40-8.50006-3.

3 Dörthe Arndt and Stephan Mennicke. Notation3 as an existential rule language. In International Joint Conference on Rules and Reasoning, pages 70–85. Springer, 2023. doi:10.1007/ 978-3-031-45072-3\_5.

Dörthe Arndt and Stephan Mennicke. Existential Notation3 logic. Theory and Practice of Logic Programming, pages 1–36, June 2025. doi: 10.1017/S1471068425000055.

5 Dörthe Arndt, Tom Schrijvers, Jos De Roo, and Ruben Verborgh. Implicit quantification made explicit: How to interpret blank nodes and universal variables in Notation3 logic. Journal of Web Semantics, 58:100501, 2019. doi:10.1016/j.websem. 2019.04.001.

6 Dörthe Arndt, Ruben Verborgh, Jos De Roo, Hong Sun, Erik Mannens, and Rik Van de Walle. Semantics of Notation3 logic: A solution for implicit quantification. In Rule Technologies: Foundations, Tools, and Applications: 9th International Symposium, RuleML 2015, Berlin, Germany, August 2-5, 2015, Proceedings 9, pages 127–143. Springer, 2015. doi:10.1007/978-3-319-21542-6\_9.

7 Dörthe Arndt and William Van Woensel. Towards supporting multiple semantics of named graphs using N3 rules. In Ahmet Soylu, Sotiris Moschoyiannis, Guido Governatori, Mantas Simkus, Petros S. Stefaneas, Alexander Steen, and Adrian Giurca, editors, Proceedings of the 13th RuleML+RR 2019 Doctoral Consortium and Rule Challenge, September 16-19, 2019 - Bolzano, Italy, volume 2438 of CEUR Workshop Proceedings. CEUR-WS.org, 2019. URL: https://ceur-ws. org/Vol-2438/paper6.pdf.

8 Dörthe Arndt and Pierre-Antoine Champin. Notation3 semantics. W3C community group report, W3C, October 2025. https://w3c.github.io/N3/ reports/20230703/semantics.html.

9 Jean-François Baget, Michel Leclère, Marie-Laure Mugnier, and Eric Salvat. On rules with existential variables: Walking the decidability line. Artif. Intell., 175(9-10):1620–1654, 2011. doi: 10.1016/J.ARTINT.2011.03.002.

10 Tim Berners-Lee. Cwm, 2000–2009. http://www. w3.org/2000/10/swap/doc/cwm.html.

11 Tim Berners-Lee. Notation 3 Logic. Technical report, W3C, 2005. URL: https://www.w3.org/ DesignIssues/Notation3.

12 Tim Berners-Lee and Dan Connolly. Notation3 (N3): A readable RDF syntax. Technical report, W3C, 2011. URL: https://www.w3.org/ TeamSubmission/n3/.

13 Tim Berners-Lee, Dan Connolly, Lalana Kagal, Yosi Scharf, and Jim Hendler. N3Logic: A logical framework for the World Wide Web. Theory and Practice of Logic Programming, 8(3):249–269, 2008. URL: http://arxiv.org/abs/0711.1533, doi:10.1017/S1471068407003213.

14 Patrick Blackburn, Maarten de Rijke, and Yde Venema. Modal Logic, volume 53 of Cambridge Tracts in Theoretical Computer Science. Cambridge University Press, 2001. doi:10.1017/ CBO9781107050884.

15 Andrea Calì, Georg Gottlob, and Thomas Lukasiewicz. Datalog<sup>±</sup>: a unified approach to ontologies and integrity constraints. In Ronald Fagin, editor, Database Theory - ICDT 2009, 12th International Conference, St. Petersburg, Russia, March 23-25, 2009, Proceedings, volume 361 of ACM International Conference Proceeding Series, pages 14–30. ACM, 2009. doi:10.1145/1514894. 1514897.

16 Jeremy J. Carroll, Christian Bizer, Patrick J. Hayes, and Patrick Stickler. Named graphs, provenance and trust. In Allan Ellis and Tatsuya Hagino, editors, Proceedings of the 14th international conference on World Wide Web, WWW 2005, Chiba, Japan, May 10-14, 2005, pages 613–622. ACM, 2005. doi:10.1145/1060745.1060835.

17 Simon Coumes, Pierre-Henri Paris, François Schwarzentruber, and Fabian M. Suchanek. Qiana: A first-order formalism to quantify over contexts and formulas with temporality. J. Artif. Intell. Res., 85, 2026. doi:10.1613/JAIR.1.18402.

18 Robert David, David Habgood, Andy Seaborne, and Simon Steyskal. SPARQL 1.2 RL. W3C working draft, W3C, September 2026. https: //www.w3.org/TR/sparql12-rl/.

19 Alin Deutsch, Alan Nash, and Jefrey B. Remmel. The chase revisited. In Maurizio Lenzerini and Domenico Lembo, editors, Proceedings of the Twenty-Seventh ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS 2008, June 9-11, 2008, Vancouver, BC, Canada, pages 149–158. ACM, 2008. doi:10.1145/1376916.1376938.

20 Michael Genesereth and Eric Kao. The Herbrand manifesto. In Nick Bassiliades, Georg Gottlob, Fariba Sadri, Adrian Paschke, and Dumitru Roman, editors, Rule Technologies: Foundations, Tools, and Applications, pages 3–12, Cham, 2015. Springer International Publishing. doi:10.1007/ 978-3-319-21542-6\_1.

21 Olaf Hartig. Foundations of RDF\* and SPARQL\*:(an alternative approach to statementlevel metadata in RDF). In AMW 2017 11th Alberto Mendelzon International Workshop on Foundations of Data Management and the Web, Montevideo, Uruguay, June 7-9, 2017., volume 1912. Juan Reutter, Divesh Srivastava, 2017. URL: https://ceur-ws.org/Vol-1912/paper12.pdf.

22 Olaf Hartig, Pierre-Antoine Champin, Gregg Kellogg, and Andy Seaborne. RDF-star and SPARQLstar. W3C community group report, W3C, December 2021. https://www.w3.org/2021/12/ rdf-star.html.

23 Olaf Hartig, Andy Seaborne, Ruben Taelman, Gregory Williams, and Thomas Pellissier Tanon. SPARQL 1.2 query language. W3C working draft, W3C, September 2026. https://www.w3.org/TR/ sparql12-query/.

24 Pat Hayes and Peter F. Patel-Schneider. RDF 1.1 Semantics. W3C recommendation, W3C, February 2014. URL: https://www.w3.org/TR/ rdf11-semantics/.

25 Ian Horrocks, Peter F Patel-Schneider, Harold Boley, Said Tabet, Benjamin Grosof, and Mike Dean. SWRL: A semantic web rule language combining OWL and RuleML. W3C member submission, W3C, May 2004. https://www.w3.org/ Submission/2004/SUBM-SWRL-20040521/.

26 ISO/IEC. Information technology — Common Logic (CL): A framework for a family of logic-based languages. Standard ISO/IEC 24707:2018, International Organization for Standardization, Geneva, Switzerland, 2018.

27 Tobias Käfer and Andreas Harth. Rule-based programming of user agents for linked data. LDOW@ WWW, 2073, 2018. URL: https://ceur-ws.org/ Vol-2073/article-05.pdf.

28 Gregg Kellogg, Olaf Hartig, Pierre-Antoine Champin, and Andy Seaborne. RDF 1.2 concepts and abstract syntax. W3C candidate re-

commendation snapshot, W3C, April 2026. URL: https://www.w3.org/TR/rdf12-concepts/.

29 Nils Küchenmeister, Alex Ivliev, Dörthe Arndt, and Markus Krötzsch. Stratified negation in RDF rules: A correct approach. In Manolis Koubarakis, Maria-Esther Vidal, Axel Polleres, Marieke van Erp, Ernesto Jimenez Ruiz, Oshani Seneviratne, Lora Aroyo, Gianluca Demartini, Reham Alharbi, Roberto Barile, Claudia d’Amato, and Valentina Tamma, editors, Proceedings of the 25th International Semantic Web Conference (ISWC 2026). Springer, 2026.

30 Markus Lanthaler, David Wood, and Richard Cyganiak. RDF 1.1 concepts and abstract syntax. W3C recommendation, W3C, February 2014. URL: https://www.w3.org/TR/2014/ REC-rdf11-concepts-20140225/.

31 Vinh Nguyen, Olivier Bodenreider, and Amit Sheth. Don’t like RDF reification? Making statements about statements using singleton property. In Proceedings of the 23rd international conference on World Wide Web, pages 759–770, 2014. doi:10.1145/2566486.2567973.

32 Eric Prud’hommeaux and Gavin Carothers. RDF 1.1 turtle. W3C recommendation, W3C, February 2014. URL: https://www.w3.org/TR/2014/ REC-turtle-20140225/.

33 Eric Prud’hommeaux and Gavin Carothers. RDF 1.1 turtle. W3C recommendation, W3C, February 2014. https://www.w3.org/TR/2014/REC-turtle-20140225/.

34 Herman J. ter Horst. Completeness, decidability and complexity of entailment for RDF schema and a semantic extension involving the OWL vocabulary. J. Web Semant., 3(2-3):79–115, 2005. doi:10.1016/J.WEBSEM.2005.06.001.

35 William Van Woensel, Dörthe Arndt, Pierre-Antoine Champin, Dominik Tomaszuk, and Gregg Kellogg. Notation3 language. W3C community group report, W3C, October 2025. https://w3c. github.io/N3/reports/20230703/.

36 William Van Woensel and Patrick Hochstenbach. Notation3 builtin functions. W3C community group report, W3C, July 2023. https://w3c.github.io/N3/reports/20230703/ builtins.html.

37 Ruben Verborgh and Jos De Roo. Drawing conclusions from linked data on the web: The EYE reasoner. IEEE Software, 32(5):23–27, May 2015. doi:10.1109/MS.2015.63.

38 W3C OWL Working Group. OWL 2 web ontology language document overview (second edition). W3C recommendation, W3C, December 2012. https://www.w3.org/TR/owl2-overview/.

39 Antoine Zimmermann. RDF 1.1: On semantics of RDF datasets. W3C working group note, W3C, February 2014. URL: https://www.w3.org/TR/ rdf11-datasets/.