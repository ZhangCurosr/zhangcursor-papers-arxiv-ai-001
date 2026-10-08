# Cognitive Schemas, Laws and Tasks

Antal Jakovác<sup>1</sup> András Telcs<sup>1,\*</sup>

<sup>1</sup>HUN-REN Wigner Research Centre for Physics, Institute for Particle and Nuclear Physics, Computational Sciences Department, Data and Compute Intensive Sciences Research Group, Budapest, Hungary.

E-mail: jakovac.antal@wigner.hun-ren.hu

Corresponding author: András Telcs,

telcs.andras@wigner.hun-ren.hu

ORCID (András Telcs): 0000-0002-3205-3081

## Abstract

This paper asks how explicit representations can support reusable cognitive schemas in knowledge-based problem solving. We develop a structural framework in which schemas are organized by the information and relations required for their use, rather than introduced as unrelated primitives. The framework also distinguishes context-dependent relations from more stable structures that can be reused across diferent representations.

Tasks are described through the information available, the unknowns to be determined, and the constraints that admissible solutions must satisfy. This makes it possible to separate limitations of the representation from limitations of the solving procedure. In particular, we distinguish inconsistency, underdetermination, and contextual insuficiency, where the current representation lacks distinctions or relations required by the external task meaning. We also show formally when a reduction of representation preserves the task-relevant solution structure.

The resulting task–schema interface ofers a structured way to describe representational conditions relevant to problem solving. It supports the reuse and stabilization of derived knowledge while remaining independent of the particular mechanism used to generate candidate solutions. This may provide a useful component for future solver architectures that combine structured knowledge, verification, and learned proposal mechanisms.

Keywords: cognitive schemas, structural representation, Knowledge Space, context, task formulation, problem solving

## 1 Introduction

Knowledge representation is a central component of intelligent problem solving: the form in which a task and its objects are represented determines which relations, transformations, and solution procedures are available. This paper develops a structural account of cognitive schemas within the representational framework introduced in [1] and further discussed in [2]. That framework provides finite contexts, invariance relations, representative objects, and a Knowledge Space as a substrate for problem solving. Here we ask which higher-level schemas can be derived from this substrate without introducing additional representational primitives.

The main idea is that notions such as equivalence, order, successor, conditional dependence, causal dependence, discrete succession, number-like structure, implication, and logical operations can be represented as derived structural patterns. They are not introduced as independent cognitive faculties or algorithmic modules. Instead, their applicability follows from relations among contexts, representational objects, and admissible transformations.

The central contribution is a dependency-structured hierarchy of such schemas together with a formal task–schema interface. The hierarchy identifies the representational conditions required for progressively richer relational and process structures. A task is represented by a context, given elements, unknown placeholders, and constraints. These jointly determine the admissible instantiations of the unknowns and the schemas that can legitimately be applied. Problem solving can therefore be viewed as constrained instantiation under task-compatible representational structure.

This formulation also makes it possible to distinguish failures of representation from failures of a solving procedure. We identify three basic forms of ill-posedness: inconsistency, underdetermination, and contextual insuficiency. In each case, the task specification fails to provide the representational conditions required for a unique admissible solution.

The contribution lies in the organization of these notions within a common structural framework: contexts are task-relative representational structures, schemas specify conditions of structural applicability, laws capture relations stable across relevant contexts, and task formulation determines which schemas may participate in problem solving. The paper develops this representational layer through the schema hierarchy, the task–schema interface, and explicit conditions for task ill-posedness.

## 2 Related Work

The closest predecessors of the present framework do not form a single research tradition. Schema theory and cognitive architectures begin from organized knowledge and action; rough sets and Formal Concept Analysis from formal distinctions among objects; analogy and program induction from reusable relational structure; and recent work on language models and knowledge graphs from the interaction between linguistic competence and explicit knowledge. These lines address diferent parts of the representational problem considered here.

Classical schema theory treats schemas as organized knowledge structures that guide memory, interpretation, and action [3, 4]. Related developments in artificial intelligence introduced frames, scripts, plans, and goals as structured representations for understanding and problem solving [5, 6]. Cognitive architectures and production-system approaches, including ACT-R, Soar, and problem-space formulations, similarly organize cognition around structured states, operators, goals, productions, and memory [7, 8, 9]. The present work isolates one component of this broader architectural problem: the representational conditions under which particular schemas become applicable to a task.

Several formal representation frameworks are especially close to the present setting. Rough set theory uses indiscernibility relations and partitions to study approximation, reducts, and decision rules [10, 11, 12]. Formal Concept Analysis constructs extent–intent concepts and concept lattices from formal contexts [13, 14]; pattern structures extend this construction to richer descriptions [15], while granular computing studies granules as units of abstraction and problem solving [16]. Conceptual graphs and related knowledge-representation formalisms provide graph-based representations of concepts and relations [17, 18].

In the present framework, contexts are task-relative representational partitions, while the Concept Graph serves as a host for concepts, schemas, and stabilized laws. This gives these structures a diferent role from an FCA concept lattice, a conceptual graph in Sowa’s sense, or a domain ontology.

Relational structure is treated in several distinct ways in theories of analogy and abstraction. Structure-mapping theory locates analogical correspondence primarily in relations among objects rather than in matches between isolated attributes [19]. Constraint-based accounts add semantic and pragmatic constraints to the structural requirements of a mapping [20]. Conceptual spaces organize concepts through geometric quality dimensions and similarity [21], whereas compositional approaches study the preservation of structure under more abstract transformations [22]. These approaches therefore provide diferent formal views of comparison, similarity, and structural organization. A separate computational question is how reusable structures are learned and combined across tasks; this is addressed more directly in program induction and rule learning.

Program induction and rule learning make the dependence of rule discovery on an available representational language particularly clear. Inductive logic programming induces logical hypotheses from examples and background knowledge [23, 24], whereas DreamCoder and related approaches learn reusable abstractions that can be composed across tasks [25, 26, 27]. In each case, the space of candidate rules is delimited by the available primitives and admissible forms of composition. The corresponding representational issue is therefore concrete: a schema or relation can be applied only when the task context supplies the distinctions and relational structure required to express it.

Modern machine learning locates relational structure and invariance at several distinct levels. Relational inductive biases and graph-based models organize computation around relations among represented objects [28, 29, 30]. Group-equivariant architectures incorporate specified transformation symmetries into the model architecture [31], while invariant risk minimization treats stability across environments as a learning criterion [32]. These examples show that relational structure and invariance may enter through the representation, the architecture, or the learning objective. In the present framework, invariance recognition and representative selection operate on task-relative contexts: they determine which distinctions are preserved under admissible transformations and which may be suppressed. This places the issue at the level of task representation rather than the choice of a particular model class or loss function.

Task specification provides another point of contact. In formal methods and program synthesis, admissible solutions are defined relative to explicit specifications and constraints before search is performed [33, 34]. Causal models likewise make variables, structural relations, and admissible interventions explicit [35]. Work on benchmark design and generalization has also shown that apparent learning failures may reflect mismatches between data, representation, and evaluation criteria [36, 37]. The framework likewise makes task structure explicit, while also treating the representational context as part of the task specification. This makes it possible to distinguish inconsistency, underdetermination, and contextual insuficiency from failures of a particular search, learning, or inference procedure.

Recent work on integrating large language models with explicit knowledge structures is also relevant to the broader motivation of the framework. A recent TIST special issue identifies three complementary directions: examining the limits of structured knowledge implicitly available in LLMs, using LLMs to construct or enrich explicit knowledge structures, and using such structures to support more grounded retrieval and reasoning [38]. Experiments on public ontologies show that general-purpose pretrained LLMs reproduce exact identifier–label associations only incompletely, with successful recall strongly associated with concept frequency on the Web [39]. These results caution against identifying the implicit representation of a pretrained language model with an explicit and reliably accessible knowledge structure.

Conversely, LLMs can assist the construction and extension of explicit structures. GPTassisted entity typing can enrich entity, type-cluster, and relation representations in an existing knowledge graph [40], while FLAME uses an LLM to propose the placement of new concepts in an existing expert-curated taxonomy [41]. Language models can also provide an interface between external task descriptions and formal structured representations. Meloni et al. study the translation of natural-language scientific questions into executable SPARQL queries over knowledge graphs and show that explicit entity and relation information can substantially assist this translation [42]. These approaches illustrate diferent ways in which the broad linguistic competence of pretrained models can complement explicit structured knowledge.

Of particular relevance to future solver architectures, Zhang et al. combine an LLM with an explicit knowledge graph, meta-knowledge, short- and long-term memory, and a self-correcting mechanism [43]. Experience acquired during interaction is retained and reused across subsequent tasks, providing incremental adaptation outside the fixed pretrained model. This direction is close in motivation to the separation considered here between problem-solving mechanisms and an explicit persistent knowledge structure. The Knowledge Space proposed in our framework is nevertheless diferent from these knowledge graphs and meta-knowledge stores. Its Concept Graph is not primarily a repository of externally specified entities and factual triples, and its persistent structures are not limited to strategies for querying an existing graph. Instead, the Knowledge Space is intended to contain task-relative concepts, schemas, and stabilized laws that can be constructed and reused across contexts. The present paper provides the representational layer on which such interpretation, proposal, verification, and controlled knowledge-extension mechanisms can be organized.

Overall, existing work provides rich accounts of structured knowledge, relations, abstraction, invariance, and formal task specification. The distinctive feature of the present framework is their organization around an explicit task-relative context: contexts determine available distinc tions, schemas specify structural applicability, laws capture relational structure stable across admissible context transformations, and task formulation determines which of these structures can participate in problem solving.

## 3 Cognitive Schemas and Structural Derivation

## Schemas, Application Frames, and Applicability

We use cognitive schema as the general term for reusable structural mappings. A cognitive primitive is itself a schema. Primitive schemas belong to the initial cognitive basis, whereas derived schemas use structures made available by other schemas or by previously stabilized conceptual objects. The schemas developed below are therefore presented in a dependencycompatible order: the prerequisites of a derived schema are introduced before the schema that uses them. The dependency structure need not form a single linear chain.

Throughout the paper, $C \subseteq \Omega$ denotes the underlying class or base domain of a context, C denotes a finite context on $C ,$ and $c \in { \mathcal { C } }$ denotes a context cell. When a cell is represented by a distinguished object, we may use that representative in place of the cell when no ambiguity results.

A schema S is represented by

$$
S = ( I _ { S } , O _ { S } , P _ { S } , \Phi _ { S } ) ,
$$

where $I _ { S }$ specifies its typed input placeholders, $O _ { S }$ its typed output placeholders, $P _ { S }$ its structural prerequisites, and $\Phi _ { S }$ the structural mapping associated with the schema.

The input and output placeholders have the form

$$
I _ { S } = ( x _ { 1 } : \tau _ { 1 } , . . . , x _ { m } : \tau _ { m } ) ,
$$

and

$$
O _ { S } = ( y _ { 1 } : \sigma _ { 1 } , \dots , y _ { n } : \sigma _ { n } ) ,
$$

where $\tau _ { i }$ and $\sigma _ { j }$ are the required types. The types specify which kinds of objects may occupy the corresponding placeholders. They may refer, for example, to contexts, context cells, concepts, relations, laws, schemas, or other structured objects available in the Knowledge Space. No fixed exhaustive collection of types is required here.

The mapping $\Phi _ { S }$ is structural rather than algorithmic. It specifies which output structure becomes available when the input placeholders are instantiated by compatible objects and the prerequisites $P _ { S }$ are satisfied.

A schema is generally not applied to a context alone. Its application also depends on the objects currently available, the outputs sought, and the relevant local constraints. We represent this information by a schema application frame

$$
\mathcal { A } = ( \mathcal { C } , X , Y , \Gamma _ { \mathcal { A } } ) ,
$$

where C is the active context, X is the collection of available objects and structures, $Y$ is the collection of output placeholders available to the application, and $\Gamma _ { A }$ contains the constraints relevant to that application. Elements of $Y$ may correspond to unknowns of the task or to intermediate objects introduced during problem solving.

An application frame contains more information than the context but need not contain the complete task. For a task

$$
{ \mathcal { T } } = ( { \mathcal { C } } , G , U , \Gamma ) ,
$$

an application frame selects the part of the task and of the currently available derived structure that is relevant to one prospective schema application. Diferent schemas may therefore be applied to diferent application frames within the same task.

An instantiation of $S$ in A assigns available objects to the input placeholders of $S$ and associates its output placeholders with compatible elements of Y . We write

## App(S, A, ι)

when an instantiation ι satisfies the following conditions:

1. every input placeholder $x _ { i } : \tau _ { i }$ is assigned an object compatible with the required type $\tau _ { i } ;$

2. every output placeholder $y _ { j } : \sigma _ { j }$ is associated with an output of compatible type $\sigma _ { j } ;$

3. the structural prerequisites $P _ { S }$ are satisfied by the instantiated objects in the context $\mathcal { C } ;$

4. the instantiation is compatible with the local constraints $\Gamma _ { A }$

The schema S is applicable to $\mathcal { A }$ if at least one such instantiation exists:

$$
\operatorname { A p p } ( S , { \mathcal { A } } ) \Longleftrightarrow \exists \iota \operatorname { A p p } ( S , { \mathcal { A } } , \iota ) .
$$

For an admissible instantiation, $\Phi _ { S }$ determines the corresponding admissible output structure or structures. Applicability therefore depends jointly on type compatibility, the active context, the available objects, and the local constraints.

This formulation also gives a precise meaning to dependency between schemas. We write

$$
S _ { 1 } \triangleleft S _ { 2 }
$$

if a structure produced by an admissible application of $S _ { 1 }$ , or the stabilized schema concept corresponding to $S _ { 1 }$ , can satisfy an input placeholder or a structural prerequisite of $S _ { 2 }$ . The relation ◁ denotes direct structural dependency.

Dependency does not by itself imply applicability. If

$$
S _ { 1 } \triangleleft S _ { 2 } ,
$$

an application of $S _ { 1 }$ may provide one of the structures required by $S _ { 2 }$ , but $S _ { 2 }$ becomes applicable only when its remaining typed inputs, structural prerequisites, and local constraints are also satisfied in the corresponding application frame. In this sense, derivation denotes typed structural construction rather than logical implication between schema names.

For example,

$$
S _ { \mathrm { o r d e r } } \triangleleft S _ { \mathrm { s u c c } }
$$

expresses that an ordered structure can satisfy a prerequisite of the successor schema. Once the remaining applicability conditions are met, the successor schema can use this structure to construct the corresponding minimal-progression relation.

More generally, a derived schema is a schema whose admissible application uses structural objects made available by preceding schema applications or by their stabilized conceptual representations, rather than relying only on the initial cognitive basis.

Schemas may themselves become objects of representation. When a schema S is suficiently stable for reuse across a family of applications, it may be lifted into the Concept Graph as a schema concept. We denote this by

$$
\operatorname { L i f t } ( S ) = c _ { S } .
$$

The schema $S$ and the concept $c _ { S }$ have distinct functions. The schema is the structural mapping that can be instantiated and applied, whereas $c _ { S }$ represents that schema as a conceptual object. The lifted object may therefore be referred to, compared, generalized, specialized, or used as an input to another schema.

The same distinction applies when a schema refers to an ordinary concept. A concept node represents a stable conceptual object, whereas a particular schema application may require a context-relative realization of that concept. Let

$$
\mathrm { I n s t } _ { \mathcal { A } } ( c )
$$

denote the admissible realizations of a concept c in the application frame A. A concept may have one or more associated instantiation schemas. If $S _ { c } ^ { \mathrm { i n s t } }$ is such a schema, an admissible application has the form

$$
S _ { c } ^ { \mathrm { i n s t } } ( c , \mathcal { A } ) = x _ { \mathcal { A } } ,
$$

with

$$
x _ { \mathcal { A } } \in \operatorname { I n s t } _ { \mathcal { A } } ( c ) .
$$

Thus, when an input placeholder of a schema requires a context-relative object but is associated with a concept c, the corresponding instantiation schema supplies an admissible realization before the main schema is applied. Schematically,

$$
c \longrightarrow x _ { A } \longrightarrow \Phi _ { S } ( x _ { A } , . . . ) .
$$

This is diferent from a meta-level application. If an input placeholder explicitly requires a concept or a schema concept, the conceptual object itself may be used as the input. A higherlevel schema may therefore operate directly on previously stabilized schemas. For example, if $S _ { 1 }$ and $S _ { 2 }$ have been lifted,

$$
\mathrm { L i f t } ( S _ { 1 } ) = c _ { S _ { 1 } } , \qquad \mathrm { L i f t } ( S _ { 2 } ) = c _ { S _ { 2 } } ,
$$

a higher-level schema H may use these schema concepts as inputs and produce a new schema $S _ { 3 } { \mathrm { : } }$

$$
( c _ { S _ { 1 } } , c _ { S _ { 2 } } ) \stackrel { { \cal H } } { \longrightarrow } S _ { 3 } .
$$

If $S _ { 3 }$ subsequently becomes stable for reuse, it may itself be lifted:

$$
S _ { 3 } \xrightarrow { \mathrm { L i f t } } c _ { S _ { 3 } } .
$$

Consequently, schema construction is not restricted to a fixed hierarchy. Primitive schemas can support derived schemas; stabilized schemas can become conceptual objects; and these schema concepts can participate in the construction of further schemas. This provides a structural basis for an evolving Knowledge Space while preserving the distinction between a schema, its conceptual representation, its context-relative instantiations, and its executable realizations.

In the present paper, the role of the Knowledge Space is deliberately limited. Its graph structure provides a persistent organization in which stabilized schemas and laws can be represented as conceptual objects and made available for later schema applications. The constructions developed here use this organization through lifting, instantiation, and reuse of such objects; they do not depend on a particular graph-search, retrieval, or navigation mechanism. Those mechanisms belong to the solver level of the framework.

Thus the Knowledge Space is used here as the representational substrate that allows a derived structure to change status: a schema first appears as an applicable structural mapping, may become stabilized as a schema concept, and may then participate as an input to the construction of further schemas. In schematic form,

$$
S \longrightarrow c _ { S } \longrightarrow S ^ { \prime }
$$

denotes this transition from constructed structure to reusable conceptual object and subsequently to a component of further structural construction.

## Equivalence

Let C be a context on a base set $C \subseteq \Omega$ , i.e. a partition of C into disjoint cells. Once C is fixed, elements $\omega , \omega ^ { \prime } \in C$ are considered equivalent if they belong to the same context cell:

$$
\omega \sim _ { \mathcal { C } } \omega ^ { \prime } \quad \iff \quad \exists c \in { \mathcal { C } } \ { \mathrm { s u c h ~ t h a t ~ } } \omega , \omega ^ { \prime } \in c .
$$

Equivalence is therefore induced by the representational granularity encoded in the context. Identity corresponds to the degenerate case in which all cells are singletons. Refining or coarsening a context modifies the induced equivalence relation.

## Order and Comparison

An order schema introduces a strict asymmetric relation ≺ defined over context cells or their representatives. Formally,

$$
{ \prec } \subseteq { \mathcal { C } } \times { \mathcal { C } }
$$

is a strict partial or strict total order. Order relations operate on equivalence classes rather than on raw states and enable comparison, precedence, containment, or hierarchy without requiring numerical structure.

## Successor

Given an order $\preccurlyeq ,$ the successor schema refines it by identifying minimal progression. For $a , b \in { \mathcal { C } }$ , define

$$
\operatorname { s u c c } _ { \prec } ( a , b )
$$

if

$$
a \prec b
$$

and there is no $c \in { \mathcal { C } }$ such that

$$
a \prec c \prec b .
$$

The successor relation is defined abstractly and applies to any ordered structure. Temporal order provides a canonical instance, but no commitment to physical or metric time is assumed.

Time and successor structure. Successor structure need not be temporal. When a temporal coordinate is available, the successor relation may be interpreted as temporal precedence and thereby supports before–after reasoning. However, the framework does not assume metric, physical, or global time. The same structural role may be played by any ordered progression admitting a successor relation, such as process states, refinement chains, or stages of representation.

## Structural Dependence

Structural dependence describes how the admissible realization of one component of an application frame constrains the admissible realizations of another. It is a structural notion defined from the available objects and constraints. It is distinct from truth-functional implication and does not assume a probabilistic interpretation.

Let

$$
\mathcal { A } = ( \mathcal { C } , X , Y , \Gamma _ { \mathcal { A } } )
$$

be an application frame. Let $\Im _ { A }$ denote the set of admissible instantiations of this frame, that is, the instantiations that satisfy the relevant type requirements and the constraints $\Gamma _ { A }$

Let A and B be two typed components whose realizations are specified by an instantiation $\iota \in \Im _ { A }$ . For an admissible value a of A, define the admissible realizations of B conditional on $A = a$ by

$$
{ \mathrm { A d m } } _ { A } ( B \mid A = a ) = \{ b : \exists \boldsymbol { \varepsilon } \in \Im _ { A } { \mathrm { ~ s u c h ~ t h a t ~ } } \boldsymbol { \iota } ( A ) = a { \mathrm { ~ a n d ~ } } \iota ( B ) = b \} .
$$

We say that B is structurally dependent on A in the application frame A, and write

$$
A  _ { \mathcal { A } } B ,
$$

if there exist two admissible realizations $a _ { 1 }$ and $a _ { 2 }$ of A such that

$$
\mathrm { A d m } _ { \cal A } ( B \mid A = a _ { 1 } ) \neq \mathrm { A d m } _ { \cal A } ( B \mid A = a _ { 2 } ) ,
$$

with both sets nonempty.

Thus

$$
A  _ { \mathcal { A } } B
$$

means that changing the admissible realization of A changes the admissible possibilities for B under the constraints represented in A. The dependence is therefore relative to the application frame: changing the context, the available structures, or the relevant constraints may change whether the dependence holds.

Structural dependence is directional in the sense that the admissible possibilities for B are compared under diferent realizations of A. The relation need not, however, be asymmetric. It is possible that

$$
A  _ { \mathcal { A } } B
$$

and

$$
B  _ { A } A
$$

both hold when the constraints couple the two components in both directions. Structural dependence alone therefore does not specify causal direction.

At the schema level, the construction may be summarized as

$S _ { \mathrm { d e p } }$ : (application frame, typed components, constraints) −→ structural dependence.

The resulting dependence relation may subsequently serve as an input or structural prerequisite of higher-level schemas. In particular, when successor structure is also available, structural dependence can be refined by an explicit precedence requirement. This gives the restricted causal schema developed in the next subsection.

## General and Conditional Implication

Implication is defined separately from structural dependence. Let C be a context on $C \subseteq \Omega$ , and let A and B be predicates on C compatible with the distinctions represented by C.

A general implication holds over the complete domain of the context:

$$
A \Rightarrow c B
$$

if

$$
\forall \omega \in C , \qquad A ( \omega ) \Longrightarrow B ( \omega ) .
$$

A conditional implication holds only on a specified subset $D \subseteq C$ . We write

$$
A \Rightarrow _ { D } B
$$

if

$$
\forall \omega \in D , \qquad A ( \omega ) \Longrightarrow B ( \omega ) .
$$

Thus structural dependence,

$$
A  _ { \mathcal { A } } B ,
$$

and implication,

$$
A \Rightarrow _ { D } B ,
$$

are diferent schemas. Structural dependence concerns how admissible realizations vary under the constraints of an application frame, whereas implication concerns preservation of truth over a specified domain.

## Structural Causal Precedence

We introduce here one restricted causal schema constructed from two previously available structures: structural dependence and successor. It captures a class of causal relations in which a directed dependence is compatible with an explicit predecessor structure. The construction is not intended as an exhaustive definition of causality; rather, it provides one causal schema that can be derived from the structural elements available at this level of the framework.

Let

$$
\mathcal { A } = ( \mathcal { C } , X , Y , \Gamma _ { \mathcal { A } } )
$$

be an application frame. Let A and B be Boolean-valued typed components represented in A whose values are well defined on the cells of C. They therefore induce predicates

$$
A , B : { \mathcal { C } } \to \{ { \mathsf { T } } , \bot \} .
$$

Assume further that

$$
A  _ { \mathcal { A } } B
$$

holds in the sense of structural dependence defined above, and that C is equipped with a successor relation succ.

We write

$$
c \prec _ { \mathrm { s u c c } } ^ { + } d
$$

when d is reachable from c by one or more successor steps. Thus there exist context cells

$$
c = c _ { 0 } , c _ { 1 } , \ldots , c _ { k } = d , \qquad k \geq 1 ,
$$

such that

$$
\mathrm { s u c c } ( c _ { i } , c _ { i + 1 } )
$$

for every $i < k$

For a predicate or event-like pattern A, define its occurrence set by

$$
\operatorname { O c c } _ { \mathcal { C } } ( A ) = \{ c \in { \mathcal { C } } : A ( c ) = { \mathsf { T } } \} .
$$

We say that A has structural causal precedence over B in A, and write

$$
\mathrm { C P } _ { \mathcal { A } } ( A , B ) ,
$$

when:

1. B occurs in the represented configuration,

$$
\operatorname { O c c } _ { \boldsymbol { C } } ( B ) \neq \varnothing ;
$$

2. every occurrence of B has a preceding occurrence of A,

$$
\forall d \in \operatorname { O c c } _ { \mathcal { C } } ( B ) , \qquad \exists c \in \operatorname { O c c } _ { \mathcal { C } } ( A ) { \mathrm { ~ s u c h ~ t h a t ~ } } c \prec _ { \operatorname { s u c c } } ^ { + } d .
$$

The first condition supplies directed structural dependence, whereas the second constrains that dependence by successor structure. The resulting relation is directional because predecessor and successor positions are distinguished by the context.

At the schema level, the construction has the form

Structural Dependence + Successor −→ Structural Causal Precedence.

Equivalently, in terms of schema dependency,

$$
S _ { \mathrm { d e p } } \triangleleft S _ { \mathrm { c p } } , \qquad S _ { \mathrm { s u c c } } \triangleleft S _ { \mathrm { c p } } .
$$

The present construction therefore defines one restricted causal relation: a structurally dependent pattern must precede the pattern that depends on it along the available successor structure. It does not identify all causal relations with precedence. Rather, the framework permits diferent causal schemas to be constructed when additional representational structure is available. Later extensions may, for example, introduce interventional or counterfactual causal schemas [35], or causal relations defined for deterministic or stochastic processes.

## Discrete Succession

Discrete succession is obtained by repeated application of the successor relation. A finite succession is a sequence of context cells

$$
( c _ { 0 } , c _ { 1 } , \ldots , c _ { k } )
$$

such that

$$
\mathrm { s u c c } ( c _ { i } , c _ { i + 1 } )
$$

for every $i < k$ . Longer successions are obtained by repeated admissible successor extension. No numerical magnitude is assumed at this stage; only repeated successor structure is required.

## Number-Like Structure

The sequence from order to number-like structure provides a simple example of how derived schemas use the outputs of their predecessors.

Let C be a context in which an admissible order relation ≺ is available. The successor schema can then be applied to the ordered structure,

$$
( { \mathcal { C } } , \prec ) \longmapsto \operatorname { s u c c } _ { \prec } ,
$$

where $\mathrm { s u c c } _ { \prec }$ identifies minimal progression with respect to $\prec .$

The resulting successor relation satisfies an input requirement of the discrete-succession schema. Given an initial element $c _ { 0 }$ , this schema produces finite successor chains

$$
( c _ { 0 } , c _ { 1 } , \ldots , c _ { k } ) ,
$$

with

$$
\mathrm { s u c c } _ { \prec } ( c _ { i } , c _ { i + 1 } )
$$

for every $i < k$

At the next level, the particular elements occurring in such a chain can be suppressed while preserving its finite successor structure. Successions with the same successor structure are treated as equivalent, and the resulting equivalence classes provide number-like objects. Numerical sameness is thereby associated with sameness of finite succession structure rather than with the identity of the elements realizing the succession.

The construction can be summarized at the schema level as

$$
S _ { \mathrm { o r d e r } } \triangleleft S _ { \mathrm { s u c c } } \triangleleft S _ { \mathrm { d i s c } } \triangleleft S _ { \mathrm { n u m } } .
$$

Each occurrence of $\triangleleft$ represents a typed structural dependency: the structure produced at one stage supplies an input or prerequisite required by the next schema. The last step also illustrates abstraction, since the resulting object no longer depends on the particular elements used to realize the succession.

If such a number-like schema becomes stable and reusable, it may itself be lifted as a schema concept and become available as an input to further schemas. Thus the construction is not intended as an axiomatization of the natural numbers, but as an example of how progressively higher-level structures can be built and subsequently reused within the Knowledge Space.

## Higher-Order Structural Correspondence

The schemas considered so far construct relations and structures within a context. A further step becomes possible when previously available relations or schema concepts are themselves used to identify a common organization across diferent configurations. The resulting schema operates not only on individual objects but on relational structures formed from them.

Consider two finite collections of typed objects

$$
X = \{ a , b , c , d \} , \qquad X ^ { \prime } = \{ r , p , s , q \} .
$$

The order in which the elements of these sets are displayed has no structural meaning. Suppose that the first configuration contains a binary relation R with

$$
R ( a , b ) , \qquad R ( a , c ) , \qquad R ( b , d ) , \qquad R ( c , d ) ,
$$

while the second contains a possibly diferent binary relation $R ^ { \prime }$ with

$$
R ^ { \prime } ( p , q ) , \qquad R ^ { \prime } ( p , r ) , \qquad R ^ { \prime } ( q , s ) , \qquad R ^ { \prime } ( r , s ) .
$$

The two configurations have the same relational organization under the correspondence

$$
\varphi : X \to X ^ { \prime }
$$

defined by

$$
\varphi ( a ) = p , \qquad \varphi ( b ) = q , \qquad \varphi ( c ) = r , \qquad \varphi ( d ) = s .
$$

For the participating pairs,

$$
R ( x , y ) \Longleftrightarrow R ^ { \prime } ( \varphi ( x ) , \varphi ( y ) ) .
$$

The correspondence is therefore determined by preservation of relational structure rather than by the names or displayed order of the objects.

This construction can be stated more generally. Let X and $X ^ { \prime }$ be finite collections of typed objects, and let R and $\mathcal { R } ^ { \prime }$ be collections of relations represented in the corresponding application frames. A structural correspondence consists of an object mapping

$$
\varphi : X \to X ^ { \prime }
$$

together with, when the participating relations are not identical, a correspondence

$$
\psi : \mathcal { R } \to \mathcal { R } ^ { \prime }
$$

between the relevant relations. For each participating relation $R \in \mathcal { R }$ , structural preservation requires

$$
R ( x _ { 1 } , \ldots , x _ { k } ) \Longleftrightarrow \psi ( R ) { \big ( } \varphi ( x _ { 1 } ) , \ldots , \varphi ( x _ { k } ) { \big ) }
$$

for the tuples involved in the correspondence.

The mappings $\varphi$ and $\psi$ are subject to the type requirements and structural prerequisites of the application frame; they are not induced by an enumeration of the participating objects or relations. The structural correspondence schema may therefore be represented as

$$
S _ { \mathrm { c o r r } } : ( \mathrm { t y p e d ~ o b j e c t s , t y p e d ~ r e l a t i o n s } ) \longrightarrow \mathrm { s t r u c t u r a l ~ c o r r e s p o n d e n c e } .
$$

The construction becomes higher-order when the participating relations or structural patterns are outputs of previously available schemas, or when stabilized schemas have already been lifted as schema concepts. Several such concepts may jointly support the construction of a new higher-order schema,

$$
( c _ { S _ { 1 } } , c _ { S _ { 2 } } , \ldots , c _ { S _ { m } } ) \longrightarrow S _ { H } .
$$

If the resulting schema becomes stable for reuse, it may itself be lifted,

$$
S _ { H } \xrightarrow { \mathrm { L i f t } } c _ { S _ { H } } .
$$

The correspondence therefore preserves an organization among typed objects and relations without requiring the participating objects or relations to be identical across the two configurations.

The constructions above illustrate two forms of schema derivation. Compositional derivation occurs when the output of one schema supplies an input or prerequisite of another, whereas higher-order construction may combine several previously available structures. Table 1 summarizes the principal dependencies developed in this section.

One compositional dependency chain is

Table 1: Structural prerequisites and resulting structures for the principal schemas developed in this section. The rows summarize dependencies and do not represent a single linear sequence of schema construction.
<table><tr><td>Schema</td><td>Required structure</td><td>Resulting structure</td></tr><tr><td>Equivalence</td><td>Context partition</td><td>Equivalence relation induced by common cell membership</td></tr><tr><td>Order</td><td>Typed objects or context cells to- gether with an admissible compar- ison relation</td><td>Ordered relational structure</td></tr><tr><td>Successor Structural depen- dence</td><td>Ordered structure Application frame, typed compo- nents, admissible instantiations,</td><td>Minimal progression relation Directed dependence of admissible realizations within the application</td></tr><tr><td>General and condi-</td><td>and local constraints Predicates together with a speci-</td><td>frame Truth-functional implication over</td></tr><tr><td>tional implication Structural causal precedence</td><td>fied domain of validity Structural dependence together with successor structure and cell-</td><td>the corresponding domain Restricted causal relation combin- ing structural dependence with</td></tr><tr><td>Discrete succession</td><td>wise occurrence predicates Successor relation and an initial</td><td>predecessor structure Finite or iterated successor chain</td></tr><tr><td>Number-like struc- ture</td><td>object Finite successor chains and equiva- lence under preservation of succes-</td><td>Equivalence classes of finite suc- cessions and their induced struc-</td></tr><tr><td>Higher-order struc- tural correspondence</td><td>sor structure Two typed relational configura- tions together with compatible ob- jects, relations, or previously de-</td><td>tural organization Structure-preserving correspon- dence and a reusable higher-order structural pattern</td></tr></table>

$$
S _ { \mathrm { o r d e r } } \triangleleft S _ { \mathrm { s u c c } } \triangleleft S _ { \mathrm { d i s c } } \triangleleft S _ { \mathrm { n u m } } .
$$

Higher-order structural correspondence instead combines several available structures and therefore need not have a single predecessor.

The constructions considered so far concern the derivation and application of schemas within contexts or application frames. We next consider when a relation can be reused across diferent contexts and thereby represented as a stable law.

## 4 Relations and Laws

Relations are structured links defined relative to a context. Let C be a context on a base domain $C \subseteq \Omega$ . A context-relative binary relation has the form

$$
R c \subseteq { \mathcal { C } } \times { \mathcal { C } } ,
$$

or equivalently

$$
R c : { \mathcal { C } } \times { \mathcal { C } } \to \{ { \mathsf { T } } , \bot \} .
$$

Such a relation is evaluated with respect to the distinctions represented by C. It may therefore depend on the granularity and organization of that context and, by itself, is not yet a law.

A relation defined on the underlying domain may induce a context-relative relation when it is invariant with respect to the context cells. Let

$$
q c : C \to { \mathcal { C } }
$$

be the canonical quotient map. Let

$$
R : C \times C \to \{ { \top } , \bot \} .
$$

The relation R descends to C if it is constant on the fibers of

$$
q c \times q c .
$$

Explicitly, whenever

$$
q c ( \omega _ { 1 } ) = q c ( \omega _ { 1 } ^ { \prime } )
$$

and

$$
q c ( \omega _ { 2 } ) = q c ( \omega _ { 2 } ^ { \prime } ) ,
$$

we require

$$
R ( \omega _ { 1 } , \omega _ { 2 } ) = R ( \omega _ { 1 } ^ { \prime } , \omega _ { 2 } ^ { \prime } ) .
$$

Under this condition there exists a unique relation $R _ { \cal { C } }$ such that

$$
R = R c \circ ( q c \times q c ) .
$$

Representative selection provides an equivalent realization of this context-relative relation. Let

$$
\rho : { \mathcal { C } }  C
$$

satisfy

$$
q c \circ \rho = \operatorname { i d } _ { \mathcal { C } } .
$$

Then, for every $c _ { 1 } , c _ { 2 } \in { \mathcal { C } } .$

$$
R c ( c _ { 1 } , c _ { 2 } ) = R { \big ( } \rho ( c _ { 1 } ) , \rho ( c _ { 2 } ) { \big ) } .
$$

Because R is constant on context cells, this value is independent of the particular representative selection $\rho .$ Representative independence is therefore a well-definedness condition for a context-relative realization, rather than a transformation between contexts.

We next consider relations across diferent contexts. Let $\mathcal { C } ^ { \prime }$ be a context on $C ^ { \prime }$ and C a context on C. A transformation

$$
p : C ^ { \prime } \to C
$$

is compatible with the contexts if it induces a map

$$
\pi : { \mathcal { C } } ^ { \prime } \to { \mathcal { C } }
$$

such that

$$
q c \circ p = \pi \circ q c ^ { \prime } .
$$

Thus objects that are indistinguishable in $\mathcal { C } ^ { \prime }$ are mapped to objects that remain indistinguishable in C. Only transformations satisfying this compatibility condition are treated here as admissible context transformations.

Given a relation $R _ { \cal { C } }$ on ${ \mathcal { C } } _ { : }$ , its pullback along $\pi$ is the relation on $\mathcal { C } ^ { \prime }$ defined by

$$
( \pi ^ { * } R c ) ( c _ { 1 } ^ { \prime } , c _ { 2 } ^ { \prime } ) = R c \big ( \pi ( c _ { 1 } ^ { \prime } ) , \pi ( c _ { 2 } ^ { \prime } ) \big ) .
$$

For a refinement, $\mathcal { C } ^ { \prime }$ is finer than $\mathcal { C }$ and π is the canonical map sending each finer cell to the coarser cell containing it. The pullback therefore gives a canonical realization of a coarser relation on the finer context.

The converse direction requires an additional condition. A relation $R _ { C ^ { \prime } }$ on the finer context descends to $\mathcal { C }$ only if it is constant on the fibers of

$$
\pi \times \pi .
$$

Thus, whenever

$$
\pi ( c _ { 1 } ^ { \prime } ) = \pi ( d _ { 1 } ^ { \prime } )
$$

and

$$
\pi ( c _ { 2 } ^ { \prime } ) = \pi ( d _ { 2 } ^ { \prime } ) ,
$$

we require

$$
\begin{array} { r } { R c \prime ( c _ { 1 } ^ { \prime } , c _ { 2 } ^ { \prime } ) = R c \prime ( d _ { 1 } ^ { \prime } , d _ { 2 } ^ { \prime } ) . } \end{array}
$$

When this condition holds, there exists a unique relation $R _ { \cal { C } }$ satisfying

$$
R _ { \mathit { C ^ { \prime } } } = \pi ^ { * } R _ { \mathit { C } } .
$$

Hence refinement naturally supports pullback, whereas coarsening requires descent. A projection between represented domains is admissible only when it induces a well-defined map between the corresponding context cells as specified above.

Let C be a family of compatible contexts together with their admissible context transformations. A law $L$ over C is a family of context-relative realizations

$$
L = \left( R _ { \mathcal { C } } ^ { L } \right) _ { \mathcal { C } \in \mathfrak { C } }
$$

such that, for every admissible map

$$
\pi : { \mathcal { C } } ^ { \prime } \to { \mathcal { C } } ,
$$

the realizations satisfy

$$
R _ { C ^ { \prime } } ^ { L } = \pi ^ { * } R _ { C } ^ { L } .
$$

The same compatibility equation has two complementary interpretations. From a coarser context to a finer one, it specifies pullback. From a finer context to a coarser one, it can hold only when the finer realization satisfies the required descent condition.

The compatibility condition is itself stable under composition. Suppose

$$
\begin{array} { r } { \mathcal { C } ^ { \prime \prime } \overset { \sigma } { \to } \mathcal { C } ^ { \prime } \overset { \pi } { \to } \mathcal { C } } \end{array}
$$

are admissible context transformations and

$$
R _ { { \mathcal C } ^ { \prime } } ^ { L } = \pi ^ { * } R _ { { \mathcal C } } ^ { L } , \qquad R _ { { \mathcal C } ^ { \prime \prime } } ^ { L } = \sigma ^ { * } R _ { { \mathcal C } ^ { \prime } } ^ { L } .
$$

Then

$$
{ \cal R } _ { { \mathcal { C } } ^ { \prime \prime } } ^ { L } = \sigma ^ { * } ( \pi ^ { * } R _ { { \mathcal { C } } } ^ { L } ) = ( \pi \circ \sigma ) ^ { * } R _ { { \mathcal { C } } } ^ { L } .
$$

Thus the realization of a law does not depend on whether a compatible change of context is performed directly or through intermediate admissible contexts.

A law is therefore not identified with any single relation $R _ { \mathcal { C } } ^ { L }$ . It is the stable relational structure represented by the compatible family of its context-relative realizations. The family C specifies its domain of applicability.

When such a law is suficiently stable for reuse, it may be lifted into the Concept Graph as a conceptual object. Its context-relative relations remain realizations of that law rather than independent lifted relations. A lifted law with explicitly relational content may be represented as a relation concept.

## 5 Tasks, Contexts, and Schema Applicability

Before problem solving can take place, a task must be represented in a context in which its unknowns, admissible distinctions, and constraints can be stated. A task is therefore specified relative to a context rather than identified with an algorithmic objective, loss function, or input sentence.

Let C be a finite context. A task is represented by

$$
T = ( C , G , U , \Gamma ) ,
$$

where $G$ contains the given elements or facts, U the unknowns to be instantiated, and Γ the constraints on admissible instantiations

The external presentation of a task may be verbal, symbolic, diagrammatic, data-based, or already formal. Here we assume that this presentation has been converted into T. The questions considered below are then which distinctions and schemas are available in $C ,$ , and whether the unknowns can be instantiated consistently with Γ.

## 5.1 Illustrative Examples

The following examples illustrate the consequences of applying an appropriate representational transformation once the relevant invariance, dependency, or structural requirement is available. They do not address how such a transformation is discovered. In some cases the external task semantics indicates the relevant invariance without prescribing its representational realization; in others, it specifies that a relevant structure exists without identifying it completely. The general claims concerning schema applicability, structural dependency, and cross-context stability follow from the formal constructions developed above rather than from these examples.

## Example 1: Permutation-Invariant Structure

External task statement. Given a finite sequence of symbols, determine whether the sequence contains at least two identical elements. The order of symbols is irrelevant to the answer.

Task specification. Let $C _ { 1 }$ be the domain of finite sequences

$$
x = ( x _ { 1 } , \ldots , x _ { n } )
$$

over a finite alphabet. Let $\mathcal { C } _ { 1 }$ be the discrete context on $C _ { 1 }$ , so that each sequence is initially represented by a distinct context cell. The task

$$
\mathcal { T } _ { 1 } = ( \mathcal { C } _ { 1 } , G _ { 1 } , U _ { 1 } , \Gamma _ { 1 } )
$$

is defined by:

• Context $\mathcal { C } _ { 1 } \colon$ the discrete context on $C _ { 1 }$ , in which every sequence is represented by a distinct context cell and the sequence representation retains positional order and symbol identity.

• Given elements $G _ { 1 } { : }$ : a finite set of labeled examples

$$
\{ ( \boldsymbol x ^ { ( i ) } , \boldsymbol y ^ { ( i ) } ) \} ,
$$

where $\boldsymbol { x } ^ { ( i ) } \in C _ { 1 }$ and $y ^ { ( i ) } \in \{ 0 , 1 \}$

• Unknown $U _ { 1 } { : }$ a function

$$
f : C _ { 1 } \to \{ 0 , 1 \} .
$$

• Constraints $\Gamma _ { 1 } \mathbf { : }$ : consistency with the given examples and invariance under permutations of sequence positions.

In the raw context $\mathcal { C } _ { 1 }$ , two distinct sequences are represented by distinct singleton cells, even when they difer only by a permutation of positions. Define an equivalence relation on $C _ { 1 }$ by

$$
x \sim _ { \mathrm { p e r m } } x ^ { \prime }
$$

if $x ^ { \prime }$ can be obtained from x by a permutation of positions. The corresponding quotient context

$$
\mathcal { C } _ { 1 } ^ { \mathrm { p e r m } } = C _ { 1 } / { \sim } _ { \mathrm { p e r m } }
$$

groups permutation-equivalent sequences into the same context cell and therefore suppresses order distinctions that are irrelevant to the task.

Let

$$
q _ { \mathrm { p e r m } } : C _ { 1 } \to \mathcal { C } _ { 1 } ^ { \mathrm { p e r m } }
$$

denote the canonical quotient map. Permutation invariance means that f is constant on the cells of ${ \mathcal { C } } _ { 1 } ^ { \mathrm { p e r m } }$ . Hence there exists a unique function

$$
\bar { f } : \mathcal { C } _ { 1 } ^ { \mathrm { p e r m } } \to \{ 0 , 1 \}
$$

such that

$$
f = \bar { f } \circ q _ { \mathrm { p e r m } } .
$$

Data-based variant. The same task may be $\mathrm { g i }$ ven through labeled observations such as

$$
G _ { 1 } = \{ ( ( a , b , c ) , 0 ) , ( ( a , b , a ) , 1 ) , ( ( d , e , f ) , 0 ) \} .
$$

These observations constrain f only on the displayed sequences. The additional condition in $\Gamma _ { 1 }$ specifies permutation invariance, so after passing to ${ \mathcal { C } } _ { 1 } ^ { \mathrm { p e r m } }$ the task is represented by

$$
{ \bar { f } } : { \mathcal { C } } _ { 1 } ^ { \mathrm { p e r m } } \to \{ 0 , 1 \} ,
$$

which depends only on the permutation class. In this example, the relevant class property is whether some symbol occurs at least twice.

## Example 2: Abstraction and Generalization by Removing Irrelevant Coordinates

External task statement. Given a binary vector of fixed length, determine an output bit that depends only on a small, unknown subset of input coordinates. All remaining coordinates are irrelevant to the correct answer.

Task specification. The task $\mathcal { T } _ { 2 } = ( \mathcal { C } _ { 2 } , G _ { 2 } , U _ { 2 } , \Gamma _ { 2 } )$ is defined by:

• Context $\mathcal { C } _ { \mathrm { 2 } } \colon$ binary vectors

$$
x = ( x _ { 1 } , \ldots , x _ { d } ) \in \{ 0 , 1 \} ^ { d } ,
$$

with all coordinates treated symmetrically.

• Given elements $G _ { 2 } \colon$ labeled examples $\{ ( \boldsymbol x ^ { ( i ) } , \boldsymbol y ^ { ( i ) } ) \}$

• Unknown $U _ { \mathrm { 2 } } { \mathrm { : } }$ a function

$$
f : { \mathcal { C } } _ { 2 } \to \{ 0 , 1 \} .
$$

• Constraints $\Gamma _ { 2 } \mathbf { : }$ consistency with the given labels.

Within the raw context $\mathcal { C } _ { 2 }$ , all coordinates are equally represented, so the representation does not identify which coordinates may be ignored without afecting the output. The task states that a relevant subset exists but does not specify its members; discovering that subset is outside the present example.

Once the relevant coordinates have been identified, projection suppresses distinctions associated with the remaining coordinates. In the reduced context, admissible laws can be expressed using only the relevant coordinates and may therefore have simpler representations. This realizes abstraction by removing task-irrelevant distinctions and supports generalization when the same reduced description applies across related task instances.

## Example 3: Relational Composition under Contextual Constraints

External task statement. Given three objects $A , B , C$ and the statements $^ { 6 6 }$ is next to $A ^ { \prime \prime }$ and $^ { 6 6 } C$ is next to $B ^ { \prime \prime }$ , determine whether $^ { 6 6 } A$ is next to $C ^ { \dprime \ d }$ holds.

The task refers to objects and a binary relation next to. No metric distance, global geometry, or transitivity assumptions are specified. The meaning of next to is restricted to the task context.

Task specification. The task $\mathcal { T } _ { 3 } = ( \mathcal { C } _ { 3 } , G _ { 3 } , U _ { 3 } , \Gamma _ { 3 } )$ is specified as follows.

The context consists of finite configurations of objects equipped with a binary relation $\mathrm { N e x t } ( \cdot , \cdot )$ . The context distinguishes adjacency relations but does not encode distance, order, or spatial embedding.

The given elements are the relational facts

$$
\mathrm { N e x t } ( B , A ) , \qquad \mathrm { N e x t } ( C , B ) .
$$

The unknown is the truth value of the relation

$$
\mathrm { N e x t } ( A , C ) .
$$

Constraints $\Gamma _ { 3 }$ . No constraints beyond consistency with the given relations are imposed. In particular, no assumptions about transitivity or symmetry of Next are included in the task specification.

Within the given context, adjacency is treated as a local relation. The task context does not determine whether adjacency is transitive or symmetric, and therefore does not uniquely determine the value of

$$
\mathrm { N e x t } ( A , C ) .
$$

The original task is consequently underdetermined. Introducing relation composition or path structure does not by itself resolve this adjacency question. Such an extension may instead define a diferent relation, for example indirect connectivity or reachability, whose truth conditions are not the same as those of Next.

A reformulated task could instead ask whether A and $C$ are connected through intermediate objects. Such a question may become decidable once an appropriate path or composition schema is available, but the conclusion then concerns a new relation rather than the original adjacency predicate. Consequently, adding constraints may resolve underdetermination of Next(A, C), whereas introducing a new relational schema constitutes a reformulation of the task.

## 5.2 Solution preservation under representational reduction.

$$
\mathcal { T } ^ { \prime } = ( \mathcal { C } ^ { \prime } , G ^ { \prime } , U ^ { \prime } , \Gamma ^ { \prime } )
$$

be a task represented in a context $\mathcal { C } ^ { \prime } .$ , and let

$$
\pi : { \mathcal { C } } ^ { \prime } \to { \mathcal { C } }
$$

be a surjective admissible context transformation that suppresses distinctions not required by the task. Let $\sim _ { \pi }$ denote the corresponding equivalence on admissible assignments: two assignments are equivalent when they difer only in distinctions identified by π.

Assume that satisfaction of the task constraints is invariant under this equivalence, so that

$$
u _ { 1 } ^ { \prime } \sim _ { \pi } u _ { 2 } ^ { \prime } \quad \Longrightarrow \quad \left( u _ { 1 } ^ { \prime } \left| = c \cdot \Gamma ^ { \prime } \Longleftrightarrow u _ { 2 } ^ { \prime } \right| = c \prime \Gamma ^ { \prime } \right) .
$$

The constraints therefore induce a well-defined task

$$
\boldsymbol { \mathcal { T } } = ( \mathcal { C } , G , U , \Gamma )
$$

on the quotient representation.

Proposition 1. Under the conditions above, the solutions of the reduced task are in one-to-one correspondence with the equivalence classes of solutions of the original task:

$$
\operatorname { S o l } ( { \mathcal { T } } ) \cong \operatorname { S o l } ( { \mathcal { T } } ^ { \prime } ) / { \sim } _ { \pi } .
$$

Proof. Constraint invariance implies that either every assignment in a $\sim _ { \pi ^ { - } } \mathrm { c l a s s }$ satisfies $\Gamma ^ { \prime }$ or none does. Satisfaction therefore depends only on the equivalence class. Each solution class of $\tau ^ { \prime }$ consequently determines exactly one solution of the quotient task $\tau ,$ and every solution of $\tau$ is obtained from such a class. □

Two consequences are immediate. First, a task with no admissible solution cannot acquire one merely by suppressing distinctions in a solution-preserving representation. Second, several solutions that difer only in distinctions suppressed by π become a single solution in the reduced context, whereas solutions belonging to diferent $\sim _ { \pi }$ -classes remain distinct. Representational reduction can therefore remove apparent underdetermination caused solely by irrelevant distinctions without changing the task-relevant solution structure.

## 6 Ill-Posedness of Tasks

The task representation makes it possible to distinguish diferent forms of ill-posedness at the level of task formulation. Internal inconsistency and underdetermination are properties of the formalized task, whereas contextual insuficiency additionally concerns the adequacy of that representation relative to the external task meaning.

Type I: Internal inconsistency. A task is internally inconsistent if no admissible instantiation of the unknowns satisfies the constraints in the given context. Formally, there is no assignment $u \in U$ such that

$$
u \Vdash _ { C } \Gamma .
$$

Inconsistency may arise from contradictory constraints, incompatible givens, or a mismatch between the assumed schema and the available context.

Type II: Underdetermination. A task is underdetermined if several mutually incompatible instantiations of the unknowns satisfy the constraints in the same context. Thus there exist distinct admissible assignments

$$
u _ { 1 } , u _ { 2 } \in U
$$

such that

$$
u _ { 1 } \left| = _ { C } \Gamma , \qquad u _ { 2 } \right| = _ { C } \Gamma ,
$$

while $u _ { 1 }$ and $u _ { 2 }$ are not equivalent with respect to the distinctions relevant to the task. Resolving underdetermination requires additional constraints, additional data, a more specific context, or an evaluation criterion.

Type III: Contextual insuficiency. A task representation is contextually insuficient when the external task semantics requires distinctions, relations, schemas, or laws that cannot be expressed in the current context C. In this case, the external task may be meaningful, but its current formal representation

$$
\boldsymbol { \mathcal { T } } = ( \mathcal { C } , G , U , \Gamma )
$$

does not contain the representational structure required to formulate the intended question or its admissible solution adequately.

Examples include tasks that require metric distance when only adjacency is represented, transitivity when only a local relation is available, or temporal order when no successor structure is represented.

Unlike inconsistency and underdetermination, contextual insuficiency is not diagnosed from the formal task tuple alone. It is identified relative to the external task meaning from which the formal representation was constructed. A more explicit treatment in terms of richer reference contexts and representational projections can be introduced in later work.

Resolving contextual insuficiency requires a change of representation, for example by refinement, coarsening, projection, extension, or reformulation, depending on the missing structure.

Granularity mismatch, scope ambiguity, and schema mismatch can be treated as refinements of the three cases above. Internal inconsistency and underdetermination can be diagnosed from the formal task representation, whereas contextual insuficiency additionally requires comparison with the external task semantics. Once both the task representation and its relation to the external task meaning are explicit, these mismatches can be classified as inconsistency, underdetermination, or contextual insuficiency.

The classification identifies whether failure lies in the constraints Γ, in the available unknowns $U ,$ in schema applicability, or in the expressive capacity of the context $C .$

## Objects Lifted into the Concept Graph

The lifting criterion is stability across contexts. Concepts, schema concepts, laws, and recurring diagnostic structures may be represented as nodes in the Concept Graph when they remain stable across an appropriate family of contexts. Context-specific realizations, including particular relations, equivalences, and concept instantiations, remain local and are linked to the corresponding stable objects rather than lifted independently. Families of compatible contexts may themselves provide reusable domains of applicability for laws and schemas. Procedures and algorithmic primitives are represented separately in the Procedure Graph and linked to Concept Graph objects through typing and applicability relations.

## 7 Discussion

The results above place task formulation and schema applicability at the same representational level. A context fixes which distinctions are available, while the task specification determines which of these distinctions and relations are required by the givens, unknowns, and constraints. Problem solving therefore depends not only on the available procedures, but also on what can be expressed in the current context.

This distinction clarifies several diferent sources of dificulty. Inconsistency arises when no admissible instantiation satisfies the constraints, underdetermination when several nonequivalent instantiations remain admissible, and contextual insuficiency when the distinctions or relations required by the intended solution are absent from the context. The examples exhibit these cases in elementary form: quotienting removes irrelevant order information, projection suppresses irrelevant coordinates, and insuficient relational structure leaves a query unresolved. A change of representation can therefore alter the admissible solution space without changing the task data themselves.

The distinction between relations and laws extends the same idea across contexts. A relation may depend on the granularity and organization of a particular context. A law has compatible realizations across a family of contexts under admissible transformations such as refinement, coarsening, projection, or representative selection. This stability allows a law to be represented as a reusable object in the Concept Graph, while its context-relative realizations remain tied to particular tasks or contexts.

The schema hierarchy, task–schema interface, and distinction between relations and laws together define a structural layer for problem solving. Within this layer, quotienting, projection, context transformation, and schema applicability can be analyzed independently of the particular mechanism used to discover or select them. This separation also provides a basis for later solver architectures in which proposal generation, verification, and persistent knowledge can be implemented by diferent components.

## 8 Conclusion

This paper developed a hierarchy of cognitive schemas for knowledge-based problem solving. Equivalence, order, successor, dependence, implication, discrete succession, and number-like structure are derived from contexts, relations, representative objects, and admissible transfor mations rather than introduced as independent primitives.

Task specification connects these schemas to problem formulation. A task context determines which distinctions and relations are available for instantiating the unknowns under the stated constraints. This also separates three forms of ill-posedness: inconsistency, underdetermination, and contextual insuficiency. In the last case, the required distinctions or relations are absent from the current context, so the representation must be changed before a solution can be expressed.

Relations and laws are separated in the same way across contexts. Relations may be specific to a particular context, whereas laws have compatible realizations across a family of contexts under admissible transformations. Such stable structures can be reused and represented in the Concept Graph.

A broader research direction is to investigate how this explicit structural layer can interact with language-model architectures. Two complementary possibilities are of particular interest:

building problem-solving capabilities from an initially untrained or minimally pretrained model whose persistent knowledge is organized through the explicit Knowledge Space, and coupling the framework to a pretrained LLM that provides rich natural-language interpretation and candidate proposals while explicit representation, verification, and knowledge stabilization remain outside the model. In both cases, the objective is to separate proposal generation from verified persistent knowledge and controlled structural learning. These directions provide a route for extending the present representational framework toward solver architectures that combine flexible proposal generation with explicit verification and persistent structural learning.

## 9 Declarations

## 9.1 Availability of Data and Material

Data availability statement: not applicable.

## 9.2 Competing Interests

The authors declare that they have no competing interests.

## 9.3 Funding

No funding was received for this work.

## 9.4 Authors’ Contributions

The authors contributed equally to all aspects of the work.

## 9.5 Acknowledgements

The authors have no acknowledgements to declare.

## References

[1] Antal Jakovác and András Telcs. Representation and abstraction. Mathematics, 13(10):1666, 2025.

[2] Antal Jakovác and András Telcs. A structural theory of cognitive representation and problem solving: Contexts, invariance, and the knowledge space, 2026. Manuscript.

[3] Frederic C. Bartlett. Remembering: A Study in Experimental and Social Psychology. Cambridge University Press, Cambridge, 1932.

[4] David E. Rumelhart. Schemata: The building blocks of cognition. In Rand J. Spiro, Bertram C. Bruce, and William F. Brewer, editors, Theoretical Issues in Reading Comprehension: Perspectives from Cognitive Psychology, Linguistics, Artificial Intelligence, and Education, pages 33–58. Lawrence Erlbaum Associates, Hillsdale, NJ, 1980.

[5] Marvin Minsky. A framework for representing knowledge. In Patrick H. Winston, editor, The Psychology of Computer Vision, pages 211–277. McGraw-Hill, New York, 1975.

[6] Roger C. Schank and Robert P. Abelson. Scripts, Plans, Goals, and Understanding: An Inquiry into Human Knowledge Structures. Lawrence Erlbaum Associates, Hillsdale, NJ, 1977.

[7] Allen Newell and Herbert A. Simon. Human Problem Solving. Prentice-Hall, Englewood Clifs, NJ, 1972.

[8] John R. Anderson, Daniel Bothell, Michael D. Byrne, Scott Douglass, Christian Lebiere, and Yulin Qin. An integrated theory of the mind. Psychological Review, 111(4):1036–1060, 2004.

[9] John E. Laird, Allen Newell, and Paul S. Rosenbloom. SOAR: An architecture for general intelligence. Artificial Intelligence, 33(1):1–64, 1987.

[10] Zdzisław Pawlak. Rough sets. International Journal of Computer and Information Sciences, 11(5):341–356, 1982.

[11] Zdzisław Pawlak. Rough Sets: Theoretical Aspects of Reasoning about Data, volume 9 of Theory and Decision Library D. Kluwer Academic Publishers, Dordrecht, 1991.

[12] Andrzej Skowron and Cecylia Rauszer. The discernibility matrices and functions in information systems. In Roman Słowiński, editor, Intelligent Decision Support: Handbook of Applications and Advances of the Rough Sets Theory, pages 331–362. Kluwer Academic Publishers, Dordrecht, 1992.

[13] Rudolf Wille. Restructuring lattice theory: An approach based on hierarchies of concepts. In Ivan Rival, editor, Ordered Sets, pages 445–470. D. Reidel Publishing Company, Dordrecht, 1982.

[14] Bernhard Ganter and Rudolf Wille. Formal Concept Analysis: Mathematical Foundations. Springer, Berlin, 1999.

[15] Bernhard Ganter and Sergei O. Kuznetsov. Pattern structures and their projections. In Conceptual Structures: Broadening the Base, volume 2120 of Lecture Notes in Computer Science, pages 129–142, Berlin, 2001. Springer.

[16] Yiyu Yao. A unified framework of granular computing. In Witold Pedrycz, Andrzej Skowron, and Vladik Kreinovich, editors, Handbook of Granular Computing. John Wiley & Sons, Chichester, 2008.

[17] John F. Sowa. Conceptual Structures: Information Processing in Mind and Machine. Addison-Wesley, Reading, MA, 1984.

[18] Franz Baader, Diego Calvanese, Deborah L. McGuinness, Daniele Nardi, and Peter F. Patel-Schneider, editors. The Description Logic Handbook: Theory, Implementation, and Applications. Cambridge University Press, Cambridge, 2003.

[19] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2):155–170, 1983.

[20] Keith J. Holyoak and Paul Thagard. Analogical mapping by constraint satisfaction. Cognitive Science, 13(3):295–355, 1989.

[21] Peter Gärdenfors. Conceptual Spaces: The Geometry of Thought. MIT Press, Cambridge, MA, 2000.

[22] Steven Phillips and William H. Wilson. Categorial compositionality: A category theory explanation for the systematicity of human cognition. PLOS Computational Biology, 6(7):e1000858, 2010.

[23] Stephen Muggleton. Inductive logic programming. New Generation Computing, 8(4):295– 318, 1991.

[24] Andrew Cropper, Sebastijan Dumančić, Richard Evans, and Stephen H. Muggleton. Inductive logic programming at 30. Machine Learning, 111(1):147–172, 2022.

[25] Kevin Ellis, Lionel Wong, Maxwell Nye, Mathias Sable-Meyer, Luc Cary, Lore Anaya Pozo, Luke Hewitt, Armando Solar-Lezama, and Joshua B Tenenbaum. Dreamcoder: growing generalizable, interpretable knowledge with wake–sleep bayesian program learning. Philosophical Transactions of the Royal Society A, 381(2251):20220050, 2023.

[26] Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building machines that learn and think like people. MIT Press, 2017.

[27] Jacob Andreas, Dan Klein, and Sergey Levine. Modular multitask learning with policy sketches. In International Conference on Machine Learning, 2017.

[28] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.

[29] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? International Conference on Learning Representations, 2019.

[30] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. International Conference on Learning Representations, 2019.

[31] Taco Cohen and Max Welling. Group equivariant convolutional networks. In Proceedings of the International Conference on Machine Learning, 2016.

[32] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

[33] Sumit Gulwani, Oleksandr Polozov, and Rishabh Singh. Program synthesis. Foundations and Trends in Programming Languages, 4(1–2):1–119, 2017.

[34] Ib Holm Sørensen et al. Specification of Software Systems. Springer, 2008.

[35] Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009.

[36] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020.

[37] Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, and Aleksander Madry. Characterizing out-of-distribution generalization with synthetic tasks. NeurIPS, 2021.

[38] Qi He, Wei Wang, and Hao Wang. Introduction to the special issue on integrating large language models and knowledge graphs for generative ai. ACM Transactions on Intelligent Systems and Technology, 17(6), 2026.

[39] Marco Bombieri, Paolo Fiorini, Simone Paolo Ponzetto, and Marco Rospocher. Do LLMs dream of ontologies? ACM Transactions on Intelligent Systems and Technology, 17(6), 2026.

[40] Hongbin Zhang, Tao Wang, Zhuowei Wang, Nankai Lin, Chong Chen, and Lianglun Cheng. A GPT-assisted multi-granularity contrastive learning approach for knowledge graph entity typing. ACM Transactions on Intelligent Systems and Technology, 17(6), 2026.

[41] Sahil Mishra, Ujjwal Sudev, and Tanmoy Chakraborty. FLAME: Self-supervised lowresource taxonomy expansion using large language models. ACM Transactions on Intelligent Systems and Technology, 17(6), 2026.

[42] Antonello Meloni, Diego Reforgiato Recupero, Francesco Osborne, Angelo Salatino, Enrico Motta, Sahar Vahdati, and Jens Lehmann. Exploring large language models for scientific question answering via natural language to SPARQL translation. ACM Transactions on Intelligent Systems and Technology, 17(6), 2026.

[43] Wei Zhang, Guojun Dai, Ding Luo, Yan Wang, and Chen Ye. From hallucination to certainty: Meta-knowledge guided self-correcting large language models. ACM Transactions on Intelligent Systems and Technology, 17(6), December 2026.