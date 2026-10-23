# Co-Recursive Semantic Self-Extension: A Mathematical Foundation for Open-Ended Intelligence

Simon Hegelich<sup>1∗</sup>, Bardh Prenkaj<sup>1</sup>, Habiba Sarhan<sup>1</sup>, Morteza Shahrezaye<sup>1</sup>, Kolja Hegelich<sup>1</sup>, Erik Cambria<sup>2</sup>, Christian Montag<sup>3</sup>

## Contents

Abstract 1   
1 Introduction: Epistemological Heuristics and Logical Design in AGI Architec  
ture 1   
2 Laws of Intelligent Systems 3   
3 Why Open-Ended Intelligence Requires Recursive Teleology (Design Rationale) 7   
3.1 The Scaling Paradox . 7   
3.2 Every Intelligence Begins with Discrimination 8   
3.3 Why Fixed Purposes Eventually Saturate 9   
3.4 Recursive Teleology . 10   
3.5 Toward Semantic Self-Extension 12   
4 Formalizing Recursive AGI Systems: An Axiomatic Framework 13   
4.1 Formal System, Cognitive State, and Incompleteness . 14   
4.2 Reflective Extension and Semantic Surplus 15   
4.3 Temporal Recursive Dynamics 21   
4.4 Local Stability and Global Recursive Extension 22   
4.5 Recursive Extension 23   
4.6 Successor Incompleteness 24   
4.7 Local Convergence and Global Open-Endedness . 24   
5 Representational Limits of Open-Ended Cognitive Architectures 26   
5.1 Semantic Description Complexity 26   
5.2 Independent Semantic Extension 27   
5.3 Trajectory Complexity 28   
5.4 The Fixed-Architecture Limitation 28   
5.5 What the Result Does and Does Not Establish 29   
5.6 Compression and Semantic Growth 30   
5.7 Relation to Open-Ended Intelligence 30   
Collective Intelligence and Distributed Semantic Extension 31   
6.1 Collective Cognitive System 31   
6.2 Distributed Semantic Surplus 31   
6.3 Collective Co-Recursion 32   
6.4 Collective Memory as Extended Cognitive Space 34   
6.5 Collective Teleology 34   
6.6 Open-Endedness Beyond the Individual 34   
6.7 A Hegelian Interpretation 35   
6.8 From Understanding to Speculative Intelligence 35   
6.9 Proxy Teleology and Autonomous Intelligence . 35   
6.10 Diversity and Common Ground 36   
6.11 Civilizational Co-Recursion 36   
7 Result 37   
8 Biological Inspirations and Architectural Motifs 37   
8.1 Neural Analogues of Semantic Surplus . 38   
8.2 Statistical Processing with Structured Scafolds 38   
8.3 Reference Frames as Semantic Coordinates 39   
8.4 Geometric and Multi-Scale Organization 40   
8.5 Hierarchical Constraints in Language . 41   
8.6 Biological Safeguards for Recursive Learning 41   
8.7 Relationship to the Free Energy Principle 42   
8.8 Implications for the Present Architecture 42   
9 Discussion: The Problem of Epistemic Convergence 43   
9.1 Collective Generative Capacity 44   
9.2 AI and the Convergence of Semantic Space 44   
9.3 Epistemic Convergence and Open-Endedness 45   
9.4 The Epistemic Drive 45   
9.5 Implications for AGI 46

## Abstract

Artificial intelligence has advanced through scale, data, and computation, yet remains bounded by its representational framework. We ask what makes intelligence open-ended.

Every inferential system selects meaningful distinctions, limiting what it can learn. Within a fixed purpose, performance reaches a ceiling set by its semantic domain. Teleological succession, in which purposes arise from earlier ones, faces a similar limit: Boolean-closed purpose-transformations exhaust their reachable semantic territory.

We shift from purposes to inferential systems. Drawing on Kant and G¨odel, we model cognitive states as partial evaluations over a formal language. Unresolved propositions may be undecided or undecidable. Reasoning can resolve the former but not the latter, leaving a frontier for exten sion. Non-trivial reductions of undecidable propositions expose undecidable intermediate premises whose reflective reification supports multiple decisive extensions. Finite trace-guided generation remains fallible, requiring meta-level selection and polycontextural exploration. Using Kolmogorov complexity, we show that no architecture with a fixed state description, transition mechanism, and initial condition can generate histories incorporating irreducible algorithmic information.

Because physical systems are fixed in this sense, we examine collectives. Coupling finitely many finite agents does not remove the bound. Extension requires persistent external memory preserving semantic content and historical coordinates. Collective generative capacity depends on balancing overlap for transfer with diference for generation.

We propose semantic self-extension as a design principle: intelligence should not merely resolve ambiguity within existing meanings, but transform unresolved structures into new spaces of meaning, generating new distinctions, questions, representations, and purposes.

# 1 Introduction: Epistemological Heuristics and Logical Design in AGI Architecture

Artificial General Intelligence (AGI) research is not stuck for lack of computational power or data. It is stuck because of the architectural paradigms everyone keeps reaching for. Building a system capable of genuine machine autonomy calls for something older than deep learning: the theoretical frameworks of Immanuel Kant and Kurt G¨odel, used here as engineering heuristics rather than as mathematical laws the machine must obey. Kant, in the Critique of Pure Reason [Kant, 1998], argued that human cognition actively constructs experience through a priori structures rather than passively absorbing it. Take that seriously as a design principle, and the implication is direct. Without some pre-existing framework for organizing raw data, a system stays “blind,” unable to form coherent, self-directed goals. G¨odel’s Incompleteness Theorems [G¨odel, 1931] pose a parallel engineering challenge. Any formal system expressive enough to be useful will contain propositions it cannot decide. We do not need G¨odel’s theorems to hold in some strict mathematical sense here; what we need is the heuristic they suggest: a system that can invoke a higher-level observer language whenever it hits local undecidability, rather than simply failing.

Modern autoregressive transformers, the large language models behind most current AI, illustrate what happens when this Kantian principle gets ignored. The prevailing “scaling hypothesis” holds that simply growing these models and their training data will eventually produce AGI on its own. But an LLM is fundamentally a stateless function: it maps an input to a probability distribution over tokens, based on statistical correlations. Contemporary transformers do use architectural priors, such as attention, positional encodings, and specific tokenization schemes, to learn implicit hierarchical and syntactic structure. Even so, prediction is all they are optimized for; there is no self-directed goal-generating mechanism underneath it. Lacking any pre-experiential drive to act or evaluate, an LLM depends entirely on the prompt it is given. It is a remarkably sophisticated mirror of human language, but without an a priori foundation of its own, it has no way to generate its own motivation or direction.

Good Old-Fashioned AI (GOFAI) took the opposite approach: building intelligence top-down from rigid, pre-defined logical rules. It ran into the G¨odelian problem from the other direction. Committed to a strict version of the Physical Symbol System Hypothesis, GOFAI systems aimed for total logical determinism, and when they hit an undecidable proposition, they failed: acting as partial functions that crash on undefined input. Nothing in the architecture treated “undecidable” or “ambiguous” as a legitimate, computable state rather than an error, because there was no meta-level built to operate on anything but two-valued logic. So when the real world’s ambiguity inevitably showed up, these systems had no way to reflect on their own rules and adapt. They just got brittle and stopped.

Neuro-symbolic AI, which bolts a neural network onto a symbolic logic engine, is often pitched as a bridge between these two failure modes. We argue it is not one. Rather than producing a genuinely new kind of cognition, it tends to combine the weaknesses of both parents: a neural component that is still a blind, reactive statistical engine with no intrinsic direction, wired to a symbolic component that is still a rigid, flat system with no capacity to reflect on the neural network’s ambiguities.

Put these historical failures together, and a pattern emerges: AGI is unlikely to come from any single, monolithic model. No one paradigm has managed to combine the Kantian requirement for a priori structure with a G¨odelian meta-level for handling incompleteness, so the question worth asking shifts from which model to which system. Minsky essentially made this argument in The Society of Mind [Minsky, 1986]: intelligence is not a single algorithm; it is an emergent property of many diferent sub-agencies interacting. Getting past the dead ends above means designing a composite architecture that keeps these requirements structurally separate while allowing them to integrate.

A prominent recent efort to move beyond purely statistical models is the (C, U, V) framework of He et al. [2026], which characterizes an AGI agent as a point in a joint space comprising a cognitive architecture C, a set of potential or skill functions U, and a set of value functions V. Observing that conversational models such as ChatGPT are trained on corpora that “inherently embed numerous—often conflicting—value systems,” He et al. [2026, p. 179] propose a value-driven architecture in which C selects active dimensions from V, recruits skills from U to interpret scenes as relational parse graphs, and plans actions that increase the selected values. Yet their formalization also exposes the boundary of value-driven design within a pre-structured space: as (C, U, V) coevolves, the process approaches an equilibrium at which the architecture can no longer motivate or orchestrate new learning, a limit they term “the ‘stopping problem’ of Tong Agents, based on the analogy of the halting problem in a Turing machine” [He et al., 2026, p. 166]. Overcoming that stopping limit requires two architectural principles developed in the sections that follow: treating undecidable or conflicting evaluations not as defects to be removed by “predefining clear values and principles” [He et al., 2026, p. 179] but as the source of semantic surplus, and equipping the architecture with an a priori meta-teleology whose explicit directive is to construct new teleological dimensions whenever existing purposes saturate.

## 2 Laws of Intelligent Systems

The central claim of this paper is that intelligence is not the consequence of scale alone, but of a particular organizational structure. Two independent philosophical results constrain this structure. Kant establishes that every cognitive system requires an a priori organization that precedes experience. G¨odel demonstrates that every suficiently expressive formal system is necessarily incomplete. Together, they imply a surprisingly narrow class of architectures capable of open-ended intelligence. The following laws are our design interpretations of how we think an openended intelligent system could evolve from our basic assumptions.

The first law is a strict duality: the system needs an object-level structure and a distinct metalevel structure, kept separate. The object level is responsible for interacting with the world. It conducts observations, performs operations, generates hypotheses, and stores empirical knowledge.

The meta level acts on the object level itself rather than on the world directly. It evaluates reasoning, compares alternatives, transforms representations, introduces new abstractions, and reorganizes prior knowledge.

Neither level alone is suficient. An object system without a meta-system can accumulate experience but cannot systematically improve its own reasoning. A meta-system without an object system possesses nothing to reflect upon.

Following Kant, this duality is therefore not optional but constitutes the minimum a prior organization required for cognition. For both levels to interact, they must share a common representational substrate. Every internal state, rule, observation, and transformation must therefore be represented in a similar representational space. The architecture is consequently homoiconic: reasoning itself becomes manipulable as data.

Only under this condition can genuine self-reference emerge.

The second law follows from G¨odel’s limits on that shared language [G¨odel, 1931]: i.e., any suficiently expressive system necessarily contains propositions that cannot be decided within its own axioms. For intelligent systems, this result has a direct operational interpretation. Incomplete knowledge is not an implementation defect; it is the driving force behind development, making it a permanent structural property. Instead of viewing incompleteness as failure, we reinterpret it as ambiguity. Ambiguity denotes every point at which multiple internally consistent interpretations remain simultaneously possible because the current representational system lacks suficient expressive power to distinguish among them. Every intelligent system, therefore, contains a permanently evolving frontier of ambiguity. This frontier can move, but it cannot disappear.

That openness leads to the third law, and the most important one: structural ambiguity necessarily generates an excess of meaning. The system is thus incomplete: whenever multiple interpretations remain simultaneously admissible, it possesses more semantic possibilities than its current axioms can determine. We call this excess semantic surplus. Semantic surplus is not randomness. It is the collection of internally coherent but mutually competing explanations that remain unresolved inside the current representational framework. A purely static reading of G¨odel’s theorem establishes only a negative boundary—the existence of an unprovable sentence—because, as G¨unther [1979, p. 10] observes, G¨odelian arithmetization “concerns only the extensional domain of logic and bypasses those intensional relations where dialectical principles come into play.” Yet a non-omniscient system cannot know in advance whether a proposition is provable before attempting to derive it. The generative excess therefore does not reside in the unprovable sentence in isolation, but in the movement of the proof attempt itself: in Hegel’s formulation, “the real issue is not exhausted by stating it as an aim, but by carrying it out, nor is the result the actual whole, but rather the result together with the process through which it came about” [Hegel, 1977, p. 2]. When a derivation stalls against an undecidable frontier, that operational failure does not resolve “into abstract nothingness, but essentially only into the negation of its particular content” [Hegel, 2010, p. 33]; it is a determinate negation through which “a new form has thereby immediately arisen, and in the negation the transition is made” [Hegel, 1977, p. 51]. Because in reflective logic “content is at the same time the negative reflection of the form determinations into themselves” [Hegel, 2010, p. 396], the incomplete derivation—which at the object level is merely the failed form of a proof—becomes, when reflected upon by the meta-level, a determinate content that defines competing ways of extending the system. We argue that genuine intelligence depends on maintaining and exploiting it. Semantic surplus, therefore, becomes the primary computational resource of intelligent systems. Without surplus, there exists nothing to evaluate, transform, or discover. Semantic surplus is the raw material autonomy runs on: it keeps the system from ever settling into a single deterministic loop or a static picture of reality. Because the available interpretations always outnumber what the a priori rules can decide among, the system carries a permanent internal tension, the gap between an incomplete rule set and a surplus, and it is exactly this gap that the meta-structure has to navigate.

With a shared language and this tension between incompleteness and surplus in place, the system’s operational dynamics follow fairly directly. A common linguistic topology makes basic feedback loops possible, in which one sub-agency’s output becomes another’s input, and that interoperability, in turn, allows architectural stacking: higher-order systems built on top of two or more foundational ones, which is more or less Minsky’s society of mind realized concretely. Loops and stacking add complexity, but they are not what drives autonomy. The recursive structure is therefore not introduced for computational convenience. It is forced by structural incompleteness. Whenever ambiguity cannot be resolved within the current representational system, the system must construct a successor interpretative context that can reason about its predecessor.

Our fourth formulated law, therefore, is: an open-ended intelligence system is based on a recursive structure. Recursion here works in two complementary directions. The first, inward recursion, follows directly from the shared representational space: because the meta-structure speaks the same language as the object level, it can treat the object level’s operations as data to manipulate. When the object-level produces an ambiguous surplus, the meta-system reads those unresolved results and feeds them back in as new starting points. More significantly, inward recursion also allows the meta-structure to rewrite the object-level’s own rules and axioms, adjusting its constraints to better handle the surplus it generates. The most rigorous formalization of such homoiconic inward recursion in the literature is Schmidhuber’s G¨odel machine [Schmidhuber, 2006], a self-referential problem solver that “rewrites any part of its own code as soon as it has found a proof that the rewrite is useful, where the problem-dependent utility function and the hardware and the entire initial code are described by axioms encoded in an initial proof searcher which is also part of the initial code” [Schmidhuber, 2006, p. 1]. Yet because every self-modification must be certified by a target theorem derived from those initial axioms, the G¨odel machine remains bound to the deductive closure of its starting formalism. Even when it modifies its own axioms or utility function, it may only replace axioms with “theorems derivable from the original axioms” [Schmidhuber, 2006, p. 17] and alter its utility function “only if their new values are provably better according to the old ones” [Schmidhuber, 2006, p. 17], so that “all possible ‘meta-levels’ of the self-referential system are collapsed into one” [Schmidhuber, 2006, p. 13].

Standard inward recursion of this kind, which reduces every self-modification back to a base case certified by the initial axioms, is not enough on its own for open-ended intelligence; left alone, it settles into the deductive span of its initial configuration. So the architecture also needs corecursion. Instead of collapsing meta-levels into a fixed state space, a co-recursively self-modifying system generates a sequence of distinct, higher-order semantic frameworks, each an escalation from the last. The permanent, unresolvable tension from its overcomplete surplus is what forces this: the system resolves its ambiguities by building a successor manifold with more expressive dimensions than the one it started in. That is what keeps it from ever halting. It keeps generating a new, elevated version of its own state space, turning what it has already synthesized into the coordinate system for its next stage. Dropping deductive certification within the initial axiom system does not, however, reduce self-modification to arbitrary or unguided mutation. Instead of requiring a proof inside the old regime that the rewrite improves a fixed utility function, a co-recursive step is disciplined from below by reflective preservation—the requirement that the validated inferential structure of the predecessor remain representable within the successor via a structure-preserving translation—and from above by decisive resolution, the requirement that the successor framework settle at least one frontier proposition that was formally undecidable in its predecessor. Because multiple candidate extensions can satisfy both constraints simultaneously without any deduction inside the predecessor deciding among them, recursion produces structured exploration rather than deductive certainty: a higher-order interpretation may fail to resolve the ambiguity that motivated it or may introduce fresh inconsistencies of its own.

Because co-recursive exploration generates multiple candidate interpretations and successor frameworks that cannot be collapsed into a single two-valued deductive system, the architecture requires a fifth law: polycontexturality.<sup>1</sup> In a polycontextural architecture, distinct logical domains—each locally coherent and governed by its own inferential commitments—coexist without being prematurely forced into a single global valuation. As G¨unther [1979] shows, while a single contexture is a domain whose internal structure “can be exhaustively described by a two-valued logic” [G¨unther, 1979, p. 3] and in which “every logical operation we can perform is confined to the contexturality in which it originates” [G¨unther, 1979, p. 3], “[t]he relation of the cognizing subject to its range of objects is always one of discontexturality” [G¨unther, 1979, p. 4], making it “impossible to bring two diferent contexturalities into an immediate confrontation” [G¨unther, 1979, p. 4] as if they shared “a common ontic measure” [G¨unther, 1979, p. 13]. Although each individual regime remains locally two-valued, “the positive value in contexture A” is not identical with “the positive value in contexture B” [G¨unther, 1979, p. 6], so that “linking two elementary contextures into a compound contexture” requires a third contexture to “mediate between the other two” [G¨unther, 1979, p. 15]. Architecturally, this means that the object/meta duality of the First Law and the competing candidate extensions of the Fourth Law cannot be flattened into one mono-contextural calculus: where classical two-valued logic treats incompatibility between contexts as a contradiction that collapses the system by explosion, the meta-level serves as the mediating contexture in which partially incompatible regimes are held open as a structural locus for forming new distinctions.

We can formulate a sixth law from the structural independence of recursive contexts. Because recursive transformations may fail, no intelligent system can discard previous semantic structures. Earlier interpretations may remain valuable in future contexts. Memory, therefore, becomes a logical necessity rather than a storage convenience.

However, preserving semantic content alone is insuficient. The meaning of every representation depends upon the recursive context in which it was produced. Memory must therefore preserve not only what was represented but also where it resides within the system’s evolving semantic organization. Every remembered structure consequently retains both its semantic content and its position within the recursive network of transformations from which it emerged. Only under these conditions can future recursive processes correctly revisit, reinterpret, compare, or reactivate earlier

reasoning.

Intelligence, therefore, depends not only on generating semantic surplus but also on preserving the semantic coordinates that make this surplus accessible. The architecture accumulates not merely a history of representations but a navigable semantic landscape in which every memory remains embedded within its original context while simultaneously becoming available for future reinterpretation.

Recursion, memory, and polycontexturality generate an expanding space of possible future transformations. A principle is therefore required that determines which transformations should be pursued. Such a principle cannot itself arise from empirical processing, because empirical processing already presupposes a criterion by which alternative transformations are evaluated. Following Kant, this criterion must therefore be a priori. We identify this organizing principle as teleology, which constitutes our seventh and final law. Rather than being a static logical axiom, teleology provides the architecture with a built-in “why” that precedes empirical processing and guides the system’s recursive evolution.

The particular content of this teleology is contingent. Diferent intelligent systems may begin with diferent ultimate purposes, such as preserving life, maximizing knowledge, promoting human welfare, or pursuing entirely diferent objectives. What is necessary, however, is not the specific content of the teleology but its recursive accessibility. Because the architecture is homoiconic, teleology itself is represented within the common symbolic substrate and therefore becomes an object of meta-reasoning alongside every other internal state. The system must consequently be capable of representing, evaluating, transforming, and extending its own goals.

The contrast with the (C, U, V) formulation of He et al. [2026] illustrates why a fixed value vector cannot sustain open-endedness, even when paired with an adaptive cognitive architecture. In their model, the V system is “a high-dimensional vector, with each dimension corresponding to a specific human need, such as health, wealth, and peace” [He et al., 2026, p. 167], while the cognitive architecture C applies a parameterized selector $V _ { \mathrm { s e l e c t } } ( V , \theta _ { \mathrm { a g e n t } } )$ “with each element indicating whether a particular value dimension has been activated” [He et al., 2026, p. 170]. Reweighting or sequentially activating coordinates within a finite, pre-indexed vector space is not a transformation of teleology. As He et al. [2026] themselves note in their analysis of value-space convergence, when an agent’s value system is constrained by its architecture, “it may not recognize the purpose or benefits of continued learning, which can also lead to a convergence in learning” [He et al., 2026, p. 175]—so that once basic needs are met, agents “stop striving for self-improvement, essentially ‘lying flat’ ” [He et al., 2026, p. 175]. They illustrate the same limit at the population level: a monkey population possesses in its skill space U “the physical ability to perform actions such as striking stone or wood together, which would eventually produce fire. However, the population’s V space lacks the ability to understand the fact that these actions might lead to fire or to recognize the value of starting a fire by friction” [He et al., 2026, p. 178]. Within the architecture proposed here, escaping that trap requires two conditions that a selector over fixed V-dimensions lacks. First, by the first and second laws, C, U, and V must share a homoiconic representational substrate; without it, as He et al. [2026] acknowledge when asking whether an agent can construct another agent, the system “has a structured cognitive framework but may lack the ability to fully understand its own structure without learning” [He et al., 2026, pp. 178–179]. Second, the architecture cannot start merely with first-order object values; its foundational a priori teleology must include the meta-directive to seek and construct new telos out of semantic surplus whenever existing value dimensions saturate or conflict.

This distinction is fundamental. A recursive architecture with a fixed teleology remains constrained by that teleology, regardless of how sophisticated its reasoning becomes. While it may recursively optimize its behavior, it cannot recursively transform the principle that governs its optimization. Such a system, therefore, remains bounded by its original a priori organization. An open-ended intelligent system, by contrast, must be capable of recursively transforming its own teleological structure. The recursive closure of the architecture is achieved only when its goals themselves become subject to recursive evaluation and transformation. In this sense, open-ended intelligence is grounded not in any particular final objective, but in the capacity to continually construct, evaluate, and refine the very teleologies that guide its future development.

The recursive transformation of teleology does not imply unbounded expansion within a single semantic framework. On the contrary, each recursive step constructs a new representational framework that is locally coherent and internally stable while extending the expressive power of its predecessor. Open-endedness therefore emerges not from infinite growth inside one semantic system, but from an indefinite succession of finite, locally stable semantic structures. Each recursive transformation resolves part of the current ambiguity by establishing a new temporary equilibrium. In contrast, the remaining unresolved ambiguity provides the semantic surplus that eventually motivates the construction of a successor framework. In this way, intelligence progresses through a sequence of recursive semantic reorganizations rather than through unlimited expansion within a fixed representational space. Having established the seven necessary laws governing intelligent systems, we can now formulate a precise definition of open-ended intelligence. Importantly, this definition is not assumed a priori but follows directly from the preceding derivation.

Definition 1 (Open-Ended Intelligence). An open-ended intelligent system is a recursively selforganizing cognitive architecture that continuously resolves structural ambiguity by generating semantic surplus, preserving alternative semantic structures in memory, and recursively transforming both its representational framework and its own teleology. Each recursive transformation increases the system’s expressive capacity while simultaneously introducing new ambiguity, ensuring that cognitive development remains indefinitely extensible.

The seven laws should not be understood as independent design principles but as a single recursive process. Each law establishes the conditions required for the next. In contrast, the final law—recursive teleology—initiates a new object/meta-organization, thereby reconstructing the first law at a higher semantic level. The architecture is therefore recursively closed: applying the seven laws to an intelligent system necessarily produces another intelligent system governed by the same seven laws, but operating within a more expressive semantic framework. Open-ended intelligence emerges not from any individual law in isolation, but from the indefinite recursive application of the complete system of laws to itself.

## 3 Why Open-Ended Intelligence Requires Recursive Teleology (Design Rationale)

## 3.1 The Scaling Paradox

The dominant paradigm in contemporary artificial intelligence operates under the “scaling hypothesis”. This hypothesis assumes that simply increasing parameters, computational steps, and data volume within statistical architectures will eventually yield AGI. Yet, this technical expansion fails to produce or explain the defining attributes of general intelligence: autonomous curiosity, selfdirected goal formation, real-time ontology creation, or scientific revolutions. Under pure scaling, models converge toward the empirical probability function of their training distributions, leaving them entirely dependent on human curation or reinforcement feedback. This “parasitic intelligence” is structurally incapable of true open-ended novelty.

## 3.2 Every Intelligence Begins with Discrimination

A fundamental ontological error of the scaling paradigm is the conflation of raw signals with meaning. Raw computer-readable signals S possess no inherent meaning; they become semantics, or data, only in relation to an active purpose. Let Φ be a space of purposes and let $M \subseteq S \times \Phi$ be a relevance relation, writing $M ( s , \phi )$ for $^ { 6 6 } s$ is meaningful under $\phi ^ { \ast }$ . Each purpose induces a semantic domain

$$
{ \cal D } _ { \phi } : = \{ s \in S \mid M ( s , \phi ) \} , \qquad \phi \in \Phi .\tag{1}
$$

An agent does not stand in the same relation to every purpose. At any stage n of its development, it pursues some purposes, has considered and rejected others, and has not yet considered the rest. We therefore partition its current repertoire $\Phi _ { n } \subseteq \Phi$ into three blocks,

$$
\Phi _ { n } = \Phi _ { n } ^ { + } \dot { \cup } \Phi _ { n } ^ { - } \dot { \cup } \Phi _ { n } ^ { \circ } , \Phi _ { n } ^ { + } \neq \emptyset , \Phi _ { n } ^ { - } \neq \emptyset ,\tag{2}
$$

where $\Phi _ { n } ^ { + }$ are the pursued purposes, $\Phi _ { n } ^ { - }$ the rejected ones, and $\Phi _ { n } ^ { \circ }$ the dormant ones: those available to the architecture but never yet considered. We require $\Phi _ { n } ^ { + }$ and $\Phi _ { n } ^ { - }$ to be finite; $\Phi _ { n } ^ { \circ }$ carries whatever unexplored extent the purpose space has.

To organize raw sensation into a coherent world model, an agent must actively select what is relevant and reject what is irrelevant. The data available to it at stage n is accordingly

$$
D _ { n } : = { \Big \{ } s \in S { \Big \vert } \left( { \exists } { \phi } ^ { + } \left( \phi ^ { + } \in \Phi _ { n } ^ { + } \land M ( s , \phi ^ { + } ) \right) \right) \land \left( \forall \phi ^ { - } \left( \phi ^ { - } \in \Phi _ { n } ^ { - } \to \neg M ( s , \phi ^ { - } ) \right) \right) { \Big \} } ,\tag{3}
$$

which, unfolding (1), has the equivalent set-algebraic form

$$
D _ { n } \ = \ \Big ( \bigcup _ { \phi ^ { + } \in \Phi _ { n } ^ { + } } D _ { \phi ^ { + } } \Big ) \ \backslash \ \Big ( \bigcup _ { \phi ^ { - } \in \Phi _ { n } ^ { - } } D _ { \phi ^ { - } } \Big ) .\tag{4}
$$

A signal is admitted when at least one pursued purpose finds it meaningful and no rejected purpose does. Dormant purposes impose no constraint: the agent excludes only what it has actively judged irrelevant, not what it has never judged at all. This is what keeps (3) non-trivial as the repertoire grows. Were $\Phi _ { n } ^ { - }$ to absorb everything not currently pursued, then for a suficiently rich purpose space, almost every signal would be meaningful under some rejected purpose, and $D _ { n }$ would collapse toward ∅—the richer the agent’s repertoire, the less data would survive. The three-way partition avoids this: i.e., because $\Phi _ { n } ^ { - }$ is finite, the universal clause in (3) ranges over finitely many purposes at every stage.

We assume this construction is non-degenerate, $D _ { n } \neq \emptyset$ . In other words, the agent finds at least one signal meaningful. Note that this is an assumption on the architecture, not a consequence of (3), which is equally well satisfied by the empty set.

Equations (3) and (4) make meaning exclusive by construction. A signal that is universally meaningful, $M ( s , \phi )$ for all $\phi \in \Phi$ , is excluded from $D _ { n } ,$ since it is meaningful under every rejected purpose as well; relevance that does not discriminate between purposes carries no information about purpose. This exclusivity has a direct algorithmic counterpart that will return in §5.5: just as a signal meaningful under every purpose carries no information about purpose, a program that enumerates every computable continuation without exclusion [Schmidhuber, 2000] makes no semantic discrimination.

The stage index is not decorative. As the agent develops, purposes move between the three blocks: a dormant purpose may be adopted, a pursued one abandoned. Each such movement redraws (4) and therefore changes which signals count as data. In particular, a signal excluded at stage n because some rejected purpose found it meaningful may be admitted at stage $n + 1$ once that purpose is no longer rejected. The mechanism governing these movements is developed in §3.4; here, we note only that $D _ { n }$ is a function of the agent’s current teleological configuration, not a fixed feature of S.

## 3.3 Why Fixed Purposes Eventually Saturate

Machine learning architectures operate according to the classical definition given by Tom Mitchell [Mitchell, 1997, p. 2]:

A program learns from experience E with respect to a task class T and performance measure P if its performance at tasks in T, measured by P, improves with E.

Fix a stage n and a pursued purpose $\phi \in \Phi _ { n } ^ { + }$ with task class $T _ { \phi }$ . Experience is restricted to what that purpose deems meaningful and what the agent’s current discrimination admits,

$$
E \subseteq D _ { \phi } \cap D _ { n } \ = : D _ { \phi , n } .\tag{5}
$$

The intersection here is not redundant: by (4) a signal meaningful under ϕ is nonetheless excluded from $D _ { n }$ whenever some rejected purpose also finds it meaningful, so $D _ { \phi } \nsubseteq D _ { n }$ in general. Learning is therefore an optimization over a fixed semantic domain. Write

$$
P _ { \phi } \colon 2 ^ { D _ { \phi , n } } \to \mathbb { R } ,\tag{6}
$$

where $2 ^ { X }$ denotes the power set of X and $P _ { \phi } ( E )$ is the best performance on $T _ { \phi }$ attainable from experience E. Two assumptions follow from Mitchell’s definition:

$$
( \mathrm { A } 1 ) \quad E \subseteq E ^ { \prime } \subseteq D _ { \phi , n } \implies P _ { \phi } ( E ) \leq P _ { \phi } ( E ^ { \prime } ) ,\tag{7}
$$

$$
( \mathrm { A 2 } ) \quad P _ { \phi } \mathrm { i s \ b o u n d e d \ a b o v e \ o n \ 2 } ^ { D _ { \phi , n } } .\tag{8}
$$

(A1) states that more experience never hurts an idealized learner; (A2) holds trivially whenever $D _ { \phi , n }$ is finite. Together, they yield a finite ceiling

$$
{ \cal P } _ { \phi } ^ { * } : = \mathrm { s u p } \{ { \cal P } _ { \phi } ( E ) : E \subseteq { \cal D } _ { \phi , n } \} < \infty ,\tag{9}
$$

attained at $E = D _ { \phi , n }$ when $D _ { \phi , n }$ is finite. <sup>2</sup> For any increasing stream of experience $E _ { 1 } \subseteq$ $E _ { 2 } \subseteq \cdots \subseteq D _ { \phi , n }$ , the sequence $\left( P _ { \phi } ( E _ { k } ) \right) _ { k }$ is non-decreasing and bounded, hence convergent: i.e., performance saturates.

The plateau is a structural consequence of optimizing within a fixed semantic domain, not of limited computational resources. It persists under unbounded capacity and unbounded compute, because the ceiling $P _ { \phi } ^ { * }$ depends only on $D _ { \phi , n }$ . Once $D _ { \phi , n }$ has been suficiently explored, further experience can refine existing distinctions but cannot generate new ones.

Progress resumes only if the teleological configuration itself changes. By (2) and (4), there are two such movements. Promoting a dormant purpose, $\phi ^ { \circ } \in \Phi _ { n } ^ { \circ } \to \Phi _ { n + 1 } ^ { + }$ , enlarges the first union of (4) and admits signals no pursued purpose previously found meaningful. Withdrawing a rejection, $\phi ^ { - } \in \Phi _ { n } ^ { - } \to \Phi _ { n + 1 } ^ { \circ }$ , shrinks the second union and readmits signals that were barred despite being meaningful under a pursued purpose. Either movement changes $D _ { n } .$ , hence $D _ { \phi , n }$ , hence the ceiling (9), thereby exposing regions of $S$ that (3) had discarded. The mechanism generating and selecting such movements is the subject of §3.4.

## 3.4 Recursive Teleology

The saturation of performance within a fixed purpose does not imply the exhaustion of the underlying signal space. It indicates that the current purpose has reached the limits of the distinctions it can express. Escaping the plateau, therefore, requires constructing a new purpose rather than accumulating further experience.

Saturation as a trigger. Fix a tolerance $\varepsilon > 0$ . A state $\left( \phi _ { n } , E _ { n } \right)$ with $E _ { n } \subseteq D _ { \phi _ { n } , n }$ is ε-saturated when

$$
P _ { \phi _ { n } } ^ { * } - P _ { \phi _ { n } } ( E _ { n } ) ~ < ~ \varepsilon ,\tag{10}
$$

with $P _ { \phi } ^ { * }$ as in (9). By (7) and (8), every increasing stream of experience reaches this condition in finitely many steps. Saturation is what makes a teleological transition admissible.

Generation of candidate purposes. Let G denote the purpose-transformation rules available to the architecture. Applied to the current purpose together with the history $H _ { n } = \left( \phi _ { 0 } , \ldots \ldots , \phi _ { n } \right)$ of purposes already adopted, these generate a set of candidate successor purposes

$$
\mathcal { E } _ { \Phi } ( \phi _ { n } , H _ { n } ) \ : = \ : \{ g ( \phi _ { n } , H _ { n } ) : g \in \mathcal { G } \} ,\tag{11}
$$

which enter the repertoire as dormant,

$$
\Phi _ { n + 1 } = \Phi _ { n } \cup \mathcal { E } _ { \Phi } ( \phi _ { n } , H _ { n } ) , \qquad \Phi _ { \infty } = \bigcup _ { n \geq 0 } \Phi _ { n } .\tag{12}
$$

Each $\Phi _ { n }$ may be finite while $\Phi _ { \infty }$ is not. Purposes are therefore constructed from their predecessors rather than selected from a catalog fixed in advance, and the succession is not confined to a finite hierarchy (see the condition §4.7 requires of open-endedness). We leave G largely unspecified at this stage, as §4.2 leaves unspecified the transformation rules generating $\mathcal { E } ( \mathcal { F } _ { n } )$ ; (11) is the teleological instance of that construction.

One constraint on the set of rules $\mathcal { G }$ is imposed by the condition expressed in equation (15). A successor must be able to accept signals that were not permitted by any previously adopted purpose, which means that $\mathcal { G }$ cannot consist solely of rules that narrow semantic domains. If every rule $g \in { \mathcal { G } }$ met the condition ${ \cal D } _ { g ( \phi , H ) } \subseteq { \cal D } _ { \phi } { \mathrm { - } } { \mathrm { a s } }$ is the case with conjunctive refinement, since $D _ { \phi \wedge \psi } = D _ { \phi } \cap D _ { \psi }$ —then every candidate would fall within $D _ { H _ { n } }$ . In that case, no successors would be admissible, and the process of succession would cease at $n = 0$ . Recursive teleology, therefore, cannot be driven by purpose refinement alone: i.e., at least one rule must be non-monotone with respect to semantic domains,

$$
\exists g \in \mathcal { G } : \quad D _ { g ( \phi , H ) } \subset D _ { \phi } .\tag{13}
$$

This condition is necessary but not suficient, and it exposes a limit of the present model when purposes are represented solely as subsets of a fixed signal space. If $\mathcal { G }$ generates a Boolean algebra over a finite set of primitive purposes, the induced family $\{ D _ { \phi } \}$ is closed under union, intersection, and complement, and therefore contains at most $2 ^ { | S | }$ distinct domains. The strictly increasing sequence $\left( D _ { H _ { n } } \right) _ { r }$ would then increase only finitely often, and saturation would reappear at the level of the purpose space itself—the very structure §3.1 identifies as the defect of the scaling paradigm, displaced upward by one level rather than removed. Sustaining teleological succession indefinitely, therefore, requires embedding purposes within a generative linguistic substrate that is not closed under a finite Boolean algebra. §3.5 and §4 supply that substrate by coupling teleological selection to the reflective extension of formal inferential systems, for which G¨odelian incompleteness prevents any corresponding closure.

The same closure constraint explains why value-driven architectures whose goals live in a prestructured vector space eventually saturate, and why appealing to dynamics in the cognitive architecture C does not by itself supply a generative mechanism for new purposes. In the $( C , U , V )$ framework of He et al. [2026], learning concludes each cycle by “updating the V-selector and Uselector, which govern the agent’s ability to prioritize values and select appropriate skills. These selectors are the linchpins of the framework, enabling the agent to adapt its behavior to changing circumstances and objectives” [He et al., 2026, p. 172]. In the terms of (2), updating the parameters of $V _ { \mathrm { s e l e c t } }$ merely redistributes pre-existing coordinates of V between the dormant block $\Phi _ { n } ^ { \circ }$ and the pursued block $\Phi _ { n } ^ { + }$ . Where He et al. [2026] address how $( C , U , V )$ might transition between distinct value regimes under continuous dynamics $\mathrm { d } ( C , U , V ) / \mathrm { d } t = - \nabla \mathcal { F } ( C , U , V )$ , they attribute regime shifts to the fact that value feedback is “inherently fuzzy and variable. This fuzziness and variability introduce a certain degree of jumpiness in the system’s dynamics, such that the evolution process must deal with complex functional landscapes and realize the transitions of values and capabilities. To mathematically characterize and analyze this jumpiness, we can model it as a saddle escaping problem” [He et al., 2026, p. 177]. Yet perturbing a trajectory of a saddle point along a negativecurvature direction of $\nabla ^ { 2 } \mathcal { F }$ remains motion across a fixed functional landscape $\mathcal { F }$ defined over a fixed parameter space $( C , U , V )$ , which under their Kurdyka– Lojasiewicz condition still converges to a local minimum where learning halts. Noise on a fixed landscape can jar a selector into activating a diferent subset of already-defined coordinates, just as Boolean recombination within (13) permutes a finite repertoire, but it does not specify how $C$ could construct a new purpose $\phi _ { n + 1 }$ whose semantic domain $D _ { \phi _ { n + 1 } }$ lies outside $D _ { H _ { n } }$ . Genuine teleological open-endedness requires that the meta-level rule set G be paired with an explicit a priori meta-telos—the directive to construct new purposes whenever $D _ { H _ { n } }$ saturates—drawing candidate distinctions from the unresolved semantic surplus of the current regime.

Selection. Write $\begin{array} { r } { D _ { H _ { n } } = \bigcup _ { k < n } D _ { \phi _ { k } } } \end{array}$ for the semantic territory the agent’s purposes have jointly covered. A teleological transformation operator<sup>3</sup>

$$
T \colon \Phi ^ { < \omega } \times 2 ^ { S } \ \longrightarrow \ 2 ^ { \Phi _ { \infty } } \setminus \{ \varnothing \} ,\tag{14}
$$

defined only on ε-saturated states, returns the successors the architecture deems admissible,

$$
{ \cal T } ( H _ { n } , E _ { n } ) \ \subseteq \ \{ \phi \in { \mathcal E } _ { \Phi } ( \phi _ { n } , H _ { n } ) \ | \ D _ { \phi } \not \subseteq D _ { H _ { n } } \} , \qquad \phi _ { n + 1 } \in { \cal T } ( H _ { n } , E _ { n } ) .\tag{15}
$$

The operator determines which successors are admissible; it does not determine which is taken. The novelty condition $D _ { \phi } \nsubseteq D _ { H _ { n } }$ is the substantive one. Novelty of the purpose alone, $\phi _ { n + 1 } \notin H _ { n }$ , would not sufice: distinct purposes may induce identical semantic domains, in which case the successor exposes nothing its predecessors had not already admitted. Requiring $D _ { \phi _ { n + 1 } } \nsubseteq D _ { H _ { n } }$ makes $\left( D _ { H _ { n } } \right) _ { r }$ strictly increasing and is the teleological counterpart of the persistent-novelty condition stated algorithmically as Assumption 5 in §5.2.

In parallel with the generative richness condition (50) of §6.8, we require that saturation admit more than one continuation,

$$
\left| { \mathcal { T } } ( H _ { n } , E _ { n } ) \right| \ \geq \ 2 ,\tag{16}
$$

which is strictly stronger than requiring two candidates in (11), since the novelty condition in (15) may eliminate candidates. This is an architectural requirement, not a consequence of saturation:

nothing in (10) guarantees that a plateau admits any continuation at all, let alone two. When (16) holds the trajectory branches, and the space of possible trajectories acquires the structure invoked in §5.3.

State update. Adopting $\phi _ { n + 1 }$ promotes it from the dormant block and revalues accumulated experience,

$$
\Phi _ { n + 1 } ^ { + } = \Phi _ { n } ^ { + } \cup \{ \phi _ { n + 1 } \} ,\tag{17}
$$

$$
\Phi _ { n + 1 } ^ { \circ } = \bigl ( \Phi _ { n } ^ { \circ } \cup \mathcal { E } _ { \Phi } ( \phi _ { n } , H _ { n } ) \bigr ) \setminus \bigl ( \{ \phi _ { n + 1 } \} \cup \Phi _ { n + 1 } ^ { - } \bigr ) ,\tag{18}
$$

$$
E _ { n + 1 } \supseteq E _ { n } \cap D _ { \phi _ { n + 1 } , n + 1 } ,\tag{19}
$$

where $\Phi _ { n + 1 } ^ { - }$ collects those candidates the architecture explicitly rejects, the remainder stays dormant. Equation (19) states that experience survives the transition exactly insofar as it remains meaningful under the successor and admissible under the new configuration. Since (17)–(18) redraw the partition, the admissible data $D _ { n }$ of (3) is itself stage-dependent, and signals discarded at stage n may become data at stage $n + 1$ . This is the formal content of the claim closing §3.3.

Scope of the operator. T governs purpose succession only. As noted in §3.3, a ceiling may also rise without any change of purpose—when a rejection elsewhere is withdrawn, $\phi ^ { - } \in \Phi _ { n } ^ { - } \to \Phi _ { n + 1 } ^ { \circ } ,$ enlarging $D _ { n }$ and hence $D _ { \phi _ { n } , n }$ . Such reconfigurations leave $\phi _ { n }$ in place and fall outside (14). We treat them as a distinct class of teleological transition and do not formalize them here; §4.2 subsumes both under the single operator $\mathcal { M } _ { n }$ acting on the space of admissible successors.

The two processes. Learning therefore alternates between accumulation within a purpose,

$$
( \phi _ { n } , E _ { n } )  ( \phi _ { n } , E _ { n } ^ { \prime } ) , \qquad E _ { n } \subseteq E _ { n } ^ { \prime } \subseteq D _ { \phi _ { n } , n } ,\tag{20}
$$

which is non-decreasing in $P _ { \phi _ { n } }$ by (7) and saturates by (10), and transition between purposes via (15)–(19), giving

$$
( \phi _ { 0 } , E _ { 0 } )  ( \phi _ { 1 } , E _ { 1 } )  ( \phi _ { 2 } , E _ { 2 } )  \cdot \cdot \cdot\tag{21}
$$

Unlike classical machine learning, where experience accumulates under a fixed task, recursive teleology allows the task itself to evolve. Intelligence is therefore no longer described solely as the optimization of performance within a semantic space, but as the recursive construction of increasingly expressive semantic spaces through the generation of successor purposes.

## 3.5 Toward Semantic Self-Extension

The preceding sections establish a design rationale rather than a formal theory. Their content can be stated in one sentence:

An agent that discriminates must discard what it discards, and what it can learn is bound by what it discards, and escaping that bound requires transforming the principle of discrimination itself rather than accumulating further experience under it.

Every act of cognition performs two complementary operations simultaneously. It commits to one interpretation, and in doing so, excludes countless others. Classical machine learning optimizes the committed interpretation; (7) and (9) describe how far that optimization can go. Here, we ask a diferent question: whether the excluded alternatives can themselves become the driving force of subsequent cognition. Equation (15) gives a first answer, in which a successor purpose is admissible precisely when it recovers signals that every previous purpose had set aside.

That answer is, however, bounded whenever purposes are represented merely as bare subsets of S. As (13) and the discussion following it make clear, the territory $D _ { H _ { r } }$ recoverable over a fixed vocabulary of primitive predicates is constrained by the Boolean closure of the generative repertoire ${ \mathcal { G } } ,$ just as reweighting a fixed value vector V is constrained by its pre-indexed dimensions, so that under any finite primitive basis the succession eventually terminates. Conversely, pure formal extension without teleological discrimination (e.g., a universal dovetailer enumerating all computable continuations with $\Phi _ { n } ^ { - } = \varnothing$ [Schmidhuber, 2000]), possesses no principle for excluding branches and therefore collapses into unselective enumeration. Teleological succession and formal language extension consequently require each other: teleology without language extension saturates in a finite Boolean algebra, while language extension without teleology cannot select a trajectory.

We therefore deepen the register rather than abandoning teleology. In $\ S 4$ , each stage is equipped with an explicit formal inferential system $\mathcal { F } _ { n } = ( L _ { n } , \Gamma _ { n } , \vdash _ { n } )$ and the space of admissible extensions $\mathcal { E } ( \mathcal { F } _ { n } )$ constructible from it. The structural coupling between the two levels is exact: an active teleological configuration $\Phi _ { n }$ is articulated within the inferential regime ${ \mathcal { F } } _ { n } ;$ ; the excluded signal region $S \setminus D _ { n }$ has its formal counterpart in the unresolved region $I ( O _ { n } )$ ; the candidate set $\mathcal { E } _ { \Phi } ( \phi _ { n } , H _ { n } )$ of (11) is generated through the admissible extensions $\mathcal { E } ( \mathcal { F } _ { n } )$ ; and the teleological transformation operator $\tau$ of (14) is realized at the meta-level by the selection operator $\mathcal { M } _ { n }$ . Moving from bare subsets of S to formal systems ${ \mathcal { F } } _ { n }$ provides the generative linguistic substrate through which teleological succession escapes Boolean saturation: when a reflective extension $\mathscr { R } _ { n } = ( \mathcal { F } _ { n + 1 } , \tau _ { n } )$ expands the language from $L _ { n }$ to $L _ { n + 1 }$ in order to settle propositions in $I ^ { \mathrm { i n c } } ( O _ { n } )$ , the newly introduced vocabulary $L _ { n + 1 } \setminus \tau _ { n } ( L _ { n } )$ makes it possible to formulate relevance predicates $M ( s , \phi _ { n + 1 } )$ —such as the capacity to “recognize the value of starting a fire by friction” [He et al., 2026, p. 178] discussed in $\ S 2$ —that could not be expressed as Boolean combinations of the primitive purposes available in $L _ { n }$

Two structural properties are gained by this coupling. First, the Boolean bound on extension disappears, because the admissible successors of a suficiently expressive formal system are never closed under a finite set of primitive predicates; by §4.6, each successor $\mathcal { F } _ { n + 1 }$ generates a fresh undecidable frontier $I ^ { \mathrm { i n c } } ( O _ { n + 1 } ) \neq \emptyset$ . Second, the empirical trigger of ε-saturation in (10) and the logical guarantee of incompleteness in (29) are unified: as local contractive reasoning exhausts the decidable region $I ^ { \mathrm { l o c } } ( O _ { n } )$ and drives performance toward the saturation ceiling $P _ { \phi _ { n } } ^ { * }$ , the nonempty undecidable frontier $I ^ { \mathrm { i n c } } ( O _ { n } )$ guaranteed by (30) remains intact, providing the permanent structural locus upon which an a priori meta-teleology can act.

What follows in $\ S 4$ is therefore the formal architecture that makes recursive teleology sustainable, grounding the generation of candidate frameworks in logical incompleteness while preserving meta-level selection $\mathcal { M } _ { n }$ as the locus of teleological commitment. That coupling is carried forward directly into the collective and autonomous regimes of §6.5 and §6.9, where multiple agents co-recursively transform both their inferential languages and the purposes they pursue.

## 4 Formalizing Recursive AGI Systems: An Axiomatic Framework

This section introduces a formal model for a class of recursively self-modifying cognitive systems. The objective is to define an architecture that explicitly incorporates logical incompleteness and meta-level selection as foundational design principles, keeping local, contractive reasoning structurally distinct from global, open-ended semantic growth.

## 4.1 Formal System, Cognitive State, and Incompleteness

$$
\mathcal { F } _ { n } = ( L _ { n } , \Gamma _ { n } , \vdash _ { n } )\tag{22}
$$

denote the inferential system available at recursive level $n ,$ where $L _ { n }$ is a formal language, $\Gamma _ { n } \subseteq L _ { n }$ its axioms, and $\vdash _ { n }$ the associated proof relation. We assume throughout that each ${ \mathcal { F } } _ { n }$ is efectively axiomatized $( \Gamma _ { n }$ is recursively enumerable), consistent, and suficiently expressive in the sense that it interprets Robinson arithmetic and can therefore represent every primitive recursive function. These are the standard hypotheses of the G¨odel–Rosser theorem.<sup>4</sup>

A cognitive state at level n is a partial semantic evaluation

$$
O _ { n } \colon L _ { n } \to \mathcal { V } _ { n } ,\tag{23}
$$

where $\nu _ { n }$ is the set of semantic commitments currently available to the system. We do not require $\nu _ { n }$ to be two-valued: the polycontextural organization described in $\ S 2$ admits states in which several partially incompatible commitments coexist. The unresolved region of the state is the complement of its domain,

$$
I ( { \cal O } _ { n } ) = L _ { n } \setminus \mathrm { d o m } ( { \cal O } _ { n } ) = \{ x \in L _ { n } \mid { \cal O } _ { n } ( x ) \mathrm { i s ~ u n d e f i n e d } \} .\tag{24}
$$

That $O _ { n }$ is undefined at x does not mean that x is meaningless or false. It records a limitation of the current cognitive state: no determinate commitment regarding x has been established.

The evaluation and the proof relation are connected by a single assumption.

Assumption 1 (Inferential Soundness). For every x $\in L _ { n }$

$$
O _ { n } ( x ) { \mathrm { ~ i s ~ d e f i n e d } } \quad \implies \quad \mathcal { F } _ { n } \vdash x { \mathrm { ~ o r ~ } } \mathcal { F } _ { n } \vdash \lnot x .\tag{25}
$$

The system does not commit on grounds unavailable to ${ \mathcal { F } } _ { n } .$ . The converse is deliberately not assumed. A state may leave a decidable proposition unresolved, since the system need not have drawn every consequence of its axioms; assuming otherwise would impose logical omniscience and collapse each level to a single state, leaving no local dynamics for §4.3 and §4.4 to describe.

The unresolved region therefore decomposes into two parts of quite diferent character. Write

$$
I ( O _ { n } ) \ = \ I ^ { \mathrm { l o c } } ( O _ { n } ) \ \dot { \cup } \ I ^ { \mathrm { i n c } } ( O _ { n } ) ,\tag{26}
$$

where

$$
I ^ { \mathrm { l o c } } ( O _ { n } ) = \{ x \in I ( O _ { n } ) \mid { \mathcal { F } } _ { n } \vdash x { \mathrm { ~ o r ~ } } { \mathcal { F } } _ { n } \vdash \lnot x \} ,\tag{27}
$$

$$
I ^ { \mathrm { i n c } } ( O _ { n } ) = \{ x \in I ( O _ { n } ) \mid { \mathcal F } _ { n } { \mathcal K } x { \mathrm { ~ a n d ~ } } { \mathcal F } _ { n } { \mathcal K } \neg x \} .\tag{28}
$$

The first part is unresolved because the system has not yet done the work; the second is because the work cannot be done within $\mathcal { F } _ { n }$ at all. Local transformation within a regime (§4.3) reduces $I ^ { \mathrm { l o c } }$ and leaves $I ^ { \mathrm { i n c } }$ untouched.

That $I ^ { \mathrm { i n c } }$ is inhabited reflects the essence of incompleteness. According to the G¨odel–Rosser theorem, any ${ \mathcal { F } } _ { n }$ that meets the above hypotheses allows for a sentence $G _ { n } \in L _ { n }$ with

$$
{ \mathcal { F } } _ { n } \gamma \varepsilon G _ { n } \qquad \mathrm { a n d } \qquad { \mathcal { F } } _ { n } \gamma \varepsilon \neg G _ { n } ,\tag{29}
$$

whence, by Assumption 1 and (28),

$$
G _ { n } \in I ^ { \mathrm { i n c } } ( O _ { n } ) , \qquad \mathrm { s o } \qquad I ^ { \mathrm { i n c } } ( O _ { n } ) \neq \emptyset .\tag{30}
$$

This is the first recursive pressure of the theory. An unresolved region, when non-empty, is not particularly remarkable on its own; any system that has not completed its reasoning will have one. However, what (30) establishes is stronger: i.e., part of that region is immune to further reasoning within the current system, regardless of the extent of reasoning performed. A system capable of representing its own reasoning thus faces a boundary that cannot be eliminated through continued inference at the same level. Instead, addressing this boundary requires a change in the inferential regime.

Two qualifications limit what (30) delivers on its own. Taken purely extensionally, it asserts only that $I ^ { \mathrm { i n c } } ( O _ { n } )$ is non-empty; it does not by itself guarantee that this unresolved region is large, structured, or productive, nor can a non-omniscient system without an external oracle inspect an arbitrary $x \in I ( O _ { n } )$ in advance to tell whether $x \in I ^ { \mathrm { l o c } } ( O _ { n } )$ or $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ prior to attempting a derivation. The condition stating that unresolved structure allows for at least two admissible continuations is stated for general architectures in (50) and does not follow from bare incompleteness alone; §4.2 shows how, in any architecture that reflectively reifies its own failed proof searches, a multi-branched semantic surplus is constructively generated from the operational trace of attempting to resolve $I ^ { \mathrm { i n c } } ( O _ { n } )$ . Additionally, $G _ { n }$ serves as a witness rather than a description: $I ^ { \mathrm { i n c } } ( O _ { n } )$ will generally include numerous sentences of much greater cognitive interest than a self-referential construction. It is these more interesting sentences that motivate the extensions discussed in §4.2.

## 4.2 Reflective Extension and Semantic Surplus

Resolving a proposition in $I ^ { \mathrm { i n c } } ( O _ { n } )$ requires, by (28), an extension of the inferential resources available to the system. No amount of local transformation reaches it. We therefore consider the construction of successor systems.

Admissible extensions. Let $\mathcal { E } ( \mathcal { F } _ { n } )$ denote the set of admissible successor systems constructible from ${ \mathcal { F } } _ { n }$ under the transformation rules available to the architecture. A successor

$$
\mathcal { F } _ { n + 1 } = ( L _ { n + 1 } , \Gamma _ { n + 1 } , \vdash _ { n + 1 } ) \ \in \ \mathcal { E } ( \mathcal { F } _ { n } )\tag{31}
$$

is admissible when it satisfies two conditions. First, regularity: $\mathcal { F } _ { n + 1 }$ is itself efectively axiomatized, consistent, and suficiently expressive in the sense of §4.1. Second, reflective adequacy: the construction supplies a translation

$$
\tau _ { n } \colon L _ { n } \longrightarrow L _ { n + 1 }\tag{32}
$$

under which the successor can represent the syntax and proof relation of its predecessor, so that statements about ${ \mathcal { F } } _ { n }$ are expressible in $L _ { n + 1 }$ . We write $\mathscr { R } _ { n } = ( \mathcal { F } _ { n + 1 } , \tau _ { n } )$ for the resulting recursive extension, carrying both the successor system and the translation into it; §4.5 develops its properties.

Reflective adequacy is less stringent than inclusion. We do not require that

$$
{ \mathcal { F } } _ { n } \subseteq { \mathcal { F } } _ { n + 1 } ,\tag{33}
$$

nor must $\tau _ { n }$ preserve the interpretation of what it translates. A successor may clarify an ambiguity present in its predecessor, maintain it, reinterpret it, or introduce new ambiguities of its own. However, it must not lose the ability to discuss its origins: an architecture that cannot represent

its own history cannot reflect on it. §4.5 and §6.4 both rely on this principle. As a result, recursive intelligence is not defined by guaranteed convergence toward a single, complete representation but rather by the capacity to construct and reason about successive representational systems.

Semantic surplus. The set $\mathcal { E } ( \mathcal { F } _ { n } )$ is generally not a singleton. Where it is not, the architecture faces several admissible ways of continuing, none of which ${ \mathcal { F } } _ { n }$ itself determines. We call this underdetermination the semantic surplus of the state and write

$$
\mathrm { S u r p } ( { \cal O } _ { n } ) = \mathcal { E } ( \mathcal { F } _ { n } ) , \qquad \mathrm { w i t h ~ s u r p l u s ~ } p r e s e n t ~ \mathrm { w h e n ~ } \big | \mathrm { S u r p } ( { \cal O } _ { n } ) \big | \ge 2 .\tag{34}
$$

Semantic surplus is therefore neither randomness nor an arbitrarily large set of alternative answers. It is the space of admissible transformations left open relative to the current formal system. Its cardinality measures how many distinct continuations the architecture has available, which is the quantity that the collective analysis of §9.1 aggregates across agents.

Within this space, we distinguish the successors that make a diference to what is decidable. Let

$$
\mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n } ) = \{ { \mathcal F } ^ { \prime } \in \mathrm { S u r p } ( O _ { n } ) ~  ~ \exists x \in I ^ { \mathrm { i n c } } ( O _ { n } ) : { \mathcal F } ^ { \prime } \vdash \tau _ { n } ( x ) ~ { \mathrm { o r } } ~ { \mathcal F } ^ { \prime } \vdash \neg \tau _ { n } ( x ) ~ \} ~ .\tag{35}
$$

denote the decisive surplus: those admissible successors that settle some proposition their predecessor could not. The inclusion $\operatorname { S u r p } ^ { \mathrm { d e c } } ( O _ { n } ) \subseteq \operatorname { S u r p } ( O _ { n } )$ may be strict, since a successor may reorganize a representation without deciding anything new. Both notions are needed. Generative capacity depends on the cardinality of the full surplus, whereas the recursive pressure of §4.1 is relieved only by its decisive part.

This gives the third law of §2 a formal referent. Surplus is a resource in a precise sense: it is what the meta-level operates on. Where $| \mathrm { S u r p } ( O _ { n } ) | \leq 1$ , there is nothing to evaluate, compare, or select, and the architecture has no move available beyond local transformation—which by (30) cannot reach $I ^ { \mathrm { i n c } }$

Proof-attempt traces and the conversion of form into content. Why should encountering an undecidable frontier generate a plurality of admissible extensions in ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ rather than a mere halting boundary? And why does the derivation attempt itself perform non-trivial mathematical work beyond simply adjoining x or ¬x by fiat? Considered purely extensionally, (30) establishes only the negative fact that some $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ cannot be proved or refuted in ${ \mathcal { F } } _ { n }$ , and Lindenbaum’s lemma already implies that adjoining x or ¬x directly to $\Gamma _ { n }$ yields two consistent extensions. Yet such a direct adjunction treats x as an unanalyzed atom. In an actual cognitive system, by contrast, two structural features make the proof search indispensable. First, because theoremhood in ${ \mathcal { F } } _ { n }$ is only recursively enumerable, the system is not handed the partition $I ( O _ { n } ) = I ^ { \mathrm { l o c } } ( O _ { n } ) \dot { \cup } I ^ { \mathrm { i n c } } ( O _ { n } )$ by an oracle and cannot know prior to attempting a derivation whether an unresolved sentence $x \in I ( O _ { n } )$ lies in $I ^ { \mathrm { l o c } } ( O _ { n } )$ or in $I ^ { \mathrm { i n c } } ( O _ { n } )$ . Second, attempting to prove or refute x decomposes the surface question into intermediate subgoals and conditional dependencies, exposing the underlying lemmas on which x actually turns.

To separate what holds mathematically of the unresolved frontier from what a finite proof search can operationally compute, we formalize the structure of a non-trivial proof attempt. For any unresolved sentence $x \in I ( O _ { n } )$ and finite search budget $t \geq 1$ , let the derivation trace generated by the object-level proof searcher up to step t be

$$
\pi _ { n , t } ( x ) \ : = \ : \big ( { \mathcal D } _ { n , t } ( x ) , { \mathcal B } _ { n , t } ( x ) \big ) ,\tag{36}
$$

where $\mathcal { D } _ { n , t } ( x )$ is the finite directed acyclic graph of inference steps explored between $\Gamma _ { n }$ and $\{ x , \lnot x \}$ within t steps, and

$$
B _ { n , t } ( x ) \ = \ \{ b _ { 1 } , \ldots , b _ { m } \} \ \subseteq \ I ( O _ { n } ) \setminus \{ x , \lnot x \}\tag{37}
$$

is the finite blocking frontier of proper intermediate subgoals (excluding the trivial self-reduction $b = x \ \mathrm { o r } \ b = \lnot x )$ at which $\mathcal { D } _ { n , t } ( x )$ stalls at step t. We say that $\pi _ { n , t } ( \boldsymbol { x } )$ is a proper reduction trace if the verified inference links in $\mathcal { D } _ { n , t } ( x )$ establish in $\mathcal { F } _ { n }$ a truth-functional reduction of x to $B _ { n , t } ( x )$ , a Boolean formula $\Psi _ { \pi } ( u _ { 1 } , \ldots , u _ { m } )$ synthesized from the conditional branches of $\mathcal { D } _ { n , t } ( x )$ such that

$$
\Gamma _ { n } \vdash _ { n } ( x  \Psi _ { \pi } ( b _ { 1 } , \ldots , b _ { m } ) ) .\tag{38}
$$

Condition (38) is satisfied by any standard goal-reduction calculus (such as backward sequent decomposition, resolution refutation, or horn-clause unfolding) after $t \geq 1$ non-trivial expansion steps: the interior nodes of $\mathcal { D } _ { n , t } ( x )$ record theorems of ${ \mathcal { F } } _ { n }$ linking x to the frontier leaves $b _ { 1 } , \dotsc , b _ { m } \in$ $I ( O _ { n } ) \setminus \{ x , \neg x \}$ . When $x \in I ^ { \log } ( O _ { n } )$ , a suficiently large budget t discharges enough leaves $b _ { i }$ in $\Gamma _ { n }$ to evaluate $\Psi _ { \pi }$ to a theorem or refutation, closing the trace $( B _ { n , t } ( x ) = \varnothing )$ . When $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ , no finite t can close the trace.

Within the object regime ${ \mathcal { F } } _ { n } ,$ an unclosed trace $\pi _ { n , t } ( \boldsymbol { x } )$ yields no unconditional theorem and leaves $O _ { n } ( x )$ undefined. Hegel’s analysis of dialectical method in the Phenomenology of Spirit and The Science of Logic identifies why treating such an operational failure as a mere nullity mistakes the nature of cognition. A cognitive process cannot be reduced to its bare terminal theorem in isolation from the derivation attempted to reach it, “[f]or the real issue is not exhausted by stating it as an aim, but by carrying it out, nor is the result the actual whole, but rather the result together with the process through which it came about” [Hegel, 1977, p. 2]. When a mode of knowing stalls at its internal limit, its failure is not an empty nothingness, “[f]or it is only when it is taken as the result of that from which it emerges, that it is, in fact, the true result; in that case it is itself a determinate nothingness, one which has a content” [Hegel, 1977, p. 51]. In The Science of Logic, Hegel formulates this principle as the general mechanism of conceptual progression: “what is selfcontradictory does not resolve itself into a nullity, into abstract nothingness, but essentially only into the negation of its particular content; or that such a negation is not just negation, but is the negation of the determined fact which is resolved, and is therefore determinate negation; that in the result there is therefore contained in essence that from which the result derives” [Hegel, 2010, p. 33]. Because “method is the consciousness of the form of the inner self-movement of the content of logic” [Hegel, 2010, p. 33] and “content is at the same time the negative reflection of the form determinations into themselves” [Hegel, 2010, p. 396], reflecting on the failed operation converts its form into a new object of thought: “the new object shows itself to have come about through a reversal of consciousness itself” [Hegel, 1977, p. 55], and through that reflection “the subject matter has received a determinateness for itself — and this determinateness is a content, for the negativity that has withdrawn into simplicity is the sublated form” [Hegel, 2010, p. 748].

Reflective adequacy (32) gives this Hegelian conversion of operational form into semantic content a direct formal realization. Under $\tau _ { n } \colon L _ { n } \to L _ { n + 1 }$ , the unclosed derivation trace $\pi _ { n , t } ( \boldsymbol { x } )$ —which at level n was only the operational form of an incomplete proof search—is reified in $L _ { n + 1 }$ as a syntactic object $\Gamma _ { \pi _ { n , t } ( x ) } \big ]$ accompanied by a finite reflective diagram $\Phi _ { \mathrm { t r } } \mathopen { } \mathclose \bgroup \left( \pi _ { n , t } \mathopen { } \mathclose \bgroup \left( x \aftergroup \egroup \right) \aftergroup \egroup \right) \subseteq L _ { n + 1 }$ encoding the verified inference links of $\mathcal { D } _ { n , t } ( x )$ and the reduction equivalence $\tau _ { n } ( x )  \Psi _ { \pi } { \bigl ( } \tau _ { n } ( b _ { 1 } ) , \ldots , \tau _ { n } ( b _ { m } ) { \bigr ) }$ What was a failed operational process in $\mathcal { F } _ { n }$ thereby becomes determinate propositional content in $L _ { n + 1 }$ . For any blocking premise $b \in B _ { n , t } ( x )$ isolated by the trace, reification induces a canonical pair of candidate trace-extensions,

$$
\widehat { \mathcal { F } } _ { n + 1 } ^ { + } ( b ) \ = \ \big ( L _ { n + 1 } , \tau _ { n } ( \Gamma _ { n } ) \cup \Phi _ { \mathrm { t r } } \big ( \pi _ { n , t } ( x ) \big ) \cup \{ \tau _ { n } ( b ) \} , \vdash _ { n + 1 } \big ) ,\tag{39}
$$

$$
\widehat { \mathcal { F } } _ { n + 1 } ^ { - } ( b ) \ = \ \big ( L _ { n + 1 } , \tau _ { n } ( \Gamma _ { n } ) \cup \Phi _ { \mathrm { t r } } \big ( \pi _ { n , t } ( x ) \big ) \cup \{ \neg \tau _ { n } ( b ) \} , \vdash _ { n + 1 } \big ) ,
$$

which preserve the predecessor’s validated inferential structure while resolving the specific intermediate blockage $b \neq x$ in opposite directions.

We now establish two distinct results corresponding to the logical and operational levels of this construction: first, an existence theorem showing that any proper reduction trace of an undecidable sentence $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ must expose at least one genuinely undecidable intermediate premise $b \neq x$ that generates non-trivial branching in ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ ; and second, an operational proposition characterizing what happens when a non-omniscient system applies (39) to finite open traces without knowing in advance which leaves belong to $I ^ { \mathrm { i n c } } ( O _ { n } )$

Proposition 1 (Existence of Non-Trivial Trace-Reduced Decisive Extensions). Let $\mathcal { F } _ { n } = ( L _ { n } , \Gamma _ { n } , \vdash _ { n }$ ) be a regular formal system and let $O _ { n }$ be a cognitive state satisfying Assumption 1. Suppose the admissible transformations $\mathcal { E } ( \mathcal { F } _ { n } )$ include the consistent members of the trace-objectifying family (39) for proper reduction traces $\pi _ { n , t } ( \boldsymbol { x } )$ satisfying $( 3 7 )  { - } ( 3 8 )$ whose reflective diagram $\Phi _ { \mathrm { t r } } { \big ( } \pi _ { n , t } ( x ) { \big ) }$ is conservative over $\tau _ { n } ( \Gamma _ { n } )$ . Then for every $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ and every proper reduction trace $\pi _ { n , t } ( \boldsymbol { x } )$

1. (Non-trivial undecidable frontier.) The blocking frontier $B _ { n , t } ( x ) \subseteq I ( O _ { n } ) \setminus \{ x , \neg x \}$ contains at least one non-trivial undecidable premise,

$$
B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } ) \ \neq \ \varnothing .\tag{40}
$$

2. (Branching decisive surplus.) For every $b \in \mathcal { B } _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ , both extensions ${ \mathcal { F } } _ { n + 1 } ^ { + } ( b ) : =$ ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { + } ( b )$ and $\mathcal { F } _ { n + 1 } ^ { - } ( b ) : = \widehat { \mathcal { F } } _ { n + 1 } ^ { - } ( b )$ in (39) are regular, reflectively adequate, and decisive over $b \neq x$ , so that

$$
\{ \mathcal { F } _ { n + 1 } ^ { + } ( b ) , \ \mathcal { F } _ { n + 1 } ^ { - } ( b ) \} \ \subseteq \ \mathrm { S u r p } ^ { \deg } ( O _ { n } ) \qquad a n d \qquad | \mathrm { S u r p } ( O _ { n } ) | \ \geq \ \left| \mathrm { S u r p } ^ { \deg } ( O _ { n } ) \right| \ \geq \ 2 .\tag{41}
$$

Whenever $k \geq 1$ blocking premises $b _ { 1 } , \dots , b _ { k } \in \mathcal { B } _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ are mutually independent over $\Gamma _ { n }$ , simultaneous resolution of $B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ generates $2 ^ { k } \geq 2$ distinct decisive extensions in ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ , each of which also decides $\tau _ { n } ( x )$ via (38) once the remaining leaves in $B _ { n , t } ( x ) \cap I ^ { \mathrm { l o c } } ( O _ { n } )$ are resolved by local deduction.

Proof. (1) For the first part, suppose toward a contradiction that $B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } ) = \emptyset$ . Since $\mathscr { B } _ { n , t } ( x ) \subseteq I ( O _ { n } ) = I ^ { \mathrm { l o c } } ( O _ { n } ) \ \dot { \cup } \ I ^ { \mathrm { i n c } } ( O _ { n } )$ by (26), every $b _ { i } \in B _ { n , t } ( x )$ must lie in $I ^ { \mathrm { l o c } } ( O _ { n } )$ . By (27), each $b _ { i }$ is either provable or refutable from $\Gamma _ { n } ( \mathrm { i . e . , } \Gamma _ { n } \vdash _ { n } b _ { i }$ or $\Gamma _ { n } \vdash _ { n } \lnot b _ { i } )$ . Substituting these provable literals into the truth-functional equivalence $\Gamma _ { n } \vdash _ { n } ( x  \Psi _ { \pi } ( b _ { 1 } , \ldots , b _ { m } ) )$ from (38) fixes the truth value of $\Psi _ { \pi }$ in ${ \mathcal { F } } _ { n }$ , yielding either $\Gamma _ { n } \vdash _ { n } x$ or $\Gamma _ { n } \vdash _ { n } \lnot x .$ which contradicts $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ Thus, (40) holds. Furthermore, because x, ¬x $\not \in \ B _ { n , t } ( x )$ by (37), every $b \in B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ is distinct from x and $\neg x$

(2) For the second part, fix $b \in B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ . Because $\mathcal { F } _ { n } \vdash b$ and $\mathcal { F } _ { n } \not \vdash \neg b$ , and since $\Phi _ { \mathrm { t r } } \big ( \pi _ { n , t } ( x ) \big )$ is conservative over $\tau _ { n } ( \Gamma _ { n } )$ , neither $\tau _ { n } ( b )$ nor its negation is refutable from $\tau _ { n } ( \dot { \Gamma } _ { n } ) \cup \Phi _ { \mathrm { t r } } ( \pi _ { n , t } ( x ) )$ . Since $\tau _ { n } ( \Gamma _ { n } )$ is recursively enumerable and $\Phi _ { \mathrm { t r } } ( \pi _ { n , t } ( x ) ) \cup \{ \pm \tau _ { n } ( b ) \}$ is finite, both ${ \mathcal { F } } _ { n + 1 } ^ { + } ( b )$ and ${ \mathcal { F } } _ { n + 1 } ^ { - } ( b )$ are consistent, efectively axiomatized, and suficiently expressive via $\tau _ { n }$ (hence regular and reflectively adequate members of $\mathcal { E } ( \mathcal { F } _ { n } ) )$ Because $\mathcal { F } _ { n + 1 } ^ { + } ( b ) \vdash _ { n + 1 } \ \tau _ { n } ( b )$ and $\mathcal { F } _ { n + 1 } ^ { - } ( b ) \vdash _ { n + 1 } \neg \tau _ { n } ( b )$ with $b \ \in \ I ^ { \mathrm { i n c } } ( O _ { n } )$ , both extensions belong to ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ by (35), establishing (41).

(3) Finally, if $b _ { 1 } , \dots , b _ { k } \in \mathcal { B } _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ are mutually independent over $\Gamma _ { n }$ , all $2 ^ { k }$ sign assignments $\{ \pm \tau _ { n } ( b _ { 1 } ) , \ldots , \pm \tau _ { n } ( b _ { k } ) \}$ are consistent with $\tau _ { n } ( \Gamma _ { n } ) \cup \Phi _ { \mathrm { t r } } \big ( \pi _ { n , t } ( x ) \big )$ and define $2 ^ { k }$ distinct decisive successors. Since every remaining local element $b _ { j } \in B _ { n , t } ( x ) \cap I ^ { \mathrm { l o c } } ( O _ { n } )$ is already decided by $\Gamma _ { n }$ , the translated equivalence $\tau _ { n } ( x ) \  \ \Psi _ { \pi } \bigl ( \tau _ { n } ( b _ { 1 } ) , \ldots , \tau _ { n } ( b _ { m } ) \bigr )$ in $\Phi _ { \mathrm { t r } } { \big ( } \pi _ { n , t } ( x ) { \big ) }$ uniquely decides $\tau _ { n } ( x )$ in each of the $2 ^ { k }$ extensions.

Proposition 1 establishes the existence of non-trivial decisive extensions, but its statement quantifies over $b \in \mathcal { B } _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } )$ . Because $I ^ { \mathrm { i n c } } ( O _ { n } )$ is not recursively decidable from finite derivation attempts, an actual agent at step t sees only the unclosed blocking frontier $B _ { n , t } ( x ) \subseteq$ $I ( O _ { n } ) \setminus \{ x , \neg x \}$ as a whole, without knowing which leaves $b \in B _ { n , t } ( x )$ are genuinely undecidable $( b \in I ^ { \operatorname { i n c } } ( O _ { n } ) )$ and which have merely not yet been proved or refuted within t steps $( b \in I ^ { \mathrm { l o c } } ( O _ { n } ) )$ The next proposition characterizes the operational behavior of trace-guided candidate generation under this epistemic constraint.

Proposition 2 (Trace-Guided Candidate Generation and Fallibility). Let ${ \mathcal { F } } _ { n }$ be a regular formal system, let $O _ { n }$ satisfy Assumption 1, and let $\pi _ { n , t } ( x ) \ : = \ : \big ( \mathcal { D } _ { n , t } ( x ) , \mathcal { B } _ { n , t } ( x ) \big )$ be an unclosed proper reduction trace at finite search step $t \geq 1$ . When the architecture applies trace reification (39) to an arbitrary open premise $b \in B _ { n , t } ( x )$ whose status in $I ^ { \mathrm { l o c } } ( O _ { n } ) \dot { \cup } I ^ { \mathrm { i n c } } ( O _ { n } )$ is unknown at step t, the resulting candidate pair $( \widehat { \mathcal { F } } _ { n + 1 } ^ { + } ( b ) , \widehat { \mathcal { F } } _ { n + 1 } ^ { - } ( b ) )$ satisfies the following exact trichotomy:

1. (Latent provability: $b \ \in \ I ^ { \mathrm { l o c } } ( O _ { n } )$ with $\mathcal { F } _ { n } \ \vdash \ b . )$ The negative candidate ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { - } ( b )$ is inconsistent, whereas the positive candidate ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { + } ( b )$ is a conservative deductive shortcut over $\tau _ { n } ( \Gamma _ { n } ) \cup \Phi _ { \mathrm { t r } } \big ( \pi _ { n , t } ( x ) \big )$ that decides no sentence of $I ^ { \mathrm { i n c } } ( O _ { n } )$ via b.

2. (Latent refutability: $b \in I ^ { \mathrm { l o c } } ( O _ { n } )$ with $\mathcal { F } _ { n } \vdash \neg b . )$ Symmetrically, ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { + } ( b )$ is inconsistent, whereas ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { - } ( b )$ is a conservative deductive shortcut.

3. (Genuine undecidability: $b \in I ^ { \mathrm { i n c } } ( O _ { n } ) . )$ Both ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { + } ( b )$ and ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { - } ( b )$ are consistent, regular, and belong to ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ by Proposition $\displaystyle 1 ( 2 )$

Moreover, if $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ , then generating the candidate family $\{ \widehat { \mathcal { F } } _ { n + 1 } ^ { \pm } ( b ) \ | \ b \in \mathcal { B } _ { n , t } ( x ) \}$ across the finite frontier $B _ { n , t } ( x )$ is guaranteed to include at least one genuine decisive pair from Case $( 3 )$ even though no efective procedure within ${ \mathcal { F } } _ { n }$ can decide in advance which $b \in B _ { n , t } ( x )$ falls under Case (3) or certify the global consistency of ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { \pm } ( b )$

Proof. Since $b \in B _ { n , t } ( x ) \subseteq I ( O _ { n } )$ , the decomposition (26) partitions b exhaustively into Cases (1)– (3).

In Case (1), $\Gamma _ { n } \vdash _ { n }$ b implies $\tau _ { n } ( \Gamma _ { n } ) \vdash _ { n + 1 } \tau _ { n } ( b )$ by reflective adequacy (32); adjoining $\neg \tau _ { n } ( b )$ in ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { - } ( b )$ therefore yields an inconsistent theory, while adjoining $\tau _ { n } ( b )$ in ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { + } ( b )$ adds a theorem already entailed by $\tau _ { n } ( \Gamma _ { n } )$

Case (2) holds by replacing b with ¬b.

Case (3) is Proposition 1.2. When $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ , Proposition 1.1 ensures $B _ { n , t } ( x ) \cap I ^ { \mathrm { i n c } } ( O _ { n } ) \neq \emptyset$ so at least one $b \in B _ { n , t } ( x )$ falls under Case (3).

Finally, deciding within ${ \mathcal { F } } _ { n }$ whether an arbitrary $b \in B _ { n , t } ( x )$ belongs to $I ^ { \mathrm { l o c } } ( O _ { n } )$ or $I ^ { \mathrm { i n c } } ( O _ { n } )$ would decide theoremhood in ${ \mathcal { F } } _ { n }$ , which is undecidable for any regular system by Church’s theorem, and certifying the consistency of a consistent extension ${ \widehat { \mathcal { F } } } _ { n + 1 } ^ { \pm } ( b )$ within ${ \mathcal { F } } _ { n }$ is excluded by G¨odel’s second incompleteness theorem. □

Propositions 1 and 2 jointly clarify why the transition from incompleteness to semantic surplus is neither automatic in a non-reflective calculus nor reducible to an infallible algorithm. Proposition 1 guarantees that whenever a proof search reduces an undecidable question $x \in I ^ { \mathrm { i n c } } ( O _ { n } )$ to a nontrivial blocking frontier $B _ { n , t } ( x )$ , genuinely branching decisive extensions exist in ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ that act on the intermediate lemmas $b \neq x$ uncovered by the trace rather than merely stipulating x or ¬x by fiat. Proposition 2 shows why turning those open traces into successor systems is an inherently speculative, fallible operation: because a finite search cannot distinguish $I ^ { \mathrm { i n c } } ( O _ { n } )$ from $I ^ { \mathrm { l o c } } ( O _ { n } )$ by inspection, trace-guided candidate generation inevitably produces a mixture of genuine decisive extensions $\left( \mathrm { C a s e } ^ { \sim } 3 \right)$ , conservative shortcuts, and latent inconsistencies (Cases˜1–2) that can only be weeded out through subsequent inference, relative consistency checks, and empirica or teleological evaluation by $\mathcal { M } _ { n }$ . This is the exact mathematical reason why an open-ended architecture requires both polycontexturality (§2) and distributed collective exploration (§6.3): competing trace-extensions ${ \widehat F } _ { n + 1 } ^ { \pm } ( b )$ must be pursued across parallel contextures or agents so that the collapse of an inconsistent branch in $\mathrm { C a s e s } ^ { \sim } ( 1 ) { - } ( 2 )$ refutes only that candidate contexture while preserving the viable decisive branches of Case˜(3). As Hegel writes of determinate negation, “[b]ecause the result, the negation, is a determinate negation, it has a content. It is a new concept but one higher and richer than the preceding — richer because it negates or opposes the preceding and therefore contains it, and it contains even more than that, for it is the unity of itself and its opposite” [Hegel, 2010, p. 33]. We retain Assumptions 2 and 3 as general architectural hypotheses below so that our downstream results apply to any admissible pair $( \mathcal { E } , \mathcal { M } _ { n } )$ , while noting that trace-objectifying architectures guarantee the existence of such extensions by Proposition 1 and generate them operationally via Proposition 2.

Assumption 2 (Decisive Extension). At every level n at which recursive extension is triggered, the decisive surplus is non-empty,

$$
{ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } ) \neq \emptyset ,\tag{42}
$$

and the meta-level selection operator restricts its choices to decisive successors, ${ \mathcal { M } } _ { n } { \big ( } { \mathrm { S u r p } } ( O _ { n } ) { \big ) } \subseteq$ ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$

Meta-level selection. A meta-level operator

$$
\mathcal { M } _ { n } \colon 2 ^ { \mathcal { E } ( \mathcal { F } _ { n } ) } \ \longrightarrow \ 2 ^ { \mathcal { E } ( \mathcal { F } _ { n } ) } \setminus \{ \emptyset \} , \qquad \mathcal { M } _ { n } ( X ) \subseteq X ,\tag{43}
$$

narrows the admissible successors to those that the architecture is prepared to adopt. In doing so, it may compare candidates, evaluate their consequences, or prune those that fail its structura criteria; what it returns is not a single successor but the set that survives. As with (14), selection is relational rather than functional: the architecture determines which continuations are available, but not which is taken. The recursive transition is then

$$
{ \mathcal { F } } _ { n } \ { \overset { \varepsilon } { \longrightarrow } } \ \operatorname { S u r p } ( O _ { n } ) \ { \overset { { \mathcal { M } } _ { n } } { \longrightarrow } } \ { \mathcal { F } } _ { n + 1 } \in { \mathcal { M } } _ { n } { \big ( } \operatorname { S u r p } ( O _ { n } ) { \big ) } .\tag{44}
$$

The essential point is that the meta-level does not choose among answers produced within a fixed system. It operates on the space of possible successor systems themselves. Because the architecture is homoiconic (§2), the agent’s teleological partition $\Phi _ { n } = ( \Phi _ { n } ^ { + } , \Phi _ { n } ^ { - } , \Phi _ { n } ^ { \circ } )$ of (2) is itself represented within $L _ { n } .$ , making $\mathcal { M } _ { n }$ the formal realization of the teleological transformation operator $\tau$ of (14). Under Assumption $2 , { \mathcal { M } } _ { n }$ selects a decisive successor ${ \mathcal { F } } _ { n + 1 } \in { \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ whose extended vocabulary $L _ { n + 1 } \setminus \tau _ { n } ( L _ { n } )$ can articulate a successor purpose $\phi _ { n + 1 }$ with a non-contained semantic domain $D _ { \phi _ { n + 1 } } \nsubseteq D _ { H _ { n } }$ in (15), without the Boolean closure constraint (13) that bounded T over a fixed language.

Where the teleological model of §3.4 expresses novelty set-theoretically via $D _ { \phi _ { n + 1 } } \nsubseteq D _ { H _ { n } }$ in (15), the formal model requires an algorithmic measure of the information introduced when $\mathcal { M } _ { n }$ commits to $\mathcal { F } _ { n + 1 }$ over competing branches in ${ \mathrm { S u r p } } ( O _ { n } )$ . That stronger requirement is stated separately as Assumption 5 in §5.2. The two conditions play the same architectural role: each ensures that successive extensions introduce genuinely new discriminations rather than relabelled repetitions of prior structure.

The decomposition $I ( O _ { n } ) = I ^ { \mathrm { l o c } } ( O _ { n } ) \ \dot { \cup } \ I ^ { \mathrm { i n c } } ( O _ { n } )$ together with (39) and (44) marks the exact boundary between deductively certified self-improvement and co-recursive self-extension. In a G¨odel machine [Schmidhuber, 2006], the proof searcher cannot know in advance which candidate code transformations have provable utility; it must run an object-level search that generates a derivation trace $\pi _ { 0 , t } ( x )$ , but it is licensed to execute a rewrite only when that trace closes within $I ^ { \mathrm { l o c } } ( O _ { 0 } )$ into a theorem derived from the initial axiom scheme $\mathcal { F } _ { 0 }$ . In his discussion of the limitations of G¨odel machines, Schmidhuber notes that “the fundamental limitations are closely related to those first identified by G¨odel’s celebrated paper on self-referential formulae” [Schmidhuber, 2006, p. 6], because any formal system encompassing arithmetic “is either flawed or allows for unprovable but true statements. Hence, even a G¨odel machine with unlimited computational resources must ignore those self-improvements whose efectiveness it cannot prove, $\mathrm { e . g . }$ , for lack of suficiently powerful axioms in $\mathrm { A } ^ { \prime \prime }$ [Schmidhuber, 2006, p. 6]. Confronted with a candidate modification whose utility lies in $I ^ { \mathrm { i n c } } ( O _ { 0 } )$ , the machine “cannot profit from a hypothetical useful self-improvement whose utility is undecidable, and will therefore simply ignore $\mathrm { i t } ^ { \dag }$ [Schmidhuber, 2006, p. 24]—discarding the unclosed derivation trace $\pi _ { 0 , t } ( x )$ because it yields no object-level theorem. In the present framework, that limitation is inverted. By Propositions 1 and 2, an unclosed proper reduction trace on $I ^ { \mathrm { i n c } } ( O _ { n } )$ is not an empty failure at which self-modification halts, but a determinate negation whose operational form $\pi _ { n , t } ( \boldsymbol { x } )$ is reflectively reified via (39) into the semantic content of candidate extensions [Hegel, 1977, pp. 2, 51], at least one pair of which is guaranteed to lie in ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ Because no finite search within ${ \mathcal { F } } _ { n }$ can decide in advance which open premise $b \in B _ { n , t } ( x )$ lies in $I ^ { \mathrm { i n c } } ( O _ { n } )$ (Proposition 2) and no element of ${ \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } )$ can be certified as superior by a proof within ${ \mathcal { F } } _ { n }$ without collapsing $\mathcal { F } _ { n + 1 }$ back into the deductive closure of $\Gamma _ { n } .$ , the operator $\mathcal { M } _ { n }$ cannot be reduced to theorem proving in ${ \mathcal { F } } _ { n } ;$ ; it mediates across discontextural regimes [G¨unther, 1979, p. 15] by filtering out inconsistent candidate branches and making a non-deductive commitment to a genuinely new formal contexture.

Successor incompleteness. Because admissibility requires regularity, §4.1 applies verbatim to $\mathcal { F } _ { n + 1 }$ . There is therefore a sentence $G _ { n + 1 } \in L _ { n + 1 }$ with

$$
{ \mathcal { F } } _ { n + 1 } \nvdash G _ { n + 1 } \qquad { \mathrm { a n d } } \qquad { \mathcal { F } } _ { n + 1 } \nvdash { \neg G } _ { n + 1 } ,\tag{45}
$$

and hence $I ^ { \mathrm { i n c } } ( O _ { n + 1 } ) \neq \emptyset$

Resolving an ambiguity at one level does not eliminate incompleteness; it just relocates it. Schematically, the architecture moves from an undecidable sentence through a successor system to a fresh undecidable sentence, and this is the fundamental recursive structure of the theory: a suficiently expressive system can construct a stronger system that transcends some of its limitations, which in turn is subject to its own limitations. The consequences of iterating this construction are developed in §4.6.

The central question is therefore no longer whether a single formal system can become complete. It is whether a cognitive architecture can sustain the construction, preservation, evaluation, and selection of successor systems without reducing the process to a fixed formal space—the question §5answers negatively for fixed architectures.

## 4.3 Temporal Recursive Dynamics

The recursive structure introduced above unfolds in time. Let $\mathcal { F } _ { n , t }$ denote the inferential system available at recursive level n and temporal step t, and $O _ { n , t }$ the corresponding cognitive state. Two

kinds of transition must be distinguished.

A local transformation operates within the current formal system,

$$
\mathcal { F } _ { n , t } \longrightarrow \mathcal { F } _ { n , t + 1 } , \qquad L _ { n , t + 1 } = L _ { n , t } , \quad \Gamma _ { n , t + 1 } = \Gamma _ { n , t } ,\tag{46}
$$

changing only the cognitive state. By (25), such a transition can only extend dom $( O _ { n , t } )$ within the decidable fragment, so

$$
I ^ { \mathrm { l o c } } ( O _ { n , t + 1 } ) \subseteq I ^ { \mathrm { l o c } } ( O _ { n , t } ) , \qquad I ^ { \mathrm { i n c } } ( O _ { n , t + 1 } ) = I ^ { \mathrm { i n c } } ( O _ { n , t } ) .\tag{47}
$$

Local work consumes the first part of the unresolved region and cannot touch the second. This is the formal content of the distinction drawn at (26).

A recursive extension constructs a new inferential system,

$$
\mathcal { F } _ { n , t } \longrightarrow \mathcal { F } _ { n + 1 , 0 } ,\tag{48}
$$

via (44). Here, L and Γ themselves change, and (47) no longer applies: the successor’s unresolved region is a region of a diferent language, related to its predecessor’s only through $\tau _ { n }$

The temporal trajectory of a system is therefore

$$
\mathrm { T r a j } ( \mathcal { F } _ { 0 } ) = ( \mathcal { F } _ { 0 , 0 } , \mathcal { F } _ { 0 , 1 } , \ldots , \mathcal { F } _ { 1 , 0 } , \mathcal { F } _ { 1 , 1 } , \ldots , \mathcal { F } _ { 2 , 0 } , \ldots ) ,\tag{49}
$$

alternating runs of local transformation with extension steps.

Generative richness. For general transformation repertoires $\mathcal { E } ( \mathcal { F } _ { n , t } )$ that do not explicitly require trace-objectification (39), we state the branching requirement proved for trace-reifying systems in Proposition 1 as a general architectural condition.

Assumption 3 (Generative Richness). At every level n and step t at which extension is attempted,

$$
\left| \operatorname { S u r p } ( O _ { n , t } ) \right| \ \geq \ 2 .\tag{50}
$$

When $\mathcal { E } ( \mathcal { F } _ { n , t } )$ is left unrestricted, (50) is not implied by (30) alone, since a system without reflective trace reification could face an unresolved proposition while possessing at most one admissible successor; when $\mathcal { E } ( \mathcal { F } _ { n , t } )$ includes (39), Proposition 1 guarantees (50) constructively from any unclosed proof search on $I ^ { \mathrm { i n c } } ( O _ { n , t } )$ . It is what §5.3 needs for the space of trajectories to carry a binary-tree structure, and what §6.8 restates at the level of an individual agent within a collective.

Recursive extension does not imply success. A successor may resolve an ambiguity present in $\mathcal { F } _ { n } .$ leave it unresolved, reinterpret the relevant structure, or introduce new ambiguities of its own. Recursion, therefore, produces exploration rather than certainty, and the trajectory (49) should not be read as monotone progress toward a final system.

## 4.4 Local Stability and Global Recursive Extension

The preceding distinction can be sharpened by assigning a state space to each inferential regime. Let

$$
( X _ { n } , d _ { n } ) , \qquad X _ { n } = \{ O \colon L _ { n } \longrightarrow \mathcal { V } _ { n } \mid O { \mathrm { ~ s a t i s f i e s ~ } } ( 2 5 ) \} ,\tag{51}
$$

be a metric space, where $X _ { n }$ is the set of cognitive states admissible within regime ${ \mathcal { F } } _ { n }$ and

$$
d _ { n } \colon X _ { n } \times X _ { n } \to \mathbb { R } _ { \geq 0 }\tag{52}
$$

measures the semantic distance between states. Note that $X _ { n }$ is non-trivial precisely because (25) was not strengthened to a biconditional: i.e., had commitment been required wherever proof is available, $X _ { n }$ would contain a single point and the dynamics below would be empty.

A local transformation is a map $f _ { i } ^ { ( n ) } \colon X _ { n } \to X _ { n }$ . For some applications, it is useful to require these to be contractive.

Assumption 4 (Local Contractivity). For each relevant transformation $f _ { i } ^ { ( n ) }$ there exists $0 \leq c _ { n } < 1$ with

$$
d _ { n } \bigl ( f _ { i } ^ { ( n ) } ( x ) , f _ { i } ^ { ( n ) } ( x ^ { \prime } ) \bigr ) \ \leq \ c _ { n } d _ { n } ( x , x ^ { \prime } ) \qquad \forall x , x ^ { \prime } \in X _ { n } .\tag{53}
$$

This expresses a local engineering principle: i.e., within a fixed semantic regime, reasoning reduces uncertainty and converges toward a stable interpretation. If the family

$$
\mathcal { H } _ { n } = \{ f _ { 1 } ^ { ( n ) } , \dots , f _ { k _ { n } } ^ { ( n ) } \}\tag{54}
$$

consists of contractive maps on a complete metric space, the corresponding iterated function system admits a unique compact attractor $\Lambda _ { n } \subseteq X _ { n }$ under the standard conditions of Hutchinson’s theorem.

The significance of this result for the present theory is limited but important. It establishes that local stability and open-ended global evolution are not contradictory requirements. Local convergence has the form

$$
X _ { n } \longrightarrow \Lambda _ { n } ,\tag{55}
$$

while recursive extension has the form

$$
X _ { n } \longrightarrow X _ { n + 1 } .\tag{56}
$$

The two operate at diferent structural levels. Convergence to $\Lambda _ { n }$ substantially exhausts $I ^ { \mathrm { l o c } }$ but, by (47), leaves $I ^ { \mathrm { i n c } }$ intact—so reaching the attractor is the condition under which (44) becomes the only remaining move. In this sense, the attractor is the formal counterpart of the ε-saturation criterion (10) of §3.4, with the diference that its residue is guaranteed non-empty by (30) rather than assumed.

## 4.5 Recursive Extension

We now state the successor operation in general form.

Definition 2 (Recursive Extension Operator). A recursive extension at level n is a pair

$$
\mathcal { R } _ { n } = ( \mathcal { F } _ { n + 1 } , \tau _ { n } ) , \qquad \mathcal { F } _ { n + 1 } \in \mathcal { E } ( \mathcal { F } _ { n } ) , \qquad \tau _ { n } : L _ { n } \to L _ { n + 1 } ,\tag{57}
$$

selected by (44), where $\mathcal { F } _ { n + 1 }$ satisfies regularity and $\tau _ { n }$ satisfies reflective adequacy in the sense of §4.2. The operator may modify the language, axioms, inference rules, representational structures, or meta-level resources of the system.

We write

$$
{ \mathcal { F } } _ { n } \xrightarrow { { \mathcal { R } } _ { n } } { \mathcal { F } } _ { n + 1 }\tag{58}
$$

for the resulting transition. $\mathcal { R } _ { n }$ carries selected structures forward while potentially transforming their interpretation. What reflective adequacy secures is that $L _ { n + 1 }$ can express statements about ${ \mathcal { F } } _ { n } .$ so that the predecessor becomes an object of reasoning for the successor rather than merely a stage it has passed through. This is the formal counterpart of recursive reflection, and the property on which the memory requirements of §6.4 depend.

## 4.6 Successor Incompleteness

The significance of recursive extension follows directly from §4.1. Suppose ${ \mathcal { F } } _ { n }$ is regular and $\mathcal { R } _ { n }$ constructs a successor ${ \mathcal { F } } _ { n + 1 } ;$ admissibility requires $\mathcal { F } _ { n + 1 }$ to be regular in turn. Then by (45) there is $G _ { n + 1 } \in L _ { n + 1 }$ with

$$
{ \mathcal { F } } _ { n + 1 } \nvdash G _ { n + 1 } \qquad { \mathrm { a n d } } \qquad { \mathcal { F } } _ { n + 1 } \nvdash { \neg G } _ { n + 1 } .\tag{59}
$$

Iterating, the architecture generates

$$
\mathcal { F } _ { 0 } \xrightarrow { \mathcal { R } _ { 0 } } \mathcal { F } _ { 1 } \xrightarrow { \mathcal { R } _ { 1 } } \mathcal { F } _ { 2 } \xrightarrow { \mathcal { R } _ { 2 } } \cdot \cdot \cdot\tag{60}
$$

with, at every level,

$$
G _ { n } \in I ^ { \mathrm { i n c } } ( O _ { n } ) \qquad \mathrm { a n d } \qquad I ^ { \mathrm { i n c } } ( O _ { n } ) \neq \emptyset .\tag{61}
$$

Recursive extension, therefore, does not eliminate incompleteness. It produces a new level at which incompleteness arises again. This is not a defect of the construction but of its engine: i.e., by (61), the recursive pressure of §4.1 is renewed at every level, so the architecture never reaches a state in which extension becomes unmotivated.

Two limits of this result should be stated plainly. First, (61) guarantees that each level has something undecidable, whereas ensuring that the successor decides what its predecessor could not relies on the decisive surplus (35), which requires reflective trace reification (Proposition 1, stated generally in Assumption 2) rather than bare extensional incompleteness alone. Second, (60) establishes that the sequence can be continued indefinitely, not that it must be, and not that its members difer in any way a reader would call progress. The stronger condition that successive extensions carry irreducible novelty is introduced separately as Assumption 5 in §5.2. The essential property of the architecture is accordingly not that it reaches a complete state, but that it can construct successor systems in which unresolved questions can be reconsidered from a diferent representational position.

## 4.7 Local Convergence and Global Open-Endedness

The distinction between local dynamics and recursive extension allows open-endedness to be defined without requiring unbounded expansion inside any single semantic space.

At each level, the system may operate within a bounded or locally contractive regime, converging by (55) to an attractor $\Lambda _ { n }$ . Yet the architecture may continue to generate successor regimes,

$$
( X _ { 0 } , d _ { 0 } ) \longrightarrow ( X _ { 1 } , d _ { 1 } ) \longrightarrow ( X _ { 2 } , d _ { 2 } ) \longrightarrow \cdot \cdot \cdot\tag{62}
$$

Open-endedness, therefore, does not require that any individual semantic space be unbounded. It requires that the process of constructing successor spaces not be restricted to a fixed finite hierarchy—the condition invoked at (12) in the teleological model, now without the closure constraint (13) that bounded it there.

This yields the central distinction of the section:

$$
\mathrm { l o c a l \ c o n v e r g e n c e \ \neq \ g l o b a l \ t e r m i n a t i o n . }\tag{63}
$$

The inequality is not merely definitional. Local convergence means $I ^ { \mathrm { l o c } } ( O _ { n } ) = \emptyset$ , attained at the attractor $\Lambda _ { n }$ . Global termination would additionally require $I ^ { \mathrm { i n c } } ( O _ { n } ) = \emptyset$ , which (30) forbids for any regular system. A system can therefore be stable with respect to its current regime while remaining capable of constructing new ones, and the stability is not evidence that the process has ended.

Relation (63) directly resolves the “stopping problem” identified by He et al. [2026, p. 166] for $( C , U , V )$ agents. In their formalization, an agent’s evolution converges, lim<sub>t</sub> ${ \bf \Gamma } _ {  \infty } ( C _ { t } , U _ { t } , V _ { t } ) =$ $( C ^ { * } , U ^ { * } , V ^ { * } )$ , once the system attains “cognitive completeness”—where the parse-graph parameters $\theta _ { \mathrm { P G } } ~ ^ { * } \mathrm { c a n }$ express all reasonable human cognition and possible actions for the given scene” [He et al., 2026, p. 174]—and “task completeness”—where the agent has “the capability to find all possible target PGs that can increase the values in its V-dimension, as well as the capability to determine a path from the agent’s current PG to the future target $\mathrm { P G } ^ { \mathfrak { s } }$ [He et al., 2026, p. 174]. At that point, “when an agent’s cognitive architecture determines that the agent’s current skills have reached the upper limit of what the agent can master, or that the agent’s current value functions have reached the optimal level the agent can achieve, the cognitive architecture cannot further orchestrate the learning process, leading to the convergence of learning” [He et al., 2026, p. 175]. In the terms of §4.1–§4.4, cognitive and task completeness describe the exhaustion of the decidable unresolved region $I ^ { \mathrm { l o c } } ( O _ { n } ) \to \emptyset$ as the trajectory approaches the local attractor $\Lambda _ { n } \subseteq X _ { n }$ in (55). That convergence becomes a terminal stopping problem in $( C , U , V )$ only because cognitive completeness is defined relative to a fixed function class $\theta _ { \mathrm { P G } }$ and a fixed value specification $V _ { \mathrm { s e l e c t } }$ . Within any suficiently expressive regime ${ \mathcal { F } } _ { n } .$ however, exhausting $I ^ { \mathrm { l o c } } ( O _ { n } )$ at $\Lambda _ { n }$ leaves the undecidable region $I ^ { \mathrm { i n c } } ( O _ { n } )$ intact by (47). What He et al. [2026] characterize as the terminal boundary of learning within a given $( C , U , V )$ space is here the trigger for the co-recursive transition $X _ { n } \to X _ { n + 1 }$ in (56), provided the architecture is governed by a meta-teleology that treats saturation at $\Lambda _ { n }$ as the signal to extend the representational and teleological framework itself.

We can make this coupling between local convergence and global co-recursion explicit by formalizing the a priori meta-teleological trigger introduced in §2 and §3.4.

Definition 3 (A Priori Meta-Teleological Trigger). Let $\left( ( X _ { n } , d _ { n } ) , \mathcal { H } _ { n } \right)$ be a locally contractive cognitive regime satisfying Assumption 4 with compact attractor $\Lambda _ { n } \subseteq X _ { n }$ driven by the active first-order teleology $\Phi _ { n } ^ { + }$ . Given a saturation threshold $\varepsilon > 0$ , an a priori meta-teleological trigger is a second-order transition rule that monitors the metric proximity $\begin{array} { r } { d _ { n } ( O _ { n , t } , \Lambda _ { n } ) : = \operatorname* { i n f } _ { y \in \Lambda _ { n } } d _ { n } ( O _ { n , t } , y ) } \end{array}$ and, at the first step $t _ { n } ^ { * }$ for which

$$
d _ { n } \bigl ( O _ { n , t _ { n } ^ { * } } , \Lambda _ { n } \bigr ) \ < \ \varepsilon ,\tag{64}
$$

invokes the meta-level selection operator $\mathcal { M } _ { n }$ on the candidate extensions (39) generated from the unclosed proper reduction traces $\pi _ { n , t _ { n } ^ { * } } ( x )$ of $x \in I ( O _ { n , t _ { n } ^ { * } } )$ —which by (30), (47), and Proposition 2 are guaranteed to contain at least one genuine decisive pair in $\mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n , t _ { n } ^ { * } } )$ alongside any latent local branches—to select a decisive extension ${ \mathcal { F } } _ { n + 1 , 0 } \in { \mathcal { M } } _ { n } \big ( \mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n , t _ { n } ^ { * } } ) \big )$ under Assumption 2 (or Propositions 1–2), initiating the regime transition $( X _ { n } , d _ { n } ) \to ( X _ { n + 1 } , d _ { n + 1 } )$

Proposition 3 (Escape from Local Convergence). Let A be a cognitive architecture whose local dynamics on each regime $( X _ { n } , d _ { n } )$ satisfy Inferential Soundness (Assumption 1) and Local Contractivity (Assumption $\it 4 )$ with contraction factor $0 \leq c _ { n } < 1$

1. (Absorption under intra-regime selection.) Any architecture whose self-modification rule requires deductive certification within ${ \mathcal { F } } _ { n }$ (acting only on $I ^ { \mathrm { l o c } } ( O _ { n , t } )$ , as in a G¨odel machine [Schmidhuber, 2006]) or only reweights objectives within the fixed space $X _ { n }$ (as in $( C , U , V )$ selector convergence [He et al., 2026]) satisfies $d _ { n } ( O _ { n , t } , \Lambda _ { n } ) \leq c _ { n } ^ { t } d _ { n } ( O _ { n , 0 } , \Lambda _ { n } ) \to 0 \ a$ s t → ∞ and remains permanently confined to $X _ { n }$

2. (Unbounded regime succession under meta-teleological triggering.) If A satisfies Decisive Extension (Assumption 2) and is equipped with the a priori meta-teleological trigger of Definition 3 for some $\varepsilon > 0$ , then at every level $n \geq 0$ the trigger condition (64) is reached in at

most

$$
t _ { n } ^ { * } \leq \mathrm { m a x } \Big \{ 0 , \Big \lfloor \frac { \ln \big ( d _ { n } ( O _ { n , 0 } , \Lambda _ { n } ) / \varepsilon \big ) } { \ln ( 1 / c _ { n } ) } \Big \rfloor + 1 \Big \}\tag{65}
$$

local steps, at which point $I ^ { \mathrm { i n c } } ( O _ { n , t _ { n } ^ { * } } ) \neq \emptyset b y \ ( 3 0 )$ and (47), inducing an infinite sequence of distinct regimes $( X _ { 0 } , d _ { 0 } )  ( X _ { 1 } , d _ { 1 } )  ( X _ { 2 } , d _ { 2 } )  \cdots i n$ which no local attractor $\Lambda _ { n }$ is a global halting state.

Proof. For (1), every local transformation $f _ { i } ^ { ( n ) } \in \mathcal { H } _ { n }$ preserves $L _ { n }$ and $\Gamma _ { n }$ by (46) and acts solely on $I ^ { \mathrm { l o c } } ( O _ { n , t } )$ by (47), while leaving $I ^ { \mathrm { i n c } } ( O _ { n , t } )$ invariant. By Assumption 4 and Hutchinson’s theorem, the induced trajectory on $( X _ { n } , d _ { n } )$ contracts toward $\Lambda _ { n }$ at rate $c _ { n } ^ { t } .$ , and because no transition rule is licensed to adopt an element of $\mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n , t } )$ whose validity lies in $I ^ { \mathrm { i n c } } ( O _ { n , t } )$ , the state remains in $X _ { n }$ for all $t \geq 0$

For (2), contractivity implies $d _ { n } ( O _ { n , t } , \Lambda _ { n } ) \leq c _ { n } ^ { t } d _ { n } ( O _ { n , 0 } , \Lambda _ { n } )$ , which falls below ε no later than the bound $t _ { n } ^ { * }$ in (65). At step $t _ { n } ^ { * }$ , regularity of ${ \mathcal { F } } _ { n }$ guarantees $I ^ { \mathrm { i n c } } ( O _ { n , t _ { n } ^ { * } } ) = I ^ { \mathrm { i n c } } ( O _ { n , 0 } ) \neq \emptyset$ by (30) and (47), and Assumption 2 guarantees $\mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n , t _ { n } ^ { * } } ) \neq \emptyset$ . Definition 3 then executes $\mathcal { F } _ { n , t _ { n } ^ { * } } \to \mathcal { F } _ { n + 1 , 0 }$ which by regularity of $\mathcal { F } _ { n + 1 , 0 }$ and (61) renews the hypotheses at level $n { \mathrel { + { 1 } } }$ , continuing the sequence indefinitely by induction on n. □

The mathematical question is consequently shifted. It is not whether a single state space can contain every future cognitive state, but whether an architecture can sustain an indefinitely extensible sequence of representational systems—and, as §5shows, whether any fixed architecture can do so at all.

## 5 Representational Limits of Open-Ended Cognitive Architectures

The preceding section establishes a recursive architecture in which a formal system constructs successor systems when its current representational resources prove insuficient. We now ask what this process costs in representational terms.

The question is whether a fixed internal description and a fixed transition mechanism can completely represent an indefinitely self-extending architecture. This is distinct from the question of computational scale. A system may possess arbitrarily large computational resources while remaining within a fixed representational framework; conversely, a system may recursively extend its framework without any individual regime being unbounded. The issue is not how much computation is available at a given level, but whether the information required to specify the successive regimes can remain bounded.

## 5.1 Semantic Description Complexity

Let $K ( x )$ denote the Kolmogorov complexity of an object x with respect to a fixed universal prefix-free Turing machine, and $K ( x \mid y )$ the corresponding conditional complexity. We assume throughout that formal systems are encoded as finite binary strings under a fixed efective encoding; this is possible because each ${ \mathcal { F } } _ { n }$ is efectively axiomatized by §4.1.

For the sequence of systems generated by (60), define the accumulated history

$$
H _ { n } = ( { \mathcal { F } } _ { 0 } , { \mathcal { F } } _ { 1 } , \ldots , { \mathcal { F } } _ { n } ) .\tag{66}
$$

The quantity $K ( H _ { n } )$ measures the minimum algorithmic information required to specify the complete semantic history up to level n.

The distinction between $K ( H _ { n } )$ and $K ( \mathcal { F } _ { n } )$ matters. The complexity of an individual successor need not exceed that of its predecessor: i.e., a successor may contain a highly compressed representation of information previously distributed across several structures, so that $K ( { \mathcal { F } } _ { n + 1 } ) < K ( { \mathcal { F } } _ { n } )$ is entirely possible. What matters for open-ended evolution is whether the accumulated history can remain algorithmically bounded.

## 5.2 Independent Semantic Extension

Let $\gamma > 0$ be a constant for which the chain rule for prefix Kolmogorov complexity holds, for all $n \geq 2$ , in the form

$$
K ( x , y ) \ \geq \ K ( x ) + K ( y \mid x ) - \gamma \log _ { 2 } n ,\tag{67}
$$

where n bounds the lengths of the objects involved; the additive constant of the usual statement is absorbed into $\gamma \log _ { 2 } n$

Before stating the algorithmic novelty condition, we note how the architectural postulates of §3.4 and §4.2 give rise to conditional description complexity at each extension step. When the meta-level operator selects a successor ${ \mathcal { F } } _ { n + 1 } \in { \mathcal { M } } _ { n } { \big ( } { \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } ) { \big ) }$ , three structural features distinguish $\mathcal { F } _ { n + 1 }$ from a deductive continuation of $H _ { n }$ . First, by Decisive Extension (Assumption 2), $\mathcal { F } _ { n + 1 }$ settles at least one frontier proposition $x _ { n } \in I ^ { \mathrm { i n c } } ( O _ { n } )$ that is formally independent of $\Gamma _ { n }$ and therefore not deductively determined by $H _ { n }$ . Second, by Generative Richness (Assumption 3), ${ \mathrm { S u r p } } ( O _ { n } )$ ofers multiple competing continuations at each level. Third, by the non-monotone teleologica succession condition (15), the extended vocabulary $L _ { n + 1 } \setminus \tau _ { n } ( L _ { n } )$ is selected in response to newly discriminated environmental signals $D _ { \phi _ { n + 1 } } \setminus D _ { H _ { n } } \neq \emptyset \ ( \mathrm { o r }$ , in the collective setting of §6.2, crossagent polycontextural inputs) that were excluded from every prior domain in $H _ { n }$ . In a closed deductive rewriter such as a G¨odel machine [Schmidhuber, 2006], where axioms are replaced only by “theorems derivable from the original axioms” [Schmidhuber, 2006, p. 17], the successor at proof-search step $t _ { n }$ is a computable function of $\mathcal { F } _ { 0 }$ and $t _ { n } ,$ so that $K ( { \mathcal { F } } _ { n + 1 } \mid H _ { n } ) \leq O ( \log n )$ . By contrast, whenever the commitment ${ \mathcal { F } } _ { n + 1 } \in { \mathcal { M } } _ { n } { \big ( } { \mathrm { S u r p } } ^ { \mathrm { d e c } } ( O _ { n } ) { \big ) }$ encodes genuinely new semantic discrimination that no fixed finite program can compute from $H _ { n }$ alone, its conditional complexity $K ( \mathcal { F } _ { n + 1 } \mid H _ { n } )$ exceeds the logarithmic prefix-coding overhead $\gamma \log _ { 2 } n$ of (67) by a positive margin $c > 0$

Assumption 5 (Persistent Algorithmic Novelty). There is a constant $c > 0$ such that, for every level n,

$$
K ( { \mathcal { F } } _ { n + 1 } \mid H _ { n } ) \geq c + \gamma \log _ { 2 } n .\tag{68}
$$

Applying (67) to the history and its successor,

$$
K ( H _ { n + 1 } ) \ = \ K ( H _ { n } , { \mathcal F } _ { n + 1 } ) \ \ge \ K ( H _ { n } ) + K ( { \mathcal F } _ { n + 1 } \ | \ H _ { n } ) - \gamma \log _ { 2 } n ,\tag{69}
$$

so that under Assumption 5 the logarithmic terms cancel and

$$
K ( H _ { n + 1 } ) \ \geq \ K ( H _ { n } ) + c ,\tag{70}
$$

giving, by induction,

$$
K ( H _ { n } ) ~ \geq ~ K ( H _ { 0 } ) + c n \qquad { \mathrm { a n d } } \qquad \left| \operatorname* { l i m } _ { n \to \infty } K ( H _ { n } ) = \infty \right| .\tag{71}
$$

The conclusion is not

$$
K ( { \mathcal { F } } _ { n + 1 } ) > K ( { \mathcal { F } } _ { n } ) ,\tag{72}
$$

which need not hold at any level, but the divergence in (71): the accumulated history grows at least linearly in the number of extensions.

Remark (Step-Wise vs. Amortized Algorithmic Novelty). The additive term $\gamma \log _ { 2 } n$ in (68) is included solely so that the prefix chain rule (67) yields the clean step-wise increment $K ( H _ { n + 1 } ) \ –$ $K ( H _ { n } ) \ \geq \ c$ at every single transition in (70). For the impossibility theorem of §5.4, however, neither step-wise uniformity nor the chain-rule constant $\gamma$ is needed. What Theorem 1 actually requires is only that the accumulated history complexity $K ( H _ { n } )$ eventually exceed the logarithmic self-delimiting index bound $2 \log _ { 2 } n$ of a fixed simulator. In particular, Theorem 1 holds unchanged under the much weaker amortized novelty condition (positive lower asymptotic density)

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { n } { \frac { 1 } { n } } K ( H _ { n } ) \ \geq \ c > \ 0 ,\tag{73}
$$

and indeed under any super-logarithmic divergence lim $\iota _ { n \to \infty } ( K ( H _ { n } ) - 2 \log _ { 2 } n ) = \infty$ . Under (73), arbitrarily long runs of recursive extensions may consolidate or compress prior structure with $K ( { \mathcal { F } } _ { n + 1 } \mid H _ { n } ) = O ( 1 )$ , provided that genuinely incompressible semantic discriminations enter intermittently with positive asymptotic density $c > 0$

## 5.3 Trajectory Complexity

Two distinct sources of complexity should be kept apart.

First, the space of possible trajectories. By Assumption 3, every extension step ofers at least two admissible continuations. If branching occurs at infinitely many levels and the choices are independent, the set of trajectories through (49) contains a subset order-isomorphic to the infinite binary tree, and hence a subset in bijection with Cantor space $\{ 0 , 1 \} ^ { \mathbb { N } }$ . Its cardinality is therefore $2 ^ { \aleph _ { 0 } }$

Second, the complexity of a single realized trajectory. By (71), one such trajectory has finite prefixes of unbounded algorithmic complexity.

These are diferent claims, and neither implies the other. Uncountability concerns how many trajectories the architecture could take; unbounded complexity concerns how much information one of them contains. A system could have uncountably many possible trajectories, all of which are algorithmically trivial, or a unique trajectory of unbounded complexity.

## 5.4 The Fixed-Architecture Limitation

Theorem 1 (Unbounded Representational Complexity of Open-Ended Trajectories). Let

$$
\mathcal { A } = ( \Sigma , \Theta , \sigma _ { 0 } )\tag{74}
$$

be a cognitive architecture with a fixed finite state description $\Sigma ,$ a fixed computable transition mechanism Θ, and a fixed initial condition $\sigma _ { 0 } \in \Sigma$ . Then A cannot generate, from $\sigma _ { 0 }$ alone, a semantic history $H _ { 0 } , H _ { 1 } , H _ { 2 } , . . .$ . satisfying Assumption 5.

Proof. Suppose it could. Since Θ and $\sigma _ { 0 }$ are fixed, the state sequence $\sigma _ { t + 1 } = \Theta ( \sigma _ { t } )$ is determined by the finite description $( \Theta , \sigma _ { 0 } )$ together with the index n. A program that reads this description, iterates Θ from $\sigma _ { 0 }$ , and outputs the first $n + 1$ systems therefore specifies $H _ { n }$ completely, so that

$$
K ( H _ { n } ) \ \leq \ K ( \Theta , \sigma _ { 0 } ) + 2 \log _ { 2 } n + O ( 1 ) ,\tag{75}
$$

where $2 \log _ { 2 }$ n encodes n self-delimitingly and the additive constant covers the fixed simulation program. Apart from that logarithmic term, the bound is independent of $n$

By Assumption 5 and (71), however,

$$
K ( H _ { n } ) ~ \ge ~ K ( H _ { 0 } ) + c n\tag{76}
$$

with $c > 0$ . Combining (75) and (76),

$$
K ( H _ { 0 } ) + c n \ : \le \ : K ( \Theta , \sigma _ { 0 } ) + 2 \log _ { 2 } n + O ( 1 ) ,\tag{77}
$$

whose left-hand side grows linearly in n while its right-hand side grows logarithmically. The inequality therefore fails for all suficiently large n, and no such A exists. □

A fixed closed transition system cannot, from a fixed finite initial description alone, generate an indefinitely self-extending semantic trajectory carrying persistent algorithmic novelty. Moreover, inspection of (77) confirms that the contradiction depends only on asymptotic growth exceeding $2 \log _ { 2 } n ,$ and therefore holds verbatim when the step-wise condition of Assumption 5 is relaxed to the amortized density condition (73).

## 5.5 What the Result Does and Does Not Establish

Theorem 1 does not say that a fixed computer cannot represent an infinite or uncountable state space, nor does it deny that a program of fixed Kolmogorov complexity can generate open-ended complexity. In his work on algorithmic theories of everything, Schmidhuber [Schmidhuber, 2000] shows that a compact, fixed program can produce boundless apparent complexity in two distinct ways, and examining why neither yields open-ended intelligence clarifies the exact force of Assumption 5.

First, a single history ${ \boldsymbol { x } } = ( x _ { 1 } , x _ { 2 } , \dots )$ can be generated by $^ { 6 6 } \mathrm { a }$ finite algorithm A that computes $x _ { k + 1 } \ ( k \geq 1 )$ from $x _ { k }$ and additional information noise<sub>k</sub>” [Schmidhuber, 2000, p. 4] where noise<sub>k</sub> is “calculated by invoking a finite pseudorandom generator subroutine” [Schmidhuber, 2000, p. 4]. To an observer inside such a process, states may appear perpetually novel. Yet as Schmidhuber notes of histories “deterministically computed by a short algorithm,” “the algorithmic entropy (compare [98]) of the universe does not increase over time, because a finite program conveying a finite amount of information is responsible for everything” [Schmidhuber, 2000, p. 37]. In the terms of Theorem 1, such a generator is a fixed closed architecture $\mathcal { A } = ( \Sigma , \Theta , \sigma _ { 0 } )$ whose history satisfies (75): its complexity is fixed from the outset, and no extension step introduces irreducible algorithmic information.

Second, Schmidhuber considers universal dovetailing algorithms such as FAST, which “simultaneously compute all computable universes in an interleaving fashion that outputs each individual universe as quickly as its fastest algorithm running just by itself, save for a constant factor independent of the universe’s size” [Schmidhuber, 2000, p. 5]. Because “computing all universes with all possible types of physical laws tends to be much cheaper in terms of information requirements than computing just one particular, arbitrarily chosen one, because there is an extremely short algorithm that systematically enumerates and runs all computable universes, while most individual universes have very long shortest descriptions” [Schmidhuber, 2000, p. 40], a program of fixed O(1) complexity will eventually enumerate every finite prefix $H _ { n }$ of every computable semantic trajec tory alongside all other strings. Yet enumerating the entire branching tree of §5.3 is not open-ended intelligence. As Schmidhuber himself remarks when noting that FAST outputs prefixes of Chaitin’s incompressible $\Omega _ { n }$ within $O ( n 2 ^ { n } )$ steps, it does so “only because it also outputs many other strings besides $\Omega _ { n }$ — there is still no fast way of identifying $\Omega _ { n }$ among all the outputs” [Schmidhuber, 2000, p. 30]. By the discrimination principle of §3.2, a procedure that admits every continuation without exclusion $( \Phi _ { n } ^ { - } = \varnothing$ in (4)) makes no distinction at all. Open-ended intelligence is not the unselective enumeration of all possible branches, which carries negligible algorithmic information precisely because nothing is rejected; it is the realization of a single selective trajectory $H _ { n } = ( { \mathcal { F } } _ { 0 } , \ldots , { \mathcal { F } } _ { n } )$ through the branching surplus via successive commitments $\mathcal { F } _ { k + 1 } = \mathcal { M } _ { k } ( \operatorname { S u r p } ( O _ { k } ) )$ . Identifying one such trajectory that satisfies Assumption 5 among all strings printed by a dovetailer requires specifying the sequence of selections at each level, and by (71) that specification carries at least cn bits of algorithmic information that the fixed dovetailer does not possess.

What Theorem 1 rules out is therefore specific: a fixed closed architecture cannot by itself select and realize a semantic history satisfying Assumption 5. The conflict is between (71) and (75), and it arises whenever each extension step commits to a distinction whose algorithmic information cannot be compressed into the prior history.

To sustain such a process, at least one of the following must therefore change over time:

1. the representational language of the system;

2. the transition rules governing it;

3. the external information available to it;

4. its own meta-level description.

Recursive self-extension supplies precisely such a mechanism. Rather than requiring a single fixed architecture to encode every future semantic distinction, the architecture constructs successors (60), distributing the representational burden across a sequence of formal systems rather than imposing it on a single fixed state space. Item (3) is the route taken in §7: a bounded agent coupled to other agents receives information not derivable from its own history.

## 5.6 Compression and Semantic Growth

The result clarifies the relation between compression and semantic extension.

Compression seeks the smallest description of an existing semantic domain,

$$
K ( \mathrm { r e p r e s e n t a t i o n } ) \downarrow\tag{78}
$$

subject to preserving the relevant information or predictive behavior. Recursive semantic extension addresses a diferent problem: when the current system cannot adequately represent or resolve a relevant ambiguity, the architecture constructs a successor (58), and the objective is not to minimize the complexity of the current representation but to enlarge the space of representable distinctions.

These are compatible. Under Assumption 5 the accumulated history diverges by (71), while each individual $K ( \mathcal { F } _ { n } )$ may be small and may even decrease—a system may compress aggressively within a regime while the sequence of regimes accumulates irreducible information. This yields the central distinction of the section:

compression within a semantic space $\neq$ extension of the semantic space itself. (79)

## 5.7 Relation to Open-Ended Intelligence

The result provides a formal reason why open-ended intelligence cannot be reduced to optimization within a fixed representational space. A system may become increasingly capable within a fixed language—improving predictions, compressing observations, optimizing parameters—without thereby acquiring the ability to construct a new regime. That ability requires (58) with a successor introducing distinctions not derivable from its predecessor, which is (35) on the logical side and Assumption 5 on the algorithmic side.

The significance of recursive teleology becomes clearer here. The problem is not that an intelligent system must compute more. It must determine when its current regime is insuficient, generate a successor, and continue reasoning within it—the three operations that §3.4 assigned to saturation detection, $\mathcal { E } _ { \Phi }$ , and T respectively, and that §4.2 reassigns to (30), E, and $\mathcal { M } _ { n }$

Theorem 1 also sets up the dificulty §6 addresses. Every physical system is finite, so no individual architecture can be the unbounded source of novelty that Assumption 5 requires. Either the assumption fails in real systems, or the novelty comes from outside the individual, which is the possibility that §6 develops.

## 6 Collective Intelligence and Distributed Semantic Extension

Theorem 1 establishes that no fixed closed architecture generates a semantic history satisfying Assumption 5. Every physical cognitive system is fixed in this sense: its memory, processing capacity, energy, and temporal horizon are bounded, and its transition mechanism is finite. The theorem, therefore, applies to any individual agent.

This limitation is not specific to artificial systems. No individual human possesses unbounded memory or representational capacity, yet human knowledge accumulates across generations through language, writing, institutions, and external records—a semantic structure exceeding the state space of any individual participant.

We therefore ask which of the escape routes listed in §5.5 a collective takes. It is route (3): external information. A bounded agent embedded in a collective receives representations not derivable from its own history, and a persistent external memory accumulates what no agent retains. This section makes that precise. It does not evade Theorem 1; a collective of finitely many fixed architectures is itself a fixed architecture and falls under the same result.

## 6.1 Collective Cognitive System

Let $C = \{ A ^ { 1 } , \ldots , A ^ { N } \}$ be a finite collection of cognitive agents. Each $A ^ { k }$ carries its own inferential system $\mathcal { F } _ { n , t } ^ { k } .$ , cognitive state $O _ { n , t } ^ { k }$ , and state space $X _ { n } ^ { k }$ in the sense of (51). Agents need not share a language.

Communication requires translation rather than identity of representation. For agents $A ^ { i } , A ^ { j }$ let

$$
\sigma _ { n } ^ { i  j } : L _ { n } ^ { i }  L _ { n } ^ { j }\tag{80}
$$

be a partial translation, defined on the fragment of $A ^ { i } \mathrm { { ^ { s } } }$ language that $A ^ { j }$ can interpret. Its domain

$$
L _ { n } ^ { i , j } = \operatorname { d o m } ( \sigma _ { n } ^ { i  j } ) \subseteq L _ { n } ^ { i }\tag{81}
$$

is the common ground between the two agents. Partiality is essential: $L _ { n } ^ { i , j } \ = \ L _ { n } ^ { i }$ would make the agents interchangeable, while $L _ { n } ^ { i , j } = \emptyset$ would make them incommunicable. Neither extreme supports collective extension, as $\ S 6 . 1 0$ shows.

The collective state is not the union of individual states but the individual states together with the relations coupling them,

$$
\mathbb { X } _ { n , t } = \left( \{ X _ { n } ^ { k } \} _ { k = 1 } ^ { N } , \ \{ \sigma _ { n } ^ { i \to j } \} _ { i \neq j } \right) .\tag{82}
$$

## 6.2 Distributed Semantic Surplus

Each agent has a semantic surplus $\operatorname { S u r p } ( O _ { n , t } ^ { k } )$ in the sense of (34). The collective surplus is

$$
\mathrm { S u r p } _ { C } ( n , t ) = \bigcup _ { k = 1 } ^ { N } \mathrm { S u r p } ( O _ { n , t } ^ { k } ) .\tag{83}
$$

The cardinality of this union is not what makes a collective more than the sum of its agents; that is measured in §9.1. The structurally important property is that one agent’s surplus can become another’s unresolved region. If $A ^ { i }$ constructs a successor $\mathcal { F } ^ { \prime } \in \mathrm { S u r p } ( O _ { n , t } ^ { i } )$ and communicates it, the translated content $\sigma _ { n } ^ { i  j } ( { \mathcal { F } } ^ { \prime } )$ may contain propositions that $A ^ { j }$ can express but not decide,

$$
\sigma _ { n } ^ { i \to j } \bigl ( \operatorname { S u r p } ( O _ { n , t } ^ { i } ) \bigr ) \cap I ^ { \operatorname { i n c } } ( O _ { n , t } ^ { j } ) \neq \emptyset ,\tag{84}
$$

so that what one agent has resolved becomes a source of recursive pressure for another. By (30), this pressure can then drive $A ^ { j }$ to construct successors of its own, which may in turn return to $A ^ { i }$ in transformed form.

## 6.3 Collective Co-Recursion

Collective co-recursion is the process in which semantic transformations cross the boundaries of individual systems and generate successor states in others. For agents $A ^ { i } , A ^ { j }$ a collective transition has the form

$$
I ^ { \mathrm { i n c } } ( O _ { n , t } ^ { i } ) \stackrel { \varepsilon } { \longrightarrow } \mathrm { S u r p } ( O _ { n , t } ^ { i } ) \stackrel { \sigma _ { n } ^ { i \to j } } { \longrightarrow } I ^ { \mathrm { i n c } } ( O _ { n , t } ^ { j } ) \stackrel { \varepsilon } { \longrightarrow } \mathrm { S u r p } ( O _ { n , t + 1 } ^ { j } ) ,\tag{85}
$$

and a collective trajectory is a sequence

$$
{ \cal O } _ { n _ { 0 } , t _ { 0 } } ^ { k _ { 0 } }  { \stackrel {  } {  } } { \cal O } _ { n _ { 1 } , t _ { 1 } } ^ { k _ { 1 } }  {  } { \cal O } _ { n _ { 2 } , t _ { 2 } } ^ { k _ { 2 } }  {  { \kdots } }\tag{86}
$$

in which the agent index may change at each step. The semantic process thereby becomes independent of the physical identity of any single agent. In other words, a transformation need not remain within the system that generated it.

What interaction alone does not achieve. Interaction between finitely many bounded agents does not by itself escape Theorem 1. If each $A ^ { k }$ has a finite state space $B ^ { k }$ and no state persists outside the agents, the joint state space $\Pi _ { k = 1 } ^ { N } B ^ { k }$ is finite, the collective is a fixed architecture in the sense of (74), and (75) applies to it unchanged. The collective trajectory (86) must then eventually revisit a joint state and cycle. Coupling redistributes cognitive work; it does not create algorithmic novelty.

What changes the situation is persistent external memory.

Proposition 4 (Bounded Agents with Persistent Memory). Let C be a collective in which

1. each agent $A ^ { k }$ has bounded state space $B ^ { k }$ ;

2. the translations (80) are non-trivial and partial;

3. there is a memory Mem<sub>C</sub>, external to every agent, whose contents persist independently of the agents’ states, and whose size is not bounded in advance;

4. agents may read from and write to Mem<sub>C</sub>.

Then the collective semantic history is not bounded by the state space of any individual agent, nor by $\textstyle \prod _ { k } B ^ { k }$

Proof. The collective’s state at time t is the pair $\big ( ( O _ { t } ^ { 1 } , \dots , O _ { t } ^ { N } ) , \mathrm { M e m } _ { C } ( t ) \big )$ . By Condition (3), the second component is not drawn from a fixed finite set, so the collective state space is not $\textstyle \prod _ { k } B ^ { k }$ and the cycling argument above does not apply. By Condition (2) and (84), a successor constructed by $A ^ { i }$ and written to Mem<sub>C</sub> may be read by $A ^ { j }$ and interpreted under a diferent inferential system, yielding by (44) successors unavailable within ${ \mathcal { F } } ^ { i }$ . By Condition (3), this content persists after $A ^ { i }$ has moved to another state, so the accumulated history is not bounded by any $B ^ { k }$ □

While Proposition 4 lifts the state-space bound at the level of the collective through Mem $C ^ { - }$ realizing route (3) of §5.5—the coupling of heterogeneous agents also explains how an individual bounded agent $A ^ { j }$ can experience a local semantic trajectory $H _ { n } ^ { j } = ( \mathcal { F } _ { 0 } ^ { j } , \ldots , \mathcal { F } _ { n } ^ { j } )$ satisfying Assumption 5 relative to its own internal description, and why teleological diversity across agents is necessary to sustain that novelty.

Corollary 1 (Cross-Agent Algorithmic Novelty and Teleological Divergence). Let $C = \{ A ^ { 1 } , \ldots , A ^ { N } \}$ be a collective satisfying the hypotheses of Proposition $\not \langle 4 ,$ in which each bounded agent $A ^ { k }$ operates with internal transition rule $\Theta ^ { k }$ , local history $H _ { n } ^ { k }$ , and active teleological partition $\Phi _ { n } ^ { k } = $ $( \boldsymbol { \Phi } _ { n } ^ { k , + } , \boldsymbol { \Phi } _ { n } ^ { k , - } , \boldsymbol { \Phi } _ { n } ^ { k , \circ } )$ as in (2).

1. (Realization of local novelty via heterogeneous discrimination.) Suppose $A ^ { i }$ and $A ^ { j }$ hold distinct teleological partitions $\Phi _ { n } ^ { i } \neq \Phi _ { n } ^ { j }$ such that $A ^ { i }$ admits environmental signals $D _ { n } ^ { i } \backslash D _ { n } ^ { j } \neq \emptyset$ excluded by $A ^ { j }$ under (4). When $A ^ { i }$ constructs a decisive extension ${ \mathcal { F } } ^ { \prime } \in { \mathrm { ~ S u r p } } ^ { \mathrm { d e c } } ( O _ { n , t } ^ { i } )$ encoding distinctions from $D _ { n } ^ { i } \setminus D _ { n } ^ { j }$ and transmits it across $\sigma _ { n } ^ { i  j }$ in (84) (or deposits it in Mem<sub>C</sub>), any successor $\mathcal { F } _ { n + 1 } ^ { j }$ constructed by $A ^ { j }$ that incorporates $\sigma _ { n } ^ { i  j } ( { \mathcal { F } } ^ { \prime } ) \cap I ^ { \mathrm { i n c } } ( O _ { n , t } ^ { j } )$ satisfies

$$
K \big ( \mathcal { F } _ { n + 1 } ^ { j } ~ \big \mid ~ H _ { n } ^ { j } \big ) ~ \ge ~ K \big ( \sigma _ { n } ^ { i \to j } ( \mathcal { F } ^ { \prime } ) ~ \big \mid ~ H _ { n } ^ { j } \big ) - O ( 1 ) .\tag{87}
$$

In particular, whenever the stream of cross-agent and memory-mediated distinctions drawn from $\textstyle \bigcup _ { i \neq j } ( D _ { n } ^ { i } \setminus D _ { n } ^ { j } )$ carries positive asymptotic algorithmic density $c > 0$ conditioned on $( \Theta ^ { j } , \sigma _ { 0 } ^ { j } ) , A ^ { j }$ ’s realized trajectory $H _ { n } ^ { j }$ satisfies the amortized novelty condition (73) (and, when uniform, Assumption 5), despite A<sup>j</sup> having a finite internal state space $B ^ { j }$

2. (Collapse under teleological homogenization.) Conversely, if all agents share an identical initial formal system $\mathcal { F } _ { 0 }$ and deterministic proof searcher (as in an ensemble of identical G¨odel machines [Schmidhuber, 2006]), or if multi-agent alignment drives every agent’s value selector toward a common stationary distribution $P _ { V } ^ { * }$ at which ”the value system of the entire intelligent population may converg ${ \ ? } { \stackrel { \prime } { \prime } } \left[ H e \right.$ et al., 2026, p. 178], then the symmetric diference of admitted domains vanishes $( D _ { n } ^ { i } \triangle D _ { n } ^ { j } \to \emptyset )$ , the cross-agent conditional complexity collapses to $K ( \mathcal { F } _ { n , t } ^ { i } \mid H _ { n } ^ { j } ) \leq O ( \log n )$ , and inter-agent coupling can no longer supply the irreducible information required by Assumption 5 or (73).

Proof. For (1), since $A ^ { j }$ reconstructs $\sigma _ { n } ^ { i  j } ( { \mathcal { F } } ^ { \prime } ) \cap I ^ { \mathrm { i n c } } ( O _ { n , t } ^ { j } )$ from $\mathcal { F } _ { n + 1 } ^ { j }$ and $H _ { n } ^ { j }$ via the partial translation (80), standard subadditivity of prefix complexity gives (87). Because the signals in $D _ { n } ^ { i } \setminus D _ { n } ^ { j }$ were excluded from $A ^ { j } \mathrm { { ^ , } } { } _ { \mathrm { s } }$ own experience domain $D _ { n } ^ { j }$ by (4), they are not computable from $( \Theta ^ { j } , \sigma _ { 0 } ^ { j } )$ alone; accumulating those inputs across n extensions therefore drives lim in $\mathrm { f } _ { n \to \infty } K ( H _ { n } ^ { j } ) / n \geq c > 0$ For (2), when all agents execute the same deterministic transition rule from identical initial axioms $\mathcal { F } _ { 0 }$ or converge to a shared value specification $P _ { V } ^ { * }$ with $D _ { n } ^ { i } = D _ { n } ^ { j }$ , the state $\mathcal { F } _ { n , t } ^ { i }$ is computable from $H _ { n } ^ { j }$ together with the step index $( n , t )$ , bounding $K ( \mathcal { F } _ { n , t } ^ { i } \mid H _ { n } ^ { j } )$ by $O ( \log ( n + t ) )$ and reducing the coupled ensemble to the single closed regime of Theorem 1. □

## 6.4 Collective Memory as Extended Cognitive Space

Proposition 4 makes memory constitutive rather than convenient. Mem<sub>C</sub> holds pairs

$$
( x , \xi ) \in \operatorname { M e m } _ { C } ,\tag{88}
$$

where x is a content representation and $\xi$ its position in the collective semantic structure: the originating agent, the inferential system under which it was produced, the transformations preceding it, and those that subsequently depended upon it.

Both components are necessary. Content alone is insuficient because, by (33), the meaning of a representation is not preserved across recursive extension: $\tau _ { n }$ may transform what it carries forward. A representation retrieved without its coordinates cannot be correctly reinterpreted, since there is no way to determine what it meant where it was produced. Collective memory, therefore, preserves both what was generated and where it belongs, and it is reflective adequacy (§4.2) that makes the second component expressible at all.

## 6.5 Collective Teleology

Agents may hold distinct purposes. Writing $\phi _ { t } ^ { k }$ for the active purpose of $A ^ { k }$ in the sense of §3.4, the collective teleological configuration is

$$
\Psi _ { t } = \{ \phi _ { t } ^ { 1 } , \dots , \phi _ { t } ^ { N } \} .\tag{89}
$$

Collective intelligence does not require these to coincide; identity would eliminate a source of semantic diversity, as §6.10 shows. What is required is that the collective be able to evaluate relations among purposes and, where necessary, generate higher-order purposes governing their interaction:

$$
\{ \phi _ { t } ^ { k } \} _ { k } \longrightarrow \Psi _ { t } \longrightarrow \Psi _ { t + 1 } .\tag{90}
$$

A collective teleology thereby becomes an object of meta-reasoning in the same way an individual one does, introducing a further level of recursive structure.

## 6.6 Open-Endedness Beyond the Individual

The individual/collective distinction resolves an apparent physical contradiction. No finite system contains infinite information, so open-ended intelligence cannot mean that a machine holds an infinite state space at any finite time. It means instead that the system can continue constructing successor states without a fixed upper bound on the semantic structures that may become relevant over its lifetime.

At the individual level, this is realized through recursion and co-recursion; at the collective level, through the transfer of surplus between bounded agents and its accumulation in Mem<sub>C</sub>. The architecture is therefore a hierarchy of bounded systems participating in an unbounded process:

$$
\mathrm { b o u n d e d ~ a g e n t s \to \mathrm { s e m a n t i c ~ c o u p l i n g } \to \mathrm { c o l l e c t i v e ~ m e m o r y } }
$$

$$
\mathrm { r e c u r s i v e ~ t r a n s f o r m a t i o n }  \mathrm { n e w ~ s e m a n t i c ~ s t r u c t u r e } .\tag{91}
$$

The physical substrate remains finite at every moment; the semantic process need not terminate.

## 6.7 A Hegelian Interpretation

At the level of the collective trajectory, the trace-objectifying mechanism formalized in $\ S 4 . 2$ expands into the broader triadic movement of Hegel’s speculative logic. As shown in §2 and §4.2, a concept or formal regime cannot be understood as a static identity: determination produces internal diference at the undecidable frontier, and the failed attempt to resolve that diference within the existing regime forces a more comprehensive determination.

Within the architecture, a state $O _ { n , t }$ carries an unresolved region $I ( O _ { n , t } )$ marking the limits of its organization. A recursive extension $\mathcal { R } _ { n }$ constructs a successor that, by reflective adequacy, retains the ability to represent its predecessor while adding resources to address what the predecessor could not,

$$
X _ { 0 } \hookrightarrow X _ { 1 } \hookrightarrow X _ { 2 } \hookrightarrow \cdots\tag{92}
$$

This is a formal analogue of Aufhebung: a determination that is simultaneously preserved, transformed, and integrated into a more comprehensive structure. Specifically, as shown in (39) and Proposition 1, when an object-level proof search on $I ^ { \mathrm { i n c } } ( O _ { n , t } )$ fails to close, its derivation trace $\pi _ { n , t } ( \boldsymbol { x } )$ does not vanish into an empty nothingness [Hegel, 1977, p. 51]; reflective translation $\tau _ { n }$ converts that failed operational form into determinate propositional content at level $n + 1$ , so that “the new object shows itself to have come about through a reversal of consciousness itself” [Hegel, 1977, p. 55] and yields a concept “higher and richer than the preceding — richer because it negates or opposes the preceding and therefore contains $\mathrm { i t } ^ { \dag }$ [Hegel, 2010, p. 33]. The hooked arrows in (92) represent reflective adequacy rather than set-theoretic inclusion, which (33) explicitly denies.

## 6.8 From Understanding to Speculative Intelligence

The formalism suggests three qualitatively diferent regimes.

Identity: local determination. Within a stable regime, cognition proceeds through established representations and contractive transformations (Assumption 4), converging to $\Lambda _ { n }$ . The objective is a reliable determination. This corresponds to what Hegel calls Verstand.

Diference: generation of surplus. Where the current representation meets unresolved ambiguity, more than one admissible transformation may be available. This is Assumption 3, $| \mathrm { S u r p } ( O _ { n , t } ) | \geq$ 2, applied here to the individual agent within a collective. As noted in (50), it does not follow from bare incompleteness alone, but holds constructively by Proposition 1 whenever the agent reflectively reifies its unclosed derivation traces via (39), ensuring that ambiguity yields genuine branching.

Totality: recursive integration. A higher-order transformation may incorporate several previously distinct structures into a new context (56), constructing a framework in which previously separate distinctions can be related. This corresponds to speculative intelligence in the limited sense relevant here: cognition becomes able to treat its own distinctions, ambiguities, and purposes as objects of further determination.

## 6.9 Proxy Teleology and Autonomous Intelligence

In contemporary machine-learning systems, the teleological structure is supplied externally: a human specifies a task, objective function, reward signal, or prompt, and the system transforms within the resulting problem space. We call this proxy teleology. Such a system may reason with great sophistication while the criterion determining what counts as successful reasoning remains outside it.

The present architecture requires more. The teleological principle must itself be accessible to recursive revision, which is what (14) formalizes: a successor purpose is selected by the system, not supplied to it. Note that (14) is relational, so the architecture determines which successor purposes are admissible without determining which is adopted—autonomy here is underdetermination, not arbitrariness.

This does not imply that every goal must be continuously replaced. It requires that the teleological structure remain available as an object of higher-order reasoning. The distinction is therefore

$$
\mathrm { { f i x e d ~ t e l o s } \quad \mathrm { { v s . } \quad \ r e c u r s i v e l y ~ a c c e s s i b l e ~ t e l o s , } }\tag{93}
$$

and the latter is the condition for the architecture developed here.

Even architectures that make value functions an explicit system component remain within proxy teleology whenever those values are fixed from outside: He et al. [2026], for instance, propose to remedy the conflicting values embedded in statistical training corpora by “predefining clear values and principles to guide the behavior of intelligent agents” [He et al., 2026, p. 179], yet a predefined value space V still withholds recursive access to the teleology itself and eliminates at the outset the inter-perspective tension from which collective semantic surplus (§6.2) is generated.

## 6.10 Diversity and Common Ground

Collective recursion imposes a structural requirement on the relation between agents. If all agents generate identical surpluses, interaction adds nothing to what an individual already has; if they share no representational structure, their transformations cannot be combined. Productive interaction requires

$$
\begin{array} { r } { \emptyset \subsetneq \operatorname { S u r p } ( O _ { n , t } ^ { i } ) \cap \operatorname { S u r p } ( O _ { n , t } ^ { j } ) \subsetneq \operatorname { S u r p } ( O _ { n , t } ^ { i } ) \cup \operatorname { S u r p } ( O _ { n , t } ^ { j } ) , } \end{array}\tag{94}
$$

with both inclusions strict. The intersection provides the substrate through which representations can be translated and compared—it is the surplus-level counterpart of the common ground (81). The diference provides the diversity from which transformations arise that neither agent would reach independently.

Redundancy is therefore not the opposite of diversity but its precondition: too much yields convergence without extension, too little prevents interaction altogether. §9.1 develops the consequences of the degenerate case.

## 6.11 Civilizational Co-Recursion

Human civilization instantiates this architecture. Individual minds remain physically bounded while their states are coupled through language, artifacts, institutions, and persistent records, so that a transformation generated by one individual becomes an unresolved problem for another, whose response becomes the starting point for a third. The resulting history persists beyond the lifetime of any participant—condition (3) of Proposition 4 realized in the form of writing.

The limiting collective semantic space may be represented as a direct limit, with two qualifications that the notation alone does not convey. Take the limit over the agent state spaces rather than over the collective states (82). Each agent’s recursive extensions induce total maps

$$
\iota _ { n } ^ { k } \colon X _ { n } ^ { k } \longrightarrow X _ { n + 1 } ^ { k } ,\tag{95}
$$

so that $\left( X _ { n } ^ { k } , \iota _ { n } ^ { k } \right) _ { n \geq 0 }$ is a directed system and

$$
X _ { \infty } ^ { k } = \varinjlim _ { n } \bigl ( X _ { n } ^ { k } , \iota _ { n } ^ { k } \bigr )\tag{96}
$$

is well defined. The inter-agent translations $\sigma _ { n } ^ { i  j }$ are partial by (81) and are therefore carried alongside the limit as structure on the family $\{ X _ { \infty } ^ { k } \} _ { k } ,$ rather than being included in the directed system itself. The collective limit is the family together with that structure,

$$
\mathbb { X } _ { \infty } = \bigl ( \{ X _ { \infty } ^ { k } \} _ { k = 1 } ^ { N } , \{ \sigma _ { \infty } ^ { i \to j } \} _ { i \neq j } \bigr ) ,\tag{97}
$$

in the sense of (82).

The maps (95) are not canonical. A recursive extension supplies a translation $\tau _ { n }$ which, by (33), need not preserve interpretation, and the successor adopted is left underdetermined by $\mathcal { M } _ { n }$ Diferent selections, therefore, yield diferent directed systems. $\mathbb { X } _ { \infty }$ is accordingly the limit of one realized trajectory, not an invariant of the architecture, and by §5.3 there are continuum-many trajectories to choose among.

Subject to these qualifications, a civilization provides a mechanism by which finite cognitive systems participate in a semantic process whose representational history is not bounded by the state of any one component.

## 7 Result

The formal development supports the following general conclusion:

Open-ended intelligence is a property of a recursively extended cognitive process, not of an individual physical substrate. A finite cognitive system can participate in openended intelligence when its semantic representations, memory, transformations, and teleological structures can be recursively coupled to those of other bounded systems.

Individual recursion provides the mechanism of self-extension. Memory preserves the resulting semantic history. Polycontexturality permits multiple interpretations to coexist. Teleology selects which transformations to pursue. Collective coupling allows these processes to cross the boundaries of individual agents.

The resulting architecture replaces the notion of an isolated, infinitely capable intelligence with a more general principle: intelligence can remain physically finite while its semantic process remains open-ended.

## 8 Biological Inspirations and Architectural Motifs

The formal architecture developed in the preceding sections is not intended as a neuroscientific theory. Mammalian cognition nevertheless provides a rich source of biological analogues for several of the architectural principles proposed here. These analogies should not be interpreted as empirical validation of the formal theorems. Rather, they indicate that several of the structural requirements of the present framework have plausible counterparts in biological cognitive systems.

In particular, biological intelligence combines statistical inference with hierarchical organization, distributed representations, structured reference frames, persistent memory, and mechanisms for learning without complete destruction of previously acquired structure. These properties provide useful analogues for the distinction developed in this paper between local semantic stabilization and open-ended semantic extension.

## 8.1 Neural Analogues of Semantic Surplus

Independent of the Free Energy Principle discussed below, several behavioral and neural findings are consistent with cognitive processes resembling those described in the formal sections. The correspondences below are analogical. They are ofered as evidence that the architectural distinctions drawn in §4 have plausible biological counterparts, not as empirical support for the formal claims, which are mathematical and neither require nor admit such support.

Retention of unintegrated information. Nagy et al. [2025] argue that semantic memory continually compresses regularities in the environment while episodic memory preserves surprising or poorly compressed experiences for later reinterpretation. Functionally, this keeps information that does not yet fit the current semantic organization available until a later stage of processing.

The relevant formal counterpart is $I ^ { \mathrm { l o c } }$ of (27): the part of the unresolved region that the system has not yet integrated, as distinct from $I ^ { \mathrm { i n c } }$ , which no amount of local work can reach. The distinction matters for the analogy. What the episodic system retains is material the organism could in principle assimilate but has not, which is precisely the character of $I ^ { \mathrm { l o c } }$ ; nothing in these findings suggests a biological counterpart of formal undecidability, and we do not claim one. What the analogy does support is the architectural point of (47): a system that discards unintegrated information as noise loses the material from which later reorganization would be built.

Generation of remote relations. Herault et al. [2024] report that judging remote concepts as related recruits the default-mode and semantic-control networks, and that this tendency is associated with measures of real-world creative achievement. This is consistent with creative cognition involving the generation of semantic relations extending beyond established associations.

This is closer to semantic surplus in the sense of (34) than the preceding finding is. Surplus is not unresolved content, but the availability of several admissible ways of continuing, and the capacity to relate concepts that the current organization treats as unrelated is a capacity to generate such continuations. The connection is to Proposition 1 and Assumption 3, which require that unresolved structure be reflectively transformed into at least two competing continuations rather than left as a static boundary.

Diferentiation and its costs. Vives et al. [2023] report that individuals more averse to uncertainty tend to diferentiate and expand their semantic representations, trading reduced semantic interference against poorer generalization.

This bears on the trade-of developed in (94) rather than on individual-level surplus, and we note it here because it concerns the same representational machinery. Finer diferentiation increases the distinctions a system can draw while reducing what it can carry across contexts—the individuallevel counterpart of the observation in §6.10 that too much semantic diference prevents interaction while too little prevents generation. That the trade-of appears within a single cognitive system, and varies with a dispositional measure, suggests it is a general property of representational economies rather than an artifact of the collective model.

## 8.2 Statistical Processing with Structured Scafolds

Modern neuroscience provides extensive evidence that the brain performs statistical and probabilistic inference. Predictive processing, probabilistic population coding, and distributed neural representations all contribute to the construction and continuous revision of internal models.

The brain is therefore neither adequately described as a purely symbolic recursive machine nor as a flat statistical classifier. A more useful architectural analogy is a system in which statistical processes operate within hierarchical, distributed, and structurally constrained representations.

This distinction is important for the present framework. Statistical inference can provide the computational substrate for evaluating possible states, predictions, and actions. The recursive architecture, by contrast, concerns how the representational structures within which such computations occur can themselves become objects of transformation.

An influential, although controversial, example is the Thousand Brains Theory [Hawkins, 2021]. The theory posits that the neocortex contains numerous semi-autonomous cortical columns, each constructing predictive models from its own sensorimotor perspective. Communication between columns is proposed to support a distributed consensus process rather than a single centralized representation.

This provides a useful biological analogue for the distinction between local semantic states and meta-level selection. Individual columns can be interpreted, at a highly abstract level, as local predictive models, while their interaction provides an analogue of the comparison and integration of multiple locally generated hypotheses.

## 8.3 Reference Frames as Semantic Coordinates

A particularly relevant aspect of the Thousand Brains framework is its emphasis on reference frames. In this view, knowledge is organized not merely as a collection of features but relative to structured coordinate systems.

Grid cells and place cells are well established in the hippocampal-entorhinal system as mechanisms involved in spatial navigation, and grid-like activity has been reported in several neocortical regions. The stronger hypothesis that grid-cell-like mechanisms are universally present throughout the neocortex remains controversial and should not be treated as an established consensus. We use reference frames as a biological analogue, not as evidence for a literal implementation of the semantic organization described in §6.4.

The architectural idea is nevertheless important, and the formal sections give it a precise motivation. Information stored in memory need not consist solely of content; it may also require information about its relational position within a larger structure—where it originated, which other structures it relates to, and where the current cognitive state lies relative to it. This is the pair (88),

$$
( x , \xi ) \in \operatorname { M e m } _ { C } ,
$$

whose second component §6.4 identifies as a requirement rather than a convenience. The reason is (33). A recursive extension supplies a translation $\tau _ { n }$ that need not preserve the interpretation of what it carries forward, so the meaning of a stored representation is not invariant across the sequence (60). A representation retrieved without its coordinates is therefore not merely harder to use; it is ambiguous, since there is no way to recover what it meant in the regime that produced it. Content-only storage is adequate for a system that never changes regime and inadequate for one that does.

The biological analogy is consequently relevant beyond spatial memory. If cognition organizes knowledge relative to reference frames rather than as unstructured collections of features, it exhibits the kind of storage that recursive extension requires. Within the Thousand Brains framework, displacement-like representations are further proposed to encode relations between reference frames—which, as an engineering analogy, suggests that relations between structures could themselves serve as coordinates for higher-order representations, and so provides a possible analogue of the nested organization described in §4.5.

We note one connection to teleology without developing it. Semantic coordinates describe where a system’s knowledge is situated; they do not by themselves determine which transformations are worth pursuing. The selection operator (43) requires a criterion, and coordinates supply at most part of what such a criterion would range over. What the analogy suggests is an engineering principle about representation: open-ended intelligence requires storage that encodes both semantic content and the relational structure needed to interpret it across successive regimes. Whether reference frames also provide the substrate on which selection operates is a further question this framework does not settle.

## 8.4 Geometric and Multi-Scale Organization

The mathematical framework developed above uses geometric concepts such as semantic manifolds, local contraction, attractors, and successor spaces. Biological systems exhibit several forms of geometric and multi-scale organization that serve as useful analogues, although these phenomena should not be conflated.

Three distinct classes are particularly important:

1. periodic geometric structures used for coordinate encoding;

2. statistical self-similarity and scale-free dynamics; and

3. morphological branching structures found in neural anatomy.

These phenomena have diferent mathematical properties and should not all be described as fractality.

Periodic Coordinate Systems. Grid cells in the medial entorhinal cortex provide a prominent example of a biological coordinate system. Individual grid cells exhibit spatially periodic activity, with firing locations forming an approximately triangular lattice. These structures are periodic and geometric rather than fractal. Their relevance to the present framework lies in illustrating how biological cognition can organize information relative to structured coordinate systems rather than relying exclusively on undiferentiated feature representations. Within the Thousand Brains framework, displacement-like representations are further proposed to encode relations between different reference frames. As an engineering analogy, such representations suggest that relations between objects could themselves serve as coordinates for higher-order representations. This provides a possible biological analogue of nested semantic representations, in which the output of one representational level becomes part of the coordinate system of another.

Morphological Fractality. Fractal structure is also widespread in neural anatomy. Dendritic trees exhibit recursive branching geometries that increase receptive surface area while constraining physical volume. Similarly, cortical folding exhibits statistical scaling properties that contribute to the eficient packing of cortical tissue. These phenomena should not, however, be interpreted as evidence that cognition itself operates through fractal semantic spaces. They are primarily morphological adaptations to physical and developmental constraints. Their relevance to the present framework is therefore limited to a more general observation: biological systems repeatedly employ recursive organization to construct complex structures from bounded local components.

Scale-Free Neural Dynamics. Neural activity also exhibits statistical scale invariance. Electrical recordings such as EEG often display approximately scale-free or $( 1 / f )$ -like spectral dynamics across temporal scales. Such dynamics are compatible with a system capable of operating across multiple temporal regimes and levels of organization. They do not, however, establish deterministic fractal computation or the specific Iterated Function Systems used in the formal model. The appropriate correspondence is therefore again analogical: biological systems exhibit multi-scale organization, while the formal model provides a mathematical abstraction of how local and global levels of organization might be coupled.

## 8.5 Hierarchical Constraints in Language

Language provides another potentially relevant biological analogue. Human language processing combines distributed statistical representations with hierarchical and constituent-based organization.

Cai et al. [2026] report evidence from single-neuron recordings in the human frontotemporal cortex that diferent neural populations participate in representing grammatical roles, phrase transitions, and larger syntactic structures during speech production. Such findings are consistent with multiple levels of linguistic organization rather than a purely flat sequence of independent word predictions. Similarly, Zou et al. [2026] report evidence that neural responses to linguistic surprisal are modulated by grammatical constituent boundaries. This suggests that prediction in the human brain is constrained by hierarchical linguistic structure.

The key point is that symbolic grammar does not replace statistical learning; instead, biological language processing serves as an analogy for a system in which statistical prediction operates within a structured, recursively organized framework. This concept is closely tied to the distinction made in this work between object-level processing and meta-level transformation. Statistical mechanisms can operate within a specific representational language, while higher-order mechanisms can work on the structures that organize these computations.

## 8.6 Biological Safeguards for Recursive Learning

The brain also ofers helpful analogies for the challenge of maintaining previously acquired knowledge while engaging in continuous learning.

The developmental metaphor of a tabula plena emphasizes that biological learning does not begin from a completely unstructured state. Rather, early neural circuits may contain dense connectivity that is subsequently refined through developmental processes such as selective pruning. Vargas-Barroso et al. [2026] provide an example in hippocampal CA3 networks in which early circuitry exhibits dense connectivity that is subsequently reorganized.

In our proposed framework, we emphasize that a robust initial structure can coexist with subsequent specialization. A recursively evolving system need not build every representational element from scratch. Instead, it can start with a pre-existing structural foundation and gradually reorganize it over time.

A second important mechanism relates to the protection of consolidated information during online learning. Gonzalez et al. [2026] describe a mechanism in the hippocampal-retrosplenial axis that involves distinct activity patterns within the same physical neuronal populations. These findings suggest that partially overlapping physical substrates can support functionally distinct representational subspaces. This concept serves as a useful analogy in engineering for recursive selfmodification. A system should be able to modify its representations without necessarily overwriting structures that remain valuable for future recursion. In this framework, this means preserving semantic history while allowing new semantic manifolds to emerge.

The biological analogy, therefore, supports a broader architectural principle: recursive learning requires both plasticity and protection. A completely rigid system cannot generate successor structures, while a system that continuously overwrites its own representational foundations cannot preserve the semantic history required for meaningful recursion.

## 8.7 Relationship to the Free Energy Principle

The Free Energy Principle (FEP) provides a related but distinct theoretical perspective on the organization of biological intelligence [Friston, 2010, Friston et al., 2006, Parr et al., 2022].

Under FEP, a cognitive system minimizes variational free energy, which provides an upper bound on sensory surprise, by maintaining internal probabilistic models of hidden causes of observations. Let o denote observations and z latent causes. The system maintains an approximate posterior $q ( z )$ over latent states that explains the observed data. Because latent variables are not directly contained in the sensory signal, the construction of $q ( z )$ already introduces internal structure beyond the immediately observed data. Hierarchical predictive processing extends this principle across multiple levels of abstraction, while active inference extends it to action by allowing the system to alter its environment in ways that reduce expected free energy [Parr et al., 2022].

FEP, therefore, provides a plausible biological account of how a bounded system can construct internal structure that goes beyond its immediate sensory input. There is, however, an important distinction between FEP and our framework. FEP primarily characterizes principles of inference and action within a generative model. The present framework concerns the conditions under which the representational structure itself can become an object of recursive modification. The distinction can be expressed schematically as

FEP: reduce expected prediction error within a generative model,

whereas our framework additionally requires

Epistemic Drive: identify productive unresolved structure and transform the model itself.

These objectives are not necessarily contradictory. A system may reduce prediction error within a current semantic manifold while simultaneously using persistent prediction errors, model failures, or unexplained observations as signals to extend its representation.

The critical distinction is therefore between local inference and representational evolution. Within a fixed semantic manifold $X _ { n }$ , prediction and inference may be strongly contractive. At the meta-level, persistent ambiguity can become an input to the construction of a successor manifold $X _ { n + 1 }$ . This suggests a testable empirical hypothesis. During periods of genuine conceptual innovation, neural processes associated with stabilization and compression of existing representations should coexist with processes associated with the generation of new semantic associations and the maintenance of unresolved alternatives. The relevant empirical question is therefore not whether the brain minimizes prediction error, but how prediction-error reduction is coupled to the formation and transformation of representational structures.

## 8.8 Implications for the Present Architecture

Taken together, these biological observations suggest several architectural motifs relevant to the formal theory:

1. Local stabilization. Biological systems construct stable predictive representations that permit eficient inference within a local regime.

2. Distributed perspectives. Multiple partially autonomous representations can coexist and interact rather than being immediately reduced to a single centralized state.

3. Reference frames. Information can be organized according to structured coordinates, creating an analogy of semantic navigation through relational state spaces.

4. Semantic preservation. Surprising or unresolved information can remain available for later reinterpretation rather than being immediately discarded.

5. Recursive hierarchy. Representations can participate in multiple levels of organization, allowing local structures to become components of higher-order representations.

6. Plasticity with protection. Learning can modify internal structure while preserving previously consolidated information.

7. Compression with extension. Eficient cognition requires the compression of established structure, while open-ended cognition requires mechanisms that extend the representational system when the existing structure becomes insuficient.

These observations do not prove the mathematical theory. They suggest that the main diference in this work—between local thinking that is limited and a flexible, ongoing change in how we understand things—has several likely examples in biology. The strongest implication is therefore not that the brain literally implements semantic manifolds, direct limits, or co-recursive operators. Rather, biological intelligence demonstrates that a finite physical substrate can support apparently open-ended cognitive development through the combination of bounded local mechanisms, hierarchical organization, persistent memory, structured reference frames, and distributed interaction.

This distinction is especially important for AGI. The physical structure of each artificial agent, like that of the human brain, is inherently finite. Thus, open-endedness does not necessarily imply an infinite internal state space. Instead, it can emerge from the recursive construction, preservation, and interaction of finite semantic structures over time and, ultimately, across multiple agents. In this regard, biological intelligence serves as a useful proof of concept for the architectural principle developed in this paper: finite cognitive systems can engage in an open-ended process when their semantic structures can be preserved, recursively transformed, and connected to successor structures through shared representational coordinates.

## 9 Discussion: The Problem of Epistemic Convergence

The formal results suggest that the central challenge of open-ended intelligence is not the accumulation or compression of knowledge but the preservation of the capacity to generate new distinctions.

Three processes should be distinguished. Accumulation is the acquisition and retention of empirical information. Compression is the reduction of representational and search costs by organizing existing information into stable structures: formally, local transformation within a regime, converging by (55) to $\Lambda _ { n }$ . Generation is the construction of new semantic structures when the current system can no longer resolve its ambiguities, which is (44).

Contemporary connectionist systems are efective at the first two. The theoretical question is whether a system optimized primarily for prediction and compression can sustain the third.

## 9.1 Collective Generative Capacity

Let $C = \{ A ^ { 1 } , \ldots , A ^ { N } \}$ be a collective in the sense of §6.1. Not every admissible successor an agent could construct is one it will pursue; selection by (43) narrows the surplus to those the agent’s teleological organization admits. Write

$$
\mathrm { S u r p } _ { n , t } ^ { k } = \mathcal { M } _ { n } ^ { k } \big ( \mathrm { S u r p } ( O _ { n , t } ^ { k } ) \big )\tag{98}
$$

for the pursued surplus of agent $A ^ { k } ;$ the continuations that survive its own selection. The collective pursued surplus is

$$
\mathrm { S u r p } _ { C } ( n , t ) = \bigcup _ { k = 1 } ^ { N } \mathrm { S u r p } _ { n , t } ^ { k } ,\tag{99}
$$

and we write

$$
\mathrm { G e n } _ { C } ( n , t ) = \left| \mathrm { S u r p } _ { C } ( n , t ) \right|\tag{100}
$$

for the number of distinct continuations available to the collective.

The degenerate case. If all agents pursue the same continuations,

$$
\operatorname { S u r p } _ { n , t } ^ { i } = \operatorname { S u r p } _ { n , t } ^ { j } \qquad \forall i , j \in \{ 1 , \dots , N \} ,\tag{101}
$$

then (99) collapses and

$$
\mathrm { G e n } _ { C } ( n , t ) = \left| \mathrm { S u r p } _ { n , t } ^ { i } \right| \qquad \mathrm { f o r ~ a n y } ~ i .\tag{102}
$$

Adding agents does not expand the number of distinct semantic possibilities. This statement is an identity rather than a conclusion, and we mention it solely to establish the scale’s endpoint; total identity is an idealized limit rather than a condition that any real collective achieves.

The interesting case is partial correlation. What matters is not whether agents are identical but how much their pursued surpluses overlap. Two quantities move in opposite directions as overlap increases. Generative capacity Gen $c ( n , t )$ falls, since the union of highly similar sets is little larger than any member. But the translatability on which §6.3 depends rises, since by (81) an extension generated by $A ^ { i }$ becomes available to $A ^ { j }$ only within the fragment $L _ { n } ^ { i , j }$ that $A ^ { j }$ can interpret.

The condition in (94) of §6.10 requires strict inclusions, meaning we need suficient overlap for transfer and distinctiveness for generation. Both extremes are unproductive. The term $G e n _ { C }$ does not fully capture this balance, as it is maximized by complete divergence, which, as stated in (81), would hinder the collective from combining efectively. Thus, we should view generative capacity as part of a trade-of rather than as a quantity to maximize. This section aims to provide a qualitative understanding rather than proposing a scalar value that combines the two aspects.

## 9.2 AI and the Convergence of Semantic Space

This creates a structural concern for increasingly centralized information systems. If many agents rely on similar models, datasets, optimization criteria, and feedback mechanisms, their pursued surpluses may become increasingly correlated—not identical, but overlapping enough that ${ \mathrm { G e n } } _ { C } ( n , t )$ approaches $| \mathrm { S u r p } _ { n , t } ^ { i } |$ for a typical i. The relevant quantity is not the amount of information available to the agents but the degree of independence among the continuations they pursue.

Two distinct mechanisms could produce this. Agents may generate similar surpluses because they share architecture and training data, so that $\operatorname { S u r p } ( O _ { n , t } ^ { k } )$ is already similar across k. Or they may generate diverse surpluses but select similarly, so that the $\mathcal { M } _ { n } ^ { k }$ of (98) converge even where the underlying spaces do not. The second is the more likely route under shared evaluation criteria and reward models, and is harder to detect, since diversity remains in principle while disappearing from what is pursued.

This does not make consensus undesirable. Shared representation is indispensable for communication, coordination, and accumulation—it is the intersection in (94), without which nothing transfers. The concern is narrower: that compression may eliminate the diferences on which further synthesis depends. Schematically,

$$
{ \mathrm { c o m p r e s s i o n } }  { \mathrm { c o n v e r g e n c e } }  { \mathrm { r e d u c e d ~ d i v e r s i t y } }  { \mathrm { r e d u c e d ~ g e n e r a t i v e ~ c a p a c i t y } } .\tag{103}
$$

We do not claim that contemporary systems inevitably produce this outcome. The claim is that an information system that consistently rewards convergence lacks a mechanism to preserve the diversity that collective generation requires, and that this is a structural property of the reward, not a contingent feature of any implementation.

## 9.3 Epistemic Convergence and Open-Endedness

The concern connects directly to the definition of open-endedness. An open-ended system must generate successor structures when the current one proves insuficient. If all agents converge on the same regime and the same admissible transformations, collective interaction does not by itself produce a new successor: by (102) the collective’s options are a single agent’s options, and Proposition 4 then adds nothing that Theorem 1 does not already exclude.

The distinction here lies between performing more computations within a given space and expanding that space itself. A system can become very eficient at solving problems that can be expressed in $X _ { n }$ without necessarily gaining the ability to develop $X _ { n + 1 }$ . This development requires that unresolved structures remain available as resources for transformation. This introduces a second role for surplus: at the individual level, it provides alternatives for local reasoning, while at the collective level, its distribution among agents creates the variability necessary for new structures to emerge. Therefore, collective open-endedness relies not on the size of any individual agent’s state space but on the nature of interactions among them.

A parallel phenomenon appears in the multi-agent mean-field analysis of He et al. [2026], where a population’s skill, value, and cognitive distributions converge to stationary limits $( P _ { U } ^ { * } , P _ { V } ^ { * } , P _ { C } ^ { * } )$ once “the value system of the entire intelligent population may converge, leading to the belief that there is no need to continue learning” [He et al., 2026, p. 178]. Equations (102) and (103) show why such collective value convergence is a structural failure mode rather than a desirable alignment equilibrium: without an epistemic drive (§9.4) that continually generates new teleological distinctions across agents, shared optimization collapses ${ \mathrm { G e n } } _ { C } ( n , t )$ to the ceiling of a single closed system.

## 9.4 The Epistemic Drive

Within this framework, an intelligent system’s task is not exhausted by resolving ambiguity. It must also determine which ambiguities are worth pursuing, and by (26), that choice ranges over two regions of quite diferent character. A proposition in $I ^ { \mathrm { l o c } } ( O _ { n } )$ can be settled by local work within the current regime; one in $I ^ { \mathrm { i n c } } ( O _ { n } )$ can be addressed only by constructing a successor. A system that attends exclusively to the first never extends its regime, while one that attends exclusively to the second never consolidates what it has.

The resulting distinction is between two forms of optimization,

$$
\mathrm { e x p l o i t a t i o n } \longrightarrow \mathrm { r e s o l u t i o n ~ w i t h i n } ~ X _ { n } ,\tag{104}
$$

$$
\mathrm { e x p l o r a t i o n \longrightarrow t r a n s f o r m a t i o n ~ t o w a r d ~ } X _ { n + 1 } ,\tag{105}
$$

and neither replaces the other. Exploitation provides the local stability of (55); exploration enables semantic growth. An open-ended system must maintain a recursive relation between them. While the contingent first-order content of $\Phi _ { 0 }$ remains open to the particular agent or collective, the structural coupling between the two modes is fixed by the architecture: the a priori meta-teleological trigger (Definition 3 and Proposition 3 in §4.7) governs the switch from local exploitation within $X _ { n }$ to co-recursive exploration toward $X _ { n + 1 }$ whenever local decidable structure saturates near $\Lambda _ { n } .$ This also separates structural exploration from expected-utility exploration within a fixed formalism: a G¨odel machine, for instance, “embodies a principled way of dealing with the exploration vs exploitation problem” [Schmidhuber, 2006, p. 17] only when its proof searcher can derive from the initial axioms that “the expected utility . . . of conducting some experiment may exceed the one of not conducting it” [Schmidhuber, 2006, p. 17], whereas transformation toward $X _ { n + 1 }$ in response to $I ^ { \mathrm { i n c } } ( O _ { n } )$ cannot be deductively certified within ${ \mathcal { F } } _ { n }$ prior to adopting the successor regime.

This marks a fundamental inversion in the role of computation on unanswerable problems. In contemporary AI systems—and even in deductively gated self-improvers such as the G¨odel machine—expending compute on a question that cannot be answered within the current representational framework is purely a waste of resources, to be suppressed by early stopping, timeouts, or reward penalties once search fails to return an object-level solution. In the present architecture, by contrast, spending compute on the unsolvable region $I ^ { \mathrm { i n c } } ( O _ { n } )$ is the primary generative engine of the system. Because a non-omniscient agent cannot know prior to trying whether a question belongs to $I ^ { \mathrm { l o c } } ( O _ { n } )$ or $I ^ { \mathrm { i n c } } ( O _ { n } )$ , it must invest computation in attempting a derivation; and by Propositions 1 and 2, when that derivation stalls against $I ^ { \mathrm { i n c } } ( O _ { n } )$ , the resulting proper reduction trace $\pi _ { n , t } ( \boldsymbol { x } )$ exposes non-trivial blocking premises $b \in B _ { n , t } ( x ) \setminus \{ x , \neg x \}$ whose reflective translation $\tau _ { n }$ converts the failed operational form into the semantic content of candidate successor regimes $X _ { n + 1 }$ . Without expending compute on problems that resist resolution in $X _ { n }$ , no non-trivial blocking frontier $B _ { n , t } ( x )$ is articulated and the co-recursive transition $X _ { n } \to X _ { n + 1 }$ cannot take place.

## 9.5 Implications for AGI

This yields a more precise account of general intelligence. It does not require eliminating ambiguity. It requires the capacity to recognize, preserve, and transform it—not merely to answer questions but to construct frameworks in which previously unanswerable questions become tractable. The pivotal sequence is therefore

$$
X _ { n } \longrightarrow X _ { n + 1 } \longrightarrow X _ { n + 2 } \longrightarrow \cdots\tag{106}
$$

rather than indefinite optimization within a fixed $X _ { n }$

Each component of the architecture has a defined role in sustaining this sequence. Teleology determines which transformations are pursued (43). Memory preserves the semantic history and the coordinates that make it interpretable (88). Polycontexturality maintains multiple partially independent interpretations. Surplus supplies the alternatives from which transformations are constructed (34). Recursion connects these into a single process of self-extension.

Because open-ended intelligence is driven by the productive failure of computation at the undecidable frontier, the quality of an AI system can no longer be defined or measured as a static snapshot, such as a score on a fixed benchmark. Any static benchmark evaluates a system within a single frozen regime $( X _ { n } , d _ { n } )$ , measuring how closely its current state $O _ { n , t }$ approximates the local attractor $\Lambda _ { n }$ over decidable problems $I ^ { \mathrm { l o c } } ( O _ { n } )$ under an externally fixed purpose $\phi _ { n }$ . By Theorem 1 and Proposition 3, however, evaluating a single snapshot of $X _ { n }$ mistakes local contraction toward $\Lambda _ { n }$ for general intelligence while treating the non-terminating exploration on $I ^ { \mathrm { i n c } } ( O _ { n } )$ that generates $\mathrm { S u r p } ^ { \mathrm { d e c } } ( O _ { n , t } )$ as unproductive overhead. Just as Hegel insists that the truth of cognition is never a bare static result severed from its becoming, but “the result together with the process through which it came about” [Hegel, 1977, p. 2], only the movement itself—the trajectory Traj(F ) across recursive regime transitions $X _ { n } \to X _ { n + 1 }$ in (106) and the conditional algorithmic novelty $K ( \mathcal { F } _ { n + 1 } \mid H _ { n } )$ it sustains—can serve as the unit of evaluation.

This conclusion speaks directly to the contemporary debate over AGI evaluation and to the Tong Test proposed by Peng et al. [2024]. Peng et al. [2024] rightly diagnose why neither the classical Turing test nor finite task-oriented benchmarks can capture general intelligence, noting that whereas narrow AI focuses on data generalization, “the generality of AGI emphasizes task generalization. Here, task generalization means that AGI must be able to adapt and behave well in a dynamic environment in which it may encounter an infinite number of unexpected scenarios” [Peng et al., 2024, p. 12], and that “[i]f N tasks do not constitute ‘general’ intelligence, then neither do N + 1 tasks” [Peng et al., 2024, p. 14]. By grounding evaluation in “dynamic embodied physical and social interactions (DEPSI)” [Peng et al., 2024, p. 12], “shifting from task-oriented to ability- and value-oriented evaluations of AGI” [Peng et al., 2024, p. 16], and using open-ended scenarios—“[b]y designing testing scenarios that do not specify clear task goals, the values of AGIs can be observed from their autonomous behaviors” [Peng et al., 2024, p. 18]—so that an agent must “go beyond human-defined tasks and be self-driven (i.e., initiating actions driven by its own ‘purpose’)” [Peng et al., 2024, p. 14], the Tong Test moves in the right direction away from static input–output snapshots. Yet the Tong Test still realizes its infinite task generation “by sampling configurations (e.g., combinations of diferent objects, physical and social fluent states, and actions) in the continuous space of DEPSI environments” [Peng et al., 2024, p. 17] over a predefined “parse graph with fluent space” [Peng et al., 2024, p. 17], and still condenses the evaluation into a static profile in which “the AGI agent yields a set of scores representing the estimated ability and value levels across individual dimensions” [Peng et al., 2024, p. 18] across five predefined ability and value milestones. In the terms of §3.4 and $\ S 4 .$ , combinatorially sampling tasks from a fixed parse-graph fluent space and scoring agents against a pre-established five-level value hierarchy still measures coverage within a single predetermined representational regime $X _ { n } ,$ remaining subject to the closure bound of (13). Evaluating open-ended intelligence requires taking the dynamic impulse of the Tong Test one step further: rather than scoring where an agent settles on a fixed capability-and-value radar plot after solving combinatorially sampled tasks, evaluation must observe the recursive movement $X _ { n } \to X _ { n + 1 }$ itself—testing whether, when confronted with problems that cannot be solved within its current parse graph and value coordinates, the system expends compute on that unsolvable frontier and reifies the resulting failed derivation traces into genuinely new representational primitives and purposes.

The central implication is accordingly:

Open-ended intelligence involves recursively transforming ambiguity into a new semantic   
space rather than eliminating it.

The objective of such a system is not to reduce the range of possible meanings as far as possible, but to create and maintain the conditions under which productive meanings can continue to emerge.

## References

Dominik Bartmanski. Imbricated–a conceptual morphology of polycontexturality. Sozialraum. de, 2021.

Jing Cai et al. Mapping the neuronal building blocks of human language with language models. Nature, 2026. doi: 10.1038/s41586-026-10691-5.

Karl Friston. The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11 (2):127–138, 2010. doi: 10.1038/nrn2787.

Karl Friston, James Kilner, and Lee Harrison. A free energy principle for the brain. Journal of Physiology-Paris, 100(1-3):70–87, 2006. doi: 10.1016/j.jphysparis.2006.10.001.

Kurt G¨odel. Uber formal unentscheidbare s¨atze der Principia Mathematica und verwandter Systeme <sup>¨</sup> I. Monatshefte f¨ur Mathematik und Physik, 38:173–198, 1931.

Joaquin Gonzalez, Mihaly V”or”oslakos, Deniz Aykan, Natalie Soto, Noa Nitzan, Roman Swanson, Melis Karadas, Zhe S. Chen, and Gy”orgy Buzs´aki. Subspace communication in the hippocampal–retrosplenial axis. Nature, 2026. doi: 10.1038/s41586-026-10481-z.

Gotthard G¨unther. Life as poly-contexturality. Beitr¨age zur Grundlegung einer operationsf¨ahigen Dialektik, 2:283–306, 1979. Page citations refer to the digital edition in vordenker.de (J. Paul, Ed., 2004, pp. 1–15).

Jef Hawkins. A thousand brains: A new theory of intelligence. Basic books, 2021.

Di He, Cong Fang, Yisen Wang, Yujia Peng, Yizhou Wang, and Song-Chun Zhu. A mathematical formulation of AGI in the (C, U, V) framework. Engineering, 62:166–180, 2026.

Georg Wilhelm Friedrich Hegel. Phenomenology of Spirit. Oxford University Press, 1977. Translated by A. V. Miller with Analysis of the Text and Foreword by J. N. Findlay. Original work published 1807.

Georg Wilhelm Friedrich Hegel. The Science of Logic. Cambridge University Press, 2010. Translated by G. di Giovanni. Original work published 1812–1816.

Caroline Herault, Marcela Ovando-Tellez, Izabela Lebuda, Yoed N. Kenett, Benoit Beranger, Mathias Benedek, and Emmanuelle Volle. Creative connections: the neural correlates of semantic relatedness are associated with creativity. Communications Biology, 7:810, 2024. doi: 10.1038/s42003-024-06493-y.

Immanuel Kant. Critique of Pure Reason. Cambridge University Press, 1998. Translated by P. Guyer and A. Wood. Original work published 1781.

Niklas Luhmann. Die gesellschaft der gesellschaft. Tijdschrift Voor Filosofie, 61(2), 1999.

Marvin Minsky. The Society of Mind. Simon & Schuster, 1986.

Tom M Mitchell. Machine learning, volume 1. McGraw-hill New York, 1997.

David G. Nagy, Gerg˝o Orb´an, and Charley M. Wu. Adaptive compression as a unifying framework for episodic and semantic memory. Nature Reviews Psychology, 4:484–498, 2025. doi: 10.1038/ s44159-025-00458-6.

Thomas Parr, Giovanni Pezzulo, and Karl J. Friston. Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press, 2022.

Yujia Peng, Jiaheng Han, Zhenliang Zhang, Lifeng Fan, Tengyu Liu, Siyuan Qi, Xue Feng, Yuxi Ma, Yizhou Wang, and Song-Chun Zhu. The Tong test: Evaluating artificial general intelligence through dynamic embodied physical and social interactions. Engineering, 34:12–22, 2024. doi: 10.1016/j.eng.2023.07.006.

J¨urgen Schmidhuber. Algorithmic theories of everything. arXiv preprint quant-ph/0011122, 2000.

J¨urgen Schmidhuber. G¨odel machines: Self-referential universal problem solvers making provably optimal self-improvements. arXiv preprint cs/0309048, 2006.

Vier Vargas-Barroso, Jason F. Watson, Andrea Navas-Olive, Alois Schl”ogl, and Peter Jonas. Developmental emergence of sparse and structured synaptic connectivity in the hippocampal CA3 memory circuit. Nature Communications, 2026. doi: 10.1038/s41467-026-71914-x.

Marc-Llu´ıs Vives, Daantje de Bruin, Jeroen M. van Baar, Oriel FeldmanHall, and Apoorva Bhandari. Uncertainty aversion predicts the neural expansion of semantic representations. Nature Human Behaviour, 7:765–775, 2023. doi: 10.1038/s41562-023-01561-5.

Jing Zou, David Poeppel, and Nai Ding. Constituent-constrained word prediction during language comprehension. Nature Neuroscience, 29:1498–1509, 2026. doi: 10.1038/s41593-026-02272-6.