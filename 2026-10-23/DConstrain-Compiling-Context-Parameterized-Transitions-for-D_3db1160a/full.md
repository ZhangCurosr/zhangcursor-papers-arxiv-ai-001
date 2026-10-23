# DConstrain: Compiling Context-Parameterized Transitions for Diffusion Language Model Constrained Decoding

Shidong Cao<sup>1</sup> Jing Ma<sup>1†</sup> Wenhao Chai<sup>2</sup> Yuchen Tian<sup>1</sup> Lun Du<sup>3†</sup> Da Zheng<sup>3†</sup>

<sup>1</sup>Hong Kong Baptist University <sup>2</sup>Princeton University <sup>3</sup>Ant Group

{cssdcao,majing}@comp.hkbu.edu.hk {dulun.dl,zhengda.zheng}@antgroup.com

## Abstract

Diffusion language models (DLMs) decode multiple tokens in parallel, so individually valid tokens can be incompatible together under a schema. Exact joint inference over a grammar graph enforces compatibility, but it requires token-labeled successors across many hypothetical parser configurations. Encoding stacks and field histories in compiled states can cause exponential growth, which forces prior exact decoders to fix property orders and exclude valid outputs. We present DConstrain, which compiles schemas into context-parameterized transition programs. Cold compilation specializes token consumption to local grammar states while retaining context-dependent stack operations, environment guards, and caller-dependent deferred parsing symbolically. Tracking field histories as runtime values supports arbitrary property order without enumerating field subsets offline. At runtime, warm compilation binds reached configurations to these programs to construct exact successors, overlapping with the model forward pass because graph construction requires no model scores. Furthermore, token-effect grouping merges tokens with identical compiled effects before runtime contexts are bound, so groups are reused across stacks and environments even when concrete successors differ. Objective-specific score aggregation and token restoration provably preserve both the Viterbi optimum and the token-labeled path distribution of ungrouped inference. Across three benchmarks and two DLM families, DConstrain compiles 100% of BFCL schemas under arbitrary property orders, whereas an optimized finite-state baseline compiles only 71.30–85.81%. Under arbitrary property order, it reduces BFCL end-to-end decoding latency by 6.65–46.26× with 100% schema validity and raises DiffusionGemma’s BFCL accuracy from 37.27% under fixed order to 81.56%.

## 1 Introduction

Diffusion language models (DLMs) generate text by iteratively denoising multiple tokens in parallel (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024; Nie et al., 2025; Ye et al., 2025). Parallel decoding reduces latency and makes DLMs attractive for agentic tasks such as tool calling (Schick et al., 2023; Qin et al.,

2024), API interaction, and structured data extraction. These tasks require outputs that strictly follow a schema (Patil et al., 2025; Geng et al., 2025), yet unconstrained DLMs often violate schema rules (Suresh et al., 2025; Dang & Ermon, 2026). In autoregressive (AR) models, schema compliance is enforced by masking invalid next tokens for a fixed left-to-right prefix (Willard & Louf, 2023; Moskal et al., 2025; Dong et al., 2025). This mechanism does not carry over directly to DLMs. Because a DLM fills several masked positions at once, two tokens that are each valid on their own can be invalid together.

Existing DLM constrained decoders follow two main paradigms. Verification-based methods sample masked positions independently and then check whether the proposed output can still be completed (Mündler et al., 2026; Zhang et al., 2026; Jin & Han, 2026). Because independent proposals ignore dependencies among positions, decoding many tokens at once causes frequent conflicts and requires repeated resampling or heuristic fallbacks under finite budgets (Mündler et al., 2026; Jin & Han, 2026). Conversely, automata-based methods enforce compatibility through joint inference (Suresh et al., 2025; Dang & Ermon, 2026; Thimmaiah et al., 2026). For the active denoising window (the canvas), they propagate the grammar position by position into a canvas-specific grammar graph whose paths represent sequences of mutually compatible tokens. The decoder then identifies the highest-scoring path or samples according to the graph’s path weights. However, these inference algorithms assume a finite automaton with readily accessible token transitions, leaving the cost of constructing this representation for complex schemas largely unaddressed.

This bottleneck arises because DLM joint inference requires a richer successor-construction interface than AR decoding. An AR decoder only needs a boolean mask of valid next tokens for a single realized prefix (Dong et al., 2025). In contrast, joint inference must compute the exact successor parser configurations for candidate tokens across many hypothetical configurations before any token is committed. Each configuration consists of a local grammar state (a position inside one grammar rule) and a runtime context (a stack of return points and a parsing environment, such as the set of fields already written). Finite-state methods encode runtime contexts as separate automaton states, and DINGO additionally precomputes transitions across the full vocabulary (Suresh et al., 2025). In real schemas, nested objects multiply possible stack contexts, while unordered properties introduce exponentially many field histories. Even context-free grammars (CFGs) can require exponential size to represent all permutations of distinct fields (Ellul et al., 2005). To keep compiled automata tractable, prior exact decoders restrict schemas to a fixed property order (Suresh et al., 2025; Dang & Ermon, 2026; Thimmaiah et al., 2026), excluding valid JSON outputs that the model may prefer.

Our key observation is that context-specific successors need not be compiled as context-specific transitions. Consider a token containing two closing braces, }}, in a nested JSON object. After the first brace closes the inner object, parsing the remaining brace requires returning to the caller. If the parent still lacks a required field, the token is rejected; otherwise, it may close the parent. The return point depends on the stack, and field completion depends on the parsing environment. These contexts can share one compiled procedure: check field completion, return to the caller, and resume parsing the remaining bytes. Rather than precomputing successors for every combination of stack and field history, we compile how to compute them with the runtime context left as a parameter. This separation allows exact successors to be constructed at runtime without enumerating all possible contexts during compilation.

Building on this insight, we propose DConstrain, which compiles schemas into context-parameterized transition programs (Figure 1). Cold compilation specializes token consumption to local grammar states: it precomputes byte-level parsing and local transitions wherever inputs are known, and retains contextdependent stack operations, environment guards, and deferred parsing symbolically. Tracking field histories as runtime values supports arbitrary property order without enumerating field subsets offline. At runtime, warm compilation instantiates these programs for configurations reachable in the current canvas, resolves context-dependent operations, and constructs the exact successor relations required for joint inference. Since warm graph construction does not require model scores, it overlaps with the model forward pass. Finally, token-effect grouping identifies equivalent compiled token effects during cold compilation, before runtime contexts are bound. The resulting groups can be reused across different stacks and environments, sharing successor computation even when the concrete successors differ across contexts. Objective-specific score aggregation and token restoration preserve the exact Viterbi optimum and token-labeled path distribution.

Our contributions are summarized as follows:

• Successor-Construction Bottleneck. We identify token-labeled successor construction as the primary scalability bottleneck in automata-based DLM decoding. Materializing context-specific transitions incurs combinatorial growth with nesting and field histories. Even CFGs can require exponential size for arbitrary permutations of distinct fields, motivating fixed property orders in prior exact decoders.

• Context-Parameterized Compilation. We introduce context-parameterized transition programs paired with a two-stage compiler and compile-time token-effect grouping. We prove that program execution preserves the reference successor relation and that grouping preserves both the Viterbi optimum and the full token-labeled path distribution of ungrouped inference. The compiled programs can be reused by any constrained-decoding algorithm that consumes token-labeled grammar successors.

• Exact Unordered Decoding. Across three structured-output benchmarks and multiple DLM tokenizers, DConstrain substantially improves compilation coverage and latency under arbitrary property order. On BFCL, exact unordered constraints also improve task accuracy over both fixed-order constraints and unconstrained decoding.

## 2 Related Work

Autoregressive Constrained Decoding. Grammar-constrained decoding is widely used to guarantee structured outputs in autoregressive LLMs. At each step, the decoder masks invalid tokens for the realized prefix and advances the parser with the selected token. Early methods introduced incremental parsing and semantic constraints for structured NLP tasks (Scholak et al., 2021; Poesia et al., 2022; Geng et al., 2023). Subsequent work corrected the distributional distortion of token-level masking through adaptive sampling (Park et al., 2024). Later systems focused on efficient tokenizer-aware mask generation (Willard & Louf, 2023; Beurer-Kellner et al., 2024; Koo et al., 2024; Moskal et al., 2025; Dong et al., 2025), and XGrammar 2 added JIT compilation with cross-grammar cache reuse (Li et al., 2026). Our compiler design is closest to two of these systems. XGrammar (Dong et al., 2025) precomputes stack-independent valid tokens at each grammar position and checks only the remaining tokens at runtime. Koo et al. (2024) compose a detokenizing transducer with a pushdown automaton so that each token transition carries explicit stack operations. DConstrain also compiles context-independent work once, but it targets a different successor-construction interface. A validity mask only identifies admissible next tokens for a single prefix. In contrast, DLM joint inference requires the full transition map $( c , v ) \mapsto \mathsf { T } _ { \mathcal { G } } ( c , v )$ across many hypothetical configurations before any token is committed. DConstrain further keeps the written-field history as a runtime parameter, which no compact grammar can encode (Proposition 3.2).

DLM Constrained Decoding. Early diffusion language models supported soft control by steering continuous latents with classifier gradients (Li et al., 2022), but soft guidance cannot guarantee valid syntax. Decoders for hard constraints fall into two families. The first family consists of verification-based decoders, which sample candidate tokens independently and resample or recover when a grammar check fails. Constrained Diffusion (Mündler et al., 2026) checks whether the partial output can still be completed. LAVE (Zhang et al., 2026) validates each proposed token against sampled lookahead completions. EPIC (Jin & Han, 2026) accelerates completion checks with memoized lexing and Earley-style parsing (Earley, 1970) to commit verified token batches in parallel. Although these decoders avoid finite-state materialization, a feasibility check only ensures that some valid continuation exists. At each step, it recovers neither the highest-scoring joint assignment nor an exact sample from the constrained distribution. In our BFCL experiments, LAVE-Opt and EPIC-Opt have lower completion rates and throughput than DConstrain (Tables 2 and 3).

The second family consists of automata-based methods, which couple token choices through joint inference over finite automata (Suresh et al., 2025; Dang & Ermon, 2026; Thimmaiah et al., 2026). DINGO (Suresh et al., 2025) builds a DFA and precomputes vocabulary-level token transitions. Dang & Ermon (2026) perform chain and tree inference over a finite automaton whose edges group tokens and sum their probabilities.

Building on such step-exact samplers, Thimmaiah et al. (2026) use twisted sequential Monte Carlo to correct trajectory bias across denoising steps. However, all three methods compile JSON schemas under a fixed property order to keep the automaton finite and small. Interpreting a pushdown automaton directly at runtime would avoid materializing context-specific states, but it would repeatedly execute token consumption logic for every reached configuration. DConstrain resolves this tension by compiling reusable local grammar–token programs while leaving the runtime context unbound. Within this framework, our token-effect grouping generalizes the edge-level aggregation of Dang & Ermon (2026) to parameterized transitions. As a result, a single token partition per local grammar state serves every runtime context.

## 3 Preliminaries

A JSON schema G (Wright et al., 2022) defines a grammar whose rules can nest. For example, an object property can itself hold an object. We therefore parse $\mathcal { G }$ with a pushdown parser (Hopcroft et al., 2007), where each grammar rule is represented as a small automaton and entering a nested rule pushes a return point onto a stack. At any point, the parser state is captured by a configuration $c = ( q , s , e )$ . Here q is the local grammar state, which marks a node inside the current rule’s automaton. The stack s holds pending return points, and the parsing environment e records earlier parsing choices, such as which object fields have already been written. Together, $( s , e )$ form the runtime context of c. Given a configuration c and a vocabulary token $v \in \mathcal V .$ , the token-transition relation ${ \sf T } _ { \mathcal { G } } ( c , v )$ returns the set of valid successor configurations $c ^ { j }$ after consuming v. This set omits statically certified dead configurations that can never lead to a complete valid output (see Appendix A.1 for formal parser semantics).

Automata-Based DLM Constrained Decoding. Automata-based methods apply ${ \mathsf { T } } _ { \mathcal { G } }$ across an active decoding window to coordinate token choices at multiple unresolved positions (Suresh et al., 2025; Dang & Ermon, 2026). Consider a denoising iteration t and its canvas of H token positions. Let $\mathcal { V } _ { i } ^ { t } \subseteq \mathcal { V }$ denote the candidate tokens at position i; a position that is already fixed has a singleton candidate set. Starting from the parser configurations $\mathcal { C } _ { 0 }$ compatible with the fixed prefix $\scriptstyle { \pmb { u } } ,$ the token-labeled transitions and reachable configurations at each position are defined recursively as

$$
\begin{array} { r l } & { E _ { i } = \big \{ ( c , v , c ^ { \prime } ) ~ | ~ c \in \mathcal { C } _ { i - 1 } , ~ v \in \mathcal { V } _ { i } ^ { t } , ~ c ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( c , v ) \big \} , } \\ & { \mathcal { C } _ { i } = \big \{ c ^ { \prime } ~ | ~ \exists c , v : ~ ( c , v , c ^ { \prime } ) \in E _ { i } \big \} , \qquad i = 1 , \dots , H . } \end{array}\tag{1}
$$

The position-indexed layers $\{ \mathcal { C } _ { i } \} _ { i = 0 } ^ { H }$ and transitions $\{ E _ { i } \} _ { i = 1 } ^ { H }$ form a layered canvas-specific grammar graph $\mathcal { F } _ { t }$ A full-window path through $\mathcal { F } _ { t }$ is admissible if its terminal configuration satisfies the required acceptance or continuation condition.

Each denoising iteration constructs and resolves $\mathcal { F } _ { t }$ in two stages. First, grammar propagation builds the graph layers from left to right: starting from $\mathcal { C } _ { 0 } ,$ it advances every reached configuration with every candidate token to collect its successors. Notably, propagation depends only on parser configurations and candidate sets, not on model scores. Second, joint inference weights each edge $\left( c , v , c ^ { \prime } \right) \in E _ { i }$ using the token probability $p _ { i } ( v )$ from the DLM predictions $P _ { t }$ (defined in Appendix A.2). Max-product dynamic programming then extracts a highest-weight admissible path via the Viterbi algorithm (Viterbi, $1 9 6 7 ) .$ , while sum-product computes path-weight sums for exact probabilistic sampling (Goodman, 1999; Dang & Ermon, 2026).

Problem Formulation. All automata-based decoders share this two-stage pipeline, and their efficiency hinges on how grammar propagation answers each successor query ${ \sf T } _ { \mathcal { G } } ( c , v )$ . In AR constrained decoding, each forward pass predicts a single position for a fixed prefix. The decoder only needs a boolean mask of valid next tokens, after which it samples one token v and advances the parser once (Dong et al., 2025). In contrast, a DLM leaves many positions unresolved at once, and each position can be reached by many hypothetical parser configurations. Before committing any token, grammar propagation must return exact successor sets rather than boolean masks, and it must evaluate every candidate token at every reached configuration. Without query reuse or token grouping, exhaustive propagation requires

$$
N _ { \mathrm { q u e r y } } ^ { \mathrm { n a i v e } } = \sum _ { i = 1 } ^ { H } \left| \mathcal { C } _ { i - 1 } \right| \left| \mathcal { V } _ { i } ^ { t } \right|\tag{2}
$$

configuration–token queries per canvas.

To avoid expensive online parsing during propagation, finite-state pipelines such as DINGO (Suresh et al., 2025) precompute token-level transitions offline during cold compilation. Given a materialized finite state space $Q _ { \mathcal G } ,$ , a dense transition table stores an entry for each of the $| Q g | | \nu |$ state–token pairs, turning each online query into a table lookup. However, flattening runtime contexts into explicit finite states duplicates shared grammar logic and precomputes transitions for contexts that a canvas may never reach. The following propositions quantify this state explosion.

Proposition 3.1 (Context expansion; Appendix A.5). Let $Q _ { \mathrm { l o c } }$ be the set of local grammar states, $g > 1$ the number of continuation-frame symbols, $d _ { \mathrm { m a x } }$ the stack-depth bound, and B the maximum number of environment values per local-state–stack pair. A context-expanded state space contains

$$
S _ { d _ { \mathrm { m a x } } } ~ \le ~ | Q _ { \mathrm { l o c } } | B \sum _ { \ell = 0 } ^ { d _ { \mathrm { m a x } } } g ^ { \ell } ~ = ~ { \cal O } \Big ( | Q _ { \mathrm { l o c } } | B g ^ { d _ { \mathrm { m a x } } } \Big )\tag{3}
$$

states. In particular, for bounded-depth brackets $o f g$ types, a minimal complete DFA requires $\begin{array} { r } { 1 + \sum _ { \ell = 0 } ^ { d _ { \operatorname* { m a x } } } g ^ { \ell } = \Theta ( g ^ { d _ { \operatorname* { m a x } } } ) } \end{array}$ states. A dense context-expanded transition table therefore stores $S _ { d _ { \mathrm { m a x } } } | \nu |$ entries, whereas a factorized compiler stores at most $| Q _ { \mathrm { l o c } } | | \mathcal { V } |$ local-state–token entries.

Proof sketch. Each context-expanded state pairs a local state with a stack of depth at most $d _ { \mathrm { m a x } }$ over g frame symbols and one environment value; counting these triples yields Eq. (3). For bounded-depth brackets, any two distinct stacks of open brackets require distinct closing sequences. By the Myhill–Nerode theorem (Hopcroft et al., 2007), a minimal complete DFA must assign a separate state to every valid stack plus one sink state, which bounds the number of table entries required.

Proposition 3.1 leaves the environment factor B abstract. When a schema includes unordered object properties, B itself grows exponentially, and replacing finite automata with context-free grammars does not avoid this blow-up.

Proposition 3.2 (Unordered properties; Appendix A.5). For every $k \geq 1$ , there exists an object schema with k required unordered properties such that every DFA accepting its valid serializations has at least $\dot { 2 ^ { k } }$ states, and every context-free grammar generating them has size $2 ^ { \Omega ( k ) }$ . A factorized compiler instead requires only $O ( k )$ local grammar states for this object and tracks its field history in a k-bit runtime environment value.

Proof sketch. Any two prefixes that have written different subsets of required fields admit different sets of valid completions, so the Myhill–Nerode theorem separates all $2 ^ { k }$ subsets. For the CFG bound, intersecting the schema language with a regular language that fixes field values and counts, and then projecting onto property names, yields the language of all permutations of k distinct symbols. Every CFG generating this permutation language has size $2 ^ { \overset { \smile } { \Omega } ( k ) }$ (Ellul et al., 2005). Because regular intersection and homomorphic projection increase CFG size by at most a polynomial factor, the original grammar must also have size $2 ^ { \mathbf { \dot { \Omega } } } { \boldsymbol { \Omega } } ( k )$

Propositions 3.1 and 3.2 bound the size of static finite-state and CFG representations rather than the runtime cost of parsing a single string. For offline finite-state compilers, however, representation size directly dictates compilation cost, which explains the sharp growth with nesting depth and property count observed in Section 5.3.

To overcome this bottleneck, we seek a compiled artifact $\mathcal { A } _ { \mathcal { G } }$ containing executable successor-construction procedures, with the runtime context supplied at query time. These procedures must implement the exact successor-construction interface

$$
\operatorname { S u c c } ( A _ { \mathcal { G } } , c , v ) = \mathsf { T } _ { \mathcal { G } } ( c , v )\tag{4}
$$

![](images/63248e227740d1e3778bb8de055bfd68ac2212cb0606aca26cec449402615fce.jpg)  
Figure 1. Overview of DConstrain on a nested-object schema with required properties a and b sharing the same Pet rule. Top: A finite-state pipeline such as DINGO encodes the grammar position and runtime context together in each state and precomputes transitions across the full vocabulary. States $c _ { 1 }$ and $c _ { 2 }$ share the local position $q _ { o }$ but carry their own stacks and field histories. The token }} is rejected from $c _ { 1 } ,$ , whose parent still requires $\mathsf { b } ,$ but closes both objects from $c _ { 2 }$ . Bottom: Cold compilation takes a local grammar position q and token $v ,$ leaving (s, e) unbound, and produces context-parameterized program sets. The local-position legend identifies $q _ { t }$ as the string body and $q _ { o }$ as the child-object close. Tokens with identical canonical effects at each position form compile-time groups. For }}, the shared program set $\mathcal { E } _ { 2 }$ defers parsing the second brace until the caller is known. Warm compilation binds the concrete stack and environment, returning an empty successor set for $c _ { 1 }$ and an accepting configuration for $c _ { 2 } .$ Right: Joint inference uses group-level edges with sum or max score aggregation before restoring concrete tokens. Timeline: Cold compilation runs once offline. Warm compilation overlaps with the first forward pass of each canvas, and grammar propagation overlaps with every forward pass before the inference solve. DINGO runs propagation and solve sequentially after each forward pass. Graphs and block lengths are schematic.

for every configuration–token pair queried during grammar propagation. Our goal is to minimize coldcompilation cost, compiled storage, and online query latency without materializing the exponentially large context-expanded state space.

## 4 DConstrain

DConstrain compiles schemas into context-parameterized transition programs (Figure 1). These programs implement the successor interface in Eq. (4), independently of the downstream constrained-decoding algorithm. Cold compilation specializes token consumption to local grammar states while leaving the runtime context unbound and retaining context-dependent operations symbolically (Section 4.1). Warm compilation executes these programs for reached configurations to construct exact successors (Section 4.2). Token-effect grouping partitions tokens by their canonical compiled effects during cold compilation and reuses these partitions during warm compilation (Section 4.3). Appendices A.3 and A.5 provide correctness and complexity analyses.

## 4.1 Context-Parameterized Transition Compilation

Given a target schema ${ \mathcal { G } } ,$ DConstrain represents local grammar rules as a recursive transition network (RTN) (Woods, 1970), keeping the call stack s and parsing environment $e$ outside the RTN as runtime context. Specifically, we decompose the environment as $e = ( d , \eta )$ , where $d$ tracks the active container depth and η tracks the choice environment (such as the written-field history), giving the full configuration $c = ( q , s , d , \eta )$ . Cold compilation specializes token consumption to each local grammar state and token pair (q, v) while leaving (s, d, η) unbound:

$$
\left( \boldsymbol { q } , \boldsymbol { v } \right) \xrightarrow { \mathrm { C o l d C o m p i l e } } \mathcal { E } \left( \boldsymbol { q } , \boldsymbol { v } \right) = \left\{ P _ { k } \right\} _ { k = 1 } ^ { K _ { q , v } } .\tag{5}
$$

Each program $P _ { k } = ( I _ { k , 1 } ; \ldots ; I _ { k , \ell _ { k } } )$ is an ordered symbolic execution alternative for consuming v from q. Its guards may reject a concrete runtime context, and its continuation may depend on a caller selected by the incoming stack. Cold compilation evaluates byte matching and local grammar operations wherever their inputs are known, retaining context-dependent operations and the input and execution state needed to resume deferred parsing. The resulting programs take two forms: residual programs with a deferred caller-dependent continuation, and compact summaries with a statically determined target state.

When token consumption or token-boundary processing must pop a return frame from the incoming stack, the caller needed to continue parsing is unknown during cold compilation. The compiler resolves pushes and pops whose return destinations can be determined locally. At the first return that requires the incoming stack, it records a residual program with the remaining token input and the current symbolic execution state. Warm compilation uses the concrete stack to resume the caller-dependent continuation. A shared program can therefore compute exact successors even when its final local state cannot be determined offline.

When a program does not require a caller from the incoming stack s, symbolic execution reduces it to a compact summary $P = ( q ^ { \prime } , \rho , \Delta , m , \Omega )$ . Here $q ^ { \prime }$ is the resulting local grammar state, and $\rho$ is the sequence of new return frames pushed (and not yet popped) while consuming v. The integers $\Delta$ and m record the net change and minimum intermediate change in container depth d. Finally, Ω is an ordered sequence of environment guards and updates on $\eta ,$ , such as verifying that field a has not yet appeared and recording its presence. During warm compilation, binding a summary $P$ to a concrete runtime context $( s , d , \eta )$ evaluates directly to

$$
\begin{array} { r } { \mathrm { B i n d } ( P ; s , d , \eta ) = \left\{ \begin{array} { l l } { \{ ( q ^ { \prime } , s \parallel \rho , d + \Delta , \Omega ( \eta ) ) \} , } & { d + m \geq 0 , \Omega ( \eta ) \neq \perp , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{6}
$$

where ∥ appends frames to the stack and ⊥ indicates a violated guard (see Appendix A.3 for the full specification with auxiliary metadata). For these summaries, binding evaluates the remaining guards and updates under $( s , d , \eta )$ and constructs the successor without recompiling grammar logic for that context. Together, summaries and residuals allow configurations with the same local state $q$ to reuse ${ \mathcal { E } } ( q , v )$ even when their successors differ.

The choice environment is likewise handled through guarded runtime operations. For an object with k fields under arbitrary property order, η stores the set of written fields as a runtime bitmask, while the compiled operator Ω encodes the guards for field uniqueness and required-field completion. A single program set ${ \mathcal { E } } ( q , v )$ is shared across the $2 ^ { k }$ possible field histories, so cold compilation does not enumerate those subsets as local grammar states. Concrete histories remain runtime values, and program execution preserves the order and scope of their guards and updates.

## 4.2 Warm Compilation

Warm compilation executes the remaining operations of a compiled program under a concrete runtime context. We first specify its token-level successor interface: for a reached configuration $c = ( q , s , e ) \in \mathcal { C } _ { i - 1 }$ and a candidate token $v \in \mathcal { V } _ { i } ^ { t }$ at position $i ,$ it returns

$$
{ \widehat { \mathsf { T } } } _ { \mathcal { G } } ( c , v ) = \bigcup _ { P \in { \mathcal { E } } ( q , v ) } { \mathrm { E x e c } } ( P , c ) .\tag{7}
$$

Here Exec $( P , c )$ applies Eq. (6) to ordinary summaries, resumes residual programs under the concrete caller, and completes token-boundary processing. Failed guards contribute no successor. The resulting context-specific successors drive the layer-by-layer propagation in Eq. (1). The implementation evaluates this interface through the compile-time partitions introduced in Section 4.3, using one representative token per reached configuration and active group. Transitions for unreachable contexts are never instantiated.

The following theorem establishes that the two-stage compiler and program execution preserve the reference successor relation, including caller-dependent continuations.

Theorem 4.1 (Exact successor construction; Appendix A.3). For every reached configuration c and candidate token $v ,$ warm compilation returns

$$
\widehat { \mathsf { T } } _ { \mathcal { G } } ( c , v ) = \mathsf { T } _ { \mathcal { G } } ( c , v )\tag{8}
$$

under the reference parser semantics. Consequently, running grammar propagation with $\widehat { \mathsf { T } } _ { \mathcal { G } }$ reproduces the exact layers $\{ { \mathcal { C } } _ { i } \}$ and transitions $\left\{ E _ { i } \right\} o f E q . \ ( 1 )$ from the same initial set $\mathcal { C } _ { 0 }$ and candidate sets $\{ \mathcal { V } _ { i } ^ { t } \}$ }. Under the same acceptance or continuation condition, both graphs admit the exact same token-labeled window language.

## 4.3 Token-Effect Grouping

Parameterized programs share transition logic across runtime contexts, but evaluating candidate tokens separately at each reached configuration retains the vocabulary factor in Eq. (2). We extend token aggregation in finite-state inference (Dang & Ermon, 2026) to context-parameterized programs: during cold compilation, tokens are grouped by their canonical compiled effects at each local state, before any concrete stack or environment is bound. For an unconstrained string rule, for example, many text tokens compile to the same continuation program (Figure 1). We define the grouping relation at local state $q$ by

$$
v _ { 1 } \sim _ { q } v _ { 2 } \quad \Longleftrightarrow \quad { \mathrm { C a n o n } } ( { \mathcal { E } } ( q , v _ { 1 } ) ) = { \mathrm { C a n o n } } ( { \mathcal { E } } ( q , v _ { 2 } ) ) .\tag{9}
$$

Cold compilation records the partition $\Pi _ { q } = \mathcal { V } / \sim _ { q } .$ . Canonicalization retains instruction order, operands, stack and environment operations, and execution-relevant boundary and residual metadata. Canonical equality is a sufficient, not necessary, condition for identical successor sets under every admissible runtime context (Appendix A.3). The same partition can therefore be reused across different stacks and environments, even though their concrete successor sets may differ.

The offline partition $\Pi _ { q }$ is shared by all configurations with local state q. At position i, the candidate set selects its active groups, $\Pi _ { q , i } ^ { t } = \big \{ \boldsymbol { G } ^ { * } \in \Pi _ { q } : \boldsymbol { G } \cap \mathcal { V } _ { i } ^ { t } \neq \emptyset \big \}$ . For each reached configuration $c = ( q , s , e )$ and active group $G \in \Pi _ { q , i } ^ { t } ,$ warm compilation evaluates the successor relation once using any representative token $v _ { G } \in G \cap \mathcal { V } _ { i } ^ { t }$

$$
\widehat { \mathsf { T } } _ { \mathcal { G } } ( c , G ) = \widehat { \mathsf { T } } _ { \mathcal { G } } ( c , v _ { G } ) , \qquad v _ { G } \in G \cap \mathcal { V } _ { i } ^ { t } .\tag{10}
$$

Using these representative successor sets, warm compilation directly constructs one group-level edge $( c , G , ^ { - } c ^ { \prime } )$ for each returned successor $c ^ { \prime } \in \widehat { \mathsf { T } } _ { \mathcal { G } } ( c , G )$ , without first materializing the corresponding token-level edges. This yields the grouped transition layer

$$
\boldsymbol { E } _ { i } ^ { \mathrm { I I } } = \{ ( c , G , c ^ { \prime } ) ~ | ~ \boldsymbol { c } \in \mathcal { C } _ { i - 1 } , ~ G \in \Pi _ { q ( c ) , i } ^ { t } , ~ c ^ { \prime } \in \widehat { \boldsymbol { \mathrm { I } } } _ { \mathcal { G } } ( c , G ) \} .\tag{11}
$$

As analyzed in Appendix ${ \mathrm { A . 5 } } ,$ , the number of successor evaluations now scales with active groups rather than individual tokens:

$$
N _ { \mathrm { i n s t } } = \sum _ { i = 1 } ^ { H } \sum _ { c \in \mathcal { C } _ { i - 1 } } \left| \Pi _ { q ( c ) , i } ^ { t } \right| \ \le \ \operatorname* { m i n } \Bigl \{ N _ { \mathrm { q u e r y } } ^ { \mathrm { n a i v e } } , \ R _ { t } \operatorname* { m a x } _ { q } | \Pi _ { q } | \Bigr \} , \qquad R _ { t } = \sum _ { i = 1 } ^ { H } | \mathcal { C } _ { i - 1 } | .\tag{12}
$$

Grouping reduces successor evaluations and the number of edges traversed during joint inference. The bound above concerns successor evaluations; objective-specific score aggregation still uses the individual probabilities of active group members, whose identities remain available for token restoration.

The compile-time partition is independent of model scores and of the inference objective. Joint inference assigns weights to its group-level edges using the current predictions $P _ { t }$ . For nonnegative token weights $p _ { i } ( v )$ , define the objective-specific aggregates

$$
M _ { i } ( G ) = \sum _ { v \in G \cap \mathcal { V } _ { i } ^ { t } } p _ { i } ( v ) , \qquad m _ { i } ( G ) = \operatorname* { m a x } _ { v \in G \cap \mathcal { V } _ { i } ^ { t } } p _ { i } ( v ) .\tag{13}
$$

Sum-product inference weights each group-level edge by $M _ { i } ( G )$ , whereas max-product inference uses $m _ { i } ( G )$ . For any fixed source configuration $c ,$ the active groups are disjoint and all tokens within each group have the same successor set. These objective-specific reductions therefore preserve the weighted transition operator between c and every successor configuration $c ^ { \prime }$ for $\oplus \in \{ + , \operatorname* { m a x } \}$ :

$$
A _ { i } ^ { \oplus } ( c , c ^ { \prime } ) = \bigoplus _ { v \in \mathcal { V } _ { i } ^ { t } : c ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( c , v ) } p _ { i } ( v ) = \bigoplus _ { G \in \Pi _ { q ( c ) , i } ^ { t } : c ^ { \prime } \in \hat { \mathsf { T } } _ { \mathcal { G } } ( c , G ) } r _ { i } ^ { \oplus } ( G ) , ~ r _ { i } ^ { + } = M _ { i } , ~ r _ { i } ^ { \operatorname* { m a x } } = m _ { i } .\tag{14}
$$

After selecting a group-level edge, DConstrain restores a token $v \in G \cap \mathcal { V } _ { i } ^ { t }$ according to the same inference objective. Sum-product restoration is applied to selected groups with positive mass $M _ { i } ( G ) > 0$ . The restoration rules are

$$
\operatorname* { P r } ( v \mid G ) = { \frac { p _ { i } ( v ) } { M _ { i } ( G ) } } { \mathrm { ~ ( s u m . p r o d u c t ) } } , \qquad v \in \underset { v ^ { \prime } \in G \cap V _ { i } ^ { t } } { \arg \operatorname* { m a x } } p _ { i } ( v ^ { \prime } ) { \mathrm { ~ ( m a x - p r o d u c t ) } } .\tag{15}
$$

The following theorem shows that grouping preserves weighted inference.

Theorem 4.2 (Grouping preserves inference; Appendix A.4). Let $\Pi _ { q }$ be the token-effect partitions of $E q .$ (9), and assume a positive partition function and exact arithmetic. Under both chain and tree inference, grouped sum-product inference with the restoration rule in Eq. (15) preserves both the partition function and the exact token-labeled path distribution of ungrouped inference. Furthermore, grouped max-product inference attains the exact Viterbi optimum and recovers every optimal token sequence when all ties are retained.

Solver-agnostic inference interface. Because DConstrain preserves the exact successor relation ${ \sf T } _ { \mathcal { G } } ( c , v )$ and both weighted transition operators, the same grouped graph representation supports different automatabased solvers. Appendix D.2 evaluates both Viterbi decoding and the tree-based sampling algorithm of Dang & Ermon (2026).

Runtime integration. Because neither warm compilation nor grammar propagation depends on model logits, CPU workers execute both stages concurrently while the GPU computes the DLM forward pass. Warm compilation runs only during the first forward pass of each canvas. Subsequent denoising iterations on the same canvas only shrink the candidate sets $\mathcal { V } _ { i } ^ { t }$ as positions are resolved, so their required successor queries are a subset of those already computed. In every iteration, grammar propagation builds $\mathcal { F } _ { t }$ from these cached successors during the forward pass, and joint inference begins as soon as both $\mathcal { F } _ { t }$ and $P _ { t }$ are ready. Compact compressed sparse row (CSR) layouts and specialized GPU kernels further accelerate the joint inference solve. Appendix B.3 details the full compiler–runtime pipeline and the optimizations enabled in each experimental setting.

## 5 Experiments

Workloads and models. We evaluate on BFCL (Patil et al., 2025), JSONSchemaBench (Geng et al., 2025), and ComplexFuncBench (Zhong et al., 2025). BFCL and ComplexFuncBench provide tool declarations for both compilation and task-level evaluation, while JSONSchemaBench stress-tests compilation with diverse and complex schemas. For end-to-end evaluation, we select two recent DLM families with different decoding granularities: LLaDA2.2 (Bie et al., 2026) with a block size of 32 and DiffusionGemma (DiffusionGemma Team et al., 2026) with a block size of 256. Both decode a long output block by block, as in block diffusion (Arriola et al., 2025). To reflect realistic serving conditions, we run LLaDA2.2 in SGLang (Zheng et al., 2024) and DiffusionGemma in vLLM (Kwon et al., 2023) rather than in standalone research inference loops. Cold-compilation experiments need only the corresponding tokenizers.

Comparisons. For cold compilation, we compare with DINGO’s DFA compiler (Suresh et al., 2025), an NFA adaptation, and DINGO-Opt, our engineering-optimized version of DINGO, under matched schemas and tokenizers. The main BFCL runtime tables compare unconstrained Native-Opt decoding, the verification-based methods LAVE-Opt (Zhang et al., 2026) and EPIC-Opt (Jin & Han, 2026), DINGO-Opt, and DConstrain. Table 6 compares DConstrain and DINGO-Opt on their common successful cases, including Any-order. The -Opt suffix identifies baseline implementations with our optimizations and adaptations to the decoding setup. Appendix D adds alternative solvers and further baselines.

Cold-compilation targets. Table 1 compares compilers that prepare token-labeled successor construction for joint solving. Verification-based methods prepare a different object: a parser and auxiliary structures for deciding whether a proposed partial output admits a legal completion (Mündler et al., 2026; Zhang et al., 2026; Jin & Han, 2026). A Boolean verifier can return after finding one witness. A joint solver needs the compatible successor alternatives so that max- or sum-product inference can combine their model weights. DConstrain compiles the programs that construct these alternatives, leaving the inference algorithm downstream. Verifier preparation and successor compilation thus have different output contracts; Tables 2 and 3 compare their resulting decoding pipelines in completion, accuracy, and throughput.

## 5.1 Main Results

Table 1 evaluates cold compilation under fixed and arbitrary property order, with nesting depth limited to four for comparability with the finite-state baselines. The BFCL runtime evaluation uses two complementary cohorts. Table 2 counts cold admission, runtime completion, strict-schema validity, and official semantic accuracy over all 2,495 cases; TPS and TPF use each row’s completed cases. Table 3 defines a common runtime-success cohort independently for each model and property-order condition by intersecting the four reported methods within that condition. Schema-invalid completions remain in each cohort. DINGO-Opt compiles 71.30% of BFCL schemas with LLaDA2.2 and 85.81% with DiffusionGemma under Any-order. Table 2 reports its Any-order outcomes on the full pool. Including it in the intersection would substantially shrink the common-success cohorts in Table 3; Table 6 compares DINGO-Opt and DConstrain on their paired successful cases. Each order condition has its own common-success cohort. Appendix B.2 specifies the accounting rules.

Cold-compilation efficiency. DConstrain reduces compilation cost across both tokenizers and all datasets. Under fixed order and depth four, it is 3.88–4.83× faster than DINGO-Opt on paired cases, even though DINGO-Opt already shares our C++ pipeline and CSR representation (Appendix B.3). The gap widens under arbitrary property order: DConstrain keeps 100% compilation coverage on BFCL and CFB, whereas DINGO-Opt loses substantial coverage. On the cases that both methods compile, DConstrain remains 3.54– 10.39× faster. On JSB, most remaining DConstrain failures with the LLaDA2.2 tokenizer come from JSON Schema features outside the supported fragment, such as oneOf exclusivity and general allOf intersection (13.7% of all schemas); 4.8% exceed compilation resource or construction budgets (Appendix D.5). Mature engines also support only part of this benchmark: on its GitHub-Hard split, XGrammar and Guidance declare support for 69% and 60% of the schemas (Geng et al., 2025).

Runtime completion and throughput. DConstrain completes all 2,495 cases with 100% strict-schema validity under both models and both property-order settings (Table 2). On the common cohorts in Table 3, it has the highest TPS among the constrained methods in all four settings. Its TPS is 1.56–17.03× that of LAVE-Opt and 13.15–30.21× that of EPIC-Opt. Under Schema-order, it reduces overhead above Native-Opt by 10.14× for LLaDA2.2 and 21.75× for DiffusionGemma relative to DINGO-Opt. On the paired Any-order cohorts in Table 6, DConstrain reduces end-to-end latency by 6.65× and 46.26×, and solve latency per forward by 26.40× and 108.54×, respectively.

Table 1. Cold-compilation performance on BFCL, JSONSchemaBench (JSB), and ComplexFuncBench (CFB). Success@180s is the fraction of all attempted schemas compiled successfully within 180 s; P50/P95 use each method’s successful cases. DINGO-Opt<sup>∗</sup> retains DINGO’s finite-state strategy with the same engineering optimizations as DConstrain (Appendix B.3). Speedup<sup>†</sup> is the row method’s mean latency divided by DConstrain’s on paired successful cases; timeouts count only toward Success@180s.
<table><tr><td></td><td colspan="4">LLaDA2.2</td><td colspan="4">DiffusionGemma</td></tr><tr><td>Method</td><td>Success (%)</td><td>P50 (s)</td><td>P95 (s)</td><td>Speedup†</td><td>Success (%)</td><td>P50 (s)</td><td>P95 (s)</td><td>Speedup+</td></tr><tr><td colspan="9">BFCL — fixed order</td></tr><tr><td>DINGO</td><td>90.3</td><td>54.30</td><td>132.90</td><td>92.9×</td><td>91.7</td><td>38.45</td><td>115.41</td><td>140.4×</td></tr><tr><td>NFA</td><td>82.5</td><td>61.30</td><td>148.50</td><td>113.4×</td><td>86.0</td><td>45.52</td><td>122.17</td><td>167.4×</td></tr><tr><td>DINGO-Opt*</td><td>100.0</td><td>2.21</td><td>5.64</td><td>3.92×</td><td>100.0</td><td>1.20</td><td>3.29</td><td>3.88×</td></tr><tr><td>DConstrain</td><td>100.0</td><td>0.58</td><td>1.48</td><td></td><td>100.0</td><td>0.31</td><td>0.82</td><td></td></tr><tr><td colspan="9">BFCL — any order</td></tr><tr><td>DINGO-Opt*</td><td>71.30</td><td>5.85</td><td>12.25</td><td>9.43×</td><td>85.81</td><td>3.76</td><td>14.16</td><td>10.39×</td></tr><tr><td>DConstrain</td><td>100.0</td><td>0.66</td><td>1.23</td><td></td><td>100.0</td><td>0.41</td><td>0.90</td><td></td></tr><tr><td colspan="9">JSB — fixed order</td></tr><tr><td>DINGO</td><td>67.7</td><td>38.00</td><td>129.11</td><td>85.0×</td><td>67.7</td><td>31.97</td><td>96.34</td><td>122.9×</td></tr><tr><td>NFA</td><td>63.7</td><td>45.26</td><td>128.40</td><td>102.8×</td><td>71.3</td><td>30.71</td><td>113.85</td><td>114.0×</td></tr><tr><td>DINGO-Opt*</td><td>74.8</td><td>2.51</td><td>70.50</td><td>4.83×</td><td>77.6</td><td>1.65</td><td>37.66</td><td>4.53×</td></tr><tr><td>DConstrain</td><td>81.4</td><td>0.60</td><td>14.14</td><td></td><td>83.3</td><td>0.37</td><td>8.27</td><td></td></tr><tr><td colspan="9">CFB — fixed order</td></tr><tr><td>DINGO</td><td>34.8</td><td>115.18</td><td>159.24</td><td>121.9×</td><td>44.7</td><td>84.73</td><td>124.04</td><td>174.6×</td></tr><tr><td>NFA</td><td>32.5</td><td>131.29</td><td>175.14</td><td>132.0×</td><td>74.0</td><td>116.63</td><td>171.69</td><td>221.9×</td></tr><tr><td>DINGO-Opt*</td><td>100.0</td><td>4.66</td><td>7.34</td><td>4.02×</td><td>100.0</td><td>2.21</td><td>3.02</td><td>3.96×</td></tr><tr><td>DConstrain</td><td>100.0</td><td>1.17</td><td>1.78</td><td></td><td>100.0</td><td>0.56</td><td>0.76</td><td></td></tr><tr><td colspan="9">CFB — any order</td></tr><tr><td>DINGO-Opt*</td><td>9.9</td><td>7.63</td><td>16.69</td><td>8.12×</td><td>13.7</td><td>2.90</td><td>9.40</td><td>3.54×</td></tr><tr><td>DConstrain</td><td>100.0</td><td>0.99</td><td>1.41</td><td></td><td>100.0</td><td>0.80</td><td>1.16</td><td></td></tr></table>

Table 6 reports end-to-end and per-forward solver latency on the DConstrain/DINGO-Opt common-success cohorts. Warm compilation overlaps the model forward pass. In every plotted setting of Figure 2, at least 94% of warm-compilation calls finish within the mean forward time. Appendix D.2 reports additional solver experiments.

Tokens per forward and GPU computation. TPF measures how many model-generated tokens are produced per generation forward, making it an important measure of GPU computation per output token in a DLM. Raising the token-acceptance threshold commits fewer tokens per step and can improve accuracy through additional refinement, reducing TPF. LAVE also uses additional model forwards for lookahead to reduce uncertainty (Zhang et al., 2026). Both approaches spend more GPU computation per generated token. DConstrain combines the highest Any-order semantic accuracy with higher TPF than the verification-based methods on both models: 81.60% accuracy at 8.10 TPF for LLaDA2.2 and 81.56% at 8.15 TPF for DiffusionGemma on the full pool. The common-success results show the same combination of high accuracy and high TPF. TPF counts non-stop model token IDs over all generation forwards for the selected cases; CPU-generated completion text is excluded.

![](images/9ee014580903107ef4b5b9403d6e229d174894775f08f1221a8c9a5dcebfd024.jpg)  
Figure 2. Warm-compilation latency under asynchronous execution. CPU warm compilation overlaps with GPU model execution. Dashed lines indicate mean model-forward durations.

## 5.2 Ablation Study

Table 4 isolates three mechanisms on BFCL with the LLaDA2.2 tokenizer and arbitrary property order. w/o Two-Stage Env binds parsing-environment values during cold compilation instead of deferring them to runtime. w/o Local-State Reuse disables memoization of transition programs for repeated occurrences of the same local grammar state but still keeps the factorized runtime environment and continuation stack. w/o Token Grouping processes tokens with equivalent effects independently. All variants share the same grammar semantics and decoding setup.

The three mechanisms remove different forms of redundancy. Deferring environment specialization gives the largest cold-compilation benefit: eager binding raises mean compilation latency to 14.61× that of the full system. Local-state reuse avoids recompiling identical local transition logic. Token-effect grouping mainly reduces vocabulary-level work in downstream joint inference, and removing it raises mean solve latency to 6.81× that of the full system. A pushdown interpreter also avoids finite-state expansion, so Appendix D.1 further compares DConstrain with Direct-PDA-Opt, which interprets the parser at every reached configuration. Even with token grouping before interpretation, it is 6.43× slower end to end on 300 BFCL cases, because it re-executes the shared transition logic for every configuration.

## 5.3 Analysis

Property order changes task accuracy. Schema-order is a substantive decoding restriction for DLMs. On the full 2,495-case BFCL pool in Table 2, allowing arbitrary property order raises DConstrain’s semantic accuracy from 72.71% to 81.60% for LLaDA2.2 and from 37.27% to 81.56% for DiffusionGemma: gains of 8.89 and 44.29 percentage points, respectively, with 100% schema validity throughout. Native-Opt accuracy changes by only −0.12 and +0.16 points under the corresponding prompt conditions. The much larger constrained-decoding gains indicate that fixing the field order excludes schema-valid outputs that the models prefer. These comparisons use the same full pool; the separately selected cohorts in Table 3 support within-order comparisons. The higher LLaDA2.2 TPS in the Any-order common panel reflects its easier case composition: the shared cohort contains 2,149 cases, compared with 2,448 under Schema-order.

Verification, sampling bias, and recovery. A verifier answers whether a proposed update leaves some legal completion. Weighted joint inference additionally accounts for the model probability mass of compatible continuations. For a fixed canvas graph, the sampling objective assigns each accepting token-labeled path π probability proportional to ∏ p<sub>i</sub>(v<sub>i</sub>). An existential completion test supplies the support of this objective, but not the continuation masses needed to sample from it. Exact rejection of complete assignments from a fixed proposal samples that proposal conditioned on validity; local token acceptance, batch selection, and

Table 2. BFCL outcomes over all 2,495 cases, reported separately for Schema-order and Any-order. N counts each method’s runtime completions. Cold admission, runtime completion (Run), strict-schema validity (Valid.), and official semantic accuracy (Acc.) use 2,495 as the denominator; TPS and TPF use each row’s N completed cases.
<table><tr><td rowspan="2"></td><td colspan="8">DiffusionGemma</td><td colspan="10">LLaDA2.2</td></tr><tr><td>N</td><td>Cold (%)</td><td>Run (%)</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td>TPS</td><td>Overhead (%) ↓</td><td>N</td><td>Cold (%)</td><td>Run (%)</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td></td><td>TPS</td><td>Overhead (%) ↓</td></tr><tr><td colspan="10">Native-Opt 2495</td><td colspan="10">2495</td></tr><tr><td colspan="10"></td><td colspan="10"></td></tr><tr><td colspan="10">Verification-based</td><td colspan="10"></td></tr><tr><td colspan="10">LAVE-Opt</td><td colspan="10">2480 100.00</td></tr><tr><td colspan="10">2272 100.00 EPIC-Opt</td><td colspan="10">99.40</td></tr><tr><td colspan="10">2024100.00</td><td colspan="10">2463 98.88 98.72 98.72</td></tr><tr><td colspan="10">Automata-based DINGO-Opt 2495 100.00 100.00 100.00 8.11</td><td colspan="10"></td></tr><tr><td colspan="10">DConstrain 2495 100.00 100.00 100.00 8.13</td><td colspan="10">2495100.00100.00100.006.86  $7 2 . 6 3 \_ 8 . 1 7$  101.32</td></tr><tr><td colspan="10">Any-order</td><td colspan="10">72.71 8.09  $7 1 . 8 _ { 8 . 9 5 \times }$ </td></tr><tr><td colspan="10"></td><td colspan="10">LLaDA2.2</td></tr><tr><td>Method</td><td colspan="10">Cold N (%)</td><td colspan="10">Cold Run Valid.</td></tr><tr><td>Native-Opt</td><td>2495</td><td></td><td>(%) 100.00</td><td>(%) 90.348.14</td><td>TPF</td><td>(%) 76.55</td><td>TPS 167.90</td><td>(%) ↓ 0.0</td><td>N</td><td>(%)</td><td></td><td>(%)</td><td>(%)</td><td>TPF</td><td>(%)</td><td>TPS</td><td></td><td>(%) ↓</td></tr><tr><td colspan="9"></td><td colspan="9">2495 100.00</td></tr><tr><td colspan="10">Verification-based</td><td colspan="9">94.957.21</td></tr><tr><td colspan="10">LAVE-Opt</td><td colspan="10"></td></tr><tr><td colspan="10">EPIC-Opt 2212 92.38</td><td colspan="10">97.07 96.99 96.995.74</td></tr><tr><td colspan="10">Automata-based</td><td colspan="10">86.41 86.21 86.21  $6 . 0 9 \ 6 5 . 0 5 _ { \downarrow 1 5 . 6 3 }$ </td></tr><tr><td colspan="10">DINGO-Opt 2141 85.81 85.81 85.818.13  $6 9 . 8 5 _ { \downarrow 6 . 7 0 }$ </td><td colspan="10"></td></tr><tr><td colspan="10">DConstrain 2495 100.00 100.00 100.00 8.15</td><td colspan="10">177971.30 71.30 71.308.11  $5 9 . 3 0 _ { \downarrow 2 1 . 3 8 }$  42.39</td></tr></table>

Bold marks the best values within each model and order panel, including ties, and highlights DConstrain overhead. Acc. subscripts show percentage-point changes from Native-Opt: ↑ indicates an increase and ↓ a decrease. Overhead is $\left( \mathrm { T P S _ { N a t i v e - O p t } / T P S - 1 } \right) \times 1 0 0 \%$ . DConstrain TPS and Overhead subscripts give throughput speedups and overhead-reduction factors over DINGO-Opt in the same panel, respectively. Table 6 reports the paired latency comparison. Schema-order fixes property order; Any-order permits arbitrary order. For Native-Opt, these identify prompt conditions. TPS is model tokens/s; TPF counts non-stop model token IDs per generation forward, excluding CPU completion text. TPS timing excludes prefill for LLaDA2.2 and includes it for DiffusionGemma. The -Opt suffix denotes our implementation optimizations and decoding adaptations for the baselines.

recovery introduce different update rules.

Consider an unambiguous two-position grammar accepting only aa and bb, with $p _ { 1 } ( a ) = p _ { 2 } ( a ) = 0 . 9$ and $p _ { 1 } ( b ) = p _ { 2 } ( b ) = 0 . 1$ . Weighted joint sampling selects aa with probability $0 . 8 1 / ( 0 . 8 1 + 0 . 0 1 ) = 9 8 . 7 8 \%$ . A policy that samples the first token and commits any feasible choice selects aa with probability 90%, because the second token is then forced. Every completed output is valid, yet the distributions differ. The missing factor is the probability mass of legal continuations, which a Boolean verifier replaces by an existence decision. The same distinction affects MAP decoding: selecting high-confidence feasible proposals does not jointly maximize the product of token probabilities.

Finite witness search adds another source of selection bias. LAVE tests model-sampled lookahead completions and uses cached-prefix recovery (Zhang et al., 2026). If a proposal u has proposal probability q(u) and witness-search success probability a(u), its probability among accepted proposals is proportional to $q ( u ) a ( u )$ . Thus a feasible proposal can be rejected when its valid witnesses are missed, and acceptance depends on the lookahead distribution and search budget. EPIC uses confidence-based subset selection and shrinking before exact verification; on exhausting the rejection budget, it completes the output using a witness that is not sampled from the model distribution (Jin & Han, 2026). These policies favor different proposals and continuations from those selected by weighted joint inference.

Verification also pays for rejected work. Independent proposals can conflict across positions, so an unsuccessful check triggers resampling, a smaller batch, or recovery. Parallel batch verification reduces the number of sequential checks, but a failed batch still requires additional selection and validation. This cost depends on how often proposals are accepted, whereas automata-based inference resolves compatibility within the weighted graph. Native DLM refinement can revise temporarily incompatible assignments. Maintaining feasibility of committed updates restricts these denoising trajectories, and proposal filtering and recovery further change which trajectories are followed. On LLaDA2.2’s Any-order common cohort, Native-Opt reaches 82.78% semantic accuracy, LAVE-Opt and EPIC-Opt reach 82.60% and 75.48%, and DConstrain reaches 83.15%. The verification-based methods obtain 100% schema validity on this cohort while losing semantic accuracy relative to Native-Opt.

Table 3. BFCL quality and throughput on common runtime-success cohorts, defined separately for each model and property-order condition. Within each panel, we intersect the successful-run sets of Native-Opt, LAVE-Opt, EPIC-Opt, and DConstrain. Schema-invalid completions are retained.
<table><tr><td>Schema-order</td><td colspan="6">DiffusionGemma (N = 1,874)</td><td colspan="4">LLaDA2.2 (N = 2,448)</td></tr><tr><td>Method</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td>TPS</td><td>Overhead (%) ↓</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td>TPS</td><td>Overhead (%) ↓</td></tr><tr><td>Native-Opt</td><td>89.707.07</td><td></td><td>78.50</td><td>145.71</td><td>0.0</td><td></td><td>94.97 7.42</td><td>81.04</td><td>754.40</td><td>0.0</td></tr><tr><td colspan="9">Verification-based</td><td></td><td></td></tr><tr><td>LAVE-Opt</td><td>100.005.89</td><td></td><td> $4 8 . 1 9 _ { \downarrow 3 0 . 3 1 }$ </td><td>6.76</td><td>2055.5</td><td>100.004.67</td><td></td><td> $7 2 . 7 4 _ { \downarrow 8 . 3 0 }$ </td><td>221.32</td><td>240.9</td></tr><tr><td>EPIC-Opt</td><td>99.73 3.19</td><td></td><td> $4 8 . 7 2 _ { \downarrow 2 9 . 7 8 }$ </td><td>3.81</td><td>3724.4</td><td>100.005.63</td><td></td><td> $6 4 . 6 5 _ { \downarrow 1 6 . 3 9 }$ </td><td>25.00</td><td>2917.6</td></tr><tr><td colspan="9">Automata-based</td><td></td><td></td></tr><tr><td>DINGO-Opt 100.00 7.13</td><td></td><td></td><td> $4 8 . 8 8 _ { \downarrow 2 9 . 6 2 }$ </td><td>21.48</td><td>578.4</td><td>100.00 6.95</td><td></td><td> $7 2 . 9 9 _ { \downarrow 8 . 0 5 }$ </td><td>93.03</td><td>710.9</td></tr><tr><td colspan="9">DConstrain 100.00 7.13  $4 8 . 8 8 _ { \downarrow 2 9 . 6 2 }$   $1 1 5 . 1 0 _ { 5 . 3 6 \times }$ </td><td> $4 4 3 . 4 7 _ { 4 . 7 7 \times }$ </td><td> $7 0 . 1 _ { 1 0 . 1 4 \times }$ </td></tr><tr><td>Any-order</td><td colspan="5">DiffusionGemma</td><td colspan="5">LLaDA2.2 (N = 2,149)</td></tr><tr><td>Method</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td>TPS</td><td>Overhead (%) ↓</td><td>Valid. (%)</td><td>TPF</td><td>Acc. (%)</td><td>TPS</td><td>Overhead (%) ↓</td></tr><tr><td>Native-Opt</td><td>91.46</td><td>7.81</td><td>79.46</td><td>165.63</td><td>0.0</td><td>95.53 7.41</td><td></td><td>82.78</td><td>756.59</td><td>0.0</td></tr><tr><td colspan="9">Verification-based</td><td></td><td></td></tr><tr><td>LAVE-Opt</td><td></td><td>100.007.43</td><td> $8 3 . 5 5 _ { \uparrow 4 . 0 9 }$ </td><td>9.92 6.30</td><td>1569.7 2529.0</td><td>100.005.75</td><td></td><td> $8 2 . 6 0 _ { \downarrow 0 . 1 8 }$ </td><td>315.63 37.34</td><td>139.7 1926.2</td></tr><tr><td>EPIC-Opt</td><td>99.45 7.51</td><td></td><td> $8 4 . 5 5 _ { \uparrow 5 . 0 9 }$ </td><td></td><td></td><td>100.006.09</td><td></td><td> $7 5 . 4 8 _ { \downarrow 7 . 3 0 }$ </td><td></td><td></td></tr><tr><td colspan="9">Automata-based</td><td></td><td></td></tr><tr><td>DConstrain 100.00 7.81</td><td></td><td></td><td> $8 4 . 8 7 _ { \uparrow 5 . 4 1 }$ </td><td>110.50</td><td>49.9</td><td>100.00 8.22</td><td></td><td> $\mathbf { 8 3 . 1 5 } _ { \uparrow 0 . 3 7 }$ </td><td>490.98</td><td>54.1</td></tr></table>

Bold marks the best values within each model and order panel, including ties, and highlights DConstrain overhead. Acc. subscripts show percentage-point changes from Native-Opt: ↑ indicates an increase and ↓ a decrease. Overhead is $\left( \mathrm { T P S _ { N a t i v e - O p t } / T P S - 1 } \right)$ × 100%. Under Schema-order, DConstrain TPS and Overhead subscripts give throughput speedups and overhead-reduction factors over DINGO-Opt, respectively. The Any-order comparison with DINGO-Opt appears in Table 6. TPS/TPF accounting follows Table 2. The -Opt suffix denotes our implementation optimizations and decoding adaptations for the baselines.

Table 4. Compiler ablations on BFCL with arbitrary property order. Values are means; gray numbers are ratios to the full system.
<table><tr><td>Variant</td><td>Cold (s) ↓</td><td>Solve (ms) ↓1</td><td>Memory (MiB) ↓</td></tr><tr><td>Full (any order)</td><td>0.72</td><td>3.72</td><td>5.94</td></tr><tr><td>w/o Two-Stage Env</td><td>10.52 (14.61×)</td><td>3.83 (1.03×)</td><td>20.97 (3.53×)</td></tr><tr><td>w/o Local-State Reuse</td><td>1.87 (2.60×)</td><td>3.75 (1.01×)</td><td> $1 2 . 9 8 \left( 2 . 1 9 \times \right)$ </td></tr><tr><td>w/o Token Grouping</td><td> $0 . 9 6 \left( 1 . 3 3 \times \right)$ </td><td> $2 5 . 3 2 \left( 6 . 8 1 \times \right)$ </td><td> $2 4 . 4 2 \left( 4 . 1 1 \times \right)$ </td></tr></table>

Table 5. Compiled representation statistics with the LLaDA2.2 tokenizer.
<table><tr><td>Dataset</td><td>Method</td><td>States</td><td colspan="2">Edges (103)</td><td colspan="2">Recorded payload (MB)</td></tr><tr><td></td><td></td><td>Mean</td><td>P50</td><td>P95</td><td>P50</td><td>P95</td></tr><tr><td>BFCL</td><td>DINGO</td><td>228.74</td><td>524.77</td><td>1226.27</td><td>54.20</td><td>114.91</td></tr><tr><td></td><td>NFA</td><td>233.74</td><td>352.71</td><td>1052.08</td><td>31.95</td><td>93.55</td></tr><tr><td></td><td>DConstrain</td><td>184.00</td><td>3.45</td><td>14.63</td><td>4.74</td><td>6.31</td></tr><tr><td>JSONSchemaBench</td><td>DINGO</td><td>182.42</td><td>350.97</td><td>1224.94</td><td>36.86</td><td>118.83</td></tr><tr><td></td><td>NFA</td><td>214.94</td><td>353.62</td><td>1054.28</td><td>36.99</td><td>130.63</td></tr><tr><td></td><td>DConstrain</td><td>163.09</td><td>3.60</td><td>264.38</td><td>5.03</td><td>31.35</td></tr><tr><td>ComplexFuncBench</td><td>DINGO</td><td>404.39</td><td>1053.46</td><td>1405.18</td><td>90.61</td><td>107.51</td></tr><tr><td></td><td>NFA</td><td>409.39</td><td>1053.28</td><td>1405.17</td><td>121.60</td><td>139.51</td></tr><tr><td></td><td>DConstrain</td><td>309.13</td><td>13.08</td><td>20.62</td><td>5.11</td><td>5.94</td></tr></table>

Property order has opposite runtime effects. DINGO-Opt’s solve latency per forward rises from 32.45 to 74.18 ms for LLaDA2.2 and from 216.72 to 3195.39 ms for DiffusionGemma when arbitrary order is allowed (Table 6). Enumerating field histories enlarges the finite-state graph, increasing the transitions processed by Viterbi inference; Figure 3 and Appendix D.2 show the associated state-space and solver costs. LAVE-Opt and EPIC-Opt show the opposite trend: LLaDA2.2 overhead falls from 240.9% to 139.7% and from 2917.6% to 1926.2%, respectively; DiffusionGemma shows the same direction. In our integrations, reusable schema-specific verification structures are prepared during cold compilation, and both orders use the same online verification pipeline. A broader accepted language admits more proposals and reduces repeated resampling and verification per committed token. TPF increases reflect this change: LAVE-Opt rises from 4.67 to 5.75 on LLaDA2.2, and EPIC-Opt from 3.19 to 7.51 on DiffusionGemma.

Efficient joint inference under arbitrary order. DConstrain makes arbitrary-order joint inference practical. Under Any-order, its overhead is 54.1% for LLaDA2.2 and 49.9% for DiffusionGemma on the common cohorts. It achieves the highest TPS among the evaluated constrained methods in every model–order setting and the highest Any-order semantic accuracy on the common cohorts, at 83.15% and 84.87%. On the full pool, it reaches 81.60% and 81.56% with complete coverage and 100% schema validity. The paired comparison in Table 6 shows 6.65× and 46.26× lower end-to-end latency than DINGO-Opt under Any-order. The following analyses connect these results to the compiled representation and its scaling with runtime context.

Factorized representation. Table 5 compares the stored representations. Finite-state compilation materializes context-expanded states and token-level transitions, whereas DConstrain stores the local RTN structure and symbolic transition programs. Its median recorded payload is 4.74–5.11 MB across the three datasets, compared with 36.86–90.61 MB for DINGO. The runtime experiments measure the cost of deferring work to warm compilation.

Scaling with property order and nesting depth. Figure 3 varies the two runtime-context dimensions: property order enlarges the parsing environment, and nesting enlarges the continuation stack. As either grows, DINGO-Opt increasingly hits the time or memory limit and fails all evaluated cases at depths 5–8, while DConstrain still compiles them, consistent with Table 1.

![](images/720ad3dc8a2d1e7a297d613c7b4bf5fd59edf0bb817f4fef0b49a58f5736517a.jpg)  
(a) Any order: time

![](images/1cae18bd803244d0c5670dcf72d830932b47b10f1934933d43ea4c1fdd8e1a7e.jpg)  
(b) Any order: states

![](images/9c940a79f13bf67352592ebbdd1657e932cf540806e8cb44f5602dd1cacf2238.jpg)  
(c) Depth: time

![](images/38597a15f18e31d5e7f10cf67d06d71cfbdbedc0601c34f73def16b5ab5b6cb6.jpg)  
(d) Depth: states  
Figure 3. Scaling with runtime-context complexity on a fixed 300-case BFCL subset. (a,b) Arbitrary-order objects are ranked by maximum parameter-object field count; (c,d) nesting depth varies from 4 to 8.

## 6 Conclusion

We presented DConstrain, a compiler for constrained decoding in diffusion language models. Finite-state compilers expand runtime contexts into states. For unordered properties, this blow-up is exponential even for context-free grammars, so existing exact decoders fix the property order. DConstrain instead compiles reusable context-parameterized transition programs, which keep the stack and the field history as runtime parameters and still give joint inference exact successors. Warm compilation and grammar propagation overlap the model forward pass, and token-effect grouping shrinks the graph while preserving the ungrouped graph’s Viterbi optimum and token-labeled path distribution. Together, these choices substantially improve compilation coverage and latency and reduce constrained-decoding overhead while preserving schema validity and task accuracy. They also make arbitrary property order practical, which further improves task accuracy over fixed-order constraints.

## References

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=tyEyYT267x.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured Denoising Diffusion Models in Discrete State-Spaces. In Advances in Neural Information Processing Systems, volume 34, pp. 17981–17993, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 958c530554f78bcd8e97125b70e6973d-Abstract.html.

Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Guiding LLMs the right way: Fast, non-invasive constrained generation. In International Conference on Machine Learning, pp. 3658–3673, 2024. URL https://proceedings.mlr.press/v235/beurer-kellner24a.html.

Tiwei Bie, Jiansheng Cai, Maosong Cao, Xiang Cao, Binbin Chen, Fuyuan Chen, Kun Chen, Yuqi Ding, Lun Du, Zhuocheng Gong, Yanmei Gu, Kaiyuan Guan, Xiaoxu Guo, Kairong Han, Zenan Huang, Zijin Huang, Zhenzhong Lan, Chengxi Li, Jianguo Li, Zhe Li, Ruiqi Liang, Xinyao Lin, Huabin Liu, Lin Liu, Xinyi Liu, Xuejie Liu, Guoshan Lu, Yuan Lu, Yuxin Ma, Chenhui Mao, Xiaojing Qi, Kaida Qiu, Xiaoyu Shi, Yiding Tian, Huanyu Wang, Qi Wang, Rui Wang, Shuaidi Wang, Lanning Wei, Tao Wu, Siyang Xiao, Yipeng Xing, Chenkai Xu, Song Yan, Ying Yan, Jingyi Yang, Yichun Yang, Liangyu Zha, Tianze Zhang, Yifan Zhang, Junbo Zhao, Wenqian Zhao, Da Zheng, Jianyuan Zhong, Jiahao

Zhou, Junlin Zhou, Tianyu Zhou, Liwang Zhu, and Yihong Zhuang. LLaDA2.2: Enabling agentic diffusion language models via levenshtein editing. Technical report, Inclusion AI, 2026. URL https: //github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2\_2\_tech\_report.pdf.

Meihua Dang and Stefano Ermon. Constrained decoding for diffusion language models via efficient inference over finite automata, 2026. URL https://arxiv.org/abs/2607.07026. arXiv:2607.07026.

DiffusionGemma Team, Adrien Ali Taïga, James Assiene, Daniele Calandriello, Rahma Chaabouni, João Gante, Tamara von Glehn, Nate Keating, Chris Knutsen, Martin Kukla, Tianlin Liu, Ivan Lobov, Ofir Nabati, João Gabriel Oliveira, Nicolas Perez-Nieves, Nastasia Prutianova, Bobak Shahriari, Jean Tarbouriech, Pavel Tyletski, Ça˘glar Ünlü, Cindy Wu, Glenn Cameron, Jerome Connor, Sertan Girgin, Maarten Grootendorst, Alon Levkovitch, Eliya Nachmani, Omar Sanseviero, Piotr Stanczyk, Quentin Berthet, Andrew Campbell, Clément Crepy, Valentin De Bortoli, Arnaud Doucet, Romuald Elie, Alexan dre Galashov, Klaus Greff, Alexis Jacq, David Ruhe, Yu-Han Wu, Sebastian Flennerhag, Brendan O’Donoghue, George Scrivener, and Shantanu Thakoor. DiffusionGemma Technical Report, 2026. URL https://arxiv.org/abs/2608.00146. arXiv:2608.00146.

Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, and Tianqi Chen. XGrammar: Flexible and efficient structured generation engine for large language models. In Proceedings of Machine Learning and Systems, volume 7. MLSys, 2025. URL https://proceedings.mlsys.org/paper\_files/ paper/2025/hash/5c20ca4b0b20b0bd2f1d839dc605e70f-Abstract-Conference.html.

Jay Earley. An efficient context-free parsing algorithm. Communications of the ACM, 13(2):94–102, 1970. doi: 10.1145/362007.362035. URL https://doi.org/10.1145/362007.362035.

Keith Ellul, Bryan Krawetz, Jeffrey Shallit, and Ming wei Wang. Regular expressions: New results and open problems. Journal of Automata, Languages and Combinatorics, 10(4):407–437, 2005. URL https: //jalc.de/issues/2005/issue\_10\_4/abs-407.pdf.

Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. Grammar-constrained decoding for structured NLP tasks without finetuning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 10932–10952. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.674. URL https://aclanthology.org/2023.emnlp-main.674/.

Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins, Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz, and Harsha Nori. JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models, 2025. URL https://arxiv.org/abs/2501.10868. arXiv:2501.10868.

Joshua Goodman. Semiring parsing. Computational Linguistics, 25(4):573–606, 1999. URL https:// aclanthology.org/J99-4004/.

John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman. Introduction to Automata Theory, Languages, and Computation. Pearson/Addison-Wesley, third edition, 2007. ISBN 978-0-321-45536-9.

Hyundong Jin and Yo-Sub Han. EPIC: Efficient and parallel inference under CFG constraints for diffusion language models, 2026. URL https://arxiv.org/abs/2606.00722. arXiv:2606.00722.

Terry Koo, Frederick Liu, and Luheng He. Automata-based constraints for language model decoding. In Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=BDBdblmyzY.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613165.

Linzhang Li, Yixin Dong, Guanjie Wang, Ziyi Xu, Alexander Jiang, and Tianqi Chen. XGrammar-2: Dynamic and efficient structured generation engine for agentic LLMs, 2026. URL https://arxiv.org/abs/2601. 04426. arXiv:2601.04426.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Advances in Neural Information Processing Systems, volume 35, pp. 4328–4343, 2022. doi: 10.52202/068431-0313. URL https://doi.org/10.52202/068431-0313.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. In International Conference on Machine Learning, pp. 32819–32848, 2024. URL https://proceedings.mlr.press/v235/lou24a.html.

Michał Moskal, Harsha Nori, Hudson Cooper, and Loc Huynh. LLGuidance: Making structured outputs go brrr. Microsoft technical article, 2025. URL https://guidance-ai.github.io/llguidance/llg-go-brrr. June 11, 2025.

Niels Mündler, Jasper Dekoninck, and Martin Vechev. Constrained decoding of diffusion LLMs with context-free grammars. In International Conference on Learning Representations, pp. 99362–99408, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ a1622fc2f8e37ea5fc696d1c13861c36-Abstract-Conference.html.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large Language Diffusion Models. In Advances in Neural Information Processing Systems, volume 38, pp. 50608–50646, 2025. doi: 10.52202/085713-1689. URL https://proceedings.neurips.cc/ paper\_files/paper/2025/hash/48b383b24230e0e6e649d9c98dae4d8c-Abstract-Conference.html.

Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, and Loris D’Antoni. Grammar-Aligned Decoding. In Advances in Neural Information Processing Systems, volume 37, pp. 24547–24568, 2024. doi: 10.52202/079017-0774. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. In International Conference on Machine Learning, pp. 48371–48392, 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

Gabriel Poesia, Oleksandr Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. Synchromesh: Reliable code generation from pre-trained language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=KmtVD97J43e.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=dHng2O0Jjr.

Subham S. Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models. In Advances in Neural Information Processing Systems, volume 37, pp. 130136–130184, 2024. doi: 10.52202/079017-4135. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pp. 68539–68551, 2023. doi: 10.52202/075280-2997. URL https://doi.org/10.52202/075280-2997.

Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 9895–9901. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.779. URL https://aclanthology.org/2021.emnlp-main.779/.

Tarun Suresh, Debangshu Banerjee, Shubham Ugare, Sasa Misailovic, and Gagandeep Singh. DINGO: Constrained inference for diffusion LLMs. In Advances in Neural Information Processing Systems, volume 38, pp. 160518–160551. Curran Associates, Inc., 2025. doi: 10.52202/085713-5362. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ eb17a2030d1bd4a1bd29531bcd626705-Abstract-Conference.html.

Aditya Thimmaiah, Lara Marinov, Jayanth Srinivasa, Haris Vikalo, Junyi Jessy Li, and Milos Gligoric. Twist, don’t tilt: Trajectory-exact constrained decoding for masked diffusion models, 2026. URL https: //arxiv.org/abs/2609.35609. arXiv:2609.35609.

A. Viterbi. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2):260–269, 1967. doi: 10.1109/tit.1967.1054010. URL https: //doi.org/10.1109/tit.1967.1054010.

Brandon T. Willard and Rémi Louf. Efficient Guided Generation for Large Language Models, 2023. URL https://arxiv.org/abs/2307.09702. arXiv:2307.09702.

William A. Woods. Transition network grammars for natural language analysis. Communications of the ACM, 13(10):591–606, 1970. doi: 10.1145/355598.362773. URL https://doi.org/10.1145/355598.362773.

Austin Wright, Henry Andrews, Ben Hutton, and Greg Dennis. JSON Schema: A media type for describing JSON documents, 2022. URL https://json-schema.org/draft/2020-12/json-schema-core. Internet-Draft draft-bhutton-json-schema-01 (JSON Schema Draft 2020-12).

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion Large Language Models, 2025. URL https://arxiv.org/abs/2508.15487. arXiv:2508.15487.

Yitong Zhang, Yongmin Li, Yuetong Liu, Jia Li, Xiaoran Jia, Zherui Li, and Ge Li. Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars, 2026. URL https://arxiv.org/abs/2602.00612. arXiv:2602.00612.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H. Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient Execution of Structured Language Model Programs. In Advances in Neural Information Processing Systems, volume 37, pp. 62557–62583, 2024. doi: 10.52202/079017-2000. URL https://proceedings.neurips.cc/paper\_ files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.

Lucen Zhong, Zhengxiao Du, Xiaohan Zhang, Haiyi Hu, and Jie Tang. ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario, 2025. URL https://arxiv. org/abs/2501.10132. arXiv:2501.10132.

## A Additional Background and Theoretical Analysis

This appendix formalizes the parser interface used in Section 3, establishes correctness of the compiler in Section 4, establishes weighted inference preservation under token-effect grouping, and analyzes compilation and graph-construction costs.

## A.1 Grammar Automata and Parser Configurations

Let $\mathcal { G }$ define an output language $\mathcal { L } _ { \mathcal { G } }$ . Finite-state constraints admit an automaton $\mathcal { M } _ { \mathcal { G } } = ( Q , \Sigma , \delta _ { \mathrm { F A } } , q _ { 0 } , F )$ with transitions $\delta _ { \mathrm { F A } } : Q \times \bar { \Sigma } \stackrel { . . } {  } 2 ^ { Q } ;$ a DFA has at most one successor per state–symbol pair, whereas an NFA may have several. For nested grammars, a pushdown representation augments local control with a runtime stack (Dong et al., 2025). Its logical configuration is

$$
c = ( q , s , e ) ,\tag{16}
$$

where s stores continuations and $e$ contains parsing variables. Guards, stack operations, and environment updates allow the same local transition logic to operate across different contexts.

Token-level transitions consume tokenizer tokens rather than individual grammar symbols (Willard & Louf, 2023; Dong et al., 2025). Let $\Gamma _ { \mathcal { G } } ^ { \mathrm { r a w } } ( c , v )$ contain all successors after consuming v, including required stack, environment, and token-boundary operations. We use the normalized relation

$$
\mathsf { T } _ { \mathcal { G } } ( c , v ) : = \mathsf { T } _ { \mathcal { G } } ^ { N } ( c , v ) = N ( \mathsf { T } _ { \mathcal { G } } ^ { \mathrm { r a w } } ( c , v ) ) ,\tag{17}
$$

where N removes statically certified dead configurations, as defined below. Relations extend to frontiers by union over their configurations. For finite automata, c reduces to a state $q .$

## A.2 Diffusion Language Models

A partially resolved window $x ^ { t } \in ( \mathcal { V } \cup \{ \left[ \mathsf { M A S K } \right] \} ) ^ { H }$ follows a fixed prefix u. At denoising iteration t, the model predicts distributions at all positions (Sahoo et al., 2024; Nie et al., 2025):

$$
\begin{array} { c } { P _ { t } = \mathrm { M o d e l } _ { \theta } ( \pmb { u } , \pmb { x } ^ { t } ) , } \\ { P _ { t } [ i , \pmb { v } ] = p _ { \theta } ( y _ { i } = \pmb { v } \mid \pmb { u } , \pmb { x } ^ { t } ) . } \end{array}\tag{18}
$$

A decoding policy selects updates for the next iteration. Multiple unresolved positions permit joint grammar-constrained token selection (Suresh et al., 2025; Dang & Ermon, 2026). Let $\mathcal { V } _ { i } ^ { t }$ be the admitted tokens at position i; fixed positions can use singleton sets. All comparisons below use identical candidate sets, prefixes, protocols, and boundary conditions.

## A.3 Correctness of Factorized Compilation

We expose $e = ( d , \eta )$ , where d is container depth and $\eta$ is the choice environment, and write $c = ( q , s , d , \eta )$ Execution states also retain input, cursor, and protocol phase. A token-complete state has finished all required boundary processing; an exhausted cursor alone is insufficient. We assume agreement only between corresponding primitive parser operations. All results use set semantics and concern completed computations; resource exhaustion is an error, not an empty result. Byte execution, compilation, and assembly are defined for every finite $w \in \mathbb { B } ^ { \ast }$ over the byte alphabet B. Token compilation specializes $w = \mathrm { b y t e s } ( v )$ with its initialization/finalization. Suffixes need not encode vocabulary tokens and retain the enclosing metadata and phase without fresh token initialization.

Static normalization. The abstract RTN soundly overapproximates concrete execution by relaxing environment guards while preserving call/return structure. For the set $\operatorname { R e t } ( q )$ of finite abstract derivations returning from $q ^ { \prime } \mathbf { s }$ current rule, define

$$
D ( q ) = \operatorname* { m i n } _ { \pi \in \operatorname { R e t } ( q ) } | \pi | _ { \mathrm { b y t e s } } , \qquad \operatorname* { m i n } \varnothing = \infty .\tag{19}
$$

Rule exits, including complete acceptance as a distinguished root exit, are productive terminals of distance zero. Consequently,

$$
\begin{array} { l l } { { D ( q ) \leq D ( q ^ { \prime } ) } } & { { \mathrm { f o r ~ z e r o - b y t e ~ e d g e s , } } } \\ { { D ( q ) \leq 1 + D ( q ^ { \prime } ) } } & { { \mathrm { f o r ~ b y t e ~ e d g e s , } } } \\ { { D ( q ) \leq D ( a ) + D ( r ) } } & { { \mathrm { f o r ~ c a l l s ~ t o ~ } a \mathrm { ~ r e t u r n i n g ~ t o ~ } r , } } \\ { { D ( q ) = 0 } } & { { \mathrm { a t ~ r u l e ~ e x i t s . } } } \end{array}\tag{20}
$$

Min-plus relaxation starts with zero at terminals and infinity elsewhere. Each finite value has a derivation witness, and saturation covers every finite return, computing Eq. (19). Thus $D ( q ) = \infty$ excludes any concrete return; finite distance need not imply concrete feasibility. Define

$$
P ( c ) = [ D ( q ) < \infty ] \wedge \bigwedge _ { r \in s } [ D ( r ) < \infty ] , \qquad N ( X ) = \{ c \in X : P ( c ) \} ,\tag{21}
$$

Complete-output accepting configurations satisfy P by their terminal distance and empty stack. Protocol steps obey the same abstract control rules and do not bypass required returns. Infinity is used only after the distance computation completes.

Lemma A.1 (Normalization safety). For raw byte execution $\delta _ { \mathrm { r a w } }$ with its prescribed zero-byte closures,

$$
N ( \delta _ { \mathrm { r a w } } ( N ( X ) , w ) ) = N ( \delta _ { \mathrm { r a w } } ( X , w ) ) .\tag{22}
$$

The same property holds for closures and complete token execution, including normalization at intermediate stages.

Proof. Every primitive step satisfies $c  c ^ { \prime } \wedge P ( c ^ { \prime } ) \Rightarrow P ( c )$ . For local and byte edges this follows from the distance inequalities. For calls, productivity of the successor implies finite distances for the callee and pushed return address. For returns, the source is an exit, and the popped address and remaining stack have finite distances. Consequently, every finite execution ending in a retained configuration contains only retained configurations. Such a path survives initial and intermediate normalization; the opposite inclusion follows from $N ( X ) \subseteq X$ □

Write $\delta _ { N } ( X , w ) = N ( \delta _ { \mathrm { r a w } } ( X , w ) )$ ). Unlike the raw relation, this semantics excludes token endpoints that are statically known never to complete the output.

Representations and admissibility. A parameterized bundle $B = \left( \mathbf { a } , \mathbf { R } , \eta _ { 0 } \right)$ contains atoms $a _ { i } = \left( q _ { i } , d _ { i } , \Omega _ { i } \right)$ and their own stack relations $R _ { i } .$ Its logical interpretation is

$$
\mathcal { D } ( B ) = \bigcup _ { i } \{ ( q _ { i } , s , d _ { i } , \Omega _ { i } ( \eta _ { 0 } ) ) : s \in R _ { i } , \ \Omega _ { i } ( \eta _ { 0 } ) \neq \bot \} .\tag{23}
$$

Here $\Omega _ { i }$ is an ordered program of partial choice operations. A public representation is admissible when its atom–stack–program associations are preserved; control, frames, operands, and cursor/protocol metadata are well formed in their binding scope; and its decoded frontier is normalized and in entry-closure form. Initialization creates aligned atoms and applies closure and N. The transfers below preserve these invariants; internal seeds carry their specified suspended phases.

Updates modify the current environment; a non-committing guard obeys

$$
{ \mathsf { c h e c k } } _ { \{ U _ { j } \} } ( \eta ) = { \left\{ \begin{array} { l l } { \eta , } & { \exists j : U _ { j } ( \eta ) \neq \bot , } \\ { \bot , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{24}
$$

Each trial starts from the same current environment. Ordered composition satisfies $( U ; V ) ( \eta ) = V ( U ( \eta ) )$ including failure.

Source-indexed decoding. At each token query’s entry, label every source c by its immutable origin $o = c .$ . Source selectors and stack transformations denote a relation $\mathcal { R } ( \widehat { Y } )$ between origins and current configurations. Define

$$
\mathcal { D } _ { o } ( \widehat { Y } ) = \{ c ^ { \prime } : ( o , c ^ { \prime } ) \in \mathcal { R } ( \widehat { Y } ) \} , \qquad \mathcal { D } ( \widehat { Y } ) = \bigcup _ { o } \mathcal { D } _ { o } ( \widehat { Y } ) .\tag{25}
$$

Initially $\mathcal { R } = \{ ( c , c ) : c \in \mathcal { D } ( \widehat { X } ) \}$ . Transfers keep the origin while changing the current configuration. Stack operations act pointwise on the selected atom’s input stack, not just its index. Merging unions these relations; shared suffixes compose them through the same intermediate configuration, never through a Cartesian product. Origins are proof annotations on existing selectors and stack maps, fixed within a query and reset at the next query.

Independent byte execution. The byte executor matches source atoms, appends call frames, partitions returns by their actual top frame, and appends choice operations in order. Its closure explores all prescribed zero-byte alternatives, deduplicates full current states with their source associations, and applies N to each atom’s own stack relation. It does not invoke local-effect assembly.

Lemma A.2 (Source-indexed byte simulation). For every origin o and byte string w, the specified byte executor with matching closure conventions satisfies

$$
\begin{array} { r } { \mathcal { D } _ { o } \Big ( \widehat { \delta } _ { \mathrm { b y t e s } } ( \widehat { Y } , w ) \Big ) = \delta _ { N } \Big ( \mathcal { D } _ { o } ( \widehat { Y } ) , w \Big ) . } \end{array}\tag{26}
$$

Closure and token-finalization satisfy the analogous equations, and full public operations return admissible representations.

Proof. Fix o. Primitive agreement gives the same byte, stack, and choice successors for each of its configurations, retaining o. Extracting an identical, equally bound prefix U preserves $V _ { i } ( U ( \eta _ { 0 } ) ) = ( U ; V _ { i } ) ( \eta _ { 0 } )$ and the origin associations. Merging identical current states unions their origins and shares their future relation. Newly attached origins receive those successors by propagation or reuse of that relation; saturation therefore covers every origin’s finite closure paths. Normalization acts per endpoint, so Lemma A.1 applies to each origin. Induction over bytes, followed by the same finalization argument, proves the equation. Transfers preserve scopes and phases; public exit closure and N restore admissibility. □

Local summaries and worklist propagation. For arbitrary $w \in \mathbb { B } ^ { \ast }$ , cold compilation scans bytes and expands local closures using items $h = ( a , \rho , \Delta , m , \Omega , \nu )$ . Here a includes control and execution phase, $\rho$ is the unmatched local return stack, ∆ is net depth change, m is its minimum prefix change including zero, and ν retains additional used metadata. Known calls and local returns modify $\rho ;$ choice operations append to Ω. Initially $\rho = \Omega = \varepsilon$ and $\Delta = m = 0$ . A depth change $\zeta$ updates

$$
( \Delta , m ) \gets ( \Delta + \zeta , \operatorname* { m i n } \{ m , \Delta + \zeta \} ) .\tag{27}
$$

Every explicit alternative is explored. A return with empty $\rho$ emits a residual before accessing the unknown incoming stack. An ordinary summary $P = ( q ^ { \prime } , \rho , \Delta , m , \Omega , \nu )$ binds as

$$
\begin{array} { r } { \mathrm { B i n d } ( P ; s , d , \eta ) = \left\{ \begin{array} { l l } { \{ ( q ^ { \prime } , s \parallel \rho , d + \Delta , \Omega ( \eta ) ) \} , } & { d + m \geq 0 , \Omega ( \eta ) \neq \perp , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{28}
$$

Frames are appended bottom-to-top. Additional checks retain their primitive tests and operands; a depthdependent check other than underflow also records its relative evaluation offset, or sufficient trace statistics, in keyed metadata ν. Binding evaluates these tests and preserves the source selector and input-stack argument. Structural steps have no unrecorded choice effects; execution phase determines remaining closure.

The worklist keys by $( a , \rho , \Delta , \Omega , \nu )$ , excluding only m. Each new key or strict increase of its best m is enqueued. A popped item is skipped only when its m is stale; otherwise all successors are explored. Ordinary and residual output tables also retain the largest m.

Lemma A.3 (Summary and worklist preservation). For every concrete input binding, saturation preserves its local reference endpoints, cut before the first unknown-input-stack return, and attributes them to that same source.

Proof. Along a local trace the stack is $s \parallel \rho .$ If $h _ { j }$ denotes its cumulative depth change, then $\Delta = h _ { k }$ and $m = \operatorname* { m i n } _ { j } h _ { j } ;$ the underflow checks succeed exactly when $d + m \ge 0$ . Ordered interpretation gives precisely $\Omega ( \eta ) ;$ ; other retained tests use their recorded operands and offsets. Each summary therefore has the same endpoint and failure domain as its witnessing trace, for that fixed input binding.

For equal keys and $m _ { 1 } \leq m _ { 2 }$ , the first success domain is contained in the second, and outputs agree. A common suffix with summary $\left( \Delta _ { U } , m _ { U } \right)$ yields

$$
( \Delta , m ) ; ( \Delta _ { U } , m _ { U } ) = ( \Delta + \Delta _ { U } , \mathrm { m i n } \{ m , \Delta + m _ { U } \} ) ,\tag{29}
$$

which is monotone in m. Dominance therefore persists through successors. Every retained item has a trace witness, giving soundness. Conversely, each prefix of any finite trace is retained or dominated by a retained item. Re-enqueuing improvements propagates this coverage to outputs at saturation, giving completeness. The inclusion and propagation argument holds for each input separately, and binding never changes its origin. Applying N preserves this sourcewise equality. □

Residual resolution. Whole-token replay consumes the original token from its original source, without first applying the compiled prefix. Source-indexed byte simulation gives that source’s full normalized successor set; union with ordinary results already contained in that set does not change it or its origin.

Suffix restoration instead applies the prefix once, pops the actual source-stack return frame once, and continues in that caller with its retained environment, input, and protocol phase. The compiler’s initial residual set is empty; new records appear only after a byte is consumed, with cursor equal to its zero-based index plus one. Propagation preserves that cursor, and empty input generates only the identity effect. Hence compiler-generated residuals satisfy

$$
1 \leq j \leq | w | , \qquad | w [ j : ] | < | w | .\tag{30}
$$

Entry and caller closures handle zero-byte returns independently.

Lemma A.4 (Suffix assembly preservation). For every finite byte string w, compiler-generated effects, and origin o, assembly with its required closures satisfies

$$
\begin{array} { r } { \mathcal { D } _ { o } \Big ( \widehat { A } ^ { \mathrm { c l } } ( \widehat { Y } , w ) \Big ) = \delta _ { N } \Big ( \mathcal { D } _ { o } \big ( \widehat { Y } \big ) , w \Big ) . } \end{array}\tag{31}
$$

Proof. Fix o and induct strongly on |w| over all finite byte strings, with compatible input and protocol phases. Empty input uses the identity effect and source-indexed closure simulation. Otherwise, summary and worklist preservation cover every ordinary endpoint and first residual cut from this origin. At a cut, prefix binding, the actual return, and caller closure recover its reference continuation without changing o. The suffix is shorter by Eq. (30), so induction applies whether or not it is a tokenizer token. Its results are composed with the prefix through the same caller configuration. An empty suffix can return an intermediate seed; the enclosing closure still processes its pending returns and checks. Union over alternatives and normalization commutation prove the claim for o. □

Token effects and transition preservation. Let ${ \mathcal { E } } ( q , v )$ be the token specialization of the byte-string compiler, including its prescribed initialization/finalization and residual modes. Define

$$
\displaystyle \mathrm { C a n o n } ( \mathcal { E } ) = \mathrm { S e r } ( \mathbb { R } \mathrm { e d } ( \mathcal { E } ) ) ,\tag{32}
$$

where Red applies the proved duplicate and dominance reductions, and Ser completely encodes the resulting alternatives. It retains ordered operations, bindings, and all ordinary and residual token-boundary, initialization/finalization, and protocol information used by execution. Residuals additionally retain input, cursor, source bindings, and replay mode. Whole-token selectors retain token identity. Interned operands are compared within the same immutable registry and binding scope; hashing must resolve full key equality. Let $\Pi _ { q }$ partition tokens by these keys.

Lemma A.5 (Token-effect preservation). For fixed q and scope, equal canonical effects imply identical normalized successor sets for each admissible concrete source, with the same source attribution.

Proof. The reductions preserve unions by the summary and worklist lemma. Structural equality identifies the same ordinary bindings, residual computations, and finalization for a fixed source. Their completed outcomes agree and retain that source. This is a sufficient equivalence test, not a complete semanticequivalence procedure. □

Theorem A.6 (Sourcewise normalized token-transition preservation). For an admissible public source $\widehat { X }$ and compiler-generated effects, completed factorized token execution returns an admissible representation and, for every $c \in { \dot { \mathcal { D } } } ( { \widehat { X } } )$ , satisfies

$$
\boxed { \mathcal { D } _ { c } \Big ( \widehat { \mathsf { T } } ( \widehat { X } , v ) \Big ) = \mathsf { T } _ { \mathcal { G } } ( c , v ) . }\tag{33}
$$

Consequently,

$$
\begin{array} { r } { \mathcal { D } \Big ( \widehat { \mathsf { T } } ( \widehat { X } , v ) \Big ) = \mathsf { T } _ { \mathcal { G } } \Big ( \mathcal { D } ( \widehat { X } ) , v \Big ) . } \end{array}\tag{34}
$$

Proof. Fix the entry origin c. Source-indexed byte simulation proves replay; pointwise summary binding, worklist propagation, and suffix induction prove local-effect execution. Both preserve c through closure, normalization, and finalization. Grouping reuses the same per-source relation. Public exit closure restores the established representation invariants. Union over sources gives the aggregate equation. □

Window and complete-output languages. For normalized $C _ { 0 } .$ , allowed sets $\mathbf { A } = \left( A _ { 0 } , \dotsc , A _ { H - 1 } \right)$ , and $Y \in \{ N , \mathrm { f a c } \}$ , define

$$
\begin{array} { r l } & { \quad C _ { 0 } ^ { Y } = C _ { 0 } , } \\ & { C _ { i + 1 } ^ { Y } = \bigcup _ { \substack { c \in C _ { i } ^ { Y } } } \bigcup _ { v \in A _ { i } } \mathsf { T } ^ { Y } ( c , v ) , } \\ & { \qquad E _ { i } ^ { Y } = \{ ( c , v , c ^ { \prime } ) : c \in C _ { i } ^ { Y } , v \in A _ { i } , c ^ { \prime } \in \mathsf { T } ^ { Y } ( c , v ) \} . } \end{array}\tag{35}
$$

Here $\mathsf { T } ^ { N } = \mathsf { T } _ { \mathcal { G } }$ and $\mathsf { T } ^ { \mathrm { f a c } } ( c , v ) = { \mathcal { D } } _ { c } ( \widehat { \mathsf { T } } ( \widehat { X } _ { i } , v ) )$ for a representation of $C _ { i } ^ { \mathrm { f a c } }$ . Stored edges use these source– successor pairs, not Cartesian products of frontiers; graph nodes retain their position. Let $\mathcal { L } _ { H , B _ { H } } ^ { Y }$ contain the token sequences labeling paths from $C _ { 0 }$ to a configuration satisfying the common normalized boundary predicate $B _ { H }$

Theorem A.7 (Normalized window-language preservation). Complete on-demand construction preserves $C _ { i } ^ { \mathrm { { f a c } } } = C _ { i } ^ { N } , E _ { i } ^ { \mathrm { { f a c } } } = E _ { i } ^ { N }$ , and

$$
\boxed { \mathcal { L } _ { H , B _ { H } } ^ { \mathrm { f a c } } \big ( C _ { 0 } ; \mathbf { A } \big ) = \mathcal { L } _ { H , B _ { H } } ^ { N } \big ( C _ { 0 } ; \mathbf { A } \big ) . }\tag{36}
$$

Proof. Induct on position. Equal frontiers and Eq. (33) give the same successors for each pair $( c , v )$ , hence identical triples $\left( c , v , c ^ { \prime } \right)$ and next frontiers. Output admissibility permits the next query. Effect groups reconstruct these same edges when every admitted token label is retained. Equal labeled paths and the same boundary predicate give equal window languages. □

For a fixed token sequence, normalization commutation also gives $X _ { H } ^ { N } = N ( X _ { H } ^ { \mathrm { r a w } } )$ . Thus raw-to-normalized boundary agreement holds whenever $B _ { H } ^ { \mathrm { r a w } } ( X ) = \overline { { B } } _ { H } ( N ( X ) \bar { ) }$ . This extra condition is unnecessary when comparing factorized execution directly with the normalized reference.

Let $\mathcal { L } _ { \mathrm { c o m p l e t e } } ^ { Y }$ contain finite token sequences reaching complete-output acceptance under semantics Y.

Theorem A.8 (Complete-output language preservation). $\mathcal { L } _ { \mathrm { c o m p l e t e } } ^ { N } = \mathcal { L } _ { \mathrm { c o m p l e t e } } ^ { \mathrm { r a w } }$ . Factorized acceptance agrees with both on completed queries.

Proof. Every complete accepting configuration satisfies $P ,$ so backward closure ensures that every raw accepting path survives normalization. Normalization only deletes alternatives, proving the reverse inclusion. The token theorem then establishes factorized agreement. The same argument preserves the existence of a finite legal completion, but not observations of dead threads. □

These statements preserve logical transitions and token languages, not derivation multiplicities, arbitrary path weights, or resource-truncated results.

## A.4 Weighted Inference under Token-Effect Grouping

Fix one denoising iteration and a finite layered window graph with initial configuration $c _ { 0 } ,$ horizon $H ,$ terminal predicate $B _ { H } ,$ , and fixed nonnegative token weights $p _ { i } ( v )$ . Forbidden tokens have zero weight. All comparisons use the same normalized, sourcewise transition relation ${ \mathsf { T } } _ { \mathcal { G } }$ and count each logical edge $( i , c , v , c ^ { \prime } )$ once. Let $\Pi _ { q }$ partition V and assume effect equivalence:

$$
\mathsf { T } _ { \mathcal { G } } ( c , v ) = \mathsf { T } _ { \mathcal { G } } ( c , G ) , \qquad v \in G \in \Pi _ { q ( c ) } .\tag{37}
$$

Positions, endpoints, and group membership are retained. An accepting path π has weight $W ( \pi ) =$ $\textstyle \prod _ { i = 1 } ^ { H } p _ { i } ( v _ { i } )$ ; assume $\begin{array} { r } { Z = \sum _ { \pi \mathrm { a c c e p t i n g } } W ( \pi ) > 0 } \end{array}$ . The following equalities use exact arithmetic.

Define

$$
M _ { i } ( G ) = \sum _ { v \in G } p _ { i } ( v ) , \qquad m _ { i } ( G ) = \operatorname* { m a x } _ { v \in G } p _ { i } ( v ) .\tag{38}
$$

For $\oplus \in \{ + , \operatorname* { m a x } \}$ , let $r _ { i } ^ { + } = M _ { i }$ and $r _ { i } ^ { \mathrm { m a x } } = m _ { i }$ . Disjoint group membership and Eq. (37) give

$$
\begin{array} { r l } & { { \cal A } _ { i } ^ { \oplus } ( \boldsymbol { c } , \boldsymbol { c } ^ { \prime } ) : = \bigoplus _ { v : c ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( \boldsymbol { c } , v ) } p _ { i } ( v ) } \\ & { \quad \quad \quad = \bigoplus _ { \boldsymbol { G } : \boldsymbol { c } ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( \boldsymbol { c } , \boldsymbol { G } ) } r _ { i } ^ { \oplus } ( \boldsymbol { G } ) . } \end{array}\tag{39}
$$

Both cases use multiplication to compose edge weights, with empty reductions equal to zero. Thus grouping preserves the entire sum-product or max-product transition operator, not only its support.

Lemma A.9 (Sum-product inference preservation). Group-sum reduction with conditional token restoration preserves $Z ,$ the full token-labeled path distribution, and all token marginals under chain or tree inference using the conditionals below.

Proof. Equation (39) preserves every forward, backward, and interval message. Set $\beta _ { H } = B _ { H }$ and $\beta _ { i - 1 } =$ $A _ { i } ^ { + } \beta _ { i }$ . Chain sampling selects a valid group edge and then a token using

$$
\operatorname* { P r } ( G , c ^ { \prime } \mid c ) = { \frac { M _ { i } ( G ) \beta _ { i } ( c ^ { \prime } ) } { \beta _ { i - 1 } ( c ) } } , \qquad \operatorname* { P r } ( v \mid G , c , c ^ { \prime } ) = { \frac { p _ { i } ( v ) } { M _ { i } ( G ) } } .\tag{40}
$$

Their product is $p _ { i } ( v ) \beta _ { i } ( c ^ { \prime } ) / \beta _ { i - 1 } ( c ) ,$ , exactly the expanded edge conditional. Multiplication along a path cancels the messages and $\mathrm { g i v e s } ~ W ( \pi ) / Z$

For tree inference, write $K _ { \ell : r } = A _ { \ell + 1 } ^ { + } \cdot \cdot \cdot A _ { r } ^ { + }$ . Sample the terminal state with probability $K _ { 0 : H } \big ( c _ { 0 } , b \big ) B _ { H } \big ( b \big ) / Z$ and an interval midpoint by

$$
\operatorname* { P r } ( C _ { k } = z \mid C _ { \ell } = a , C _ { r } = b ) = \frac { K _ { \ell : k } ( a , z ) K _ { k : r } ( z , b ) } { K _ { \ell : r } ( a , b ) } .\tag{41}
$$

At leaf i with endpoints $a , b ,$ choose G with probability $M _ { i } ( G ) \mathbf { 1 } [ b \in \mathsf { T } _ { \mathcal { G } } ( a , G ) ] / A _ { i } ^ { + } ( a , b )$ , then restore v with probability $p _ { i } ( v ) / M _ { i } ( G )$ . Associativity preserves the interval operators; cancellation of the interval and leaf normalizers again gives $W ( \pi ) / Z .$ Only positive-mass events are sampled. Equality of full path distributions implies equality of all token marginals. □

Theorem A.10 (Equivalence to Dang–Ermon sampling). Consider Algorithm 1 of Dang $\boldsymbol { \mathcal { E } }$ Ermon (2026), expressed in the equivalent state-space formulation of their Appendix A.2. Use the same finite token-labeled graph, initial and terminal factors, and logical edge-counting convention as above, with per-token factors

$$
g _ { i } ( c , v , c ^ { \prime } ) = p _ { i } ( v ) \mathbf { 1 } [ c ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( c , v ) ] .\tag{42}
$$

Under Lemma A.9, the grouped tree sampler has the same partition function and output-token distribution as this instance of Algorithm 1.

Proof. Their state-space transition matrix is

$$
\mathsf { M } _ { i } ( c , c ^ { \prime } ) = \sum _ { v } g _ { i } ( c , v , c ^ { \prime } ) = A _ { i } ^ { + } ( c , c ^ { \prime } ) .\tag{43}
$$

Equation (39) preserves this matrix under group-sum reduction. Thus all bottom-up interval messages, the root boundary distribution, and the conditional midpoint distributions are unchanged.

At a leaf with endpoints $c , c ^ { \prime } .$ , selecting a group and restoring $v \in G$ gives

$$
\frac { M _ { i } ( G ) \mathbf { 1 } [ c ^ { \prime } \in \mathsf { T } _ { \mathcal { G } } ( c , G ) ] } { A _ { i } ^ { + } ( c , c ^ { \prime } ) } \frac { p _ { i } ( v ) } { M _ { i } ( G ) } = \frac { g _ { i } ( c , v , c ^ { \prime } ) } { A _ { i } ^ { + } ( c , c ^ { \prime } ) } ,\tag{44}
$$

which is the same leaf token conditional. Only positive-mass events are sampled. Induction over the sampling tree therefore gives the same state-space joint distribution and hence the same output-token distribution. The identical interval operators and boundary factors also give the same partition function.

Lemma A.11 (Max-product inference preservation). Group-max reduction preserves the Viterbi optimum under chain or tree composition. All maximizing token sequences can be recovered when all tied token and state-path choices are retained

Proof. Equation (39) preserves every max-product message. For a fixed grouped state path,

$$
\operatorname* { m a x } _ { v _ { i } \in G _ { i } } \prod _ { i = 1 } ^ { H } p _ { i } ( v _ { i } ) = \prod _ { i = 1 } ^ { H } m _ { i } ( G _ { i } ) .\tag{45}
$$

Every token replacement is feasible by effect equivalence. Hence each optimal grouped path lifts to an optimal token path by choosing group maximizers, and every positive-weight expanded optimum is recovered this way. Retaining all ties preserves the full maximizing sequence set. A single argmax and backpointer return one optimum; identical selected outputs additionally require the same tie-breaking rule. □

Parsing ambiguity and the probability measure. Let $n ( y )$ be the number of accepting paths labeled by token sequence y in the chosen token-level graph, and let $\begin{array} { r } { w ( y ) = \prod _ { i } p _ { i } ( y _ { i } ) } \end{array}$ . Summing path weights gives

$$
Q _ { \mathrm { p a t h } } ( y ) = \frac { n ( y ) w ( y ) } { \sum _ { z } n ( z ) w ( z ) } , \qquad Q _ { \mathrm { s e q } } ( y ) = \frac { \mathbf { 1 } [ n ( y ) > 0 ] w ( y ) } { \sum _ { z : n ( z ) > 0 } w ( z ) } .\tag{46}
$$

The second distribution counts each legal sequence once. Cancelling positive $w ( y )$ shows that the two distributions coincide if and only if $n ( y )$ is constant on the positive-weight legal support; unambiguity is sufficient. A constant $n ( y ) = k$ gives the same normalized distribution but scales its partition function by k. For example, equal-weight sequences with path counts two and one have path probabilities $2 / 3$ and ${ } ^ { 1 / 3 , }$ rather than $1 / 2$ each. Thus language equivalence alone does not establish sum-product equivalence across different parser representations. Grouping preserves the chosen graph’s measure, not an assumed absence of parsing ambiguity.

These results require the stated weighted conditionals; greedy selection by group mass is generally not equivalent. Extra token-dependent scores require corresponding weighted reductions and token-restoration rules. Distributional equality does not imply identical outputs under the same random seed or floating-point reduction order.

## A.5 Complexity Analysis

We separate compilation and grammar propagation from model inference, token-probability processing, and joint inference.

Context expansion. Let $Q _ { \mathrm { l o c } }$ be the local grammar states, $g > 1$ the number of possible continuation-frame symbols, $d _ { \mathrm { m a x } }$ the stack-depth bound, and B the maximum number of remaining context assignments per local-state–stack pair. Then

$$
S _ { d _ { \mathrm { m a x } } } \leq | Q _ { \mathrm { l o c } } | B \sum _ { \ell = 0 } ^ { d _ { \mathrm { m a x } } } g ^ { \ell } = O ( | Q _ { \mathrm { l o c } } | B g ^ { d _ { \mathrm { m a x } } } ) .\tag{47}
$$

The stack factor is $d _ { \operatorname* { m a x } } + 1$ for $g = 1$ . The bound need not be attained. For bounded-depth brackets with $g$ types, however, each opening-only prefix has a distinct matching completion. A minimal complete DFA needs one state per stack plus a sink:

$$
S _ { \mathrm { D F A } } ^ { \mathrm { m i n } } = 1 + \sum _ { \ell = 0 } ^ { d _ { \mathrm { m a x } } } g ^ { \ell } = \Theta ( g ^ { d _ { \mathrm { m a x } } } ) .\tag{48}
$$

This is a worst-case DFA representation gap, not a universal runtime bound.

Unordered properties. We prove Proposition 3.2. Fix $k \geq 1$ and distinct Unicode letters $a _ { 1 } , \ldots , a _ { k }$ . Let $\mathcal { G } _ { k }$ be the object schema with properties $a _ { 1 } , \ldots , a _ { k } .$ , all required, no additional properties, and the value schema $\lbrace " 0 \boldsymbol { \mathrm { n s t } } " : \boldsymbol { \mathbb { O } } \rbrace$ for every property. Let $L _ { k }$ be the set of its serializations, and write $P _ { i }$ for the pair $" a _ { i } " : \mathbb { 0 }$

DFA bound. For $S \subseteq \{ 1 , \ldots , k \}$ , let u<sub>S</sub> open the object and write the pairs $P _ { i }$ with $i \in S$ in increasing order of $i ,$ separated by commas. When $\emptyset \neq S \neq \left\{ 1 , \ldots , k \right\}$ , u<sub>S</sub> also ends with a comma. Let $w _ { S }$ write the remaining pairs in the same way and close the object, so that $u _ { S } w _ { S } \in L _ { k }$ . Now take $S \ne T$ and assume without loss of generality that some $i \in S \setminus T$ exists. The string u<sub>T</sub>w<sub>S</sub> never writes $P _ { i } ,$ because $i \not \in T$ and $w _ { S }$ writes only properties outside S. Since $a _ { i }$ is required, $u _ { T } w _ { S } \notin L _ { k }$ . The $2 ^ { k }$ prefixes are therefore pairwise inequivalent, and by the Myhill–Nerode theorem (Hopcroft et al., 2007) every DFA for $L _ { k }$ has at least $2 ^ { k }$ states.

Grammar bound. Let G be a context-free grammar for $L _ { k } ,$ and let s be its size, i.e., the total length of its productions. Let $R _ { k }$ be the regular language of strings that open an object, write exactly k pairs from $\left\{ P _ { 1 } , \ldots , P _ { k } \right\}$ separated by commas, and close the object. A DFA for $R _ { k }$ has $O ( k )$ states. Every property is required and $R _ { k }$ allows exactly k pairs, so $L _ { k } \cap R _ { k }$ contains exactly the strings that write each $P _ { i }$ once, in some order. The product construction for intersecting a context-free language with a regular language (Hopcroft et al., 2007) gives a grammar for $L _ { k } \cap R _ { k }$ whose size is polynomial in s and k. Let h be the homomorphism that keeps each $a _ { i }$ and erases every other character. Replacing every terminal of a grammar by its image under h does not enlarge the grammar, and $h \big ( L _ { k } \cap R _ { k } \big )$ is the set of all permutations of $a _ { 1 } , \ldots , a _ { k }$ . Every context-free grammar for this set has size $2 ^ { \Omega ( k ) }$ (Ellul et al., 2005). Hence poly $\boldsymbol { \mathbf { \ell } } ( s , k ) \geq 2 ^ { \Omega ( k ) }$ , and therefore $s = 2 ^ { \Omega ( k ) }$

Factorized representation. Under the parsing-environment factorization of Section 4.1, the object needs local grammar states only for its delimiters, property names, separators, and values, so O(k) of them suffice. The environment η stores the set of properties already written, a k-bit value. Programs that complete a property name carry the guard $^ { \prime \prime } a _ { i } \notin \eta ^ { \prime \prime }$ and the update “add $a _ { i }$ to $\eta ^ { \prime \prime }$ . The program that closes the object carries the guard $\mathbf { \boldsymbol { \eta } }$ contains every required property”. All 2<sup>k</sup> field histories share these programs. $2 ^ { k }$

Entry and instantiation counts. A dense context-expanded table has $S _ { d _ { \mathrm { m a x } } } | \mathcal { V } |$ entries; factorized compilation has at most $| Q _ { \mathrm { l o c } } | | \mathcal { V } |$ local-state–token entries. These are counts, not constant-time operations. For reachable frontiers $\mathcal { C } _ { i - 1 }$ , define

$$
\begin{array} { r l } & { \quad R _ { t } = \displaystyle \sum _ { i = 1 } ^ { H } | { \mathcal C } _ { i - 1 } | , } \\ & { \Pi _ { q , i } ^ { t } = \big \{ G \in \Pi _ { q } : G \cap \mathcal V _ { i } ^ { t } \neq \emptyset \big \} , \qquad K = \operatorname* { m a x } _ { q } | \Pi _ { q } | , } \\ & { \quad { \cal N } _ { \mathrm { i n s t } } = \displaystyle \sum _ { i = 1 } ^ { H } \displaystyle \sum _ { c \in { \mathcal C } _ { i - 1 } } \big | \Pi _ { q ( c ) , i } ^ { t } \big | \leq \operatorname* { m i n } \{ { \cal N } _ { \mathrm { q u e r y } } , { \cal R } _ { t } K \} , } \end{array}\tag{49}
$$

where $\begin{array} { r } { N _ { \mathrm { q u e r y } } = \sum _ { i } \sum _ { c \in \mathcal { C } _ { i - 1 } } | \mathcal { V } _ { i } ^ { t } | } \end{array}$ . Repeated configurations are counted per position. With full-vocabulary candidates, the ungrouped count is $R _ { t } | \nu |$

Construction cost. Let $T _ { \mathrm { s e t u p } }$ include shared schema lowering and static-distance analysis. Let $C ( q , v )$ include a cold entry’s queue processing, improvement propagation, and effect/key construction. Let $W _ { j }$ include a top-level instantiation’s binding, all residual descendants, closures, suffix compilation, and successor creation. For cold pairs $\mathcal { P } _ { \mathrm { c o l d } } \subseteq Q _ { \mathrm { l o c } } \times \mathcal { V } _ { \mathrm { . } }$ , actual top-level queries $| { \mathcal { I } } _ { \mathrm { i n s t } } | \leq N _ { \mathrm { i n s t } }$ , and additional indexing and label/graph work $A _ { t }$

$$
T _ { \mathrm { c o n s t r u c t } } = O \left( T _ { \mathrm { s e t u p } } + \sum _ { ( q , v ) \in \mathcal { P } _ { \mathrm { c o l d } } } C ( q , v ) + \sum _ { j \in \mathcal { I } _ { \mathrm { i n s t } } } W _ { j } + A _ { t } \right) .\tag{50}
$$

Each shared construction is charged once; recursive work included in $W _ { j }$ is not counted again. $\begin{array} { r } { \operatorname { I f } C ( q , v ) \le \tau _ { c } } \end{array}$ and $W _ { j } \leq \tau _ { w } ,$ , then

$$
T _ { \mathrm { c o n s t r u c t } } = O \left( T _ { \mathrm { s e t u p } } + \tau _ { c } | Q _ { \mathrm { l o c } } | | \mathcal { V } | + \tau _ { w } R _ { t } K + A _ { t } \right) .\tag{51}
$$

Neither per-entry cost is assumed constant for arbitrary grammars. Grouping also does not remove the work needed to discover effects or emit individual token labels.

Scope. For an L-byte entry, if every depth prefix is at least $- \kappa L ,$ then $- \kappa L \leq m \leq 0$ and each complete key admits at most κL + 1 retained values, including its first admission. This bounds re-expansion per key, not the number or size of keys. Similarly, residual progress bounds a recursive chain by L strict decreases, not the total work of its branching tree. Local stacks and pending programs may still grow, $R _ { t }$ may approach $H S _ { d _ { \mathrm { m a x } } } ,$ and K may approach |V|. Cold assets are reusable across requests with the same grammar and binding scope; graph reuse additionally requires compatible candidate sets and boundary/protocol conditions. The analysis therefore characterizes avoided replication without claiming an unconditional exponential decoding speedup.

## B Additional Experimental Details

This section provides additional implementation and evaluation settings for the experiments in Section 5.

## B.1 Implementation and Decoding Settings

Inference uses NVIDIA H20-3e GPUs with approximately 140 GiB of visible memory per device. Both models run in BF16 with model CUDA Graphs enabled. LLaDA2.2 (Bie et al., 2026) uses SGLang (Zheng et al., 2024) with tensor parallelism across four GPUs, while DiffusionGemma (DiffusionGemma Team et al., 2026) uses vLLM (Kwon et al., 2023) 0.28.0 with one GPU per model instance. DConstrain implements cold and warm compilation in C++ and joint inference in CUDA.

Cold-compilation experiments run on an Intel Xeon Platinum 8575C CPU. Each schema compilation is single-threaded, with up to 64 schemas compiled concurrently. Cold timing starts after tokenizer and service initialization and ends when the reusable compiled artifact is ready for online use. Warm compilation and model inference are excluded from cold-compilation time.

LLaDA2.2. The adaptive runtime comparison uses 32-token physical blocks, with the first block’s available positions determined by the prompt suffix. Generation is greedy with seed zero. MASK-to-token updates use a confidence threshold of 0.5, falling back to the highest-confidence MASK position when none qualifies. The token-to-token threshold is 0.0; DELETE and SPLIT are disabled. A block is committed after all MASK positions are filled and token assignments stabilize. Generation terminates at an accepted EOS or the 2,048-token output limit, with bounded update/refinement rounds; budget exhaustion is counted as failure.

DiffusionGemma. Generation uses 256-token canvases, seed zero, and the native entropy-bound sampler. The entropy budget is 0.1, and temperature decreases linearly from 0.8 to 0.4. Gumbel-max candidates are retained at entropy-selected positions while the remaining positions are re-noised, with native soft self-conditioning preserved. Generation uses at most 48 denoising iterations per canvas and terminates at EOS/EOT or the 2,048-token output limit.

## B.2 Compilation and Evaluation Settings

The BFCL pool contains 2,495 cases, obtained by screening all required cases in BFCL and excluding cases whose schemas admit no valid instance. The JSONSchemaBench (JSB) pool contains all 5,754 training schemas, and the ComplexFuncBench (CFB) pool contains all 1,000 benchmark cases.

The main cold-compilation experiments and ablation study use the full corresponding pools. The main BFCL runtime tables report both all-attempted and common runtime-success cohorts, as defined below. Unless otherwise specified, other analyses and appendix experiments use fixed subsets of 300 BFCL cases and 300 JSB schemas, and the full 1,000-case CFB pool, as applicable. Table 6 uses cases successfully completed by both DINGO-Opt and DConstrain within the designated pool. Failed compilation cases are not replaced.

For cold compilation, the nominal timeout is 180 s per schema. Success rates include all attempted cases, while compilation latency statistics use each method’s successfully compiled cases. Reported speedups divide the baseline’s arithmetic mean latency by DConstrain’s over paired successful cases. Timeouts and failures affect success rates but are excluded from these latency ratios.

BFCL completion and quality cohorts. In Table 2, cold-stage admission, runtime completion, strict-schema validity, and official BFCL semantic accuracy all use the full denominator of 2,495. N counts runtime completions regardless of schema validity; TPS and TPF use those N cases. Native-Opt has no constraintcompilation stage. DConstrain completes all 2,495 cases with 100% cold admission, runtime completion, and strict-schema validity under both models and both order settings.

For Table 3, we define four independent cohorts, one for each combination of model and property-order condition, by intersecting the runtime-success sets of Native-Opt, LAVE-Opt, EPIC-Opt, and DConstrain. Schema-invalid completions are retained. The Schema-order and Any-order cohorts contain 1,874 and 2,201 cases for DiffusionGemma, and 2,448 and 2,149 cases for LLaDA2.2, respectively. DINGO-Opt completes all Schema-order cases; its validity, TPF, and semantic accuracy on the Schema-order common cohorts match DConstrain. Under Any-order, its cold-compilation success is 85.81% for DiffusionGemma and 71.30% for LLaDA2.2. Table 2 reports its Any-order outcomes over the full pool. Including DINGO-Opt would substantially reduce the common-success cohorts in Table 3, so Table 6 compares it with DConstrain on their paired successful cases. In Table 2, Schema-order DINGO-Opt has TPF 8.11 for DiffusionGemma and 6.86 for LLaDA2.2, semantic accuracy 37.19% and 72.63%, and TPS 23.13 and 101.32, respectively.

Schema validity is evaluated independently of solver acceptance and semantic correctness. The two main BFCL runtime tables report strict-schema validity and the official semantic evaluator separately; semantic accuracy is not filtered by schema validity. Other reported joint-correctness metrics additionally require schema validity. JSONSchemaBench validates outputs against the original schema. Timeouts and exhausted decoding budgets count as runtime failures; schema-invalid completions count against validity but remain completed runs.

Table 6. Paired runtime comparison on BFCL and JSONSchemaBench (JSB). Each setting uses the common successful cases of DINGO-Opt and DConstrain. Native-Opt is unconstrained decoding. E2E is the mean request latency; Solve/forward is the mean joint-inference latency per generation forward. Subscripts give DConstrain’s speedup over DINGO-Opt.
<table><tr><td rowspan="2">Method</td><td colspan="3">LLaDA2.2</td><td colspan="3">DiffusionGemma</td></tr><tr><td>Valid. (%)↑</td><td>E2E (ms) ↓</td><td>Solve/forward (ms)↓</td><td>Valid. (%)↑</td><td>E2E (ms) ↓</td><td>Solve/forward (ms)↓</td></tr><tr><td>BFCL — fixed order</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Native-Opt</td><td>94.87</td><td>85.42</td><td></td><td>89.50</td><td>1948.23</td><td></td></tr><tr><td>DINGO-Opt</td><td>100.0</td><td>264.23</td><td>32.45</td><td>100.0</td><td>12874.32</td><td>216.72</td></tr><tr><td>DConstrain</td><td>100.0</td><td> $\mathbf { 1 1 5 . 2 1 } _ { 2 . 2 9 \times }$ </td><td> $3 . 1 6 _ { 1 0 . 2 7 \times }$ </td><td>100.0</td><td> $2 5 2 1 . 4 2 _ { 5 . 1 1 \times }$ </td><td> $9 . 9 5 _ { 2 1 . 7 8 \times }$ </td></tr><tr><td>BFCL — any order</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINGO-Opt</td><td>100.0</td><td>813.20</td><td>74.18</td><td>100.0</td><td>145529.00</td><td>3195.39</td></tr><tr><td>DConstrain</td><td>100.0</td><td> $1 2 2 . 3 5 _ { 6 . 6 5 \times }$ </td><td> $2 . 8 1 _ { 2 6 . 4 0 \times }$ </td><td>100.0</td><td> $3 1 4 5 . 6 4 _ { 4 6 . 2 6 \times }$ </td><td> $2 9 . 4 4 _ { 1 0 8 . 5 4 \times }$ </td></tr><tr><td>JSB —fixed order</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Native-Opt</td><td>73.7</td><td>169.42</td><td></td><td>55.3</td><td>1894.82</td><td></td></tr><tr><td>DINGO-Opt</td><td>100.0</td><td>2282.87</td><td>172.57</td><td>100.0</td><td>54323.33</td><td>853.25</td></tr><tr><td>DConstrain</td><td>100.0</td><td> $3 2 7 . 6 1 _ { 6 . 9 7 \times }$ </td><td> $\mathbf { 1 1 . 7 8 } _ { 1 4 . 6 5 \times }$ </td><td>100.0</td><td> $\mathbf { 5 3 8 8 . 2 1 } _ { 1 0 . 0 8 \times }$ </td><td> $3 4 . 1 7 _ { 2 4 . 9 7 \times }$ </td></tr></table>

Throughput accounting. For the main BFCL tables, TPF is the number of model non-stop token IDs divided by all generation forwards for the selected cases. CPU-generated completion text is excluded from model-token counts. The Overhead columns report inverse-TPS overhead over Native-Opt in the same model–order panel, $( \mathrm { T P S _ { N a t i v e - O p t } / T P S _ { m e t h o d } - 1 \bar { ) } \times 1 0 0 \% }$ . In Table 2 and the Schema-order panel of Table 3, DConstrain TPS subscripts give $\mathrm { T P S } _ { \mathrm { D C o n s t r a i n } } / \mathrm { T P S } _ { \mathrm { D I N G O - O p t } }$ in the same table and panel; its bold Overhead entries show the reduction factor $h _ { \mathrm { D I N G O } } / h _ { \mathrm { D C o n s t r a i n } } ,$ where $h = \mathrm { T P S } _ { \mathrm { N a t i v e { \mathrm { - } } O p t } } / \mathrm { T P S } - 1$ . Schema and Any denote schema-order and arbitrary-order settings; for Native-Opt, they identify the prompt condition.

Runtime timing. End-to-end latency is measured by wall-clock time at the request level, while GPU joint-inference stages are measured with CUDA events. Warm compilation measures the request-local successor-construction work, which overlaps with model execution; these intervals are therefore not additive. Cold compilation and artifact loading are excluded from runtime latency. Runtime latency statistics in Table 6 are arithmetic means over the paired cases successfully completed by both compared methods.

Paired latency comparison. Table 6 reports BFCL and JSB latency on the cases successfully completed by both DINGO-Opt and DConstrain in each dataset–order setting. E2E is the mean request latency. Solve/forward is the mean joint-inference latency per generation forward. On BFCL under Any-order, DConstrain reduces end-to-end latency by 6.65× for LLaDA2.2 and 46.26× for DiffusionGemma, and reduces solve latency per forward by 26.40× and 108.54×, respectively.

## B.3 Engineering Optimizations

We complement DConstrain’s compilation design with engineering optimizations that reduce execution overhead and graph-storage cost.

Native compilation pipeline. Cold and warm compilation execute in a native C++ pipeline, reducing interpreter overhead in grammar processing and transition instantiation. Reusable schema artifacts are stored in packed native arrays, avoiding per-edge Python objects and their allocation overhead.

Compact sparse graph storage. We store canvas-specific grammar graphs in compressed sparse row (CSR) format, using row offsets and contiguous edge arrays. This representation stores only existing transitions rather than allocating dense graph structures. CUDA joint-inference kernels consume the CSR representation directly, connecting grammar propagation with GPU execution.

DINGO-Opt adopts the same C++ pipeline and CSR representation while retaining finite-state compilation. Direct-PDA-Opt uses edge merging, shared token-membership storage, and grouped GPU reductions while interpreting reached parser configurations on demand, without compiling symbolic transition programs. Its + TG variant additionally applies conservative token grouping before interpretation (Appendix D.1).

Asynchronous execution. Native CPU workers run warm compilation during the first forward pass of each canvas and grammar propagation during every forward pass. Neither stage reads the model scores. The scheduling interface supports both eager execution and model CUDA Graph replay. Data transfer and readiness tracking connect warm compilation to GPU joint inference, which begins when both graph data and model scores are available. Score-dependent operations therefore remain on the critical path. This overlap limits exposed warm-compilation latency, while workspace reuse reduces repeated setup costs. We measure end-to-end latency directly rather than summing overlapping components, and report compilation and exposed costs in Section 5.1.

## C Additional Analyses of Constraint Semantics and Runtime

## C.1 Warm Compilation Scaling and Runtime Overlap

DConstrain avoids materializing context-expanded token transitions during cold compilation, but exact joint inference still requires the runtime contexts that are reachable from the current decoding canvas. Warm compilation instantiates context-specific successors on demand from shared transition programs. This section examines whether the resulting deferred work becomes a runtime bottleneck as parsingenvironment and stack complexity increase, and how this work is overlapped with model computation during execution.

Scaling with runtime-context complexity. Figure 4 complements the cold-compilation scaling analysis in Figure 3 by measuring the corresponding online cost. We consider the two sources of runtime-context growth targeted by the factorized representation. For arbitrary-order objects, increasing the number of parameter fields increases the number of possible parsing-environment configurations. For nested structures, increasing nesting depth increases the diversity of continuation-stack contexts that may be reached during grammar propagation.

On the CFB any-order sweep in Figure 4(a), median CPU warm-compilation time per 32-token block increases gradually with the maximum number of parameter fields per object. Even at the largest field counts, the median remains well below the corresponding mean model-forward time of 9.09 ms. Thus, supporting arbitrary schema-valid property order introduces additional online work, but the measured warm-compilation cost remains small relative to the model computation with which it can overlap.

Figures 4(b) and 4(c) examine nesting depth on BFCL. As depth increases from four to eight, warmcompilation work increases, consistent with a larger set of reachable stack contexts. The increase is visible both in the total warm-compilation work accumulated within a request and in the mean cost of an individual warm-compilation call. At depth eight, the mean per-call CPU cost remains below the approximately 9 ms mean model-forward time, while end-to-end request latency remains comparatively stable across the depth sweep. Across the depth sweep, mean warm-compilation cost stays below the mean model-forward time, and end-to-end latency remains comparatively stable.

(a) Fields: median and IQR  
![](images/52303d73ef2889f9a03d269ddbbedafcde53098c78d09be681fe245779d62632.jpg)

![](images/92544b1d438c353c2783d7993e5396aae7d58e1950fce0634809cab8669ac7b2.jpg)  
(b) Per-request mean and range

![](images/d7752f61668b61c99397a42fe725737f6bfb158bfae5e1f0d0a79d91c4361721.jpg)  
Figure 4. Warm-compilation scaling with runtime-context complexity. (a) CPU warm-compilation time per 32-token block on CFB with arbitrary property order, grouped by the maximum number of parameter fields per object; the line and band report the median and interquartile range. (b) Per-request end-to-end latency and accumulated CPU warm-compilation time on BFCL as nesting depth increases. (c) Mean CPU warm-compilation time per call over the same depth sweep. Bands in (b,c) show the observed range across requests. Dashed lines denote the corresponding mean model-forward time.

Warm compilation during asynchronous execution. The scaling experiments above characterize how online work changes with specific sources of context complexity. Figure 2 in the main text reports warm-compilation timings from actual asynchronous execution across the broader evaluation workloads. Native CPU workers run score-independent warm compilation concurrently with GPU model forward computation. Joint inference starts when both the graph and model scores are ready, so only warm compilation remaining at that point is exposed on the decoding critical path. The figure sorts the measured CPU warm-compilation times and shows the corresponding mean model-forward duration as a reference; the percentage annotations summarize the fraction of warm-compilation calls below that duration.

For LLaDA2.2 on CFB, arbitrary property order increases warm-compilation cost relative to schema order, as expected from the additional parsing-environment alternatives, while the distribution remains below the corresponding mean forward time over the plotted samples. On JSONSchemaBench, which contains substantially more heterogeneous schemas, the distribution has a longer tail: 97.17% of warm-compilation calls remain below the 8.90 ms mean model-forward duration, while the largest observed call reaches 34.61 ms.

The same pattern holds with the substantially larger 256-token canvases of DiffusionGemma. On BFCL and CFB with arbitrary property order, 99.47% and 94.20% of warm-compilation calls, respectively, are below the corresponding mean model-forward durations of 26.4 and 26.7 ms. On JSONSchemaBench, 99.3% are below the 25.64 ms mean forward duration, with a maximum observed warm-compilation time of 30.32 ms.

Implications for deferred instantiation. Together, Figures 4 and 2 clarify the trade-off introduced by two-stage compilation. DConstrain shares compiled transition logic across the runtime contexts reached during grammar propagation. It performs context-specific instantiation only when the corresponding configurations become reachable. The resulting online work grows with runtime-context complexity, as expected, but remains modest relative to model-forward computation over the evaluated regimes and is overlapped with model execution by the asynchronous schedule. The request-level means in Table 6 account for this overlap directly.

## D Additional Experimental Results

These experiments complement the main runtime and compilation results in Section 5. We examine direct pushdown interpretation, tree-parallelized automata inference, compiled representations, and compilation coverage.

## D.1 Comparison with Direct Pushdown Interpretation

The ablations in Section 5.2 vary reuse within DConstrain. Here we additionally evaluate a baseline that computes $( c , v ) \mapsto \mathsf { T } _ { \mathcal { G } } ( c , v )$ by directly interpreting the pushdown parser at reached runtime configurations, without materializing a context-expanded DFA or compiling symbolic transition programs. This comparison tests whether on-demand interpretation alone can provide efficient successor construction.

Baseline variants. Direct-PDA-Opt already merges output edges with common destinations and uses shared token-membership storage and grouped GPU reductions. Direct-PDA-Opt + TG additionally groups tokens before interpretation: bytes are equivalent only when their raw transitions agree at every local RTN node, and token-trie paths with identical byte-class sequences are merged. It also reuses unions of token memberships. Calls, returns, and environment updates still execute in the interpreter, and all original token identities and model scores are retained. This conservative grouping uses no symbolic transition programs and differs from DConstrain’s local token-effect grouping.

Evaluation setting. We evaluate all variants on the same fixed set of 300 BFCL cases, with LLaDA2.2, BF16, tensor parallelism of four, batch size one, and 32-token blocks. All variants use the same name-first tool-call format, arbitrary parameter-field order, a nesting-depth limit of four, prompt, and MAP decoding. These controlled runs execute graph construction and model forward serially, with a fresh runtime per request. E2E spans the first prefill through runtime and parser cleanup after generation; cold preparation is excluded. Figure 5 uses the original paired runs. Figure 6 reruns DConstrain and the grouped baseline in the same service on the same 300 cases. All 300 cases produce identical output tokens across variants and satisfy their schemas.

Runtime comparison. Pre-interpretation grouping substantially reduces the direct interpreter’s online cost, but a large gap remains. In the grouped paired run, mean E2E is 878.10 ms for Direct-PDA-Opt + TG and 136.63 ms for DConstrain, a 6.43× speedup. The warm-compilation curves show that repeated runtime interpretation remains expensive even after conservative grouping. Together with the compiler ablations, these results support compiling and reusing transition programs as a contribution beyond simply avoiding DFA expansion.

## D.2 Tree-Parallelized Automata Inference

DConstrain changes how the canvas-specific grammar graph is constructed while preserving the tokenlabeled successor relation consumed by downstream automata-based inference. It is therefore solveragnostic rather than specialized to the Viterbi solver used in our main experiments. We verify this compatibility by applying both Viterbi inference and the inference algorithm of Dang & Ermon (2026) directly to DConstrain graphs, using both sequential and tree-parallelized implementations.

We further examine whether tree parallelism can reduce the solver cost underlying Table 6. All four solver variants are evaluated on the same 300 BFCL cases using both DINGO and DConstrain graph representations. Table 7 reports results for two cases near the 25th and 95th percentiles of DINGO graph node count, denoted P25 and P95. The selected cases and inference objectives are matched across graph representations.

We consider the log-depth tree construction proposed by Dang and Ermon (Dang & Ermon, 2026) and apply the same reduction to MAP inference for a direct comparison with the Viterbi-style decoding used by DINGO (Suresh et al., 2025). Let L denote the decoding-window length, S the automaton states, and E the grammar-transition structure. Conventional Viterbi inference performs position-wise dynamic programming, where state scores at position i depend on those at position i − 1, yielding a sequential dependency across the decoding window. Tree-based inference instead composes adjacent transition operators in a binary reduction tree, reducing the sequential depth from O(L) to O(log L). For MAP inference, the same construction can be performed over the max-product semiring (Goodman, 1999), preserving the Viterbi objective (Viterbi, 1967) while exposing tree-level parallelism.

![](images/fff1adc8ec5243f3f9d0e28c98340d8d0980f3e8062fa3741ce7a642b3a33ab7.jpg)

![](images/ff345039d75d262852e4ef67212931a57ac47b5347d793abcd94518072a20a5c.jpg)  
(b) Total warm compilation

![](images/535e02f01c8c076b5db1fdfc6c4c547a0261fb5f5f8e06d00af353d6eb454ed1.jpg)  
(c) Mean warm compilation

![](images/9db865a4a05137ca14f750c27cf584800f8cdc0b23d081da30e48f62e057bac6.jpg)  
Figure 5. DConstrain versus Direct-PDA-Opt without pre-interpretation token grouping on 300 BFCL cases with LLaDA2.2. Panels report (a) complete-request E2E, (b) accumulated warm-compilation time per request, (c) each request’s mean time per warm-compilation call, and (d) cold-compilation time. Each curve shows its latency percentiles from P0 to P95; the upper 5% is omitted without rescaling the percentile axis.

![](images/56d6f15794f848e16640ddff27ef94dd061f9bcfcd36e0f36ad6cb99c1563497.jpg)

![](images/1f5506dc52a490741b0eb7fcfabedebbde99f8f52ee508b78f63c4334a42f5e9.jpg)  
(b) Total warm compilation

![](images/ee7ea61e4bd0bd27e850ac8f322609ff0e190753a92c186e76e59241373d0898.jpg)  
(c) Mean warm compilation

![](images/4eaf42205b63429132be983f51427d4644ff4d2ebb2aadb1bcdc904faf3bce9c.jpg)  
Figure 6. DConstrain versus Direct-PDA-Opt + TG, which adds conservative pre-interpretation token grouping and membership reuse. Both methods are rerun on the same 300 BFCL cases. Panel definitions and the P0–P95 display follow Figure 5.

This reduction trades sequential depth for substantially more computation and intermediate state. Under the state-space formulation, sequential message passing requires

$$
O \Big ( L \Big ( | S | ^ { 2 } + | E | | \mathcal { V } | \Big ) \Big )\tag{52}
$$

work, whereas the tree construction requires

$$
O \Big ( L \Big ( | S | ^ { 3 } + | E | | \mathcal { V } | \Big ) \Big )\tag{53}
$$

work while reducing the critical path to $O ( \log L )$ . Dang and Ermon characterize this trade-off as an |S|-fold increase in message-passing FLOPs in exchange for greater GPU parallelism. The memory cost can also increase substantially: each interval in the reduction tree materializes a state-to-state transition summary of size $O ( | S | ^ { 2 } )$ . Retaining the summaries required by the subsequent top-down pass therefore requires ${ \dot { O } } ( L | S | ^ { 2 } )$ intermediate storage, compared with $O ( L | S | )$ backpointer storage for conventional Viterbi reconstruction, or $O ( | S | )$ score storage when reconstruction is unnecessary. This quadratic dependence on the automaton state space can become expensive for large structured-output graphs.

Table 7. Solver latency (ms) and peak solver GPU memory (GiB). All four solver variants are evaluated on 300 BFCL cases; the table displays cases near the 25th and 95th percentiles of DINGO graph node count. P25/P95 refer to graph size, not latency quantiles. “Ours” denotes DConstrain; “seq.” and $\mathrm { \Delta ^ { \prime \prime } t r e e ^ { \prime \prime } }$ denote sequential and tree-parallelized execution. DANG denotes the inference method of Dang & Ermon (2026).
<table><tr><td colspan="2"></td><td colspan="2">P25 BFCL schema</td><td colspan="2">P95 BFCL schema</td></tr><tr><td>Graph</td><td>Method</td><td>Time (ms) ↓</td><td>Mem. (GiB) ↓</td><td>Time (ms) ↓</td><td>Mem. (GiB) ↓</td></tr><tr><td rowspan="4">Ours</td><td>Viterbi seq.</td><td>3.90</td><td>0.01</td><td>30.85</td><td>0.03</td></tr><tr><td>DANG seq.</td><td>4.70</td><td>0.01</td><td>48.97</td><td>0.09</td></tr><tr><td>Viterbi tree</td><td>30.80</td><td>1.14</td><td>7824.19</td><td>12.55</td></tr><tr><td>DANG tree</td><td>60.70</td><td>1.56</td><td>4344.17</td><td>12.30</td></tr><tr><td rowspan="4">DINGO</td><td>Viterbi seq.</td><td>39.80</td><td>0.01</td><td>425.08</td><td>0.08</td></tr><tr><td>Viterbi tree</td><td>7.13</td><td>0.21</td><td>880.38</td><td>12.84</td></tr><tr><td>DANG seq.</td><td>331.32</td><td>0.02</td><td>13 536.26</td><td>0.09</td></tr><tr><td>DANG tree</td><td>11.40</td><td>0.29</td><td>3577.13</td><td>17.63</td></tr></table>

Table 7 shows that the benefit of tree parallelism depends strongly on graph size. On the P25 DINGO graph, tree-parallelized Viterbi reduces latency from 39.8 ms to 7.13 ms, a 5.58× speedup, but remains slower than sequential Viterbi on the DConstrain graph at 3.90 ms. On the P95 DINGO graph, tree-parallelized Viterbi instead increases latency from 425.08 ms to 880.38 ms, while sequential Viterbi on the DConstrain graph requires only 30.85 ms. Peak solver GPU memory on the DINGO graph increases from 0.01 GiB to 0.21 GiB at P25 and from 0.08 GiB to 12.84 GiB at P95, approximately 21× and $1 6 7 \times ,$ respectively. Tree parallelism also increases both latency and memory for Viterbi on the DConstrain graph in these two cases. These measurements illustrate that reducing sequential depth does not guarantee lower solver cost: as the graph grows, the additional dense intermediate computation and storage can dominate the available parallelism.

## D.3 Compiled Representations with DiffusionGemma

Table 8 complements the LLaDA2.2 results in Table 5. DConstrain’s median recorded memory footprint is 4.10 MB on BFCL, 4.23 MB on JSONSchemaBench, and 4.57 MB on ComplexFuncBench; the corresponding P95 values are 5.39 MB, 28.41 MB, and 5.09 MB. State counts use the three-method successful intersection, while edge and memory quantiles use each method’s successful cases.

To complement the any-order compilation results in Table 1, Table 9 compares the compiled representations produced by DINGO-Opt and DConstrain under the native tool-call protocols. All representation statistics are computed on the same successful intersection within each setting. Because DConstrain succeeds on every evaluated case, the intersection size n is determined by the finite-state baseline’s compilation coverage. The results show that arbitrary property order leads to very large context-expanded transition structures for DINGO-Opt, whereas DConstrain retains compact local RTN nodes and symbolic transitions.

## D.4 Cold Compilation with Additional Tokenizers

To evaluate whether DConstrain’s compilation behavior depends on the two tokenizers used in the main experiments, we additionally compile the same structured-output workloads with the Dream 7B and LLaDA 8B tokenizers. These experiments evaluate compilation only and do not run model inference. All settings use arbitrary property order and the same recursive structured-output grammar configuration.

Table 8. Compiled representations with the DiffusionGemma tokenizer on fixed sets of 300 BFCL cases, 300 JSON-SchemaBench schemas, and 1,000 ComplexFuncBench cases. Edges are reported in thousands; edge and memory columns show P50 and P95. Mean state counts use the three-method successful intersection (258, 202, and 421 cases, respectively); edge and memory quantiles use each method’s successful cases. States denote automaton states for baselines and local RTN nodes for DConstrain; edges are token-level versus symbolic. <sup>‡</sup>Memory measures baseline edge-array payloads versus DConstrain’s recorded native-snapshot payloads, in decimal MB, rather than total process memory.
<table><tr><td colspan="6">Compiled representation and memory footprint</td></tr><tr><td>Dataset</td><td>Method</td><td>States ↓</td><td colspan="2">Edges (103) ↓</td><td colspan="2">Memory  $( \mathrm { M B } ) ^ { \ddagger } \downarrow$ </td></tr><tr><td></td><td></td><td>Mean</td><td>P50</td><td>P95</td><td>P50</td><td>P95</td></tr><tr><td rowspan="3">BFCL</td><td>DINGO</td><td>235.21</td><td>884.93</td><td>2070.61</td><td>42.49</td><td>99.46</td></tr><tr><td>NFA</td><td>240.21</td><td>593.88</td><td>1774.72</td><td>28.53</td><td>85.25</td></tr><tr><td>DConstrain</td><td>208.61</td><td>0.76</td><td>4.00</td><td>4.10</td><td>5.39</td></tr><tr><td rowspan="2">JSONSchemaBench</td><td>DINGO</td><td>187.28</td><td>591.32</td><td>2066.61</td><td>28.40</td><td>99.24</td></tr><tr><td>NFA</td><td>223.56</td><td>596.35</td><td>3001.78</td><td>28.65</td><td>144.16</td></tr><tr><td></td><td>DConstrain</td><td>169.92</td><td>0.85</td><td>51.80</td><td>4.23</td><td>28.41</td></tr><tr><td rowspan="2">ComplexFuncBench</td><td>DINGO</td><td>460.21</td><td>1776.40</td><td>3252.16</td><td>85.32</td><td>156.17</td></tr><tr><td>NFA</td><td>465.21</td><td>2365.89</td><td>3842.17</td><td>113.62</td><td>184.51</td></tr><tr><td></td><td>DConstrain</td><td>424.23</td><td>2.35</td><td>3.70</td><td>4.57</td><td>5.09</td></tr></table>

Table 9. Compiled representations under native tool-call protocols with arbitrary parameter-field order. Statistics use the common successful intersection within each setting. States denote DFA states for DINGO-Opt and local RTN nodes for DConstrain; edges denote token-level and symbolic transitions, respectively. DConstrain successfully compiles every evaluated case.
<table><tr><td rowspan="2">Tokenizer</td><td rowspan="2"></td><td rowspan="2">Dataset Depth Success</td><td rowspan="2"></td><td rowspan="2">Method</td><td rowspan="2">States</td><td colspan="2">Edges (10³)</td><td colspan="2">Payload (MB)</td></tr><tr><td>Mean</td><td>P50</td><td>P95 P50</td><td>P95</td></tr><tr><td rowspan="4">LLaDA2.2</td><td>BFCL</td><td>4</td><td>213/300</td><td>DINGO-Opt</td><td>655.92</td><td>49124.94</td><td>161 307.59</td><td>2358.16</td><td>7743.37</td></tr><tr><td></td><td></td><td></td><td>DConstrain</td><td>164.98</td><td>3.27</td><td>8.87</td><td></td><td></td></tr><tr><td rowspan="2">CFB</td><td>4</td><td>99/1000</td><td>DINGO-Opt</td><td>409.19</td><td>77163.56</td><td>169703.33</td><td>3704.11</td><td>8146.34</td></tr><tr><td></td><td></td><td>DConstrain</td><td>247.52</td><td>9.65</td><td>15.14</td><td></td><td></td></tr><tr><td rowspan="4">DiffusionGemma</td><td>BFCL</td><td>4</td><td>2141/2495</td><td>DINGO-Opt</td><td>1737.94</td><td>9178.19</td><td>65604.92</td><td>440.59</td><td>3149.94</td></tr><tr><td></td><td></td><td></td><td>DConstrain</td><td>481.19</td><td>0.96</td><td>3.58</td><td>3.46</td><td>4.20</td></tr><tr><td rowspan="2">CFB</td><td>4</td><td>137/1000</td><td>DINGO-Opt</td><td>1379.69</td><td>19669.07</td><td>62942.46</td><td>944.23</td><td>3021.61</td></tr><tr><td></td><td></td><td>DConstrain</td><td>693.85</td><td>2.28</td><td>3.61</td><td>3.66</td><td>3.86</td></tr></table>

Table 10 shows that DConstrain retains high compilation coverage and sub-second median cold-compilation latency across both additional tokenizers. BFCL and ComplexFuncBench compile successfully in all cases, while JSONSchemaBench succeeds on 281 of 300 evaluated schemas. The identical local-state counts across tokenizers reflect the tokenizer-independent pushdown grammar structure, while token-dependent symbolic transitions, payload size, and compilation latency vary with tokenization. These results indicate that the factorized compilation design is not specific to the LLaDA2.2 or DiffusionGemma tokenizers used in the main experiments.

Table 10. Any-order cold compilation with Dream 7B and LLaDA 8B tokenizers. All runs use the same recursive structured-output grammar configuration and nesting bounds of 8/8. Statistics use successful compilations; Success reports coverage. States are mean local RTN node counts, edges are symbolic transitions, and payload is the schemaexclusive native-snapshot footprint in decimal MB. Cold compilation uses 64 concurrent single-threaded workers.
<table><tr><td rowspan="2">Tokenizer</td><td rowspan="2">Dataset</td><td rowspan="2">Success</td><td rowspan="2">States</td><td colspan="2">Edges (103)</td><td colspan="2">Payload (MB)</td><td colspan="2">Cold (s)</td></tr><tr><td>Mean</td><td>P50</td><td>P95</td><td>P50</td><td>P95 P50</td><td>P95</td></tr><tr><td rowspan="3">Dream 7B</td><td>BFCL</td><td>300/300</td><td>416.25</td><td>0.92</td><td>4.24</td><td>2.77</td><td>4.08</td><td>0.36</td><td>1.05</td></tr><tr><td>JSONSchemaBench</td><td>281/300</td><td>616.65</td><td>1.03</td><td>157.53</td><td>2.78</td><td>24.71</td><td>0.38</td><td>12.85</td></tr><tr><td>ComplexFuncBench</td><td>1000/1000</td><td>770.05</td><td>2.83</td><td>4.39</td><td>3.22</td><td>3.69</td><td>0.67</td><td>0.97</td></tr><tr><td rowspan="3">LLaDA 8B</td><td>BFCL</td><td>300/300</td><td>416.25</td><td>0.85</td><td>4.00</td><td>2.09</td><td>3.03</td><td>0.31</td><td>0.81</td></tr><tr><td>JSONSchemaBench</td><td>281/300</td><td>616.65</td><td>0.93</td><td>138.53</td><td>2.08</td><td>14.90</td><td>0.33</td><td>10.78</td></tr><tr><td>ComplexFuncBench</td><td>1000/1000</td><td>770.05</td><td>2.63</td><td>4.06</td><td>2.45</td><td>2.80</td><td>0.60</td><td>0.84</td></tr></table>

Table 11. Failure breakdown for DConstrain on JSONSchemaBench with the LLaDA2.2 tokenizer. Percentages are relative to all 5,754 schemas and rounded to one decimal place; the total is calculated from counts before rounding.
<table><tr><td>Failure cause Cases Full set (%)</td></tr><tr><td>Schema coverage and semantic limitations</td></tr><tr><td>Unsupported keyword or keyword combination 285 5.0</td></tr><tr><td>one0f branch exclusivity not provable 178 3.1</td></tr><tr><td>General al1of intersection unsupported 127 2.2</td></tr><tr><td>Nested resource scopes (id/$id) unsupported 127 2.2</td></tr><tr><td>Nested schema-dialect switching unsupported 28 0.5</td></tr><tr><td>Dynamic-property constraints conflict with empty-object subset 38 0.7</td></tr><tr><td>Regular expression outside supported dialect 4 0.1</td></tr><tr><td>Tuple-style items unsupported 1 0.0</td></tr><tr><td>$ref and outer-type intersection not provable 1 0.0</td></tr><tr><td>Resource and construction limits</td></tr><tr><td>68 1.2</td></tr><tr><td>State/transition/closure or pattern-size budget exceeded RSS limit exceeded 134 2.3</td></tr><tr><td>Compilation timeout 72 1.3</td></tr><tr><td></td></tr><tr><td>Implementation and unresolved failures 0.1</td></tr><tr><td>_deduplicate_json_values implementation error 4 Duplicate enum values rejected 3 0.1</td></tr><tr><td>Compiled language unexpectedly empty 2 0.0</td></tr><tr><td>Total failures 1,072 18.6</td></tr></table>

## D.5 Failure Analysis on JSONSchemaBench

We further analyze the remaining DConstrain compilation failures on JSONSchemaBench with the LLaDA2.2 tokenizer. Table 11 categorizes all 1,072 failed schemas out of 5,754, corresponding to an 18.6% failure rate and an 81.4% success rate. Most failures arise from unsupported schema semantics rather than compiler execution failures: semantic-coverage limitations account for 789 cases (13.7% of the full benchmark and 73.6% of all failures). Resource limits account for another 274 cases (4.8% of the benchmark), primarily RSS exhaustion and the 180 s compilation timeout. Only nine cases (0.2%) are attributable to known implementation errors or unresolved compiler behavior.

The dominant failure modes therefore reflect the current supported schema fragment rather than failures of the factorized compilation mechanism itself. In particular, oneOf/allOf reasoning and nested resource semantics account for a substantial fraction of unsupported cases. Resource failures form the second-largest category, showing that although DConstrain substantially reduces context expansion, sufficiently complex schemas can still exceed finite construction or memory budgets.

Table 12. Targeted semantic audit of 305 historical DConstrain rejections. All counts refer to schemas. Counterexamples are complete JSON texts accepted by XGrammar but rejected under the original schema. Rows use DConstrain’s first rejection cause, which need not match the constraint violated by the witness. Counterexample counts are lower bounds from a bounded search; absence of a witness does not establish semantic preservation.
<table><tr><td colspan="3"></td><td rowspan="2">Counterexamples XG 0.2.8</td><td rowspan="2">Same witnesses XG 0.1.27</td></tr><tr><td>Historical rejection</td><td>Cases</td><td>Compiled XG 0.2.8</td></tr><tr><td>one0f disjointness</td><td>178</td><td>175</td><td>43</td><td>42</td></tr><tr><td>General al10f</td><td>127</td><td>123</td><td>81</td><td>81</td></tr><tr><td>Total</td><td>305</td><td>298</td><td>124</td><td>123</td></tr></table>

Localizing frontend coverage limitations. To separate frontend coverage from factorized execution, we replayed the 178 oneOf branch-disjointness rejections and 127 general allOf rejections in Table 11 using the archived frontend from the reported experiment. All 305 rejections were reproduced by the schema-toparser frontend alone, without invoking the factorized token compiler. These failures therefore precede factorized execution; their localization does not establish that downstream compilation would succeed after extending the frontend.

Frontend acceptance versus semantic preservation. We examined the same 305 original schemas using XGrammar (Dong et al., 2025). With strict\_mode=True, XGrammar 0.2.8 returned grammars for 298 schemas. A bounded search identified 124 distinct schemas for which the native matcher accepted a complete JSON text that violated the original schema. We independently replayed all 124 counterexamples and confirmed their invalidity with jsonschema 4.26.0; 104 were also confirmed by fastjsonschema 2.22.2. We additionally found valid instances for 86 schemas, establishing their satisfiability. Replaying the same invalid texts in XGrammar 0.1.27 reproduced violations for 123 schemas. Table 12 summarizes these results.

The audit used an empty tokenizer and the native complete-string matching interface, requiring an accepting terminal state. It therefore tests membership in the compiled character language, rather than compilation against the LLaDA2.2 vocabulary or the likelihood of generating these texts during model inference. Validation followed each schema’s declared dialect, defaulting to Draft 2020-12 when unspecified. Optional format assertions and network reference retrieval were disabled.

For example, JSB schema o41599 specifies oneOf over integer, string, number, boolean, and null. Both tested XGrammar versions accept 0, which satisfies both the integer and number branches and thus violates the exactly-one requirement. Similarly, schema o44455 explicitly requires an object and contains a multi-branch allOf, yet both matchers accept 0. Both examples were confirmed by both validators.

Scope of the conclusions. Frontend acceptance alone is insufficient evidence that the original schema constraints are preserved. This targeted audit does not estimate overall semantic accuracy, establish DConstrain’s semantic correctness on all accepted schemas, or imply that every rejected schema is intrinsically difficult. We distinguish execution correctness relative to supported reference-parser semantics from frontend coverage and resource-bounded construction. Incomplete coverage, resource failures, and implementation issues remain limitations of the current system.

Limitations. DConstrain substantially reduces the cost of nested constraints during cold compilation by retaining a symbolic RTN representation rather than expanding stack contexts into finite states, and this representation can in principle express context-free structure. However, factorization does not eliminate the worst-case complexity of joint inference. During warm compilation, uncertainty over unresolved token choices can propagate across the decoding canvas and induce a rapidly growing set of reachable runtime contexts; in the worst case, this expansion can still be exponential in nesting depth or other sources of branching. DConstrain therefore extends the practical nesting depth substantially relative to finite-state baselines, but it does not provide unbounded or uniformly efficient compilation for arbitrary deeply nested grammars.