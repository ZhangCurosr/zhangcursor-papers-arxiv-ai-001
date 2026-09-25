# Epistemic-Probabilistic Model for Guarded Multi-Agent LLM Coordination

Mehdi Nasiri<sup>1</sup>, Mohammad Saeed Arvenaghi<sup>2</sup>, Sadegh Vaezi<sup>1</sup>, and Ebrahim Ardeshir-Larijani<sup>∗2,1</sup>

<sup>1</sup>Pasargad Institute for Advanced Innovative Solutions <sup>2</sup>Iran University of Science and Technology

## Abstract

Large language model (LLM)-based multi-agent systems (MAS) are increasingly used in applied AI, yet their connection to established multi-agent theory remains underdeveloped. Many such systems lack explicit representations of social knowledge and protocol-governed coordination. La Malfa et al. [9] identify these issues among four shortcomings of current LLM-based MAS. We focus on the social-epistemic and coordination aspects of this gap.

We introduce Epistemic Probabilistic Language Agents (EPLA model), a neuro-symbolic architecture for multi-agent coordination under uncertainty. EPLA model combines an Epistemic Logic Core, a Conditional Belief Engine, retrieval-augmented memory, a Policy LLM, and a Symbolic Guard that controls execution against the authoritative symbolic state. Guard feedback, Adversarial Representation Engineering (ARE), and reinforcement learning with linear temporal logic (LTL) objectives provide interfaces for later adaptation. We formalize the epistemic layer in a gossip testbed through epistemic lottery gossip models, combining view-based call histories with agent-indexed probability weights. For a restricted knowledge fragment, we prove lottery transparency and invariance under positive admissible reweighting. For a precisely defined source-compatible gossip instance, an explicit modal-depth-one translation into Apt and Wojtczak’s decidable language yields decidability of source-compatible Guards. We also prove a conditional ranking-progress bound for stochastic action selection.

Keywords: neuro-symbolic agents; epistemic gossip; epistemic probability; symbolic guards; LLM coordination.

## 1 Introduction

Large language model (LLM)-based multi-agent systems are often engineered as message-passing workflows. An agent may recommend a tool call, delegate to another agent, or forward information retrieved from a document. Fluent generation alone does not establish that the action is licensed by the agent’s information state or by the interaction protocol. In the position paper Large Language Models Miss the Multi-Agent Mark, La Malfa et al. [9] identify a recurring mismatch between LLMagent practice and multi-agent theory: current systems use multi-agent terminology, but often omit explicit social epistemics and protocol-governed coordination.

EPLA model addresses this gap through a neuro-symbolic division of responsibility: learned components propose and adapt, while explicit symbolic components represent state and control execution. The policy module uses a large language model (Policy LLM) to interpret language and propose typed communication or tool actions. The Epistemic Logic Core (ELC) maintains the authoritative symbolic state. A Conditional Belief Engine (CBE) represents graded and conditional uncertainty, drawing on conditional-belief logic [16]. Retrieval-augmented generation (RAG) supplies retrieved context [11]; EPLA model additionally requires retained records to be durable and source-linked. The Symbolic Guard determines whether a candidate action may execute. Guard feedback is intended to supervise Adversarial Representation Engineering (ARE), an empirical model-editing method [20]. A reinforcement-learning (RL) head may optimize long-horizon behavior against objectives expressed in linear temporal logic (LTL) [3, 6]. These are intended interfaces and empirical hypotheses, not implementation results. For example, an LLM may propose a call between two agents, but the Guard permits it only when the exact symbolic state satisfies the call’s specified precondition.

The formal setting is gossip-structured coordination: agents communicate pairwise, calls change what agents know, and later actions are permitted only when their epistemic preconditions are satisfied. This setting is deliberately controlled, not a claim that arbitrary tool traces already have gossip semantics. Epistemic gossip protocols have precise call-history semantics [14, 15], while epistemic probability logic simplified (EPLS) gives a compact semantics in which knowledge is probability one [17]. Combining these ideas distinguishes an LLM’s graded uncertainty from a Guard’s crisp knowledge condition. The narrow formal question is whether adding positive uncertainty weights to source-style gossip histories changes any crisp knowledge-based Guard decision.

This paper makes two contributions. The contribution is limited to the EPLAspecific architecture specification, formalization, and stated derivations; the paper does not claim priority for the individual methods or proof techniques in isolation.

1. It specifies an intended EPLA model architecture in which an exact symbolic state, a conditional-belief state, and source-linked retrieval records inform a Policy LLM; a Symbolic Guard controls execution and returns diagnostic feedback; and representation editing and temporal-objective policy learning may use that feedback to shape later proposals.

2. It defines epistemic lottery gossip models (ELGMs), which enrich gossip callhistory models with agent-indexed lottery weights. Their accessibility relation is based on an agent’s view, not merely on the subsequence of calls involving that agent. The paper proves lottery transparency and positive-reweighting invariance, transfers Guard decidability only for an explicit modal-depth-one source fragment, and gives a conditional ranking bound for stochastic action selection.

Because this version is an extended abstract, the results are accompanied by proof sketches; the full manuscript contains the complete proofs.

The formal and architectural components have diferent roles. Theorems about ELGMs concern the stated gossip abstraction; they do not imply that an implementation realizes that abstraction. The architecture locates the intended feedback paths, while the efects of its learned components remain empirical questions.

## 2 EPLA model Architecture

Figure 1 summarizes the EPLA model architecture. EPLA model treats coordination as a controlled loop rather than a free-form exchange of messages. At time t, an agent receives an observation, consults persistent memory, forms a candidate action, and submits that action to the Guard. The Guard checks the authoritative state, not a summary generated by the LLM. An accepted action is passed to the ELC for the state transition; a rejected action returns a diagnostic and structured labels that may shape later proposals.

The responsibility boundaries are deliberately asymmetric. The ELC alone maintains and commits the authoritative epistemic state; the CBE maintains graded uncertainty; and retrieval returns bounded, source-linked evidence. The Policy LLM turns these inputs into typed candidate actions, but cannot execute them. The Guard checks the exact state and returns permit or deny; its feedback supplies diagnostic labels that ARE and the RL/LTL module may use to shape later proposals, without overriding the Guard.

![](images/018b9012ac678b52e642760903f17310173b0e9cc5896f487a7068249fda0c84.jpg)  
Figure 1: Simplified single-step architecture of EPLA model. The observation supplies both the Policy LLM context and the retrieval query. The Policy LLM proposes a typed candidate using the observation, retrieved memory, and a CBE summary. The Guard checks the candidate against the authoritative Guard state, which contains the exact symbolic and protocol/resource predicates required for execution; the CBE summary remains non-authoritative context. The ELC commits an accepted symbolic update. Guard diagnostics may supervise ARE and inform RL/LTL, while post-transition outcomes and the next LTL monitor state provide an additional RL/LTL input. Every later candidate re-enters the ordinary Guard path. Eventdriven memory and CBE updates and external efects are specified in the execution model.

## 2.1 Execution Model

Let the operational state be

$$
z _ { t } = ( G _ { t } , \Sigma _ { t } , \Pi _ { t } , \mathcal { R } _ { t } , q _ { t } ) ,
$$

where $G _ { t }$ is the ELC state, $\Sigma _ { t }$ is the CBE state, $\Pi _ { t }$ is the protocol and resource state, $\mathcal { R } _ { t }$ is the retrieval memory, and $q _ { t }$ is the state of an LTL monitor when a temporal objective is active. One decision cycle has nine stages:

1. observe the current task and multi-agent state;

2. retrieve relevant events, messages, rules, and evidence from $\mathcal { R } _ { t }$ ;

3. construct a bounded context that identifies the current task, symbolic summary, and retrieved evidence;

4. ask the Policy LLM for a typed candidate action;

5. check the action against epistemic, protocol, consistency, and budget conditions;

6. return a denial and diagnostic, or prepare an accepted transition without yet performing an external side efect;

7. atomically record the accepted action and let the ELC commit $G _ { t + 1 }$ ; then perform any external side efect under a durable intent record and update $\Sigma _ { t }$ $\Pi _ { t } .$ , and $\mathcal { R } _ { t }$ from the resulting event;

8. when ARE is enabled, form representation-level supervision from the Guard labels; and

9. when the RL head is enabled, update its policy and value estimates from the next monitor state and acceptance signal.

A conforming implementation must evaluate the Guard and commit the accepted ELC successor against the same versioned state. It must either serialize those operations or revalidate the action when the state version changes. External side efects also require a durable intent record and an idempotent or compensating execution rule; these concurrency and recovery mechanisms are requirements, not features established by the present formal model.

The formal model has no human override. At the architecture level, a denied request for a policy exception, or a high-impact action whose semantics are not formalized, is sent to a designated human reviewer. Human approval may authorize a new typed action or a rule change, but it does not make a failed Guard predicate true; any approved action must enter through the Guard and ELC again.

The action schema separates natural-language generation from state transition. A candidate action includes an action type, participating agents, structured arguments, an evidence reference set, and an optional natural-language realization. The ELC determines the operational efect of an accepted action. The architecture retains three communication action types: Call, Message, and Announce. This paper gives operational semantics only to a pairwise push-pull Call, because that action has both a precise secret-union transition and source-verified view semantics [2, 14]. Formalizing Message or Announce would additionally require a content language, recipient and observability rules, a truth and ambiguity policy, and exact preconditions and state-update rules. Those choices are not fixed in this draft, so the two action types remain in the architecture rather than receiving invented formal semantics.

## 2.2 Conditional Belief, Retrieval, and Guard Feedback

The CBE maintains qualitative conditional-belief queries of the form $\mathsf { B } _ { a } ( \varphi \mid \psi )$ following van Eijck and Li [16], and may separately maintain numerical belief estimates as proposed for EPLA model. The cited conditional-belief operator itself is not a numerical degree. CBE state is updated after communication and may be summarized for the Policy LLM. The exact symbolic state, rather than a lossy summary, remains the reference used by the Guard. The ELGM developed below supplies a precise formal abstraction of graded uncertainty over gossip histories; it does not assume that every CBE implementation uses the same representation. In that formalization, theorem-backed Guard predicates are restricted to a crisp ELC knowledge fragment defined in Section 3; numerical or conditional CBE queries require separate semantics and proof obligations.

RAG supplies persistent context that would otherwise be lost from a bounded prompt. Each memory record should retain an event identifier, source or provenance link, timestamp or logical order, access scope, and confidence or validation metadata. Retrieval is useful for reconstructing task context and protocol rules, but retrieved text is evidence for a decision, not a substitute for the state predicates checked by the Guard.

The Guard has two outputs. Its execution output is a binary decision over the typed action. Its learning output is a critique that names the failed condition, the relevant state facts, and a machine-readable label such as call-permission, truthfulness, or protocol-compliance. This second output permits failures to become diagnostic data rather than silent discarded candidates.

## 2.3 Representation Editing and Temporal Policy Learning

ARE and LTL-constrained RL address diferent parts of the architecture. EPLA model proposes using Guard-derived target and anti-target labels to edit representations associated with behavioral concepts such as permission, truthfulness, and protocol compliance. Whether those labels produce valid and efective ARE supervision is an empirical hypothesis. ARE does not replace the Guard, and it is not identified with the lottery-reweighting operator in Section 6.

The RL head addresses action selection over a trajectory. For example, a temporal objective may require protocol violations never to occur and every agent eventually to learn every secret. The monitor state and Guard outcome then provide signals for optimizing policy behavior over multiple decisions. ARE and RL/LTL occupy diferent architectural roles, and neither is assumed to replace the Guard.

## 2.4 Scalability and Conformance Requirements

We identify four engineering issues that a future implementation should address. First, Guard checks may need tiered validation, caching, incremental state updates, or compiled monitors so that exact checking does not become an uncontrolled bottleneck. Second, the ELC/CBE state and retrieval memory may need event-sourced logs, replayable updates, and checks for disagreement between compact Policy summaries and exact Guard state. Third, multi-agent scaling requires explicit topology and memory-partition choices rather than an assumption of all-to-all communication. Fourth, ARE and RL updates need an evaluated training schedule that can detect and limit interference; possible strategies include timescale separation, partial freezing, and policy anchoring. These are proposed implementation strategies and evaluation obligations, not methods or performance properties established by the formal results.

For the formal analysis below, we project the architecture to a call-only abstraction: retain the accepted pairwise Call history, secret facts, agent views, and lottery weights, and omit budgets, retrieved text, free-form messages and announcements, ARE parameters, and the LTL monitor. At the secret-only root, a permitted $\mathrm { C A L L } ( a , b )$ appends ab and the ELC replaces both callers’ secret sets by their union; this is the transition modeled below. Message and Announce candidates have no transition in the current formal model. This projection explains the theorem boundary; it is not evidence that an implementation conforms to it.

For example, start with three agents, each of whom initially knows only its own secret. The Policy LLM proposes a call between agents a and b, with the precondition that a does not yet know b’s secret. The Guard reads the root ELC state, finds the call structurally available and the precondition true, and permits it. The ELC records the call and updates both callers so that each knows both secrets. If the same precondition is used for a second call between a and b, the Guard denies it because a now knows b’s secret; the history is unchanged and the failed-precondition label is returned as diagnostic feedback.

## 3 Preliminaries

Let Ag be a finite set of agents. Each agent initially owns one secret, and the call alphabet is $C : = \{ a b : a , b \in \mathsf { A g } , a \neq b \}$ , with ab denoting an unordered pair. A finite call history is a word $\mathbf { c } \in C ^ { < \omega }$ . The factual state used here records, for each agent, which secrets are locally available; phone-number exchange is outside the present formal instance. In the push-pull setting used in the standard gossip literature, a call ab makes the callers exchange the secrets they currently know.

The relevant epistemic object is the agent’s view. The view is richer than the bare subsequence of calls in which the agent participates: it also records the local information the agent obtains through those calls. This is the notion used in source gossip semantics [2, 14].

Definition 3.1 (Views and source-style accessibility). For each agent a, let $\mathsf { v i e w } _ { a } ( \mathbf { c } )$ be the sequence of local observations available to a after history c: calls not involving a are invisible, while a call involving a appends the call together with a’s resulting

local gossip state. Define

$$
\begin{array} { r } { { \bf c } R _ { a } { \bf c ^ { \prime } } \quad \mathrm { i f f } \quad \mathsf { v i e w } _ { a } ( { \bf c } ) = \mathsf { v i e w } _ { a } ( { \bf c ^ { \prime } } ) . } \end{array}
$$

This definition intentionally follows the view-based semantics of epistemic gossip. Equality of the call subsequence involving a is not enough: if a calls c after c has already learned another secret, then a observes a diferent local state than in a history where c had not learned it.

We use atoms $S _ { a b }$ to mean that agent a is familiar with agent $b \mathrm { ^ { \prime } s }$ secret at the current history. Let $V ( S _ { a b } )$ be the set of histories where this is true. These atoms are local for their first index: by construction of the view, if $\mathbf { c } R _ { a } \mathbf { c } ^ { \prime }$ , then $S _ { a b }$ has the same truth value at c and $\mathbf { c } ^ { \prime }$ . The factual expertise condition is

$$
\mathsf { E } _ { 0 } : = \bigwedge _ { a , b \in \mathsf { A g } } S _ { a b } .
$$

EPLS represents epistemic probabilities by lotteries over possible worlds [17]. We use the same idea, but with histories as worlds.

Definition 3.2 (Admissible epistemic lottery gossip model). An admissible epistemic lottery gossip model is a tuple

$$
\mathcal { M } ^ { G } = ( H , V , \{ R _ { a } \} _ { a \in \mathsf { A g } } , \{ L _ { a } \} _ { a \in \mathsf { A g } } )
$$

where:

1. $H \subseteq C ^ { < \omega }$ is a nonempty set of finite histories;

2. V is the factual valuation induced by the gossip state after each history;

3. $R _ { a }$ is the view-equivalence relation from Definition 3.1, restricted to $H ;$

4. $L _ { a } : H \to \mathbb { R } _ { > 0 }$ is a strictly positive lottery such that, for every $\mathbf c \in H$

$$
0 < \sum _ { \mathbf { c } ^ { \prime } \in [ \mathbf { c } ] _ { a } } L _ { a } ( \mathbf { c } ^ { \prime } ) < \infty , \qquad [ \mathbf { c } ] _ { a } : = \{ \mathbf { c } ^ { \prime } \in H : \mathbf { c } R _ { a } \mathbf { c } ^ { \prime } \} .
$$

Definition 3.3 (Knowledge fragment). The guard fragment $\mathcal { L } _ { K }$ is generated by

$$
\varphi : : = { \mathsf { T } } \mid p \mid \neg \varphi \mid ( \varphi \land \psi ) \mid { \mathcal { K } } _ { a } ( \varphi ) ,
$$

where p ranges over factual gossip atoms. It contains no numerical probability atoms such as $\mathsf { P } _ { a } ( \varphi ) \ge q$ . The source-compatible subfragment below uses only atoms $S _ { a b }$

For atoms $p ,$ write $\mathcal { M } ^ { G } , \mathbf { c _ { \alpha } } \in \mathbf { \boldsymbol { p } }$ exactly when $\mathbf { c } \in V ( p )$ The clauses for ⊤, negation, and conjunction are the usual Boolean clauses. These clauses, together with the knowledge clause below, define satisfaction for the knowledge fragment.

The finite-sum condition is not cosmetic. If H contains an infinite a-accessibility class, a constant-weight prior on that class is not admissible; a finite protocol domain or a summable length-decaying prior is therefore needed. EPLA model’s use of positive real weights is a modeling generalization. EPLS itself permits finite or countable worlds and defines lotteries with positive rational values bounded on every

$R _ { a } .$ -equivalence class; countability does not require real weights. Finite rational instances are EPLS-compatible special cases.

For an admissible ELGM, define the probability assigned by a at c to a formula $\varphi$ by

$$
\mathsf { P } _ { a } ( \varphi ) ( \mathbf { c } ) = \frac { \sum _ { \mathbf { c } ^ { \prime } \in [ \mathbf { c } ] _ { a } , \mathcal { M } ^ { G } , \mathbf { c } ^ { \prime } \vdash \varphi } L _ { a } ( \mathbf { c } ^ { \prime } ) } { \sum _ { \mathbf { c } ^ { \prime } \in [ \mathbf { c } ] _ { a } } L _ { a } ( \mathbf { c } ^ { \prime } ) } .
$$

Knowledge is probability one:

$$
\mathcal { M } ^ { G } , \mathbf { c } \in \mathcal { K } _ { a } ( \varphi ) \quad \mathrm { i f f } \quad \mathsf { P } _ { a } ( \varphi ) ( \mathbf { c } ) = 1 .
$$

Because the atoms $S _ { a b }$ are local for their first index, this knowledge semantics gives $S _ { a b }  { \ K } _ { a } ( S _ { a b } )$ Thus $\mathsf { E } _ { 0 }$ is equivalent to the usual knowledge formulation $\textstyle \bigwedge _ { a , b \in { \mathsf { A g } } } { K _ { a } ( S _ { a b } ) }$ in this model.

For the source-compatible decidability theorem only, we use the narrower gossip language ${ \mathcal { L } } _ { \mathsf { w n } }$ defined by Krzysztof R. Apt and Dominik Wojtczak in On Decidability of a Logic of Gossips and Common Knowledge in a Logic of Gossips $[ 1 , 2 ]$ In logic, a language is a grammar: it specifies which formulas are permitted. $\mathcal { L } _ { \mathrm { w n } }$ permits factual gossip formulas and an epistemic or common-knowledge modality applied directly to a factual formula, but it does not permit probability formulas or a modality inside another modality. Thus the source-compatible formula $\boldsymbol { \kappa } _ { a } ( S _ { b c } )$ is permitted, whereas $\mathcal { K } _ { a } ( \mathcal { K } _ { b } ( S _ { c d } ) )$ is not. The general EPLA model Guard fragment $\mathcal { L } _ { K }$ is diferent: it permits nested individual-knowledge formulas but excludes numerical probability atoms. Definition 3.5 defines $\mathcal { L } _ { K , \mathrm { s r c } }$ , the smaller part of $\mathcal { L } _ { K }$ that is translated into $\mathcal { L } _ { \mathrm { w n } }$

Definition 3.4 (Apt–Wojtczak source instance). Fix a finite, linearly ordered agent set A with $| { \cal A } | \geq 3$ and a bijection $b \mapsto p _ { b }$ from agents to distinct secrets. Let $C _ { \mathrm { s r c } }$ contain exactly the canonically written pairs ab with $a < b ,$ , and let $H _ { \mathrm { s r c } } : = C _ { \mathrm { s r c } } ^ { < \omega }$ be the full set of finite call sequences. The initial factual state is the secret-only root in which each agent a knows exactly $p _ { a }$ , and every call applies the source push-pull union update.

An Apt–Wojtczak source instance is an admissible ELGM whose history domain is $H _ { \mathrm { s r c } }$ , whose valuation satisfies

$$
\mathbf { c } \in V ( S _ { a b } ) \quad \mathrm { i f f } \quad p _ { b } \in \mathbf { c } ( \mathsf { r o o t } ) _ { a } ,
$$

where $\mathbf { c ( r o o t ) } _ { a }$ denotes agent $a \mathrm { { : } } \mathrm { { s } }$ secret set after executing c from the initial root state, and whose accessibility relation is exactly the source relation ${ \sim } _ { a } ^ { \mathrm { A W } }$ on call sequences. Equivalently, it is equality of Apt–Wojtczak’s recursively constructed source views, which their equivalence theorem identifies with $\sim _ { a } ^ { \mathrm { A W } } ~ [ 2 ]$ . This specialized instance fixes Definition 3.1 to the source’s secret-only call and view semantics; it does not include phone-number exchange, protocol-restricted history domains, or other EPLA model action types.

Definition 3.5 (Source-compatible Guard fragment and translation). Let the factual source formulas $\theta$ and the source-compatible Guard formulas $\varphi$ be generated by

$$
\begin{array} { r l } & { \theta \mathrel { \mathop : } = \top \mid S _ { a b } \mid \neg \theta \mid ( \theta \land \theta ) , } \\ & { \varphi \mathrel { \mathop : } = \theta \mid \neg \varphi \mid ( \varphi \land \varphi ) \mid \mathcal { K } _ { a } ( \theta ) . } \end{array}
$$

Write $\mathcal { L } _ { K , \mathrm { s r c } }$ for this modal-depth-one subfragment of $\mathcal { L } _ { K }$ . For a fixed $a _ { 0 } \in A$ , define a translation $\tau$ into the Apt–Wojtczak language recursively by

$$
\begin{array} { r l r } & { } & { \tau ( S _ { a b } ) = F _ { a } p _ { b } , \qquad \tau ( \top ) = F _ { a _ { 0 } } p _ { a _ { 0 } } \vee \neg F _ { a _ { 0 } } p _ { a _ { 0 } } , } \\ & { } & { \tau ( \neg \varphi ) = \neg \tau ( \varphi ) , \quad \tau ( \varphi \wedge \psi ) = \tau ( \varphi ) \wedge \tau ( \psi ) , } \\ & { } & { \tau ( \mathcal { K } _ { a } ( \theta ) ) = K _ { a } \tau ( \theta ) . \qquad } \end{array}
$$

Here $F _ { a } p _ { b }$ is the source familiarity atom stating that agent a is familiar with agent $b \mathrm { ^ { \prime } s }$ secret, and $K _ { a }$ is the singleton-agent epistemic modality (equivalently, $C _ { \{ a \} }$ in the source notation). The symbol ∨ is the usual Boolean abbreviation. Factual formulas translate into the source propositional language, and each knowledge formula translates to one unnested individual-knowledge modality over a propositional formula. Hence the image of $\mathcal { L } _ { K , \mathrm { s r c } }$ lies in $\mathcal { L } _ { \mathrm { w n } }$ . The translation introduces neither probability terms nor nonsingleton common-knowledge operators.

## 4 Basic Epistemic Properties

The ELGM adds graded uncertainty to a standard gossip model without changing the underlying view-based accessibility relation. The lottery weights say which histories an agent takes to be more likely among the histories compatible with its view. The knowledge operator remains crisp because knowledge is the probability-one case.

Proposition 4.1 (S5 behavior of knowledge). In every admissible ELGM, the operator $\textstyle { \mathcal { K } } _ { a } ( \cdot )$ satisfies the S5 axioms over the language in Definition 3.3.

Proof sketch. Since $R _ { a }$ is an equivalence relation, every world in $[ \mathbf { c } ] _ { a }$ has the same accessible class. Strictly positive weights make probability one equivalent to truth at every accessible history. The usual S5 argument for equivalence relations therefore applies. □

Example 4.2 (A view distinction). Let the agents be $a , b , c .$ Compare histories ac and bc; ac. The subsequence of calls involving a is ac in both histories. But after ac, agent a learns only what c had at that point. In bc; ac, agent c has first learned from b, so the later call gives a more information. Hence the views of a difer. This is why Definition 3.1 uses views rather than call subsequences.

## 5 Guards and Lottery Transparency

An action guard is an epistemic precondition. In an LLM-agent interpretation, the LLM may propose an action, but the symbolic layer decides whether the action is permitted. The formal core models pairwise gossip calls only. For $\mathbf c \in H$ , let

$$
\begin{array} { r } { \mathcal A _ { H } ( \mathbf c ) : = \{ a b \in C : \mathbf c \cdot a b \in H \} . } \end{array}
$$

Thus a call is structurally executable only when its successor history is in the model domain. Messages, retrieval operations, and other actions in the wider EPLA model architecture require a separate operational semantics.

Definition 5.1 (Symbolic guard). Let each call $\alpha \in C$ have a precondition $\mathsf { p r e } ( \alpha ) \in$ $\mathcal { L } _ { K }$ . The Guard in $\mathcal { M } ^ { G }$ is

$$
\Gamma _ { \mathcal { M } } \boldsymbol { \sigma } ( \alpha , \mathbf { c } ) = \left\{ \begin{array} { l l } { \mathsf { p e r m i t } , } & { \mathrm { i f ~ } \alpha \in \mathcal { A } _ { H } ( \mathbf { c } ) \mathrm { ~ a n d ~ } \mathcal { M } ^ { G } , \mathbf { c } \vdash \mathsf { p r e } ( \alpha ) , } \\ { \mathsf { d e n y } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Guard soundness is immediate from the definition: if $\Gamma _ { \mathcal { M } } ( \alpha , \mathbf { c } ) = \mathsf { p e r m i t }$ , then the stated precondition is true at c and the call has a successor history in H. This is a semantic property of the formal predicate. A concrete Guard must conform to the same state representation, parser, and rule semantics for the property to govern its executions. The nontrivial point is that, for $\mathcal { L } _ { K }$ , the truth value does not depend on the exact positive lottery weights.

Let $M _ { 0 } = ( H , V , \{ R _ { a } \} _ { a \in \mathsf { A g } } )$ be the lottery-free Kripke model underlying $\mathcal { M } ^ { G }$ using the standard Kripke clause for $K _ { a }$

Lemma 5.2 (Lottery transparency). For every admissible ELGM, every $\textbf { c } \in \ H$ ， and every $\varphi \in { \mathcal { L } } _ { K }$ ，

$$
\mathcal { M } ^ { G } , \mathbf { c } \in \varphi \quad i f f \quad M _ { 0 } , \mathbf { c } \in \varphi .
$$

Proof sketch. The proof is by structural induction. Atoms and Boolean cases are immediate because $\mathcal { M } ^ { G }$ and $M _ { 0 }$ share $H , V ,$ and $R _ { a }$ . For $\textstyle { \mathcal { K } } _ { a } ( \psi )$ , admissibility gives strictly positive weights and a finite positive denominator on $[ \mathbf { c } ] _ { a }$ . Therefore $\mathsf { P } _ { a } ( \psi ) ( \mathbf { c } ) = 1$ if no accessible history falsifies $\psi .$ . By the induction hypothesis, this is equivalent to the Kripke truth condition for $K _ { a } \psi$ in $M _ { 0 }$ □

Lemma 5.3 (Source correspondence). For every Apt–Wojtczak source instance, every $\mathbf { c } \in H _ { \mathrm { s r c } }$ , and every $\varphi \in \mathcal { L } _ { K , \mathrm { s r c } }$

$$
\mathcal { M } ^ { G } , \mathbf { c } \in \varphi \quad i f f \quad ( M _ { \mathrm { A W } } , \mathbf { c } ) \lneq \tau ( \varphi ) ,
$$

where $M _ { \mathrm { A W } }$ is the source gossip model on the same agent set and root state.

Proof sketch. Lemma 5.2 first replaces the ELGM by its lottery-free Kripke structure. Definition 3.4 identifies that structure’s valuation and accessibility relations with those of $M _ { \mathrm { A W } }$ . A structural induction on $\varphi$ then gives the result: the atomic case is the defining clause for $V ( S _ { a b } )$ , Boolean cases are immediate, and the knowledge case uses the common relation $\mathrm { \sim } _ { a } ^ { \mathrm { A W } }$ and the translated $K _ { a }$ clause. □

Theorem 5.4 (Guard decidability for the source-compatible fragment). Let $\mathcal { M } ^ { G }$ be an Apt–Wojtczak source instance, let $\mathbf { c } \in H _ { \mathrm { s r c } } ,$ let $\alpha \in C _ { \mathrm { s r c } }$ and let $\mathsf { p r e } ( \alpha ) \in \mathcal { L } _ { K , \mathrm { s r c } } .$ Given finite encodings of α, c, and pre(α), checking $\Gamma _ { \mathcal { M } } ( \alpha , \mathbf { c } )$ is decidable.

Proof sketch. Because $H _ { \mathrm { s r c } }$ contains every finite source call sequence, $\mathbf { c } \cdot \boldsymbol { \alpha } \in H _ { \mathrm { s r c } } ,$ so the structural part of the Guard is decidable. By Lemma 5.3, the remaining precondition test is equivalent to satisfaction of $\tau ( \mathsf { p r e } ( \alpha ) ) \in \mathcal { L } _ { \mathsf { w n } }$ in $M _ { \mathrm { A W } }$ . Apt and Wojtczak prove that this satisfaction problem is decidable for every finite source call sequence and every formula in $\mathcal { L } _ { \mathrm { w n } }$ [2]. No complexity-class bound is claimed here. □

Remark 5.5 (What the theorem does not say). The theorem does not cover probability threshold guards, arbitrary factual atoms outside $\mathcal { L } _ { K , \mathrm { s r c } }$ , nested individual knowledge, common knowledge, arbitrary admissible ELGMs with an inefective or protocol-restricted history domain, phone-number exchange, or arbitrary LLM tool traces. It also does not import complexity results for dynamic EPLS model checking [17]; that is a diferent problem.

## 6 State Updates and Candidate-Action Selection

There are two diferent dynamics in the model. A call changes the factual gossip state and hence the histories and views. A lottery reweighting changes an agent’s graded uncertainty over histories without changing the underlying history domain, view relation, or valuation.

## 6.1 Structural Call Update

For a call ab $\in { \mathcal { A } } _ { H } ( \mathbf { c } )$ permitted by $\Gamma _ { \mathcal { M } ^ { G } }$ , the structural update appends the call to the actual history and updates the factual gossip state by the usual push-pull rule. This paper uses the direct call-history update ${ \mathbf { c } } \mapsto { \mathbf { c } } \cdot a b ,$ which is defined because $a b \in \mathcal { A } _ { H } ( \mathbf { c } )$ . We do not claim that a singleton dynamic epistemic logic (DEL) event model is suficient to encode the information exchanged in a gossip call; a correct action model would need to encode the participants’ observations.

## 6.2 Probabilistic Reweighting

The probabilistic update is a normalized product rule, inspired by dynamic update with probabilities [12]. It is a modeling definition for ELGMs, not a theorem imported wholesale from that literature. The factors below are positive likelihood factors. This notation does not assert that they are normalized probability kernels over a separately defined event or observation space.

Definition 6.1 (Three-source lottery reweighting). Let e be an event and $O _ { a }$ the observation received by agent a. Suppose the prior lottery is admissible and the factors

$$
P _ { \mathrm { o c c } } ( e \mid { \bf c } ) > 0 , \qquad P _ { \mathrm { o b s } } ( o _ { a } \mid e , { \bf c } ) > 0
$$

are bounded and chosen so that the normalizer below is finite and nonzero. Define

$$
L _ { a } ^ { \prime } ( { \bf c } ) = \frac { 1 } { Z _ { a } } L _ { a } ( { \bf c } ) P _ { \mathrm { o c c } } ( e \mid { \bf c } ) P _ { \mathrm { o b s } } ( o _ { a } \mid e , { \bf c } ) ,
$$

where

$$
Z _ { a } = \sum _ { { \bf c } ^ { \prime } \in H } L _ { a } ( { \bf c } ^ { \prime } ) P _ { \mathrm { o c c } } ( e \mid { \bf c } ^ { \prime } ) P _ { \mathrm { o b s } } ( o _ { a } \mid e , { \bf c } ^ { \prime } ) .
$$

The stated finite, nonzero normalizer is an additional assumption: classwise admissibility of the prior alone does not imply it. Under the positivity and boundedness conditions above, the resulting lottery has finite positive class sums and is therefore admissible.

Lemma 6.2 (Positive reweightings preserve guards). Suppose every lottery that is reweighted has the form $L _ { a } ^ { \prime } ( \mathbf { c } ) = L _ { a } ( \mathbf { c } ) k _ { a } ( \mathbf { c } )$ , where $0 < k _ { a } ( \mathbf { c } ) < \infty$ , and every resulting lottery is admissible. If H, V , and all accessibility relations $R _ { a }$ are unchanged, then every $\varphi \in { \mathcal { L } } _ { K }$ has the same truth value before and after the reweighting. In particular, every Guard in Definition 5.1 has the same truth value.

Proof sketch. The update changes numeric weights but preserves positive support and leaves H, V, and all $R _ { a }$ fixed. Lemma 5.2 says $\mathcal { L } _ { K }$ truth depends only on those shared structures under admissibility. Hence Guard truth is invariant. □

## 6.3 LLM Candidate-Action Policies

For the formal core, an LLM policy is represented as a stochastic generator of gossipcall candidates

$$
\pi _ { \mathrm { L L M } } ( \alpha \mid { \bf c } ) , \qquad \alpha \in { \cal C } .
$$

At a decision step, the LLM submits α; the Guard permits or denies it. Permitted calls update the call history. A denied candidate does not become safe merely because the LLM assigned it high probability. This is the intended separation between graded model confidence and symbolic knowledge. Natural-language messages, retrieval calls, and other action types lie outside this formal transition system until their parsers and operational interfaces are specified.

Representation-editing methods such as ARE [20] can change an LLM’s distribution over candidate actions. This paper does not identify ARE with the lottery reweighting in Definition 6.1, and no theorem below depends on ARE. Connecting a neural intervention to the positive-kernel lemma requires an empirical mapping from model behavior to epistemic alternatives, together with a check of the lemma’s assumptions.

## 7 Decidability and Conditional Progress

Guard checking is decidable only in the source-compatible fragment. Termination needs further protocol assumptions. We state a general condition for later transfer instead of applying a gossip bound to all LLM-agent executions.

Theorem 7.1 (Conditional ranking termination). Let $( X _ { t } ) _ { t \geq 0 }$ be a discrete-time execution process on a state space X, adapted to a filtration $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ that records the complete execution prefix through step t, with $X _ { 0 } = x _ { 0 }$ . Suppose there is a ranking function $\rho : X  \{ 0 , 1 , \ldots , B \}$ with goal set $G = \rho ^ { - 1 } ( 0 )$ , and define the hitting time $T : = \operatorname* { i n f } \{ t \geq 0 : X _ { t } \in G \}$ , with $T = \infty$ if this set is empty. Let $\beta : X \backslash G  C$ select a Guard-permitted call at each non-goal state, such that every outcome of executing $\beta ( x )$ reaches x<sup>′</sup> with $\rho ( x ^ { \prime } ) < \rho ( x )$ Conditional on $\mathcal { F } _ { t }$ , the policy must select $\beta ( X _ { t } )$ at the next candidate step with probability at least $\varepsilon > 0$ whenever $X _ { t } \notin G$ . Finally, every other permitted call and every rejected candidate must leave the rank nonincreasing. Then $\mathbb { E } [ T ] \le \rho ( x _ { 0 } ) / \varepsilon \le B / \varepsilon$

Proof sketch. At any non-goal state, the waiting time to the next strict rank decrease is stochastically dominated by a geometric random variable of mean $1 / \varepsilon$ The remaining assumptions prevent a diferent candidate from increasing the rank or undoing progress. At most $\rho ( x _ { 0 } )$ strict rank decreases are needed before rank zero is reached. Linearity of expectation gives the bound. □

Remark 7.2 (Instantiating the rank). Classical gossip results can instantiate the theorem only after the implemented transition system exhibits the required rank, nonincreasing alternative transitions, and lower policy bound. Complete-graph push-pull Learn New Secrets (LNS) protocols have source-specific call bounds [14]; dynamic partial networks have diferent success conditions [15]. This paper therefore does not state a universal closed-form bound for arbitrary guarded LLM-agent executions.

## 8 Related Work and Future Work

Epistemic gossip and DEL. Dynamic epistemic logic provides the standard semantics for information-changing events [13]. Epistemic gossip protocols specialize this tradition to pairwise calls and higher-order knowledge of secrets [4, 7, 14]. EPLA model imports the call-history perspective, but adds lottery weights so that agents can have graded uncertainty over histories.

Epistemic probability and probabilistic update. EPLS motivates the use of lotteries and the identification of knowledge with probability one [17]. Earlier distributed-systems work shows that the choice of agent probability spaces matters in runs-and-systems models [5]; ELGMs instead fix histories as worlds and sourcestyle views as the epistemic relation for a narrow gossip setting. Kooi [8] provides broad probabilistic dynamic epistemic logic background. Van Benthem, Gerbrandy, and Kooi [12] specifically distinguish prior, occurrence, and observation probabilities in dynamic update. EPLA model uses these ideas in a restricted gossip setting and keeps the full-support assumptions explicit.

LLM agents and formal methods. Recent work contrasts formal multi-agent systems (MAS) theory with LLM-agent practice [9]. Zhang et al. [19] provide a formal-methods roadmap for trustworthy AI agents. Yu et al. [18] instead study model checking for multi-agent systems modeled in an epistemic process calculus. Contract-like or neurosymbolic layers for agents make a similar engineering move: a learned model generates candidate behavior, while a symbolic layer constrains execution [10]. EPLA model instantiates this pattern with an epistemic state layer, a guarded action interface, and learning interfaces.

Representation editing. ARE is an empirical adversarial representation engineering method for editing LLM behavior [20]. It uses hidden state representations, a discriminator, and fine-tuning objectives. In EPLA model, it is an architectural component whose efects must be tested empirically after implementation.

Implementation and empirical assessment remain the main line of our future work, which is currently being conducted.

## 9 Conclusion

EPLA model combines a Policy LLM that generates candidate actions, a Symbolic Guard that controls execution, and an epistemic state layer that makes the Guard’s restricted conditions precise. The central technical result is lottery transparency for the knowledge fragment. Under admissibility and full support, positive lottery weights do not afect which knowledge-fragment Guards are true. For the explicitly defined modal-depth-one source fragment, the correspondence with Apt–Wojtczak’s model makes Guard checking decidable. Positive admissible reweighting therefore preserves these Guards, and the conditional ranking result bounds expected progress when its explicit policy and rank assumptions hold. The broader architecture integrates conditional belief, retrieval, diagnostic Guard feedback, representation editing, and LTL-constrained policy learning. To support the efectiveness of our approach, we are currently conducting limited experimental validation.

## References

[1] Krzysztof R. Apt and Dominik Wojtczak. On decidability of a logic of gossips. In Logics in Artificial Intelligence, volume 10021 of Lecture Notes in Computer Science, pages 18–33, 2016. doi: 10.1007/978-3-319-48758-8 2.

[2] Krzysztof R. Apt and Dominik Wojtczak. Common knowledge in a logic of gossips. In Proceedings of the 16th Conference on Theoretical Aspects of Rationality and Knowledge (TARK 2017), volume 251 of Electronic Proceedings in Theoretical Computer Science, pages 10–27, 2017.

[3] Alberto Camacho, Rodrigo Toro Icarte, Toryn Q. Klassen, Richard A. Valenzano, and Sheila A. McIlraith. LTL and beyond: Formal languages for reward function specification in reinforcement learning. In Proceedings ofthe 28th International Joint Conference on Artificial Intelligence (IJCAI), pages 6065–6073, 2019.

[4] Martin C. Cooper, Andreas Herzig, Faustine Mafre, Fr´ed´eric Maris, and Pierre R´egnier. The epistemic gossip problem. Discrete Mathematics, 342(3):654–663, 2019. doi: 10.1016/j.disc.2018.10.041.

[5] Joseph Y. Halpern and Mark R. Tuttle. Knowledge, probability, and adversaries. Journal of the ACM, 40(4):917–962, 1993.

[6] Lewis Hammond, Alessandro Abate, Julian Gutierrez, and Michael Wooldridge. Multi-agent reinforcement learning with temporal logic specifications. In Proceedings of the International Joint Conference on Autonomous Agents and Multiagent Systems, pages 583–592, 2021. doi: 10.65109/qtwt7869.

[7] Andreas Herzig and Faustine Mafre. How to share knowledge by gossiping. AI Communications, 30(1):1–17, 2017.

[8] Barteld Pieter Kooi. Probabilistic dynamic epistemic logic. Journal of Logic, Language and Information, 12(4):381–408, 2003. doi: 10.1023/A: 1025050800836.

[9] Emanuele La Malfa, Gabriele La Malfa, Samuele Marro, Jie M. Zhang, Elizabeth Black, Michael Luck, Philip H. S. Torr, and Michael Wooldridge. Large language models miss the multi-agent mark. In Advances in Neural Information Processing Systems, volume 38, 2025. arXiv:2505.21298.

[10] Claudiu Leoveanu-Condrei. A DbC inspired neurosymbolic layer for trustworthy agent design, 2025. arXiv:2508.03665.

[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.

[12] Johan van Benthem, Jelle Gerbrandy, and Barteld Kooi. Dynamic update with probabilities. Studia Logica, 93(1):67–96, 2009. doi: 10.1007/ s11225-009-9209-y.

[13] Hans van Ditmarsch, Wiebe van der Hoek, and Barteld Kooi. Dynamic Epistemic Logic, volume 337 of Synthese Library. Springer, Dordrecht, 2007.

[14] Hans van Ditmarsch, Jan van Eijck, Pere Pardo, Rahim Ramezanian, and Fran¸cois Schwarzentruber. Epistemic protocols for dynamic gossip. Journal of Applied Logic, 20:1–31, 2017. doi: 10.1016/j.jal.2016.12.001.

[15] Hans van Ditmarsch, Malvin Gattinger, Louwe B. Kuijer, and Pere Pardo. Strengthening gossip protocols using protocol-dependent knowledge, 2019. arXiv:1907.12321.

[16] Jan van Eijck and Kai Li. Conditional belief, knowledge and probability. In Electronic Proceedings in Theoretical Computer Science, volume 251, pages 188– 206, 2017. doi: 10.4204/EPTCS.251.14.

[17] Jan van Eijck and Fran¸cois Schwarzentruber. Epistemic probability logic simplified. Advances in Modal Logic, 10:1–27, 2014.

[18] Qixian Yu, Zining Cao, Zong Hui, and Yuan Zhou. Model checking for multiagent systems modeled by epistemic process calculus, 2025. arXiv:2501.18155.

[19] Yedi Zhang, Yufan Cai, Xinyue Zuo, Xiaokun Luan, Kailong Wang, Zhe Hou, Yifan Zhang, Zhiyuan Wei, Meng Sun, Jun Sun, Jing Sun, and Jin Song Dong. The fusion of large language models and formal methods for trustworthy AI agents: A roadmap, 2024. arXiv:2412.06512.

[20] Yihao Zhang, Zeming Wei, Jun Sun, and Meng Sun. Adversarial representation engineering: A general model editing framework for large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-4010.