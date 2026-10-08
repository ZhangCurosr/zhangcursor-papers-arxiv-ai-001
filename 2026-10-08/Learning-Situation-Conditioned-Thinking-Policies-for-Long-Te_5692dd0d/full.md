# Learning Situation-Conditioned Thinking Policies for Long-Term LLM Agents

Hong Su

Abstract—Long-running autonomous agents must reuse accumulated reasoning experience without allowing explicit historical memory and LLM context to grow indefinitely. However, existing memory mechanisms mainly retrieve, summarize, or compress past content and do not directly learn when particular kinds of thinking should be activated or discover new thinking knowledge from temporally dispersed experiences. This paper proposes a situation-conditioned thinking memory framework that transforms historical reasoning experience into a lightweight policy for predicting what should be thought about in the current situation, while leaving detailed reasoning to a large language model. Situations may represent temporal or spatiotemporal evolution rather than only current states. Temporary experiences are also periodically analyzed across multiple independent episodes to identify repeated long-range regularities, which are consolidated into new thinking knowledge and further internalized by the lightweight policy. Experiments show that the learned policy achieves 1.000 F1 on temporal-rule generalization, improves DeepSeek reasoning F1 from 0.789 to 0.868, reduces online processing time from 0.3636 ms to 0.0382 ms per query at 30,000 historical situations, and reaches 1.000 relation-discovery F1 and future-thinking accuracy after sufficient repeated crossexperience evidence.

Index Terms—Situation-Conditioned Thinking; Long-Term Agent Memory; Cross-Experience Knowledge Discovery; LLM Agents

## I. INTRODUCTION

Long-running large language model (LLM) [1] [2] agents increasingly operate in settings where they must accumulate experience, revisit earlier situations, and reuse previously effective reasoning over extended periods. This requirement is particularly important for autonomous agents and robots, whose decisions may depend not only on the current observation but also on how a situation has evolved over time. As interaction continues, however, the amount of historical observations, reasoning traces, outcomes, and reflections grows continuously. Simply retaining all of this material as explicit memory makes subsequent reasoning increasingly dependent on retrieval, context construction, and repeated LLM processing. A more scalable agent therefore needs not only to remember past content, but also to progressively internalize reusable knowledge about what should be thought about when similar situations reappear.

Existing long-term memory mechanisms mainly address this problem through retrieval, summarization, reflection, or structured memory organization [3] [4]. These approaches preserve valuable historical evidence and can improve subsequent reasoning, but they usually continue to depend on an external memory whose size grows with experience. Summary-based compression can reduce this burden, but may discard temporal ordering and subtle transitions that determine which kind of reasoning is required. Reflection methods can produce higherlevel textual knowledge, yet the resulting reflections are still typically stored and later retrieved rather than learned as an automatic situation-to-thinking policy. Reasoning distillation provides another alternative, but usually trains a smaller model to reproduce answers or reasoning traces, whereas many longterm agents still benefit from retaining a powerful LLM for detailed reasoning. In addition, some useful knowledge is not explicitly present in any single historical record: it may only become apparent after multiple temporally separated experiences exhibit a repeated pattern, such as an event � followed much later by � and eventually by �. Therefore, a long-term agent requires a mechanism that can both internalize recurring thinking requirements and discover new reusable thinking knowledge across accumulated experiences.

In this paper, we propose a situation-conditioned thinking policy for long-term LLM agents. Instead of repeatedly retrieving complete historical reasoning traces, the framework learns a lightweight mapping from a situation to the thinking requirement that should be activated under that situation. A situation may describe a static state, a temporal transition, or a longer spatiotemporal evolution, allowing the same current state to trigger different thinking when it is reached through different histories. The lightweight model predicts what or when to think, while the LLM remains responsible for how to reason over the current evidence. In this way, historical reasoning is used as training material rather than being treated only as permanent online context.

The framework further introduces periodic cross-experience consolidation for knowledge that cannot be extracted reliably from a single record. Temporally dispersed experiences are first organized into candidate long-range relations, after which repeated support, counterexamples, reverse-order evidence, and cross-period stability are considered during consolidation. Stable relations are formulated as new thinking knowledge and transformed into additional training material for the lightweight policy. Consequently, a repeatedly observed pattern such as $A \ldots B \ldots C$ can eventually be internalized as a future thinking rule $A \dots B \to ^ { \cdot }$ “consider possible $C ^ { \mathfrak { s } }$ before � is observed. This mechanism allows the agent to move progressively from explicit historical experience to reusable thinking behavior, while still preserving temporary or episodic memory for unresolved and newly emerging situations.

The main contributions of this work are summarized as follows:

• We introduce a situation-conditioned thinking memory mechanism that learns a lightweight policy for predicting what kind of thinking should be activated, while leaving detailed reasoning to the LLM.

• We represent situations using temporal or spatiotemporal evolution rather than only the current state, enabling the learned policy to distinguish histories that end in the same observation but require different reasoning.

• We develop a periodic cross-experience consolidation mechanism that identifies stable long-range regularities across repeated experiences, formulates them as new thinking knowledge, and converts them into training material for future automatic thinking activation.

The remainder of this paper is organized as follows. Section II reviews long-term LLM memory, reflection and experience reuse, and temporal cross-experience knowledge discovery. Section III presents the proposed situation-conditioned thinking framework and its periodic consolidation mechanism. Section IV describes the experimental setup and evaluates the proposed method across six complementary experiments. Finally, Section V concludes the paper and discusses the implications of learning reusable thinking behavior from longterm experience.

## II. RELATED WORK

## A. Long-Term Memory and Retrieval for LLM Agents

Long-term memory has become a central component of LLM-based agents because interaction histories can quickly exceed the context window and because useful past experience may need to be reused after long delays. Generative Agents stores natural-language observations, retrieves memories according to relevance, recency, and importance, and periodically synthesizes higher-level reflections to support subsequent planning and behavior [5]. MemGPT approaches the same longcontext problem from a systems perspective by introducing hierarchical memory tiers and virtual-context management, allowing an agent to move information between limited incontext memory and larger external storage [6]. More recent work has further explored structured memory organization. For example, THEANINE links memories according to temporal and cause–effect relations and retrieves memory timelines rather than isolated entries [7], while MAGMA represents memories through separate semantic, temporal, causal, and entity graphs and performs policy-guided traversal over these relational views [8]. These methods substantially improve an agent’s ability to preserve and retrieve information over long interaction histories.

The main advantage of retrieval-based memory is that original evidence remains explicitly accessible. Complete histories can therefore preserve detailed information that may be difficult to compress safely, and structured retrieval can expose relations that similarity-only retrieval might overlook. This is especially useful when a past event itself is required as evidence for the current decision. However, explicit memory also creates a scaling dependency: as interaction history grows, the system must continue storing, indexing, ranking, and often inserting historical information into the LLM context. Even when the retrieved subset is small, the online reasoning process remains dependent on an external history and on the quality of the retrieval mechanism. Summary-based compression reduces this burden but can remove temporal ordering or subtle transition information that later becomes important.

Our work addresses a different question from conventional memory retrieval. Instead of asking which historical record should be retrieved, we ask whether repeated historical reasoning can be transformed into a lightweight policy that predicts what kind of thinking should be activated in the present situation. Historical records are therefore training material rather than necessarily permanent online context. This distinction also separates our method from structured retrieval approaches such as THEANINE and MAGMA: they organize temporal or causal memories so that an LLM can retrieve them more effectively, whereas our method learns a situationto-thinking mapping and can consequently operate without an ever-growing online history after consolidation. Explicit memory is still useful for unresolved or newly emerging patterns, but already learned thinking requirements can be internalized.

B. Reflection, Lifelong Experience Reuse, and Reasoning Internalization

A second line of work improves agents by processing previous interaction outcomes rather than merely retrieving raw observations. Reflexion converts feedback from previous trials into verbal reflections and stores them in episodic memory so that later attempts can avoid earlier mistakes [9]. Generative Agents similarly introduces a reflection process that synthesizes lower-level observations into higher-level statements [5]. In embodied lifelong learning, Voyager continually explores Minecraft, refines executable skills using environmental feedback, and stores successful programs in an expanding skill library for later reuse [10]. These approaches demonstrate that experience becomes more useful when it is transformed into reusable higher-level representations instead of being preserved only as raw trajectories.

Reflection and skill libraries nevertheless retain the learned result mainly as explicit textual memories, plans, or executable behaviors. Their strength is interpretability and flexibility: the LLM can inspect a reflection or retrieve a learned skill and directly reuse or adapt it. Their limitation, relative to the present problem, is that the agent still needs to decide which reflection or skill to retrieve and often invokes the LLM to interpret the retrieved content. Furthermore, reflection commonly improves what the agent should conclude or do after a previous trial, whereas our target is narrower and complementary: learning when a particular type of thinking should be invoked before detailed reasoning is carried out.

Our approach is also different from reasoning distillation. Distilling Step-by-Step, for example, uses LLM-generated rationales as additional supervision for training a smaller task model, allowing a compact model to reproduce more of the teacher’s task-solving capability [11]. Such distillation is valuable when the desired outcome is to replace or approximate the large model. In our framework, the lightweight model is not trained to reproduce the entire rationale or final answer. It learns the mapping from a situation to a compact thinking requirement, while the LLM remains responsible for detailed reasoning over the current evidence. Thus, the division of labor is small model: what/when to think; large model: how to reason now. This preserves the flexibility of a general LLM while transferring frequently reusable thinking selection into a cheaper learned policy.

## C. Temporal Reasoning and Cross-Experience Knowledge Discovery

Long-term agent memory increasingly incorporates temporal structure because isolated memory items are insufficient when current meaning depends on how events evolved. THEA-NINE explicitly links memories through temporal and cause– effect relations and constructs timelines that expose changes across long conversations [7]. TReMu combines memory with neuro-symbolic temporal reasoning for multi-session dialogue, targeting temporal relations that are difficult to answer from isolated utterances [12]. Temporal Semantic Memory further distinguishes actual event time from dialogue time and consolidates semantically related information into durative memories that represent persistent or evolving states [13]. These approaches show the benefit of preserving temporal organization rather than treating memory as an unordered collection of semantically similar items.

Recent memory research is also moving beyond storage toward abstraction across trajectories. A recent survey characterizes this evolution as progressing from Storage, through Reflection, to Experience, where cross-trajectory abstraction can produce more general knowledge from multiple interactions [14]. This perspective is closely related to our periodic consolidation mechanism. Nevertheless, detecting a recurring temporal relation is not equivalent to summarizing several records. A relation such as $A \dots B  C$ may be absent from every individual record as an explicit statement and become apparent only after multiple temporally separated episodes are compared. Moreover, apparent relations may arise by chance; therefore repeated independent evidence, counterexamples, reverse-order evidence, and temporal stability are important before a candidate is promoted to reusable knowledge.

Our framework combines low-level temporal candidate discovery with LLM-based knowledge consolidation. The lowlevel stage identifies statistically supported long-range candidates and preserves evidence such as recurrence across independent episodes and enrichment over order-shuffled histories. The LLM then organizes these candidates, compares supporting and contradictory evidence, and formulates stable relations as reusable thinking knowledge. Crucially, the discovered relation is not retained only as another memory item: it is converted into training material for the situation-conditioned thinking policy. Consequently, a repeatedly observed pattern $A \ldots B \ldots C$ can eventually produce a future policy $A \dots B $ “consider possible $C ^ { \mathfrak { s } }$ before � is observed. This closes the gap between temporal memory organization and learned thinking activation: cross-experience regularities are not only retrieved or described, but progressively internalized into when the agent should think about a particular issue.

## III. SITUATION-CONDITIONED THINKING MEMORY WITH CROSS-EXPERIENCE KNOWLEDGE DISCOVERY

## A. Problem Setting and Overall Principle

A long-running autonomous agent repeatedly encounters situations, invokes a large language model (LLM), receives environmental feedback, and accumulates experience. Directly retaining all previous reasoning traces and repeatedly inserting them into the LLM context is not scalable because explicit history grows with operation time. The objective of our model is therefore not to memorize every previous reasoning result, but to learn when a reusable type of thinking should be activated. A lightweight policy determines what should be thought about, while the LLM remains responsible for detailed reasoning about the current situation.

At interaction step �, an experience is represented as

$$
e _ { t } = ( s _ { t } , a _ { t } , h _ { t } , y _ { t } , f _ { t } ) ,\tag{1}
$$

where $s _ { t }$ is the observed situation, $a _ { t }$ is the performed action when one exists, $h _ { t }$ is the LLM reasoning or other processing trace, $y _ { t }$ is the observed outcome, and $f _ { t }$ denotes later feedback. A situation is not restricted to a single instantaneous state. It may contain a sequence

$$
s _ { t } = S ( x _ { t - K + 1 } , . ~ . ~ . , x _ { t } ) ,\tag{2}
$$

where $K = 1$ represents a static situation and $K > 1$ allows temporal, spatial, or spatiotemporal evolution to determine the required thinking. This distinction is important because two situations can end in the same current state while requiring different reasoning due to their different histories.

The model contains two complementary routes for producing reusable thinking knowledge. First, a single sufficiently informative experience may directly yield a thinking requirement. Second, some regularities are invisible in any individual record and emerge only after multiple temporally dispersed experiences are jointly examined. The second route is essential for long-range patterns such as

$$
A \stackrel { \Delta _ { 1 } } { \longrightarrow } B \stackrel { \Delta _ { 2 } } { \longrightarrow } C ,\tag{3}
$$

where $A , B ,$ and � may be separated by long and variable intervals. The model therefore maintains temporary experience records and periodically asks the LLM to discover repeated cross-experience regularities rather than merely summarize them.

## B. Direct Thinking Abstraction from Individual Experience

For an experience that already contains sufficient evidence, an abstraction operator extracts a reusable thinking requirement

$$
g _ { i } = \mathcal { A } _ { \mathrm { L L M } } ( e _ { i } ) ,\tag{4}
$$

where $g _ { i }$ describes a reasoning direction rather than a final answer. Examples include checking a cross-signal inconsistency,

analyzing a possible hidden dependency, examining future impact, or seeking additional information. The corresponding direct training material is

$$
d _ { i } ^ { \mathrm { d i r } } = ( s _ { i } , g _ { i } ) .\tag{5}
$$

This material allows the lightweight model to learn that situations with similar structure should activate similar thinking, even when their surface objects are different.

However, Eq. (4) is insufficient when the useful regularity spans multiple experiences. For example, a high-load event may occur, a weak vibration may appear much later, and a positioning error may emerge after an additional delay. None of the three records alone reveals the relation. Treating the records independently would therefore fail to produce the thinking rule that the first two events should trigger consideration of the later failure.

## C. Periodic Cross-Experience Knowledge Discovery

The agent keeps a bounded temporary experience buffer

$$
\mathcal { B } _ { t } = \{ e _ { t - L + 1 } , \ldots , e _ { t } \} ,\tag{6}
$$

where � denotes the currently retained horizon. Temporary memory is not intended to become a permanently growing prompt. Instead, it provides a working set from which previously unknown regularities can be discovered.

A consolidation operation is triggered periodically or when sufficient new evidence has accumulated. Additional triggers may include repeated unexplained outcomes, persistent prediction errors, or novelty. We denote the trigger by

$$
\chi _ { t } = \mathbb { I } \big [ \mathrm { p e r i o d } ( t ) \vee | \mathcal { B } _ { t } | \ge B \vee \mathrm { n o v e l t y } _ { t } \vee \mathrm { r e p e a t e d F a i l u r e } _ { t } \big ]\tag{7}
$$

When $\chi _ { t } = 1$ , the LLM performs cross-experience knowledge discovery

$$
\mathcal { K } _ { t } = \mathcal { D } _ { \mathrm { L L M } } ( \mathcal { B } _ { t } ) ,\tag{8}
$$

where $\mathcal { D } _ { \mathrm { L L M } }$ is explicitly instructed to search for repeated temporal, predictive, relational, or candidate-causal regularities. Its objective is not to shorten the buffer into a narrative summary. Instead, it seeks new relations that are not explicitly stated in any individual record.

A discovered knowledge unit is represented as

$$
k _ { j } = ( P _ { j } , R _ { j } , \Delta _ { j } , g _ { j } , c _ { j } , E _ { j } ) ,\tag{9}
$$

where $P _ { j }$ is the antecedent event/state pattern, $R _ { j }$ is the discovered relation type, $\Delta _ { j }$ records temporal constraints, $g _ { j }$ is the thinking requirement induced by the relation, $c _ { j }$ is its confidence, and $E _ { j }$ is the set of supporting experience identifiers. For example,

$$
P _ { j } : A \xrightarrow { \Delta _ { 1 } } B \xrightarrow { \Delta _ { 2 } } C \quad \implies \quad g _ { j } = \ " { \mathrm { c o n s i d e r ~ d e l a y e d ~ } } C \mathrm { - r e l a t e d }\tag{10}
$$

The event symbols may have no semantic implication by themselves; evidence for the rule is provided by its repeated occurrence across the history.

The model distinguishes a discovered predictive regularity from a proven causal statement. A newly induced relation is initially treated as a candidate. Its confidence can be updated

as additional supporting or contradicting experience becomes available,

$$
c _ { j } ^ { ( t + 1 ) } = \rho c _ { j } ^ { ( t ) } + ( 1 - \rho ) \operatorname { S u p p o r t } ( k _ { j } ; \mathcal { B } _ { t + 1 } ) ,\tag{11}
$$

where $0 \leq \rho < 1$ controls persistence. Relations that repeatedly predict future observations or survive active validation may be promoted, whereas unsupported relations can be weakened or removed. This design prevents an LLM-generated hypothesis from being treated automatically as confirmed causality.

D. Conversion of Discovered Knowledge into a Lightweight Thinking Policy

Cross-experience discovery becomes useful for future autonomy only when the discovered knowledge changes subsequent processing. We therefore convert both direct abstractions and newly discovered regularities into training materials. Let

$$
\mathcal { D } _ { t } ^ { \mathrm { d i r } } = \{ ( s _ { i } , g _ { i } ) \} \quad \mathrm { a n d } \quad \mathcal { D } _ { t } ^ { \mathrm { d i s c } } = \{ ( P _ { j } , g _ { j } ) : k _ { j } \in \mathcal { K } _ { t } \} .\tag{12}
$$

The cumulative learning set is updated by

$$
\mathcal { D } _ { t + 1 } = \mathcal { D } _ { t } \cup \mathcal { D } _ { t } ^ { \mathrm { d i r } } \cup \mathcal { D } _ { t } ^ { \mathrm { d i s c } } .\tag{13}
$$

Thus, LLM consolidation is not the final memory representation. It produces new learning material that can subsequently be internalized.

A lightweight model $M _ { \theta }$ is trained to estimate

$$
M _ { \theta } ( s ) = P _ { \theta } ( g \mid s ) ,\tag{14}
$$

where $g$ may be a single thinking activity or a multi-label set. Sequence models such as Transformers or LSTMs are suitable when order and long-range evolution matter, while simpler machine-learning models can be used when the situation representation is static. The framework therefore does not require one specific small-model architecture; the key requirement is that the learned carrier preserves the situation-to-thinking relation.

For a newly discovered long-range rule, the training situation need not include the eventual outcome �. Instead, a prefix containing the relevant antecedent pattern, such as $A \ldots B ,$ can be paired with a future-oriented thinking requirement derived from $k _ { j }$ :

$$
( A \ldots R , \ g _ { j } = \mathrm { { ^ { * } c o n s i d e r / p r e d i c t } } \ C \mathrm { - r e l a t e d \ c o n s e q u e n c e ^ { * } } ) .\tag{15}
$$

Consequently, after learning, the agent can activate the appropriate thinking before � actually occurs.

## E. Runtime Thinking Activation and Continual Development

At runtime, the current situation is first processed by the lightweight policy. The activated thinking set is

$$
{ \widehat { \mathcal { G } } } _ { t } = \{ g : P _ { \theta } ( g \mid s _ { t } ) \geq \tau \} ,\tag{16}
$$

where � is an activation threshold. The LLM then performs current reasoning under these thinking directions,

$$
z _ { t } = \mathrm { L L M } ( s _ { t } , \widehat { G } _ { t } ) ,\tag{17}
$$

and the resulting action and later outcome create a new experience $e _ { t + 1 }$ . The lightweight model therefore answers

what should be considered now, whereas the LLM answers how the current problem should be reasoned through.

The complete continual-development loop is

$$
\mathrm { i n t e r a c t i o n \to t e m p o r a r y \ e x p e r i e n c e s }
$$

$$
 \mathrm { c r o s s - e x p e r i e n c e ~ d i s c o v e r y }
$$

→ new thinking knowledge

→ policy learning

(18)

→ future thinking activation

→ new interaction and validation.

This loop allows the system to acquire knowledge that was not predefined and was not evident in a single observation. Repeated experiences can reveal a previously unknown longrange relation; the relation can then be converted into a thinking requirement; and the requirement can finally become an automatically activated lightweight policy. Exact episodic records may still be retained when needed for auditing or factual recall, but reusable thinking knowledge no longer requires replaying the entire growing history to the LLM.

## IV. VERIFICATION

## A. Experimental Setting, Scenarios, and Metrics

We evaluate whether long-running experience can be transformed into reusable thinking knowledge rather than remaining as an ever-growing collection of explicit historical records. The controlled setting emulates an autonomous robot that continuously observes several heterogeneous signals, including temperature, current, vibration, object distance, visibility, motion state, and load. A situation is represented by either a single observation or a sequence of observations, so that the experiments can distinguish a current state from the temporal or spatial evolution that produced it. The experiments are intentionally synthetic and controlled: the objective is not to reproduce a particular robot platform, but to isolate the memory and thinking mechanisms while keeping the correct thinking requirements unambiguous.

The lightweight situation-conditioned thinking policy is implemented by a Transformer encoder with four encoder layers, model dimension 192, six attention heads, a feedforward dimension of 768, and dropout of 0.1. It is trained as a multi-label predictor using binary cross-entropy and AdamW. Five independent random seeds are used for the learning experiments. The downstream large language model (LLM) is the online deepseek-v4-flash model. The lightweight model does not generate the final natural-language reasoning result. Instead, it predicts which relation or issue should be considered, and DeepSeek performs the detailed reasoning over the current situation.

Six experiments test complementary mechanisms. Experiment 1 tests whether a learned policy avoids the storage and online-retrieval growth of explicit long-term history. Experiment 2 tests transfer to unknown objects when the required thinking depends on a hidden temporal-order rule. Experiment 3 isolates the need to represent situation evolution by giving different histories with the same final state. Experiment 4 tests whether learned thinking guidance improves actual DeepSeek reasoning. Experiment 5 measures how a rare thinking pattern is acquired as historical experience accumulates. Experiment 6 tests the newly introduced periodic crossexperience knowledge-discovery mechanism: temporally separated events must recur across multiple independent episodes before they can be consolidated into new knowledge and then internalized as a future thinking policy.

The compared methods are selected according to the mechanism tested. Summary-RAG retrieves historical situations represented by aggregate statistics and is therefore compact but insensitive to temporal order. Sequence-RAG stores the complete temporal sequence and provides a stronger retrieval reference. RF-Summary is a random-forest multi-label classifier trained on the same aggregate features. Current-State-Only uses only the last observation rather than the complete situation evolution. For the downstream LLM test, DeepSeek-Only receives the current observations without learned guidance, RAG+DeepSeek receives thinking guidance from retrieved histories, Proposed+DeepSeek receives guidance predicted by the learned policy, and Oracle+DeepSeek receives the groundtruth thinking requirement. In Experiment 6, No Consolidation does not form cross-experience knowledge, Unordered destroys the long-range order, Local preserves only short-range order, Statistics-Only ranks long-range candidates using their accumulated temporal evidence without LLM consolidation, and Oracle directly uses the hidden regularities.

Thinking prediction is evaluated with micro-precision, micro-recall, and micro-F1. End-to-end LLM reasoning is measured using factor precision, factor recall, factor F1, and final decision accuracy. Scalability is measured by stored bytes, online historical records, estimated history-context tokens, and measured retrieval or policy-inference time. Experiment 6 additionally reports candidate recall, relation-discovery F1, and future-thinking activation accuracy. Unless otherwise stated, values are means and standard deviations over five seeds.

## B. Experiment 1: Long-History Consolidation and Scalability

The first experiment asks whether reusable thinking knowledge can be consolidated into a fixed policy instead of repeatedly searching a growing explicit history. The history is increased from 500 to 30,000 situations. Summary-RAG and Sequence-RAG retrieve the top four records for every new query. Summary-RAG stores order-insensitive aggregate features, whereas Sequence-RAG stores the complete temporal sequence. The proposed method learns from accumulated histories and subsequently uses no historical record in its online context.

Table I shows that Sequence-RAG and the proposed method both maintain a micro-F1 of 1.000 at all four history sizes. Hence, the scalability result is not obtained by deliberately weakening retrieval accuracy. Sequence-RAG storage, however, grows from 0.156 MB at 500 histories to 9.360 MB at 30,000 histories, whereas the proposed model remains approximately 7.358 MB. From the measured per-record index size, the Sequence-RAG index reaches the learned-model size at approximately $2 . 3 6 \times 1 0 ^ { 4 }$ historical situations. Thus, explicit retrieval is initially more compact for a small history, but its memory grows approximately linearly while the learned policy has a fixed parameter size after consolidation.

![](images/94073b7a917f6f226e9d086aa52cdc8e02f9b7a6c00d9756105fbcf9432922b1.jpg)  
Fig. 1. Storage as explicit historical experience grows. Sequence-RAG grows with the number of stored histories, while the consolidated thinking policy remains fixed.

![](images/5bbf4b38826fd4811ce906f2be5b62cdf1ec43c70c4b6af4b0925eb7a0a58ed0.jpg)  
Fig. 2. Online cost under growing history size. Sequence-RAG must search a larger explicit memory, whereas the learned policy performs fixed-size inference.

The online-time trend is similar. Sequence-RAG retrieval rises from 0.0186 ms/query at 500 histories to 0.3636 ms/query at 30,000 histories, whereas policy inference remains approximately 0.0382 ms/query. At 30,000 histories, the measured Sequence-RAG retrieval time is therefore about 9.53× the policy-inference time. Summary-RAG uses a smaller representation but attains only $0 . 4 9 5 \pm 0 . 0 7 1$ F1 at 30,000 histories because the aggregate representation discards temporal order. Figures 1 and 2 visualize these two scalability effects. The experiment therefore demonstrates the intended trade-off: complete retrieval can preserve accuracy but incurs a historydependent storage and online-search cost; simple history compression is cheaper but can lose temporally structured thinking information; learned consolidation retains the reusable thinking behavior with zero online historical records.

## C. Experiment 2: Generalization of Non-Obvious Temporal Thinking Rules

Experiment 2 tests whether the learned memory captures a reusable thinking rule rather than recognizing an object name or an obvious physical relation. Training situations contain one set of robot-component identities, while testing situations contain unseen identities. More importantly, different classes are constructed to have the same start value, final value, set of intermediate values, mean, standard deviation, and global change. The classes differ only in the ordering of their intermediate observations, and the mapping from each ordering to its required thinking activity is an environmentspecific rule rather than common world knowledge.

Table II shows that the proposed sequence Transformer obtains $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ micro-F1. Summary-RAG reaches $0 . 5 6 6 \pm$ 0.045, while RF-Summary reaches $0 . 3 8 3 \pm 0 . 0 3 3$ . Because aggregate sequence statistics were deliberately matched, the approximately 0.434 F1 gap between Proposed and Summary-RAG cannot be attributed to a different final state or a different mean value; it arises from preserving and learning the temporal ordering itself. The result therefore supports the claim that the learned thinking memory can transfer an environmentspecific situation-to-thinking relation to unknown objects when the same latent evolution reappears.

## D. Experiment 3: Same Final State, Different Situation Evolution

The third experiment isolates the importance of situation evolution even further. Each pair ends at exactly the same moderate temperature and current values. In one history, however, the signals rise toward the endpoint and require trend, cause, and hazard analysis; in the other, they fall toward the same endpoint and indicate recovery. A current-state-only model therefore sees identical final evidence for two cases that should activate different thinking.

The proposed sequence-aware policy achieves 1.000±0.000 micro-F1, compared with $0 . 7 1 1 \pm 0 . 0 4 4$ for Current-State-Only (Table II). This is an absolute difference of approximately 0.289. The result demonstrates that “what should be thought about” may depend on how the present state was reached, not merely on the current observation itself. Together, Experiments 2 and 3 justify representing a situation as a temporal or spatiotemporal sequence when the relevant thinking requirement is transition-dependent.

## E. Experiment 4: End-to-End Thinking Guidance with DeepSeek

Experiment 4 tests whether the learned thinking policy improves downstream LLM reasoning rather than only reproducing thinking labels. Every test situation contains several simultaneous numerical signal sequences. The lightweight policy outputs a reasoning direction, such as comparing temperature with current, examining visibility jointly with distance, comparing vibration with load, or inspecting a late current change. These directions are not the final answers. DeepSeek must still inspect the current numerical observations and determine which canonical factor is actually present. Test objects use unseen identities so that the result cannot be obtained from a memorized component name.

TABLE I  
LONG-HISTORY SCALABILITY WITH TOP-4 RETRIEVAL. STORAGE USES DECIMAL MB (10<sup>6</sup> BYTES).
<table><tr><td>History N</td><td>Summary-RAG F1</td><td>Sequence-RAG F1</td><td>Seq.-RAG MB</td><td>Seq.-RAG ms/query</td><td>Proposed F1</td><td>Proposed MB</td><td>Proposed ms/query</td></tr><tr><td>500</td><td> $\overline { { 0 . 5 2 8 \pm 0 . 0 6 8 } }$ </td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td>0.156</td><td>0.0186</td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td>7.358</td><td>0.0382</td></tr><tr><td>2,000</td><td> $0 . 5 1 8 \pm 0 . 0 4 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>0.624</td><td>0.0309</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>7.358</td><td>0.0382</td></tr><tr><td>10,000</td><td> $0 . 4 9 9 \pm 0 . 0 7 3$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>3.120</td><td>0.1533</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>7.358</td><td>0.0382</td></tr><tr><td>30,000</td><td> $0 . 4 9 5 \pm 0 . 0 7 1$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>9.360</td><td>0.3636</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>7.358</td><td>0.0382</td></tr></table>

TABLE II  
MECHANISM VERIFICATION FOR TEMPORAL-ORDER GENERALIZATION AND SITUATION EVOLUTION.
<table><tr><td>Experiment</td><td>Method</td><td>Micro-F1</td></tr><tr><td rowspan="4">Exp. 2</td><td>Proposed Transformer</td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td></tr><tr><td>Summary-RAG</td><td> $0 . 5 6 6 \pm 0 . 0 4 5$ </td></tr><tr><td>RF-Summary</td><td> $0 . 3 8 3 \pm 0 . 0 3 3$ </td></tr><tr><td>Proposed Sequence</td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td></tr><tr><td rowspan="2">Exp. 3</td><td>Current-State-Only</td><td> $0 . 7 1 1 \pm 0 . 0 4 4$ </td></tr><tr><td></td><td></td></tr></table>

As shown in Table III, DeepSeek-Only obtains a factor F1 of $0 . 7 8 9 \pm 0 . 0 3 8$ . Proposed+DeepSeek increases this value to $0 . 8 6 8 { \pm } 0 . 0 3 5$ , an absolute increase of 0.079 and approximately a 10.0% relative improvement. Recall is already 1.000 for all four methods, whereas factor precision increases from $0 . 7 0 4 \pm$ 0.055 to $0 . 8 1 5 \pm 0 . 0 4 8$ . The improvement therefore mainly reflects fewer irrelevant or incorrect factors, i.e., the learned policy helps the LLM focus on the relevant relation rather than simply encouraging it to produce more candidate factors. Decision accuracy also increases from $0 . 4 8 4 { \pm } 0 . 1 1 5$ to 0.575± 0.140.

RAG+DeepSeek and Oracle+DeepSeek both obtain 0.877± 0.029 factor F1, only 0.009 above Proposed+DeepSeek. A paired analysis over the 320 saved situation-level comparisons gives a Proposed–DeepSeek factor-F1 difference of +0.0786 with 95% bootstrap interval [0.0609, 0.0974] and paired randomization $\begin{array} { r l r } { p } & { { } < } & { 0 . 0 0 1 } \end{array}$ . Decision correctness improves by +0.0906, with interval [0.0594, 0.1250] and $p ~ < ~ 0 . 0 0 1$ . In contrast, the Proposed–RAG factor-F1 difference is −0.0089 with interval $[ - 0 . 0 3 0 2 , 0 . 0 1 3 5 ]$ and $p = 0 . 4 7 2$ , so there is no reliable accuracy difference between these two methods in this controlled task. This result is consistent with the intended role of consolidation: Proposed approaches the quality of explicit retrieval while Experiment 1 shows that it does not require a growing online history.

## F. Experiment 5: Formation ofThinking Knowledgefrom Rare Experience

Experiment 5 tests whether a thinking policy can actually emerge from accumulated experience. A delayed-onset event is made rare in the historical stream: signals remain apparently normal for a long prefix and then exhibit a late abnormal acceleration. Correct handling requires trend, hazard, cause, future-impact, and information-seeking thinking. The test set contains this rare pattern, while the number of available historical situations is increased from 100 to 8,000.

![](images/c836a707d3fb20f5786c987c6d95a5a7a2cc9cca37d6d8d5caf6e3a06d863fbb.jpg)  
Fig. 3. Factor F1 in the end-to-end DeepSeek experiment. Learned thinking guidance improves over unguided DeepSeek and approaches the explicitretrieval and oracle references.

![](images/d2c8682552e4aa4c815fa242c8e1cc18472669c1651818c15ec69b52d7b606a0.jpg)  
Fig. 4. Learning of a rare thinking pattern as the number of historical situations increases. Error bars show one standard deviation over five seeds.

Figure 4 shows a clear acquisition curve. With 100 histories, rare-pattern micro-F1 is $0 . 7 7 7 \pm 0 . 0 7 7 .$ . It increases to $0 . 9 1 2 \pm 0 . 0 5 1$ at 250 histories and $0 . 9 9 6 \pm 0 . 0 0 5$ at 500. Performance then approaches saturation, reaching $0 . 9 9 8 \pm 0 . 0 0 3$ at 2,000 histories and $0 . 9 9 9 \pm \ : 0 . 0 0 1$ at 8,000. Thus, the thinking policy is not assumed to exist from the beginning: it becomes reliable after enough relevant historical evidence has accumulated, with diminishing returns once the rare pattern has been internalized.

## G. Experiment 6: Periodic Cross-Experience Discovery of New Thinking Knowledge

The final experiment evaluates the new periodic crossexperience consolidation mechanism. Its purpose is different from Experiments 2–5: the target thinking rule is not explicitly available in any individual record. Each seed uses randomly generated event codes with no semantic meaning, such as $E _ { 0 1 6 }$ $E _ { 1 7 6 }$ , and $E _ { 0 3 0 }$ . A hidden regularity has the form

TABLE III  
END-TO-END REASONING WITH DEEPSEEK-V4-FLASH. VALUES ARE MEAN±STANDARD DEVIATION OVER FIVE SEEDS.
<table><tr><td>Method</td><td>Factor Precision</td><td>Factor Recall</td><td>Factor F1</td><td>Decision Accuracy</td><td>Avg. Tokens</td></tr><tr><td>DeepSeek-Only</td><td> $\overline { { 0 . 7 0 4 \pm 0 . 0 5 5 } }$ </td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td> $\overline { { 0 . 7 8 9 \pm 0 . 0 3 8 } }$ </td><td> $0 . 4 8 4 \pm 0 . 1 1 5$ </td><td> $\overline { { 7 1 9 . 6 1 \pm 0 . 9 8 } }$ </td></tr><tr><td>RAG+DeepSeek</td><td> $0 . 8 3 5 \pm 0 . 0 3 9$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 8 7 7 \pm 0 . 0 2 9$ </td><td> $0 . 5 5 9 \pm 0 . 1 0 6$ </td><td> $7 2 7 . 9 4 \pm 0 . 8 2$ </td></tr><tr><td>Proposed+DeepSeek</td><td> $0 . 8 1 5 \pm 0 . 0 4 8$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 8 6 8 \pm 0 . 0 3 5$ </td><td> $0 . 5 7 5 \pm 0 . 1 4 0$ </td><td> $7 2 8 . 1 3 \pm 1 . 1 8$ </td></tr><tr><td>Oracle+DeepSeek</td><td> $0 . 8 3 5 \pm 0 . 0 3 9$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 8 7 7 \pm 0 . 0 2 9$ </td><td> $0 . 5 5 9 \pm 0 . 1 0 6$ </td><td> $7 2 7 . 9 4 \pm 0 . 8 2$ </td></tr></table>

TABLE IV

PAIRED PER-SITUATION ANALYSIS IN EXPERIMENT 4 $( n = 3 2 0 )$
<table><tr><td>Comparison / Metric</td><td>Mean Diff.</td><td>95% CI</td><td>p</td></tr><tr><td>Proposed – DeepSeek / F1</td><td>+0.0786</td><td>[0.0609,0.0974]</td><td> $\overline { { < 0 . 0 0 1 } }$ </td></tr><tr><td>Proposed – DeepSeek / Decision</td><td>+0.0906</td><td> $[ 0 . 0 5 9 4 , 0 . 1 2 5 0 ]$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Proposed – RAG / F1</td><td>-0.0089</td><td> $[ - 0 . 0 3 0 2 , 0 . 0 1 3 5 ]$ </td><td> $_ { 0 . 4 7 2 }$ </td></tr><tr><td>Proposed – RAG / Decision</td><td>+0.0156</td><td> $\left[ - 0 . 0 2 8 1 , 0 . 0 5 9 4 \right]$ </td><td>0.587</td></tr></table>

TABLE V  
RARE-PATTERN LEARNING AS HISTORICAL EXPERIENCE ACCUMULATES.
<table><tr><td>Historical situations</td><td>Proposed micro-F1</td></tr><tr><td>100</td><td> $\overline { { 0 . 7 7 7 \pm 0 . 0 7 7 } }$ </td></tr><tr><td>250</td><td> $0 . 9 1 2 \pm 0 . 0 5 1$ </td></tr><tr><td>500</td><td> $0 . 9 9 6 \pm 0 . 0 0 5$ </td></tr><tr><td>1,000</td><td> $0 . 9 9 5 \pm 0 . 0 0 6$ </td></tr><tr><td>2,000</td><td> $0 . 9 9 8 \pm 0 . 0 0 3$ </td></tr><tr><td>4,000</td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td></tr><tr><td>8,000</td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td></tr></table>

where �, �, and � are separated by long random intervals. The same relation must recur across multiple independent episodes before it should be regarded as reusable knowledge. Since the event codes are randomly assigned, DeepSeek cannot infer the answer from pretrained common knowledge.

$$
A \ \cdot \cdot \cdot \ B \ \cdot \cdot \cdot \ C ,
$$

The history also contains deliberately difficult counterevidence: isolated occurrences of �, �, and $C ; A \cdots B$ episodes in which � does not follow; reversed $B \cdots A$ sequences; neutral episodes; and decoy triples that repeat strongly in only one temporal partition. A low-level temporal stage therefore produces an over-complete set of candidate relations and measures support across independent episodes, counterexamples, reverse evidence, cross-partition stability, and enrichment relative to event-order shuffling. This stage does not declare the final knowledge. DeepSeek periodically compares the accumulated evidence and converts accepted relations into explicit reusable thinking knowledge. The accepted relation is then materialized as training data so that a lightweight policy can activate “think about possible $C ^ { \mathfrak { s } }$ when a future $A \cdots B$ prefix reappears before � occurs.

Table VI shows a pronounced evidence-accumulation threshold. With only one or two positive episodes per hidden rule, candidate recall and Proposed relation F1 are both 0. At four repetitions, candidate recall rises to $0 . 9 3 3 \pm 0 . 1 4 9$ but Proposed relation F1 is only $0 . 3 3 3 \pm 0 . 4 7 1$ . Thus, most true rules have begun to enter the candidate set, but the accumulated evidence is not yet sufficient for stable consolidation across seeds. At eight independent repetitions, candidate recall reaches $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ , and Proposed+DeepSeek reaches 1.000±0.000 relation F1. This transition supports a three-stage interpretation: insufficient evidence, candidate emergence, and stable knowledge consolidation.

The ablations at eight repetitions further isolate what makes the knowledge discoverable. No Consolidation, Unordered, and Local all obtain 0 relation F1, whereas Statistics-Only, Proposed+DeepSeek, and Oracle all obtain 1.000. Destroying long-range order or restricting the analysis to local structure therefore removes the signal, while repeated long-range evidence is sufficient to identify the hidden relation. The fact that Statistics-Only matches Proposed in this controlled synthetic task is important: it indicates that the LLM is not required to outperform statistical evidence ranking for numerical relation identification. Its role is instead to organize the accumulated cross-experience evidence, formulate accepted relations as explicit reusable knowledge, and convert them into new thinking materials.

This distinction is confirmed by the downstream internalization test. At eight repetitions, the future-thinking policy trained from Statistics-Only, Proposed, or Oracle relations reaches an accuracy of $1 . 0 0 0 \pm 0 . 0 0 0 .$ , whereas No Consolidation, Unordered, and Local remain at $0 . 3 3 3 \pm 0 . 0 0 0 .$ , which is the chance level for three possible future-thinking targets. Hence, once a stable $A \cdots B  C$ regularity has been consolidated, it can be transformed into a policy that activates the appropriate thinking about � before � is observed. Figure 5 summarizes the relation-discovery transition as repeated independent evidence accumulates.

Overall, the six experiments form a connected verification chain. Experiment 1 shows why reusable thinking knowledge should be consolidated rather than indefinitely retrieved from a growing explicit history. Experiments 2 and 3 show that the policy can preserve non-obvious and transition-dependent thinking requirements. Experiment 4 demonstrates that the learned guidance improves actual DeepSeek reasoning. Experiment 5 shows that a thinking policy emerges as relevant experience accumulates. Experiment 6 extends this process beyond single experiences: repeated temporally dispersed events can first be recognized as a stable cross-experience regularity, converted into new thinking knowledge, and finally internalized so that the corresponding thinking is activated before the future outcome occurs.

TABLE VI  
CROSS-EXPERIENCE KNOWLEDGE DISCOVERY. CANDIDATE RECALL AND RELATION F1 ARE REPORTED OVER FIVE SEEDS. FUTURE-THINKING ACCURACY IS EVALUATED AT EIGHT REPETITIONS.
<table><tr><td>Repetitions/rule</td><td>Candidate Recall</td><td> $\overline { { \mathrm { S t a t i s t i c s  – O n l y ~ F 1 } } }$ </td><td>Proposed+DeepSeek F1</td><td>Oracle F1</td></tr><tr><td>1</td><td> $\overline { { 0 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td> $\overline { { 0 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td> $\overline { { 0 . 0 0 0 \pm 0 . 0 0 0 } }$ </td><td> $\overline { { 1 . 0 0 0 \pm 0 . 0 0 0 } }$ </td></tr><tr><td>2</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td></td><td> $0 . 9 3 3 \pm 0 . 1 4 9$ </td><td> $0 . 3 3 3 \pm 0 . 4 7 1$ </td><td> $0 . 3 3 3 \pm 0 . 4 7 1$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>8</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

At 8 repetitions, No-Consolidation, Unordered, and Local relation F1 are all 0.000 ± 0.000.

TABLE VII  
FUTURE-THINKING ACTIVATION AFTER CROSS-EXPERIENCE CONSOLIDATION AT EIGHT REPETITIONS PER HIDDEN RULE.
<table><tr><td>Knowledge source</td><td>Future-thinking accuracy</td></tr><tr><td>No Consolidation</td><td> $\overline { { 0 . 3 3 3 \pm 0 . 0 0 0 } }$ </td></tr><tr><td>Unordered</td><td> $0 . 3 3 3 \pm 0 . 0 0 0$ </td></tr><tr><td>Local</td><td> $0 . 3 3 3 \pm 0 . 0 0 0$ </td></tr><tr><td>Statistics-Only</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Proposed+DeepSeek</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Oracle</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

![](images/97915ac91f073f36e5efe0979f1399bb0015c301429496df6bc381a9a8f1a12b.jpg)  
Fig. 5. Cross-experience relation discovery as the number of independent supporting episodes increases. Markers with lower opacity show individual random-seed results, while dashed lines show the mean performance. The large variation at four repetitions indicates an intermediate stage in which true relations have begun to emerge but are not yet consolidated consistently across runs.

## V. CONCLUSION

This paper proposed a situation-conditioned thinking memory framework for long-term LLM agents. Instead of repeatedly retrieving an ever-growing reasoning history, the framework learns a lightweight policy that predicts what should be thought about under the current situation, while leaving detailed reasoning to the LLM. The model further represents situations as temporal or spatiotemporal evolutions and periodically consolidates temporally dispersed experiences to identify stable cross-experience regularities, transform them into new thinking knowledge, and internalize them for future activation. Experimental results show that the learned policy preserves non-obvious temporal thinking rules, improves

DeepSeek factor F1 from 0.789 to 0.868, maintains fixedsize online reasoning as explicit history grows, and reduces retrieval time by about 9.53× at 30,000 historical situations. Cross-experience experiments further show that sufficiently repeated long-range evidence can be consolidated into reusable knowledge and subsequently activate correct future thinking before the predicted outcome occurs. These results indicate that long-term agent memory can move beyond storing or retrieving past content toward progressively learning what to think about from accumulated experience.

## REFERENCES

[1] W. X. Zhao, K. Zhou, J. Li, T. Tang, Z. Dong, Y. Hou, B. Zhang, Y. Min, J. Zhang, P. Liu et al., “A survey of large language models,” Frontiers of Computer Science, vol. 20, no. 12, p. 2012627, 2026.

[2] H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman, N. Akhtar, N. Barnes, and A. Mian, “A comprehensive overview of large language models,” ACM Transactions on Intelligent Systems and Technology, vol. 16, no. 5, pp. 1–72, 2025.

[3] W. Wang, L. Dong, H. Cheng, X. Liu, X. Yan, J. Gao, and F. Wei, “Augmenting language models with long-term memory,” Advances in neural information processing systems, vol. 36, pp. 74 530–74 543, 2023.

[4] Q. Wang, Y. Fu, Y. Cao, S. Wang, Z. Tian, and L. Ding, “Recursively summarizing enables long-term dialogue memory in large language models,” Neurocomputing, vol. 639, p. 130193, 2025.

[5] J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” in Proceedings ofthe 36th annual acm symposium on user interface software and technology, 2023, pp. 1–22.

[6] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, “Memgpt: Towards llms as operating systems,” arXiv preprint arXiv:2310.08560, 2023.

[7] K. T.-i. Ong, N. Kim, M. Gwak, H. Chae, T. Kwon, Y. Jo, S.-w. Hwang, D. Lee, and J. Yeo, “Towards lifelong dialogue agents via timeline-based memory management,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 8631–8661.

[8] D. Jiang, Y. Li, G. Li, and B. Li, “Magma: A multi-graph based agentic memory architecture for ai agents,” in Proceedings of the 64th Annual Meeting of the Association

for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 36 848–36 865.

[9] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” Advances in neural information processing systems, vol. 36, pp. 8634–8652, 2023.

[10] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, “Voyager: An openended embodied agent with large language models,” arXiv preprint arXiv:2305.16291, 2023.

[11] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes,” in Findings of the association for computational linguistics: ACL 2023, 2023, pp. 8003–8017.

[12] Y. Ge, S. Romeo, J. Cai, R. Shu, Y. Benajiba, M. Sunkara, and Y. Zhang, “Tremu: Towards neurosymbolic temporal reasoning for llm-agents with memory in multi-session dialogues,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 18 974–18 988.

[13] M. Su, Y. Guo, Z. Hou, L. Bai, Z. Li, Y. Zhang, G. Yin, W. Lin, X. Jin, J. Guo et al., “Beyond dialogue time: Temporal semantic memory for personalized llm agents,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 29 935–29 951.

[14] J. Luo, Y. Tian, C. Cao, Z. Luo, H. Lin, K. Li, C. Kong, R. Yang, and J. Ma, “From storage to experience: A survey on the evolution of llm agent memory mechanisms,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 41 622–41 652.

Hong Su received the MS and PhD degrees, in 2006 and 2022, respectively, from Sichuan University, Chengdu, China. He is currently a researcher of Chengdu University of Information Technology Chengdu, China. His research interests include blockchain, cross-chain and smart contract.