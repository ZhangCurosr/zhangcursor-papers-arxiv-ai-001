# PERSONALIZING LLM AGENT MEMORY USING BIOMETRICS

Yanhong Qian, Qingguo Meng, Shihao Ding, Xingbo Dong, Zhe Jin, Hanrui Wang, and Isao Echizen (Senior Member, IEEE) Anhui Provincial Key Laboratory of Secure Artificial Intelligence,   
School of Artificial Intelligence, Anhui University, Hefei 230093, China yanhongqian@stu.ahu.edu.cn; mqg1024@163.com; shihaoding@stu.ahu.edu.cn; xingbo.dong@ahu.edu.cn; jinzhe@ahu.edu.cn

National Institute of Informatics, Tokyo 101-8430, Japan hanrui\_wang@nii.ac.jp; iechizen@nii.ac.jp

## ABSTRACT

Personalized memory helps LLM agents deliver stable, tailored assistance by storing and reusing user-specific data across interactions. In multi-user scenarios, however, retrieval must consider not only semantic similarity but also whether the current requester matches the identity associated with the stored memory. We propose Bio-Memory, a biometric-aware memory architecture that conditions memory retrieval on both semantic similarity and biometric matching. Built on top of A-Mem, Bio-Memory augments each atomic memory note with a biometric embedding and uses biometric matching to form the retrieval candidate pool before semantic ranking. We evaluate Bio-Memory on LoCoMo in a 10-user shared-agent setting over 7 face benchmarks and 10 palmprint protocols. Across datasets, Bio-Memory consistently separates owner and non-owner queries. Under face-based personalization, the largest average gap reaches 27.29% / 21.15% in F1 / BLEU-1 on CALFW; under palmprint-based personalization, the corresponding gap is 25.75% / 19.22% on MS\_Blue. These results support biometrics as a practical control signal for personalized memory retrieval in shared environments.

Keywords face recognition · palmprint recognition · identity-aware memory · multi-user LLM agents

## 1 Introduction

Large language model (LLM) agents are increasingly expected to function as persistent personalized assistants rather than one-shot question answering systems [1]. To support long-term interaction, recent memory architectures allow agents to store and reuse user-related information such as preferences, schedules, historical interactions, and taskspecific facts across sessions [2]. A representative example is A-Mem [3], which organizes memories as structured notes containing fields such as timestamp, content, keywords, context, tags, and semantic embeddings. More broadly, systems such as MemoryBank [4], Mem0 [5] and PersonaMem-v2 [6] show that memory can substantially improve interaction continuity and user adaptation [2].

However, as memory-augmented LLM agents move from single-user devices to shared environments, personalization brings a new challenge: the system must not only retrieve relevant memories, but also retrieve the right user’s memories. Existing personalized memory systems are generally built around an externally supplied identity, such as an account, user ID, or session boundary [2]. This assumption is fragile in shared settings, where a smart home assistant, public terminal, or enterprise assistant may serve multiple people via the same device or active session. Accounts, passwords, and session states can be shared, reused, or left active, so identifying the active profile differs from identifying the current human requester. For personalized memory retrieval, this distinction is critical: the same query, such as “What is my next appointment?”, should retrieve different memories depending on who is physically making the request.

![](images/65adc59c71d19409742a3e9063d70d43e10a92dd9695e141212d2108da0ac539.jpg)  
(a) Memory in A-Mem

![](images/e29a4d6567085e0034cd52a9dab3818238fea0ca8479e4ac299d478ab8682e75.jpg)  
(b) Memory in Bio-Memory  
Figure 1: Comparison of memory note structures in (a) A-Mem and (b) the proposed Bio-Memory. The atomic memory note of A-Mem consists of timestamps, content, keywords, context, tags, and semantic embeddings. Bio-Memory inherits this structure and additionally integrates biometric embeddings. Different colors in (b) denote memories from distinct users, supporting joint semantic and biometric matching for accurate memory retrieval in shared scenarios.

Figure 1 shows the structural difference between A-Mem and Bio-Memory. Bio-Memory preserves the semantic fields of atomic memory notes while adding a biometric embedding that ties each note to a physical user. This changes personalization from purely semantic retrieval to retrieval conditioned on both semantic relevance and the current user’s biometric identity. Rather than using biometrics only for session authentication, we use them to determine which memories may enter the candidate pool. This design is particularly useful in shared environments, where the session state may remain unchanged even when the active requester changes.

Based on this idea, we propose Bio-Memory, a biometric-aware memory architecture built on top of A-Mem. Bio-Memory preserves the atomic note organization of A-Mem while augmenting each note with an additional biometric embedding. At inference time, the current user’s probe biometric embedding is matched against the stored biometric embedding attached to each memory note, and semantic retrieval is then performed only over the resulting biometricmatched candidate memories. We evaluate Bio-Memory in a 10-user shared-agent setting using LoCoMo together with public face and palmprint benchmarks.

Our contributions are threefold: i) we augment atomic memory notes with biometric identity evidence, grounding agent memories in both semantic content and the physical identity of the requester; ii) we propose Bio-Memory, a biometric-aware extension of A-Mem that constructs a biometric-matched candidate pool before semantic retrieval and response generation; iii) we present an extensive 10-user shared-agent evaluation on LoCoMo, covering seven face benchmarks and ten palmprint protocols.

## 2 Related Work

## 2.1 Memory-Augmented LLM Agents

Long-interaction LLM agents rely on persistent external memory to retain user state beyond the limited context window of a single prompt [7, 8]. Early memory-augmented agent frameworks primarily target long-term dialogue continuity. Generative Agents records all user observations into a textual memory stream, ranks stored records via recency, relevance and importance, and periodically generates high-level reflective summaries to support downstream planning and decision-making [9]. To resolve identical context-length bottlenecks, MemGPT adopts an OS-inspired memory architecture. It regards the prompt as finite working memory and implements explicit memory paging to swap content between in-context windows and external storage [10]. Collectively, these pioneer works establish two foundational design principles for agent memory. Personal history must be decoupled from the prompt context, and retrieval rules dominate which historical information enters model reasoning.

Subsequent research evolves generic long-term memory toward user-personalized storage. MemoryBank equips dialogue agents with dedicated long-term memory, event summaries and static user profiles, and introduces an Ebbinghaus-inspired forgetting mechanism to dynamically adjust memory activation levels instead of treating all records as equally persistent [4]. Treating memory management as a data maintenance pipeline, Mem0 automatically extracts core factual knowledge from conversations, consolidates sparse records into compact storage, and supports standard CRUD operations to maintain memory consistency. Its graph-enhanced extension further models relational dependencies between stored facts [5]. PersonaMem-v2 explores implicit user personalization across disjoint multi-turn sessions, demonstrating that compactly compressed user memory drastically cuts token overhead compared to feeding full dialogue history, while preserving strong personalized reasoning performance [6]. Personalized dialogue agents push this direction further. LD-Agent combines long- and short-term memory banks with persona modeling for sustained interaction, while RMM improves personalized retrieval through prospective and retrospective reflection over long-horizon dialogue history [11, 12].

Recent system-level research redefines memory as a standalone infrastructure layer rather than a single retrieval module. MemOS partitions memory systems into modular storage, update, retrieval and generation components, and advocates a unified OS-style abstraction to manage heterogeneous memory resources [13]. Concurrent survey literature also highlights that memory systems should be evaluated based on core primitives including consolidation, update, indexing, forgetting and retrieval, rather than merely framed as simple context window extensions [2]. Existing studies collectively frame memory design as a joint optimization problem of storage layout, record organization and retrieval strategy. Nevertheless, nearly all prior work operates under a single implicit assumption. Retrieval only needs to prioritize task-relevant memories. This assumption fails in multi-user shared agent environments, where the system must additionally validate whether candidate memories belong to the current requesting user before reasoning.

## 2.2 A-Mem

Among existing structured memory frameworks, A-Mem bears the closest connection to our bio-memory pipeline, as it organizes long-term memory into graph-connected atomic notes instead of flat unstructured dialogue archives [3]. Drawing inspiration from Zettelkasten note-taking systems, each memory unit stores complete metadata such as interaction text, timestamps, keywords, tags, descriptive context, semantic embeddings, and bidirectional links to related records. Instead of appending raw dialogue text naively when new interactions arrive, A-Mem invokes the LLM to generate structured attributes for new notes, matches them against historical records to identify semantic connections, and embeds new entries into a dynamically evolving memory graph via dedicated indexing and link construction logic. This transforms memory storage from passive log recording into an active, structured organization process.

A core highlight of A-Mem is its support for continuous memory evolution [3]. Incoming new observations trigger iterative updates to historical notes, enriching them with supplementary context, refined tag labels and strengthened cross-note links over time. Unlike prior flat memory systems that treat stored text as static unmodifiable chunks, A-Mem’s graph architecture enables lifelong refinement of memory representations as user experience accumulates. This design delivers strong performance on memory-intensive QA benchmarks. Retrieval operates over semantically enriched, interconnected note units, allowing downstream reasoning to recover not only isolated facts but complete contextual evidence chains.

A-Mem’s modular note structure makes it a natural base for our identity-aware extension. Since each note function as an independent retrieval unit, we can attach biometric identity metadata without rewriting its core note creation, graph linking or semantic ranking modules. However, A-Mem’s retrieval logic solely relies on semantic similarity ranking, which fits single-user scenarios perfectly. The only retrieval target is matching personal history from one fixed user. In multi-user shared memory pools, semantically identical notes from different users coexist and create a critical unaddressed security risk. Semantic ranking cannot distinguish cross-user private records. To resolve this limitation, bio-memory builds upon A-Mem’s graph note architecture and inserts a biometric identity gating filter prior to semantic retrieval, rather than overhauling its fundamental memory organization design.

## 2.3 Identity and Biometrics in Multi-User Agent Settings

As LLM agents move into shared environments, logical credentials or session identifiers no longer reliably specify the physical requester [14, 15]. Family assistants, public kiosks, and enterprise terminals may all serve multiple people within the same active session, so account-level identity and physical identity can diverge. Prior work has identified this ambiguity as a central challenge for multi-user agents [14] and has shown that memory systems may expose stored personal information under mismatched or crafted queries [16]. Existing multi-user agent research therefore highlights the need to align the active requester with the correct personalized context, but generally does not use physical identity evidence as the primary signal for constructing the retrieval candidate set.

![](images/d4ac4a5ab57239b685ccace8404c39a0730f4dc4e9e18c18788ae0c7fa7541ba.jpg)  
Figure 2: Overall pipeline of Bio-Memory. (a) Note Construction: Raw agent-environment interactions are processed into structured memory notes with semantic attributes. (b) Note Storage: Each note is stored with full attributes including temporal information, textual content, contextual details, keywords, tags, semantic embeddings and biometric embeddings to support dual semantic and biometric memory management. (c) Link Generation and Memory Evolution: The LLM builds semantic links between memories and dynamically updates attributes of historical memories via new memory inputs, enabling continuous knowledge evolution and interconnected memory boxes. (d) Memory Retrieval: Biometric matching is performed prior to semantic ranking to filter valid memory candidates. Qualified memories are fed into the LLM for personalized response generation, while the model directly leverages parametric knowledge if no valid memories are available.

Biometric recognition offers a direct way to ground memory access in the current requester. Face recognition methods such as ArcFace [17] provide strong non-contact verification, while palmprint recognition captures complementary structural and textural cues and remains reliable across different sensing conditions [18]. Although biometrics are commonly used for login or session authentication, prior work on continuous authentication shows that one-time identity verification is often insufficient once the active user can change during ongoing interaction [19]. However, biometric evidence is still less often integrated into the retrieval pipeline itself. Bio-Memory addresses this gap by attaching biometric embeddings to memory notes and using biometric matching to determine which memories are eligible for semantic retrieval.

This distinction between session authentication and retrieval conditioning matters. A user may be authenticated at system entry, yet the memory layer must still decide which subset of notes should be considered for a query, especially when sessions are shared or persistent. By moving biometric evidence to the retrieval stage, Bio-Memory uses identity not only as an access credential but also as a constraint on memory search.

## 3 Methodology

## 3.1 Overview

Bio-Memory is a biometric-aware memory architecture for personalized LLM agents. Built on top of A-Mem [3], it augments each personal memory record with a biometric embedding, so that memory retrieval depends on both semantic similarity and biometric matching. The framework preserves the original agentic memory capabilities of A-Mem while introducing identity-conditioned retrieval at query time.

Figure 2 illustrates the full Bio-Memory pipeline, including note construction, note storage with biometric embeddings, memory evolution, and biometric-aware retrieval.

Bio-Memory is evaluated under two query-time roles within the shared-agent protocol. In the Owner condition, the probe biometric embedding matches the stored biometric embedding attached to the target memory records, so those memories participate in retrieval. In the Non-owner condition, the probe biometric embedding does not match, so those memories are excluded from the matched candidate pool. For each question, Bio-Memory constructs a matched memory pool, sorts it chronologically, ranks it by semantic similarity, selects the top-K results, and passes them to the LLM. If matching leaves no available memory, the model receives an empty memory context. Bio-Memory is therefore a lightweight retrieval-layer extension rather than a new end-to-end agent stack.

## 3.2 Memory Construction

Following the design of A-Mem, for each interaction input, Bio-Memory first constructs a structured memory note $m _ { i }$ with rich semantic attributes. We extend this structure by incorporating a biometric embedding $b _ { i }$ for personalization, yielding the unified representation:

$$
m _ { i } = \left\{ c _ { i } , t _ { i } , K _ { i } , G _ { i } , X _ { i } , e _ { i } , L _ { i } , b _ { i } \right\} ,\tag{1}
$$

where $c _ { i }$ denotes the original interaction content, $t _ { i }$ the timestamp, $K _ { i }$ LLM-generated keywords, $G _ { i }$ LLM-generated tags, $X _ { i }$ the LLM-produced contextual description, $e _ { i }$ the semantic embedding of the memory note, $L _ { i }$ the set of semantically linked memories, and $b _ { i }$ the stored biometric embedding for this memory record.

These biometric embeddings are extracted via a pre-trained biometric encoder. In our evaluation setup, all memory records from the same conversational sample are assigned to a single user $u ,$ sharing the same stored biometric template $b _ { i } = b _ { u } ^ { \mathrm { g a l } }$ <sup>l</sup>. For face-based experiments, we use ArcFace; for palmprint-based experiments, we use CCNet. All biometric embeddings are L2-normalized and matched with cosine similarity.

To construct the structured attributes $K _ { i } , G _ { i }$ , and $X _ { i }$ , we prompt the LLM with a dedicated template $P _ { s 1 }$ :

$$
K _ { i } , G _ { i } , X _ { i } \gets \mathrm { L L M } \left( c _ { i } \parallel t _ { i } \parallel P _ { s 1 } \right) .\tag{2}
$$

The semantic embedding $e _ { i }$ is computed by encoding the concatenation of all textual components using a text encoder $f _ { \mathrm { e n c } } \colon$

$$
e _ { i } = f _ { \mathrm { e n c } } \left( \mathrm { c o n c a t } \left( c _ { i } , K _ { i } , G _ { i } , X _ { i } \right) \right) .\tag{3}
$$

Bio-Memory otherwise retains the original A-Mem mechanisms for link generation and memory evolution without additional modifications.

## 3.3 Memory Retrieval

Unlike conventional semantic-only retrieval, Bio-Memory conditions the retrieval candidate set on biometric matching before assessing semantic similarity. At query time step t, the current requester submits a query $q _ { t }$ alongside their real-time probe biometric embedding, denoted as $b _ { t }$ .

Bio-Memory first performs online biometric matching against the memory store to determine whether the probe biometric embedding matches the stored biometric embedding of each memory record. The biometric matching output for a specific memory record $m _ { i }$ is formalized as an indicator function:

$$
\begin{array} { r } { M _ { \mathrm { b i o } } ( b _ { t } , b _ { i } ) = \mathbb { I } \big [ \mathrm { s i m } _ { \mathrm { b i o } } ( b _ { t } , b _ { i } ) \geq \tau _ { \mathcal { D } } \big ] , } \end{array}\tag{4}
$$

where $\mathrm { s i m } _ { \mathrm { b i o } } ( \cdot )$ computes the biometric similarity (e.g., cosine similarity), and $\tau _ { \mathcal { D } }$ is a predefined verification threshold optimized on the respective biometric benchmark $\mathcal { D } .$ In the experimental protocol, $\tau _ { \mathcal { D } }$ is selected independently for each biometric benchmark based on its verification pairs. Specifically, cosine similarity scores are first computed for owner and non-owner comparisons, and $\tau _ { \mathcal { D } }$ is defined as the intersection of the corresponding score distributions that lies between their principal modes. The resulting benchmark-specific threshold is then fixed for all subsequent LoCoMo Owner and Non-owner trials on that benchmark.

Memories with $M _ { \mathrm { b i o } } = 1$ form a query-specific matched memory pool:

$$
\mathcal { M } _ { t } ^ { \mathrm { m a t c h } } = \{ m _ { i } \in \mathcal { M } : M _ { \mathrm { b i o } } ( b _ { t } , b _ { i } ) = 1 \} .\tag{5}
$$

Because this pool is constructed dynamically, personalization is determined directly by the requester’s biometric embedding at query time.

This order of operations is central to the method. If semantic ranking were applied before biometric filtering, notes from different users could already compete in the same candidate set. By matching biometrics first, Bio-Memory narrows the search space before relevance ranking and makes identity consistency easier to maintain.

Subsequently, we sort the matched pool chronologically to preserve temporal coherence:

$$
\mathcal { M } _ { t } ^ { \mathrm { t i m e } } = \mathrm { S o r t T i m e } \left( \mathcal { M } _ { t } ^ { \mathrm { m a t c h } } \right) .\tag{6}
$$

Semantic retrieval is then performed within the time-sorted pool. We encode the query $q _ { t }$ into $e _ { t }$ using the text encoder $f _ { \mathrm { e n c } }$ . The semantic similarity between the query and a candidate memory is computed as cosine similarity:

$$
\mathrm { s i m } _ { \mathrm { s e m } } ( q _ { t } , m _ { i } ) = \frac { e _ { t } ^ { \top } e _ { i } } { \| e _ { t } \| _ { 2 } \| e _ { i } \| _ { 2 } } .\tag{7}
$$

Table 1: Face recognition evaluation datasets.
<table><tr><td>Dataset</td><td>Protocol / Subset</td><td>Type / Group</td><td>#Subjects</td><td>#Images</td></tr><tr><td>AgeDB-30 [20]</td><td>AgeDB-30</td><td>Age</td><td>568</td><td>16,488</td></tr><tr><td>CALFW [21]</td><td>CALFW</td><td>Age</td><td>5,749</td><td>12,174</td></tr><tr><td>CFP-FF [22]</td><td>CFP-FF</td><td>Frontal</td><td>500</td><td>7,000</td></tr><tr><td>CFP-FP [22]</td><td>CFP-FP</td><td>Pose</td><td>500</td><td>7,000</td></tr><tr><td>CPLFW [23]</td><td>CPLFW</td><td>Pose</td><td>5,749</td><td>11,652</td></tr><tr><td>LFW [24]</td><td>LFW</td><td>Frontal</td><td>5,749</td><td>13,233</td></tr><tr><td>VGG2-FP [25]</td><td>VGG2-FP</td><td>Pose</td><td>9,131</td><td>~3.31M</td></tr></table>

Table 2: Palmprint recognition evaluation datasets.
<table><tr><td>Dataset</td><td>Protocol / Subset</td><td>Type / Group</td><td>#Subjects</td><td>#Images</td></tr><tr><td>CasiaM [26]</td><td>CasiaM_460</td><td>460nm</td><td>200</td><td>1,200</td></tr><tr><td>CasiaM [26]</td><td>CasiaM_700</td><td>700nm</td><td>200</td><td>1,200</td></tr><tr><td>CasiaM [26]</td><td>CasiaM_850</td><td>850nm</td><td>200</td><td>1,200</td></tr><tr><td>IITD [27]</td><td>IITD</td><td>Contactless</td><td>460</td><td>2,300</td></tr><tr><td>MS [28]</td><td>MS_Blue</td><td>Blue</td><td>500</td><td>6,000</td></tr><tr><td>MS [28]</td><td>MS_Green</td><td>Green</td><td>500</td><td>6,000</td></tr><tr><td>MS [28]</td><td>MS_NIR</td><td>NIR</td><td>500</td><td>6,000</td></tr><tr><td>MS [28]</td><td>MS_Red</td><td>Red</td><td>500</td><td>6,000</td></tr><tr><td>PolyU [29]</td><td>PolyU</td><td>Contact</td><td>378</td><td>7,560</td></tr><tr><td>Tongji [30]</td><td>Tongji</td><td>Contactless</td><td>600</td><td>12,000</td></tr></table>

Bio-Memory selects the top-K most relevant memories from the time-sorted pool, denoted as $\begin{array} { r l } { \mathcal { R } _ { t } } & { { } = } \end{array}$ $\mathrm { T o p K } _ { m _ { i } \in \mathcal { M } _ { \epsilon } ^ { \mathrm { t i m e } } } \operatorname { s i m } _ { \mathrm { s e m } } ( q _ { t } , m _ { i } )$ . The value of K is fixed to 10 across both Owner and Non-owner conditions, ensuring that the probe biometric embedding is the only experimental difference between the two settings. The final retrieved memory set is formally defined as:

$$
\mathcal { R } _ { t } = \left\{ \begin{array} { l l } { \mathrm { T o p K } _ { m _ { i } \in \mathcal { M } _ { t } ^ { \sf t i m e } } \sin ( q _ { t } , m _ { i } ) , } & { \mathcal { M } _ { t } ^ { \mathrm { m a t c h } } \neq \emptyset , } \\ { \emptyset , } & { \mathcal { M } _ { t } ^ { \mathrm { m a t c h } } = \emptyset . } \end{array} \right.\tag{8}
$$

Bio-Memory combines biometric matching, chronological ordering, and semantic ranking to determine which memories are passed to the LLM. Users asking the same question can therefore be routed to entirely different personalized memory sets.

This formulation also clarifies the role of an empty matched set. When $\mathcal { M } _ { t } ^ { \mathrm { m a t c h } } = \emptyset .$ , the system withholds user-specific memory because the identity condition is unmet. In shared-agent settings, this is preferable to retrieving semantically plausible but identity-mismatched memories.

## 3.4 Response Generation

The retrieved memory records are serialized into a temporary context:

$$
C _ { t } = \mathrm { F o r m a t } ( \mathscr { R } _ { t } ) ,\tag{9}
$$

and the final answer is generated by the LLM conditioned on the question and the retrieved memory context:

$$
\hat { y } _ { t } = \mathrm { L L M } ( q _ { t } , C _ { t } ) .\tag{10}
$$

When $\mathcal { R } _ { t } = \emptyset .$ , the formatted context is empty, so Bio-Memory falls back to answering the question without personal memory support. This behavior is particularly important in the Non-owner setting: personal memories remain entirely absent from retrieval, while the model may still generate a reasonable response based only on its parametric knowledge.

## 4 Experimental Setup

## 4.1 Datasets

We evaluate Bio-Memory using LoCoMo [31] together with public biometric matching benchmarks. The face and palmprint benchmarks are summarized in Tables 1 and 2. Each LoCoMo conversational sample is paired with one

stored biometric embedding, and all memory entries from that sample share the same biometric template. For each biometric benchmark, we build independent same-identity and different-identity pairings between LoCoMo samples and biometric identities.

## 4.2 Evaluation Protocol

Our goal is to test whether biometric personalization at query time can isolate personal memory retrieval in a sharedagent setting. Since LoCoMo does not include biometrics, we bind each LoCoMo conversation to one stored biometric embedding from a biometric benchmark and store all memory entries from that conversation with the same template.

To simulate multi-user usage, we construct a 10-user shared-agent memory store by merging ten LoCoMo conversations into a single memory space containing ten biometric-linked user partitions. Each conversation is assigned one biometric identity from the corresponding benchmark, and all memories from that conversation share the same stored gallery template. For a target partition, the Owner query uses a probe embedding sampled from the same biometric identity as the stored gallery template, while the Non-owner query uses a probe embedding from a different user within the same shared setting. Biometric matching is applied before semantic retrieval, so only matched memories enter the retrieval candidate pool. If no entry satisfies biometric matching, the model answers without personal memory support.

For evaluation, we report Multi-Hop, Temporal, and Single-Hop questions, which directly test memory retrieval and memory-grounded reasoning. To reduce evaluation cost, we use a fixed 10% subset of the LoCoMo test split by setting the sampling ratio to 0.1. Within each run, the same sampled question subset is reused across biometric datasets and across the Owner and Non-owner conditions, so the probe biometric embedding remains the primary experimental difference. We compare model predictions with the LoCoMo reference answers using token-level F1 and BLEU-1. All scores are reported as percentages, the reported Average is the macro-average over the three question types, and Avg Gap is computed as Owner Average minus Non-owner Average.

We emphasize these three question types because they most directly reflect the effect of retrieval quality on downstream answering. Single-Hop questions test whether one key fact can be recovered from the correct personal memory partition. Multi-Hop questions require the model to combine multiple pieces of stored evidence, making them more sensitive to noisy or identity-misaligned retrieval. Temporal questions are especially informative because they depend on the ordering of personal events and are difficult to answer from generic parametric knowledge alone.

## 4.3 Implementation Details

Our implementation builds upon the A-Mem memory layer and adds biometric personalization at retrieval time. Semantic indexing uses a sentence-transformer retriever based on all-MiniLM-L6-v2, while the answer generation model and inference backend follow the evaluation script. To ensure fair comparison, owner and non-owner trials share the same memory store, retrieval configuration, and evaluation subset; the only inference difference is the query probe biometric embedding.

## 5 Results

## 5.1 Quantitative Results

Tables 3 and 4 summarize the quantitative results on palmprint and face benchmarks in the 10-user shared-agent setting. Across both tables, the same pattern appears repeatedly: once retrieval is conditioned on biometric identity, owner queries preserve access to the correct personal evidence, whereas non-owner queries lose that evidence before semantic ranking begins.

Overall trend. The owner advantage is uniform across all datasets, all question types, and both evaluation metrics. This gap is not small. In the palmprint results, the average owner–non-owner gap ranges from 20.13 to 25.75 in F1 and from 14.97 to 19.53 in BLEU-1; the strongest separation appears on MS\_Blue, where the owner average reaches 30.68 / 23.56 while the non-owner average drops to 4.93 / 4.34. The face benchmarks show the same behavior, with average gaps from 23.58 to 27.29 in F1 and from 18.34 to 21.15 in BLEU-1. CALFW is the clearest example: the owner average is 30.89 / 24.31, compared with only 3.60 / 3.16 for the non-owner. Because the owner and non-owner settings use the same memory store, the same question subset, and the same downstream generator, these consistent margins indicate that the decisive change happens at the biometric filtering stage.

Authorized utility. The owner and non-owner rows also make clear that the biometric filtering step is not achieved by sacrificing useful memory for the legitimate user. In the palmprint table, owner averages remain stable between 26.27 / 21.25 and 31.32 / 24.00 across all ten protocols, whereas non-owner averages stay much lower, mostly between 4.49 /

Table 3: Downstream LoCoMo QA results under palmprint-based retrieval personalization in the 10-user shared-agent setting. Each entry is reported as F1 / BLEU-1 in percentages (%). Average is the macro-average over Multi-Hop, Temporal, and Single-Hop questions, and Avg. Gap is Owner Average minus Non-owner Average.
<table><tr><td>Dataset</td><td>Condition</td><td>Multi-Hop</td><td>Temporal</td><td>Single-Hop</td><td>Average</td><td>Avg. Gap</td></tr><tr><td rowspan="2">PolyU</td><td>Owner</td><td>23.70 / 16.32</td><td>31.16/21.19</td><td>27.08 / 22.67</td><td>27.31 / 20.06</td><td rowspan="2">22.82 / 16.18</td></tr><tr><td>Non-owner</td><td>5.26 / 4.67</td><td>0.36 / 0.27</td><td>7.84 / 6.71</td><td>4.49 / 3.88</td></tr><tr><td rowspan="2">IITD</td><td>Owner</td><td>20.64 /17.79</td><td>23.73 / 16.10</td><td>34.43 / 29.85</td><td>26.27 / 21.25</td><td rowspan="2">20.75 / 16.75</td></tr><tr><td>Non-owner</td><td>5.27 / 4.84</td><td>3.09 / 1.66</td><td>8.21 / 7.00</td><td>5.52 / 4.50</td></tr><tr><td rowspan="2">CasiaM_850</td><td>Owner</td><td>20.41 / 16.55</td><td>28.14 / 18.77</td><td>31.65 / 25.41</td><td>26.73 / 20.24</td><td rowspan="2">20.13 / 14.97</td></tr><tr><td>Non-owner</td><td>7.51 / 6.07</td><td>3.45 / 1.93</td><td>8.85 / 7.80</td><td>6.60 / 5.27</td></tr><tr><td rowspan="2">MS_Green</td><td>Owner</td><td>26.46 / 21.09</td><td>29.58 / 21.24</td><td>34.49 / 29.42</td><td>30.18 / 23.92</td><td rowspan="2">24.47 / 19.11</td></tr><tr><td>Non-owner</td><td>5.62 / 5.16</td><td>2.82 / 1.49</td><td>8.70 / 7.79</td><td>5.71 / 4.81</td></tr><tr><td rowspan="2">CasiaM_460</td><td>Owner</td><td>25.71 / 19.86</td><td>31.71 / 21.86</td><td>36.53 / 30.28</td><td>31.32 / 24.00</td><td rowspan="2">25.73 / 19.13</td></tr><tr><td>Non-owner</td><td>5.72 / 6.35</td><td>3.56 / 2.06</td><td>7.48 / 6.19</td><td>5.59 / 4.87</td></tr><tr><td rowspan="2">MS_Red</td><td>Owner</td><td>19.62 / 17.50</td><td>30.70 / 19.89</td><td>33.45 / 28.11</td><td>27.92 / 21.83</td><td rowspan="2">21.58 / 16.82</td></tr><tr><td>Non-owner</td><td>5.83 / 4.88</td><td>3.20 / 1.79</td><td>9.99 / 8.36</td><td>6.34 / 5.01</td></tr><tr><td rowspan="2">CasiaM_700</td><td>Owner</td><td>25.72 / 20.32</td><td>27.48 / 18.90</td><td>34.38 / 29.69</td><td>29.19 / 22.97</td><td rowspan="2">24.47 / 18.59</td></tr><tr><td>Non-owner</td><td>5.39 /5.39</td><td>1.83 / 1.39</td><td>6.95 / 6.37</td><td>4.72 / 4.38</td></tr><tr><td rowspan="2">MS_NIR</td><td>Owner</td><td>27.50 / 22.80</td><td>30.35 / 20.61</td><td>31.53 / 27.96</td><td>29.79 / 23.79</td><td rowspan="2">24.52 / 19.53</td></tr><tr><td>Non-owner</td><td>2.88 / 3.07</td><td>3.48 / 1.96</td><td>9.46 / 7.75</td><td>5.27 / 4.26</td></tr><tr><td rowspan="2">MS_Blue</td><td>Owner</td><td>22.26 / 16.97</td><td>35.89 / 24.28</td><td>33.90 / 29.43</td><td>30.68 / 23.56</td><td rowspan="2">25.75 / 19.22</td></tr><tr><td>Non-owner</td><td>5.61 / 5.48</td><td>2.43 / 1.19</td><td>6.76 / 6.35</td><td>4.93 / 4.34</td></tr><tr><td rowspan="2">Tongji</td><td>Owner</td><td>23.21 / 18.20</td><td>29.10 / 20.12</td><td>29.89 / 24.63</td><td>27.40 / 20.98</td><td rowspan="2">22.19 / 16.68</td></tr><tr><td></td><td>4.31 / 4.70</td><td>4.23 / 1.95</td><td>7.08 / 6.25</td><td>5.21 / 4.30</td></tr><tr><td rowspan="2"></td><td>Non-owner</td><td></td><td></td><td></td><td></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

3.88 and 6.60 / 5.27. The face table exhibits the same structure: owner averages stay in a narrow and healthy band from 28.59 / 22.57 to 31.41 / 23.07, while non-owner averages remain near the floor, between 3.60 / 3.16 and 5.24 / 4.44. This contrast matters for interpretation. If biometric conditioning simply removed too much memory from everyone, both rows in each dataset would collapse together. Instead, the owner rows remain strong, which means the filtering step is selective: it suppresses identity-mismatched evidence while preserving enough user-aligned notes for effective memory-grounded answering.

Question-type behavior. The per-task columns further explain where the separation comes from. Temporal questions are the most sensitive. In the palmprint results, owner Temporal F1 scores range from 23.73 to 35.89, but the corresponding non-owner scores are only 0.36 to 4.23; in the face results, owner Temporal F1 remains between 31.81 and 36.53, whereas non-owner Temporal F1 stays between 1.08 and 3.12. This is consistent with the nature of Temporal QA: once the correct user’s event chain is removed, the model has little basis for reconstructing the timeline. The Single-Hop and Multi-Hop columns show a related but slightly different pattern. Single-Hop owner scores stay strong in both tables because one key personal fact is often recoverable when the correct partition is retained, while Multi-Hop owner scores remain competitive because several related notes can still be retrieved together. The non-owner rows, by contrast, stay low across both columns, indicating that biometric filtering breaks not only isolated fact access but also evidence composition.

Cross-modal consistency. This effect is not tied to one biometric modality. Palmprint protocols and face benchmarks differ in sensing conditions, feature variation, and verification difficulty, yet both result tables preserve the same directional pattern: owner scores remain high and non-owner scores remain suppressed. The main difference lies in the sharpness of the margin. For example, the face benchmarks reach a 27.29 / 21.15 average gap on CALFW, while the palmprint protocols reach 25.75 / 19.22 on MS\_Blue. Even the weaker cases still maintain large separations, such as 20.13 / 14.97 on PolyU and 23.58 / 18.34 on CPLFW. This consistency across modalities supports the same interpretation: better verification produces a sharper memory boundary, but even when verification is less favorable, the identity-aware retrieval mechanism still preserves a clear distinction between matched and mismatched users.

Table 4: Downstream LoCoMo QA results under face-based retrieval personalization in the 10-user shared-agent setting. Each entry is reported as F1 / BLEU-1 in percentages (%). Average is the macro-average over Multi-Hop, Temporal, and Single-Hop questions, and Avg. Gap is Owner Average minus Non-owner Average.
<table><tr><td>Dataset</td><td>Condition</td><td>Multi-Hop</td><td>Temporal</td><td>Single-Hop</td><td>Average</td><td>Avg. Gap</td></tr><tr><td rowspan="2">AgeDB-30</td><td rowspan="2">Owner Non-owner</td><td>22.99 / 15.60</td><td>36.46 / 25.70</td><td>34.77 / 27.90</td><td>31.41 / 23.07</td><td rowspan="2">26.85 / 19.58</td></tr><tr><td>3.22 / 1.88</td><td>1.38 / 1.03</td><td>9.07 / 7.55</td><td>4.56 / 3.49</td></tr><tr><td rowspan="2">CALFW</td><td rowspan="2">Owner Non-owner</td><td>23.16 / 20.15</td><td>33.92 / 23.02</td><td>35.58 / 29.77</td><td>30.89 / 24.31</td><td rowspan="2">27.29 / 21.15</td></tr><tr><td>2.04 / 2.19</td><td>2.21 / 1.11</td><td>6.56 / 6.17</td><td>3.60 / 3.16</td></tr><tr><td rowspan="2">CFP-FF</td><td rowspan="2">Owner Non-owner</td><td>21.17 / 18.85</td><td>31.81 / 22.27</td><td>37.69 / 31.11</td><td>30.22 / 24.08</td><td rowspan="2">26.34 / 20.72</td></tr><tr><td>1.92 / 1.49</td><td>1.47 / 1.12</td><td>8.26 / 7.48</td><td>3.88 / 3.36</td></tr><tr><td rowspan="2">CFP-FP</td><td rowspan="2">Owner Non-owner</td><td>24.25 / 19.85</td><td>32.42 / 21.30</td><td></td><td>31.28 / 24.09</td><td rowspan="2">26.18 / 19.32</td></tr><tr><td>5.40 / 5.45</td><td>1.08 / 0.82</td><td>37.16/31.12 8.81 / 8.03</td><td>5.10/4.77</td></tr><tr><td rowspan="2">CPLFW</td><td rowspan="2">Owner Non-owner</td><td>20.54 / 17.61</td><td>31.27 / 21.58</td><td></td><td></td><td rowspan="2">23.58 / 18.34</td></tr><tr><td>4.47 / 3.99</td><td>2.36 / 1.42</td><td>33.96 / 28.52 8.19 / 7.28</td><td>28.59 / 22.57 5.01 / 4.23</td></tr><tr><td rowspan="2">LFW</td><td rowspan="2">Owner Non-owner</td><td rowspan="2">21.77 / 18.23</td><td rowspan="2">36.53 / 27.31</td><td rowspan="2"></td><td rowspan="2">30.42 / 24.40</td><td rowspan="2">25.81 / 20.60</td></tr><tr><td>32.96 / 27.65</td></tr><tr><td rowspan="2">VGG2-FP</td><td rowspan="2">Owner</td><td rowspan="2">2.19 / 2.21 22.28 / 18.58</td><td rowspan="2">2.73 / 1.39</td><td rowspan="2">8.91 / 7.80</td><td rowspan="2">4.61 / 3.80</td><td rowspan="2"></td></tr><tr><td>31.88 / 21.57</td></tr><tr><td rowspan="2"></td><td rowspan="2">Non-owner</td><td rowspan="2">5.85 / 5.06</td><td rowspan="2">3.12 / 1.69</td><td rowspan="2">38.49 / 33.10 6.75 / 6.57</td><td rowspan="2">30.88 / 24.42 5.24 / 4.44</td><td rowspan="2">25.64 / 19.98</td></tr><tr><td></td></tr></table>

Main takeaway. Taken together, these results show that Bio-Memory is not merely improving generic semantic retrieval. The crucial intervention happens earlier, at the stage where the candidate pool is constructed. Across all rows, owner queries retain access to the personalized notes needed for high-quality answers, while non-owner queries are consistently restricted to incomplete or generic context. The fact that this pattern holds simultaneously across palmprint protocols, face benchmarks, and all three question types indicates that the method acts as retrieval-time access control over memory evidence rather than as a post hoc adjustment of generated responses.

## 5.2 Qualitative Retrieval Analysis

Qualitative examples. Figure 3 gives concrete examples of the same effect. In the Multi-Hop case, the owner retrieves the right notes and answers “Three dogs,” while the non-owner falls back to “Unknown.” In the Temporal case, the owner answers “In July, 2022,” while the non-owner responds “Not specified.” In the Single-Hop case, the owner retrieves the relevant personal fact, whereas the non-owner produces an unsupported alternative. These examples are easy to read because they show the same question under the same system with only one change: the identity signal used at retrieval time.

Interpretation. The qualitative traces confirm that the main separation happens before answer generation. Owner and non-owner queries diverge as soon as candidate memories are selected. Once the correct notes are filtered out, Multi-Hop reasoning loses its evidence chain, Temporal questions lose the user-specific timeline, and even Single-Hop questions lose the one fact they need. The figure therefore supports the same conclusion as the tables: Bio-Memory works by controlling access to user-specific memory evidence at retrieval time.

## 6 Conclusion

This paper presented Bio-Memory, a biometric-aware memory architecture for personalized LLM agents that conditions personal memory retrieval on both semantic relevance and biometric matching at query time. By augmenting each memory note with a stored biometric embedding, Bio-Memory enables a shared agent to construct user-aligned retrieval contexts without changing the underlying LLM. Across seven face benchmarks, ten palmprint protocols, and a 10-user shared-agent evaluation, owners retain strong memory-grounded question answering performance, whereas non-owners lose access to the user-specific evidence needed for retrieval-based answering. These results support the feasibility and practical value of biometrics as a control signal for personalized memory retrieval in shared-agent settings.

More broadly, the study suggests that personalized memory for LLM agents is not only a relevance problem but also a user-alignment problem: the system must determine not only what memory is relevant, but also whose memory should be retrieved. Bio-Memory offers a practical way to introduce this distinction while preserving the strengths of structured memory systems such as A-Mem.

![](images/9b9a54c8e8b10fe62ae55d6dbae968954985fe9d102e8f10d4b1667ad4588a0d.jpg)  
Figure 3: Retrieval traces for owner-versus-non-owner queries in Bio-Memory. For the same question, owner and non-owner queries follow different retrieval paths because their probe biometric embeddings align differently with the personal memory store.

## References

[1] Minxing Zhang, Yi Yang, Roy Xie, Bhuwan Dhingra, Shuyan Zhou, and Jian Pei. Generalizability of large language model-based agents: A comprehensive survey. ACM Computing Surveys, 58(10):1–44, 2026.

[2] Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong, and Jeff Z Pan. Rethinking memory in llm based agents: Representations, operations, and emerging topics. arXiv preprint arXiv:2505.00675, 2025.

[3] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for LLM agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[4] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.

[5] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[6] Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, et al. Personamem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. arXiv preprint arXiv:2512.06688, 2025.

[7] Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, and Michal Shmueli-Scheuer. A survey on evaluation of llm-based agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 26690–26714, 2026.

[8] Jiacheng Yao, Guoxiu He, and Xin Xu. A collaborative reasoning framework for large language models in long-context q&a. Expert Systems with Applications, 299:129960, 2026.

[9] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

[10] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[11] Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua. Hello again! llm-powered personalized agent for long-term dialogue, 2025.

[12] Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents, 2025.

[13] Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Junpeng Ren, Huayi Lai, Hao Wu, Bo Tang, Zhenren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofeng Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. Memos: A memory os for ai system. arXiv preprint arXiv:2507.03724, 2025.

[14] Shu Yang, Shenzhe Zhu, Hao Zhu, José Ramón Enríquez, Di Wang, Alex Pentland, Michiel A Bakker, and Jiaxin Pei. Multi-user large language model agents. arXiv preprint arXiv:2604.08567, 2026.

[15] Keyeun Lee, Seo Hyeong Kim, Seolhee Lee, Jinsu Eun, Yena Ko, Hayeon Jeon, Esther Hehsun Kim, Seonghye Cho, Soeun Yang, and Eun-mee Kim. Spectrum: A grounded framework for multidimensional identity representation in llm-based agent. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025.

[16] Bo Wang, Weiyi He, Shenglai Zeng, Zhen Xiang, Yue Xing, Jiliang Tang, and Pengfei He. Unveiling privacy risks in LLM agent memory. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 25241–25260, Vienna, Austria, jul 2025. Association for Computational Linguistics.

[17] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.

[18] Ziyuan Yang, Huijie Huangfu, Lu Leng, Bob Zhang, Andrew Beng Jin Teoh, and Yi Zhang. Comprehensive competition mechanism in palmprint recognition. IEEE Transactions on Information Forensics and Security, 18:5160–5170, 2023.

[19] Jiajia Li, Qian Yi, Ming K. Lim, Shuping Yi, Pengxing Zhu, and Xingjun Huang. Mbbfauth: Multimodal behavioral biometrics fusion for continuous authentication on non-portable devices. IEEE Transactions on Information Forensics and Security, 19:10000–10015, 2024.

[20] Stylianos Moschoglou, Athanasios Papaioannou, Christos Sagonas, Jiankang Deng, Irene Kotsia, and Stefanos Zafeiriou. Agedb: the first manually collected, in-the-wild age database. In proceedings ofthe IEEE conference on computer vision and pattern recognition workshops, pages 51–59, 2017.

[21] Tianyue Zheng, Weihong Deng, and Jiani Hu. Cross-age lfw: A database for studying cross-age face recognition in unconstrained environments. arXiv preprint arXiv:1708.08197, 2017.

[22] Soumyadip Sengupta, Jun-Cheng Chen, Carlos Castillo, Vishal M Patel, Rama Chellappa, and David W Jacobs. Frontal to profile face verification in the wild. In 2016 IEEE winter conference on applications ofcomputer vision (WACV), pages 1–9. IEEE, 2016.

[23] Tianyue Zheng and Weihong Deng. Cross-pose lfw: A database for studying cross-pose face recognition in unconstrained environments. Technical Report 18-01, Beijing University of Posts and Telecommunications, February 2018.

[24] Gary B Huang, Marwan Mattar, Tamara Berg, and Erik Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. In Workshop onfaces in’Real-Life’Images: detection, alignment, and recognition, 2008.

[25] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automaticface & gesture recognition (FG 2018), pages 67–74. IEEE, 2018.

[26] Z Sun, T Tan, Y Wang, and SZ Li. Ordinal palmprint representation for personal identification. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, volume 1, pages 279–284, 2005.

[27] Ajay Kumar. Incorporating cohort information for reliable palmprint authentication. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 583–590. IEEE, 2008.

[28] David Zhang, Zhenhua Alexander Guo, Guangming Lu, Lei Zhang, and Wangmeng Zuo. An online system of multispectral palmprint verification. IEEE transactions on instrumentation and measurement, 59(2):480–490, 2010.

[29] David Zhang, Wai-Kin Kong, Jane You, and Michael Wong. Online palmprint identification. IEEE Transactions on pattern analysis and machine intelligence, 25(9):1041–1050, 2003.

[30] Lin Zhang, Lida Li, Anqi Yang, Ying Shen, and Meng Yang. Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach. Pattern Recognition, 69:199–212, 2017.

[31] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long term conversational memory of llm agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, 2024.