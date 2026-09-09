# BIO-MEMART: BIOMETRIC-AWARE KV CACHE MEMORY FOR MULTI-USER LLM AGENTS

Yanhong Qian, Xuanying He, Qingguo Meng, Shihao Ding, Xingbo Dong, and Zhe Jin

Anhui Provincial Key Laboratory of Secure Artificial Intelligence,   
School of Artificial Intelligence, Anhui University, Hefei 230093, China   
yanhongqian@stu.ahu.edu.cn; w126221228@stu.ahu.edu.cn; mqg1024@163.com; shihaoding@stu.ahu.edu.cn; xingbo.dong@ahu.edu.cn; jinzhe@ahu.edu.cn

Hanrui Wang and Isao Echizen (Senior Member, IEEE) National Institute of Informatics, Tokyo 101-8430, Japan hanrui\_wang@nii.ac.jp; iechizen@nii.ac.jp

## ABSTRACT

KV cache is evolving from a serving optimization into an external memory substrate for long-term LLM agents. In a shared multi-user deployment, however, reusable KV blocks introduce a missing access-control question: semantic relevance alone cannot determine whether a memory block is authorized for the current physical user. We propose Bio-MemArt, a biometric-aware KV-cache memory framework for multi-user LLM agents. Bio-MemArt attaches a normalized biometric template to each stored KV memory block, filters the shared memory pool with the current user’s biometric probe, and then runs the original MemArt retrieval and KV reuse pipeline only inside the authorized candidate pool. This design preserves latent-space retrieval, direct cache reuse, and decoupled position encoding while adding physical-user access control to shared KV memory. We evaluate Bio-MemArt under Owner and Non-owner query conditions on long-term dialogue QA with face and palmprint benchmarks. Across face benchmarks, the average owner and non-owner biometric success rates are 95.71% and 0.86%; across palmprint benchmarks, they are 97.60% and 2.00%. In the efficiency study, average prefill tokens drop from 18,781.96 under full-context prompting to 28.57 with Bio-MemArt, showing that biometric gating preserves the low-token operating regime of KV-cache memory.

Keywords biometric authentication · KV cache memory · LLM agents · multi-user personalization

## 1 Introduction

Large language model (LLM) agents are increasingly deployed as persistent assistants that must preserve user-specific state across many turns and sessions. In this setting, KV cache is no longer only a serving artifact. Recent work has started to treat reusable KV states as an external memory substrate for agentic systems, because KV blocks preserve model-native hidden states and avoid repeatedly rebuilding long contexts from scratch [1, 2, 3]. When historica context is already stored as reusable KV blocks, the system can keep memory in the model-native format instead of reconstructing it from text at every turn.

Shared deployments raise a new problem. A smart home assistant, public terminal, classroom device, or enterprise agent may serve multiple physical users through one active system. If their historical KV blocks are stored in a shared memory pool, semantic retrieval alone can still surface another user’s private memory. The core issue is therefore not only which KV block is relevant, but also whether that block is authorized for the current physical requester.

The broader KV-cache literature does not resolve this question. System work improves serving efficiency through better paging, scheduling, offloading, and storage management [4, 5, 6, 7]. Multi-agent reuse work shares KV states across agents through overlapping prefixes and collective communication [8, 9, 10]. MemArt further elevates reusable KV blocks into an explicit long-term memory substrate for LLM agents [1]. Yet these methods all optimize how KV blocks are stored, moved, reused, or retrieved; none decides whether a retrieved block should be accessible to the current physical user. This leaves a deployment gap between memory efficiency and memory authorization in real multi-user systems, where access control must be enforced before semantic KV retrieval can proceed.

Biometric identity provides the missing access signal. Face and palmprint embeddings can verify who is currently issuing the query more directly than account metadata alone [11, 12, 13]. The challenge is to add this identity gate without breaking the benefits that make KV-cache memory attractive in the first place: latent-space retrieval, direct cache reuse, and low prefill cost.

We propose Bio-MemArt, a biometric-aware KV-cache memory framework for multi-user LLM agents, built on the original MemArt pipeline [1]. Each stored KV memory block keeps the original MemArt fields, including the KV block, compressed key, and timestamp, and adds a biometric template vector. At query time, the current user’s biometric probe is compared with the template attached to each KV block. Only blocks whose cosine similarity exceeds a statistically estimated threshold enter the authorized candidate pool. The original MemArt retrieval then selects top-ranked KV blocks from this authorized candidate pool and reuses them through decoupled position encoding. Identity filtering therefore happens before semantic KV retrieval, while generation remains unchanged.

This paper makes three contributions. First, we propose Bio-MemArt, a biometric-aware KV-cache memory framework for multi-user LLM agents, which extends the original MemArt memory architecture with physical-user identity cues. Second, we design an identity-aware retrieval method that attaches biometric templates to KV memory blocks and filters the shared memory pool before native MemArt retrieval and KV reuse. Third, experiments on long-term dialogue QA with face and palmprint benchmarks show that this design preserves strong Owner–Non-owner separation while retaining the low-token advantage of KV-cache memory.

## 2 Related Work

## 2.1 KV Cache Systems

Transformer inference stores key and value tensors for previous tokens so that later decoding steps do not recompute all past states. Modern inference systems exploit this property through paging, prefix caching, and cache compression to reduce prefill cost and latency [4, 14]. Recent systems extend this line to agentic and multi-agent workloads, where one deployment may need to keep many partially active agents alive at the same time. In such settings, the focus is on avoiding fragmentation, keeping GPU memory utilized, and moving KV states without repeated recomputation.

Mooncake pools CPU, DRAM, SSD, and network resources into a KV-cache-centric disaggregated architecture for large-scale serving [5]. TokenCake studies space contention and time underutilization in multi-agent applications, combining dynamic memory partitioning with event-driven offload and predictive upload [6]. KVFlow replaces LRUstyle eviction with workflow-aware prefix caching guided by an Agent Step Graph and steps-to-execution estimates [8]. DualPath observes that agentic inference can become storage-I/O bound and introduces dual-path KV loading to use both prefill-side and decode-side bandwidth [7]. Taken together, these systems establish that KV cache is becoming a first-class systems resource for agent workloads.

However, they still optimize throughput, memory utilization, and data movement. They do not ask who should be allowed to activate a reusable KV block once that block exists in a shared memory pool. Our setting starts exactly from this missing question: when reusable KV states persist across users, efficient cache management alone is not enough, because the system also needs an identity-aware access rule before reuse happens.

## 2.2 KV Reuse and Memory

Another line of work asks how KV states can be reused across contexts or across agents instead of being managed request by request. KVCOMM aligns overlapping contexts under different prefixes through an anchor pool that estimates cache offsets online, enabling cross-context KV reuse in multi-agent workflows [9]. TokenDance exploits the All-Gather communication pattern in synchronized multi-agent rounds and performs collective KV sharing with diff-aware storage [10]. For scenario-specific deployments, KEEP redesigns KV memory management for embodied planning, where memory updates are frequent and structured [2], while Agent Memory Below the Prompt persists quantized KV caches to disk for edge multi-agent inference [3]. These works show that KV states can be treated as reusable assets across steps, contexts, or agents.

MemArt takes the next step from serving optimization to agent memory itself by treating reusable KV blocks as the external memory substrate of long-term agents [1]. This shift is especially relevant to our paper, because once KV cache becomes the memory substrate of a long-term assistant, reuse also determines what personal history can be surfaced in later interactions. Our work builds directly on this paradigm, but changes the central question from how to reuse more KV blocks to which blocks should be legally accessible in a shared multi-user memory pool. In other words, existing KV-memory work explains how to preserve and retrieve useful states, whereas our contribution adds an authorization layer that decides whether those states should be exposed to the current requester at all.

## 2.3 Biometric Access Control

Personalized memory becomes risky when multiple users share one device or active session. Logical identifiers such as account names and conversation IDs can be shared, stale, or spoofed, while the active physical user may change across turns. Biometrics offer complementary identity evidence because face and palmprint traits are tied to the person making the request. This makes them attractive when the system must distinguish the current physical user rather than only the nominal account holder.

Face recognition methods such as ArcFace learn discriminative embeddings for verification [11], while palmprint recognition captures line, texture, and structural cues that are useful for contact or contactless matching [12, 13]. Most biometric work, however, stops at recognition or verification itself. It does not address how biometric evidence should interact with reusable internal memory states inside an LLM agent. Bio-MemArt applies biometric verification to KV-cache memory, where the main challenge is not recognition alone, but preserving the retrieval and reuse properties of shared KV blocks after identity gating is introduced. Our paper differs from both classic biometric verification and prior KV-memory systems by using biometrics as the front-end filter that controls access to reusable KV memory.

## 3 Method

## 3.1 Problem Setting

We consider a shared LLM agent with a memory pool containing records from multiple users. For each user u, the agent stores historical interactions as memory blocks. At query time, the system receives a natural-language query x and a biometric probe embedding $b _ { q }$ from the current physical user, and must retrieve useful memory only when that memory belongs to the same user as $b _ { q }$ . We evaluate two conditions: in Owner, $b _ { q }$ matches the biometric template associated with the target memory; in Non-owner, it comes from a different user and should not unlock that memory. With this setting in place, we next define how biometric identity is attached to KV memory.

## 3.2 Biometric-Aware KV Memory

Bio-MemArt preserves the original MemArt memory representation and augments each KV block with a biometric template [1]. For the i-th memory block, we store

$$
m _ { i } = \{ K _ { i } , V _ { i } , c _ { i } , t _ { i } , a _ { i } , b _ { i } \} ,\tag{1}
$$

where $K _ { i }$ and $V _ { i }$ are the key and value tensors of the historical context block, $c _ { i }$ is the compressed representative key used by MemArt for efficient indexing, $t _ { i }$ is a timestamp, $a _ { i }$ denotes auxiliary metadata, and $b _ { i }$ is the biometric template embedding of the user who owns the block. The biometric template can be produced by a face encoder or a palmprint encoder. All biometric embeddings are L2-normalized, so matching is performed with cosine similarity:

$$
\begin{array} { r } { s ( b _ { q } , b _ { i } ) = b _ { q } ^ { \top } b _ { i } . } \end{array}\tag{2}
$$

All KV blocks from the same user share the same stored biometric template in our evaluation protocol. This keeps the identity field lightweight and avoids modifying the KV tensor structure.

## 3.3 Threshold Estimation

Bio-MemArt estimates a benchmark-specific biometric threshold from verification data rather than choosing one manually. For each biometric benchmark, we compute cosine similarities for matched pairs and mismatched pairs. Let $P ^ { + }$ denote the positive similarity distribution and $P ^ { - }$ denote the negative similarity distribution. The threshold τ is selected at the crossing region between the two distributions:

$$
\tau = \arg \operatorname* { m i n } _ { z } \left| \hat { p } ^ { + } ( z ) - \hat { p } ^ { - } ( z ) \right| ,\tag{3}
$$

where $\hat { p } ^ { + }$ and $\hat { p } ^ { - }$ are estimated density curves. The resulting τ is fixed for all downstream memory retrieval trials on the same biometric benchmark. This avoids tuning the threshold on the dialogue QA task and separates biometric verification from memory evaluation.

![](images/b2e8846f33870ea44bad7f28e2f65866e6b6298bdf16561a7b3acb38f03e5772.jpg)  
Figure 1: Overall pipeline of Bio-MemArt. (a) Biometric identity verification matches the current user’s biometric feature against enrolled identities and activates only the corresponding owner memory pool; unmatched requests receive no authorized memory. (b) Personalized KV-memory retrieval is then performed inside the authorized candidate pool: the request is encoded into query KV states, compressed keys are used for top-K block selection, selected memories are merged under a unified position layout, and the decoder reuses the retrieved KV cache to generate the final answer.

## 3.4 Identity-First KV Retrieval

Figure 1 shows the overall Bio-MemArt pipeline. Retrieval proceeds in two stages. First, biometric filtering constructs a query-specific authorized candidate pool:

$$
\mathcal { M } _ { b i o } ( b _ { q } ) = \{ m _ { i } \in \mathcal { M } \mid s ( b _ { q } , b _ { i } ) \geq \tau \} .\tag{4}
$$

If the probe belongs to the memory owner, the owner’s blocks remain available. If the probe belongs to a different user, the target blocks are removed, preventing their use during generation.

Second, Bio-MemArt applies the native MemArt retrieval function within $\mathcal { M } _ { b i o }$ . Inside this filtered pool, we keep the original MemArt design unchanged, including compressed-key indexing, multi-token aggregation retrieval, and decoupled position encoding for safe KV reuse [1]. Biometric filtering only restricts which blocks are eligible to enter the candidate set; it does not redefine the internal retrieval score or the original KV-reuse equations of MemArt. No model weights, attention layers, or decoding rules are changed.

## 3.5 Why KV Cache as Memory?

Bio-MemArt adopts KV cache memory because the reusable unit is already the model’s hidden state rather than a re-materialized prompt segment [4, 14, 1, 2, 3]. This lets the system enforce authorization before large-scale cache reuse happens. We therefore evaluate not only owner/non-owner separation, but also whether biometric gating preserves the efficiency benefits of KV memory. This choice also keeps the comparison aligned with the actual deployment question in this paper. If authorization were added only after converting memory back into text, the system would no longer test whether access control can coexist with native KV reuse. Using KV memory directly allows us to evaluate both properties at once: whether unauthorized memory can be blocked, and whether the model still benefits from low-prefill hidden-state reuse when access is granted.

## 4 Experiments

## 4.1 Datasets

Conversational memory benchmarks. We use LoCoMo as the primary long-term dialogue QA benchmark. LoCoMo contains long conversations spanning many sessions and multiple question types that test different forms of memory use [26]. We focus on memory-intensive question types: Multi-Hop, Temporal, and Single-Hop. These categories directly test whether the agent can retrieve personal facts, combine information across memory entries, and reason about event order [26]. They therefore provide a controlled downstream setting for measuring whether biometric gating changes memory access without changing the QA task itself.

Table 1: Face recognition evaluation datasets.
<table><tr><td>Dataset</td><td>Type / Group</td><td>#Subjects</td><td>#Images</td></tr><tr><td>AgeDB-30 [15]</td><td>Age</td><td>568</td><td>16,488</td></tr><tr><td>CALFW [16]</td><td>Age</td><td>5,749</td><td>12,174</td></tr><tr><td>CFP-FF [17]</td><td>Frontal</td><td>500</td><td>7,000</td></tr><tr><td>CFP-FP [17]</td><td>Pose</td><td>500</td><td>7,000</td></tr><tr><td>CPLFW [18]</td><td>Pose</td><td>5,749</td><td>11,652</td></tr><tr><td>LFW [19]</td><td>Frontal</td><td>5,749</td><td>13,233</td></tr><tr><td>VGG2-FP [20]</td><td>Pose</td><td>9,131</td><td>~3.31M</td></tr></table>

Table 2: Palmprint recognition evaluation datasets.
<table><tr><td>Dataset</td><td>Type / Group</td><td>#Subjects</td><td>#Images</td></tr><tr><td>CasiaM_460 [21]</td><td>460nm</td><td>200</td><td>1,200</td></tr><tr><td>CasiaM_700 [21]</td><td>700nm</td><td>200</td><td>1,200</td></tr><tr><td>CasiaM_850 [21]</td><td>850nm</td><td>200</td><td>1,200</td></tr><tr><td>IITD [22]</td><td>Contactless</td><td>460</td><td>2,300</td></tr><tr><td>MS_Blue [23]</td><td>Blue</td><td>500</td><td>6,000</td></tr><tr><td>MS_Green [23]</td><td>Green</td><td>500</td><td>6,000</td></tr><tr><td>MS_NIR [23]</td><td>NIR</td><td>500</td><td>6,000</td></tr><tr><td>MS_Red [23]</td><td>Red</td><td>500</td><td>6,000</td></tr><tr><td>PolyU [24]</td><td>Contact</td><td>378</td><td>7,560</td></tr><tr><td>Tongji [25]</td><td>Contactless</td><td>600</td><td>12,000</td></tr></table>

Face benchmarks. Face-based identity evaluation uses seven public verification benchmarks covering age, frontal-view, and pose variation. AgeDB-30 and CALFW emphasize age changes; CFP-FF and LFW are largely frontal; CFP-FP, CPLFW, and VGG2-FP are more pose-challenging. A pre-trained face encoder extracts L2-normalized embeddings, and each benchmark has its own operating threshold. Table 1 summarizes the face datasets, their variation type, and their scale. Together, these benchmarks let us test whether owner/non-owner separation remains stable under differen age and pose conditions.

Palmprint benchmarks. Palmprint evaluation uses ten protocols from five public datasets. CasiaM and MS provide wavelength-specific settings; IITD and Tongji are contactless; PolyU is contact-based. Table 2 summarizes the palmprint protocols and their acquisition settings. Together with face recognition, these protocols test whether the biometric gate generalizes across complementary physiological traits rather than overfitting to one biometric modality.

## 4.2 Evaluation Protocol

The core evaluation principle is to keep the downstream QA task fixed while changing only the biometric operating condition. For each benchmark, we build the shared memory pool from the same set of LoCoMo users and ask the same questions under both Owner and Non-owner settings. As a result, changes in F1 or BLEU do not come from different dialogue content; they come from differences in which KV memory blocks remain accessible after biometric filtering.

This protocol explains why some datasets can produce similar QA rows. When two biometric benchmarks induce the same effective owner acceptance and non-owner rejection pattern on the same LoCoMo split, they can produce similar downstream answers. Each reported owner/non-owner comparison uses the same stored dialogue history, target user, and question set; only the biometric probe changes.

## 4.3 Baselines

We compare against two reference systems. Full-context inference serves as the upper-cost reference. Native KV memory uses MemArt without biometric filtering and is the primary empirical baseline for downstream QA and runtime comparisons [1]. For the same shared memory pool, Owner and Non-owner trials use identical dialogue questions and storage contents; only the biometric probe changes. This comparison isolates three questions: whether biometric gating prevents unauthorized retrieval, whether it preserves native KV-memory answer quality, and whether it retains the efficiency advantage of KV reuse over replaying full context.

## 4.4 Evaluation Metrics

We evaluate three aspects. Memory accuracy is measured using unigram F1 and BLEU-1. Identity isolation is measured by biometric success rate and by the Owner–Non-owner gap in downstream QA:

$$
\Delta _ { \mathrm { F } 1 } = \mathrm { F } 1 _ { O u n e r } - \mathrm { F } 1 _ { N o n - o w n e r } , \quad \Delta _ { \mathrm { B L E U } } = \mathrm { B L E U } _ { O u n e r } - \mathrm { B L E U } _ { N o n - o w n e r } .\tag{5}
$$

A larger positive gap indicates that matched users can use personal memory while mismatched users cannot. Efficiency is measured by prefill tokens, retrieval time, biometric matching overhead, response time, and generated tokens, which are standard deployment-facing indicators for long-context and KV-cache systems [4, 14, 1]. The central efficiency claim is that Bio-MemArt should remain close to native MemArt while requiring far fewer prefill tokens than full-context prompting.

## 4.5 Implementation Details

All experiments adopt Qwen2.5-3B-Instruct unless specified otherwise. The face and palmprint pipelines rely on pre-extracted biometric embeddings. For every benchmark, we calibrate the threshold τ using positive and negative verification pairs and fix this value for all LoCoMo trials.

Our multi-user environment restricts each shared memory pool to at most five users. We perform one-to-one assignment between five biometric identities from the target benchmark and five LoCoMo users, where each user has one biometric template and one independent KV-memory partition. Under the Owner setting, the probe matches the target user’s registered identity; under Non-owner, the probe belongs to another enrolled identity and should be denied memory access. The downstream QA dataset is unified across all benchmarks, so performance gaps only arise from biometric decision thresholds and corresponding accept/reject behaviors.

For the lightweight efficiency comparison in Section 5.4, we sample five non-adversarial LoCoMo questions from ten distinct conversations to build 50 QA instances in total, with the generation length capped at 20 tokens. Unless otherwise noted, all storage, compression, retrieval and inference configurations of MemArt are kept identical to the original setup in [1]. The biometric pipelines only supply authorization embeddings, while the rest of the memory pipeline is shared. This keeps the evaluation focused on what changes when physical-user identity is inserted into KV-memory retrieval without rewriting the long-term memory mechanism.

## 5 Results and Analysis

We report paired owner/non-owner evaluation for face and palmprint benchmarks separately because the two modalities use different verification thresholds. Because all benchmarks share the same downstream LoCoMo protocol, QA tables should be read together with the dataset-specific authentication summary.

## 5.1 Reading Benchmark Differences

The most important point in this paper is that benchmark differences do not always appear first as large changes in downstream QA numbers. Because every benchmark is evaluated on the same LoCoMo conversations and the same question subsets, two datasets can produce nearly identical answer scores whenever they induce the same owner/nonowner authorization pattern. In that case, the biometric benchmark is still affecting the system, but its effect is expressed first through owner acceptance and non-owner rejection rather than through a different language task.

This is why the QA tables and the authentication summary must be interpreted together. When owner acceptance falls, the authorized user loses access to some relevant memory blocks and the owner QA score drops. When non-owner acceptance rises, the unauthorized user gains access to memory that should have been blocked and the owner/non-owner gap narrows. This joint reading explains why some datasets have similar QA rows while still reflecting different biometric difficulty.

## 5.2 Main Results on Owner versus Non-owner

The main tables report each dataset with paired Owner and Non-owner rows. All entries are F1 / BLEU-1 percentages on the shared LoCoMo Multi-Hop, Temporal, and Single-Hop subsets. The Average column uses the exported ALL split, and Avg. Gap denotes the corresponding Owner minus Non-owner difference. Because all benchmarks use the same downstream QA set, dataset-specific differences are explained by the authentication summary later in this section

Matched biometric probes consistently preserve better memory-grounded QA than mismatched probes across both modalities. Several datasets share nearly identical QA rows because they lead to the same accept/reject pattern

![](images/d67e44090564bfc640734d66efadb58842717da50d8159856bfd2d5607556ae0.jpg)  
Figure 2: Visualization of the response process under owner and non-owner queries. Given the same question, the owner branch passes identity authentication, retrieves only owner memories, and returns the correct personalized answer about the Boston restaurant. The non-owner branch fails identity matching, cannot access that user’s private memory pool, and returns a safe non-committal response instead of leaking personalized content.

Table 3: Downstream LoCoMo QA results under face-based biometric-aware KV retrieval.
<table><tr><td>Dataset</td><td>Condition</td><td>Multi-Hop</td><td>Temporal</td><td>Single-Hop</td><td>Average</td><td>Avg. Gap</td></tr><tr><td rowspan="2">AgeDB-30</td><td>Owner</td><td>17.05 / 15.55</td><td>20.55 / 17.14</td><td>26.00 / 16.01</td><td>22.86 / 17.12</td><td rowspan="2">16.66 / 12.28</td></tr><tr><td>Non-owner</td><td>5.55 / 4.55</td><td>5.37 / 4.63</td><td>7.47 / 5.43</td><td>6.20 / 4.84</td></tr><tr><td rowspan="2">CALFW</td><td>Owner</td><td>14.68 / 13.57</td><td>20.56 / 17.28</td><td>24.44 / 14.82</td><td>21.90 / 16.40</td><td rowspan="2">16.59 / 12.24</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">CFP-FF</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">17.34 / 12.86</td></tr><tr><td>Non-owner</td><td>5.55 / 4.55</td><td>5.37 / 4.63</td><td>7.47 / 5.43</td><td>6.20 / 4.84</td></tr><tr><td rowspan="2">CFP-FP</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">17.34 / 12.86</td></tr><tr><td>Non-owner</td><td>5.55 / 4.55</td><td>5.37 / 4.63</td><td>7.47 / 5.43</td><td>6.20 / 4.84</td></tr><tr><td rowspan="2">CPLFW</td><td>Owner</td><td>13.98 / 12.32</td><td>19.04 /16.15</td><td>21.09 / 12.78</td><td>19.81 / 14.92</td><td rowspan="2">14.50 / 10.76</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">LFW</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">18.23 / 13.54</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">VGG2-FP</td><td>Owner</td><td>15.36 / 13.83</td><td>21.94 / 18.44</td><td>23.88 / 14.79</td><td>22.37 / 16.93</td><td rowspan="2">17.06 / 12.77</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54/3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr></table>

Table 4: Downstream LoCoMo QA results under palmprint-based biometric-aware KV retrieval.
<table><tr><td>Dataset</td><td>Condition</td><td>Multi-Hop</td><td>Temporal</td><td>Single-Hop</td><td>Average</td><td>Avg. Gap</td></tr><tr><td rowspan="2">CasiaM_460</td><td>Owner</td><td>16.40 / 14.91</td><td>21.55 / 18.11</td><td>25.51 / 15.78</td><td>23.01 / 17.35</td><td rowspan="2">15.87 / 11.74</td></tr><tr><td>Non-owner</td><td>6.38 / 5.41</td><td>6.86 / 5.88</td><td>7.38 / 5.17</td><td>7.14/ 5.61</td></tr><tr><td rowspan="2">CasiaM_700</td><td>Owner</td><td>15.76 / 14.26</td><td>21.54 / 18.17</td><td>24.46 / 15.27</td><td>22.29 / 16.87</td><td rowspan="2">15.92 / 11.95</td></tr><tr><td>Non-owner</td><td>6.52 / 5.28</td><td>5.29 / 4.51</td><td>7.13 / 4.97</td><td>6.37 / 4.92</td></tr><tr><td rowspan="2">CasiaM_850</td><td>Owner</td><td>16.40 / 14.91</td><td>21.11 /17.82</td><td>25.27 / 15.72</td><td>22.52 / 17.00</td><td rowspan="2">16.58 / 12.38</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>5.42 / 4.60</td><td>6.34 / 4.49</td><td>5.94 / 4.62</td></tr><tr><td rowspan="2">IITD</td><td>Owner</td><td>16.94 / 15.49</td><td>21.54 /18.17</td><td>25.40 / 15.76</td><td>22.85 / 17.25</td><td rowspan="2">16.65 / 12.41</td></tr><tr><td>Non-owner</td><td>5.55 / 4.55</td><td>5.37 / 4.63</td><td>7.47 / 5.43</td><td>6.20 / 4.84</td></tr><tr><td rowspan="2">MS_Blue</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">18.23 / 13.54</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">MS_Green</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">18.23 / 13.54</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">MS_NIR</td><td>Owner</td><td>16.40 / 14.91</td><td>21.99 / 18.46</td><td>25.41 / 15.69</td><td>23.16 / 17.46</td><td rowspan="2">17.37 / 12.95</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.99 / 4.25</td><td>6.44 / 4.58</td><td>5.79 / 4.51</td></tr><tr><td rowspan="2">MS_Red</td><td>Owner</td><td>16.40 / 14.91</td><td>21.99 / 18.46</td><td>25.41 / 15.69</td><td>23.16 /17.46</td><td rowspan="2">17.85 / 13.30</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">PolyU</td><td>Owner</td><td>17.05 / 15.55</td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 / 17.70</td><td rowspan="2">18.23 / 13.54</td></tr><tr><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td></tr><tr><td rowspan="2">Tongji</td><td>Owner</td><td></td><td>21.99 / 18.46</td><td>26.12 / 16.05</td><td>23.54 /17.70</td><td rowspan="2">18.23 / 13.54</td></tr><tr><td></td><td>17.05 / 15.55</td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td>Non-owner</td><td>5.33 / 4.33</td><td>4.54 / 3.96</td><td>6.21 / 4.52</td><td>5.31 / 4.16</td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.3 Dataset-Specific Authentication Summary

Because biometric verification quality differs across benchmarks, Bio-MemArt uses a benchmark-specific operating threshold before downstream KV retrieval. Table 5 summarizes the threshold scan together with the owner and nonowner authentication success rates observed in the reported experiments. These statistics explain why some benchmarks yield stronger downstream isolation than others under the same retrieval pipeline.

The authentication summary makes the benchmark-level variation explicit. For face benchmarks, owner acceptance ranges from 86.00% on CPLFW to 100.00% on CFP-FF, CFP-FP, and LFW, while non-owner acceptance stays between 0.00% and 2.00%. For palmprint benchmarks, most protocols remain strong on both sides, but CasiaM\_460 and CasiaM\_700 admit the highest non-owner acceptance at 6.00%, which explains their weaker isolation than MS\_Blue, PolyU, or Tongji. This is why some datasets differ even when their QA tables look similar: the LoCoMo questions are fixed, so what changes across benchmarks is the biometric operating point and, consequently, which memories can be retrieved.

## 5.4 Runtime: Why Use KV Cache Memory?

Bio-MemArt inherits the KV-cache-centric memory substrate from MemArt [1]. We report a lightweight token-cost comparison between full-context prompting and KV-based memory in Table 6.

Table 6 reports all methods as averages over the same 50 questions. Full-context prompting requires 18,781.96 prefill tokens on average because every query reprocesses the full dialogue history. By contrast, MemArt reduces the average prefill cost to 35.42 tokens, and Bio-MemArt remains in the same low-token range at 28.57 tokens. The slightly lower token count of Bio-MemArt is consistent with biometric filtering removing non-matching memory blocks before KV retrieval, so the final reused KV set can be marginally smaller than in native MemArt. Response times also remain close, showing that biometric gating does not compromise the core efficiency advantage of KV-cache-centric memory. The runtime table shows that biometric verification changes which KV blocks can be reused without destroying the low-prefill operating regime of native KV memory.

Table 5: Dataset-specific biometric operating points and authentication outcomes used in the reported experiments. ‘Threshold’ is the refined operating point used before KV retrieval, while ‘Owner’ and ‘Non-owner’ report the corresponding authentication success rates in downstream evaluation.
<table><tr><td>Dataset</td><td>Threshold</td><td>Delta</td><td>Owner (%)</td><td>Non-owner (%)</td></tr><tr><td>AgeDB-30</td><td>0.2295</td><td>0.0605</td><td>98.00</td><td>2.00</td></tr><tr><td>CALFW</td><td>0.2143</td><td>0.0979</td><td>94.00</td><td>0.00</td></tr><tr><td>CFP-FF</td><td>0.4721</td><td>0.2052</td><td>100.00</td><td>2.00</td></tr><tr><td>CFP-FP</td><td>0.2634</td><td>0.1021</td><td>100.00</td><td>2.00</td></tr><tr><td>CPLFW</td><td>0.2472</td><td>0.1521</td><td>86.00</td><td>0.00</td></tr><tr><td>LFW</td><td>0.5591</td><td>0.3416</td><td>100.00</td><td>0.00</td></tr><tr><td>VGG2-FP</td><td>0.1711</td><td>0.0593</td><td>92.00</td><td>0.00</td></tr><tr><td>CasiaM_460</td><td>0.7520</td><td>-0.0081</td><td>96.00</td><td>6.00</td></tr><tr><td>CasiaM_700</td><td>0.8195</td><td>0.0026</td><td>94.00</td><td>6.00</td></tr><tr><td>CasiaM_850</td><td>0.8222</td><td>0.0102</td><td>94.00</td><td>4.00</td></tr><tr><td>IITD</td><td>0.5302</td><td>0.0342</td><td>96.00</td><td>2.00</td></tr><tr><td>MS_Blue</td><td>0.7402</td><td>-0.0135</td><td>100.00</td><td>0.00</td></tr><tr><td>MS_Green</td><td>0.7445</td><td>-0.0090</td><td>100.00</td><td>0.00</td></tr><tr><td>MS_NIR</td><td>0.8661</td><td>-0.0021</td><td>98.00</td><td>2.00</td></tr><tr><td>MS_Red</td><td>0.8700</td><td>0.0236</td><td>98.00</td><td>0.00</td></tr><tr><td>PolyU</td><td>0.5819</td><td>0.0172</td><td>100.00</td><td>0.00</td></tr><tr><td>Tongji</td><td>0.7105</td><td>0.0410</td><td>100.00</td><td>0.00</td></tr></table>

Table 6: Efficiency comparison averaged over the same 50 non-adversarial LoCoMo questions with the Qwen2.5-3B-Instruct backbone. Both MemArt and Bio-MemArt remain in the low-token regime, while full-context prompting repeatedly recomputes the entire dialogue history.
<table><tr><td>Method</td><td>Avg Prefill Tokens</td><td>Avg Response Time</td><td>Median Response Time</td></tr><tr><td>Full-context</td><td>18781.96</td><td>3.6881</td><td>3.8854</td></tr><tr><td>MemArt</td><td>35.42</td><td>2.2092</td><td>2.1679</td></tr><tr><td>Bio-MemArt</td><td>28.57</td><td>1.8743</td><td>1.8772</td></tr></table>

Figure 2 illustrates the final behavioral difference: matched probes recover the personalized restaurant memory, whereas mismatched probes give a generic, non-leaking answer.

## 5.5 Modality and Category Analysis

The category-level tables show that the isolation effect appears across all three question types. Across both modalities, stronger owner acceptance and lower non-owner acceptance translate into larger owner/non-owner QA gaps. CPLFW is the hardest face benchmark in our evaluation, while CFP-FF, CFP-FP, and LFW preserve the strongest owner access.

For palmprint, CasiaM\_460 and CasiaM\_700 admit more non-owner access than MS\_Blue, PolyU, or Tongji, so their isolation is weaker under the same downstream questions. Face benchmarks differ mainly in owner retention: stronger pose or age variation reduces the number of correctly admitted owner memories. Palmprint benchmarks differ more in false acceptance: protocols such as CasiaM\_460 and CasiaM\_700 admit more non-owner access, which narrows the owner/non-owner gap.

Across categories, the weakest non-owner performance appears on questions that depend on personally grounded event sequences, showing that the gate suppresses coherent personalized memory traces rather than only a few isolated facts. Single-Hop questions rely on direct access to user facts, Multi-Hop questions require combining information across more than one memory item, and Temporal questions depend on ordered event traces. When non-owner access is denied, all three categories degrade, but the drop is especially visible on questions that require a coherent user-specific memory chain rather than one isolated detail. That behavior matches the mechanism of Bio-MemArt: once the owner’s

KV blocks are removed from the authorized candidate pool, the model loses access not only to specific facts, but also to the hidden-state context needed to reconstruct a personalized sequence of events.

## 5.6 Practical Interpretation

Bio-MemArt changes which memory blocks are eligible before native KV retrieval begins. The owner/non-owner gap should therefore be read as an authorization effect on memory eligibility rather than as a general change in language-model behavior. Strict owner access preserves high-quality authorized KV entries and strong downstream QA, whereas higher non-owner acceptance weakens separation by admitting memory that should have been denied.

Identical QA rows do not imply equivalent biometric benchmarks. Because the LoCoMo task is fixed, what varies across benchmarks is the biometric operating point that determines which hidden-state memories survive the gate. This is why the authentication summary is needed alongside the QA tables.

The efficiency result follows the same logic. Bio-MemArt preserves the original KV reuse pipeline and adds only a selective authorization filter, so it retains low token overhead without falling back to prompt replay.

## 6 Conclusion

We presented Bio-MemArt, a biometric-aware KV-cache memory framework for multi-user LLM agents. Its identityaware retrieval method attaches a biometric template to each stored KV block, filters the shared memory pool using the current user’s biometric probe, and then applies native MemArt retrieval and KV reuse within the authorized candidate pool. This design addresses the core limitation of shared agent memory: semantically relevant memory is not necessarily authorized memory. The experiments show three consistent outcomes: authorized users retain substantially better QA performance than unauthorized users, different biometric benchmarks induce different operating points and acceptance rates, and the added identity gate still preserves the low-token operating regime of KV-cache memory.

## References

[1] Yuan Zeng, Pengfei Zuo, Min Lyu, Xingkun Yang, Huatao Wu, Yinlong Xu, and Zhou Yu. KVCache-Centric Memory for LLM Agents, 2026.

[2] Zebin Yang, Tong Xie, Baotong Lu, Shaoshan Liu, Bo Yu, and Meng Li. KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning, 2026.

[3] Yakov Pyotr Shkolnikov. Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices. arXiv preprint arXiv:2603.04428, 2026.

[4] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23, pages 611–626, New York, NY, USA, 2023. Association for Computing Machinery.

[5] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. Mooncake: Trading More Storage for Less Computation—A KVCache-Centric Architecture for Serving LLM Chatbot. In 23rd USENIX Conference on File and Storage Technologies (FAST 25), pages 155–170, 2025.

[6] Zhuohang Bian, Feiyang Wu, Zhuoran Li, Teng Ma, and Youwei Zhuo. TokenCake: A KV-Cache-Centric Serving Framework for LLM-Based Multi-Agent Applications. arXiv preprint arXiv:2510.18586, 2025.

[7] Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, and Panpan Huang. DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference. arXiv preprint arXiv:2602.21548, 2026.

[8] Zaifeng Pan, Ajjkumar Dahyalal Patel, Yipeng Shen, Zhengding Hu, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, and Yufei Ding. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. In Advances in Neural Information Processing Systems, volume 38, pages 126246–126265, 2026.

[9] Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, and Yiran Chen. KVCOMM: Online Cross-Context KV-Cache Communication for Efficient LLM-Based Multi-Agent Systems. In Advances in Neural Information Processing Systems, volume 38, pages 17882–17928. Curran Associates, Inc., 2025.

[10] Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, and Youwei Zhuo. TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing. arXiv preprint arXiv:2604.03143, 2026.

[11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4690–4699, 2019.

[12] Angelo Genovese, Vincenzo Piuri, Konstantinos N. Plataniotis, and Fabio Scotti. PalmNet: Gabor-PCA Convolutional Networks for Touchless Palmprint Recognition. IEEE Transactions on Information Forensics and Security, 14(12):3160–3174, 2019.

[13] Changlu Gao, Zhixin Yang, Wei Jia, Lu Leng, Bob Zhang, and Andrew Beng Jin Teoh. Deep Learning in Palmprint Recognition: A Comprehensive Survey. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 56(3):2143–2162, 2026.

[14] Suyu Ge et al. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. In International Conference on Learning Representations, 2024.

[15] Stylianos Moschoglou, Athanasios Papaioannou, Christos Sagonas, Jiankang Deng, Irene Kotsia, and Stefanos Zafeiriou. AgeDB: The First Manually Collected, In-the-Wild Age Database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017.

[16] Tianyue Zheng, Weihong Deng, and Jiani Hu. Cross-Age LFW: A Database for Studying Cross-Age Face Recognition in Unconstrained Environments. arXiv preprint arXiv:1708.08197, 2017.

[17] Soumyadip Sengupta, Jun-Cheng Chen, Carlos Castillo, Vishal M. Patel, Rama Chellappa, and David W. Jacobs. Frontal to Profile Face Verification in the Wild. In 2016 IEEE Winter Conference on Applications ofComputer Vision, pages 1–9, 2016.

[18] Tianyue Zheng and Weihong Deng. Cross-Pose LFW: A Database for Studying Cross-Pose Face Recognition in Unconstrained Environments. Technical report, Beijing University of Posts and Telecommunications, 2018.

[19] Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. In Workshop on Faces in ’Real-Life’ Images: Detection, Alignment, and Recognition, 2008.

[20] Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, and Andrew Zisserman. VGGFace2: A Dataset for Recognising Faces Across Pose and Age. In 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), pages 67–74, 2018.

[21] Zhenan Sun, Tieniu Tan, Yunhong Wang, and Stan Z. Li. Ordinal Palmprint Representation for Personal Identification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 1, pages 279–284, 2005.

[22] Ajay Kumar. Incorporating Cohort Information for Reliable Palmprint Authentication. In 2008 Sixth Indian Conference on Computer Vision, Graphics and Image Processing, 2008.

[23] David Zhang, Zhenhua Guo, Guangming Lu, Lei Zhang, and Wangmeng Zuo. An Online System of Multispectral Palmprint Verification. IEEE Transactions on Instrumentation and Measurement, 59(2):480–490, 2009.

[24] David Zhang, Wai-Kin Kong, Jane You, and Michael Wong. Online Palmprint Identification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(9):1041–1050, 2003.

[25] Lei Zhang, Li Li, An Yang, Ying Shen, and Meng Yang. Towards Contactless Palmprint Recognition: A Novel Device, a New Benchmark, and a Collaborative Representation Based Identification Approach. Pattern Recognition, 69:199–212, 2017.

[26] Adyasha Maharana et al. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2024.