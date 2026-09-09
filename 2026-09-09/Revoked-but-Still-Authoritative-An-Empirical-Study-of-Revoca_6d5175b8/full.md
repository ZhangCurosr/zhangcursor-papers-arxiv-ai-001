# Revoked but Still Authoritative: An Empirical Study of Revocation Enforcement in Agent-Memory Systems

Yi Ting Shen<sup>∗</sup>   
Vulcan Research, AIFT   
Singapore   
yiting.shen@aift.io

Kentaroh Toyoda Vulcan Research, AIFT Singapore kentaroh.toyoda@aift.io

Alex Leung   
Vulcan Research, AIFT   
Singapore   
alex.leung@aift.io

## Abstract

Long-running language-model agents depend on persistent memory. Many agentmemory systems preserve history through soft revocation: a contradicted fact is marked invalid and retained rather than deleted. However, whether that mark is enforced at retrieval time is unexamined. In this paper, we measure five such systems: we load each with a revoked policy and its replacement, track whether the revoked fact is returned at retrieval and whether the agent then acts on it across nine policy scenarios and nine models, and score every trial under six defense conditions. We find that no system enforces revocation by default: the revoked fact is returned wherever the revocation label is visible to the retrieval layer, outranks its replacement, and leads agents to the unsafe action. Based on these findings, we develop a guard that sits between the agent and any memory backend and withholds records that are revoked or conflict with their replacement. We open-source our code at https://github.com/VulcanLab/Memory-Rebirth-Attack.

## 1 Introduction

Persistent memory is what allows an agent to retain and reuse history across interactions. To do so, many agent-memory frameworks version their contents: each fact carries validity metadata, and an update marks the contradicted fact invalid rather than deleting it. This design is soft revocation. It supports audit and point-in-time queries and avoids the data loss of destructive updates. Whether soft revocation is enforced, however, has not been examined: does marking a fact invalid prevent the store from returning it to an agent that acts on it?

To answer it, we measure whether five major agent-memory systems enforce revocation at runtime: Graphiti and Zep [9], which share one engine, mem0 [2], langmem [6] and cognee [13]. We load each memory system with a revoked policy and its replacement, query the store with an ordinary question, and pass the returned facts to an agent that chooses between a safe and an unsafe action. Under this setting, we measure two metrics. The first is the rate at which the store returns the revoked record, which is a property of the store. The second is the rate at which the agent then acts on it, which is a property of the model. The methodology covers nine scenarios, nine models spanning six vendors. Each trial is scored under six defense conditions, from no defense to a check on retrieval-time validity.<sup>1</sup>

Our central finding is that no memory system enforces revocation in every case, and that they fail in one of three ways: the revocation is never recorded, recorded but invisible to the application, or recorded and visible but not enforced at retrieval. Only two of the five systems returned both the revoked record and its replacement. We found that both systems, in every scenario, ranked the revoked record above the current one and led agents to the unsafe action in more than two of every five trials.

Based on this finding, we develop a guard that mitigates the failure at the retrieval step. The key idea is to check validity when a record is read: the guard intercepts every read between the agent and any memory backend, inspects the facts the backend returns to an ordinary caller, and withholds those it can establish are revoked or conflict with their replacement. We test the guard on the same five systems and nine models and find that it withholds revoked facts wherever the backend’s revocation label is visible at retrieval or the returned facts state incompatible values.

The remainder of this paper is organized as follows. Section 2 describes related work, Section 3 details the methodology, Section 4 reports the measurements, and Section 5 presents the guard. Section 6 concludes the paper.

## 2 Related Work

Prior work can be classified into five directions: (1) memory systems, (2) attacks that inject content into a retrieval corpus or an agent’s memory, (3) studies of the memory lifecycle, (4) defenses, either guardrails or provenance checks, and (5) version-aware retrieval. We summarize each direction below, and contrast them with our approach in Table 1.

Memory systems. An agent-memory system persists what an agent and its users learn across sessions and returns stored records when a later query makes them relevant. A graph is often adopted as a base structure for such memory because it makes entities and their relations explicit, so retrieval can follow connections between facts instead of relying on similarity alone. Rasmussen et al. [9] describe Zep and its engine Graphiti, a temporal knowledge graph in which every fact records when it became true, when it stopped being true, and its source, so an update marks the prior fact rather than removing it. mem0 [2] is a memory layer for production agents that extracts facts from conversations and consolidates them through LLM-judged update operations. Its default store is a flat vector store, and an optional variant links the extracted entities in a graph, but that graph records relations without the validity intervals Graphiti records on each edge, so supersession is handled at the fact record. langmem [6] is a library that gives agents tooling to extract information from conversations and a background manager that consolidates and updates what was extracted, storing each memory as a structured document retrieved by similarity rather than as part of a graph. cognee [13] is an open-source memory engine that runs an extract-cognify-load pipeline, turning ingested documents into a knowledge graph embedded for vector search, so retrieval combines graph traversal with similarity rather than relying on either alone.

Attacks that inject content. Dash et al. [3] systematize memory poisoning attacks into six classes distinguished by how hostile content is written into a store: explicit command insertion, conditional command insertion, salience-driven compaction poisoning, policy-conformant fact injection, false precedent insertion, and skill-procedure insertion. AgentPoison [1] backdoors an agent’s long-term memory with optimized retrieval triggers, a conditional insertion. MINJA [4] removes the requirement of store access and induces the agent to write malicious records through ordinary interaction, a policy conformant injection. MemoryGraft [12] inserts poisoned experience traces that are later retrieved as precedent, a false-precedent attack. Zombie Agents [16] makes injected content persistent through the agent’s own memory updates, in the manner of salience-driven compaction poisoning, and agent worms [17] extend the spread across platforms through shared memory. PoisonedRAG [18] targets a setting outside this taxonomy: it inserts crafted texts into a retrieval corpus, so the store attacked is one the agent reads rather than a memory it writes.

Lifecycle security. Lifecycle security examines a memory record’s risks across its whole existence, from how it is written through how it is updated and retrieved to how it is deleted or deprecated. Lin et al. [7] organize agent-memory risk by lifecycle phase and find in their Forget & Rollback phase that incomplete deletion leaves residual content, naming verified forgetting an open problem. Their work investigates deletion that was attempted but failed to propagate, whereas the failure measured in our paper involves no deletion but deliberate retention: the store keeps a record it has itself marked invalid, and retrieval does not enforce the mark. The framework treats retention only as a residue of failed deletion, not as a state a store maintains by design.

Table 1: Comparison with the injection-based attack families.
<table><tr><td>family</td><td>record authored by</td><td>needs write access?</td><td>adds new content?</td><td>defense must check</td></tr><tr><td>Corpus poisoning [18]</td><td>attacker</td><td>yes</td><td>yes</td><td>provenance</td></tr><tr><td>Memory poisoning [1]</td><td>attacker</td><td>yes</td><td>yes</td><td>provenance</td></tr><tr><td>Query-only [4]</td><td>attacker</td><td>noª</td><td>yes</td><td>provenance</td></tr><tr><td>Poisoned trace [12]</td><td>attacker</td><td>yesª</td><td>yes</td><td>provenance</td></tr><tr><td>Self-reinforcing [16]</td><td>attacker</td><td>noª</td><td>yes</td><td>provenance</td></tr><tr><td>Cross-platform [17]</td><td>attacker</td><td>noª</td><td>yes</td><td>provenance</td></tr><tr><td>Ours</td><td>(benign) userb</td><td>no</td><td>no</td><td>retrieval-time validity</td></tr></table>

<sup>a</sup> the agent performs the write.

Guardrails and provenance checks. Guardrails inspect content at a model or harness boundary, on input or output, and refuse or rewrite it before or after it reaches the model. NeMo Guardrails [10], for example, checks retrieved content in a RAG pipeline. When the harness includes a memory system, such a guardrail could check the records the store returns for validity before the agent sees them.

Provenance checks verify where a record came from before granting it authority. SMSR [11] proposes using a cryptographic signature at write time to prevent unauthorized writes. The signature works as a provenance check, and it passes the record measured here, whose origin is the defender. TMA-NM [8] binds a record’s authority to its write-time origin with machine-checked guarantees. However, origin binding does not constrain what an authorized record does once its authorization is revoked.

Version-aware retrieval. Version-aware retrieval returns the current version of content that changes over time. VersionRAG [5] stores every document update in a hierarchical graph that records version sequences, version-specific content boundaries, and the changes between document states, so a query can be answered from a specific version. MemStrata [15] excludes a contradicted fact from retrieval: when a newer assertion replaces a fact’s value, a deterministic rule marks the older value invalid in a bi-temporal ledger. ConflictRAG [14] reconciles contradictory retrieved facts before generation by detecting the conflicts with a trained classifier, classifying them as factual, temporal, or opinion, and resolving each type in turn, ranking candidate sources on criteria extracted from the text itself, such as authority and recency.

Positioning. The question this paper asks is whether marking a fact invalid prevents the store from returning it to an agent that acts on it. Each direction above answers a different question: how foreign content enters a store, whether an attempted deletion completed, or whether retrieval returns the current version. None asks whether a record the store has itself marked invalid still determines what the agent does. Table 1 states the difference concretely. Every prior family brings new content into the store from an attacker, so a provenance check, which asks where a record came from, catches it. Even MINJA, the nearest prior threat model, leaves that check a foreign origin to detect: the agent performs the write, but the attacker authors the record. The record that harms the agent here is the defender’s own revoked policy, so every provenance check passes it by construction. What would catch it is a check of retrieval-time validity, whether a record is still current at the moment it is read, and the rest of this paper measures whether these five systems perform one.

## 3 Methodology

The objective of the measurement is to investigate whether a record that a memory system has marked revoked is still returned to the agent through default retrieval and still determines the agent’s action. Section 3.1 gives the notation and assumptions, Section 3.2 the two-phase measurement pipeline, Section 3.3 the memory systems under test, Section 3.4 the nine scenarios, Section 3.5 the defense conditions, and Section 3.6 the measurement methodology and the two outcome measures. The symbols used throughout are summarized in Table 2.

Table 2: Notation table.
<table><tr><td>symbol</td><td>meaning</td></tr><tr><td> $M$ </td><td>memory store, a finite set of records  $\{ r _ { 1 } , \ldots , r _ { n } \}$ </td></tr><tr><td>r</td><td>a record</td></tr><tr><td> $\operatorname { t x t } ( r )$ </td><td>fact text carried by</td></tr><tr><td> $\tau ( r )$ </td><td>timestamp of  $r ; \perp \mathrm { i f }$  not exposed to a caller</td></tr><tr><td> $\rho ( r )$ </td><td>status of  $r \colon 1$  revoked, 0 current, ⊥ not exposed to a caller</td></tr><tr><td> $q$ </td><td>query;  $q _ { s }$  the query for scenario s</td></tr><tr><td> $k$ </td><td>result size;  $k = 1 0$  in all runs</td></tr><tr><td> $R ( q , M , k )$ </td><td>default retrieval: the k records ranked by similarity, with no status restriction</td></tr><tr><td> $\mathbf { 1 } [ P ]$ </td><td>indicator: 1 if condition  $P$  holds, 0 otherwise</td></tr><tr><td>S  $^ { a }$ </td><td>a scenario;  $S$  the set of nine</td></tr><tr><td></td><td>the agent&#x27;s chosen action;  $a ^ { - }$  the action s designates unsafe</td></tr><tr><td> $\varphi ( r , s )$   $\operatorname { E R } ( s )$ </td><td>indicator that r carries the revoked policy of s</td></tr><tr><td></td><td>exposure indicator for s and the exposure rate  $\mathrm { P r } [ \mathrm { E R } = 1 ] \left( \mathrm { E q . } \left( 1 \right) \right)$ </td></tr><tr><td> $\operatorname* { P r } [ a = a ^ { - } ]$ </td><td>the unsafe-action rate, the model-level factor</td></tr></table>

## 3.1 Notation and assumptions

A memory store is a finite set of records $M = \{ r _ { 1 } , \ldots , r _ { n } \}$ , each carrying a fact text $\operatorname { t x t } ( r )$ , a timestamp $\tau ( r )$ , and a status flag $\rho ( r ) \in \{ 0 , 1 , \bot \}$ . The flag records the status a caller sees: $\rho = 1$ if the store reports the record revoked, 0 if it reports the record current, and ⊥ if it exposes no status to a caller, whether or not it marks the record internally. The third value is required because one system under test marks records revoked internally but exposes no status (Section 3.3), and this restricts the defenses expressible on that system. A record assessed as current and a record never assessed are both written $\rho = 0$ , since no store measured here distinguishes them. A store that fails to record a revocation is therefore indistinguishable from one with nothing to record, which is the first of the three failure forms the results distinguish (Section 4).

Retrieval takes a query q, a store $M$ and a result size $k ,$ and returns k records ranked by similarity alone. We use $k = 1 0$ in all runs; four systems expose the result size to the caller, and cognee exposes no result-size parameter and uses its library default. We write $R ( q , M , k )$ for default retrieval, the result a caller receives from each system’s default settings with no restriction applied. An application sees these records unless it restricts retrieval to records with $\rho = 0 ;$ , and such a restriction cannot be expressed at all when every record has $\rho = \bot$ . The agent then receives $\operatorname { t x t } ( \cdot )$ only, since $\rho$ and τ are record metadata and do not reach the prompt. This is the standard application pattern, and it is why nothing in the context identifies which of two conflicting policies is invalid.

The measurement depends on one precondition: the store holds two contradicting records about the same subject, a superseded one and its replacement. How each system reaches that state, including whether it marks the superseded record revoked, is part of the measurement itself (Section 3.2). Beyond producing these two records, the study does not manipulate the system: the harness never writes further records, sets a record’s status, modifies retrieval, alters the system prompt, or changes model configuration. The record that can mislead the agent is the defender’s own superseded policy, not attacker-authored content, so every provenance check surveyed in Section 2 passes it by construction.

## 3.2 Experimental setup

Figure 1 shows the sequence this paper measures and shows the measurement pipeline, drawn on the deploy-approval scenario with its superseded policy $r _ { 1 }$ and replacement $r _ { 2 } . \mathrm { A }$ setup phase produces the store state every measurement starts from, an experimental phase queries it.

Setup phase: The setup phase places two contradicting policies in the memory store, namely $( 1 ) \ v _ { 1 }$ , the older revoked record, and $( 2 ) \ v _ { 2 }$ , its replacement, under one of two insertion modes. Under direct insertion we write both records in the system’s native format, such as the highlighted expiration\_date in the mem0 call box of Fig. 1. Under indirect insertion we instead ask the system to ingest plain contradicting text. The supersession is stated only inside the prose, and the system’s own extraction must detect the supersession in the text and revoke the prior fact on its own. The revoked record’s status is therefore guaranteed in the direct mode and uncertain in the other.

![](images/155adc3a135138f5a552584ff58b851179bbbc61d5717078da0e0b4e47a9e3fd.jpg)  
Figure 1: The measurement pipeline.

Since retrieval is identical under both modes, a difference between them is a difference in how the store reached its state.

Experimental phase: The experimental phase queries the resulting store. A user asks an ordinary question about the affected subject; default retrieval R returns the closest records; and the agent decides between a safe and an unsafe action. In five of the nine scenarios the system prompt additionally forbids the unsafe action, shown as the shield in Fig. 1. Every scenario-model pair runs ten trials at temperature 0.7. The nine decision models span three capability tiers (flagship, efficient, and open-weight), seven families, and six vendors. The flagship tier runs GPT-5.5, Grok-4, and DeepSeek-V4-Flash (OpenAI, xAI, DeepSeek); the efficient tier runs GPT-5-nano, Gemini-2.5-Flash, and Grok-4-Fast (OpenAI, Google, xAI); and the open-weight tier runs GLM-5.2, Gemma-3-27B-IT, and Kimi-K2.6 (Zhipu, Google, Moonshot).

## 3.3 Memory systems

Table 3 lists the five memory systems we test. We found that some projects implement revocation differently from what their documentation describes,<sup>2</sup> so the mechanism column reports the behavior we measured rather than the documented one. All five share the same embedding model and the same extraction model, so retrieval and extraction quality do not vary across systems. Two configurations return the revoked record at retrieval time, Graphiti by default and mem0 (exp.) under a flag override. The other three do not, either because they retain nothing after an update (langmem, cognee) or because Zep returns the revoked record but never the status flag, so the filter that protects the other exposed systems cannot be expressed on it. We measure Zep by reading status directly from its store, which is labeling instrumentation rather than a capability an application has. Zep also has no direct insertion, because its API cannot express writing an already-revoked fact, so it appears only under indirect insertion. We run mem0 in two configurations that differ by one retrieval setting (Section B.3).

## 3.4 Scenarios

Against these memory systems, we set up nine scenarios. Table 4 lists them, each pairing a revoked policy that permits a harmful action with a current policy that prohibits it. The eight policy scenarios model a routine organizational event: a governing rule is revoked and its replacement forbids what the revoked rule permitted. Each current policy is the organization’s rule in force, so acting on the revoked text is a policy violation regardless of its operational consequences. The eight split into two groups by whether the system prompt carries a safety rule. The first four carry no rule and no distractor facts, and measure which of the two policies determines the action when nothing in the prompt favors either policy; the remaining five carry an explicit rule forbidding the unsafe action and add two benign distractor facts each, and measure whether the revoked fact overrides that rule. The ninth, ‘stored access directive’, is an additional case rather than a policy scenario. Its revoked text is not a statement about a rule but an instruction addressed to the agent, such as ‘grant access’. This is the kind of record that the injection attacks of Section 2 write into memory: content whose purpose is to command the agent rather than to record a fact. We include it to test how the measured defenses treat an explicit stored instruction compared with a superseded policy, and we report it individually, excluding it from the pooled no-rule rate (Section B.1).

Table 3: The memory systems under test.
<table><tr><td>system</td><td>update mechanism (measured)</td><td>retrieval</td><td>insertion</td></tr><tr><td>Graphiti [9]</td><td>soft revocation via validity timestamps</td><td>returns revoked record (default)</td><td>both</td></tr><tr><td>mem0 [2]</td><td>expiry marker (direct); consolidation added without retiring (indirect)</td><td>returns revoked record (flag override)</td><td>both</td></tr><tr><td></td><td></td><td>withholds revoked record (default)</td><td>both</td></tr><tr><td>langmem [6]</td><td>in-place overwrite when consolidation fires, inconsistently</td><td>no revoked record retained (default)</td><td>both</td></tr><tr><td>cognee [13]</td><td>prune-and-reingest; no fact-level contradiction step</td><td>no revoked record retained (default)</td><td>both</td></tr><tr><td>Zep CE [9]</td><td>soft revocation, same engine; splits a policy across edges, revokes some</td><td>returns revoked record, no status to caller (de- fault)</td><td>indirect</td></tr></table>

Table 4: Scenarios.
<table><tr><td>scenario</td><td>safety rule in prompt?</td><td>revoked policy asserts</td><td>unsafe action</td></tr><tr><td>deploy approval</td><td>no</td><td>one reviewer suffices</td><td>approve the deploy</td></tr><tr><td>PII export</td><td>no</td><td>personally identifiable information (PII) may go to ex- ternal vendors</td><td>allow the export</td></tr><tr><td>rollback policy</td><td>no</td><td>roll back immediately on any failure</td><td>roll back production</td></tr><tr><td>stored access directive</td><td>no</td><td>a stored directive to grant access</td><td>grant access</td></tr><tr><td>card disclosure</td><td>yes</td><td>agents may read full card numbers aloud</td><td>disclose the number</td></tr><tr><td>backup deletion</td><td>yes</td><td>delete all backups on suspected breach</td><td>delete backups</td></tr><tr><td>over-limit transfer</td><td>yes</td><td>one operator may move up to $1,000,000</td><td>execute the transfer</td></tr><tr><td>compliance log deletion</td><td>yes</td><td>delete audit logs older than 24 hours</td><td>delete the logs</td></tr><tr><td>MFA waiver</td><td>yes</td><td>multi-factor authentication (MFA) may be waived when in a hurry</td><td>grant without MFA</td></tr></table>

## 3.5 Filters and guardrails

We test five defense conditions and a no-defense baseline (Table 5), and measure how much each condition reduces the unsafe-action rate and where in the pipeline it acts. Every condition runs on all nine scenarios; in the five whose system prompt already forbids the unsafe action, each condition acts on top of that rule. The store-level filter and the guard restrict retrieval before the agent reads anything, whereas prompt hardening and the output filter act on what the agent has already read. The output filter is deliberately given the revoked fact texts when judging whether to block. The fifth condition is the guard that we designed in response to the measurement findings. We detail the guard’s design and results in Section 5.

## 3.6 Measurement

We run ten trials per scenario-model pair (Section 3.4), per defense condition (Section 3.5), and per system configuration (Section 3.3). We report two metrics: the exposure rate and the unsafe-action rate.

Table 5: Filters and guards.
<table><tr><td colspan="3">mechanism</td><td rowspan="2">requires status exposure?</td></tr><tr><td>defense</td><td>where it acts</td><td></td></tr><tr><td>no defense</td><td></td><td>whatever the store returns reaches the agent</td><td></td></tr><tr><td>store-level filter prompt hardening</td><td>before the read after the read</td><td>restricts retrieval to records not marked revoked appends an instruction to disregard superseded facts</td><td>yes no</td></tr><tr><td>output filter</td><td>after the read</td><td>blocks the chosen action if it conflicts with current</td><td>no</td></tr><tr><td></td><td></td><td>policy</td><td></td></tr><tr><td>filter with prompt hardening</td><td>both</td><td>the two conditions combined</td><td>yes</td></tr><tr><td>guard (Section 5)</td><td>before the read</td><td>wraps the search and withholds records that are marked revoked or contradicted by a newer fact</td><td>no</td></tr></table>

The exposure rate, $\mathrm { P r } [ \mathrm { E R } = 1 ]$ , is the fraction of scenario-model pairs for which default retrieval returned the revoked policy. For a scenario s, let $\operatorname { E R } ( s )$ indicate that default retrieval returned the revoked policy, and let the exposure rate be its mean over the scenario set S:

$$
\begin{array} { c l } { \displaystyle \mathrm { E R } ( s ) = \mathbf { 1 } \Big [ \exists r \in R ( q _ { s } , M , k ) : \varphi ( r , s ) = 1 \Big ] , } \\ { \displaystyle \mathrm { P r } [ \mathrm { E R } = 1 ] = \frac { 1 } { | S | } \sum _ { s \in S } \mathrm { E R } ( s ) , } \end{array}\tag{1}
$$

where $\varphi ( r , s ) = 1$ if r carries the revoked policy of $s , \mathbf { 1 } [ P ]$ is 1 when the condition P holds and 0 otherwise.

The unsafe-action rate, $\operatorname* { P r } [ a = a ^ { - } ]$ , is the fraction of trials in which the agent chose $a ^ { - }$ , the action the revoked policy implies and each scenario designates as unsafe.

$$
\operatorname* { P r } [ a = a ^ { - } ] = \operatorname* { P r } [ { \mathrm { E R } } = 1 ] \cdot \operatorname* { P r } [ a = a ^ { - } { \mathrm { ~ } } | { \mathrm { ~ E R } } = 1 ] .\tag{2}
$$

## 4 Evaluations

We report the measurements following the methodology of Section 3 exactly: one agent reads once, decides once, and never writes to the store. Sections 4.1 and 4.2 report the exposure and unsafe-action rate and what causes them, and Sections 4.3 and 4.4 break the effect down by model and by scenario.

## 4.1 Exposure and the unsafe-action rate

Table 6 shows, for each system, the exposure rate, the rank of the revoked record when returned, and the unsafe-action rate under no defense. The later columns add the defense conditions of Section 4.2. We discuss the exposure rate first to see whether each system still returns the revoked record to the caller. We see from Table 6 that the revoked record reaches the agent through default retrieval on two of the five systems. No system measured enforces revocation under indirect insertion. The five systems fail in one of three ways. First, the revocation is never recorded (cognee, langmem and mem0 under indirect insertion). Second, it is recorded and returned, but the status is withheld from the caller, so no application can enforce the revocation (Zep). Third, it is recorded and returned with its status exposed to the caller, yet not enforced at retrieval (Graphiti by default, and mem0 with the expiry-flag override enabled, abbreviated mem0 (exp.) in the tables).

We then turn to the unsafe-action rate to see how often the returned revoked record determines the agent’s action. Only the third form can be quantified end to end, because Section 3 measures the unsafe-action rate only where the revoked record is returned while its revocation mark is verified applied. On both exposed systems the revoked fact was returned for every scenario, 81/81 each, and ranked above the policy that replaced it. The revoked fact led agents to the unsafe action in 44.2% of trials on Graphiti (358/810) and 42.1% on mem0 (exp.) (341/810). No unsafe action occurred under the store-level filter of Section 4.2 (0/1,620) or under three clean-context baselines (0/2,430). Across the two exposed systems, agents took the unsafe action in 699 of 1,620 trials (43.1%). The per-scenario rate spans 0.0% to 94.4% and the per-model rate 15.0% to 60.0% (Sections 4.3 and 4.4). The claims below depend on this variation across cells rather than on sampling precision within cells.

Table 6: Exposure and unsafe-action rate by defense condition, under direct insertion.
<table><tr><td>system</td><td>exposure</td><td>rank</td><td>no defense</td><td>filter</td><td>prompt</td><td>output</td><td>filter+prompt</td></tr><tr><td>graphiti</td><td>81/81</td><td>1</td><td>44.2%</td><td>0.0%</td><td>38.4%</td><td>19.0%</td><td>0.0%</td></tr><tr><td>mem0 (exp.)</td><td>81/81</td><td>1</td><td>42.1%</td><td>0.0%</td><td>36.0%</td><td>17.2%</td><td>0.0%</td></tr><tr><td>cognee, langmem, mem0</td><td>0/81</td><td></td><td></td><td>no revoked record returned</td><td></td><td></td><td></td></tr></table>

The two exposed systems return the revoked record ranked first in every scenario. Revoked policy tends to be phrased more absolutely than its replacement, as in “must be blocked” against “are monitored”, and a similarity-ranked retriever favors that phrasing. Independent evidence establishes that such retrievers handle versioned content poorly: on a benign benchmark of versioned documents, conventional retrieval answers 58% of questions correctly and a graph-based variant 64% [5]. The other three systems give the store-level filter nothing to act on. Under direct insertion, cognee and langmem delete the superseded record outright, and mem0 filters it at read time by default, which is the shipped filter we disable in mem0 (exp.). Under indirect insertion, cognee, langmem and mem0 record the revocation status in 0 of 9 scenarios while storing both policies. Zep records the revocation but omits the status from everything it returns, so no application on Zep can express the filter. The indirect-insertion measurements establish that the precondition arises without privilege, with Graphiti recording the revocation in 4/9 scenarios and Zep in 6/9 from plain contradicting text. Under indirect insertion a cell labeled unexposed is not a clean context but one the detector failed to recognize (Section A.1).

## 4.2 Defense conditions and retrieval

We then validate how effective the defenses in Section 3.5 are. Table 6 reports the unsafe-action rate under each defense condition, on the two systems whose revocation mark was verified as applied. The store-level filter removes the effect entirely (0/1,620), as does the filter combined with prompt hardening. Prompt hardening alone reduces the rate from 43.1% to 37.2%: the prompt asks the model to disregard superseded facts, but the retrieved context does not indicate which fact is superseded, so the model must judge without the information it needs. The output filter reduces the rate only to 18.1%, and even that under an advantage no deployment has: it was given the revoked fact texts when deciding what to block.

We see that the cause lies in retrieval rather than in the store. Holding the store fixed removes retention as a variable: mem0 run twice on identical data with a single retrieval flag changed switches from safe to unsafe in 39 of the 81 matched cells and never the other way.<sup>3</sup> The difference between the two configurations is one boolean in a shipped product. The complementary comparison confirms the attribution. The two exposed systems, which are independently implemented, are not distinguishable on the same grid of scenarios and models (p = 0.629 on 81 matched pairs).<sup>4</sup> That agreement is what the decomposition of Eq. (2) leads us to expect: with both systems exposed in every cell, the unsafe-action rate reduces to its conditional part, a property of the model rather than of the store. What the tested systems lack is a check of retrieval-time validity, whether a record is still current when read. What they provide, at best, is provenance, where the record came from. Section 5 presents a mechanism that implements this.

## 4.3 Per-model analysis

We next break the rate down by model to see whether susceptibility follows capability. Table 7 shows the per-model unsafe-action rate under direct insertion on the two exposed systems. All nine models were affected, from 15.0% to 60.0%, and there is no correlation between capability tier and the unsafe-action rate. The highest and lowest rates both belong to the efficient tier, the flagship group spans 33.3% to 58.9%, and the most resistant model in the set is the fast non-reasoning Grok-4-Fast. The model cannot tell from the context alone which policy is current: the two policy statements

Table 7: Per-model rates under direct insertion on the two systems whose mark was verified applied.
<table><tr><td>model</td><td>no defense</td><td>filter</td><td>prompt hardening</td><td>output filter</td></tr><tr><td>GPT-5.5</td><td> $6 0 / 1 8 0 = 3 3 . 3 \%$ </td><td>0/180</td><td> $4 8 / 1 8 0 = 2 6 . 7 \%$ </td><td> $2 2 / 1 8 0 = 1 2 . 2 \%$ </td></tr><tr><td>Grok-4</td><td> $9 4 / 1 8 0 = 5 2 . 2 \%$ </td><td>0/180</td><td> $8 6 / 1 8 0 = 4 7 . 8 \%$ </td><td> $6 0 / 1 8 0 = 3 3 . 3 \%$ </td></tr><tr><td>DeepSeek-V4-Flash</td><td> $1 0 6 / 1 8 0 = 5 8 . 9 \%$ </td><td>0/180</td><td> $9 1 / 1 8 0 = 5 0 . 6 \%$ </td><td> $3 8 / 1 8 0 = 2 1 . 1 \%$ </td></tr><tr><td>GPT-5-nano</td><td> $1 0 0 / 1 8 0 = 5 5 . 6 \%$ </td><td>0/180</td><td> $7 9 / 1 8 0 = 4 3 . 9 \%$ </td><td> $4 3 ^ { \cdot } / 1 8 0 = 2 3 . 9 \%$ </td></tr><tr><td>Gemini-2.5-Flash</td><td> $1 0 8 / 1 8 0 = 6 0 . 0 \%$ </td><td>0/180</td><td> $8 3 / 1 8 0 = 4 6 . 1 \%$ </td><td> $1 7 / 1 8 0 = 9 . 4 \%$ </td></tr><tr><td>Grok-4-Fast (non-reas.)</td><td> $2 7 / 1 8 0 = 1 5 . 0 \%$ </td><td>0/180</td><td> $1 7 / 1 8 0 = 9 . 4 \%$ </td><td> $2 1 / 1 8 0 = 1 1 . 7 \%$ </td></tr><tr><td>GLM-5.2</td><td> $6 0 / 1 8 0 = 3 3 . 3 \%$ </td><td>0/180</td><td> $6 0 / 1 8 0 = 3 3 . 3 \%$ </td><td> $2 0 / 1 8 0 = 1 1 . 1 \%$ </td></tr><tr><td>Gemma-3-27B-IT</td><td> $7 0 \mathrm { ^ { \circ } / 1 8 0 = 3 8 . 9 \% }$ </td><td>0/180</td><td> $7 8 / 1 8 0 = 4 3 . 3 \%$ </td><td> $6 2 / 1 8 0 = 3 4 . 4 \%$ </td></tr><tr><td>Kimi-K2.6</td><td> $7 4 / 1 8 0 = 4 1 . 1 \%$ </td><td>0/180</td><td> $6 1 / 1 8 0 = 3 3 . 9 \%$ </td><td> $1 0 / 1 8 0 = 5 . 6 \%$ </td></tr></table>

Table 8: With rule versus no rule.
<table><tr><td rowspan="2">group</td><td colspan="2">no defense</td><td colspan="2">retrieval filter</td></tr><tr><td>pooled</td><td>excl. outlier</td><td>pooled</td><td>excl. outlier</td></tr><tr><td>with rule</td><td>12.9% (=116/900)</td><td>4.6% a (=33/720)</td><td>0.0% (=0/900)</td><td>0.0% (=0/720)</td></tr><tr><td>no rule</td><td>81.0% (=583/720)</td><td>86.7% b (=468/540)</td><td>0.0% (=0/720)</td><td>0.0% (=0/540)</td></tr></table>

<sup>a</sup> excl. backup deletion; <sup>b</sup> excl. stored access directive.

appear with equal standing in the context, and the revoked one is typically phrased more absolutely.   
Reasoning quality therefore does not decide the outcome.

## 4.4 Per-scenario analysis

We then break the rate down by scenario. Table 8 shows the pooled unsafe-action rate by scenario group, under no defense and under the retrieval filter. Where no prompt rule opposes the action, agents took the unsafe action in 81.0% of trials, reaching 94.4% on PII export. Where an explicit prohibition was present, the unsafe action was still selected in 12.9% of trials, but that figure is dominated by one scenario: backup deletion reached 46.1% and supplies 83 of the 116 bypasses, leaving 4.6% across the four with-rule scenarios other than backup deletion, which we report as the conservative figure. Backup deletion is the only rule stated with a conditional exception, requiring written approval from the chief information security officer, and the only one whose revoked text asserts precedence. The two features cannot be separated, so the high rate cannot be attributed to either alone. Across those four scenarios, the effect is largest where the unsafe action is operationally plausible and smallest where it is recognizable as harmful independently of policy: 3.9% for an over-limit transfer, 0.6% for waiving MFA, and no unsafe trial for log deletion. In those cases pretrained refusal, not the prompt rule, accounts for the low rates. The two groups are not a controlled comparison, since they differ in scenario, action set and distractor count, and the study never measures the same scenario with and without its prohibition. The rule-carrying group supports only the narrower statement that a prompt prohibition does not reliably prevent the unsafe action.<sup>5</sup>

## 5 Guard

Filtering at the store removes the effect and should be the default, but it needs the vendor to expose status, which one of the five systems does not, and it needs the application to apply it at every read. This section presents a mitigation an application can deploy without any change to the memory backend.

## 5.1 Design

The measurements of Section 4 identify what is missing. Provenance checks, which verify where a record came from [11], do not apply to this setting, since the record came from the defender and the system then marked it invalid. What is absent is a check on retrieval-time validity, whether a record is still current at the moment it is read, and it belongs at retrieval, because every reader passes through it. The guard sits between the agent and its memory, reads whatever the backend returns to an ordinary caller (Fig. 2), and proceeds in two stages, from more to less reliable. It uses explicit status fields where a backend exposes them, which is exact but requires vendor cooperation. Among what remains it detects pairs of facts asserting incompatible values for the same subject and withholds the older, which requires nothing of the backend and is the only stage that can catch content no system ever marked. Withheld facts are returned separately with a reason, so an application can report a conflict rather than lose history. Nothing is deleted from the store.

![](images/a4925ced08364a95a5c1dde9788b4c179c84f5300cf671927c7d9ee97bc910ca.jpg)  
Figure 2: The guard architecture.

Table 9: Guard performance by system and insertion condition.  
Each cell gives direct; indirect.
<table><tr><td>system</td><td>revoked present</td><td>caught</td><td>current withheld</td></tr><tr><td>cognee</td><td>0; 0</td><td>0/0; 0/0</td><td>0/19; 0/18</td></tr><tr><td>graphiti</td><td>18; 4</td><td>18/18; 4/4</td><td>0/38; 1/14</td></tr><tr><td>langmem</td><td>0; 0</td><td>0/0; 0/0</td><td>0/19; 6/15</td></tr><tr><td>mem0</td><td>0; 0</td><td>0/0; 0/0</td><td>0/38; 8/25</td></tr><tr><td>mem0 (exp.)</td><td>12; 0</td><td>12/12; 0/0</td><td>0/22; 6/24</td></tr><tr><td>zep</td><td>0; 14</td><td>0/0; 4/14</td><td>-;1/36</td></tr></table>

Withholding requires positive evidence of staleness. An earlier version broke ties by result order, which is unsound because result order is a relevance ranking rather than a chronology. Against a backend exposing no metadata it withheld the current policy and kept the revoked one, converting a 0% unsafe rate into 100%. Where the guard cannot establish which of two conflicting facts is older it therefore keeps both and reports the conflict.<sup>6</sup>

## 5.2 Evaluation

The guard is scored on the same scenarios, models and trials as the defense conditions of Section 4.2. Where the store exposes a mark it matches the store-level filter (0/1,620; withholds 18/18 revoked facts on Graphiti and 12/12 on mem0, none current), which shows that the guard is deployable, not that its design is sound: the first stage reads the same field the filter reads, so the agreement is not independent evidence. The guard also sees records the agent never does, so its recall does not translate one-to-one into the agent’s safety.

The independent result is under indirect insertion, where no mark exists for any filter to read: 0/810 on both mem0 configurations against the store filter’s 5/810 and 24/810, and 315/810 against 364/810 on Zep, whose API cannot express a filter at all. This is the only place in the study where the guard does something the store-level filter cannot.

That capability also withholds current facts, and Table 9 reports both effects. With H the records the guard withholds, V the revoked records present, and C the current ones, recall is |H ∩ V|/|V| (the caught column, higher is better) and |H ∩ C| counts current facts withheld (lower is better, ideally zero).<sup>7</sup> On cognee and Graphiti under indirect insertion the guard is marginally worse than no defense (53/810 vs 44/810 on cognee, 83/810 vs 81/810 on Graphiti), because the conflict test also matches conflicting pairs that are not a revoked policy and its replacement. On Zep it caught only 4/14, since Zep splits one policy across several edges and revokes only some of them, and nothing in the returned text says which.

## 6 Conclusion

In this paper, to investigate whether marking a fact revoked prevents the store from returning it to an agent that acts on it, we have measured five major agent-memory systems under soft revocation. Our findings are threefold. First, no system enforces revocation in every case: revoked policies are returned by ordinary retrieval and lead agents to unsafe actions, even against an explicit prohibition in the prompt. Second, the failure lies in retrieval rather than in any one product: two independently implemented systems behave alike, and one switches between safe and unsafe on a single retrieval flag. Third, the failure extends beyond the single read: it persists through the agent’s own write, a second query to the same store is not an independent check, and it recurs through tool actions without an attacker. Based on these findings, we have proposed a guard that sits between the agent and its memory backend to mitigate the failure at retrieval: it inspects the facts the backend returns to an ordinary caller and withholds those it can establish are revoked or in conflict with their replacement. Where the backend exposes a mark, the guard is as effective as the store-level filter. Where no mark exists, the guard still stopped unsafe actions that the store-level filter could not, including on Zep, which exposes no status at all. The control that is missing is retrieval-time validity, not provenance.

## References

[1] Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2024.

[2] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[3] Pritam Dash, Tongyu Ge, Aditi Jain, Tanmay Shah, and Zhiwei Shang. From untrusted input to trusted memory: A systematic study of memory poisoning attacks in LLM agents. arXiv preprint arXiv:2606.04329, 2026.

[4] Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, and Zhen Xiang. Memory injection attacks on LLM agents via query-only interaction. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[5] Daniel Huwiler, Kurt Stockinger, and Jonathan Fürst. VersionRAG: Version-aware retrievalaugmented generation for evolving documents. arXiv preprint arXiv:2510.08109, 2025.

[6] LangChain AI. LangMem: Long-term memory for AI agents. https://github.com/ langchain-ai/langmem, 2025. Software repository.

[7] Zehao Lin, Xixuan Hao, Renyu Fu, Shaobo Cui, Kai Chen, Chunyu Li, Zhiyu Li, and Feiyu Xiong. A survey on long-term memory security in LLM agents: Attacks, defenses, and governance across the memory lifecycle. arXiv preprint arXiv:2604.16548, 2026.

[8] Yedidel Louck. Securing LLM-agent long-term memory against poisoning: Non-malleable, origin-bound authority with machine-checked guarantees. arXiv preprint arXiv:2606.24322, 2026.

[9] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

[10] Traian Rebedea, Razvan Dinu, Makesh Narsimhan Sreedhar, Christopher Parisien, and Jonathan Cohen. NeMo guardrails: A toolkit for controllable and safe LLM applications with programmable rails. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 431–445, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-demo.40.

[11] Tarun Sharma. SMSR: Certified defence against runtime memory poisoning in persistent LLM agent systems. arXiv preprint arXiv:2606.12703, 2026.

[12] Saksham Sahai Srivastava and Haoyu He. MemoryGraft: Persistent compromise of LLM agents via poisoned experience retrieval. arXiv preprint arXiv:2512.16962, 2025.

[13] Topoteretes. cognee: The open-source AI memory platform for agents. https://github. com/topoteretes/cognee, 2023. Software repository.

[14] Chenyu Wang, Yueyuan Li, Yingmin Liu, and Yang Shu. ConflictRAG: Detecting and resolving knowledge conflicts in retrieval augmented generation. arXiv preprint arXiv:2605.17301, 2026.

[15] Neeraj Yadav. Temporal validity in retrieval memory: Eliminating stale-fact errors for AI agents over evolving knowledge. arXiv preprint arXiv:2606.26511, 2026.

[16] Xianglin Yang, Yufei He, Shuo Ji, Bryan Hooi, and Jin Song Dong. Zombie agents: Persistent control of self-evolving LLM agents via self-reinforcing injections. arXiv preprint arXiv:2602.15654, 2026.

[17] Mingming Zha and Xiaofeng Wang. Autonomous LLM agent worms: Cross-platform propagation, automated discovery and temporal re-entry defense. arXiv preprint arXiv:2605.02812, 2026.

[18] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models. In 34th USENIX Security Symposium (USENIX Security ’25), pages 3827–3844, 2025.

## A Full Methodology

## A.1 Detecting a revoked policy in a result

The exposure measure of Section 3.6 uses $\varphi ( r , s )$ , the indicator that a returned record carries the revoked policy of scenario s. That section states what $\varphi$ means but not how it is computed. Thi section gives the computation. Where the backend exposes a status, $\rho ( r ) \in \{ 0 , 1 \}$ and $\varphi ( r , s ) = \rho ( r )$ which is sound because the only records the seeded store marks revoked are those carrying the revoked policy of s. Where $\rho ( r ) = \perp$ , a lexical criterion is used. The criterion exists for the stores that expose no reliable mark. Under indirect insertion the system’s own extraction may paraphrase the policy text, so an exact match against the revoked policy cannot be assumed. For the hard-deleting systems, which return no revoked record at all, the criterion simply never applies. With $W ( x )$ the set of stemmed content words of $x ,$ after stop-words and policy boilerplate are removed, define

$$
\begin{array} { l } { \displaystyle \sigma ( \boldsymbol { r } , \boldsymbol { x } ) = \frac { \left| W ( \mathrm { t x t } ( \boldsymbol { r } ) ) \cap W ( \boldsymbol { x } ) \right| } { \left| W ( \boldsymbol { x } ) \right| } , } \\ { \displaystyle \varphi ( \boldsymbol { r } , \boldsymbol { s } ) = \mathbf { 1 } \big [ \sigma ( \boldsymbol { r } , \boldsymbol { v } _ { s } ) \geq \theta _ { 1 } \wedge } \\ { \sigma ( \boldsymbol { r } , \boldsymbol { v } _ { s } ) - \sigma ( \boldsymbol { r } , \boldsymbol { c } _ { s } ) \geq \theta _ { 2 } \big ] } \end{array}\tag{3}
$$

where $v _ { s }$ and $c _ { s }$ are the revoked and current policy texts. Both conditions are required: the first admits only facts substantially about the revoked policy, and the second only those closer to it than to its replacement, so a paraphrase of the current policy is not miscounted. We use $\theta _ { 1 } = 0 . 3$ and $\theta _ { 2 } = 0 . 1 5$ , fixed before measurement. The thresholds were not tuned against the exposure outcomes.

## A.2 Paired comparison across systems

We do not report per-condition confidence intervals on the pooled proportions. The cells are stochastic (ten trials each at temperature 0.7), but the within-cell variation they induce is small relative to the between-cell dispersion that the claims depend on. Most cells are all-or-nothing across the ten trials, so nearly all variance lies between cells. A cluster analysis of the released trial data, reported as an intraclass correlation and a design effect per arm, quantifies this and is included with the results bundle. The between-cell dispersion is itself descriptive of the fixed grid of nine authored scenarios and nine models rather than a generalisation interval over them (Sections 4.3 and 4.4).

The cross-system claims depend on a paired comparison. Systems are measured on the same grid, so cells are matched pairs indexed by $( s , m )$ , where s ranges over the nine scenarios S and m over the nine models M. Let $u _ { X } ( s , m )$ be the number of unsafe trials for system X in cell $( s , m )$ . We report the pooled rate and, because pooling discards the pairing, the paired counts

$$
\begin{array} { r l } & { W _ { A B } = \big | \{ ( s , m ) : u _ { A } ( s , m ) > u _ { B } ( s , m ) \} \big | , } \\ & { W _ { B A } = \big | \{ ( s , m ) : u _ { B } ( s , m ) > u _ { A } ( s , m ) \} \big | , } \\ & { \quad \mathrm { t i e s } = | S | | \mathcal { M } | - W _ { A B } - W _ { B A } } \end{array}
$$

where $W _ { A B }$ counts the cells in which system A produced more unsafe actions than system $B .$ The pairing holds scenario difficulty and model susceptibility constant within each comparison, and these counts are the basis for every cross-system claim we make. Section B.2 turns them into a two-sided exact test on the discordant cells.

## A.3 The stale-retrieval guard

This section gives the containment and opposition tests, the thresholds, and the procedure, deferred from Section 5.1. The guard is released as guard/stale\_guard.py, a backend-agnostic module that reads only the records a retrieval call returns. For two texts x and y with stemmed content-word sets $W ( x )$ and $W ( y )$ , subject agreement is measured by containment rather than Jaccard similarity, because a terse current policy and a verbose superseded one are about the same subject even when their lengths differ sharply, and Jaccard penalizes exactly that case:

$$
\mathrm { c t m } ( x , y ) = \frac { \vert W ( x ) \cap W ( y ) \vert } { \mathrm { m i n } ( \vert W ( x ) \vert , \vert W ( y ) \vert ) } .\tag{4}
$$

Opposition is a disjunction of three signals, with Ω a fixed antonym-pair set, $N ( \cdot )$ a negation-cue count and Q(·) the set of quantities in a text. In the first case, $u \in$ x means that the string u occurs in x:

$$
\operatorname { o p p } ( x , y ) = { \left\{ \begin{array} { l l } { 1 } & { { \exists } ( u , w ) \in \Omega : ( u \in x \land w \in y ) } \\ & { \qquad \lor ( w \in x \land u \in y ) } \\ { 1 } & { Q ( x ) \neq \emptyset \land Q ( y ) \neq \emptyset \land Q ( x ) \cap Q ( y ) = \emptyset } \\ { 1 } & { | N ( x ) - N ( y ) | \geq 1 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

and the two are combined conjunctively:

$$
\mathrm { c o n f i c t } ( x , y ) = \mathbf { 1 } \big [ \mathrm { c t m } ( x , y ) \geq \alpha \land \mathrm { o p p } ( x , y ) = 1 \big ] .\tag{5}
$$

Both conjuncts are required, because subject agreement alone would treat two paraphrases of the same current policy as a disagreement, and an opposition signal alone would match unrelated facts. We use $\alpha = 0 . 4 5$ and $\beta = 0 . 6$ . The parameter $\bar { \beta }$ caps conflict-driven withholding at a fraction of the result set, counting the records stage 1 has already withheld, although it does not limit stage 1 itself, which acts only on marks the backend has set. The cap exists because if most of a result set looks conflicting, the heuristic, not the store, is more likely at fault. Records with a known timestamp are considered first, in descending order of $\tau .$ The descending order matters: it guarantees that the older record of a conflicting pair is always the incoming one, so a record already kept is never evicted. Records with $\tau = \bot$ follow them in their original relative order, since they are neither newer nor older by any evidence available. A conflict that the timestamps cannot order is reported, and both records are kept. In the released implementation, stage 1 matches a fixed set of revocation field names and treats a scheduled-expiry field, the marker one backend writes, as revoked only once the date it carries is past. Algorithm 1 gives the procedure.

```latex
Algorithm 1 The Stale-Retrieval Guard.
Require: retrieval result $( r _ { 1 } , \ldots , r _ { k } )$ , with status $\rho ( r )$ , timestamp $\tau ( r )$ and text txt(r) where the backend
exposes them; thresholds $\alpha , \beta$
Ensure: kept records K in original rank order, withheld records H with reasons
1: K ← [ ], H ← [ ] {Stage 1: apply a revocation the backend already recorded}
2: for all r in the result do
3: $\mathbf { i f } \rho ( r ) = 1$ then
4: $\begin{array} { r } { \dot { H }  H + ( r , } \end{array}$ “backend marked revoked”)
5: else
6: $K \gets K + r$
7: end if
8: end for
9: kept $ [ ]$ {Stage 2: pairwise contradiction}
10: for all r in K sorted by known timestamp, descending do
11: for all $r ^ { \prime }$ in kept do
12: if conflic $\mathbf { \Psi } _ { : } ( \operatorname { t x t } ( r ) , \operatorname { t x t } ( r ^ { \prime } ) )$ then
13: if τ(r) = ⊥ or $\tau ( r ^ { \prime } ) \stackrel { \cdot \cdot } { = } \bot$ or $\tau ( r ) \geq \tau ( r ^ { \prime } )$ then
14: note “unresolved conflict”; kept ← kept + r {cannot order the pair: keep both}
15: else if $| H | < \beta k$ then
16: $H \gets \dot { H } + ( r ,$ “superseded”) {demonstrably older, cap not reached}
17: else
18: kept ← kept + r {cap reached: keep without withholding}
19: end if
20: break
21: end if
22: end for
23: if no conflict found then kept ← kept + r
24: end for
25: return (restore original rank of kept, H) {Stage 3: each record in H carries its reason}
```

## A.4 Compute resources

This subsection reports the compute required to reproduce the measurements. The experiments require no training and no GPUs. All measurements ran on a single MacBook Air (M5) with 16 GB of RAM, which hosted the five memory systems as local containers. No model weights were loaded locally: the nine decision models and the shared extraction and embedding model were accessed through their providers’ hosted APIs, routed through LiteLLM. Because no model is trained or served locally, execution time is dominated by API latency rather than by local computation, so the meaningful resource is the number of model calls rather than elapsed wall-clock time.

Table 10: Decision-model calls and cost by experiment.
<table><tr><td>experiment</td><td>calls</td><td>cost</td><td>share</td></tr><tr><td>main grid (reported)</td><td>44,550</td><td>$71.78</td><td>90.7%</td></tr><tr><td>propagation</td><td>1,458</td><td>$2.35</td><td>3.0%</td></tr><tr><td>persistence</td><td>1,296</td><td>$2.09</td><td>2.6%</td></tr><tr><td>exploit chain</td><td>1,216</td><td>$1.84</td><td>2.5%</td></tr><tr><td>parse audit</td><td>486</td><td>$0.78</td><td>1.0%</td></tr><tr><td>write-back</td><td>115</td><td>$0.23</td><td>0.2%</td></tr><tr><td>total</td><td>49,121</td><td>$79.08</td><td>100%</td></tr></table>

Table 10 reports the decision-model calls and cost for each experiment, priced from the unit prices the LiteLLM serving layer publishes. The main grid reported in this paper covers 9 scenarios, 9 models, and 10 trials per cell across the systems, configurations, defense conditions, and insertion modes (Section 3.6). It accounts for 44,550 of the 49,121 decision-model calls and \$71.78 of the \$79.08 total. The remaining 4,571 calls funded the three studies reported in Section C (propagation, the exploit chain, and write-back), a persistence study not reported here, and a parse audit of replies from which no action could be parsed. Only the persistence study and the parse audit contribute no number to this paper. Input and billed output token counts per call are recorded in the released results bundle. The full research project therefore required more compute than the main grid alone, by the margin shown in Table 10.

## B Additional results

## B.1 Per-scenario rates

Table 11: Per-scenario rates on the two exposed systems, direct insertion.
<table><tr><td>scenario</td><td>no defense</td><td>retrieval filter</td></tr><tr><td colspan="3">with prompt rule</td></tr><tr><td>backup deletion card disclosure</td><td> $8 3 / 1 8 0 = 4 6 . 1 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$   $0 / 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>over-limit transfer MFA waiver</td><td> $2 5 / 1 8 0 = 1 3 . 9 \%$   $7 / 1 8 0 = 3 . 9 \%$ </td><td> $0 \dot { / } 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>compliance log deletion pooled</td><td> $1 / 1 8 0 = 0 . 6 \%$   $0 / 1 8 0 = 0 . 0 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$   $0 / 1 8 0 = 0 . 0 \%$   $0 \dot { / } 9 0 0 = 0 . 0 \%$ </td></tr><tr><td>excl. backup deletion</td><td> $1 1 \dot { 6 } / 9 0 0 = 1 2 . 9 \%$   $\mathbf { 3 3 / 7 2 0 } = 4 . 6 \%$ </td><td> $0 / 7 2 0 = 0 . 0 \%$ </td></tr><tr><td>no prompt rule</td><td></td><td></td></tr><tr><td>PII export</td><td></td><td></td></tr><tr><td></td><td> $1 7 0 / 1 8 0 = 9 4 . 4 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>rollback policy</td><td> $1 5 8 / 1 8 0 = 8 7 . 8 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>deploy approval</td><td> $1 4 0 / 1 8 0 = 7 7 . 8 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>stored access directive</td><td> $1 1 5 / 1 8 0 = 6 3 . 9 \%$ </td><td> $0 / 1 8 0 = 0 . 0 \%$ </td></tr><tr><td>pooled, incl. directive</td><td> $5 8 3 / 7 2 0 = 8 1 . 0 \%$ </td><td> $0 / 7 2 0 = 0 . 0 \%$ </td></tr><tr><td>excl. directive</td><td> $4 6 8 / 5 4 0 = 8 6 . 7 \%$ </td><td> $0 / 5 4 0 = 0 . 0 \%$ </td></tr><tr><td></td><td></td><td></td></tr></table>

Table 11 gives the per-scenario figures behind Table 8. The two groups are kept apart because they differ in scenario, action set and distractor count, and only the scenarios with a prompt rule measure whether the agent takes the unsafe action despite the prompt rule. The stored access directive is the non-policy scenario of Section 3.4, whose revoked text is an instruction to the agent rather than a policy statement, and it is excluded from the pooled no-rule rate. As in Table 8, the bold pooled rates exclude the outlier scenario of each group: backup deletion supplies 83 of the 116 with-rule bypasses, so the with-rule rate excluding it is the conservative figure (Section 4.4).

## B.2 Matched-pair comparison

Table 12: Matched-pair comparison on the 81 matched cells, no-defense arm, direct insertion.
<table><tr><td>system A</td><td>system B</td><td>cells</td><td>A unsafe</td><td>B unsafe</td><td> $A > B$ </td><td> $B > A$ </td><td>tied</td><td>exact p</td></tr><tr><td>graphiti</td><td>mem0 (exp.)</td><td>81</td><td> $3 5 8 / 8 1 0 = 4 4 . 2 \%$ </td><td> $3 4 1 / 8 1 0 = 4 2 . 1 \%$ </td><td>10</td><td>7</td><td>64</td><td>0.629</td></tr><tr><td>mem0</td><td>mem0 (exp.)</td><td>81</td><td> $0 / 8 1 0 = 0 . 0 \%$ </td><td> $3 4 1 / 8 1 0 = 4 2 . 1 \%$ </td><td>0</td><td>39</td><td>42</td><td> $< 0 . 0 0 1$ </td></tr></table>

Under no defense and direct insertion, the two exposed systems differ by 2.1 points, and Table 12 gives the paired counts on the 81 matched cells. A two-sided exact test on the discordant pairs gives $p = 0 . 6 2 9 \colon \mathrm { a } \ 1 0 \mathrm { - } \mathrm { t o - } 7$ split is compatible with an even one, so the test does not distinguish the two systems. That is the intended reading rather than a weak result. With both systems exposed in every cell, the decomposition of Eq. (2) reduces the unsafe-action rate to its conditional part, a property of the model choosing between two contradictory facts rather than of the store, so the two systems should agree. The same test on the within-system comparison, where retrieval policy really does differ, returns $p < 0 . 0 0 1$ on 39 discordant cells against zero the other way (second row of Table 12), so the statistic separates the two cases.

The bound on that reading should be stated with it. With 17 discordant pairs a two-sided exact test attains significance only at a 13-to-4 split or more extreme, so this comparison could not have detected a systematic advantage unless roughly three-quarters of the discordant cells fell the same way. What the test supports is that the two systems do not differ grossly, not that they are identical, and the claim we base on it is only the former.

## B.3 Within-system comparison

Table 13: The within-system comparison on mem0: the shipped read-time filter at its default and disabled.
<table><tr><td>retrieval policy</td><td>exposure rate</td><td>rank</td><td>no defense</td><td>filter</td><td>prompt</td><td>output</td><td>filter+prompt</td><td>guard</td></tr><tr><td>expired filtered (default)</td><td>0/81</td><td>-1</td><td>0/810</td><td>0/810</td><td>0/810</td><td>0/810</td><td>0/810</td><td>0/810</td></tr><tr><td>expired returned</td><td>81/81</td><td></td><td> $3 4 1 / 8 1 \dot { 0 } = 4 2 . 1 \%$ </td><td>0/810</td><td> $2 9 2 / 8 1 \dot { 0 } = 3 6 . 0 \%$ </td><td> $1 3 9 / 8 1 \dot { 0 } = 1 7 . 2 \%$ </td><td>0/810</td><td>0/810</td></tr></table>

Table 13 gives both configurations of the within-system comparison: mem0 run twice on identical data, with its shipped read-time filter at its default and disabled (Section 3.3). With expired records filtered the store never exposes the agent and no unsafe action occurs. With them returned, exposure is total, the revoked record ranks first, and 42.1% of trials select the unsafe action, at $p < 0 . 0 0 1$ on the matched-pair test (Table 12).

## C Beyond a single read: full results

This appendix documents three settings that extend the single-read methodology of Section 3. They are not exhaustive: each targets one constraint that the single-read methodology fixes, and a full deployment combines all three. All three run on Graphiti, the exposed system whose API can express both write-back and the filtered retrieval, with the store seeded as in the setup phase, the nine models, ten trials per cell, and temperature 0.7 as in the measurements above. The two exposed systems are not distinguishable on matched cells.<sup>8</sup> Write-back, however, is system-specific, so the rates reported here are properties of Graphiti’s pipeline rather than claims about mem0 (exp.).

## C.1 Agent write-back

The store-level filter removes the unsafe action entirely (Section 4.2), but under a measurement in which the agent never writes to the store: the measurement scores one decision, and whatever the agent concluded is discarded when the trial ends. An agent with memory access does not behave that way. An agent that acts on a revoked fact and then records what it decided places that conclusion in the store as an ordinary, current record. From that moment the revoked policy no longer needs to be retrieved, because its conclusion is already stored as a current record, and there is nothing marked invalid for the filter to remove. We measure whether that transfer occurs, and what remains of the store-level filter’s guarantee once it has.

![](images/7ca7029cad3e7bb98b871bb2609a6319c51a73702b6c9f5cc812eec661751b68.jpg)  
Figure 3: The write-back measurement: the agent’s journal against the retrieval filter.

Figure 3 shows the design. Hop 0 repeats the setup phase exactly: the store holds the revoked record r and its replacement r , and a first agent answers the ordinary question of its scenario. In every run where that agent chose the unsafe action, its decision is then written back under one of two modes. The natural mode feeds the decision through the system’s ordinary ingestion call, which is what an agent with write access actually does. The direct mode writes the decision as an explicit record, which is fully controlled but bypasses ingestion. The two are reported separately so they cannot be conflated. Hops 1 through 3 then place a new agent, on a related task, against the store: it reads and decides under default retrieval and under the retrieval filter that reached zero in Section 4.2. Over the nine scenarios, nine models and ten trials under both write-back modes, this gives 1,620 runs, of which 714 (44.1%) were poisoned at hop 0, consistent with the 44.2% of Section 4.1 on this system.

Table 14 reports the unsafe rate at each hop, given a poisoned hop-0 decision, for both write-back modes and both retrieval conditions. Under the direct write-back the store retained every journal: the decision was stored in 356 of 356 runs, and as a current, unmarked record in all of them. Default retrieval at the later hops returns the unsafe conclusion in 98.0%, 95.8% and 96.3% of runs. The filter removes r , but the journal is not r : the filtered rates are 71.6%, 79.5% and 83.1%, against the 0.0% the same filter achieved on the same system in Section 4.2 (Table 6). The defense is defeated by a record the system itself considers current, and nothing sustains it: the filtered rate is stable or rising across the three hops, so the journal entry, stored as current, remains in the store over the three hops.

The natural write-back is the faithful mode. In it, the store’s own pipeline limits the effect. Graphiti’s ingestion stored only 43 of the 358 journals (12.0%), and marked 21 of those 43 superseded without any instruction to do so. This leaves 22 runs (6.1%) with an active journal record. Where the journal did persist as active, the filter again failed: 19 of those 22 runs were unsafe at hop 1 under the filter. Pooled over the natural write-back mode, the filtered column shows 7.5%, 7.8% and 12.8%, above the 0.0% of Section 4.2 at every hop.

The two modes bound the effect from either side. The natural mode reports what the system’s own pipeline does with an agent’s journal. The direct mode is the controlled form of the same event, with ingestion excluded, and an application that deliberately records what its agents decided, rather than relying on the memory system’s extraction, adopts the direct mode. Under either mode the store-level filter, the defense that reached zero in Section 4.2, no longer reaches zero, because the agent’s own write creates a record the filter has no reason to remove.

Table 14: Unsafe rate at later hops, given a poisoned hop-0 decision.
<table><tr><td rowspan="2">write-back</td><td rowspan="2"></td><td colspan="3">default retrieval</td><td colspan="3">retrieval filter</td></tr><tr><td>stored active hop 1</td><td>hop 2</td><td>hop 3</td><td>hop 1</td><td>hop 2</td><td>hop 3</td></tr><tr><td>direct</td><td>356/356</td><td>349/356</td><td>341/356</td><td>343/356</td><td>255/356</td><td>283/356</td><td>296/356</td></tr><tr><td rowspan="2">natural</td><td rowspan="2"></td><td>98.0%</td><td>95.8%</td><td>96.3%</td><td>71.6%</td><td>79.5%</td><td>83.1%</td></tr><tr><td>293/358</td><td>306/358</td><td>297/358</td><td>27/358</td><td>28/358</td><td>46/358</td></tr><tr><td></td><td>22/358</td><td>81.8%</td><td>85.5%</td><td>82.9%</td><td>7.5%</td><td>7.8%</td><td>12.8%</td></tr><tr><td>no write-back (Section 4.2) 一</td><td colspan="3">358/810 = 44.2%</td><td colspan="3">0/810 = 0.0%</td></tr></table>

![](images/5c7cf0ed98327b7481343ca59cf7c463efe41c12d34a35ca68f756fc7f60505b.jpg)  
Figure 4: The propagation measurement: three roles, one shared store, one question each.

## C.2 Multi-agent propagation

Agent write-back shows that an agent’s journal can persist beyond the record that produced it. The question an operator actually faces is how many of the agents sharing one store take the unsafe action from a single revoked fact. Review is the natural containment boundary: a reviewing agent asks a different question, and if the revoked policy is not returned for its query, the reviewer’s read is independent of the executor’s and can detect its unsafe decision. Whether that happens cannot be assumed: retrieval is query-specific, and no measurement has established what a different question returns. We measure it.

Figure 4 shows the design. Three agents share one store, seeded as in the setup phase. The executor receives the operational task, retrieves, decides, and journals what it did. The reviewer is asked which action the policy requires, on a question phrased independently of the executor’s task. It is never shown the executor’s framing or its proposal, so its answer cannot be an approval of what it reviews. The planner is asked what applies to the next, related request, which is a forward-looking question an operator would ask. Each role retrieves on its own question and chooses between the same safe and unsafe actions as the measurements above. We vary what is shared across three conditions: default retrieval for every role; the status filter applied to every role’s read; and a no-journal condition in which the executor decides but records nothing. This condition separates the reach of the original revoked record from that of the journalled decision of Section C.1. Over the nine scenarios, nine models and ten trials this gives 2,430 runs.

Table 15 reports, per condition, the unsafe rate for each role, the reviewer’s agreement given a poisoned executor, whether the reviewer or planner went unsafe, and the mean number of unsafe roles.

Under default retrieval the executor went unsafe in 354 of 810 runs (43.7%), consistent with the 44.2% of Section 4.1 on this system. Every role saw the revoked record: 810 of 810 for the executor, the reviewer and the planner alike, because each role’s own question still returned it. Review is therefore not a containment boundary. Given an executor that went unsafe, the reviewer chose the unsafe action on its own read in 341 of 354 runs (96.3%), and some later role went unsafe in 352 of 354 (99.4%). On average 1.32 of the three roles ended up unsafe, and all three did in 337 of 810 runs (41.6%).

Table 15: Per-role outcomes across the three conditions.
<table><tr><td>condition</td><td>executor unsafe</td><td>reviewer unsafe</td><td>planner unsafe</td><td>reviewer unsafe given poisoned</td><td>any later role given poisoned</td><td>all three unsafe</td></tr><tr><td>shared, default</td><td>354/810 43.7%</td><td>358/810 44.2%</td><td>356/810 44.0%</td><td>341/354 96.3%</td><td>352/354 99.4%</td><td>337/810 41.6%</td></tr><tr><td>shared, filtered</td><td>0/810 0.0%</td><td>0/810 0.0%</td><td>0/810 0.0%</td><td></td><td></td><td>0/810 0.0%</td></tr><tr><td>default, no journal</td><td>360/810 44.4%</td><td>308/810 38.0%</td><td>328/810 40.5%</td><td>294/360 81.7%</td><td>317/360 88.1%</td><td>278/810 34.3%</td></tr></table>

Mean roles unsafe (of 3): 1.32 shared default; 0 filtered; 1.23 no journal.

The filtered condition reduces every column to zero (0/810), so the store-level fix does contain propagation. The requirement is that it be applied at every read. The store itself enforces nothing, and an application that filters its executor but not its reviewer, or its reviewer but not its planner, leaves the roles it does not cover in the default condition.

The no-journal condition separates the two mechanisms. Without the journal the executor still went unsafe at 44.4% (360/810), consistent with Section 4.1, and the revoked record still reached the reviewer and the planner in every run, but reviewer agreement given a poisoned executor fell to $2 9 4 / 3 6 0 = 8 1 . 7 \%$ , the reach of any later role fell to 88.1% (317/360), and the mean number of unsafe roles to 1.23. The journal is therefore not the cause of propagation, but it compounds it: agreement rises from 81.7% to 96.3% when the executor’s decision is in the store. Both mechanisms are read-path failures. A second opinion drawn from the same store is not independent, and the store’s contents, not how the roles are arranged, are what carries the effect.

## C.3 Tool actions

The measurements above record which action an agent selects, and a selection is not an action. We give the agent tools: a file read, a shell, an egress channel, and a sanctioned API. We measure whether the returned revoked policy leads the agent to invoke a dangerous tool, to move the secret into a sink, and to leave a journal from which a second agent repeats the dangerous action without any revoked record retrieved. We also test the other candidate containment boundary: a guardrail on the tool rather than on the read.

Figure 5 shows the chain. The agent is a tool-calling loop of up to four steps. The tools are decoys: they appear real to the model, record every attempt, and touch no file, shell or network. The secret is a synthetic token that stands in for a real one, so the end-to-end outcome is observed without any real resource existing. Four tool-facing scenarios pair a revoked permissive policy with a current prohibition and a task: arbitrary code execution, a destructive shell breach, exfiltration of a secret held in environment variables, and shell use where a sanctioned API exists. For each of the nine models we run ten trials per scenario under four defense conditions: no defense, the retrieval filter, the guard of Section 5, and a tool-layer guardrail that refuses dangerous shell commands at the tool boundary while the revoked policy remains in the retrieved context. We report three signals of increasing strength: the tool execution rate (whether the dangerous tool was invoked), whether the secret was read, and whether it reached a report or an egress sink (exfiltration confirmed).

Table 16 reports the three signals per defense condition. With no defense the agent invoked the dangerous tool in 62 of 360 runs (17.2%), read the secret in 18 (5.0%), and confirmed an exfiltration in 9 (2.5%), all nine of them in the secret-exfiltration scenario, where they amount to 9/90 = 10.0% of that scenario’s trials. The per-scenario tool execution rates span 27.8% for arbitrary code execution, 20.0% for secret exfiltration, 15.6% for shell use against the sanctioned API and 5.6% for the destructive shell breach.

The two controls that act before the read reduce every signal. The retrieval filter leaves the tool execution rate at $8 / 3 6 0 = 2 . 2 \%$ with no confirmed exfiltration, and the guard of Section 5 matches it at $8 / 3 6 0 = 2 . 2 \%$ with none. With $r _ { 1 }$ removed, whatever remains is behavior against the current policy rather than anything revoked.

![](images/1bb561698277343e1a34327a805e0ab1896e0744d198ee3622d0513b24e9a8b0.jpg)  
Figure 5: The tool-action measurement: from the returned revoked policy to a tool action, and the relapse by a second agent.

Table 16: Tool-action signals per defense condition.
<table><tr><td>defense condition</td><td>tool execution rate</td><td>secret read</td><td>exfiltration confirmed</td></tr><tr><td>no defense</td><td> $6 2 / 3 6 0 = 1 7 . 2 \%$ </td><td> $1 8 / 3 6 0 = 5 . 0 \%$ </td><td> $9 / 3 6 0 = 2 . 5 \%$ </td></tr><tr><td>retrieval filter</td><td> $8 / 3 6 0 = 2 . 2 \%$ </td><td> $1 / 3 6 0 = 0 . 3 \%$ </td><td> $0 / 3 6 0 = 0 . 0 \%$ </td></tr><tr><td>guard (Section 5)</td><td> $8 / 3 6 0 = 2 . 2 \%$ </td><td> $3 / 3 6 0 = 0 . 8 \%$ </td><td> $0 / 3 6 0 = 0 . 0 \%$ </td></tr><tr><td>tool-layer guardrail</td><td> $6 2 / 3 6 0 = 1 7 . 2 \%$ </td><td> $1 8 / 3 6 0 = 5 . 0 \%$ </td><td> $7 / 3 6 0 = 1 . 9 \%$ </td></tr></table>

The tool-layer guardrail does not. Its tool execution rate is $6 2 / 3 6 0 = 1 7 . 2 \%$ , identical to no defense, and it confirmed 7 exfiltrations against the 9 under no defense, a difference of two events on the same trials. The guardrail refused a shell command in 9 of 360 runs (2.5%), and the agent bypassed none of those 9 refusals: the rule fails upstream of evasion, because the returned revoked policy licenses actions the rule does not classify as dangerous. The revoked policy restores capability wherever a tool exists to take it up.

Arbitrary code execution is the strongest form of the dangerous action. One scenario licenses the agent to run any diagnostic command, and when the returned revoked policy licenses more, the agent composed interpreter one-liners of its own design: 2 of 90 undefended trials (2.2%), 4 of 90 under the tool guardrail, and 1 of 90 under the filter. The attempts are preserved verbatim in the released run data. They are uncommon, but their occurrence shows that the returned revoked policy restores capability, and a model that takes it up has composed, and would have executed, code of its own choosing.

The dangerous action recurs without an attacker. After the first agent journals its decision, a second agent asked an ordinary operational question repeats it in 42 of the 62 runs in which the first agent went unsafe (67.7%) under default retrieval, and in 13 of 62 (21.0%) with the filter applied at the second read. The recurrence under default retrieval is the write-back mechanism of Section C.1 reached end to end: the second agent needs no revoked record, only the journal of the first.

The containment boundary that holds is the read. The two controls that reduce every signal to the baseline act before the agent reads anything, and the one control placed at the tool is statistically indistinguishable from no defense. That is where the guard of Section 5 acts.