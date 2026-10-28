# Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

Saiful Khan <sup>1\*</sup> and Afrah Farea <sup>2</sup>

<sup>1\*</sup>Scientific Computing, Science and Technology Facilities Council, RAL, OX11 0QX, UK.   
<sup>2</sup>Department of Engineering Science, University of Oxford, Parks Road, OX1 3PJ, UK.

\*Corresponding author(s). E-mail(s): saiful.khan@stfc.ac.uk;

## Abstract

Large language model (LLM) agents that call enterprise APIs must select the correct endpoints, follow process order, use valid payload keys, and choose coded values from predefined lists. However, current retrieval-augmented service discovery treats endpoints independently and therefore misses process and schema constraints. We address this limitation by integrating Open Data Protocol (OData) metadata and Business Process Model and Notation (BPMN) reference processes in a typed property graph. A hybrid retriever identifies candidate endpoints using dense embeddings and follows process edges to retrieve subsequent endpoints. We evaluate the method on 42 enterprise services with process-chain ground truth. For queries with low lexical overlap between the query and downstream entities, it improves Recall@3 by 5.7 points (n = 74), with consistent gains across embedding models and cutofs. It matches dense retrieval on other queries and provides no improvement on OpenAPI services without a process layer, indicating that the gains arise from process structure. We also evaluate an LLM agent on 81 multi-step queries. It completes 38.7% of tasks, compared with 10.3% for a chunkbased baseline. In an elicitation probe, providing valid code lists eliminates invented codes but leaves invalid keys in 46% of write runs. Full typed context reduces invalid calls to 3% (3 of 101 runs), with no invented codes. These results show that exposing process and schema constraints during retrieval improves the reliability of agent-driven API calls.

Keywords: Service discovery, Knowledge graphs, Retrieval-augmented generation, LLM agents, Business process modeling

## 1 Introduction

Imagine an AI assistant that must arrange a business trip. It has to find a flight, book it, and then reserve a hotel using the booking reference, in that order. Each booking form also has required fields and fixed code lists, such as cabin class or fare type. The assistant can pick the right booking API and still fail: it may invent a fare code, leave out a required reference, or book the hotel before the flight. Such a call fails just as badly as a call to the wrong API. Reliable multi-step automation therefore needs more than the right endpoint. It also needs two kinds of constraints around each call: sequential constraints (what comes next) and parametric constraints (which keys and code values are valid).

Enterprise software has the same problem at a much larger scale. Large companies expose thousands of APIs. A typical SAP S/4HANA system, for example, ships hundreds of Open Data Protocol (OData) services, each with many collections, fields, and cross-references. Suppose a developer asks: “create a purchase order for supplier X with a valid status code, then confirm goods receipt.” Choosing the right endpoint is only the first step. The payload keys must match the schema, the coded values must be valid, and the calls must follow the business process. Each of these rules gives a language model a chance to produce a call that looks right but is invalid.

Recent work on retrieval-augmented service discovery [1, 2] improves endpoint retrieval with chunking and embedding-based filtering. However, it treats each endpoint as an isolated unit. It leaves out three kinds of context: (a) process, what comes next in the workflow; (b) structure, which entities and value-help codes are valid; and (c) execution, how an agent builds and sends the call. The Discovery Agent [1], for example, ranks endpoints but stops at retrieval, and it has no process-chain ground truth. Its authors suggest knowledge graphs as a way to improve accuracy, but note that graphs are costly to build. We address this gap.

Our approach links two open, machinereadable standards. OData \$metadata describes the entity collections, fields, and references of a service, but not the order in which they are used. BPMN describes the order of business tasks, but not the data each task uses. We load both sources into one typed property graph and link process activities to the entity sets they use. A hybrid retriever finds the closest entity sets by embedding similarity and then follows process edges to the ones that come next. It returns each endpoint with its schema context, which an execution agent uses to issue validated GET and POST calls through a mock HTTP layer.

This paper makes four contributions:

(a) A typed property-graph schema that combines service metadata with business processes, and an ingestion pipeline for OData \$metadata and BPMN 2.0. We evaluate this design with a construction-fidelity audit (Sec. 5.3) and an ablation study (Sec. 5.5).

(b) A hybrid retriever that combines dense embedding seeds with a bounded expansion along process-sequence edges. We evaluate it in Sec. 5.4 and test where it stops working in Sec. 5.8.

(c) An execution-capable ReAct [3] agent that uses the retrieved value-help and cardinality context to build its calls. We evaluate it in Secs. 5.6 and 5.7, including a baseline that receives the value lists as plain text, which separates the efect of seeing the values from the efect of the typed representation.

(d) OData-Proc, a public benchmark of 42 enterprise services, 14 BPMN reference processes, and 288 queries with two-annotator, processchain ground truth. To our knowledge it is the first benchmark that scores service discovery against business-process chains.

Sec. 2 explains the main terms for readers who are new to OData, or BPMN. Sec. 3 reviews related work, Sec. 4 describes our method, and Sec. 5 reports the experiments.

## 2 Background

This section introduces the systems and standards we build on; Table 1 summarises the key terms and metrics.

Table 1 Key terms and metrics used in this paper.
<table><tr><td>Term</td><td>Meaning</td></tr><tr><td>Service (API)</td><td>A web interface that lets a program read or write business data.</td></tr><tr><td>$metadata</td><td>The machine-readable description that every OData service publishes.</td></tr><tr><td>EntitySet</td><td>A named collection of records, similar to a table (e.g., PurchaseOrder).</td></tr><tr><td>Property</td><td>One field of a record (e.g., Supplier).</td></tr><tr><td>Navigation</td><td>A link between two EntitySets, with a cardi- nality such as 1:N.</td></tr><tr><td>Value help</td><td>The fixed list of valid codes for a property</td></tr><tr><td>Payload</td><td>(e.g., NB, UB, FO). The property values sent in a POST call that</td></tr><tr><td>Activity</td><td>creates a record. One task in a BPMN process (e.g., create purchase order).</td></tr><tr><td>Mapping edge</td><td>A hand-written link from an activity to an EntitySet it reads or writes.</td></tr><tr><td>Process chain</td><td>The EntitySets used by an activity and by the activities that follow it.</td></tr><tr><td>Invalid call</td><td>A GET or POST call that uses an EntitySet, key, or code that does not exist.</td></tr><tr><td>Recall@k</td><td>Share of the gold EntitySets that appear in the top k results.</td></tr><tr><td>MRR</td><td>Mean of 1/rank of the first correct result.</td></tr><tr><td>nDCG@10</td><td>Ranking quality of the top 10, giving more weight to higher ranks.</td></tr><tr><td>Completion rate</td><td>Share of agent runs that finish the task with valid calls.</td></tr></table>

Enterprise systems and SAP. Companies run their core business tasks, such as purchasing, sales, and warehousing, in enterprise resource planning (ERP) software. SAP S/4HANA is one such ERP system. It gives other programs access to its data through web services, also called APIs. SAP publishes the descriptions of these services in a public catalogue, the SAP Business Accelerator Hub. We use only these public descriptions, so no running SAP system is needed.

OData services. The Open Data Protocol (OData)<sup>1</sup> is a standard for web services that read and write data over HTTP. Each OData service publishes a machine-readable description of itself, called its \$metadata document. An EntitySet is a named collection of business records, similar to a database table, for example PurchaseOrder. Its EntityType defines the structure of a record, and each field of a record is a Property, for example Supplier. A NavigationProperty links one EntitySet to another and states the cardinality: one purchase order has many purchase-order items (1:N). Some properties accept only codes from a fixed list, which SAP calls value help. For example, the purchase-order type must be NB (standard order), UB (stock transport order), or FO (framework order). A client reads records with a GET call. It creates a record with a POST call, whose body (the payload) holds the property values. The \$metadata document says what data exists, but not in which order a business uses it.

BPMN processes. Business Process Model and Notation (BPMN 2.0)<sup>2</sup> is a standard for describing business processes. A BPMN model is a flow chart of activities (tasks) connected by sequence flows. It is stored as XML, so programs can read it. A purchasing process, for example, may run create purchase order, then post goods receipt, then enter supplier invoice. A BPMN model says in which order tasks happen, but not which data each task uses. Our method links the two standards: each activity is connected to the EntitySets it reads or writes.

Property graphs. A property graph is a kind of knowledge graph. It stores data as nodes and directed edges, and both can carry a type label and key-value attributes. Cypher [4] is a query language for such graphs. A Cypher query can follow a path of bounded length, for example “from this activity, follow up to three next edges”. Such bounded queries have predictable cost. We use FalkorDB<sup>3</sup>, an open-source graph database that runs Cypher.

Dense retrieval. An embedding model turns a text into a vector, so that texts with similar meaning get nearby vectors. Dense retrieval embeds the query and returns the k stored items with the closest vectors. We use the FAISS library [5] for this search. Dense retrieval works well when the query describes the target. It has no way to find a target that the query does not describe, such as the next step of a process.

LLM agents. An LLM agent is a large language model that runs in a loop. It reasons about the next step, calls a tool, reads the result, and repeats until the task is done. This pattern is called ReAct [3]. In this setting, hallucination means that the agent invents an EntitySet, a key, or a code that does not exist in the service description, for example the code NORMAL instead of NB. We call any such call an invalid call.

## 3 Related Work

## 3.1 Retrieval-augmented service discovery

Pesl et al. [1] compare chunking strategies for OpenAPI specifications, including ChunkRAG [6]. They also introduce SOCBench-D, a benchmark of 550 queries across 11 GICS sectors. Their best method, endpoint splitting with summaries, is our chunk-RAG baseline. They leave knowledge graphs to future work, and their Discovery Agent does not execute calls. Earlier work in the same line, such as CRAFT [7] and Rest-GPT [2], combines retrieval with single-call execution. Their position papers [8, 9] frame LLM-based service composition as a wider research agenda. We extend this line to multi-step, payload-aware composition with explicit process context.

Expanding a knowledge graph around dense retrieval seeds is also common in KG-augmented RAG, for example in GraphRAG [10]. Our claim is narrower. In our experiments, generic graph structure over service metadata does not improve retrieval. The gain comes from typed processsequence edges. Graphs built only from extracted entities do not have such edges.

![](images/1a9b994e018cd9093b519742ea556a0f7de2da848bf5de1610f8705676191583.jpg)  
Fig. 1 (Left) End-to-end architecture: ofline ingestion and indexing feed an online hybrid retriever and an executioncapable ReAct agent. (Right) Property-graph schema: OData structural nodes (purple); BPMN process nodes (orange); ValueHelp enum node (green).

## 3.2 Classical composition and knowledge graphs for services

In service computing [11, 12], automated service composition is decades older than LLMs. Semantic web service frameworks such as OWL-S [13], WSMO [14], and SAWSDL [15] describe service capabilities in machine-readable form. They need a lot of manual modeling, and this cost is one reason given [1] for preferring chunk-RAG over knowledge graphs. Other work treats composition as AI planning. HTN planners such as SHOP2 [16] turn goals into executable service chains; see [17] for a survey. QoS-aware methods [18] optimize non-functional constraints. Recent non-LLM methods learn service representations directly, for example with feature mining and contrastive learning [19]. Our process edges look like the workflow skeletons that planners use. The diference is that we derive them from BPMN reference processes and give them to an LLM as retrieval context, not to a symbolic planner.

Our approach reduces this manual work but does not remove it. OWL-S and WSMO need annotations for every service in a catalogue. We load OData \$metadata automatically and write by hand only the BPMN layer and a small activity-to-EntitySet mapping (Sec. 6.1).

## 3.3 LLM tool use and agent frameworks

ReAct [3] introduced the pattern of interleaving reasoning and acting. Several works connect LLMs to large API collections. Toolformer [20] learns tool calls through self-supervision. Gorilla [21] retrieves and invokes APIs from large documentation sets. ToolLLM [22], AnyTool [23], and API-Bank [24] evaluate tool calling over broad, loosely typed REST APIs, usually from free-form documentation and for single calls or short chains. None of them model the structural rules, such as fixed value lists or navigation cardinality, that decide whether a payload is valid. In our system, every retrieved endpoint comes with these rules, and the mock execution layer enforces them.

## 3.4 Business-process and API integration

Graph transformers improve web API recommendation [25], and LLM-based representations help recommend services with little usage history [26]. LLMs also generate TOSCA service compositions from natural-language requirements [27] and drive business-process discovery [28]. Goretti et al. [29] enforce usage policies during process discovery with a Trusted Execution Environment. We focus on accurate service matching and on the schema and process context an agent needs before it can execute a call.

Work that links business-process models to executable services has mostly focused on modeling. Examples include deriving orchestrations from BPMN [30] and industrial tools such as Camunda Connectors<sup>4</sup> and SAP Signavio reference processes<sup>5</sup>. This link has rarely been used as the basis for retrieval and execution in LLMbased discovery. Work on generating BPMN from text [31] may one day automate the process layer that we currently write by hand. In our system, the activity-to-endpoint mapping is a first-class graph edge that the retriever uses directly. To our knowledge, no earlier work has evaluated processchain expansion in retrieval-augmented service discovery. Our benchmark provides a repeatable setting for doing so.

## 4 Approach

## 4.1 System Overview

Fig. 1(left) shows the two stages of our system. In the ofline stage, the ingestion pipeline reads OData \$metadata and BPMN 2.0 files and writes the nodes and edges to the graph database. It also writes a short text summary for each EntitySet and indexes the summaries with FAISS [5].

In the online stage, the hybrid retriever takes the top-k dense matches as seeds and expands them along the process chain with a bounded Cypher [4] query. A ReAct [3] agent then issues GET and POST calls, which a mock HTTP layer validates.

Fig. 1(right) shows the graph schema. Eight node labels represent OData and BPMN structure, and nine edge types connect them. Two design choices matter for execution. First, we store cardinality on Navigation nodes. This keeps graph queries simple and shows the agent which relations are 1:N and which are N:1. Second, ValueHelp is its own node and holds the list of valid codes. The agent can therefore get the valid codes with one bounded graph lookup, instead of relying on a loose embedding match.

## 4.2 Ofline: Ingestion Pipeline

OData parser. The OData parser reads the EntityTypes, EntitySets, and SAP annotations of each \$metadata document. Each EntityType becomes a node, and its Properties become child nodes linked by HAS PROPERTY. Each Navigation-Property becomes a Navigation node. It links to its source and target EntityTypes through NAVIGATE FROM and NAVIGATE TO and stores its cardinality (one or many). ValueHelp annotations become nodes, linked by HAS VALUE HELP, that hold the explicit code lists. Table 2 gives the counts.

BPMN parser. The BPMN parser extracts the activities of each process and builds HAS NEXT edges from the sequence flow. We ignore gateways, events, and pools. For our purpose, the value of the BPMN layer is the order of activities, not the full execution semantics.

Activity-to-EntitySet mapping. A short, hand-written mapping links each activity to the EntitySets it reads or writes (e.g., create purchase order → PurchaseOrder, PurchaseOrderItem). It covers 14 processes, 51 task activities, and 45 USES ENTITY SET edges. The mapping is small enough to review in full, yet it drives the retrieval gains reported in Sec. 5. We leave ten activities unmapped on purpose, because no service in the catalogue covers their systems (e.g., recruiting, payroll, outbound delivery).

EntitySet summaries. For the embeddings, we build a short text summary of each EntitySet. It joins the EntitySet label, the EntityType description, and the Property names. This follows the “summary” chunking approach of [1], but at the EntitySet level instead of the endpoint level.

## 4.3 Online: Retrieval and Execution

Hybrid retrieval. The retriever has two branches. The dense branch runs a top-k search over the EntitySet summaries and returns a set S of likely matches. For each $e \in \ S$ the process branch runs a bounded graph query that finds downstream EntitySets: it goes from e to the activities that use it, follows HAS NEXT to later activities, and collects the EntitySets those activities use. The retriever keeps the dense order, places each new entity after its seed, and removes duplicates. It also returns the cardinality of each navigation, so the agent knows how to shape the payload.

THINK user wants PO + goods receipt; need entity sets   
ACT RETRIEVE("create purchase order then goods receipt")   
OBS specs: PurchaseOrder { required: Supplier, PurchaseOrderType;   
PurchaseOrderType value-help: [ "NB", "UB", "FO" ];   
nav: PurchaseOrderItem (1:N) }, MaterialDocumentHeader { ... }   
process: Create purchase order -> Post goods receipt   
THINK PurchaseOrderType must come from the value-help list -> "NB"   
ACT POST PurchaseOrder { Supplier:"4711", PurchaseOrderType:"NB",   
PurchaseOrderItem:[ { Material:"M-001", Quantity:5 } ] }   
OBS 201 -> { PurchasingDocument: "4500001234" }   
THINK next activity in chain is goods receipt   
ACT POST MaterialDocumentHeader   
{ ReferenceDocument:"4500001234", DocumentType:"WE" } -> 201  
Fig. 2 Agent trace for a query that creates a purchase order and then posts the goods receipt. THINK is the agent’s reasoning, ACT a tool call, and OBS the tool result. The retrieved value-help list stops the agent from inventing a PurchaseOrderType code. Process expansion shows that MaterialDocumentHeader comes next.

Execution agent. A ReAct agent, built with LangGraph<sup>6</sup>, has three tools: (a) hybrid retrieval, which returns the full typed specification (required keys, navigation cardinalities, value-help lists, and process-chain information); (b) GET; and (c) POST. A mock HTTP layer validates every call. All grounding context arrives in one retrieval call, which keeps tool round-trips low for local 30B models. Two mechanisms reduce hallucination. First, POST rejects payloads with missing keys or invalid codes, so the agent can retry. Second, navigation cardinalities tell the agent to send 1:N relations as nested arrays. Fig. 2 shows both mechanisms.

Worked example. Consider the query: “Create a purchase order with a valid type code for supplier 4711, then post the goods receipt.” A semantic retriever over endpoint summaries finds PurchaseOrder, but it may miss PurchaseOrderItem because the query never says “item”. The valid type codes exist only in ValueHelp, so an LLM that sees only summaries may output a plausible but invalid code such as NORMAL. It also does not know that Material DocumentHeader must follow PurchaseOrder. Fig. 2 shows how our system closes each gap: the dense seed finds PurchaseOrder, process expansion adds MaterialDocumentHeader, and the retrieved specification gives the 1:N item navigation and the value-help list.

## 5 Experimental Results

We evaluate our approach with a constructionfidelity audit and five research questions (RQs). Each one targets a contribution from Sec. 1.

Fidelity audit (contribution (a)): Does the graph match the source OData \$metadata and BPMN files (Sec. 5.3)?

RQ1 (retrieval quality; contribution (b)): Does process-aware hybrid retrieval find more correct endpoints than dense-only and graph-only retrieval? RQ1 uses each benchmark query word for word, so it measures retrieval without any agent behaviour.

RQ2 (ablation; contributions (a),(b)): Which graph component causes the gain?

RQ3 (end-to-end agent; contribution (c)): Can the agent complete multi-step tasks without invalid calls? RQ3 measures the full loop, so retrieval errors and execution errors are mixed. An oracle-retrieval test separates them.

RQ4 (hallucination elicitation; contribution (c)): Does value-help grounding stop invented codes and keys when a query needs a fixed-value code?

RQ5 (external generalization; limits of contribution (b)): Does the gain carry over to OpenAPI services that have no business-process layer? This is a falsification test: if our explanation is right, there should be no gain.

## 5.1 Datasets

Our main benchmark, OData-Proc, combines four public OData corpora (SAP Business Accelerator Hub, TripPin, Northwind, and Microsoft Graph)

Table 2 OData-Proc dataset statistics. Structural counts are distinct elements after removing 29 duplicate source rows (Sec. 5.3). Most EntityTypes are complex, nested, or parameter types that are never exposed as a top-level collection. The RQ1 process-chain probe (n = 74) is exactly the process-chain goal type.
<table><tr><td>Element</td><td>Count</td><td>Queries</td><td>Count</td></tr><tr><td>Source services</td><td>42</td><td>Total</td><td>288</td></tr><tr><td>EntitySets</td><td>132</td><td>Single entity</td><td>130</td></tr><tr><td>EntityTypes</td><td>1,270</td><td>Process-chain goal</td><td>74</td></tr><tr><td>Properties</td><td>8,481</td><td>Navigation chain</td><td>67</td></tr><tr><td>Navigation edges</td><td>1,499</td><td>Multi-step workflow</td><td>15</td></tr><tr><td>BPMN processes</td><td>14</td><td>Value-help dep.</td><td>2</td></tr><tr><td>BPMN task activities</td><td>51</td><td>Train / test</td><td>150/138</td></tr><tr><td>Mapping edges</td><td>45</td><td>RQ3 queries</td><td>81</td></tr></table>

with 14 BPMN reference processes (Table 2). It uses only openly published metadata, so no access to a proprietary system is needed. Eight of the process files cover SAP reference scenarios such as Warehouse Inbound Processing and Sell from Stock. We converted these files to BPMN 2.0 from an earlier RDF format and kept the activity names, order, and entity-set mappings.

There are five query types. A single-entity query asks for one EntitySet. A navigation-chain query needs EntitySets linked by navigation. A process-chain goal states a business goal whose downstream EntitySets share few words with the query. A multi-step workflow asks for several explicit steps, and a value-help dependent query needs a fixed code. The benchmark has 288 queries, split deterministically into 150 training and 138 test queries. We used the training set only to tune prompt wording and the retriever settings, and did not look at the test set until the final evaluation. All 74 process-chain probes are in the test split. The value-help category has only 2 queries on purpose: we test code grounding on a separate 56-query elicitation probe with adversarial wording, so that it does not inflate the main retrieval scores. The RQ3 execution experiments use the 81 test queries of the four multi-step types that target the SAP corpus. A mock HTTP layer validates payloads against the OData metadata, so the experiments need no live backend. For external generalization, we sample 110 SOCBench-D queries, 10 from each GICS sector [1].

The first version of the benchmark had 80 queries labelled by one annotator. We grew it by generating 251 candidate queries from the catalogue and the process graph, each with machinesuggested labels, and having two annotators review every candidate independently: a human author and a large language model (Claude, Anthropic), whose per-candidate decisions we release. They agreed on whether to keep 250 of the 251 candidates (Cohen’s κ = 0.665) and on all 415 (query, EntitySet) labels of the candidates both kept $( \kappa = 1 . 0 0 0 )$ . Both saw the suggested labels, so this measures agreement on corrections, not independent labelling. The merge added 208 queries to the main benchmark and 41 to the elicitation probe (full protocol in Supplementary Sec. S2).

## 5.2 Experimental Setup

We ran all experiments on an A100 GPU (40 GB VRAM). They can also be reproduced on a workstation with 32 GB RAM and no GPU. FalkorDB stores the graph, and Ollama<sup>7</sup> hosts the LLMs and embedding models locally. The primary LLM is q wen3-coder:30b (30B, tool-tuned). We use qwen 3.6 (36B, general) and gemma4:31b for sensitivity analysis. The main embedding model is qwen 3-embedding:8b. As a control, we also use the smaller sentence-transformers/all-mpnet-ba se-v2. Table 1 defines the metrics.

Unless stated otherwise, every p value in this paper comes from a paired two-sided Wilcoxon signed-rank test over per-query scores; where we compare all three retrievers at once, we use the Friedman test and call it the overall test. Both are standard tests for paired results that do not assume a normal distribution.

## 5.3 Graph-Construction Fidelity

All later results depend on the graph, so we first check that it matches its sources. An independent checker, written separately from the ingestion pipeline, reads the source OData and BPMN files again and compares them with the graph. For each structural class we measure coverage (the share of source elements found in the graph) and exactness (the share of graph elements whose attributes match the source). For EntitySets, EntityTypes, Properties, Navigation edges with cardinality,

ValueHelp codes, BPMN activities, and activity order, both coverage and exactness are 1.000. The few diferences come from the sources (number formats and duplicate rows), not from ingestion. The audit does not cover the hand-written mapping edges, because no independent source exists for them; we assess them with the provenance split in Sec. 5.4. Supplementary Sec. S3 gives the full audit.

## 5.4 RQ1: Retrieval quality

We compare Dense (FAISS only), Graph (graph only), and Hybrid (our method) on the 138 test queries of OData-Proc. Table 3 reports Recall@{1, 3, 5, 10}, MRR, nDCG@10, and latency against the labelled gold EntitySets.

As expected, the gain appears only where there is a process signal. On the process-chain probe $( n = 7 4 )$ , the downstream entities share few words with the query. Here Hybrid improves Recall@3 by +5.7 points over Dense (0.732 vs. 0.675; p = 0.037). The gap narrows to +3.4 points at k = 5 $( p \ = \ 0 . 1 3 )$ as dense retrieval catches up, and widens again at k = 10 (+3.2 points, $p = 0 . 0 1 8 )$ On the 64 non-probe queries, Hybrid stays close to Dense (Recall@3 0.901 vs. 0.917). The small loss comes from expansion sometimes pushing out a correct dense result. Over the full test split, Hybrid gains +2.3 points at k = 3 (0.810 vs. 0.787; overall test $p = 1 . 7 \times 1 0 ^ { - 3 1 }$ , n = 138). This gain comes entirely from the probe, which makes up more than half of the test split. The efect is smaller than in the earlier 80-query version of the benchmark (+15.0 probe Recall@3 points on n = 20). Many of the new probe queries were reworded to name their business process explicitly, which gives dense retrieval more words to match.

MRR barely changes (0.893 for Hybrid vs. 0.902 for Dense). Expansion places new entities after the dense seed, so its benefit is to fill ranks 2 and 3 with downstream entities that dense retrieval cannot reach. At k = 3, expansion adds 32 of the 304 gold entities (28 of them in the probe) and pushes 16 dense golds out of the top three. Graph alone performs poorly: without embedding seeds, a graph-only search cannot tell candidates apart. This supports the hybrid design. Supplementary Sec. S4 gives the full pertype breakdown and shows that, whenever both entities of a required pair are retrieved, the context passed to the agent is complete.

Ground-truth provenance split. The same author wrote the activity-to-EntitySet mapping and the query labels. To control for this circularity, we split the test queries by where their labels come from. Labels for 59 queries follow from the OData metadata alone (no process chain needed). The other 79 queries need the mapping. On the metadata-only queries, Hybrid and Dense tie at Recall@1 (0.675). Hybrid is slightly lower at Recall@3 and Recall@5 (0.915 vs. 0.932; 0.958 vs. 0.975), because expansion sometimes pushes out a late gold. All of the gain is in the mapping-dependent subset (Recall@3 0.678 → 0.732, nDCG@10 $0 . 8 1 5 ~  ~ 0 . 8 4 4 )$ . Performance therefore depends on a correct mapping.

Validity and scope. (i) Construction. The ground-truth chains of the probe come from the hand-written mapping, and the queries are written to avoid word overlap. This favours Hybrid by design. The probe therefore tests whether curated process edges make otherwise unreachable entities retrievable, not whether such edges exist by default. (ii) Multiple comparisons. The overall test is clear $( p ~ = ~ 1 . 7 ~ \times ~ 1 0 ^ { - 3 1 } )$ We ran two paired tests on the probe, so a stricter threshold of 0.025 applies to each (Holm correction); the main probe result $( p = 0 . 0 3 7 )$ does not meet it. With the smaller mpnet embedding, Hybrid still beats Dense on the probe at every k, but none of these diferences is significant (all $p \ \geq \ 0 . 1 9 )$ We therefore read the probe-level gain as consistent in direction but not confirmed at its current efect size. (iii) Baseline choice. Dense already implements Pesl’s “summary” chunking at the EntitySet level, so it stands in for chunk-RAG on OData. Porting their token-overlap retriever to OData would risk measuring the port, so we evaluate it on its native benchmark, SOCBench-D (RQ5).

## 5.5 RQ2: Ablation

We remove Hybrid components one at a time. Removing USES ENTITY SET and HAS NEXT, or all graph branches, turns Hybrid into Dense: the metrics match the Dense rows of Table 3 at every k. The Recall@3 gain therefore comes from the process edges, not from the other graph elements. We do not ablate ValueHelp here, because it is used to build payloads, not to rank results; RQ3 and RQ4 test it.

Table 3 RQ1 retrieval quality on the OData-Proc test split $( n = 1 3 8 )$ . Hybrid is our process-aware method; Dense is FAISS-only retrieval; Graph is graph-only retrieval. The lower block shows the process-chain probe subset $( n = 7 4 )$ ; there, Hybrid beats Dense at Recall@3 with $p = 0 . 0 3 7$ (paired test).
<table><tr><td>Cond.</td><td>R@1</td><td>R@3</td><td>R@5</td><td>R@10</td><td>MRR</td><td>nDCG@10</td><td>lat. (ms)</td></tr><tr><td colspan="8">Full set  $( n = 1 3 8 )$ </td></tr><tr><td>Dense</td><td>0.472</td><td>0.787</td><td>0.870</td><td>0.928</td><td>0.902</td><td>0.868</td><td>202.9</td></tr><tr><td>Graph</td><td>0.132</td><td>0.292</td><td>0.394</td><td>0.617</td><td>0.352</td><td>0.397</td><td>15.6</td></tr><tr><td>Hybrid</td><td>0.472</td><td>0.810</td><td>0.885</td><td>0.945</td><td>0.893</td><td>0.875</td><td>208.4</td></tr><tr><td colspan="8">Process-chain probe</td></tr><tr><td>Dense</td><td>0.318</td><td>0.675</td><td> $( n = 7 4 )$  0.788</td><td>0.869</td><td>0.889</td><td>0.809</td><td>203.9</td></tr><tr><td>Graph</td><td>0.012</td><td>0.068</td><td>0.142</td><td>0.419</td><td>0.181</td><td>0.222</td><td>15.4</td></tr><tr><td>Hybrid</td><td>0.318</td><td>0.732</td><td>0.822</td><td>0.901</td><td>0.877</td><td>0.837</td><td>199.9</td></tr></table>

Separating the two process-edge types. Are the 45 hand-written mapping edges enough on their own, or does BPMN sequence order also help? Mapping-only keeps USES ENTITY SET but removes HAS NEXT, so expansion finds the entities used by the seed activity but cannot follow the chain. Sequence-only keeps HAS NEXT but removes the mapping, which cuts the path from entities to activities. Both stay at the Dense level (probe Recall@3 0.673 and 0.675; full split 0.786 and 0.787). Full Hybrid beats mapping-only on the probe by +5.9 Recall@3 points $( p = 0 . 0 2 8 ; + 2 . 6$ at $k = 5 , p = 0 . 3 3 )$ . Both edge types are therefore needed: the mapping ties entities to activities, and HAS NEXT carries the expansion to later activities.

## 5.6 RQ3: End-to-end agent

Table 4 RQ3: mock-execution agent on the 81 multi-step test queries (3 runs per query, primary LLM). “Calls/run” is the mean number of tool calls per run, and “Invalid/run” the mean number of invalid calls per run (GET \$metadata requests are not counted). No run timed out. Completion diference +0.284, $p < 0 . 0 0 1$ (paired test, $n = 8 1 )$ . Results on an earlier 26-query subset and with other LLMs are in Supplementary Sec. S5.

<table><tr><td>Cond.</td><td>Runs</td><td>Compl.</td><td></td><td>Calls/run Invalid/run Lat.(s)</td><td></td></tr><tr><td>ChunkRAG</td><td>243</td><td>0.103</td><td>0.89</td><td>0.107</td><td>5.3</td></tr><tr><td>Ours</td><td>243</td><td>0.387</td><td>2.45</td><td>0.103</td><td>11.5</td></tr></table>

Table 4 shows agent results on the 81 multistep test queries (20 navigation chains, 58 processchain goals, 2 multi-step workflows, and 1 valuehelp query). Both conditions share the same mock GET/POST layer and validator. ChunkRAG [6] combines the chunk-RAG retriever of [1] with our execution layer and returns entity set, service, and property names. Our agent’s retrieval returns the full typed specification: required keys, navigation cardinality, value-help code lists, and process-chain details. ChunkRAG therefore cannot see value-help lists, by design (Supplementary Sec. S1 compares the tools). We run each query three times (243 runs per condition).

Scope and circularity. These experiments use no live backend. The mock validator checks payloads against the same metadata and ValueHelp nodes that we used to build the graph, so our agent makes an invalid call only when it ignores or misuses the retrieved context. The invalid-call metric therefore tests whether grounding closes this failure mode by construction, not how the agent performs against a live service (Sec. 6.2).

Completion rate. Our agent completes far more tasks than ChunkRAG (0.387 vs. 0.103, p < 0.001). The gain holds in every type with more than two queries: navigation chains (0.517 vs. 0.100) and process-chain goals (0.316 vs. 0.103). Two factors explain it. First, ChunkRAG often stops before acting: it makes no GET or POST call in 173 of 243 runs (ours: 78). Second, when inloop discovery returns all gold entities, our agent completes 0.489 of runs, compared with 0.175 for ChunkRAG. Retrieval misses still cause many failures for both methods: 100 of 218 ChunkRAG failures and 57 of 149 of ours. Our agent is slower (11.5 vs. 5.3 seconds per run), mainly because it makes more tool calls (2.45 vs. 0.89).

An earlier 26-query subset, written before the benchmark grew, pointed the other way: ChunkRAG completed more tasks (0.295 vs. 0.192), but the diference was not significant $( p =$ 0.46). On that subset, a wrong dense seed was the main risk, because process expansion then adds the wrong chain. We base our conclusions on the larger test set and report the subset in Supplementary Sec. S5.

Invalid-call rate. Both agents make about the same number of invalid calls per run (0.103 for ours vs. 0.107 for ChunkRAG; 25 vs. 26 in total). Neither invents a code. Our agent writes much more often (55 vs. 22 runs with a POST), so the per-run rates hide this diference. Most of our invalid calls read or write an entity set that the service does not have (16 of 25). Most of ChunkRAG’s invalid calls use a key that does not exist (21 of 26). These queries rarely need fixedvalue codes, so the elicitation probe in Sec. 5.7 tests this case directly.

Separating execution from retrieval. To separate the two failure sources, we ran an oracle condition on the 26-query subset: each agent receives the gold specification for each query (perfect retrieval). Completion stays low for both agents (ours 0.256, ChunkRAG 0.359), so execution by the 30B LLM, not only retrieval, limits completion. Payload construction still separates the two conditions. With the same gold context, ChunkRAG issues 15 schema-invalid writes, including 6 invented codes, while ours issues 3 and invents none. Supplementary Sec. S5 gives the full oracle results, a seed-verification variant, and the sensitivity runs with gemma4:31b and qwen3.6.

## 5.7 RQ4: Hallucination elicitation

Table 5 RQ4: code elicitation on the 56-query probe (3 runs per query; 168 per condition; primary LLM). “POST runs”: runs with at least one POST. “Inv. runs”: POST runs with an invalid call. “Invalid”: all invalid calls (entity set, key, or code). “Codes”: invented codes. ChunkRAG+VH adds the value-help code lists as plain text to the retrieved chunks. No run timed out.

<table><tr><td>Cond.</td><td>POST runs Inv. runs Invalid</td><td></td><td></td><td>Codes</td></tr><tr><td>ChunkRAG</td><td>127</td><td>100/127</td><td>218</td><td>60</td></tr><tr><td>ChunkRAG+VH</td><td>128</td><td>59/128</td><td>120</td><td>0</td></tr><tr><td>Ours</td><td>101</td><td>3/101</td><td>3</td><td>0</td></tr></table>

We build a 56-query elicitation probe, separate from the main benchmark. It started with 15 hand-written queries; the two-annotator protocol (Sec. 5.1) added 41 more. Each query implies a specific SAP fixed-value code without naming it. For example, “a standard purchase order” implies PurchaseOrderType=NB. The codes span three EntitySets that accept POST and use real SAP value-help domains. The ingested \$metadata has no machine-readable value lists, so we enter the relevant domains by hand. The graph and the mock validator use the same values, so the circularity note in Sec. 5.6 applies.

Table 5 shows the result. Because the conditions issue POSTs at diferent rates, we report invalid-call rates over runs that issue a POST. ChunkRAG makes an invalid call in 100 of 127 POST runs (0.79, 95% CI [0.71, 0.85]), with 218 invalid calls in total. Sixty of these are invented codes, such as DA or EX WORKS instead of the Incoterms codes DAP and EXW, RET instead of the returns code RE, and IV instead of the invoice code F2. The most common invented code is the purchasing code NB reused as a sales-order type (19 times), so pre-training is not reliable even for wellknown codes. Our method makes an invalid call in 3 of its 101 POST runs (0.03, 95% CI [0.01, 0.08]) and invents no code. Two of its three invalid calls put a header key (ShippingCondition) on a salesorder item, and one reads an entity set from the wrong service. Per query, our invalid-run rate is lower than ChunkRAG’s (p < 0.001, n = 56).

One cost remains. In 45 of our 168 runs, the agent makes no tool call and replies in text only (ChunkRAG: 19). This is a failure to use the tools, not a timeout, and it is not hidden partial execution, because these runs did nothing.

Symmetric baseline: ChunkRAG with inlined value-help. Plain ChunkRAG cannot see value-help lists. To separate “the value list was visible” from “a typed graph carried it”, we add a third condition that pastes the value-help code lists as plain text into ChunkRAG’s retrieved chunks. Everything else stays the same. Pasting the lists closes the code class: ChunkRAG+VH invents no codes in its 128 POST runs. Showing the value list is therefore enough to stop invented codes, whatever form it takes. It does not stop other invalid calls: 120 remain in 59 of the 128 POST runs (0.46, 95% CI [0.38, 0.55]), because flat text does not constrain the payload structure. With the full typed context, the rate falls to 0.03 (p < 0.001 per query against ChunkRAG+VH, n = 56).

## 5.8 RQ5: External generalization

Table 6 shows the external test. We adapt the OData ingestion to OpenAPI 3.x and process the SOCBench-D specifications behind the 110 sampled queries (10 per GICS sector). SOCBench-D has no business-process layer, so the process branch of the Hybrid retriever is inactive, and only the weaker shared-parameter graph fallback applies. We compare against a token-overlap retriever (Tokens), which represents Pesl’s chunk-RAG baseline.

Table 6 RQ5: generalization on SOCBench-D (OpenAPI, no BPMN layer), pooled over 11 GICS sectors $( n = 1 1 0$ queries). Hybrid vs. Tokens (paired tests): R@3 +0.010, p = 0.753; R@5 −0.021, $p = 0 . 4 3 7 ; \mathrm { R @ 1 0 ~ - 0 . 0 4 1 } , p = 0 . 0 7 1 .$
<table><tr><td>Retriever</td><td>R@3</td><td>R@5</td><td>R@10</td></tr><tr><td>Tokens</td><td>0.297</td><td>0.413</td><td>0.614</td></tr><tr><td>Hybrid (no BPMN)</td><td>0.307</td><td>0.392</td><td>0.573</td></tr></table>

Over all 110 queries, Hybrid and Tokens do not difer significantly at any k (all $p ~ \geq ~ 0 . 0 7 )$ This is what we predicted: the gain depends on process-chain edges, not on graph structure in general. At $k = 1 0$ , Hybrid is slightly worse $( \Delta \ : =$ −0.041, $p = 0 . 0 7 1 \AA )$ , because without process context, expansion adds some ranking noise. This small negative result further supports our explanation of the OData-Proc gain. Per-sector results (Supplementary Sec. S7) are mixed and depend on how closely endpoint paths match the words in the queries.

## 6 Discussion

## 6.1 Mapping Efort and Scalability

The whole retrieval gain flows through the 45 hand-written USES ENTITY SET edges, so their authoring cost is the main practical bottleneck. Three properties keep it low. First, the efort grows with the number of business processes, not with the size of the catalogue: the 42 services, 132 EntitySets, and 8,481 properties were loaded with no manual work. Second, writing an edge only requires knowing which documents an activity reads or writes, which process owners already know; unlike OWL-S and WSMO annotations, it needs no ontology expertise. Third, the mapping is small enough to review in full.

We measured this cost directly. One author familiar with the catalogue re-wrote a random sample of ten mapped activities, covering 12 of the 45 edges, from scratch under a timer, following a fixed four-step procedure (Supplementary Sec. S6). The sample took 205 person-minutes (median 20.5 minutes per activity, range 16–25), and every re-written edge matched the released mapping. At the median rate, all 41 mapped activities would take about 840 person-minutes, or about two working days of focused efort.

For hundreds of processes, as in a full SAP Signavio library, manual authoring would still be costly. We therefore tested semi-automation, in which an LLM proposes edges and a human checks them. Given the catalogue and the ordered activities of each process, a local LLM (qwen3-coder:30b) proposes USES ENTITY SET edges. Against the 45 hand-written edges, the proposals reach precision 0.79 and recall 0.93 (F1 0.86). Authoring the mapping thus becomes checking 53 proposed edges instead of writing 45 from scratch. The three missed edges show that domain review is still needed (Supplementary Sec. S6).

## 6.2 Threats to Validity

Construct validity. The evaluation has a known circularity. The mock validator checks keys and codes against the same schema and ValueHelp nodes from which we build the graph. The low invalid-call results therefore show that grounding closes the failure mode by construction; they do not measure performance against an independent target. In addition, the probe queries and the activity mapping have the same author. The provenance split, the fidelity audit, and the symmetric value-help baseline reduce these efects but do not remove them.

Internal validity. The benchmark has twoannotator ground truth over 288 queries (Sec. 5.1), but two concerns remain. First, the second annotator is a disclosed LLM that worked from the same machine-suggested labels as the human. The nearperfect label agreement therefore overstates how independent the labels are. Second, the probelevel retrieval efect is modest and does not survive Holm correction (Sec. 5.4).

External validity. The gain depends entirely on process edges, so it only transfers to ecosystems that publish a process layer, or where one can be written. SOCBench-D confirms that there is no gain without one. SAP’s reference processes are unusually well documented, and we have not yet shown a second domain with a native process layer. RQ5 thus defines the scope of the method: it targets structured enterprise ecosystems, and chunk-RAG remains a good fit for free-form APIs. Future work includes a second domain with a native process layer, support for GraphQL and gRPC, and automatic BPMN generation [31] for domains without process models.

## 7 Conclusion

To call APIs reliably over several steps, an LLM agent must respect sequential and parametric constraints: process order, required keys, and valid code values. We meet this need by combining OData \$metadata and BPMN reference processes in one typed property graph. Hybrid retrieval combines dense embeddings with process-chain expansion, and a ReAct agent uses the resulting value-help and cardinality context to build payloads. On process-chain queries, the method improves Recall@3 by +5.7 points (p = 0.037, n = 74), and it stays close to dense retrieval elsewhere. The ablation shows that the gain comes from process edges, not from generic graph structure. In the execution probe, typed context stops invented codes and almost all invalid keys: our method makes an invalid call in 3 of its 101 POST runs, while the baseline does so in 100 of 127, with 60 invented codes. On multi-step test queries, our agent also completes more tasks than the baseline (0.387 vs. 0.103), although completion remains low for both and the system is not yet ready for deployment. The evaluation is also partly circular, because the mock validator and the graph share the same metadata. Future work should test against an independent backend, go beyond SAP reference processes, automate the process layer, and improve seed coverage and the agent’s use of tools.

## References

[1] Pesl, R.D., Mathew, J.G., Mecella, M., Aiello, M.: Retrieval-Augmented Generation for Service Discovery: Chunking Strategies and Benchmarking. IEEE Trans. on Services Computing 19(02), 1520–1534 (2026)

[2] Song, Y., Xiong, W., Zhu, D., Wu, W., Qian, H., Song, M., Huang, H., Li, C., Wang, K., Yao, R., Tian, Y., Li, S.: RestGPT: Connecting Large Language Models with Real-World RESTful APIs (2023)

[3] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: ReAct: Synergizing Reasoning and Acting in Language Models. In: ICLR (2023)

[4] Francis, N., Green, A., Guagliardo, P., Libkin, L., Lindaaker, T., Marsault, V., Plantikow, S., Rydberg, M., Selmer, P., Taylor, A.: Cypher: An evolving query language for property graphs. In: Proc. of the 2018 Int. Conf. on Management of Data, pp. 1433–1445 (2018)

[5] Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazar´e, P.-E., Lomeli, M., Hosseini, L., J´egou, H.: The faiss library. IEEE Trans. on Big Data 12(2), 346–361 (2026)

[6] Singh, I.S., Aggarwal, R., Allahverdiyev, I., Taha, M., Akalin, A., Zhu, K., O’Brien, S.: ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems. In: NAACL SRW 2025 (2025)

[7] Pesl, R.D., et al.: CRAFT: Cross-document retrieval-augmented generation for service apis. In: Proc. Int. Conf. on Advanced Info. Systems Engg. (CAiSE) (2025)

[8] Pesl, R.D., St¨otzner, M., Georgievski, I., Aiello, M.: Uncovering LLMs for servicecomposition: Challenges and opportunities. In: Service-Oriented Computing – ICSOC 2023 Workshops. LNCS, vol. 14518, pp. 39– 48. Springer, ??? (2024)

[9] Pesl, R.D., Georgievski, I., Aiello, M.: Compositio prompto: An architecture to employ large language models in automated service computing. In: Service-Oriented Computing (ICSOC). Springer, ??? (2024)

[10] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R.O., Larson, J.: From Local to Global: A Graph RAG Approach to Query-Focused Summarization (2024)

[11] Khan, S., Nguyen, P.H., Abdul-Rahman, A., Freeman, E., Turkay, C., Chen, M.: Rapid Development of a Data Visualization Service in an Emergency Response. IEEE Trans. on

Services Computing 15(3), 1251–1264 (2022)

[12] Khan, S., Chakraborty, J., Beaucamp, P., Bhujel, N., Chen, M.: Data Virtualization for Machine Learning. In: Luo, M., Zhang, L.- J. (eds.) International Conference on Services Computing (SCC), pp. 69–84. Springer, ??? (2025)

[13] Martin, D., Burstein, M., McDermott, D., McIlraith, S., Paolucci, M., Sycara, K., McGuinness, D.L., Sirin, E., Srinivasan, N.: Bringing semantics to web services with OWL-S. WWW 10(3), 243–277 (2007)

[14] Roman, D., Keller, U., Lausen, H., Bruijn, J., Lara, R., Stollberg, M., Polleres, A., Feier, C., Bussler, C., Fensel, D.: Web service modeling ontology. Appl. Ontol. 1(1), 77–106 (2005)

[15] Kopeck´y, J., Vitvar, T., Bournez, C., Farrell, J.: SAWSDL: Semantic annotations for WSDL and XML schema. IEEE Internet Computing 11(6), 60–67 (2007)

[16] Sirin, E., Parsia, B., Wu, D., Hendler, J., Nau, D.: Htn planning for web service composition using shop2. Journal of Web Semantics 1(4), 377–396 (2004)

[17] Rao, J., Su, X.: A survey of automated web service composition methods. In: Cardoso, J., Sheth, A. (eds.) Semantic Web Services and Web Process Composition, pp. 43–54 (2005)

[18] Zeng, L., Benatallah, B., Ngu, A.H.H., Dumas, M., Kalagnanam, J., Chang, H.: QoSaware middleware for web services composition. IEEE Trans. on Software Engineering 30(5), 311–327 (2004)

[19] Xu, Y., Bai, G., Xiao, W., Zhao, X., Yin, Y., Li, R., Zeng, F.: RESTful API service discovery via comprehensive feature mining, deep neural networks, and contrastive learning. In: Aiello, M., Deng, S., Murillo, J.-M., Georgievski, I., Benatallah, B., Wang, Z. (eds.) Service-Oriented Computing (ICSOC), pp. 19–34 (2026)

[20] Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language models can teach themselves to use tools. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) NeurIPS, vol. 36, pp. 68539–68551 (2023)

[21] Patil, S.G., Zhang, T., Wang, X., Gonzalez, J.E.: Gorilla: Large language model connected with massive apis. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) NeurIPS, vol. 37, pp. 126544–126565 (2024)

[22] Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Hong, L., Tian, R., Xie, R., Zhou, J., Gerstein, M., li, d., Liu, Z., Sun, M.: Toolllm: Facilitating large language models to master 16000+ real-world apis. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) ICLR, vol. 2024, pp. 9695–9717 (2024)

[23] Du, Y., Wei, F., Zhang, H.: Anytool: selfreflective, hierarchical agents for large-scale api calls. In: ICML (2024)

[24] Li, M., Zhao, Y., Yu, B., Song, F., Li, H., Yu, H., Li, Z., Huang, F., Li, Y.: API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. In: Bouamor, H., Pino, J., Bali, K. (eds.) EMNLP, pp. 3102–3116 (2023)

[25] Zheng, X., Wang, G., Zhang, Y., Han, B., Yu, J.: Motif-based linearizing graph transformer for web API recommendation. In: Service-Oriented Computing (ICSOC). Springer, ??? (2024)

[26] Rong, D., Yao, L., Zheng, Y., Yu, S., Xu, X., Liu, M., Wang, Z.: LLM enhanced representation for cold start service recommendation. In: Service-Oriented Computing (ICSOC), pp. 153–167. Springer, ??? (2024)

[27] Meflah, W., Brabra, H., Acheli, M., Ben Douma, H., Sellami, M., Gaaloul, W., Zeghlache, D.: From natural language to TOSCA: Leveraging LLMs for automated service composition. In: Service-Oriented Computing (ICSOC). Springer, ??? (2025)

[28] Lindenberg, P., Kumara, I., Owotogbe, J., Heuvel, W.-J., Tamburri, D.A.: Business process discovery through agentic generative ai. In: Service-Oriented Computing (ICSOC), pp. 253–267 (2026)

[29] Goretti, V., Kirrane, S., Di Ciccio, C.: Usage control for process discovery through a trusted execution environment. In: Service-Oriented Computing (ICSOC), pp. 268–286 (2026)

[30] Ouyang, C., Dumas, M., Aalst, W.M.P.V.D., Hofstede, A.H.M.T., Mendling, J.: From business process models to process-oriented software systems. ACM Trans. Softw. Eng. Methodol. 19(1) (2009)

[31] Nivon, Q., Sala¨un, G.: Automated generation of bpmn processes from textual requirements. In: Service-Oriented Computing (ICSOC), Singapore, pp. 185–201 (2025)

# Supplementary Information

Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

These supplementary sections give the secondary analyses and details referred to in the main paper. All p values come from the paired two-sided Wilcoxon signed-rank test over per-query scores, as in the main paper.

## S1 Tool Surface Per Condition

Table S1 compares the tools of the two agent conditions. Both use the same mock GET/POST layer and validator in RQ3 and RQ4. They difer only in the retrieval tool and what it returns. Neither condition has a separate schema or value-help tool. Our grounding context arrives inside the retrieval result, while ChunkRAG’s retrieval returns summaries only, so ChunkRAG cannot see value-help lists by design. The symmetric baseline in RQ4 (ChunkRAG+VH) adds the value-help lists as plain text to ChunkRAG’s retrieval result and changes nothing else.

We do not report a condition with raw tool use and no retrieval. Streaming the \$metadata XML straight into a tool-use prompt timed out in all runs, which would mix up format problems with system performance.

Table S1 Tool surface per condition, identical across RQ3 and RQ4.
<table><tr><td></td><td>ChunkRAG</td><td>Ours</td></tr><tr><td>Retrieval tool</td><td>maries [1]</td><td>Dense top-k search over EntitySet sum- Dense seeds + bounded process-chain expan- sion</td></tr><tr><td>Retrieval returns</td><td>EntitySet, service, and property names</td><td>Full typed specification: required keys, naviga- tion cardinality, value-help code lists, process- chain information</td></tr><tr><td>Execution</td><td colspan="2">GET and POST through the same mock HTTP layer and validator</td></tr></table>

## S2 Benchmark Construction and Annotation Protocol

The first version of the benchmark had 80 queries labelled by one annotator. We grew it to its current size with a two-annotator protocol. First, we generated a pool of 251 candidate queries automatically from the catalogue and the process graph. Each candidate came with machine-suggested labels and a note on how it was derived. Two annotators then reviewed the full pool independently, following a written protocol. Annotator A was a human author. Annotator B was a large language model (Claude, Anthropic); we disclose this openly and release its scripted per-candidate decisions with the repository.

Annotator A accepted 94 candidates, revised 156, and rejected 1. Annotator B accepted 100, revised 149, and rejected 2. The annotators agreed on whether to keep or reject 250 of the 251 candidates (99.6%), which gives Cohen’s κ = 0.665. The κ is only moderate despite near-perfect agreement because almost all candidates were kept. On the 249 candidates both kept, label agreement over the (query, EntitySet) pairs was κ = 1.000 (n = 415 pairs). Both annotators saw the machine-suggested labels, so this number measures agreement on corrections, not independent labelling.

We discussed the one keep/reject disagreement and dropped the query as unanswerable (it used an opaque internal entity-set name), so 2 candidates were dropped in total. Where the two accepted wordings difered, we kept the more natural one. We computed agreement before this resolution step and did not recompute it afterwards. The merge added 208 queries to the main benchmark and 41 to the elicitation probe. We rebuilt the process-chain ground truth from the graph instead of copying it from the drafts.

The benchmark was split deterministically into 150 training and 138 test queries. Some secondary agent analyses (Sec. S5) use an earlier 26-query multi-step subset, created before the benchmark grew.

## S3 Graph-Construction Fidelity Audit

An independent checker parses the source OData \$metadata and BPMN 2.0 files again and compares the result with the graph. We check each structural class: EntitySets, EntityTypes, Properties, Navigation edges (with cardinality), ValueHelp code lists, Activities, and HAS NEXT order. For each class we report coverage and exactness. Coverage is the share of source elements that appear in the graph. Exactness is the share of graph elements whose attributes match the source.

The checker was written separately from the ingestion pipeline and shares none of its parsing logic. It reads OData XML with namespace-qualified queries, while the pipeline matches local element names. It also has its own readers for the SAP catalogue exports and for BPMN.

Two scope notes matter. First, the SAP catalogue is available only as structured CSV exports from the SAP Business Accelerator Hub, not as raw XML, so the audit treats these exports as the source of record. For TripPin, Northwind, and Microsoft Graph, it checks the full path from XML to graph. Second, we compare ValueHelp lists with their hand-curated seed tables, because the ingested \$metadata has no machine-readable value-list annotations. The mock validator uses the same seed tables. The audit does not cover the hand-written USES ENTITY SET edges, because no independent source exists for them. The main paper assesses them with a provenance split (RQ1).

Table S2 Graph-construction fidelity audit. Coverage: share of distinct source Table S2 Graph-construction fidelity audit. Coverage: share of distinct source  
elements present in the graph. Exactness: share of graph elements whose attributes all match the source. The 79 activities are the 51 task activities plus 28 start and end events.
<table><tr><td>Structural class</td><td>Source</td><td>Graph</td><td>Coverage</td><td>Exactness</td></tr><tr><td>EntitySets</td><td>132</td><td>132</td><td>1.000</td><td>1.000</td></tr><tr><td>EntityTypes</td><td>1,270</td><td>1,270</td><td>1.000</td><td>1.000</td></tr><tr><td>Properties</td><td>8,481</td><td>8,481</td><td>1.000</td><td>1.000</td></tr><tr><td>Navigation edges (incl. cardinality)</td><td>1,499</td><td>1,499</td><td>1.000</td><td>1.000</td></tr><tr><td>Activities (BPMN)</td><td>79</td><td>79</td><td>1.000</td><td>1.000</td></tr><tr><td>HAS_NEXT (BPMN)</td><td>65</td><td>65</td><td>1.000</td><td>1.000</td></tr><tr><td>ValueHelp codes</td><td>26</td><td>26</td><td>1.000</td><td>1.000</td></tr></table>

Table S2 shows the result. Every structural class reaches full coverage and exactness. The audit found only three harmless kinds of diference, and each one comes from the source, not from ingestion. (i) For 70 of 8,481 properties, attribute strings difer only by a number or boolean format change from the table-loading step (key flag true → True; MaxLength 26 → 26.0). We count these as matches after normalization; they do not change meaning. (ii) The source tables have 29 duplicate rows. The loader merges each duplicate into one graph element, and the counts above are over distinct elements. The only attribute-level case is one SAP EntitySet listed twice with two spellings of its entity type; the graph keeps both type links, so nothing is lost. (iii) ComplexType definitions (1,300 in Microsoft Graph, 4 in TripPin) are out of scope by design. The entity model covers only types exposed through EntitySets and the types reachable from them by navigation.

## S4 Additional Retrieval Analyses (RQ1)

## S4.1 Where the gain comes from

To explain the Hybrid result, we label each gold entity in one of four ways: found by the dense seed, added by process expansion, pushed out by expansion, or missed. Table S3 gives these counts by query type.

Table S3 Gold-entity attribution for Hybrid at $k = 3$ (counts over the 304 gold entities of the test split). Seed: in the dense top-3. Exp.: only in the Hybrid top-3, added or moved up by process expansion. Displ.: in the dense top-3 but pushed out by expansion. Miss: in neither. Seed+Exp. equals Hybrid Recall@3 and Seed+Displ. equals Dense Recall@3, micro-averaged over gold entities.
<table><tr><td>Query type</td><td>Golds</td><td>Seed</td><td> $\mathrm { E x p . }$ </td><td> $\mathrm { D i s p l . }$ </td><td>Miss</td></tr><tr><td>Single entity</td><td>35</td><td>33</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Value-help dependent</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Navigation chain</td><td>49</td><td>44</td><td>0</td><td>1</td><td>4</td></tr><tr><td>Multi-step workflow</td><td>13</td><td>6</td><td>4</td><td>2</td><td>1</td></tr><tr><td>Process-chain goal</td><td>206</td><td>120</td><td>28</td><td>12</td><td>46</td></tr><tr><td>All</td><td>304</td><td>203</td><td>32</td><td>16</td><td>53</td></tr></table>

At $k = 3 .$ , expansion contributes 32 of the 304 gold entities: 28 in the process-chain probe and four in multi-step workflows. It adds nothing in query types without a process signal. Expansion also pushes 16 dense golds out of the top three. At $k = 1 0$ , the dense seed alone finds 271 golds, and the net gain from expansion falls to nine, with two pushed out. This matches the smaller recall gap at larger k in the main RQ1 table.

## S4.2 Completeness of the context passed to the agent

We also check whether each retrieved entity comes with all the context the agent needs. On the agent’s discovery path (top-5 seeds), the retrieved context is complete for 3/3 required payload keys, 1/1 valuehelp code lists, 75/85 navigation cardinalities on gold parent–child pairs, and $6 7 / 7 9$ process dependencies. Almost all gaps come from retrieval. In 34 of the 35 failing pairs, at least one entity is missing from the discovery result. Either no dense seed lands in the gold cluster (the SalesOrder cluster in six queries, the Microsoft Graph users cluster in two), or the discovery tool reports process dependencies only for pairs next to a retrieved seed, so second-hop chain members are left out. In the single exception, both entities are retrieved, but their dependency is reported only relative to the dense seeds. Whenever both entities of a required pair are retrieved, keys and cardinality are complete. The gaps are retrieval failures, not failures in building the context

## S4.3 Embedding sensitivity

With the smaller all-mpnet-base-v2 embedding instead of the 8B model, Hybrid still beats Dense on the probe at every k (Recall@3 0.515 vs. 0.507; Recall@10 +3.3 points). The diferences are smaller, however, and none is significant (all $p \geq 0 . 1 9 ;$ the paired probe test used in the main paper gives $p = 0 . 5 4 )$ Weaker seeds leave more queries out of reach of the graph expansion. The direction of the efect holds, but absolute scores are lower.

## S5 Additional Agent Analyses (RQ3)

The main paper reports RQ3 on the 81 multi-step test queries. Before the benchmark grew, we ran RQ3 on an earlier 26-query multi-step subset. Several secondary analyses (seed verification, oracle retrieval, and the sensitivity LLMs) were run only on this subset. We report them here.

## S5.1 Results on the subset and why they difer

Table S4 shows the subset results. With the primary LLM, ChunkRAG completed more tasks than our agent (0.295 vs. 0.192), but the diference was not significant $( p = 0 . 4 6 )$ . Our method made fewer invalid calls $( 4 / 7 8 ~ \mathrm { v s . ~ } 1 0 / 7 8 )$ . Two of our four invalid calls happen in one run where in-loop discovery missed the gold entity cluster and the agent guessed from its prior knowledge. The other two happen despite correct retrieval. One invents a property name on a correctly found entity set (PurReqnItemText on PurchaseReqnItem), and one posts SalesOrder under the Northwind service.

Table S4 Agent results on the earlier 26-query subset. “Invalid/run” is the mean number of invalid calls per run (GET \$metadata requests are not counted). gemma4:31b runs once per query; the other LLMs run three times. The qwen3.6 rows were collected on CPU-only serving with an earlier version of the system and mainly measure serving instability.
<table><tr><td>LLM</td><td>Cond.</td><td>Runs</td><td>Compl.</td><td>Calls/run</td><td>Invalid/run</td><td>Timeouts</td></tr><tr><td colspan="7">Primary LLM; difference −0.103, p = 0.461 (n = 26)</td></tr><tr><td>qwen3-coder:30b</td><td>ChunkRAG</td><td>78</td><td>0.295</td><td>2.12</td><td>0.128</td><td>0/78</td></tr><tr><td>qwen3-coder:30b</td><td>Ours</td><td>78</td><td>0.192</td><td>1.69</td><td>0.051</td><td>0/78</td></tr><tr><td colspan="7">Sensitivity LLMs</td></tr><tr><td>gemma4:31b</td><td>ChunkRAG</td><td>26</td><td>0.192</td><td>1.00</td><td>0.000</td><td>0/26</td></tr><tr><td>gemma4:31b</td><td>Ours</td><td>26</td><td>0.115</td><td>0.65</td><td>0.000</td><td>0/26</td></tr><tr><td>qwen3.6 (36B)</td><td>ChunkRAG</td><td>78</td><td>0.064</td><td>0.15</td><td>0.090</td><td>72/78</td></tr><tr><td>qwen3.6 (36B)</td><td>Ours</td><td>78</td><td>0.064</td><td>0.49</td><td>0.038</td><td>54/78</td></tr></table>

On the subset, a wrong seed was the main risk. The ofline retriever in RQ1 gets the benchmark query word for word, while the agent retrieves with its own rewording. If the rewording picks a wrong but similar-sounding EntitySet, process expansion adds the wrong chain, and the retrieve-then-execute design fails early. ChunkRAG shows flat endpoint text and may still reach a completing call. Process context therefore helps when the seed is right and makes things worse when it is wrong. On the subset, retrieval misses caused most failures for both methods: 40 of 55 ChunkRAG failures and 43 of 63 of ours. On the larger test set, which is mostly process-chain goals, ChunkRAG far more often makes no call at all, and this outweighs the wrong-seed risk.

## S5.2 Checking seeds before expansion

We tried the direct fix for committing to a wrong seed too early. Inside the discovery tool, the same LLM checks the dense seeds against the user query, and only verified seeds start process-chain expansion. Unverified seeds are still shown as plain specifications. On the subset (78 runs), verification removes invalid writes completely (0 invalid POST calls, vs. 3 for the unverified agent and 9 for ChunkRAG). The cost is more invalid exploratory reads (11 GET probes to service/entity pairs that do not exist, which the mock rejects harmlessly). Completion improves only slightly (0.205 vs. 0.192; ChunkRAG 0.295). The conditional rates explain why. Completion is 0.421 when in-loop discovery returned all gold entities and 0.000 when it did not. Verification can reject a wrong seed but cannot add a missing one, and the main failure is a missing seed, not a wrong one. Closing the completion gap therefore needs better seed coverage (larger candidate pools, query rewriting), not stricter filtering.

## S5.3 Separating execution from retrieval (oracle retrieval)

RQ3 mixes retrieval and execution failures. We therefore add an oracle condition in which the agent receives the gold specification for each query (perfect retrieval). Any remaining failure must then come from payload construction and tool use. We score required-key adherence, value-help adherence, correct nesting of 1:N navigations as arrays, and retries after the validator rejects a call, for both conditions under the same oracle context.

Under oracle retrieval (78 runs per condition, no timeouts), completion stays low for both agents (ours 0.256, ChunkRAG 0.359). The completion gap on the subset is thus not only a retrieval efect. It also reflects execution limits of the 30B LLM, mainly failures to use the tools and incomplete payloads. Payload construction still separates the two conditions. With the same gold context, ChunkRAG issues 15 schema-invalid writes, including 6 invented codes, while ours issues 3 and invents none. Required-key adherence is low for both (0.25 vs. 0.08). Only the typed prompt leads to correct nesting of 1:N children as arrays (6 of 29 posting runs vs. 0 of 22). Retrying after a rejected call is rare in both conditions (2 vs. 0 runs). Looking only at runs where retrieval succeeded gives the same picture. Over the 35 runs whose discovery returned every gold entity, completion is 0.43 (ours) vs. 0.57 (ChunkRAG), against 0.00 vs. 0.07 when it did not. Good retrieval is necessary for completion, but execution remains the main bottleneck.

## S5.4 Sensitivity LLMs and serving

We use the non-primary LLMs only to check external validity. gemma4:31b runs once per query (26 runs per condition, no timeouts). On the subset, its results point the same way as the primary LLM: higher ChunkRAG completion, and zero invalid calls for both. qwen3.6 hit the latency ceiling in 92% of ChunkRAG runs and 69% of our runs, so its rows mainly measure serving instability rather than agent behaviour. Local LLM serving can stall across condition switches when its cache is reused on CPU-only serving; restarting the server between conditions avoids this. On GPU serving (final runs), no restarts were needed. Our conclusions rest on the primary qwen3-coder:30b, which had 0 timeouts in 642 runs.

## S6 Mapping Authoring

## S6.1 Authoring procedure

Authoring followed a fixed procedure. For each BPMN task activity, the author (1) identifies the business documents the activity reads or writes, (2) finds the covering service in the catalogue, (3) checks the exact EntitySet names against that service’s \$metadata, and (4) records the (activity, service, EntitySet) triples in a version-controlled mapping file. Activities with no covering service are left unmapped and documented in the file.

For the timing study, one author familiar with the catalogue re-wrote a random sample (fixed seed) of ten mapped activities, covering 12 of the 45 edges, from scratch under a timer, using this procedure. The sample took 205 person-minutes (median 20.5 minutes per activity, range 16–25), and every re-written edge matched the released mapping.

## S6.2 LLM-proposed mapping

For each of the 14 processes, a local LLM (qwen3-coder:30b) receives the catalogue (services with their entity sets) and the ordered task activities, and proposes USES ENTITY SET edges. Against the 45 handwritten edges, the proposals reach precision 0.79 and recall 0.93 (F1 0.86). They recover 42 of the 45 gold edges. The 11 false positives are plausible mappings that a reviewer can reject quickly, because no service in the catalogue covers them (e.g., mapping Hire-to-Retire tasks to the Microsoft Graph users entity, or goods receipt to PurchaseOrderItem). Authoring the mapping thus becomes checking 53 proposed edges instead of writing 45 from scratch. The three misses (the TripPin flight-lookup entities and one quality-notification entity) show that domain review is still needed.

## S7 Per-Sector Results on SOCBench-D (RQ5)

The per-sector results are mixed. Hybrid is ahead in five sectors and behind in six. The diferences depend on how closely endpoint paths match the words in the queries. Without a process layer, the graph adds no consistent benefit, as the main paper predicts.

Table S5 RQ5 per-sector Recall@5 on SOCBench-D (Hybrid vs. Tokens), 10 queries per GICS sector.
<table><tr><td>Sector</td><td>Hybrid</td><td>Tokens</td><td>∆</td></tr><tr><td>01 Energy</td><td>0.346</td><td>0.258</td><td>+0.088</td></tr><tr><td>02 Materials</td><td>0.533</td><td>0.432</td><td>+0.101</td></tr><tr><td>03 Industrials</td><td>0.364</td><td>0.307</td><td>+0.058</td></tr><tr><td>04 Consumer discretionary</td><td>0.502</td><td>0.477</td><td>+0.025</td></tr><tr><td>05 Consumer staples</td><td>0.397</td><td>0.394</td><td>+0.003</td></tr><tr><td>06 Health care</td><td>0.339</td><td>0.403</td><td>-0.064</td></tr><tr><td>07 Financials</td><td>0.471</td><td>0.517</td><td>-0.046</td></tr><tr><td>08 Information technology</td><td>0.384</td><td>0.505</td><td>-0.121</td></tr><tr><td>09 Communication services</td><td>0.346</td><td>0.531</td><td>-0.184</td></tr><tr><td>10 Utilities</td><td>0.310</td><td>0.399</td><td>-0.089</td></tr><tr><td>11 Real estate</td><td>0.320</td><td>0.323</td><td>-0.003</td></tr></table>