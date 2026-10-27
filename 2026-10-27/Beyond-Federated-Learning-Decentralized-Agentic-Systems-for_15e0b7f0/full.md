# Beyond Federated Learning: Decentralized Agentic Systems for Self-Evolving Clinical Intelligence

Haoyuan Li, Mathias Funk and Aaqib Saeed

Eindhoven University of Technology, Eindhoven, The Netherlands.

\*Corresponding author(s). E-mail(s): h.y.li@tue.nl; Contributing authors: M.Funk@tue.nl; a.saeed@tue.nl;

## Abstract

Clinical expertise is inherently distributed: it resides not in any single dataset or institution but in the accumulated reasoning experiences of thousands of clinicians navigating diverse patient populations, resource constraints, and evolving treatment protocols. Yet current approaches to scaling medical AI, from centralized large language models to federated learning, remain anchored to a data-centric paradigm that treats knowledge as static parameters to be aggregated. In this Perspective, we argue for a fundamental shift from data aggregation to experience aggregation. We propose Decentralized Agentic Systems (DAS), a framework in which distributed AI agents operating across healthcare institutions synthesize collective clinical intelligence by sharing abstracted reasoning trajectories rather than raw data or model parameters. Unlike federated learning, which averages numerical gradients across homogeneous model architectures, DAS exchanges structured records of clinical reasoning, specifically the sequence of observations, hypotheses, tool invocations, and outcome feedback that constitute diagnostic experience. We examine the architectural principles, deployment considerations, governance requirements, and critical open challenges of this framework, laying the groundwork for a collaborative medical AI ecosystem where distributed expertise autonomously adapts to the complexities of longitudinal patient care.

Keywords: Agentic AI, Decentralized Learning, Digital Healthcare

## 1 Introduction

Medicine is, at its core, a distributed enterprise. The expertise required to manage a complex patient rarely resides in a single clinician, department, or institution. A rural primary care physician develops pattern recognition for atypical presentations under resource constraints that an academic specialist never encounters; a community cardiologist accumulates experience with medication titration across diverse socioeconomic contexts; a home health monitoring system learns to distinguish clinically significant decompensation from benign fluctuations in daily vital signs. This distributed expertise, forged through thousands of diagnostic encounters across heterogeneous clinical environments, represents the most valuable and least utilized knowledge asset in healthcare.

The rapid development of Large Language Models (LLMs) has contributed to growing interest in agentic Artificial Intelligence (AI), which seeks to extend AI systems beyond static information retrieval toward perception, reasoning, and action in dynamic environments. Prior work has reported promising results in complex domains such as software engineering, mathematical theorem proving, and scientific hypothesis generation [1–4]. In healthcare, where clinical reasoning demands synthesizing heterogeneous evidence under uncertainty, agentic AI promises to augment human expertise by orchestrating knowledge-intensive diagnostic and therapeutic workflows.

Early applications of agentic AI in medicine established foundational approaches to algorithmic clinical reasoning. Systems such as those in [5–8] employ LLMs as centralized orchestrators, using modular decomposition to partition clinical queries into sub-tasks spanning evidence retrieval, diferential diagnosis, and treatment recommendation. While efective for localized decision support, these systems operate as isolated instances constrained by static reasoning templates and single-institution deployment. Recent work has advanced toward hierarchical multi-agent systems enabling dynamic collaboration for symptom assessment, diagnostic interpretation, and treatment recommendation, simulating multi-turn clinical consultations [9–13].

However, a fundamental architectural tension pervades current systems: clinical expertise is inherently distributed across institutions, specialties, and care settings, yet agentic AI remains stubbornly centralized. This tension is deeper than the wellknown challenges of data fragmentation and privacy regulation, though it encompasses them. We identify three dimensions of this mismatch, each reflecting not merely a data problem but an expertise problem.

## Expertise fragmentation problem.

The conventional framing of healthcare data silos emphasizes regulatory barriers to data sharing. Though real, this framing obscures a more fundamental issue: even removing regulatory barriers would not pool clinical expertise. The diagnostic insight a rural emergency physician develops managing acute coronary syndrome without catheterization access is not encoded in the EHR; it resides in the physician’s reasoning process and outcomes observed. A centralized system trained on pooled records acquires statistical patterns but not the situated reasoning constituting clinical mastery. The challenge is therefore not merely to share data across institutions but to share the experience of navigating clinical complexity.

## Scaling limits of static knowledge.

The advancement of LLMs has been predicated on access to ever-larger corpora of human-generated text, yet recent analyses suggest this paradigm is approaching asymptotic limits as high-quality medical data becomes increasingly scarce [14, 15]. In healthcare, where rare conditions, evolving treatment protocols, and longitudinal patient trajectories demand continuous learning, reliance on static datasets creates systems that are perpetually outdated. More critically, static training data captures declarative medical knowledge such as facts, guidelines, and associations, but not the procedural knowledge of how to apply it under the uncertainty and time pressure of clinical practice.

## Sovereignty-utility tension.

Autonomous agents handling Protected Health Information (PHI) must comply with HIPAA, IVDR, and GDPR [16–18], yet compliance is only one dimension of a broader sovereignty challenge. Healthcare institutions increasingly assert governance over not only their data but their institutional knowledge (i.e., clinical protocols, reasoning conventions, and accumulated expertise) as competitive and intellectual assets. Current architectures ofer institutions a binary choice: contribute data to centralized systems or remain isolated. What is needed is an architecture that enables institutions to contribute to and benefit from collective intelligence while retaining sovereignty over their clinical assets, sharing insights without surrendering the experiences from which those insights derive.

These challenges are compounded by the computational realities of deployment. State-of-the-art agents rely on multi-model toolchains: EchoAgent [19] orchestrates domain-specific vision models for echocardiogram interpretation, while Echo-CoPilot [20] leverages EchoPrime [21] and PanEcho [22] for multi-view cardiac analysis. Centralizing such systems incurs substantial inference costs [23], while naive distribution faces severe infrastructure heterogeneity.

Such converging pressures cannot be addressed by existing distributed learning paradigms. Federated Learning (FL), the prevailing privacy-preserving approach, aggregates model parameters through gradient averaging, operating at the wrong level of abstraction for clinical expertise. Gradients encode statistical correlations rather than reasoning strategies, require architectural homogeneity, and provide no mechanism for transferring the procedural knowledge characterizing clinical mastery.

In this Perspective, we propose Decentralized Agentic Systems (DAS) as an alternative paradigm: a framework wherein distributed agents operating across healthcare institutions synthesize collective clinical intelligence through the sharing of abstracted reasoning trajectories, while concurrently self-evolving through the reciprocal incorporation of network-wide experiential knowledge into local reasoning capabilities. We formalize the architectural principles underlying DAS (Section 3), examine the deployment landscape and governance requirements for real-world adoption (Sections 3.5 and 4), and articulate the critical open challenges (Section 5).

## 2 Learning from Experience Beyond Human Data

The transition toward decentralized agentic architectures represents a necessary evolution aligned with fundamental shifts in how AI acquires knowledge. LLMs’ progress has been predicated on scaling human-curated datasets, a paradigm approaching asymptotic limits as high-quality clinical data grows scarce [14, 15]. In healthcare, this constraint is particularly acute: medical knowledge is continuously generated through diagnostic encounters, treatment adaptations, and clinical judgments across thousands of institutions, accessible only in part to any centralized system. We argue that the next frontier lies in enabling AI agents to learn directly from distributed clinical experiences, transforming ephemeral diagnostic interactions into durable, shareable intelligence.

Experience difers fundamentally from data. Data represents static records, laboratory values, imaging findings, documented diagnoses. Experience, by contrast, encompasses the dynamic process through which an agent perceives clinical signals, reasons about their significance under uncertainty, selects among possible interventions, and observes outcomes [24]. Consider the diference between a dataset of electrocardiograms labeled with diagnoses and the experience of an agent that detects a subtle ST-elevation pattern, retrieves relevant patient history, considers confounding factors, and refines its interpretation based on downstream clinical confirmation. The former provides declarative knowledge, facts about the world; the latter generates procedural knowledge, that is, how to navigate clinical reasoning itself.

Each institutional agent continuously generates experiential feedback through clinical interactions. A cardiology-focused agent may develop nuanced expertise in distinguishing Takotsubo cardiomyopathy from acute coronary syndrome in postmenopausal women; a primary care agent may learn to recognize atypical thyroid dysfunction in elderly populations. Such experiences represent the contextual, procedural knowledge unavailable in static datasets. The challenge lies in synthesizing this distributed intelligence into collective clinical wisdom without violating data sovereignty constraints.

To address this, we propose DAS as a framework for agent-centric insight aggregation. Unlike FL [25], which aggregates numerical model parameters through gradient averaging, DAS operates at a higher abstraction level: the sharing of semantic trajectories. We define a semantic trajectory as a structured record comprising an agent’s reasoning path: observations, hypotheses, tool invocations, and inferential steps, together with environmental feedback that validated or refuted its conclusions. Exchanging these traces rather than raw data or model weights enables knowledge transfer that is both privacy-conscious and semantically meaningful, forming the substrate through which individual agentic nodes continuously self-evolve: each agent refines its local reasoning capabilities by assimilating experiential knowledge distilled from across the distributed network.

## Why existing federated approaches are insuficient.

Gradient-based approaches assume homogeneity in model architectures, data distributions, and task definitions, while assumptions rarely hold across diverse healthcare settings. Semantic trajectories abstract away these implementation details to capture the core reasoning process. An insight such as “in patients presenting with chest pain and elevated troponin but atypical ECG findings, consider requesting strain echocardiography before concluding myocardial infarction” transfers meaningfully across institutional contexts regardless of underlying architectures or data distributions.

Table 1 Comparison of distributed learning paradigms for healthcare AI. DAS operates at a fundamentally diferent level of abstraction than parameter-based approaches. Superscript <sup>∗</sup> denotes partial or limited applicability.
<table><tr><td>Property</td><td>FedAvg</td><td>Federated Distillation</td><td>Personalized FL</td><td>DAS (Ours)</td></tr><tr><td>Unit of sharing</td><td>Gradients / weights</td><td>Soft predictions / logits</td><td>Adapted parameters</td><td>Reasoning trajectories</td></tr><tr><td>Model heterogeneity supported</td><td>x</td><td>x*</td><td>x</td><td>L</td></tr><tr><td>Captures procedural reasoning</td><td>x</td><td>x</td><td>x</td><td>1</td></tr><tr><td>Interpretability of shared knowledge</td><td>x</td><td>x*</td><td>x</td><td>(human- readable)</td></tr><tr><td>Cross-specialty transfer</td><td>x</td><td>x</td><td>√*</td><td>(semantic)</td></tr><tr><td>Auditability for clinical safety</td><td>x</td><td>x</td><td>x</td><td>(trajectory inspection)</td></tr><tr><td>Privacy risk vector</td><td>Gradient inversion</td><td>Output inversion</td><td>Gradient inversion</td><td>Semantic inference</td></tr></table>

DAS is distinct from FL variants that partially address the limitations of FedAvg [25]. Federated distillation [26, 27] exchanges soft predictions or logits, reducing architectural coupling. Split learning [28] partitions computation across clients and servers. Personalized FL [29, 30] accommodates heterogeneous data distributions through local adaptation. Each addresses a specific limitation of gradient-based aggregation, yet all share a common constraint: they operate at the level of model parameters or outputs, not at the level of reasoning.

Table 1 summarizes the key distinctions. The critical diference is representational: parameter-based approaches encode knowledge as numerical vectors requiring architectural compatibility. Semantic trajectories encode knowledge as structured reasoning sequences that are interpretable, auditable, and transferable across heterogeneous architectures. A trajectory recording reasoning from ambiguous ECG findings to a correct Wellens syndrome diagnosis transfers regardless of whether the receiving agent uses BioMistral, Med-PaLM, or a future architecture.

This comparison also highlights that DAS introduces its own privacy risk vector, namely semantic inference from reasoning patterns, which difers qualitatively from gradient inversion attacks and demands purpose-built mitigation strategies (Section 5.3).

## Relationship to adjacent paradigms.

DAS relates $^ { \mathrm { t o , } }$ but difers from, several active research directions. Retrieval-Augmented Generation (RAG) [31] enhances LLM reasoning by retrieving relevant documents; DAS can be viewed as a federated extension of RAG where the retrieval corpus consists of validated reasoning trajectories from distributed agents. Chain-ofthought distillation [32, 33] transfers reasoning via training on reasoning traces; DAS extends this from a teacher-student relationship to a network of peer agents sharing experiential knowledge. Experience replay in multi-agent reinforcement learning [34] enables learning from stored interactions; DAS adapts this to healthcare where experiences comprise clinical reasoning under real-world constraints.

The aggregation of validated semantic trajectories enables what we term collective self-evolution: the emergence of shared diagnostic intelligence exceeding any individual agent’s capabilities. This evolution occurs not through centralized retraining on pooled data but through continuous refinement of reasoning strategies based on crossinstitutional experience. When an agent successfully navigates a novel presentation to correct a diagnosis, that trajectory becomes available network-wide, enriching the collective repertoire of clinical problem-solving once abstracted and validated.

## 3 Architectural Foundations of Decentralized Agentic Systems

We now formalize the DAS architecture. The central challenge is not merely distributing computation but enabling meaningful knowledge synthesis across agents operating in heterogeneous clinical contexts: diferent patient populations, institutional protocols, specialty focuses, and infrastructure capabilities. As illustrated in Figure 1, DAS addresses this through three architectural layers: distributed agentic nodes that generate clinical experiences locally, a semantic abstraction layer that transforms heterogeneous experiences into transferable representations, and a collective intelligence layer that aggregates and redistributes insights across the agent network.

## 3.1 Distributed Agentic Nodes

The foundational unit of DAS is the distributed agentic node, an autonomous agent tailored to local clinical requirements. Formally, we define node $\mathcal { A } _ { k }$ for institution or deployment context $k \in \{ 1 , \ldots , N \}$ as a tuple:

$$
\mathcal { A } _ { k } = \langle \mathcal { M } _ { k } , \mathcal { T } _ { k } , \mathcal { E } _ { k } , \mathcal { C } _ { k } \rangle\tag{1}
$$

where $\mathcal { M } _ { k }$ denotes the local reasoning backbone (an LLM adapted to the node’s clinical specialty and institutional knowledge), $\mathcal { T } _ { k }$ represents the executable tool suite, $\mathcal { E } _ { k }$ is the local experience bufer storing interaction trajectories, and $\mathcal { C } _ { k }$ encodes local context including patient population characteristics, institutional protocols, and regulatory constraints.

![](images/ac48cffbfc10ab7e67178d8b148bd5327b889a42303b6d51e0c463058b4ca289.jpg)  
Fig. 1 Architecture of Decentralized Agentic Systems (DAS). The framework comprises distributed agentic nodes operating within local clinical contexts, connected through a semantic abstraction layer that transforms raw experiences into standardized reasoning trajectories. The collective intelligence layer aggregates these trajectories to synthesize shared clinical knowledge, which is subsequently distributed back to enhance individual agent capabilities.

Each agentic node operates through an iterative perception-reasoning-action cycle. At each step t, the reasoning backbone $\mathcal { M } _ { k }$ receives a clinical observation $o _ { t } ,$ which may include patient symptoms, diagnostic results, or queries from clinical staf. It generates a reasoning trace $r _ { t }$ explicating its inferential process and selects an action $a _ { t } \in \mathcal T _ { k }$ for execution. The environment returns feedback $f _ { t }$ , comprising tool outputs, downstream clinical outcomes, or expert corrections. This interaction constitutes an experience tuple:

$$
e _ { t } ^ { k } = \langle o _ { t } , r _ { t } , a _ { t } , f _ { t } , c _ { t } \rangle\tag{2}
$$

where $c _ { t }$ captures relevant contextual metadata, and experiences accumulate as an ordered sequence:

$$
\mathcal { E } _ { k } = \{ e _ { 1 } ^ { k } , e _ { 2 } ^ { k } , \ldots , e _ { T } ^ { k } \}\tag{3}
$$

Raw experiences remain locally stored, preserving data sovereignty and enabling compliance with institutional data governance policies.

## 3.2 Semantic Abstraction Layer

As experiences accumulate across distributed nodes, heterogeneity in clinical contexts, observation modalities, and action semantics necessitates transformation before

knowledge can be meaningfully shared. We introduce the Semantic Abstraction Layer, denoted Φ, implementing a two-stage pipeline: experience distillation and trajectory standardization.

## Experience Distillation.

Raw interaction trajectories contain substantial redundancy: routine observations, exploratory actions that yielded no insight, and noise from environmental stochasticity. The distillation function $\phi _ { \mathrm { d i s t } } : { \mathcal { E } } _ { k } \to { \mathcal { E } } _ { k } ^ { \prime }$ filters experiences to retain those carrying significant epistemic value:

$$
\mathcal { E } _ { k } ^ { \prime } = \phi _ { \mathrm { d i s t } } ( \mathcal { E } _ { k } ) = \{ e _ { t } ^ { k } \in \mathcal { E } _ { k } ~ | ~ \mathcal { U } ( e _ { t } ^ { k } ; \mathcal { C } _ { k } ) > \delta _ { k } \}\tag{4}
$$

where $\mathcal { U } ( \cdot )$ quantifies epistemic value across dimensions of task success, novelty, error correction, and generalization likelihood. The threshold $\delta _ { k }$ may be context-dependent, reflecting institutional priorities and resource constraints.

The design of $\mathcal { U } ( \cdot )$ is an open challenge. We envision hybrid approaches combining automated metrics (e.g., downstream outcome verification) with expert validation, where clinicians flag experiences of particular educational value, analogous to casebased learning in medical education.

## Trajectory Standardization.

Following distillation, retained experiences are transformed into a canonical representation enabling cross-context learning. The standardization function ϕ : $\mathcal { E } _ { k } ^ { \prime } \to S _ { k }$ maps heterogeneous local experiences into semantic trajectories:

$$
\mathcal { S } _ { k } = \phi _ { \mathrm { s t d } } ( \mathcal { E } _ { k } ^ { \prime } ) = \{ \sigma _ { i } \} _ { i = 1 } ^ { | \mathcal { E } _ { k } ^ { \prime } | } , \quad \sigma _ { i } = ( o _ { i } ^ { \mathrm { a b s } } , r _ { i } ^ { \mathrm { a b s } } , a _ { i } ^ { \mathrm { a b s } } , f _ { i } ^ { \mathrm { a b s } } )\tag{5}
$$

where each component is abstracted into a domain-general representation. Observations are mapped to standardized clinical concepts (e.g., SNOMED-CT or ICD), reasoning traces are structured according to a common inferential schema, actions are categorized by functional type rather than implementation-specific tool calls, and feedback is encoded as outcome categories with confidence indicators. This standardization enables cross-domain semantic alignment while removing institution-specific identifiers and abstracting patient-level details into population-level patterns.

## 3.3 Collective Intelligence through Federated Trajectory Learning

The collective intelligence layer synthesizes distributed knowledge through Federated Trajectory Learning, aggregating standardized semantic trajectories that preserve semantic content rather than averaging model parameters.

Let $ { \mathcal { S } } = \{  { S _ { 1 } } , \ldots ,  { S _ { N } } \}$ denote the global trajectory repository, where each $S _ { k }$ represents standardized trajectories contributed by node $\boldsymbol { \mathcal { A } } _ { k }$

$$
{ \mathcal S } _ { k } = \Phi ( { \mathcal E } _ { k } ) = \phi _ { \mathrm { s t d } } ( \phi _ { \mathrm { d i s t } } ( { \mathcal E } _ { k } ) )\tag{6}
$$

The collective intelligence mechanism $\mathcal { G }$ updates a shared knowledge representation $\theta _ { \mathrm { g l o b a l } } \colon$

$$
\theta _ { \mathrm { g l o b a l } } ^ { ( t + 1 ) } = \mathcal { G } \left( \theta _ { \mathrm { g l o b a l } } ^ { ( t ) } , \{ S _ { k } ^ { ( t ) } \} _ { k = 1 } ^ { N } \right)\tag{7}
$$

G may be instantiated through trajectory-derived insight aggregation, retrievalaugmented reasoning, clinical knowledge graph construction, or distillation into updated reasoning templates, accommodating diverse strategies without prescribing a single formulation. This architecture supports asynchronous learning: nodes contribute trajectories continuously, $\theta _ { \mathrm { g l o b a l } }$ is refined without synchronized participation, and updated knowledge enriches individual $\mathcal { M } _ { k }$ with network-wide insights.

## 3.4 Self-Evolution of Distributed Agentic Nodes

The emergence of collective intelligence creates a reciprocal feedback loop through which individual agentic nodes continuously self-evolve. Formally, node $\mathcal { A } _ { k }$ updates its local reasoning backbone by incorporating knowledge distilled from the global trajectory repository:

$$
\mathcal { M } _ { k } ^ { ( t + 1 ) } = \mathcal { F } \Big ( \mathcal { M } _ { k } ^ { ( t ) } , \ \mathcal { R } \Big ( \theta _ { \mathrm { g l o b a l } } ^ { ( t ) } , \mathcal { C } _ { k } \Big ) \Big ) ,\tag{8}
$$

where $\mathcal { R } ( \cdot )$ is a context-conditioned retrieval function that selects trajectories from $\theta _ { \mathrm { g l o b a l } }$ relevant to the node’s local clinical context $\mathcal { C } _ { k }$ , and $\mathcal F ( \cdot )$ integrates retrieved knowledge into existing reasoning capabilities through in-context augmentation [35], fine-tuning via reinforcement learning from human or AI feedback (e.g., RLHF [36, 37], DPO [38]), or reasoning template refinement.

This self-evolution operates along two complementary axes. Breadth expansion occurs as nodes assimilate reasoning trajectories from clinical domains beyond their local experience: a primary care agent encountering a rare cardiac presentation can retrieve and internalize diagnostic pathways developed by specialized cardiology agents elsewhere in the network. Depth refinement occurs as nodes receive corrective trajectories that challenge or supersede their existing reasoning patterns, enabling continuous calibration of diagnostic confidence and decision thresholds. As a result, each node converges toward a locally specialized yet globally informed reasoning profile, one that reflects both its accumulated clinical experience and the collective mastery of the distributed network.

## 3.5 Deployment Landscape across Healthcare Contexts

The DAS framework supports deployment at multiple scales of clinical organization, as illustrated in Figure 2.

At the point-of-care level, agentic nodes operate within clinical workflows $( { \mathrm { i . e . } }$ emergency department triage, outpatient consultation, and surgical planning) providing real-time decision support and accumulating fine-grained experiences from direct clinical interactions.

At the institutional level, DAS enables coordination across departments and specialties within a single healthcare organization. A hospital-wide deployment might include specialized agents for cardiology, oncology, primary care, and radiology, each developing domain-specific expertise while contributing to shared institutional knowledge, supporting quality improvement and dissemination of best practices.

![](images/9ac275e5596ca2020a4530cd03e9d64d044657e7d9b94f05a99f441b72838761.jpg)  
Fig. 2 Deployment landscape of the DAS framework across healthcare contexts. The architecture supports deployment across three scales: point-of-care agents providing real-time clinical decision support, institutional agents coordinating departmental expertise, and network-level aggregation enabling multi-center collaborative intelligence.

At the network level, DAS enables secure multi-institutional collaboration among federations of hospitals, academic medical centers, and research institutions while maintaining data sovereignty. This scale is particularly valuable for rare disease diagnosis, where individual institutions encounter insuficient case volumes, and for multi-center clinical research, where distributed agents can identify eligible patients and support protocol adherence.

Across all scales, the semantic abstraction layer ensures knowledge sharing occurs at the level of clinical reasoning rather than raw patient data. We illustrate this through two clinical scenarios: heart failure management across distributed care settings (Box 3.1) and rare disease diagnosis across a hospital network (Box 4.1).

## Box 3.1: Illustrative Scenario: Distributed Heart Failure Management

Consider a network of agents supporting heart failure care across three contexts: a rural primary care clinic, an urban cardiology practice, and a home monitoring system.

Local Experience Generation. The primary care agent learns to recognize atypical heart failure presentations in patients with limited access to echocardiography, developing expertise in risk stratification using symptoms and basic labs. The cardiology agent accumulates experience with advanced imaging interpretation and medication titration. The home monitoring agent learns to identify early decompensation patterns from daily weight trends and activity data.

Semantic Abstraction. Each agent’s experiences are distilled into transferable trajectories. The primary care agent’s identification of heart failure in a patient presenting with fatigue and subtle edema becomes: “In patients with diabetes and renal impairment presenting with nonspecific fatigue, consider heart failure even without classic dyspnea.”

Collective Intelligence and Node Self-Evolution. When the home monitoring agent encounters an ambiguous pattern of modest weight gain with preserved activity levels, it retrieves relevant trajectories from the collective knowledge base and integrates them into its local reasoning backbone, expanding its decompensation recognition beyond patterns observed in its own deployment context. Reciprocally, the primary care and cardiology agents assimilate the monitoring agent’s longitudinal decompensation trajectories, refining their own early-warning reasoning. This scenario illustrates how DAS enables knowledge sharing impossible under data-centralization constraints: the rural clinic cannot share patient records with the urban practice, but all three nodes continuously contribute to and self-evolve through collective clinical reasoning.

## 4 Governance, Regulation, and Deployment Realities

The technical architecture of DAS is necessary but insuficient for real-world adoption. Healthcare AI systems operate within complex governance ecosystems involving institutional policies, regulatory frameworks, professional liability structures, and patient trust relationships. We examine the non-technical dimensions that will determine whether decentralized agentic systems can transition from architectural concept to deployed clinical infrastructure.

## 4.1 Regulatory Considerations

Agentic AI systems in healthcare face an evolving regulatory landscape. In the United States, the FDA’s framework for Software as a Medical Device (SaMD) was designed for static algorithms with well-defined inputs and outputs, a paradigm poorly suited to agentic systems that reason dynamically and adapt over time [39]. Key open questions include whether each institutional agent requires independent regulatory clearance or whether DAS can be evaluated as a system; how continuous learning from trajectory updates interacts with the FDA’s predetermined change control plan requirements; and what validation standards apply to collectively derived reasoning patterns emerging from network-level aggregation rather than explicit design.

In the EU, the AI Act classifies most clinical decision-support systems as high-risk, requiring transparency, human oversight, and robustness testing. DAS creates novel compliance challenges: when reasoning reflects contributions from agents across multiple member states, which jurisdiction governs? These questions must be confronted proactively with regulatory bodies.

## Box 4.1: Illustrative Scenario: Rare Disease Diagnosis across a Hospital Network

Consider a network of 15 academic medical centers, each encountering a small number of patients with hereditary transthyretin amyloidosis (hATTR), a rare, progressive condition frequently misdiagnosed as hypertensive heart disease or idiopathic cardiomyopathy.

Local Experience Generation. At institution A, a cardiology agent encounters a 65-year-old patient with unexplained left ventricular hypertrophy and bilateral carpal tunnel syndrome. After initially pursuing a standard hypertensive cardiomyopathy workup, the agent, prompted by subtle electrodiagnostic findings, requests a technetium pyrophosphate scan, leading to a correct hATTR diagnosis. At institution B, a neurology agent identifies a pattern linking peripheral neuropathy progression with cardiac biomarker elevation in three patients, eventually connecting the presentation to hATTR.

Semantic Abstraction. The cardiology agent’s reasoning path is distilled into a transferable trajectory: “In patients with unexplained LVH and concurrent carpal tunnel syndrome or peripheral neuropathy, particularly over age 60, consider technetium pyrophosphate scan to evaluate for cardiac amyloidosis before concluding hypertensive or idiopathic cardiomyopathy.” The neurology agent contributes a complementary trajectory linking peripheral and cardiac manifestations.

Collective Intelligence and Node Self-Evolution. When a primary care agent at institution C encounters a patient with progressive neuropathy and mildly elevated troponin, it retrieves these cross-institutional trajectories, flags the possibility of hATTR, and integrates this diagnostic pathway into its local reasoning backbone. Simultaneously, the cardiology and neurology agents at institutions A and B assimilate each other’s complementary trajectories, refining their own cross-domain recognition of hATTR. This scenario illustrates how DAS addresses a critical gap in rare disease management: no single institution accumulates suficient case volume to develop robust diagnostic expertise, but through collective intelligence and reciprocal self-evolution, the network continuously elevates diagnostic capacity across all participating nodes.

## 4.2 Institutional Governance and Trust Models

Beyond formal regulation, DAS requires governance structures addressing participation, liability, and reciprocity across three dimensions.

## Trajectory curation and quality control.

Who decides which reasoning trajectories enter the collective knowledge base? Uncurated aggregation risks knowledge degradation (see Section 5.2), while overly restrictive curation may stifle the diversity that makes collective intelligence valuable. We envision hybrid models combining automated quality metrics (outcome validation, internal consistency, novelty assessment) with periodic expert review boards analogous to clinical peer review committees.

## Liability allocation.

When a recommendation informed by cross-institutional trajectories leads to an adverse outcome, liability is nontrivial. Current frameworks struggle with distributed AI even in non-agentic contexts [40]. DAS liability models might draw on multi-site clinical guideline precedent, where contributing institutions share collective responsibility for quality while individual providers retain responsibility for application to specific patients.

## Participation incentives and fairness.

Institutions must perceive suficient benefit to justify participation costs, including computational resources, clinical time for quality validation, and governance compliance overhead. Free-rider problems, where institutions consume collective knowledge without contributing proportionally, can erode trust. Contribution-weighted access to collective intelligence and transparent accounting of trajectory utilization may help align incentives. Smaller or less-resourced institutions should not be systematically disadvantaged; enabling rural and community hospitals to benefit from academic medical center expertise is a primary motivation for DAS.

## 4.3 Integration with Clinical Infrastructure

Practical deployment requires integration with EHR platforms. Interoperability standards such as HL7 FHIR provide structured interfaces for clinical observations and decision support. However, institutional variation in EHR configuration, documentation practices, and workflow integration means the operational context surrounding each agent difers substantially, afecting experience quality and applicability of received knowledge.

We advocate beginning with standardized tasks: clinical risk scoring, medication interaction checking, or diagnostic workup recommendations, before extending to complex reasoning, enabling iterative validation of trajectory sharing while building institutional trust through demonstrated value.

## 5 Technical Challenges and Open Research Directions

The architecture of Section 3 provides foundational mechanisms for decentralized agentic collaboration, yet several substantial technical challenges remain unresolved, which we examine as follows.

## 5.1 Experience Heterogeneity across Clinical Contexts

Decentralized healthcare systems must reconcile experiences generated across radically diferent clinical environments. A diagnostic agent in a tertiary academic medical center encounters diferent patient populations, diagnostic modalities, and institutional protocols than one in a rural primary care clinic. This heterogeneity manifests across three dimensions.

Statistical heterogeneity arises from diferences in patient demographics and disease prevalence. An agent serving a pediatric oncology unit accumulates experiences with childhood malignancies of limited relevance to one in a geriatric care facility, and direct aggregation across such disparate populations risks distributional biases that degrade diagnostic accuracy.

Semantic heterogeneity emerges from diferences in clinical vocabulary, documentation practices, and reasoning conventions. The same finding may be described diferently by cardiologists and internists; the same conclusion may be reached through diferent inferential pathways depending on institutional culture. Standardization eforts (as in Equation 5) must navigate the tension between abstraction suficient for cross-context transfer and preservation of clinically meaningful nuance.

Operational heterogeneity reflects diferences in available tools, computational resources, and workflow integration. An agent with access to advanced imaging analysis generates fundamentally diferent trajectories than one limited to laboratory values and clinical notes.

The Semantic Abstraction Layer addresses heterogeneity through trajectory standardization and utility-weighted aggregation. However, determining abstraction levels that enable transfer while preserving clinical validity remains empirically open, and the contribution weights $w _ { k }$ must reflect source-to-target context relevance, requiring either explicit clinical metadata or learned representations of transferability.

## 5.2 Hallucination Propagation and Byzantine Resilience

Large language models exhibit well-documented tendencies toward hallucination [41– 43]. In agentic systems, this vulnerability extends to entire reasoning trajectories: an agent may construct internally coherent diagnostic pathways based on fabricated associations or nonexistent evidence, and when such trajectories enter the sharing mechanism they risk corrupting collective knowledge network-wide.

This represents a novel form of the classical Byzantine failure problem [44, 45]. Agentic hallucination difers critically: hallucinated trajectories may be superficially plausible, evading detection mechanisms designed for obviously erroneous outputs; hallucination propensity may correlate across agents sharing similar architectures, violating independence assumptions underlying Byzantine protocols; and clinical consequences, including incorrect diagnoses and inappropriate treatments, may be severe and dificult to reverse.

The utility threshold in experience distillation (Equation 4) ofers partial mitigation but presupposes reliable utility assessment when ground truth outcomes may be delayed. More robust approaches might incorporate multi-agent verification or external validation against established clinical knowledge bases. The governance structures in Section 4, particularly expert trajectory curation, provide a complementary non-technical defense. The optimal balance between automated and human-mediated quality control remains context-dependent, with high-stakes domains such as oncology warranting more stringent validation.

## 5.3 Semantic Privacy Leakage from Abstracted Trajectories

A foundational premise of decentralized learning is that sharing derived representations preserves privacy. However, gradient inversion attacks have demonstrated that model updates can leak substantial information about training data [46, 47], enabling reconstruction of sensitive clinical images from federated exchanges [48, 49].

This vulnerability extends to trajectory sharing. A reasoning path comprising specific tool invocations, diagnostic queries, and decision sequences constitutes a behavioral fingerprint: even without explicit patient identifiers, trajectories querying rare disease databases or following condition-specific diagnostic pathways may reveal diagnoses, genetic predispositions, or lifestyle factors patients expect to remain confidential.

The standardization function provides partial protection, but the tension between utility and privacy persists: suficiently abstracted trajectories may lose the clinical specificity that makes them valuable. Developing formal privacy guarantees, potentially drawing on diferential privacy frameworks adapted to structured reasoning representations, is an important open challenge. The appropriate tradeof likely varies across contexts, suggesting the need for configurable mechanisms rather than uniform solutions.

## 5.4 Coordination Complexity in Asynchronous Multi-Agent Systems

DAS supports asynchronous operation, allowing agents to contribute trajectories and receive updates without synchronized participation. This flexibility is essential across institutions with diferent schedules and connectivity constraints, but introduces challenges for coherent collective intelligence.

Temporal inconsistency arises when agents operate on diferent knowledge versions. Recently updated agents generate experiences reflecting current collective understanding, while those with stale knowledge produce trajectories based on outdated patterns, requiring careful versioning and potentially discounting stale inputs.

Feedback delays complicate credit assignment. Clinical outcomes validating or refuting diagnostic reasoning may emerge weeks or months after the initial interaction, by which point collective knowledge may have evolved substantially, obscuring attribution of learning signal to specific contributions.

Coordination overhead scales with network size. As participating agents grow, mechanisms for trajectory aggregation and knowledge distribution must remain computationally tractable, and ensuring asynchronous updates do not introduce inconsistencies afecting patient care requires careful protocol design and potentially formal verification.

## 5.5 Evaluating Collective Intelligence

A challenge crosscutting all others is the absence of established evaluation methodologies. Conventional metrics such as accuracy, sensitivity, and specificity apply to individual agents but do not capture the value added by trajectory sharing. We identify three dimensions future work must operationalize.

## Marginal improvement from collective knowledge.

Does an agent with access to collective trajectories outperform the same agent in isolation? This requires controlled comparisons measuring diagnostic accuracy and reasoning quality: does the agent consider a broader diferential and reach correct diagnoses more eficiently?

## Cross-context transfer efectiveness.

Do trajectories from one context (e.g., academic cardiology) improve reasoning in another (e.g., rural primary care)? Transfer efectiveness likely varies across domains, and characterizing this landscape is essential for deployment decisions.

## Longitudinal learning dynamics.

Does collective intelligence improve as the trajectory repository grows, or do saturation and diminishing returns emerge? Longitudinal evaluation requires sustained deployment in realistic clinical environments, a significant but essential validation step.

We recommend that early DAS implementations incorporate prospective evaluation protocols from the outset, including pre-registered hypotheses and clearly defined metrics for success and failure.

## Closing Remarks

The future of healthcare AI lies not in monolithic systems that encode medical knowledge centrally, but in networks of specialized agents learning continuously from clinical experience and sharing insights in privacy-preserving ways. We have proposed Decentralized Agentic Systems as an architectural foundation for this future, one that respects the distributed nature of clinical expertise, the sovereignty of institutional knowledge, and the safety imperatives of patient care. The challenges are substantial, and we have deliberately identified open questions alongside our proposals. The converging pressures of data scarcity, institutional fragmentation, and the maturation of agentic AI capabilities make this a timely moment to chart a path toward collaborative clinical intelligence. The architecture of that collaboration matters, and we believe the principles articulated here: experience over data, reasoning over parameters, and governance alongside technology, point toward a productive direction.

## Funding Statement

This work was supported by the NGF AiNed Fellowship Grant of A.S.

## References

[1] Yang, J., Jimenez, C.E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O.: Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems 37, 50528–50652 (2024)

[2] Baba, K., Liu, C., Kurita, S., Sannai, A.: Prover agent: An agent-based framework for formal mathematical proofs. arXiv preprint arXiv:2506.19923 (2025)

[3] Gottweis, J., Weng, W.-H., Daryin, A., Tu, T., Palepu, A., Sirkovic, P., Myaskovsky, A., Weissenberger, F., Rong, K., Tanno, R., et al.: Towards an ai co-scientist. arXiv preprint arXiv:2502.18864 (2025)

[4] Lange, R.T., Sun, Q., Prasad, A., Faldor, M., Tang, Y., Ha, D.: Towards robust agentic cuda kernel benchmarking, verification, and optimization. arXiv preprint arXiv:2509.14279 (2025)

[5] Ferber, D., El Nahhas, O.S., W¨olflein, G., Wiest, I.C., Clusmann, J., Leßmann, M.-E., Foersch, S., Lammert, J., Tschochohei, M., J¨ager, D., et al.: Development and validation of an autonomous artificial intelligence agent for clinical decisionmaking in oncology. Nature cancer, 1–13 (2025)

[6] Tang, X., Zou, A., Zhang, Z., Li, Z., Zhao, Y., Zhang, X., Cohan, A., Gerstein, M.: Medagents: Large language models as collaborators for zero-shot medical reasoning. In: Findings of the Association for Computational Linguistics: ACL 2024, pp. 599–621 (2024)

[7] Rose, D.P., Hung, C.-C., Lepri, M., Alqassem, I., Gashteovski, K., Lawrence, C.: Meddxagent: A unified modular agent framework for explainable automatic differential diagnosis. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13803–13826 (2025)

[8] Hao, Y., Holmes, J., Waddle, M.R., Davis, B.J., Yu, N.Y., Vickers, K.S., Preston, H., Margolin, D., L¨ockenhof, C.E., Vashistha, A., et al.: Personalizing prostate cancer education for patients using an ehr-integrated llm agent. npj Digital Medicine 8(1), 770 (2025)

[9] Almansoori, M., Kumar, K., Cholakkal, H.: Medagentsim: Self-evolving multiagent simulations for realistic clinical interactions. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 362–372 (2025). Springer

[10] Zhang, W., Qiao, M., Zang, C., Niederer, S., Matthews, P.M., Bai, W., Kainz, B.: Multi-agent reasoning for cardiovascular imaging phenotype analysis. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 429–439 (2025). Springer

[11] Qin, Y., Gamage Nanayakkara, D.S., Li, X.: Multi-agent collaboration for integrating echocardiography expertise in multi-modal large language models. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 358–368 (2025). Springer

[12] Kim, Y., Park, C., Jeong, H., Chan, Y.S., Xu, X., McDuf, D., Lee, H., Ghassemi, M., Breazeal, C., Park, H.W.: Mdagents: An adaptive collaboration of llms for medical decision-making. Advances in Neural Information Processing Systems 37, 79410–79452 (2024)

[13] Li, R., Wang, X., Berlowitz, D., Mez, J., Lin, H., Yu, H.: Care-ad: a multiagent large language model framework for alzheimer’s disease prediction using longitudinal clinical notes. npj Digital Medicine 8(1), 541 (2025)

[14] Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., Hobbhahn, M.: Position: Will we run out of data? limits of llm scaling based on human-generated data. In: Forty-first International Conference on Machine Learning (2024)

[15] Muennighof, N., Rush, A., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., Rafel, C.A.: Scaling data-constrained language models. Advances in Neural Information Processing Systems 36, 50358–50376 (2023)

[16] Neupane, S., Mittal, S., Rahimi, S.: Towards a hipaa compliant agentic ai system in healthcare. arXiv preprint arXiv:2504.17669 (2025)

[17] M¨uller, H., Holzinger, A., Plass, M., Brcic, L., Stumptner, C., Zatloukal, K.: Explainability and causability for artificial intelligence-supported medical image analysis in the context of the european in vitro diagnostic regulation. New Biotechnology 70, 67–72 (2022)

[18] Voigt, P., Bussche, A.: The eu general data protection regulation (gdpr). A practical guide, 1st ed., Cham: Springer International Publishing 10(3152676), 10–5555 (2017)

[19] Daghyani, M., Wang, L., Hashemi, N., Medhat, B., Abdelsamad, B., Velez, E.R., Li, X., Tsang, M.Y., Luong, C., Tsang, T.S., et al.: Echoagent: Guideline-centric reasoning agent for echocardiography measurement and interpretation. arXiv

[20] Heidari, M., Roohi, M.A., Khosravi, A., Hacihaliloglu, I.: Echo-copilot: A multiview, multi-task agent for echocardiography interpretation and reporting. arXiv preprint arXiv:2512.09944 (2025)

[21] Vukadinovic, M., Chiu, I.-M., Tang, X., Yuan, N., Chen, T.-Y., Cheng, P., Li, D., Cheng, S., He, B., Ouyang, D.: Comprehensive echocardiogram evaluation with view primed vision language ai. Nature, 1–3 (2025)

[22] Holste, G., Oikonomou, E.K., Tokodi, M., Kov´acs, A., Wang, Z., Khera, R.: Panecho: Complete ai-enabled echocardiography interpretation with multi-task deep learning. medRxiv, 2024–11 (2025)

[23] Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., et al.: Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618 (2025)

[24] Silver, D., Sutton, R.S.: Welcome to the era of experience. Google AI 1, 11 (2025)

[25] McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.: Communication-eficient learning of deep networks from decentralized data. In: Artificial Intelligence and Statistics, pp. 1273–1282 (2017). PMLR

[26] Lin, T., Kong, L., Stich, S.U., Jaggi, M.: Ensemble distillation for robust model fusion in federated learning. In: NeurIPS (2020)

[27] He, C., Annavaram, M., Avestimehr, S.: Group knowledge transfer: Federated learning of large cnns at the edge. NeurIPS (2020)

[28] Gupta, O., Raskar, R.: Distributed learning of deep neural network over multiple agents. Journal of Network and Computer Applications (2018)

[29] T Dinh, C., Tran, N., Nguyen, J.: Personalized federated learning with Moreau envelopes. NeurIPS (2020)

[30] Fallah, A., Mokhtari, A., Ozdaglar, A.: Personalized federated learning with metalearning. In: NeurIPS (2020)

[31] Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: NeurIPS (2020)

[32] Ho, N., Schmid, L., Yun, S.-Y.: Large language models are reasoning teachers. In: ACL (2023)

[33] Magister, L.C., Mallinson, J., Adamek, J., et al.: Teaching small language models to reason. In: ACL (2023)

[34] Foerster, J., Nardelli, N., Farquhar, G., et al.: Stabilising experience replay for deep multi-agent reinforcement learning. In: ICML (2017)

[35] Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., Huang, G.: Expel: Llm agents are experiential learners. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 19632–19642 (2024)

[36] Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017)

[37] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)

[38] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36, 53728–53741 (2023)

[39] M¨uhlberger, M., et al.: Navigating the regulatory landscape for AI-based medical devices. NPJ Digital Medicine (2024)

[40] Price, W.N., Gerke, S., Cohen, I.G.: Medical malpractice and black-box medicine. Big Data, Health Law, and Bioethics, 295–306 (2019)

[41] Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., et al.: Siren’s song in the ai ocean: A survey on hallucination in large language models. Computational Linguistics 51(4), 1373–1418 (2025)

[42] Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., et al.: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43(2), 1–55 (2025)

[43] Lin, S., Hilton, J., Evans, O.: Truthfulqa: Measuring how models mimic human falsehoods. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (volume 1: Long Papers), pp. 3214–3252 (2022)

[44] Lamport, L., Shostak, R., Pease, M.: The byzantine generals problem. In: Concurrency: the Works of Leslie Lamport, pp. 203–226 (2019)

[45] Blanchard, P., El Mhamdi, E.M., Guerraoui, R., Stainer, J.: Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems 30 (2017)

[46] Fredrikson, M., Jha, S., Ristenpart, T.: Model inversion attacks that exploit confidence information and basic countermeasures. In: Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pp. 1322–1333 (2015)

[47] Zhu, L., Liu, Z., Han, S.: Deep leakage from gradients. Advances in neural information processing systems 32 (2019)

[48] Hitaj, B., Ateniese, G., Perez-Cruz, F.: Deep models under the gan: Information leakage from collaborative deep learning. In: Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 603–618 (2017)

[49] Lu, J., Zhang, X.S., Zhao, T., He, X., Cheng, J.: April: Finding the achilles’ heel on privacy for vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10051–10060 (2022)