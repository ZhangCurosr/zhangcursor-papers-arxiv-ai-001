# Contestable Agents for Data Governance: Recording Human Stewardship of AI-Generated Metadata Across Structured and Unstructured Systems

Humza Akhtar<sup>1</sup> Brian Freeman<sup>2</sup>

<sup>1</sup>Industry Solutions, MongoDB, Inc.

<sup>2</sup>Advanced Concepts & Capabilities, Trane Technologies

humza.akhtar@mongodb.com, brian.freeman@tranetechnologies.com

## Abstract

Industrial organizations hold operational knowledge in structured records, such as telemetry, point lists, and transactions, and in unstructured sources, such as manuals, service notes, communications, and inspection reports. Governance programs often manage these sources separately. Classifying, linking, and documenting them depends on a small group of employees with deep domain knowledge. Agents built on large language models can propose metadata, cite documents, and flag quality issues, but unreviewed proposals can spread errors into downstream systems. A perfunctory approval leaves no record of what a reviewer checked. We present contestable governance: specialist agents produce scored, cited arguments for disputed metadata decisions; a steward can accept, reject, modify, or add arguments before commit; and a governance record retains the evidence, interventions, and outcome. A versioned domain glossary supplies model-specific terms and acronym definitions to the agents. We describe the workflow, a triage path that reserves full argumentation for uncertain proposals while preserving review before commit, the scoring and record models, and a gate that controls which verified decisions become precedent. Two HVAC use cases illustrate the pattern: point classification using Brick Schema class labels, supported by an open reference implementation, and an illustrative warranty failure-mode decision linking claim fields and telemetry to service notes and revisioned bulletins.

Keywords: data governance, metadata management, AI agents, contestable AI, computational argumentation, data lineage, domain vocabulary, Brick Schema, structured and unstructured data

## 1 Introduction

Manufacturers of commercial HVAC equipment depend on data in two forms. Connected chillers, rooftop units, and air handlers stream structured telemetry from hundreds of points per unit, while installation, operation, and maintenance manuals define those points in prose that varies by model and firmware revision. Factory end-of-line tests produce structured pass and fail results, while PDF specifications and change notices explain the test limits and engineering deviations. Warranty and service systems hold claim codes and part numbers, while technicians describe failures in free-text notes. Governance programs typically assign these sources to different tools and owners, leaving employees to remember the links between fields and the documents that define them.

That arrangement breaks down as the installed base grows. Every new product line, firmware update, and connected site generates data that must be classified and linked to documentation before remote monitoring and service teams can use it reliably. Few employees have the equipment knowledge to do this work. Until a steward reviews a new model, its points may remain unclassified or carry incorrect classes. If a mixed-air temperature sensor is labeled as discharge air, for example, a remote monitoring system could interpret normal mixing as a failed heating coil and send a technician to the site with an incorrect diagnosis.

AI agents built on large language models change the economics of this work. Foundation models can match schemas, clean entries, and infer the meaning of cryptic field names with little task-specific training (Narayan et al., 2022). Agents that combine a model with retrieval and extraction tools can read a manual, compare a field with prior decisions, and propose a classification (Wang et al., 2024; Lewis et al., 2020). They also introduce a governance problem. An agent that writes metadata directly to a catalog can propagate an error to every report and application that consumes it. A wrong class assigned to one air handler can spread across a portfolio when its mapping is copied to other units of the same model. Reviewers who approve output with a single click risk rubber-stamping it, a documented pattern in which people accept automated recommendations with little scrutiny (Parasuraman and Manzey, 2010; Buçinca et al., 2021). The resulting record says that a person approved the decision without showing what they checked. When a reviewer overrules an agent, for example by reclassifying a supplier’s part number, the correction may stay in an email or ticket that the agent never sees. The agent then repeats the mistake. Months later, an engineer investigating a fault rule that fires on the wrong point cannot tell whether its classification came from a model, a reviewer, or a bulk import.

Closing this gap requires seven properties, which we use as requirements throughout the paper: proposals must cite the structured fields and documents they rely on (R1); a reviewer must be able to contest them before commit (R2); every decision must be attributable to an agent, a model version, and a named reviewer, as applicable (R3); history must be append-only (R4); the full lineage of a decision must be retrievable in one query (R5); verified decisions must pass a gate before becoming precedent (R6); and reviewer interventions must be stored as measurable data (R7).

Here, lineage means the trace from source evidence through agent proposals and human edits to committed metadata. We use PROV, the World Wide Web Consortium’s provenance data model, to describe that trace: its entities include documents and decision records, its activities include generating and reviewing arguments, and its agents are the models and people responsible (Moreau and Missier, 2013). PROV provides a conceptual mapping in this paper; the reference implementation does not store records in a PROV serialization.

This paper argues that contestability belongs in the data layer of agent-assisted governance. A contestable system lets a qualified person challenge the reasoning before an output is committed and retains that intervention in the decision record (Hirsch et al., 2017; Alfrink et al., 2023). In our proposed pattern, agents attach cited evidence to each decision, specialist agents argue disputed cases, and a steward can contest a proposal before commit. A governance record shows who made each decision and how the steward intervened (Section 3). We discuss two HVAC use cases (Section 4). The first is point classification, supported by an open reference implementation. The second is an illustrative warranty failure-mode classification that shows how a steward could resolve conflicting evidence across structured service records and revisioned documents.

## 2 Background and Related Work

Khatri and Brown (2010) frame data governance as a set of decision domains covering data principles, data quality, metadata, data access, and data lifecycle, with explicit assignment of who holds decision rights in each. Weber et al. (2009) show that the right governance structure depends on organizational context, and Otto (2011) documents how large firms organize stewardship roles in practice. Abraham et al. (2019) consolidate this literature into a framework of structural, procedural, and relational mechanisms, and the DAMA body of knowledge sets out the operational practices stewards follow (DAMA International, 2017). Across this work, stewardship is a human activity performed alongside operations, and its outputs, such as approved definitions and classifications, are stored separately from the reasoning that produced them.

Large data catalogs show what happens when documentation depends entirely on engineers: Google’s Goods system infers metadata for billions of datasets after creation because requiring documentation up front did not scale (Halevy et al., 2016). Provenance research distinguishes why a result was produced from where its data came from (Buneman et al., 2001). PROV formalizes this account through relationships among entities, activities, and agents (Moreau and Missier, 2013). Other work describes what metadata should enable. The FAIR principles call for data to be findable, accessible, interoperable, and reusable (Wilkinson et al., 2016), while datasheets ask producers to document how datasets were collected and processed (Gebru et al., 2021). Data quality provides a way to judge the result across dimensions that depend on fitness for use (Wang and Strong, 1996). Together, these works motivate recording evidence and responsibility close to the decision that creates metadata.

Point naming in building automation makes the metadata problem concrete. A controller exposes thousands of points, and one vendor may label an air handler’s supply-air temperature sensor AHU3\_SA\_T, while another uses SupplyAirTemp\_03. Brick defines a shared ontology of building equipment, points, and their relationships (Balaji et al., 2016). Vendors still name points differently, so someone must map raw names to Brick classes before energy reports and fault detection rules can use them consistently. Automated approaches infer mappings from names, for example by learning parsing rules from a few expert examples (Bhattacharya et al., 2015) or transferring learned mappings across buildings through an intermediate representation (Koh et al., 2018). A name alone can mislead when vendor conventions differ, and relevant manuals may resolve the ambiguity. Foundation models can read those manuals (Narayan et al., 2022); the question is how to use both kinds of evidence while keeping a reviewer in control of what is committed.

Freeman et al. (2026) compared four sequential single-task agents with one multi-task agent on a fixed ERP incident, then added a reconciler to inspect contradictions. They also tested glossary injection on a separate HVAC troubleshooting query. Their judge-based results motivate role separation and explicit vocabulary here, but do not establish that either measure improves the factual correctness of metadata decisions in this workflow.

Contestability has been studied as a design goal for machine learning systems whose users need to question outputs as part of their professional practice (Hirsch et al., 2017). Later work examined how people understand challenges to algorithmic decisions (Lyons et al., 2021) and proposed design practices for contestable AI (Alfrink et al., 2023). These studies describe what a person should be able to do; computational argumentation offers one way to represent what they can challenge. Abstract argumentation frameworks model arguments and their attacks on one another (Dung, 1995), while bipolar frameworks add support relations (Cayrol and Lagasquie-Schiex, 2005). Quantitative bipolar argumentation frameworks (QBAFs) assign base scores and compute how support and attack affect argument strength, as in DF-QuAD (Rago et al., 2016). An argument graph divides a decision into claims that a person can challenge individually. Such graphs have been proposed for contestable AI (Leofante et al., 2024), and recent work constructs them from language-model-generated arguments (Freedman et al., 2024). We apply this approach to data governance and add a storage model that preserves each reviewer’s edits alongside the arguments they changed.

## 3 Our Approach

## 3.1 Workflow

The full contestation path has four phases. In data analysis, an analyzer reads structured inputs, such as a point list or schema, gathers relevant documents and glossary entries, and retrieves similar verified decisions from a precedent library. In multi-agent argumentation, specialist agents submit supporting or challenging arguments only when their evidence bears on a candidate; each argument has a base score and a source reference (R1). Lack of evidence for one candidate does not, by itself, support another. The agents have distinct evidence roles: one might inspect a point name, another the equipment manual, and a third the applicability of a document revision. Their disagreements show a reviewer where the evidence diverges. In human contestation, a steward can accept, reject, modify, or add arguments. The system rescores after each change, and no decision is committed before review ends (R2). In selection and storage, the scoring engine selects the highest-scoring candidate, a finalizer explains the outcome and propagation steps, and the system stores the evidence, arguments, interventions, and decision in a governance record (R3 to R5). The finalizer does not choose the class; an auditor can recompute it from the stored arguments without rerunning a model. Algorithm 1 summarizes this proposed sequence.

Algorithm 1 Simplified Contestation Workflow with Human Oversight   
1: procedure CONTEST(input)   
Phase 1: Data analysis   
2: candidates ← AnalyzerIdentifyCandidates(input)   
3: evidence ← AnalyzerGatherDocumentsGlossaryAndPrecedents(input)   
Phase 2: Multi-agent argumentation   
4: arguments ← AgentsCollectRelevantArguments(candidates, evidence)   
5: scores ← ScoringEngineCalculateScores(candidates, arguments)   
Phase 3: Human contestation   
6: stewardActions ← []   
7: while review is open do   
8: action ← HumanStewardAcceptRejectModifyOrAddArgument()   
9: Apply(action)   
10: Append(stewardActions, action)   
11: scores ← ScoringEngineRecalculateScores(candidates, arguments)   
12: end while   
Phase 4: Selection, explanation, and storage   
13: selected ← ScoringEngineSelectHighestScoringCandidate(scores)   
14: explanation ← FinalizerExplainResultAndPropagation(selected, arguments, scores)   
15: record ← StoreGovernanceRecord(input, evidence, arguments,   
stewardActions, selected, explanation)   
16: return record   
17: end procedure   
18: procedure AUDIT(record)   
19: recomputed ← RecomputeClassFromStoredArguments(record.arguments)   
20: return Compare(recomputed, record.selected)   
21: end procedure

Role constraints. The human steward may accept, reject, modify, or add arguments during review. The scoring engine selects the highest-scoring candidate. The finalizer explains the result and propagation steps but does not choose the class. The auditor can recompute the class from the stored arguments without rerunning a model.

## 3.2 Domain vocabulary

Acronyms and equipment terms often change meaning across vendors, product families, and revisions. Before the agents evaluate a case, the analyzer can supply a controlled glossary containing only terms present in its evidence packet. Each entry needs an expansion, a definition, its equipment and revision scope, the source passage that authorizes it, a version, and a steward owner. For example, DISCH may identify the discharge side of a mixing box on one air handler but a post-coil location on another. A universal expansion would conceal that difference. Selecting relevant entries also avoids filling each agent’s context with unrelated definitions, a tradeoff explored in the glossary experiments of Freeman et al. (2026).

The glossary is governed metadata, not an unquestionable prompt instruction. Agents should cite both an applicable glossary entry and its source document when a term affects a proposal. A reviewer who finds an obsolete or ambiguous definition can challenge it, record the correction, and prevent the unverified entry from becoming precedent. The proposed decision record would store the glossary version used so that later changes can identify decisions needing review.

## 3.3 Triage

Running every decision through full argumentation requires model calls and reviewer time, even though many points on a new unit are unambiguous. A typed decision model such as Jev, which returns a bounded choice with a probability instead of generated text (TypeSafe AI, 2026), could classify points first and send only low-confidence proposals through multi-agent argumentation. In a separate controlled benchmark of 100 agent tasks, a similar routing design reduced calls to a strong language model by 72.7% while reaching 95% task success (Wu and Lim, 2026). Those figures do not measure this governance workflow.

High-confidence proposals still go to a steward with their source evidence, model version, and probability. Each remains contestable before an explicit review decision and catalog commit (R2). Its governance record stores the question, evaluated state, reviewer action, and outcome even when no argument graph was generated (R3). The threshold that selects the review path is itself a governance decision: it needs a named owner and must be stored with every record it affects. A reviewer can escalate any proposal to full argumentation because a probability alone gives no specific reasoning to challenge.

## 3.4 Argument scoring

Each decision on the full contestation path uses a quantitative bipolar argumentation framework (QBAF) $\langle A , R ^ { + } , R ^ { - } , \tau \rangle$ , drawing on bipolar and quantitative argumentation (Cayrol and Lagasquie-Schiex, 2005; Rago et al., 2016). “Bipolar” means that arguments can support or attack one another; “quantitative” means that they have numerical strengths. The set A contains the active arguments. For an argument x, $R ^ { + } ( x )$ contains its supporters, R<sup>−</sup>(x) contains its attackers, and $\tau ( x ) \in [ 0 , 1 ]$ is its unitless base score. The framework specifies the range of $\tau ( x )$ but does not determine its value: an assessor must assign that input before the graph is scored. Within a candidate, supporting arguments support one another, as do challenging arguments. Every supporting argument attacks every challenging argument, and vice versa. Let $f ^ { t } ( x )$ denote the computed strength of argument x after iteration t, with $f ^ { 0 } ( x ) = \tau ( x )$

We use the signed-sum logistic update from the open reference implementation as an illustrative QBAF scoring rule. Its support and attack contributions have explicit signs, its inputs can be stored for independent recomputation, and the logistic function bounds strengths in (0,1). This choice preserves the reference implementation’s arithmetic for the worked examples; it is not evidence that this rule is preferable. DF-QuAD (Rago et al., 2016) and other modular weighted semantics for bipolar graphs (Potyka, 2019) offer alternative ways to aggregate and propagate support and attack. The unitless coefficient α scales every support and attack contribution before the logistic transform. We fix $\alpha = 0 . 3$ as an illustrative uniform edge weight, not a parameter fitted to outcomes. These strengths are comparison scores, not calibrated probabilities. For a positive integer n, we update all strengths simultaneously for n iterations:

$$
f ^ { t + 1 } ( x ) = \mathrm { s i g } \Big ( \tau ( x ) + \alpha \sum _ { y \in R ^ { + } ( x ) } f ^ { t } ( y ) - \alpha \sum _ { z \in R ^ { - } ( x ) } f ^ { t } ( z ) \Big ) , \qquad t = 0 , \ldots , n - 1 .\tag{1}
$$

Here, $\mathrm { s i g } ( u ) = 1 / ( 1 + e ^ { - u } )$ is the logistic function. The summation variable y ranges over arguments supporting x, and z ranges over arguments attacking $x ; f ^ { t } ( y )$ and $f ^ { t } ( z )$ are their strengths at iteration t. The value $f ^ { n } ( x )$ is the computed strength of argument x after n updates. A candidate’s raw margin is the sum of its supporting arguments’ final strengths minus the sum of its challenging arguments’ final strengths.

The first update uses each neighbor’s base score $f ^ { 0 } = \tau$ , before that neighbor’s own supporters and attackers have changed its strength. Later updates propagate those changes through the graph, including its reciprocal links, until the strengths approach a fixed point. Iteration seeks a self-consistent graph score; it does not add evidence or change τ.

For the worked warranty example, we set $n = 1 0$ , matching the reference implementation’s default and serving as a case-specific numerical stopping count. S1 moves from $f ^ { 1 } = 0 . 6 6 5$ to $f ^ { 1 0 } = 0 . 6 8 3$ In the largest candidate graph of that example (Section 4.2), each of five arguments has at most four incoming relations. Since the logistic derivative is at most $1 / 4$ , Equation 1 has a sup-norm contraction bound $q \leq ( 1 / 4 ) ( 0 . 3 ) ( 4 ) = 0 . 3$ . The saved reproducibility script finds a maximum max $_ x | f ^ { 1 0 } ( x ) - f ^ { 9 } ( x ) |$ of $3 . 2 0 \times 1 0 ^ { - 7 }$ across the candidate graphs before and after review. The contraction-mapping error bound (Rudin, 1976) multiplies this difference by $q / ( 1 - q )$ , making the remaining error at most $1 . 3 7 \times 1 0 ^ { - 7 }$ per argument and $6 . 8 5 \times 1 0 ^ { - 7 }$ per raw margin, below the three-decimal precision used to compare candidates. A local sensitivity check at $\alpha = 0 . 2$ and 0.4 preserves Sensor as the pre-review winner and Linkage as the post-review winner, although the margins change. These checks establish numerical stability for this example, not decision validity, calibration, or general robustness; larger graphs or different weights require a new convergence check.

The reference implementation normalizes candidate margins for display; the worked examples report raw margins. Reviewer operations change the active graph: a rejected argument is removed before rescoring but retained in the record; an added argument enters with a base score of 0.85; a modified argument retains its original text; and an accepted argument is marked as reviewed. With no incoming support or attack, the logistic transform maps base scores in [0,1] to strengths between 0.50 and 0.73, rather than preserving an isolated argument’s base score. This behavior should be assessed against published axioms for weighted bipolar argumentation (Amgoud and Ben-Naim, 2018). The presence or absence of an argument can therefore move a candidate’s score more than a modest change in its base score, making rejection a strong review action. In the reference implementation, the agent that writes an argument also assigns its base score. For the synthetic warranty example, we selected the base scores in Table 1 to illustrate the scoring and review steps; they were not derived from a rubric, empirical calibration, or a rating study. In a deployment, a versioned rubric would first check whether the cited source applies to the item and revision: an inapplicable source should not contribute a scored argument. For each eligible argument, the rubric could assess the directness and specificity of the evidence and record the assessor, rationale, and rubric version. Independent adjudication would then be needed to test whether these assignments are repeatable and improve decisions. A separate decision model could score arguments, separating authorship from assessment.

## 3.5 Governance record

The governance record connects a decision with its evidence and review history, so an auditor can retrieve its lineage without manually reconciling logs, tickets, and catalogs. The reference implementation uses MongoDB documents for arguments and audit entries, time series collections for sensor data, and vector search over embedded documents. In the proposed cross-system schema, evidence sits in a context document; each argument has its own document with base and final scores and edit fields; and the decision document contains the outcome, scores before and after review, and an entry for every human edit. A shared session identifier connects these documents, allowing one aggregation pipeline to return the decision’s lineage (R5):

```javascript
db.governance_records.aggregate([
{ $match: { session_id: "a7c3..." } },
{ $lookup: { from: "argument_records", localField: "session_id",
foreignField: "session_id", as: "arguments" } },
{ $lookup: { from: "session_contexts", localField: "session_id",
foreignField: "session_id", as: "evidence" } },
{ $unwind: "$evidence" }
])
```

This structure can be expressed in PROV terms (Moreau and Missier, 2013): source documents and arguments are entities; generation, review, and edits are activities; and agents and reviewers are responsible actors. The decision record is the resulting entity. Storing interventions as data also supports operational questions: which agent’s arguments are rejected most often, how long review takes for a new equipment model, and how many points the triage threshold routes to full argumentation (R7). To meet the append-only requirement in a deployment, edits would need a sequenced event log and database roles that allow inserts but prohibit updates and deletes (R4). Schema validation should require a named reviewer, model and prompt versions, and a rationale for each intervention (R3).

## 3.6 Promotion gate

A verified record is useful both as a decision and as precedent for a similar case. Its embedding could enter the vector index searched during data analysis, allowing the next unit of the same equipment model to retrieve the reviewer’s corrections. An incorrect precedent could spread across a portfolio. The proposed promotion gate therefore admits only verified records (R6), stores each promotion as an attributable event, and lets a reviewer demote a record. Demotion would flag decisions that relied on the record for review.

## 4 Use Cases

## 4.1 Use case 1: HVAC point classification

Each connected HVAC unit exposes hundreds of points with vendor-specific names. Mapping them to standard classes lets building management systems and analytics platforms interpret them consistently. The open reference implementation https://github.com/mongodb-industry-solutions/iom t-contestable-ai demonstrates contestable governance for this task on synthetic data using CrewAI agents, GPT-4o, Voyage AI embeddings, and MongoDB (Figure 1).

The synthetic site contains 247 points, 18 of them flagged as ambiguous. Under the proposed triage path, 229 points would receive a shorter evidence-based steward review, while the 18 ambiguous points would receive full argumentation and review. None would be committed before review. The reference implementation runs the full workflow on one point: AHU-1-DISCH-T, on an air handler from a fictional EQ-7700 series. The analyzer retrieves precedents showing why this point needs review: DISCH-T maps to mixed air on EQ-7700 air handlers and supply air on EQ-7700 fan coil units, while a similar name on another vendor’s equipment refers to post-coil discharge air. Two agents argue over three option labels: brick: Supply\_Air\_Temperature\_Sensor, brick:Discharge\_Air\_Temperature\_Sensor, and brick:Mixed\_Air\_Temperature\_Sensor. Brick 1.4.2 declares the supply-air and discharge-air sensor classes equivalent and marks the latter as an alias of the former (Brick Consortium, Inc., 2024). The three labels therefore represent two distinct ontology choices; the reference implementation nevertheless scores the labels separately. The String Matcher sees only the point name, the evidence available to namebased methods (Bhattacharya et al., 2015; Koh et al., 2018), and favors discharge air. The Documentation Reader sees manual fragments and cites section 4.2, which places the sensor in the mixed-air plenum, and Appendix A, which explains that DISCH refers to the discharge side of the mixing box.

![](images/460985f9e90faba43af5346df5bc5f49ccc7b24cfc9193620c06067ad0b653d9.jpg)  
Figure 1: Contestable governance for HVAC point classification: data analysis, multi-agent argumentation between two agents, human contestation, and selection and storage of a verified onboarding record.

A model-scoped glossary would preserve these different meanings of DISCH with the applicable equipment type and manual revision. A single global expansion would conceal the conflict that the reviewer needs to inspect.

With illustrative base scores, the two leading candidates finish argumentation nearly tied: a margin of 0.045 for discharge air and 0.044 for mixed air. The reviewer rejects the String Matcher’s challenge to mixed air because it rests on the abbreviation alone, raising the mixed-air margin to 0.821. The reviewer then adds an argument citing section 4.2, raising the margin to 1.758. The system commits mixed air, consistent with the documentation. The record attributes the class to two named agents and a reviewer, cites the decisive manual section, and preserves the rejected reasoning and the reason for rejection. If promoted, the decision would give the next EQ-7700 air handler a verified precedent that includes the reviewer’s knowledge of this vendor’s use of DISCH.

The scenario also exposes decisions a production deployment would need to make. Normalized scores display the near tie above as 1.00 and 0.99, which a reviewer might mistake for high confidence; raw margins should appear alongside them. Base scores are assigned by the model itself, remain uncalibrated, and can vary between runs. The reference implementation overwrites an argument on a second edit, seeds its precedent library without writing verified decisions back, and has no triage step. Append-only history, the promotion gate, and triage remain proposed extensions.

## 4.2 Use case 2: Governing warranty failure-mode metadata across systems

A fictional air handler, model DEMO-AHU with serial DEMO-2407, has a claim record C1 containing a supply air temperature (SAT) fault code and a replaced supply-air sensor. A telemetry window T1 shows a changing damper command while measured position stays fixed. A technician note span N1 reports a loose damper linkage and says the sensor replacement did not clear the fault. A service bulletin, B1, describes sensor drift only for serial suffixes 1000–1999. Glossary entry G1 records this model-specific expansion of SAT. The governed field is one failure-mode tag with two candidates: sensor drift (Sensor) and damper linkagefailure (Linkage). A claim-to-bulletin link is a separate decision. The original claim code remains source data.

The analyzer snapshots C1, T1, N1, B1, and G1 in Figure 2, capturing the state of all relevant evidence at the time of analysis. A claims-and-telemetry agent receives C1, T1, and G1; a service-note agent receives N1 and G1; and a bulletin-scope agent receives B1 and C1’s serial number. These agents submit arguments with cited evidence; a reconciler flags contradictions but cannot select a tag or authorize a commit. Table 1 lists each candidate’s supporting and challenging arguments with its cited basis and illustrative base score. The author-selected scores create a contested decision, rather than estimate evidential reliability. In this example, the bulletin title produces a provisional Sensor argument (S3) with τ = 0.80, while the serial-range check challenges it (S5). Because B1 does not apply to this unit, S3 is an erroneous argument to remove during review, not an applicable argument that merely deserves a lower score.

![](images/4df4655f3f41b9034045f6ea86eb82546a33395d2a90971ac4681b1805fe3666.jpg)  
Figure 2: Proposed workflow for the fully synthetic warranty example. Versioned evidence and glossary entries feed specialist agents; a steward contests the arguments before the selected tag and its lineage are stored. The promotion gate is a proposed extension.

Table 1: Synthetic arguments for the failure-mode decision. A plus sign supports a candidate and a minus sign challenges it. S3 is rejected and S6 is added during review. The author-selected τ values are illustrative inputs, not probabilities or the computed $f ^ { 1 0 }$ strengths.
<table><tr><td>ID</td><td>Effect</td><td>Cited basis</td><td>τ</td></tr><tr><td>S1</td><td>Sensor +</td><td>C1: SAT fault code</td><td>0.70</td></tr><tr><td>S2</td><td>Sensor +</td><td>C1: supply-air sensor replaced</td><td>0.65</td></tr><tr><td>S3</td><td>Sensor +</td><td>B1: bulletin title mentions sensor drift</td><td>0.80</td></tr><tr><td>S4</td><td>Sensor –</td><td>T1: damper command changes; position stays fixed</td><td>0.75</td></tr><tr><td>S5</td><td>Sensor –</td><td>B1 and C1: serial ranges do not overlap</td><td>0.75</td></tr><tr><td>D1</td><td>Linkage +</td><td>T1: damper command-position mismatch</td><td>0.80</td></tr><tr><td>D2</td><td>Linkage +</td><td>N1: technician observed a loose linkage</td><td>0.85</td></tr><tr><td>D3</td><td>Linkage –</td><td>C1: fault code points toward a sensor</td><td>0.70</td></tr><tr><td>S6</td><td>Sensor –</td><td>N1: sensor swap did not resolve fault; steward-added</td><td>0.85</td></tr></table>

For a worked calculation, consider S1 before review. $\mathrm { A t } t = 0 .$ its supporters S2 and S3 have base scores 0.65 and 0.80, while its attackers S4 and S5 each have base score 0.75. The first synchronous update of Equation 1 is

$$
\begin{array} { r l } & { f ^ { 1 } ( \mathrm { S 1 } ) = \mathrm { s i g } \big ( 0 . 7 0 + 0 . 3 ( 0 . 6 5 + 0 . 8 0 ) - 0 . 3 ( 0 . 7 5 + 0 . 7 5 ) \big ) } \\ & { \qquad = \mathrm { s i g } ( 0 . 6 8 5 ) = 0 . 6 6 5 . } \end{array}
$$

After ten updates, $f ^ { 1 0 } ( \mathbf { S } 1 ) = 0 . 6 8 3$ . The other Sensor strengths before review are 0.673 and 0.703 for S2 and S3, and 0.576 each for S4 and S5. Their signed sum gives a raw Sensor margin of $0 . 6 8 3 + 0 . 6 7 3 + 0 . 7 0 3 -$ $0 . 5 7 6 - 0 . 5 7 6 = 0 . 9 0 7$ ; the corresponding Linkage margin is 0.839. The arXiv ancillary files include the script reproduce\_warranty\_scores.py, which recomputes the per-argument $f ^ { 1 }$ and $f ^ { 1 0 }$ values and candidate margins, and its output, warranty\_score\_results.txt.

The reconciler flags the conflict between S3 and S5. The steward rejects S3 because B1 excludes serial DEMO-2407 and adds S6 from N1. Rescoring changes $f ^ { 1 0 } ( { \bf S } 1 )$ to 0.557 and the Sensor margin to −1.016; the Linkage margin remains 0.839. The scoring engine selects Linkage, and the steward checks the result before authorizing the commit. These numbers show how the defined scoring rule responds to edits; they are neither calibrated confidence values nor measured classification accuracy. Table 2 contrasts the ambiguous shorthand in the synthetic records with the evidence-linked language available after glossary and source checks.

Table 2: Synthetic examples of ambiguous language and clear, evidence-linked results. The right column states what the records support after glossary and source checks.
<table><tr><td>Ambiguous shorthand</td><td>Clear, reviewable result</td></tr><tr><td>SAT fault; changed sensor.</td><td>G1 expands SAT as supply-air temperature. C1 records a code and part replacement; neither proves sensor drift.</td></tr><tr><td>Bulletin confrms drift</td><td>B1 revision B covers serial suffixes 1000–1999. C1 has suffix 2407, so this bulletin does not apply to the unit.</td></tr><tr><td>Damper seems bad.</td><td>T1 shows command-position mismatch, and N1 records a loose linkage and unsuccessful sensor replacement. Proposed tag: damper linkage failure.</td></tr></table>

The proposed governance record retains versions of C1 through G1, the original claim code, both candidate tags, every argument, the S3 rejection and S6 addition, the reviewer identity, and the final tag.

The separate claim-to-bulletin link would be rejected on applicability grounds. Only a verified record could become precedent; a later bulletin revision could flag dependent decisions for review.

The following abbreviated JSON documents illustrate how this synthetic decision and its S1 argument could be stored. The decision contains the selected tag, candidate margins, and steward edits; the argument contains the base score τ and computed $f ^ { 1 0 }$ strengths. Both share a session\_id with the evidence snapshot, allowing retrieval through the aggregation in Section 3.5.

Decision document (governance\_records).

"\_id": "decision-DEMO-2407",   
"session\_id": "session-DEMO-2407",   
"decision\_type": "failure\_mode\_tag",   
"subject": {"model": "DEMO-AHU", "serial": "DEMO-2407"},   
"evidence\_snapshot\_id": "context-DEMO-2407",   
"glossary\_version": "G1-v1",   
"scoring": {"rule": "logistic\_qbaf\_v1", "alpha": 0.3,   
"iterations": 10},   
"raw\_margins": {   
"before": {"Sensor": 0.907130, "Linkage": 0.838551},   
"after": {"Sensor": -1.015798, "Linkage": 0.838551}   
},   
"review\_events": [   
{"seq": 1, "actor\_id": "demo-steward", "action": "reject",   
"argument\_id": "S3", "reason": "B1 excludes serial suffix 2407"},   
{"seq": 2, "actor\_id": "demo-steward", "action": "add",   
"argument\_id": "S6", "evidence\_ref": "N1",   
"reason": "Sensor replacement did not resolve the fault"},   
{"seq": 3, "actor\_id": "demo-steward", "action": "authorize",   
"reason": "Linkage tag matches reviewed evidence"}   
],   
"reviewer\_id": "demo-steward",   
"authorized": true,   
"selected\_tag": "damper\_linkage\_failure",   
"precedent\_status": "pending\_gate"

Argument document (argument\_records).

"\_id": "argument-DEMO-2407-S1",   
"session\_id": "session-DEMO-2407",   
"argument\_id": "S1",   
"candidate": "Sensor",   
"effect": "support",   
"agent\_role": "claims\_telemetry",   
"model\_version": "demo-model-v1",   
"prompt\_version": "demo-prompt-v1",   
"source\_ref": {"id": "C1", "field": "fault\_code"},   
"tau": 0.70,   
"f10\_before": 0.682928,   
"f10\_after": 0.556808,   
"status": "active"

The embedded review events summarize a separate sequenced, append-only event log. Each remaining argument would have its own record with a cited source, base score, final strengths, and review status. A rejected argument such as S3 would retain its pre-review strength and rejection event rather than disappear from the audit trail. These records are proposed, not output from a deployed warranty system. An evaluation on independently adjudicated synthetic cases would need to measure tag accuracy, citation and applicability validity, reviewer interventions, and consistency across repeated runs.

## 5 Discussion

The two cases show why a reviewer needs to challenge individual arguments, not only the selected metadata value. In Use case 1, the point name suggests one class until the manual locates the sensor in the mixed-air plenum. In Use case 2, the bulletin title appears to support sensor drift until its serial range excludes the unit. Retaining the original proposals, cited sources, and steward corrections makes these changes of reasoning inspectable. A versioned glossary helps interpret local terms, but it cannot establish whether a source applies to a particular unit or revision.

Use case 1 demonstrates a choice among Brick class labels, not validation of a Brick instance graph. The reported margins reproduce the reference implementation’s three-label scoring and should not be read as ontology-normalized scores. An ontology-aware deployment would collapse equivalent class labels before scoring, record the ontology version and canonical class, link the point instance to its equipment through brick:hasPoint or brick:isPointOf, and validate the resulting graph against that Brick version. The warranty failure-mode tag in Use case 2 is governed metadata outside the Brick point-classification example.

The examples also expose limits of the illustrative scoring method. The warranty calculation demonstrates that the stored graph can be rescored after review; it does not establish that the selected tag is more accurate than an alternative method. The signed-sum margin depends on which claims are represented as separate arguments. S1 and S2 both derive from C1, for example, so repeated or correlated evidence could increase a margin without adding independent support. A deployment would need to identify shared provenance and test decision stability under alternative argument graphs as well as alternative values of τ and α.

An auditable record makes a decision reconstructable, but it does not make the decision correct. Stewards still need to check document scope, resolve conflicting evidence, and authorize commits. Evaluation should measure review time, rejected arguments, erroneous commits, and later reversals alongside classification accuracy. The point-classification reference implementation demonstrates part of the workflow; the crosssystem warranty record and promotion gate remain proposed. Their effects on review workload and decision quality have not been measured.

## 6 Conclusion and Future Research

Agents can perform much of the classification, linking, and documentation work on which data governance depends. Governance programs still need control over what enters the catalog. Contestable governance attaches cited structured and unstructured evidence to each proposal, lets a steward challenge it before commit, and stores the result as an attributable record retrievable in one query. A promotion gate controls what becomes precedent, while triage reserves full specialist argumentation for uncertain cases. The synthetic point-classification case illustrates how a reviewer can correct a name-based inference and preserve the reason for the correction. The illustrative warranty case shows how specialist agents and a versioned glossary could expose conflicts among claim codes, telemetry, technician observations, and bulletin scope before a steward commits a failure-mode tag. Requiring a steward for every correction may delay resolution, even when an edit is low risk and reversible. A future risk-based path could automatically correct a typographical error in a point description without changing its class, provided the evidence is unambiguous and the system records the original value, rationale, and means to reverse the edit. The workflow described here still requires review before commit; the conditions and effects of such an automatic path need evaluation.

Future research should test the learning loop. Once verified decisions are written back through the promotion gate, an evaluation could measure whether precedent improves accuracy on later units of the same equipment series, how far an incorrect precedent spreads before demotion, and when a firmware revision makes an older precedent unreliable. Separate comparisons should test whether versioned glossary entries and specialist roles improve citation validity and repeated-run consistency over a shared, unstructured prompt.

Validating the scoring rule requires formal and empirical tests. Formal checks should assess boundedness, the direction of support and attack effects, the treatment of isolated and duplicated arguments, and convergence on cyclic graphs against published axioms and analyses (Amgoud and Ben-Naim, 2018; Potyka, 2019). An empirical comparison should hold the evidence, candidate set, and adjudicated labels fixed while comparing the signed-sum logistic rule with DF-QuAD (Rago et al., 2016) and a specified modular alternative, documenting how each handles reciprocal cycles. A versioned rubric should define how assessors assign τ from applicable evidence and test agreement among assessors. The rubric and α should be selected on development cases, then tested on held-out cases for top-class accuracy and ranking stability under small score changes, missing or duplicated arguments, and repeated agent runs. An error-bounded stopping rule should replace the fixed ten iterations so that each graph is scored to a stated numerical tolerance.

## References

Rene Abraham, Johannes Schneider, and Jan vom Brocke. Data governance: A conceptual framework, structured review, and research agenda. International Journal ofInformation Management, 49:424–438, 2019.

Kars Alfrink, Ianus Keller, Gerd Kortuem, and Neelke Doorn. Contestable AI by design: Towards a framework. Minds and Machines, 33, 2023.

Leila Amgoud and Jonathan Ben-Naim. Weighted bipolar argumentation graphs: Axioms and semantics. In Proceedings ofthe 27th International Joint Conference on Artificial Intelligence (IJCAI), pages 5194–5198, 2018. doi: 10.24963/ijcai.2018/720.

Bharathan Balaji, Arka Bhattacharya, Gabriel Fierro, Jingkun Gao, Joshua Gluck, Dezhi Hong, Aslak Johansen, Jason Koh, Joern Ploennigs, Yuvraj Agarwal, Mario Bergés, David Culler, Rajesh Gupta, Mikkel Baun Kjærgaard, Mani Srivastava, and Kamin Whitehouse. Brick: Towards a unified metadata schema for buildings. In Proceedings of the 3rd ACM International Conference on Systems for Energy-Efficient Built Environments (BuildSys), 2016.

Arka Bhattacharya, Dezhi Hong, David Culler, Jorge Ortiz, Kamin Whitehouse, and Eugene Wu. Automated metadata construction to support portable building applications. In Proceedings ofthe 2nd ACM International Conference on Embedded Systemsfor Energy-Efficient Built Environments (BuildSys), 2015.

Brick Consortium, Inc. Brick schema ontology, version 1.4.2, 2024. URL https://brickschema.or g/schema/1.4/Brick.ttl. Accessed October 11, 2026.

Zana Buçinca, Maja Barbara Malaya, and Krzysztof Z. Gajos. To trust or to think: Cognitive forcing functions can reduce overreliance on AI in AI-assisted decision-making. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW1), 2021.

Peter Buneman, Sanjeev Khanna, and Wang-Chiew Tan. Why and where: A characterization of data provenance. In Proceedings ofthe International Conference on Database Theory (ICDT), 2001.

Claudette Cayrol and Marie-Christine Lagasquie-Schiex. On the acceptability of arguments in bipolar argumentation frameworks. In Symbolic and Quantitative Approaches to Reasoning with Uncertainty (ECSQARU), volume 3571 of Lecture Notes in Computer Science, 2005.

DAMA International. DAMA-DMBOK: Data Management Body of Knowledge. Technics Publications, 2nd edition, 2017.

Phan Minh Dung. On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial Intelligence, 77(2):321–357, 1995.

Gabriel Freedman, Adam Dejl, Deniz Gorur, Xiang Yin, Antonio Rago, and Francesca Toni. Argumentative large language models for explainable and contestable decision-making. arXiv preprint arXiv:2405.02079, 2024.

Brian Freeman, Adam Kicklighter, Matt Erdman, and Zach Gordon. Toward epistemic stability: Engineering consistent procedures for industrial LLM hallucination reduction. Computer and Information Science, 19 (2), 2026. doi: 10.5539/cis.v19n2p1. URL https://ccsenet.org/journal/index.php/ci s/article/view/0/53307.

Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Communications ofthe ACM, 64(12):86–92, 2021.

Alon Halevy, Flip Korn, Natalya F. Noy, Christopher Olston, Neoklis Polyzotis, Sudip Roy, and Steven Euijong Whang. Goods: Organizing Google’s datasets. In Proceedings of the 2016 International Conference on Management ofData (SIGMOD), 2016.

Tad Hirsch, Kritzia Merced, Shrikanth Narayanan, Zac E. Imel, and David C. Atkins. Designing contestability: Interaction design, machine learning, and mental health. In Proceedings ofthe 2017 ACM Conference on Designing Interactive Systems (DIS), 2017.

Vijay Khatri and Carol V. Brown. Designing data governance. Communications of the ACM, 53(1):148–152, 2010.

Jason Koh, Bharathan Balaji, Dhiman Sengupta, Julian McAuley, Rajesh Gupta, and Yuvraj Agarwal. Scrabble: Transferrable semi-automated semantic metadata normalization using intermediate representation. In Proceedings of the 5th ACM International Conference on Systems for Built Environments (BuildSys), 2018.

Francesco Leofante, Hamed Ayoobi, Adam Dejl, Gabriel Freedman, Deniz Gorur, Junqi Jiang, Guilherme Paulino-Passos, Antonio Rago, Anna Rapberger, Fabrizio Russo, Xiang Yin, Dekai Zhang, and Francesca Toni. Contestable AI needs computational argumentation. In Proceedings of the 21st International Conference on Principles ofKnowledge Representation and Reasoning (KR), 2024.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrievalaugmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS), 2020.

Henrietta Lyons, Eduardo Velloso, and Tim Miller. Conceptualising contestability: Perspectives on contesting algorithmic decisions. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW1), 2021.

Luc Moreau and Paolo Missier. PROV-DM: The PROV data model. W3C Recommendation, 2013.

Avanika Narayan, Ines Chami, Laurel Orr, and Christopher Ré. Can foundation models wrangle your data? Proceedings ofthe VLDB Endowment, 16(4), 2022.

Boris Otto. Organizing data governance: Findings from the telecommunications industry and consequences for large service providers. Communications ofthe Associationfor Information Systems, 29, 2011.

Raja Parasuraman and Dietrich H. Manzey. Complacency and bias in human use of automation: An attentional integration. Human Factors, 52(3):381–410, 2010.

Nico Potyka. Extending modular semantics for bipolar weighted argumentation. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 1722–1730, 2019. doi: 10.65109/degv1211.

Antonio Rago, Francesca Toni, Marco Aurisicchio, and Pietro Baroni. Discontinuity-free decision support with quantitative argumentation debates. In Proceedings of the 15th International Conference on Principles of Knowledge Representation and Reasoning (KR), 2016.

Walter Rudin. Principles ofMathematical Analysis. McGraw-Hill, 3 edition, 1976.

TypeSafe AI. Jev: A system one model for typed decisions. Product documentation, 2026.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6), 2024.

Richard Y. Wang and Diane M. Strong. Beyond accuracy: What data quality means to data consumers. Journal ofManagement Information Systems, 12(4):5–33, 1996.

Kristin Weber, Boris Otto, and Hubert Österle. One size does not fit all: A contingency approach to data governance. ACM Journal of Data and Information Quality, 1(1), 2009.

Mark D. Wilkinson et al. The FAIR guiding principles for scientific data management and stewardship. Scientific Data, 3:160018, 2016.

Tiantong Wu and Wei Yang Bryan Lim. REFLEX with Jev for efficient selective control in LLM agents. arXiv preprint arXiv:2609.26532, 2026.