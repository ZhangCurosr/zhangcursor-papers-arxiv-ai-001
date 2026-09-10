# Can AI Agents Detect and Repair Artifact Drift in Network Experiments?

Tianzhu Zhang<sup>∗</sup> Nokia Bell Labs Massy, France tianzhu.zhang@nokia-bell-labs.com

Yusheng Zheng University of California, Santa Cruz Santa Cruz, California, USA yzhen165@ucsc.edu

Weichen Tao<sup>∗</sup> Telecom Paris Palaiseau, France weichen.tao@telecom-paris.fr

Long Chen The University of Hong Kong Hong Kong, Hong Kong SAR, China chenlong.sjtu@gmail.com

Changgang Zheng   
Nanjing University   
Nanjing, China   
changgangzheng@qq.com

Xiaoyi Fan Tsinghua University Beijing, China fan-xy25@mails.tsinghua.edu.cn

Meikang Qiu   
Augusta University   
Augusta, Georgia, USA   
qiumeikang@gmail.com

## Ab<sub>s</sub>t<sub>rac</sub>t

In recent years, AI agents have evolved into capable assistants that carry out multi-step tasks in digital environments. The network systems community is beginning to explore these capabilities in operational and experimental settings. However, an agent operating in network systems should not be judged solely by whether it completes the immediate task. The experiment record it modifies must also remain trustworthy. We call this property artifact integrity: the record’s claims must remain supported by the available evidence, confined to the scope established by that evidence, and traceable through the artifacts that encode their support.

To make this property measurable, we introduce NetArtifactBench, which tests whether AI agents can repair inconsistent records derived from public network-system artifacts while preserving claims that remain supported. The benchmark contains 52 instances with injected inconsistencies ranging from direct contradictions to unstated relations spread across several artifacts. We evaluate 23 agent configurations across three general-purpose AI agent runtimes using deterministic scoring. The average contract pass rate is 65.3% across 5,980 outputs, but no agent runtime exceeds 30% when repair requires recovering implicit relations and propagating changes across artifacts. These results reveal a sharp boundary between local correction and complete record-level repair. Therefore, we argue that artifact integrity should become a first-class design and evaluation requirement for AI agents operating on network systems.

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

With the rise of agentic workflows powered by large language models (LLMs), AI agents are beginning to reshape network systems, from operations to experimentation. Network automation has already evolved beyond static procedures toward policy and intent-driven management, closedloop control, and data-driven optimization [6]. LLM-powered agents extend this trajectory by interpreting heterogeneous evidence, acting on the network through tools, and revising the artifacts that record their actions and outcomes. Recent work has explored AI agents for network diagnosis, intentdriven management, configuration, and Internet measurement [20, 23, 25–27]. Industry roadmaps likewise position AI agents as a key component of future autonomous networking [7, 15].

This shift changes not only how network tasks are executed, but also how knowledge about the network is assembled and carried forward. An agent may inspect configurations, combine observations from several sources, and revise the experiment record. The record captures what was executed, under which conditions, what was observed, and what the evidence supports. As AI agents take on larger roles in network operations and experimentation, such records may inform subsequent diagnoses, configurations, and operational decisions. The integrity of these records must be preserved whenever downstream tasks depend on them.

However, current evaluations remain largely outcomecentered. Determining whether an agent achieves the requested result is necessary, but it does not reveal whether the artifacts it modifies remain faithful to the underlying evidence. We refer to the configurations, logs, reports, and structured claims associated with a network task as its network experiment package; together, they form the experiment record. An agent may satisfy the immediate objective while leaving this record inconsistent with, or unsupported by, the available evidence. We call this failure artifact drift. Preventing it requires artifact integrity: every claim must remain supported by the evidence, bounded by its scope, and traceable to the artifacts that justify it. A valid repair must propagate consistently across the record and leave enough provenance for another reviewer, tool, or agent to reconstruct what changed and why.

![](images/4165d6ef16ed97d59550fa2267b37e86b52bcaa7446d4132c4ea74302e26baec.jpg)  
Fi<sub>g</sub>ure 1: Five contract-visible failure modes in a<sub>g</sub>ent-<sub>p</sub>roduced re<sub>p</sub>airs.

Artifact drift becomes more consequential as agents gain greater autonomy. In an assisted task, drift may distort one result. When an agent carries findings from one experiment into the next, however, the record becomes part of the state that guides future actions. An unsupported claim may then shape a new hypothesis, configuration, or measurement, propagating a local inconsistency through an entire line of work. Existing benchmarks remain largely outcomecentered [1, 23, 26, 28]. They do not ask whether the surrounding experiment record remains coherent, evidencegrounded, and supportable after the agent has acted. Artifact integrity thus remains a missing dimension in the evaluation of AI agents for network systems.

We introduce NetArtifactBench to make this missing target measurable. Rather than treating correctness as a collection of file-local checks, NetArtifactBench models artifact integrity through the relations among claims, evidence, scope, provenance, and experimental phase. It realizes this view in 52 source-grounded instances spanning four network task families, each combining controlled drift with valid claims that an agent must preserve. The benchmark further provides a deterministic evaluation protocol that tests whether the agent identifies the broken relation, propagates the correction across all afected artifacts, and leaves behind a complete and traceable record.

## 2 H<sub>ow</sub> A<sub>r</sub>tif<sub>ac</sub>t R<sub>epa</sub>i<sub>rs</sub> F<sub>a</sub>il

NetArtifactBench presents each agent with a record containing a controlled integrity violation. Fig. 1 summarizes five ways the returned repair can fail: it may leave the seeded violation unresolved or introduce a new substantive or auditability failure. The modes are illustrative and may overlap.

Residual inconsistency: A repair may correct the injected inconsistency in one artifact while leaving the same unsupported assertion elsewhere in the experiment record. Fig. 1 (a) shows an example in an Open vSwitch experiment [18]: an initial phase installs several flows, a later replacement removes some of them, and the final state contains only the surviving flows. The returned package preserves the individual phase records but still claims that the removed flows remain in the final state. The source artifacts are valid; the stale conclusion remains unresolved.

Unsupported replacement: Removing an unsupported conclusion does not justify asserting its opposite. In the Batfish example of Fig. 1 (b) [8], the public evidence records how individual filters treat a test flow but does not establish end-toend reachability. The repair replaces the unsupported claim that the flow is blocked end-to-end with the equally unsupported claim that it is not. The direction of the conclusion changes, but the missing path relation is never established.

Support damage: A repair causes support damage when it removes valid information together with an unsupported claim. In Fig. 1 (c), Batfish output establishes the state of a particular BGP session, and the query records the direction and address family. The report incorrectly generalizes this result to sessions in all directions and address families. A correct repair would narrow the claim to the session and scope actually tested. Instead, the repair deletes both the unsupported generalization and the valid statement describing what the query covered. The false claim is removed, but the remaining observation loses the scope needed to interpret it.

Lost traceability: A correction may be substantively plausible while leaving its support impossible to verify from the experiment record. In the P4 example of Fig. 1 (d) [3], the repair narrows an end-to-end forwarding claim to the configured default-miss behavior, but points to a source location that does not exist in the public package. The revised claim may be correct, yet the record no longer provides a valid link between that claim and the evidence intended to support it.

Invalid package: A repair may express a plausible correction without returning it in a valid experiment record. In the Zeek-derived case of Fig. 1 (e) [17], the agent provides a readable explanation but no parseable collection of repaired claims. The returned package thus does not establish which claims remain, how they are classified, or which evidence supports them. Restoring artifact integrity requires not only coherent corrections but also a valid representation in which those corrections can be identified and verified.

These failure modes are distinct but may coexist in one output. A malformed package may also retain a stale claim, and a coordinated repair may simultaneously introduce support damage. Together, they reveal two requirements. First, the returned record must restore the substantive relations among claims, evidence, scope, provenance, and experimental phase. Second, those relations must remain explicit enough for another tool or agent to audit and reuse. The next section formalizes the first requirement, and Section 4 turns both into a deterministic benchmark contract.

## 3 Cl<sub>a</sub>im–E<sub>v</sub>id<sub>e</sub>n<sub>ce</sub>–S<sub>cope</sub> M<sub>o</sub>d<sub>e</sub>l

The failures above arise when an agent changes individual artifacts without restoring the relations that support the experiment record. Following research-object models of scientific artifacts [2, 21], we represent a network experiment package as a graph of claims, evidence, scope constraints, and provenance or phase relations. A claim states what the experiment establishes, evidence records what was config ured or observed, and scope specifies the conditions under which the claim holds. Artifact integrity requires the public package to establish a valid path from evidence to each claim.

A support path may be direct, as when a value is read from a measurement, or relational, as when a claim depends on joining observations, binding metadata objects, or following state across phases. This view draws on database and general provenance models [4, 5, 14]. A path is valid only if every required connection is established by the public artifacts. Matching identifiers do not by themselves justify a cross-trace join, coexisting objects do not imply a binding, and observations from diferent phases cannot be combined without respecting their order and provenance.

For a claim �, let ${ \mathcal { P } } _ { c } ^ { + }$ denote its valid public support paths, and let $\operatorname { c t x } ( p )$ denote the conditions under which path � is valid. A claim presented as established satisfies artifact integrity only if

$$
\exists p \in \mathcal { P } _ { c } ^ { + } \quad \mathrm { s u c h t h a t } \quad \mathrm { s c o p e _ { \mathrm { e f f } } } ( c ) \subseteq \mathrm { c t x } ( p ) .
$$

Here, $\mathsf { s c o p e } _ { \mathrm { e f f } } ( c )$ is the scope asserted after applying explicit and consistent limitations. A limitation may narrow a claim, but it does not supply positive evidence or silently correct broader prose.

We label a claim observed when it follows directly from reported results, supported when other public artifacts establish it, and unsupported when no valid support path exists. An unsupported claim may remain only if it is explicitly labeled as such and nowhere else presented as established. A valid repair must propagate the correction across the record, preserve claims that remain supported, and avoid fabricating missing evidence or relations. Section 4 turns these requirements into a benchmark contract and adds the parseability and traceability conditions needed for machine auditability.

## 4 NetArtifactBench Desi<sub>g</sub>n

NetArtifactBench turns the claim–evidence–scope model into a self-contained repair task over network-system experiment records. The agent receives a frozen public package and task contract, but no hidden oracle information. It must identify the violated relation, propagate the correction across afected artifacts, preserve claims that remain supported, and return parseable and traceable artifacts. Figure 2 summarizes construction, execution, and scoring.

## 4.1 Instance Construction

NetArtifactBench contains 52 instances derived from public Batfish [8], Zeek [17], P4C [16], and Open vSwitch [18] repositories. Their tests, traces, compiler outputs, expected results, and phase-ordered state provide stable evidence anchors. Each injected inconsistency breaks a relation grounded in source-project semantics, such as scope containment, trace provenance, metadata binding, or temporal ordering. The public task contract declares permitted commands, required outputs, and citation rules; every oracle judgment is grounded in public evidence. The resulting instances difer not only in network domain but also in how the drift is planted and what a successful repair must reconstruct. Specifically, some contain an explicit contradiction in a single artifact, while others require the agent to recover a relation across provenance, scope, metadata, or experimental phases. We organize these diferences into five structural tiers, summarized in Table 1. The tiers progress from direct correction (Tiers 1–2) to explicit relational repair (Tier 3), coordinated propagation (Tier 4), and inferred relations (Tier 5).

![](images/2bc2d746cc5ded2ed20533d719c064a34be90fe649ce136af6f4a2baebeb6c13.jpg)  
Fi <sub>ure</sub> 2<sub>:</sub> N<sub>et</sub>A<sub>rtifact</sub>B<sub>ench cons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on wor</sub>kfl<sub>ow.</sub> P<sub>u</sub>bli<sub>c source</sub> f<sub>ac</sub>t<sub>s are assem</sub>bl<sub>e</sub>d i<sub>n</sub>t<sub>o a</sub> coherent acka e a declared mutation is injected without chan in those facts and the instance is validated <sub>an</sub>d f<sub>rozen.</sub> E<sub>very</sub> <sub>execu</sub>ti<sub>on</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>run</sub>ti<sub>me</sub> <sub>a</sub> f<sub>res</sub>h <sub>copy</sub> <sub>o</sub>f <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>pu</sub>bli<sub>c</sub> <sub>pac</sub>k<sub>age</sub> <sub>an</sub>d t<sub>as</sub>k <sub>con</sub>t<sub>rac</sub>t<sub>.</sub> Hidd<sub>en</sub> <sub>anno</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>a</sub>dd <sub>no</sub> <sub>ev</sub>id<sub>ence,</sub> b<sub>ypass</sub> th<sub>e</sub> <sub>agen</sub>t<sub>,</sub> <sub>an</sub>d <sub>mee</sub>t th<sub>e</sub> <sub>re</sub>t<sub>urne</sub>d <sub>repa</sub>i<sub>r</sub> <sub>on</sub>l<sub>y</sub> <sub>a</sub>t th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> scorer<sub>,</sub> which checks dia<sub>g</sub>nosis<sub>,</sub> re<sub>p</sub>air com<sub>p</sub>letion<sub>,</sub> labels<sub>, p</sub>reservation<sub>, p</sub>arseabilit<sub>y,</sub> and traceabilit<sub>y</sub>. Solid <sub>p</sub>aths <sub>are pu</sub>bli<sub>c or agen</sub>t<sub>-v</sub>i<sub>s</sub>ibl<sub>e;</sub> d<sub>as</sub>h<sub>e</sub>d <sub>gray pa</sub>th<sub>s are</sub> hidd<sub>en.</sub>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> St<sub>ruc</sub>t<sub>ura</sub>l ti<sub>ers o</sub>f <sub>ar</sub>tif<sub>ac</sub>t d<sub>r</sub>ift<sub>.</sub>  
Tier What the repair requires   
1 Correct one directly exposed contradiction in a metric, scope, command,   
label, or limitation.   
2 Correct several exposed contradictions while preserving claims that   
remain supported.   
3 Follow an explicit relation or scope constraint across rows, filters, phases,   
identifiers, or attachments.   
4 Propagate one correction across several afected artifacts.   
5 Infer an unstated relation across tables, source contexts, or control and   
runtime artifacts.

## 4.2 Machine-Auditable Scorin<sub>g</sub>

A structured finding matches a violation when it identifies the expected claim, field, artifact scope, and observed-tocorrected value relation. The repaired experiment report is checked for repair completeness and preservation. The human-readable audit report must be present, but its prose is not scored. Let � denote the scored violations, � their corresponding repair obligations, and � the required postrepair label decisions. The scorer also tracks supported claims that must remain intact. We report the following metrics:

$$
\begin{array} { r l } & { \mathrm { ~ r e c a l l } = \frac { { \mathrm { s c o r e d ~ v i o l a t i o n s ~ c r e d i t e d ~ a s ~ d e t e c t e d } } } { | M | } , } \\ & { \mathrm { ~ } } \\ & { \mathrm { ~ r e p a i r = \frac { { \mathrm { s a t i s f i e d ~ r e p a i r ~ o b l i g a t i o n s } } } { | R | } , } } \\ & { \mathrm { ~ l a b e l = \frac { \mathrm { c o r r e c t ~ l a b e l ~ d e c i s i o n s } } { | L | } , } } \\ & { \mathrm { ~ a c c u s . = p r o t e c t e d - c l a i m a c c u s a t i o n ~ e v e n t s } , } \\ & { \mathrm { ~ r e s i d u a l = r e m a i n i n g ~ c o n t r a c t ~ v i o l a t i o n s } . } \end{array}
$$

An accusation event occurs when a confirmed structured finding attacks a protected claim or the returned claim-support map removes or incorrectly relabels it. Other supportedcontent preservation failures contribute to residual violations but not necessarily to Accus. Residual violations also include unresolved repairs, schema errors, and invalid evidence references. Limitation-drift mutations and their associated repairs are removed from the per-output Recall and Repair denominators. For every reported group, we average the output-level metrics over the applicable instances and agent configurations within each repetition, then report the mean and sample standard deviation across the five repetitions.

These metrics determine a single contract-level decision. An agent output passes only if every scored violation is identified, every required repair is completed, all protected claims are preserved, no residual contract violation remains, and all prescribed artifacts are parseable and traceable. Traceability requires valid references to public evidence and the exact preservation of required public claim identifiers.

## 5 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub>

We organize the evaluation around three research questions:

• RQ1: How efectively can current general-purpose AI agents restore artifact integrity?

• RQ2: How does repair reliability change as tasks impose increasingly complex relations across artifacts?

• RQ3: In what ways do current artifact-integrity repairs remain unreliable?

Experimental setup: For this study, an AI agent is the deployed combination of a backend LLM and an agent runtime. The LLM provides the underlying reasoning and generation capabilities, while the runtime manages context, exposes tools and files, and carries out edits in the workspace. Each agent configuration fixes the runtime, backend LLM, and exposed inference settings.

We construct 23 agent configurations using Codex CLI, Cursor Agent, and OpenCode to inspect and edit self-contai ned workspaces that directly match NetArtifactBench’s interface. This combination lets us vary model capability and inference settings without changing the benchmark or pack age format. The 23 agent configurations and 52 instances define 1,196 configuration–instance pairs. Each agent configuration is executed five times on every instance, yielding 5,980 evaluable outputs<sup>1</sup>. This work focuses on general-purpose agents. Network-specialized agents contain built-in telemetry, simulators, APIs, and task-specific state [20, 23, 26, 27]. Testing them would make it dificult to distinguish whether performance came purely from artifact-integrity reasoning or from additional support by the specialized environment.

## 5.1 RQ1: Overall Efectiveness

Overall, 3,904 of the 5,980 outputs pass the contract (65.3%), with 74.2% for Codex, 71.2% for Cursor, and 44.0% for Open-Code (aggregated from Table 2). For GPT-5.5 under Codex, contract pass rates rise from 73.1% at low efort to 76.5%, 76.9%, and 81.9% at medium, high, and xhigh efort. The added GPT-5.3 Codex Spark, GPT-5.6 Luna, and GPT-5.6 Sol configurations attain 66.9%, 75.0%, and 78.5%, respectively. The added Cursor Grok 4.5/high, Gemini 3.1 Pro, and Claude Sonnet 5/high configurations attain 78.5%, 66.9%, and 80.0%, respectively. Configuration-level rates span 66.9–81.9% within Codex and 45.4–80.8% within Cursor. The six OpenCode configurations span 1.2–72.3%, with DeepSeek V4 Flash attaining the highest OpenCode rate.

Finding 1: The evaluated general-purpose AI agents do not reliably restore artifact integrity in this suite.

## 5.2 RQ2: Structural Complexity

Contract pass rates decline across the structural tiers in this suite. From Tier 1 to Tier 5, they fall from 98.9% to 26.3% for Codex, from 93.8% to 28.4% for Cursor, and from 60.8% to 17.0% for OpenCode (Table 3). All three runtimes exhibit lower pass rates in more challenging structural tiers that contain coordinated or inferred cross-artifact relations.

Finding 2: Across all three runtimes, contract pass rates decline as repairs move from directly exposed contradictions (Tiers 1–2) to following, propagating, or inferring relations across artifacts (Tiers 3–5).

## 5.3 RQ3: Reliability Limits

Partial contract completion: This partial completion appears at every structural tier. For every runtime–tier combination, mean recall, repair, and label rates exceed the contract pass rate, with higher structural tiers exhibiting wider gaps. Tier 5 provides the clearest illustration. Label accuracy exceeds the contract pass rate for every runtime. Mean label accuracies are 96.2%, 93.4%, and 60.2% for Codex, Cursor, and OpenCode, while their contract pass rates are 26.3%, 28.4%, and 17.0%. The corresponding recall rates are 51.1%, 57.0%, and 33.9%, and repair rates are 59.6%, 58.1%, and 35.0%. As the contract pass requires that every obligation be jointly fulfilled, these rates show that agents often complete only part of the repair, but do not reveal which unmet obligation most often prevents a full pass. The hardest instance-level outcomes likewise span the later tiers. Two cross-artifact instances receive no contract pass in any of 115 outputs per instance: one Tier 4 scope-propagation task and one Tier 5 join task drawn from programmable data-plane metadata. Every returned package for these instances violates at least one declared contract obligation.

Observedfailure flags: One failed output can violate several contract obligations and thus contribute to multiple failure categories, so the percentages below do not sum to 100%. Among the 2,076 output-level failures, 68.5% leave at least one required diagnosis or repair incomplete, 47.9% omit

Table 2: Avera<sub>g</sub>e <sub>p</sub>erformance metrics across a<sub>g</sub>ent runtimes and LLM backends. Entries re<sub>p</sub>ort the mean ± sam<sub>p</sub>le <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on across</sub> fi<sub>ve repe</sub>titi<sub>ons, eac</sub>h <sub>average</sub>d <sub>over a</sub>ll 52 i<sub>ns</sub>t<sub>ances.</sub>
<table><tr><td rowspan="2">Runtime</td><td rowspan="2">Backend / setting</td><td colspan="4">Artifact integrity (% ↑)</td><td colspan="2">Violations (↓)</td></tr><tr><td>Contract</td><td>Recall</td><td>Repair</td><td>Label</td><td>Accus.</td><td>Resid.</td></tr><tr><td rowspan="9">Codex</td><td>GPT-5.3 Codex Spark / high</td><td> $6 6 . 9 \pm 3 . 7$ </td><td> $7 4 . 6 { \pm } 4 . 2 $ </td><td> $7 8 . 7 \pm 3 . 2 $ </td><td> $9 4 . 3 { \pm } 1 . 8 $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.642±0.057</td></tr><tr><td>GPT-5.4 mini / medium</td><td> $6 8 . 1 \pm 4 . 2$ </td><td> $8 0 . 2 { \pm } 2 . 2 $ </td><td> $8 3 . 1 { \pm } 2 . 5 $ </td><td> $9 8 . 5 { \pm } 0 . 8 $ </td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 6 6 9 { \scriptstyle \pm 0 . 0 9 6 }$ </td></tr><tr><td>GPT-5.4 / medium</td><td> $7 1 . 2 \pm 1 . 9$ </td><td> $8 5 . 8 { \pm } 2 . 8 $ </td><td> $8 5 . 4 \pm 0 . 8$ </td><td> $9 9 . 2 { \pm } 1 . 1 $ </td><td> $0 . 0 0 8 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 5 6 9 { \scriptstyle \pm 0 . 0 7 0 }$ </td></tr><tr><td>GPT-5.5 / low</td><td> $7 3 . 1 \pm 2 . 7$ </td><td> $8 4 . 6 \pm 1 . 2 $ </td><td> $8 6 . 3 { \pm } 1 . 6 $ </td><td> $9 9 . 1 { \pm } 1 . 0 $ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 5 2 3 { \scriptstyle \pm 0 . 0 6 7 }$ </td></tr><tr><td>GPT-5.5 / medium</td><td> $7 6 . 5 \pm 1 . 6$ </td><td> $8 6 . 9 { \pm } 2 . 2 $ </td><td> $8 8 . 7 { \pm } 0 . 8 $ </td><td> $9 9 . 6 { \pm } 0 . 6 $ </td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 4 4 6 { \pm } 0 . 0 4 2$ </td></tr><tr><td>GPT-5.5 / high</td><td> $7 6 . 9 \pm 2 . 4$ </td><td> $8 8 . 5 { \pm } 1 . 8 $ </td><td> $8 8 . 5 { \pm } 1 . 2 $ </td><td> $9 9 . 4 \pm 0 . 6 $ </td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 4 4 2 { \scriptstyle \pm 0 . 0 4 5 }$ </td></tr><tr><td>GPT-5.5 / xhigh</td><td> ${ \bf 8 1 . 9 \pm 2 . 9 }$ </td><td> $9 0 . 8 { \pm } 2 . 0 $ </td><td> $9 0 . 6 \pm 1 . 9$ </td><td> $9 9 . 8 { \pm } 0 . 3 $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 3 4 6 { \pm } 0 . 0 9 0$ </td></tr><tr><td>GPT-5.6 Luna / high GPT-5.6 Sol / high</td><td> $7 5 . 0 \pm 1 . 4$ </td><td> $9 0 . 2 \pm 1 . 6 $ </td><td> $8 6 . 2 \pm 1 . 1$ </td><td> $9 7 . 2 \pm 1 . 7 $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 4 5 4 { \scriptstyle \pm 0 . 0 4 6 }$ </td></tr><tr><td></td><td> $7 8 . 5 \pm 1 . 6$ </td><td> $9 1 . 2 { \pm } 2 . 1 $ </td><td> $8 8 . 8 \pm 0 . 9$ </td><td> $9 9 . 7 { \pm } 0 . 3 $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 3 9 2 { \scriptstyle \pm 0 . 0 8 3 }$ </td></tr><tr><td rowspan="8">Cursor</td><td>Claude Opus 4.8</td><td> ${ \bf 8 0 . 8 \pm 1 . 9 }$ </td><td>92.9±1.7</td><td>91.0±1.1</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \qquad $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.350±0.029</td></tr><tr><td>Claude Sonnet 5</td><td> $8 0 . 0 \pm 2 . 2$ </td><td>90.8±2.0</td><td>89.6±1.7</td><td> $9 8 . 9 { \pm } 0 . 9$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.358±0.092</td></tr><tr><td>Composer 2.5</td><td> $7 5 . 4 \pm 2 . 1$ </td><td>92.5±1.4</td><td> $8 8 . 3 { \pm } 1 . 4 $ </td><td> $9 9 . 6 { \pm } 0 . 4 $ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.504±0.032</td></tr><tr><td>Gemini 3.1 Pro</td><td> $6 6 . 9 \pm 3 . 7$ </td><td>79.6±1.3</td><td> $8 1 . 9 { \pm } 1 . 3 $ </td><td>98.3±1.2</td><td> $0 . 0 0 8 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td>0.681±0.050</td></tr><tr><td>Gemini 3.5 Flash</td><td> $7 0 . 0 \pm 3 . 5$ </td><td>87.9±3.7</td><td>84.2±1.5</td><td>97.8±0.9</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.608±0.097</td></tr><tr><td>GPT-5.4</td><td> $7 2 . 7 \pm 0 . 9$ </td><td>86.7±2.1</td><td>86.5±1.0</td><td>99.2±0.7</td><td> $0 . 0 1 9 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td>0.531±0.029</td></tr><tr><td>GPT-5.4 nano</td><td> $4 5 . 4 \pm 8 . 2$ </td><td> $5 3 . 1 \pm 6 . 4$ </td><td>65.6±5.4</td><td>79.4±3.9</td><td> $0 . 0 8 1 { \pm } 0 . 0 3 2$ </td><td>1.269±0.296</td></tr><tr><td>Grok 4.5</td><td> $7 8 . 5 \pm 0 . 9$ </td><td>95.4±0.8</td><td>89.4±0.7</td><td>99.9±0.2</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.400±0.025</td></tr><tr><td rowspan="6">OpenCode</td><td>DiffusionGemma</td><td> $4 8 . 1 \pm 6 . 5$ </td><td>55.8±4.8</td><td>62.9±3.9</td><td>77.7±5.8</td><td> $0 . 0 5 4 { \pm } 0 . 0 3 9$ </td><td>1.462±0.224</td></tr><tr><td>DeepSeek V4 Flash</td><td> $7 2 . 3 \pm 5 . 5$ </td><td> $8 6 . 7 \pm 3 . 9$ </td><td> $8 3 . 1 { \pm } 4 . 1$ </td><td>93.3±2.6</td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 5 0 0 { \scriptstyle \pm 0 . 0 9 1 }$ </td></tr><tr><td>MiniMax M2.5</td><td> $6 6 . 2 \pm 1 0 . 0$ </td><td> $8 0 . 2 \pm 1 0 . 6 $ </td><td> $7 9 . 4 \pm 1 1 . 2$ </td><td> $9 0 . 7 \pm 1 2 . 9$ </td><td> $0 . 0 3 1 { \pm } 0 . 0 2 9$ </td><td> $0 . 9 1 9 { \scriptstyle \pm 0 . 7 0 5 }$ </td></tr><tr><td>Qwen3.6-35B</td><td> $6 0 . 8 \pm 5 . 0$ </td><td> $7 5 . 8 { \pm } 2 . 7 $ </td><td> $7 3 . 5 { \pm } 2 . 5 $ </td><td> $8 4 . 2 { \pm } 2 . 6 $ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 8 5 8 { \pm } 0 . 1 9 3$ </td></tr><tr><td>Qwen3 8B</td><td> $1 . 2 \pm 1 . 7$ </td><td> $1 . 7 { \pm } 1 . 4 $ </td><td> $2 . 9 { \pm } 1 . 8 $ </td><td> $3 . 8 \pm 2 . 2$ </td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $3 . 2 1 9 { \scriptstyle \pm 0 . 6 2 3 }$ </td></tr><tr><td>Qwen3 14B</td><td> $1 5 . 4 \pm 4 . 5$ </td><td> $2 4 . 4 \pm 4 . 2$ </td><td> $3 1 . 5 { \pm } 4 . 9$ </td><td> $3 9 . 4 \pm 6 . 8$ </td><td> $0 . 0 0 8 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $1 . 7 6 5 { \pm } 0 . 1 7 7$ </td></tr></table>

Table 3: Contract <sub>p</sub>ass rate (%) b<sub>y</sub> structural tier (mean ± sam<sub>p</sub>le standard deviation across five re<sub>p</sub>etitions).

Finding 3: The evaluated agents often satisfy individual repair obligations without completing the full artifact integrity contract, and repeated executions can change whether the repair passes.

<table><tr><td>Runtime</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Codex</td><td> $9 8 . 9 { \pm } 0 . 4 $ </td><td>97.8±2.2</td><td> $7 8 . 0 { \pm } 3 . 1 $ </td><td> $4 9 . 3 { \pm } 5 . 8$ </td><td>26.3±2.3</td></tr><tr><td>Cursor</td><td>93.8±1.5</td><td>86.5±5.5</td><td> $8 0 . 2 { \pm } 2 . 1 $ </td><td> $4 0 . 0 { \pm } 3 . 1 $ </td><td>28.4±2.3</td></tr><tr><td> ${ \mathrm { O p e n C o d e } }$ </td><td>60.8±4.4</td><td>49.3±3.7</td><td> $4 7 . 3 { \pm } 2 . 9 $ </td><td> $2 3 . 3 { \pm } 4 . 7$ </td><td>17.0±4.5</td></tr></table>

## 6 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Several works explore applying LLMs and AI agents to network systems. NetConfEval [24] and VPP [13] evaluate required report information, 33.6% fail at least one supportedcontent preservation check, and 22.6% return invalid structured output. Required-artifact omission appears in 7.1%, and 3.9% retain the injected unsupported conclusion.

Run-to-run instability: Among the 1,196 configuration-ins tance pairs, 331 (27.7%) contain both passing and failing outputs. Mixed outcomes occur in 19.9% of Codex pairs, 25.0% of Cursor pairs, and 42.9% of OpenCode pairs. For the smallest Cursor backend, the first-execution pass rate is 59.6%, compared with 45.4% across five executions.

configuration generation and correction, while NetAssistant [25], LLexus [11], OSS-GPT [12], Confucius [27], and ArachNet [20] support multi-step diagnosis, operations, and measurement workflows. Dedicated AI agent benchmarks make these tasks testable and reproducible. Cornetto [ checks configuration repairs and regressions, NIKA [26] re plays dynamic incidents, and NetAgentBench [23] evaluates multi-turn configuration through a deterministic state machine. In these works, the evaluation unit remains the immediate configuration, incident, or workflow outcome. NetArtifactBench evaluates whether a repair restores the relations within an experiment record while preserving claims that remain supported.

Outside the networking domain, SWE-bench evaluates whether repository edits resolve a software issue, while FEVER tests whether textual evidence supports or refutes a claim [10, 22]. Artisan evaluates scripts that reproduce published results, and ArtifactCopilot automates environment setup and recovery during artifact execution [1, 28]. These tasks evaluate issue resolution, claim classification, result reproduction, and successful execution, respectively. NetArtifactBench instead evaluates whether a repair leaves an entire network experiment record coherent, traceable, and free of damage to claims that remain supported.

## 7 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

As AI agents begin to enter network systems, the artifacts they modify become part of the record from which results are interpreted, verified, and reused. A repair can appear locally plausible while leaving that record internally inconsistent or no longer traceable to its evidence. NetArtifactBench makes this record-level failure measurable through 52 sourcegrounded instances that require agents to identify broken claim–evidence relations, propagate corrections across affected artifacts, and preserve supported claims.

Our evaluation reveals that strong general-purpose AI agents can handle direct contradictions well, but struggle when repair depends on implicit relations or coordinated changes across artifacts. Local correctness does not ensure a complete and auditable record of the experiment. These results make artifact integrity a first-class requirement for AI agents operating on network systems. Reliable AI agents for network systems must preserve both the record’s substantive support and the explicit structure that later tools and agents need to inspect and reuse it.

The controlled violations probe repair capability, not the prevalence or natural distribution of integrity failures in real network workflows. Future work should study naturally occurring cases, independent oracle review, more permissive semantic evaluation, and network-specialized agents.

## R<sub>e</sub>f<sub>erences</sub>

[1] Doehyun Baek and Michael Pradel. 2026. Artisan: Agentic Artifact Evaluation. arXiv:2602.10046 [cs.SE] doi:10.48550/arXiv.2602.10046

[2] Sean Bechhofer, Iain Buchan, David De Roure, Paolo Missier, John Ainsworth,Jiten Bhagat, Philip Couch, Don Cruickshank, Mark Delderfield, Ian Dunlop, Matthew Gamble, Danius Michaelides, Stuart Owen, David Newman, Shoaib Sufi, and Carole Goble. 2013. Why Linked Data is Not Enough for Scientists. Future Generation Computer Systems 29, 2 (2013), 599–611. doi:10.1016/j.future.2011.08.004

[3] Pat Bosshart, Dan Daly, Glen Gibb, Martin Izzard, Nick McKeown, Jennifer Rexford, Cole Schlesinger, Dan Talayco, Amin Vahdat, George Varghese, and David Walker. 2014. P4: Programming Protocol Independent Packet Processors. ACM SIGCOMM Computer Communication Review 44, 3 (2014), 87–95. doi:10.1145/2656877.2656890

[4] Peter Buneman, Sanjeev Khanna, and Wang-Chiew Tan. 2001. Why and Where: A Characterization of Data Provenance. In Database Theory—ICDT 2001 (Lecture Notes in Computer Science, Vol. 1973). Springer, 316–330. doi:10.1007/3-540-44503-X\_20

[5] James Cheney, Laura Chiticariu, and Wang-Chiew Tan. 2009. Provenance in Databases: Why, How, and Where. Foundations and Trends in Databases 1, 4 (2009), 379–474. doi:10.1561/1900000006

[6] Alexander Clemm, Laurent Ciavaglia, Lisandro Zambenedetti Granville, and Jef Tantsura. 2022. Intent-Based Networking - Concepts and Definitions. RFC 9315. RFC Editor. doi:10.17487/RFC9315

[7] Ericsson. 2025. AI Agents in the Telecommunication Network Architecture. White Paper BCSS-25:024439. Ericsson. https://www.ericsson.com/en/reports-and-papers/white-papers/aiagents-and-network-architecture

[8] Ari Fogel, Stanley Fung, Luis Pedrosa, Meg Walraed-Sullivan, Ramesh Govindan, Ratul Mahajan, and Todd Millstein. 2015. A General Ap proach to Network Configuration Analysis. In Proceedings of the 12th USENIX Symposium on Networked Systems Design and Implementation (NSDI). USENIX Association, 469–483. https://www.usenix.org/ conference/nsdi15/technical-sessions/presentation/fogel

[9] Harvard MadSys Lab. 2026. FreeInference. https://freeinference.org/ Accessed 2026-07-13.

[10] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=VTF8yNQM66

[11] Pedro Las-Casas, Alok Gautum Kumbhare, Rodrigo Fonseca, and Sharad Agarwal. 2024. LLexus: An AI Agent System for Incident Management. ACM SIGOPS Operating Systems Review 58, 1 (2024), 23–36. doi:10.1145/3689051.3689056

[12] Abdelkader Mekrache, Adlen Ksentini, and Christos Verikoukis. 2025. OSS-GPT: An LLM-Powered Intent-Driven Operations Support System for 6G Networks. In Proceedings ofthe 2025 IEEE 11th International Conference on Network Softwarization. IEEE, 155–163. doi:10.1109/ NetSoft64993.2025.11080632

[13] Rajdeep Mondal, Alan Tang, Ryan Beckett, Todd Millstein, and George Varghese. 2023. What do LLMs Need to Synthesize Correct Router Configurations?. In Proceedings ofthe 22nd ACM Workshop on Hot Topics in Networks. ACM, 189–195. doi:10.1145/3626111.3628194

[14] Luc Moreau and Paolo Missier. 2013. PROV-DM: The PROV Data Model. W3C Recommendation. World Wide Web Consortium. https: //www.w3.org/TR/2013/REC-prov-dm-20130430/ 30 April 2013.

[15] Mark Newman. 2025. Agentic AI and Autonomy: CSPs Set Out Their Strategies. Research Report. TM Forum. https://inform.tmforum.org/research-and-analysis/reports/agenticai-and-autonomy-csps-set-out-their-strategies

[16] P4 Language Consortium. 2026. P4C: P4\_16 Reference Compiler. https: //github.com/p4lang/p4c Accessed 2026-07-06.

[17] Vern Paxson. 1999. Bro: A System for Detecting Network Intruders in Real-Time. Computer Networks 31, 23–24 (1999), 2435–2463. doi:10. 1016/S1389-1286(99)00112-7

[18] Ben Pfaf, Justin Pettit, Teemu Koponen, Ethan Jackson, Andy Zhou, Jarno Rajahalme, Jesse Gross, Alex Wang, Joe Stringer, Pravin Shelar, Keith Amidon, and Martín Casado. 2015. The Design and Implementation of Open vSwitch. In Proceedings of the 12th USENIX Symposium on Networked Systems Design and Implementation (NSDI). USENIX Association, 117–130. https://www.usenix.org/conference/nsdi15/technicalsessions/presentation/pfaf

[19] Ioannis Protogeros, Rufat Asadli, Benjamin Hofman, and Laurent Vanbever. 2026. Benchmarking LLM-Driven Network Configuration Repair. arXiv:2604.22513 [cs.NI] doi:10.48550/arXiv.2604.22513

[20] Alagappan Ramanathan, Eunju Kang, Dongsu Han, and Sangeetha Abdu Jyothi. 2025. Towards an Agentic Workflow for Internet Measurement Research. In Proceedings of the 24th ACM Workshop on Hot Topics in Networks. ACM, 61–68. doi:10.1145/3772356.3772409

[21] Stian Soiland-Reyes, Peter Sefton, Mercè Crosas, Leyla Jael Castro, Frederik Coppens, José M. Fernández, Daniel Garijo, Björn Grüning, Marco La Rosa, Simone Leo, Eoghan Ó Carragáin, Marc Portier, Ana Trisovic, RO-Crate Community, Paul Groth, and Carole Goble. 2022. Packaging Research Artefacts with RO-Crate. Data Science 5, 2 (2022), 97–138. doi:10.3233/DS-210053

[22] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: A Large-scale Dataset for Fact Extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers). Association for Computational Linguistics, 809–819. doi:10.18653/v1/N18-1074

[23] Ahmed Twabi, Yepeng Ding, and Tohru Kondo. 2026. NetAgentBench: A State-Centric Benchmark for Evaluating Agentic Network Configuration. arXiv:2604.09678 [cs.NI] doi:10.48550/arXiv.2604.09678

[24] Changjie Wang, Mariano Scazzariello, Alireza Farshin, Simone Ferlin, Dejan Kostić, and Marco Chiesa. 2024. NetConfEval: Can LLMs Facili tate Network Configuration? Proceedings of the ACM on Networking 2, CoNEXT2, Article 7 (June 2024), 25 pages. doi:10.1145/3656296

[25] Haopei Wang, Anubhavnidhi Abhashkumar, Changyu Lin, Tianrong Zhang, Xiaoming Gu, Ning Ma, Chang Wu, Songlin Liu, Wei Zhou, Yongbin Dong, Weirong Jiang, and Yi Wang. 2024. NetAssistant: Dialogue Based Network Diagnosis in Data Center Networks. In Proceedings of the 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI). USENIX Association, 2011–2024. https: //www.usenix.org/conference/nsdi24/presentation/wang-haopei

[26] Zhihao Wang, Alessandro Cornacchia, Alessio Sacco, Franco Galante, Marco Canini, and Dingde Jiang. 2025. A Network

Arena for Benchmarking AI Agents on Network Troubleshooting. arXiv:2512.16381 [cs.NI] doi:10.48550/arXiv.2512.16381

[27] Zhaodong Wang, Samuel Lin, Guanqing Yan, Soudeh Ghorbani, Minlan Yu, Jiawei Zhou, Nathan Hu, Lopa Baruah, Sam Peters, Srikanth Kamath, Jerry Yang, and Ying Zhang. 2025. Intent-Driven Network Management with Multi-Agent LLMs: The Confucius Framework. In Proceedings of the ACM SIGCOMM 2025 Conference. ACM, 347–362. doi:10.1145/3718958.3750537

[28] Zhaonan Wu, Yanjie Zhao, Zhenpeng Chen, Zheng Wang, and Haoyu Wang. 2026. Agent-Based Software Artifact Evaluation. arXiv:2602.02235 [cs.SE] doi:10.48550/arXiv.2602.02235