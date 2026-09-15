# Externalizing Requirement-to-Repair Artifacts as Observable Traces for LLM-Based Program Repair

Zewen Tao and Shin-nosuke Ishikawa

Rikkyo University

26vr047w@rikkyo.ac.jp, shinnosuke-ishikawa@rikkyo.ac.jp

## Abstract

Repository-level repair requires not only correct patches but also inspectable records that explain how issue requirements are translated into code changes and post-edit evidence. We contribute THEMIS, a stageaware repair workflow that externalizes this requirement-to-repair process through semantic interpretation, a runtime requirement–code graph, graph-derived Developer guidance, retained repair rationale and patches, and postedit audit records. A retrospective audit of 300 SWE-bench Lite cases demonstrates that these artifacts provide broad support for cross-stage inspection: a complete Developer rationale is available for 288 cases, and 214 cases (71.3%) retain a complete audited field set connecting the selected stages. The retained records further enable systematic measurement of crossstage correspondence: target symbols recur in 62.6% of Developer rationales and in 62.8% of patches, rising to 75.8% when related symbols are included. In a paired 100-case comparison, the relational workflow resolves 19 cases versus 9 for the direct same-input condition; because the two conditions also differ in Analyzer output, graph-derived distillation, and Judge records, we report this as preliminary, workflow-level evidence rather than a causal effect of the graph component. Together, these results show that THEMIS makes otherwise implicit requirement-to-repair transitions inspectable, enabling systematic examination of how repair decisions persist, align, and evolve across stages.

## 1 Introduction

Automatically repairing real software issues requires more than generating plausible code edits. A GitHub issue may describe observed behavior, expected behavior, failing tests, examples, and implicit domain constraints. Benchmarks such as SWE-bench make this setting concrete by asking systems to resolve real repository-level issues from natural-language reports and code contexts (Jimenez et al., 2024). LLMs have expanded the space of possible repairs, but repository-level automated program repair remains difficult because a system must jointly understand the report, localize relevant code, infer the intended behavior, and produce a minimal patch that satisfies the test oracle (Monperrus, 2018; Jiang et al., 2023).

The central information problem arises between repair stages: a system must carry issue requirements into code edits and post-edit audit while preserving records that distinguish each stage and make their relations inspectable. The design therefore begins with semantic interpretation of the issue, makes the resulting requirement-code relations explicit, distills selected relations into a bounded rationale and edit, and retains post-edit evidence for audit. This why-chain connects the system’s stages: semantic interpretation supplies the meaning needed to relate requirements to code; explicit relation state makes those connections inspectable; bounded rationale and editing preserve a focused execution record; and post-edit evidence shows what can be examined after the change. This paper retrospectively and descriptively audits retained records from 300 SWE-bench Lite cases. Conversational repair systems add feedback from tests or prior attempts (Xia and Zhang, 2023; Xia et al., 2024b); this paper examines one repair pass.

We ask: RQ1: What stage-specific information can be externalized to make the transition from issue requirements to code edits and postedit audit inspectable, and where do continuity gaps arise? RQ2: What observable correspondences and disagreements arise among target localization, repair rationale, generated edits, and post-edit evidence in repository-level repair? RQ3: What preliminary outcome differences are observed between a relational repair workflow and a direct same-input repair condition? RQ1 is operationalized through selectedfield availability and continuity gaps; RQ2 through retained post-edit target summaries, Developer rationale, and patch-text lexical comparisons; and RQ3 through a paired 100-case workflow comparison.

We present THEMIS as one instantiation of an explicit selected stage-aware audit projection.<sup>1</sup> It constructs repair context from a requirement-code graph, then retains selected graph-derived summaries, Developer rationale, patches, and postedit diagnostics. Its Developer receives a textual distillation. The projection is inspectable through its readable retained fields and traceable through the workflow’s asserted links among them; the study audits those selected serialized records in an augmented-input, single-revision configuration.

This paper makes three contributions.

1. Stage information and continuity for RQ1: We define the selected stage-aware projection and report retained-field availability and continuity gaps in Table 2.

2. Observable correspondence and disagreement for RQ2: We characterize lexical correspondences and disagreements among retained post-edit target summaries, Developer rationale, and patches in Table 3 and two illustrative cases in Section 5.3.

3. Preliminary workflow outcomes for RQ3: We compare 100 paired cases between the relational workflow and a direct same-input condition: 19/100 versus 9/100 official resolutions and 89/100 versus 43/100 nonempty patches. This workflow-level comparison spans conditions with distinct Analyzer output, graph-derived distillation, and Judge records.

## 2 Related Work

Automated program repair. Automated program repair is commonly formulated as generateand-validate search: candidate edits are generated and checked against tests, specifications, crashes, or semantic constraints (Monperrus, 2018). Classical systems include search-based repair such as GenProg (Weimer et al., 2009), empirical studies of test-suite repair (Le Goues et al., 2012), learned patch priors such as Prophet (Long and Rinard, 2016), and semantic/symbolic systems such as SemFix, Angelix, and Nopol (Nguyen et al., 2013; Mechtaev et al., 2016; Xuan et al., 2017). THEMIS follows the constraint-guided repair tradition, but its constraints are derived from issue text and represented as requirement-code graph relations.

LLM-based repository repair. LLMs have shifted repair from hand-designed templates toward flexible code generation. Prior work studies LLMs for bug fixing and repository-level issue resolution, including conversational repair, SWE-bench agents, and agentless repair pipelines (Sobania et al., 2023; Xia and Zhang, 2023; Xia et al., 2024b; Jimenez et al., 2024; Yang et al., 2024; Xia et al., 2024a). THEMIS complements these systems by structuring the information given to a Developer model through explicit issue requirements and code mappings.

Graphs and semantic guidance for code. Graph representations have long been used to encode code structure beyond token sequences. Code property graphs combine syntax, control flow, and dependence information for program analysis (Yamaguchi et al., 2014); neural code models use syntactic and semantic graph relations for code understanding (Allamanis et al., 2018; Alon et al., 2019a,b; Guo et al., 2021). THEMIS adds issuederived requirement nodes and violation/advisory edges, using the graph as both a code representation and a repair-context state. The selected projection declares stage roles and selected relations explicitly, and structured logging can make the same declarations. The direct same-input baseline compares the reported workflow conditions across their representation formats.

## Software traceability and agent observability.

A separate line of work studies traceability directly: software and systems traceability surveys and IR-based trace-link recovery connect requirements to downstream artifacts and evaluate how reliably such links can be recovered (Cleland-Huang et al., 2012; Borg et al., 2014). Recent surveys of LLM agents similarly call for unified schemas that record how evidence, tool calls, and intermediate reasoning shape an agent’s final action, so that execution provenance can be audited after the fact (Wang et al., 2026). THEMIS targets this same auditability goal for repository-level repair specifically, externalizing a requirement-torepair trace rather than a general-purpose agent ex-

ecution log.

## 3 System Architecture

THEMIS contains four modules: an Advanced Code Analyzer, an Enhanced Graph Manager, a Developer, and a Judge. Its graph implements explicit requirement-code relations within the reported workflow. Together the modules define an explicit, selected stage-aware audit projection that captures pre-edit interpretation and guidance, repair execution and rationale, and post-edit evidence. This separation serves the method’s inspectability objective: the analyzer keeps issuelevel semantic hypotheses distinct from structural relation claims; the graph manager exposes selected requirement-target-constraint relations as inspectable anchors; the Developer retains the guidance-to-rationale-to-patch transition as an inspectable execution record; and the rebuilt graph and Judge retain post-edit evidence as a distinct audit stage. Figure 1 preserves this ordering across the reported single-revision boundary: pre-edit guidance precedes one repair execution, followed by post-edit audit.

## 3.1 Advanced Code Analyzer

The Advanced Code Analyzer is introduced to retain issue-level semantic hypotheses from the issue text and local code separately from structural relation claims. This separation keeps the two forms of pre-edit information inspectable and the semantic findings distinct from graph edges. It receives the issue text, selected target files, optional code context, and analysis options. It performs bug classification, semantic extraction, context enhancement, concept mapping, pattern matching, and multi-round LLM reasoning. Its outputs are findings, recommendations, confidence scores, strategy metadata, and usage information when available. These outputs enter the Developer prompt as semantic priors: they suggest likely fault mechanisms or repair directions, remain distinct from structural targets, and occupy a separate pre-edit record category.

## 3.2 Enhanced Graph Manager

The Enhanced Graph Manager is introduced to expose selected requirement-target-constraint relations as inspectable anchors for pre-edit distillation. Its expected benefit is that these selected relations and code neighborhoods can be examined as explicit structural records. The Enhanced Graph

Manager builds the requirement-code graph. A structural extractor parses Python source into code nodes such as functions, classes, and variables. A semantic injector decomposes issue text into RequirementNodes and maps them to candidate code nodes. A dependency tracer adds code-neighborhood information. A violation flagger analyzes requirement-code pairs and produces ViolationEdges.

The graph distinguishes blocking and nonblocking guidance. High-signal blocking mismatches are recorded as VIOLATES edges. Lower-confidence or non-blocking signals are recorded as ADVISORY edges. The graph manager then summarizes these edges in an analysis report containing violation counts, prioritized violations, evidence scores, confidence values, and dependency statistics.

## 3.3 Developer

The Developer consumes selected actionable context to produce one bounded edit. It is introduced to preserve the guidance-to-rationaleto-patch transition as an inspectable execution record. Textual distillation is a pre-edit artifact that presents selected priorities, targets, plans, and local ingredients in readable form. The Developer rationale and patch are execution artifacts that retain the transition from guidance to execution. The Developer receives the current file map, original requirements, Advanced Analyzer findings, graph-derived violation priorities, target symbols, repair operator plans, and local code ingredients. Graph artifacts reach the Developer as concise textual guidance. The Developer produces bounded edits in either search/replace form or symbolrewrite form. The runner accepts edits that satisfy file, syntax, and substantive-change checks.

## 3.4 Judge

The Judge creates a post-edit record from the graph rebuilt after the single edit, separating remaining blocking conflicts from advisory findings. It is introduced to retain post-edit evidence as a distinct audit stage in the single-revision workflow. This timing positions the post-edit conditions and repair brief for inspection after Developer execution. The Judge consumes a revised graph, the requirements, and optionally the baseline graph. Its hard check scans graph edges of type VIOLATES and ADVISORY, separates blocking conflicts from advisory findings, and builds a conflict report. It also constructs a repair brief with a target symbol, related symbols, issue summary, expected behavior, and minimalchange hint. In the current experiments, these outputs serve exclusively as post-edit audit records after the workflow’s one Developer revision.

![](images/a67890bb7789c9bb7f600fd8e0c028b8dfefb5ec3187ac8ee7325cd35cbff1a2.jpg)  
Figure 1: THEMIS artifact flow in the reported single-revision configuration. The Advanced Code Analyzer supplies soft semantic findings; the Enhanced Graph Manager makes requirements, targets, and constraints explicit; the Developer consumes readable distilled context and records its rationale and patch; and the Judge records post-edit conflict history and repair briefs. Dashed-border boxes are runtime-only graph states; shaded boxes are selected persisted artifacts. Solid arrows show reported data flow. Persisted artifacts and dashed links denote recorded, workflow-asserted relations. Judge outputs follow the edit and serve exclusively as post-edit audit records in the single-revision workflow.

## 3.5 First-Pass Repair Context

The first-pass context implements the pre-edit portion of the design chain: it keeps raw requirements and files, soft semantic clues, and explicit graph-derived constraints distinguishable while presenting their selected forms together for a bounded edit. The first Developer pass receives three kinds of information. First, raw inputs provide the requirements and selected source files. Second, the Advanced Code Analyzer provides semantic findings and recommendations. Third, the graph manager provides baseline graph-derived constraints: prioritized requirement-code violations, target symbols, repair operator plans, and local code ingredients. Judge conflict reports and repair briefs are generated after a Developer edit and graph rebuild, where they serve as post-edit audit records.

## 3.6 Inspectable Trace Artifacts

The architecture serializes selected artifacts so that pre-edit interpretation and targets, repair execution, and post-edit audit remain distinct for inspection. Canonical logs retain selected requirement and target summaries, Developer rationale, patch, and post-edit diagnostics. During analysis and graph rebuild, the runtime graph holds requirement nodes, code nodes, mappings, and conflict relations; the selected serialization defines the audit scope. The Developer receives textual distillation of selected graph-derived items. The audit measures availability, surface overlap, and quality flags in the retained records. Table 1 distinguishes their timing, consumers, and persistence.

We define an observable trace schema as τ = $( r , t , d , p , q )$ , where r is a serialized requirement summary, t is a selected target summary, d is Developer rationale, p is the patch record, and q is a post-edit diagnostic. A trace is inspectable when these stored artifacts are readable. It is traceable when the workflow asserts links among them. This tuple captures selected workflow-asserted records and relations for availability, surface-overlap, and quality-flag analysis.

## 4 Experimental Setup

We evaluate THEMIS on SWE-bench Lite, a 300- instance subset of real GitHub issue-resolution tasks from Python repositories (Jimenez et al., 2024). Each instance provides a base commit, issue text, optional hints, and tests used by the official evaluation harness. THEMIS consumes the issue text and selected source files, produces a patch as a git diff, and writes predictions in the JSONL format expected by the harness.

All reported runs use gpt-5.1-codex-mini for the repair workflow and the Advanced Code Analyzer, chosen for its balance of coding capability and inference cost relative to larger code-specialized models; we leave comparison across model families and scales to future work. The main run uses the integrated configuration with Advanced Analysis, graph construction, one Developer revision, graph rebuild, and post-edit Judge audit. Repair briefs and Judge diagnostics are generated after that edit and rebuild. Thus, the evaluation covers a single graph-guided repair pass and its post-edit audit records.

<table><tr><td>Artifact</td><td>Producer and timing</td><td>Consumer</td><td>Persistence in canonical logs</td></tr><tr><td>Runtime requirement- code graph</td><td>Graph Manager before edit; rebuilt after edit</td><td>Distillation and Judge</td><td>Runtime graph during analysis; se- lected summaries are serialized</td></tr><tr><td>Developer-facing distilla- tion</td><td>Graph Manager before the single Developer edit</td><td>Developer</td><td>Selected textual guidance derived from graph artifacts</td></tr><tr><td>Developer rationale and patch</td><td>Developer during the single edit</td><td>Runner and retrospec- tive audit</td><td>Rationale fields and patch record when available</td></tr><tr><td>Repair brief and conflict history</td><td>Judge after edit and graph rebuild</td><td>Retrospective audit</td><td>Selected post-edit summaries and conflict history</td></tr><tr><td>Official outcome</td><td>SWE-bench harness af- ter submission</td><td>Retrospective audit</td><td>Harness resolution record</td></tr></table>

Table 1: Artifact timing, consumers, and persistence. The repair brief and Judge diagnostics are post-edit records in the reported single-revision run.

The reported condition is explicitly augmented. THEMIS uses FAIL\_TO\_PASS test identifiers during file selection and requirement construction. The runner extracts explicit Python paths from issue text and failing-test metadata when available, otherwise searches the repository for issue terms and ranks candidate files. The requirement string concatenates the problem statement, available hints, FAIL\_TO\_PASS identifiers, and any manually supplied semantic contracts. The reported evidence applies to this augmented-input SWE-bench condition; issue-only systems are a future comparison target. Selected presets add bounded context, retrieval, or semantic-contract information for controlled slices.

We operationalize the three general research questions from retained records. RQ1 measures selected-field retention and gaps in the retained summary chain. RQ2 measures string-level surface correspondence or disagreement among the retained post-edit target summary, the last Developer rationale, and patch text. RQ3 is a paired 100-case direct same-input workflow comparison. All 100 pairs match instance IDs, gpt-5.1- codex-mini, seed 42, one Developer revision, the requirement-string builder, the selected-file strategy, and the actual selected files. Advanced Analyzer output, graph-derived distillation and relations, and Judge records occur exclusively in the relational workflow; the two conditions use different token budgets and interfaces. Full-run lineage predates input\_protocol and is established retrospectively from experiment metadata and baseline declarations. This comparison supports preliminary workflow-level interpretation across the reported condition differences. The target lineage is exactly post-edit meta. repair\_brief.target\_symbol: generated after the edit, it serves as a retrospective lexical target, distinct from pre-edit guidance and patch generation. The rationale concatenates the last recorded hypothesis\_root\_ cause, expected\_invariant, and patch\_ strategy; JSON null marks missing values, and None is a string value. Canonical patch text is model\_patch.

For matching, we apply Unicode-aware casefold(), replace :: with ., and match either the qualified or leaf form as a substring of casefolded rationale or model\_ patch. An expanded patch row is a hit when target\_symbol or any non-empty related\_ symbols[] entry matches model\_patch under that rule. This string-level rule measures surface correspondence among the retained records. The rationale cohort of 214 requires repair-brief core, non-empty conflict\_ metrics\_history, and non-empty last Developer rationale fields. The patch cohort of 207 requires repair-brief core and non-empty canonical model\_patch; the 205-case intersection is the joint cohort. Low confidence is meta. repair\_brief.confidence ≤ 0.2; a generic issue summary is trimmed exact membership in No clear relationship found or No validation functions found in codebase. Both flags use the 222 repairbrief-core cases as their denominator. We report official-harness resolved rate, resolved rate among non-empty-patch instances, and patch generation rate. Patch production records an edit, while official resolution records a benchmark-passing edit.

<table><tr><td>Serialized artifact</td><td>Available cases</td></tr><tr><td>Conflict history</td><td>300/300 (100.0%)</td></tr><tr><td>Complete Developer rationale</td><td>288/300 (96.0%)</td></tr><tr><td>Repair-brief core</td><td>222/300 (74.0%)</td></tr><tr><td>Complete audited field set</td><td>214/300 (71.3%)</td></tr></table>

Table 2: Retrospective availability of selected serialized artifacts across exactly nine default-run log directories.

## 5 Results

## 5.1 Availability of Stage-Specific Repair Records

RQ1 asks whether selected records form a complete field chain available for inspection. We check complete chains because an auditor needs all three retained field groups to read the cross-stage record together. We therefore audited exactly nine existing default-run log directories covering the 300 cases. Conflict history is a non-empty conflict\_metrics\_history list. Developer rationale requires non-empty hypothesis\_root\_cause, expected\_ invariant, and patch\_strategy. Repairbrief core requires non-empty requirement\_ id, target\_symbol, and expected\_ behavior. An audited field set is complete when all three field groups are retained.

For RQ1, Table 2 reports conflict history for all 300 cases and complete field-chain coverage for 214/300 (71.3%). The 86 remaining cases identify the retained fields that require attention for joint cross-stage inspection. The 214/300 figure reports selected-field coverage in the retained record set.

## 5.2 Surface Correspondence and Record-Quality Flags

For RQ2, Table 3 measures lexical overlap to make visible target recurrence in rationales and patches under the stated string rule. It reports descriptive quality flags separately to make uncertainty and limited specificity in the retained summaries visible. Under the stated string rule, recorded targets appear in 134/214 rationales and 130/207 patches;

<table><tr><td>Characterization</td><td>Cases</td></tr><tr><td>Target in rationale</td><td>134/214 (62.6%)</td></tr><tr><td>Target in patch</td><td>130/207 (62.8%)</td></tr><tr><td>Target or related symbol in patch</td><td>157/207 (75.8%)</td></tr><tr><td>Low-confidence repair brief</td><td>131/222 (59.0%)</td></tr><tr><td>Generic issue summary</td><td>138/222 (62.2%)</td></tr></table>

Table 3: Lexical alignment and quality flags for selected serialized artifacts.

allowing related symbols raises patch overlap to 157/207. These counts characterize lexical surface correspondence and disagreement among retained records.

Within the 205-case joint cohort, both rationale and patch match in 110 cases; the remaining 95/205 distribute across 18 rationale-only matches, 19 patch-only matches, and 58 records in the zero-overlap cell. Their respective resolved counts are 29 (26.4%), 8 (44.4%), 2 (10.5%), and 8 (13.8%). The matrix makes visible both stringsurface agreement and disagreement. Across the repair-brief-core cohort, 131/222 repair briefs are low confidence and 138/222 have generic issue summaries. Together, the overlap and flag results make correspondence, mismatch, uncertainty, and limited specificity visible in the retained records.

## 5.3 Two Illustrative Stage-Specific Record Cases

The two cases make RQ2 patterns inspectable beyond aggregate counts. They serve as illustrative records that connect the aggregate surface measures to one readable correspondence and one record-level mismatch.

The resolved case django\_\_ django-11620 is the correspondence case. It selected django/views/debug.py; its repair brief identifies REQ-029, targets resolve, and lists the related symbol technical\_404\_ response; the Developer rationale states that the relevant path should catch Http404 alongside Resolver404; and the patch makes that change. The official harness outcome is resolved. The retained records expose readable surface correspondence.

The unresolved case django\_\_ django-16139 is the disagreement case. It selected django/contrib/auth/forms.py; the Developer rationale targets UserChangeForm and a \_to\_field-aware URL; in contrast, the graph repair brief targets

<table><tr><td>Condition</td><td>Resolved</td></tr><tr><td>Complete audited field set Incomplete audited field set</td><td>47/214 (22.0%)</td></tr><tr><td>Target in rationale</td><td>11/86 (12.8%) 37/134 (27.6%)</td></tr><tr><td>Target absent from rationale</td><td>10/80 (12.5%)</td></tr><tr><td>Target in patch</td><td>31/130 (23.8%)</td></tr><tr><td>Target absent from patch</td><td>17/77 (22.1%)</td></tr></table>

Table 4: Descriptive official outcome associations for available artifact subsets.

SetPasswordForm, reports confidence 0.2, and states “No clear relationship found.” The patch modifies UserChangeForm, and the official harness outcome is unresolved. The retained records expose an inspectable record-level target mismatch.

## 5.4 Patch Production and Official Outcomes

This subsection reports outcome context through inspectable patch production and benchmark success, which answer different questions. In the 300- instance augmented-input, single-revision run, 279 cases produce non-empty patches (93.0%), 58 of 300 are officially resolved (19.3%), and 58/279 non-empty-patch cases are resolved (20.8%). The distinction keeps produced patch records and benchmark-passing outcomes separately visible. These results provide descriptive outcome context for the reported condition and identify causaleffect validation as a future target.

## 5.5 Descriptive Outcome Context by Recorded Condition

We compare recorded-condition strata for descriptive official-outcome context; denominators require their respective available fields.

The separations are 22.0% versus 12.8% for complete versus incomplete field chains, 27.6% versus 12.5% for rationale-target presence versus absence, and 23.8% versus 22.1% for patchtarget presence versus absence. Rationale-target presence separates these descriptive strata more than patch-target presence. These associations are correlational and likely confounded by case difficulty: cases with incomplete field chains may simply be harder cases where the pipeline broke down for reasons that also make them harder to resolve.

## 5.6 Paired Same-Input Workflow Comparison

RQ3 compares the paired 100-case direct sameinput workflow conditions in Table 5 to place their outcomes in workflow-level context. THEMIS resolves 19/100 versus 9/100; the paired cells are 9 both resolved, 10 THEMIS-only resolved, 0 direct-only resolved, and 81 unresolved in both conditions. The paired risk difference is +0.10, with an approximate 95% confidence interval of [0.0412, 0.1588]. The exact two-sided McNemar test uses the 10 discordant pairs, 10 THEMIS-only and 0 direct-only, and yields $p ~ = ~ 0 . 0 0 2$ . Nonempty patches occur in 89/100 THEMIS cases and 43/100 direct-condition cases. These results provide preliminary workflow-level evidence of a difference between the two conditions.

<table><tr><td>Paired outcome or patch aggregate</td><td>Cases</td></tr><tr><td>THEMIS resolved</td><td>19/100</td></tr><tr><td>Direct condition resolved Both resolved</td><td>9/100 9</td></tr><tr><td>THEMIS-only resolved Direct-only resolved</td><td>10 0</td></tr><tr><td>Neither resolved THEMIS non-empty patch</td><td>81</td></tr><tr><td>Direct condition non-empty patch</td><td>89/100 43/100</td></tr></table>

Table 5: Paired 100-case official outcomes and nonempty patch aggregates for the same-input workflow comparison. The paired risk difference is +0.10, the approximate 95% confidence interval is [0.0412, 0.1588], and the exact two-sided McNemar p is 0.002.

This is preliminary workflow-level evidence across conditions with distinct Analyzer output, graph-derived distillation and relations, Judge records, interfaces, and token budget. The observed difference supports workflow-level interpretation.

## 6 Discussion

RQ1: externalized stage information exposes continuity gaps. The selected projection externalizes conflict history for 300/300 cases, complete Developer rationale for 288/300, and repairbrief core fields for 222/300. Evaluating their joint availability matters because stage separation supports inspection when the retained pre-edit, edit, and post-edit records can be read together. All three field groups are jointly available in 214/300 cases; the remaining 86 cases locate continuity gaps for targeted serialization improvements. Thus, the availability evaluation identifies the retained records that sustain the stage-aware audit design and the fields that require attention within the selected serialization.

RQ2: separately retained records make lexical correspondence and disagreement observable. Separately retaining targets, rationales, and patches makes the lexical comparison informative because it exposes agreement and disagreement among records as distinct observations. Recorded targets occur in 134/214 Developer rationales and in 130/207 patches under the stated string rule; in the 205-case joint cohort, 95/205 cases divide among rationale-only, patchonly, and zero-overlap cells. The stage-specific records therefore make surface correspondence, disagreement, and target mismatches inspectable. The quality flags complement this comparison as design signals: among the 222 repair-brief cores, 131/222 are low confidence and 138/222 have generic issue summaries, making uncertainty and limited specificity visible for review. They remain record-surface observations under the stated lexical proxy, separate from RQ3 workflow outcomes, and quality-improvement validation remains a future target.

RQ3: the paired direct same-input comparison shows a bounded workflow-level difference. Comparing paired same-input cases matters because it places the two workflow conditions in a common outcome context while retaining the workflow-level interpretation. Across the paired 100 cases, THEMIS resolves 19/100 cases and the direct condition resolves 9/100. The discordant outcomes are 10 THEMIS-only resolutions and 0 direct-only resolutions. The paired risk difference is +0.10, with an approximate 95% confidence interval of [0.0412, 0.1588], and the exact two-sided McNemar test gives $p = 0 . 0 0 2$ . This comparison is therefore informative as preliminary evidence about the reported workflow conditions, within the confounds documented in the experimental setup and limitations.

## 7 Conclusion

Repository-level repair should be evaluated not only by whether a patch passes its tests, but also by whether the process connecting issue requirements, repair decisions, code edits, and postedit evidence can be inspected. This paper contributes THEMIS, a stage-aware repair workflow that externalizes this process through semantic interpretation, explicit requirement–code relations, graph-derived Developer guidance, retained rationale and patch records, and post-edit diagnostics. The empirical results demonstrate the value of this approach in three ways. First, across 300 SWE-bench Lite cases, complete Developer rationales are retained for 288 cases, while 214 cases (71.3%) provide a complete field chain across the selected stages. These results show that THEMIS enables joint inspection of the requirement-to-repair process at benchmark scale. Second, the retained artifacts make cross-stage correspondence systematically measurable: target symbols recur in 62.6% of Developer rationales and 62.8% of patches, increasing to 75.8% when related symbols are considered. This makes it possible to observe how repair targets and decisions persist or evolve across stages rather than remaining hidden behind the final patch. Third, in the paired 100-case comparison, THEMIS resolves 19 cases compared with 9 for the direct same-input condition, providing preliminary, confounded workflow-level evidence for the relational approach; because the two conditions also differ in Analyzer output, distillation, and audit records, this comparison does not yet isolate the graph component’s causal contribution. Taken together, these contributions establish externalized repair artifacts as a practical foundation for inspecting and evaluating the process by which repositorylevel repairs are produced. By turning otherwise implicit requirement-to-repair transitions into observable traces, THEMIS extends repair evaluation beyond final outcomes and enables systematic analysis of how requirements are interpreted, translated into edits, and reflected in post-edit evidence.

## 8 Limitations

Workflow and comparison boundaries. The documented experiments use one Developer revision, with the Judge acting only as a post-edit audit record. RQ3 compares paired workflows that differ in Analyzer output, graph-based distillation and relations, Judge records, token budget, and interface. Lineage for this comparison is reconstructed from experiment metadata and baseline declarations. The study reports workflow-level outcomes for these configurations; future work can test representation, module, artifact, lexicalpattern, causal, and general-superiority questions.

Record and proxy boundaries. The selected serialization captures requirement and target summaries, Developer rationale, patch records, and post-edit diagnostics; the full runtime graph, complete requirements, and complete reasoning remain runtime material. Requirement decomposition, requirement-code mapping, and violation flagging are heuristic. Lexical matching measures string-level surface correspondence. Semantic correctness, trace correctness, and target-topatch causal validation remain future targets, as does validation of the contents and relations represented by field availability and asserted links. A principled limitation applies to this proxy: the retrospective target used for RQ2 (repair\_brief.target\_symbol) is generated by the same Judge component after the edit and graph rebuild, rather than fixed before editing or drawn from an independent reference (e.g., the gold patch’s changed lines). High lexical overlap may therefore partly reflect the Judge’s agreement with the Developer’s own edit rather than independently established cross-stage traceability.

Utility, scope, and model dependence. Human-utility evaluation of whether these records help readers diagnose failures, debug systems, or make better decisions remains a future validation target. Evaluation covers Python repositories in SWE-bench Lite and one model configuration, gpt-5.1-codex-mini. Future validation across languages, benchmarks, models, model scales, and repair settings can assess language-specific extraction and edit logic alongside workflow outcomes.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 24K15077.

Following the ACL Policy on AI Writing Assistance, we disclose our use of AI assistants in preparing this work: OpenCode assisted with implementing the THEMIS codebase, and Codex assisted with translating and polishing portions of the manuscript text. This writing-assistance use of Codex is unrelated to the gpt-5.1-codex-mini configuration evaluated as a Developer model in Section 4. All AIassisted content was reviewed and verified by the authors, who take full responsibility for it.

## References

Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. 2018. Learning to represent pro-

grams with graphs. Preprint, arXiv:1711.00740.

Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. 2019a. code2seq: Generating sequences from structured representations of code. Preprint, arXiv:1808.01400.

Uri Alon, Meital Zilberstein, Omer Levy, and Eran Yahav. 2019b. code2vec: Learning distributed representations of code. Preprint, arXiv:1803.09473.

Markus Borg, Per Runeson, and Anders Ardö. 2014. Recovering from a decade: A systematic mapping of information retrieval approaches to software traceability. Empirical Software Engineering, 19(6):1565–1616.

Jane Cleland-Huang, Olly Gotel, and Andrea Zisman, editors. 2012. Software and Systems Traceability. Springer.

Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, and Ming Zhou. 2021. Graphcodebert: Pre-training code representations with data flow. Preprint, arXiv:2009.08366.

Nan Jiang, Kevin Liu, Thibaud Lutellier, and Lin Tan. 2023. Impact of code language models on automated program repair. In Proceedings of the 45th International Conference on Software Engineering, pages 1430–1442.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues? Preprint, arXiv:2310.06770.

Claire Le Goues, Michael Dewey-Vogt, Stephanie Forrest, and Westley Weimer. 2012. A systematic study of automated program repair: Fixing 55 out of 105 bugs for \$8 each. In Proceedings of the 34th International Conference on Software Engineering, pages 3–13.

Fan Long and Martin Rinard. 2016. Automatic patch generation by learning correct code. In Proceedings of the 43rd Annual ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, pages 298–312.

Sergey Mechtaev, Jooyong Yi, and Abhik Roychoudhury. 2016. Angelix: Scalable multiline program patch synthesis via symbolic analysis. In Proceedings of the 38th International Conference on Software Engineering, pages 691–701.

Martin Monperrus. 2018. Automatic software repair: A bibliography. ACM Computing Surveys, 51(1):1– 24.

Hoang Duong Thien Nguyen, Dawei Qi, Abhik Roychoudhury, and Satish Chandra. 2013. Semfix: Program repair via semantic analysis. In Proceedings of the 35th International Conference on Software Engineering, pages 772–781.

Dominik Sobania, Martin Briesch, Carol Hanna, and Justyna Petke. 2023. An analysis of the automatic bug fixing performance of chatgpt. Preprint, arXiv:2301.08653.

Yiqi Wang, Jiaqi Zhang, Zhangkai Wu, Taotao Cai, Zirui Liu, Zequn Sun, Manqing Dong, Mingkai Zheng, Yiqun Duan, Xuefei Yin, and Yanming Zhu. 2026. From agent traces to trust: A survey of evidence tracing and execution provenance in llm agents. Preprint, arXiv:2606.04990.

Westley Weimer, ThanhVu Nguyen, Claire Le Goues, and Stephanie Forrest. 2009. Automatically finding patches using genetic programming. In Proceedings ofthe 31st International Conference on Software Engineering, pages 364–374.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2024a. Agentless: Demystifying llm-based software engineering agents. Preprint, arXiv:2407.01489.

Chunqiu Steven Xia, Yuxiang Wei, and Lingming Zhang. 2024b. Automated program repair via conversation: Fixing 162 out of 337 bugs for \$0.42 each using chatgpt. In Proceedings ofthe 33rd ACM SIG-SOFT International Symposium on Software Testing and Analysis.

Chunqiu Steven Xia and Lingming Zhang. 2023. Conversational automated program repair. Preprint, arXiv:2301.13246.

Jifeng Xuan, Matias Martinez, Favio Demarco, Maxime Clément, Sebastian Lamelas Marcote, Thomas Durieux, Daniel Le Berre, and Martin Monperrus. 2017. Nopol: Automatic repair of conditional statement bugs in java programs. IEEE Transactions on Software Engineering, 43(1):34–55.

Fabian Yamaguchi, Nico Golde, Daniel Arp, and Konrad Rieck. 2014. Modeling and discovering vulnerabilities with code property graphs. In Proceedings of the 2014 IEEE Symposium on Security and Privacy, pages 590–604.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. Preprint, arXiv:2405.15793.